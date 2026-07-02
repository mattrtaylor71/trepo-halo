// lib/signals.mjs — signal pollers.
//
// RULE: every poller asserts on a concrete metric datapoint, log line, or
// DynamoDB row. NEVER assert on CloudWatch *alarm state* — an alarm can be OK
// simply because nothing has happened yet. We prove the underlying signal is
// readable, not that an alarm happens to be green.

import {
  CloudWatchClient,
  GetMetricStatisticsCommand,
} from '@aws-sdk/client-cloudwatch';
import {
  CloudWatchLogsClient,
  FilterLogEventsCommand,
} from '@aws-sdk/client-cloudwatch-logs';
import {
  DynamoDBClient,
} from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  QueryCommand,
} from '@aws-sdk/lib-dynamodb';
import { REGION, ANALYTICS_TABLE } from './config.mjs';

const cw = new CloudWatchClient({ region: REGION });
const cwl = new CloudWatchLogsClient({ region: REGION });
const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({ region: REGION }));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const nowMs = () => Date.now();

/**
 * Generic poll loop. `fn` returns {ok, detail} — poll until ok or timeout.
 * @returns {ok, detail, elapsedMs, timedOut}
 */
async function poll(fn, { timeoutMs = 300_000, intervalMs = 10_000, label = 'signal' } = {}) {
  const start = nowMs();
  let last = { ok: false, detail: 'no attempt yet' };
  while (nowMs() - start < timeoutMs) {
    try {
      last = await fn();
    } catch (err) {
      last = { ok: false, detail: `poll error: ${err.message}` };
    }
    if (last.ok) {
      return { ...last, elapsedMs: nowMs() - start, timedOut: false };
    }
    await sleep(intervalMs);
  }
  return { ...last, elapsedMs: nowMs() - start, timedOut: true };
}

/**
 * Assert a CloudWatch metric has a Sum > 0 datapoint since `sinceMs`.
 */
export async function cloudwatchMetric({
  namespace, metricName, dimensions = [], sinceMs,
  timeoutMs = 300_000, intervalMs = 10_000,
}) {
  const startTime = new Date(sinceMs - 60_000); // pad 1 min back for filter latency
  const dims = dimensions.map((d) => ({ Name: d.name, Value: d.value }));
  return poll(async () => {
    const res = await cw.send(new GetMetricStatisticsCommand({
      Namespace: namespace,
      MetricName: metricName,
      Dimensions: dims,
      StartTime: startTime,
      EndTime: new Date(nowMs() + 60_000),
      Period: 60,
      Statistics: ['Sum'],
    }));
    const pts = (res.Datapoints || []).filter((p) => (p.Sum || 0) > 0);
    const total = pts.reduce((a, p) => a + p.Sum, 0);
    if (pts.length > 0) {
      return { ok: true, detail: `${metricName} Sum=${total} across ${pts.length} datapoint(s)` };
    }
    return { ok: false, detail: `${metricName} no Sum>0 datapoint yet (since ${startTime.toISOString()})` };
  }, { timeoutMs, intervalMs, label: metricName });
}

/**
 * Assert a log line matching `pattern` appears in `logGroup` since `sinceMs`.
 * `pattern` is a CloudWatch Logs filter pattern (quote literals) OR a plain
 * substring we grep client-side (set `clientMatch:true`).
 */
export async function logLine({
  logGroup, pattern, sinceMs, clientMatch = false,
  timeoutMs = 300_000, intervalMs = 10_000,
}) {
  return poll(async () => {
    const params = {
      logGroupName: logGroup,
      startTime: sinceMs,
      limit: 50,
    };
    if (!clientMatch) params.filterPattern = pattern;
    const res = await cwl.send(new FilterLogEventsCommand(params));
    let events = res.events || [];
    if (clientMatch) {
      events = events.filter((e) => (e.message || '').includes(pattern));
    }
    if (events.length > 0) {
      const sample = (events[0].message || '').trim().slice(0, 160);
      return { ok: true, detail: `${events.length} match(es); e.g. "${sample}"` };
    }
    return { ok: false, detail: `no log line matching ${JSON.stringify(pattern)} since ${new Date(sinceMs).toISOString()}` };
  }, { timeoutMs, intervalMs, label: 'logLine' });
}

/**
 * Assert a DynamoDB analytics event row exists for `ownerId` with `eventName`
 * at or after `sinceIso`. Optionally require `propsMatch` (subset of
 * properties) to match. Uses the base table (PK owner_id, SK ts_id begins/gte)
 * by default; set `useGsi:true` to query EventNameIndex.
 */
export async function dynamoEvent({
  ownerId, eventName, sinceIso, propsMatch = null, useGsi = false,
  timeoutMs = 120_000, intervalMs = 10_000,
}) {
  return poll(async () => {
    let items = [];
    if (useGsi) {
      const res = await ddb.send(new QueryCommand({
        TableName: ANALYTICS_TABLE,
        IndexName: 'EventNameIndex',
        KeyConditionExpression: 'event_name = :en',
        ExpressionAttributeValues: { ':en': eventName },
        ScanIndexForward: false,
        Limit: 100,
      }));
      items = (res.Items || []).filter((it) => it.owner_id === ownerId);
    } else {
      // Base table: PK owner_id, SK ts_id is `${iso}#${suffix}` so ts_id >= sinceIso works.
      const res = await ddb.send(new QueryCommand({
        TableName: ANALYTICS_TABLE,
        KeyConditionExpression: 'owner_id = :o AND ts_id >= :s',
        ExpressionAttributeValues: { ':o': ownerId, ':s': sinceIso },
        ScanIndexForward: false,
        Limit: 100,
      }));
      items = res.Items || [];
    }
    let matches = items.filter((it) => !eventName || it.event_name === eventName);
    if (propsMatch) {
      matches = matches.filter((it) => {
        const p = it.properties || {};
        return Object.entries(propsMatch).every(([k, v]) => String(p[k]) === String(v));
      });
    }
    if (matches.length > 0) {
      return { ok: true, detail: `${matches.length} row(s) for ${ownerId}/${eventName} (ts_id ${matches[0].ts_id})` };
    }
    return { ok: false, detail: `no ${eventName} row for ${ownerId} since ${sinceIso}` };
  }, { timeoutMs, intervalMs, label: 'dynamoEvent' });
}

/**
 * Poll GET {baseUrl}/job/{jobId} until status === `want` (or one of `want[]`).
 */
export async function jobStatus({
  baseUrl, jobId, want, timeoutMs = 300_000, intervalMs = 10_000,
}) {
  const wants = Array.isArray(want) ? want.map((w) => w.toUpperCase()) : [String(want).toUpperCase()];
  return poll(async () => {
    const res = await fetch(`${baseUrl}/job/${encodeURIComponent(jobId)}`);
    let body = {};
    try { body = await res.json(); } catch (_) { /* non-json */ }
    const status = String(body.status || body.state || '').toUpperCase();
    if (status && wants.includes(status)) {
      return { ok: true, detail: `job ${jobId} status=${status}` };
    }
    return { ok: false, detail: `job ${jobId} status=${status || res.status} (want ${wants.join('|')})` };
  }, { timeoutMs, intervalMs, label: 'jobStatus' });
}

/**
 * Simple HTTP request; assert status is in `wantStatuses`.
 * @returns {ok, status, detail, body}
 */
export async function httpStatus({ url, method = 'GET', headers = {}, body = null, wantStatuses = [200] }) {
  const opts = { method, headers: { ...headers } };
  if (body !== null && body !== undefined) {
    if (typeof body === 'string') {
      opts.body = body;
    } else {
      opts.body = JSON.stringify(body);
      opts.headers['Content-Type'] = opts.headers['Content-Type'] || 'application/json';
    }
  }
  let status = 0;
  let text = '';
  try {
    const res = await fetch(url, opts);
    status = res.status;
    text = await res.text();
  } catch (err) {
    return { ok: false, status: 0, detail: `request failed: ${err.message}`, body: '' };
  }
  const ok = wantStatuses.includes(status);
  return {
    ok,
    status,
    detail: `${method} ${url} -> ${status} (want ${wantStatuses.join('|')})`,
    body: text,
  };
}

export { sleep, nowMs };
