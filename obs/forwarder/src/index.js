'use strict';

const zlib = require('zlib');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, PutCommand } = require('@aws-sdk/lib-dynamodb');

const client = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.TABLE_NAME || 'TrepoAnalyticsEvents';

const ALLOWED_EVT = new Set(['analysis_failed', 'backend_error', 'ai_op']);
// Additional soft-failure markers that should surface in the errors feed. Each is
// normalised to a backend_error item with a derived service so it groups sensibly.
const SOFT_EVT_SERVICE = {
  shopping_peruser_write_miss: 'voice',
  discard_peruser_write_miss: 'voice',
  bulk_gemini_failover: 'capture',
  recipe_batch_partial_failure: 'recipes',
};
const FREE_TEXT_MARKER = 'Background processing error';

function truncate(v, n) {
  if (v === undefined || v === null) return undefined;
  const s = String(v);
  return s.length > n ? s.slice(0, n) : s;
}

// Extract an 8-char suffix, preferring a requestId from the marker or a UUID in the raw line.
function suffix8(rawMessage, marker) {
  const rid = marker && (marker.requestId || marker.request_id || marker.awsRequestId);
  if (rid) return String(rid).replace(/-/g, '').slice(0, 8).padEnd(8, '0');
  const m = String(rawMessage || '').match(
    /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i
  );
  if (m) return m[0].replace(/-/g, '').slice(0, 8);
  return Math.random().toString(16).slice(2, 10).padEnd(8, '0');
}

// Try to parse a JSON object out of a log line that may be prefixed with a
// timestamp / requestId. Find the first '{' and JSON.parse from there.
function parseJsonMarker(message) {
  if (typeof message !== 'string') return null;
  const idx = message.indexOf('{');
  if (idx === -1) return null;
  try {
    const obj = JSON.parse(message.slice(idx));
    return obj && typeof obj === 'object' && !Array.isArray(obj) ? obj : null;
  } catch (_) {
    return null;
  }
}

// Turn a single log event into a marker object (or null to skip).
function markerFromLogEvent(message) {
  const parsed = parseJsonMarker(message);
  if (parsed && ALLOWED_EVT.has(parsed.evt)) {
    return parsed;
  }
  // Soft-failure markers (voice write-miss, gemini failover, recipe partial): normalise
  // to a backend_error so they land in the errors feed with a sensible service + code.
  if (parsed && SOFT_EVT_SERVICE[parsed.evt]) {
    return {
      ...parsed,
      evt: 'backend_error',
      service: parsed.service || SOFT_EVT_SERVICE[parsed.evt],
      op: parsed.op || parsed.evt,
      code: parsed.code || parsed.evt,
    };
  }
  // Free-text fallback marker for legacy "Background processing error" lines.
  if (typeof message === 'string' && message.includes(FREE_TEXT_MARKER)) {
    return {
      evt: 'backend_error',
      service: 'bulk',
      op: 'background_processing',
      error: truncate(message.trim(), 500),
    };
  }
  // Free-text fallback for the second worker failure path: "[IdentifyAsync] Job <id> failed:" / "[BulkCommit] Job <id> failed:".
  if (typeof message === 'string') {
    const m = message.match(/\[(IdentifyAsync|BulkCommit)\] Job (\S+)(?: identify)? failed:/);
    if (m) {
      return {
        evt: 'backend_error',
        service: 'bulk',
        op: m[1] === 'IdentifyAsync' ? 'identify_job_failed' : 'commit_job_failed',
        job_id: m[2],
        error: truncate(message.trim(), 500),
      };
    }
  }
  return null;
}

function buildItem(marker, logEvent, logGroup) {
  const service = marker.service || 'unknown';
  const ownerId = marker.owner_id || marker.ownerId || `backend#${service}`;

  // Prefer the log event's own timestamp (epoch ms) over Date.now().
  const epochMs =
    logEvent && typeof logEvent.timestamp === 'number'
      ? logEvent.timestamp
      : Date.now();
  const iso = new Date(epochMs).toISOString();

  const props = {};
  const rawMsg = logEvent ? logEvent.message : '';
  const setProp = (k, v) => {
    const t = truncate(v, k === 'error' ? 500 : 1024);
    if (t !== undefined && t !== '') props[k] = t;
  };
  setProp('service', marker.service);
  setProp('op', marker.op);
  setProp('code', marker.code);
  setProp('error', marker.error);
  setProp('job_id', marker.job_id !== undefined ? marker.job_id : marker.jobId);
  setProp('log_group', logGroup);
  setProp('kind', marker.kind);
  setProp('stage', marker.stage);
  // ai_op markers: per-LLM-call telemetry (model, latency, status, io previews).
  setProp('model', marker.model);
  setProp('status', marker.status);
  setProp('input', truncate(marker.input, 400));
  setProp('output', truncate(marker.output, 1200));
  if (marker.latency_ms !== undefined && marker.latency_ms !== null) {
    props.latency_ms = Number(marker.latency_ms) || 0;
  }
  if (marker.tokens_in != null) props.tokens_in = Number(marker.tokens_in) || 0;
  if (marker.tokens_out != null) props.tokens_out = Number(marker.tokens_out) || 0;

  return {
    owner_id: ownerId,
    ts_id: `${iso}#${suffix8(rawMsg, marker)}`,
    event_name: marker.evt === 'ai_op' ? 'ai_op' : 'backend_error',
    device_id: 'backend',
    app_version: 'backend',
    timestamp: iso,
    created_at: iso,
    properties: props,
  };
}

exports.handler = async (event) => {
  let payload;
  try {
    const buff = Buffer.from(event.awslogs.data, 'base64');
    payload = JSON.parse(zlib.gunzipSync(buff).toString('utf8'));
  } catch (err) {
    console.error('Failed to decode awslogs data:', err);
    return { ok: false, reason: 'decode_failed' };
  }

  // CONTROL_MESSAGE is sent by CloudWatch when a subscription is created; ignore it.
  if (payload.messageType && payload.messageType !== 'DATA_MESSAGE') {
    return { ok: true, skipped: 'control_message' };
  }

  const logGroup = payload.logGroup;
  const logEvents = Array.isArray(payload.logEvents) ? payload.logEvents : [];

  let written = 0;
  let matched = 0;
  for (const le of logEvents) {
    let marker;
    try {
      marker = markerFromLogEvent(le.message);
    } catch (err) {
      console.error('marker parse error (skipping record):', err);
      continue;
    }
    if (!marker) continue;
    matched += 1;

    try {
      const item = buildItem(marker, le, logGroup);
      await client.send(new PutCommand({ TableName: TABLE, Item: item }));
      written += 1;
    } catch (err) {
      // Never throw on a single bad record — log and continue.
      console.error('PutItem failed (skipping record):', err, 'message=', le.message);
    }
  }

  console.log(
    JSON.stringify({ logGroup, events: logEvents.length, matched, written })
  );
  return { ok: true, matched, written };
};
