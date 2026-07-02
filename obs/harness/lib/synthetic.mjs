// lib/synthetic.mjs — synthetic signal injection.
//
// Inject a marker line into a log group (create a dedicated stream), then assert
// (a) the metric-filter datapoint fires, and (b) IF the group is subscribed to
// the analytics forwarder, the forwarded TrepoAnalyticsEvents row appears
// (owner_id `backend#<service>`).
//
// The forwarder (obs/forwarder) recognises JSON markers with evt in
// {'analysis_failed','backend_error'} and writes event_name='backend_error',
// owner_id = marker.owner_id || `backend#<service>`.

import {
  CloudWatchLogsClient,
  CreateLogStreamCommand,
  PutLogEventsCommand,
  DescribeSubscriptionFiltersCommand,
  DescribeMetricFiltersCommand,
} from '@aws-sdk/client-cloudwatch-logs';
import { REGION } from './config.mjs';
import { cloudwatchMetric, dynamoEvent, nowMs } from './signals.mjs';

const cwl = new CloudWatchLogsClient({ region: REGION });

/**
 * Is `logGroup` subscribed to any forwarder destination?
 */
export async function isSubscribed(logGroup) {
  try {
    const res = await cwl.send(new DescribeSubscriptionFiltersCommand({ logGroupName: logGroup }));
    return (res.subscriptionFilters || []).length > 0;
  } catch (_) {
    return false;
  }
}

/**
 * Does a metric filter on `logGroup` produce `namespace:metricName`?
 * Auto-adapts as concurrent WPs add filters. Returns {exists, pattern}.
 */
export async function metricFilterInfo(logGroup, namespace, metricName) {
  try {
    const res = await cwl.send(new DescribeMetricFiltersCommand({ logGroupName: logGroup }));
    for (const f of res.metricFilters || []) {
      for (const t of f.metricTransformations || []) {
        if (t.metricNamespace === namespace && t.metricName === metricName) {
          return { exists: true, pattern: f.filterPattern };
        }
      }
    }
  } catch (_) { /* group may not exist */ }
  return { exists: false, pattern: null };
}

/**
 * Inject a marker line into `logGroup` on a fresh stream.
 * @returns the marker object actually written.
 */
export async function injectMarker({ logGroup, marker, tsSuffix = '' }) {
  const streamName = `obs-harness-${nowMs()}${tsSuffix ? '-' + tsSuffix : ''}`;
  await cwl.send(new CreateLogStreamCommand({ logGroupName: logGroup, logStreamName: streamName }));
  const line = JSON.stringify(marker);
  await cwl.send(new PutLogEventsCommand({
    logGroupName: logGroup,
    logStreamName: streamName,
    logEvents: [{ timestamp: nowMs(), message: line }],
  }));
  return { streamName, line };
}

/**
 * Full synthetic check for a metric-filter-backed signal.
 *
 * @param logGroup    log group to inject into
 * @param marker      JSON marker object (must contain the literal the filter matches)
 * @param metricName  Trepo/Capture metric expected to fire
 * @param service     backend service name (for the forwarded backend#<service> row)
 * @param namespace   metric namespace (default Trepo/Capture)
 * @returns {status:'PASS'|'FAIL'|'SKIP', mode:'synthetic', detail}
 *
 * If no metric filter for `namespace:metricName` exists on the log group yet
 * (e.g. the owning WP hasn't created it), the test SKIPs rather than FAILs.
 */
export async function syntheticSignal({
  logGroup, marker, metricName, service, namespace = 'Trepo/Capture',
  metricTimeoutMs = 300_000,
}) {
  const filter = await metricFilterInfo(logGroup, namespace, metricName);
  if (!filter.exists) {
    return {
      status: 'SKIP', mode: 'synthetic',
      detail: `no metric filter for ${namespace}:${metricName} on ${logGroup} yet (not created by owning WP)`,
    };
  }
  const sinceMs = nowMs();
  let injected;
  try {
    injected = await injectMarker({ logGroup, marker });
  } catch (err) {
    return { status: 'FAIL', mode: 'synthetic', detail: `inject failed: ${err.message}` };
  }

  const metricRes = await cloudwatchMetric({
    namespace, metricName, sinceMs, timeoutMs: metricTimeoutMs,
  });
  if (!metricRes.ok) {
    return {
      status: 'FAIL', mode: 'synthetic',
      detail: `injected on ${injected.streamName}; metric NOT seen: ${metricRes.detail}`,
    };
  }

  // Metric fired. If the group is subscribed to the forwarder, also assert the row.
  const subscribed = await isSubscribed(logGroup);
  if (!subscribed) {
    return {
      status: 'PASS', mode: 'synthetic',
      detail: `metric fired (${metricRes.detail}); group not subscribed to forwarder -> Dynamo row not expected`,
    };
  }

  const ownerId = marker.owner_id || `backend#${service}`;
  const rowRes = await dynamoEvent({
    ownerId, eventName: 'backend_error',
    sinceIso: new Date(sinceMs - 60_000).toISOString(),
    timeoutMs: 120_000,
  });
  if (!rowRes.ok) {
    return {
      status: 'FAIL', mode: 'synthetic',
      detail: `metric fired (${metricRes.detail}) but forwarded row missing: ${rowRes.detail}`,
    };
  }
  return {
    status: 'PASS', mode: 'synthetic',
    detail: `metric fired (${metricRes.detail}); forwarded row present (${rowRes.detail})`,
  };
}
