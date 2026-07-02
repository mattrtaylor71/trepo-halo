// flows/kitchen.mjs — KitchenApi (main API 7tn3gvwvh7).
// Body shape from playground6_groceryRec/kitchen_api/app.py:_build_create_payload.
import { API, TEST_OWNER, HARNESS_ITEM_NAME, CAPTURE_NS, APIGW } from '../lib/config.mjs';
import { httpStatus, cloudwatchMetric, nowMs } from '../lib/signals.mjs';

export const name = 'kitchen';

async function getItems(owner) {
  const res = await httpStatus({ url: `${API.main}/kitchen/${owner}`, method: 'GET', wantStatuses: [200] });
  let items = [];
  try { items = (JSON.parse(res.body).items) || []; } catch (_) { /* */ }
  return { res, items };
}

export const tests = [
  {
    label: 'GET /kitchen/{owner} -> 200 items[]',
    mode: 'e2e',
    async run() {
      const { res, items } = await getItems(TEST_OWNER);
      if (!res.ok) return { status: 'FAIL', mode: 'e2e', detail: res.detail };
      if (!Array.isArray(items)) return { status: 'FAIL', mode: 'e2e', detail: 'response missing items[] array' };
      return { status: 'PASS', mode: 'e2e', detail: `200; items[] len=${items.length}` };
    },
  },
  {
    label: 'add item -> present -> delete (cleanup)',
    mode: 'e2e',
    async run(ctx) {
      const jobId = `obs-harness-${ctx.runId}`;
      const body = {
        product_name: HARNESS_ITEM_NAME,
        device_id: 'obs-harness',
        job_id: jobId,
        action: 'IN',
        defer_recipes: true,
      };
      const add = await httpStatus({
        url: `${API.main}/kitchen/${TEST_OWNER}`,
        method: 'POST',
        body,
        wantStatuses: [200, 201],
      });
      if (!add.ok) return { status: 'FAIL', mode: 'e2e', detail: `add POST ${add.status}: ${add.body.slice(0, 160)}` };
      let itemId = null;
      try { itemId = JSON.parse(add.body)?.item?._id; } catch (_) { /* */ }
      if (!itemId) {
        // Fall back to locating by name.
        const { items } = await getItems(TEST_OWNER);
        itemId = (items.find((i) => i.product_name === HARNESS_ITEM_NAME) || {})._id;
      }
      if (!itemId) return { status: 'FAIL', mode: 'e2e', detail: `created but no _id resolvable; add body=${add.body.slice(0, 160)}` };

      // Register cleanup regardless of the verify outcome below.
      ctx.cleanup.push(async () => {
        await httpStatus({ url: `${API.main}/kitchen/${TEST_OWNER}/${itemId}`, method: 'DELETE', wantStatuses: [200, 204, 404] });
      });

      // Verify present.
      const { items } = await getItems(TEST_OWNER);
      const present = items.some((i) => i._id === itemId);
      if (!present) return { status: 'FAIL', mode: 'e2e', detail: `item ${itemId} not present after add` };

      // Delete now (cleanup) and verify gone.
      const del = await httpStatus({ url: `${API.main}/kitchen/${TEST_OWNER}/${itemId}`, method: 'DELETE', wantStatuses: [200, 204] });
      if (!del.ok) return { status: 'FAIL', mode: 'e2e', detail: `delete ${del.status}: ${del.body.slice(0, 120)}` };
      return { status: 'PASS', mode: 'e2e', detail: `add ${add.status}, present, delete ${del.status} (id ${itemId.slice(0, 8)}…)` };
    },
  },
  {
    label: 'malformed POST -> 4xx (or 5xx-with-metric)',
    mode: 'e2e',
    async run() {
      const sinceMs = nowMs();
      const res = await httpStatus({
        url: `${API.main}/kitchen/${TEST_OWNER}`,
        method: 'POST',
        body: {}, // missing product_name -> 400
        wantStatuses: [400, 401, 403, 422],
      });
      if (res.ok) return { status: 'PASS', mode: 'e2e', detail: `${res.status} (clean 4xx)` };
      if (res.status >= 500) {
        // 5xx is also an observable failure — assert the API Gateway 5xx metric.
        const m = await cloudwatchMetric({
          namespace: 'AWS/ApiGateway',
          metricName: '5xx',
          dimensions: [{ name: 'ApiId', value: APIGW.mainId }],
          sinceMs,
          timeoutMs: 180_000,
        });
        return {
          status: m.ok ? 'PASS' : 'FAIL',
          mode: 'e2e',
          detail: m.ok ? `${res.status}; ApiGateway 5xx metric fired (${m.detail})` : `${res.status} but 5xx metric not seen: ${m.detail}`,
        };
      }
      return { status: 'FAIL', mode: 'e2e', detail: `unexpected ${res.detail}; body=${res.body.slice(0, 120)}` };
    },
  },
];
