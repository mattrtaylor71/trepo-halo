// flows/analytics_ingest.mjs — analytics ingest pipeline.
import { API, TEST_OWNER } from '../lib/config.mjs';
import { httpStatus, dynamoEvent, nowMs } from '../lib/signals.mjs';

const rand8 = () => Math.random().toString(16).slice(2, 10);

export const name = 'analytics_ingest';

export const tests = [
  {
    label: 'ingest harness_probe -> 201 -> Dynamo row',
    mode: 'e2e',
    async run(ctx) {
      const iso = new Date().toISOString();
      const sinceIso = new Date(nowMs() - 60_000).toISOString();
      const event = {
        event_name: 'harness_probe',
        owner_id: TEST_OWNER,
        device_id: 'obs-harness',
        timestamp: iso,
        event_id: rand8(),
        properties: { test_run_id: ctx.runId },
        app_version: 'harness',
      };
      const post = await httpStatus({
        url: `${API.analytics}/analytics/events`,
        method: 'POST',
        body: { events: [event] },
        wantStatuses: [200, 201, 202],
      });
      if (!post.ok) {
        return { status: 'FAIL', mode: 'e2e', detail: `ingest POST ${post.status}: ${post.body.slice(0, 160)}` };
      }
      const row = await dynamoEvent({
        ownerId: TEST_OWNER,
        eventName: 'harness_probe',
        sinceIso,
        propsMatch: { test_run_id: ctx.runId },
        timeoutMs: 120_000,
      });
      if (!row.ok) {
        return { status: 'FAIL', mode: 'e2e', detail: `POST ${post.status} but row not found: ${row.detail}` };
      }
      return { status: 'PASS', mode: 'e2e', detail: `POST ${post.status}; ${row.detail}` };
    },
  },
  {
    label: 'malformed body -> 400',
    mode: 'e2e',
    async run() {
      const res = await httpStatus({
        url: `${API.analytics}/analytics/events`,
        method: 'POST',
        body: {},
        wantStatuses: [400, 422],
      });
      return {
        status: res.ok ? 'PASS' : 'FAIL',
        mode: 'e2e',
        detail: res.detail + (res.ok ? '' : ` body=${res.body.slice(0, 120)}`),
      };
    },
  },
];
