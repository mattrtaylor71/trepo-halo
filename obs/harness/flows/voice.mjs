// flows/voice.mjs — HALO voice quick-ack path.
import { API, TEST_OWNER, LOG_GROUPS, CAPTURE_NS } from '../lib/config.mjs';
import { httpStatus } from '../lib/signals.mjs';
import { syntheticSignal } from '../lib/synthetic.mjs';

export const name = 'voice';

export const tests = [
  {
    label: 'POST /voice-ack minimal body -> 2xx',
    mode: 'e2e',
    async run(ctx) {
      const res = await httpStatus({
        url: API.voiceAck,
        method: 'POST',
        headers: {
          'x-owner-id': TEST_OWNER,
          'x-device-id': 'obs-harness',
          'x-client-surface': 'app',
          'Content-Type': 'application/json',
        },
        body: { message: 'obs harness readability ping', session_id: `obs-harness-${ctx.runId}` },
        wantStatuses: [200, 201, 202],
      });
      return { status: res.ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: res.detail + (res.ok ? '' : ` body=${res.body.slice(0, 140)}`) };
    },
  },
  {
    label: 'synthetic backend_error -> VoiceBackendError',
    mode: 'synthetic',
    async run(ctx) {
      const marker = {
        evt: 'backend_error',
        service: 'voice',
        op: 'quick_ack_async_worker',
        error: `obs-harness synthetic voice backend_error (run ${ctx.runId})`,
      };
      return syntheticSignal({ logGroup: LOG_GROUPS.voiceWorker, marker, metricName: 'VoiceBackendError', service: 'voice', namespace: CAPTURE_NS });
    },
  },
];
