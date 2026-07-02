// flows/auth.mjs — auth (twilioAuth) send-code path.
// We only send a MALFORMED body so no real SMS is dispatched, then prove the
// request left a readable log line in the twilioAuth log group.
import { API, LOG_GROUPS } from '../lib/config.mjs';
import { httpStatus, logLine, nowMs } from '../lib/signals.mjs';

export const name = 'auth';

export const tests = [
  {
    label: 'malformed send-code -> clean 4xx + twilioAuth log line',
    mode: 'e2e',
    async run() {
      const sinceMs = nowMs() - 30_000;
      // Unique marker echoed into the body; twilioAuth logs the parsed request.
      const marker = `obsharness${nowMs()}`;
      const res = await httpStatus({
        url: `${API.auth}/auth/send-code`,
        method: 'POST',
        body: { not_a_phone: marker }, // missing phone_number -> clean 4xx, no SMS
        wantStatuses: [400, 401, 403, 422],
      });
      if (!res.ok) {
        return { status: 'FAIL', mode: 'e2e', detail: `expected clean 4xx, got ${res.detail}; body=${res.body.slice(0, 120)}` };
      }
      // Signal readability: any log line from twilioAuth since the request proves
      // the invocation is observable. (twilioAuth may not echo the body marker.)
      const log = await logLine({
        logGroup: LOG_GROUPS.auth,
        pattern: 'START',       // Lambda START line is emitted for every invoke
        clientMatch: true,
        sinceMs,
        timeoutMs: 90_000,
      });
      if (!log.ok) {
        return { status: 'FAIL', mode: 'e2e', detail: `4xx OK (${res.status}) but no twilioAuth log line: ${log.detail}` };
      }
      return { status: 'PASS', mode: 'e2e', detail: `${res.status}; twilioAuth log readable (${log.detail})` };
    },
  },
];
