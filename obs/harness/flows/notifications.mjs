// flows/notifications.mjs — notifications API.
import { API, LOG_GROUPS, CAPTURE_NS } from '../lib/config.mjs';
import { httpStatus } from '../lib/signals.mjs';
import { syntheticSignal } from '../lib/synthetic.mjs';

export const name = 'notifications';

export const tests = [
  {
    label: 'GET /notifications/users -> 200',
    mode: 'e2e',
    async run() {
      const res = await httpStatus({ url: `${API.notifications}/notifications/users`, method: 'GET', wantStatuses: [200] });
      return { status: res.ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: res.detail + (res.ok ? '' : ` body=${res.body.slice(0, 120)}`) };
    },
  },
  {
    label: 'synthetic backend_error -> NotificationsBackendError',
    mode: 'synthetic',
    async run(ctx) {
      const marker = {
        evt: 'backend_error',
        service: 'notifications',
        op: 'list_users',
        error: `obs-harness synthetic notifications backend_error (run ${ctx.runId})`,
      };
      return syntheticSignal({ logGroup: LOG_GROUPS.notifications, marker, metricName: 'NotificationsBackendError', service: 'notifications', namespace: CAPTURE_NS });
    },
  },
];
