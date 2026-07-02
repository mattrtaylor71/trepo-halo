// flows/list.mjs — trepo-list-handler via POST {auth}/v1/list.
// Request shape mirrors iOS TrepoAPIService.addItem/removeItem.
import { API, TEST_OWNER, HARNESS_ITEM_NAME } from '../lib/config.mjs';
import { httpStatus } from '../lib/signals.mjs';

export const name = 'list';

export const tests = [
  {
    label: 'add item -> delete by itemUUID (cleanup)',
    mode: 'e2e',
    async run(ctx) {
      const add = await httpStatus({
        url: `${API.auth}/list`,
        method: 'POST',
        body: {
          operation: 'add',
          ownerId: TEST_OWNER,
          device: 'obs-harness',
          product_name: HARNESS_ITEM_NAME,
        },
        wantStatuses: [200, 201],
      });
      if (!add.ok) return { status: 'FAIL', mode: 'e2e', detail: `add ${add.status}: ${add.body.slice(0, 160)}` };

      let itemUUID = null;
      try {
        const j = JSON.parse(add.body);
        const first = (j.items || [])[0] || {};
        itemUUID = first.itemUUID || first.household_item_uuid || first.item_uuid || null;
      } catch (_) { /* */ }
      if (!itemUUID) return { status: 'FAIL', mode: 'e2e', detail: `added but no itemUUID in response: ${add.body.slice(0, 200)}` };

      ctx.cleanup.push(async () => {
        await httpStatus({
          url: `${API.auth}/list`,
          method: 'POST',
          body: { operation: 'remove', ownerId: TEST_OWNER, device: 'obs-harness', itemUUID },
          wantStatuses: [200, 404],
        });
      });

      const del = await httpStatus({
        url: `${API.auth}/list`,
        method: 'POST',
        body: { operation: 'remove', ownerId: TEST_OWNER, device: 'obs-harness', itemUUID },
        wantStatuses: [200],
      });
      if (!del.ok) return { status: 'FAIL', mode: 'e2e', detail: `remove ${del.status}: ${del.body.slice(0, 120)}` };
      return { status: 'PASS', mode: 'e2e', detail: `add ${add.status}, remove ${del.status} (uuid ${String(itemUUID).slice(0, 8)}…)` };
    },
  },
];
