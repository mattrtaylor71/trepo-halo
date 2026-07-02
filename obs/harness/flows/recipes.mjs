// flows/recipes.mjs — saved recipes API.
import { API, TEST_OWNER, LOG_GROUPS, CAPTURE_NS } from '../lib/config.mjs';
import { httpStatus } from '../lib/signals.mjs';
import { syntheticSignal } from '../lib/synthetic.mjs';

export const name = 'recipes';

export const tests = [
  {
    label: 'GET /saved-recipes/{owner} -> 200',
    mode: 'e2e',
    async run() {
      const res = await httpStatus({ url: `${API.main}/saved-recipes/${TEST_OWNER}`, method: 'GET', wantStatuses: [200] });
      return { status: res.ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: res.detail + (res.ok ? '' : ` body=${res.body.slice(0, 120)}`) };
    },
  },
  {
    label: 'synthetic backend_error -> RecipesBackendError',
    mode: 'synthetic',
    async run(ctx) {
      const marker = {
        evt: 'backend_error',
        service: 'recipes',
        op: 'saved_recipes_api',
        error: `obs-harness synthetic recipes backend_error (run ${ctx.runId})`,
      };
      return syntheticSignal({ logGroup: LOG_GROUPS.savedRecipes, marker, metricName: 'RecipesBackendError', service: 'recipes', namespace: CAPTURE_NS });
    },
  },
];
