// flows/receipt_dish_discard.mjs — grocery/dish/discard analyze failure signals.
// Synthetic-only for now: inject an `analysis_failed` marker into each Analyze*
// log group and assert the matching Trepo/Capture datapoint (+ forwarded Dynamo
// row if the group is subscribed to the forwarder).
import { LOG_GROUPS, CAPTURE_NS } from '../lib/config.mjs';
import { syntheticSignal } from '../lib/synthetic.mjs';

export const name = 'receipt_dish_discard';

function mk(service, logGroup, metricName) {
  return {
    label: `${service} analysis_failed -> ${metricName}`,
    mode: 'synthetic',
    async run(ctx) {
      const marker = {
        evt: 'analysis_failed',
        service,
        op: 'analyze_on_upload',
        code: 'obs_harness_synthetic',
        error: `obs-harness synthetic ${service} analysis_failed (run ${ctx.runId})`,
      };
      return syntheticSignal({ logGroup, marker, metricName, service, namespace: CAPTURE_NS });
    },
  };
}

export const tests = [
  mk('grocery', LOG_GROUPS.grocery, 'GroceryAnalysisFailed'),
  mk('dish', LOG_GROUPS.dish, 'DishAnalysisFailed'),
  mk('discard', LOG_GROUPS.discard, 'DiscardAnalysisFailed'),
];
