# Thyme household food and planning access — October 18, 2026

Author: Codex, requested by Matt. Status: deployed and verified on live legacy and native Thyme.

## Result and boundaries

Old and new Thyme now read the saved recipes of every current household member, with who saved each recipe, ingredients, instructions and source details. The collection no longer stops at the first API page. Current household shopping items are reconciled before filtering and limits. Automatic meal plans are household-scoped; hand-planned calendar entries and generated UWIH suggestions already used shared APIs and were verified. Current membership is checked from the database, not a model-supplied or cached member list.

Shared food reads do not broaden personal health, dish logs, private chats, or personal recipe edit/delete permissions. Recipe edits/deletes remain owner-specific; reading, recommending, copying, cooking, or scheduling a partner recipe is supported. Separate saved copies retain their own IDs and authors, even when they came from the same URL. Ambiguous titles still require disambiguation.

## Root causes and changes

1. Legacy Thyme called the personal saved-recipes API without pagination. New Thyme directly selected only the acting user's saved recipe rows. Both now use the current household collection. Limited results include a complete compact catalog; detail resolution and calendar snapshots search the complete collection. Native tool results page that collection and preserve attribution.
2. Legacy shopping reads included removed history. Household reads now choose the latest state of each shared item before excluding removed rows. Checked items remain marked bought. Equal-time removals win; otherwise equal replicas prefer the acting member's ID so existing write selectors retain their expected IDs.
3. Meal plans were mirrored but failed with “Food preferences changed during generation.” The generator incorrectly used the general brain activity revision/cache TTL as its publication fence. The existing recipe-generation dependency guard is now reused: initial memory freshness, current membership, actual kitchen stock, dietary restrictions, explicit brain controls and a bounded generation lifetime remain checked. Ordinary activity and cache TTL expiration alone no longer cancel the job. The general conversational memory fence was not weakened. Failed meal plans and incomplete imports are omitted from the automatic recommendation shortlist.
4. The final native canary exposed a separate existing bug: every `S#...` DynamoDB record update triggered chat-history invalidation, including the native assistant's own progress, provider/session bookkeeping and recipe proposals. These updates repeatedly invalidated the answer's memory fence. The brain stream consumer now considers only actual user-message content/retention and session identity/retention. Real edits, deletes, role changes, and retention changes still retract evidence. Missing old images are treated conservatively. Native streams were verified to provide both old and new images.

## Verification

- Legacy suite: 120 passed, 3 failed. The same three failures were reproduced on the original serving baseline in the prior kitchen fix: multi-recipe confirmation and two item-edit tests missing the newer request journal. No new failing tests.
- Native suite: 120 passed, zero failures, including full household pagination, older partner recipe details and mutation boundaries.
- Household food regressions: 10 passing; seven failed before the scope fix.
- Real MySQL statement-local fixtures: 21 assertions passed across recipes, shopping and meal plans. Includes mixed collations, foreign/solo/empty households, missing/stale membership and member departure. No customer tables or rows changed by these SQL fixtures.
- Meal-plan guard: six regressions passed; real database capture/publication also passed. The old fence rejected simulated changed revision/expired tokens while unchanged actual dependencies remained publishable.
- Brain stream: four failures reproduced on the exact live module; all 10 regressions pass after filtering bookkeeping. Actual user corrections/deletions and expiry remain tested.
- Authenticated live legacy streaming requests from both Matt and Anna correctly recommended actual saved recipes from both authors and listed the same three current shopping items and stores.
- Live native canary initially reproduced the self-invalidating progress bug twice. After the stream fix it completed in 17.3 seconds with correct titles from both authors, the exact three-item shopping list, no error and no mutation proposals. Anna’s native pilot enrollment was unchanged; her live checks used legacy Thyme, while Matt’s native session read both members’ recipes.
- Both accounts see 127 saved records: 57 saved by Matt, 70 by Anna. Not every historical import is complete; this is the collection count, not a claim that all imports are cookable. Detailed lookup of complete partner recipes passed.
- Both accounts see the same nine calendar entries for September 1–October 31; canonical stored IDs matched exactly. The calendar API bounds date ranges; tests respect its range rather than treating a clipped response as missing data.
- Generated UWIH results matched across both accounts: 10 kitchen-only and 10 grocery-needed suggestions at audit time. API freshness/version behavior was preserved.
- After the generator fix, the existing failed household meal plan regenerated successfully in about 12 seconds. Both accounts read the same ready 12-slot plan with ingredients and steps. Existing focus and image settings were preserved.

No iOS code changed; authenticated server paths were exercised, not a simulator/visual walkthrough. The legacy tests omitted session IDs. Native verification creates dedicated QA conversations; it does not edit kitchen items, shopping items or saved recipes.

## Deployment and recovery

This is an overlay release over exact captured live ZIPs, not a full deployment of this older repository checkout. `scripts/prepare_household_food_overlay.py` only replaces verified matching function bodies in service-specific modules. `scripts/deploy_household_food_overlay.py` checks live revision and code hash, verifies the exact changed ZIP-member set and preserves every other member and service configuration. Native and legacy service-specific code remains intact. The brain stream change replaces one module only.

Private deployment manifests, prior ZIPs, configuration, test results and customer-specific canary evidence are stored in `/Users/MattTaylor/Library/Caches/trepo-thyme-household-data-20261018`. Credentials and customer payloads are excluded from this repository. Use the recorded original ZIP only after checking that a rollback would not overwrite a newer deployment.

Shared context consulted: `apps/trepo/sources/20260923/ios/trepo-v0/technical-overview` (historical per-user replication design). Live deployed modules, current membership and canonical tables supplied current operational evidence. No shared Gbrain page was written for this task.

## Deployed package checksums

| Service | Code SHA-256 (base64) |
| --- | --- |
| `trepo-grocery-backend-dev-MealPlanGeneratorFunctio-4rNstnlDTS1Z` | `1c04WOSVs5sfphnycBneFfQo8G/3W+StgJ3KVksjC9M=` |
| `trepo-household-brain-v2-StreamConsumer-tcofFrGVflHI` | `GSn4JmCfC7aEA0z4MtP23Hx8J2zGt+WWzB8biVWBars=` |
| `trepo-quick-ack-async-worker-dev` | `moR4mbOQGvuJ9GTjJJOU1yNxaVBtlMPVN2zn/Dg3sbo=` |
| `trepo-quick-ack-sam-dev` | `6y6bBGOLzrWD/wxe8xtAuALBLV18PAppQAwACME5w+E=` |
| `trepo-quick-ack-stream-dev` | `VTgSvoo88OiDF3E1Sd3rfIDpLuewbmPQ0IAQKaYLHAI=` |
| `trepo-thyme-agent-v2-pilot-api` | `pOhuOtepbZ3k6xfwZkuJV3SK1Ag9emYH5Ysok4KkoPs=` |
| `trepo-thyme-agent-v2-pilot-native` | `Kka9RxyIHzTDydrxyexwGEBLC8cTkG1lstErmKRUZ+0=` |
| `trepo-thyme-agent-v2-pilot-worker` | `9nVBtcIJmsVk/qLNJsgfwh+pOombTcoIHGn84qi3RAs=` |
