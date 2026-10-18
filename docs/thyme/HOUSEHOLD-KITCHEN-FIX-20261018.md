# Thyme household inventory fix — October 18, 2026

Status: deployed; verified against live inventory and live app streaming requests.
Author: Codex, requested by Matt. No customer inventory was edited during verification.

## Symptom and root cause

Anna asked which condiments were in her household kitchen. Legacy Thyme loaded only rows whose `shared_kitchen.owner_id` matched Anna's user UUID, then instructed the model that this was the complete kitchen. It omitted items checked in by Matt. Transcription and household membership were correct.

The original incident loaded 12 of 56 active household items. At verification on October 18 the household had 48 active items; counts changed with ordinary inventory use. Both members now read the same 48 items, exactly matching the canonical household query. A live request immediately before deployment reproduced Anna's false “no obvious condiments” answer. The same request after deployment returned actual mustard, ketchup, soy sauce, hot sauces, and dressings.

## Change

- `household-kitchen.mjs` resolves current database membership and active inventory in one read-only SQL statement. It ignores stale client/session member lists, rejects missing/fallback/changed membership, isolates solo accounts, and distinguishes an empty kitchen from an unavailable read.
- `data-access.mjs` uses that reader for full/limited inventory, overview counts, household summary counts, item-ID reads, and name searches. Limits apply after household collection. Separate physical items with the same name remain distinct. Mutation selectors remain unchanged.
- `device-assistant.mjs` grounds answers in current recorded household stock, gives that data precedence over older wrong answers, and explicitly marks failed reads unavailable. It does not infer absent stock from a failed read.
- SQL explicitly bridges the different collations of `new_users` and `shared_kitchen`. Production EXPLAIN uses the actor primary key, membership indexes, and kitchen owner index; it does not scan all customer inventory.

Newer Thyme already had the correct household read. Its deployed gateway was independently read and exercised for both members: 48 items and mustard visible for each. No native gateway change was required. Historical shared context: `apps/trepo/sources/20260923/ios/trepo-v0/technical-overview` describes the older per-user replication design; current deployed shared-table settings and SQL were authoritative for this incident.

## Verification

- New read regressions: six failures on the unmodified serving code; all nine pass after the fix.
- Grounding regressions: three pass, including unavailable vs. empty and items in the compact tail.
- Nine real-MySQL assertions with statement-local JSON fixtures: no table or customer-data writes. Covers mixed collations, foreign/removed items, solo and empty kitchens, stale/missing membership, current membership changes, and duplicate names.
- Entire existing legacy suite plus new tests: **110 passed, 3 failed**. The exact same three failures occur on the unchanged live baseline (**98 passed, 3 failed**). These are existing test/runtime mismatches around multi-recipe confirmation and two item-edit tests that omit the newer request journal; no new suite failures.
- Two bounded model replays with the production prompt and current inventory passed, including correcting Anna's old answer in conversation context.
- Live streaming Lambda requests authenticated as Matt and Anna both returned the actual condiments, with no error events. Logs confirm 48 items loaded and complete answers with no actions. The tests omitted session IDs so they did not append to the users' chat sessions.
- Live synchronous request and worker startup check are recorded in the private verification evidence.

Focused suite output:

```text
Tests 12 | Passed 12 | Failed 0 | Skipped 0
```

No iOS binary or simulator test was needed for this server data-scope change. The app's streaming backend was exercised directly; no claim is made about a visual device walkthrough.

## Deployment and recovery

Three changed ZIP members per service: `data-access.mjs`, `device-assistant.mjs`, and the new `household-kitchen.mjs`. Every other member and function configuration was preserved and verified. The stream service's independent recent changes were not copied over the synchronous/worker services; only the identical grounding function was replaced there.

| Service | New code SHA-256 (base64) |
| --- | --- |
| `trepo-quick-ack-stream-dev` | `zcrpPxHwI7KMLMbxQzLhuGPDU7jkUWYxrwgro8jEAkc=` |
| `trepo-quick-ack-sam-dev` | `Tql0b3u+qdJS1iB0+Gfo+XrCOATuoCiAOzzInyxW//k=` |
| `trepo-quick-ack-async-worker-dev` | `Z6IeQwESxQE8AyDu0LMc3giz8qPC3gjw4tV83kOo54Y=` |

The deployment script guards both revision and original code hash before writing. Original ZIPs, exact runtime/dependencies, configs, test output, and live response/log evidence remain private at `/Users/MattTaylor/Library/Caches/trepo-thyme-household-fix-20261018`. Configs and user payloads are not committed. The same archives allow a guarded rollback after rechecking that no later deployment would be overwritten.

The separate baseline commit records the two current serving modules before the fix. This worktree is an overlay release, not a replacement for the full live service bundle. Run its tests against the captured deployed runtime; do not deploy the older repository's full package or SAM staging directory.
