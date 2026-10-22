# Thyme saved-action confirmations — October 22, 2026

Author: Codex. Requested by Matt Taylor. Status: deployed and verified at 03:38 Pacific on October 22 (10:38 UTC).

A successful dish insert advanced the household brain activity revision. Thyme's final full-context validation then rejected its own successful save, replaced the confirmation with a misleading food-preferences warning, and removed the result UI. The same guard could block a second action. Internal `ai_op` success was emitted before that replacement.

The screenshot's dish was already present and complete in the actual serving API. No customer records, original media, or historical conversations were backfilled or rewritten. Repeating those inserts would risk duplicates.

## Change

- Preserve full context freshness before every tool and before publishing general model text. Do not globally ignore activity revisions or disable invalidation triggers.
- Add authenticated receipt-scope validation: current actor/household, complete membership, dietary restrictions, household generation/enabled state, subject control revisions, and the original bounded turn lifetime. Ordinary activity revision and optional-context cache expiry alone do not erase a historical committed receipt. Receipt validation does not rebuild optional context.
- When full publication validation fails, build confirmation text and result UI only from typed durable action receipts. Read-tool output, old model prose, and requested quick-item names cannot supply the confirmation. A receipt describes what committed, not a guarantee that nobody subsequently edited/deleted the row.
- Before a later tool, a stale context causes the pending model batch to stop. An authorized refresh discards old model/history/read-tool text and rebuilds current grounding. The next model turn sees the original request and attempted-action receipts. A request-local canonical argument cache prevents reexecuting an identical attempted write, including uncertain timeouts; existing domain journals remain in effect. This is not semantic deduplication across differently worded customer requests.
- Async tickets retain their original signed expiry and scope. Refresh cannot mint or extend the ticket. Full answer/history validation remains strict.
- Publish one reconciled final text for buffered app streaming. Emit the final `ai_op` outcome only after publication validation. Failures report inability to verify, not an invented change in food preferences. If remaining actions cannot continue, keep a partial receipt rather than inviting a blind replay.

## Validation

- Exact deployed baseline reproduced the screenshot. Ten new behavior tests fail against it.
- Twelve full assistant-loop regressions pass on the candidate: sync/stream single and multiple dishes, stale-context replan, duplicate-attempt suppression, real scope revocation, validation outage, uncertain tool timeout, strict general-answer fence, and shopping receipts.
- Twenty-four Python scope/API regressions pass, including actor binding, controls, expiry, no context rebuild for validation, async tickets, and refresh lifetime.
- Eight real MySQL 8.4 fixture tests pass using isolated localhost databases and the actual dish invalidation trigger. Own inserts preserve receipt scope while invalidating old context; deletes remain deleted. Allergy, membership, pause/reset and preference changes revoke the scope.
- Final broader serving-runtime suite: 132 passed and three preexisting fixture failures. The same three failures reproduced against the untouched live package (multi-recipe mock lacks durable IDs; two kitchen-edit fixtures lack request-journal setup). No new baseline-suite failures.
- The twelve assistant-loop regressions also passed on the exact prepared sync and async worker packages, preserving their existing routing differences. Saved-dish tests exercise the real app-output builder and verify the confirmation and action card.
- Thirteen authenticated live contract checks passed: full freshness, stale-activity receipt scope, true dietary/member/control/actor/expiry rejection, bounded refresh, and async receipt/refresh behavior. Test fences were changed in memory; no household data was changed.
- A real signed founder app request completed in 5.81 seconds with `answer=complete`, `actions=none`, one `get_recent_dishes` action, a result card, and the final publication marker. The returned card ID/time matched the actual dishes API; its null-name fallback was verified against serving code. The `ai_op` marker described the actual returned answer. This was a read-only production smoke test, not a production meal-insert test or a visual phone check.

## Release

`scripts/deploy_thyme_confirmation_overlay.py` updates exactly two ZIP members per function: `family_brain/api.py` and `household_brain/store.py` on HomeSuggestions; `device-assistant.mjs` and new `brain-turn.mjs` on legacy stream, sync, and async worker. It checks exact live revision/code hashes and preserves all other package entries and configuration. The existing service-specific leftover-routing differences remain intact. No schema, grants, firmware, native-agent rollout, or customer-data changes.

Private source ZIPs, configs, test outputs and operational receipts: `/Users/MattTaylor/Documents/Codex/2026-10-22/thyme-log-investigation`. Never commit these private captures. The previous source baseline is recorded separately in commit `2e27ea6`; the HomeSuggestions October 22 changes were preserved.

Shared context consulted: `apps/trepo`, `apps/trepo/screens/cook-and-dish-log`; customer copy checked against `brand/trepo-brand-guidelines` with no approved exception required. Operational findings come from actual deployed code, live trigger metadata and app API reads, not historical shared-memory summaries. No company-memory save. The Antonio monitor remains paused.

## Verified deployed packages

| Function | SHA-256 (base64) |
| --- | --- |
| `trepo-grocery-backend-dev-HomeSuggestionsFunction-eq77AzDop23s` | `I7cuANBzOOtRCkrTeQKuez9mVVIAoRTAVbL9Ogxj01s=` |
| `trepo-quick-ack-stream-dev` | `1++a9Y4OyiTpQ1bgXiQsB99LtPYpS4V64yzwWnejsc4=` |
| `trepo-quick-ack-sam-dev` | `dasN10wK39mssa38GVmgOSyRlWdyP1Hh8m0+ZUm21rw=` |
| `trepo-quick-ack-async-worker-dev` | `dFK+O5GYGr3YtQM/mZOfn6aIkcDITCiZ0F2Jx0SV++0=` |

Implementation commit: `c4c56d0`, branch `codex/thyme-write-confirmation-20261022`, pushed to origin. This is a scoped live ZIP overlay, not a merge or wholesale deploy of the repository. Rollback must compare the current live revision/hash first so it cannot overwrite a later release.

## Live write testing and follow-up repairs (October 22, 03:45–04:00 Pacific)

Matt requested real testing after the initial release. Created one isolated, explicitly synthetic QA user/household with no email, phone number, registered device or push token. Authenticated requests used the actual deployed stream, sync, async-ingest and async-worker functions. Saved rows were checked through the same `/dishes/{user}` API as the app. These checks do not claim a visual phone test or independently verified nutritional estimates.

Live scenarios:

- Oatmeal: one completed record, populated macros, confirmation and card; the save advanced the household input revision.
- Two separate dishes (banana and coffee): exactly two completed records and two successful actions. This uncovered a preexisting card builder that returned only the last tool's card. The targeted repair now returns one card per distinct saved dish for multiple-dish acknowledgments, preserving existing card formats. A subsequent egg-and-orange request returned both cards; IDs, names and macros matched the serving API.
- The MaryRuth's liquid supplement wording: the ordinary synchronous path saved one completed record and returned its confirmation/card instead of the false preference warning.
- Generated spoken request for toast with peanut butter: the real audio ingest/transcription/background worker saved one completed dish and confirmed it. Its signed brain ticket remained valid through the write. Re-delivery returned the same terminal voice job, with one worker attempt and no extra dish.
- Retry testing exposed a second bug: response-cache and durable-operation admission keys included the changing brain input revision. Retrying even a modern `v1` operation could execute it again. This was reproduced only in the synthetic household.
- Short legacy retry caches now use the stable household/control scope. Modern operation admission uses a stable actor/operation key, with a separately stored authorization scope. A changed household or control scope hides the original receipt and prevents another execution; it does not select a new admission key. Client render acknowledgments also require the stored authorization scope.
- A fresh legacy request and immediate retry returned the same yogurt ID with no extra row. A fresh modern request and retry returned the same pear ID; the actual GET recovery route returned that receipt. Advancing only the synthetic household's control generation then returned `scope_changed`, no old card and no extra row.
- Existing operation IDs issued before the guarded cutover at **2026-10-22 10:58:45 UTC** cannot be admitted under the new namespace. A replay of the pre-upgrade canary returned `pre_upgrade_request` and did not write. During the short rolling deployment interval, new modern IDs older than that cutover were likewise refused conservatively. Historical operation rows were not rewritten. Legacy clients retain their short replay window; this is not indefinite semantic deduplication of repeated meal descriptions.

Validation: the final targeted suite has **22 passing tests** (12 confirmation/fence tests, four card regressions, six admission/scope/replay tests). The earlier SQL and live receipt-scope checks remain applicable; they were not represented as new real-write tests. Production replay and recovery checks passed after the follow-up deployment.

`scripts/deploy_thyme_dish_cards_overlay.py` changed only `app-response.mjs` on stream/sync/async worker. `scripts/deploy_thyme_retry_overlay.py` changed `family-brain.mjs`, `operation-bridge.mjs`, and `operation-recovery.mjs` on stream/sync; the worker only needed `family-brain.mjs`, since its existing one-way voice-job claim already prevents repeated execution. Both scripts guarded exact previous hashes/revisions and preserved unrelated code/configuration. HomeSuggestions remained unchanged after the initial fix.

| Function | Final SHA-256 (base64) |
| --- | --- |
| `trepo-quick-ack-stream-dev` | `lafMiOF8gGcpmypL8BOz0m1UD5agZ8EVRldGUcfSvcY=` |
| `trepo-quick-ack-sam-dev` | `4WdnLOFYgIbvGdW/4VCd5UaYCutx5uBWvMDrDyirdk4=` |
| `trepo-quick-ack-async-worker-dev` | `WDSvtswO4Ol7jkj5R2Gn3Xn8bioRaib7Z2zwRtLJqvY=` |

Private evidence is under `thyme-log-investigation/live-write`: requests, app/API responses, failed-test reproductions, deployment receipts, control-scope test and guarded cleanup ledger. Original test audio and audit evidence are retained. No customer account was used for write testing, and no customer meals were changed.

Cleanup completed and read back: removed all 12 synthetic dish IDs from both shared and legacy serving tables, the single derived triage entry, and the QA signup. The serving API returned an empty dish list afterward. The first cleanup attempt rejected an incorrect household-versus-actor ownership assumption and rolled back without deleting anything; the subsequent exact-actor transaction succeeded. No schema changes or media deletion were needed. Final live ZIP readback matched all changed source modules and every function reported a successful deployment.
