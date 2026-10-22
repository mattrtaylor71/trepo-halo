# Thyme saved-action confirmations — October 22, 2026

Author: Codex. Requested by Matt Taylor. Status: implemented and tested; rollout receipt follows after verification.

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
- Broader serving-runtime suite: 131 passed and three preexisting fixture failures before the final additional shopping regression. The same three failures reproduced against the untouched live package (multi-recipe mock lacks durable IDs; two kitchen-edit fixtures lack request-journal setup). No new baseline-suite failures.
- No phone UI was visually checked. Live app-serving validation and deployment results are recorded below when completed.

## Release

`scripts/deploy_thyme_confirmation_overlay.py` updates exactly two ZIP members per function: `family_brain/api.py` and `household_brain/store.py` on HomeSuggestions; `device-assistant.mjs` and new `brain-turn.mjs` on legacy stream, sync, and async worker. It checks exact live revision/code hashes and preserves all other package entries and configuration. The existing service-specific leftover-routing differences remain intact. No schema, grants, firmware, native-agent rollout, or customer-data changes.

Private source ZIPs, configs, test outputs and operational receipts: `/Users/MattTaylor/Documents/Codex/2026-10-22/thyme-log-investigation`. Never commit these private captures. The previous source baseline is recorded separately in commit `2e27ea6`; the HomeSuggestions October 22 changes were preserved.

Shared context consulted: `apps/trepo`, `apps/trepo/screens/cook-and-dish-log`; customer copy checked against `brand/trepo-brand-guidelines` with no approved exception required. Operational findings come from actual deployed code, live trigger metadata and app API reads, not historical shared-memory summaries. No company-memory save. The Antonio monitor remains paused.
