# Private Thyme pilot: household data and response time

Status: implemented and verified in the private pilot, October 1, 2026 PDT (October 2 UTC). Author: Codex for Matt. Scope: the founder web pilot; existing iOS/customer Thyme functions were not redeployed.

## What was wrong

Matt's authenticated actor was correctly bound to `7d7df434…`, but the legacy kitchen reader used only that actor's rows. The iOS Kitchen API reads active kitchen rows belonging to every current household member. The pilot showed 15 instead of 60 items. The shopping reader returned 301 removed historical rows plus 4 current rows. These were real records from the correct account, but the wrong collection boundaries.

The reported dinner request took 41.173 seconds on Balanced/Sol and made seven model tool calls. The model repeatedly fetched data already available to the server; the page's generic status did not show the completed work.

## Changes

- Read the shared kitchen using current household membership and `action='IN'`, matching the native API's collection. Fail closed when the acting user is no longer a member. Preserve all matching rows; do not hide duplicate IDs or truncate a kitchen.
- Read the acting user's mirrored shopping list once, including `ADDED` and `CHECKED`, excluding removed history. Keep checked items available for unchecking. Verify `ADDED` as a successful uncheck state.
- Resolve proposed kitchen changes to an exact current household item before selecting its owning member for the legacy resolver; preserve the authenticated acting user. Missing/ambiguous IDs fail closed. An opt-in private-package guard prevents the legacy resolver falling back to a similarly named item if the selected ID vanishes.
- Fetch kitchen, shopping and preferences concurrently for every message. Supply a fresh compact snapshot to the model. Preserve quantities, zero, open state, dates and ingredients; missing amounts remain unknown. Above the size bound, explicitly mark the snapshot incomplete and require a tool read rather than silently truncate it.
- Keep the same default model. Independent provider status reads run concurrently; a temporary lack of turn ownership waits for evidence rather than executing an unowned tool.
- Show a compact live status and completed checks from actual server results. No invented stages, percentage or time-based activity claims. Stop polling once complete; prevent overlapping browser polls and stale responses after changing conversations.

## Evidence

| Check | Verified result |
|---|---|
| Direct production SQL, read only | Correct `7d7df434…` actor; pilot IDs exactly match native household selection: 60 kitchen / 4 shopping |
| Automated regressions | 69/69 pass: household membership, missing/ambiguous item IDs, actor retention, current list states, fresh snapshots between turns, unknown/zero amounts, failed reads, turn ownership, recovery, dietary changes, idempotency and existing recipe/proposal tests |
| Real model qualification with synthetic data | 8/8 turns pass across Balanced/Sol and Quick/Luna: grounded recipe, targeted ingredient edit, clarification with unchanged recipe, pending shopping proposal; zero account writes |
| Deployed AWS canary | HMAC tamper rejected 401; unenrolled and scope override rejected 403; read-only principal cannot approve; duplicate message returns same conversation; correct 60/4 context; answer completed |
| Same dinner prompt, same Balanced model | Original 41.173s / 7 tool calls; new live browser runs 20.320s and 19.3s, with one recipe-creation tool call in the first measured run |
| Simpler actual-data answers | Quick/Luna canary 7.996s worker / 9.510s including API polling; Balanced shopping-list follow-up 6.883s |
| Desktop/mobile | Real checked-item progress captured at 1280px and 390px; mobile document width 390px, no horizontal overflow; zero browser console errors |
| Reopening | Reload and reopen the saved conversation: current four-item shopping answer and recipe retained |
| Frontend | TypeScript and production build pass |

Small-sample timings establish an improvement, not a guaranteed latency percentile. Complete recipes still take roughly 20 seconds in these live tests. These fixes do not claim instant answers or production-wide rollout.

No customer kitchen, shopping, preferences or saved recipes were modified for qualification. Private conversation cards and test conversations were created. Actual customer mutation approvals were deliberately not exercised; household mutation routing was tested with isolated gateway fixtures, not a live inventory write. The existing legacy kitchen-edit adapter's retry-journal configuration still needs separate end-to-end qualification before claiming those writes work in this pilot (its legacy journal expects a different key schema from the pilot store).

## Deployment

- Private backend archive SHA-256: `bde5bd25ebbf6157c6f739bfb589f0889512da9a7ae766c1e1494ceb1cd1ae60`.
- Updated only `trepo-thyme-agent-v2-pilot-api` and `trepo-thyme-agent-v2-pilot-worker` using revision fences.
- Frontend source: `9ae4fe9ef73f224b4d95d0926015470a62738d2d` (pushed).
- Sites deployment: `appgdep_6abf4e86f4b48191b2f78bdbaf4b137d`, succeeded at `2026-10-02T06:26:22.655498Z`. Owner-private access preserved.
- URL: https://trepo-thyme-pilot.trepo-2057.chatgpt.site/

Refresh the page and start a new conversation for the updated model instructions. Existing conversations retain their original provider instructions, though their tools now read the corrected household collections.

Evidence folder: `/Users/MattTaylor/Documents/Codex/2026-10-01/thyme-agent-pilot/evidence/`. Relevant files: `native-inventory-comparison.json`, `live-data-regression.log`, `speed-qualification.json`, `speed-cloud-check.log`, `live-speed-comparison.json`, `progress-ui-build.log`, `live-progress.png`, `mobile-live-progress.png`.

Workflow: gstack investigation (trace root cause, narrow corrections, regression and functional verification). Company/brand evidence: `brand/trepo-brand-guidelines` and `brand/trepo-app-design-system`, source `trepo-company`. Existing cream/teal palette and quiet outlined presentation retained; no brand amendment. This is a local engineering record, not a shared gbrain save.
