# Bought-date compatibility rollout — October 26, 2026

Status: deployed to the serving Kitchen and Thyme streaming backends at approximately 16:14 Pacific, after Matt explicitly requested deployment. Live readback confirms the reported avocado now returns six days old for iOS build 150 and build 205. No customer data backfill or date re-entry was necessary.

## Scope

The existing authorized Kitchen GET now projects explicit saved bought-date corrections for identified iOS clients through build 205. `amount_operations.annotate` supplies `acquired_on` only from the item's `__acquired_on` override; selection is limited to those already-authorized response rows. There are no additional database queries or writes. The original check-in timestamp is preserved in storage and exposed as `checked_in_at`; only the legacy response's age timestamp changes. Completed items only, valid nonfuture civil dates only, valid client time zone required. Pending captures, unedited items, archived rows, errors and other routes are untouched. Null/unknown dates still require the native update for honest unknown-age UI.

Builds 206 and later, and clients sending `X-Trepo-Kitchen-Age: bought-date-v1`, receive canonical timestamps. The iOS release owner was notified that the next build must include native fix `7e0762d`; this deployment does not claim any new phone installation or App Store release.

The Thyme stream emits sanitized `tool_start` / `tool_end` refresh events after a verified successful bought-date edit on the app surface. Older apps otherwise miss those names because the household-brain wrapper converts raw tool events into progress events. No arguments, model drafts or private result text are added to these events.

The old app continues to label this as “days in the kitchen” and may group the item by the projected date. Separate purchase/check-in wording requires the native fix. This was disclosed before the approved test and rollout.

## Verification

- Nine Python test groups pass, including explicit correction selection, unchanged archived/unrelated records, build boundary, capability opt-out, invalid/future/unknown dates, pending captures, timezone/DST, methods and authorization responses.
- Three Node tests pass for successful-date refresh, failure/read suppression and exclusion of private arguments.
- The actual packaged Kitchen handler was executed with a stubbed data source: build 205 corrected; build 206 unchanged; authorization denial never reaches the data source.
- The actual Node package handler imported successfully. Only each entrypoint and its new helper changed relative to fresh serving ZIP captures; unrelated production code was preserved.
- Authenticated public API reads of Matt's existing avocado changed from zero to six days for builds 150 and 205. Canonical and capability-aware reads retain `_createdDate=2026-10-26T22:17:54`, with `acquired_on=2026-10-20`. An independent database read confirms the stored check-in timestamp is unchanged. All other item response fields were compared and preserved.
- Both deployed code hashes and successful update status were read back. Environment, role, runtime, handler, layers, network, memory, timeout and architecture were unchanged.
- The unchanged App Store 1.21/build-150 product source passed the single Release simulator journey against the final deployed packages: **1 test, 0 failures, 114.933 seconds**. Real Thyme edit, immediate six-day display, tab-return six-day display and cold-relaunch six-day display all passed. Run `20261026-161423-5f6bf4`; result `/Users/matthewtaylor/Projects/Trepo/app-pipeline/tests/runs/20261026-161423-5f6bf4/results/personas-20261026-161427-90565.xcresult`. The runner released its lock. Exact synthetic-item cleanup passed, with an independent database absence check. Production compatibility remains enabled.

## Deployment and rollback evidence

| Serving function | Deployed SHA-256 (base64) | Pre-deploy SHA-256 |
|---|---|---|
| `trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k` | `vdiHG1q0c36xh8tmNNhSJqBcn2tP/icVq4tnPyw/uVY=` | `QlTAYyMlnMEmUHXfI/Ei+dR4vKquvAlqIdNUH2u1sfE=` |
| `trepo-quick-ack-stream-dev` | `eoHP8WyU1QG0dPsZmMmJMBjuQGkuKRXdN76zY4AX6e8=` | `BecBNWx22k15BoiqVdl48clnmyjwETuoW9J3mzVYIKM=` |

Fresh original packages, candidate manifests, revision-guarded deployment receipts, private configuration captures and live verification receipts are under `/Users/MattTaylor/Library/Caches/trepo-bought-date-rollout-20261026/`. Configuration files contain credentials and must not be committed or printed. `apply_rollout.py restore` in that private directory checks current code/configuration before restoring captured originals; it refuses to overwrite concurrent code changes. Production remains enabled after successful testing.

Context source: `apps/trepo/screens/kitchen-and-capture`. No new design or brand exception, and no shared-brain save was requested.
