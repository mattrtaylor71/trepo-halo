# Capture / Upload Failure Alerts

CloudWatch alarms that page **matt@trepo.ai** (SNS topic `trepo-capture-alerts`) when an
upload or its analysis fails. Created 2026-06-24. matt-cli was granted the needed actions via
managed policy `TrepoListSyncMonitoring` (v2 broadened `logs:PutMetricFilter` to
`/aws/lambda/*`).

## Pipeline → function map
| Upload type | Entry/analysis Lambda |
|---|---|
| Bulk grocery check-in | `grocery-identifier-dev-identify-async` (async identify) → `AnalyzeOnUpload` → `grocery-identifier-dev-bulk-commit` (commit) |
| Receipt | `trepo-grocery-backend-dev-AnalyzeOnUpload` (shares grocery analysis, receipt path via quickIdentify/geminiReceipt) |
| Leftovers / dishes | `trepo-grocery-backend-dev-AnalyzeDishOnUpload` |
| Discards | `trepo-grocery-backend-dev-AnalyzeDiscardOnUpload` |

## Alarms (all → SNS trepo-capture-alerts, Sum≥1 / 5min, missing=notBreaching)
**Hard failures** — Lambda `Errors` metric (unhandled throw / timeout / OOM on the async invoke):
- `trepo-capture-errors-grocery-bulk-and-receipt-analysis` → AnalyzeOnUpload
- `trepo-capture-errors-leftovers-dish-analysis` → AnalyzeDishOnUpload
- `trepo-capture-errors-discard-analysis` → AnalyzeDiscardOnUpload
- `trepo-capture-errors-bulk-async-identify` → identify-async
- `trepo-capture-errors-bulk-commit` → bulk-commit

**Soft failures** — log metric filters on the terminal failure marker `"Background processing
error"` (logged only in the `.catch` that also marks the job `failed`):
- `trepo-capture-bulk-identify-job-failed` ← filter `bulk-identify-analysis-failed` on identify-async → `Trepo/Capture:BulkIdentifyAnalysisFailed`
- `trepo-capture-bulk-commit-job-failed` ← filter `bulk-commit-failed` on bulk-commit → `Trepo/Capture:BulkCommitFailed`

## Soft-failure alarms (gap now CLOSED 2026-06-24)
The 3 S3 analyze functions now emit a structured `{"evt":"analysis_failed","kind":...,"stage":...}`
log in their **non-throwing** failure paths (the swallowed catches that return 200 — e.g. the dish
`imageUrl` TDZ bug, which failed the preliminary write silently). Helper `reportAnalysisFailed()`
added to each; instrumented at the preliminary/primary result-write swallowed catch + the
`owner_missing` non-throwing FAILED (+ discard shopping-list add). The noisy `fast_status:FAILED`
paths are intentionally NOT instrumented (deep analysis recovers from those).

Metric filters `"analysis_failed"` → metrics + alarms:
- `trepo-capture-dish-analysis-soft-failed` ← AnalyzeDishOnUpload → `Trepo/Capture:DishAnalysisFailed`
- `trepo-capture-grocery-receipt-analysis-soft-failed` ← AnalyzeOnUpload → `GroceryAnalysisFailed`
- `trepo-capture-discard-analysis-soft-failed` ← AnalyzeDiscardOnUpload → `DiscardAnalysisFailed`

Verified live: triggered the dish owner_missing path → marker logged → `DishAnalysisFailed`=1.0.

## Coverage summary
- **Bulk**: hard (Lambda Errors) + soft (job marked failed) ✅
- **Receipt / Leftovers / Discard**: hard (Lambda Errors) + soft (`analysis_failed` on the
  swallowed result-loss paths) ✅. Note: only the *instrumented* swallowed catches are covered;
  if a brand-new silent-failure path is added later, instrument it the same way.

## Action required
Confirm the SNS email subscription — AWS emailed a link to matt@trepo.ai (status
PendingConfirmation until clicked).

## Coverage expansion (2026-07-13) — closed audit gaps
New metric filters + alarms → SNS trepo-capture-alerts (artifact 72b40b1e). All threshold-1 unless noted:
- `VoiceShoppingListWriteMiss` / `VoiceDiscardWriteMiss` ← `{$.evt="shopping_/discard_peruser_write_miss"}` on
  trepo-quick-ack{,-stream-dev,-async-worker-dev}. Voice item invisible to user (per-user table miss).
- `BulkGeminiFailover` ← `{$.evt="bulk_gemini_failover"}` on identify-async. **thr=5/5min** (degradation, recovers via OpenAI).
- `IdentifyJobRedrive` ← `"Re-drove job"` on identify-async. **thr=3/5min** (provider struggling early-warning).
- `RecipeBatchPartialFailure` ← `{$.evt="recipe_batch_partial_failure"}` on SavedRecipesApi. **thr=3/5min** (some photos in a batch failed).
- Instrumented (roll into existing KitchenBackendError / RecipesBackendError): shared_kitchen insert/update/delete
  exceptions; recipe-link refine-unreadable (save_recipe_refine); saved-recipes unhandled-500.
NOTE: bulk-identify-failed metric filter greps literal `"failed:"` — do NOT change that log substring without
updating the filter (it silently broke 2026-07-13 when the re-drive edit changed it). Prefer structured `{$.evt=…}`.
Removed: push-notification-failed alarm (dead-token 404s = benign baseline noise; real notify outage → 5xx-notifications).
