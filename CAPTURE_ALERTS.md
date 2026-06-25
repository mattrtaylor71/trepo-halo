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

## Coverage & the one gap
- **Bulk**: hard + soft (job marked failed) ✅ fully covered.
- **Receipt / Leftovers / Discard**: **hard failures covered** (Lambda Errors). **Soft failures
  NOT yet covered** — these S3-triggered analyze functions catch errors and return 200 (e.g. the
  dish `imageUrl` TDZ bug returned 200 while the preliminary write silently failed). They have no
  single clean failure-marker today, so a low-noise log filter isn't possible without a code
  change.
- **To close the gap** (recommended follow-up): add a structured `{"evt":"analysis_failed",
  "kind":"dish|grocery|receipt|discard",...}` log line in each analyze function's main
  failure/catch branch (same pattern as `list_fanout_miss`), then one metric filter
  `"analysis_failed"` across those log groups + an alarm. Small, testable, but touches 3 live
  capture functions → deploy via surgical update-function-code with the usual verification.

## Action required
Confirm the SNS email subscription — AWS emailed a link to matt@trepo.ai (status
PendingConfirmation until clicked).
