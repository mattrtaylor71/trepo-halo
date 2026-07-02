# Trepo Observability Verification Harness

Proves that **every backend flow's failure/success signal is actually readable** —
not that an alarm is green, but that the underlying metric datapoint, log line, or
DynamoDB row can be observed. If a signal can't be read, an incident goes unseen.

## Setup

```bash
export AWS_PROFILE=trepo-dev AWS_REGION=us-east-1
cd /Users/MattTaylor/Desktop/trepov2/obs/harness
npm install          # @aws-sdk/client-cloudwatch, -cloudwatch-logs, -dynamodb, lib-dynamodb
```

## Running

```bash
node verify.mjs --flow <name>            # run one flow
node verify.mjs --all                    # run every flow
node verify.mjs --all --mode e2e         # only real end-to-end tests
node verify.mjs --all --mode synthetic   # only injected-marker tests
node verify.mjs --all --slow             # also run tests marked slow (junk-image bulk upload)
node verify.mjs --purge                  # delete harness_probe/api_error rows for TEST_OWNER
```

Every run prints a results matrix and writes `results-<iso>.json`.

## What the result labels mean

| Result             | Meaning |
|--------------------|---------|
| **PASS(e2e)**      | The *real* flow ran end-to-end (real API call / real failure) **and** its signal was read back (metric / log line / Dynamo row / HTTP status). Highest confidence. |
| **PASS(synthetic)**| A marker line was injected into the flow's log group and the corresponding metric-filter datapoint (and, if the group is subscribed to the analytics forwarder, the forwarded `TrepoAnalyticsEvents` row) was read back. Proves the *signal path* is wired, without needing to force a real failure. |
| **FAIL**           | The signal could not be read (API returned an unexpected status, or the metric/log/row never appeared within the timeout). |
| **SKIP**           | The signal doesn't exist yet — e.g. a metric filter a sibling work-package is still creating, or a `slow` test not requested. Not a failure. |

**Cardinal rule:** pollers assert on metrics / logs / rows — **never** on CloudWatch
alarm *state*. An alarm can read OK simply because nothing has happened.

## Flows

| Flow | e2e | synthetic |
|------|-----|-----------|
| `analytics_ingest` | POST `harness_probe` → 201 → Dynamo row; malformed `{}` → 400 | — |
| `auth` | malformed `send-code` → clean 4xx + `twilioAuth` log line (no SMS sent) | — |
| `kitchen` | GET items[]; add → verify present → delete; malformed POST → 4xx (5xx → ApiGateway 5xx metric) | — |
| `list` | add + delete an item via `POST /v1/list` (operation add/remove, by `itemUUID`) | — |
| `bulk` | GET `/health` → 200; GET bogus `/job/{id}` → 404. **slow:** junk image → job FAILED + `"Background processing error"` log + `BulkIdentifyAnalysisFailed` metric | — |
| `receipt_dish_discard` | — | inject `analysis_failed` into the three Analyze* groups → `Grocery/Dish/DiscardAnalysisFailed` metric (+ forwarded `backend#<svc>` row) |
| `voice` | POST `/voice-ack` minimal body → 2xx | inject `backend_error` → `VoiceBackendError` (SKIP until WP4 creates the filter) |
| `recipes` | GET `/saved-recipes/{owner}` → 200 | inject `backend_error` → `RecipesBackendError` (SKIP until WP3 creates the filter) |
| `notifications` | GET `/notifications/users` → 200 | inject `backend_error` → `NotificationsBackendError` (SKIP until filter exists) |

Synthetic tests **auto-detect** whether the target metric filter exists (via
`DescribeMetricFilters`) and whether the group is subscribed to the forwarder (via
`DescribeSubscriptionFilters`), so they adapt as sibling work-packages land their
filters — reporting SKIP (not FAIL) when a filter isn't there yet.

## Cleanup semantics

- **`test_run_id`** stamps harness-created analytics data; every created item carries
  the `obs-harness` device id / `obs-harness-item` name.
- **Per-flow cleanup:** `kitchen` and `list` register deferred cleanup that deletes any
  item they created (runs even if a later assertion fails).
- **`--purge`** deletes only `harness_probe` and `api_error` rows **for `TEST_OWNER`**
  from `TrepoAnalyticsEvents`. It never touches any other owner_id.
- Synthetic `backend_error` rows are written under `backend#<service>` owner ids by the
  forwarder (not `TEST_OWNER`); they are harmless test markers and are left in place.

## TEST_OWNER

`TEST_OWNER = 1f4db6b6-2f62-4558-aa40-c8e82527dc74` — the only owner_id permitted for
mutations. This is the known-safe deletion test user. Its household owner_id is `94041`,
but all of its analytics data is keyed by the UUID (confirmed: 214 rows under the UUID,
0 under `94041`), so the UUID is required for the analytics-row assertions. The user is
not currently in MySQL `new_users` (purged during deletion testing) — this does not
affect the harness, which only needs a stable, safe owner_id. See `lib/config.mjs`.
