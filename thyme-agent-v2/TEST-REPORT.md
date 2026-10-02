# Thyme private pilot — verification report

Date: October 1, 2026 (America/Los_Angeles). Status: private deployment; existing iOS/customer routes unchanged. Matt authorized the pilot and automated testing. Automated tests made no changes to real kitchen, shopping, saved-recipe or memory data.

## Results

| Layer | Evidence | Result |
|---|---|---|
| Core automation | 60 Node tests | Pass: signatures/replay, cross-account access, idempotency, concurrent work, stale approvals, interrupted recovery, uncertain write reconciliation, canonical recipe edits, quantity/slot/content verification, dietary checks, exact owned targets and daily limits |
| Real SQL fixture | 6 integration checks | Pass: exact current and legacy recipe saves, portions notes, duplicate replay, changed intent, household change and deleted-account rejection |
| Real managed models | 24 synthetic conversation turns | Pass: Sol, Astra and Luna; breakfast from stock, targeted ingredient replacement, method clarification without recipe changes, shopping proposal with zero writes |
| Cloud worker | AWS API → DynamoDB stream → managed agent → Trepo | Passed read-only canary using Matt's real 15 kitchen items and 305 shopping entries. Final deployed-artifact canary: 21.603 seconds total, 20.885 seconds worker metric. No comparison with old Thyme claimed. |
| Browser UI | Private deployed account connection | Passed: authenticated Matt connection, actual kitchen count and persistent history recovered after full-page reload |
| Live recipe behavior | Private browser and actual kitchen | First run found chat-only recipe output and excessive missing core ingredients; added stronger inventory constraint and canonical-text safeguard. Retest produced a two-serving card using actual listed halloumi, peas, cabbage and garlic, marked amounts unknown and pantry staples optional. Follow-up explanation left the entire rendered recipe unchanged. |
| Live approval preparation | Matt’s exact current recipe | Passed after date/recovery fixes: one pending card, exact canonical title/ingredients/steps and Serves 2, version unchanged. No approval clicked and no account mutation applied. |
| Responsive UI | Local synthetic desktop/390px checks | Passed: data panel, recipe disclosure, approve/reject cards and feedback. Synthetic approval affects fixture data only. No console warnings/errors in the checked page after dependency reload. |
| Production frontend | Build, typecheck and route checks | Passed. Development-only /preview returns 404; unauthenticated API returns 401. Client artifact scanned: no bridge secret or provider key. |

## Model sample

Initial identical four-turn run (seconds; not a large statistical benchmark):

| Model | New recipe | Targeted edit | Clarification | List proposal |
|---|---:|---:|---:|---:|
| GPT-6.1 Sol | 32.550 | 18.583 | 16.326 | 18.259 |
| GPT-6 Astra | 28.385 | 21.457 | 14.606 | 19.023 |
| GPT-6 Luna | 18.700 | 14.192 | 6.312 | 13.184 |

The final Sol qualification after recipe hardening completed in 24.045, 18.575, 10.317 and 12.949 seconds. Luna was fastest in the initial sample; Sol remains the default pending broader evaluation. Actual billing usage arrives eventually: missing usage is shown as pending, never counted as zero. No per-answer cost savings or universal speed improvement is claimed.

## Review findings addressed

Three native gstack adversarial passes found actionable defects. Fixes include scoping/removing unsafe legacy operations, exact target selection, normalization of reviewed amounts and stores, verifying actual requested fields, freezing calendar contents, rejecting missing membership during SQL writes, preserving request recovery controls and clearing old errors after reconciliation. Legacy operations that cannot meet the pilot's approval contract are not exposed.

The last review pass required fixes and is therefore recorded as `issues_found / converged:false`. It is not represented as an independent clean certification of subsequent edits. Final automated checks cover the repaired cases. Outside-provider reviews are disabled by company policy. The later live UI tests also drove the recipe-card safeguard and its four regression cases, plus a SQL-date serialization fix and two regression tests. The saved-recipe proposal failed because MySQL Date objects could not be marshalled by DynamoDB. Both added tests reproduced the error before the fix; durable records now use the same JSON date representation as the tool response and snapshot fingerprint. The failed provider turn had already timed out while waiting for its result; resuming preserved the conversation but could not retroactively create an approval. A new regression check ensures every subsequent turn receives the complete current proposal state, including an explicit empty state, so Thyme need not guess whether an earlier approval exists.

## Limits and remaining rollout gates

- No automated real-account mutation was approved. Before broader adoption, manually approve a harmless list/recipe change and verify it in the native app. All dangerous or ambiguous outcomes stay unconfirmed; an uncertain write is never repeated automatically.
- This is a private text-chat pilot, not an iOS release. No TestFlight build, App Store update, voice integration or all-customer cutover occurred.
- No purchases, external messages, bulk clearing, legacy dish-log editing, URL imports or asynchronous legacy meal-plan regeneration are enabled here.
- Longer conversations, large recipe libraries, broader household/timezone coverage, load/cost benchmarking and more varied food constraints need expansion before general rollout. The current lexical food checks are not an exhaustive allergen or food-safety guarantee.
- Live domain adapters do not all offer a single transaction combining review revision with mutation. Snapshot checks and exact targets reduce races; full transactional adapters remain a rollout gate.

## Deployment and reproducibility

- Backend artifact SHA-256: `5e20423b23e47976df15135dc1f507fcb70cb61c233d1a2c0c962b1768fd9198`.
- API: `trepo-thyme-agent-v2-pilot-api`; worker: `trepo-thyme-agent-v2-pilot-worker`; dedicated DynamoDB state table and stream.
- Site: `appgprj_6abf3a2970fc819188a587e3383c4d92`, owner-private, environment revision 1.
- Site source/deployment revisions are recorded in `DEPLOYMENT.json` after final frontend publication.
- Detailed local evidence: `/Users/MattTaylor/Documents/Codex/2026-10-01/thyme-agent-pilot/evidence`. It contains synthetic model results, sanitized cloud checks, test logs and UI captures. Full real-account responses and credentials remain in the private cache and are not committed.

Brand: approved app guidance from `brand/trepo-app-design-system`, alongside `brand/trepo-brand-guidelines`; original Thyme artwork, cream/teal/yellow palette and outlined controls. No approved brand exception or amendment was introduced.
