# Cloud reliability repairs — October 23, 2026

Status: implemented and regression-tested locally. Not deployed. No production customer records, permissions, firmware, notifications or saved company memory were changed by this implementation. The release ZIPs preserve the serving functions' unmodified files and dependencies.

## Changes and evidence

| Repair | Final behavior | Verification |
|---|---|---|
| Household kitchen actions | Resolve current server membership, lock it through the mutation, and update the exact active item. Partner-owned items work; transferred, foreign and deleted items fail closed. Discard copies and inventory commit together. Shopping receipts return the actual UUID rather than `0`. | Real MySQL: partner removal, same-name separation, foreign IDs, transfer during mutation, sink rollback, quantity edits, legacy null-household roots and receipt resolution. |
| Meal-plan recovery | Ignore stage-only differences while keeping semantic/version guards. Durable requests coalesce by household; retries use current focus, original verified consent actor, a 15-second debounce and at most three attempts per unchanged input revision. Leases recover dead workers. Old tokens cannot publish or fail newer work. | SQL claims, retry exhaustion, duplicate delivery, killed-worker lease, simultaneous requests, atomic plan/queue commit, changed focus and household, consent denial, deadline/provider failure fixtures. |
| Dietary validation | Check both `steps` and `instructions`. Require well-formed recipes. Repair only rejected slots and validate the complete result again. Provider work respects the invocation deadline and permission wrapper. | Allergen-only-in-steps, both fields together, plant substitutes, malformed slots, malformed output and unsafe repair rejection. This is a regression fix, not a certification of every dietary interpretation. |
| Recipe status polling | Both serving implementations return current owner-scoped saved results without image/model work, kitchen matching, DDL or availability writes. Edited and deleted results remain current. Availability is explicitly pending. | 100 concurrent fixture polls, owner isolation, old/missing tables versus real DB outages, partial/terminal result contracts. No production latency/load claim. |
| Recipe recovery | Try retained social/web/PDF source first, recognize “How to make” and numbered methods, checkpoint extraction independently of verification, and keep owner/evidence/version/expiry isolation. Never turn the text gate into a verification bypass. | Retained sources survive unavailable URLs, verifier timeouts reuse candidates, changed evidence/owner/version invalidate them, incomplete sources fetch more evidence, unsupported temperature remains rejected. |
| Deadlock recovery | Retry the complete SQL insert plus expiration-provenance transaction only after MySQL 1213 rollback. At most three attempts, stable item ID, no repeated model work. Ambiguous network/commit errors are not blindly retried. | An actual two-connection InnoDB deadlock; rollback, exhaustion, provenance failure, replay and preservation of a customer's edited name. Historical lock-cycle attribution is still unknown. |
| Outcome monitoring | Read actual meal-plan/import failures, retry states, structured action receipts, retained Halo voice outcomes, missing original objects, and capacity errors. Failed reads become unknown coverage rather than zero. | Failed-plan, partial-action, absent-media/late-arrival, permissions/read-failure, throttling and metric-delivery fixtures. CloudFormation alarm validation. |
| Immutable icon delivery | A CloudFront template admits only the three approved generic icon namespaces and hash-shaped paths. Origin still checks content hashes. No bucket is made public. Catalog responses can switch to the validated CDN hostname. | Allowed/denied methods, query/path restrictions, original bytes, invalid hashes and invalid CDN hosts. CloudFormation validation. Real edge hit rate and cold-cache load await deployment. |
| iMessage context refresh | One owner's failure no longer replays every successful owner. Transient source/fence errors get one fresh-snapshot retry; scope/permission failures stop. Per-owner reason codes and a failure metric remain visible. GET still validates current membership before using cached context. | Source failure isolation, changed-fence retry, scope/permission denial, deadline reserve, state-read failure and stale-cache denial. This does not establish the historical cause of every `BRAIN_UNAVAILABLE`. |

## Test results

Run `RELIABILITY_MYSQL_PORT=33387 bash ops/cloud-reliability/run-tests.sh` with a disposable localhost MySQL server and the Python dependencies in `ops/cloud-reliability/test-requirements.txt` and the voice service npm lockfile. The tested database was MySQL 8.4.11; all tests used disposable schemas on port 33387.

- 109 Python tests passed, with six additional subtests.
- 153 voice Node tests passed, with no skips.
- Three edge-routing tests and seven context-projection tests passed.
- **272 tests in total**, plus six subtests.
- The eight household MySQL tests also passed against the assembled SAM entry-point variant, outside that total.
- Three infrastructure templates passed AWS `ValidateTemplate`; no stack was created.
- All 11 release packages passed Linux Lambda startup checks, which import the assembled handler and every changed Python module, with networking disabled and dummy credentials. Streaming uses a minimal `awslambda.streamifyResponse` runtime shim; this checks import closure, not streaming behavior.

The baseline voice suite had nine failures from stale fixtures and a missing declared `jose` dependency. Fixtures now model the actual serving transaction/result contract; `jose@6.1.0` is declared. These fixture fixes are not nine extra customer bug fixes.

Private reproducible evidence is in `/Users/MattTaylor/Documents/Codex/2026-10-23/halo-cloud-audit/deep/final-regression.log`, `deep/sam-assembled-integration.log`, and `release/startup-results.json`. Runtime/package hashes are in `release/manifest.json`. Raw customer evidence is not copied into this source repository.

## Release construction

`ops/cloud-reliability/release/build.py --out <private-directory>` downloads current serving ZIPs read-only, verifies their digest and reviewed source baseline, and replaces only named repaired modules. It retains each function's handler path, native dependencies and consent wrappers. Before any deployment, use the manifest's base revision as a compare-and-swap condition and recheck code/configuration drift.

The SAM voice variant differs from the async/streaming worker in its existing consent integration. The builder accepts only its explicitly reviewed file hash and applies a three-way merge; it does not copy the worker's consent integration into SAM or remove the worker's checks. Unknown drift or a merge conflict stops construction. Regression tests exercise this refusal behavior.

The Meal Plan API gains the exact first-party modules required by its current-membership/read-validation path. The new dispatcher uses the generator ZIP but only reads the durable queue and invokes the existing consent-guarded generator. It has no provider key or permission to call a model directly.

## Rollout order and prerequisites

1. Recheck serving revisions and preserve rollback ZIPs/configuration. Do not deploy the checkout wholesale: some baseline source reconciliation was substantial.
2. Apply the additive `meal_plan_generator/generation_jobs.sql` migration. No GET performs this migration. Install the retry scheduler with its default **DISABLED** state.
3. Release narrow voice, dietary, status, retained-source recovery and kitchen SQL changes to the approved review-account/canary path. Confirm real saved records and receipt IDs; do not rely on assistant response text.
4. Drain or account for old meal-plan workers before enabling new retries. Deploy the fenced generator, then run controlled current-input generation. The new Meal Plan API requires publication provenance: **do not turn it on globally before legacy plans have been reconciled/regenerated.** An old plan with no trustworthy provenance is withheld, never silently stamped with today's household.
5. Enable retry dispatch only after queue, worker and consent/focus tests pass against the deployed canary. Observe outcomes for at least 30 minutes and enough completed operations; quiet time alone is not a pass.
6. Give the monitor only the required DynamoDB scan/read tables, original-media `s3:GetObject` prefixes and `cloudwatch:GetMetricStatistics`. Configure `VOICE_OPERATION_TABLE`, `MONITORED_LAMBDAS`, `HALO_MEDIA_BUCKETS` and the Halo voice-jobs table. Connect alarms to the existing approved operations destination. Missing access must continue to alarm as incomplete coverage.
7. Create the icon distribution against the verified icon API origin. Test bytes, privacy rejection, a cold request and a subsequent cache hit that avoids Lambda. Then set `ICON_CDN_HOST` on catalog-producing paths. Old persisted or client-cached API URLs will continue using the origin until refreshed; the patch does not retroactively rewrite every saved URL.
8. Deploy context projection with its per-owner failure alarm. Preserve current privacy validation. Do not infer successful user messaging from a successful projection refresh.

Stop expansion for unauthorized household access, duplicate writes, edits/deletions overwritten, new missing imports, more failures or unbounded retries. Disable new dispatch before rolling back workers. Keep additive tables and evidence; never restore a database snapshot over newer customer edits.

## Customer recovery and remaining qualification

Historical data recovery is intentionally separate from code deployment. No customer backfill has been applied in this change set.

- Reconcile the known partial yogurt action against the current exact row/revision and original operation. Recover only the still-missing discard; never repeat the successful shopping add.
- Deduplicate failed meal-plan account rows into current households, regenerate those with no valid plan first, then remaining eligible failures against current membership/preferences.
- Build an exact revision manifest for recipe failures. Independently review recovered source-supported recipes before applying them through the existing guarded route. Preserve edits, deletions, original media and notification state.
- Completed recipe jobs affected only by status throttling need reliable delivery, not extraction replays.
- Absent original Halo media cannot be reconstructed by a cloud backfill. Monitoring now distinguishes missing media from a completed result.
- The five packaged-nutrition/dish failures require original-image review before changing nutrition behavior; no precise nutrition was guessed here.
- No native iOS backoff change, phone kill/reopen journey, production model replay, or live CloudFront load test is claimed. The server polling cause is fixed; these remain rollout qualification/client follow-up.
- No firmware patch or OTA qualification was part of this cloud implementation. Expected no-token push responses were not turned into notification retries.

## Investigation provenance

Root causes and reproductions are recorded in the private `halo-cloud-audit/FIX-PLAN.md`. Serving source is the authority for deployment state. The relevant recalled company page was `apps/trepo/sources/20260923/ios/docs/codex/architecture-fixes-20260913/kitchen-refresh-version`; it helped preserve already-completed version bookkeeping. No company memory was saved.

The gstack investigation workflow was used: concrete failing behavior, source tracing, regression coverage, then repair. No outside-provider review, telemetry or artifact syncing was enabled. Cache configuration was checked against AWS's [DefaultCacheBehavior documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-cloudfront-distribution-defaultcachebehavior.html) and [query-string caching rules](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/QueryStringParameters.html).

The limited new user-facing meal-plan status wording was checked against `brand/trepo-brand-guidelines` (source modified September 15, 2026) and the amendment search. No applicable approved copy amendment or brand exception was found; no visual assets or brand rules were changed.
