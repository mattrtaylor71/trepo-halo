# Empty bulk upload follow-up — local candidate qualified, not deployed

2026-09-24. Scope approved by parent release task: restore nonreviewable terminal semantics for ordinary `bulk_inventory_deep` scans with an actual empty `result.items` array, including older retained jobs. No serving deployments, customer record changes, canonical edits, app edits, or recipe edits were performed by this task.

## Root cause and exact serving baseline

Read-only API Gateway inspection (`routes.json`) confirms POST `/identify-async` serves `grocery-identifier-dev-identify-async`; GET `/job/{job_id}` serves the separately packaged `grocery-identifier-dev-GetJobFunction-xL9ddcVmBIaC`.

The worker unconditionally stored `completed`, sent a ready push, and counted the session job as completed after valid empty analysis. The independently deployed GET handler exposed that as HTTP 200/status completed. The initial pre-release capture source also unconditionally completed empty analysis: this is a pre-existing status-contract problem, not caused by the newer honest “No items identified” stage text. Private diagnostic job contents remain in their original release diagnostics and are not copied here.

The independently downloaded GetSessions package already excludes completed empty items from the signed review inbox. Its source differs from the unused older copy inside the capture ZIP. No GetSessions or GetSession deployment is proposed. Their original ZIPs, hashes, revisions, and configuration hashes are preserved in `baseline/` and `baseline.json`.

## Minimal change

- Worker: valid empty bulk analysis stores terminal `failed` with `result.items=[]`, aggregate `capture_outcome=no_items`, `error_code=no_items_identified`, and the message “No grocery items were found. Try a clearer photo.” It does not automatically repeat the same provider analysis, persist kitchen items, or send a single-job ready notification. Only the winning terminal write updates session failure counters.
- GET/job: projects retained completed empty bulk jobs and new explicitly marked empty failures to HTTP 200/status failed. HTTP 200 is necessary because installed 1.16 pollers only inspect 200–202 responses. This is a pure response projection; it does not update stored job status, result, ownership, or session records. The existing queue's unrelated lazy TTL-retention behavior is unchanged.
- Generic failed jobs retain HTTP 500. Nonempty bulk, receipt typed recovery, noninventory modes, pending/processing work, malformed/missing results, and any commit marker are not reinterpreted. Existing server commit markers are `committing` and `committed`; unknown markers are conservatively preserved.
- The helper adds only aggregate outcome metadata. It invents no per-image index, source classification, or recovery capability. Photo reuse remains disabled.

A completed all-empty set is recorded in `failed_job_ids`, while the existing aggregate session processing/TTL semantics remain unchanged. No ready-notification claim is consumed: a later valid photo admitted into the same appendable session can still send its one ready notification. A real Dynamo roundtrip proves this and rejects a foreign-owner append. This is per-job terminal settlement, not a claim that the aggregate row becomes terminal.

## Compatibility and limits

Actual frozen 1.16 source at commit `d1a61a9`, and frozen App Store 113 `BulkSessionManager.swift`, accept 200–202 before reading status and handle `failed` as terminal. They compute completion from per-photo uploading/processing states. The 1.16 tree contains only definitions, no calls, for `fetchSession`/`fetchActiveSessions`, so the unchanged aggregate status does not create an installed-iOS polling wait. Current Android source `ReviewSession.kt` derives source job IDs without waiting for aggregate status, and `TrepoApi.reviewSession` re-reads each owner-checked job; failed jobs are excluded by `reviewSessionJob`. These are source traces, not a new Android device run.

Legacy GET/job remains its existing UUID-based polling API; no authentication contract was changed. Signed inbox authentication, exact owner checks, strong source rereads, and owner-bound pagination cursors are unchanged and tested against the actual deployed package with local Dynamo.

Already cached native `completed` state cannot be invalidated by a backend response unless the client reads and applies it. Current ordinary Fridge/Pantry uses `ReceiptSessionManager`; its old failed branch does not parse typed capture metadata. The release team owns native cached-state/review-source filtering and any typed no-items recovery presentation. Backend qualification alone does not qualify those screens.

## Reproduction and validation

`worker-before.txt`: 9 actual-handler tests, 4 passed / 5 failed before repair. Failures show empty worker results remained completed and emitted ready effects. `poll-before.txt`: 9 actual-handler tests, 7 passed / 2 failed before repair; legacy completed empty stayed completed, and newly failed no-items would return HTTP 500.

`focused-final.txt`: **90 passed, 0 failed, 0 skipped** on final frozen source. Includes actual handler execution, real job queue/lease/session Dynamo roundtrips, winning/losing terminal races, repeat/overlap delivery, no-redrive/no-ready effects, later valid admission, receipt/noninventory/nonempty preservation, pure legacy polling projection, and independently deployed GetSessions signed inbox restoration/owner/cursor/no-write checks.

Tests inject analysis outcomes and transport clients; they do not assert model detection accuracy or perform provider calls. Prompts, providers, IAM, environment, runtime, and layers are unchanged. Parent owns final cloud runtime/canary qualification and native reruns.

Reproduce from `candidates/capture`:

```sh
NODE_PATH=/Users/MattTaylor/Desktop/trepov2/grocery-identifier/node_modules node --experimental-test-module-mocks --test --test-reporter=tap test/*.test.cjs test/*.test.mjs ../../tests/*.test.cjs
```

Local Dynamo must run at 127.0.0.1:38001. Every test creates/deletes its own unique tables. The only external test dependency above is the already installed TypeScript parser; production source loads from the exact isolated candidates/baselines. `scene-source-manifest.test.cjs` was adapted from a broader unpublished feature test to assert the approved serving writer still keeps `s3_key=null` (photo reuse disabled). The earlier broad run's failure expecting unpublished binding is retained in `capture-regression-final.txt`; it is not misrepresented as a final-source regression. No production writer change was made.

## Exact proposed artifacts

| Function | Candidate ZIP CodeSha256 | Expected baseline RevisionId |
| --- | --- | --- |
| identify-async | `m0s046cogIP8mE7Jz+IrIrSUSpaKT1I6UnBpTHPOEe4=` | `186ef2a2-f755-4da1-ab42-4be3cfb62bd7` |
| GetJob | `1S0Y+omz2yjGKwTeMHmaAHetd9fRU793pTnyiScOlSs=` | `3cd6f42e-b17e-49bc-af6c-df972b299c7c` |

`capture.zip` changes only `identifyAsync/app.js` and adds `emptyBulkOutcome.js`; 13,733 other entries are byte-identical to the serving ZIP. `getJob.zip` changes only `getJob/app.js` and adds the identical helper; 13,633 other entries are byte-identical. No removed entries, dependency changes, or configuration changes. `manifest.json` records full source hashes, ZIP hashes, independent baselines and counts; `empty-bulk.patch` is the three-file production patch. Original serving ZIPs are retained for parent-controlled rollback.

Relevant company context: `apps/trepo/architecture` and `brand/trepo-brand-guidelines` (source PDF dated 2026-09-15, retrieved 2026-09-23). The brief recovery message changes no visual brand treatment; no applicable approved copy amendment or brand exception was identified. No shared brain write, telemetry, external-provider review, or upgrade was performed.

## Portable test-source handoff (final reconstruction gate)

`test-source.zip` SHA256 `805dbfad569615c35ad5fedb9acdce8f94ca5ef32155f47fc18d84b9025ca40e` contains **11** test files, six exact unchanged serving GetSessions source/package files, the pinned TypeScript 5.9.3 parser and license, and `run-tests.sh`. `test-source-manifest.json` enumerates every required file and SHA256; test dependencies are outside the proposed deployment ZIPs.

To reproduce without any canonical checkout, create a new empty directory. Extract `capture.zip` into `candidates/capture`, `getJob.zip` into `candidates/getJob`, and `test-source.zip` at the new directory root. Start isolated DynamoDB Local at 127.0.0.1:38001 with `-inMemory -sharedDb -disableTelemetry`, then run `bash run-tests.sh` from that directory. The runner contains the complete 90-test invocation and resolves all AWS dependencies from capture.zip; every GetSessions dependency file was independently verified byte-identical to the corresponding capture dependency. The source bundle deliberately includes no provider evaluation/category scripts that are not part of this test invocation.

This exact reconstruction was performed in `reconstructed/`: **90 passed, 0 failed, 0 skipped**, recorded in `reconstructed-final.txt`. Production ZIP hashes remain unchanged. Node used locally was v23.6.1; parent owns the deployed Node20 cloud/native qualification. Cleanup verified zero local fixture tables, stopped owned Dynamo PIDs82543 and90606, and left MySQL33317/PID2186 running for the parent fixture. No cloud fixture was created here.
