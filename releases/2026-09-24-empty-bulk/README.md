# Empty bulk scan terminal outcomes — September 24, 2026

Status: source committed in `f369342`; both code-only deployments completed with unchanged configuration, and real Node20 cloud admission/polling canaries passed on September24. Final native qualification and TestFlight remain separate gates.

This is a narrow follow-up to the customer journey release. A scan that identifies no grocery items must not claim that items are ready to review. The worker now stores a terminal no-items failure for ordinary bulk inventory scans and sends no empty-ready notification. The separately deployed polling handler projects retained legacy completed-empty jobs to the same terminal response, without rewriting their stored records. Receipt-specific typed recovery remains unchanged.

The new app candidate also needs its own fix: candidate 124 removed a legacy empty-result safeguard too broadly, allowing an empty job to contaminate a later review. That unshipped native regression is distinct from the older backend status contract. Native fixes and full rerun evidence belong in the iOS release record.

## Compatibility

The no-items result is HTTP 200 with status `failed` because installed 1.16 accepts only 200–202 before inspecting job status. Generic failures retain HTTP 500. Nonempty results, receipts, pending work, commit markers and unrelated analysis modes are preserved. Photo reuse stays disabled. No permissions, schema, environment, provider, runtime or dependency changes are included.

The existing aggregate session processing/TTL behavior remains unchanged; installed iOS completion is computed per photo. No ready-notification claim is consumed for an all-empty session, so a later valid admission can still notify once. No Android device run was performed. Cached client state still requires the native repair.

## Reproduction and tests

The actual worker initially failed 5 of 9 focused tests; the actual polling handler failed 2 of 9. The final frozen source passes **90 tests, zero failures or skips**, including real local Dynamo lease/session/inbox roundtrips, concurrent terminal settlement, no redrive or ready push, later valid admission, foreign-owner rejection, pure legacy projection, receipt preservation and signed inbox owner/cursor checks. The exact archived ZIPs and independent test-source bundle were extracted afresh and passed all 90 again.

Local tests used Node 23.6.1 and injected provider analysis/transport results. Deployed Node 20 and real-provider accuracy need the separate live canary. Earlier failed reproduction and broader unpublished-test mismatch evidence is retained, not relabeled as passing.

`manifest.json`, overlays and `rebuild.py` reproduce each exact candidate ZIP from its checksum-pinned serving baseline. Run `python3 rebuild.py capture BASE.zip NEW.zip`, or use `getJob`. The script fails if any artifact differs. Private versioned bucket receipts in `evidence/` identify the exact originals and candidates; the capture baseline is the previously archived journey capture release.

The complete portable test-source bundle, including the pinned TypeScript parser and license, is archived at the exact version in `evidence/test-source-backup.json` (SHA256 `805dbfad569615c35ad5fedb9acdce8f94ca5ef32155f47fc18d84b9025ca40e`). The source/test runner is also committed under `test-source`; the third-party parser is in that archived bundle only. To reproduce in an empty directory, extract the two candidate ZIPs into `candidates/capture` and `candidates/getJob`, extract the test-source bundle at the root, start isolated DynamoDB Local at 127.0.0.1:38001 with telemetry disabled, then run `bash run-tests.sh`. Test dependencies are never deployed.

## Rollout and rollback

Deploy GetJob first, then identify-async, each with its pinned expected RevisionId and full unchanged configuration verification. Original ZIPs are privately archived and fully downloaded to verify their checksums. Rollback is code-only against the current revision, using those original artifacts; do not overwrite concurrent unrelated deployments. The final native full suite and health gate remain required before TestFlight upload.

Company sources: `apps/trepo/architecture`, `brand/trepo-brand-guidelines` (source dated September 15), and `brand/trepo-app-design-system`. No brand exception, shared-memory save, external review service, or telemetry was used.

## Live cloud verification

Exact deployed artifacts matched the qualified ZIPs. Actual public HTTP admission and polling verified: a synthetic legacy completed-empty job projects to HTTP200/failed; a newly analyzed blank bulk photo stores failed/no_items; a fridge photograph returns three items; a receipt returns three items; and a blank receipt retains its typed no_items recovery result. Photo reuse remained OFF. Kitchen inventory IDs were unchanged. Every uniquely owned test job was deleted after terminal processing and absence verified; current offloaded source objects were removed where present. Existing bucket version retention is unchanged.

The first live test used an overly broad whole-row equality assertion. The unchanged queue removes TTL from retained uncommitted review rows, so the corrected assertion allows only that exact pre-existing retention-metadata change and requires every other field to match. The failed attempt, original test, correction rationale and successful rerun are retained under `evidence/`. No serving code was changed to accommodate the test.
