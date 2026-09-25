# Saved Recipes media timeout follow-up — September 24, 2026

Status: reviewed, regression-tested candidate. Deployment, final live canaries and final native Release checks are pending. This folder does not claim that its package is serving yet.

During the journey release health check, one URL import exhausted the service's 120-second limit while downloading a 45.92 MiB social video. AWS retried it; a retained recipe eventually appeared after roughly three minutes. Other oversized videos reached transcription and received size errors. Comparing the relevant functions against the package serving before the journey release confirmed that this behavior predates that release. No affected customer's data was edited or replayed to investigate it.

## Behavior change

The service now gives external providers a bounded amount of time and leaves 15 seconds for saving the result. Downloads stop after at most 30 seconds; transcription stops after at most 45 seconds and does not repeat SDK retries. The actual remaining invocation time can shorten either limit. Both media-download paths enforce a 24,000,000-byte ceiling before transcription. Existing audio format selection and acceptance of genuine spoken recipes are preserved.

If a source cannot be processed within those limits, the accepted link is retained with an honest incomplete outcome. The same terminal job does not keep scheduling identical repair work. A short synchronous request can still defer once to the longer asynchronous path. A usable recipe remains usable; optional image or category enrichment cannot consume all completion time. Human edits, recipe identity, ownership checks and explicit recovery remain intact.

Only `app.py` and `recipe_work_budget.py` change. No model, prompt, layer, binary, schema, permission or runtime configuration changes are included. Timer boundaries surround provider work, not SQL transactions or commits. A separate database or service outage can still prevent persistence and must remain a visible failure.

## Exact source record

- `manifest.json` pins the initial journey recipe package and this candidate. The initial six-service release remains recorded, unchanged, in `../2026-09-24-journey/`.
- `overlays/recipes/` contains the two complete changed source files; `recipes.patch` is the readable delta.
- `new-entry-metadata.json` preserves ZIP metadata. `rebuild.py recipes BASE_ZIP NEW_OUTPUT_ZIP` reconstructs and verifies the exact package without cloud access. Reproduction was verified before committing this record.
- `tests/` contains the 16 exact regression source files, pinned in `test-source-manifest.json`. Application code was loaded from the candidate, not substituted from the dirty development checkout.
- `evidence/agent-candidate-manifest.json` describes the review agent's package. Its ZIP metadata differs from the parent's final package, but every uncompressed entry is identical; `package-parity.json` verifies this. The authoritative candidate digest is in this folder's `manifest.json`.

## Verification

| Check | Result | What it establishes |
| --- | --- | --- |
| Same 28 new cases against the previous application | 22 failed, 6 passed | The tests reproduce the missing deadlines, size guards and terminal outcomes |
| Candidate regression suite | 224 passed, 0 failed, 0 skipped | Includes the 28 new cases and 196 existing cases; counts overlap other release suites |
| Real slow HTTP body and actual SDK 429 response | Passed | A trickling body is interrupted; a transcription rate limit makes one SDK attempt |
| Real isolated MySQL and DynamoDB Local | Passed | One retained recipe and shared mirror, stable poll identity, no repeated extraction on duplicate delivery, other owner's data unchanged |
| Exact Python 3.9 cloud runtime and existing layers | Passed | Native deadline, timer restoration, async-stage refusal, size rejection and hard interruption work in the deployed runtime |
| Real spoken MP4 through transcription and refinement | Passed in 5.268 seconds | Three ingredients and four instructions produced from the synthetic narration; downloaded bytes matched the source |
| First audio fixture attempt | Failed: provider rejected M4A format | Kept as a fixture failure; the valid MP4 retry is the passing evidence, not a relabeled result |

The cloud probe used unchanged candidate application code with an unshipped diagnostic handler. It made no customer/database changes. Its temporary function and exact synthetic audio object version were deleted and verified absent. The separate diagnostic log group was not removed. Local database fixture cleanup is recorded; the unrelated native-test fixture remained active for the release coordinator.

Tests require the repository's existing test dependencies and isolated database fixtures. For a fresh test workspace, reconstruct the package, extract its root Python application modules to `saved_recipes_api/` beside a copy of `tests/`, put that directory on `PYTHONPATH`, and point both `RECIPE_CANDIDATE_SOURCE` and `TREPO_SAVED_RECIPE_TEST_APP` at it (the latter at `app.py`). Use only the isolated MySQL port 33317 and DynamoDB Local endpoint 38001. Missing database fixtures are not a full qualification pass. Do not run these tests against production databases.

## Remaining release gates and rollback

Wait for the initial full native run to finish before changing its backend baseline. Deploy this recipe package with the expected current CodeSha256 and RevisionId, verify unchanged configuration, run controlled live canaries on the dedicated review account, and then run the final native Release checks against this new baseline. TestFlight upload remains gated until those checks pass.

To roll back this follow-up alone, restore the checksum-pinned initial journey recipe ZIP with a current RevisionId condition, only if the function still serves this follow-up. To reverse the entire journey recipe release, use the earlier release's original serving ZIP instead. Never overwrite a newer deployment or deploy the whole dirty development checkout. Preserve both deployment histories.

Company context: `apps/trepo/architecture`. No customer-facing design or brand exception is introduced. This is repository documentation; no automatic shared-brain save was made.
