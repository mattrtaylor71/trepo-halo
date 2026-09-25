# Customer journey release — September 24, 2026

Status: all six backend fixes are deployed; controlled live canaries passed. Authorized by Matt (“Yes and commit and document. And cut to test fight”). TestFlight target1.17(124) has compiled, signed/exported and passed Apple's package validation. The full native qualification is running; upload and tester availability remain pending. Build123 does not contain these incident UI changes.

## Release boundaries

Only the six serving-derived ZIPs in the release manifest are approved for deployment. Function code updates use the previously verified AWS revision and verify that configuration is unchanged. The older, dirty development folders are not deployment inputs. Original ZIPs are retained for service-specific rollback. No migration or IAM change is part of this release.

Capture photo reuse remains OFF. Typed wrong-upload/no-items outcomes and the native retake guidance are included. Enabling retained-photo recovery still requires the complete recovered-child-to-Kitchen-commit journey and configuration qualification; this release does not claim that feature is activated.

All live mutations use the dedicated App Store review account. Original incident/customer records are never modified by these tests.

## Live functional evidence

- Kitchen: create a uniquely identified test item; canonically edit name, brand, amount, storage and opened state; verify refreshed values; retry the exact operation and obtain the same receipt; reject changed payload under the same operation ID without changing saved values; verify legacy name-only editing; delete the exact fixture and verify absence.
- The first Kitchen canary expected an old-style quantity update to succeed after a versioned amount edit. It correctly received the **pre-existing** stale-amount protection409. That failed test attempt is retained. The corrected test explicitly verifies the409 and legacy name-only success. No compatibility rule was weakened to make a test pass.
- Saved Recipes: real text extraction to usable ingredients and steps; optimistic revision conflict for an old edit; same-ID paste recovery through the real provider; preservation of the manually edited title and note; fresh GET matches recovered ingredients/instructions; exact fixture deletion/absence verified.
- Capture: actual admission and background processing in production. Receipt→3 items; pantry image submitted as receipt→wrong_capture_type with0 items; blank image→no_items with0 items. All three have recovery unavailable while the feature is OFF. The three test jobs were deleted after terminal processing; the oversized pantry source object was deleted as well.

Private evidence: `/Users/MattTaylor/Library/Caches/trepo-jill-implementation-20260924/rollout/`. Files `kitchen-recipes-canary.json`, `kitchen-recipes-attempt1.json`, `capture-canary.json` and individual service deployment receipts preserve exact outcomes, timings and release hashes. The frozen app release is `/Users/MattTaylor/Library/Caches/trepo-codex-release/1.17-124-journey-20260924/`.

- Thyme sync and stream: real-model rename of a uniquely identified test item, authoritative Kitchen readback, then removal of only that test session's completed short-term response-cache entry. A fresh invocation with the same operation ID returned without advancing the item's revision. Exact test items were deleted and absence verified. Test session records were removed; immutable operation journals/receipts are retained as designed.
- Async voice: actual audio admission, immutable source storage, transcription and deployed worker completion passed. This used the existing synthetic `readlist.wav` fixture, not a physical microphone. Its normal short-lived voice job/source remain subject to the service's existing retention; they are not customer data or Kitchen inventory.
- Fresh post-rollout CloudWatch snapshot reported0 Lambda Errors and0 Throttles across all six functions (2 stream,2 sync,1 worker,57 Kitchen,96 recipe and6 capture invocations). This is a short observation window of unhandled errors/throttling, not proof of device delivery or absence of handled application errors.

## Durable retry follow-up

The final gstack audit reproduced a lost-response retry conflict against the original ZIP. The fix retains an immutable actor/household-scoped canonical request before mutation and reads it before target resolution. All original Kitchen hash and revision checks remain. A later manual edit or deletion is not undone by a retry. Separate accepted legacy commands receive distinct generation IDs so repeated words later remain usable; explicit operation IDs retain stable replay.

Final exact Thyme candidate suites pass143/143/144 with0 skipped, plus13 real Node→Python→MySQL round trips per candidate. Four new retry scenarios fail on the original ZIP and pass after repair. The earlier39-model evaluation does not exercise datastore writes; final live sync/stream tests above do. No schema/IAM/configuration changes were required.

## Source history

The development iOS branch was still committed at the older build115 checkpoint. A separate release branch, `codex/journey-release-124`, first records the exact already-distributed123 app manifest in commit `cd2d4af`. This avoids describing accumulated116–123 work as this incident fix. The incident124 commit will contain the ten reviewed Swift changes, the two version files, tests and release documentation. The original dirty working tree is preserved.

Backend release source will be recorded as exact per-function overlays, patches and hash manifests on `codex/journey-backend-release-20260924`. The serving sync/stream/worker functions have intentional base differences, so a single development folder is not an honest representation of all three deployed artifacts. Overlay records preserve each function's exact reviewed bytes, including those base differences, without sweeping unrelated work into a deployment.

## Coverage limits

The earlier targeted results are recorded in TEST-REPORT.md. Counts overlap and are not a unique total. Older iOS decoders and Android JVM contracts qualify the tested wire formats, not complete old-client UI. Physical camera, microphone, Kitchen Assistant hardware, push delivery, new alert thresholds and client-render delivery deduplication are not marked passed. Existing receipt-row category/status pill clipping remains a separate UX-audit finding. Brand sources: `brand/trepo-brand-guidelines`, `brand/trepo-app-design-system`; no brand exception. Architecture source: `apps/trepo/architecture`. No automatic shared-brain save was performed.
