# Journey repair test report — September 24, 2026

Current rollout status and live evidence: [RELEASE.md](RELEASE.md). The qualification snapshot below records the pre-deployment phase and its limits.

Targeted qualification complete; controlled production checks remain. Counts below are separate suites and overlap. Do not add them into a unique total. All backend fixtures use isolated MySQL33317/DynamoDB Local38001 unless explicitly described otherwise. No destructive test used the incident customer's household.

| Test layer | Latest verified result | Evidence in private implementation directory |
|---|---:|---|
| Full recipe/saved-recipe regression | 310 passed, 2 existing datetime warnings | `recipe-regression-final3.txt` |
| Minimal serving recipe candidate: recovery, preservation, completion, atomic edits | 73 passed | `recipe-minimal-source-final3.txt` |
| Kitchen isolated regressions including 9 cross-language round trips | 47 passed | `kitchen-isolated-final.txt` |
| Minimal serving Kitchen source | 38 passed, 3 existing datetime warnings | `kitchen-minimal-source-tests.txt` |
| Full Thyme source regression | 982 passed, 0 skipped | `thyme-regression-final4.txt` |
| Exact serving-derived Thyme stream candidate | 134 passed | `thyme-minimal-final.txt` |
| Exact serving-derived Thyme sync candidate | 134 passed | `thyme-sync-minimal-final.txt` |
| Exact serving-derived Thyme async worker candidate | 135 passed | `thyme-worker-minimal-final.txt` |
| Capture full regression | 566 passed, 0 skipped | `capture-regression-final.txt` |
| Minimal capture candidate focused suite | 43 passed | `capture-minimal-final2.txt` |
| Live assistant model: 13 cases × 3 repetitions | 39/39 on final packaged candidate | `model-eval-candidate-final.json`; `model-eval-final-release.json` |
| Actual receipt image provider: receipt, pantry scene, blank × 2 | 6/6; receipt 3 items each, scene/blank 0 | `capture-model-eval.json` |
| Actual receipt image provider: synthetic product-label sheet × 2 | 2/2; product/wrong_capture_type, zero receipt items | `capture-label-model-eval.json` |
| Frozen iOS 1.16 and 1.17 actual Swift wire decoders | 44 assertions each | `legacy-116.swift`, `legacy-117.swift`, `legacy-wire-fixtures.json` |
| Actual native recipe state contract | 32 assertions | `native-recipe-contract` |
| Actual Swift capture decode/persistence: optional capability | 8 assertions | `native-capture-contract.swift` |
| Android actual parser and existing import/notes/bookmark/dead-row contracts | 15 passed, 0 skipped; seven exact candidate recipe envelopes | `android-contract-verified.json`; `android-results/` native JUnit XML |
| Six final cloud ZIPs under exact deployed runtimes/layers | All boot; no FunctionError | `cloud-runtime-probes-final.json`; updated capture: `cloud-capture-capability-final.json` |
| Native Debug smoke after nine-file integration | 1/1 | Mac mini run `20260924-163128-174a57` |
| Final native Debug: recipe recovery, capture empty/mixed/expired, content and cold launch | 5/5 | Mac mini run `20260924-171048-d952c6`; all 278 source inputs independently verified |
| Final native Release: same five journeys, plus edited receipt quantity surviving recovery navigation | 5/5 | Mac mini run `20260924-171611-ba3b99`; all 278 source inputs independently verified |
| Final normal-app Debug smoke, with no private transport | 1/1 | Mac mini run `20260924-172951-1c30fe`; all 278 source inputs independently verified |

Private evidence root: `/Users/MattTaylor/Library/Caches/trepo-jill-implementation-20260924/`.

Native recovery uses the real app screens, serializer, HTTP transport and candidate SQL update/revision/dual-write functions. Only extraction is deterministic in this native fixture; separate live-provider tests are listed above. Two cases exercise paste, source replacement, required replacement confirmation, cancellation, authoritative identity/status/note readback and cold relaunch. The final strengthening also asserts visible ingredient/direction text. The private transport is generated only into a frozen test snapshot, restricted to the review account and exact fixture paths. It is not part of the distributed app.

Failures are preserved: source retention retry reproduction; pending-note paste KeyError; unrelated confirmation rejection; local candidate tests initially omitted their import search path; the first native recovery selector run could not locate visible buttons by inherited accessibility IDs and was corrected to exact visible labels. None was relabeled a pass. Final tests must match their source manifest; passes before a later source change do not qualify that later source.

Limits: six cloud boot probes do not establish full cloud functional behavior. Android UI, physical devices, push delivery, capture-recovery native commit and controlled production canaries are not marked passed. Android's real application and tests compiled using JDK17/SDK36; this is contract qualification, not emulator qualification. The new fixes are not live or in TestFlight123. See IMPLEMENTATION.md for gates and deployment boundaries.

Capture availability regression: two new tests failed before implementation, then all five outcome tests passed. An independent reviewer also ran7 worker scenarios for enabled/off/leases-off/missing-secret/missing-manifest/wrong-owner/automatic-persist cases. Optional `recovery_available` is set only by the server from the verified retained source; missing/false means retake. The retained manifest proves source upload at completion, not indefinite future object availability. The endpoint checks bytes and digest again and returns410 when expired.

A candidate-only scene test referenced an unrelated unshipped writer and failed import. That is retained in `capture-minimal-final.txt`; the scoped shipped worker/recovery suite passed 43 in `capture-minimal-final2.txt`. The complete source suite includes the writer and passed 566.

Release mixed-capture reproduction `20260924-170113-430665`:4passed/1failed. The empty capture and expired-source screens passed, as did both SQL recipe journeys; mixed recovery failed because an outer safe-area inset overlaid the header and placed the banner hit target under the status bar. The final correction places the action inside BulkCheckInReviewView. The new native suite also checks that an existing receipt quantity edit survives recovery navigation. Original screenshots/hierarchy/video retained under `screens-capture-before/`.

The corrected mixed-banner Debug and final Release runs both passed all five native cases. Release adds a receipt quantity edit before recovery and verifies 2 remains after returning; its actual screenshot was inspected. Recipe recovery screenshots show ingredients, directions and the preserved human note after cold launch. Final scoped package validation checks all six ZIP hashes and changed entries, plus all ten native incident files (`final-source-package-verification.json`).

The native fixture server and reverse tunnel are stopped. Its exact isolated SQL schema was dropped and absence independently verified (`native-recovery/cleanup-verified.json`). The synthetic simulator draft was removed, with no synthetic receipt persisted. The owned local MySQL33317 and DynamoDB Local38001 test processes were stopped; MySQL fixture files and all reports remain available. Temporary cloud probe functions were deleted and absence verified; their diagnostic log groups remain under the existing permission limit.

Screenshots: [unreachable banner before correction](screens/capture-banner-before.png), [corrected banner and quantity 2](screens/capture-banner-release.png), [expired photo guidance](screens/capture-expired.png), [recovered recipe after relaunch](screens/recipe-recovered-release.png). The compact category/status pill truncation visible in the receipt row is an existing design-audit finding; it is not claimed repaired by this incident patch.

## Final deployment and retry follow-up

See RELEASE.md for live evidence and build124 status. Final packaged Thyme source after the durable replay repair: stream143/143, sync143/143, worker144/144, zero skipped;13 real Node→Python→MySQL round trips for each candidate. Four replay cases failed on the pre-fix ZIP. Focused local source118/118. New legacy request-generation test prevents later same-worded commands from replaying an old edit forever. Existing39-model harness uses mocked mutation; it is not claimed as datastore qualification for this final change. Production canaries directly exercise final sync/stream writes and fresh-handler replay. The actual audio async worker and three capture pipelines passed. All six final ZIPs were independently rebuilt from their original packages plus committed overlays and matched byte-for-byte.
