# September 24 customer-journey backend release

This is the exact deployed-source record for six production Lambda code updates. The resources' `dev` names do not mean staging. Matt explicitly authorized the fixes, deployment, commits, documentation and TestFlight.

## Why per-function overlays

The development branch contains extensive unrelated uncommitted work, and sync/stream/worker intentionally have different deployed bases. Shipping that directory wholesale would change unrelated runtime behavior. Each release ZIP was therefore built from that function's verified serving ZIP and only the reviewed incident entries. Every unlisted entry is unchanged. These are release-specific source overlays, not a claim that the development checkout was clean or all its work shipped.

- `manifest.json`: original CodeSha256/RevisionId, final code/archive digest and old/new source hashes.
- `overlays/<service>/`: complete source bytes for every new/modified entry (64 total entries across the six archives).
- `patches/`: readable exact diffs against each original serving package.
- `new-entry-metadata.json` and `rebuild.py`: reproduce exact reviewed archives using the checksum-pinned original ZIP. No network or deployment occurs.
- `tests/`: scoped regression source snapshot, with original repository paths preserved. These tests use the normal repository test dependencies/fixtures and isolated database setup; they are not a standalone production test runner. `test-source-manifest.json` pins their bytes.
- `evidence/`: sanitized deployment receipts, functional canaries and test results. No credentials, customer transcripts, binary dependencies or database dumps are committed.

Original and final ZIPs are retained privately at `/Users/MattTaylor/Library/Caches/trepo-jill-implementation-20260924/{serving,release}/`. To reproduce one: `python3 rebuild.py kitchen /path/to/original/kitchen.zip /path/to/new-output.zip`. All six reconstructed archives were byte-compared before this record was committed. All six originals and six final packages are also preserved in the existing private, versioned SAM artifact bucket under `trepo-incident-releases/2026-09-24-journey/`. Each object uses the bucket’s KMS encryption and was downloaded in full to verify its SHA-256 after upload. Exact object/version references are in `evidence/artifact-backup.json`. No bucket, access or runtime configuration was changed. This Git record contains source, not third-party package binaries.

## Fixes

Kitchen edits verify authoritative stored values before committing an operation receipt. Thyme routes corrections through that canonical operation, clarifies ambiguous targets and preserves compound labels. An immutable actor/household-scoped journal retains the original edit request before mutation; a fresh handler replays the original revision before target lookup. Conflicting intent is rejected. Old clients without an explicit operation ID get a new generation for a separately accepted request, avoiding permanent deduplication of later same-worded commands.

The journal uses the existing session table and permissions. It stores the exact edit body and an intent hash, with no raw transcript or bearer credential. It has no automatic TTL, matching durable Kitchen operation receipts; account/data lifecycle handling must treat it as application data. Read/write uncertainty fails closed.

Recipe repair rechecks revision/source/existence after extraction and preserves manual edits. A recipe is ready only with usable ingredients and instructions. Same-ID source/paste recovery preserves human content except explicitly confirmed replacements. Public-page recovery is bounded by URL/network/time limits.

Receipt processing separates identified items, wrong capture type and no-items outcomes. New retained-photo reuse is **OFF**, with configuration unchanged; only typed outcomes/retake guidance ship now. Activation requires a successful child-recovery-to-Kitchen native journey and complete configuration qualification. No schema, IAM, broad environment, notification or customer-account changes were made.

## Tests and limitations

See `evidence/TEST-REPORT.md`, `evidence/RELEASE.md` and the final replay handoff. Test counts overlap. Final Thyme candidate suites pass143/143/144 and13 real cross-language SQL round trips each; four retry cases fail on the original ZIP and pass after repair. Production canaries cover Kitchen readback/replay, live recipe extraction/recovery, both Thyme response modes and forced fresh-handler replay, actual async audio admission/worker completion, and receipt/pantry/blank processing.

Older iOS and Android contract tests establish the tested wire formats. They do not establish every older-client recovery screen, Android emulator UI, physical microphone/camera, hardware, push delivery or new client-stage monitoring. Photo reuse remains deferred. Separate TestFlight124 qualification is recorded in the iOS release report.

## Rollback and continuation

For a service failure, first confirm its current CodeSha256 still equals this release. Restore only that service's retained original ZIP with the current RevisionId condition and verify the previous CodeSha256/configuration. Never roll back over a newer deployment or deploy an entire dirty development folder. No migrations need reversal.

Future development should port an explicitly reviewed delta back to the canonical source paths, preserving the serving sync/stream/worker differences; then rebuild/test fresh artifacts. The original dirty checkout was preserved. Shared context: `apps/trepo/architecture`, `brand/trepo-brand-guidelines`, `brand/trepo-app-design-system`. Documentation is in Git; no automatic company-brain save was made.

## Superseding follow-ups

This folder remains the exact initial six-service record. The subsequently deployed recipe media limits are recorded in `../2026-09-24-recipe-timeouts/`; empty bulk terminal outcomes and the independently deployed GetJob reader are recorded in `../2026-09-24-empty-bulk/`. Their exact source, failed-before tests, reconstruction metadata, unchanged-configuration deployment receipts and real review-account canaries are preserved separately. The initial recipe timeout at18:22 Pacific remains historical evidence; follow-up observation windows must not erase it. Current serving rollout therefore includes seven distinct functions.
