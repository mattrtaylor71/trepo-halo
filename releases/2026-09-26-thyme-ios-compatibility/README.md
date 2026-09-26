# Thyme iPhone session compatibility — September 26, 2026

Status: **Fixed and qualified; not deployed. Awaiting founder release approval.**

## Incident and root cause

A customer reported repeated interrupted requests and a failing “Check result” action. The shipped iOS 1.17 (125) source creates sessions as `chat_` followed by the first 12 characters of a UUID (for example `chat_01234567-89A`) and sends a v1 operation receipt identity. The September 25 backend operation-recovery overlay allowed only complete UUID session IDs. It rejected iOS text and audio before assistant admission. The GET recovery/render-ack bridge independently rejected the same session format.

A real production streaming request with the iOS format returned `invalid_session_id` before authentication/assistant execution. A signed dedicated-QA recovery GET returned HTTP400 `invalid_request_identity`. These reproduce the two screenshot error paths. The early rejections do not contain customer IDs in the assistant outcome logs, so the specific customer's individual failed operations were not traceable there; do not claim they were replayed or repaired.

## Repair

One shared bounded validator accepts the two actual shipped formats: full UUIDs and 17-character iOS chat IDs. Admission, recovery and render acknowledgments use it. Preserve the exact session string; do not normalize historical keys. Signed actor checks, current household membership, expiry, fingerprint conflict checks, conditional admission, response storage, and uncertain-operation replay protection are unchanged. Plain correlation-only older clients retain their legacy path.

`serving-diff.patch` is the authoritative small diff against live code. The Git base predates the already-deployed recovery overlay; the two full modules are recorded here from verified serving artifacts with only that patch applied. Do not deploy this entire historical checkout. `package.py` replaces exactly those two ZIP entries in each captured serving archive and asserts that every other entry is byte-identical.

## Verification

- Exact serving handler suite: 19 existing checks pass, all 7 new iOS compatibility checks fail.
- Patched exact handler suite: all 26 pass. Includes text/audio, sync/stream, result recovery, render acknowledgments, concurrent admission, replay, uncertain completion, malformed sessions, unsigned/wrong-owner/wrong-session requests and legacy compatibility.
- AWS private candidates under Node22: four real-model cases pass (iOS/Android × sync/stream), each with exact saved-answer recovery, cross-transport replay, authorization checks, and render acknowledgment. Dedicated QA account only; read-only assistant prompts. Temporary functions deleted and absence confirmed.
- Mac mini: the shipped Swift recovery storage and transport compile and pass 34 native contracts using actual iOS-form session IDs. This is not a simulator UI or customer-device test.

Raw evidence and rollback ZIPs are private at `/Users/MattTaylor/Library/Caches/trepo-thyme-stuck-20260926/`. Committed reports contain no credentials or customer transcripts. `candidate-manifest.json` pins both serving and candidate hashes. Production configuration and family-memory/Android code remain outside the change.

## Review

Gstack critical/checklist review and an independent in-host adversarial review found no actionable issues in the two-file production diff. The independent reviewer checked exact-session preservation and downstream consumers; test fixtures were reviewed in summary mode. Outside-provider reviews are disabled by company policy.

## Release and follow-up

Approval is required under the founder's existing release rule. On approval, recheck both live code hashes/revisions, update code only with revision fences, then verify public sync/stream/GET/ack paths with the dedicated QA identity. Keep the original ZIPs for rollback. Do not blindly resend customer requests: old unconfirmed cards remain until the user checks or dismisses them; this repair restores new requests and valid saved-result recovery, not automatic replay of failed commands.

Shared context consulted: `apps/trepo/architecture` and the dated Thyme reliability pages under `apps/trepo/sources/20260923/ios/docs/codex/thyme-audit-20260917/`. Historical documents inform scope; current endpoint/package checks establish this incident. No company-brain write was requested or performed.
