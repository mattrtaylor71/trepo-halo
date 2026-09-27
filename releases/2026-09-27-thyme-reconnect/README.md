# Temporary Thyme reconnect for iOS 1.17 (124)

Status: **LIVE — approved by Matt and deployed September 27, 2026 at 07:47 UTC. Public old124 text/audio, signed iOS/Android/legacy contracts, full simulator reconnect/lifecycle journey and Release smoke all passed.** Configuration and app binaries were unchanged. No TestFlight/App Store release or branch merge. See [deployment record](DEPLOYMENT.md).

## User experience

The public build omits the login credential on both Thyme streaming transports, but its existing **Check result** request includes it. The temporary bridge uses that authenticated request to establish a short-lived session on the owned streaming host.

1. An unsigned new request is rejected before account lookup, transcription, assistant execution or mutation. The stream shows: “Tap ‘Check result’ once to reconnect Thyme, then send a new message. Check your items before repeating an earlier change.”
2. The user taps **Check result**. That signed, membership-checked GET establishes the stream session through a one-use redirect. It still retrieves the original receipt and **never repeats the original command**.
3. The user can send new text and voice requests normally. The existing signed identity, current household membership, family-brain fences, durable admission and replay checks still run.

**Limitations:** Build124's generic unconfirmed warning remains for a request without a receipt; it can be dismissed manually. We do not fabricate a success receipt to hide it. Never advise blindly repeating an uncertain write. A receipt older than 24 hours cannot reconnect; use the Check result button on a new failed request. The reconnect session lasts at most 24 hours, cannot outlive the signed credential, and works across new chat sessions. The bridge automatically sunsets October11 UTC; corrected native releases are still the permanent repair. This is not a transparent/no-tap fix and does not prove every historical app build has the required signed recovery path.

## Bounded change and security

`serving-diff.patch` is the authoritative three-file change against captured serving packages. `overlay/` contains the full changed modules. The historical Git checkout is NOT a deployable full backend baseline. `package.py` overlays only two entries per Lambda and proves every other entry is byte-identical. Production configuration and database schemas are unchanged.

- Only an authenticated, membership-checked iOS recovery GET can issue a reconnect ticket. An owner ID, device string or IP address cannot authenticate anyone.
- Tickets use 256-bit random values, are stored by SHA256 digest, expire in60 seconds independently of DynamoDB TTL, and are consumed atomically once. Their stored contents are encrypted.
- Cookies are authenticated-encrypted with Node's AES-256-GCM and HKDF-SHA256 using distinct cookie/ticket purpose and exact destination origin. They are host-only, Secure, HttpOnly, SameSite=Strict, and expire server-side as well as client-side. No new dependencies.
- The original signed JWT is reverified on every use. Actor mismatch, duplicate/disagreeing cookies, invalid supplied Authorization, browser-origin requests, malformed/expired credentials, and disabled/sunset bridge all fail closed.
- Ticket redemption returns the authorized receipt snapshot obtained at issuance, within the60-second ticket window. A membership/fence change during that short interval does not revoke that already-issued snapshot; subsequent commands always resolve current access again.
- The authenticated redirect target is server configuration, never a caller-provided redirect URL. Tokens, cookies and raw tickets are not logged by the new modules.
- `THYME_IOS_RECONNECT_DISABLED=true` is an immediate kill switch. The code-only rollback is the two original ZIPs. Do not remove either original signed-operation checks or the expired-operation/unknown-commit guards.

## Verification

**41/41 exact packaged handler tests pass**, no failures/skips. Covers prior26 operation regressions plus reconnect issuance, first rejection, text/audio, account isolation, explicit bad-token precedence, header casing, browser origin, ciphertext tampering, expiry, purpose/origin binding, one-use concurrent redemption, uncertain writes, request fingerprint conflict, kill switch, sunset, and AWS's mirrored cookie fields.

**Mac mini native transport + real HTTPS:** Compiled the actual final distributed124 `APIConfig`, `TrepoAuth`, `ThymeRequestRecovery`, `VoiceAssistantService`, and `VoiceAssistantStreamService`. Only the two test endpoint constants changed; a QA Keychain accessor and no-op logging/analytics supplied the harness. The authentication, recovery and text/audio request code remained unchanged. Tested against isolated Node22 AWS candidates restricted to the dedicated review account:

- Initial unsigned request rejected with reconnect instruction.
- Signed Check result follows HTTPS redirect; Secure/HttpOnly host cookie is present. Original operation remains `not_found`, with no fictitious completion.
- New unsigned124 text and synthesized WAV requests both complete a cookie recipe using pantry context.
- Same-operation replay returns the exact saved answer; signed GET returns the same answer; render acknowledgment succeeds.
- New chat session IDs work with the established cookie.

An earlier private HTTPS run failed because AWS includes the same cookie in both `event.cookies` and the Cookie header. Fixed normalization to accept identical mirrored single values while rejecting actual duplicates/disagreement. Added the regression and reran the exact ZIP suite and real HTTPS native test successfully. The original failure was not a customer deployment.

The compiled-native transport check used synthesized voice audio. It is not a physical microphone or customer-device test. No customer credential, recipe, kitchen, list or conversation was modified or replayed.

**Full iPhone simulator UI qualification, September27:** Built the final distributed124 Release source on the Mac mini. All278 app inputs were compared against the frozen release: only the two Thyme endpoint constants differed, pointing to the isolated QA candidates. Actual UI journey passed in81.972 seconds: reproduced unsigned failure and visible reconnect instructions; tapped Check result; retained the truthful original unconfirmed warning; dismissed its pending card; generated/opened a cookie recipe; repeated after backgrounding; repeated after app termination/relaunch without another reconnect. Three new requests produced tappable cards and ingredient screens, with no new pending warning. Nine screenshots and a continuous91-second video were retained. The old red warning messages remain in that chat until it is closed; this backend repair cannot replace124's hardcoded UI.

Release smoke also passed in21.840 seconds: signed-in cold launch and all four main tabs. Both selected tests executed completely, 0 failures/skips, verified remote status and SHA256 evidence. This is a source-equivalent simulator qualification, not the encrypted App Store binary or all historical versions. [Simulator report and video](/Users/MattTaylor/Library/Caches/trepo-thyme-reconnect-20260927/SIMULATOR-REPORT.md).

Independent in-host gstack adversarial review found no remaining actionable authentication/replay defect after the cookie and reconnect-hint corrections. Test fixtures were reviewed in summary mode; outside-provider reviews are disabled by company policy. The bounded receipt-snapshot window is documented above.

Private evidence: `/Users/MattTaylor/Library/Caches/trepo-thyme-reconnect-20260927/`. It contains ZIP identity/hash manifests, exact handler test log, native source hashes and HTTPS results, and cleanup readback. Both temporary QA functions/URLs were deleted and absence verified. Serving production hashes AND revisions remained unchanged; both functions stayed Active/Successful.

## Release procedure used after approval

1. Confirm exact serving hashes and revisions still match `candidate-manifest.json`; if changed, stop and rebase/requalify only the overlay.
2. Verify candidate ZIP hashes, qualification manifest and native test result.
3. Deploy the **stream candidate first**, then the sync candidate, with revision fences and no configuration changes.
4. Use only the dedicated QA account for public verification: reject unsigned request, signed Check result, encrypted-cookie text/audio completion, exact replay, saved-result recovery and render acknowledgment. Also verify corrected signed iOS/Android clients. Do not replay a real customer's ambiguous commands.
5. If public verification fails, restore both captured original ZIPs with revision fences. Preserve the release receipt and report the outcome accurately.

Shared context consulted: `apps/trepo/architecture`, `brand/trepo-brand-guidelines`, and relevant historical recovery records. Current operational conclusions come from frozen release source and AWS tests. No approved brand exception or visual change; no company-brain write was requested or performed.
