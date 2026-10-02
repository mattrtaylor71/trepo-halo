# Private native Thyme pilot

Status: implementation and isolated native endpoint deployed on October 2, 2026; iOS build 131 release qualification is in progress. TestFlight availability is not established by this document until the release receipt is added.

Matt approved trying the new backend inside his iOS app. The audience is one signed-in actor, enforced by the server. This is an opt-in private trial with a return to current Thyme. It does not replace the public app's existing Thyme endpoint, make Zach eligible, or authorize a customer rollout.

## App behavior

In Ask Thyme, an enrolled account sees **Try new Thyme**. Selecting it opens native chat with current kitchen-backed agent tools, progress, versioned recipe cards, conversation history, explicit change proposals and recorded-voice transcription. The menu returns to current Thyme. Ordinary accounts continue using the existing UI and backend.

Voice is transcribed into an editable draft before sending. Approve/Not now applies to the concrete stored proposal, not an arbitrary natural-language “yes.” A verified applied receipt refreshes Kitchen, List, Dish Log, Meal Plan and saved-recipe caches. Calendar fetches fresh data when opened.

The app saves pending requests before transmission with one durable request ID. An uncertain response blocks another send and offers reconnection/result checking. Relaunch uses the same request ID; an unresolved mutation is never automatically repeated. A changed household drops the previous local request before starting the new scope. Local files are account-specific, protected and excluded from backups; tokens are not persisted there.

## Endpoint and security

`src/native.mjs` is a separate Lambda entry point. It uses the pinned serving Trepo JWT verifier, checks the native actor allowlist before runtime initialization, requires the signed owner to equal the actor, rejects client-supplied identity fields and resolves household membership server-side. A context hash fences every request to that actor and current household. The client rejects redirects and uses the bundled HTTPS endpoint without shared cookies.

The endpoint uses the existing private pilot Service, DynamoDB store and stream worker. The existing web API, worker code, public routes and installed older apps are unchanged by this native deployment. Async durable jobs run in the cloud independently of the laptop.

Transcription accepts only bounded mono 16 kHz 16-bit PCM WAV (0.5–60 seconds), uses a scoped request/audio hash and a 45-second lease, permits 60 recordings per actor/day, and rechecks membership after the provider returns. Its 25-second provider timeout and zero SDK retries are bounded below the Lambda/client timeouts. Transcription has no customer write capability.

## Evidence and limits

- Backend: 113 tests passed, including 13 new native-entry cases.
- Pinned real JWT verifier: six checks passed using an ephemeral synthetic test key; no customer token was manufactured.
- Live signed documented review-account token: enrollment denied, spoofing Matt's owner header denied. Missing/invalid tokens denied. No customer account writes in these probes.
- Synthetic spoken recording through the real transcription service: exact transcript returned in 729 ms. This is one observation, not a latency guarantee.
- Native UI: initial four scenarios passed (proposal approval/relaunch, acknowledgement-loss recovery, transcription draft/switch-back, ordinary-account isolation). Final full/release verification remains tracked in the release evidence folder.
- Actual microphone test on the dedicated Mac mini failed because it has output devices only: CoreAudio reported a zero-channel input and converter error -50. This is preserved evidence, not claimed as a successful recording test. The fifth case checks real recorder failure leaves text entry usable. Physical iPhone recording is unverified.
- Real Matt JWT and a harmless founder-approved live write still require Matt's interactive app use. Fixtures prove native state transitions and isolated SQL tests prove the adapters; they do not prove every real customer/device combination.

Private operational evidence: `/Users/MattTaylor/Library/Caches/trepo-thyme-native-20261002` (backend/auth/transcription) and `/Users/MattTaylor/Library/Caches/trepo-codex-release/thyme-native-20261002` (frozen native source, manifests, tests, signing and Apple receipts). No credentials or raw customer contents belong in this document.

## Rollback

Matt can select **Use current Thyme** immediately. To revoke new native access, clear the dedicated native function's allowlist using a revision-fenced config update; existing public Thyme remains unchanged. This does not cancel an already accepted worker operation. Reconcile durable applying/verifying jobs before stopping/restarting workers or reissuing any action. Preserve server records and local pending-request evidence.

## Brand and scope

Sources: `brand/trepo-brand-guidelines` and `brand/trepo-app-design-system`. Native UI uses the existing Thyme mascot and TrepoTheme palette, outlined cards, rounded headings and yellow actions; no brand exception or amendment. No automatic company-brain save, scheduled Codex task, outside-provider review or telemetry sync.
