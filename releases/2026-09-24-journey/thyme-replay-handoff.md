# Thyme durable edit retry repair

September 24, 2026, 6:10 p.m. Pacific. Implemented and packaged by the delegated gstack ship test-gap audit. Parent owns rollout, commits, release documentation and TestFlight. No serving deployment, config/IAM changes or commit made by this agent.

A lost Kitchen response previously left a fresh handler resolving an old name or sending a newer revision under the original operation UUID. Kitchen correctly rejected the changed request hash. The same logical retry now reads an immutable canonical edit request before target resolution, then resends exactly those bytes through the unchanged Kitchen authorization, hash, revision and receipt checks. A later edit remains untouched; deletion is not reversed. Changed intent with the same operation conflicts before lookup/transport.

The journal uses the existing SESSION_TABLE_NAME table, with household partition and actor/operation key. Conditional Put plus a consistent Get handles concurrent preparation and uncertain write acknowledgments. Missing, unavailable or invalid journal state fails closed. Records have no TTL to match the durable SQL receipt lifetime, and contain no authorization token or raw transcript. An intent hash binds the resolved request's original arguments and transcript. Existing local request-identity code was preserved.

Each accepted legacy response-cache request receives a distinct random generation token. Sync and stream fallback mutation/correlation identities use that token rather than the session-plus-transcript key. Explicit client operation IDs retain precedence; async retains voice-job:<job_id>. Thus a later intentionally repeated legacy command does not replay an older edit indefinitely.

## Source scope

Five production files under playground12_voice_ack/trepo-quick-ack: index.js, index-stream.js, lib/data-access.mjs, lib/kitchen-edit-client.mjs, lib/session-store.mjs. Changes are present in local source and all three exact serving-derived candidates. No Kitchen code or request-hash checks changed.

Tests: new test/kitchen-edit-journal.test.mjs, updated test/kitchen-assistant-integration.test.mjs and test/fixtures/kitchen-edit-roundtrip.mjs, plus playground6_groceryRec/tests/test_thyme_kitchen_roundtrip.py. The integration harness substitutes only the DB-connection provider and HTTP transport; actual resolver, DynamoDB Local, Python canonical transaction and MySQL persistence execute.

## Verification

- Exact stream candidate: 143 tests pass, zero skipped.
- Exact sync candidate: 143 tests pass, zero skipped.
- Exact async-worker candidate: 144 tests pass, zero skipped.
- Each candidate separately passes all 13 Node-to-Python-to-MySQL round trips.
- Local focused source suite: 118 pass, zero skipped; initial local 13 database cases pass.
- Nine journal cases cover fresh handler before target lookup; field/reference/transcript conflict; actor/household scope; concurrent preparation; unavailable storage; lost journal acknowledgment; corrupted body; data minimization/no TTL; and separate accepted legacy generations.
- Four fresh-handler database scenarios failed against the original packaged code with actual old-target 404 or changed-hash 409 results, then pass on the final candidates. They cover ordinary replay, subsequent human edit, deletion and changed retry payload. Evidence: thyme-replay-fail-before2.txt; earlier environment failures remain separately retained.
- Existing sync/stream tests still prove two distinct edits execute serially with caller authorization intact; their journal dependency now uses an isolated Dynamo table.
- The existing 39-case live-model harness mocks updateKitchenItemDetails, so it cannot exercise this datastore change. Prompt/provider behavior is unchanged; no paid model evaluation was rerun.

## Packages and cleanup

release/manifest.json and final-source-package-verification.json reflect the three new ZIPs. thyme-replay-release-verification.json records final hashes and verifies every other ZIP entry is unchanged. Original pre-repair ZIPs are retained under replay-before/packages. Updated *-serving.patch files describe the final serving-derived deltas.

All generated quantity_fixture_* schemas and per-test Dynamo tables were removed by their fixtures. The temporary thyme_replay_fixture MySQL user was dropped. At the parent's request, owned MySQL33317 PID2186 and DynamoDB Local38001 PID2253 remain running for native release qualification. PID file: mysql-replay.pid. Source hashes and cleanup handoff: thyme-replay-cleanup-and-inputs.json.

Production fresh-handler replay/canary is still the parent's next gate. No physical-device, full TestFlight or retained-photo activation pass is implied.
