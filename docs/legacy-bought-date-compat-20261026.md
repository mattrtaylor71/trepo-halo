# Bought-date compatibility test — October 26, 2026

Status: tested prototype; no customer rollout remains enabled. Requested by Matt: test whether existing app users can receive the corrected bought age without installing the native fix.

## Result

One live Release simulator journey passed against the unchanged product source for App Store 1.21 build 150 (`a844cd860344b1eaafd4fc3ecc1f3611f620d436`). Public Apple lookup reports version 1.21 released October 12. The release owner's records bind its submitted build 150 to this source; the release owner independently confirmed that baseline. This is a simulator rebuild of the shipped source, not execution of Apple's distribution-signed iPhone binary.

The test created one uniquely identified avocado in the documented review account, primed the old Kitchen cache, asked the real Thyme service to change its bought date to six days earlier, and verified:

- Thyme confirmed the saved date and canonical Kitchen GET returned it.
- Canonical GET retained the exact original check-in timestamp and item name.
- A legacy build-150 request received a six-day age anchor, with original check-in time separately retained in the response.
- A newer build-206 request remained unprojected. Build 206 here is a protocol test input, not a claim of a shipped release.
- The visible old Kitchen detail displayed six days immediately after closing Thyme, after switching tabs, and after a cold app relaunch.
- Exact-item deletion and absence verification succeeded. A separate database read confirmed the synthetic item was absent after teardown.

The native case ran in 108.765 seconds with zero failures. Initial and relaunch screenshots were exported; the initial screenshot was visually inspected and reads **6 days in the kitchen.** Product sources under `trepo_v0` remained byte-identical to the shipped-source commit, checked before and after testing.

## Prototype behavior and limits

`legacy_bought_date.py` is disabled by default. It projects only caller-allowlisted corrected items for explicitly eligible iOS builds and GET routes, after ordinary authorization. It converts a valid bought civil date to client-local midnight in UTC for the old age parser. It does not write the database. Pending items, invalid/future/unknown dates, unrecognized clients, newer builds, non-GET routes and errors are unchanged. Client time zone is required; compatibility responses are marked no-store.

`legacy-date-refresh.mjs` emits sanitized `tool_start`/`tool_end` events for a verified bought-date action after final validation. This repairs the older UI's reliance on streamed tool names; the brain wrapper otherwise converts those events into newer progress events. No model arguments, private draft text or result summaries are emitted by the helper.

The old app still interprets this input as check-in time. Its wording and date grouping therefore use the supplied bought date. Clearing an unknown bought date is not made into an honest unknown-age display by this bridge; that requires the proper native fix. These limitations remain even though the numeric age test passes.

The canary was narrower than a rollout: exactly review actor `8fcb37c1-0927-4c31-856a-5b761efc226c`, synthetic item `5c92790c-a7ca-4f3d-be6c-4ae31756dca4`, and build 150. Before wider activation, deployment wiring must select genuine explicit date corrections, set the reviewed supported-build boundary, and preserve exclusion of corrected clients. The preparer in `scripts/prepare_legacy_bought_date_canary.py` builds only these review-scoped packages and never deploys them.

## Evidence and cleanup

- Python compatibility checks: 7 tests passed, including multiple gate, malformed-date and timezone/DST cases.
- Node refresh checks: 3 tests passed, including failure/no-action suppression and private-argument exclusion.
- Package startup: Python compilation and actual Node handler import passed.
- Native passing run: `/Users/matthewtaylor/Projects/Trepo/app-pipeline/tests/runs/20261026-155611-bd73f9/results/personas-20261026-155614-35959.xcresult`.
- Original source manifest, coverage and runner-cleanup receipt are adjacent. The receipt confirms the simulator lock was removed.
- Exported native screenshots: `/Users/matthewtaylor/Projects/Trepo/legacy-bought-date-canary-20261026/evidence-passed/`.
- Local screenshot: `canary/evidence/legacy-six-days.png` (not committed).
- Private deployment captures and receipts: `/Users/MattTaylor/Library/Caches/trepo-bought-compat-20261026/`. Credentials and service configurations are not committed.
- Original Kitchen code hash restored and read back: `QlTAYyMlnMEmUHXfI/Ei+dR4vKquvAlqIdNUH2u1sfE=`.
- Original streaming code hash restored and read back: `BecBNWx22k15BoiqVdl48clnmyjwETuoW9J3mzVYIKM=`.
- Configuration readback matched the originals for environment, role, handler, runtime, layers, network, memory, timeout and architecture.

The first attempt built the old product successfully but could not compile the current test bundle because three newer Shop contract inputs do not exist in build 150. No user journey ran on that attempt. Its evidence is retained at run `20261026-155240-0535d0`. The isolated experimental test project was narrowed to PersonaCore and the requested Thyme journey, with all journey assertions preserved; the runner and product code were not changed. The passing run is a new source snapshot, not a relabeled earlier pass. The temporary backend packages were restored after each attempt.

The dedicated simulator is left with the tested build 150; the next serialized native job will install its own candidate. No phone install, TestFlight/App Store publication, broad customer change, or shared-brain save occurred. The proper native fix remains separately committed as `7e0762d`.

Context: `apps/trepo/screens/kitchen-and-capture`; existing Trepo UI was exercised unchanged, with no new design or brand exception.
