# Thyme runtime and visible task progress

Date: October 24, 2026. Author: Codex, requested by Matt Taylor.
Status: implemented in isolated branches; local contracts, live synthetic inference and focused simulator validation. Not deployed, enabled for customers, installed on a phone, or uploaded to TestFlight.

## What changed

Both Thyme entry points now use the same Responses transport, task classification, focused tool selection and public activity vocabulary when the runtime pilot is enabled. The legacy adapter retains its existing authenticated operation journal, tool executor, household-brain validation and receipt reconciliation. Native Thyme retains its durable requests, scoped tool gateway, recipe cards, approval cards and verified apply path. These adapters deliberately preserve their existing product behavior; this is not a replacement of every backend with a new hosting platform.

The default is GPT-6.1 Sol with low reasoning. Quick requests use a 45-second turn budget; native longer tasks use up to 150 seconds per invocation. Provider calls have a maximum 35-second deadline covering headers and body. Native requests checkpoint provider output and call IDs before tool execution, reuse deterministic tool receipts on recovery, bound stored context, and stop at a total task-round cap. No blind provider retries were added. Legacy long requests retain their prior runtime.

New inference is off unless both `THYME_RUNTIME_V1=1` and `THYME_RUNTIME_ACTORS` contains the exact authenticated actor in a JSON array. A wildcard does not enable everyone. In native Thyme, disable the flag to return to the existing Agents API; the next turn rehydrates conversation history, including direct-runtime messages. An in-flight request retains its original route and must finish or recover normally.

Public stages include checking the kitchen, looking through history, finding recipes, preparing changes for review and checking the result. They are fixed server text, not model reasoning or raw tool output. No invented percentage or “nearly finished” timer is exposed. The brain wrapper still withholds unvalidated answer drafts while forwarding safe activity.

The app replaces the three-dot wait with a Thyme task card. Native tasks can continue while the app is backgrounded, and completed steps remain available under “See Thyme’s steps.” The app says that a task is saved only after the durable server session exists. Reduce Motion uses a still track, and animation pauses while the scene is inactive. Old servers remain compatible through optional fields and a fallback label.

## Integrity and recovery

- Membership, food-preference and memory fences remain in place before tool calls and final publication.
- Native account writes still require the existing visible approval, followed by domain-specific readback. A model action claim cannot substitute for a receipt.
- For questions about an earlier approved change, confirmation is rewritten from the verified receipt's exact summary and explicitly identified as historical. It cannot authorize a new write or validate a different item named by the model.
- Provider reasoning/call items survive tool continuation without exposing them to the app. Worker recovery reuses tool receipt IDs rather than repeating completed work.
- Interrupted tools display a failed step rather than a successful checkmark. Round-budget recovery progresses from its saved state instead of getting stuck at the old limit.
- Existing brain updates remain downstream of actual writes; no extra brain rebuild is inserted into answer delivery.

## Measurement

Legacy client traces correlate a stable request ID with monotonic request start, headers, first progress, first text, final answer and visible-result acknowledgement. URLSession reports DNS, connection, TLS, upload, header wait, redirects and connection reuse. Server logs correlate acceptance, ready input, household context, model rounds and publication. The trace contains timings, not message/audio contents, authorization tokens or URLs.

The original breakfast incident had about 54 seconds from the phone operation timestamp to the server receiving its asynchronous render acknowledgement, while the model loop took about 4.5 seconds. The new implementation does not establish the cause of the pre-/post-server gaps. A current-phone Wi-Fi/cellular, text/voice, cold/warm and background test is still required; simulator/backend timings must not be presented as phone p50/p95 or a proven fix for that incident.

## Verification

- Shared runtime plus native contracts: **136 passed**, no failures/skips. Covers route isolation, household changes, encrypted reasoning continuation, proposal-only writes, verified historical receipts, interrupted tools, worker recovery and runtime rollback history.
- Legacy brain/receipt suite: **14 passed**, including enabled Responses streaming and write receipt validation. Non-opted-in legacy behavior is also covered.
- Swift production-method recovery harness: **97 passed**. Explicit macOS 15 target and installed Xcode SDK were used; this is a production-method contract test, not a physical-device run.
- The new simulator task journey passed in `20261024-133644-d34bd9`: real task labels, background/reopen without resending, completed answer and expandable step history. Original screenshots were visually inspected.
- Focused final simulator contracts: **9/9 passed** in `20261024-134024-57b0d3`; native chat journeys: **8/8 passed** in `20261024-134104-04583a`; cold-launch/all-tabs smoke: **1/1 passed** in `20261024-134554-520c06`. All three final receipts confirm complete coverage, valid evidence and cleanup. See the companion iOS `REPORT.md`.
- Three candidate Lambda ZIPs were rebuilt from SHA-pinned current serving archives. Source hashes and actual packaged entry imports passed; the deployed consent guard is preserved and test-only consent fixtures are excluded. No Lambda update was issued.

Live synthetic tests used the real provider with fixture household data and zero customer reads/writes:

| Full native path | Elapsed | Outcome |
|---|---:|---|
| Condiments | 1.521 s | Actual mustard; no invented quantity |
| Breakfast | 8.653 s | Complete canonical recipe card; 2 tool calls |
| Add milk | 2.850 s | Pending approval; no applied account change |

A separate single-inference comparison checked Luna and Sol on three small scenarios. Luna attempted a shopping proposal during a read-only breakfast request. Sol passed all three bounded scenarios (1.4–2.2 seconds per call). This small sample supports retaining Sol as the default; it is not a broad model-quality benchmark. Independent-provider candidates were researched but not benchmarked because their credentials were not configured.

Three native adversarial review passes covered both production diffs. All concrete findings were corrected; the final correction handles ISO receipt timestamps when selecting the latest verified historical change. Targeted reproduction and all 136 backend contracts passed afterward. The final review is recorded `completed=true, converged=false` because that third pass contained a fix; it is not an untouched-tree clean certification. Outside-provider reviews were disabled, and release integration review remains required.

## Integration and controlled rollout

Backend branch: `codex/thyme-runtime-20261024`, based on `42cdfa7`, a scoped import of the current deployed Thyme sources. The earlier repository checkout was stale; review the runtime delta against this baseline.

iOS branch: `codex/thyme-task-progress-20261024`, based on `75a948dc47dad610e11f18ccc7a8c91c3957fc40`. It lives separately from the active release checkout. The companion report contains simulator receipts, screenshots and physical-device gaps. Its default Git remote points to another working checkout and must not receive a blind push.

Release order: inspect current serving hashes again, rebuild the three scoped overlays against approved fresh bases, deploy with the runtime flag off, integrate the app commit through the normal release owner, then enable exact Matt/Zach actor IDs in both the native ingress and worker plus the legacy stream environment. Confirm accepted/first-progress/render traces on the phone before broadening. Preserve AI-consent configuration and existing receipt/data tables. Disabling the flag is the runtime rollback; do not delete request or proposal records.

Local candidates and evidence: `/Users/MattTaylor/Library/Caches/trepo-thyme-latency-20261024/candidates/verification.json`, `runtime-native-tests.log`, `runtime-legacy-tests.log`, `runtime-model-canary.jsonl`, `runtime-native-canary.jsonl`, and `ios-final-tests.log`. Other files in that private audit directory include live configuration secrets: do not copy or commit the directory wholesale.

Reproduction commands (from the backend root):

```sh
node --import ./shared/thyme-runtime/test/register.mjs --test shared/thyme-runtime/test/runtime.test.mjs thyme-agent-v2/test/*.test.mjs
```

From `playground12_voice_ack/trepo-quick-ack`:

```sh
node --import ../../shared/thyme-runtime/test/register.mjs --experimental-test-module-mocks --test test/brain-write-confirmation.test.mjs
```

The test import replaces Lambda's absolute consent-guard path only inside the test process. Candidate packaging never includes that fixture. Live canaries require `OPENAI_API_KEY` through a private environment; scripts do not retrieve or print credentials.

## Design sources and limits

Shared context: `apps/trepo/screens/account-and-assistant`, the historical Thyme audit, `brand/trepo-brand-guidelines`, and `brand/trepo-app-design-system`. Current deployment source/logs, not historical memory, established serving behavior. The UI reuses TrepoTheme and the existing Thyme mascot, with no recreated wordmark or new brand exception.

No production deployment, customer flag change, notification, external message, shared-brain save, TestFlight upload or physical-device validation occurred in this implementation task.
