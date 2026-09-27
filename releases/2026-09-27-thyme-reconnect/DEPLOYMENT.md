# Approved backend release — September 27, 2026

Matt explicitly approved the tested temporary workaround in this Codex conversation after the simulator report. The stream package was updated first at 07:47:29 UTC, then the recovery package at 07:47:40 UTC. Both updates used the tested ZIPs and current revision fences. Function configuration, IAM, database schemas and app binaries were unchanged.

| Serving function | Approved code SHA256 (base64) | New revision |
| --- | --- | --- |
| trepo-quick-ack-stream-dev | V9I7HDLrpzCJIbJPNPBHZqae0qzbC1eHhjyaqbW8F5Q= | 931198c5-32b6-4d3b-ae4d-5a03d4d3e542 |
| trepo-quick-ack-sam-dev | SRgbVQOH1fjgbe654z8B2mY4wJD6P8QI7QtXUn13nNQ= | c42bbb25-7752-4ab2-8a4f-bf5431a02b09 |

Exact readback confirmed both Active/Successful with the approved hashes and revisions. The previous whole-package rollback ZIPs remain retained; no rollback was needed during the completed HTTP checks.

## Production verification

Using only the dedicated review account, the unchanged frozen 124 networking source passed on the Mac mini against the real public endpoints:

- An unsigned initial request is rejected and shows the reconnect instruction.
- Signed Check result follows the live API Gateway/Function URL redirect and stores the Secure/HttpOnly host cookie. The original request remains not_found and is never replayed.
- New old 124 text and synthesized-audio requests complete. A replay with the same operation returns the identical answer, signed recovery returns that answer, and rendering acknowledgment succeeds. New chat session IDs work.
- Additional signed public HTTP contract checks passed for current iOS, Android, and the legacy iOS ordinary correlation-UUID path. Android recovery stays on its original endpoint; legacy IDs are not forced into the v1 receipt protocol. These are protocol checks, not a native Android UI test.
- The bounded post-deploy log scan through 07:50:29 UTC found 0 matching runtime/error signals and 0 timeout entries across both functions, with all pages read. This short window is not a guarantee about every customer or subsequent traffic.

The initial manually constructed preflight session omitted the hyphen that real iOS session IDs contain and was correctly rejected 400. Using the exact shipped UUID-prefix format returned 200/not_found. No production code was changed to relax that validation.

The additional full public-endpoint iPhone simulator qualification **passed**, run `20260927-004909-d5527d`. All 278 app input files were byte-identical to the final distributed 124 source, including the original public endpoints. The reconnect/background/cold-launch journey passed in 76.994 seconds; Release smoke passed in 22.208 seconds. Both selected tests executed with zero failures, skips or missing cases. The reconnect instructions, completed chat and recipe after cold launch were visually inspected. Fourteen screenshot attachments and a continuous screen recording were retained. This is a Release simulator build of the original source, not an encrypted App Store binary or a physical microphone test.

[Production simulator recording](/Users/MattTaylor/Library/Caches/trepo-thyme-reconnect-20260927/live-simulator-proof.mp4). The retrieved evidence hashes and remote exit status were verified before completion. `deployment.json` and `verification.json` retain the sanitized release receipt and test summary beside the exact runtime overlay.

Prior candidate qualification remains 41 exact handler tests, a full old 124 reconnect/background/cold-launch journey, and Release smoke, all passing. The production suite above is additional evidence, not a restatement of the earlier isolated-endpoint run.

## User-facing limits

Affected 124 users tap **Check result** on a new failed request, dismiss its old pending card, then send a new message. Existing red warnings remain in that chat until it closes. Do not blindly repeat uncertain earlier list/kitchen changes. The session lasts at most 24 hours and cannot outlive its signed credential; the temporary bridge sunsets October 11 UTC. The native Authorization fix remains the permanent repair. No TestFlight or App Store release was performed here.

No customer command was replayed and no customer records were edited. Evidence is under `/Users/MattTaylor/Library/Caches/trepo-thyme-reconnect-20260927/`: `release-approval.json`, `deployment-receipt.json`, `live-readback.json`, `live-native.json/log`, `live-signed.json`, `live-canary.json`, and the earlier exact-package/QA reports. The local test credentials are not included in this repository or report.

Applied gstack release readiness, exact-artifact verification, rollback and canary principles to the already-approved direct Lambda artifact release. GitHub PR merge/CI-auto-deploy steps are inapplicable to this deployment path; no branch was merged. No automatic artifact sync, external-provider review, or company-brain write was performed. Shared source consulted: `apps/trepo/operations`, a dated runbook; live claims above come from AWS readback and direct tests.
