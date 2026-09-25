# Journey release completion — September 24, 2026

Verified Apple availability: 2026-09-25T06:51:40.131188+00:00. **Trepo 1.17 (124) is VALID / IN_BETA_TESTING for Me and Testers.**

The seven scoped backend functions are live. Source overlays, unchanged-configuration receipts, scoped tests and canaries are committed through `b0a77b9`; this final documentation commit does not alter serving code. The initial six-function deployment, recipe deadline follow-up, and empty-bulk/GetJob follow-up remain separate historical records. No schema, IAM, layer, runtime or environment change was introduced. Retained-photo reuse remains OFF.

## Final native qualification

Final app/source harness commit `9637242b514a01de4a9726e889e2b4f654ab094a`; runtime commit `ea86505e32f94d5b2fac2f9905886894ccbd6d2a`. All 482 build/test inputs match, including 278 app inputs matching the signed archive. Full suite: 117/117 and 31 additional focused/Release executions passed; the 148 execution count includes repeated checks. A further 94 compiler-executed contracts passed. The original 124 candidate's 97-pass/20-fail run remains failed in the evidence and was never uploaded.

IPA SHA-256: `7eed127a22854af7be69f3bb0544d44d4f9a2e06046e10e0223bebc33c9c4d65`. Apple build ID: `6e1b893e-727b-425d-8bc8-f0b949b9c9aa`. Full native report, inspected screenshots and source-bound receipts are in the iOS repository at `docs/codex/testflight-124-20260924/` and `docs/codex/journey-investigation-20260924/`.

## Final pre-upload cloud observation

Checked at 2026-09-25T06:46:34.118303+00:00. Windows start at each function's own current deployment, so their durations differ. All seven code hashes and revisions match the independently verified final configuration receipts.

| Function | Observed invocations | Errors | Throttles |
| --- | ---: | ---: | ---: |
| thyme | 64 | 0 | 0 |
| thyme-sync | 2 | 0 | 0 |
| thyme-worker | 1 | 0 | 0 |
| kitchen | 2514 | 0 | 0 |
| recipes | 1070 | 0 | 0 |
| capture | 76 | 0 | 0 |
| getJob | 528 | 0 | 0 |

Selected timeout/runtime/backend-error log queries had no matches. Metrics can lag, and the selected patterns do not detect every handled error or prove client delivery. The 18:22 recipe timeout remains in the earlier deployment history; the post-fix window does not erase it. Logs and exact query references are retained in `final-qualification/`.

## Boundaries and cleanup

Live mutations were restricted to the dedicated review account and exact synthetic fixtures. The release's final native fixture-absence receipt is included. The isolated recipe fixture schema was dropped and absence independently verified; its owned server, reverse tunnel and local MySQL process were stopped, with all evidence retained. No incident customer records were modified or customer messages sent. New telemetry/alert thresholds and physical camera, microphone, Kitchen Assistant hardware, Android UI and APNs delivery are not marked qualified. Frozen iOS 1.16/1.17 and Android JVM contract results cover the tested wire formats, not all older screens. Family memory remains paused.

Source evidence and rollback are in the three release folders. Do not deploy the broadly dirty development checkout. Company context: `apps/trepo/architecture`; brand references: `brand/trepo-brand-guidelines` and `brand/trepo-app-design-system`. No brand exception or automatic shared-brain save.
