# Legacy URL and explicit recovery deadline follow-up, v2

Prepared September 24, 2026 (America/Los_Angeles). Implemented and locally qualified; no serving deployment or commit by this agent. Frozen v1 source/ZIP/evidence remain intact. Only `app.py` changes relative to v1; the deadline helper and configuration are unchanged.

## Confirmed gap and correction

Frozen v1's legacy synchronous URL handler lets an initial extraction `WorkLimit` reach the generic HTTP exception handler. Real bounded-provider and SQL tests reproduce a 500 with no retained record. V2 catches that limit, applies existing pure URL/platform validation, persists the normalized link through the existing save path, and permits one longer background repair.

Legacy URL clients require 200/201 recipe receipts and cannot handle a new 202 job contract. V2 returns 201 for the newly retained link and 200 for its deduped retry. Those codes acknowledge persistence, **not ready recipe content**. The recipe keeps its actual `repairing`/`failed` status and `processing`/`link_retained` outcome, with `recipe_content_incomplete` and the existing recovery details. It does not invent ingredients or steps. Failed repair dispatch terminalizes the saved link through the existing guarded helper; a failed INSERT still returns an error. A complete existing recipe remains unchanged.

Explicit source replacement and pasted-text recovery previously caught `ValueError` from their recovery deadline but not the inner provider's `WorkLimit`. Nested timers can therefore return a generic 500 even when the intended 18-second recovery boundary expires. V2 handles both timeout types with the existing actionable 422 wording: the saved recipe has not changed; retry or paste its text. SQL tests verify all personal/shared fields and the content revision remain unchanged for both actions and both native/recovery deadline precedence cases.

## Evidence

- `legacy-before.txt`: same 11 new cases against frozen v1: **8 failed, 3 passed**.
- `legacy-after.txt`: **11 passed** against v2.
- `focused-final.txt`: previous 224 cases plus these 11: **235 passed, 0 failed, 0 skipped**.
- Existing real MySQL and DynamoDB cases, provider/format tests, readiness contracts, concurrency, edit/repair/recovery guards and media-limit cases remain included.
- Candidate application files in the source test view are byte-identical to the proposed ZIP. Host Python 3.8 executed tests; Python 3.9 syntax passed. Parent owns requalification in the actual cloud runtime and the final release gates.

## Frozen artifact

- `app.py`: `9f1b885e1b796b1af7768310ddc509e1517116f1563b0591eb10700ba2c84157`
- Unchanged helper: `77432c2fbb608ec5fb1daa8e90bb37291e2af0c1b53b342e95e58eafd1161116`
- Agent ZIP CodeSha256: `kWrPSL7NY8d7d7rFeJT2uCUSqM4UTSggipB8hQ7pQmk=`
- Patch: `recipes-v2.patch`, relative to frozen v1.
- Manifest: `manifest.json`, including before/after counts and source/package hashes.

The parent may reconstruct different ZIP metadata; application entry byte parity is recorded separately. No canonical source, initial release manifest, live customer record, deployment configuration, binary, layer or permission was changed by this agent.

## Cleanup and remaining work

Restarted owned DynamoDB Local PID 63920 was stopped after verification with zero remaining fixture tables; telemetry was disabled. MySQL33317 and the parent's one native fixture schema remain running and unmodified. See `cleanup.json`.

Parent-owned remaining gates are exact cloud-runtime requalification, final archive/commit/deployment, final live canary and native Release qualification. This handoff does not claim those complete.
