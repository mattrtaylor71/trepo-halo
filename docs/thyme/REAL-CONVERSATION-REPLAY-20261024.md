# Thyme real-conversation replay — October 24, 2026

Status: implemented and tested in `codex/thyme-runtime-20261024`. Not deployed or enabled for customers. No TestFlight build was created in this work.

## What was tested

Reviewed 1,083 user turns across 241 household identifiers in the legacy conversation store, dated October 1–24. Excluded identifiable automated canary traffic when selecting the corpus. Selected 50 real requests from 26 accounts; 24 include their actual preceding conversation messages. The selected cases cover kitchen contents, recipes from current stock, partners' saved recipes, dietary constraints, substitutions, short follow-ups, shopping lists, mixed removal/addition requests, meal-calendar changes, dish logging, purchase history and canceled actions.

Replayed the actual candidate Service → Runner → ToolGateway → OpenAI Responses path using `gpt-6.1-sol`, with low reasoning effort. Fresh read-only household snapshots were captured through the deployed gateway. The primary snapshot was captured at 2026-10-24T21:27:44Z; exact purchase-history responses were captured separately afterward. Current consent checks remained active before provider calls.

Each case runs in an isolated MemoryStore. The snapshot gateway checks household scope and throws on a write. No customer list, calendar, kitchen, preference or dish record was changed. Review proposals remain unapplied. The browser viewer is loopback-only, behind a random URL path, with a strict route allowlist and no-cache responses. Raw conversations, results, deployment configuration and credentials are excluded from Git.

This is an independent-turn replay with real prior messages, not a new model-generated conversation chained through all turns. It evaluates present answers against captured current data; it cannot reconstruct the inventory that existed when the original question was asked.

## Problems found and repaired

| Problem | Resulting change |
| --- | --- |
| Keyword-based tool selection hid dish-log or kitchen-removal tools in mixed requests. | Keep the supported tool catalog available; authorization and review boundaries remain in the gateway. |
| “Saved recipes I checked” was mistaken for a claim that a recipe had been saved. | Bind the completed-action predicate to the matching action, preserving legitimate read-only answers. |
| “Nothing has been added” was mistaken for a successful account mutation. | Recognize negative subjects and “rather than” narration without weakening checks on positive action claims. |
| Large kitchens and saved libraries produced oversized repeated reads or timeouts. | Compact initial inventory; use bounded, searchable pages with exact continuation offsets. Explicit detail reads preserve item dates, product information, nutrition and storage guidance; saved recipe detail preserves its method. |
| Successful history responses appeared as failed steps because their evidence envelope omitted `ok`. | Normalize success only for the expected schema and matching actor/household. Preserve actual errors and reject foreign evidence. |
| Generic completion labels concealed which resource was read; unrelated events could prematurely check off a tool. | Retain resource-specific labels and finish tool steps only on their own result. |
| Initial setup and inference emitted duplicate “Putting your answer together” stages. | Give household reads one stable stage, completed after the reads, followed by the answer stage. |
| Broad cooking questions triggered avoidable intake questions; historical dish logs could be prepared for the wrong day. | Give a useful stock-based starting point; preserve mixed intent; explicitly reject unsupported backdating instead of silently logging today. |
| Output could use tables unsuitable for a narrow chat. | Prefer short paragraphs and labeled lists; code blocks only for explicitly requested copy-paste text. |

The larger multi-meal requests also receive the task budget. Tool definitions remain explicit; no provider-hosted discovery loop is required.

## Verification

- All 50 latest recorded cases completed and passed structural checks: nonempty answer, mobile-friendly format, recipe structure, no unresolved template/raw payload, no false generic action rewrite, no live writes, unapplied proposals, and no running final steps.
- 43 cases have additional explicit intent expectations, such as using saved-library/history tools, respecting cancellation, retaining ambiguous-item clarification, and not preparing a current-day dish entry for a historical request. All passed.
- Reviewed answer grounding against captured stock, saved recipes and receipt evidence. Purchase answers distinguish upload dates from purchase dates and do not invent currency. Saved-library answers use actual returned recipes. Missing inventory and unsupported actions are stated rather than silently fabricated.
- Browser QA rendered all 50 final answers at 375px with larger text: no horizontal overflow or empty answers. Opened all 13 generated recipe cards: each had ingredients and steps and no dialog overflow. Verified Quick hop, reduced motion, final answer transition and zero browser warning/error logs.
- 146 native/runtime regression tests and 24 legacy action-confirmation tests passed. Coverage includes household membership changes, consent/recovery boundaries, failed reads, review/apply receipts, uncertain writes, false action claims, pagination and detail preservation.
- Packaged the native candidate over a SHA-pinned fresh deployed ZIP and verified every overlaid file against source. Unchanged deployed files and the actual consent adapter were preserved. Packaging does not deploy or enable the runtime.

The latest private assessment records a median of 5.7 seconds, p95 of 15.6 seconds and maximum of 18.2 seconds. These are inference-plus-local-snapshot timings, not mobile-network or live database latency. Repeated runs were used to expose variation; failures remain in the private evidence and only explicitly rerun cases supersede them in the latest view. Focused reruns covered the last action-claim and detail-read changes; their updated browser results were rechecked as well.

## Limits and release boundary

- Production configuration still has the earlier runtime; this candidate is opt-in and remains off. Both legacy and native deployed backends were inspected, but the 50-case inference replay exercises the native candidate path. Legacy action-claim regressions were separately tested.
- Browser rendering is not an iPhone simulator/device test. The native client still needs its release QA and Quick hop design integration. The existing design prototype remains a set of scripted scenarios; the separate replay viewer displays recorded real agent outputs.
- Customer mutation application was intentionally not performed. Existing domain/recovery tests verify application behavior; a release still needs a controlled test-account end-to-end approval check.
- External web search/import and background job completion were not exercised live by the snapshot gateway. A missing captured source is reported as unavailable, never empty. One earlier web-search attempt correctly hit that explicit limitation; the model did not claim to have retrieved web results.
- The current tools cannot backdate dish logs or persist aisle sorting. Tested responses explain these limits. This work does not add those capabilities.
- These checks improve the tested failure modes; 50 examples cannot prove correctness for every possible future request. Date/timezone boundary journeys and continuous multi-turn interactions remain part of release QA.

## Reproducing the checks

Run the deterministic suites from this checkout:

```sh
node --import ./shared/thyme-runtime/test/register.mjs --test shared/thyme-runtime/test/runtime.test.mjs thyme-agent-v2/test/*.test.mjs
```

From `playground12_voice_ack/trepo-quick-ack`:

```sh
node --import ../../shared/thyme-runtime/test/register.mjs --experimental-test-module-mocks --test test/brain-write-confirmation.test.mjs test/action-claim-guard.test.mjs
```

`scripts/thyme/replay-conversations.mjs` requires absolute private corpus/output paths, a scoped inference credential, and the real deployed consent adapter. Do not use the unit-test consent stub for customer replays. Each output file is created exclusively with mode 0600; it contains customer data and must remain outside the repository.

`scripts/thyme/assess-replays.mjs` evaluates private JSONL evidence without inference or network calls. Supply `--corpus`, comma-separated chronological `--results` files, and a new `--output` path. Optional case expectations live in the private corpus. It exits nonzero on missing cases or failed assertions.

Brand references: `brand/trepo-brand-guidelines` and `brand/trepo-app-design-system`. The viewer reuses the original Thyme mascot, cream/teal/yellow palette, outlined surfaces and the founder-selected Quick hop motion. This local browser QA view uses system rounded typography as the existing prototype does; it is not an approved change to corporate typography or the native design system. No brand amendment or shared-brain save was made.
