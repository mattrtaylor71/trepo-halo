# Customer journey failures: code investigation and proposed repairs

September 24, 2026 · Prepared for Matt Taylor · Internal engineering plan

**Status: investigation, implementation and targeted qualification complete; controlled production and capture activation gates remain. See IMPLEMENTATION.md and TEST-REPORT.md for current evidence. No incident fix deployed.** The current TestFlight 1.17 (123) release is separate and does not fix these issues. This investigation changed no customer records, sent no messages and made no production configuration changes.

The supplied [failure analysis](/Users/MattTaylor/Documents/Codex/2026-09-20/halo-customer-experience/outputs/jill-experience-audit-2026-09-24/FAILURE-ANALYSIS.md) is substantially supported by the code. The central problem is that a successful tool call, upload or saved row does not necessarily mean the user's requested task succeeded. Fix the write contract and completion criteria before trying to solve this with more prompts or a family-memory service.

## What I independently verified

- Read the deployed Thyme, Kitchen and Saved Recipes source retained with the incident evidence; checked current iOS source separately.
- Rechecked the three live function hashes at **1:09 p.m. Pacific**. All match the report's serving versions: Thyme `vzhdBuD6…`, Saved Recipes `fx6QVMX0…`, Kitchen `vCo7hLgj…`. The failures are not explained by a stale source copy.
- Ran **21 offline characterization scenarios on the Mac mini: 17 voice/guard checks and four recipe-repair checks**. These reproduce current behavior with synthetic rows and substituted storage/extraction dependencies. They are not passing acceptance tests for a fix, live model evaluations or proof of what appeared on the customer's phone.
- Applied the installed **gstack-investigate** workflow: evidence, source trace, hypothesis, executable reproduction and repair/test plan. Recalled `apps/trepo/architecture` from `trepo-company`; operational claims use the source artifacts and fresh function metadata, not memory alone.
- No independent staging environment was established. Services named `dev` are serving production. Future qualification must distinguish isolated tests, dedicated review-account checks and physical-device checks.

### Further findings from the code

1. **Argument normalization is only part of the rename failure.** The kitchen normalizer drops unsupported names and a name-only edit fails validation. However, the full validator subsequently spreads the original arguments back into the payload. If another accepted field such as brand is present, `new_name` survives that merge. The writer still ignores it. The fix needs a real schema field, validation and writer support, plus a strict per-tool allowlist; merely changing the normalizer is insufficient.
2. **The pantry retry guard also misses an explicit command.** The exact deployed guard returns no recovery tool for “Add beef broth to my pantry”; it recognizes “Check in beef broth.” Its kitchen regex recognizes `check in`, `stock/restock` and a narrow `add…kitchen` form, but not ordinary add-to-pantry phrasing. A normal model turn can still succeed, as the incident shows. This is specifically a corrective-routing gap, not proof that all explicit pantry requests fail.
3. **Rename confirmations lack coverage in the action-claim guard.** The exact detector does not recognize “Renamed Garlic Oil to Garlic Olive Oil.” Existing guard domains cover calendar, dish logging, shopping adds, kitchen adds and removals. The verified mutation result must become the authority for edit/rename confirmations too; a broader wording regex alone cannot validate a write.

The first two refinements were discovered when initial probe assumptions failed. The probes were corrected to describe observed behavior; no application code or acceptance assertions were weakened.

## Proposed implementation order

| Order | Change | User outcome | Priority |
|---|---|---|---|
| 1 | Real rename + verified transactional writes + correction field boundaries | A correction changes the intended name once, preserves everything else and confirms the saved value | P1 |
| 2 | Protect recipes from late/background repair | A background retry cannot erase manual work or replace a newer source | P1 |
| 3 | Honest recipe completion + source replacement | Users know whether they have a usable recipe or a retained link requiring action | P1 |
| 4 | Bounded pantry-entry context + better corrective routing | Follow-up products/quantities work without repeating the whole request | P2 |
| 5 | Compound-name handling + wrong-mode capture recovery | One product remains one product; an accidental receipt-mode scan can recover using its original photo | P2 |
| Throughout | Outcome verification and request correlation | Monitoring measures completed user tasks and exposes repeat corrections | P1 for write verification; P2 for broader reporting |

Ship these as separate, narrowly scoped changes with their own evidence. Do not deploy the broadly modified backend checkout as an incident fix.

## 1. Real corrections that preserve unrelated fields

**Confirmed source:** [tool schema](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/shared/voice-assistant/tool-definitions.mjs:310), [detail validation](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/shared/voice-assistant/tool-executor.mjs:387), [generic argument merge](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/shared/voice-assistant/tool-executor.mjs:415), [writer](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs:4020).

Add `new_name` to `update_item_details`, explicitly distinct from the lookup `item_name`. Resolve and carry a stable `item_id`; map only the validated new name to `product_name`. Name-only edits become valid. Reject unknown mutation fields rather than allowing the generic argument spread to carry them forward. Omitted/null fields must not accidentally become quantity zero, empty brand or closed state.

Persist a small pending correction record after an ambiguous request: authenticated actor, household, conversation, target candidates, proposed field patch, expected revision, creation/expiry and originating operation ID. “Garlic Oil” or “most recent” resolves the target without losing “Garlic Olive Oil.” Scope it to the correct account/session and expire or cancel it when the user changes tasks. Start with a bounded lifetime and tune it from evaluation; this is task state, not long-term family memory.

While the pending operation is a rename, permit only the name field unless the user clearly requests additional changes. “Drizzle it” must not open, consume, move or delete the bottle. “I opened that bottle” must still work through the explicit opened-state operation. Do not rely solely on a prompt asking the model to be careful.

**Use the canonical Kitchen edit transaction.** The deployed voice writer performs direct SQL and returns the old row plus proposed fields; the synthetic protected-value probe reproduces a response claiming a new brand while storage still holds the old one. Existing [Kitchen operations](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/serving_amount_operations.py:129) already provide revision checks, protected overrides, operation identity, transaction rollback and authoritative readback. Reuse that implementation through a narrow trusted internal adapter rather than creating another independent SQL mutation path in Thyme. Derive actor/membership from authenticated server context; never accept model-supplied owner or household overrides. If an internal Lambda invocation is used, keep it IAM restricted and unreachable as a public body-controlled bypass.

The adapter should:

1. Lock/revalidate the active item and its revision; reject deleted/foreign/stale targets.
2. Update protected edit metadata and the item together. Preserve compatibility projections required by the actual serving configuration.
3. Read back authoritative fields, compare them to the normalized authorized patch, and commit only a matching result. A mismatch rolls back and reports a conflict/failure.
4. Store the result under a stable operation ID; the same retry returns the same result. Reusing the ID with a different patch conflicts. A model tool-call ID regenerated on retry is not a sufficient idempotency key.
5. Return the confirmed item/revision and an applied/no-change/conflict outcome. Produce the confirmation from these values, never from the requested payload.
6. Use the existing cache, recipe-availability and Shelf Life invalidation hooks. A legitimate opened/storage change must preserve the agreed lifecycle rules; a name-only correction must not reset admission/opening clocks or move an existing shelf deadline incidentally.

**Acceptance:** replay all six failed correction attempts across the four products, including target selection, “most recent” and spelled names. Assert persisted names, unchanged quantities/brands/storage/opened dates, shared reads, immediate app UI, refresh and relaunch. Add protected-field, stale revision, deletion, foreign household, no-op, identical retry, changed-payload retry and forced persisted-mismatch cases. Run both streaming and synchronous Thyme paths, plus the Kitchen Assistant path if the shared tool change reaches it. Do not broaden ambiguous device removals as part of this fix.

## 2. Make background recipe repair non-destructive

**Confirmed source:** [_repair_saved_recipe](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/serving_saved_recipes_app.py:5224). Its comment promises not to downgrade, but the empty-extraction branch can clear ingredients and instructions. It reads once before slow extraction and writes without checking whether the user edited the record meanwhile. An ingredient-only extraction can also be marked ready.

All four risks were reproduced with the actual function and synthetic storage. **This does not establish that a repair erased this customer's manual edits**; her recorded repair preceded those edits.

Capture a recipe content revision, source identity and existence before extraction. Perform network/model work outside the database transaction. Then lock/re-read the row and compare those values before applying a result. A changed revision/source, deleted record or already-superseded job produces a skipped result. Never recreate a deleted save. Update the serving legacy/shared representations in the same required transaction.

An empty extraction is not evidence that existing content was fabricated. Preserve stored content and report the new extraction's limitation. Track per-field provenance for subsequent machine versus user edits; do not infer provenance from list length. A longer ingredient list must not replace newer user-edited instructions or silently combine unrelated versions. Use counts only as one completeness signal, not a write-authority rule.

**Acceptance:** late empty repair after manual edit; richer repair after newer source replacement; deletion during extraction; duplicate worker delivery; projection failure/rollback; partial extraction; real complete recovery. Assert human content and deletion survive every ordering. Run the existing saved-recipe extraction, edit, sharing/Own scope and async suites before any narrowly packaged deploy.

## 3. Separate a retained link from a ready recipe

**Confirmed backend:** [save state](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/serving_saved_recipes_app.py:6310), [manual edit](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/serving_saved_recipes_app.py:7094).

**Current app:** [job completion](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/RecipeLinkSaveManager.swift:265) calls `finishSuccess` from job `completed`; [synchronous save](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/RecipeLinkSaveManager.swift:477) also returns success from a valid 2xx reply without checking recipe readiness. `finishSuccess` emits `recipe_saved` and can notify that the recipe is ready. The [model](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/SavedRecipesAPIService.swift:30) retains a status string, but that does not govern these confirmations. This is current-source evidence, not a claim about her exact installed build.

Keep the existing `repairing`, `ready`, `failed` wire states and job protocol. Add optional structured outcome/recovery information without breaking older decoders. Define one readiness policy shared by initial save, repair and user edits: grounded usable content, including ingredients and preparation directions where the source requires them. Title-only or a URL pasted as an ingredient is not enough; source-backed simple recipes must not be rejected just for being short.

For updated clients use three clear presentations: processing; recipe ready; link retained/needs recipe content. Keep the retained link visible and offer **Replace source**, **Paste recipe** and **Open original**. Source replacement should be an explicit server operation using the same recipe identity, source deduplication, validation and revision checks; do not silently crawl arbitrary URLs found in ingredient text. Reevaluate readiness after complete manual edits, including when a previously failed recipe becomes usable. A title-only edit must not flip it ready.

If following a clearly marked recipe article from a social note, keep traversal bounded, validate each URL/redirect and block internal/private network destinations. Do not invent missing content. Handle source/photo download failure separately from recipe-content readiness.

**Older builds:** preserving response shape is necessary but cannot fix old client copy that declares every completed job ready. Audit the frozen 1.16 and 1.17 parsers. For legacy imports, only advertise recipe completion when usable; a terminal incomplete import should use the existing supported error/recovery channel while retaining its bookmark and stable identity. Do not change job semantics globally before confirming that retrying the failed job cannot create duplicate saves. Document residual old-UI limitations honestly.

**Acceptance:** invalid input; social note without content; grounded linked article; initial acceptance followed by repair failure; complete manual repair; title-only edit; mixed batch with ready and incomplete recipes; background/relaunch; offline retry; deleted/replaced source; inaccessible original image with saved cover. Verify UI, notification wording and analytics independently of HTTP/job success.

## 4. Keep pantry-entry context without weakening authorization

**Confirmed source:** [write-intent patterns](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/playground12_voice_ack/trepo-quick-ack/lib/device-assistant.mjs:916), [correction guard](/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/thyme-serving-source/playground12_voice_ack/trepo-quick-ack/lib/device-assistant.mjs:1094). Conversation messages reach the model, but the deterministic recovery gate receives only the latest transcript.

Keep the false-confirmation guard. Extend it with authenticated, bounded pantry-entry state established by an explicit user action, and broaden well-defined explicit pantry/fridge/freezer add commands. Carry the parsed product/quantity forward. Uncertain continuations should ask one targeted question instead of “Please tell me what you'd like me to do.” An informational question may preserve recent context but must not itself authorize a write; negation, task switch, account switch and expiry revoke or suspend mutation authority.

Compound product labels should remain one candidate when grounded as such. Where a phrase could be one infused oil or three groceries, ask a brief clarification/review. Do not automatically merge existing records based on similar names or infer the same physical bottle from close volumes.

**Acceptance:** the three incident continuations; explicit add-to-pantry wording; changed quantity; “yes” to a specific pending question; intervening informational question; negated/quoted command; shopping-list switch; stale session; multiple household members; real multi-item list versus compound oil name. Count writes and assert at most one intended addition. Use deterministic fixtures and repeated model evaluations; passing regex tests alone is insufficient.

## 5. Recover a product photo submitted as a receipt

The incident evidence identifies a product-label photo and zero extracted items. The current [receipt adapter](/Users/MattTaylor/Desktop/trepov2/grocery-identifier/quickIdentify/geminiReceipt.js:654) carries analysis text and items but no stable typed capture-mismatch outcome. This local worker file was inspected, not independently verified as the deployed worker artifact.

The native [empty review guard](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/MainTabView.swift:875) routes an empty completed review into [“Still fetching your items”](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/ReceiptReviewRecoveryView.swift:34), which can say “They are saved” even with an expected count of zero. That conflates genuinely empty analysis with a result that has not been downloaded. It is a reachable code path, not confirmed footage of this customer's display.

Add structured `no_items` / `wrong_capture_type` / `result_unavailable` outcomes, preserving legacy result fields. Download failure keeps Retry; a genuine product-label mismatch offers **Identify this product** using the retained source image. Require confirmation and normal quantity review before adding it. Scope image access to the owner, preserve original job identity, and make repeated recovery taps idempotent. Expired/lost media needs an honest retake path. Never auto-switch modes and commit inventory invisibly.

**Acceptance:** genuine receipt, product label, unreadable/blank image, valid zero result, temporarily unavailable positive result, mixed receipt batch, restart, retained-image recovery and double tap. Verify analysis-completed, items-identified and items-committed are separate outcomes.

## 6. Measure verified outcomes

Reuse the current client's operation identity support rather than introducing a second unrelated ID. [The stream service](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/VoiceAssistantStreamService.swift:101) already sends `x-operation-id`; the inspected serving quick-ack package has no matching operation-ID handling, and the ordinary [voice-send analytics](/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/trepo_v0/trepo_v0/VoiceAssistantView.swift:485) omit that join key.

Carry one request identity through received, transcribed/no-speech, clarification, tool attempt, verified write, response delivery, cancellation and visible completion. Server-generated IDs can cover older clients after receipt, but cannot retroactively identify a request that never reached the server. Deduplicate stage events. Keep transport success, model success and user-task completion as separate dimensions. General dashboards need IDs, field names, counts, state and timings, not raw recordings or conversation content.

Add alerts for repeated corrections to the same target, requested/persisted mismatch, repeated noncompletion, repair failure and zero-item recovery abandonment. Establish thresholds from a baseline instead of inventing a failure percentage from the 481 mixed events. The two unmatched recordings remain **unresolved**, not proven lost requests.

## Qualification and release gates

| Layer | Required evidence before claiming the repair works |
|---|---|
| Offline regressions | Convert these characterization cases into fail-before/pass-after tests; schema, guard, transaction and repair-race coverage |
| Isolated database | Real MySQL triggers/transactions with synthetic households; concurrent writes, rollback, deletion, idempotency and both read/projection modes |
| Model evaluation | Exact recorded text sequences in scoped evaluation fixtures; multiple runs; inspect intended fields and authoritative outcomes, not only assistant prose |
| Backend candidate | Build from freshly downloaded serving artifacts or a reviewed minimal patch; identify all packaged shared-tool consumers; verify auth, signed identity and legacy contracts |
| Mac mini Debug + Release | Dedicated review account: correct four products, refresh, force-close/reopen; failed/ready recipe recovery; zero-item capture recovery; screenshots and videos |
| Older iOS / Android / device | Frozen 1.16 and 1.17 wire decoders plus current Android contract checks; physical speech/camera/Kitchen Assistant checks where affected; simulator alone cannot prove these |
| Controlled production | Explicit deployment approval; scoped candidate rollout, readback and outcome comparison; rollback on mismatched writes, lost edits, duplicate commits, auth regression or elevated errors |

Do not perform destructive or corrective tests in the customer's household. Her previously repaired records, deliberate deletion and possible duplicate should remain untouched; the provided report is the source of that recovery status, not a fresh customer-record read in this turn.

The current TestFlight work may continue under its existing gates because these defects predate build 123 and the release does not change these backend functions. Its notes must not claim these fixes. Subsequent backend repairs should benefit older app versions where the contract permits; new recovery UI requires a later client release.

## Evidence and handoff

- Mac mini execution: `/Users/mikehunt/Projects/trepo-jill-investigation-20260924/`.
- Private readback and outputs: `/Users/MattTaylor/Library/Caches/trepo-jill-proposal-20260924/` (`deployed.json`, `voice-result.json`, `repair-result.json`).
- Reusable offline probes: [voice](probe-voice.mjs), [repair](probe-repair.py). They deliberately assert existing defects and must not be used as release acceptance gates unchanged.
- Original evidence stays in `/Users/MattTaylor/Library/Caches/trepo-jill-repair-20260924/`; no imported source document was overwritten.
- Brain context: `apps/trepo/architecture` (September 23 checkpoint). No shared-brain write was made.

**Recommendation:** implement orders 1 and 2 first as correctness repairs, then order 3 as the completion/recovery change. Keep context/capture improvements in their own patches. The architectural lesson is to make every entry point use the same verified mutation contract, and distinguish a job finishing from a user's task succeeding.
