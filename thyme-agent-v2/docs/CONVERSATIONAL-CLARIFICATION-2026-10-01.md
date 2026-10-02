# Thyme clarification and conversation preferences

Author: Codex for Matt Taylor. Date: October 1, 2026 (PDT; tests continue October 2 UTC).
Status: completed, committed and verified on the private web pilot October 2, 2026. No existing customer route or iOS release changed.

## Request and scope

Matt requested useful clarification, especially kitchen-only versus shopping for generic recipe requests, remembered for the rest of a chat. He also requested learning from Thyme conversation patterns since July. This changes the pilot harness, instructions and evaluation suite, **not model weights**. No bulk customer rollout or iOS release is included.

## Evidence and limitations

Read-only investigation started July 1 and ended October 2 UTC. Full paginated scan of the live session table: 13 pages, 8,662 records, 2,738 user messages. Excluding the known qualification account and explicitly test-marked sessions leaves **2,047 candidate customer messages in 911 sessions**, September 2–October 2: September 1,997; October 50. Unmarked founder/test traffic may remain. Pattern categories overlap and are rough lexical counts, not satisfaction or failure rates: 996 recipe/cooking requests, 631 replies of three words or fewer, 131 explicit stock references, 227 list/action references. Representative complete exchanges were reviewed across these categories.

The old SQL `ai_request_logs` table yielded no rows for the requested interval. Retained CloudWatch logs were also searched across five relevant Lambda groups (105,633 records scanned; 746 matching log lines returned without truncation). Deduplicating transcript lines by invocation recovered 212 request texts: **July 113, August 20, September 71, October 8**. Older logs include device/development traffic and explicit probes, so these are historical examples, not a clean customer cohort. July/August transcripts are sparse, not an exhaustive account of every request. The realtime voice log retains only 14 days. There is no claim that all history since July is available.

Raw transcripts/account identifiers stay in the restricted local audit cache. No customer corpus was uploaded for training or saved into company gbrain. This document and test cases use aggregates and short, generic or paraphrased patterns.

## Patterns that changed the design

| Pattern | What goes wrong | New behavior |
| --- | --- | --- |
| Broad dinner/recipe request | Guesses whether shopping is allowed | One concrete kitchen-versus-shopping question |
| Specific dish or cooking technique | An intake questionnaire adds needless friction | Answer directly; state useful assumptions |
| “Another one,” “with chicken,” “not with egg” | Loses the current recipe/task or restarts it | Use current conversation, preserve unrelated recipe details |
| “Yes” after two alternatives | Guesses which choice was meant | Disambiguate briefly |
| “Only what I have, for two, 20 minutes” | Forgets one or more constraints later | Store all three explicit constraints in this conversation |
| “Actually I can shop” | Sticks with obsolete scope, or discards all other choices | Override shopping scope while preserving servings/time |
| “Just pick” | Keeps asking instead of helping | Choose a sensible default and label the assumption |
| Isolated item/garbled voice fragment | Invents an action or changes the wrong destination | Ask the missing action/destination; use prior task only when clear |
| “Make it healthier” | Invents a nutrition or medical goal | Ask which concrete goal matters |
| Straightforward list addition | Asks irrelevant recipe questions | Prepare the existing reviewed action directly |

Before the change, a real-model baseline “Can you give me a dinner recipe?” produced a recipe immediately, with no question, in 20.253 seconds. The failure matches the old policy: a general instruction to avoid questions, no explicit decision rules, and no dedicated durable preference record.

## Implementation

- Bounded typed conversation preferences: `inventory_mode` (`existing_only`, `shopping_ok`, `either`), servings (1–40), time cap (1–1440 minutes). A null update clears one field. Each update requires an exact evidence quote from the current user message; updates are atomic and revision-fenced.
- Data is scoped by actor, household **and conversation**. No transfer to a new chat, no change to household food memory or allergies, no account-write approval. Preferences survive reloads and worker restarts and are supplied every turn.
- One recorded clarifying question is shown as the answer. The same request cannot then create a recipe or propose a mutation. The normal message input accepts any answer; no additional UI panels or mandatory form.
- Acknowledgment-only replies still record the supplied choices. Same-turn preference writes cannot remove a pending question and reopen action tools. The provider adapter explicitly filters item ownership because the live provider ignored its turn filter; earlier questions are not rendered again on later replies.
- User history and explicit preferences are retained when an idle older provider session moves to the updated instructions/tools. In-flight requests are never migrated or replayed. Oversized history fails with an honest new-chat request rather than silently dropping context. Canonical recipe records and approval records remain authoritative.
- Existing action approval, dietary checks, current kitchen/list reads and recipe versioning remain in force. This work does not resolve the separately documented legacy kitchen-mutation qualification limitation.

## Validation

**82 automated tests pass.** Coverage includes actor/household/chat isolation, overrides and clearing, evidence quotes, invalid fields, atomic updates, retries, stale workers, concurrency conflicts, same-turn question/action blocking, idle-chat migration, in-flight protection, large histories, recipe edit preservation, dietary gates, approvals and provider pagination/turn ownership.

The real-model matrix passed **57/57 turns across 12 scenarios and three models** (Sol, Luna and Astra). It checks broad recipes, multiple constraints, continuation, overriding shopping scope, ambiguous “yes,” “just pick,” ordinal answers, grocery-store context, named dishes, explicit pantry meals, clear and unclear list actions, voice fragments, “healthier,” and cooking technique questions. A further **9/9 turns** check acknowledgment-only answers and memory across all three models. The post-provider-fix continuation regression passed **4/4 turns**, including an assertion that the earlier question occurs only once. Total reported successful model qualification: **70 turns**. These are bounded qualifications, not a guarantee for every possible phrase.

Failures found during qualification were fixed and retained in local evidence:

- Luna originally proposed raw rice under a 20-minute limit. The policy now requires total elapsed time and a reliably quick cooking method; reruns were manually checked for credible timing.
- Astra initially wrote a clarifying question without recording it. The tool requirement was made explicit and reruns passed.
- Luna initially repeated a shopping-permission question for a user already heading to the store. The policy now recognizes that context; all three models passed the held-out wording.
- Live cloud testing found an acknowledgment-only reply remembered in model history but absent from durable preferences. Explicit recording guidance and a new three-model regression cover this case.
- Browser testing revealed the provider returned earlier messages despite the requested turn filter, which could repeat a recorded question. Local turn ownership filtering and a paginated regression fix the cause.

Two test expectations were corrected rather than treating valid behavior as a product bug: “rice dish” is a broad request that may merit clarification, and “dinner ideas” need not produce a full recipe card. Raw failures and reruns are retained; passing numbers do not count those earlier failed attempts as successes.

Real-model evaluation uses synthetic account data with a peanut allergy and a small kitchen. Live checks use Matt’s verified pilot account and do not approve any account mutations. See `test/conversation.test.mjs`, `scripts/qualify-clarification.mjs`, and `scripts/qualify-clarification-cloud.mjs`. Raw account/transcript artifacts are not committed. The pilot remains an experimental model-driven system; remembering a preference does not prove every generated recipe is perfect, and actual meals were not cooked during these tests.

### Final deployed checks

- Private API/worker code hashes verified against packaged artifact `566ebe4c4efd533df21ee5c2486d63efd502194b8b49dc7f55648fdbe0b66e7e`. Code-only update; credentials, permissions, environment and all existing customer routes unchanged. Source commit `9967aed`.
- Four live cloud conversation turns passed: broad question; acknowledgment-only choice recording; recall after a new API read/worker invocation; older chat upgraded while retaining its history. Read-only DynamoDB verification independently confirmed all three values and cleared pending questions in both the new and upgraded chat.
- Authentication negatives passed: forged signature, unenrolled user, client household override and approval through the read-only qualification principal are rejected. Request deduplication passed. Live data still resolves to Matt’s 60 kitchen items and 4 active shopping items. One real-model canary completed with no account mutations.
- Browser: “Help me choose dinner tonight” displayed one kitchen-versus-shopping question; “Only what I already have, for two, in 20 minutes” produced a two-serving recipe using current stock and a quick prepared-rice method. Reloading the page, reopening that chat and asking for the choices returned all three correctly. No duplicate earlier question, additional recipe on the reminder, or browser warning/error. Three final browser turns verified.
- Evidence: `CLARIFICATION-QUALIFICATION-2026-10-02.json`; screenshots `clarifying-question.png` and `clarification-after-reload.png` in the local pilot evidence directory. Earlier failed trials remain in the restricted audit cache.

Pilot: https://trepo-thyme-pilot.trepo-2057.chatgpt.site/

## Brand and context sources

Existing private-pilot UI is retained. Wording follows `brand/trepo-brand-guidelines` (original Sep15, retrieved Sep23): brief, approachable, direct. No brand exception. Shared context `sources/analytics/2026-09-22-power-user-archetypes` also cautions against treating every household as an inventory-first user. Live source/log evidence determines behavior and coverage.
