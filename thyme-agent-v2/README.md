# Thyme private agent pilot

Matt approved this pilot on October 1, 2026. This does not replace the public Thyme endpoint. An opt-in native integration is now implemented for Matt only; see [native pilot status and verification limits](docs/NATIVE-PILOT-2026-10-02.md). The private tester is https://trepo-thyme-pilot.trepo-2057.chatgpt.site. It requires the site owner's ChatGPT sign-in and a server-side account binding. Current web audience: Matt only.

## What is implemented

- Persistent managed agent conversations, deferred tool discovery and live scoped Trepo data.
- Canonical conversation recipes, stable ingredient/step IDs, revision checks and targeted edits. A conservative chat-text safeguard can preserve a single complete, explicitly portioned recipe exactly when the model omits its card tool; ambiguous text is not guessed, and a contradictory full-text rewrite cannot replace an existing recipe card.
- Real kitchen, complete shopping list, saved recipes, suggestions, meal plan, calendar, dish history, discarded items, strict dietary preferences and existing family memory.
- All 57 original Thyme tools have an equivalent read or reviewed action, plus six explicit list/item/recipe/dish editing tools. Shopping groups, kitchen quantities/details, personal Dish Log, saved recipes/categories, dated calendar plans, social imports and automatic meal-plan refresh are supported. Canonical conversation recipes and soft food-memory updates remain available. A user must review and approve each concrete operation; ordinary chat text cannot approve it.
- Durable request acceptance, duplicate detection, worker leases, interrupted-turn recovery and result reconciliation. The worker persists a write result before reading it back. A lost or uncertain write is never replayed automatically.
- Phone-friendly private chat, conversation history, thinking-style comparison, live data panel, recipe cards, approval/rejection controls and helpful/unhelpful feedback.
- Three model choices: GPT-6.1 Sol, GPT-6 Astra, GPT-6 Luna. No automatic routing decision is claimed from this small comparison.

## Scope boundaries

Purchases, outgoing messages, native camera entry, notifications, account deletion and Kitchen Assistant device control are not implied by tool parity. Recorded voice is provided separately by the private native pilot. Those require separate native/transactional integrations. “All 57” refers to the pinned existing Thyme catalog, not every screen or every possible app operation. Recipe ingredients and calendar entries are frozen as explicit reviewed content before approval. Dish Log and saved-library edits are scoped to the enrolled actor. Empty store groups cannot persist because the app derives groups from list items.

URL imports and automatic meal-plan refresh use existing asynchronous jobs. A queued response never earns a completed receipt; the pilot polls/read-reconciles without resubmitting uncertain writes. Other legacy batch adapters can partially succeed; the pilot reports an unconfirmed result unless every requested change is visible. See [TOOL-PARITY-REPORT.md](TOOL-PARITY-REPORT.md) for coverage and boundaries.

Existing app routes, login compatibility and unrelated production resources are unchanged. Native iOS integration and voice entry are covered by the private native pilot status document. Push notifications and a general customer rollout remain separate work. This is not evidence that every legacy action or every iOS build has been tested.

## Runtime

`entry.mjs` has an HMAC-protected API and a DynamoDB stream worker. The server maps authenticated Sites identity to an enrolled Trepo actor. Browser/model identity overrides are rejected. State keys include actor and household; current membership is rechecked. The signed request binds HTTP method, path, body, timestamp and one-time nonce.

DynamoDB stores sessions, jobs, tool receipts, proposals, versioned recipes and feedback. The change stream dispatches queued jobs one at a time per batch. No scheduled Codex task or laptop process is needed to run an accepted request. State TTL is 90 days, nonce TTL 180 seconds, table point-in-time recovery is enabled, operational logs retain 14 days. The stream retains changes for 24 hours; a prolonged stream outage still requires operational recovery from the durable request records.

Limits: one active request per conversation, 100 new messages per actor/day, 30 tool rounds and a 180-second worker budget. Worker concurrency is two. Dish/recipe/discard tool reads provide query, offset, count and next_offset. Personal edit approvals store the selected row; broad clears retain the full approved snapshot. Requests exceeding storage/tool-response limits still fail explicitly rather than truncating an approval. Runtime date context currently uses America/Los_Angeles for this founder pilot.

`gateway.mjs` wraps a pinned deployed tool package. `package.py` checks its SHA before packaging. The package patches add complete shopping reads for explicit `all:true`, persisted shopping quantity, row-mapper/normalizer exports and exact kitchen-ID handling. New pilot adapters perform scoped transactions and use the serving canonical kitchen operation endpoint with revision checks. Existing default behavior is preserved. Canonical recipe saving uses a dedicated transaction to persist the exact reviewed recipe and servings notes to both current and older saved-recipe schemas, without running another model.

## Testing

Use Node 22 and `npm ci`, then `npm test`. Run `scripts/qualify-sql.mjs` only against the named disposable local MySQL fixture on port 33317. Its credentials are fixture-only. `scripts/qualify.mjs` uses `THYME_PRIVATE_CONFIG` to read an existing provider key and `THYME_EVIDENCE` to write synthetic results. Never commit that private config. Run `scripts/assert-qualification.mjs <evidence files>` for behavioral assertions.

`scripts/cloud-check.mjs` uses the special read-only qualification identity. It may read Matt's data and create a private test conversation, but cannot approve or resume account writes. Its full response is stored only in the mode-0700 private cache; shareable evidence omits household contents. Human-approved pilot changes are real changes visible in Trepo.

## Deploy and rollback

Build with `scripts/package.py --base <immutable serving.zip> --out <private-cache>/pilot.zip`. Deploy with `scripts/deploy.py --archive <pilot.zip> --private-dir <private-cache> --base-config <private config>`. The script updates only newly named `trepo-thyme-agent-v2-pilot-*` infrastructure and uses AWS revision checks. Credentials and account bindings stay out of Git. Preserve the previous pilot archive and configuration before updating.

Rollback: redeploy the prior pilot archive/configuration or clear the private web principal binding to stop new access. Disable the pilot stream mapping if stopping workers is necessary, then inspect durable jobs and reconcile any applying/verifying operation before re-enabling. Do not replay uncertain writes. None of this changes existing customer routes. Site source and runtime environment revisions are separately recorded in the deployment report.

## Known limits and next release gates

- The deployed browser connected to Matt, generated a recipe from his live kitchen and preserved it during a clarification and full-page reload. See TEST-REPORT.md for detailed evidence.
- Real-account automated tests are read-only. Approved writes are tested in fixtures; a founder should exercise a harmless live write and verify it in iOS before broader adoption.
- Lexical allergy rejection supplements model instructions; it is not a complete food-safety or allergen guarantee. No estimated shelf-life date proves an item safe to eat.
- Legacy writes still use existing domain adapters. Snapshot/revision checks narrow races but do not create a single cross-system database transaction for every action. This is a founder pilot.
- Billing usage can arrive after the model finishes. Missing usage stays pending. Cost and old-versus-new latency have not been proven better.
- An in-host gstack adversarial review of the parity update identified ten concrete issues. They were fixed and rechecked; the reviewer reported no remaining P1/P2 findings. Tests/fixtures were reviewed in summary mode only. Outside-provider review remains disabled by company policy.

Brand sources consulted: `brand/trepo-brand-guidelines` and approved app guidance `brand/trepo-app-design-system`. Original Thyme artwork is used. No brand amendment or automatic company-brain save was made.
