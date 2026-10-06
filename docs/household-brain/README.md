# Household brain backend

Author: Codex for Matt Taylor. October 6, 2026.
Status: implemented and locally tested; **not deployed or enabled in the live app**.
Matt requested an internally managed brain for every household. No customer-facing
dashboard or new app screens are included.

## What a brain contains

One household state controls learning, serving, timezone and history boundaries.
It produces separate contexts for the household and each current member:

- Explicit soft preferences: cuisines, favorite foods, avoided foods, preferred
  products, cooking style and maximum prep time. Personal unless an operator
  deliberately sets household visibility. A person's own preference takes priority.
- Observations: saved recipe interest, attributed dish logs, recorded recipe
  ingredient use and meal-calendar plans. These remain personal.
- Product evidence: inventory observations, receipt imports, recorded consumption
  and discard, and trusted service receipts when their producers are connected.
  Product variants are not merged by guesses.
- Suggestions: check recorded low stock, consider a product near its recorded
  purchase interval, use items approaching their recorded date, or check past dates.
- Coverage and provenance: a 90-day maximum learning window, source names, evidence
  type/date/scope, stable evidence identifiers and qualifications about uncertainty.

Allergies, diets, religious restrictions, health restrictions and custom dietary
records remain separate, current authoritative inputs. Memory cannot invent or
overwrite them. A generated suggestion cannot authorize a purchase or an app edit.

## When and how it changes

1. A successful canonical database transaction changes an app record. Its trigger
   marks the household dirty and advances its input revision in that **same
   transaction**. Rollback rolls back the signal too. The signal contains IDs and
   counters, not food text. Multiple rapid edits can be combined into one refresh.
2. The worker rereads current committed sources and deterministically rebuilds the
   context. It makes no model calls. Deployment includes a disabled-by-default
   one-minute schedule, processing at most 20 households per invocation.
3. A consumer read checks current membership and revisions. A dirty/missing/expired
   context is rebuilt on demand, so it does not wait for the schedule. Normal cache
   lifetime is at most five minutes; explicit preference expiry can shorten it.
4. Before publishing an answer, consumers validate the original revision and
   household boundary again. A concurrent correction/reset/membership change
   rejects the stale answer. A refresh failure does not serve an old projection.

Only operator commands directly edit preference/control records. Ordinary app work
changes the evidence from which the brain is rebuilt. Opening a screen or producing
an unapproved proposal does not become a fact.

## Exact action matrix

The first group is implemented against canonical MySQL sources and becomes active
when the migration and capture switch are enabled. It does not require an iOS UI
change. Endpoints that write only legacy per-user tables are outside this capture
contract until their canonical write path is verified.

| Committed action | What changes | What it does not claim |
|---|---|---|
| Add a kitchen item, including confirmed scan/voice/text check-in | Current product evidence; low-stock/date candidates | A check-in alone is not a purchase |
| Finish analysis or confirm a reviewed scan row | An eligible `ready` row can enter evidence | Upload acceptance, processing, failure or `needs_review` does not qualify |
| Edit item name, brand, variant, amount or expiration | Rebuilds from corrected current values | Unknown amounts never mean empty |
| Remove/archive/restore a kitchen item | Changes current stock context; retained canonical archive can preserve historical inventory/receipt evidence | Removing stock alone does not prove it was eaten |
| Commit a receipt import | Records receipt evidence; a valid saved purchase date can establish a purchase occasion | An import timestamp is not a purchase date |
| Add/edit/remove/check/uncheck a shopping-list item | Updates active-list exclusions from restock suggestions | Checking an item is not a verified purchase |
| Successfully save a recipe | Records personal recipe interest when status is `ready` | Saving does not prove liking or cooking |
| Edit/unsave/delete a saved recipe | Replaces or removes that current contribution | No obsolete save counter survives deletion |
| Add/edit/delete an attributed Dish Log entry | Records or corrects personal logged-dish evidence | Does not identify who cooked or who liked it |
| Commit recipe ingredient use | Records consumed products and the provided recipe title | Does not infer cooking duration or enjoyment |
| Commit a discard through the amount-operation journal | Records attributed discard evidence | A manual quantity edit is not a discard |
| Undo consumption/discard | Removes the reversed operation's contribution and rereads restored stock | Retries do not count as another occasion |
| Add/edit/remove a meal-calendar entry | Changes personal planned-meal interest | A plan is not a cooked meal |
| Change authoritative dietary preferences | Invalidates old answers and supplies the fresh restrictions separately | Does not turn restrictions into inferred memory |
| Create/join/leave/move/delete a member | Provisions/invalidate household records and audience contexts; removes departed-member contributions from serving | Does not expose another member's private history |
| Operator set/forget a preference | Immediately changes the next eligible read, with author/reason/scope/audit | Household sharing is never inferred |
| Operator pause/resume/reset | Hides learning or starts a new learning window, retaining explicit notes | Resume/reset do not replay pre-reset or paused-period history |
| Operator suppress/restore a product | Suppression survives refresh until explicitly restored | Does not delete inventory, change restrictions or cancel an order |

These service contracts are also implemented and tested, but their producers are
**not activated** by this change:

| Producer | Accepted event | Required integration |
|---|---|---|
| Shop | `product_selected` | A durable receipt from a committed selection, with stable ID and current canonical household identity |
| Order service | `purchase` | A verified successful order/purchase receipt, not a pending cart or checkout click |
| Recipe feedback | `recipe_liked` / `recipe_disliked` | A committed explicit feedback record |
| Same producer | `reverse` | Original event ID; reversal works even if it arrives before the original |

No current order-completion or recipe-feedback producer was verified, so this
implementation does not claim those are already learning. The existing Shop D1
brain remains in place until its authenticated relay adopts the shared adapter.

Browsing, tapping, chat/model guesses, failed writes, abandoned carts, previews,
unapproved agent actions and job acceptance do not add learned facts.

## Internal management

Use `python -m household_brain.cli` with `PYTHONPATH=playground6_groceryRec`.
The CLI verifies the actual AWS STS caller against `HOUSEHOLD_BRAIN_OPERATORS`.
Access to the database secret and its database grants remain the authorization
boundary; the environment allowlist is an additional operator check.

`HOUSEHOLD_BRAIN_DB_SECRET` identifies a Secrets Manager JSON secret with `host`,
`port`, `username`, `password`, `dbname`, and, for RDS, the approved PEM `ca_bundle`.
The dedicated service connection verifies TLS certificates and hostnames. Do not
place secret contents in request files, logs or this repository.

Commands:

- `install`: preflight canonical tables, create additive tables/triggers; serving
  and capture stay unchanged/off on first installation. Repeated install verifies
  existing trigger definitions rather than silently replacing them.
- `capture-on/off --reason '...'`: audited global capture/serving kill switch;
  invalidates existing contexts. Reenable cannot reuse a cache from the disabled period.
- `default-serving-on/off --reason '...'`: chooses serving for subsequently
  provisioned households. Does not enable existing disabled households.
- `provision`: create missing household states from current membership, idempotently.
- `status`: counts, pending work and global configuration; no customer food data.
- `inspect --household ID`: internal state, preferences, contexts, suppression,
  recent service receipts and the last 50 operator edits.
- `change --request-file FILE`: revision-checked, audited, idempotent edit.
- `refresh --household ID`: rebuild one household; omit ID for a bounded worker pass.

Example internal change (substitute IDs and the revision returned by inspection):

```json
{
  "operation_id": "a-stable-unique-operation-id",
  "household_id": "canonical-household-id",
  "expected_revision": 12,
  "action": "set",
  "actor_id": "current-member-id",
  "key": "cuisines",
  "value": ["Thai"],
  "visibility": "household",
  "reason": "Household preference explicitly confirmed to Trepo support"
}
```

Reuse the same operation ID for a retry. A changed request under the same ID is
rejected. A stale revision requires a fresh inspection. Optional `expires_at` is
validated, with a maximum one-year lifetime. `forget` removes that explicit field;
it does not erase historical support audit records.

Other actions: `enable` (optional `backfill_days`, 0–90; default 0), `disable`,
`pause`, `resume`, `reset`, `timezone`, `suppress_product`, `restore_product`.
Pause/resume/reset may target an `actor_id` or the whole household. Product
suppression uses the exact `product_key` from inspection, optionally personal.
Reset/resume advance a boundary and retain explicit preferences and suppression.
Reset is not a legal data-erasure mechanism. Event text ages out after 90 days
through bounded worker maintenance; stable IDs/hashes/reversal records remain to
prevent replay. Audit records and original app records retain their own lifecycle.

## Integrations delivered

- Existing Python Family Brain entry points delegate to the shared backend when
  `HOUSEHOLD_BRAIN_V2=1`. Existing JWT checks and long-running publication guards
  remain in use. Shared feeds receive only the shared context.
- The legacy Thyme adapter supports the new rollout flag. Its customer memory
  mutation path is blocked in V2, including direct calls to the Python control API.
- Native Thyme V2 loads the private shared reader, removes its memory-edit tool,
  preserves the original input boundary across request recovery, and validates it
  before publishing. It rotates hidden provider context after brain changes.
- `household-brain-client.mjs` exports the shared context/validation adapter and
  `shoppingBrain(verifiedSession, ...)` for the server-side shopping relay.
  It is **not yet called by the separately running shopping repository**. That
  integration must supply canonical IDs from its verified phone session, never a
  `trepo-copy-*` ID, and revalidate before order/planner publication.
- Private event ingestion accepts only the producer's fixed allowed event types,
  checks current membership/window/generation and handles duplicates/reversals.
  Give each future producer a separate IAM-restricted function with a fixed
  `HOUSEHOLD_BRAIN_PRODUCER`; never expose this handler through HTTP.

## Rollout and rollback

The deployment template creates a private reader and a worker, with the schedule
disabled. There is no function URL, API Gateway route or customer operator API.
Limit `lambda:InvokeFunction` to the exact authenticated backend/relay roles that
need the reader; the caller is responsible for verified session identity.

1. Review this diff together with the active iOS/backend fixes. Fetch fresh serving
   packages. Build overlays with the exact package SHA; source drift aborts. Do not
   deploy the complete old repository checkout over the serving backend.
2. Test canonical table coverage, ownership, indexes, trigger grants/definer and
   lock timing against a staging database. The source adapter intentionally fails
   on missing or oversized sources: at most 64 members and 2,000 rows per source.
   These limits are coverage errors, not silent truncation. Large-household/query
   latency qualification is still required before broad rollout.
3. Install additive schema/triggers while capture is off. Keep their definer
   available. Enable capture, then provision existing households. Start with
   serving off by default and explicitly enable the founder households.
4. Deploy the private runtime and minimal consumer overlays with V2 flags off.
   Configure database secret/VPC/IAM; runtime DB grants should allow only required
   canonical reads and brain maintenance. Operator DDL/control grants are separate.
5. Enable the one-minute worker, verify real refresh/error/latency metrics, then
   turn on V2 in one reader and its relevant consumers. Native Thyme additionally
   needs `HOUSEHOLD_BRAIN_FUNCTION` and IAM invoke permission. Verify an actual
   edit, undo, membership change and internal reset in the pilot.
6. Extend serving to approved households/default provisioning only after those
   checks. Connect the shopping relay and service event producers independently.

The trigger signal is transactionally coupled to the original app write. A
misconfigured trigger/grant could therefore fail that write; staging permission
and contention checks are a rollout requirement. Projection/model failures happen
after the source commit and never repeat or reverse it.

Rollback: turn consumer V2 flags off to restore the prior app behavior. Disable
the schedule and use `capture-off` to stop capture and invalidate V2 serving. Keep
tables/triggers until a reviewed removal; dropping tables under active triggers
would break writes. Audits, explicit preferences and operational data are preserved.

## Source evidence

The baseline is the September 25 founder service, checked against the serving
Home Lambda on October 6, plus the current Thyme V2 source at `e856563` and the
shopping pilot's evidence distinctions. `imported-v1-manifest.json` records source
hashes; `source-schema-20261006.json` contains column names/types, no customer rows.
Shared company context: `trepo-company/apps/trepo/architecture` and
`trepo-company/sources/product/2026-09-23-agentic-ordering-requirements`.
These historical pages are context, not deployment evidence. No shared-company
memory write, purchase, message, merge or production deployment was performed.
