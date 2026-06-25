# Shopping List Sync — Durable Fix Plan

**Status:** proposed (2026-06-24). Investigation owner: list divergence between household members.
**Live app — every change staged behind a flag, additive-first, no destructive steps.**

## Problem (root cause)
The shopping list is not a single shared object. Each household member has a private
`{user_id}_new_list` mirror table. "Sync" = best-effort **fan-out on write** (`listHandler.js`
loops `memberIds` and writes a copy into each). Reads (`operation:'view'`) return **only the
acting user's own table**. There is **no reconciliation** after a missed write. The only heal
path (lazy backfill, lines ~115-154) runs **only when a member's list is completely empty**,
copies **one-directional** from the first non-empty member, and — until 2026-06-24 — was
**throwing `Unknown column 'sort_order'`** for ~188 drifted tables and aborting.

Consequence: any item that fails to fan out (missed write, non-fan-out add path, transient
membership hiccup) is **permanently orphaned**. Confirmed cases: "raspberries" (Anna, Mar 9–14,
never on Matt's side), "tallow fries" (Anna, Jun 7 — healed manually 2026-06-24). Systemic:
2 of 6 multi-member households diverged.

## Design principle
Stop treating write-time fan-out as the source of truth. Make **reads tolerant** (union across
the household) so a single missed write is invisible to users, and make **writes match by the
shared id** so mutations always propagate. Keep the per-user tables (no schema migration of the
storage model in v1) — just change how we read and reconcile.

---

## Stage 0 — Safety net already in place ✅
- `sort_order` column added to all 188 drifted `_new_list` tables (2026-06-24). Backfill no
  longer throws. **Done.**
- Manual reconcile of Matt/Anna lists (tallow fries). **Done.**

## Stage 1 — Union read (the core fix, low risk, behind a flag)
**Goal:** `operation:'view'` returns the *merged* household list, deduped by
`household_item_uuid`, instead of only the caller's table. A missed fan-out becomes invisible.

- New env flag `LIST_UNION_READ` (default `false`).
- When true, in the `view` branch:
  1. `getHouseholdMemberIds(pool, ownerId)` (already called).
  2. `SELECT ... FROM` each `{memberId}_new_list` (skip `ER_NO_SUCH_TABLE`).
  3. Merge rows keyed by `household_item_uuid` (fallback `_id` only when uuid null). On
     conflict prefer the row with the most recent `updated_at`; for `action`, prefer `CHECKED`
     only if the newest write is CHECKED (respect the latest toggle).
  4. Order by `sort_order` (nulls last) then `_createdDate DESC`, as today.
- **No writes in the read path** (remove the lazy-backfill write-on-read entirely — it is the
  source of one-directional partial heals and adds write load to a GET). Reconciliation moves to
  Stage 3.
- Rollout: enable for the Matt/Anna household first (hardcode allowlist or a per-owner flag),
  watch, then flip globally.

**Risk:** read-only; reversible by flipping the flag. Main cost is N small SELECTs per view —
households are ≤3 members, so negligible. Keep the single-table path when flag is off.

## Stage 2 — Match-by-shared-id on writes (stop new divergence at the source)
**Goal:** every mutation propagates across all copies; never silently hit only one table.

- `remove` / `set_action` / `update_item` / `reorder`: **drop the `_id` fallback.** Match only
  by `household_item_uuid`. The `_id` is per-table and meaningless across mirrors — the fallback
  is precisely what lets a delete/check/edit land on only the actor's copy.
- Guarantee every list row has a `household_item_uuid`: one-time `UPDATE ... SET
  household_item_uuid = (SELECT ... )` / generate where null, so the uuid-only match never misses.
- `add` / `batch_add`: wrap each per-member INSERT in try/catch that logs + continues (today
  `add` has no try/catch → one member's failure 500s the whole request). Emit a structured
  `list_fanout_miss` log when any member insert fails, so misses stop being silent.

**Risk:** behavioral change on mutations. Ship behind `LIST_STRICT_UUID_MATCH` flag; the union
read (Stage 1) covers any row still lacking a uuid during transition.

## Stage 3 — One-time reconcile pass (heal the existing fleet)
**Goal:** union every multi-member household's lists once, so current divergence is gone even
before all clients pick up the new read path.

- Offline script (pattern: the `sort_order` batch from 2026-06-24): for each `owner_id` with
  >1 member, compute the union of `household_item_uuid`s across members' `_new_list` tables;
  insert any missing rows into the members that lack them (same uuid, `_device='reconcile'`),
  preferring the newest `action`/`store`/`sort_order`. Idempotent (skip if uuid present).
  Per-table try/except, sequential, single connection.
- Dry-run first (report what *would* change per household), then execute.
- Known targets today: household `38673` (Matt/Anna — already done manually), `50928`
  (Randy — Topo Chico, Coors Light).

**Risk:** additive inserts only, idempotent. Same shape as the validated `sort_order` batch.

## Stage 4 — Observability + guardrail (catch the next one automatically)
- CloudWatch metric on `list_fanout_miss` (Stage 2) → alarm if > 0 sustained.
- Lightweight daily job: count households whose member lists' uuid-sets differ; emit a metric.
  Divergence should trend to ~0 after Stages 1–3; a rising count flags a new regression.

---

## Sequencing & rollback
1. Stage 1 (union read) behind flag → enable Matt/Anna → verify → global. **Biggest user win,
   lowest risk.**
2. Stage 3 (reconcile pass, dry-run → execute) — heals everyone now.
3. Stage 2 (strict uuid match + fan-out logging) behind flag → global.
4. Stage 4 (metrics/alarm).

Each stage is independently shippable and flag-gated. If anything misbehaves, flip the flag —
the storage model is unchanged, so there is nothing to roll back at the data layer.

## Files
- `APP/Lambdas/listHandler.js` — view branch (Stage 1), mutation branches (Stage 2).
- `APP/Lambdas/householdSync.js` — unchanged (membership resolution already correct).
- New: `APP/Lambdas/scripts/reconcile_household_lists.py` (Stage 3).
- `template.yaml` — `LIST_UNION_READ`, `LIST_STRICT_UUID_MATCH` env flags + metric filter.
