# Shared-Table Migration — FINAL PHASE Plan

> **Status:** Scout inventory + plan. READ-ONLY. No code, DB, or infra was changed producing this doc.
> **Author:** scout agent · **Date:** 2026-07-12
> **Goal:** Retire the ~16,164 per-user MySQL tables (`{uuid}_new_list`, `_dishes`, `_saved_recipes`, `_meal_plan`, `_recipes`, `_discards`, `_new_kitchen`, `_prod_kitchen`, `_feed_events`, `_master_feed`, `_new_feed`, `-metrics`) in favor of `shared_*` tables keyed by `owner_id`.

---

## 0. TL;DR — where the migration actually stands

Live flag state (read from deployed Lambda env vars, 2026-07-12):

| Family | Shared target exists? | Reads cut over? | Writes cut over? | Per-user tables still written by? |
|---|---|---|---|---|
| **kitchen** (`_prod_kitchen`/`_new_kitchen`) | ✅ `shared_kitchen` (13,817 rows) | ✅ `USE_SHARED_TABLES=true` | ✅ `WRITE_SHARED_ONLY=true` on grocery-backend | ⚠️ `trepo-kitchen-handler` (APP/Lambdas, **no flags**, Nov-2025 deploy) still writes `_new_kitchen` |
| **dishes** (`_dishes`) | ✅ `shared_dishes` (382 rows) | ✅ `USE_SHARED_TABLES` | 🟡 dual-write (`DUAL_WRITE_ENABLED`) | dish upload worker + kitchen_api dual-write both |
| **list** (`_new_list`) | ✅ `shared_shopping_list` (134) + legacy `shared_list` (1,365, dead) | ✅ `LIST_UNION_READ=true` | ✅ dual-write live; per-user still populated (**1,575 rows / 197 non-empty tables** exact) | listHandler fan-out (still writes per-user) |
| **discards** (`_discards`) | ⚠️ `shared_discards` exists but only **17 rows** | 🟡 flag set, mapping partial | ❌ upload discard-writer does **NOT** dual-write shared_discards | discard upload worker + discards_api (per-user) |
| **saved_recipes** | ❌ none | ❌ | ❌ (`WRITE_SHARED_ONLY=true` is set but code has no shared mapping → silently writes per-user) | saved_recipes_api |
| **meal_plan** | ❌ none | ❌ | ❌ | meal_plan_api / meal_plan_generator |
| **recipes** | ❌ none | ❌ | ❌ | recipes_api / recipes_generator |
| **feeds** (`_new_feed`/`_master_feed`/`_feed_events`) | ❌ none | ❌ | ❌ | feedHandler + upload workers + master_feed.py |
| **metrics** (`-metrics`) | ❌ none | ❌ | ❌ | metrics_api / kitchen_analysis_generator / redemptions_api / upload workers |

**Key insight:** `WRITE_SHARED_ONLY=true` and `USE_SHARED_TABLES=true` are live on the grocery-backend APIs, but the resolver code (`_resolve_kitchen_table` / `_resolve_table`) **only remaps `_prod_kitchen`→`shared_kitchen` and `_dishes`→`shared_dishes`.** Every other family ignores the flag and still resolves to a per-user table. So five families (saved_recipes, meal_plan, recipes, feeds, metrics) need a **shared target designed and built** before they can be cut over — they are not "flip a flag" work.

**Two straggler write-paths must be closed** even for the already-migrated families:
1. `trepo-kitchen-handler` (`APP/Lambdas/kitchenHandler.js`) — writes `_new_kitchen`, no flags, deployed 2025-11-14.
2. `trepo-feed-handler` (`APP/Lambdas/feedHandler.js`) — writes `_new_feed`, no flags, deployed 2025-11-14.
3. Discard upload worker never dual-writes `shared_discards` → discards migration is nominal only.

---

## 1. Database census (live, 2026-07-12)

Total tables in `mysqlTutorial`: **16,712**. Per-user family tables: **~16,164** across **~1,582 users**. `shared_*`: 6 tables. Other (non-family, non-shared): 542.

Row counts below are **exact `COUNT(*)` per table** (gathered after the RDS maintenance window; see §3). Sorted by exact rows.

| Family suffix | # tables | **Exact rows** | Non-empty tables | On-disk (data+idx) | Shared target |
|---|---|---|---|---|---|
| `_master_feed` | 339 | **47,391** | 339 | 109.3 MB | ❌ |
| `-metrics` | 1,582 | **16,201** | 344 | 104.9 MB | ❌ |
| `_feed_events` | 52 | **6,191** | 52 | 15.2 MB | ❌ |
| `_new_feed` | 1,568 | **4,858** | 25 | 74.6 MB | ❌ |
| `_discards` | 1,570 | **2,843** | 20 | 213.2 MB | ⚠️ shared_discards (17 rows) |
| `_saved_recipes` | 1,576 | **2,032** | 231 | 114.2 MB | ❌ |
| `_new_list` | 1,573 | **1,575** | 197 | 73.8 MB | ✅ shared_shopping_list |
| `_prod_kitchen` | 1,581 | **1,473** | 35 | 211.5 MB | ✅ shared_kitchen (13,817) |
| `_dishes` | 1,572 | **1,064** | 30 | 174.1 MB | ✅ shared_dishes (382) |
| `_recipes` | 1,580 | **682** | 346 | 192.8 MB | ❌ |
| `_meal_plan` | 1,580 | **349** | 349 | 49.4 MB | ❌ |
| `_new_kitchen` | 1,568 | **30** | 2 | 49.0 MB | ✅ shared_kitchen |

**Total per-user rows: ~84,689.** The migration payload is dominated by **feeds** (`_master_feed` 47,391 + `_feed_events` 6,191 + `_new_feed` 4,858 = **58,440 rows**, ~63% of everything) and **metrics** (16,201). The already-targeted families are small/stale: `_prod_kitchen` 1,473 rows in 35 tables (vs 13,817 live in shared_kitchen → kitchen genuinely moved), `_new_kitchen` 30 rows in 2 tables (dead), `_dishes` 1,064. Data-sizing takeaway: the real cost is in the five no-target families (Phase D), especially the feed merge.

`shared_*` tables (exact, with `owner_id` column present in all):

| Table | Rows | Notes |
|---|---|---|
| `shared_kitchen` | 13,817 | Live, primary kitchen store |
| `shared_archive_kitchen` | 1,812 | Archive of removed kitchen items |
| `shared_list` | 1,365 | **Legacy write-only dead weight — decommission-ready** |
| `shared_dishes` | 382 | Live (dual-write) |
| `shared_shopping_list` | 134 | Live shopping list |
| `shared_discards` | 17 | Nominal — discard writer not dual-writing |

---

## 2. Dependency matrix — readers & writers by family (file:line)

Scope covered: `APP/Lambdas/`, `playground6_groceryRec/*`, `playground12_voice_ack/trepo-quick-ack/`, `playground_explore/`, `obs/`, `shared/voice-assistant/`, `grocery-identifier/` (root src), `Lambda/household_membership_lambda/`, `twilio-verify-lambda/`. Build artifacts (`.aws-sam/build/`, `dist/`, `node_modules/`) excluded. All line numbers are repo source, not deployed.

### 2a. KITCHEN — `_prod_kitchen` / `_new_kitchen` → `shared_kitchen` ✅
Flags: `USE_SHARED_TABLES` (read routing), `WRITE_SHARED_ONLY` (write routing) — both **live=true** on grocery-backend.

**Writes**
- `playground6_groceryRec/kitchen_api/app.py:1061` — INSERT `{owner}_prod_kitchen` (resolver routes to shared_kitchen when flag on); `:383-401` `_resolve_kitchen_table()`.
- `playground6_groceryRec/analyze_on_upload_nodejs/src/utils/mysqlWriter.ts:21,375-382,408-413,448,629` — INSERT/UPDATE/DELETE `shared_kitchen` (gated `WRITE_SHARED_ONLY`, line 7). No per-user write when flag on.
- `APP/Lambdas/kitchenHandler.js:84,117,133` — ⚠️ fan-out INSERT/DELETE to household members' `_new_kitchen`. **No flag, no shared write.** (deployed `trepo-kitchen-handler`, 2025-11-14)
- `analyze_discard_on_upload_nodejs/app.js:1390` — `removed_from_table: {owner}_prod_kitchen` write record.

**Reads** (all gated `USE_SHARED_TABLES`, resolve to shared_kitchen)
- `kitchen_api/app.py:1069,1091`; `saved_recipes_api/app.py:796`; `recipes_generator/app.py:411`; `swap_review_api/app.py:89`; `home_suggestions_api/app.py:143`; `kitchen_insights_api/app.py:164`; `kitchen_analysis_generator/app.py:115,122`(`_new_kitchen`); `metrics_api/app.py:145,152`(`_new_kitchen`); `meal_plan_generator/app.py:233`; `get_job/app.py:61-64`.
- `playground_explore/explore_recipes_api/app.py:330` — READ `{owner}_prod_kitchen` (⚠️ **not confirmed flag-gated** — verify before cutover).
- `analyze_on_upload_nodejs/src/utils/suggestKitchenSwaps.ts:232` — READ `{owner}_prod_kitchen`.

### 2b. DISHES — `_dishes` → `shared_dishes` ✅ (dual-write)
Flag: `DUAL_WRITE_ENABLED`.
- `kitchen_api/app.py:250-271` `_shared_dishes_insert()` — INSERT `shared_dishes` (non-blocking, errors swallowed).
- `analyze_dish_on_upload_nodejs/src/utils/mysqlDishWriter.ts:14` INSERT `shared_dishes` (dual-write, flag line 6); `:158` INSERT/UPDATE `{memberId}_dishes` (per-user primary).
- `dishes_api/app.py:402` READ/WRITE `{owner}_dishes`; `recipes_generator/app.py:331`, `get_job/app.py:63` mapping `_dishes`→`shared_dishes`.
- `analyze_discard_on_upload_nodejs/recharacterize.js:33,351` `{owner}_dishes` reclassification.

### 2c. LIST — `_new_list` → `shared_shopping_list` + legacy `shared_list` ✅
Flags (live on `trepo-list-handler`): `DUAL_WRITE_ENABLED=true`, `LIST_UNION_READ=true`, `LIST_STRICT_UUID_MATCH=true`, `LIST_UNION_READ_OWNERS=''`.
- `APP/Lambdas/listHandler.js`: CREATE `:76`; READ single `:192`, union `:200-206`; backfill `:246,256`; fan-out add `:313`, dual-write shared_list `:354-358`; batch add `:408`+`:443-447`; remove `:471`+`:494-495`, **shared_shopping_list mark-REMOVED `:506-507` (always, flag-independent)**; remove-by-barcode `:525,542-543,555-556`; set_action `:578,600-601`; update `:630,650-651`; reorder `:669`.
- `playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs:669` + ~24 sites (2090,2163,2214,2390,2588,2697,2879,2921,3175,3749,3820,3877,3949,4038,4112,4140,4166,4362,4427,4465,4939,5051,5115,5195) — writes gated by `WRITE_SHARED_ONLY` (line 20).
- `analyze_discard_on_upload_nodejs/app.js:310` `{owner}_new_list`.
- `obs/voice-dualwrite-canary/index.mjs:156,164` — READ `{member}_new_list` (canary consistency audit).

### 2d. DISCARDS — `_discards` → `shared_discards` ⚠️ (writer NOT dual-writing)
- `discards_api/app.py:314,373,397` READ; `:447,493,509` WRITE `{owner|member}_discards` (**no shared write**).
- `analyze_discard_on_upload_nodejs/src/utils/mysqlDiscardWriter.ts:255,312,356` WRITE `{memberId}_discards` — **no dual-write to shared_discards**.
- `kitchen_analysis_generator/app.py:129`, `metrics_api/app.py:159` READ `{owner}_discards`.
- `analyze_discard_on_upload_nodejs/app.js:1320,1340` source_table refs.

### 2e. SAVED_RECIPES — `_saved_recipes` ❌ NO shared target
- `saved_recipes_api/app.py:502-506` `_saved_recipes_table()`; `:527-528` CREATE; reads/writes at 3520,3623,3631,3673,3775,3788,3803,3813,3875,3883. Flag `USE_SHARED_TABLES` present (`:29`) but no shared mapping → always per-user.

### 2f. MEAL_PLAN — `_meal_plan` ❌ NO shared target
- `meal_plan_api/app.py:105-110` table; `:136-137` CREATE; `:190,246` R/W.
- `meal_plan_generator/app.py:159-164` table; `:210-211` CREATE.
- `obs/voice-dualwrite-canary/_poll.mjs:9` READ `{owner}_meal_plan`.

### 2g. RECIPES — `_recipes` ❌ NO shared target
- `recipes_generator/app.py:287-291` table; `:341-342` CREATE; R/W 474,484,1010,1052,1155.
- `recipes_api/app.py:106-110` table; `:113-114` CREATE; `:212-215` READ.

### 2h. FEEDS — `_new_feed` / `_master_feed` / `_feed_events` ❌ NO shared target
- `APP/Lambdas/feedHandler.js:84,117,134` — fan-out WRITE `_new_feed` (no flags; deployed `trepo-feed-handler` 2025-11-14).
- `kitchen_api/master_feed.py:92`, `discards_api/master_feed.py:92`, `dishes_api/master_feed.py:92` — WRITE `{member}_master_feed`.
- `analyze_dish_on_upload_nodejs/masterFeedWriter.js:84`, `analyze_discard_on_upload_nodejs/masterFeedWriter.js:84`, `analyze_on_upload_nodejs/masterFeedWriter.js:84` — WRITE `{owner}_master_feed`.
- `analyze_dish_on_upload_nodejs/src/utils/mysqlFeedWriter.ts:53`, `analyze_discard_on_upload_nodejs/src/utils/mysqlFeedWriter.ts:52`, `analyze_on_upload_nodejs/src/utils/mysqlFeedWriter.ts:52` — WRITE `{memberId}_feed_events`.

### 2i. METRICS — `-metrics` ❌ NO shared target
- `metrics_api/app.py:138` WRITE, `:145,152,159` reads; `redemptions_api/app.py:165` WRITE; `kitchen_analysis_generator/app.py:108` WRITE.
- `analyze_dish_on_upload_nodejs/app.js:249`, `analyze_discard_on_upload_nodejs/app.js:286`, `analyze_on_upload_nodejs/app.js:271,521` WRITE snapshots.

### 2j. SIGNUP + HOUSEHOLD MIRRORING (creators of the whole batch)
- **`twilio-verify-lambda/index.mjs:489-786` `ensureUserTables()`** — CREATE TABLE IF NOT EXISTS for all 10 families on every signup: new_list `:502-521`, new_kitchen `:523-537`, new_feed `:539-555`, prod_kitchen `:557-598`, discards `:600-639`, dishes `:641-682`, recipes `:684-696`, saved_recipes `:698-724`, meal_plan `:726-740`, metrics `:742-764`. Called from `findOrCreateUser():935` and `findOrCreateUserByApple():1087`. (deployed `twilioAuth`, 2026-07-03)
- **Household mirroring** copies all families template→joiner: `twilio-verify-lambda/index.mjs:1114-1125` MIRRORED_TABLES, `:1233-1262` `mirrorHouseholdTables()`; `Lambda/household_membership_lambda/index.mjs:11-22` + `:227-256` (join), `:258-262`/`:354-408` (leave-clear). Both lists include `_redemptions` which is **never created at signup** (schema gap).

### 2k. INTROSPECTION / OPS (must not break on drop)
- `playground6_groceryRec/diagnostic_dashboard/server.js:1211-1213,1325-1327` and `analytics-routes.js:21-30` — regex-enumerate per-user tables at runtime.
- `playground6_groceryRec/sync_kitchen_tables.py` — utility syncing all `_new_kitchen`.
- `obs/voice-dualwrite-canary/*` + `obs/harness/flows/*.mjs` + `obs/alarm-triage/index.mjs` — the self-healing hourly canary reads per-user `_new_list`/`_meal_plan` to audit shopping/discard consistency.

---

## 3. EXACT ROW COUNTS — COMPLETE

Exact `COUNT(*)` per table, run 2026-07-12 after the RDS maintenance window (with reconnect retries). Full results are in §1's table. Summary:

- **Total per-user rows: ~84,689** across ~16,164 tables.
- Largest payloads (Phase D, no shared target): `_master_feed` 47,391 · `-metrics` 16,201 · `_feed_events` 6,191 · `_new_feed` 4,858 · `_saved_recipes` 2,032 · `_recipes` 682 · `_meal_plan` 349.
- Already-targeted families are small/residual: `_new_list` 1,575 · `_prod_kitchen` 1,473 (vs shared_kitchen 13,817 — kitchen genuinely migrated) · `_dishes` 1,064 (vs shared_dishes 382) · `_discards` 2,843 (vs shared_discards 17 — **discards NOT migrated**) · `_new_kitchen` 30 (dead).

Sizing implication: S3 archive export volume is modest (~85k rows total; on-disk ~1.4 GB dominated by empty-table overhead). Backfill effort is concentrated in the feed families (58k rows, 3 schemas to reconcile) and metrics (16k rows).

---

## 4. Deployed-vs-repo drift (read-only assessment)

Cannot byte-diff without downloading each package; below is a risk ranking from deploy timestamps + known facts. **Before editing any HIGH-risk fn, download deployed code (`aws lambda get-function --query Code.Location`) and diff against repo.**

| Deployed fn | Repo source | Last deploy | Drift risk | Why |
|---|---|---|---|---|
| `trepo-kitchen-handler` | `APP/Lambdas/kitchenHandler.js` | 2025-11-14 | **HIGH** | 8 months stale; repo has MIRRORED_TABLES/household logic that may post-date deploy. Still a live per-user writer. |
| `trepo-feed-handler` | `APP/Lambdas/feedHandler.js` | 2025-11-14 | **HIGH** | Same as above. |
| `grocery-identifier-dev-*` (bulk-commit, identify*, enrich-kitchen-item) | `grocery-identifier/*.js` | 2026-07-11/12 | **MED–HIGH** | Flagged as known-drifted "root writers". Root `bulkKitchenWriter.js` calls `KITCHEN_API_BASE_URL` (indirect write), but deployed variant may write per-user kitchen directly — **must diff before trusting**. |
| `twilioAuth` | `twilio-verify-lambda/index.mjs` | 2026-07-03 | **MED** | Owns signup table creation; repo may have edits not deployed (or vice-versa). Critical to get right before removing CREATE statements. |
| `trepo-list-handler` | `APP/Lambdas/listHandler.js` | 2026-07-11 | LOW | Fresh; flags match repo. |
| grocery-backend APIs (Kitchen/Saved/Meal/Recipes/Discards/etc.) | `playground6_groceryRec/*/app.py` | 2026-07-12 | LOW | Deployed today; flags match. |
| `trepo-quick-ack*` | `playground12_voice_ack/trepo-quick-ack` | 2026-07-12 | LOW | Fresh. |

---

## 5. Proposed sequence (SAFE order)

Guiding rule: **read cutover (flag) → verify via canary → stop dual-write → stop table creation at signup → archive → rename `zz_archive_` → 30-day soak → drop.** Never drop before a 30-day S3 export + rename soak. Families split into "cutover-only" (target exists) and "build-then-cutover" (target must be designed).

### Phase A — Decommission dead weight (lowest risk, do first)
1. **`shared_list`** (legacy write-only, 1,365 rows, no readers): stop the `DUAL_WRITE_ENABLED` writes to it in `listHandler.js` → rename `zz_archive_shared_list` → 30-day soak → drop. (No per-user table involved.)

### Phase B — Close straggler per-user writers for already-migrated families
2. **Kitchen writer parity:** make `trepo-kitchen-handler` (`kitchenHandler.js`) write `shared_kitchen` (or route its callers to `kitchen_api`), then stop `_new_kitchen` writes. Diff deployed vs repo first (HIGH drift).
3. **Feed writer:** feeds have no shared target yet — defer to Phase D; for now just inventory `feedHandler.js` callers.
4. **Discards dual-write fix:** add `shared_discards` write to `mysqlDiscardWriter.ts` + `discards_api` (currently 17 rows = essentially unmigrated), backfill per-user `_discards`→`shared_discards`, verify with canary.

### Phase C — Cut over & retire families WITH a shared target
For **list, kitchen, dishes, discards** (once B4 done):
5. Confirm 100% read cutover (`USE_SHARED_TABLES`/`LIST_UNION_READ` already true) — audit the un-flagged reader `explore_recipes_api/app.py:330`.
6. Backfill any residual per-user rows into shared (list: 1,575 rows across 197 tables still need reconciling vs shared_shopping_list; kitchen already live; dishes/discards backfill).
7. Flip to shared-only writes (already `WRITE_SHARED_ONLY=true` for kitchen; extend to dishes/discards; stop listHandler per-user fan-out).
8. Canary green for 1–2 weeks → `zz_archive_` rename the 4 families' per-user tables → 30-day S3 export soak → drop.

### Phase D — Design + build shared targets, then cut over (the real work)
For **saved_recipes, meal_plan, recipes, feeds, metrics** (no target today):
9. Design shared schemas (§6), create tables + indexes.
10. Add dual-write in each API/worker (saved_recipes_api, meal_plan_api/generator, recipes_api/generator, feed writers, metrics writers).
11. Backfill per-user → shared (owner_id = table prefix).
12. Add flag-gated read cutover; verify parity via canary/harness.
13. Flip to shared-only writes.
14. Archive/rename/drop per-user tables per the standard soak.

### Phase E — Stop creation & mirroring at the source
15. Once a family is fully retired, remove its `CREATE TABLE` from `twilio-verify-lambda/index.mjs:ensureUserTables()` and its entry from both `MIRRORED_TABLES` lists (twilio + household_membership). Do this **last per family** so in-flight/rolled-back deploys don't recreate dropped tables.
16. Fix the `_redemptions` gap (mirrored but never created) as part of this cleanup.
17. Update `diagnostic_dashboard` + `sync_kitchen_tables.py` + `obs` canary/harness to stop scanning dropped families.

### Phase F — Final drop
18. After all families archived + soaked: drop `zz_archive_*` per-user tables in batches; keep the 30-day S3 exports as cold backup.

---

## 6. New shared tables to design (Phase D)

> **✅ PHASE 1 UPDATE (2026-07-12): 4 of the 5 tables are BUILT with the REAL schemas.**
> The SQL sketches below were GUESSES and diverged materially from the actual
> per-user schemas — corrected and created via `playground6_groceryRec/migrations/shared_phase1_create_tables.sql`.
> **Real-schema corrections (authoritative, pulled from `saved_recipes_api._ensure_*` + twilio `ensureUserTables()`):**
> - **`shared_saved_recipes`** — 24 cols (full recipe row incl. `source_image_url(s)`, `image_storage_key`, `raw_caption/content`). Dedupe unique is **per-owner**: `UNIQUE(owner_id, resolved_url_hash)` (the per-user table's global unique becomes per-owner — two users can save the same URL).
> - **`shared_recipes`** & **`shared_meal_plan`** — these are **one-row-per-user CACHES**, not recipe/plan logs: the per-user INSERT literally uses `_id='current'`, columns are `status`/`kitchen_only`/`need_grocery`/`error_message` (recipes) and `status`/`focus`/`explanation_title`/`explanation_paragraph`/`plan`/`error_message` (meal_plan). Because `_id='current'` is identical for every user, **`owner_id` is the PRIMARY KEY** (one row per owner) and the dual-write is an upsert (`INSERT … ON DUPLICATE KEY UPDATE`).
> - **`shared_metrics`** — denormalized snapshot LOG (append; `_id` is a UUID, `_createdDate=NOW()`): `IQ`/`Points`/`UPF`/`harmful_ingredients` + `*_what`/`*_suggestions` + `kitchen_analysis_*`. `owner_id` indexed (NOT unique). Source used `utf8mb4_unicode_ci`; **standardized `owner_id` (and table) to `utf8mb4_0900_ai_ci`** to match `shared_kitchen` and avoid JOIN-collation errors.
> - **`owner_id VARCHAR(36)`** (not 64) to match the UUID sources; every table has `INDEX(owner_id)` + `INDEX(owner_id, _createdDate)`.
> - **Feeds (`shared_feed`)** — DEFERRED: the 3 sources (`_new_feed` product-centric, `_master_feed` event-centric, `_feed_events`) are genuinely different schemas; shape TBD after a reader-consumption analysis.

Pattern (mirror existing `shared_kitchen`): add `owner_id VARCHAR(64)` as the tenancy key + keep `_owner`/`_device` for back-compat, `INDEX(owner_id)`, plus family-specific indexes. `owner_id` value = the UUID prefix of the source per-user table.
> _(The original sketch prose + SQL below is retained for history but is SUPERSEDED by the real schemas above / the migration SQL file.)_

**`shared_saved_recipes`** (from `_saved_recipes`, ~1.6k rows / 1,576 tables)
```sql
CREATE TABLE shared_saved_recipes (
  _id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
  _owner VARCHAR(64), _device VARCHAR(64), _createdDate DATETIME, _updatedDate DATETIME,
  recipe_title TEXT, source_url TEXT, image_url TEXT, ingredients JSON, steps JSON,
  attributed_source VARCHAR(255), saved_from VARCHAR(64), metadata JSON,
  INDEX ix_owner (owner_id), INDEX ix_owner_created (owner_id, _createdDate)
);  -- final columns: mirror saved_recipes_api CREATE at app.py:527-528
```

**`shared_meal_plan`** (from `_meal_plan`, ~343 rows / 1,580 tables)
```sql
CREATE TABLE shared_meal_plan (
  _id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
  _owner VARCHAR(64), _device VARCHAR(64), _createdDate DATETIME, _updatedDate DATETIME,
  plan_date DATE, meal_slot VARCHAR(32), recipe_ref VARCHAR(64), payload JSON,
  INDEX ix_owner (owner_id), INDEX ix_owner_date (owner_id, plan_date)
);  -- mirror meal_plan_api CREATE at app.py:136-137
```

**`shared_recipes`** (from `_recipes`, ~2.3k rows / 1,580 tables)
```sql
CREATE TABLE shared_recipes (
  _id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
  _owner VARCHAR(64), _device VARCHAR(64), _createdDate DATETIME, _updatedDate DATETIME,
  title TEXT, ingredients JSON, steps JSON, source VARCHAR(64), generated_from JSON,
  INDEX ix_owner (owner_id), INDEX ix_owner_created (owner_id, _createdDate)
);  -- mirror recipes_api CREATE at app.py:113-114
```

**`shared_feed`** (unify `_new_feed` + `_master_feed` + `_feed_events`; largest by rows, ~53k combined)
```sql
CREATE TABLE shared_feed (
  _id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
  _owner VARCHAR(64), _device VARCHAR(64), _createdDate DATETIME,
  event_type VARCHAR(48), source_feed ENUM('new_feed','master_feed','feed_events'),
  entity_ref VARCHAR(64), payload JSON,
  INDEX ix_owner_created (owner_id, _createdDate), INDEX ix_owner_type (owner_id, event_type)
);  -- reconcile 3 feed schemas before building; feedHandler.js + master_feed.py + mysqlFeedWriter.ts
```

**`shared_metrics`** (from `-metrics`, ~14k rows / 1,582 tables — highest row count)
```sql
CREATE TABLE shared_metrics (
  _id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
  _owner VARCHAR(64), _device VARCHAR(64), snapshot_at DATETIME,
  metric_name VARCHAR(64), metric_value DOUBLE, dimensions JSON,
  INDEX ix_owner_time (owner_id, snapshot_at), INDEX ix_owner_metric (owner_id, metric_name)
);  -- reconcile metrics_api CREATE (twilio index.mjs:742-764) + redemptions writes
```

> Column lists above are sketches — pull exact columns from `ensureUserTables()` CREATE blocks (`twilio-verify-lambda/index.mjs`) and each API's `_ensure_*_table()` before building. Consider partitioning `shared_metrics`/`shared_feed` by `owner_id` hash or a monthly range if they grow past a few M rows.

---

## 7. Risk register

| Risk | Family | Severity | Mitigation |
|---|---|---|---|
| Deployed `kitchenHandler`/`feedHandler` (Nov-2025) drift from repo → editing repo doesn't match prod | kitchen, feed | HIGH | Download + diff deployed code before any change; redeploy from repo to re-baseline first. |
| grocery-identifier "root writers" may write per-user kitchen directly in prod despite repo using API path | kitchen | HIGH | Diff deployed grocery-identifier package; trace actual DB writes before assuming shared_kitchen coverage. |
| `WRITE_SHARED_ONLY=true` is set on saved_recipes/meal_plan/recipes APIs but has **no effect** (no shared mapping) → false sense of "migrated" | 3 families | HIGH | Treat these as fully per-user; do not drop on flag-state alone. |
| Discards "migrated" but writer never dual-writes shared_discards (17 rows) | discards | HIGH | Fix dual-write + backfill before any archive. |
| `explore_recipes_api:330` reads `_prod_kitchen` possibly un-gated | kitchen | MED | Confirm flag-gating; route to shared_kitchen before dropping `_prod_kitchen`. |
| Signup `ensureUserTables()` recreates dropped tables on next signup/rollback | all | MED | Remove CREATE + mirror entries only AFTER family fully retired (Phase E, last). |
| Household mirror copies large per-user tables on join; and `_redemptions` mirrored-but-not-created | all | MED | Remove from MIRRORED_TABLES in lock-step; fix `_redemptions` gap. |
| Canary/harness/diagnostic-dashboard read per-user tables → error/alert storm on drop | list, meal_plan, all | MED | Update obs canary, harness flows, diagnostic_dashboard, sync_kitchen_tables.py before drop; canary is the parity gate — keep it green as go/no-go. |
| Irreversible DROP | all | HIGH | `zz_archive_` rename + 30-day S3 export soak before every DROP; keep exports as cold backup. |
| Rollback mid-cutover | all | — | Per-step rollback = flip flag back / re-enable dual-write. Archive-rename is reversible (rename back) within soak window. Only DROP is terminal. |

**Canary's role:** the self-healing hourly canary (`obs/voice-dualwrite-canary/`) is the **go/no-go gate** — a family is only eligible to advance from dual-write → shared-only → archive when the canary reports parity green for the agreed soak. It must be extended to cover each new family (currently audits list/shopping + discard).

---

## 8. Effort estimate (per step)

| Phase | Work | Est. effort |
|---|---|---|
| A | Retire `shared_list` dead weight | 0.5 day + 30-day soak |
| B2 | Kitchen straggler writer parity (diff + fix + deploy `kitchen-handler`) | 1–2 days |
| B4 | Discards dual-write + backfill + verify | 1–2 days |
| C | Cut over + archive list/kitchen/dishes/discards | 2–3 days + soak |
| D (×5) | Design+build+dual-write+backfill+cutover per no-target family | 2–4 days **each** (10–20 days total); feeds hardest (3-schema merge) |
| E | Strip signup CREATE + mirror entries + fix `_redemptions` + update ops tooling | 2–3 days |
| F | Final drops after soaks | 1 day (spread over weeks of soak) |

**Rough total: ~4–6 weeks of engineering** + overlapping 30-day archive soaks. The five build-then-cutover families (Phase D) dominate; the four already-targeted families (Phase C) are days, not weeks.

---

## 9. Open questions to resolve before executing

1. Is `trepo-kitchen-handler`/`trepo-feed-handler` still in the live request path, or superseded by grocery-backend `kitchen_api`? (Determines whether B2 is "fix" or "delete".)
2. Do the deployed `grocery-identifier` fns write per-user kitchen directly? (Diff required.)
3. `_new_list` still holds 1,575 rows / 197 tables while `shared_shopping_list` has only 134 — reconcile which is authoritative before archiving (dual-write is on, so per-user should be a superset; confirm shared has all live items).
4. Should feeds collapse into one `shared_feed` or keep `_new_feed` (app feed) separate from analytics `_master_feed`/`_feed_events`?
5. Confirm no analytics/BI job (`trepo-analytics-*`, `bench/`) reads per-user tables outside the scanned scope.
