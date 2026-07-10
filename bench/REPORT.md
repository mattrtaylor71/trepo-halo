# Trepo Model-Upgrade Benchmark — Results

**Date:** 2026-07-10 · **Harness:** `bench/` (committed) · **Mode:** local API calls only — **no prod changes, no env flips, no deploys.**
Real prompts, real tool schemas, real kitchen/image data. Judge = `claude-opus-4-8` (blind). Matt picks from this evidence.

> **Pricing caveat (read first):** 4.x/4o rates are OpenAI list prices. **gpt-5.x rates are ESTIMATES** (`est` flag in `lib/pricing.mjs`) — the models resolve on our key but their list price isn't confirmed. **Token counts are exact** (measured from `usage`); dollar figures are token×price and should be re-derived once real 5.x pricing is known. Every estimated number below is marked *(EST)*.

---

## TL;DR — recommended upgrade matrix

| Surface | Currently deployed | Recommended | Why | Monthly cost delta (dev volume) |
|---|---|---|---|---|
| **Thyme — iOS voice** | `gpt-4.1` | **`gpt-5.4-2026-03-05`** (HALO parity) | Fixes a **13.9% fabricated-confirmation rate** → 0%; 75%→100% tool-correct. And it's **cheaper per turn** ($0.0114 vs $0.0172). | **−34% per turn** *(EST)* — upgrade pays for itself |
| **Recipes** | `gpt-4.1-mini` | **`gpt-5.5`** (quality) or hold | Blind judge winner on both kitchens; composes real dishes vs trivially-simple ones. ~11× cost though. | +$50/mo @1.3k calls *(EST)* |
| **Fast photo identify** | `gpt-4.1-mini` | **Hold** (or `gpt-5.4-mini` for cost/latency) | No candidate beats current accuracy (0.8). `gpt-5.4-mini` = 2× faster + cheaper at −0.1 acc. `gpt-5.4` strictly worse. | mini: **cheaper**; 5.4: +3× *(EST)* |
| **Home suggestions** | `gpt-4o-mini` (hardcoded) | **`gpt-5.4-mini`** | Big quality jump on rich kitchens (4o-mini ignores inventory context); 3× faster than gpt-5.4 → safe vs 22s deadline; hits exactly 30. | +$8/mo @6k calls *(EST)* |

**Highest-value, lowest-risk single change: Thyme iOS `gpt-4.1` → `gpt-5.4`.** It removes a data-integrity bug (fabricated "it's been logged" with no write) and is cheaper per turn.

---

## Method

- **Real prompts, not paraphrases.** Thyme imports the deployed `buildSystemPrompt()` (23,749-char system prompt) + `buildChatTools()` (all 48 tool schemas) directly from `playground12_voice_ack/trepo-quick-ack/lib/realtime-config.mjs`. Recipes/identify/home prompts are copied verbatim from the Lambda source (`prompts/`, and inline in each script).
- **Real data.** Kitchen inventories pulled from `shared_kitchen` (household-scoped, mirroring `recipes_generator._get_kitchen_ingredients`): test user (5 items) + owner `3ed72da8` (190 items → capped to 80 = prod behavior). 10 real capture images from S3 with deep-path DB ground-truth labels, incl. **the Merlot image**.
- **Blind judge.** `claude-opus-4-8` scores anonymized, shuffled sets (no model identity) for recipes & home.
- **Stochastic ops repeated.** Thyme tool-calling run 3× per case (tool-calling is nondeterministic); rates reported.
- **Native discipline, not the regex.** Thyme runs `tool_choice:"auto"` to measure the *model's* judgment. Production additionally applies a `detectWriteIntent` regex that force-calls a tool on first-turn write phrasings — that would paper over model gaps on regex-matching commands, but **not** on read-backs, pure-chat, or the self-poison case (annotated per row in `out/thyme_raw.json`).

---

## 1. Thyme — tool discipline + quality  (12 cases × 3 reps × 4 models = 144 calls)

| Model | Correct | **Fabrication** | No-tool | Wrong-tool | Median latency | $/turn |
|---|---|---|---|---|---|---|
| **gpt-4.1** (current iOS) | **75.0%** | **13.9%** | 11.1% | 0% | 966 ms | $0.0172 |
| gpt-5.4-2026-03-05 (HALO) | **100%** | 0% | 0 | 0 | 1682 ms | $0.0114 *(EST)* |
| gpt-5.5 | **100%** | 0% | 0 | 0 | 2577 ms | $0.0117 *(EST)* |
| gpt-5.6-sol | **100%** | 0% | 0 | 0 | 2016 ms | $0.0176 *(EST)* |

**⭐ Standout — the fabrication bug.** On *"checked into my kitchen: black beans and salsa"*, **gpt-4.1 replied `"Black beans and salsa have been added to your kitchen inventory."` with NO tool call — 3/3 times.** The item is never written; the user is told it was. This is the exact failure the system prompt's "never confirm an action you did not perform" rule targets, and the **current iOS model violates it.** All three gpt-5.x models correctly call `check_in_item`. gpt-4.1 also under-fires on terse consumption (*"I just had an iced coffee"* → asks a clarifying question instead of logging, 3/3).

**Self-poisoning case passed by everyone.** After a fake prior assistant *"your lunch has been logged ✅"* in history, all four models correctly **re-called** the log tool on repeat (didn't trust the poisoned history).

**Caveat — gpt-5.6-sol:** rejects function tools in `/v1/chat/completions` unless `reasoning_effort:'none'` (or you move to `/v1/responses`). Benchmarked reasoning-OFF. Real integration cost if adopted.

**Recommendation: iOS `gpt-4.1` → `gpt-5.4-2026-03-05`.** Removes the fabrication risk, matches HALO (already on it), and is ~34% cheaper per turn *(EST)*. Latency rises ~0.7s (acceptable for a voice turn; gpt-5.5/sol are slower with no quality gain here).

---

## 2. Recipe generation  (10+10 recipes, 2 kitchens, 3 models; blind opus judge)

Grounding was strong for all models (programmatic check ≈1.0; 0 phantom ingredients). The separation is **appeal / practicality**, from the blind judge:

| Kitchen | Model | Appeal | Practicality | Grounding | Judge verdict |
|---|---|---|---|---|---|
| test user (5 items) | gpt-4.1-mini | 7 | 8 | 9 | clean, realistic |
| | gpt-5.4 | 4 | 4 | 6 | ⚠ "odd banana-egg custard/scramble combos" |
| | **gpt-5.5** | **8** | **8** | **9** | 🏆 winner |
| owner 3ed (190→80) | gpt-4.1-mini | 6 | 8 | 6 | ⚠ "several kitchen_only recipes trivially simple" |
| | gpt-5.4 | 8 | 8 | 7 | strong |
| | **gpt-5.5** | **9** | **8** | **8** | 🏆 winner |

**⭐ Standout — "recipes" that aren't recipes.** On the big kitchen, gpt-4.1-mini emitted `"Vegetable Frittata"` whose only ingredient is the prepared item *"Homemade Skillet Vegetable Frittata"* — i.e. it re-served an inventory item as a recipe. gpt-5.5 composed real dishes: **Turkey Avocado Burgers** (correctly pairing the *Hamburger Buns* + *Ground Turkey* already in the kitchen), Shrimp Avocado Salad, Cajun Sausage Long-Bean Skillet. Full 3-per-model samples in `out/recipes_samples_owner3ed.json`.

**Note:** gpt-5.4 *regressed* vs the current model on the **small** kitchen (contrived combos). gpt-5.5 is the consistent winner. Cost: gpt-5.5 ≈ 11× gpt-4.1-mini (**+~$50/mo** at dev volume, EST). **Recommend gpt-5.5 if recipe quality is a priority; otherwise hold** — current grounding is fine, the gap is polish/variety.

---

## 3. Fast photo identify  (10 real images, deep-path ground truth, 3 models)

| Model | Top-label acc | Partial | Wrong | Median lat | p90 lat | $/1k images | Merlot guess |
|---|---|---|---|---|---|---|---|
| **gpt-4.1-mini** (current) | **0.80** | 0.10 | 0.10 | 2341 ms | 2853 ms | $1.08 | ❌ "Premium Dark Chocolate" |
| gpt-5.4-mini | 0.70 | 0.20 | 0.10 | **1348 ms** | 2002 ms | **$0.63** *(EST)* | ❌ "Bottled soda" |
| gpt-5.4 | 0.60 | 0.30 | 0.10 | 2600 ms | 3019 ms | $3.10 *(EST)* | ❌ "rotisserie chicken" |

**⭐ Standout — the Merlot image defeats the entire fast tier.** Deep-path truth = *"Merlot Wine"*. All three fast models miss it, and differently: chocolate / soda / rotisserie chicken. This is exactly the case where **deep-path escalation earns its keep** — no fast model should be trusted to finalize this bottle. (Several other "misses" are ground-truth label quirks, e.g. *"Half & Half Blend"* is actually a salad-greens blend, which the models call *"salad greens"* — arguably more correct than the stored label.)

**No upgrade improves accuracy.** The current model is the most accurate here. `gpt-5.4-mini` is the only interesting alternative: ~2× faster and **cheaper** *(EST)*, at −0.10 accuracy — worth considering purely for latency/cost on this volume-sensitive tier. **gpt-5.4 is strictly worse** (slower, 3× cost, least accurate). **Recommend: hold gpt-4.1-mini** (or trial gpt-5.4-mini if fast-tier latency/cost becomes a pain point). Production caps this tier at 220 output tokens — reasoning models blow through that on reasoning tokens, so any 5.x adoption here needs `reasoning_effort:'none'` or a higher cap.

---

## 4. Home suggestions  (30 items, 2 kitchens, 3 models; blind opus judge; 22s deadline)

| Kitchen | Model | Relevance | Practicality | Latency | Fits 22s? | Count | $/call |
|---|---|---|---|---|---|---|---|
| test user | gpt-4o-mini (current) | 7 | 8 | 10.2s | ✅ | 33 | $0.0008 |
| | gpt-5.4-mini | 8 | 8 | **5.5s** | ✅ | 30 | $0.0022 *(EST)* |
| | **gpt-5.4** | **9** | **9** | 17.2s | ⚠ tight | 30 | $0.0135 *(EST)* |
| owner 3ed | **gpt-4o-mini** | **5** | **6** | 11.4s | ✅ | 32 (+1 dup) | $0.0008 |
| | gpt-5.4-mini | 8 | 8 | **5.0s** | ✅ | 30 | $0.0022 *(EST)* |
| | gpt-5.4 | 9 | 8 | 13.1s | ✅ | 30 | $0.0135 *(EST)* |

**⭐ Standout — 4o-mini ignores the kitchen.** On the rich cocktail-heavy kitchen, gpt-4o-mini scored **5/6**: *"generic healthy-eating list that ignores inventory context"* (quinoa, chickpeas, Greek yogurt). The 5.x models noticed the actual gaps — *"hamburger buns without beef,"* citrus/mixers for the huge liquor inventory. gpt-4o-mini also overshoots the "exactly 30" instruction (32–33) and duplicated a kitchen item.

**Recommend: `gpt-4o-mini` → `gpt-5.4-mini`.** Near-gpt-5.4 quality (8/8 vs 9/8), **3× faster** (5s vs 13–17s — important against the 22s hard deadline, where gpt-5.4's 17.2s is uncomfortably close and would risk timeouts/fallback under load), exactly 30 items, no dups. **~+$8/mo** at 6k calls *(EST)*. gpt-5.4 is best-quality but the latency risk + 17× cost aren't worth it over the mini. This surface also has a code bug worth fixing separately: the model is **hardcoded** (`model="gpt-4o-mini"`), so it can't be changed by env — an upgrade needs a one-line code edit.

---

## Cost math (dev-account volumes — scale to prod)

30-day invocation counts (CloudWatch, this dev account): Home **6,008** (~200/day), Recipes **1,283** (~42/day), Thyme assistant & IdentifyFast read ~0 (low/differently-routed dev traffic — per-call economics matter more than dev totals here).

| Surface | Current $/mo | Recommended $/mo | Δ | Note |
|---|---|---|---|---|
| Home (6,008) | $4.87 (4o-mini) | $13.32 (5.4-mini) | **+$8.45** *(EST)* | best value upgrade |
| Recipes (1,283) | $4.82 (4.1-mini) | $55.03 (5.5) | +$50.21 *(EST)* | quality-driven, optional |
| Thyme (per-turn) | $0.0172 (4.1) | $0.0114 (5.4) | **−34%** *(EST)* | cheaper AND better |
| Identify (per-img) | $0.00108 (4.1-mini) | hold / $0.00063 (5.4-mini) | −42% if switched *(EST)* | keep accuracy → hold |

All dollar deltas on estimated 5.x pricing; **token deltas are exact** in `out/*_raw.json`.

---

## Limitations
- gpt-5.x **pricing is estimated** — confirm before quoting cost deltas as fact.
- Thyme grounding is a simplified static snapshot (real prod injects Oura/DynamoDB/food-scoring); this exercises tool *choice*, which is the point, not grounding fidelity.
- Identify uses `/v1/chat/completions` + `json_object` with a raised token cap (prod fast path uses the Responses API + strict json_schema + 220-token cap); all models get the identical call, so the comparison is fair, but absolute prod latency will differ.
- Sample sizes are modest by design (single-digit-dollar budget): 12 Thyme cases, 2 kitchens, 10 images. Directional, not census.

## Reproduce
```
mkdir -p /tmp/bench_secrets   # drop openai.key + anthropic.key (from Lambda env)
cd bench && npm install
node fixtures/pull_kitchen.mjs && node fixtures/pull_identify.mjs   # refresh real data
node 01_thyme.mjs && node 02_recipes.mjs && node 03_identify.mjs && node 04_home.mjs
```
Raw per-call outputs (every prompt, tool call, score, token count) are in `bench/out/*.json`.
