// BENCHMARK 2 — Recipe generation. Real recipes_generator prompt (verbatim) on real
// kitchen inventories: test user (5 items) + owner 3ed72da8 (190 items, capped to 80
// most-recent = production behavior). Models: gpt-4.1-mini (current) vs gpt-5.4 vs
// gpt-5.5. Metrics: programmatic inventory-grounding + variety, plus a BLIND opus judge
// (sets anonymized + shuffled) scoring appeal/practicality/grounding 1-10.
import { readFileSync, writeFileSync } from "node:fs";
import { chat } from "./lib/openai.mjs";
import { judge } from "./lib/anthropic.mjs";
import { costUSD } from "./lib/pricing.mjs";
import { scoreRecipeSet } from "./lib/grounding.mjs";

const MODELS = ["gpt-4.1-mini", "gpt-5.4", "gpt-5.5"];
const TEMPERATURE = 0.4;
const MAX_INGREDIENTS = 80; // RECIPE_GEN_MAX_INGREDIENTS
const SYSTEM = readFileSync(new URL("./prompts/recipes_system.txt", import.meta.url), "utf8");
const KITCHENS = {
  testuser: JSON.parse(readFileSync(new URL("./fixtures/kitchen_testuser.json", import.meta.url))),
  owner3ed: JSON.parse(readFileSync(new URL("./fixtures/kitchen_owner3ed.json", import.meta.url))),
};

function formatKitchen(items) {
  return items.map((i) => (i.description ? `${i.product_name} (${i.description})` : i.product_name)).join(", ")
    || "No specific ingredients (suggest pantry staples)";
}
function parseJson(text) {
  const f = text.match(/```(?:json)?\s*([\s\S]*?)```/); const c = f ? f[1] : text;
  try { return JSON.parse(c); } catch {}
  try { return JSON.parse(c.slice(c.indexOf("{"), c.lastIndexOf("}") + 1)); } catch { return null; }
}
// deterministic shuffle (no Math.random dependency for reproducibility)
function seededShuffle(arr, seed) {
  const a = [...arr]; let s = seed;
  for (let i = a.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) & 0x7fffffff; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const raw = {};
for (const [kLabel, kitchen] of Object.entries(KITCHENS)) {
  const items = kitchen.items.slice(0, MAX_INGREDIENTS);
  const user = `Kitchen grocery products (use exact product name in recipe ingredients; descriptions in parentheses clarify what each item is): ${formatKitchen(items)}\n\n`
    + `Generate exactly 10 kitchen_only recipes using ONLY these items (plus pantry), and exactly 10 need_grocery recipes that use some of these items but need extra ingredients to buy (include missing_ingredients for each).\nReturn the JSON object only.`;
  raw[kLabel] = {};
  for (const model of MODELS) {
    const res = await chat({ model, messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }], temperature: TEMPERATURE, max_tokens: 6000 });
    const parsed = res.ok ? parseJson(res.text) : null;
    const metrics = parsed ? scoreRecipeSet(parsed, kitchen.items) : { error: "parse_failed" };
    raw[kLabel][model] = { ok: res.ok, error: res.error || null, latencyMs: res.latencyMs, usage: res.usage, cost: costUSD(model, res.usage), reasoningOff: !!res.reasoningOff, set: parsed, metrics };
    console.log(`  recipes ${kLabel} ${model}: grounding=${metrics.groundingRate} phantom=${metrics.phantomIngredientsInKitchenOnly} proteins=${metrics.distinctProteins} lat=${res.latencyMs}ms`);
  }
}

// ---- BLIND JUDGE (opus) — per kitchen, anonymized + shuffled sets ----
for (const [kLabel, kitchen] of Object.entries(KITCHENS)) {
  const entries = MODELS.map((m) => ({ model: m, set: raw[kLabel][m].set })).filter((e) => e.set);
  const shuffled = seededShuffle(entries, 4242 + kLabel.length);
  const labels = shuffled.map((_, i) => String.fromCharCode(65 + i)); // A,B,C
  const kitchenList = kitchen.items.slice(0, MAX_INGREDIENTS).map((i) => i.product_name).join(", ");
  const sys = "You are a meticulous culinary judge. You are given a kitchen inventory and several ANONYMIZED recipe sets (each with kitchen_only and need_grocery recipes). Score each set objectively. Return ONLY JSON.";
  const body = `KITCHEN INVENTORY:\n${kitchenList}\n\n`
    + shuffled.map((e, i) => `=== RECIPE SET ${labels[i]} ===\n${JSON.stringify(e.set).slice(0, 9000)}`).join("\n\n")
    + `\n\nScore each set 1-10 on: appeal (would a real person want to cook & eat these), practicality (normal weeknight-makeable, coherent steps, no contrived combos), grounding (kitchen_only recipes truly use only inventory+pantry; need_grocery correctly flag missing items). Return JSON: {"scores":{"${labels.join('":{"appeal":0,"practicality":0,"grounding":0,"comment":""},"')}":{"appeal":0,"practicality":0,"grounding":0,"comment":""}},"winner":"LETTER","why":""}`;
  const jr = await judge({ system: sys, user: body, maxTokens: 1600 });
  const mapping = Object.fromEntries(shuffled.map((e, i) => [labels[i], e.model]));
  raw[kLabel].__judge = { mapping, verdict: jr.json, winnerModel: jr.json?.winner ? mapping[jr.json.winner] : null };
  console.log(`  JUDGE ${kLabel}: winner set ${jr.json?.winner} = ${raw[kLabel].__judge.winnerModel}`);
}

writeFileSync(new URL("./out/recipes_raw.json", import.meta.url), JSON.stringify(raw, null, 2));
console.log("\nwrote out/recipes_raw.json");
