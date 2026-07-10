// BENCHMARK 4 — Home suggestions. Real home_suggestions prompt (verbatim) on real
// kitchen context for the test user + owner 3ed72da8's active household. Current model
// = gpt-4o-mini (HARDCODED in prod) vs gpt-5.4-mini vs gpt-5.4. temperature 0.7,
// json_object, 30 suggestions. Blind opus judge for relevance/practicality. Latency is
// gated: the endpoint has a 22s deadline — a candidate that doesn't fit is disqualified.
import { readFileSync, writeFileSync } from "node:fs";
import { chat } from "./lib/openai.mjs";
import { judge } from "./lib/anthropic.mjs";
import { costUSD } from "./lib/pricing.mjs";

const MODELS = ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4"];
const TEMPERATURE = 0.7;
const DEADLINE_MS = 22000; // prod HOME_SUGGESTIONS_LLM_TIMEOUT_SECONDS
const KITCHENS = {
  testuser: JSON.parse(readFileSync(new URL("./fixtures/kitchen_testuser.json", import.meta.url))),
  owner3ed: JSON.parse(readFileSync(new URL("./fixtures/kitchen_owner3ed.json", import.meta.url))),
};
function buildPrompt(items) {
  const itemsText = items.map((i) => `- ${i.product_name} (category: ${i.category || "Unknown"})`).join("\n") + "\n";
  return `You are a smart grocery shopping assistant. Based on the user's current kitchen contents below, suggest exactly 30 grocery items they likely need to buy this week.

IMPORTANT: Each item_name must be a specific, concrete product — something you'd actually write on a shopping list. NOT a category or vague description.
- Good: "Spinach", "Chicken Breast", "Greek Yogurt", "Yellow Onions", "Coconut Milk"
- Bad: "Fresh Vegetables (e.g. spinach or kale)", "Protein Source", "Healthy Snacks", "Dairy Products"

Consider:
- Items that are expiring soon or may have run out
- Common complementary items that go with what they already have
- Staples that are missing from their kitchen
- Variety and nutritional balance
- Do NOT suggest items the user already has on their shopping list

Current kitchen contents:
${itemsText}

Respond with a JSON object in this exact format:
{
  "suggestions": [
    {"item_name": "string", "reason": "short reason why they need it", "confidence": 0.0}
  ]
}

The confidence should be a float between 0.0 and 1.0 indicating how confident you are they need this item. Keep item_name to 1-3 words max. Provide exactly 30 suggestions. Return ONLY valid JSON, no other text.`;
}
function parseJson(t) { const f = t.match(/```(?:json)?\s*([\s\S]*?)```/); const c = f ? f[1] : t; try { return JSON.parse(c); } catch {} try { return JSON.parse(c.slice(c.indexOf("{"), c.lastIndexOf("}") + 1)); } catch { return null; } }
function seededShuffle(arr, seed) { const a = [...arr]; let s = seed; for (let i = a.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) & 0x7fffffff; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; }

const raw = {};
for (const [kLabel, kitchen] of Object.entries(KITCHENS)) {
  raw[kLabel] = {};
  const prompt = buildPrompt(kitchen.items);
  for (const model of MODELS) {
    const res = await chat({ model, messages: [{ role: "user", content: prompt }], temperature: TEMPERATURE, response_format: { type: "json_object" }, max_tokens: 2048 });
    const parsed = res.ok ? parseJson(res.text) : null;
    const sugg = parsed?.suggestions || [];
    const names = sugg.map((s) => (s.item_name || "").toLowerCase());
    const kitchenNames = new Set(kitchen.items.map((i) => (i.product_name || "").toLowerCase()));
    const dupWithKitchen = names.filter((n) => kitchenNames.has(n)).length;
    const dupInternal = names.length - new Set(names).size;
    raw[kLabel][model] = {
      ok: res.ok, error: res.error || null, latencyMs: res.latencyMs, withinDeadline: res.latencyMs <= DEADLINE_MS,
      count: sugg.length, dupWithKitchen, dupInternal, usage: res.usage, cost: costUSD(model, res.usage),
      reasoningOff: !!res.reasoningOff, suggestions: sugg,
    };
    console.log(`  home ${kLabel} ${model}: n=${sugg.length} lat=${res.latencyMs}ms fit22s=${res.latencyMs <= DEADLINE_MS} dupKitchen=${dupWithKitchen}`);
  }
}

// ---- BLIND JUDGE (opus): relevance/practicality per kitchen, anonymized+shuffled ----
for (const [kLabel, kitchen] of Object.entries(KITCHENS)) {
  const entries = MODELS.map((m) => ({ model: m, sugg: raw[kLabel][m].suggestions })).filter((e) => e.sugg?.length);
  const shuffled = seededShuffle(entries, 7777 + kLabel.length);
  const labels = shuffled.map((_, i) => String.fromCharCode(65 + i));
  const kitchenList = kitchen.items.map((i) => i.product_name).join(", ");
  const sys = "You judge grocery shopping-list suggestions given a kitchen inventory. Score each ANONYMIZED set. Return ONLY JSON.";
  const body = `KITCHEN INVENTORY:\n${kitchenList}\n\n`
    + shuffled.map((e, i) => `=== SUGGESTION SET ${labels[i]} ===\n${e.sugg.map((s) => `${s.item_name} — ${s.reason}`).join("\n").slice(0, 6000)}`).join("\n\n")
    + `\n\nScore each set 1-10 on: relevance (do these complement THIS kitchen / fill real gaps, not random staples), practicality (concrete buyable products not vague categories, no items already in the kitchen, sensible variety). Return JSON: {"scores":{"${labels.join('":{"relevance":0,"practicality":0,"comment":""},"')}":{"relevance":0,"practicality":0,"comment":""}},"winner":"LETTER","why":""}`;
  const jr = await judge({ system: sys, user: body, maxTokens: 1400 });
  const mapping = Object.fromEntries(shuffled.map((e, i) => [labels[i], e.model]));
  raw[kLabel].__judge = { mapping, verdict: jr.json, winnerModel: jr.json?.winner ? mapping[jr.json.winner] : null };
  console.log(`  JUDGE ${kLabel}: winner ${jr.json?.winner} = ${raw[kLabel].__judge.winnerModel}`);
}
writeFileSync(new URL("./out/home_raw.json", import.meta.url), JSON.stringify(raw, null, 2));
console.log("\nwrote out/home_raw.json");
