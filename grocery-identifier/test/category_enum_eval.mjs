// Name->category eval for the AI-driven category constraint (Task 3). Exercises the
// EXACT enum + guidance that identifyGrocery.ts injects (flag-gated) against the real
// miscategorized set + must-not-regress controls, via a real OpenAI call with the same
// model + strict json_schema enum. This is the test-first gate; iterate the guidance
// until it passes. (identify is image-based; this name-based harness faithfully tests
// whether the guidance makes the model pick the right category for the hard names.)
import OpenAI from "openai";

const CATEGORY_ENUM = ["leftovers","produce","dairy_eggs","meat_seafood","pantry","spices","snacks_sweets","beverages","prepared_other"];

// keep in lock-step with CATEGORY_ENUM_GUIDANCE in src/openai/identifyGrocery.ts
const GUIDANCE =
  "\n\nCATEGORY — choose EXACTLY ONE of these 9 values, by what the product fundamentally IS, NOT by incidental words in its name: leftovers, produce, dairy_eggs, meat_seafood, pantry, spices, snacks_sweets, beverages, prepared_other." +
  "\nRules + hard examples: Potato Bread / Blueberry Bread = pantry (it IS bread, shelf-stable). Blueberry Muffins / cakes / cookies / pastries = snacks_sweets. Green Onion Pancakes and other frozen/prepared foods = prepared_other. A bag of potatoes / loose bananas / a bunch of celery or herbs = produce. Strawberry yogurt = dairy_eggs. Chicken broth = pantry. Fresh raw meat/poultry/fish = meat_seafood — including ANY beef/pork/lamb/veal cut such as steak, ribeye, sirloin, NY strip, skirt steak, flank, T-bone, brisket, filet, pork steak, ham steak, or chops (a raw steak is NEVER a beverage; but steak sauce = pantry and steak seasoning = spices). Any drink = beverages. Home leftover food = leftovers. Spices, seasonings, spice blends, spice rubs, and dried/ground herbs = spices, NOT pantry. When a produce word appears in a processed product's name (potato bread, blueberry muffin, green onion pancake), pick the PROCESSED category, not produce. Tofu / tempeh / seitan / plant-based meat substitutes = prepared_other.";

const SYSTEM = "You categorize grocery products into the app's fixed category enum." + GUIDANCE;

const CASES = [
  // must-fix (were wrongly 'produce')
  ["Potato Bread", "pantry"],
  ["L'Oven Fresh Potato Bread", "pantry"],
  ["Blueberry Bread", "pantry"],
  ["Blueberry Muffins", "snacks_sweets"],
  ["Mixed Berry Biscuits", "snacks_sweets"],
  ["Trader Joe's Green Onion Pancakes", "prepared_other"],
  ["Green Onion Pancakes", "prepared_other"],
  // must-not-regress controls
  ["Bananas Bunch", "produce"],
  ["Cilantro Bunch", "produce"],
  ["Bag of Potatoes", "produce"],
  ["Strawberry Yogurt", "dairy_eggs"],
  ["Chicken Broth", "pantry"],
  ["Oreo Cookies", "snacks_sweets"],
  ["Coca-Cola 12 oz Can", "beverages"],
  // must-fix: raw steaks were landing in 'beverages' on the Gemini bulk path
  ["Steak", "meat_seafood"],
  ["Ribeye Steak", "meat_seafood"],
  ["NY Strip Steak", "meat_seafood"],
  ["Sirloin", "meat_seafood"],
  ["Pork Steak", "meat_seafood"],
  ["Ham Steaks", "meat_seafood"],
  ["Skirt Steak", "meat_seafood"],
  // steak controls must NOT be pulled into meat
  ["Steak Seasoning", "spices"],
  ["A1 Steak Sauce", "pantry"],
  ["Protein Shake", "beverages"],
];

const schema = {
  type: "object", additionalProperties: false, required: ["category"],
  properties: { category: { type: "string", enum: CATEGORY_ENUM } },
};

const model = process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function classify(name) {
  const resp = await openai.responses.create({
    model,
    ...((/^(o[1-9]|gpt-5)/.test(model)) ? { reasoning: { effort: "medium" } } : {}),
    max_output_tokens: 900,
    text: { format: { type: "json_schema", name: "cat", strict: true, schema } },
    input: [
      { role: "system", content: [{ type: "input_text", text: SYSTEM }] },
      { role: "user", content: [{ type: "input_text", text: `Product name: ${name}` }] },
    ],
  });
  const txt = resp.output_text || (resp.output?.map(o => o.content?.map(c => c.text).join("")).join("")) || "{}";
  return JSON.parse(txt).category;
}

let fails = 0;
for (const [name, want] of CASES) {
  let got;
  try { got = await classify(name); } catch (e) { got = "ERR:" + (e?.message || e); }
  const ok = got === want;
  if (!ok) fails++;
  console.log(`  ${ok ? "OK " : "FAIL"}  ${name.padEnd(34)} -> ${String(got).padEnd(14)} (want ${want})`);
}
console.log(`\n${fails === 0 ? "EVAL PASS" : "EVAL FAIL: " + fails + " miss"} (${CASES.length} cases)`);
process.exit(fails === 0 ? 0 : 1);
