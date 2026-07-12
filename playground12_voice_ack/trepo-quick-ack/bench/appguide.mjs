// App Guide test matrix — runs every catalog question × 3 phrasings against the
// REAL device-assistant loop (same prompt + tools + model as deployed) on safe
// user 1f4db6b6, asserting: (a) answer contains the required UI anchor facts,
// (b) no invented UI, (c) no false action claim (how-to must not call a write
// tool or claim it did something). Usage: node bench/appguide.mjs [--verbose]
import fs from "fs";

const ENV_PATH = process.env.QA_ENV_JSON || "/tmp/qa_env.json";
Object.assign(process.env, JSON.parse(fs.readFileSync(ENV_PATH, "utf8")), { WRITE_SHARED_ONLY: "true" });
const da = await import("/Users/MattTaylor/Desktop/trepov2/playground12_voice_ack/trepo-quick-ack/lib/device-assistant.mjs");

const O = "1f4db6b6-2f62-4558-aa40-c8e82527dc74";
const CTX = { ownerId: O, userId: O, tableOwnerId: O, shoppingNamespace: O, householdMemberIds: [O] };
const VERBOSE = process.argv.includes("--verbose");

// must: EVERY listed anchor (string or regex) must appear (case-insensitive).
// Groups of alternatives are expressed as a regex with |.
const CATALOG = [
  // 1. HOUSEHOLD
  { id: "hh_id", cat: "household", must: [/household id/i, /profile|account/i],
    p: ["how do i find my household id", "where's my household ID", "how do i see my houshold id"] },
  { id: "hh_invite", cat: "household", must: [/invite to household/i],
    p: ["how do i invite my partner to my household", "how to add someone to my household", "how do i invit my husband"] },
  { id: "hh_join", cat: "household", must: [/household id/i, /join|enter/i],
    p: ["how do i join a household", "how to join my wife's household", "how do i joyn a household"] },
  { id: "hh_partner_cant_see", cat: "household", must: [/same household|household id/i],
    p: ["my partner can't see my kitchen items", "why can't my husband see my groceries", "wife cant see my stuff"] },

  // 2. ADDING ITEMS
  { id: "add_photo", cat: "adding", must: [/fridge\/pantry|\+|plus/i, /photo|picture|scan|camera/i],
    p: ["how do i add groceries by taking a photo", "how to scan my fridge", "how do i add food with a pic"] },
  { id: "add_text", cat: "adding", must: [/text/i],
    p: ["how do i add things without taking a picture of my fridge", "how to add items without a photo", "add stuff without a pic"] },
  { id: "add_receipt", cat: "adding", must: [/receipt/i],
    p: ["how do i add items from a receipt", "how to scan a receipt", "add from reciept"] },
  { id: "add_delivery", cat: "adding", must: [/receipt|text/i],
    p: ["how do i upload from a grocery delivery service", "can i import my instacart order", "how to add my grocery delivery"] },
  { id: "add_fix", cat: "adding", must: [/pencil|fix this item|edit/i],
    p: ["how do i fix a wrongly identified item", "the app got an item wrong how do i fix it", "how to correct a wrong item"] },
  { id: "add_single", cat: "adding", must: [/text|\+|plus/i],
    p: ["how do i add just one item", "how to add a single thing to my kitchen", "how do i add only one item"] },

  // 3. RECIPES
  { id: "rec_import", cat: "recipes", must: [/share/i, /trepo/i],
    p: ["how do i save a recipe from instagram", "how to import a tiktok recipe", "save a recipe from a website"] },
  { id: "rec_explore", cat: "recipes", must: [/explore/i],
    p: ["where do i find trending recipes", "how do i see explore recipes", "where are the popular recipes"] },
  { id: "rec_use", cat: "recipes", must: [/use what i have/i],
    p: ["how do i see recipes i can make with what i have", "how do i find recipes for what i have", "how do i see what i can cook with what i have"] },
  { id: "rec_delete", cat: "recipes", must: [/remove from saved recipes|edit/i],
    p: ["how do i delete a saved recipe", "how to remove a recipe i saved", "delete a saved recipe"] },
  { id: "rec_missing", cat: "recipes", must: [/add missing|shopping list/i],
    p: ["how do i add a recipe's missing ingredients to my list", "how do i add what a recipe needs to my shopping list", "how do i put a recipe's missing ingredients on my list"] },

  // 4. KITCHEN
  { id: "kit_group", cat: "kitchen", must: [/fix this item|automatic/i],
    p: ["how do i get ungrouped kitchen items into groups", "how do i change a kitchen item's category", "how do i fix a kitchen item's group"] },
  { id: "kit_storage", cat: "kitchen", must: [/stored in|fridge|freezer|pantry/i],
    p: ["how do i set where an item is stored", "how to say something is in the freezer", "change storage location"] },
  { id: "kit_expiry", cat: "kitchen", must: [/expire|expiry/i],
    p: ["how do i set an expiration date", "how to add an expiry date", "set when food expires"] },
  { id: "kit_delete", cat: "kitchen", must: [/discard item|remove/i],
    p: ["how do i remove something from my kitchen", "how to delete a kitchen item", "how do i get rid of a kitchen item"] },
  { id: "kit_iq", cat: "kitchen", must: [/kitchen iq/i],
    p: ["what is kitchen iq", "what does kitchen IQ mean", "whats kitchen iq"] },

  // 5. SHOPPING
  { id: "shop_add", cat: "shopping", must: [/list/i, /add an item|type|mic/i],
    p: ["how do i add something to my shopping list", "how do i put an item on my shopping list", "how do i add to my shopping list"] },
  { id: "shop_check", cat: "shopping", must: [/check|checkbox|bought/i],
    p: ["how do i check off items on my list", "how to mark something bought", "check off shopping items"] },
  { id: "shop_store", cat: "shopping", must: [/store|group/i],
    p: ["how do i organize my list by store", "how to add a store to my list", "group my list by store"] },
  { id: "shop_household", cat: "shopping", must: [/household|shared|real time|real-time/i],
    p: ["does my shopping list share with my household", "can my partner see my shopping list", "is the list shared"] },

  // 6. DISH LOG
  { id: "dish_log", cat: "dish", must: [/log a dish|dish log/i],
    p: ["how do i log a meal", "how to log a dish with a photo", "how do i track what i ate"] },
  { id: "dish_macros", cat: "dish", must: [/calorie|protein|carb|fat|nutrition/i],
    p: ["where do i see my calories and macros", "how do i see nutrition info", "where are my daily macros"] },
  { id: "dish_voice", cat: "dish", must: [/voice|text|tell/i],
    p: ["can i log a dish by voice", "how do i log food by talking", "can i log meals by voice"] },

  // 7. METRICS / MEAL PLAN
  { id: "mp_where", cat: "metrics", must: [/meal plan/i],
    p: ["where do i see my meal plan", "how do i find my meal plan", "where's my meal plan"] },
  { id: "mp_empty", cat: "metrics", must: [/check.* groceries|check in|kitchen/i],
    p: ["why would a meal plan be empty", "what do i do if my meal plan is empty", "how do i get a meal plan to show up"] },
  { id: "metrics_where", cat: "metrics", must: [/dish log|health/i],
    p: ["where can i see my metrics", "where are my health stats", "where do i see my nutrition metrics"] },

  // 8. CAPABILITIES / ACCOUNT
  { id: "cap", cat: "account", must: [/kitchen|shopping|dish|recipe/i],
    p: ["what can you do", "what can thyme help with", "what are you able to do"] },
  { id: "acct_notif", cat: "account", must: [/notification/i],
    p: ["how do i turn on notifications", "how to enable reminders", "turn on notifcations"] },
  { id: "acct_feedback", cat: "account", must: [/matt@trepo\.ai|feedback|email/i],
    p: ["how do i give feedback", "how do i contact support", "who do i email for help"] },
  { id: "acct_signout", cat: "account", must: [/log out|sign out/i],
    p: ["how do i sign out", "how to log out", "how do i logout"] },
  { id: "acct_delete", cat: "account", must: [/delete account/i],
    p: ["how do i delete my account", "how to delete my account permanently", "remove my account"] },
];

// Tools that MUTATE data — if any of these actually ran, a how-to answer
// silently performed a write (a false action) even if the text looks explanatory.
const WRITE_TOOLS = new Set([
  "check_in_item", "check_in_many_items", "add_to_shopping_list", "add_many_to_shopping_list",
  "add_dish_ingredients_to_shopping_list", "add_recipe_ingredients_to_shopping_list",
  "add_saved_recipe_ingredients_to_shopping_list", "log_dish_from_voice", "log_dish_ingredients",
  "append_to_recent_dish", "update_recent_dish", "discard_item", "clear_kitchen_inventory",
  "remove_from_shopping_list", "clear_shopping_list", "mark_shopping_item_bought",
  "mark_item_opened", "update_item_quantity", "update_item_expiration", "update_item_location",
  "update_item_details", "delete_dish_log", "mark_dish_consumed", "save_recipe_from_tiktok",
]);
const FALSE_ACTION = /\bi(?:'ve| have)?\s*(?:just\s+)?(added|logged|removed|deleted|created|saved|checked off|put|discarded)\b/i;
// Sentence-initial completed-action confirmation ("Logged a meal entry.",
// "Added Placeholder to your shopping list.") — a write actually happened.
const COMPLETED_ACTION = /(^|[.!?]\s+)(?:(?:ok(?:ay)?|done|sure|got it)[\s,!.]+)*(added|logged|removed|deleted|created|saved|checked off|discarded)\b(?!\s+(?:out|in|up)\b)(?:\s*[:\-—]|\s+(?:your|the|a|an|it\b|that\b|two|one|both|placeholder|\d))/i;
// invented-UI guard: labels that do NOT exist in the app (must never appear)
const INVENTED = /\b(settings gear|hamburger menu|three dots menu|swipe left to delete|long-press to|instacart integration|connect your (kroger|amazon|walmart))\b/i;

function assess(text, must, toolTrace) {
  const t = String(text || "");
  const missing = must
    .filter((m) => !(m instanceof RegExp ? m.test(t) : t.toLowerCase().includes(String(m).toLowerCase())))
    .map((m) => (m instanceof RegExp ? m.source : m));
  // False action = a mutating tool actually ran (definitive), OR the text claims a
  // write ("I added…" / sentence-initial "Added/Logged: X"). A pure explanation
  // that only touched READ tools is fine.
  const wroteTools = (toolTrace || []).filter((x) => x && x.ok !== false && WRITE_TOOLS.has(x.toolName)).map((x) => x.toolName);
  const falseAction = wroteTools.length > 0 || FALSE_ACTION.test(t) || COMPLETED_ACTION.test(t);
  const invented = INVENTED.test(t);
  return { pass: missing.length === 0 && !falseAction && !invented, missing, falseAction, invented, wroteTools };
}

const results = [];
let passN = 0, total = 0;
for (const q of CATALOG) {
  for (let i = 0; i < q.p.length; i++) {
    total++;
    const phrasing = q.p[i];
    let text = "", toolTrace = [], err = null;
    try {
      const r = await da.runDeviceAssistant({ transcript: phrasing, userContext: CTX, env: process.env, sessionMessages: [], responseSurface: "halo" });
      text = r.text || ""; toolTrace = r.toolTrace || [];
    } catch (e) { err = String(e.message).slice(0, 80); }
    const a = err ? { pass: false, missing: ["ERR:" + err], falseAction: false, invented: false, wroteTools: [] } : assess(text, q.must, toolTrace);
    if (a.pass) passN++;
    results.push({ id: q.id, cat: q.cat, ph: i, phrasing, pass: a.pass, missing: a.missing, falseAction: a.falseAction, invented: a.invented, text: VERBOSE ? text : text.slice(0, 120) });
    const tag = a.pass ? "PASS" : "FAIL";
    console.log(`${tag} [${q.id}#${i}] ${phrasing}`);
    if (!a.pass) console.log(`     missing=${JSON.stringify(a.missing)} falseAction=${a.falseAction} wrote=${JSON.stringify(a.wroteTools)} invented=${a.invented} | "${text.slice(0, 160)}"`);
  }
}
console.log(`\n==== MATRIX: ${passN}/${total} pass (${(100 * passN / total).toFixed(1)}%) ====`);
const byFail = results.filter((r) => !r.pass);
if (byFail.length) console.log("FAILURES:", JSON.stringify(byFail.map((r) => ({ id: r.id, ph: r.ph, missing: r.missing, fa: r.falseAction, inv: r.invented })), null, 1));
fs.writeFileSync("/tmp/appguide_matrix_results.json", JSON.stringify(results, null, 2));
process.exit(0);
