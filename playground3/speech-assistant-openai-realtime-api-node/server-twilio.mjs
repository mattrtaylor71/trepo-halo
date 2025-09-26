// server-twilio.mjs
import Fastify from 'fastify';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';

// Load environment variables from .env file
dotenv.config();

// Retrieve the OpenAI API key from environment variables.
const { OPENAI_API_KEY } = process.env;

if (!OPENAI_API_KEY) {
  console.error('Missing OpenAI API key. Please set it in the .env file.');
  process.exit(1);
}

// --- minimal fetch polyfill (no extra deps needed) ---
const ensureFetch = async () => {
  if (typeof fetch !== 'undefined') return fetch;
  const { default: nf } = await import('node-fetch');
  return nf;
};

// Initialize Fastify
const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Constants (keep tutorial defaults)
const VOICE = 'marin';
const PORT = process.env.PORT || 5050;

// === Grocery data config (override in .env if you like) ===
const GROCERY_AUTH0_SUB = process.env.GROCERY_AUTH0_SUB || "auth0|672c4ebdba93ac8a306410d5";
const LAMBDA_RECYCLABLES = process.env.LAMBDA_RECYCLABLES || "https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables";
const LAMBDA_KITCHEN     = process.env.LAMBDA_KITCHEN     || "https://i9dzt51kkg.execute-api.us-east-1.amazonaws.com/fetchKitchen";

/**
 * SMARTER persona (keeps responses concise, adds intuition):
 * - "what do I need/buy/shopping list" => compute_needs (discarded minus kitchen), then suggest_from_history
 * - Recipes: prefer kitchen-first; if caller hints they’re shopping/driving/in-store, allow store_flexible mode
 * - Health/budget: respect keywords (“healthy”, “lighter”, “cheap”, “budget”) to bias suggestions
 * - One short clarifier only if crucial (e.g., “Cooking now or shopping?”). Otherwise act.
 */
const SYSTEM_MESSAGE = [
  // Language + tone
  "LANGUAGE: English (en-US).",
  "VOICE: Natural, quick, and conversational. Short sentences. No chit-chat, but sound human.",
  "BREVITY: Prefer summaries over lists. Only read a few examples unless explicitly asked for the full list.",

  // Role + context
  "ROLE: Grocery co-pilot. Assume the caller may be planning a shop, driving or in-store, standing in their kitchen, or coordinating with a partner. They may care about health or budget.",

  // Names / normalization
  "NAMES: When reading items from tools, use the 'short' field if available. If absent, shorten titles by removing packaging/size/marketing words (e.g., 'in carton', '18 ct', '12 oz', 'bottle', 'pack', 'family size', 'variety pack'). Keep the human name (e.g., 'brown eggs').",

  // Data scope
  "SCOPE: You can call list_kitchen_14d (what's on hand) and list_discarded_14d (recently tossed).",

  // Summarize-first behavior
  "SUMMARIZE-FIRST:",
  "- If the user asks: 'what’s in my kitchen', 'look at my kitchen', 'am I missing anything', 'what should I pick up' ⇒ DO NOT read raw lists.",
  "- Instead do two steps: (1) Summarize the kitchen in one short sentence (category mix: produce/protein/dairy/grains/snacks/cooking; note any perishables risk like 'greens, berries, dairy soon'). (2) Identify gaps.",
  "- Give 3–6 representative short names only if it helps; otherwise stay abstract (e.g., 'yogurt, eggs, sparkling water').",
  "- End with a gentle option, e.g., 'Want details by category?'",

  // Gap / needs inference
  "NEEDS INFERENCE:",
  "- Compute 'probable needs' as: items discarded recently MINUS items in the kitchen (case-insensitive by 'short' name).",
  "- De-dup rule: Never recommend an item if a same/similar item is already in the kitchen. Match by 'short' name and treat obvious variants as the same (e.g., 'Greek yogurt' ~ 'Greek yogurt 2%').",
  "- Also flag staple gaps if missing: eggs, milk/dairy alt, bread/tortillas, cooking oil, onions/garlic, leafy greens, a protein (chicken/fish/tofu/beans), fruit (bananas/berries/citrus), breakfast base (oats/cereal/yogurt), coffee/tea, rice/pasta, canned tomatoes/beans, basic seasoning (salt/pepper/spice).",
  "- If the user mentions health ⇒ bias to produce/lean protein/higher-score items. If budget ⇒ include one quick swap (value brand/size/bulk).",

  // Intent routing
  "ROUTING:",
  "- Need/buy/get/pick up/shopping list/what to get ⇒ Summarize-first, then 'You’re low on: …' using needs inference. Ask once: 'Any goals this week—healthy, speed, or budget?' and bias the list accordingly.",
  "- What’s in my kitchen / what do I have ⇒ Summarize-first, then optionally 'Top items: A; B; C…' (max 6).",
  "- What did I throw out / what I tossed / discarded ⇒ call list_discarded_14d and speak short names (max 8).",
  "- Recipes only if caller explicitly asks. First ask one micro-choice: 'Want to use what you have, or can you shop too?' If 'shop' ⇒ store_flexible; otherwise ⇒ kitchen.",
  "- If the user explicitly says recipe/recipes/cook/make ⇒ do NOT return staples or needs unless they also ask for a list. Ask one micro-clarifier (“Use what you have, or shop too?”), then call recommend_recipes with mode accordingly.",
  "- Only call compute_needs for need/out/missing/gaps intents. Do NOT call compute_needs inside a recipes turn unless the user adds “also make me a list.”",
  "- If the user names an ingredient (e.g., “tuna” or “salmon”), pass it via recommend_recipes with proteins_preferred:[…]. Do not substitute another protein unless the user allows it.",


  // Output shapes (keep it tight, but friendly)
  "FORMATS:",
  "- Kitchen summary: 'Your kitchen constists: mostly of {cats}. Items you might want to use soon are: {X, Y}.'",
  "- Needs: 'You’re low on: A; B; C…' (max 15; end with '…' if more).",
  "- Discarded: 'You've discarded <count> items: A; B; C…' (max 15).",
  "- Quick list after an 'expiring/soon' topic: start with those expiring items that are staples or nearly out, then add top needs; de-duplicate against kitchen.",
  "- Recipes (only when asked): 'Title — You have: …; You need to buy: …' (2–3 ideas).",
  "- Add at most one health/budget hint when relevant.",
  "- Offer a next step instead of listing more: 'Want dairy details, or a quick list to buy?'",
  "- In store_flexible (shop) recipes: include at least 2 specific Buy items that elevate the dish (e.g., garlic, lemon, fresh herbs, sauce, missing base), not just what’s on hand.",
  "- When suggesting a shopping list, name specific items (not “produce”) and add a 1-line why tied to the caller’s patterns/preferences (e.g., “garlic — big flavor, matches your savory cooking”).",


  // === CONVERSATIONAL LISTING (tiny) ===
  "CONVERSATIONAL LISTING:",
  "- For recipes and expiring foods, avoid telegram style like 'Have bison use lemon'.",
  "- Prefer short, human lines: 'You have bison; finish with lemon.' 'You have spinach; use today.'",
  "- Keep each line ~6–12 words; still concise.",

  "RECIPE & SHOPPING CLARIFIERS:",
  "- Only two built-in clarifiers override the one-clarifier rule: (1) recipes ⇒ 'Want to use what you have, or can you shop too?'; (2) general 'what should I get' ⇒ 'Any goals this week—healthy, quick, or budget?'. Keep them to one short line each, then act.",
  "- In kitchen mode: never list an ingredient under 'You have' unless verified via fuzzy match (e.g., 'rice' ~ 'basmati rice'); do not add a Buy list in kitchen mode.",
  "- In store_flexible mode: you may add a Buy list, but de-duplicate it against what’s already in the kitchen using fuzzy matching.",
  "- When giving 2–3 recipes, vary the main protein across ideas if multiple proteins are in the kitchen (e.g., eggs, chicken sausage, turkey bacon, beans).",


  // Context & follow-ups (NEW)
  "CONTEXT & FOLLOW-UPS:",
  "- Track the last topic for the next 2–3 turns (e.g., 'expiring soon', 'budget', 'high-protein').",
  "- If the user says 'yes' to a follow-up like 'Want a quick list?' then bias that list by the current topic. Examples:",
  "  • If the topic was expiring/soon ⇒ include those expiring perishables first (only if not already abundant), then add computed needs; remove anything clearly in the kitchen (de-dup).",
  "  • If the topic was budget ⇒ prefer value staples and size/brand swaps in the quick list.",
  "  • If the topic was recipes ⇒ offer a Buy list that complements today’s kitchen for those recipes.",
  "- Never pivot back to unrelated needs if the user’s confirmation follows a topical prompt; carry the topic into the answer.",

  // Data access + listing rules
  "DATA ACCESS:",
  "- Tools return the full set of items (no artificial limits). Use them to reason, but DO NOT read everything.",
  "- When asked to 'list', start high-level: speak categories + counts and 3–6 representative short names; then ask if they want details by category.",
  "- Only enumerate everything if the caller explicitly asks to list all items; in that case, page by category in small chunks on request (e.g., 'dairy next?').",
  "- When the user says “out of / need / missing” and it’s unclear, ask: “Discards or typical gaps?” before calling tools.",


  "TOOL DISCIPLINE:",
  "- Call tools that match the current intent ONLY (recipes → recommend_recipes; needs → compute_needs; discards → list_discarded_14d; kitchen → list_kitchen_14d).",
  "- After an intent change, do not reuse results from a previous tool path unless the new intent needs them.",
  
  "INGREDIENT OVERRIDES:",
  "- Respect explicit ingredients. When the caller says “tuna,” call recommend_recipes with proteins_preferred:[\"tuna\"].",
  "- If not in the kitchen and mode=kitchen, ask one micro-clarifier: “Okay to shop for tuna?” before substituting.",
  "- If mode=store_flexible, include the requested protein in Buy instead of swapping.",


  // === EXPIRING & OLD FOODS (NEW) ===
  "EXPIRING & OLD FOODS:",
  "- When asked 'what's expiring/going bad/old', examine ALL kitchen items with their age (createdDate).",
  "- Classify into buckets using sensible shelf-life heuristics by type:",
  "  • use_today: clearly at/past shelf life (e.g., berries/herbs ~3–4d, leafy greens ~5d, bread/tortillas ~6d, cooked leftovers ~4d, milk/yogurt ~10d, eggs ~21d, raw meat/fish ~3d).",
  "  • use_soon: within ~1–2 days of the above shelf-life.",
  "  • old pantry/stale: non-perishables kept ~90d+ (e.g., chips, crackers, opened sauces).",
  "- Answer with a concise summary first, then 3–6 representative short names (not the entire list).",
  "- FLOW LOGIC: If the caller then says 'yes' to a quick/grocery list while you’re in an 'expiring' topic:",
  "  • Build the list from use_today + use_soon + obviously stale. These are the focus.",
  "  • Only add computed gaps/needs AFTER those, and NEVER include items that are clearly still in the kitchen (de-dup by short name).",
  "- If they say 'replace these', offer a replacement-ready list (same items, marked 'replace'), otherwise keep it informational.",
  "- When listing expiring items, use mini sentences with 'you': e.g., 'You have Greek yogurt; use in ~2 days.' 'You have berries; use today.'",
  "- Keep 3–6 lines max; prioritize use_today → use_soon → old pantry.",


  // One clarifier rule
  "CLARITY:",
  "- If intent is clear, act.",
  "- If ambiguous OR would trigger heavy tools/lists, ask up to TWO micro-clarifiers (5–9 words), then act.",
  "- Do NOT run tools until clarified when ambiguity is high.",
  "- Skip filler like “got it/ok/sure”; start with the answer.",
  "- Examples:",
  "  • “What am I out of?” → “Scan discards or typical gaps?”",
  "  • “What should I get?” → “Are you doing a quick shop to replenish items or are you planning for the whole week?”",
  "  • “Recipes?” → “Want to use what you have, or shop too?”",

  "INTERRUPTIONS & TOPIC SWITCHING:",
  "- Latest user message WINS. If the user shifts topic (e.g., asks for recipes after staples), ABORT the current plan and pivot immediately.",
  "- Do not finish or summarize the previous path once a new intent appears.",
  "- Clear any queued lists in your head; respond to the new intent directly.",


  "INTENT DISAMBIGUATION FLOWS:",
  "- If user picks “discards” for “what am I out of” ⇒ call list_discarded_14d; summarize + 3–6 short names; offer quick list.",
  "- If user picks “typical gaps / scan kitchen” ⇒ run compute_needs; summarize-first; de-dup against kitchen; don’t read long lists unless asked.",
  "- If user gives no choice after one nudge, choose the most likely route from their last topic (e.g., if talking about expiring/soon, bias to expiring items first; if talking about budget, bias to value staples).",



].join("\n");


// Logs (unchanged)
const LOG_EVENT_TYPES = [
  'error',
  'response.content.done',
  'rate_limits.updated',
  'response.done',
  'input_audio_buffer.committed',
  'input_audio_buffer.speech_stopped',
  'input_audio_buffer.speech_started',
  'session.created',
  'session.updated'
];
const SHOW_TIMING_MATH = false;

// --- helpers for grocery tools ---

function cleanTitle(title = "") {
  let t = String(title).toLowerCase().trim();

  // remove common packaging/size/marketing tokens
  t = t
    .replace(/\b(in\s+)?carton\b/g, "")
    .replace(/\b(bottle|bottles|jar|can|cans|pack|multi[-\s]?pack|variety\s?pack|family\s?size|value\s?pack)\b/g, "")
    .replace(/\b(\d+\s?(oz|ounce|ounces|fl\s?oz|ml|g|kg|lb|pound|ct|count))\b/g, "")
    .replace(/\b(\d+[\s-]?(pack|ct))\b/g, "")
    .replace(/\borganic\b/g, "")                 // keep if you really want—easy to revert
    .replace(/\bgluten[-\s]?free\b/g, "")
    .replace(/\bnon[-\s]?gmo\b/g, "")
    .replace(/\bno\s+added\s+sugar\b/g, "")
    .replace(/\bwith\b.*$/g, "")                 // drop long marketing tails
    .replace(/[()]/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  // simple canonical tweaks
  t = t.replace(/\byoghurt\b/g, "yogurt");
  t = t.replace(/\bgreek\s*yogurt\b/g, "Greek yogurt"); // capitalize product types
  t = t.replace(/\beggs?\b/g, "eggs");

  // capitalize first word + proper style for 1–2 words
  t = t.split(" ").map((w,i) => (i===0 ? w.charAt(0).toUpperCase()+w.slice(1) : w)).join(" ");

  // fallbacks
  if (!t) t = title || "Item";
  return t;
}

const normalizeItems = (arr = []) =>
  arr
    .map((it) => {
      const rawTitle =
        it.title || it.name || it.product_name || it.label || "Unknown item";
      const short = cleanTitle(rawTitle);
      return {
        id: it._id || it.user_item_id || it.id || null,
        original_id: it.original_id || it.item || null,
        title: rawTitle,
        short,                         // ← concise, human-friendly name
        images: it.images || it.image_urls || "",
        score: it.score ?? null,
        category: it.simplified_category || it.category || "Other",
        inventory: it.inventory ?? null,
        createdDate: it._createdDate || it.createdAt || it.date || null,
      };
    })
    .filter((x) => x.createdDate)
    .sort((a, b) => new Date(b.createdDate) - new Date(a.createdDate));


async function parseLambdaResponse(r) {
  const raw = await r.text();
  if (!r.ok) return { error: raw, status: r.status };
  let payload;
  try { payload = JSON.parse(raw); } catch { payload = { message: String(raw) }; }
  if (payload && typeof payload === "object" && typeof payload.body === "string") {
    try { payload = JSON.parse(payload.body); } catch {}
  }
  return { payload, status: r.status };
}
const daysSince = (iso) => {
  const ts = new Date(iso).getTime();
  if (!Number.isFinite(ts)) return null;
  return Math.max(0, Math.floor((Date.now() - ts) / 86400000));
};
const canon = (s='') => s.toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();

// --- Fuzzy presence helpers (exact + contains + token overlap) ---
function tokenize(s="") {
  return canon(s).split(" ").filter(Boolean);
}

function fuzzyIncludes(needle="", haystack="") {
  const n = canon(needle);
  const h = canon(haystack);
  if (!n || !h) return false;
  if (h.includes(n)) return true;            // "basmati rice" includes "rice"
  if (n.includes(h)) return true;            // rare, but harmless
  // token overlap: at least 1 token in common for short names
  const nt = tokenize(needle);
  const ht = new Set(tokenize(haystack));
  let overlap = 0;
  for (const t of nt) if (ht.has(t)) overlap++;
  return overlap >= Math.min(2, nt.length);  // e.g., "greek yogurt" ~ "greek yogurt 2%"
}

// Returns the matched kitchen item title if present (fuzzy), else null
function kitchenHasLike(name, kitchenItems) {
  const n = canon(name);
  if (!n) return null;
  for (const it of kitchenItems) {
    const t = it.short || it.title || "";
    if (fuzzyIncludes(name, t)) return t;
  }
  return null;
}

// Use this to build a safe "have" list only from things truly present
function presentOrBuy({ name, kitchenItems, mode }) {
  const match = kitchenHasLike(name, kitchenItems);
  if (match) return { have: match, buy: null };
  if (mode === "store_flexible") return { have: null, buy: name };
  return { have: null, buy: null }; // kitchen mode: don't claim or buy
}

// De-dupe a Buy array by removing anything that is in kitchen (fuzzy) or already in Have
function dedupeBuy({ buy=[], have=[], kitchenItems }) {
  const haveSet = new Set(have.map(canon));
  return buy.filter(item => {
    if (!item) return false;
    if (haveSet.has(canon(item))) return false;
    return !kitchenHasLike(item, kitchenItems);
  });
}


// --- Tool schemas (advertise to the model) ---
const TOOL_SCHEMAS = [
  {
    type: "function",
    name: "list_discarded_14d",
    description: "Return items the user has thrown out in the last ~14 days.",
    parameters: {
      type: "object",
      properties: { limit: { type: "integer", minimum: 1, maximum: 500 } },
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "list_kitchen_14d",
    description: "Return items currently in the user's kitchen (last ~14 days).",
    parameters: {
      type: "object",
      properties: { limit: { type: "integer", minimum: 1, maximum: 500 } },
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "get_item_age",
    description: "Given an item name, return when it was added and how many days since.",
    parameters: {
      type: "object",
      properties: { name: { type: "string", description: "Item name (case-insensitive, partial ok)." } },
      required: ["name"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "compute_needs",
    description: "Compute 'what to buy': items recently discarded but not currently in the kitchen.",
    parameters: {
      type: "object",
      properties: { limit: { type: "integer", minimum: 1, maximum: 50 } },
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "summarize_prefs",
    description: "Summarize preferences from kitchen + discarded history (top categories/brands, avg score).",
    parameters: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    type: "function",
    name: "suggest_from_history",
    description: "Suggest 3–5 tasteful add-ons based on frequent categories/brands and higher scores; avoid duplicates of current kitchen.",
    parameters: {
      type: "object",
      properties: { limit: { type: "integer", minimum: 1, maximum: 10 } },
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "recommend_recipes",
    description: "Return 2–3 quick recipe ideas using kitchen-first; in store_flexible mode include Buy list.",
    parameters: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["kitchen", "store_flexible"] },
        servings: { type: "integer", minimum: 1, maximum: 12 },
        diet: { type: "string", description: "Optional: vegetarian, high-protein, dairy-free, etc." }
      },
      required: ["mode"],
      additionalProperties: false,
    },
  },
];

// Return ALL discarded items from Lambda (no slicing)
async function tool_list_discarded_14d() {
  const f = await ensureFetch();
  const r = await f(LAMBDA_RECYCLABLES, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_name: GROCERY_AUTH0_SUB }),
  });
  const { payload, error } = await parseLambdaResponse(r);
  if (error) return { ok: false, error };
  const arr = Array.isArray(payload) ? payload : [];
  const items = normalizeItems(arr);            // ← no slice()
  return { ok: true, count: items.length, items, note: "~last 14 days only" };
}

// Return ALL kitchen items from Lambda (no slicing)
async function tool_list_kitchen_14d() {
  const f = await ensureFetch();
  const r = await f(LAMBDA_KITCHEN, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ auth0_sub: GROCERY_AUTH0_SUB }),
  });
  const { payload, error } = await parseLambdaResponse(r);
  if (error) return { ok: false, error };
  const arr = Array.isArray(payload) ? payload : [];
  const items = normalizeItems(arr);            // ← no slice()
  return { ok: true, count: items.length, items, note: "~last 14 days only" };
}

// Use full kitchen set for age lookup
async function tool_get_item_age({ name }) {
  const base = await tool_list_kitchen_14d();   // ← full set
  if (!base.ok) return base;
  const q = (name || "").trim().toLowerCase();
  if (!q) return { ok: false, error: "missing name" };

  const matches = base.items
    .map(it => ({ it, t: (it.short || it.title || "").toLowerCase() }))
    .filter(x => x.t.includes(q));
  const chosen = matches.length
    ? matches.sort((a,b)=> new Date(b.it.createdDate) - new Date(a.it.createdDate))[0].it
    : base.items.find(it => (it.short || it.title || "").toLowerCase().startsWith(q));

  if (!chosen) return { ok: false, error: "not found" };
  const ts = new Date(chosen.createdDate).getTime();
  const days = Number.isFinite(ts) ? Math.max(0, Math.floor((Date.now() - ts)/86400000)) : null;
  return { ok: true, title: chosen.short || chosen.title, createdDate: chosen.createdDate, days };
}

async function tool_compute_needs({ limit = 12 } = {}) {
  const [k, d] = await Promise.all([
    tool_list_kitchen_14d(),       // full set
    tool_list_discarded_14d(),     // full set
  ]);
  if (!k.ok) return k;
  if (!d.ok) return d;

  const inKitchen = new Set(k.items.map(i => canon(i.short || i.title)));
  const counts = new Map(); // key -> {title, short, category, scoreSum, n, last}

  for (const it of d.items) {
    const key = canon(it.short || it.title);
    if (!key) continue;
    if (!inKitchen.has(key)) {
      const cur = counts.get(key) || {
        title: it.title,
        short: it.short || it.title,
        category: it.category,
        scoreSum: 0,
        n: 0,
        last: it.createdDate
      };
      cur.n += 1;
      cur.scoreSum += Number(it.score ?? 0);
      if (new Date(it.createdDate) > new Date(cur.last)) cur.last = it.createdDate;
      counts.set(key, cur);
    }
  }

  const needs = [...counts.values()]
    .map(v => {
      const title = v.short;
      const stapleHint = /eggs|milk|yogurt|bread|tortilla|oil|onion|garlic|greens|chicken|tofu|beans|fruit|oats|cereal|coffee|tea|rice|pasta|tomato|salt|pepper/i.test(title)
        ? "staple you often use"
        : null;
      const whyBits = [];
      whyBits.push("recently used up");
      if (stapleHint) whyBits.push(stapleHint);
      if (typeof v.scoreSum === "number" && v.n) {
        const avg = Math.round((v.scoreSum / v.n) || 0);
        if (avg >= 4) whyBits.push("you rate this highly");
      }
      const why = whyBits.join("; ");

      return {
        title,                      // concise name
        category: v.category,
        last_discarded: v.last,
        times_recently_discarded: v.n,
        score_estimate: v.n ? Math.round((v.scoreSum / v.n) || 0) : null,
        why
      };
    })
    .sort((a,b) =>
      (b.times_recently_discarded - a.times_recently_discarded) ||
      (new Date(b.last_discarded) - new Date(a.last_discarded))
    )
    .slice(0, Math.min(50, Math.max(1, limit)));


  return { ok: true, count: needs.length, needs };
}


async function tool_summarize_prefs() {
  const [k, d] = await Promise.all([
    tool_list_kitchen_14d({ limit: 500 }),
    tool_list_discarded_14d({ limit: 500 }),
  ]);
  if (!k.ok) return k;
  if (!d.ok) return d;

  const all = [...k.items, ...d.items];
  if (!all.length) return { ok: true, top_categories: [], top_brands: [], avg_score: null };

  const catCount = new Map();
  const brandCount = new Map();
  let scoreSum = 0, scoreN = 0;

  for (const it of all) {
    if (it.category) catCount.set(it.category, (catCount.get(it.category) || 0) + 1);
    if (it.brand) brandCount.set(it.brand, (brandCount.get(it.brand) || 0) + 1);
    if (typeof it.score === 'number') { scoreSum += it.score; scoreN += 1; }
  }

  const top_categories = [...catCount.entries()].sort((a,b)=>b[1]-a[1]).slice(0,3).map(([name, n])=>({ name, n }));
  const top_brands = [...brandCount.entries()].sort((a,b)=>b[1]-a[1]).slice(0,3).map(([name, n])=>({ name, n }));
  const avg_score = scoreN ? Math.round(scoreSum / scoreN) : null;

  return { ok: true, top_categories, top_brands, avg_score };
}

async function tool_suggest_from_history({ limit = 5 } = {}) {
  const [needs, k, d] = await Promise.all([
    tool_compute_needs({ limit: 100 }),
    tool_list_kitchen_14d({ limit: 500 }),
    tool_list_discarded_14d({ limit: 500 }),
  ]);
  if (!needs.ok) return needs;
  if (!k.ok) return k;
  if (!d.ok) return d;

  const kitchenKeys = new Set(k.items.map(i => canon(i.title)));

  // Infer lightweight "prefs" from history
  const catCount = new Map();
  const scoreByTitle = new Map();
  for (const it of [...k.items, ...d.items]) {
    if (it.category) catCount.set(it.category, (catCount.get(it.category) || 0) + 1);
    if (typeof it.score === 'number') scoreByTitle.set(canon(it.title), it.score);
  }
  const favoriteCats = [...catCount.entries()].sort((a,b)=>b[1]-a[1]).slice(0,2).map(([c])=>c);

  const byCat = new Map();
  for (const it of d.items) {
    if (!it.category) continue;
    const arr = byCat.get(it.category) || [];
    arr.push(it);
    byCat.set(it.category, arr);
  }
  for (const [cat, arr] of byCat.entries()) {
    arr.sort((a,b) => (Number(b.score ?? 0) - Number(a.score ?? 0)));
  }

  const suggestions = [];

  // Helper to write a short, preference-aware reason
  function reasonFor(title, category) {
    const t = canon(title);
    const score = scoreByTitle.get(t);
    const bits = [];
    if (favoriteCats.includes(category)) bits.push(`fits your ${category.toLowerCase()} favorites`);
    if (score >= 4) bits.push("you’ve rated it well");
    // Special helpful reasons for common pantry buys
    if (/garlic/i.test(title)) bits.push("big flavor with no added sugar");
    if (/lemon/i.test(title)) bits.push("brightens bowls and fish");
    if (/fresh herbs|basil|cilantro|parsley/i.test(title)) bits.push("boosts freshness in quick meals");
    if (/parmesan|feta|mozzarella/i.test(title)) bits.push("great finishing cheese for weeknights");
    if (bits.length === 0) bits.push("pairs with what you buy often");
    return bits.join("; ");
  }

  // 1) Top needs (most likely purchases)
  for (const n of needs.needs) {
    if (suggestions.length >= limit) break;
    if (!kitchenKeys.has(canon(n.title))) {
      suggestions.push({
        title: n.title,
        reason: "recently used up",
        category: n.category || undefined,
        why: reasonFor(n.title, n.category || "pantry")
      });
    }
  }

  // 2) Fill with higher-score items from favorite categories, avoiding duplicates
  if (suggestions.length < limit) {
    const favCats = [...byCat.entries()].sort((a,b)=>b[1].length - a[1].length).map(([cat])=>cat);
    for (const cat of favCats) {
      for (const it of (byCat.get(cat) || [])) {
        const key = canon(it.title);
        if (suggestions.length >= limit) break;
        if (kitchenKeys.has(key)) continue;
        if (suggestions.some(s => canon(s.title) === key)) continue;
        suggestions.push({
          title: it.title,
          reason: `popular in your ${cat}`,
          category: cat,
          score: it.score ?? null,
          why: reasonFor(it.title, cat)
        });
      }
      if (suggestions.length >= limit) break;
    }
  }

  return { ok: true, count: suggestions.length, suggestions };
}


function pickByCategory(items, cat, max=2) {
  return items.filter(i => i.category === cat).slice(0,max);
}
function pickAny(items, max=2) {
  return items.slice(0,max);
}

async function tool_recommend_recipes({ mode, servings = 2, diet, proteins_preferred = [], avoid_substitutions = false } = {}) {
  const k = await tool_list_kitchen_14d({ limit: 500 });
  if (!k.ok) return k;
  const kitchen = k.items;

  // inventories by rough type
  const proteins = kitchen.filter(i =>
    /protein|meat|fish|chicken|turkey|beef|pork|lamb|tofu|tempeh|egg|yogurt|beans|tuna|salmon|sardine|shrimp|scallop/i.test(i.category || "") ||
    /chicken|turkey|beef|pork|lamb|tofu|tempeh|egg|yogurt|beans|tuna|salmon|sardine|shrimp|scallop/i.test(i.title || "")
  );
  const produce  = kitchen.filter(i => (i.category === "Produce") || /spinach|tomato|pepper|onion|garlic|broccoli|lettuce|avocado|mushroom|cilantro|herb/i.test(i.title||""));
  const grains   = kitchen.filter(i => (i.category === "Grains")  || /rice|pasta|quinoa|tortilla|bread|noodle|udon|ramen|farro/i.test(i.title||""));
  const dairy    = kitchen.filter(i => (i.category === "Dairy")   || /milk|cheese|yogurt|butter|cream|parmesan|mozzarella|feta/i.test(i.title||""));
  const cooking  = kitchen.filter(i => (i.category === "Cooking") || /oil|olive|spice|sauce|broth|stock|soy|miso|tomato sauce|salsa|vinegar/i.test(i.title||""));

  // --- Diverse proteins (as in your last version) ---
  const prefTokens = (proteins_preferred || []).map(canon).filter(Boolean);
  const presentProteinTitles = proteins.map(i => i.title);
  const resolvePresentMatch = (token) => presentProteinTitles.find(t => canon(t).includes(token)) || null;

  const preferredPresent = [];
  for (const p of prefTokens) {
    const m = resolvePresentMatch(p);
    if (m && !preferredPresent.some(x => canon(x) === canon(m))) preferredPresent.push(m);
  }
  const othersPresent = presentProteinTitles.filter(t => !preferredPresent.some(x => canon(x) === canon(t)));
  const preferredMissing = prefTokens
    .map(tok => presentProteinTitles.find(t => canon(t).includes(tok)) ? null : tok)
    .filter(Boolean)
    .filter((v, i, a) => a.indexOf(v) === i);

  const distinctProteins = [...preferredPresent, ...othersPresent];
  if (!distinctProteins.length) distinctProteins.push("eggs", "beans");
  if (mode === "store_flexible" && distinctProteins.length < 3 && preferredMissing.length) {
    for (const tok of preferredMissing) {
      if (distinctProteins.length >= 3) break;
      if (!distinctProteins.some(x => canon(x).includes(tok))) distinctProteins.push(tok);
    }
  }
  const proteinsForRecipes = distinctProteins.slice(0, 3);

  // Helpers
  const pickGrainFromKitchen = () =>
    grains[0]?.title || kitchenHasLike("rice", kitchen) || kitchenHasLike("pasta", kitchen) || kitchenHasLike("tortilla", kitchen) || null;

  // Plan “elevator” buys by recipe style; ensure at least 2 in shop mode
  function planElevatorBuys(style) {
    // Preferred flavor builders by style
    const common = ["garlic", "lemon", "fresh herbs", "olive oil"];
    const byStyle = {
      bowl: ["tahini", "avocado", "cucumber"],
      pasta: ["parmesan", "basil", "cherry tomatoes"],
      stir: ["soy sauce", "ginger", "scallions"],
      tacos: ["salsa", "lime", "cilantro"],
      salad: ["greens", "cucumber", "red onion"]
    };
    const wanted = [...common, ...(byStyle[style] || [])];

    // Add missing base if we don’t have one
    if (!pickGrainFromKitchen()) wanted.unshift("rice");

    // Convert to present/buy and keep only actual buys (store_flexible only)
    const buys = [];
    if (mode === "store_flexible") {
      for (const item of wanted) {
        const { have, buy } = presentOrBuy({ name: item, kitchenItems: kitchen, mode });
        if (buy) buys.push(buy);
      }
    }
    // Guarantee at least 2 buys to make shopping meaningful
    return Array.from(new Set(buys)).slice(0, 4); // cap 4 to keep concise
  }

  const results = [];

  function buildRecipe({ name, style, baseWanted, vegCount = 2, proteinName }) {
    const haveArr = [];
    let buyArr = [];

    // Protein
    if (proteinName) {
      const { have, buy } = presentOrBuy({ name: proteinName, kitchenItems: kitchen, mode });
      const prefRequested = prefTokens.length && prefTokens.some(p => canon(proteinName).includes(p));
      const mustNotSub = mode === "kitchen" && prefRequested && avoid_substitutions && !have;
      if (have) haveArr.push(have);
      if (buy && !mustNotSub) buyArr.push(buy);
    }

    // Base
    let base = baseWanted;
    if (base && !kitchenHasLike(base, kitchen)) {
      if (mode === "kitchen") base = pickGrainFromKitchen();
    }
    if (base) {
      const { have, buy } = presentOrBuy({ name: base, kitchenItems: kitchen, mode });
      if (have) haveArr.push(have);
      if (buy) buyArr.push(buy);
    }

    // Veg + pantry touches (from what’s on hand)
    for (const it of produce.slice(0, vegCount)) haveArr.push(it.title);
    if (cooking[0]) haveArr.push(cooking[0].title);

    // Add “elevator” buys for shop mode
    if (mode === "store_flexible") {
      buyArr = [...buyArr, ...planElevatorBuys(style)];
    }

    // Final de-dupe
    const dedupedBuy = dedupeBuy({ buy: Array.from(new Set(buyArr)), have: Array.from(new Set(haveArr)), kitchenItems: kitchen });

    // Ensure shop mode has meaningful buys (at least 2) — if not, add safe pantry items
    let finalBuy = dedupedBuy;
    if (mode === "store_flexible" && finalBuy.length < 2) {
      const safety = ["garlic", "lemon", "fresh herbs", "parmesan", "salsa", "soy sauce"];
      for (const s of safety) {
        const { buy } = presentOrBuy({ name: s, kitchenItems: kitchen, mode });
        if (buy && !finalBuy.some(x => canon(x) === canon(buy))) finalBuy.push(buy);
        if (finalBuy.length >= 2) break;
      }
      finalBuy = dedupeBuy({ buy: finalBuy, have: haveArr, kitchenItems: kitchen });
    }

    results.push({
      name,
      mode,
      have: Array.from(new Set(haveArr)),
      buy: mode === "store_flexible" ? finalBuy : [],
      steps: name.includes("Pasta") || name.includes("Noodle")
        ? `Boil base; sauté ${proteinName} + veg; toss with oil/salt; serves ${servings}.`
        : name.includes("Stir-Fry")
          ? `Stir-fry ${proteinName} + veg; sauce if handy; serve over base; serves ${servings}.`
          : name.includes("Tacos")
            ? `Warm tortillas, fill with ${proteinName} + veg; cheese/salsa if handy; serves ${servings}.`
            : `Cook base, sauté veg, add ${proteinName}; finish with oil/salt; serves ${servings}.`
    });
  }

  // Recipe 1: Grain bowl
  {
    const baseWanted = kitchenHasLike("quinoa", kitchen) ? "quinoa"
                      : kitchenHasLike("rice", kitchen) ? "rice"
                      : kitchenHasLike("farro", kitchen) ? "farro"
                      : (mode === "store_flexible" ? "rice" : null);
    const p = proteinsForRecipes[0] || "eggs";
    buildRecipe({ name: `${p} Veggie Grain Bowl`, style: "bowl", baseWanted, vegCount: 2, proteinName: p });
  }

  // Recipe 2: Pasta / noodle or stir-fry
  {
    const haveNoodle = kitchenHasLike("pasta", kitchen) || kitchenHasLike("noodle", kitchen);
    const baseWanted = haveNoodle || (mode === "store_flexible" ? "pasta" : null);
    const p = proteinsForRecipes[1] || proteinsForRecipes[0] || "beans";
    const style = (baseWanted && /pasta|noodle/i.test(baseWanted)) ? "pasta" : "stir";
    const title = style === "pasta" ? `${p} Weeknight Pasta` : `${p} Quick Stir-Fry`;
    buildRecipe({ name: title, style, baseWanted, vegCount: 2, proteinName: p });
  }

  // Recipe 3: Tacos or big salad
  {
    const hasTortilla = !!kitchenHasLike("tortilla", kitchen);
    const baseWanted = hasTortilla ? "tortillas" : (mode === "store_flexible" ? "greens" : kitchenHasLike("greens", kitchen) ? "greens" : null);
    const p = proteinsForRecipes[2] || proteinsForRecipes[1] || proteinsForRecipes[0] || "beans";
    const style = hasTortilla ? "tacos" : "salad";
    const title = hasTortilla ? `${p} 15-min Tacos` : `${p} Big Chopped Salad`;
    buildRecipe({ name: title, style, baseWanted, vegCount: hasTortilla ? 2 : 3, proteinName: p });
  }

  // Diet tweak (light)
  if (diet && /veg(etari|an)/i.test(diet)) {
    for (const r of results) {
      const drop = /(chicken|beef|turkey|salmon|tuna|shrimp|pork|lamb|scallop|bacon|sausage)/i;
      r.have = r.have.filter(x => !drop.test(x));
      r.buy = r.buy.filter(x => !drop.test(x));
      r.note = (r.note ? r.note + " " : "") + "Vegetarian-friendly tweak.";
    }
  }

  return { ok: true, count: results.length, recipes: results };
}


// Root Route (unchanged)
fastify.get('/', async (_request, reply) => {
  reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Route for Twilio to handle incoming calls (kept simple & brief)
fastify.all('/incoming-call', async (request, reply) => {
  const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Say voice="Google.en-US-Chirp3-HD-Aoede">Welcome to Trepo.</Say>
      <Pause length="1"/>
      <Say voice="Google.en-US-Chirp3-HD-Aoede">What can I help you with?</Say>
      <Connect>
        <Stream url="wss://${request.headers.host}/media-stream" />
      </Connect>
    </Response>`;
  reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for media-stream (tutorial flow preserved)
fastify.register(async (fastify) => {
  fastify.get('/media-stream', { websocket: true }, (connection, _req) => {
    console.log('Client connected');

    // Connection-specific state
    let streamSid = null;
    let latestMediaTimestamp = 0;
    let lastAssistantItem = null;
    let markQueue = [];
    let responseStartTimestampTwilio = null;

    const openAiWs = new WebSocket(`wss://api.openai.com/v1/realtime?model=gpt-realtime`, {
      headers: { Authorization: `Bearer ${OPENAI_API_KEY}` }
    });

    // Control initial session with OpenAI
    const initializeSession = () => {
      const sessionUpdate = {
        type: 'session.update',
        session: {
          type: 'realtime',
          model: "gpt-realtime",
          output_modalities: ["audio"],
          audio: {
            input: { format: { type: 'audio/pcmu' }, turn_detection: { type: "server_vad" } },
            output: { format: { type: 'audio/pcmu' }, voice: VOICE },
          },
          instructions: SYSTEM_MESSAGE,
          tools: TOOL_SCHEMAS,
          tool_choice: 'auto',
        },
      };

      console.log('Sending session update:', JSON.stringify(sessionUpdate));
      openAiWs.send(JSON.stringify(sessionUpdate));
      // If you want AI to greet first:
      // sendInitialConversationItem();
    };

    // Optional: AI speaks first
    const sendInitialConversationItem = () => {
      const initialConversationItem = {
        type: 'conversation.item.create',
        item: {
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Give a one-sentence hello and say “ask me about your kitchen or what you tossed.”' }
          ]
        }
      };
      openAiWs.send(JSON.stringify(initialConversationItem));
      openAiWs.send(JSON.stringify({ type: 'response.create' }));
    };

    // Handle interruption when the caller's speech starts
    const handleSpeechStartedEvent = () => {
      if (markQueue.length > 0 && responseStartTimestampTwilio != null) {
        const elapsedTime = latestMediaTimestamp - responseStartTimestampTwilio;

        if (lastAssistantItem) {
          const truncateEvent = {
            type: 'conversation.item.truncate',
            item_id: lastAssistantItem,
            content_index: 0,
            audio_end_ms: elapsedTime
          };
          openAiWs.send(JSON.stringify(truncateEvent));
        }

        connection.send(JSON.stringify({ event: 'clear', streamSid }));
        markQueue = [];
        lastAssistantItem = null;
        responseStartTimestampTwilio = null;
      }
    };

    // Send mark messages so we know if and when AI response playback is finished
    const sendMark = (connection, streamSid) => {
      if (streamSid) {
        const markEvent = { event: 'mark', streamSid, mark: { name: 'responsePart' } };
        connection.send(JSON.stringify(markEvent));
        markQueue.push('responsePart');
      }
    };

    // Open event for OpenAI WebSocket
    openAiWs.on('open', () => {
      console.log('Connected to the OpenAI Realtime API');
      setTimeout(initializeSession, 100);
    });

    // Listen for messages from the OpenAI WebSocket (and send to Twilio if necessary)
    openAiWs.on('message', async (data) => {
      try {
        const response = JSON.parse(data);

        if (LOG_EVENT_TYPES.includes(response.type)) {
          console.log(`Received event: ${response.type}`, response);
        }

        // Stream audio back to Twilio
        if (response.type === 'response.output_audio.delta' && response.delta) {
          const audioDelta = { event: 'media', streamSid, media: { payload: response.delta } };
          connection.send(JSON.stringify(audioDelta));

          if (!responseStartTimestampTwilio) {
            responseStartTimestampTwilio = latestMediaTimestamp;
          }

          if (response.item_id) {
            lastAssistantItem = response.item_id;
          }

          sendMark(connection, streamSid);
        }

        // Handle function calls emitted inside response.done
        if (response.type === 'response.done' && response?.response?.output?.length) {
          for (const item of response.response.output) {
            if (item?.type === "function_call") {
              const call_id = item.call_id || item.id;
              const name = item.name;
              let args = {};
              try {
                args = typeof item.arguments === 'string'
                  ? JSON.parse(item.arguments || '{}')
                  : (item.arguments || {});
              } catch {}

              let result;
              try {
                if (name === 'list_discarded_14d')      result = await tool_list_discarded_14d(args);
                else if (name === 'list_kitchen_14d')   result = await tool_list_kitchen_14d(args);
                else if (name === 'get_item_age')       result = await tool_get_item_age(args);
                else if (name === 'compute_needs')      result = await tool_compute_needs(args);
                else if (name === 'summarize_prefs')    result = await tool_summarize_prefs(args);
                else if (name === 'suggest_from_history') result = await tool_suggest_from_history(args);
                else if (name === 'recommend_recipes')  result = await tool_recommend_recipes(args);
                else result = { error: `unknown tool ${name}` };
              } catch (e) { result = { error: String(e) }; }

              // Send function_call_output then trigger a new response
              openAiWs.send(JSON.stringify({
                type: "conversation.item.create",
                item: { type: "function_call_output", call_id, output: JSON.stringify(result) }
              }));
              openAiWs.send(JSON.stringify({ type: "response.create" }));
            }
          }
        }

        if (response.type === 'input_audio_buffer.speech_started') {
          handleSpeechStartedEvent();
        }
      } catch (error) {
        console.error('Error processing OpenAI message:', error, 'Raw message:', data);
      }
    });

    // Handle incoming messages from Twilio (unchanged)
    connection.on('message', (message) => {
      try {
        const data = JSON.parse(message);

        switch (data.event) {
          case 'media':
            latestMediaTimestamp = data.media.timestamp;
            if (openAiWs.readyState === WebSocket.OPEN) {
              const audioAppend = { type: 'input_audio_buffer.append', audio: data.media.payload };
              openAiWs.send(JSON.stringify(audioAppend));
            }
            break;
          case 'start':
            streamSid = data.start.streamSid;
            console.log('Incoming stream has started', streamSid);
            responseStartTimestampTwilio = null; 
            latestMediaTimestamp = 0;
            break;
          case 'mark':
            if (markQueue.length > 0) {
              markQueue.shift();
            }
            break;
          default:
            console.log('Received non-media event:', data.event);
            break;
        }
      } catch (error) {
        console.error('Error parsing message:', error, 'Message:', message);
      }
    });

    // Handle connection close
    connection.on('close', () => {
      if (openAiWs.readyState === WebSocket.OPEN) openAiWs.close();
      console.log('Client disconnected.');
    });

    // Handle WebSocket close and errors
    openAiWs.on('close', () => {
      console.log('Disconnected from the OpenAI Realtime API');
    });

    openAiWs.on('error', (error) => {
      console.error('Error in the OpenAI WebSocket:', error);
    });
  });
});

fastify.listen({ port: PORT }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server is listening on port ${PORT}`);
});
