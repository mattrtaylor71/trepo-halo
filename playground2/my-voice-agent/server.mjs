// server.mjs
import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import morgan from "morgan";
import chalk from "chalk";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(morgan("dev"));

// --- config ----------------------------------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("Set OPENAI_API_KEY in env");
  process.exit(1);
}
const PORT = process.env.PORT || 5174;

// Your fixed Auth0 sub
const GROCERY_AUTH0_SUB = "auth0|672c4ebdba93ac8a306410d5";

// Lambdas
const LAMBDA_RECYCLABLES =
  "https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables";
const LAMBDA_KITCHEN =
  "https://i9dzt51kkg.execute-api.us-east-1.amazonaws.com/fetchKitchen";

// --- tiny helpers ----------------------------------------------
const redact = (t = "") => (t.length > 8 ? t.slice(0, 4) + "***" + t.slice(-4) : "***");
const log = {
  info: (...a) => console.log(chalk.blue("[info]"), ...a),
  ok:   (...a) => console.log(chalk.green("[ok]  "), ...a),
  err:  (...a) => console.log(chalk.red("[err] "), ...a),
};

async function parseLambdaResponse(r) {
  const raw = await r.text();
  const tag = Math.random().toString(36).slice(2, 7);
  log.info(`[${tag}] lambda status=${r.status} len=${raw.length}`);
  log.info(`[${tag}] preview: ${raw.slice(0, 300)}`);
  if (!r.ok) return { error: raw, status: r.status };

  let payload;
  try { payload = JSON.parse(raw); } catch { payload = { message: String(raw) }; }
  if (payload && typeof payload === "object" && typeof payload.body === "string") {
    try { payload = JSON.parse(payload.body); } catch {}
  }
  return { payload, status: r.status };
}

const normalizeItems = (arr = []) =>
  arr
    .map((it) => ({
      id: it._id || it.user_item_id || it.id || null,    // user item row id
      original_id: it.original_id || it.item || null,     // product id when present
      title: it.title || it.name || "Unknown item",
      images: it.images || it.image_urls || "",
      score: typeof it.score === "number" ? it.score : null,
      category: it.simplified_category || it.category || "Other",
      brand: it.brand || it.manufacturer || null,
      inventory: it.inventory ?? null,
      createdDate: it._createdDate || it.createdAt || it.date || null,
    }))
    .filter((x) => x.createdDate && x.title)
    .sort((a, b) => new Date(b.createdDate) - new Date(a.createdDate));

const uniqByLowerTitle = (list) => {
  const seen = new Set();
  const out = [];
  for (const it of list) {
    const key = (it.title || "").toLowerCase().trim();
    if (!seen.has(key)) { seen.add(key); out.push(it); }
  }
  return out;
};

// --- health -----------------------------------------------------
app.get("/health", (_req, res) =>
  res.json({ ok: true, time: new Date().toISOString() })
);

// --- Discarded (fetchRecyclables, last ~14d) -------------------
app.get("/groceries/discarded14", async (req, res) => {
  try {
    const limit = Math.min(500, Math.max(1, parseInt(req.query.limit ?? "200", 10)));
    const r = await fetch(LAMBDA_RECYCLABLES, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_name: GROCERY_AUTH0_SUB }),
    });
    const { payload, error, status } = await parseLambdaResponse(r);
    if (error) return res.status(status || 502).json({ ok: false, error });

    if (payload?.message && /No matching items/i.test(payload.message)) {
      return res.json({ ok: true, count: 0, items: [], note: "~last 14 days only" });
    }

    const arr = Array.isArray(payload) ? payload : [];
    const items = normalizeItems(arr).slice(0, limit);

    return res.json({
      ok: true,
      count: items.length,
      items,
      note: "These are items you disposed of in the last ~14 days.",
    });
  } catch (e) {
    log.err("discarded14 route error:", e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Kitchen (fetchKitchen, last ~14d) --------------------------
app.get("/groceries/kitchen14", async (req, res) => {
  try {
    const limit = Math.min(500, Math.max(1, parseInt(req.query.limit ?? "200", 10)));
    const r = await fetch(LAMBDA_KITCHEN, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auth0_sub: GROCERY_AUTH0_SUB }),
    });
    const { payload, error, status } = await parseLambdaResponse(r);
    if (error) return res.status(status || 502).json({ ok: false, error });

    if (payload?.message && /No matching items/i.test(payload.message)) {
      return res.json({ ok: true, count: 0, items: [], note: "~last 14 days only" });
    }

    const arr = Array.isArray(payload) ? payload : [];
    const items = normalizeItems(arr).slice(0, limit);

    return res.json({
      ok: true,
      count: items.length,
      items,
      note: "These are items currently in your kitchen (feed covers ~14 days).",
    });
  } catch (e) {
    log.err("kitchen14 route error:", e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Needs = discarded - in-kitchen (by title) ------------------
app.get("/groceries/needs", async (req, res) => {
  try {
    const limit = Math.min(200, Math.max(1, parseInt(req.query.limit ?? "50", 10)));

    const [discRes, kitRes] = await Promise.all([
      fetch(LAMBDA_RECYCLABLES, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_name: GROCERY_AUTH0_SUB }),
      }),
      fetch(LAMBDA_KITCHEN, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth0_sub: GROCERY_AUTH0_SUB }),
      })
    ]);

    const [{ payload: disc }, { payload: kit }] = await Promise.all([
      parseLambdaResponse(discRes),
      parseLambdaResponse(kitRes)
    ]);

    const discarded = normalizeItems(Array.isArray(disc) ? disc : []);
    const kitchen   = normalizeItems(Array.isArray(kit)  ? kit  : []);

    const inKitchenTitles = new Set(kitchen.map(i => (i.title || "").toLowerCase().trim()));
    const needsRaw = discarded.filter(i => !inKitchenTitles.has((i.title || "").toLowerCase().trim()));
    const needs = uniqByLowerTitle(needsRaw).slice(0, limit);

    return res.json({
      ok: true,
      count: needs.length,
      items: needs,
      note: "Heuristic: recently discarded items you don't currently have.",
    });
  } catch (e) {
    log.err("needs route error:", e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Preferences from both feeds --------------------------------
app.get("/groceries/preferences", async (_req, res) => {
  try {
    const [discRes, kitRes] = await Promise.all([
      fetch(LAMBDA_RECYCLABLES, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_name: GROCERY_AUTH0_SUB }),
      }),
      fetch(LAMBDA_KITCHEN, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth0_sub: GROCERY_AUTH0_SUB }),
      })
    ]);

    const [{ payload: disc }, { payload: kit }] = await Promise.all([
      parseLambdaResponse(discRes),
      parseLambdaResponse(kitRes)
    ]);

    const all = normalizeItems([
      ...(Array.isArray(disc) ? disc : []),
      ...(Array.isArray(kit)  ? kit  : []),
    ]);

    const tallies = {
      category: new Map(),
      brand: new Map(),
    };
    let avgScore = { sum: 0, n: 0 };

    for (const it of all) {
      if (it.category) tallies.category.set(it.category, 1 + (tallies.category.get(it.category) || 0));
      if (it.brand)    tallies.brand.set(it.brand,       1 + (tallies.brand.get(it.brand) || 0));
      if (typeof it.score === "number") { avgScore.sum += it.score; avgScore.n += 1; }
    }

    const topN = (m, n=5) => [...m.entries()].sort((a,b)=>b[1]-a[1]).slice(0,n).map(([name,count])=>({name,count}));
    const favorites = {
      categories: topN(tallies.category, 5),
      brands:     topN(tallies.brand, 5),
      mean_score: avgScore.n ? +(avgScore.sum / avgScore.n).toFixed(1) : null
    };

    return res.json({ ok: true, favorites, sample_size: all.length });
  } catch (e) {
    log.err("preferences route error:", e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Light suggestions based on history (no external catalog) ---
app.get("/groceries/suggest", async (req, res) => {
  try {
    const limit = Math.min(20, Math.max(1, parseInt(req.query.limit ?? "10", 10)));
    const [discRes, kitRes] = await Promise.all([
      fetch(LAMBDA_RECYCLABLES, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_name: GROCERY_AUTH0_SUB }),
      }),
      fetch(LAMBDA_KITCHEN, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth0_sub: GROCERY_AUTH0_SUB }),
      })
    ]);

    const [{ payload: disc }, { payload: kit }] = await Promise.all([
      parseLambdaResponse(discRes),
      parseLambdaResponse(kitRes)
    ]);

    const discarded = normalizeItems(Array.isArray(disc) ? disc : []);
    const kitchen   = normalizeItems(Array.isArray(kit)  ? kit  : []);
    const all       = [...discarded, ...kitchen];

    // Restock = frequent discards not in kitchen
    const inKitchen = new Set(kitchen.map(i => (i.title || "").toLowerCase().trim()));
    const discardCounts = new Map();
    for (const d of discarded) {
      const key = (d.title || "").toLowerCase().trim();
      discardCounts.set(key, 1 + (discardCounts.get(key) || 0));
    }
    const restock = uniqByLowerTitle(
      discarded
        .filter(d => !inKitchen.has((d.title || "").toLowerCase().trim()))
        .sort((a,b)=> (discardCounts.get((b.title||"").toLowerCase().trim())||0) - (discardCounts.get((a.title||"").toLowerCase().trim())||0))
    ).slice(0, Math.ceil(limit/2));

    // “On-brand” suggestions: pick high-score items from your own history in favorite categories
    const byCat = new Map();
    for (const it of all) {
      if (!it.category) continue;
      if (!byCat.has(it.category)) byCat.set(it.category, []);
      byCat.get(it.category).push(it);
    }
    const suggestions = [];
    for (const [cat, items] of byCat.entries()) {
      const top = items
        .filter(x => typeof x.score === "number")
        .sort((a,b)=> b.score - a.score)
        .slice(0, 2); // 2 per category
      for (const x of top) suggestions.push({ title: x.title, category: cat, score: x.score, brand: x.brand });
    }
    const trimmedSuggestions = uniqByLowerTitle(suggestions).slice(0, Math.max(3, Math.floor(limit/2)));

    return res.json({
      ok: true,
      restock,
      suggestions: trimmedSuggestions,
      note: "Suggestions are based on your own history (no external catalog)."
    });
  } catch (e) {
    log.err("suggest route error:", e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Realtime session: expose smarter tools --------------------
app.post("/session", async (_req, res) => {
  try {
    const tools = [
      {
        type: "function",
        name: "list_discarded_14d",
        description: "Items the user has thrown out in the last ~14 days.",
        parameters: { type: "object", properties: { limit: { type: "integer", minimum: 1, maximum: 500 } }, additionalProperties: false }
      },
      {
        type: "function",
        name: "list_kitchen_14d",
        description: "Items currently in the user's kitchen (last ~14 days).",
        parameters: { type: "object", properties: { limit: { type: "integer", minimum: 1, maximum: 500 } }, additionalProperties: false }
      },
      {
        type: "function",
        name: "compute_needs",
        description: "What to buy next: recently discarded items not currently in the kitchen.",
        parameters: { type: "object", properties: { limit: { type: "integer", minimum: 1, maximum: 200 } }, additionalProperties: false }
      },
      {
        type: "function",
        name: "infer_preferences",
        description: "Favorite categories/brands inferred from discarded + kitchen.",
        parameters: { type: "object", properties: {}, additionalProperties: false }
      },
      {
        type: "function",
        name: "suggest_from_history",
        description: "Restock list + a few on-brand recs (based on the user's own history; no external catalog).",
        parameters: { type: "object", properties: { limit: { type: "integer", minimum: 1, maximum: 20 } }, additionalProperties: false }
      },
      {
        type: "function",
        name: "get_item_age",
        description: "Given an item name, return when it was added and how many days since (from kitchen).",
        parameters: { type: "object", properties: { name: { type: "string" } }, required: ["name"], additionalProperties: false }
      }
    ];

    const body = {
      model: "gpt-realtime",
      modalities: ["audio","text"],
      voice: "marin",
      tools,
      tool_choice: "auto",
      instructions: [
        "You are a fast grocery co-pilot. Assume the user is planning, driving to, or in a grocery store—or standing in their kitchen.",
        "",
        "INTENT RULES:",
        "- If they ask 'what do I need' or 'what should I buy', call compute_needs; then offer 3–5 extras via suggest_from_history.",
        "- If they ask 'what's in my kitchen', call list_kitchen_14d.",
        "- If they ask 'what have I tossed/discarded', call list_discarded_14d.",
        "- If they ask 'recommendations' or 'what should I get', call suggest_from_history and optionally infer_preferences.",
        "- If they ask 'recipes', assume kitchen-first. If they mention 'store', allow missing ingredients. Offer 2–3 ideas.",
        "",
        "OUTPUT STYLE:",
        "- Keep replies to 2–4 compact sentences or a tight list.",
        "- If a tool returns a note, briefly paraphrase it.",
        "- Use names, not dates, unless asked. If asked 'when/age', call get_item_age.",
        "- Health/budget: include one sensible swap or tip if relevant.",
      ].join("\n"),
    };

    const r = await fetch("https://api.openai.com/v1/realtime/sessions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
      },
      body: JSON.stringify(body),
    });

    const txt = await r.text();
    if (!r.ok) return res.status(r.status).send(txt);
    const json = JSON.parse(txt);
    return res.json(json);
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
});

// --- start ------------------------------------------------------
app.listen(PORT, () => {
  console.log(`Ephemeral session server on http://localhost:${PORT}`);
  console.log(`OPENAI_API_KEY = ${redact(OPENAI_API_KEY)}`);
});
