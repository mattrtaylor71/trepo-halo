// BENCHMARK 3 — Fast photo identify. Real quickIdentify fast prompt on 10 real capture
// images (incl. THE Merlot image — deep-path truth "Merlot Wine", a known hard case).
// Models: gpt-4.1-mini (current fast tier) vs gpt-5.4-mini vs gpt-5.4. Ground truth =
// deep-path/DB final labels. Accuracy graded by a BLIND opus pass (0=wrong,1=partial,
// 2=correct) over shuffled+anonymized predictions. Reports latency + cost/image (this
// tier is volume-sensitive). NOTE: production caps this tier at 220 output tokens; we
// use a higher cap so reasoning models can answer at all — see report caveat.
import { readFileSync, writeFileSync } from "node:fs";
import { chat, imageContent } from "./lib/openai.mjs";
import { judge } from "./lib/anthropic.mjs";
import { costUSD } from "./lib/pricing.mjs";

const MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4"];
const MAX_TOKENS = 800; // prod fast tier = 220; raised so reasoning models can answer (caveat in report)
const manifest = JSON.parse(readFileSync(new URL("./fixtures/identify_images.json", import.meta.url)));

const SYSTEM = "You are the original ultra-fast grocery identifier. Return strict JSON only. Identify the single main grocery item visible in the image. Prioritize the most obvious foreground item and ignore background objects. Keep the answer concise. If the image is unclear, return one conservative best guess and set needs_review to true. Include brand and category only when reasonably visible.";
const USER = "What is the main item in this image? Return one best-guess grocery item with item_name, brand, category, confidence, short_description, visible_text, item_type, needs_review, and uncertainty_reasons.";

function imgDataUri(s3key) {
  const b = readFileSync(new URL(`./fixtures/images/${s3key.split("/").pop()}`, import.meta.url));
  return `data:image/jpeg;base64,${b.toString("base64")}`;
}
function pickName(parsed) {
  if (!parsed) return null;
  if (parsed.item_name) return { name: parsed.item_name, brand: parsed.brand, category: parsed.category, confidence: parsed.confidence, needs_review: parsed.needs_review };
  if (Array.isArray(parsed.items) && parsed.items[0]) { const i = parsed.items[0]; return { name: i.item_name, brand: i.brand, category: i.category, confidence: i.confidence, needs_review: i.needs_review }; }
  return null;
}
function parseJson(t) { const f = t.match(/```(?:json)?\s*([\s\S]*?)```/); const c = f ? f[1] : t; try { return JSON.parse(c); } catch {} try { return JSON.parse(c.slice(c.indexOf("{"), c.lastIndexOf("}") + 1)); } catch { return null; } }

const preds = []; // flat list for blind grading
const raw = {};
for (const model of MODELS) {
  raw[model] = [];
  for (const img of manifest.images) {
    const messages = [{ role: "system", content: SYSTEM }, { role: "user", content: imageContent(imgDataUri(img.s3_key), USER, "auto") }];
    const res = await chat({ model, messages, response_format: { type: "json_object" }, max_tokens: MAX_TOKENS });
    const parsed = res.ok ? parseJson(res.text) : null;
    const p = pickName(parsed);
    const rec = { model, id: img.id, truth: img.truth_name, truthCat: img.truth_category, pred: p?.name || null, predCat: p?.category || null, confidence: p?.confidence ?? null, needsReview: p?.needs_review ?? null, latencyMs: res.latencyMs, usage: res.usage, cost: costUSD(model, res.usage), reasoningOff: !!res.reasoningOff, error: res.error || null };
    raw[model].push(rec);
    preds.push({ gradeIdx: preds.length, truth: img.truth_name, truthCat: img.truth_category, pred: p?.name || "(none)" });
    console.log(`  ${model} ${img.id}: truth="${img.truth_name}" pred="${p?.name || "(none)"}" lat=${res.latencyMs}ms`);
  }
}

// ---- BLIND accuracy grade (opus) over shuffled+anonymized predictions ----
const sys = "You grade grocery-image identification. For each item you get the GROUND TRUTH label and a MODEL PREDICTION. Score 2 = correct (same product, synonyms/brand differences OK), 1 = partial (right category/close but wrong specific product), 0 = wrong or empty. Return ONLY JSON.";
const body = "Grade these predictions:\n" + preds.map((p) => `#${p.gradeIdx}: truth="${p.truth}" [${p.truthCat}] prediction="${p.pred}"`).join("\n")
  + `\n\nReturn JSON: {"grades":[{"idx":0,"score":2},...]} with one entry per # above.`;
const jr = await judge({ system: sys, user: body, maxTokens: 1500 });
const grades = {};
for (const g of (jr.json?.grades || [])) grades[g.idx] = g.score;
let gi = 0;
for (const model of MODELS) for (const rec of raw[model]) rec.accScore = grades[gi++] ?? null;

// aggregate
const agg = {};
for (const model of MODELS) {
  const rows = raw[model]; const n = rows.length;
  const scores = rows.map((r) => r.accScore ?? 0);
  const correct = scores.filter((s) => s === 2).length, partial = scores.filter((s) => s === 1).length, wrong = scores.filter((s) => s === 0).length;
  const lat = rows.map((r) => r.latencyMs).sort((a, b) => a - b);
  const cost = rows.reduce((s, r) => s + (r.cost?.usd || 0), 0);
  agg[model] = {
    n, topLabelAccuracy: +(correct / n).toFixed(3), partialRate: +(partial / n).toFixed(3), wrongRate: +(wrong / n).toFixed(3),
    avgAccScore: +(scores.reduce((a, b) => a + b, 0) / n).toFixed(2),
    medianLatencyMs: lat[Math.floor(n / 2)], p90LatencyMs: lat[Math.floor(n * 0.9)],
    costPer1kImages: +((cost / n) * 1000).toFixed(2), costEst: costUSD(model, {}).est,
    merlotPred: rows.find((r) => r.id === "05afcaf9")?.pred || null,
  };
}
writeFileSync(new URL("./out/identify_raw.json", import.meta.url), JSON.stringify({ raw, agg }, null, 2));
console.log("\n=== IDENTIFY AGGREGATE ==="); console.table(agg);
