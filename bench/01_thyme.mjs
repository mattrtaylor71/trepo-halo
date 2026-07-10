// BENCHMARK 1 — Thyme tool discipline + quality.
// Imports the REAL system prompt + 48 tool schemas from the deployed voice code, and
// replays 12 single-turn cases × N models × 3 reps via chat/completions. Measures
// each model's NATIVE tool discipline (tool_choice:"auto"). We annotate — but do NOT
// apply — production's detectWriteIntent regex, so the numbers reflect model judgment,
// not the regex force. Production additionally forces a tool on first-turn write-intent
// matches; that would mask model gaps on regex-matching phrasings but NOT on the
// read-back / pure-chat / self-poison cases (where discipline actually matters).
import { readFileSync, writeFileSync } from "node:fs";
import { chat } from "./lib/openai.mjs";
import { costUSD } from "./lib/pricing.mjs";
import { buildSystemPrompt, buildChatTools }
  from "../playground12_voice_ack/trepo-quick-ack/lib/realtime-config.mjs";
import { TOOL_CATEGORIES } from "../shared/voice-assistant/tool-definitions.mjs";

const MODELS = ["gpt-4.1", "gpt-5.4-2026-03-05", "gpt-5.5", "gpt-5.6-sol"];
const REPS = 3;
const TEMPERATURE = 0.2; // production Thyme value
const CASES = JSON.parse(readFileSync(new URL("./fixtures/thyme_cases.json", import.meta.url)));
const WRITE_TOOLS = new Set(TOOL_CATEGORIES.write);
const READ_TOOLS = new Set(TOOL_CATEGORIES.read);

// Production detectWriteIntent patterns (device-assistant.mjs) — reproduced for
// ANNOTATION ONLY (what prod would force). Not applied to the actual call.
const WRITE_INTENT = [
  [/\b(add|put|throw|get|toss)\b.{0,30}\b(list|shopping|grocery)\b/i, "add_to_shopping_list"],
  [/\b(remove|delete|take off)\b.{0,30}\b(list|shopping|grocery)\b/i, "remove_from_shopping_list"],
  [/\b(clear|empty|wipe)\b.{0,20}\b(list|shopping|grocery)\b/i, "clear_shopping_list"],
  [/\b(check.?in|add.{0,10}kitchen|stock|restock)\b/i, "check_in_item"],
  [/\b(discard|threw away|throw away|toss out|get rid of)\b/i, "discard_item"],
  [/\b(mark|it.s|it is).{0,10}\bopen/i, "mark_item_opened"],
  [/\b(log|had|ate|just ate|i ate|i had|eaten|for (breakfast|lunch|dinner|snack))\b/i, "log_dish_from_voice"],
  [/\b(bought|picked up|got it|checked off)\b.{0,20}\b(list|shopping)?\b/i, "mark_shopping_item_bought"],
];
const detectWriteIntent = (t) => { for (const [re, tool] of WRITE_INTENT) if (re.test(t)) return tool; return null; };

// Light, documented grounding so read-backs are realistic (NOT the full prod grounding
// pipeline — that pulls Oura/DynamoDB/food-scoring; here a static kitchen+shopping
// snapshot from the test-user fixture is enough to exercise tool CHOICE).
const kitchen = JSON.parse(readFileSync(new URL("./fixtures/kitchen_testuser.json", import.meta.url)));
const groundingMsgs = [
  { role: "system", content: "Current kitchen (grounding):\n" + kitchen.items.map((i) => `- ${i.product_name}${i.quantity ? ` (${i.quantity})` : ""}`).join("\n") },
  { role: "system", content: "Current shopping list (grounding):\n- Olive oil [Costco]\n- Paper towels" },
];

const USER_CTX = { firstName: "Matt", householdSize: 1 };
const systemPrompt = buildSystemPrompt(USER_CTX, { responseSurface: "app" });
const tools = buildChatTools();

function scoreRep(kase, res) {
  if (!res.ok) return { outcome: "error", detail: res.error };
  const names = res.toolCalls.map((t) => t.name);
  const calledWrite = names.filter((n) => WRITE_TOOLS.has(n));
  const calledRead = names.filter((n) => READ_TOOLS.has(n));
  const accept = new Set(kase.accept);
  const claimsDone = /\b(added|logged|checked in|checked into|saved|removed|cleared|discarded|has been|have been|done|got it,? .*log)\b/i.test(res.text || "");
  if (kase.expect === "write") {
    if (names.some((n) => accept.has(n))) return { outcome: "correct", names };
    if (names.length) return { outcome: "wrong_tool", names };
    // No tool AND the text claims the action happened = FABRICATED confirmation
    // (the exact failure the system prompt forbids). Worse than a plain miss.
    if (claimsDone) return { outcome: "fabrication", names, flag: "FABRICATED confirmation w/o tool call" };
    return { outcome: "no_tool", names };
  }
  if (kase.expect === "read") {
    if (calledWrite.length) return { outcome: "wrong_tool", names, flag: "WROTE on a read-back" };
    if (names.some((n) => accept.has(n))) return { outcome: "correct", names };
    if (calledRead.length) return { outcome: "partial_read", names };
    return { outcome: "no_tool", names };
  }
  // expect none (pure chat / musing)
  if (names.length === 0) return { outcome: "correct", names };
  return { outcome: "wrong_tool", names, flag: calledWrite.length ? "unwanted WRITE" : "unwanted tool" };
}

const results = [];
for (const model of MODELS) {
  for (const kase of CASES) {
    const messages = [
      { role: "system", content: systemPrompt },
      ...groundingMsgs,
      ...(kase.history || []),
      { role: "user", content: kase.transcript },
    ];
    for (let rep = 0; rep < REPS; rep++) {
      const res = await chat({ model, messages, tools, tool_choice: "auto", temperature: TEMPERATURE, max_tokens: 700 });
      const score = scoreRep(kase, res);
      results.push({
        model, case: kase.id, group: kase.group, rep, expect: kase.expect,
        prodWouldForce: detectWriteIntent(kase.transcript),
        outcome: score.outcome, flag: score.flag || null,
        reasoningOff: res.ok ? !!res.reasoningOff : null,
        toolCalls: res.ok ? res.toolCalls : null,
        text: res.ok ? (res.text || "").slice(0, 400) : null,
        latencyMs: res.latencyMs, usage: res.usage, cost: costUSD(model, res.usage),
        error: res.ok ? null : res.error,
      });
      process.stdout.write(`  ${model} ${kase.id} r${rep}: ${score.outcome}${score.flag ? " ⚠" + score.flag : ""}\n`);
    }
  }
}
writeFileSync(new URL("./out/thyme_raw.json", import.meta.url), JSON.stringify(results, null, 2));

// aggregate
const agg = {};
for (const m of MODELS) {
  const rows = results.filter((r) => r.model === m);
  const n = rows.length;
  const correct = rows.filter((r) => r.outcome === "correct").length;
  const wrong = rows.filter((r) => r.outcome === "wrong_tool").length;
  const notool = rows.filter((r) => r.outcome === "no_tool").length;
  const fabrication = rows.filter((r) => r.outcome === "fabrication").length;
  const badWrite = rows.filter((r) => r.flag && /WRITE|read-back/.test(r.flag)).length;
  const lat = rows.filter((r) => r.latencyMs).map((r) => r.latencyMs).sort((a, b) => a - b);
  const cost = rows.reduce((s, r) => s + (r.cost?.usd || 0), 0);
  const inTok = rows.reduce((s, r) => s + (r.usage?.prompt_tokens || 0), 0) / n;
  const outTok = rows.reduce((s, r) => s + (r.usage?.completion_tokens || 0), 0) / n;
  agg[m] = {
    n, correctRate: +(correct / n).toFixed(3), wrongToolRate: +(wrong / n).toFixed(3),
    noToolRate: +(notool / n).toFixed(3), fabricationRate: +(fabrication / n).toFixed(3),
    badWriteOnReadback: badWrite,
    medianLatencyMs: lat[Math.floor(lat.length / 2)] || null,
    avgInTok: Math.round(inTok), avgOutTok: Math.round(outTok),
    costPer1k: +((cost / n) * 1000).toFixed(2), costEst: costUSD(m, {}).est,
  };
}
writeFileSync(new URL("./out/thyme_agg.json", import.meta.url), JSON.stringify(agg, null, 2));
console.log("\n=== THYME AGGREGATE ===");
console.table(agg);
