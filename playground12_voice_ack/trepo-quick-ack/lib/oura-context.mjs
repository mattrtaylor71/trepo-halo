// Oura ring body context for Thyme. Reads the compact body_state that
// trepo-oura-refresher maintains in TrepoOuraState (keyed by user UUID).
// Fail-safe by design: no state, stale state, table missing, IAM denied,
// or slow read (>800ms) all resolve to null and Thyme behaves exactly as
// it did before Oura existed.
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";

const TABLE = process.env.OURA_STATE_TABLE || "TrepoOuraState";
const MAX_AGE_MS = Number(process.env.OURA_CONTEXT_MAX_AGE_MS || 6 * 60 * 60 * 1000);
const READ_TIMEOUT_MS = Number(process.env.OURA_CONTEXT_TIMEOUT_MS || 800);

const docClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));

async function getStateItem(ownerCandidates) {
  for (const key of ownerCandidates) {
    if (!key) continue;
    const res = await docClient.send(new GetCommand({ TableName: TABLE, Key: { owner_id: String(key) } }));
    if (res?.Item?.body_state) return res.Item;
  }
  return null;
}

function describe(state, illness, firstName, updatedAgoMin) {
  const lines = [];
  const r = state.readiness || {};
  const s = state.sleep || {};
  const n = state.last_night || {};
  const a = state.activity || {};
  const bits = [];
  if (r.score != null) {
    let t = `readiness ${r.score}`;
    if (r.temperature_deviation_c != null) t += ` (body temp ${r.temperature_deviation_c >= 0 ? "+" : ""}${Number(r.temperature_deviation_c).toFixed(2)}°C vs baseline)`;
    bits.push(t);
  }
  if (s.score != null) {
    let t = `sleep score ${s.score}`;
    if (n.total_sleep_h != null) t += ` (${n.total_sleep_h}h slept${n.avg_hrv != null ? `, avg HRV ${n.avg_hrv}` : ""})`;
    bits.push(t);
  }
  if (a.steps != null) bits.push(`activity today: ${a.steps} steps, ${a.active_calories ?? "?"} active cal`);
  if (state.stress_summary) bits.push(`stress: ${state.stress_summary}`);
  if (Array.isArray(state.workouts_today) && state.workouts_today.length > 0) {
    bits.push(`workouts today: ${state.workouts_today.map((w) => `${w.activity} (${w.intensity})`).join(", ")}`);
  }
  if (bits.length === 0) return null;

  lines.push(`Oura ring body data for ${firstName || "the user"} (synced ${updatedAgoMin} min ago): ${bits.join("; ")}.`);
  const flags = Array.isArray(state.flags) ? state.flags : [];
  if (flags.length > 0) lines.push(`Signals: ${flags.join(", ").replace(/_/g, " ")}.`);
  if (illness?.active) {
    lines.push("Heads up: early illness-onset signals were detected today (" +
      (illness.signals || []).filter((x) => x !== "FORCED TEST TRIGGER").join("; ") +
      "). Lean suggestions toward gentle food — soup, hydration, light dinners — and an early night.");
  }
  lines.push("When food questions come up, tailor suggestions to this body state naturally " +
    "(lighter/earlier dinner after poor sleep or low readiness, protein after hard workouts, " +
    "gentle food if temperature is elevated). Mention the reason briefly and conversationally. " +
    "Never make medical claims or diagnoses; do not bring up this data when it isn't relevant.");
  return lines.join("\n");
}

export async function buildOuraGroundingMessage(userContext, env) {
  if ((env?.OURA_CONTEXT_ENABLED || "true") === "false") return null;
  try {
    const read = getStateItem([userContext?.userId, userContext?.ownerId]);
    const item = await Promise.race([
      read,
      new Promise((resolve) => setTimeout(resolve, READ_TIMEOUT_MS, null))
    ]);
    if (!item) return null;
    const updatedAt = Date.parse(item.body_state_updated_at || "");
    if (!updatedAt || Date.now() - updatedAt > MAX_AGE_MS) return null;
    const state = typeof item.body_state === "string" ? JSON.parse(item.body_state) : item.body_state;
    return describe(state, item.illness, userContext?.firstName, Math.round((Date.now() - updatedAt) / 60000));
  } catch (err) {
    console.warn("[OURA] grounding skipped:", err?.message || err);
    return null;
  }
}
