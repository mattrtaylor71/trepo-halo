import { Fault, key, scope, now } from "./core.mjs";
const object = (properties, required) => ({ type: "object", properties, required, additionalProperties: false });
export const CONVERSATION_TOOLS = [
  {
    type: "function", name: "remember_conversation_preferences",
    description: "Remember ONLY explicit answers or constraints from the CURRENT user message for this chat. Call before generating when shopping allowance, servings, or time is supplied or changed. Exact source_quote must occur in that message. Does not save household memory, change dietary restrictions, grant approval, or change app data. Null clears a field. Do not remember your own assumptions.",
    parameters: object({ updates: { type: "array", minItems: 1, maxItems: 3, items: object({
      field: { type: "string", enum: ["inventory_mode", "servings", "max_minutes"] },
      value: { anyOf: [{ type: "string", enum: ["existing_only", "shopping_ok", "either"] }, { type: "integer", minimum: 1, maximum: 1440 }, { type: "null" }] },
      source_quote: { type: "string", minLength: 1, maxLength: 1000 },
    }, ["field", "value", "source_quote"]) } }, ["updates"]),
  },
  {
    type: "function", name: "ask_clarifying_question",
    description: "Ask ONE short question only when missing information materially changes the answer. Record it here, then repeat the question as your final reply and stop. Do not also generate a recipe or prepare changes. Give 2-3 concrete choices in the question when useful; free-text replies are always welcome. Use conversation_preferences and history to avoid repeating answered questions.",
    parameters: object({ question: { type: "string", minLength: 8, maxLength: 350 } }, ["question"]),
  },
];
export async function readConversation(store, a, sid) {
  const row = await store.get(scope(a), key(sid, "C", "preferences"));
  return { scope: "this_conversation_only", values: row?.values || {}, pending_question: row?.pending || null };
}
export async function conversationTool(store, a, s, call) {
  const pk = scope(a), sk = key(s.id, "C", "preferences");
  const [req, current] = await Promise.all([store.get(pk, "Q#" + s.activeRequest), store.get(pk, key(s.id))]);
  if (!req || current?.activeRequest !== req.id || req.sessionId !== s.id || req.kind !== "message" || req.actor !== a.actor || req.status !== "running")
    throw new Fault("conversation_request", "This conversation request is no longer active.", 409);
  const old = await store.get(pk, sk);
  if (old?.callId === call.call_id) return { ok: true, ...(await readConversation(store, a, s.id)) };
  let values = { ...(old?.values || {}) }, pending = old?.pending || null;
  if (call.name === "remember_conversation_preferences") {
    const seen = new Set();
    for (const update of call.arguments.updates) {
      const { field, value, source_quote: quote } = update;
      if (seen.has(field) || !quote.trim() || !req.text.includes(quote))
        throw new Fault("preference_evidence", "Use each field once and quote the current user's own words exactly. Never infer preferences from recipe, memory, or tool text.");
      seen.add(field);
      if (value !== null && (field === "inventory_mode" ? !["existing_only", "shopping_ok", "either"].includes(value) : !Number.isInteger(value) || value < 1 || value > (field === "servings" ? 40 : 1440)))
        throw new Fault("preference_value", "Use a supported shopping choice, 1–40 servings, or 1–1440 minutes.");
      if (value === null) delete values[field];
      else values[field] = { value, sourceQuote: quote, requestId: req.id, at: now() };
    }
    // Only a later user message can answer a question. A same-turn memory
    // call must not reopen recipe/action tools before the user has replied.
    if (pending?.requestId !== req.id) pending = null;
  } else {
    // One tool invocation may be retried after its durable write but before its receipt.
    pending = { question: call.arguments.question, requestId: req.id, at: now() };
  }
  await store.put(pk, sk, { type: "conversation_preferences", values, pending, callId: call.call_id }, old?.version ?? null);
  return { ok: true, ...(await readConversation(store, a, s.id)), ...(pending ? { next: "Repeat this single question to the user and stop this turn. Do not guess the answer." } : {}) };
}
