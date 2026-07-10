// Minimal OpenAI Chat Completions caller for the benchmark. Raw HTTP (fetch) so the
// harness has no SDK-version coupling and can call every candidate model uniformly.
// Handles tools, vision (image_url), and JSON response formats, with graceful
// parameter fallback for newer models that reject temperature/max_tokens.
import { OPENAI_KEY } from "./keys.mjs";

const ENDPOINT = "https://api.openai.com/v1/chat/completions";

export async function chat({ model, messages, tools = null, tool_choice = null,
  temperature = null, response_format = null, max_tokens = null }) {
  const body = { model, messages };
  if (tools) body.tools = tools;
  if (tool_choice) body.tool_choice = tool_choice;
  if (temperature != null) body.temperature = temperature;
  if (response_format) body.response_format = response_format;
  if (max_tokens != null) body.max_completion_tokens = max_tokens;

  let attempt = 0;
  const t0 = Date.now();
  while (true) {
    const r = await fetch(ENDPOINT, {
      method: "POST",
      headers: { Authorization: `Bearer ${OPENAI_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (r.ok) {
      const latencyMs = Date.now() - t0;
      const msg = j.choices?.[0]?.message || {};
      return {
        ok: true, model, latencyMs,
        reasoningOff: body.reasoning_effort === "none",
        text: msg.content || "",
        toolCalls: (msg.tool_calls || []).map((tc) => ({
          name: tc.function?.name,
          args: safeParse(tc.function?.arguments),
          rawArgs: tc.function?.arguments,
        })),
        finish: j.choices?.[0]?.finish_reason,
        usage: j.usage || {},
        raw: j,
      };
    }
    // Graceful param fallback: some newer models reject a custom temperature or the
    // max_tokens field name. Strip the offending param and retry (max 3 adjustments).
    const emsg = (j.error?.message || "").toLowerCase();
    if (attempt < 3) {
      if (emsg.includes("temperature") && "temperature" in body) { delete body.temperature; attempt++; continue; }
      if (emsg.includes("max_completion_tokens") || (emsg.includes("max_tokens") && emsg.includes("unsupported"))) {
        delete body.max_completion_tokens; attempt++; continue;
      }
      if (emsg.includes("'max_tokens'") && body.max_completion_tokens != null) {
        body.max_tokens = body.max_completion_tokens; delete body.max_completion_tokens; attempt++; continue;
      }
      // Some reasoning models (e.g. gpt-5.6-sol) reject function tools in
      // chat/completions unless reasoning is off. Retry with reasoning_effort:'none'
      // (benchmarked reasoning-OFF — flagged in the caller/report).
      if (emsg.includes("reasoning_effort") && body.reasoning_effort !== "none") {
        body.reasoning_effort = "none"; attempt++; continue;
      }
    }
    return { ok: false, model, latencyMs: Date.now() - t0, error: `${r.status}: ${j.error?.message || ""}`, usage: {}, raw: j };
  }
}

function safeParse(s) { try { return JSON.parse(s); } catch { return { __unparsed: s }; } }

// A user content array with an image (data URI) + text, for vision calls.
export function imageContent(dataUri, text, detail = "auto") {
  return [
    { type: "text", text },
    { type: "image_url", image_url: { url: dataUri, detail } },
  ];
}
