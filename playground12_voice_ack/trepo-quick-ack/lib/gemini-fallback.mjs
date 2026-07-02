// Gemini fallback for when OpenAI is unavailable (429 insufficient_quota, 5xx, timeout).
// All Trepo voice/chat OpenAI calls funnel through openAiRequest() in device-assistant.mjs;
// chat + streaming both hit /chat/completions, transcription hits /audio/transcriptions.
//
// Chat/streaming fall back to Google's OpenAI-COMPATIBLE endpoint (same request/response shape,
// tools + SSE streaming supported) so callers are unchanged. Transcription falls back to the
// native Gemini generateContent API with inline audio.
//
// Dormant unless env.GEMINI_API_KEY is set. Use a durable AI-Studio API key ("AIza…"); ephemeral
// Live-API tokens ("AQ.…") work but expire in ~30 min.

const GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai";
const GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta";

export function geminiFallbackEnabled(env) {
  return Boolean(env && env.GEMINI_API_KEY);
}

// True for the failures worth failing over on: OpenAI quota exhaustion, 5xx, or transport aborts.
export function isOpenAiOutageError(error) {
  const msg = String(error?.message || "");
  if (error?.name === "AbortError") return true;
  if (msg.includes("fetch failed")) return true;
  if (msg.includes("insufficient_quota")) return true;
  return /failed with (429|500|502|503|504)\b/.test(msg);
}

// Chat completions via the Gemini OpenAI-compatible endpoint. `bodyString` is the exact
// OpenAI request body the caller built; we only swap the model. Returns a fetch Response so both
// expectJson (response.json()) and streaming (response.body reader) callers work identically.
export async function geminiChatFallbackResponse(bodyString, env) {
  const model = env.GEMINI_FALLBACK_MODEL || "gemini-2.5-flash-lite";
  let payload;
  try { payload = JSON.parse(bodyString); } catch { payload = {}; }
  payload.model = model;
  // Gemini's compat layer ignores unknown OpenAI-only fields; keep messages/tools/tool_choice/stream.
  const res = await fetch(`${GEMINI_OPENAI_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GEMINI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gemini chat fallback failed with ${res.status}: ${text}`);
  }
  return res;
}

// Native Gemini transcription of a WAV buffer → plain text. Needs a durable AIza key (the native
// generateContent endpoint rejects ephemeral AQ. tokens).
export async function geminiTranscribe(wavBuffer, env) {
  const model = env.GEMINI_TRANSCRIBE_MODEL || "gemini-2.5-flash";
  const b64 = Buffer.from(wavBuffer).toString("base64");
  const res = await fetch(`${GEMINI_NATIVE_BASE}/models/${model}:generateContent`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GEMINI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      contents: [{
        parts: [
          { text: "Transcribe the following audio to plain text. Return ONLY the verbatim transcription with no preamble, quotes, or commentary." },
          { inlineData: { mimeType: "audio/wav", data: b64 } },
        ],
      }],
      generationConfig: { temperature: 0 },
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gemini transcription failed with ${res.status}: ${text}`);
  }
  const json = await res.json();
  const parts = json?.candidates?.[0]?.content?.parts || [];
  return parts.map((p) => p?.text || "").join("").trim();
}
