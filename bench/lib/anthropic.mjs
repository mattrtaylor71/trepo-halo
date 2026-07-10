// Blind judge — claude-opus-4-8 via the official Anthropic SDK. Returns parsed JSON
// scores. Used to score recipe sets and home-suggestion sets WITHOUT knowing which
// OpenAI model produced them (sets are shuffled + anonymized by the caller).
import Anthropic from "@anthropic-ai/sdk";
import { ANTHROPIC_KEY } from "./keys.mjs";

const client = new Anthropic({ apiKey: ANTHROPIC_KEY });

export async function judge({ system, user, maxTokens = 2000 }) {
  const t0 = Date.now();
  const msg = await client.messages.create({
    model: "claude-opus-4-8",
    max_tokens: maxTokens,
    system,
    messages: [{ role: "user", content: user }],
  });
  const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  return { text, json: extractJson(text), latencyMs: Date.now() - t0, usage: msg.usage };
}

function extractJson(text) {
  // tolerate ```json fences or leading prose
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : text.slice(text.indexOf("{") >= 0 ? text.indexOf("{") : 0);
  try { return JSON.parse(candidate); } catch {}
  try { return JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1)); } catch {}
  return null;
}
