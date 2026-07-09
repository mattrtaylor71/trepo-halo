import { buildChatTools, buildSystemPrompt } from "./realtime-config.mjs";
import { buildOuraGroundingMessage } from "./oura-context.mjs";
import {
  getKitchenItems,
  getMealPlan,
  getRecentDiscards,
  getRecentDishes,
  getRecipeSuggestions,
  getSavedRecipes,
  getShoppingItems
} from "./data-access.mjs";
import { executeToolAction } from "./tool-actions.mjs";
import { buildVoiceUiResponse } from "./ui-response.mjs";
import {
  geminiFallbackEnabled,
  isOpenAiOutageError,
  geminiChatFallbackResponse,
  geminiTranscribe
} from "./gemini-fallback.mjs";

const OPENAI_API_BASE = "https://api.openai.com/v1";
const DEFAULT_OPENAI_TIMEOUT_MS = 45000;
const DEFAULT_OPENAI_MAX_RETRIES = 2;
const DEFAULT_OPENAI_RETRY_BASE_MS = 1000;

function getOpenAiHeaders(env) {
  return {
    Authorization: `Bearer ${env.OPENAI_API_KEY}`,
    "Content-Type": "application/json"
  };
}

export function pcmToWavBuffer(audioBuffer, options = {}) {
  if (audioBuffer?.subarray?.(0, 4)?.toString("ascii") === "RIFF") {
    return audioBuffer;
  }

  const sampleRate = Number(options.sampleRate || 24000);
  const channels = Number(options.channels || 1);
  const bitsPerSample = Number(options.bitsPerSample || 16);
  const blockAlign = channels * bitsPerSample / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = audioBuffer.length;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(channels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  audioBuffer.copy(buffer, 44);

  return buffer;
}

function getTextContent(message) {
  if (!message) {
    return "";
  }

  if (typeof message.content === "string") {
    return message.content.trim();
  }

  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        return part?.text || "";
      })
      .join("")
      .trim();
  }

  return "";
}

function parseToolArgs(value) {
  if (!value) {
    return {};
  }

  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getOpenAiTimeoutMs(env) {
  const parsed = Number(env?.OPENAI_TIMEOUT_MS || DEFAULT_OPENAI_TIMEOUT_MS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_OPENAI_TIMEOUT_MS;
}

function getOpenAiMaxRetries(env) {
  const parsed = Number(env?.OPENAI_MAX_RETRIES || DEFAULT_OPENAI_MAX_RETRIES);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : DEFAULT_OPENAI_MAX_RETRIES;
}

function getOpenAiRetryBaseMs(env) {
  const parsed = Number(env?.OPENAI_RETRY_BASE_MS || DEFAULT_OPENAI_RETRY_BASE_MS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_OPENAI_RETRY_BASE_MS;
}

function isRetryableOpenAiStatus(status) {
  return status === 429 || status === 408 || status === 409 || status >= 500;
}

function getRetryDelayMs(response, attempt, env) {
  const retryAfterHeader = response?.headers?.get?.("retry-after");
  const retryAfterSeconds = Number(retryAfterHeader);
  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
    return retryAfterSeconds * 1000;
  }
  return getOpenAiRetryBaseMs(env) * (2 ** attempt);
}

async function openAiRequestPrimary(path, { env, method = "POST", headers = {}, body, expectJson = true, deadline = null }) {
  const maxRetries = getOpenAiMaxRetries(env);
  const configuredTimeoutMs = getOpenAiTimeoutMs(env);
  let lastError = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    // Deadline guard: bail out if not enough time remains for another attempt
    if (deadline) {
      const remaining = deadline - Date.now();
      if (remaining <= 5000) {
        console.warn("[WARN] OpenAI request skipping attempt due to deadline:", JSON.stringify({ path, attempt, remainingMs: remaining }));
        throw lastError || new Error(`OpenAI request aborted: Lambda deadline too close (${remaining}ms remaining)`);
      }
    }

    const timeoutMs = deadline
      ? Math.min(configuredTimeoutMs, deadline - Date.now() - 2000)
      : configuredTimeoutMs;
    if (timeoutMs <= 0) {
      console.warn("[WARN] OpenAI request computed non-positive timeout, aborting:", JSON.stringify({ path, attempt, timeoutMs }));
      throw lastError || new Error("OpenAI request aborted: no time remaining for request");
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${OPENAI_API_BASE}${path}`, {
        method,
        headers,
        body,
        signal: controller.signal
      });

      if (response.ok) {
        return expectJson ? response.json() : response;
      }

      const text = await response.text();
      if (attempt < maxRetries && isRetryableOpenAiStatus(response.status)) {
        const delayMs = getRetryDelayMs(response, attempt, env);
        console.warn("[WARN] OpenAI request retrying:", JSON.stringify({ path, status: response.status, attempt: attempt + 1, delayMs }));
        await sleep(delayMs);
        continue;
      }

      throw new Error(`OpenAI request failed with ${response.status}: ${text}`);
    } catch (error) {
      const isAbortError = error?.name === "AbortError";
      const retryableError = isAbortError || String(error?.message || "").includes("fetch failed");
      lastError = error;
      if (attempt < maxRetries && retryableError) {
        const delayMs = getOpenAiRetryBaseMs(env) * (2 ** attempt);
        console.warn("[WARN] OpenAI transport retrying:", JSON.stringify({ path, attempt: attempt + 1, delayMs, reason: error?.message || String(error) }));
        await sleep(delayMs);
        continue;
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  throw lastError || new Error("OpenAI request failed");
}

// Wraps the OpenAI call with a Gemini fallback for /chat/completions — covers BOTH the
// non-streaming caller (runDeviceAssistant → HALO) and the streaming caller
// (createChatCompletionStream → app Thyme), since both hit this path. On an OpenAI outage
// (429 insufficient_quota / 5xx / transport abort) the same request is retried against Gemini's
// OpenAI-compatible endpoint so Thyme keeps responding and tool-calls keep working. Dormant unless
// GEMINI_API_KEY is set. If Gemini also fails, the original OpenAI error is surfaced.
async function openAiRequest(path, opts) {
  try {
    return await openAiRequestPrimary(path, opts);
  } catch (error) {
    const { env, body, expectJson = true } = opts || {};
    if (path === "/chat/completions" && geminiFallbackEnabled(env) && isOpenAiOutageError(error)) {
      console.warn("[WARN] OpenAI chat failed — falling back to Gemini:", JSON.stringify({ reason: String(error?.message || "").slice(0, 140) }));
      try {
        const res = await geminiChatFallbackResponse(body, env);
        return expectJson ? res.json() : res;
      } catch (gemErr) {
        console.error("[ERROR] Gemini chat fallback also failed:", String(gemErr?.message || gemErr).slice(0, 200));
        throw error;
      }
    }
    throw error;
  }
}

const GROUNDING_STOP_WORDS = new Set([
  "a", "about", "all", "am", "an", "and", "any", "are", "at", "be", "can", "do", "for", "from",
  "get", "got", "had", "has", "have", "hey", "i", "im", "i'm", "if", "in", "into", "is", "it",
  "its", "it's", "know", "later", "like", "me", "my", "of", "on", "or", "please", "put", "recipe",
  "recipes", "shopping", "that", "the", "their", "them", "there", "these", "they", "this", "those",
  "to", "up", "use", "want", "was", "we", "with", "you", "your"
]);

const RECIPE_TOPIC_REGEX = /\b(recipe|recipes|ingredient|ingredients|cook|cooking|make|making|meal\s*plan|meal prep|breakfast|lunch|dinner|dessert|snack|instructions?|steps?|saved|imported|bookmarked|creami|tiktok|instagram|reel)\b/i;
const BROAD_FOOD_QUERY_REGEX = /\b(what\s+(have|did)\s+i\s+(eat|have)|what foods|what ingredients|what do i have|what(?:'s| is) in (?:the )?kitchen|what did i (?:discard|throw away)|what should i (?:eat|make)|what can i make)\b/i;
const AMBIGUOUS_REFERENCE_REGEX = /\b(that|this|those|these|it|one|ones)\b/i;
const RECIPE_READ_ONLY_QUESTION_REGEX = /\b(what do i need to buy|what should i buy|what do i need|what ingredients do i need|what(?:'s| is) missing|which ingredients|which groceries|what groceries|do i need to buy)\b/i;
const EXPLICIT_LIST_WRITE_REGEX = /\b(add|put|place|save|send|move|throw)\b[\s\w]{0,30}\b(on|onto|to|in)\b[\s\w]{0,20}\b(shopping|list)\b|\badd (?:them|those|it|these|ingredients|items)\b|\bput (?:them|those|it|these|ingredients|items)\b/i;

function normalizeGroundingText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function tokenizeGroundingText(value) {
  const seen = new Set();
  const tokens = [];
  for (const token of normalizeGroundingText(value).split(/\s+/)) {
    if (!token || token.length < 2 || GROUNDING_STOP_WORDS.has(token)) {
      continue;
    }
    if (seen.has(token)) {
      continue;
    }
    seen.add(token);
    tokens.push(token);
  }
  return tokens;
}

function buildSessionGroundingText(sessionMessages) {
  return (sessionMessages || [])
    .map((message) => String(message?.content || "").trim())
    .filter(Boolean)
    .join(" ");
}

function trimGroundingNote(value, maxLength = 100) {
  const text = String(value || "").trim().replace(/\s+/g, " ");
  if (!text) {
    return null;
  }
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function looksRecipeRelated(transcript, sessionMessages) {
  const combined = `${transcript || ""} ${buildSessionGroundingText(sessionMessages)}`;
  return RECIPE_TOPIC_REGEX.test(combined) || /https?:\/\/\S+/.test(String(transcript || ""));
}

function isBroadFoodQuestion(transcript) {
  return BROAD_FOOD_QUERY_REGEX.test(String(transcript || ""));
}

function isReadOnlyRecipeShoppingQuestion(transcript, sessionMessages) {
  const combined = `${transcript || ""} ${buildSessionGroundingText(sessionMessages)}`;
  return RECIPE_READ_ONLY_QUESTION_REGEX.test(combined) && !EXPLICIT_LIST_WRITE_REGEX.test(combined);
}

function buildGroundingCandidate({ label, searchParts = [], source, note = null, sourcePriority = 0, recency = 0 }) {
  return {
    label: String(label || "").trim(),
    searchText: normalizeGroundingText(searchParts.join(" ")),
    source: String(source || "").trim() || "household data",
    note: trimGroundingNote(note),
    sourcePriority,
    recency
  };
}

function scoreGroundingCandidate(candidate, transcript) {
  const normalizedTranscript = normalizeGroundingText(transcript);
  const transcriptTokens = tokenizeGroundingText(transcript);
  const normalizedLabel = normalizeGroundingText(candidate.label);
  const searchText = candidate.searchText || normalizedLabel;
  let score = 0;

  if (normalizedTranscript && normalizedLabel === normalizedTranscript) {
    score += 160;
  } else if (normalizedTranscript && normalizedTranscript.length >= 4 && normalizedLabel.includes(normalizedTranscript)) {
    score += 110;
  } else if (normalizedTranscript && normalizedLabel.length >= 4 && normalizedTranscript.includes(normalizedLabel)) {
    score += 90;
  }

  for (const token of transcriptTokens) {
    if (normalizedLabel === token) {
      score += 60;
      continue;
    }
    if (normalizedLabel.includes(token)) {
      score += token.length >= 5 ? 28 : 18;
      continue;
    }
    if (searchText.includes(token)) {
      score += token.length >= 5 ? 16 : 10;
    }
  }

  return score;
}

function pickGroundingCandidates(candidates, transcript, options = {}) {
  const allowFallback = Boolean(options.allowFallback);
  const minimumScore = Number.isFinite(Number(options.minimumScore)) ? Number(options.minimumScore) : 18;
  const limit = Number.isFinite(Number(options.limit)) ? Number(options.limit) : 5;
  const scored = (candidates || [])
    .filter((candidate) => candidate?.label)
    .map((candidate, index) => ({
      ...candidate,
      score: scoreGroundingCandidate(candidate, transcript),
      index
    }))
    .sort((left, right) => (
      right.score - left.score
      || left.sourcePriority - right.sourcePriority
      || left.recency - right.recency
      || left.index - right.index
    ));

  const filtered = allowFallback
    ? scored
    : scored.filter((candidate) => candidate.score >= minimumScore);

  const seen = new Set();
  const picks = [];
  for (const candidate of filtered) {
    const key = normalizeGroundingText(candidate.label);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    picks.push(candidate);
    if (picks.length >= limit) {
      break;
    }
  }
  return picks;
}

function formatGroundingCandidate(candidate) {
  return `- ${candidate.label} (${candidate.source}${candidate.note ? `; ${candidate.note}` : ""})`;
}

function summarizeKitchenGroundingArray(values, limit = 4) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const items = values
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .slice(0, limit);
  if (items.length === 0) {
    return null;
  }
  return items.join(", ");
}

function formatKitchenGroundingLine(item) {
  const parts = [item.item_name];
  if (item.id) {
    parts.push(`id ${item.id}`);
  }
  if (item.brand) {
    parts.push(`brand ${item.brand}`);
  }
  if (item.variant) {
    parts.push(`variant ${item.variant}`);
  }
  if (item.category) {
    parts.push(`category ${item.category}`);
  }
  if (item.state_label) {
    parts.push(item.state_label);
  }
  if (item.storage_location) {
    parts.push(`in ${item.storage_location}`);
  }
  if (item.expiration_date) {
    parts.push(`expires ${item.expiration_date}`);
  }
  if (item.estimated_price) {
    parts.push(`price ${item.estimated_price}`);
  }
  if (item.upf) {
    parts.push(`upf ${item.upf}`);
  }
  const harmful = summarizeKitchenGroundingArray(item.harmful_ingredients, 3);
  if (harmful) {
    parts.push(`harmful ${harmful}`);
  }
  const ingredients = summarizeKitchenGroundingArray(item.ingredients, 4);
  if (ingredients) {
    parts.push(`ingredients ${ingredients}`);
  }
  if (item.storage_guidance) {
    const sg = item.storage_guidance;
    if (sg.summary) {
      parts.push(`storage guidance: ${sg.summary}`);
    } else if (sg.min_days != null && sg.max_days != null) {
      parts.push(`keeps ${sg.min_days}-${sg.max_days} days (${sg.storage_zone || "mixed"})`);
    }
  }
  if (item.nutrition_summary) {
    parts.push(`nutrition: ${item.nutrition_summary}`);
  }
  return `- ${parts.join(" - ")}`;
}

async function buildShoppingGroundingMessage(userContext, env) {
  if (env?.ACTION_MODE !== "real" || !userContext?.ownerId) {
    return null;
  }

  try {
    const items = await getShoppingItems(userContext, { env, limit: 50 });
    if (!Array.isArray(items) || items.length === 0) {
      return null;
    }

    const itemLines = items
      .slice(0, 50)
      .map((item) => `- ${item.item_name}${item.store ? ` @ ${item.store}` : ""}`);

    return [
      "Current shopping list for grounding spoken item names:",
      ...itemLines,
      "If the transcript is slightly off, prefer the closest existing shopping-list item name.",
      "Do not split one spoken shopping phrase into multiple items unless the current list strongly supports that interpretation."
    ].join("\n");
  } catch (error) {
    console.warn("[WARN] shopping grounding preload failed:", error?.message || error);
    return null;
  }
}

async function buildKitchenGroundingMessage(userContext, env) {
  if (env?.ACTION_MODE !== "real" || !userContext?.ownerId) {
    return null;
  }

  try {
    const items = await getKitchenItems(userContext, { env, limit: 50 });
    if (!Array.isArray(items) || items.length === 0) {
      return null;
    }

    const itemLines = items
      .slice(0, 50)
      .map((item) => formatKitchenGroundingLine(item));

    return [
      "Current kitchen inventory for grounding food references, follow-up updates, and product-detail questions:",
      ...itemLines,
      "The kitchen data may include item_id, brand, variant, category, price, ingredients, nutrition_summary, harmful_ingredients, UPF, quantity, location, expiration, and storage_guidance (how long items typically keep, storage zone, and timing).",
      "If the user refers to food vaguely such as 'that rice', 'it', 'those', or a broad food name after a recent kitchen action, prefer the strongest current kitchen match.",
      "If multiple similar kitchen items match a broad reference, ask one short clarifying question and name the likely candidates."
    ].join("\n");
  } catch (error) {
    console.warn("[WARN] kitchen grounding preload failed:", error?.message || error);
    return null;
  }
}

async function buildFoodGroundingMessage(userContext, env, transcript) {
  if (env?.ACTION_MODE !== "real" || !userContext?.ownerId) {
    return null;
  }

  try {
    const [kitchenItems, recentDiscards, recentDishes] = await Promise.all([
      getKitchenItems(userContext, { env, limit: 12 }),
      getRecentDiscards(userContext, { env, limit: 8 }),
      getRecentDishes(userContext, { env, limit: 8 })
    ]);

    const candidates = [
      ...(Array.isArray(recentDishes?.items) ? recentDishes.items : []).map((dish, index) => buildGroundingCandidate({
        label: dish.dish_name,
        searchParts: [dish.dish_name, ...(Array.isArray(dish.ingredients) ? dish.ingredients : [])],
        source: "recent dish",
        note: Array.isArray(dish.ingredients) && dish.ingredients.length > 0
          ? `ingredients: ${dish.ingredients.slice(0, 4).join(", ")}`
          : null,
        sourcePriority: 0,
        recency: index
      })),
      ...(Array.isArray(kitchenItems) ? kitchenItems : []).map((item, index) => buildGroundingCandidate({
        label: item.item_name,
        searchParts: [
          item.item_name,
          item.brand,
          item.variant,
          item.category,
          item.estimated_price,
          item.nutrition_summary,
          ...(Array.isArray(item.ingredients) ? item.ingredients : []),
          ...(Array.isArray(item.harmful_ingredients) ? item.harmful_ingredients : [])
        ],
        source: "kitchen",
        note: [
          item.id ? `id ${item.id}` : null,
          item.brand || null,
          item.state_label || null,
          item.storage_location ? `in ${item.storage_location}` : null,
          item.estimated_price ? `price ${item.estimated_price}` : null,
          item.upf ? `upf ${item.upf}` : null,
          summarizeKitchenGroundingArray(item.harmful_ingredients, 2)
            ? `harmful ${summarizeKitchenGroundingArray(item.harmful_ingredients, 2)}`
            : null
        ].filter(Boolean).join("; ") || null,
        sourcePriority: 1,
        recency: index
      })),
      ...(Array.isArray(recentDiscards) ? recentDiscards : []).map((item, index) => buildGroundingCandidate({
        label: item.item_name,
        searchParts: [item.item_name, item.brand, item.category, item.discard_reason],
        source: "recent discard",
        note: item.discard_reason || item.brand || null,
        sourcePriority: 2,
        recency: index
      }))
    ];

    const shortlist = pickGroundingCandidates(candidates, transcript, {
      limit: 5,
      allowFallback: isBroadFoodQuestion(transcript) || AMBIGUOUS_REFERENCE_REGEX.test(String(transcript || ""))
    });

    if (shortlist.length === 0) {
      return null;
    }

    const strongestByMargin = shortlist[0]?.score > 0 && (!shortlist[1] || shortlist[0].score >= shortlist[1].score + 18);
    return [
      strongestByMargin ? "Likely current food reference for this turn:" : "Likely food references for this turn:",
      ...shortlist.map(formatGroundingCandidate),
      "Use this shortlist before sounding unsure. If one candidate is clearly strongest, phrase a confident confirmation like 'If you mean X...' and only ask a short clarifying question when multiple candidates remain plausible."
    ].join("\n");
  } catch (error) {
    console.warn("[WARN] food grounding preload failed:", error?.message || error);
    return null;
  }
}

async function buildRecipeGroundingMessage(userContext, env, transcript, sessionMessages) {
  if (env?.ACTION_MODE !== "real" || !userContext?.ownerId || !looksRecipeRelated(transcript, sessionMessages)) {
    return null;
  }

  try {
    const [savedRecipes, mealPlan, recipeSuggestions] = await Promise.all([
      getSavedRecipes(userContext, { env, limit: 8 }),
      getMealPlan(userContext, { env }),
      getRecipeSuggestions(userContext, { env, limit: 8 })
    ]);

    const mealPlanRecipes = (Array.isArray(mealPlan?.plan) ? mealPlan.plan : [])
      .filter((slot) => slot?.recipe?.title)
      .map((slot, index) => buildGroundingCandidate({
        label: slot.recipe.title,
        searchParts: [slot.recipe.title, ...(Array.isArray(slot.recipe.ingredients) ? slot.recipe.ingredients : [])],
        source: "meal plan",
        note: [slot.day_label, slot.meal_type].filter(Boolean).join(" "),
        sourcePriority: 0,
        recency: index
      }));

    const suggestionRecipes = [
      ...((Array.isArray(recipeSuggestions?.kitchen_only) ? recipeSuggestions.kitchen_only : []).map((recipe, index) => buildGroundingCandidate({
        label: recipe.title,
        searchParts: [recipe.title, ...(Array.isArray(recipe.ingredients) ? recipe.ingredients : []), ...(Array.isArray(recipe.missing_ingredients) ? recipe.missing_ingredients : [])],
        source: "recipe suggestion",
        note: "ready now",
        sourcePriority: 2,
        recency: index
      }))),
      ...((Array.isArray(recipeSuggestions?.need_grocery) ? recipeSuggestions.need_grocery : []).map((recipe, index) => buildGroundingCandidate({
        label: recipe.title,
        searchParts: [recipe.title, ...(Array.isArray(recipe.ingredients) ? recipe.ingredients : []), ...(Array.isArray(recipe.missing_ingredients) ? recipe.missing_ingredients : [])],
        source: "recipe suggestion",
        note: "needs groceries",
        sourcePriority: 3,
        recency: index
      })))
    ];

    const candidates = [
      ...(Array.isArray(savedRecipes?.recipes) ? savedRecipes.recipes : []).map((recipe, index) => buildGroundingCandidate({
        label: recipe.title,
        searchParts: [recipe.title, ...(Array.isArray(recipe.ingredients) ? recipe.ingredients : [])],
        source: "saved recipe",
        note: recipe.platform || recipe.author_name || null,
        sourcePriority: 1,
        recency: index
      })),
      ...mealPlanRecipes,
      ...suggestionRecipes
    ];

    const shortlist = pickGroundingCandidates(candidates, transcript, {
      limit: 6,
      allowFallback: true,
      minimumScore: 16
    });

    if (shortlist.length === 0) {
      return null;
    }

    const strongestByMargin = shortlist[0]?.score > 0 && (!shortlist[1] || shortlist[0].score >= shortlist[1].score + 18);
    return [
      strongestByMargin ? "Possibly related recipe in user data:" : "Possibly related recipes in user data:",
      ...shortlist.map(formatGroundingCandidate),
      "This shortlist is for context only — it shows what the user has saved or suggested. Do NOT recommend these unless the user is specifically asking about their saved recipes or suggestions. If the user is asking for a general recipe (e.g. 'give me a lemon ninja creami recipe'), answer from your own knowledge instead of redirecting to a partial match from this list."
    ].join("\n");
  } catch (error) {
    console.warn("[WARN] recipe grounding preload failed:", error?.message || error);
    return null;
  }
}

function buildReadOnlyIntentMessage(transcript, sessionMessages) {
  if (!isReadOnlyRecipeShoppingQuestion(transcript, sessionMessages)) {
    return null;
  }

  return [
    "This turn is a read-only recipe shopping question.",
    "The user is asking what they would need or what ingredients are missing, not asking you to modify the shopping list.",
    "Use read tools such as get_saved_recipe_detail or get_recipe_detail first, answer with the relevant ingredients or groceries, and do not call any write tool unless the user explicitly asks to add items."
  ].join("\n");
}

function normalizeSessionMessages(sessionMessages) {
  if (!Array.isArray(sessionMessages)) {
    return [];
  }

  return sessionMessages
    .map((message) => {
      const role = ["system", "assistant", "user"].includes(message?.role) ? message.role : null;
      const content = typeof message?.content === "string" ? message.content.trim() : "";
      if (!role || !content) {
        return null;
      }
      return { role, content };
    })
    .filter(Boolean);
}

async function createChatCompletion(messages, env, { deadline = null, toolChoice = "auto" } = {}) {
  return openAiRequest("/chat/completions", {
    env,
    headers: getOpenAiHeaders(env),
    body: JSON.stringify({
      model: env.OPENAI_MODEL || "gpt-5.4-2026-03-05",
      temperature: 0.2,
      messages,
      tools: buildChatTools(),
      tool_choice: toolChoice
    }),
    deadline
  });
}

// ── Intent detection for forced tool_choice ────────────────────────────
// When the transcript clearly expresses a write intent, force the model to
// call the appropriate tool instead of relying on tool_choice: "auto" which
// can hallucinate confirmations without actually executing the action.
const WRITE_INTENT_PATTERNS = [
  { pattern: /\b(add|put|throw|get|toss)\b.{0,30}\b(list|shopping|grocery)\b/i,       tool: "add_to_shopping_list" },
  { pattern: /\b(remove|delete|take off)\b.{0,30}\b(list|shopping|grocery)\b/i,        tool: "remove_from_shopping_list" },
  { pattern: /\b(clear|empty|wipe)\b.{0,20}\b(list|shopping|grocery)\b/i,              tool: "clear_shopping_list" },
  { pattern: /\b(check.?in|add.{0,10}kitchen|stock|restock)\b/i,                       tool: "check_in_item" },
  { pattern: /\b(discard|threw away|throw away|toss out|get rid of)\b/i,                tool: "discard_item" },
  { pattern: /\b(mark|it.s|it is).{0,10}\bopen/i,                                      tool: "mark_item_opened" },
  { pattern: /\b(log|had|ate|just ate|i ate|i had|eaten|for (breakfast|lunch|dinner|snack))\b/i, tool: "log_dish_from_voice" },
  { pattern: /\b(bought|picked up|got it|checked off)\b.{0,20}\b(list|shopping)?\b/i,  tool: "mark_shopping_item_bought" },
];

function detectWriteIntent(transcript) {
  for (const { pattern, tool } of WRITE_INTENT_PATTERNS) {
    if (pattern.test(transcript)) {
      return { type: "function", function: { name: tool } };
    }
  }
  return "auto";
}

// ── Hallucination guard ────────────────────────────────────────────────
// FALSE-CONFIRMATION GUARD. Detects when the model CLAIMS it performed a write
// action but did not actually call the tool this turn (a hallucinated "logged/
// added" confirmation). Table-driven claim -> expected-tool map so we can (a)
// tell the model exactly which tool to call on the corrective retry and (b) show
// an honest, domain-appropriate failure if it still won't. Patterns match BOTH
// word orders — "logged your lunch" AND "your lunch has been logged" (the latter
// slipped the old regex, which is how Matt's 07-06 lunch turn produced a green
// confirmation with no DB write).
const ACTION_CLAIM_RULES = [
  {
    domain: "dishes",
    expectedTool: "log_dish_ingredients",
    okTools: new Set(["log_dish_ingredients", "log_dish_from_voice", "update_recent_dish", "append_to_recent_dish", "delete_dish_log", "mark_dish_consumed"]),
    failText: "I wasn't able to log that just now — please try again.",
    patterns: [
      /\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack)\b[^.?!\n]{0,40}\blogged\b/i,
      /\blogged\b[^.?!\n]{0,40}\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack|it|that|your)\b/i,
      /\bi(?:'ve| have)\s+logged\b/i,
      /\b(has|have|had|is|was|were|been)\s+logged\b/i,
      /\badded\b[^.?!\n]{0,30}\bto\s+your\s+(?:dish|meal)\s+log\b/i,
    ],
  },
  {
    domain: "shopping_add",
    expectedTool: "add_to_shopping_list",
    okTools: new Set(["add_to_shopping_list", "add_many_to_shopping_list", "add_dish_ingredients_to_shopping_list", "add_recipe_ingredients_to_shopping_list", "add_saved_recipe_ingredients_to_shopping_list"]),
    failText: "I wasn't able to add that to your shopping list just now — please try again.",
    patterns: [
      /\badded\b[^.?!\n]{0,40}\bto\s+your\s+(?:shopping\s+)?(?:list|cart)\b/i,
      /\bput\b[^.?!\n]{0,30}\bon\s+your\s+(?:shopping\s+)?list\b/i,
    ],
  },
  {
    domain: "kitchen_add",
    expectedTool: "check_in_item",
    okTools: new Set(["check_in_item", "check_in_many_items"]),
    failText: "I wasn't able to add that to your kitchen just now — please try again.",
    patterns: [
      /\badded\b[^.?!\n]{0,40}\bto\s+your\s+(kitchen|pantry|fridge)\b/i,
      // "checked in" AND "checked into" (into is one token, so \bin\b fails there).
      /\bchecked\s+in(?:to)?\b[^.?!\n]{0,50}\b(kitchen|pantry|fridge)\b/i,
      /\bchecked[\s-]*in\b[^.?!\n]{0,30}\b(kitchen|pantry|fridge|item|it|that)\b/i,
    ],
  },
];

// Every write tool across the claim rules — used to detect whether ANY write
// succeeded this turn (recovery signal).
const WRITE_TOOL_NAMES = new Set(ACTION_CLAIM_RULES.flatMap((r) => [...r.okTools]));

// Returns the matching claim rule (domain + expected tool) or null.
function detectActionClaim(text) {
  const value = String(text || "");
  for (const rule of ACTION_CLAIM_RULES) {
    if (rule.patterns.some((pattern) => pattern.test(value))) {
      return rule;
    }
  }
  return null;
}

function looksLikeHallucinatedAction(text) {
  return detectActionClaim(text) !== null;
}

async function* createChatCompletionStream(messages, env, { deadline = null, toolChoice = "auto" } = {}) {
  const response = await openAiRequest("/chat/completions", {
    env,
    headers: getOpenAiHeaders(env),
    body: JSON.stringify({
      model: env.OPENAI_MODEL || "gpt-5.4-2026-03-05",
      temperature: 0.2,
      messages,
      tools: buildChatTools(),
      tool_choice: toolChoice,
      stream: true
    }),
    expectJson: false,
    deadline
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
        try {
          yield JSON.parse(trimmed.slice(6));
        } catch {
          // skip malformed chunks
        }
      }
    }
  }
}

export async function* runDeviceAssistantStreaming({ transcript, userContext, env, sessionMessages = [], responseSurface = "halo", lambdaDeadline = null }) {
  const deadline = lambdaDeadline || (Date.now() + 140000);
  const normalizedResponseSurface = String(responseSurface || "halo").trim().toLowerCase() === "app" ? "app" : "halo";
  const normalizedSessionMessages = normalizeSessionMessages(sessionMessages);

  const [
    shoppingGroundingMessage,
    kitchenGroundingMessage,
    foodGroundingMessage,
    recipeGroundingMessage,
    ouraGroundingMessage
  ] = await Promise.all([
    buildShoppingGroundingMessage(userContext, env),
    buildKitchenGroundingMessage(userContext, env),
    buildFoodGroundingMessage(userContext, env, transcript),
    buildRecipeGroundingMessage(userContext, env, transcript, normalizedSessionMessages),
    buildOuraGroundingMessage(userContext, env)
  ]);
  const readOnlyIntentMessage = buildReadOnlyIntentMessage(transcript, normalizedSessionMessages);
  const messages = [
    { role: "system", content: buildSystemPrompt(userContext, { responseSurface: normalizedResponseSurface }) },
    ...(shoppingGroundingMessage ? [{ role: "system", content: shoppingGroundingMessage }] : []),
    ...(kitchenGroundingMessage ? [{ role: "system", content: kitchenGroundingMessage }] : []),
    ...(foodGroundingMessage ? [{ role: "system", content: foodGroundingMessage }] : []),
    ...(recipeGroundingMessage ? [{ role: "system", content: recipeGroundingMessage }] : []),
    ...(ouraGroundingMessage ? [{ role: "system", content: ouraGroundingMessage }] : []),
    ...(readOnlyIntentMessage ? [{ role: "system", content: readOnlyIntentMessage }] : []),
    ...(normalizedSessionMessages.length > 0
      ? [{
        role: "system",
        content: "Use the recent conversation for follow-up context like clarifications, pronouns, and references such as 'that', 'it', or broad food names. Always use tools and current household data as the source of truth."
      }]
      : []),
    ...normalizedSessionMessages,
    { role: "user", content: transcript }
  ];
  const toolTrace = [];
  const toolEvents = [];
  const quickItems = [];
  let fullText = "";
  let falseConfirmCorrections = 0;
  let falseConfirmRecovered = false;
  // When the false-confirmation guard fires, the corrective retry FORCES a tool
  // call (tool_choice: "required") — "auto" let the model re-claim without calling
  // the tool (kitchen_add on 07-09: retry produced text again in ~1s, recovered:false).
  let forceToolChoiceNext = null;

  for (let turn = 0; turn < 6; turn += 1) {
    const turnToolChoice = forceToolChoiceNext || "auto";
    forceToolChoiceNext = null;
    const stream = createChatCompletionStream(messages, env, { deadline, toolChoice: turnToolChoice });
    let assembledContent = "";
    const assembledToolCalls = [];

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;

      // Accumulate text content and yield deltas
      if (delta.content) {
        assembledContent += delta.content;
        yield { type: "text_delta", delta: delta.content };
      }

      // Accumulate tool call deltas
      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? assembledToolCalls.length;
          if (!assembledToolCalls[idx]) {
            assembledToolCalls[idx] = {
              id: tc.id || "",
              type: "function",
              function: { name: tc.function?.name || "", arguments: "" }
            };
          }
          if (tc.id) assembledToolCalls[idx].id = tc.id;
          if (tc.function?.name) assembledToolCalls[idx].function.name = tc.function.name;
          if (tc.function?.arguments) assembledToolCalls[idx].function.arguments += tc.function.arguments;
        }
      }
    }

    // Use latest non-empty text (matches sync handler behavior).
    // Concatenating across turns doubles text when the model repeats
    // a preface after tool execution.
    if (assembledContent) {
      fullText = assembledContent;
    }

    // No tool calls — assistant is done
    if (assembledToolCalls.length === 0) {
      // FALSE-CONFIRMATION GUARD: the model claims a write action but the tool
      // for that claim did not succeed this turn. Correct once; if it still
      // won't call the tool, replace the confirmation with an honest failure —
      // NEVER let an unbacked confirmation reach the user.
      const claim = detectActionClaim(fullText);
      const backed = claim && toolTrace.some((t) => t.ok && claim.okTools.has(t.toolName));
      if (claim && !backed) {
        if (falseConfirmCorrections < 1) {
          falseConfirmCorrections += 1;
          console.log(JSON.stringify({ evt: "assistant_false_confirm", tool: claim.expectedTool, domain: claim.domain, userId: userContext?.userId || null, recovered: null, phase: "retrying", reason: "claim_without_tool" }));
          messages.push({ role: "assistant", content: fullText });
          messages.push({
            role: "user",
            content: `You stated the action was completed but you did not call ${claim.expectedTool}. Call it now with the details from the user's message, then confirm. If you genuinely cannot do it, say so honestly instead of claiming success.`
          });
          // Force a tool call on the retry so the model can't just re-claim in text.
          forceToolChoiceNext = "required";
          fullText = "";
          yield { type: "text_delta", delta: "\n" };
          continue;
        }
        // Retry already happened and it STILL produced an unbacked claim. Record WHY
        // so we're not guessing next time: did the domain tool run and fail, or was
        // it never called at all?
        const domainToolAttempted = toolTrace.some((t) => claim.okTools.has(t.toolName));
        const reason = domainToolAttempted ? "retry_tool_call_failed" : "retry_no_tool_call";
        console.log(JSON.stringify({ evt: "assistant_false_confirm", tool: claim.expectedTool, domain: claim.domain, userId: userContext?.userId || null, recovered: false, reason }));
        fullText = claim.failText;
        yield { type: "text_delta", delta: `\n${claim.failText}` };
      }

      const uiResponse = buildVoiceUiResponse({
        text: fullText || "Okay.",
        quickItems,
        toolEvents,
        responseSurface: normalizedResponseSurface
      });
      return {
        text: fullText || "Okay.",
        quickItems,
        toolTrace,
        toolEvents,
        version: uiResponse.version,
        type: uiResponse.type,
        ui: uiResponse.ui
      };
    }

    // Execute tool calls
    messages.push({
      role: "assistant",
      content: assembledContent || "",
      tool_calls: assembledToolCalls
    });

    for (const toolCall of assembledToolCalls) {
      const toolName = toolCall.function.name;
      const args = parseToolArgs(toolCall.function.arguments);
      yield { type: "tool_start", tool_name: toolName, args };

      const result = await executeToolAction({ toolName, args, env, userContext, responseSurface: normalizedResponseSurface });

      if (toolName === "add_to_shopping_list" && result?.ok && result?.args?.item_name) {
        quickItems.push(result.args.item_name);
      }
      if (toolName === "add_many_to_shopping_list" && result?.ok && Array.isArray(result?.args?.items)) {
        quickItems.push(...result.args.items.map((item) => item.item_name).filter(Boolean));
      }

      toolEvents.push({ toolName, args: result?.args || args, result, quickItems: [...quickItems] });
      toolTrace.push({
        toolName,
        args: result?.args || args,
        ok: Boolean(result?.ok),
        statusCode: result?.statusCode || 200,
        actionSummary: result?.actionSummary || null,
        error: result?.error || null
      });

      yield {
        type: "tool_end",
        tool_name: toolName,
        ok: Boolean(result?.ok),
        summary: result?.actionSummary || null
      };

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(result)
      });
    }

    // If a corrective retry (false-confirmation guard) produced a successful
    // write, record the recovery for the corpus/metric.
    if (falseConfirmCorrections > 0 && !falseConfirmRecovered && toolTrace.some((t) => t.ok && WRITE_TOOL_NAMES.has(t.toolName))) {
      falseConfirmRecovered = true;
      console.log(JSON.stringify({ evt: "assistant_false_confirm", userId: userContext?.userId || null, recovered: true, reason: "corrective_tool_succeeded" }));
    }

    // If the model produced no visible text this turn (went straight to tools),
    // emit a brief synthetic status so the user isn't waiting in silence
    if (!assembledContent && turn === 0) {
      yield { type: "text_delta", delta: "One moment..." };
    }
  }

  // Exhausted turns
  const uiResponse = buildVoiceUiResponse({
    text: fullText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolEvents
  });
  return {
    text: fullText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolTrace,
    toolEvents,
    version: uiResponse.version,
    type: uiResponse.type,
    ui: uiResponse.ui
  };
}

function validateAudioQuality(audioBuffer) {
  if (!audioBuffer || audioBuffer.length < 3200) {
    return { ok: false, reason: "audio_too_short" };
  }
  const sampleCount = Math.floor(audioBuffer.length / 2);
  const windowSize = 1600; // 100ms at 16kHz
  const windowCount = Math.floor(sampleCount / windowSize);
  if (windowCount < 2) {
    return { ok: true };
  }

  const rmsValues = [];
  for (let w = 0; w < windowCount; w++) {
    let sumSq = 0;
    const offset = w * windowSize;
    for (let i = 0; i < windowSize; i++) {
      const sample = audioBuffer.readInt16LE((offset + i) * 2);
      sumSq += sample * sample;
    }
    rmsValues.push(Math.sqrt(sumSq / windowSize));
  }

  const meanRms = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length;
  const rmsStd = Math.sqrt(rmsValues.reduce((a, r) => a + (r - meanRms) ** 2, 0) / rmsValues.length);
  const rmsVariation = meanRms > 0 ? rmsStd / meanRms : 0;

  console.log("[AUDIO_QC] windows=" + windowCount + " meanRms=" + meanRms.toFixed(0) +
    " rmsStd=" + rmsStd.toFixed(1) + " variation=" + (rmsVariation * 100).toFixed(1) + "%");

  // Speech has energy variation >5%. Constant-energy noise has <2%.
  if (rmsVariation < 0.02 && meanRms > 5000) {
    return { ok: false, reason: "constant_noise", meanRms: Math.round(meanRms), variation: rmsVariation };
  }
  return { ok: true };
}

export async function transcribeAudio(audioBuffer, env, options = {}) {
  const qc = validateAudioQuality(audioBuffer);
  if (!qc.ok) {
    console.warn("[AUDIO_QC] rejected:", JSON.stringify(qc));
    return "";
  }

  const wavBuffer = pcmToWavBuffer(audioBuffer, {
    sampleRate: Number(options.sampleRate || env.AUDIO_SAMPLE_RATE || 24000),
    channels: 1,
    bitsPerSample: 16
  });

  const form = new FormData();
  form.append("model", env.TRANSCRIPTION_MODEL || "gpt-4o-mini-transcribe");
  form.append("language", env.TRANSCRIPTION_LANGUAGE || "en");
  form.append("response_format", "json");
  form.append("file", new Blob([wavBuffer], { type: "audio/wav" }), "audio.wav");

  let json;
  try {
    json = await openAiRequest("/audio/transcriptions", {
      env,
      headers: {
        Authorization: `Bearer ${env.OPENAI_API_KEY}`
      },
      body: form
    });
  } catch (error) {
    // OpenAI transcription down (429/5xx/abort) → fall back to native Gemini so HALO/app voice
    // still gets transcribed. Needs a durable AIza GEMINI_API_KEY (native rejects AQ. tokens).
    if (geminiFallbackEnabled(env) && isOpenAiOutageError(error)) {
      console.warn("[WARN] OpenAI transcription failed — falling back to Gemini:", String(error?.message || "").slice(0, 140));
      return await geminiTranscribe(wavBuffer, env);
    }
    throw error;
  }
  return String(json?.text || "").trim();
}

export async function runDeviceAssistant({ transcript, userContext, env, sessionMessages = [], responseSurface = "halo", lambdaDeadline = null }) {
  const deadline = lambdaDeadline || (Date.now() + 140000);
  const normalizedResponseSurface = String(responseSurface || "halo").trim().toLowerCase() === "app" ? "app" : "halo";
  const normalizedSessionMessages = normalizeSessionMessages(sessionMessages);
  console.log("[DEBUG] assistant turn start:", JSON.stringify({
    transcript,
    ownerId: userContext?.ownerId || null,
    userId: userContext?.userId || null,
    tableOwnerId: userContext?.tableOwnerId || null,
    shoppingNamespace: userContext?.shoppingNamespace || null,
    sessionMessageCount: normalizedSessionMessages.length,
    responseSurface: normalizedResponseSurface
  }));

  const [
    shoppingGroundingMessage,
    kitchenGroundingMessage,
    foodGroundingMessage,
    recipeGroundingMessage,
    ouraGroundingMessage
  ] = await Promise.all([
    buildShoppingGroundingMessage(userContext, env),
    buildKitchenGroundingMessage(userContext, env),
    buildFoodGroundingMessage(userContext, env, transcript),
    buildRecipeGroundingMessage(userContext, env, transcript, normalizedSessionMessages),
    buildOuraGroundingMessage(userContext, env)
  ]);
  const readOnlyIntentMessage = buildReadOnlyIntentMessage(transcript, normalizedSessionMessages);
  const messages = [
    { role: "system", content: buildSystemPrompt(userContext, { responseSurface: normalizedResponseSurface }) },
    ...(shoppingGroundingMessage ? [{ role: "system", content: shoppingGroundingMessage }] : []),
    ...(kitchenGroundingMessage ? [{ role: "system", content: kitchenGroundingMessage }] : []),
    ...(foodGroundingMessage ? [{ role: "system", content: foodGroundingMessage }] : []),
    ...(recipeGroundingMessage ? [{ role: "system", content: recipeGroundingMessage }] : []),
    ...(ouraGroundingMessage ? [{ role: "system", content: ouraGroundingMessage }] : []),
    ...(readOnlyIntentMessage ? [{ role: "system", content: readOnlyIntentMessage }] : []),
    ...(normalizedSessionMessages.length > 0
      ? [{
        role: "system",
        content: "Use the recent conversation for follow-up context like clarifications, pronouns, and references such as 'that', 'it', or broad food names. Always use tools and current household data as the source of truth."
      }]
      : []),
    ...normalizedSessionMessages,
    { role: "user", content: transcript }
  ];
  const toolTrace = [];
  const toolEvents = [];
  const quickItems = [];
  let lastAssistantText = "";
  let falseConfirmCorrections = 0;
  let falseConfirmRecovered = false;
  // The false-confirmation corrective retry FORCES a tool call (see streaming path).
  let forceToolChoiceNext = null;

  // On the first turn, detect write intent and force tool_choice if appropriate
  const initialToolChoice = detectWriteIntent(transcript);
  if (initialToolChoice !== "auto") {
    console.log("[DEBUG] forcing tool_choice on first turn:", JSON.stringify(initialToolChoice));
  }

  for (let turn = 0; turn < 6; turn += 1) {
    // Force a tool on a corrective retry; else force on the first turn if write-intent; else auto.
    const turnToolChoice = forceToolChoiceNext || (turn === 0 ? initialToolChoice : "auto");
    forceToolChoiceNext = null;
    const completion = await createChatCompletion(messages, env, { deadline, toolChoice: turnToolChoice });
    const message = completion?.choices?.[0]?.message;

    if (!message) {
      break;
    }

    lastAssistantText = getTextContent(message) || lastAssistantText;

    if (!Array.isArray(message.tool_calls) || message.tool_calls.length === 0) {
      // FALSE-CONFIRMATION GUARD (see runDeviceAssistantStreaming for the full
      // rationale): claim of a write action whose tool did not succeed this turn.
      // Correct once, then honest failure — never surface an unbacked confirmation.
      const claim = detectActionClaim(lastAssistantText);
      const backed = claim && toolTrace.some((t) => t.ok && claim.okTools.has(t.toolName));
      if (claim && !backed) {
        if (falseConfirmCorrections < 1) {
          falseConfirmCorrections += 1;
          console.log(JSON.stringify({ evt: "assistant_false_confirm", tool: claim.expectedTool, domain: claim.domain, userId: userContext?.userId || null, recovered: null, phase: "retrying", reason: "claim_without_tool" }));
          messages.push({ role: "assistant", content: lastAssistantText });
          messages.push({
            role: "user",
            content: `You stated the action was completed but you did not call ${claim.expectedTool}. Call it now with the details from the user's message, then confirm. If you genuinely cannot do it, say so honestly instead of claiming success.`
          });
          forceToolChoiceNext = "required"; // make the retry actually call the tool
          continue; // retry this turn
        }
        const domainToolAttempted = toolTrace.some((t) => claim.okTools.has(t.toolName));
        const reason = domainToolAttempted ? "retry_tool_call_failed" : "retry_no_tool_call";
        console.log(JSON.stringify({ evt: "assistant_false_confirm", tool: claim.expectedTool, domain: claim.domain, userId: userContext?.userId || null, recovered: false, reason }));
        lastAssistantText = claim.failText;
      }

      const uiResponse = buildVoiceUiResponse({
        text: lastAssistantText || "Okay.",
        quickItems,
        toolEvents,
        responseSurface: normalizedResponseSurface
      });

      console.log("[DEBUG] assistant final response:", JSON.stringify({
        text: lastAssistantText || "Okay.",
        type: uiResponse.type,
        quickItems,
        toolCount: toolTrace.length
      }));

      return {
        text: lastAssistantText || "Okay.",
        quickItems,
        toolTrace,
        toolEvents,
        version: uiResponse.version,
        type: uiResponse.type,
        ui: uiResponse.ui
      };
    }

    messages.push({
      role: "assistant",
      content: message.content || "",
      tool_calls: message.tool_calls
    });

    const toolResults = await Promise.all(
      message.tool_calls.map(async (toolCall) => {
        const toolName = toolCall?.function?.name;
        const args = parseToolArgs(toolCall?.function?.arguments);
        console.log("[DEBUG] assistant requested tool:", JSON.stringify({ toolName, args }));
        const result = await executeToolAction({
          toolName,
          args,
          env,
          userContext,
          responseSurface: normalizedResponseSurface
        });
        return { toolCall, toolName, args, result };
      })
    );

    for (const { toolCall, toolName, args, result } of toolResults) {
      if (toolName === "add_to_shopping_list" && result?.ok && result?.args?.item_name) {
        quickItems.push(result.args.item_name);
      }
      if (toolName === "add_many_to_shopping_list" && result?.ok && Array.isArray(result?.args?.items)) {
        quickItems.push(...result.args.items.map((item) => item.item_name).filter(Boolean));
      }

      toolEvents.push({
        toolName,
        args: result?.args || args,
        result,
        quickItems: [...quickItems]
      });

      toolTrace.push({
        toolName,
        args: result?.args || args,
        ok: Boolean(result?.ok),
        statusCode: result?.statusCode || 200,
        actionSummary: result?.actionSummary || null,
        error: result?.error || null
      });

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(result)
      });
    }

    // Corrective retry produced a successful write -> record the recovery.
    if (falseConfirmCorrections > 0 && !falseConfirmRecovered && toolTrace.some((t) => t.ok && WRITE_TOOL_NAMES.has(t.toolName))) {
      falseConfirmRecovered = true;
      console.log(JSON.stringify({ evt: "assistant_false_confirm", userId: userContext?.userId || null, recovered: true, reason: "corrective_tool_succeeded" }));
    }
  }

  const uiResponse = buildVoiceUiResponse({
    text: lastAssistantText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolEvents
  });

  console.log("[DEBUG] assistant exhausted turns:", JSON.stringify({
    text: lastAssistantText || "I heard you, but I couldn't finish that request.",
    type: uiResponse.type,
    quickItems,
    toolCount: toolTrace.length
  }));

  return {
    text: lastAssistantText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolTrace,
    toolEvents,
    version: uiResponse.version,
    type: uiResponse.type,
    ui: uiResponse.ui
  };
}
