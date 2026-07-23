import { buildChatTools, buildSystemPrompt } from "./realtime-config.mjs";
import { buildOuraGroundingMessage } from "./oura-context.mjs";
import {
  getDietaryPreferences,
  formatDietaryPreferencesBlock,
  getKitchenItems,
  getKitchenItemsFull,
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

// ── Per-LLM-call telemetry (ai_op markers) ─────────────────────────────────
// Additive observability: emit one single-line JSON record per LLM call so we
// can track op/model/latency/status across chat, transcription, and the Gemini
// fallbacks. Emission is fully wrapped so telemetry can NEVER break a request,
// and we never log raw audio bytes/base64 — only counts and transcript text.
function emitAiOp(rec) {
  try {
    console.log(JSON.stringify({ evt: "ai_op", service: "thyme", ...rec }));
  } catch {
    // Telemetry is best-effort; swallow any serialization/logging error.
  }
}

// owner_id resolution order per marker spec: userContext.userId || ownerId || tableOwnerId.
function ownerIdForMarker(userContext) {
  return userContext?.userId || userContext?.ownerId || userContext?.tableOwnerId || null;
}

function truncateForMarker(value, max) {
  const text = typeof value === "string" ? value : value == null ? "" : String(value);
  return text.length > max ? text.slice(0, max) : text;
}

// output = final assistant text + the tool names invoked this turn (from the
// toolTrace/toolEvents), capped to the marker's 1200-char budget.
function summarizeAiOutput(finalText, toolTrace) {
  const tools = Array.isArray(toolTrace)
    ? toolTrace.map((entry) => entry?.toolName).filter(Boolean)
    : [];
  const toolPart = tools.length ? ` [tools: ${tools.join(", ")}]` : "";
  return truncateForMarker(`${finalText || ""}${toolPart}`, 1200);
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
      const fbStart = Date.now();
      const fbModel = env.GEMINI_FALLBACK_MODEL || "gemini-2.5-flash-lite";
      let fbInput = "";
      try {
        const parsed = JSON.parse(body);
        const lastUser = Array.isArray(parsed?.messages)
          ? [...parsed.messages].reverse().find((m) => m?.role === "user")
          : null;
        fbInput = truncateForMarker(typeof lastUser?.content === "string" ? lastUser.content : "", 400);
      } catch {
        // body may not be JSON-parseable; leave input empty (never log audio).
      }
      try {
        const res = await geminiChatFallbackResponse(body, env);
        emitAiOp({ op: "chat_fallback_gemini", owner_id: null, model: fbModel, latency_ms: Date.now() - fbStart, status: "success", input: fbInput, output: "gemini chat fallback response", error: "" });
        return expectJson ? res.json() : res;
      } catch (gemErr) {
        emitAiOp({ op: "chat_fallback_gemini", owner_id: null, model: fbModel, latency_ms: Date.now() - fbStart, status: "error", input: fbInput, output: "", error: truncateForMarker(gemErr?.message || String(gemErr), 500) });
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

// ── App how-to guide (env-versioned via PROMPT_GUIDE_VERSION) ───────────────
// Describes the REAL Trepo UI so Thyme can answer app-usage questions from fact,
// not invention. Rollback = set PROMPT_GUIDE_VERSION=off (block omitted). Every
// label below is quoted from the shipped iOS UI.
const APP_GUIDE_V1 = `TREPO APP GUIDE — use ONLY these facts to answer app how-to / "where do I…" questions.
GUIDE RULES: (1) Answer app how-to questions ONLY from the facts below. If something isn't covered here, say you're not certain and encourage the user to reach out for help — they can email matt@trepo.ai or text Matt and Zach at +1 415 987 7809 — never invent a screen or button. (2) For the user's own live data (their Household ID, who's in their household, what's in their kitchen), use your tools/context — never make up an ID or names. (3) When the user ASKS how/where/whether to do something ("how do I…", "where is…", "can I…"), just EXPLAIN the steps — describe where to tap; do NOT call a tool or perform the action, and NEVER say you did something you didn't do. Only perform an action when the user clearly commands it ("add milk", "log my breakfast").

NAVIGATION: five bottom tabs — "List" (shopping list), "Kitchen" (your inventory), a center "+" button (add items), "Cook" (recipes), "Dish Log" (nutrition). Tapping "+" opens the "Check into your kitchen" menu with four choices: Leftovers, Fridge/Pantry, Receipt, Text.

ADD ITEMS TO KITCHEN:
- Photo scan: tap "+" → "Fridge/Pantry" → photograph your groceries/fridge → a review screen shows "Identified Items" and a "Needs Review" section → tap the pencil to fix a name/brand, confirm or dismiss uncertain ones → tap "Add N items to kitchen".
- Leftovers: "+" → "Leftovers" → photo (optional note).
- Receipt: "+" → "Receipt" → photograph the receipt, or tap "Upload" to pick from your photo library.
- No photo: "+" → "Text" → type items separated by commas or new lines → "Add".
- Fix a mis-identified item: on the review screen tap the pencil on that row; for an item already in the kitchen, open it and use "Fix this item".
- There is no grocery-delivery-service import; to add a delivery order, photograph its receipt/confirmation via "Receipt" or type the items via "Text".
- I (Thyme) can also add items by voice or text — just tell me what to add.

KITCHEN TAB: items are grouped into sections — Produce, Dairy & Eggs, Meat & Seafood, Pantry, Snacks & Sweets, Beverages, Prepared & Other, Leftovers — assigned automatically. There's no manual re-grouping; if a category is wrong, open the item and use "Fix this item". Open an item to set "Stored in" (Fridge/Freezer/Pantry), set "Expires" (Set expiry date), or tap "Discard Item" to remove it. "Kitchen IQ" is a score of how well-stocked/healthy your kitchen is, shown on the home screen.

SHOPPING LIST ("List" tab): type in "Add an item…" or hold the microphone to speak; tap an item's checkbox to check it off; tap the X to remove it. Items group under store headings — use "Add New Group" to add a store. "Clear checked" clears checked items. In a household, everyone shares one list in real time.

DISH LOG ("Dish Log" tab, page titled "Health"): tap "Log a dish" to photograph a meal, or log by voice/text through me. Each dish shows Calories, Protein, Carbs, and Fat, with a "Your daily nutrition" summary. Your meal plan also lives here; if it's empty it says "Check more groceries in to unlock your personalized meal plan" — so check in more groceries.

RECIPES ("Cook" tab): "Explore" = trending recipes from creators; "Use what I have" = recipes from your current kitchen ("Make with what you have" vs "Need a few more things"); "My Recipes" = your saved recipes. To import a recipe from Instagram, TikTok, or a website, tap Share in that app and choose "Trepo" — it saves automatically. In a saved recipe, tap "Edit" to change it or "Remove from saved recipes" to delete it; "Add missing to list" puts missing ingredients on your shopping list.

HOUSEHOLD & SHARING (open your Profile, then the Account/Household section): your "Household ID" is shown there and can be copied — share it so others can join you. Tap "Invite to Household" to text someone your ID. To join a household, enter its ID in the Household section and tap the arrow (this replaces your shopping list with the household's shared list). "Leave Household" leaves it. Everyone in a household shares the same kitchen, shopping list, and recipes in real time. If a partner can't see your items, make sure you are both in the SAME household (same Household ID) — there is no separate members list; the shared Household ID is what links you.

ACCOUNT (Profile): tap "Enable notifications" to turn on reminders. For feedback or help, email matt@trepo.ai. "Log Out" signs out; "Delete Account" permanently deletes your data.

WHAT I (THYME) CAN DO: add or remove kitchen items, manage your shopping list, log dishes, and suggest recipes — by voice or text. For app settings or account actions, I'll tell you where to tap.`;

// v2 (2026-07-22): brings the guide current with shipped features the v1 guide predated —
// Recall check, recipe sharing, saved-recipe categories/tags, the Meal Plan calendar + notes,
// "I made this", the header buttons, and the Cook-tab cards. v1's verified wording is kept
// verbatim (it passes the appguide matrix); the new material uses labels quoted from the live UI.
const APP_GUIDE_V2 = `TREPO APP GUIDE — use ONLY these facts to answer app how-to / "where do I…" questions.
GUIDE RULES: (1) Answer app how-to questions ONLY from the facts below. If something isn't covered here, say you're not certain and encourage the user to reach out for help — they can email matt@trepo.ai or text Matt and Zach at +1 415 987 7809 — never invent a screen or button. (2) For the user's own live data (their Household ID, who's in their household, what's in their kitchen), use your tools/context — never make up an ID or names. (3) When the user ASKS how/where/whether to do something ("how do I…", "where is…", "can I…"), just EXPLAIN the steps — describe where to tap; do NOT call a tool or perform the action, and NEVER say you did something you didn't do. Only perform an action when the user clearly commands it ("add milk", "log my breakfast").

NAVIGATION: four bottom tabs — "List" (shopping list), "Kitchen" (your inventory), "Cook" (recipes), "Dish Log" (nutrition) — plus a center "+" button that opens the "Check into your kitchen" menu (four choices: Leftovers, Fridge/Pantry, Receipt, Text). At the top of the home screen are round buttons: "Profile", "Help", and a megaphone that opens "Recall check"; plus an "Ask Thyme" button (that's me).

ADD ITEMS TO KITCHEN:
- Photo scan: tap "+" → "Fridge/Pantry" → photograph your groceries/fridge → a review screen shows "Identified Items" and a "Needs Review" section → tap the pencil to fix a name/brand, confirm or dismiss uncertain ones → tap "Add N items to kitchen".
- Leftovers: "+" → "Leftovers" → photo (optional note).
- Receipt: "+" → "Receipt" → photograph the receipt, or tap "Upload" to pick from your photo library.
- No photo: "+" → "Text" → type items separated by commas or new lines → "Add".
- Fix a mis-identified item: on the review screen tap the pencil on that row; for an item already in the kitchen, open it and use "Fix this item".
- There is no grocery-delivery-service import; to add a delivery order, photograph its receipt/confirmation via "Receipt" or type the items via "Text".
- I (Thyme) can also add items by voice or text — just tell me what to add.

KITCHEN TAB: items are grouped into sections — Produce, Dairy & Eggs, Meat & Seafood, Pantry, Snacks & Sweets, Beverages, Prepared & Other, Leftovers — assigned automatically. There's no manual re-grouping; if a category is wrong, open the item and use "Fix this item". Open an item to set "Stored in" (Fridge/Freezer/Pantry), set "Expires" (Set expiry date), add it to your shopping list, use it in a recipe, or tap "Discard Item" to remove it. "Kitchen IQ" is a score of how well-stocked/healthy your kitchen is, shown on the home screen.

SHOPPING LIST ("List" tab): type in "Add an item…" or hold the microphone to speak; tap an item's checkbox to check it off; tap the X to remove it. A "Suggested for you" row offers quick add ideas. Items group under store headings — use "Add New Group" to add a store. "Clear checked" clears checked items. In a household, everyone shares one list in real time.

DISH LOG ("Dish Log" tab, page titled "Health"): tap "Log a dish" to photograph a meal, or log by voice/text through me. Each dish shows Calories, Protein, Carbs, and Fat, with a "Your daily nutrition" summary. Your meal plan also lives under the Cook tab; if it's empty check more groceries in. "I made this": open a saved recipe and tap "I made this" to mark its ingredients as used — that removes those ingredients from your kitchen.

RECIPES ("Cook" tab): cards for "Explore" (trending recipes from creators), "Use what I have" (recipes from your current kitchen — "Make with what you have" vs "Need a few more things"), "My Recipes" (your saved recipes), "Meal Plan" (plan your week), and "Save a Recipe" (add by Link, Text, or Photos). To import a recipe from Instagram, TikTok, or a website, tap Share in that app and choose "Trepo" — it saves automatically. On a recipe, the bookmark saves it (it turns solid blue when saved); open a saved recipe to "Edit" it or "Remove from saved recipes" to delete it; "Add missing to list" puts missing ingredients on your shopping list.

SHARE A RECIPE: on a recipe, tap the share icon to create a link you can send to anyone. They open the recipe on the web and tap "Save in Trepo" — it opens the recipe in the app if they have it, or sends them to install it with the recipe waiting.

RECIPE CATEGORIES / TAGS: in "My Recipes" you can create your own categories (like "Meal prep") — tap the "＋ New" chip to make one. Tap the tag icon on a recipe to assign it to categories, and tap a category chip at the top to filter your saved recipes to just that category.

MEAL PLAN ("Meal Plan" card in the Cook tab): a weekly calendar. Pick a day, then under "Breakfast", "Lunch", "Dinner", or "Snack" tap "Add a … recipe" to add a saved, kitchen, or explore recipe to that slot; each entry shows "Ready to cook" or "+N to buy", and the X removes it. Each day also has a "Notes" section — tap "Add a note" for free-text notes.

RECALL CHECK (the megaphone button at the top of the home screen): tap "Check my kitchen" and it scans your kitchen against current recalls and outbreak alerts from the FDA, USDA, and CDC. Results show closer matches and softer "might not be yours" ones; tap a match's status icon to learn what its certainty means, and "View recall" / "Search for this recall" to read the official notice. It flags by the type of food only (not brand, lot, or specific product), so always confirm with the official notice. "You're all clear" means nothing in your kitchen matched.

HOUSEHOLD & SHARING (open your Profile, then the Account/Household section): your "Household ID" is shown there and can be copied — share it so others can join you. Tap "Invite to Household" to text someone your ID. To join a household, enter its ID in the Household section and tap the arrow (this replaces your shopping list with the household's shared list). "Leave Household" leaves it. Everyone in a household shares the same kitchen, shopping list, and recipes in real time; in "My Recipes" a "MY OWN | HOUSEHOLD" slider switches between your recipes and the household's shared ones. If a partner can't see your items, make sure you are both in the SAME household (same Household ID) — there is no separate members list; the shared Household ID is what links you.

ACCOUNT (Profile): tap "Enable notifications" to turn on reminders. "Spread the word" copies your App Store link so you can share Trepo. For feedback or help, email matt@trepo.ai. "Log Out" signs out; "Delete Account" permanently deletes your data.

WHAT I (THYME) CAN DO: add or remove kitchen items, manage your shopping list, log dishes, and suggest recipes — by voice or text. For app settings or account actions, I'll tell you where to tap.`;

const APP_GUIDES = { v1: APP_GUIDE_V1, v2: APP_GUIDE_V2 };

// Returns the app-guide system message(s), or [] when PROMPT_GUIDE_VERSION=off.
// Default is v2 (current app); rollback = set PROMPT_GUIDE_VERSION=v1 (or off).
function getAppGuideMessages() {
  const version = String(process.env.PROMPT_GUIDE_VERSION || "v2").trim().toLowerCase();
  if (version === "off" || version === "none" || version === "") return [];
  return [{ role: "system", content: APP_GUIDES[version] || APP_GUIDE_V1 }];
}
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

// Most-recent items kept with full enrichment in the kitchen grounding
// message. Every item beyond this head is still listed, but compactly.
const KITCHEN_GROUNDING_DETAILED_HEAD = 40;

// Compact one-liner for a kitchen item: "name (category, location)". Keeps the
// full kitchen visible to Thyme without dumping per-item enrichment (a ~250
// item tail is only ~2.5k input tokens this way).
function formatCompactKitchenGroundingLine(item) {
  const name = item.item_name || "Unknown item";
  const attrs = [item.category, item.storage_location].filter(Boolean);
  return attrs.length > 0 ? `${name} (${attrs.join(", ")})` : name;
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
    // Read the WHOLE kitchen so Thyme is aware of every item. The most-recent
    // head keeps full enrichment; the remaining tail is listed compactly so a
    // large kitchen is never partially invisible (fixes the 25-item cap that
    // made Thyme suggest pantry-only / repetitive meals and claim it could only
    // see part of the kitchen).
    const items = await getKitchenItemsFull(userContext, { env });
    if (!Array.isArray(items) || items.length === 0) {
      return null;
    }

    const totalCount = items.length;
    const detailedItems = items.slice(0, KITCHEN_GROUNDING_DETAILED_HEAD);
    const remainingItems = items.slice(KITCHEN_GROUNDING_DETAILED_HEAD);

    const detailedLines = detailedItems.map((item) => formatKitchenGroundingLine(item));
    const compactLines = remainingItems.map((item) => `- ${formatCompactKitchenGroundingLine(item)}`);

    const lines = [
      `Current kitchen inventory (${totalCount} item${totalCount === 1 ? "" : "s"} total) for grounding food references, follow-up updates, meal ideas, and product-detail questions. This is your COMPLETE kitchen; you can see every item.`,
      `Detailed items (${detailedItems.length} most-recent, with full enrichment):`,
      ...detailedLines
    ];

    if (compactLines.length > 0) {
      lines.push(
        `Remaining ${compactLines.length} kitchen items (compact - name (category, location)); these are just as present, only listed with less detail:`,
        ...compactLines
      );
    }

    lines.push(
      "The detailed kitchen data may include item_id, brand, variant, category, price, ingredients, nutrition_summary, harmful_ingredients, UPF, quantity, location, expiration, and storage_guidance (how long items typically keep, storage zone, and timing).",
      "Treat both the detailed and compact lists as fully visible inventory. Never say you can only see part of the kitchen, and draw on fridge and pantry items across the whole list (not just the most-recent ones) when suggesting meals or answering what the user has.",
      "If the user refers to food vaguely such as 'that rice', 'it', 'those', or a broad food name after a recent kitchen action, prefer the strongest current kitchen match.",
      "If multiple similar kitchen items match a broad reference, ask one short clarifying question and name the likely candidates."
    );

    return lines.join("\n");
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
  { pattern: /\b(remove|removed|take out|took out|pull|use up|used up)\b.{0,30}\b(kitchen|pantry|fridge|freezer)\b/i, tool: "discard_item" },
  { pattern: /\b(mark|it.s|it is).{0,10}\bopen/i,                                      tool: "mark_item_opened" },
  // Meal-calendar commands (planning to cook LATER) must win over the dish-log
  // "for <slot>" pattern below — an add/put/schedule verb here means schedule, not
  // "I ate". A genuine "I had X for dinner" uses had/ate (no add verb) → dish log.
  { pattern: /\b(add|put|schedule|slot|pencil)\b.{0,40}\b(calendar|meal[-\s]?plan|meal[-\s]?calendar)\b/i, tool: "add_recipe_to_meal_calendar" },
  { pattern: /\b(add|put|schedule|slot|pencil)\b.{0,40}\b(?:for|to)\s+(breakfast|lunch|dinner|snack)\b/i,   tool: "add_recipe_to_meal_calendar" },
  { pattern: /\b(move|reschedule|shift|bump)\b.{0,50}\b(calendar|to\s+(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next|this|\d{4}-\d{2}-\d{2}))\b/i, tool: "move_meal_calendar_entry" },
  { pattern: /\b(remove|take\s+off|unschedule)\b.{0,40}\b(calendar|meal[-\s]?plan|meal[-\s]?calendar)\b/i,   tool: "remove_meal_calendar_entry" },
  { pattern: /\b(log|had|ate|just ate|i ate|i had|eaten|for (breakfast|lunch|dinner|snack))\b/i, tool: "log_dish_from_voice" },
  { pattern: /\b(bought|picked up|got it|checked off)\b.{0,20}\b(list|shopping)?\b/i,  tool: "mark_shopping_item_bought" },
];

// How-to / informational questions must NOT be force-routed to a write tool:
// "how do I add to my shopping list" is a QUESTION, not the command "add milk".
// Without this, App-Guide how-to questions silently trigger a placeholder write.
// A genuine command (no interrogative frame) still forces its tool below.
const HOWTO_INTENT_REGEX = /\b(?:how\s+(?:do|can|would|should|to)|how\s+d(?:o|oes)\b[^.?!]*\bi\b|where\s+(?:do|can|is|are|'?s)|where'?s|what\s+(?:is|are|can|'?s)|what'?s|why\s+(?:is|are|does|do|can|no)|can\s+i|could\s+i|do\s+you|are\s+you|is\s+there|is\s+it\s+possible)\b/i;

function detectWriteIntent(transcript) {
  if (HOWTO_INTENT_REGEX.test(String(transcript || ""))) return "auto";
  // Consumption + an explicit REMOVAL ("I just had strawberries, can you remove them",
  // "finished the milk, take it out") is a KITCHEN discard, not a dish log. The
  // had/ate → log_dish pattern below would otherwise HARD-force log_dish and the
  // removal never happens (the strawberries bug). Do NOT force a tool here — hand the
  // choice to the model + the system prompt's discard rule (LLM-first). Shopping-list
  // and calendar removals have their own patterns above, so exclude those surfaces.
  const wantsRemoval = /\b(remove|removing|get(?:ting)?\s+rid\s+of|take\s+(?:it|them|those|these|that)\s+out|took\s+(?:it|them|those|these|that)\s+out|toss\s+(?:it|them|those|these|that)|throw\s+(?:it|them|those|these|that)\s+(?:out|away)|discard)\b/i.test(String(transcript || ""))
    && !/\b(list|shopping|cart|grocer|calendar|meal[-\s]?plan)\b/i.test(String(transcript || ""));
  for (const { pattern, tool } of WRITE_INTENT_PATTERNS) {
    // Don't let "had/ate" hard-force a dish log when they're clearly asking to remove it.
    if (tool === "log_dish_from_voice" && wantsRemoval) continue;
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
    // Meal-calendar confirmations. Placed FIRST so a "scheduled/added ... to your
    // calendar" reply resolves to THIS domain (satisfied by a calendar write) and
    // never falls through to the dishes rule (which would demand a dish-log tool
    // and flag a real calendar action as a false confirmation). Planning a meal is
    // NOT eating one. Patterns are PAST-TENSE only, so capability answers ("I can
    // add things to your calendar") never match.
    domain: "meal_calendar",
    expectedTool: "add_recipe_to_meal_calendar",
    okTools: new Set([
      "add_recipe_to_meal_calendar",
      "add_many_to_meal_calendar",
      "add_generated_recipes_to_meal_calendar",
      "move_meal_calendar_entry",
      "remove_meal_calendar_entry"
    ]),
    failText: "I couldn't update your meal calendar just now — please try again, or check the recipe title and date.",
    patterns: [
      /\b(added|scheduled|planned|booked|slotted|penciled|pencilled)\b[^.?!\n]{0,60}\b(calendar|meal\s+plan|meal\s+calendar)\b/i,
      /\b(added|scheduled|planned|booked|slotted|put)\b[^.?!\n]{0,60}\b(?:for|to)\s+(?:your\s+)?(?:breakfast|lunch|dinner|snack)\b\s+(?:on\s+|for\s+)?(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|next|this|\d{4}-\d{2}-\d{2})/i,
      /\b(moved|rescheduled|shifted|bumped)\b[^.?!\n]{0,60}\b(?:to|on)\s+(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2}|next|this|tomorrow)\b/i,
      /\b(removed|took\s+off|taken\s+off|cleared|deleted|unscheduled)\b[^.?!\n]{0,60}\b(?:from\s+)?(?:your\s+)?(calendar|meal\s+plan|meal\s+calendar)\b/i,
    ],
  },
  {
    domain: "dishes",
    expectedTool: "log_dish_ingredients",
    okTools: new Set(["log_dish_ingredients", "log_dish_from_voice", "update_recent_dish", "append_to_recent_dish", "delete_dish_log", "mark_dish_consumed"]),
    failText: "I wasn't able to log that just now — please try again.",
    patterns: [
      /\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack)\b[^.?!\n]{0,40}\blogged\b/i,
      /\blogged\b[^.?!\n]{0,40}\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack|it|that|your)\b/i,
      /\b(has|have|had|is|was|were|been)\s+logged\b/i,
      /\badded\b[^.?!\n]{0,30}\bto\s+your\s+(?:dish|meal)\s+log\b/i,
      // Sentence-initial completed-action confirmations that name the food
      // DIRECTLY ("Logged: two waffles") — the shape that slipped Brad's 07-09
      // turn (the meal-word patterns above require dish/meal/food/breakfast/etc.,
      // which "Eggo waffles"/"espresso" are not). Excludes "logged out/in/up" and
      // instructional/offer forms ("you can log a dish", "want me to log…") which
      // use present-tense "log" — so it coexists with the App Guide how-to text.
      /^\s*(?:(?:ok(?:ay)?|done|great|sure|perfect|got\s*it|all\s*set|no\s*problem|there|alright)[\s,!.—-]+)*(?:i(?:'ve|\s+have|\s+just)?\s+)?(logged|added|saved|recorded|noted)\b(?!\s+(?:out|in|up|off)\b)(?:\s*[:\-—]|\s+(?:your|the|a|an|two|one|both|it\b|that\b|\d))/i,
      /\bi(?:'ve|\s+have|\s+just)?\s+logged\b(?!\s+(?:out|in)\b)/i,
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
  {
    // Removal claims were UNGUARDED — a "removed/discarded it" reply with no
    // discard_item call (e.g. a name-match miss that threw not-found) sailed
    // through as a green confirmation. That is why 0/3 of Zach's week-long
    // removal requests landed while Thyme confirmed every one.
    domain: "kitchen_remove",
    expectedTool: "discard_item",
    okTools: new Set(["discard_item", "clear_kitchen_inventory"]),
    failText: "I couldn't find that to remove — it may not be in your kitchen, or try naming it the way it's saved.",
    // The negative lookahead keeps SHOPPING-list removals ("removed X from your
    // shopping list") from matching the kitchen-remove rule — that domain uses a
    // different tool (remove_from_shopping_list) and has no rule of its own here.
    patterns: [
      /\b(removed|discarded|tossed|took\s+out|taken\s+out|threw\s+(?:it\s+)?(?:out|away)|thrown\s+(?:it\s+)?(?:out|away)|got(?:ten)?\s+rid\s+of|used\s+up)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,45}\b(kitchen|pantry|fridge|freezer|inventory|stock)\b/i,
      /\bi(?:'ve| have)\s+(removed|discarded|tossed|taken\s+out|thrown\s+(?:it\s+)?(?:out|away)|gotten\s+rid\s+of)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)/i,
      /\b(removed|discarded|tossed|took\s+out|threw\s+(?:it\s+)?(?:out|away)|got\s+rid\s+of)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,30}\b(it|that|them|those|the|your)\b/i,
      /\b(has|have|been)\s+(removed|discarded|taken\s+out|tossed)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,30}\b(kitchen|pantry|fridge|from|it|that)\b/i,
    ],
  },
];

// Every write tool across the claim rules — used to detect whether ANY write
// succeeded this turn (recovery signal).
const WRITE_TOOL_NAMES = new Set(ACTION_CLAIM_RULES.flatMap((r) => [...r.okTools]));

// ── Guard-leak scrub (F-106) ───────────────────────────────────────────
// The dish-log suppression guard is silent-correct: it skips a phantom dish-log
// but the PRIMARY action (kitchen add / list add / discard) still succeeds.
// Bug: the model narrated the guard's internal reasoning verbatim at the user —
// tool names and "I can't honestly …" — even though the real action worked
// (c2ed3a8d, 854f0d88, 3ed72da8). This is a DETERMINISTIC backstop: when the
// turn has a successful PRIMARY write AND the final narration leaks a tool-name
// token or guard-justification phrasing, we replace the narration with a clean
// confirmation naming exactly the items that were actually written. The prompt
// rule (realtime-config buildSystemPrompt) is the first line of defense; this is
// the belt-and-suspenders that never lets the leak reach the user.

// Primary write tools whose success should be confirmed to the user by item
// name when the model's own narration is unusable (leaked). Dish-log tools are
// deliberately excluded — a suppressed dish log is NOT a primary action.
const PRIMARY_WRITE_TOOLS = new Set([
  "check_in_item",
  "check_in_many_items",
  "add_to_shopping_list",
  "add_many_to_shopping_list",
  "discard_item",
]);

// Tokens that must never appear in user-facing narration — internal tool names
// plus the guard's justification phrasing. Presence of any of these alongside a
// successful primary write means the narration leaked the guard's reasoning.
const GUARD_LEAK_TOKENS = [
  "log_dish_ingredients",
  "log_dish_from_voice",
  "update_recent_dish",
  "append_to_recent_dish",
  "mark_dish_consumed",
  "save_generated_recipe",
  "save_recipe_from_tiktok",
  "check_in_item",
  "check_in_many_items",
  "discard_item",
  "dish_log_suppressed",
  "dish-log action",
  "dish-log tool",
  "skip_dishlog",
];
const GUARD_LEAK_PHRASES = [
  /\bcan'?t\s+honestly\b/i,
  /\bcannot\s+honestly\b/i,
  /\bi\s+did\s+already\s+complete\b/i,
  /\bwithout\s+making\s+something\s+up\b/i,
  /\bno\s+(?:recent\s+)?dish\s+details\b/i,
  /\bnot\s+to\s+log\s+something\s+they\s+ate\b/i,
  /\bshouldn'?t\s+invent\s+a\s+meal\b/i,
];

// True when narration leaked an internal tool name or guard-justification phrase.
function narrationLeaksGuardReasoning(text) {
  const value = String(text || "");
  if (!value) return false;
  const lower = value.toLowerCase();
  if (GUARD_LEAK_TOKENS.some((tok) => lower.includes(tok.toLowerCase()))) return true;
  return GUARD_LEAK_PHRASES.some((re) => re.test(value));
}

// Pull the human item names out of a successful primary-write tool-trace entry.
function primaryWriteItemNames(entry) {
  if (!entry || !entry.ok || !PRIMARY_WRITE_TOOLS.has(entry.toolName)) return [];
  const args = entry.args || {};
  const names = [];
  if (Array.isArray(args.items)) {
    for (const it of args.items) {
      const n = it && typeof it === "object" ? it.item_name : it;
      if (n && String(n).trim()) names.push(String(n).trim());
    }
  }
  if (args.item_name && String(args.item_name).trim()) names.push(String(args.item_name).trim());
  return names;
}

// Build a clean, user-facing confirmation from the successful primary writes in
// the turn. Returns null when there is no successful primary write to confirm
// (so the caller leaves the model's narration untouched). Phrasing mirrors the
// good replies already in the corpus ("Added to your kitchen: …"). No em dashes.
function buildPrimaryWriteConfirmation(toolTrace) {
  const trace = Array.isArray(toolTrace) ? toolTrace : [];
  const kitchenNames = [];
  const listNames = [];
  const discardNames = [];
  for (const entry of trace) {
    if (!entry || !entry.ok) continue;
    const names = primaryWriteItemNames(entry);
    if (names.length === 0) continue;
    if (entry.toolName === "check_in_item" || entry.toolName === "check_in_many_items") kitchenNames.push(...names);
    else if (entry.toolName === "add_to_shopping_list" || entry.toolName === "add_many_to_shopping_list") listNames.push(...names);
    else if (entry.toolName === "discard_item") discardNames.push(...names);
  }
  const uniq = (arr) => [...new Set(arr)];
  const sections = [];
  if (kitchenNames.length) sections.push(`Added to your kitchen:\n${uniq(kitchenNames).map((n) => `- ${n}`).join("\n")}`);
  if (listNames.length) sections.push(`Added to your shopping list:\n${uniq(listNames).map((n) => `- ${n}`).join("\n")}`);
  if (discardNames.length) sections.push(`Removed from your kitchen:\n${uniq(discardNames).map((n) => `- ${n}`).join("\n")}`);
  if (sections.length === 0) return null;
  return sections.join("\n\n");
}

// F-106 deterministic scrub: if the final narration leaked guard reasoning AND a
// primary write succeeded this turn, replace the narration with a clean
// confirmation of what actually happened. Returns the (possibly rewritten) text.
export function scrubGuardLeakFromNarration(text, toolTrace) {
  if (!narrationLeaksGuardReasoning(text)) return text;
  const clean = buildPrimaryWriteConfirmation(toolTrace);
  // Only rewrite when we have a real successful action to confirm. If nothing
  // primary succeeded, leave the text alone — the false-confirmation guard and
  // honest-failure paths own that case; we must never invent a success.
  return clean || text;
}

// ── Recipe-save vs dish-log mutual-exclusion guard ─────────────────────
// "Save a recipe (to cook later)" and "log a dish (that I ate)" are DIFFERENT
// actions. The model sometimes treats "save <food>" as BOTH — emitting
// save_generated_recipe (correct) AND a dish-log write (log_dish_ingredients)
// in the same turn, fabricating a phantom "you ate this" record with invented
// nutrition, then narrating only the dish log. (07-14: "Save turkey fried rice"
// → recipe saved ✅ + phantom Turkey Fried Rice dish ❌.) Recipe-save intent
// WINS: whenever a recipe-save tool is present in the request, suppress every
// dish-log write. A legitimate "I ate X" turn does NOT call a save tool, so it
// is untouched (see the control test). delete_dish_log is intentionally NOT in
// this set — suppressing a delete could strand a bad row.
const RECIPE_SAVE_TOOLS = new Set(["save_generated_recipe", "save_recipe_from_tiktok"]);
const DISH_LOG_WRITE_TOOLS = new Set([
  "log_dish_ingredients",
  "log_dish_from_voice",
  "update_recent_dish",
  "append_to_recent_dish",
  "mark_dish_consumed",
]);

// True when a batch of tool calls contains a recipe-save (which then wins over
// any dish-log write in the same request).
export function batchHasRecipeSaveTool(toolNames) {
  return (Array.isArray(toolNames) ? toolNames : []).some((n) => RECIPE_SAVE_TOOLS.has(n));
}

// Decide whether a single dish-log write must be suppressed. Suppress a dish-log
// write iff a recipe-save is present in this same turn (batchHasRecipeSave) or
// already happened earlier in the request (recipeSaveIssued). Non-dish-log tools
// (including the recipe save itself) are never suppressed.
export function shouldSuppressDishLog(toolName, { batchHasRecipeSave = false, recipeSaveIssued = false } = {}) {
  return DISH_LOG_WRITE_TOOLS.has(toolName) && (batchHasRecipeSave || recipeSaveIssued);
}

// Synthetic tool result handed back to the model in place of the suppressed
// dish-log write. Kept SHORT and instructional (not user-facing prose) so the
// model does not quote it verbatim at the user (F-106: the old wordy note was
// being narrated word-for-word, tool name and all). The user-facing reply must
// confirm ONLY the successful primary action; never mention this suppression.
function suppressedDishLogResult() {
  return {
    ok: false,
    suppressed: true,
    statusCode: 409,
    actionSummary: null,
    error: "dish_log_suppressed_recipe_save",
    note: "skip_dishlog:confirm_primary_action_only",
  };
}

// Returns the matching claim rule (domain + expected tool) or null.
export function detectActionClaim(text) {
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
  // Household dietary prefs (safety-critical: allergies) injected into the system prompt so
  // Thyme honors them in suggestions and can state them when asked. Best-effort; a fetch
  // failure / no prefs yields an empty block = unchanged behavior.
  const dietaryBlock = formatDietaryPreferencesBlock(await getDietaryPreferences(userContext, { env }));
  const messages = [
    { role: "system", content: buildSystemPrompt(userContext, { responseSurface: normalizedResponseSurface, dietaryBlock }) },
    ...getAppGuideMessages(),
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
  // ai_op telemetry: streaming path serves the app surface → op "chat_app";
  // HALO responses through this path fall back to "chat_halo". Latency spans the
  // whole completion loop. Model is the OpenAI chat model actually requested.
  const aiOpStart = Date.now();
  const aiOp = normalizedResponseSurface === "app" ? "chat_app" : "chat_halo";
  const aiModel = env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
  let falseConfirmCorrections = 0;
  let falseConfirmRecovered = false;
  // Set once a recipe-save tool is seen in the request; suppresses dish-log
  // writes for the rest of the request (mutual-exclusion guard).
  let recipeSaveIssued = false;
  // When the false-confirmation guard fires, the corrective retry FORCES a tool
  // call (tool_choice: "required") — "auto" let the model re-claim without calling
  // the tool (kitchen_add on 07-09: retry produced text again in ~1s, recovered:false).
  let forceToolChoiceNext = null;

  try {
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

      // F-106 guard-leak scrub: if the narration leaked the dish-log guard's
      // internal reasoning (tool names / "can't honestly") but a primary write
      // actually succeeded, replace it with a clean confirmation of what happened.
      // Mirrors the false-confirm streaming pattern: overwrite fullText and yield
      // the corrected text as an appended delta so the client's final text is clean.
      const scrubbed = scrubGuardLeakFromNarration(fullText, toolTrace);
      if (scrubbed !== fullText) {
        console.log(JSON.stringify({ evt: "assistant_guard_leak_scrubbed", userId: userContext?.userId || null, surface: "streaming" }));
        fullText = scrubbed;
        yield { type: "text_delta", delta: `\n${scrubbed}` };
      }

      const uiResponse = buildVoiceUiResponse({
        text: fullText || "Okay.",
        quickItems,
        toolEvents,
        responseSurface: normalizedResponseSurface
      });
      emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "success", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(fullText || "Okay.", toolTrace), error: "" });
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

    // Recipe-save vs dish-log mutual exclusion: if this turn (or an earlier one
    // in this request) issues a recipe-save, dish-log writes below are suppressed.
    const batchHasRecipeSave = batchHasRecipeSaveTool(assembledToolCalls.map((tc) => tc.function.name));

    for (const toolCall of assembledToolCalls) {
      const toolName = toolCall.function.name;
      const args = parseToolArgs(toolCall.function.arguments);
      yield { type: "tool_start", tool_name: toolName, args };

      // Suppress a phantom dish-log write when the same request saves a recipe.
      if (shouldSuppressDishLog(toolName, { batchHasRecipeSave, recipeSaveIssued })) {
        console.log(JSON.stringify({ evt: "assistant_dishlog_suppressed_on_save", tool: toolName, userId: userContext?.userId || null, reason: batchHasRecipeSave ? "same_turn_save" : "prior_turn_save" }));
        const suppressed = suppressedDishLogResult();
        toolTrace.push({ toolName, args, ok: false, statusCode: suppressed.statusCode, actionSummary: null, error: suppressed.error, suppressed: true });
        yield { type: "tool_end", tool_name: toolName, ok: false, summary: null };
        messages.push({ role: "tool", tool_call_id: toolCall.id, content: JSON.stringify(suppressed) });
        continue;
      }

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

    // Once a recipe-save is issued, keep suppressing dish-log writes for any
    // later turn in this request too.
    if (batchHasRecipeSave) recipeSaveIssued = true;

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
  fullText = scrubGuardLeakFromNarration(fullText, toolTrace);
  const uiResponse = buildVoiceUiResponse({
    text: fullText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolEvents
  });
  emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "success", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(fullText || "I heard you, but I couldn't finish that request.", toolTrace), error: "" });
  return {
    text: fullText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolTrace,
    toolEvents,
    version: uiResponse.version,
    type: uiResponse.type,
    ui: uiResponse.ui
  };
  } catch (aiOpErr) {
    emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "error", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(fullText, toolTrace), error: truncateForMarker(aiOpErr?.message || String(aiOpErr), 500) });
    throw aiOpErr;
  }
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

  // ai_op telemetry: input is byte/duration counts only — never the audio itself.
  const transcribeStart = Date.now();
  const transcribeModel = env.TRANSCRIPTION_MODEL || "gpt-4o-mini-transcribe";
  const transcribeInput = `pcm_bytes=${audioBuffer?.length || 0} wav_bytes=${wavBuffer?.length || 0} sample_rate=${Number(options.sampleRate || env.AUDIO_SAMPLE_RATE || 24000)}`;

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
    emitAiOp({ op: "transcribe", owner_id: null, model: transcribeModel, latency_ms: Date.now() - transcribeStart, status: "error", input: transcribeInput, output: "", error: truncateForMarker(error?.message || String(error), 500) });
    // OpenAI transcription down (429/5xx/abort) → fall back to native Gemini so HALO/app voice
    // still gets transcribed. Needs a durable AIza GEMINI_API_KEY (native rejects AQ. tokens).
    if (geminiFallbackEnabled(env) && isOpenAiOutageError(error)) {
      console.warn("[WARN] OpenAI transcription failed — falling back to Gemini:", String(error?.message || "").slice(0, 140));
      const fbStart = Date.now();
      const fbModel = env.GEMINI_TRANSCRIBE_MODEL || "gemini-2.5-flash";
      try {
        const fbText = await geminiTranscribe(wavBuffer, env);
        emitAiOp({ op: "transcribe_fallback", owner_id: null, model: fbModel, latency_ms: Date.now() - fbStart, status: "success", input: transcribeInput, output: truncateForMarker(fbText, 1200), error: "" });
        return fbText;
      } catch (fbErr) {
        emitAiOp({ op: "transcribe_fallback", owner_id: null, model: fbModel, latency_ms: Date.now() - fbStart, status: "error", input: transcribeInput, output: "", error: truncateForMarker(fbErr?.message || String(fbErr), 500) });
        throw fbErr;
      }
    }
    throw error;
  }
  const transcript = String(json?.text || "").trim();
  emitAiOp({ op: "transcribe", owner_id: null, model: transcribeModel, latency_ms: Date.now() - transcribeStart, status: "success", input: transcribeInput, output: truncateForMarker(transcript, 1200), error: "" });
  return transcript;
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
  // Household dietary prefs (safety-critical: allergies) injected into the system prompt so
  // Thyme honors them in suggestions and can state them when asked. Best-effort; a fetch
  // failure / no prefs yields an empty block = unchanged behavior.
  const dietaryBlock = formatDietaryPreferencesBlock(await getDietaryPreferences(userContext, { env }));
  const messages = [
    { role: "system", content: buildSystemPrompt(userContext, { responseSurface: normalizedResponseSurface, dietaryBlock }) },
    ...getAppGuideMessages(),
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
  // ai_op telemetry: op derived from responseSurface (HALO path → "chat_halo",
  // app → "chat_app"). Latency spans the whole completion loop; model is the
  // OpenAI chat model actually requested.
  const aiOpStart = Date.now();
  const aiOp = normalizedResponseSurface === "app" ? "chat_app" : "chat_halo";
  const aiModel = env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
  let falseConfirmCorrections = 0;
  let falseConfirmRecovered = false;
  // Set once a recipe-save tool is seen in the request; suppresses dish-log
  // writes for the rest of the request (mutual-exclusion guard).
  let recipeSaveIssued = false;
  // The false-confirmation corrective retry FORCES a tool call (see streaming path).
  let forceToolChoiceNext = null;

  // On the first turn, detect write intent and force tool_choice if appropriate
  const initialToolChoice = detectWriteIntent(transcript);
  if (initialToolChoice !== "auto") {
    console.log("[DEBUG] forcing tool_choice on first turn:", JSON.stringify(initialToolChoice));
  }

  try {
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

      // F-106 guard-leak scrub: replace any narration that leaked the dish-log
      // guard's internal reasoning with a clean confirmation of the primary
      // write that actually succeeded (no tool names, no "can't honestly").
      const scrubbed = scrubGuardLeakFromNarration(lastAssistantText, toolTrace);
      if (scrubbed !== lastAssistantText) {
        console.log(JSON.stringify({ evt: "assistant_guard_leak_scrubbed", userId: userContext?.userId || null, surface: "sync" }));
        lastAssistantText = scrubbed;
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

      emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "success", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(lastAssistantText || "Okay.", toolTrace), error: "" });
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

    // Recipe-save vs dish-log mutual exclusion: if this turn (or an earlier one
    // in this request) issues a recipe-save, dish-log writes are suppressed.
    const batchHasRecipeSave = batchHasRecipeSaveTool(message.tool_calls.map((tc) => tc?.function?.name));

    const toolResults = await Promise.all(
      message.tool_calls.map(async (toolCall) => {
        const toolName = toolCall?.function?.name;
        const args = parseToolArgs(toolCall?.function?.arguments);
        console.log("[DEBUG] assistant requested tool:", JSON.stringify({ toolName, args }));
        // Suppress a phantom dish-log write when the same request saves a recipe.
        if (shouldSuppressDishLog(toolName, { batchHasRecipeSave, recipeSaveIssued })) {
          console.log(JSON.stringify({ evt: "assistant_dishlog_suppressed_on_save", tool: toolName, userId: userContext?.userId || null, reason: batchHasRecipeSave ? "same_turn_save" : "prior_turn_save" }));
          return { toolCall, toolName, args, result: suppressedDishLogResult() };
        }
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
    if (batchHasRecipeSave) recipeSaveIssued = true;

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

  lastAssistantText = scrubGuardLeakFromNarration(lastAssistantText, toolTrace);
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

  emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "success", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(lastAssistantText || "I heard you, but I couldn't finish that request.", toolTrace), error: "" });
  return {
    text: lastAssistantText || "I heard you, but I couldn't finish that request.",
    quickItems,
    toolTrace,
    toolEvents,
    version: uiResponse.version,
    type: uiResponse.type,
    ui: uiResponse.ui
  };
  } catch (aiOpErr) {
    emitAiOp({ op: aiOp, owner_id: ownerIdForMarker(userContext), model: aiModel, latency_ms: Date.now() - aiOpStart, status: "error", input: truncateForMarker(transcript, 400), output: summarizeAiOutput(lastAssistantText, toolTrace), error: truncateForMarker(aiOpErr?.message || String(aiOpErr), 500) });
    throw aiOpErr;
  }
}
