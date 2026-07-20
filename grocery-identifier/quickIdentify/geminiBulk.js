const fetch = require("node-fetch");

const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_BULK_TIMEOUT_MS || 180000);
const GEMINI_MAX_DIMENSION = Number(process.env.GEMINI_BULK_MAX_DIMENSION || 2048);
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
const GEMINI_MAX_ATTEMPTS = Number(process.env.GEMINI_BULK_ATTEMPTS || 3);

// Cross-model deep fallback (defense-in-depth for a gemini-3.1-pro-preview outage).
// OFF by default (AI_FALLBACK_DEEP). When the primary model exhausts its retries with
// a TRANSIENT error, fall to a secondary model (gemini-2.5-flash) reusing the exact
// same prompt/schema/parser. Budget-capped so the fallback attempt cannot push the
// invocation past the Lambda timeout. ON flip is Matt's. Marker: ai_fallback_used
// surface:bulk_deep.
// App category enum (must stay in lock-step with CATEGORY_ENUM in
// src/openai/identifyGrocery.ts and KITCHEN_CATEGORY_ENUM in
// kitchen_api/category_normalizer.py + data-access.mjs). Constrains the Gemini
// bulk `category` output to these 9 values so raw meat can never land in
// `beverages` from a free-text guess (root cause of the steak->beverages bug:
// the schema was free-text `{type:"string"}` and the prompt only showed
// "Produce, Dairy, Condiment, Beverage" as examples — no meat, no enum).
const CATEGORY_ENUM = [
  "leftovers", "produce", "dairy_eggs", "meat_seafood",
  "pantry", "spices", "snacks_sweets", "beverages", "prepared_other",
];

const AI_FALLBACK_DEEP = String(process.env.AI_FALLBACK_DEEP || "").toLowerCase() === "true";
const AI_FALLBACK_DEEP_MODEL = process.env.AI_FALLBACK_DEEP_MODEL || "gemini-2.5-flash";
const AI_FALLBACK_DEEP_TIMEOUT_MS = Number(process.env.AI_FALLBACK_DEEP_TIMEOUT_MS || 120000);
const FN_BUDGET_MS = Number(process.env.GEMINI_BULK_FN_BUDGET_MS || 900000);
const FN_FINALIZE_BUFFER_MS = Number(process.env.GEMINI_BULK_FINALIZE_BUFFER_MS || 60000);
const MIN_FALLBACK_BUDGET_MS = 15000;

function emitLog(evt, fields) {
  try {
    console.log(JSON.stringify({ evt, ...fields }));
  } catch (_e) {
    /* logging must never throw */
  }
}

const geminiResponseSchema = {
  type: "object",
  properties: {
    visual_analysis_log: { type: "string" },
    inventory: {
      type: "array",
      items: {
        type: "object",
        properties: {
          item_name: { type: "string" },
          brand: { type: "string", nullable: true },
          variant: { type: "string", nullable: true },
          position_hint: { type: "string", nullable: true },
          estimated_state: { type: "string" },
          fill_level: { type: "string", nullable: true },
          // Constrained to the 9 app enums (was free-text `{type:"string"}`). Gemini
          // v1beta responseSchema accepts `enum` on a string (see toGeminiSchema in
          // src/openai/geminiFallback.ts). This forces raw meat -> meat_seafood
          // instead of a free-text guess the downstream normalizer can't rescue.
          category: { type: "string", enum: CATEGORY_ENUM },
          visible_text_OCR: { type: "string", nullable: true },
        },
        required: ["item_name", "brand", "variant", "position_hint", "estimated_state", "fill_level", "category", "visible_text_OCR"],
      },
    },
  },
  required: ["visual_analysis_log", "inventory"],
};

let sharpModule = null;
let sharpLoadAttempted = false;

function getSharp() {
  if (sharpLoadAttempted) {
    return sharpModule;
  }

  sharpLoadAttempted = true;
  try {
    sharpModule = require("sharp");
  } catch (error) {
    console.warn("[GeminiBulk] sharp unavailable, skipping resize optimization:", error?.message || error);
    sharpModule = null;
  }

  return sharpModule;
}

function cleanNullable(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function tokenize(value) {
  return normalizeText(value)
    .split(" ")
    .map((token) => token.trim())
    .filter(Boolean);
}

function uniqueStrings(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

function inferItemType(category, itemName) {
  const joined = normalizeText([category, itemName].filter(Boolean).join(" "));
  const produceTokens = [
    "produce",
    "vegetable",
    "vegetables",
    "fruit",
    "fruits",
    "lettuce",
    "pepper",
    "scallion",
    "onion",
    "radish",
    "cabbage",
    "broccoli",
    "cauliflower",
    "apple",
    "banana",
    "grape",
  ];

  if (produceTokens.some((token) => joined.includes(token))) {
    return "produce";
  }

  if (!joined) {
    return "unknown";
  }

  return "packaged";
}

function extractVisibleText(value) {
  const normalized = cleanNullable(value);
  if (!normalized) {
    return [];
  }

  return uniqueStrings(
    normalized
      .split(/\n|,|;|\|/)
      .map((part) => part.trim())
      .filter(Boolean)
  ).slice(0, 6);
}

function normalizePositionHint(value) {
  const normalized = cleanNullable(value);
  if (!normalized) {
    return null;
  }

  const cleaned = normalized
    .toLowerCase()
    .replace(/[^a-z0-9\s-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleaned) {
    return null;
  }

  return cleaned
    .split(" ")
    .map((part) => {
      if (!part) return part;
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function normalizeFillLevel(value, estimatedState) {
  const explicit = cleanNullable(value);
  const explicitText = normalizeText(explicit);
  if (explicitText) {
    if (explicitText.includes("unopened") || explicitText.includes("sealed")) return "unopened";
    if (explicitText.includes("full")) return explicitText.includes("mostly") ? "mostly_full" : "full";
    if (explicitText.includes("three quarter") || explicitText.includes("75")) return "mostly_full";
    if (explicitText.includes("half")) return "half_full";
    if (explicitText.includes("low") || explicitText.includes("quarter")) return "low";
    if (explicitText.includes("nearly empty")) return "nearly_empty";
    if (explicitText.includes("empty")) return "empty";
    if (explicitText.includes("unknown")) return "unknown";
  }

  const stateText = normalizeText(estimatedState);
  if (stateText.includes("unopened") || stateText.includes("sealed")) return "unopened";
  if (stateText.includes("full")) return stateText.includes("mostly") ? "mostly_full" : "full";
  if (stateText.includes("75")) return "mostly_full";
  if (stateText.includes("half")) return "half_full";
  if (stateText.includes("quarter") || stateText.includes("low")) return "low";
  if (stateText.includes("nearly empty")) return "nearly_empty";
  if (stateText.includes("empty")) return "empty";
  return "unknown";
}

function buildCheckInLabel(entry) {
  const brand = cleanNullable(entry.brand);
  const itemName = cleanNullable(entry.item_name);
  const variant = cleanNullable(entry.variant);

  let label = itemName || brand || "Unknown item";
  const normalizedLabel = normalizeText(label);

  if (brand && !normalizedLabel.includes(normalizeText(brand))) {
    label = `${brand} ${label}`.trim();
  }

  if (variant) {
    const variantTokens = tokenize(variant);
    const labelTokens = tokenize(label);
    const missingVariantTokens = variantTokens.filter((token) => !labelTokens.includes(token));
    if (missingVariantTokens.length > 0) {
      label = `${label} ${variant}`.trim();
    }
  }

  return label.trim();
}

const GENERIC_CONTAINER_NAMES = new Set([
  "bag",
  "bottle",
  "box",
  "can",
  "carton",
  "container",
  "jar",
  "package",
  "packet",
  "pouch",
  "tub",
  "wrapper",
]);

const GENERIC_FOOD_NAMES = new Set([
  "condiment",
  "dip",
  "dressing",
  "sauce",
  "spread",
]);

function isGenericShopperLabel(value, genericNames) {
  const normalized = normalizeText(value);
  return Boolean(normalized) && genericNames.has(normalized);
}

function estimateCount(estimatedState) {
  const text = normalizeText(estimatedState);
  const approximateMatch = text.match(/(?:approx|about|around)?\s*(\d+)\s+(?:items|bottles|jars|eggs|cups|cans|boxes|containers|heads|peppers|pieces)/);
  if (approximateMatch) {
    return Math.max(1, Number(approximateMatch[1]));
  }

  const singleMatch = text.match(/\b(\d+)\b/);
  if (singleMatch) {
    return Math.max(1, Number(singleMatch[1]));
  }

  return 1;
}

function buildUncertaintyReasons(entry, visibleText) {
  const reasons = [];
  const state = normalizeText(entry.estimated_state);
  const category = normalizeText(entry.category);
  const fillLevel = normalizeText(entry.fill_level);

  if (!entry.brand) {
    reasons.push("missing_brand");
  }
  if (!entry.variant) {
    reasons.push("missing_variant_detail");
  }
  if (visibleText.length === 0) {
    reasons.push("no_visible_text");
  }
  if (state.includes("approx")) {
    reasons.push("approximate_state");
  }
  if (state.includes("partially") || state.includes("occluded")) {
    reasons.push("partial_visibility");
  }
  if (state.includes("likely") || state.includes("appears") || state.includes("unclear") || state.includes("difficult")) {
    reasons.push("uncertain_identification");
  }
  if (category.includes("unknown")) {
    reasons.push("unknown_category");
  }
  if (!fillLevel || fillLevel === "unknown") {
    reasons.push("unknown_fill_level");
  }
  return uniqueStrings(reasons);
}

function inferConfidence(entry, visibleText, uncertaintyReasons) {
  let confidence = 0.72;
  const itemName = cleanNullable(entry?.item_name);
  const variant = cleanNullable(entry?.variant);
  const brand = cleanNullable(entry?.brand);
  const isGenericContainer = isGenericShopperLabel(itemName, GENERIC_CONTAINER_NAMES);
  const isGenericFood = isGenericShopperLabel(itemName, GENERIC_FOOD_NAMES);

  if (entry.brand) confidence += 0.08;
  if (visibleText.length > 0) confidence += 0.08;
  if (uncertaintyReasons.includes("partial_visibility")) confidence -= 0.1;
  if (uncertaintyReasons.includes("uncertain_identification")) confidence -= 0.14;
  if (uncertaintyReasons.includes("unknown_category")) confidence -= 0.12;
  if (isGenericContainer) confidence -= 0.18;
  if (isGenericFood) confidence -= 0.12;
  if (!variant && isGenericFood) confidence -= 0.06;
  if (!brand && (isGenericContainer || isGenericFood)) confidence -= 0.04;
  return Math.max(0.3, Math.min(0.98, confidence));
}

function normalizeInventoryEntry(entry) {
  const visibleText = extractVisibleText(entry.visible_text_OCR);
  const fillLevel = normalizeFillLevel(entry.fill_level, entry.estimated_state);
  const normalizedEntry = { ...entry, fill_level: fillLevel };
  const rawItemName = cleanNullable(entry.item_name) || "Unknown item";
  const variant = cleanNullable(entry.variant);
  const brand = cleanNullable(entry.brand);
  const displayItemName = buildCheckInLabel({ item_name: rawItemName, brand, variant });
  const itemType = inferItemType(entry.category, rawItemName);
  const uncertaintyReasons = buildUncertaintyReasons(normalizedEntry, visibleText);
  return {
    item_name: displayItemName,
    brand,
    variant,
    category: cleanNullable(entry.category) || "Unknown",
    item_type: itemType,
    count: estimateCount(entry.estimated_state),
    position_hint: normalizePositionHint(entry.position_hint),
    confidence: inferConfidence({ ...normalizedEntry, item_name: rawItemName, brand, variant }, visibleText, uncertaintyReasons),
    short_description: cleanNullable(entry.estimated_state) || "Gemini fridge inventory result.",
    needs_review: uncertaintyReasons.length > 0,
    uncertainty_reasons: uncertaintyReasons,
    visible_text: visibleText,
    estimated_state: cleanNullable(entry.estimated_state),
    fill_level: fillLevel,
    visible_text_OCR: cleanNullable(entry.visible_text_OCR),
    check_in_label: displayItemName,
  };
}

function normalizeImageUrl(rawUrl) {
  const cleaned = cleanNullable(rawUrl)?.replace(/^['"]+|['"]+$/g, "").replace(/&amp;/gi, "&");
  if (!cleaned) {
    throw new Error("Image URL is empty");
  }

  const withProtocol = cleaned.startsWith("//") ? `https:${cleaned}` : cleaned;
  const parsed = new URL(withProtocol);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Image URL must use http or https");
  }
  return parsed.toString();
}

async function loadImageAsset(input) {
  if (input.imageBuffer) {
    return {
      buffer: input.imageBuffer,
      mimeType: input.mimeType || "image/jpeg",
    };
  }

  if (!input.imageUrl) {
    throw new Error("Either imageBuffer or imageUrl is required");
  }

  const response = await fetch(normalizeImageUrl(input.imageUrl), {
    headers: { "user-agent": "Mozilla/5.0" },
    timeout: 30000,
  });
  if (!response.ok) {
    throw new Error(`Failed to download image from URL: HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "image/jpeg";
  return {
    buffer: await response.buffer(),
    mimeType: contentType,
  };
}

async function resizeImageAsset(imageAsset) {
  const sharp = getSharp();
  if (!sharp) {
    return imageAsset;
  }

  const image = sharp(imageAsset.buffer, {
    sequentialRead: true,
  });
  try {
    const metadata = await image.metadata();
    const width = metadata.width || 0;
    const height = metadata.height || 0;
    const maxDimension = Math.max(width, height);

    if (!maxDimension || maxDimension <= GEMINI_MAX_DIMENSION) {
      return imageAsset;
    }

    return {
      buffer: await image
        .rotate()
        .resize({
          width: width >= height ? GEMINI_MAX_DIMENSION : undefined,
          height: height > width ? GEMINI_MAX_DIMENSION : undefined,
          fit: "inside",
          withoutEnlargement: true,
        })
        .jpeg({
          quality: 85,
          mozjpeg: true,
        })
        .toBuffer(),
      mimeType: "image/jpeg",
    };
  } catch (error) {
    console.warn("[GeminiBulk] Image resize skipped:", error?.message || error);
    return imageAsset;
  }
}

async function reportStage(options, stage, message, progress, details = {}) {
  if (typeof options?.onStage !== "function") {
    return;
  }

  await options.onStage({
    stage,
    message,
    progress,
    ...details,
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildGeminiPrompt() {
  return `You are an advanced inventory management AI agent specializing in high-fidelity grocery inventory extraction. I have provided you with a dense, high-resolution image of groceries in a refrigerator, pantry, or shelf scene.

You are tasked with returning a perfectly structured JSON object with two main sections:

Section 1: visual_analysis_log (String)
Keep this brief. In 1 to 3 short sentences, mention any especially hard-to-identify or partially occluded items.

Section 2: inventory (Array of Objects)
Only after completing your reasoning, create the final inventory list. Focus first on identifying the correct visible grocery items in the scene. Each object in this array must contain:
- item_name: A shopper-facing grocery title. Make this as detailed as the visible evidence supports. Prefer the actual product or food name over generic container words like "jar", "bottle", or "packet" whenever readable label text gives you enough evidence.
- brand: The brand name only when clearly verifiable from readable label text, otherwise null.
- variant: Flavor, size, pack count, fat percentage, variety, or other variant details when clearly visible, otherwise null.
- position_hint: A coarse human-readable location only when visually obvious, such as "top shelf left", "middle shelf right", "bottom drawer center", or "door shelf upper-right". Do not guess precise coordinates. If location is unclear, return null.
- estimated_state: A short factual state such as "unopened box", "partially full bottle", or "produce in bag".
- fill_level: One of "unopened", "full", "mostly_full", "half_full", "low", "nearly_empty", "empty", or "unknown". Use "unknown" if the fill level is not clear.
- category: Choose EXACTLY ONE of these 9 values by what the item fundamentally IS, NOT by incidental words in its name: leftovers, produce, dairy_eggs, meat_seafood, pantry, spices, snacks_sweets, beverages, prepared_other. Hard rules: fresh or raw meat, poultry, or fish — including ANY beef/pork/lamb/veal cut such as steak, ribeye, sirloin, NY strip, skirt steak, flank, T-bone, brisket, filet, pork steak, ham steak, chops — = meat_seafood (a raw steak is NEVER a beverage). Any drink = beverages. Chicken or beef broth = pantry. Spices, seasonings, rubs, and spice blends (incl. steak seasoning) = spices — this ALSO includes culinary/finishing SALTS (black salt, sea salt, kosher salt, Himalayan/pink salt, seasoning salt, garlic/onion salt), MASALA and Indian spice blends (garam masala, chaat masala, tandoori masala, curry powder), whole or ground PEPPER and peppercorns (black/white pepper, Sichuan peppercorns, red pepper flakes), and whole spices (cumin/coriander/mustard seeds, cardamom, cloves, nutmeg, saffron, allspice, turmeric). BUT a food that only mentions salt, pepper, or a spice as a flavor or minor ingredient keeps its OWN category — canned beans 'No Salt Added' = pantry, tuna 'Lemon Pepper' = meat_seafood, a Tikka Masala simmer-sauce/ready-meal = prepared_other, garlic cloves = produce. Steak sauce / marinade = pantry. Yogurt = dairy_eggs. Bread/bagels/tortillas = pantry. Muffins/cookies/cakes/pastries = snacks_sweets. Tofu/tempeh/seitan/plant-based meat substitutes = prepared_other. Home leftover food = leftovers.
- visible_text_OCR: Any text you literally read from the label that guided identification. If nothing readable, return null.

Title examples:
- Better than "Jar": "Greek Peperoncini"
- Better than "Bottle": "Primal Kitchen Dressing"
- Better than "Packet": "Reese's Peanut Butter Cups"
- If brand is visible, include it in the title when natural: "Jeff's Garden Greek Peperoncini"
- If size is visible, include it in variant: "Orange Juice", variant: "52 fl oz"
- Good position_hint examples: "top shelf left", "middle shelf center-right", "bottom shelf right"

Be extremely factual. Rely only on what you see. Do not invent brands. If detail is unclear, keep the item_name simpler rather than guessing, but still prefer a real grocery/product noun over a generic container noun whenever the visible label supports it. Location should stay coarse and conservative: prefer null over a bad position_hint. Prefer identifying more real visible items over writing long reasoning.`;
}

function isTransientGeminiFetchError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  const code = String(error?.code || "").toUpperCase();
  const type = String(error?.type || "").toLowerCase();
  return (
    code === "ETIMEDOUT" ||
    code === "ECONNRESET" ||
    code === "ECONNREFUSED" ||
    code === "EAI_AGAIN" ||
    type === "request-timeout" ||
    message.includes("network timeout") ||
    message.includes("timed out") ||
    message.includes("socket hang up") ||
    message.includes("fetch failed") ||
    message.includes("temporary failure in name resolution")
  );
}

function buildRetryDelayMs(attempt) {
  const baseDelayMs = 2000 * (attempt + 1);
  const jitterMs = Math.floor(Math.random() * 500);
  return baseDelayMs + jitterMs;
}

// Pure decision for the cross-model deep fallback (exported for unit testing — no
// network). Fall back ONLY when: the flag is on, the primary error is transient, and
// enough of the Lambda budget remains for one capped secondary attempt.
function deepFallbackDecision(error, { enabled, elapsedMs } = {}) {
  if (!enabled) return { fallback: false, reason: "disabled" };
  if (!isTransientGeminiFetchError(error)) return { fallback: false, reason: "non_transient" };
  const remainingMs = FN_BUDGET_MS - (Number(elapsedMs) || 0) - FN_FINALIZE_BUFFER_MS;
  const timeoutMs = Math.min(AI_FALLBACK_DEEP_TIMEOUT_MS, remainingMs);
  if (timeoutMs < MIN_FALLBACK_BUDGET_MS) {
    return { fallback: false, reason: "no_budget", remainingMs };
  }
  return { fallback: true, reason: "primary_transient", timeoutMs, model: AI_FALLBACK_DEEP_MODEL };
}

async function callGemini(imageAsset, options = {}) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required");
  }

  // Model / timeout / attempt count are overridable (used by the cross-model deep
  // fallback to re-run this exact request against a secondary model with a capped
  // single attempt). Defaults preserve the primary-model behavior.
  const model = options.modelOverride || GEMINI_MODEL;
  const timeoutMs = options.timeoutOverride || GEMINI_TIMEOUT_MS;
  const maxAttempts = options.attemptsOverride || GEMINI_MAX_ATTEMPTS;

  const body = {
    systemInstruction: {
      parts: [{ text: buildGeminiPrompt() }],
    },
    contents: [
      {
        role: "user",
        parts: [
          {
            text: "Analyze this grocery scene carefully. Prioritize identifying the correct visible grocery items. Keep the visual_analysis_log short, and use null or unknown for any detail that is not clearly visible.",
          },
          {
            inline_data: {
              mime_type: imageAsset.mimeType,
              data: imageAsset.buffer.toString("base64"),
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.1,
      topP: 0.95,
      responseMimeType: "application/json",
      responseSchema: geminiResponseSchema,
    },
  };

  let lastError = null;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          timeout: timeoutMs,
        }
      );

      if (response.ok) {
        const payload = await response.json();
        const text = payload?.candidates?.[0]?.content?.parts?.map((part) => part.text).filter(Boolean).join("\n") || "{}";
        return JSON.parse(text);
      }

      const errorText = await response.text();
      if (response.status === 429) {
        throw new Error("Gemini quota exceeded or billing is not enabled for this API key. Enable Gemini paid usage, then try the bulk scan again.");
      }

      if (response.status >= 500 && response.status < 600) {
        lastError = new Error(`Gemini Pro transient HTTP ${response.status}: ${errorText}`.trim());
        if (attempt < maxAttempts - 1) {
          await reportStage(
            options,
            "gemini_reasoning",
            `Hmm, that took a sec. Giving it another shot (${attempt + 2}/${maxAttempts})...`,
            42,
            { model, retry_attempt: attempt + 1, retry_reason: `http_${response.status}` }
          );
          await sleep(buildRetryDelayMs(attempt));
          continue;
        }
        throw new Error("Gemini Pro is experiencing high demand right now. Please try the bulk scan again in a moment.");
      }

      throw new Error(`Gemini bulk scan failed: HTTP ${response.status} ${errorText}`.trim());
    } catch (error) {
      if (isTransientGeminiFetchError(error)) {
        lastError = error instanceof Error ? error : new Error(String(error || "Unknown Gemini timeout"));
        if (attempt < maxAttempts - 1) {
          await reportStage(
            options,
            "gemini_reasoning",
            `Still working on it \u2014 retrying (${attempt + 2}/${maxAttempts})...`,
            42,
            { model, retry_attempt: attempt + 1, retry_reason: "network_timeout" }
          );
          await sleep(buildRetryDelayMs(attempt));
          continue;
        }
      }
      throw error;
    }
  }

  throw lastError || new Error("Gemini bulk scan failed.");
}

async function identifyItemGeminiBulk(input, options = {}) {
  await reportStage(options, "loading_image", "Getting your photo ready...", 5, { model: GEMINI_MODEL });
  const loadedAsset = await loadImageAsset(input);
  await reportStage(options, "preparing_image", "Taking a closer look...", 16, { model: GEMINI_MODEL });
  const imageAsset = await resizeImageAsset(loadedAsset);
  const startedAt = Date.now();

  await reportStage(options, "gemini_reasoning", "Identifying your groceries...", 42, { model: GEMINI_MODEL });
  let geminiResult;
  try {
    geminiResult = await callGemini(imageAsset, options);
  } catch (primaryErr) {
    // Cross-model deep fallback (defense-in-depth). Only engages when the flag is on,
    // the primary failure is transient, and enough Lambda budget remains for one capped
    // secondary attempt (else it would risk blowing the 900s function timeout).
    const decision = deepFallbackDecision(primaryErr, {
      enabled: AI_FALLBACK_DEEP,
      elapsedMs: Date.now() - startedAt,
    });
    if (!decision.fallback) {
      if (AI_FALLBACK_DEEP && decision.reason === "no_budget") {
        emitLog("ai_fallback_skipped", { surface: "bulk_deep", reason: "no_budget", remaining_ms: decision.remainingMs });
      }
      throw primaryErr;
    }
    emitLog("ai_fallback_triggered", {
      surface: "bulk_deep",
      reason: decision.reason,
      from_model: GEMINI_MODEL,
      to_model: decision.model,
      timeout_ms: decision.timeoutMs,
    });
    await reportStage(options, "gemini_reasoning", "Trying a faster model...", 42, {
      model: decision.model,
      retry_reason: "cross_model_fallback",
    });
    try {
      geminiResult = await callGemini(imageAsset, {
        ...options,
        modelOverride: decision.model,
        timeoutOverride: decision.timeoutMs,
        attemptsOverride: 1,
      });
      emitLog("ai_fallback_used", { surface: "bulk_deep", provider: "gemini", model: decision.model, ok: true });
    } catch (fallbackErr) {
      emitLog("ai_fallback_failed", {
        surface: "bulk_deep",
        provider: "gemini",
        model: decision.model,
        error: String(fallbackErr?.message || fallbackErr),
      });
      throw primaryErr; // preserve the original error surfaced to the user/job
    }
  }
  await reportStage(options, "finalizing_inventory", "Wrapping things up...", 88, { model: GEMINI_MODEL });

  const items = Array.isArray(geminiResult?.inventory) ? geminiResult.inventory.map(normalizeInventoryEntry) : [];
  const itemTypeCounts = items.reduce(
    (summary, item) => {
      if (item.item_type === "produce") summary.produce += 1;
      else if (item.item_type === "packaged") summary.packaged += 1;
      else summary.unknown += 1;
      return summary;
    },
    { produce: 0, packaged: 0, unknown: 0 }
  );

  return {
    items,
    visual_analysis_log: cleanNullable(geminiResult?.visual_analysis_log),
    debug: {
      elapsed_ms: Date.now() - startedAt,
      mode: "bulk_inventory_deep",
      model: GEMINI_MODEL,
      provider: "gemini",
      item_count: items.length,
      item_type_counts: itemTypeCounts,
      review_required_count: items.filter((item) => item.needs_review).length,
      total_quantity: items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0),
    },
  };
}

module.exports = {
  identifyItemGeminiBulk,
  buildCheckInLabel,
  deepFallbackDecision,
};
