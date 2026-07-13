const fetch = require("node-fetch");

const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_BULK_TIMEOUT_MS || 180000);
const GEMINI_MAX_DIMENSION = Number(process.env.GEMINI_BULK_MAX_DIMENSION || 2048);
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
const GEMINI_MAX_ATTEMPTS = Number(process.env.GEMINI_BULK_ATTEMPTS || 3);

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
          category: { type: "string" },
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
- category: e.g. Produce, Dairy, Condiment, Beverage.
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

async function callGemini(imageAsset, options = {}) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required");
  }

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

  for (let attempt = 0; attempt < GEMINI_MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${process.env.GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          timeout: GEMINI_TIMEOUT_MS,
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
        if (attempt < GEMINI_MAX_ATTEMPTS - 1) {
          await reportStage(
            options,
            "gemini_reasoning",
            `Hmm, that took a sec. Giving it another shot (${attempt + 2}/${GEMINI_MAX_ATTEMPTS})...`,
            42,
            { model: GEMINI_MODEL, retry_attempt: attempt + 1, retry_reason: `http_${response.status}` }
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
        if (attempt < GEMINI_MAX_ATTEMPTS - 1) {
          await reportStage(
            options,
            "gemini_reasoning",
            `Still working on it \u2014 retrying (${attempt + 2}/${GEMINI_MAX_ATTEMPTS})...`,
            42,
            { model: GEMINI_MODEL, retry_attempt: attempt + 1, retry_reason: "network_timeout" }
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
  const geminiResult = await callGemini(imageAsset, options);
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
};
