const { z } = require("zod");
const fetch = require("node-fetch");
const { Jimp } = require("jimp");
const { getOpenAIClient, parseJsonResponse } = require("../dist/openai/client");
const { appendUncertaintySignals, applyCatalogMatch, classifyItemType } = require("./catalog");
const { extractGoogleVisionContext } = require("./googleVision");
const { identifyItemGeminiBulk } = require("./geminiBulk");
const { identifyReceiptGeminiDeep } = require("./geminiReceipt");

const MAX_REGIONS = 18;
const CROP_PADDING_RATIO = 0.08;
const FAST_LEGACY_TIMEOUT_MS = Number(process.env.FAST_LEGACY_TIMEOUT_MS || 5000);
const FAST_LEGACY_ATTEMPTS = 1;
const QUICK_GLANCE_TIMEOUT_MS = Number(process.env.QUICK_GLANCE_TIMEOUT_MS || 5000);
const QUICK_GLANCE_ATTEMPTS = 1;

function getQuickIdentifyModel() {
  return process.env.OPENAI_FAST_MODEL || "gpt-4.1-mini";
}

function getDeepIdentifyModel() {
  return process.env.OPENAI_DEEP_MODEL || "gpt-5.4-2026-03-05";
}

function isDeepModelMode(mode = "fast") {
  return mode === "deep" || mode === "bulk";
}

function getModelForMode(mode = "fast") {
  return isDeepModelMode(mode) ? getDeepIdentifyModel() : getQuickIdentifyModel();
}

function getPipelineConfig(mode = "fast") {
  if (mode === "bulk") {
    return {
      mode: "bulk",
      publicMode: "bulk_inventory_deep",
      model: getModelForMode("bulk"),
      maxRegionIdentifications: 18,
      identifyConcurrency: 3,
      structuredRetryAttempts: 3,
      regionDetectTimeoutMs: 35000,
      regionIdentifyTimeoutMs: 18000,
      wholeSceneTimeoutMs: 45000,
      reviewTimeoutMs: 45000,
      missingItemsTimeoutMs: 25000,
      maxOutputItems: 36,
      missingItemsMaxOutput: 14,
    };
  }

  if (mode === "deep") {
    return {
      mode: "deep",
      publicMode: "quick_identify_deep",
      model: getModelForMode("deep"),
      maxRegionIdentifications: 16,
      identifyConcurrency: 3,
      structuredRetryAttempts: 2,
      regionDetectTimeoutMs: 25000,
      regionIdentifyTimeoutMs: 12000,
      wholeSceneTimeoutMs: 20000,
      reviewTimeoutMs: 18000,
      missingItemsTimeoutMs: 15000,
      maxOutputItems: 24,
      missingItemsMaxOutput: 10,
    };
  }

  return {
    mode: "fast",
    publicMode: "fast",
    model: getModelForMode("fast"),
    maxRegionIdentifications: 3,
    identifyConcurrency: 3,
    structuredRetryAttempts: 1,
    regionDetectTimeoutMs: 6000,
    regionIdentifyTimeoutMs: 4000,
    wholeSceneTimeoutMs: 7000,
    reviewTimeoutMs: 8000,
    missingItemsTimeoutMs: 5000,
    maxOutputItems: 10,
    missingItemsMaxOutput: 10,
  };
}

function getReasoningConfig(model, effort = "low") {
  return model.startsWith("gpt-5") ? { effort } : undefined;
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

const FastItemSchema = z.object({
  item_name: z.string().nullable(),
  brand: z.string().nullable(),
  category: z.string().nullable(),
  item_type: z.enum(["produce", "packaged", "unknown"]).default("unknown"),
  count: z.number().int().min(1).default(1),
  position_hint: z.string().nullable(),
  confidence: z.number().min(0).max(1),
  short_description: z.string().nullable(),
  needs_review: z.boolean().default(false),
  uncertainty_reasons: z.array(z.string()).default([]),
  visible_text: z.array(z.string()).default([]),
});

const RegionSchema = z.object({
  label_guess: z.string().nullable(),
  position_hint: z.string().nullable(),
  count_estimate: z.number().int().min(1).default(1),
  confidence: z.number().min(0).max(1),
  is_group: z.boolean().default(false),
  x: z.number().min(0).max(1),
  y: z.number().min(0).max(1),
  width: z.number().min(0.02).max(1),
  height: z.number().min(0.02).max(1),
});

const RegionResponseSchema = z.object({
  regions: z.array(RegionSchema).default([]),
});

const regionJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["regions"],
  properties: {
    regions: {
      type: "array",
      maxItems: MAX_REGIONS,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["label_guess", "position_hint", "count_estimate", "confidence", "is_group", "x", "y", "width", "height"],
        properties: {
          label_guess: { type: ["string", "null"] },
          position_hint: { type: ["string", "null"] },
          count_estimate: { type: "integer", minimum: 1 },
          confidence: { type: "number", minimum: 0, maximum: 1 },
          is_group: { type: "boolean" },
          x: { type: "number", minimum: 0, maximum: 1 },
          y: { type: "number", minimum: 0, maximum: 1 },
          width: { type: "number", minimum: 0.02, maximum: 1 },
          height: { type: "number", minimum: 0.02, maximum: 1 },
        },
      },
    },
  },
};

const itemJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "item_name",
    "brand",
    "category",
    "item_type",
    "count",
    "position_hint",
    "confidence",
    "short_description",
    "needs_review",
    "uncertainty_reasons",
    "visible_text",
  ],
  properties: {
    item_name: { type: ["string", "null"] },
    brand: { type: ["string", "null"] },
    category: { type: ["string", "null"] },
    item_type: { type: "string", enum: ["produce", "packaged", "unknown"] },
    count: { type: "integer", minimum: 1 },
    position_hint: { type: ["string", "null"] },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    short_description: { type: ["string", "null"] },
    needs_review: { type: "boolean" },
    uncertainty_reasons: {
      type: "array",
      items: { type: "string" },
    },
    visible_text: {
      type: "array",
      items: { type: "string" },
    },
  },
};

function cleanNullable(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function normalizeForCompare(value) {
  return cleanNullable(value)?.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim() || "";
}

function tokenize(value) {
  return normalizeForCompare(value)
    .split(" ")
    .map((token) => token.trim())
    .filter(Boolean);
}

function tokenSet(value) {
  return new Set(tokenize(value));
}

function overlapCount(a, b) {
  let count = 0;
  for (const token of a) {
    if (b.has(token)) {
      count += 1;
    }
  }
  return count;
}

function hasAnyToken(set, values) {
  return values.some((value) => set.has(value));
}

function uniqueStrings(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isGenericTitle(value) {
  const tokens = tokenSet(value);
  if (tokens.size === 0) {
    return true;
  }
  if (hasAnyToken(tokens, ["unknown", "unidentified", "generic", "misc", "assorted"])) {
    return true;
  }

  const genericWords = ["item", "product", "food", "beverage", "container", "package", "pack", "can", "bottle", "jar", "box", "bag", "bowl", "aluminum"];
  const genericCount = genericWords.filter((word) => tokens.has(word)).length;
  return genericCount >= Math.max(2, Math.ceil(tokens.size * 0.6));
}

function smartTitleCase(value) {
  const cleaned = cleanNullable(value);
  if (!cleaned) {
    return null;
  }

  return cleaned
    .split(/\s+/)
    .map((word) =>
      word
        .split("-")
        .map((part) => {
          if (!part) return part;
          if (part === part.toUpperCase() && part.length <= 5) return part;
          if (/\d/.test(part)) return part;
          return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
        })
        .join("-")
    )
    .join(" ");
}

function normalizeBrand(value) {
  const cleaned = cleanNullable(value);
  if (!cleaned) {
    return null;
  }
  const upperBrands = ["ZYN", "RXBAR", "M&M'S", "KIND", "G2G"];
  const exactUpper = upperBrands.find((brand) => brand.toLowerCase() === cleaned.toLowerCase());
  return exactUpper || smartTitleCase(cleaned);
}

function ensureDetailedItemName(itemName, brand) {
  const normalizedItemName = smartTitleCase(itemName);
  const normalizedBrand = normalizeBrand(brand);
  if (!normalizedBrand) {
    return normalizedItemName || "Unknown Item";
  }
  if (!normalizedItemName) {
    return normalizedBrand;
  }
  if (normalizedItemName.toLowerCase().includes(normalizedBrand.toLowerCase())) {
    return normalizedItemName;
  }
  return `${normalizedBrand} ${normalizedItemName}`;
}

function buildFallbackName(item) {
  const parts = [cleanNullable(item.brand), cleanNullable(item.item_name), cleanNullable(item.category)].filter(Boolean);
  if (parts.length === 0) {
    return "Unknown item";
  }
  const [first, ...rest] = parts;
  const deduped = [first];
  for (const part of rest) {
    if (!first.toLowerCase().includes(part.toLowerCase())) {
      deduped.push(part);
    }
  }
  return deduped.join(" ");
}

function normalizePositionHint(value) {
  return smartTitleCase(cleanNullable(value));
}

const QUICK_GLANCE_PACKAGED_CUE_TOKENS = [
  "bag",
  "box",
  "carton",
  "jar",
  "bottle",
  "pack",
  "package",
  "container",
  "mix",
  "kit",
  "salad",
  "slaw",
  "coleslaw",
  "washed",
  "ready",
  "refrigerated",
  "gum",
  "mints",
  "yogurt",
  "milk",
  "cheese",
  "tortillas",
  "pasta",
  "oats",
  "coffee",
  "beans",
  "rice",
  "cakes",
];

const QUICK_GLANCE_LABEL_HINT_TOKENS = new Set([
  "mix",
  "kit",
  "salad",
  "slaw",
  "coleslaw",
  "spinach",
  "greens",
  "lettuce",
  "gum",
  "mints",
  "yogurt",
  "milk",
  "cheese",
  "tortillas",
  "pasta",
  "oats",
  "coffee",
  "beans",
  "rice",
  "cakes",
  "fries",
  "tots",
  "broccoli",
  "grapes",
  "bananas",
]);

const QUICK_GLANCE_NON_PRODUCT_TOKENS = new Set([
  "organic",
  "natural",
  "fresh",
  "washed",
  "ready",
  "keep",
  "refrigerated",
  "perishable",
  "net",
  "wt",
  "oz",
  "lb",
  "serving",
  "calories",
  "whole",
  "foods",
  "market",
]);

function normalizeBaseItem(item) {
  const parsed = FastItemSchema.parse(item);
  const normalized = {
    item_name: cleanNullable(parsed.item_name),
    brand: normalizeBrand(parsed.brand),
    category: smartTitleCase(parsed.category),
    item_type: parsed.item_type || "unknown",
    count: Number.isFinite(parsed.count) ? parsed.count : 1,
    position_hint: normalizePositionHint(parsed.position_hint),
    confidence: parsed.confidence,
    short_description: cleanNullable(parsed.short_description),
    needs_review: Boolean(parsed.needs_review),
    uncertainty_reasons: uniqueStrings((parsed.uncertainty_reasons || []).map((value) => cleanNullable(value)).filter(Boolean)).slice(0, 6),
    visible_text: uniqueStrings(
      parsed.visible_text
        .map((value) => cleanNullable(value))
        .filter(Boolean)
        .slice(0, 8)
    ),
  };

  if (!normalized.item_name) {
    normalized.item_name = buildFallbackName(normalized);
  }
  normalized.item_name = ensureDetailedItemName(normalized.item_name, normalized.brand);

  if (!normalized.short_description) {
    normalized.short_description = normalized.category
      ? `Likely a ${normalized.category.toLowerCase()} item.`
      : "Quick best-effort item identification.";
  }

  return normalized;
}

function hasQuickGlancePackagingCues(item) {
  const visibleText = Array.isArray(item?.visible_text) ? item.visible_text : [];
  const visibleTokens = tokenSet(visibleText.join(" "));
  const combinedTokens = tokenSet([item?.item_name, item?.brand, item?.category, visibleText.join(" ")].filter(Boolean).join(" "));
  const packagedCueCount = QUICK_GLANCE_PACKAGED_CUE_TOKENS.filter((token) => combinedTokens.has(token)).length;
  const hasStructuredLabelText = visibleText.some((value) => tokenize(value).length >= 2);
  return packagedCueCount > 0 || (Boolean(item?.brand) && visibleTokens.size > 0) || hasStructuredLabelText;
}

function selectLabelDrivenItemName(item) {
  const currentName = smartTitleCase(item?.item_name);
  const currentTokens = tokenSet(currentName);
  const candidates = uniqueStrings((item?.visible_text || []).map((value) => smartTitleCase(value)).filter(Boolean));

  let bestPhrase = null;
  let bestScore = -Infinity;
  for (const phrase of candidates) {
    const phraseTokens = tokenize(phrase);
    if (phraseTokens.length === 0) {
      continue;
    }

    const phraseTokenSet = new Set(phraseTokens);
    const hintHits = phraseTokens.filter((token) => QUICK_GLANCE_LABEL_HINT_TOKENS.has(token)).length;
    const nonProductHits = phraseTokens.filter((token) => QUICK_GLANCE_NON_PRODUCT_TOKENS.has(token)).length;
    const overlap = overlapCount(phraseTokenSet, currentTokens);
    let score = Math.min(phraseTokens.length, 5) * 2 + hintHits * 3 + overlap - nonProductHits * 2;
    if (phraseTokens.length === 1 && hintHits === 0) score -= 3;
    if (phraseTokens.every((token) => QUICK_GLANCE_NON_PRODUCT_TOKENS.has(token))) score -= 6;
    if (score > bestScore) {
      bestScore = score;
      bestPhrase = phrase;
    }
  }

  if (!bestPhrase || bestScore < 4) {
    return currentName;
  }
  if (!currentName) {
    return bestPhrase;
  }

  const overlap = overlapCount(tokenSet(bestPhrase), currentTokens);
  const currentLooksGeneric = isGenericTitle(currentName) || currentTokens.size <= 1;
  if (currentLooksGeneric || (item?.item_type === "packaged" && overlap === 0)) {
    return bestPhrase;
  }
  return currentName;
}

function normalizeQuickGlanceItem(item) {
  const normalized = normalizeBaseItem(item);
  normalized.item_type = hasQuickGlancePackagingCues(normalized) ? "packaged" : classifyItemType(normalized);
  normalized.item_name = selectLabelDrivenItemName(normalized) || buildFallbackName(normalized);
  normalized.item_name = ensureDetailedItemName(normalized.item_name, normalized.brand);
  return appendUncertaintySignals(normalized);
}

function normalizeLegacyFastItem(item) {
  const normalized = normalizeBaseItem(item);
  normalized.item_type = classifyItemType(normalized);
  normalized.count = 1;
  return appendUncertaintySignals(normalized);
}

function normalizeItem(item) {
  const normalized = normalizeBaseItem(item);

  normalized.item_type = classifyItemType(normalized);
  return appendUncertaintySignals(applyCatalogMatch(normalized));
}

function specificityScore(item) {
  let score = 0;
  if (item.brand) score += 4;
  if (item.item_name && !isGenericTitle(item.item_name)) score += 6;
  if (item.category) score += 2;
  if (item.position_hint) score += 1;
  score += Math.min((item.visible_text || []).length, 4);
  score += Math.round((item.confidence || 0) * 3);
  return score;
}

function positionsConflict(a, b) {
  const aTokens = tokenSet(a);
  const bTokens = tokenSet(b);
  if (aTokens.size === 0 || bTokens.size === 0) return false;
  const horizontalConflict = (aTokens.has("left") && bTokens.has("right")) || (aTokens.has("right") && bTokens.has("left"));
  const verticalConflict = (aTokens.has("top") && bTokens.has("bottom")) || (aTokens.has("bottom") && bTokens.has("top"));
  return horizontalConflict || verticalConflict;
}

function positionsCompatible(a, b) {
  const aTokens = tokenSet(a);
  const bTokens = tokenSet(b);
  if (aTokens.size === 0 || bTokens.size === 0) return true;
  if (positionsConflict(a, b)) return false;
  return overlapCount(aTokens, bTokens) > 0 || aTokens.has("center") || bTokens.has("center");
}

function sameOrMissing(a, b) {
  const normalizedA = normalizeForCompare(a);
  const normalizedB = normalizeForCompare(b);
  if (!normalizedA || !normalizedB) return true;
  return normalizedA === normalizedB;
}

function titlesSimilar(a, b) {
  const aTokens = tokenSet(a);
  const bTokens = tokenSet(b);
  if (aTokens.size === 0 || bTokens.size === 0) return false;
  const overlap = overlapCount(aTokens, bTokens);
  return overlap >= Math.min(aTokens.size, bTokens.size) || overlap >= 2;
}

function visibleTextOverlap(a, b) {
  const aSet = new Set((a.visible_text || []).flatMap((value) => tokenize(value)));
  const bSet = new Set((b.visible_text || []).flatMap((value) => tokenize(value)));
  return overlapCount(aSet, bSet);
}

function choosePreferredItem(a, b) {
  return specificityScore(a) >= specificityScore(b) ? a : b;
}

function mergeDuplicatePair(a, b) {
  const preferred = choosePreferredItem(a, b);
  const secondary = preferred === a ? b : a;
  const merged = {
    ...preferred,
    brand: preferred.brand || secondary.brand,
    category: preferred.category || secondary.category,
    item_type: preferred.item_type !== "unknown" ? preferred.item_type : secondary.item_type,
    count: Math.max(preferred.count || 1, secondary.count || 1),
    position_hint: preferred.position_hint || secondary.position_hint,
    confidence: Math.max(preferred.confidence || 0, secondary.confidence || 0),
    needs_review: Boolean(preferred.needs_review || secondary.needs_review),
    uncertainty_reasons: uniqueStrings([...(preferred.uncertainty_reasons || []), ...(secondary.uncertainty_reasons || [])]).slice(0, 6),
    short_description:
      (preferred.short_description || "").length >= (secondary.short_description || "").length
        ? preferred.short_description
        : secondary.short_description,
    visible_text: uniqueStrings([...(preferred.visible_text || []), ...(secondary.visible_text || [])]).slice(0, 8),
  };
  merged.item_name = ensureDetailedItemName(preferred.item_name || secondary.item_name, merged.brand);
  return appendUncertaintySignals(applyCatalogMatch(merged));
}

function areLikelyDuplicateObjects(a, b) {
  let score = 0;
  if (positionsCompatible(a.position_hint, b.position_hint)) score += 2;
  if (sameOrMissing(a.brand, b.brand)) score += 2;
  if (sameOrMissing(a.category, b.category)) score += 1;
  if (titlesSimilar(a.item_name, b.item_name)) score += 2;
  if (visibleTextOverlap(a, b) > 0) score += 2;
  if ((isGenericTitle(a.item_name) && !isGenericTitle(b.item_name)) || (!isGenericTitle(a.item_name) && isGenericTitle(b.item_name))) {
    score += 1;
  }
  if (positionsConflict(a.position_hint, b.position_hint)) score -= 3;
  return score >= 5;
}

function dedupeLikelyDuplicateObjects(items) {
  const deduped = [];
  for (const item of items) {
    const existingIndex = deduped.findIndex((candidate) => areLikelyDuplicateObjects(candidate, item));
    if (existingIndex === -1) {
      deduped.push(item);
    } else {
      deduped[existingIndex] = mergeDuplicatePair(deduped[existingIndex], item);
    }
  }
  return deduped;
}

function buildGroupingKey(item) {
  const identity = [normalizeForCompare(item.brand), normalizeForCompare(item.item_name), normalizeForCompare(item.category)].join("|");
  if (!isGenericTitle(item.item_name)) {
    return identity;
  }
  const textKey = (item.visible_text || []).map(normalizeForCompare).slice(0, 2).join("|");
  return `${identity}|${textKey}`;
}

function mergeSameTypeGroup(items) {
  if (items.length === 1) {
    return items[0];
  }
  const preferred = items.reduce((best, current) => (specificityScore(current) >= specificityScore(best) ? current : best));
  const positions = uniqueStrings(items.map((item) => item.position_hint).filter(Boolean));
  const totalCount = items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0);
  const merged = {
    ...preferred,
    count: totalCount,
    confidence: Math.max(...items.map((item) => item.confidence || 0)),
    visible_text: uniqueStrings(items.flatMap((item) => item.visible_text || [])).slice(0, 8),
    position_hint: positions.length === 1 ? positions[0] : positions.length > 1 ? "Multiple Positions" : preferred.position_hint,
    needs_review: items.some((item) => item.needs_review),
    uncertainty_reasons: uniqueStrings(items.flatMap((item) => item.uncertainty_reasons || [])).slice(0, 6),
  };
  if (totalCount > 1 && merged.short_description) {
    merged.short_description = `${merged.short_description.replace(/\.$/, "")}. ${totalCount} visible in the image.`;
  }
  return appendUncertaintySignals(applyCatalogMatch(merged));
}

function collapseRepeatedSameType(items) {
  const groups = new Map();
  for (const item of items) {
    const key = buildGroupingKey(item);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(item);
  }
  return Array.from(groups.values()).map(mergeSameTypeGroup);
}

function applyBulkInventoryConservatism(items) {
  return items.map((item) => {
    const next = {
      ...item,
      uncertainty_reasons: [...(item.uncertainty_reasons || [])],
    };
    const visibleTextCount = (next.visible_text || []).filter(Boolean).length;
    const packaged = classifyItemType(next) === "packaged";
    const genericPackagedTitle = packaged && isGenericTitle(next.item_name);

    if (packaged && !next.brand && visibleTextCount === 0) {
      next.needs_review = true;
      next.confidence = Math.min(Number(next.confidence || 0), 0.62);
      next.uncertainty_reasons = uniqueStrings([...next.uncertainty_reasons, "limited_label_evidence"]);
    }

    if (genericPackagedTitle) {
      next.needs_review = true;
      next.confidence = Math.min(Number(next.confidence || 0), 0.58);
      next.uncertainty_reasons = uniqueStrings([...next.uncertainty_reasons, "generic_packaged_title"]);
    }

    if (Number(next.confidence || 0) < 0.55) {
      next.needs_review = true;
      next.uncertainty_reasons = uniqueStrings([...next.uncertainty_reasons, "low_scene_confidence"]);
    }

    return appendUncertaintySignals(applyCatalogMatch(next));
  });
}

function rankFinalItems(items) {
  return [...items].sort((a, b) => {
    const countDelta = Math.max(b.count || 1, 1) - Math.max(a.count || 1, 1);
    if (countDelta !== 0) {
      return countDelta;
    }

    const confidenceDelta = (b.confidence || 0) - (a.confidence || 0);
    if (confidenceDelta !== 0) {
      return confidenceDelta;
    }

    return specificityScore(b) - specificityScore(a);
  });
}

function buildFallbackItems(mode = "fast") {
  return [
    {
      item_name: mode === "bulk" ? "Unknown Grocery Item" : "Unknown Item",
      brand: null,
      category: null,
      item_type: "unknown",
      count: 1,
      position_hint: null,
      confidence: 0,
      needs_review: true,
      uncertainty_reasons: ["fallback_item"],
      short_description:
        mode === "bulk"
          ? "Bulk inventory scan could not confidently identify any visible grocery items."
          : "Quick best-effort item identification.",
      visible_text: [],
    },
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeRegion(region) {
  const parsed = RegionSchema.parse(region);
  const x = clamp(parsed.x, 0, 0.98);
  const y = clamp(parsed.y, 0, 0.98);
  const width = clamp(parsed.width, 0.02, 1 - x);
  const height = clamp(parsed.height, 0.02, 1 - y);
  return {
    label_guess: cleanNullable(parsed.label_guess),
    position_hint: normalizePositionHint(parsed.position_hint),
    count_estimate: parsed.count_estimate,
    confidence: parsed.confidence,
    is_group: parsed.is_group,
    x,
    y,
    width,
    height,
  };
}

function normalizeDetectedRegions(payload) {
  const parsed = RegionResponseSchema.parse(payload);
  return parsed.regions
    .map(normalizeRegion)
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .slice(0, MAX_REGIONS);
}

function buildImageContentFromUrl(imageUrl) {
  return {
    type: "input_image",
    image_url: imageUrl,
    detail: "original",
  };
}

function buildDataUrl(buffer, mimeType) {
  return `data:${mimeType || "image/jpeg"};base64,${buffer.toString("base64")}`;
}

function buildImageContentFromBuffer(buffer, mimeType) {
  return {
    type: "input_image",
    image_url: buildDataUrl(buffer, mimeType),
    detail: "original",
  };
}

function normalizeImageUrl(rawUrl) {
  const cleaned = cleanNullable(rawUrl)?.replace(/^['"]+|['"]+$/g, "").replace(/&amp;/gi, "&");
  if (!cleaned) {
    throw new Error("Image URL is empty");
  }

  const withProtocol = cleaned.startsWith("//") ? `https:${cleaned}` : cleaned;

  let parsed;
  try {
    parsed = new URL(withProtocol);
  } catch {
    throw new Error("Invalid image URL");
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Image URL must use http or https");
  }

  return parsed.toString();
}

function extractOutputText(response) {
  if (typeof response?.output_text === "string" && response.output_text.trim()) {
    return response.output_text;
  }

  const texts = [];
  const outputs = Array.isArray(response?.output) ? response.output : [];
  for (const output of outputs) {
    const contents = Array.isArray(output?.content) ? output.content : [];
    for (const content of contents) {
      if (typeof content?.text === "string" && content.text.trim()) {
        texts.push(content.text);
      }
    }
  }

  return texts.join("\n").trim();
}

function parseStructuredResponse(response) {
  try {
    return parseJsonResponse(response);
  } catch (error) {
    const outputText = extractOutputText(response);
    if (outputText) {
      return parseJsonResponse({ output_text: outputText });
    }
    throw error;
  }
}

function isRetryableStructuredError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("No structured output returned from OpenAI") ||
    message.includes("Failed to parse structured OpenAI output") ||
    message.includes("Rate limit") ||
    message.includes("temporarily unavailable")
  );
}

async function createStructuredResponse(requestFactory, parser, attempts, timeoutMs) {
  const openai = getOpenAIClient();
  let lastError = null;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await openai.responses.create(requestFactory(attempt), { timeout: timeoutMs });
      return parser(response);
    } catch (error) {
      lastError = error;
      if (!isRetryableStructuredError(error) || attempt === attempts - 1) {
        throw error;
      }
      await sleep(400 * (attempt + 1));
    }
  }

  throw lastError || new Error("Structured response failed");
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

  const normalizedUrl = normalizeImageUrl(input.imageUrl);
  const response = await fetch(normalizedUrl, {
    headers: { "user-agent": "Mozilla/5.0" },
  });
  if (!response.ok) {
    throw new Error(`Failed to download image from URL: HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.startsWith("image/")) {
    throw new Error("URL does not point to an image file");
  }

  return {
    buffer: await response.buffer(),
    mimeType: contentType,
  };
}

function isBulkInventoryConfig(config) {
  return config?.mode === "bulk";
}

function getBulkSceneExamples() {
  return "milk cartons, yogurt cups, eggs, butter, shredded cheese, salad kits, tortillas, deli meat, condiments, jam jars, sauces, canned goods, cereal, pasta, oats, bread, produce drawers, fruit bowls, and freezer bags.";
}

function buildOcrPromptContext(ocrContext) {
  if (!ocrContext?.usedGoogleVision) {
    return null;
  }

  const sections = [];
  if (ocrContext.promptText) {
    sections.push(`OCR text extracted from the image:\n${ocrContext.promptText}`);
  }
  if (Array.isArray(ocrContext.objectNames) && ocrContext.objectNames.length > 0) {
    sections.push(`Google Vision object localization hints: ${ocrContext.objectNames.join(", ")}`);
  }
  return sections.join("\n\n");
}

async function detectRegions(imageAsset, config) {
  const bulkMode = isBulkInventoryConfig(config);
  return createStructuredResponse(
    (attempt) => ({
      model: config.model,
      ...(getReasoningConfig(config.model, attempt === 0 ? "medium" : "low")
        ? { reasoning: getReasoningConfig(config.model, attempt === 0 ? "medium" : "low") }
        : {}),
      max_output_tokens: bulkMode ? 1800 : 1400,
      text: {
        format: {
          type: "json_schema",
          name: "item_regions",
          strict: true,
          schema: regionJsonSchema,
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text:
                bulkMode
                  ? `You are an item-region detector for dense fridge, pantry, and grocery shelf scenes. Return strict JSON only. Find every distinct visible grocery item or grouped cluster that a shopper would list separately: produce, cartons, bags, boxes, cans, tubs, jars, bottles, condiments, dairy, deli packs, frozen foods, bread, tortillas, and pantry staples. Door-bin items, partially occluded labels, and stacked shelf products still count when clearly visible. If multiple identical loose items are together, return one region with count_estimate > 1. Avoid duplicate regions for the same physical object, and do not return kitchenware unless it clearly contains a grocery item group. Use normalized coordinates from 0 to 1 for x, y, width, height.`
                  : "You are an item-region detector for crowded grocery scenes. Return strict JSON only. Find every distinct visible grocery item or grouped cluster that a shopper would list separately: produce, cartons, bags, boxes, cans, tubs, jars, bread, tortillas, eggs, dairy, frozen foods, and pantry staples. Separate overlapping packages when both are visible. Produce in a bowl, bag, or bunch still counts as a visible item group. If multiple identical loose items are together, return one region with count_estimate > 1. Avoid duplicate regions for the same physical object. Use normalized coordinates from 0 to 1 for x, y, width, height.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                attempt === 0
                  ? bulkMode
                    ? `Detect the distinct visible grocery item regions in this fridge, pantry, or shelf image. Be exhaustive for ${getBulkSceneExamples()} Include clearly visible repeated products, produce groups, and condiment bottles. Separate adjacent products when both are visible, but group identical loose produce with a count estimate.`
                    : "Detect the distinct visible grocery item regions in this image. Be exhaustive for produce, bread, tortillas, pasta, oats, rice cakes, milk, eggs, yogurt, shredded cheese, peanut butter, canned goods, boxed meals, and frozen potato products. Do not duplicate the same object. For each region, provide a best-guess label, position hint, count estimate, confidence, whether it is a grouped item region, and the normalized bounding box."
                  : "Return only valid JSON for visible item regions. If unsure, return fewer but high-confidence regions rather than malformed output or duplicate regions.",
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => normalizeDetectedRegions(parseStructuredResponse(response)),
    config.structuredRetryAttempts,
    config.regionDetectTimeoutMs
  );
}

async function cropRegion(imageAsset, region) {
  const image = await Jimp.read(imageAsset.buffer);
  const width = image.bitmap?.width || 0;
  const height = image.bitmap?.height || 0;
  if (!width || !height) {
    throw new Error("Could not determine image dimensions for crop");
  }

  const padX = Math.round(region.width * width * CROP_PADDING_RATIO);
  const padY = Math.round(region.height * height * CROP_PADDING_RATIO);

  const left = clamp(Math.round(region.x * width) - padX, 0, width - 1);
  const top = clamp(Math.round(region.y * height) - padY, 0, height - 1);
  const cropWidth = clamp(Math.round(region.width * width) + padX * 2, 32, width - left);
  const cropHeight = clamp(Math.round(region.height * height) + padY * 2, 32, height - top);

  const croppedImage = image.clone().crop({
    x: left,
    y: top,
    w: cropWidth,
    h: cropHeight,
  });
  const cropped = await croppedImage.getBuffer("image/jpeg");

  return {
    buffer: cropped,
    mimeType: "image/jpeg",
  };
}

async function identifyRegion(regionAsset, region, regionIndex, regionCount, config) {
  const bulkMode = isBulkInventoryConfig(config);
  const normalized = normalizeItem(
    await createStructuredResponse(
      (attempt) => ({
        model: config.model,
        ...(getReasoningConfig(config.model, attempt === 0 ? "medium" : "low")
          ? { reasoning: getReasoningConfig(config.model, attempt === 0 ? "medium" : "low") }
          : {}),
        max_output_tokens: bulkMode ? 650 : 500,
        text: {
          format: {
            type: "json_schema",
            name: "identified_item",
            strict: true,
            schema: itemJsonSchema,
          },
        },
        input: [
          {
            role: "system",
            content: [
              {
                type: "input_text",
                text:
                  bulkMode
                    ? "You identify one crop from a larger bulk grocery scene. Return strict JSON only. Prefer a detailed properly capitalized grocery title with brand when visible. Treat this crop as part of a fridge, pantry, or shelf inventory sweep. Read partial labels aggressively, but if the crop is ambiguous mark needs_review true instead of overclaiming. If the crop contains produce in bulk, identify the produce type and approximate visible count. Distinguish similar shelf items like oats vs granola, tortillas vs wraps, shredded cheese vs yogurt, condiments vs sauces, and jars vs canned goods. Do not invent brands or merge different overlapping products into one if two distinct grocery items are visible. Always include item_type as one of produce, packaged, or unknown, plus uncertainty_reasons as short machine-readable strings."
                    : "You identify one crop from a larger grocery scene. Return strict JSON only. Prefer a detailed properly capitalized grocery title with brand when visible. If the crop contains produce in bulk, identify the produce type and approximate visible count. Use packaging text aggressively when readable. Distinguish similar pantry items like oats vs rice cakes, shredded cheese vs yogurt, or bread vs tortillas. Do not invent brands or merge different overlapping products into one if two distinct grocery items are visible. Always include item_type as one of produce, packaged, or unknown. Always include needs_review as a boolean and uncertainty_reasons as an array of short machine-readable strings.",
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "input_text",
                text:
                  `This is crop ${regionIndex + 1} of ${regionCount} from a larger scene. Detector hint: ${JSON.stringify({
                    label_guess: region.label_guess,
                    position_hint: region.position_hint,
                    count_estimate: region.count_estimate,
                    is_group: region.is_group,
                    confidence: region.confidence,
                  })}. Identify the main item or item group visible inside this crop only. Do not describe background objects outside the crop. ${
                    attempt === 0
                      ? bulkMode
                        ? "Give the best title, brand if visible, category, item_type, quantity within this crop, confidence, visible text, needs_review, and uncertainty_reasons. If this looks like one of several repeated identical products, set count to the visible number in this crop rather than creating multiple slightly different items."
                        : "Give the best title, brand if visible, category, item_type, quantity within this crop, confidence, visible text, needs_review, and uncertainty_reasons."
                      : "Return only valid JSON. If unsure, give one conservative best-guess item instead of malformed output, and use needs_review true."
                  }`,
              },
              buildImageContentFromBuffer(regionAsset.buffer, regionAsset.mimeType),
            ],
          },
        ],
      }),
      (response) => parseStructuredResponse(response),
      config.structuredRetryAttempts,
      config.regionIdentifyTimeoutMs
    )
  );
  return {
    ...normalized,
    count: Math.max(normalized.count || 1, region.count_estimate || 1),
    position_hint: normalized.position_hint || region.position_hint || null,
    confidence: Math.max(normalized.confidence || 0, region.confidence || 0),
  };
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const results = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await mapper(items[currentIndex], currentIndex);
    }
  }

  const workerCount = Math.min(concurrency, items.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results.filter(Boolean);
}

async function identifyByRegions(imageAsset, config) {
  const detectedRegions = await detectRegions(imageAsset, config);
  if (detectedRegions.length === 0) {
    return [];
  }

  const prioritizedRegions = detectedRegions.slice(0, config.maxRegionIdentifications);
  return mapWithConcurrency(prioritizedRegions, config.identifyConcurrency, async (region, index) => {
    try {
      const cropAsset = await cropRegion(imageAsset, region);
      return await identifyRegion(cropAsset, region, index, prioritizedRegions.length, config);
    } catch (error) {
      console.warn("[QuickIdentify] Region identification failed:", error);
      return null;
    }
  });
}

async function identifyWholeSceneFallback(imageAsset, config, ocrContext = null) {
  const bulkMode = isBulkInventoryConfig(config);
  const ocrPromptContext = buildOcrPromptContext(ocrContext);
  const parsed = await createStructuredResponse(
    (attempt) => ({
      model: config.model,
      ...(getReasoningConfig(config.model, "low") ? { reasoning: getReasoningConfig(config.model, "low") } : {}),
      max_output_tokens: bulkMode ? 1800 : 900,
      text: {
        format: {
          type: "json_schema",
          name: "fallback_items",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["items"],
            properties: {
              items: {
                type: "array",
                maxItems: config.maxOutputItems,
                items: itemJsonSchema,
              },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text:
                bulkMode
                  ? "You identify all visible items in a dense fridge, pantry, or shelf grocery scene. Return strict JSON only. Be as specific as possible, include brands when visible, group identical repeated items by count, and avoid duplicate detections of the same object. Count only items that are visibly present, not assumed from hidden shelf stock. You may also receive OCR text from Google Vision; use it aggressively for brand names, label words, and dates, but only when the text plausibly matches something visible in the image. If an item is partly visible but plausible, keep it only when there is enough visual evidence and mark needs_review true when evidence is incomplete. For every item include item_type as produce, packaged, or unknown, plus needs_review and uncertainty_reasons."
                  : "You identify all visible items in a grocery scene. Return strict JSON only. Be as specific as possible, include brands when visible, group identical repeated items by count, and avoid duplicate detections of the same object. For every item include item_type as produce, packaged, or unknown, plus needs_review and uncertainty_reasons.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                attempt === 0
                  ? bulkMode
                    ? `Identify all visible distinct grocery items in this image as a bulk inventory scan. Be exhaustive for ${getBulkSceneExamples()} Include partially occluded but clearly recognizable items, door-bin condiments, and repeated products with counts. Do not list non-grocery containers unless the grocery itself is visible. Use needs_review for ambiguous labels instead of pretending certainty.${ocrPromptContext ? `\n\n${ocrPromptContext}` : ""}`
                    : config.mode === "deep"
                      ? "Identify all visible distinct grocery items in this image as a full-scene inventory. Be exhaustive for produce, bread, tortillas, pasta, oats, rice cakes, milk, eggs, yogurt, shredded cheese, canned foods, peanut butter, boxed macaroni and cheese, and frozen potato products. Include partially occluded but clearly visible items. Group identical repeated items by count and avoid duplicates. Include item_type, needs_review, and uncertainty_reasons for each item."
                    : "Identify all visible distinct grocery items in this image. Be exhaustive for produce, bread, tortillas, pasta, oats, rice cakes, milk, eggs, yogurt, shredded cheese, canned foods, peanut butter, boxed macaroni and cheese, and frozen potato products. If uncertain, return fewer but better items rather than duplicating the same object twice. Include item_type, needs_review, and uncertainty_reasons for each item."
                  : "Return only valid JSON. If uncertain, give a smaller set of high-confidence items rather than malformed or empty output.",
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => parseStructuredResponse(response),
    config.structuredRetryAttempts,
    config.wholeSceneTimeoutMs
  );

  return z.object({ items: z.array(FastItemSchema).default([]) }).parse(parsed).items.map(normalizeItem);
}

async function identifyLegacyFast(imageAsset, userHint, leftovers) {
  const model = getQuickIdentifyModel();
  let systemText = "You are the original ultra-fast grocery identifier. Return strict JSON only. Identify the single main grocery item visible in the image. Prioritize the most obvious foreground item and ignore background objects. Keep the answer concise. If the image is unclear, return one conservative best guess and set needs_review to true. Include brand and category only when reasonably visible.";
  if (userHint) {
    systemText += `\n\nCRITICAL OVERRIDE: The user manually described this item as: "${userHint}". This note tells you what the item ACTUALLY IS, regardless of what the image shows. The user may be photographing food inside a container, a reusable bottle, a Tupperware, or a bag — the container is NOT the product. Use the user's description as the item_name. For example, if the image shows a Hydro Flask but the user says "smoothie", the item is a smoothie, NOT a Hydro Flask. The user's note always takes priority over visible branding or packaging.`;
  }
  else if (leftovers) {
    systemText += `\n\nLEFTOVERS MODE: This is a photo of leftover FOOD or DRINK the user is saving — NOT a packaged grocery product. Name the item by what the food/drink actually IS, as specifically as the image allows (e.g. 'Black Coffee', 'Chicken Fried Rice', 'Half a Burrito'). A confident contextual guess beats a generic label — e.g. a Starbucks cup is 'Black Coffee' (or 'Iced Coffee' etc), not 'Prepared Food'. NEVER use 'Prepared Food', 'Leftovers', or 'Prepared Food Leftovers' as the item_name — the item is already tagged as a leftover elsewhere. Put container details (cup, tupperware) in variant/description, not the name. Only if the contents are truly unidentifiable, use a best-effort descriptive name like 'Mixed Leftover Meal'.`;
  }
  const parsed = await createStructuredResponse(
    () => ({
      model,
      ...(getReasoningConfig(model, "low") ? { reasoning: getReasoningConfig(model, "low") } : {}),
      max_output_tokens: 220,
      text: {
        format: {
          type: "json_schema",
          name: "fast_item_identification",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["items"],
            properties: {
              items: {
                type: "array",
                maxItems: 1,
                items: itemJsonSchema,
              },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text: systemText,
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                "What is the main item in this image? Return one best-guess grocery item with item_name, brand, category, confidence, short_description, visible_text, item_type, needs_review, and uncertainty_reasons.",
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => parseStructuredResponse(response),
    FAST_LEGACY_ATTEMPTS,
    FAST_LEGACY_TIMEOUT_MS
  );

  const items = z.object({ items: z.array(FastItemSchema).default([]) }).parse(parsed).items.map(normalizeLegacyFastItem);
  return items.length > 0 ? [items[0]] : buildFallbackItems();
}

function applyQuickGlanceConservatism(items) {
  const gumLikeTokens = ["eclipse", "wrigley", "wrigley's", "extra", "orbit", "trident", "gum", "chewing", "spearmint", "peppermint", "mint", "mints"];
  const frozenPotatoTokens = ["tater", "tots", "potato", "potatoes", "fries", "hashbrown", "hash", "brown"];

  return items.map((item) => {
    const next = {
      ...item,
      uncertainty_reasons: [...(item.uncertainty_reasons || [])],
    };
    const visibleTokens = tokenSet((item.visible_text || []).join(" "));
    const combinedTokens = tokenSet([item.item_name, item.brand, item.category].filter(Boolean).join(" "));
    const packaged = classifyItemType(item) === "packaged";

    if (packaged && visibleTokens.size === 0) {
      next.needs_review = true;
      next.confidence = Math.min(Number(next.confidence || 0), 0.72);
      next.uncertainty_reasons = uniqueStrings([...next.uncertainty_reasons, "limited_packaging_text"]);
    }

    if (
      (hasAnyToken(visibleTokens, gumLikeTokens) || hasAnyToken(combinedTokens, ["eclipse", "wrigley", "wrigley's", "extra", "orbit", "trident"])) &&
      hasAnyToken(combinedTokens, frozenPotatoTokens)
    ) {
      next.needs_review = true;
      next.confidence = Math.min(Number(next.confidence || 0), 0.35);
      next.uncertainty_reasons = uniqueStrings([...next.uncertainty_reasons, "brand_category_mismatch"]);
    }

    return appendUncertaintySignals(next);
  });
}

async function identifyQuickGlance(imageAsset) {
  const model = getQuickIdentifyModel();
  const parsed = await createStructuredResponse(
    (attempt) => ({
      model,
      ...(getReasoningConfig(model, "low") ? { reasoning: getReasoningConfig(model, "low") } : {}),
      max_output_tokens: 400,
      text: {
        format: {
          type: "json_schema",
          name: "quick_glance_items",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["items"],
            properties: {
              items: {
                type: "array",
                maxItems: 1,
                items: itemJsonSchema,
              },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text:
                "You are the fast first-pass grocery identifier for check-in and discard. Return strict JSON only. Identify the single main foreground item the user is most likely holding up. For packaged items, item_name must be the front-of-pack product title the user would recognize, using readable label text over ingredient guesses. Example: if a bag says coleslaw mix, return Coleslaw Mix, not Spinach or Lettuce. If a tub says Greek Yogurt, return Greek Yogurt. Use readable packaging text and logos aggressively when they are visible, and copy the strongest product words into visible_text. If the label text is only partly readable, give the closest conservative package name supported by the text and visible packaging, mark needs_review=true, and avoid rewriting packaged salad mixes, kits, or slaws into a loose produce type unless that produce word is clearly the package title. Do not guess frozen potato products unless potato nuggets, fries, or words like tater, tots, potato, or fries are actually visible. Include visible_text, item_type, needs_review, and uncertainty_reasons.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                attempt === 0
                  ? "Give the quickest best-effort identification of the single main grocery item in this image. Prefer the exact package title or closest readable label-driven product name over a generic ingredient guess. Read packaging text and logos aggressively, and include the strongest product words in visible_text. If the item is a packaged salad or produce bag, use the bag title such as coleslaw mix, salad mix, or baby spinach if that wording is visible. If readable text is insufficient, keep the guess conservative and mark needs_review true."
                  : "Return only valid JSON with a conservative quick-glance item list.",
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => parseStructuredResponse(response),
    QUICK_GLANCE_ATTEMPTS,
    QUICK_GLANCE_TIMEOUT_MS
  );

  return applyQuickGlanceConservatism(
    z.object({ items: z.array(FastItemSchema).default([]) }).parse(parsed).items.map(normalizeQuickGlanceItem)
  );
}

async function reviewSceneItems(imageAsset, provisionalItems, config, ocrContext = null) {
  if (provisionalItems.length === 0) {
    return [];
  }

  const bulkMode = isBulkInventoryConfig(config);
  const ocrPromptContext = buildOcrPromptContext(ocrContext);
  const parsed = await createStructuredResponse(
    (attempt) => ({
      model: config.model,
      ...(getReasoningConfig(config.model, attempt === 0 ? "medium" : "low")
        ? { reasoning: getReasoningConfig(config.model, attempt === 0 ? "medium" : "low") }
        : {}),
      max_output_tokens: bulkMode ? 2200 : 1800,
      text: {
        format: {
          type: "json_schema",
          name: "reviewed_scene_items",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["items"],
            properties: {
              items: {
                type: "array",
                maxItems: config.maxOutputItems,
                items: itemJsonSchema,
              },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text:
                bulkMode
                  ? `You reconcile detections for a bulk grocery inventory scene. Return strict JSON only. Start from the provisional item list, but treat it only as hints. Correct wrong titles, remove false positives, merge duplicates, fix counts, and add obvious missed grocery items visible in the full image. Produce a final inventory of all visible grocery item types. Prefer specific grocery names and brands when visible. You may also receive OCR text from Google Vision; use it aggressively for label words, brand names, and dates, but only if the text plausibly maps to visible items in the image. Mark needs_review true whenever the evidence is partial or similar products are hard to separate. Watch carefully for ${getBulkSceneExamples()} Every item must include item_type, needs_review, and uncertainty_reasons.`
                  : "You reconcile grocery detections for a full scene. Return strict JSON only. Start from the provisional item list, but treat it only as hints. Correct wrong titles, remove false positives, merge duplicates, fix counts, and add any obvious missed grocery items visible in the full image. Produce a final grocery inventory of all visible item types. Prefer specific grocery names and brands when visible. For common grocery scenes, watch carefully for grapes, bananas, leafy greens, broccoli florets, bread, tortillas, pasta, oats, milk, eggs, yogurt, shredded cheese, canned beans, peanut butter, rice cakes, boxed macaroni and cheese, and frozen potato products. Every item must include item_type, needs_review, and uncertainty_reasons.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                `${attempt === 0
                  ? bulkMode
                    ? `Review this full fridge, pantry, or shelf photo and the provisional detections below. Return the corrected final item list for a bulk inventory scan. Do not keep items that are not actually visible. Add obvious missed items. Group identical items with count. Prefer strong coverage, but if a label or category is ambiguous keep the item conservative and mark needs_review true.${ocrPromptContext ? `\n\n${ocrPromptContext}` : ""}`
                    : "Review this full grocery photo and the provisional detections below. Return the corrected final item list. Do not keep items that are not actually visible. Add obvious missed items. Group identical items with count. Prefer complete grocery coverage over overly conservative output when an item is clearly visible. Include item_type, needs_review, and uncertainty_reasons for every item."
                  : "Return only valid JSON for the corrected final grocery item list."}\n\nProvisional items:\n${JSON.stringify(
                  provisionalItems,
                  null,
                  2
                )}`,
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => parseStructuredResponse(response),
    config.structuredRetryAttempts,
    config.reviewTimeoutMs
  );

  return z.object({ items: z.array(FastItemSchema).default([]) }).parse(parsed).items.map(normalizeItem);
}

async function findMissingItems(imageAsset, currentItems, config, ocrContext = null) {
  if (currentItems.length === 0) {
    return [];
  }

  const bulkMode = isBulkInventoryConfig(config);
  const ocrPromptContext = buildOcrPromptContext(ocrContext);
  const parsed = await createStructuredResponse(
    (attempt) => ({
      model: config.model,
      ...(getReasoningConfig(config.model, attempt === 0 ? "medium" : "low")
        ? { reasoning: getReasoningConfig(config.model, attempt === 0 ? "medium" : "low") }
        : {}),
      max_output_tokens: 1200,
      text: {
        format: {
          type: "json_schema",
          name: "missing_scene_items",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["items"],
            properties: {
              items: {
                type: "array",
                maxItems: config.missingItemsMaxOutput,
                items: itemJsonSchema,
              },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text:
                bulkMode
                  ? `You look for clearly visible grocery items that are missing from an existing bulk inventory. Return strict JSON only. Only return new missing items that are visibly present in the image and not already in the inventory. You may also receive OCR text from Google Vision; use it to confirm brands or labels for items that are actually visible, not to invent hidden items. Do not repeat current items. Be especially careful with ${getBulkSceneExamples()} If a possible item is too ambiguous, skip it instead of guessing. Every item must include item_type, needs_review, and uncertainty_reasons.`
                  : "You look for clearly visible grocery items that are missing from an existing inventory. Return strict JSON only. Only return new missing items that are visibly present in the image and not already in the inventory. Do not repeat current items. Be especially careful with bananas, milk, yogurt multipacks, rice cakes, frozen potato products, boxed macaroni and cheese, broccoli florets, grapes, canned beans, peanut butter, and shredded cheese. Every item must include item_type, needs_review, and uncertainty_reasons.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                `${attempt === 0
                  ? bulkMode
                    ? `Review the full bulk grocery image and the current inventory below. Return only clearly visible missing grocery items that should be added. Use this pass to catch door-bin items, partly occluded shelf products, or produce groups the earlier passes missed. If nothing obvious is missing, return an empty items array.${ocrPromptContext ? `\n\n${ocrPromptContext}` : ""}`
                    : "Review the full grocery image and the current inventory below. Return only clearly visible missing grocery items that should be added. If nothing obvious is missing, return an empty items array."
                  : "Return only valid JSON for clearly visible missing grocery items."}\n\nCurrent inventory:\n${JSON.stringify(
                  currentItems,
                  null,
                  2
                )}`,
            },
            buildImageContentFromBuffer(imageAsset.buffer, imageAsset.mimeType),
          ],
        },
      ],
    }),
    (response) => parseStructuredResponse(response),
    config.structuredRetryAttempts,
    config.missingItemsTimeoutMs
  );

  return z.object({ items: z.array(FastItemSchema).default([]) }).parse(parsed).items.map(normalizeItem);
}

function refineHybridItems(items) {
  return items.map((item) => appendUncertaintySignals(applyCatalogMatch(item)));
}

function summarizeItemTypes(items) {
  return items.reduce(
    (summary, item) => {
      const type = classifyItemType(item);
      if (type === "produce") {
        summary.produce += 1;
      } else if (type === "packaged") {
        summary.packaged += 1;
      } else {
        summary.unknown += 1;
      }
      return summary;
    },
    { produce: 0, packaged: 0, unknown: 0 }
  );
}

function finalizeItems(items, config = {}) {
  const normalized = items.filter((item) => item && (item.item_name || item.brand || item.category));
  const refinedItems = refineHybridItems(normalized);
  const bulkAdjustedItems = config.mode === "bulk" ? applyBulkInventoryConservatism(refinedItems) : refinedItems;
  const dedupedItems = dedupeLikelyDuplicateObjects(bulkAdjustedItems);
  const collapsedItems = collapseRepeatedSameType(dedupedItems);
  const finalItems = config.mode === "bulk" ? applyBulkInventoryConservatism(collapsedItems) : collapsedItems;
  return finalItems.length > 0 ? rankFinalItems(finalItems) : buildFallbackItems(config.mode);
}

async function identifyItems(input, options = {}) {
  const config = getPipelineConfig(options.mode === "bulk" ? "bulk" : options.mode === "deep" ? "deep" : "fast");
  await reportStage(options, "loading_image", "Loading the grocery image.", 5, { mode: config.mode, model: config.model });
  const imageAsset = await loadImageAsset(input);
  const startedAt = Date.now();
  let ocrContext = null;

  if (config.mode === "bulk") {
    await reportStage(options, "reading_labels", "Reading labels and visible text with OCR.", 12);
    try {
      ocrContext = await extractGoogleVisionContext(imageAsset);
    } catch (error) {
      console.warn("[QuickIdentify:bulk] Google Vision OCR failed:", error);
    }
  }

  let regionItems = [];
  let sceneItems = [];
  await reportStage(options, "detecting_regions", "Detecting candidate grocery item regions.", 18);
  try {
    regionItems = await identifyByRegions(imageAsset, config);
  } catch (error) {
    console.warn(`[QuickIdentify:${config.mode}] Region pipeline failed, falling back to whole-scene analysis:`, error);
  }

  if (isDeepModelMode(config.mode)) {
    await reportStage(options, "full_scene_inventory", "Building a whole-scene grocery inventory.", 36);
    try {
      sceneItems = await identifyWholeSceneFallback(imageAsset, config, ocrContext);
    } catch (error) {
      console.warn(`[QuickIdentify:${config.mode}] Whole-scene inventory failed:`, error);
    }
  } else if (regionItems.length === 0) {
    regionItems = await identifyWholeSceneFallback(imageAsset, config, ocrContext);
  }

  let candidateItems =
    config.mode === "bulk" && sceneItems.length > 0
      ? [...sceneItems, ...regionItems]
      : regionItems.length > 0
        ? regionItems
        : sceneItems;
  if (candidateItems.length === 0) {
    await reportStage(options, "fallback_inventory", "Retrying with a conservative whole-scene inventory.", 42);
    candidateItems = await identifyWholeSceneFallback(imageAsset, config, ocrContext);
  }

  await reportStage(options, "routing_items", "Routing produce and packaged goods through specialist cleanup.", 52);
  candidateItems = finalizeItems(candidateItems, config);

  if (isDeepModelMode(config.mode) && candidateItems.length > 0) {
    const combinedDraftItems = finalizeItems([...(regionItems || []), ...(sceneItems || [])], config);
    await reportStage(options, "reconciling_inventory", "Reconciling crop detections with the full-scene inventory.", 68, {
      draft_counts: summarizeItemTypes(combinedDraftItems),
    });
    try {
      candidateItems = await reviewSceneItems(imageAsset, combinedDraftItems, config, ocrContext);
    } catch (error) {
      console.warn(`[QuickIdentify:${config.mode}] Scene review failed, using provisional region items:`, error);
      candidateItems = combinedDraftItems;
    }

    try {
      await reportStage(options, "finding_missing_items", "Sweeping for obvious missing grocery items.", 82);
      const missingItems = await findMissingItems(imageAsset, candidateItems, config, ocrContext);
      candidateItems = finalizeItems([...candidateItems, ...missingItems], config);
    } catch (error) {
      console.warn(`[QuickIdentify:${config.mode}] Missing-item sweep failed:`, error);
    }
  }

  const items = finalizeItems(candidateItems, config);
  const itemTypeCounts = summarizeItemTypes(items);
  await reportStage(options, "finalizing_inventory", "Finalizing the grocery inventory and uncertainty signals.", 96, {
    item_type_counts: itemTypeCounts,
  });
  return {
    items,
    debug: {
      elapsed_ms: Date.now() - startedAt,
      mode: config.publicMode || config.mode,
      model: config.model,
      item_count: items.length,
      item_type_counts: itemTypeCounts,
      review_required_count: items.filter((item) => item.needs_review).length,
      total_quantity: items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0),
      ocr_enabled: Boolean(ocrContext?.usedGoogleVision),
      ocr_line_count: ocrContext?.lineCount || 0,
      ocr_object_count: ocrContext?.objectCount || 0,
    },
  };
}

async function identifyItemFast(input, options = {}) {
  return identifyItems(input, { ...options, mode: "fast" });
}

async function identifyItemFastLegacy(input, options = {}) {
  const model = getQuickIdentifyModel();
  await reportStage(options, "fast_legacy_loading", "Loading image for the legacy fast path.", 5, { mode: "fast_legacy", model });
  const imageAsset = await loadImageAsset(input);
  const startedAt = Date.now();
  await reportStage(options, "fast_legacy_identifying", "Running the original single-pass fast identification.", 60);
  const items = await identifyLegacyFast(imageAsset, input.userHint, input.leftovers);
  await reportStage(options, "fast_legacy_done", "Legacy fast identification complete.", 100, {
    item_count: items.length,
  });
  return {
    items,
    debug: {
      elapsed_ms: Date.now() - startedAt,
      mode: "fast_legacy",
      model,
      item_count: items.length,
      review_required_count: items.filter((item) => item.needs_review).length,
      total_quantity: items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0),
    },
  };
}

async function identifyItemQuickGlance(input, options = {}) {
  const model = getQuickIdentifyModel();
  await reportStage(options, "quick_glance_loading", "Loading image for quick glance.", 5, { mode: "quick_glance", model });
  const imageAsset = await loadImageAsset(input);
  const startedAt = Date.now();
  await reportStage(options, "quick_glance_identifying", "Running a single-pass quick item identification.", 60);
  const items = finalizeItems(await identifyQuickGlance(imageAsset));
  await reportStage(options, "quick_glance_done", "Quick glance complete.", 100, {
    item_count: items.length,
  });
  return {
    items,
    debug: {
      elapsed_ms: Date.now() - startedAt,
      mode: "quick_glance",
      model,
      item_count: items.length,
      review_required_count: items.filter((item) => item.needs_review).length,
      total_quantity: items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0),
    },
  };
}

async function identifyItemDeep(input, options = {}) {
  return identifyItems(input, { ...options, mode: "deep" });
}

async function identifyItemBulkDeep(input, options = {}) {
  // Bulk fridge/pantry identify runs on Gemini (best at multi-item photos). If Gemini
  // fails (timeout/outage) after its retries, we already have the user's uploaded image
  // — so FAIL OVER to the OpenAI bulk pipeline rather than losing the whole scan. Only a
  // total Gemini outage reaches the catch; the common case still uses Gemini.
  try {
    return await identifyItemGeminiBulk(input, options);
  } catch (geminiError) {
    const msg = (geminiError && (geminiError.message || String(geminiError))) || 'unknown Gemini error';
    console.error(`[BulkIdentify] Gemini bulk failed, failing over to OpenAI bulk: ${msg}`);
    try {
      if (typeof options.onStage === 'function') {
        options.onStage({ stage: 'switching_engines', message: 'Switching engines to finish your scan…', progress: 45 });
      }
    } catch (_) { /* stage reporting is best-effort */ }
    // Marker so we can measure how often the fallback fires (grep: bulk_gemini_failover).
    try { console.log(JSON.stringify({ evt: 'bulk_gemini_failover', reason: msg.slice(0, 300) })); } catch (_) {}
    return await identifyItems(input, { ...options, mode: 'bulk' });
  }
}

async function identifyItemReceiptDeep(input, options = {}) {
  return identifyReceiptGeminiDeep(input, options);
}

module.exports = {
  identifyItemBulkDeep,
  identifyItemReceiptDeep,
  identifyItemDeep,
  identifyItemFast,
  identifyItemFastLegacy,
  identifyItemQuickGlance,
  identifyItems,
};
