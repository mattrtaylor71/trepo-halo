const { LambdaClient, InvokeCommand } = require("@aws-sdk/client-lambda");
const fetch = require("node-fetch");
const { buildCheckInLabel } = require("./quickIdentify/geminiBulk");
const { maybeEnrichPersistedBulkItem } = require("./bulkEnrichmentEngine");
const { estimateStorageGuidance } = require("./storageGuidance");

const KITCHEN_API_BASE_URL =
  (process.env.KITCHEN_API_BASE_URL || "https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com").replace(/\/$/, "");
const CREATE_RETRY_ATTEMPTS = 5;
// Cap how many kitchen creates fire at once. Committing a whole receipt (40+
// items) used to POST every item concurrently via Promise.all, which throttled
// the Kitchen API Lambda integration — API Gateway then bounced the excess with
// HTTP 503 and those confirmed items were silently dropped (job 485d2233:
// 29 of 41 persisted, 12 lost). Bounding concurrency keeps the burst under the
// throttle ceiling; the retry below recovers any that still bounce.
const CREATE_CONCURRENCY = Number(process.env.BULK_CREATE_CONCURRENCY || 5);
const lambdaClient = new LambdaClient({});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Transient HTTP statuses worth retrying: gateway throttle/unavailable + rate limit.
const RETRYABLE_HTTP_STATUSES = new Set([429, 500, 502, 503, 504]);

// Run an async mapper over items with a bounded number in flight at once.
// Preserves input order in the returned results array.
async function mapWithConcurrency(items, limit, mapper) {
  const results = new Array(items.length);
  let cursor = 0;
  const workers = new Array(Math.max(1, Math.min(limit, items.length))).fill(null).map(async () => {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      results[index] = await mapper(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

// Fast deterministic emoji assignment — no LLM needed
const CATEGORY_EMOJI_MAP = {
  produce: "🥬", fruits: "🍎", vegetables: "🥦", "fresh produce": "🥬",
  dairy: "🥛", "dairy & eggs": "🥛", "dairy alternatives": "🥛", eggs: "🥚",
  meat: "🥩", seafood: "🐟", "meat & seafood": "🥩", poultry: "🍗",
  bakery: "🍞", bread: "🍞",
  snacks: "🍿", "snacks & sweets": "🍬", candy: "🍬", chips: "🍿",
  beverages: "🥤", drinks: "🥤", coffee: "☕", tea: "🍵",
  pantry: "🫙", condiment: "🫙", condiments: "🫙", sauce: "🫙", sauces: "🫙",
  spices: "🧂", seasonings: "🧂",
  frozen: "🧊", "frozen foods": "🧊",
  deli: "🥗", "deli & prepared foods": "🥗", prepared: "🥗",
  cereal: "🥣", breakfast: "🥣",
  pasta: "🍝", rice: "🍚", grains: "🌾",
  "baby products": "🍼", "pet supplies": "🐾",
  cleaning: "🧹", "household goods": "🏠", "personal care": "🧴",
};

const NAME_EMOJI_MAP = {
  apple: "🍎", banana: "🍌", avocado: "🥑", lemon: "🍋", lime: "🍋",
  orange: "🍊", grape: "🍇", strawberry: "🍓", blueberry: "🫐",
  mango: "🥭", pineapple: "🍍", peach: "🍑", cherry: "🍒", watermelon: "🍉",
  tomato: "🍅", carrot: "🥕", corn: "🌽", pepper: "🫑", onion: "🧅",
  garlic: "🧄", potato: "🥔", broccoli: "🥦", cucumber: "🥒", lettuce: "🥬",
  spinach: "🥬", mushroom: "🍄", coconut: "🥥",
  chicken: "🍗", beef: "🥩", pork: "🥩", bacon: "🥓", steak: "🥩",
  fish: "🐟", salmon: "🐟", shrimp: "🦐", tuna: "🐟",
  cheese: "🧀", butter: "🧈", yogurt: "🥛", milk: "🥛", egg: "🥚",
  bread: "🍞", tortilla: "🫓", bagel: "🥯", croissant: "🥐",
  pizza: "🍕", taco: "🌮", burrito: "🌯", pasta: "🍝",
  rice: "🍚", soup: "🍜", salad: "🥗", sandwich: "🥪",
  cookie: "🍪", cake: "🎂", chocolate: "🍫", donut: "🍩", ice_cream: "🍦",
  coffee: "☕", juice: "🧃", wine: "🍷", beer: "🍺", water: "💧", soda: "🥤",
  oil: "🫒", honey: "🍯", sauce: "🫙", vinegar: "🫙",
};

function assignEmojiFast(productName, category) {
  const nameLower = (productName || "").toLowerCase();
  // Check name-based matches first (more specific)
  for (const [key, emoji] of Object.entries(NAME_EMOJI_MAP)) {
    if (nameLower.includes(key)) return emoji;
  }
  // Fall back to category
  const catLower = (category || "").toLowerCase().trim();
  if (CATEGORY_EMOJI_MAP[catLower]) return CATEGORY_EMOJI_MAP[catLower];
  // Partial category match
  for (const [key, emoji] of Object.entries(CATEGORY_EMOJI_MAP)) {
    if (catLower.includes(key)) return emoji;
  }
  return "🍽️";
}

function getEnrichKitchenItemFunctionName() {
  return cleanNullable(process.env.ENRICH_KITCHEN_ITEM_FUNCTION_NAME);
}

async function enqueueKitchenItemEnrichment({ owner, itemId, preliminaryItem, initialPayload }) {
  const functionName = getEnrichKitchenItemFunctionName();
  const payload = {
    owner,
    item_id: itemId,
    preliminary_item: preliminaryItem,
    initial_payload: initialPayload,
  };

  if (functionName && process.env.AWS_LAMBDA_FUNCTION_NAME) {
    await lambdaClient.send(
      new InvokeCommand({
        FunctionName: functionName,
        InvocationType: "Event",
        Payload: Buffer.from(
          JSON.stringify({
            enrich_kitchen_item_internal: true,
            payload,
          })
        ),
      })
    );
    return { attempted: true, status: "queued" };
  }

  if (process.env.AWS_LAMBDA_FUNCTION_NAME) {
    return { attempted: false, status: "skipped_missing_worker" };
  }

  maybeEnrichPersistedBulkItem({
    owner,
    itemId,
    preliminaryItem,
    initialPayload,
  }).catch((error) => {
    console.error(`[BulkKitchenWriter] Background enrichment failed for kitchen item ${itemId}:`, error);
  });

  return { attempted: true, status: "processing" };
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
    .replace(/[^a-z0-9%]+/g, " ")
    .trim();
}

function parseOptionalNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function parseOptionalInteger(value) {
  const parsed = parseOptionalNumber(value);
  if (parsed == null) {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round(parsed)));
}

function parseOptionalBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (value === 1) {
      return true;
    }
    if (value === 0) {
      return false;
    }
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "yes", "y", "1"].includes(normalized)) {
      return true;
    }
    if (["false", "no", "n", "0"].includes(normalized)) {
      return false;
    }
  }
  return null;
}

function normalizeStringArray(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => cleanNullable(entry))
    .filter(Boolean);
}

function parseSourceIndex(value) {
  const parsed = parseOptionalNumber(value);
  if (parsed == null || !Number.isInteger(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function inferQuantityUnit(item) {
  const estimatedState = normalizeText(item.estimated_state);
  const unitMatch = estimatedState.match(/\b\d+(?:\.\d+)?\s+(bottles?|jars?|cans?|boxes?|bags?|cartons?|containers?|packs?|heads?|bunches?|pieces?|items?)\b/);
  if (unitMatch) {
    return unitMatch[1].replace(/s$/, "");
  }
  if (typeof item.count === "number" && item.count > 1) {
    return "count";
  }
  if (item.item_type === "produce") {
    return "count";
  }
  return null;
}

function inferFillPercent(item) {
  const estimatedState = cleanNullable(item.estimated_state);
  const explicitMatch = estimatedState?.match(/(\d{1,3})\s*%/);
  if (explicitMatch) {
    return Math.max(0, Math.min(100, Number(explicitMatch[1])));
  }

  switch (item.fill_level) {
    case "unopened":
    case "full":
      return 100;
    case "mostly_full":
      return 75;
    case "half_full":
      return 50;
    case "low":
      return 25;
    case "nearly_empty":
      return 10;
    case "empty":
      return 0;
    default:
      return null;
  }
}

function inferIsOpened(item, fillPercent) {
  const estimatedState = normalizeText(item.estimated_state);
  if (estimatedState.includes("unopened") || estimatedState.includes("sealed") || item.fill_level === "unopened") {
    return false;
  }
  if (
    estimatedState.includes("opened") ||
    estimatedState.includes("partially used") ||
    estimatedState.includes("partially full") ||
    estimatedState.includes("half full") ||
    estimatedState.includes("mostly full") ||
    estimatedState.includes("low") ||
    estimatedState.includes("nearly empty")
  ) {
    return true;
  }
  if (typeof fillPercent === "number" && fillPercent >= 0 && fillPercent < 100) {
    return true;
  }
  return null;
}

function buildConfirmedItemFromAnalysisItem(item, sourceIndex) {
  const fillPercent = inferFillPercent(item);
  const quantityValue = typeof item.count === "number" && item.count > 0 ? item.count : 1;
  const quantityUnit = inferQuantityUnit(item);
  const isOpened = inferIsOpened(item, fillPercent);

  return {
    source_index: sourceIndex,
    manual: false,
    product_name: buildCheckInLabel(item),
    item_name: cleanNullable(item.item_name),
    brand: cleanNullable(item.brand),
    variant: cleanNullable(item.variant),
    category: cleanNullable(item.category) || "Bulk grocery scan",
    item_type: cleanNullable(item.item_type),
    remaining_quantity: cleanNullable(item.estimated_state),
    quantity_value: quantityValue,
    quantity_unit: quantityUnit,
    is_opened: isOpened,
    fill_percent: fillPercent,
    confidence: typeof item.confidence === "number" ? item.confidence : null,
    explanation: null,
    product_description: cleanNullable(item.short_description) || cleanNullable(item.estimated_state),
    visible_text: normalizeStringArray(item.visible_text),
    uncertainty_reasons: normalizeStringArray(item.uncertainty_reasons),
    fill_level: cleanNullable(item.fill_level) || "unknown",
    estimated_state: cleanNullable(item.estimated_state),
    check_in_label: buildCheckInLabel(item),
    needs_review: Boolean(item.needs_review),
    scan_source: cleanNullable(item.scan_source),
  };
}

function normalizeConfirmedItem(item, index, sourceItems = []) {
  const sourceIndex = parseSourceIndex(item?.source_index);
  const sourceItem = sourceIndex != null ? sourceItems[sourceIndex] || null : null;
  if (sourceIndex != null && !sourceItem) {
    // Can happen when items from multiple receipt jobs are merged client-side
    // but committed under a single source_job_id. Treat as manual item.
    console.warn(`[BulkCommit] Confirmed item ${index + 1} references source_index ${sourceIndex} not in source job (${sourceItems.length} items) — treating as manual`);
  }

  const base = sourceItem ? buildConfirmedItemFromAnalysisItem(sourceItem, sourceIndex) : {
    source_index: null,
    manual: true,
    product_name: null,
    item_name: null,
    brand: null,
    variant: null,
    category: "Bulk grocery scan",
    item_type: null,
    remaining_quantity: null,
    quantity_value: 1,
    quantity_unit: null,
    is_opened: null,
    fill_percent: null,
    confidence: null,
    explanation: null,
    product_description: null,
    visible_text: [],
    uncertainty_reasons: [],
    fill_level: "unknown",
    estimated_state: null,
    check_in_label: null,
    needs_review: false,
    scan_source: null,
  };

  const quantityValue = parseOptionalNumber(item?.quantity_value ?? item?.count) ?? base.quantity_value ?? 1;
  const fillPercent = parseOptionalInteger(item?.fill_percent) ?? base.fill_percent ?? null;
  const isOpened = parseOptionalBoolean(item?.is_opened);
  const remainingQuantity = cleanNullable(item?.remaining_quantity) || cleanNullable(item?.estimated_state) || base.remaining_quantity;
  const productName = cleanNullable(item?.product_name) || cleanNullable(item?.check_in_label) || base.product_name;

  if (!productName) {
    throw new Error(`Confirmed item ${index + 1} is missing product_name`);
  }

  return {
    source_index: sourceIndex,
    manual: parseOptionalBoolean(item?.manual) ?? base.manual,
    product_name: productName,
    item_name: cleanNullable(item?.item_name) || base.item_name || productName,
    brand: cleanNullable(item?.brand) || base.brand,
    variant: cleanNullable(item?.variant) || base.variant,
    category: cleanNullable(item?.category) || base.category || "Bulk grocery scan",
    item_type: cleanNullable(item?.item_type) || base.item_type,
    remaining_quantity: remainingQuantity,
    quantity_value: quantityValue > 0 ? quantityValue : 1,
    quantity_unit: cleanNullable(item?.quantity_unit) || base.quantity_unit,
    is_opened: isOpened == null ? base.is_opened : isOpened,
    fill_percent: fillPercent,
    confidence: parseOptionalNumber(item?.confidence) ?? base.confidence,
    explanation: cleanNullable(item?.explanation) || base.explanation,
    product_description: cleanNullable(item?.product_description) || cleanNullable(item?.short_description) || base.product_description || remainingQuantity,
    visible_text: normalizeStringArray(item?.visible_text).length > 0 ? normalizeStringArray(item.visible_text) : base.visible_text,
    uncertainty_reasons: normalizeStringArray(item?.uncertainty_reasons).length > 0 ? normalizeStringArray(item.uncertainty_reasons) : base.uncertainty_reasons,
    fill_level: cleanNullable(item?.fill_level) || base.fill_level || "unknown",
    estimated_state: remainingQuantity,
    check_in_label: productName,
    needs_review: parseOptionalBoolean(item?.needs_review) ?? base.needs_review,
    scan_source: cleanNullable(item?.scan_source) || base.scan_source,
    // User-entered expiration ("yyyy-MM-dd") from the review/edit sheet — carried
    // through so buildKitchenPayload can persist it (was previously dropped here).
    expiration_date: cleanNullable(item?.expiration_date) || null,
  };
}

function normalizeConfirmedItems(items, sourceItems = []) {
  if (!Array.isArray(items)) {
    throw new Error("Confirmed items payload must be an array");
  }
  return items.map((item, index) => normalizeConfirmedItem(item, index, sourceItems));
}

function buildKitchenPayload(item, context, index) {
  const productName = cleanNullable(item.product_name) || cleanNullable(item.check_in_label) || cleanNullable(item.item_name) || "Unknown item";
  const fillPercent = parseOptionalInteger(item.fill_percent);
  const quantityValue = parseOptionalNumber(item.quantity_value) ?? 1;
  const quantityUnit = cleanNullable(item.quantity_unit);
  const isOpened = parseOptionalBoolean(item.is_opened);
  const scanSource = cleanNullable(item.scan_source) || "bulk_scene_scan";
  const sourceLabel = scanSource === "receipt_scan" ? "receipt scan" : "scene scan";
  const defaultCategory = scanSource === "receipt_scan" ? "Receipt scan" : "Bulk grocery scan";
  const evidenceBits = [
    cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    normalizeStringArray(item.visible_text).length > 0 ? `label text: ${normalizeStringArray(item.visible_text).slice(0, 2).join(" | ")}` : null,
    item.fill_level && item.fill_level !== "unknown" ? `fill level: ${item.fill_level.replace(/_/g, " ")}` : null,
    typeof fillPercent === "number" ? `fill percent: ${fillPercent}%` : null,
    quantityUnit ? `quantity: ${quantityValue} ${quantityUnit}` : quantityValue > 1 ? `quantity: ${quantityValue}` : null,
    typeof isOpened === "boolean" ? `opened: ${isOpened ? "yes" : "no"}` : null,
  ].filter(Boolean);
  const storageGuidance = estimateStorageGuidance({
    product_name: productName,
    item_name: cleanNullable(item.item_name),
    brand: cleanNullable(item.brand),
    variant: cleanNullable(item.variant),
    category: cleanNullable(item.category),
    product_description: cleanNullable(item.product_description) || cleanNullable(item.short_description) || cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    explanation: cleanNullable(item.explanation),
  });

  return {
    device_id: context.deviceId,
    user_id: context.userId,
    job_id: `${context.jobId}:bulk:${index + 1}`,
    product_name: productName,
    brand: cleanNullable(item.brand),
    variant: cleanNullable(item.variant),
    category: cleanNullable(item.category) || defaultCategory,
    remaining_quantity: cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    quantity_value: quantityValue,
    quantity_unit: quantityUnit,
    is_opened: isOpened,
    fill_percent: fillPercent,
    confidence: typeof item.confidence === "number" ? item.confidence : null,
    explanation: cleanNullable(item.explanation) || (evidenceBits.length > 0
      ? `Kitchen check-in from ${sourceLabel}. ${evidenceBits.join(". ")}.`
      : `Kitchen check-in from ${sourceLabel}.`),
    product_description: cleanNullable(item.product_description) || cleanNullable(item.short_description) || cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    barcode: null,
    country_guess: null,
    estimated_price: null,
    ingredients: [],
    nutrition_summary: null,
    upf: "no",
    harmful_ingredients: [],
    similar_items: [],
    alternatives: [],
    healthier_alternatives: [],
    store_availability: [],
    images: context.imageUrl || null,
    s3_key: null,
    action: "IN",
    product_expiration: cleanNullable(item.expiration_date) || null,
    storage_guidance: storageGuidance,
    product_image_url: `emoji:${assignEmojiFast(productName, cleanNullable(item.category))}`,
    product_image_key: null,
    analysis_stage: "preliminary",
    analysis_status: "ready",
    analysis_source: scanSource === "receipt_scan" ? "grocery_identifier_receipt_preliminary" : "grocery_identifier_bulk_preliminary",
    needs_review: Boolean(item.needs_review),
    defer_recipes: true,
    provisional_payload: {
      source_job_id: context.sourceJobId || context.jobId,
      source_item_index: parseSourceIndex(item.source_index),
      manual: Boolean(item.manual),
      scan_source: scanSource,
      enrichment_status: "pending",
      fill_level: item.fill_level || null,
      remaining_quantity: cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
      quantity_value: quantityValue,
      quantity_unit: quantityUnit,
      is_opened: typeof isOpened === "boolean" ? isOpened : null,
      fill_percent: fillPercent,
      visible_text: normalizeStringArray(item.visible_text),
      uncertainty_reasons: normalizeStringArray(item.uncertainty_reasons),
    },
  };
}

async function createKitchenItem(owner, payload) {
  let lastError = null;
  for (let attempt = 1; attempt <= CREATE_RETRY_ATTEMPTS; attempt += 1) {
    let response;
    try {
      response = await fetch(`${KITCHEN_API_BASE_URL}/kitchen/${encodeURIComponent(owner)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        timeout: 20000,
      });
    } catch (networkError) {
      // Connection reset / timeout — treat as transient and retry with backoff.
      lastError = networkError instanceof Error ? networkError : new Error(String(networkError));
      if (attempt === CREATE_RETRY_ATTEMPTS) throw lastError;
      await sleep(backoffDelay(attempt));
      continue;
    }

    const data = await response.json().catch(() => ({}));
    if (response.ok) {
      return data;
    }

    lastError = new Error(data.error || data.details || `Kitchen API create failed with HTTP ${response.status}`);
    const message = String(lastError.message || "");
    // Retry transient gateway/throttle responses (the burst-503 that silently
    // dropped confirmed receipt items) and DB deadlocks. A 4xx other than 429 is
    // a real client error — don't retry.
    const isRetryable =
      RETRYABLE_HTTP_STATUSES.has(response.status) ||
      message.includes("Deadlock found when trying to get lock") ||
      message.includes("(1213");
    if (!isRetryable || attempt === CREATE_RETRY_ATTEMPTS) {
      throw lastError;
    }

    await sleep(backoffDelay(attempt));
  }

  throw lastError || new Error("Kitchen API create failed");
}

// Exponential backoff with jitter so retried items don't re-burst in lockstep.
function backoffDelay(attempt) {
  const base = Math.min(2000, 200 * 2 ** (attempt - 1)); // 200,400,800,1600,2000
  return base + Math.floor(Math.random() * 200);
}

async function persistConfirmedBulkItems({ owner, userId, deviceId, jobId, sourceJobId, confirmedItems, sourceItems, imageUrl }) {
  if (!owner) {
    return {
      requested: false,
      persisted_count: 0,
      created_count: 0,
      duplicate_count: 0,
      items: [],
      errors: [],
    };
  }

  const items = normalizeConfirmedItems(confirmedItems, sourceItems);
  const summary = {
    requested: true,
    persisted_count: 0,
    created_count: 0,
    duplicate_count: 0,
    enrichment_requested_count: 0,
    enrichment_queued_count: 0,
    items: [],
    errors: [],
    enrichment_errors: [],
  };
  const rawPreparedItems = items.map((item, index) => ({
    item,
    payload: buildKitchenPayload(item, {
      owner,
      userId: userId || owner,
      deviceId: deviceId || "bulk-scan",
      jobId,
      sourceJobId: sourceJobId || jobId,
      imageUrl: imageUrl || null,
    }, index),
  }));

  // Consolidate duplicates from the same batch: merge items with the same
  // product_name into a single row with summed quantity_value.
  const consolidationMap = new Map();
  for (const entry of rawPreparedItems) {
    const key = (entry.payload.product_name || "").trim().toLowerCase();
    if (consolidationMap.has(key)) {
      const existing = consolidationMap.get(key);
      existing.payload.quantity_value = (existing.payload.quantity_value || 1) + (entry.payload.quantity_value || 1);
      // Keep the higher confidence
      if (typeof entry.payload.confidence === "number" && (entry.payload.confidence > (existing.payload.confidence || 0))) {
        existing.payload.confidence = entry.payload.confidence;
      }
    } else {
      consolidationMap.set(key, entry);
    }
  }
  const preparedItems = Array.from(consolidationMap.values());
  const consolidatedCount = rawPreparedItems.length - preparedItems.length;
  if (consolidatedCount > 0) {
    console.log(`[CONSOLIDATE] Merged ${rawPreparedItems.length} items into ${preparedItems.length} (${consolidatedCount} duplicates consolidated)`);
  }

  // Bounded concurrency (not Promise.all) so a big receipt doesn't burst the
  // Kitchen API past its throttle ceiling and 503 confirmed items onto the floor.
  const writeResults = await mapWithConcurrency(
    preparedItems,
    CREATE_CONCURRENCY,
    async ({ item, payload }) => {
      try {
        const response = await createKitchenItem(owner, payload);
        return {
          ok: true,
          item,
          payload,
          response,
        };
      } catch (error) {
        return {
          ok: false,
          item,
          payload,
          error,
        };
      }
    }
  );

  // Final resweep: never give up on a CONFIRMED item after a single pass. If any
  // still failed (despite createKitchenItem's own 5x backoff), pause to let a
  // transient throttle drain, then re-attempt ONLY those once more at low
  // concurrency. Items recovered here flip back to ok and count as persisted.
  const stillFailed = writeResults.filter((r) => !r.ok);
  if (stillFailed.length > 0) {
    console.warn(`[BulkKitchenWriter] ${stillFailed.length} item(s) failed first pass; resweeping after delay.`);
    await sleep(2000);
    const reswept = await mapWithConcurrency(
      stillFailed,
      Math.min(CREATE_CONCURRENCY, 3),
      async (failedResult) => {
        try {
          const response = await createKitchenItem(owner, failedResult.payload);
          return { result: failedResult, ok: true, response };
        } catch (error) {
          return { result: failedResult, ok: false, error };
        }
      }
    );
    let recovered = 0;
    for (const r of reswept) {
      if (r.ok) {
        r.result.ok = true;          // same object reference held in writeResults
        r.result.response = r.response;
        r.result.error = undefined;
        recovered += 1;
      } else {
        r.result.error = r.error;
      }
    }
    console.warn(`[BulkKitchenWriter] Resweep recovered ${recovered} of ${stillFailed.length} item(s).`);
  }

  const enrichmentQueue = [];
  for (const writeResult of writeResults) {
    if (!writeResult.ok) {
      summary.errors.push({
        product_name: writeResult.payload.product_name,
        job_id: writeResult.payload.job_id,
        message: writeResult.error instanceof Error ? writeResult.error.message : "Kitchen persistence failed",
      });
      continue;
    }

    summary.persisted_count += 1;
    if (writeResult.response?.created) {
      summary.created_count += 1;
    } else {
      summary.duplicate_count += 1;
    }

    const summaryItem = {
      product_name: writeResult.payload.product_name,
      created: Boolean(writeResult.response?.created),
      item_id: writeResult.response?.item?._id || null,
      job_id: writeResult.payload.job_id,
      source_index: parseSourceIndex(writeResult.item.source_index),
      manual: Boolean(writeResult.item.manual),
    };
    summary.items.push(summaryItem);

    if (writeResult.response?.item?._id) {
      enrichmentQueue.push({
        summaryItem,
        item: writeResult.item,
        payload: writeResult.payload,
        itemId: writeResult.response.item._id,
      });
    }
  }

  // Process enrichment in chunks of 5 to avoid Lambda fan-out explosion
  const ENRICHMENT_CHUNK_SIZE = 5;
  console.log(`[ENRICH] Processing ${enrichmentQueue.length} items in chunks of ${ENRICHMENT_CHUNK_SIZE}`);
  for (let i = 0; i < enrichmentQueue.length; i += ENRICHMENT_CHUNK_SIZE) {
    const chunk = enrichmentQueue.slice(i, i + ENRICHMENT_CHUNK_SIZE);
    const results = await Promise.allSettled(
      chunk.map(async (queued) => {
        const enrichment = await enqueueKitchenItemEnrichment({
          owner,
          itemId: queued.itemId,
          preliminaryItem: queued.item,
          initialPayload: queued.payload,
        });
        if (enrichment?.attempted) {
          summary.enrichment_requested_count += 1;
        }
        if (enrichment?.status === "queued") {
          summary.enrichment_queued_count += 1;
        }
        if (enrichment?.status) {
          queued.summaryItem.enrichment = enrichment.status;
        }
      })
    );
    for (let j = 0; j < results.length; j++) {
      if (results[j].status === 'rejected') {
        const queued = chunk[j];
        queued.summaryItem.enrichment = "dispatch_failed";
        summary.enrichment_errors.push({
          product_name: queued.payload.product_name,
          job_id: queued.payload.job_id,
          message: results[j].reason instanceof Error ? results[j].reason.message : "Bulk enrichment dispatch failed",
        });
        console.error('[ENRICH] Chunk item failed:', results[j].reason?.message || results[j].reason);
      }
    }
  }

  // Trigger recipe generation once after all items are persisted (instead of per-item)
  if (summary.persisted_count > 0 && owner) {
    try {
      await fetch(`${KITCHEN_API_BASE_URL}/kitchen/${encodeURIComponent(owner)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trigger_recipes_only: true }),
        timeout: 10000,
      });
    } catch (err) {
      console.warn("[BulkKitchenWriter] Failed to trigger recipe generation:", err?.message || err);
    }
  }

  if (summary.errors.length > 0) {
    const aggregatedError = new Error(
      `Bulk commit failed for ${summary.errors.length} item(s); persisted ${summary.persisted_count} of ${items.length}.`
    );
    aggregatedError.summary = summary;
    throw aggregatedError;
  }

  return summary;
}

async function persistBulkKitchenResults({ owner, userId, deviceId, jobId, analysis, imageUrl }) {
  const sourceItems = Array.isArray(analysis?.items) ? analysis.items : [];
  const confirmedItems = sourceItems.map((item, index) => buildConfirmedItemFromAnalysisItem(item, index));
  return persistConfirmedBulkItems({
    owner,
    userId,
    deviceId,
    jobId,
    sourceJobId: jobId,
    confirmedItems,
    sourceItems,
    imageUrl,
  });
}

module.exports = {
  buildConfirmedItemFromAnalysisItem,
  normalizeConfirmedItems,
  persistConfirmedBulkItems,
  persistBulkKitchenResults,
};
