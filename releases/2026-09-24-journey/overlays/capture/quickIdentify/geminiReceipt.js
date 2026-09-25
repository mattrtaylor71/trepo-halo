const {captureOutcome} = require('../captureOutcome');
const { providerMetadata, rememberProviderResult, providerResultMetadata, recordProviderAttempt } = require("./providerUsage");
const fetch = require("node-fetch");
const { Jimp } = require("jimp");
const { extractGoogleVisionContext } = require("./googleVision");
const { buildCheckInLabel } = require("./geminiBulk");

const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_RECEIPT_TIMEOUT_MS || 120000);
const GEMINI_MAX_DIMENSION = Number(process.env.GEMINI_RECEIPT_MAX_DIMENSION || 1600);
const GEMINI_MODEL = process.env.GEMINI_RECEIPT_MODEL || process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
const GEMINI_FALLBACK_MODEL = process.env.GEMINI_RECEIPT_FALLBACK_MODEL || "gemini-2.5-flash";
const GEMINI_MAX_ATTEMPTS = Number(process.env.GEMINI_RECEIPT_ATTEMPTS || 2);

const geminiResponseSchema = {
  type: "object",
  properties: {
    capture_type: {type:"string",enum:["receipt","product","scene","unreadable","unknown"]},
    receipt_analysis_log: { type: "string" },
    receipt_summary: {
      type: "object",
      properties: {
        merchant_name: { type: "string", nullable: true },
        purchase_date: { type: "string", nullable: true },
        subtotal_amount: { type: "string", nullable: true },
        total_amount: { type: "string", nullable: true },
        receipt_item_count_hint: { type: "integer", nullable: true },
      },
      required: ["merchant_name", "purchase_date", "subtotal_amount", "total_amount", "receipt_item_count_hint"],
    },
    items: {
      type: "array",
      items: {
        type: "object",
        properties: {
          raw_line_text: { type: "string", nullable: true },
          item_name: { type: "string" },
          brand: { type: "string", nullable: true },
          variant: { type: "string", nullable: true },
          category: { type: "string", nullable: true },
          item_type: { type: "string" },
          quantity: { type: "integer" },
          unit_price: { type: "string", nullable: true },
          line_total: { type: "string", nullable: true },
          confidence: { type: "number" },
          needs_review: { type: "boolean" },
          uncertainty_reasons: {
            type: "array",
            items: { type: "string" },
          },
          visible_text_OCR: { type: "string", nullable: true },
        },
        required: [
          "raw_line_text",
          "item_name",
          "brand",
          "variant",
          "category",
          "item_type",
          "quantity",
          "unit_price",
          "line_total",
          "confidence",
          "needs_review",
          "uncertainty_reasons",
          "visible_text_OCR",
        ],
      },
    },
  },
  required: ["capture_type", "receipt_analysis_log", "receipt_summary", "items"],
};

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

function uniqueStrings(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

function titleCase(value) {
  const cleaned = cleanNullable(value);
  if (!cleaned) {
    return null;
  }
  return cleaned
    .split(/\s+/)
    .map((word) => {
      if (word === word.toUpperCase() && word.length <= 4) {
        return word;
      }
      if (/\d/.test(word)) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(" ");
}

function clampConfidence(value) {
  const parsed = typeof value === "number" && Number.isFinite(value) ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    return 0.5;
  }
  return Math.max(0.2, Math.min(0.99, parsed));
}

function parseCurrency(value) {
  const cleaned = cleanNullable(value);
  if (!cleaned) {
    return null;
  }
  const normalized = cleaned.replace(/[^0-9.-]+/g, "");
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.round(parsed * 100) / 100;
}

function normalizeQuantity(value) {
  const parsed = typeof value === "number" && Number.isFinite(value) ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    return 1;
  }
  return Math.max(1, Math.round(parsed));
}

function inferItemType(category, itemName, lineText) {
  const joined = normalizeText([category, itemName, lineText].filter(Boolean).join(" "));
  const produceTokens = [
    "apple",
    "apples",
    "avocado",
    "avocados",
    "banana",
    "bananas",
    "berries",
    "blackberries",
    "blueberries",
    "broccoli",
    "cabbage",
    "carrot",
    "carrots",
    "cauliflower",
    "celery",
    "cilantro",
    "garlic",
    "grape",
    "grapes",
    "greens",
    "kale",
    "lettuce",
    "lime",
    "limes",
    "mango",
    "mangos",
    "mushroom",
    "mushrooms",
    "onion",
    "onions",
    "pepper",
    "peppers",
    "pineapple",
    "potato",
    "potatoes",
    "produce",
    "spinach",
    "strawberries",
    "tomato",
    "tomatoes",
  ];
  if (produceTokens.some((token) => joined.includes(token))) {
    return "produce";
  }
  if (!joined) {
    return "unknown";
  }
  return "packaged";
}

function extractVisibleText(entry) {
  return uniqueStrings(
    [cleanNullable(entry.raw_line_text), cleanNullable(entry.visible_text_OCR)]
      .filter(Boolean)
      .flatMap((value) => value.split(/\n|,|;|\|/))
      .map((value) => value.trim())
      .filter(Boolean)
  ).slice(0, 6);
}

function looksCrypticReceiptLine(value) {
  const cleaned = cleanNullable(value);
  if (!cleaned) {
    return false;
  }
  const tokens = cleaned.split(/\s+/).filter(Boolean);
  if (tokens.length < 2) {
    return false;
  }
  const shortTokenCount = tokens.filter((token) => token.length <= 4).length;
  const abbreviationLikeCount = tokens.filter((token) => /^[A-Z0-9/&.-]+$/.test(token)).length;
  return shortTokenCount >= 2 && abbreviationLikeCount >= 2;
}

function buildUncertaintyReasons(entry, visibleText, confidence) {
  const providedReasons = Array.isArray(entry?.uncertainty_reasons)
    ? entry.uncertainty_reasons.map((value) => cleanNullable(value)).filter(Boolean)
    : [];
  const reasons = [...providedReasons];
  if (visibleText.length === 0) {
    reasons.push("no_ocr_evidence");
  }
  if (looksCrypticReceiptLine(entry.raw_line_text)) {
    reasons.push("cryptic_receipt_abbreviation");
  }
  if (!cleanNullable(entry.brand) && inferItemType(entry.category, entry.item_name, entry.raw_line_text) === "packaged") {
    reasons.push("missing_brand");
  }
  if (confidence < 0.72) {
    reasons.push("low_receipt_confidence");
  }
  return uniqueStrings(reasons);
}

function normalizeSummary(summary) {
  const normalized = summary && typeof summary === "object" ? summary : {};
  return {
    merchant_name: titleCase(normalized.merchant_name),
    purchase_date: cleanNullable(normalized.purchase_date),
    subtotal_amount: parseCurrency(normalized.subtotal_amount),
    total_amount: parseCurrency(normalized.total_amount),
    receipt_item_count_hint: Number.isInteger(normalized.receipt_item_count_hint) ? normalized.receipt_item_count_hint : null,
  };
}

function normalizeReceiptEntry(entry, receiptSummary) {
  const rawItemName = cleanNullable(entry.item_name) || "Unknown receipt item";
  const brand = titleCase(entry.brand);
  const variant = titleCase(entry.variant);
  const rawLineText = cleanNullable(entry.raw_line_text);
  const visibleText = extractVisibleText(entry);
  const confidence = clampConfidence(entry.confidence);
  const uncertaintyReasons = buildUncertaintyReasons(entry, visibleText, confidence);
  const itemType = ["produce", "packaged", "unknown"].includes(entry.item_type)
    ? entry.item_type
    : inferItemType(entry.category, rawItemName, rawLineText);
  const displayName = buildCheckInLabel({
    item_name: rawItemName,
    brand,
    variant,
  });
  const merchantName = cleanNullable(receiptSummary?.merchant_name);
  const purchaseDate = cleanNullable(receiptSummary?.purchase_date);
  const noteBits = [
    merchantName ? `Purchased from ${merchantName}` : "Purchased from receipt",
    purchaseDate ? `on ${purchaseDate}` : null,
    rawLineText ? `receipt line ${rawLineText}` : null,
  ].filter(Boolean);

  return {
    item_name: displayName,
    brand,
    variant,
    category: titleCase(entry.category) || "Receipt scan",
    item_type: itemType,
    count: normalizeQuantity(entry.quantity),
    position_hint: null,
    confidence,
    short_description: `${noteBits.join(" ")}.`,
    needs_review: Boolean(entry.needs_review) || uncertaintyReasons.length > 0,
    uncertainty_reasons: uncertaintyReasons,
    visible_text: visibleText,
    estimated_state: "Purchased on receipt",
    fill_level: "unknown",
    visible_text_OCR: cleanNullable(entry.visible_text_OCR),
    check_in_label: displayName,
    raw_line_text: rawLineText,
    unit_price: parseCurrency(entry.unit_price),
    line_total: parseCurrency(entry.line_total),
    receipt_merchant_name: merchantName,
    receipt_purchase_date: purchaseDate,
    scan_source: "receipt_scan",
  };
}

function buildAggregateKey(item) {
  return [
    normalizeText(item.brand),
    normalizeText(item.item_name),
    normalizeText(item.variant),
    normalizeText(item.category),
  ].join("|");
}

function aggregateReceiptItems(items) {
  const grouped = new Map();

  for (const item of items) {
    const key = buildAggregateKey(item);
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        ...item,
        raw_line_text: cleanNullable(item.raw_line_text),
        visible_text_OCR: cleanNullable(item.visible_text_OCR),
      });
      continue;
    }

    const mergedRawLines = uniqueStrings([existing.raw_line_text, item.raw_line_text]).join(" | ");
    const mergedVisibleText = uniqueStrings([...(existing.visible_text || []), ...(item.visible_text || [])]).slice(0, 10);
    const mergedLineTotal =
      typeof existing.line_total === "number" && typeof item.line_total === "number"
        ? Math.round((existing.line_total + item.line_total) * 100) / 100
        : existing.line_total ?? item.line_total ?? null;
    const mergedUnitPrice =
      existing.unit_price === item.unit_price
        ? existing.unit_price
        : existing.unit_price ?? item.unit_price ?? null;

    grouped.set(key, {
      ...existing,
      count: Math.max(1, Number(existing.count || 1)) + Math.max(1, Number(item.count || 1)),
      confidence: Math.max(existing.confidence || 0, item.confidence || 0),
      needs_review: Boolean(existing.needs_review || item.needs_review),
      uncertainty_reasons: uniqueStrings([...(existing.uncertainty_reasons || []), ...(item.uncertainty_reasons || [])]),
      visible_text: mergedVisibleText,
      visible_text_OCR: cleanNullable(uniqueStrings([existing.visible_text_OCR, item.visible_text_OCR]).join(" | ")),
      raw_line_text: mergedRawLines || null,
      line_total: mergedLineTotal,
      unit_price: mergedUnitPrice,
      short_description: existing.count + item.count > 1
        ? `${existing.short_description.replace(/\.$/, "")}. Aggregated from multiple receipt lines.`
        : existing.short_description,
    });
  }

  return Array.from(grouped.values()).sort((a, b) => {
    const countDelta = Math.max(b.count || 1, 1) - Math.max(a.count || 1, 1);
    if (countDelta !== 0) {
      return countDelta;
    }
    return (b.confidence || 0) - (a.confidence || 0);
  });
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
  const image = await Jimp.read(imageAsset.buffer);
  const width = image.bitmap?.width || 0;
  const height = image.bitmap?.height || 0;
  const maxDimension = Math.max(width, height);

  if (!maxDimension || maxDimension <= GEMINI_MAX_DIMENSION) {
    return imageAsset;
  }

  const scale = GEMINI_MAX_DIMENSION / maxDimension;
  image.scale(scale);
  return {
    buffer: await image.getBuffer("image/jpeg"),
    mimeType: "image/jpeg",
  };
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

function buildGeminiPrompt(ocrContext) {
  const ocrText = cleanNullable(ocrContext?.promptText);
  return `You are an advanced grocery receipt extraction AI. Your job is to turn a photographed store receipt into a clean JSON list of grocery items that can be checked into a kitchen inventory.

You will receive:
- the receipt image
- OCR text extracted from the receipt when available

Return strict JSON with:
0. capture_type: receipt, product (one product label/package), scene (fridge/pantry/foods without a receipt), unreadable, or unknown. Classify what is actually visible. For a non-receipt image return no receipt items; never invent a purchase.
1. receipt_analysis_log: 1 to 3 short sentences about any hard-to-read areas or especially cryptic abbreviations
2. receipt_summary:
   - merchant_name
   - purchase_date in the most literal readable format from the receipt
   - subtotal_amount
   - total_amount
   - receipt_item_count_hint
3. items: all purchased items from the receipt

Important rules:
- Include every purchased item on the receipt — food, beverages, household goods, personal care, baby products, pet supplies, cleaning supplies, etc. If it was purchased, include it.
- Exclude taxes, fees, tips, bottle deposits, subtotal/total lines, payment lines, coupons, discounts, rebates, VOID lines, cashier metadata, and membership data.
- Exclude clearly non-purchasable lines such as flowers, greeting cards, or clothing. Do NOT exclude personal care, baby, pet, or household items — these should be included with needs_review=true.
- Infer shopper-friendly item names from receipt abbreviations conservatively. Costco receipts often abbreviate items heavily, and "KS" can mean Kirkland Signature.
- Prefer a natural check-in title such as "Greek Yogurt", "Organic Free-Range Eggs", "Kirkland Sliced Turkey", or "Blackberries".
- Only include a brand when supported by the receipt text or the retailer shorthand.
- Use variant for flavor, pack, size, cut, or other detail when supported.
- Set quantity from the receipt line when visible; otherwise use 1.
- Set item_type to produce, packaged, household, personal_care, baby, pet, or unknown.
- Mark needs_review true and add uncertainty_reasons whenever the abbreviation expansion is uncertain.
- For non-food/non-beverage items (household, personal care, baby, pet, etc.), set needs_review to true and add "non_grocery_item" to uncertainty_reasons.
- Keep raw_line_text close to the literal receipt line that supports the item.
- visible_text_OCR should include the literal OCR snippet that best supports the item.
- unit_price and line_total should be plain numeric strings like "5.99" when readable, otherwise null.
- confidence should be 0 to 1.

${ocrText ? `OCR text from the receipt:\n${ocrText}\n` : "OCR text was not available, so rely on the image only.\n"}

Be conservative but useful. It is better to return "Organic Free-Range Eggs" with needs_review=true than to omit an obvious purchased grocery item entirely.`;
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

async function callGemini(imageAsset, ocrContext, options = {}) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required");
  }

  const body = {
    systemInstruction: {
      parts: [{ text: buildGeminiPrompt(ocrContext) }],
    },
    contents: [
      {
        role: "user",
        parts: [
          {
            text:
              "Extract all purchased items from the receipt. Include food, beverages, household goods, personal care, baby products, pet supplies, and cleaning supplies. Ignore payment/admin lines. Use needs_review whenever abbreviations are hard to expand confidently.",
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

  // Try primary model first, then fallback model
  const models = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL].filter(Boolean);
  const uniqueModels = [...new Set(models)];

  for (const currentModel of uniqueModels) {
    const isFallback = currentModel !== uniqueModels[0];

    if (isFallback) {
      await reportStage(
        options,
        "gemini_reasoning",
        "Switching to backup model...",
        56,
        { model: currentModel, fallback: true }
      );
    }

    for (let attempt = 0; attempt < GEMINI_MAX_ATTEMPTS; attempt += 1) {
      const attemptStarted = Date.now();
      let payload = null, httpStatus = null, finishedAt = null, outcome = "transport_error";
      try {
        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${currentModel}:generateContent?key=${process.env.GEMINI_API_KEY}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            timeout: GEMINI_TIMEOUT_MS,
          }
        );

        httpStatus = response.status;
        outcome = "http_error";
        if (response.ok) {
          outcome = "parse_error";
          payload = await response.json();
          finishedAt = Date.now();
          const text = payload?.candidates?.[0]?.content?.parts?.map((part) => part.text).filter(Boolean).join("\n");
          if (!text) throw new Error('Receipt response is unavailable');
          const parsed = JSON.parse(text);
          if (!parsed || !['receipt','product','scene','unreadable','unknown'].includes(parsed.capture_type)
              || !Array.isArray(parsed.items)
              || parsed.items.some(item => !item || typeof item !== 'object' || typeof item.item_name !== 'string' || !item.item_name.trim())
              || (parsed.capture_type !== 'receipt' && parsed.items.length)) {
            throw new Error('Receipt response has an invalid shape');
          }
          outcome = "success";
          return rememberProviderResult(parsed, providerMetadata("gemini", currentModel, payload));
        }

        const errorText = await response.text();
        finishedAt = Date.now();
        if (response.status === 429) {
          // Quota exceeded — try fallback model instead of throwing immediately
          lastError = new Error(`Gemini quota exceeded on ${currentModel}`);
          break; // Break inner loop to try fallback model
        }

        if (response.status >= 500 && response.status < 600) {
          lastError = new Error(`Gemini receipt scan transient HTTP ${response.status} on ${currentModel}: ${errorText}`.trim());
          if (attempt < GEMINI_MAX_ATTEMPTS - 1) {
            await reportStage(
              options,
              "gemini_reasoning",
              `Hmm, that took a sec. Giving it another shot (${attempt + 2}/${GEMINI_MAX_ATTEMPTS})...`,
              56,
              { model: currentModel, retry_attempt: attempt + 1, retry_reason: `http_${response.status}` }
            );
            await sleep(buildRetryDelayMs(attempt));
            continue;
          }
          break; // Exhausted retries on this model — try fallback
        }

        lastError = new Error(`Gemini receipt scan failed on ${currentModel}: HTTP ${response.status} ${errorText}`.trim());
        break; // Non-retryable error — try fallback
      } catch (error) {
        finishedAt = finishedAt ?? Date.now();
        if (isTransientGeminiFetchError(error)) {
          lastError = error instanceof Error ? error : new Error(String(error || "Unknown Gemini timeout"));
          if (attempt < GEMINI_MAX_ATTEMPTS - 1) {
            await reportStage(
              options,
              "gemini_reasoning",
              `Still working on it \u2014 retrying (${attempt + 2}/${GEMINI_MAX_ATTEMPTS})...`,
              56,
              { model: currentModel, retry_attempt: attempt + 1, retry_reason: "network_timeout" }
            );
            await sleep(buildRetryDelayMs(attempt));
            continue;
          }
          break; // Exhausted retries — try fallback
        }
        lastError = error;
        break; // Non-transient error — try fallback
      } finally {
        recordProviderAttempt({provider:"gemini",surface:"receipt_inventory_deep",requestedModel:currentModel,payload,attempt:attempt+1,httpStatus,status:outcome,durationMs:(finishedAt ?? Date.now())-attemptStarted,imageCount:1}, console.log);
      }
    }
  }

  throw lastError || new Error("Gemini receipt scan failed on all models.");
}

async function identifyReceiptGeminiDeep(input, options = {}) {
  await reportStage(options, "loading_image", "Getting your receipt ready...", 5, { model: GEMINI_MODEL });
  const loadedAsset = await loadImageAsset(input);
  await reportStage(options, "preparing_image", "Sharpening up the details...", 12, { model: GEMINI_MODEL });
  const imageAsset = await resizeImageAsset(loadedAsset);
  const startedAt = Date.now();

  await reportStage(options, "reading_receipt_text", "Reading through your receipt...", 22, { model: GEMINI_MODEL });
  let ocrContext = null;
  try {
    ocrContext = await extractGoogleVisionContext(imageAsset);
  } catch (error) {
    console.warn("[ReceiptIdentify] OCR failed:", error);
  }

  await reportStage(options, "gemini_reasoning", "Figuring out what you bought...", 56, {
    model: GEMINI_MODEL,
    ocr_enabled: Boolean(ocrContext?.usedGoogleVision),
  });
  const geminiResult = await callGemini(imageAsset, ocrContext, options);

  const observedProvider = providerResultMetadata(geminiResult);
  await reportStage(options, "finalizing_inventory", "Almost done \u2014 tidying up your list...", 90, { model: observedProvider.model || observedProvider.requested_model || GEMINI_MODEL });
  const receiptSummary = normalizeSummary(geminiResult?.receipt_summary);
  const items = Array.isArray(geminiResult?.items)
    ? aggregateReceiptItems(
        geminiResult.items
        .map((entry) => normalizeReceiptEntry(entry, receiptSummary))
        .filter((entry) => cleanNullable(entry.item_name))
      )
    : [];

  return {
    items,
    capture_type: captureOutcome(geminiResult).capture_type,
    ...{capture_outcome:captureOutcome({...geminiResult,items}).outcome},
    visual_analysis_log: cleanNullable(geminiResult?.receipt_analysis_log),
    receipt_analysis_log: cleanNullable(geminiResult?.receipt_analysis_log),
    receipt_summary: receiptSummary,
    debug: {
      elapsed_ms: Date.now() - startedAt,
      mode: "receipt_inventory_deep",
      model: observedProvider.model || observedProvider.requested_model || GEMINI_MODEL,
      usage: observedProvider.usage || null,
      provider: "gemini",
      item_count: items.length,
      review_required_count: items.filter((item) => item.needs_review).length,
      total_quantity: items.reduce((sum, item) => sum + Math.max(item.count || 1, 1), 0),
      ocr_enabled: Boolean(ocrContext?.usedGoogleVision),
      ocr_line_count: ocrContext?.lineCount || 0,
      merchant_name: receiptSummary?.merchant_name || null,
      purchase_date: receiptSummary?.purchase_date || null,
    },
  };
}

module.exports = {
  identifyReceiptGeminiDeep,
};
