const fetch = require("node-fetch");
const { getOpenAIClient, getOpenAIModel, parseJsonResponse } = require("./dist/openai/client");

const KITCHEN_API_BASE_URL =
  (process.env.KITCHEN_API_BASE_URL || "https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com").replace(/\/$/, "");

const LOCAL_MIN_SCORE = Math.max(0, Math.min(1, Number(process.env.SWAP_LOCAL_MIN_SCORE || 0.32)));
const MAX_LOCAL_CANDIDATES = Math.max(1, Math.min(12, Number(process.env.SWAP_MAX_CANDIDATES || 8)));
const MAX_KITCHEN_ROWS = Math.max(1, Math.min(500, Number(process.env.BULK_KITCHEN_SIMILARITY_MAX_ROWS || 250)));
const OPENAI_SHORTLIST_LIMIT = Math.max(
  MAX_LOCAL_CANDIDATES,
  Math.min(20, Number(process.env.SWAP_OPENAI_SHORTLIST_LIMIT || 12))
);
const OPENAI_BATCH_MAX_ITEMS = Math.max(1, Math.min(30, Number(process.env.SWAP_OPENAI_BATCH_MAX_ITEMS || 16)));

const deepSwapBatchResponseJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["items"],
  properties: {
    items: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["item_index", "candidate_ids", "reason_summary"],
        properties: {
          item_index: { type: "integer", minimum: 0 },
          candidate_ids: {
            type: "array",
            items: { type: "string" },
          },
          reason_summary: { type: "string" },
        },
      },
    },
  },
};

const GENERIC_MATCH_TOKENS = new Set([
  "fresh",
  "ripe",
  "whole",
  "raw",
  "halved",
  "sliced",
  "diced",
  "item",
  "food",
  "grocery",
  "product",
  "pack",
  "package",
  "bag",
  "box",
  "container",
]);

const CONFLICT_TOKEN_GROUPS = [
  ["red", "white", "yellow", "green", "purple"],
  ["lime", "lemon", "grapefruit", "orange"],
  ["plain", "vanilla", "chocolate", "strawberry", "blueberry", "raspberry", "berry", "mango", "pineapple", "coconut"],
  ["spearmint", "peppermint", "wintergreen", "cinnamon"],
];

function cleanText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalize(value) {
  if (!value) return "";
  return value
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeMatchName(value) {
  if (!value) return "";
  let raw = value.toLowerCase();
  raw = raw.replace(/\([^)]*\d+[^)]*\)/g, " ");
  raw = raw.replace(/\b\d+(\.\d+)?\s*(oz|ounce|ounces|lb|lbs|pound|pounds|g|kg|ml|l|liter|liters|fl|floz|gal|gallon|gallons|qt|quart|quarts|pt|pint|pints)\b/g, " ");
  raw = raw.replace(/[^\w\s]/g, " ");
  raw = raw.replace(/\s+/g, " ").trim();
  if (!raw) return "";
  return raw
    .split(" ")
    .filter(Boolean)
    .filter((token) => !GENERIC_MATCH_TOKENS.has(token))
    .join(" ");
}

function tokenSet(...values) {
  const merged = values
    .map((value) => normalizeMatchName(value))
    .filter(Boolean)
    .join(" ");
  return new Set(merged.split(/\s+/).filter(Boolean));
}

function tokenSimilarity(a, b) {
  const setA = tokenSet(a);
  const setB = tokenSet(b);
  if (setA.size === 0 && setB.size === 0) return 1;
  if (setA.size === 0 || setB.size === 0) return 0;
  let overlap = 0;
  for (const token of setA) {
    if (setB.has(token)) overlap += 1;
  }
  return (2 * overlap) / (setA.size + setB.size);
}

function overlappingTokens(a, b) {
  const matches = [];
  for (const token of a) {
    if (b.has(token)) matches.push(token);
  }
  return matches;
}

function getConflictPenalty(left, right) {
  const leftTokens = tokenSet(left.product_name, left.variant);
  const rightTokens = tokenSet(right.product_name, right.variant);
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;

  let penalty = 0;
  for (const group of CONFLICT_TOKEN_GROUPS) {
    const leftGroup = group.filter((token) => leftTokens.has(token));
    const rightGroup = group.filter((token) => rightTokens.has(token));
    if (leftGroup.length > 0 && rightGroup.length > 0 && !leftGroup.some((token) => rightGroup.includes(token))) {
      penalty += 0.28;
    }
  }
  return Math.min(penalty, 0.45);
}

function clampScore(value) {
  return Math.max(0, Math.min(1, value));
}

function rankCandidate(item, row) {
  const itemBarcode = normalize(item.barcode || "");
  const rowBarcode = normalize(row.barcode || "");
  if (itemBarcode && rowBarcode && itemBarcode === rowBarcode) {
    return 1;
  }

  const itemName = normalizeMatchName(item.product_name || "");
  const rowName = normalizeMatchName(row.product_name || "");
  const nameSim = tokenSimilarity(itemName, rowName);
  const brandSim = tokenSimilarity(item.brand || "", row.brand || "");
  const variantSim = tokenSimilarity(item.variant || "", row.variant || "");
  const categorySim = tokenSimilarity(item.category || "", row.category || "");
  const itemNameTokens = tokenSet(item.product_name, item.variant);
  const rowNameTokens = tokenSet(row.product_name, row.variant);
  const itemTokens = tokenSet(item.product_name, item.variant, item.category);
  const rowTokens = tokenSet(row.product_name, row.variant, row.category);
  const sharedNameTokens = overlappingTokens(itemNameTokens, rowNameTokens);
  const sharedTokens = overlappingTokens(itemTokens, rowTokens);
  const conflictPenalty = getConflictPenalty(item, row);

  if (nameSim < 0.18 && sharedTokens.length === 0 && categorySim < 0.34) {
    return 0;
  }

  let score = 0.48 * nameSim + 0.10 * brandSim + 0.08 * variantSim + 0.14 * categorySim;
  if (itemName && rowName && itemName === rowName) score += 0.14;
  if (sharedNameTokens.length >= 1) score += 0.12;
  if (sharedNameTokens.length >= 1 && item.category && row.category && normalize(item.category) === normalize(row.category)) {
    score += 0.08;
  }
  if (sharedTokens.length >= 2) score += 0.06;
  if (item.brand && row.brand && normalize(item.brand) === normalize(row.brand)) score += 0.04;
  if (item.category && row.category && normalize(item.category) === normalize(row.category)) score += 0.05;
  score -= conflictPenalty;

  return clampScore(score);
}

function scoreCandidates(item, rows) {
  return rows
    .map((row) => ({ ...row, score: rankCandidate(item, row) }))
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return new Date(a._createdDate || 0).getTime() - new Date(b._createdDate || 0).getTime();
    });
}

function rankLocalCandidates(item, rows) {
  return scoreCandidates(item, rows)
    .filter((row) => row.score >= LOCAL_MIN_SCORE)
    .slice(0, MAX_LOCAL_CANDIDATES);
}

function defaultReasonSummary(item, count) {
  if (count === 0) {
    return `No active kitchen rows looked similar enough to swap with ${item.product_name || "this item"}.`;
  }
  return `Ranked ${count} active kitchen item(s) by product, brand, variant, and category similarity.`;
}

function buildFastSwapSuggestionsFromRows(groceryItem, rows) {
  const ranked = rankLocalCandidates(groceryItem, rows);
  return {
    candidate_ids: ranked.map((row) => row._id),
    generated_by: "fast_local",
    generated_at: new Date().toISOString(),
    reason_summary: defaultReasonSummary(groceryItem, ranked.length),
    score_metadata: ranked.map((row) => ({
      id: row._id,
      score: Number(row.score.toFixed(3)),
      product_name: row.product_name,
      brand: row.brand,
      variant: row.variant,
      category: row.category,
      product_image_url: row.product_image_url || null,
      created_at: row._createdDate || null,
    })),
  };
}

function buildDeepSwapSuggestionsFromShortlist(shortlistById, candidateIds, reasonSummary) {
  return {
    candidate_ids: candidateIds,
    generated_by: "deep_openai",
    generated_at: new Date().toISOString(),
    reason_summary: cleanText(reasonSummary) || "OpenAI reranked current kitchen candidates for this item.",
    score_metadata: candidateIds.map((id) => {
      const candidate = shortlistById.get(id);
      return {
        id,
        score: Number((candidate?.score || 0).toFixed(3)),
        product_name: candidate?.product_name || null,
        brand: candidate?.brand || null,
        variant: candidate?.variant || null,
        category: candidate?.category || null,
        product_image_url: candidate?.product_image_url || null,
        created_at: candidate?._createdDate || null,
      };
    }),
  };
}

function buildDeepNoMatchSuggestion(item, reasonSummary) {
  return {
    candidate_ids: [],
    generated_by: "deep_openai",
    generated_at: new Date().toISOString(),
    reason_summary:
      cleanText(reasonSummary) ||
      `No current kitchen item is a genuinely plausible swap target for ${item.product_name || "this item"}.`,
    score_metadata: [],
  };
}

function buildSwapInputSummary(item) {
  return [
    item.product_name && `Product: ${item.product_name}`,
    item.brand && `Brand: ${item.brand}`,
    item.variant && `Variant: ${item.variant}`,
    item.category && `Category: ${item.category}`,
    item.product_description && `Description: ${item.product_description}`,
  ].filter(Boolean).join("\n");
}

function buildShortlistSummary(shortlist) {
  return shortlist
    .map(
      (candidate, index) =>
        `${index + 1}. _id="${candidate._id}" | product=${candidate.product_name || "(none)"} | brand=${
          candidate.brand || "(none)"
        } | variant=${candidate.variant || "(none)"} | category=${candidate.category || "(none)"} | local_score=${candidate.score.toFixed(3)}`
    )
    .join("\n");
}

function chunkArray(items, size) {
  const chunks = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

async function rerankDeepSwapSuggestionsBatch(shortlistedItems) {
  if (!Array.isArray(shortlistedItems) || shortlistedItems.length === 0) {
    return new Map();
  }

  const openai = getOpenAIClient();
  const model = process.env.OPENAI_DEEP_MODEL || getOpenAIModel();
  const shortlistByIndex = new Map(shortlistedItems.map((entry) => [entry.item_index, entry]));
  const userPrompt = shortlistedItems
    .map((entry) => {
      const itemSummary = buildSwapInputSummary(entry.grocery_item);
      const shortlistSummary = buildShortlistSummary(entry.shortlist);
      return [
        `ITEM_INDEX: ${entry.item_index}`,
        "NEWLY ANALYZED ITEM:",
        itemSummary || "No product details.",
        "",
        "CURRENT KITCHEN SHORTLIST:",
        shortlistSummary || "No shortlist candidates.",
      ].join("\n");
    })
    .join("\n\n---\n\n");

  const response = await openai.responses.create({
    model,
    ...(model.startsWith("gpt-5") ? { reasoning: { effort: "low" } } : {}),
    max_output_tokens: 1600,
    text: {
      format: {
        type: "json_schema",
        name: "bulk_kitchen_swap_candidates",
        strict: true,
        schema: deepSwapBatchResponseJsonSchema,
      },
    },
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text:
              "You rank which current kitchen items are genuinely plausible swap or overlap candidates for newly analyzed grocery items. Prefer exact duplicates first, then same product or very close same-family substitutes. Be conservative and avoid broad category-only matches. Produce should not match unrelated produce just because both are vegetables or fresh items. Lemon is not a plausible swap for green onion, radish, or bell pepper. Only return candidate_ids that appear in the provided shortlist for each item.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              `${userPrompt}\n\nReturn strict JSON with this shape:\n` +
              '{ "items": [ { "item_index": 0, "candidate_ids": ["..."], "reason_summary": "..." } ] }\n' +
              "For each item_index, order candidate_ids best-first. If none are genuinely plausible, return an empty array.",
          },
        ],
      },
    ],
  });

  const parsed = parseJsonResponse(response);
  const resultMap = new Map();
  const parsedItems = Array.isArray(parsed?.items) ? parsed.items : [];

  for (const parsedItem of parsedItems) {
    const itemIndex = Number(parsedItem?.item_index);
    if (!Number.isInteger(itemIndex) || !shortlistByIndex.has(itemIndex)) {
      continue;
    }

    const shortlistEntry = shortlistByIndex.get(itemIndex);
    const shortlistById = new Map(shortlistEntry.shortlist.map((candidate) => [candidate._id, candidate]));
    const candidateIds = Array.from(
      new Set(
        (Array.isArray(parsedItem?.candidate_ids) ? parsedItem.candidate_ids : [])
          .map((id) => String(id || "").trim())
          .filter((id) => shortlistById.has(id))
      )
    ).slice(0, MAX_LOCAL_CANDIDATES);

    const suggestion =
      candidateIds.length > 0
        ? buildDeepSwapSuggestionsFromShortlist(shortlistById, candidateIds, parsedItem?.reason_summary)
        : buildDeepNoMatchSuggestion(shortlistEntry.grocery_item, parsedItem?.reason_summary);

    resultMap.set(itemIndex, suggestion);
  }

  return resultMap;
}

async function fetchKitchenCandidateRows(owner) {
  const response = await fetch(`${KITCHEN_API_BASE_URL}/kitchen/${encodeURIComponent(owner)}`, {
    headers: { "Content-Type": "application/json" },
    timeout: 20000,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || body.details || `Kitchen API list failed with HTTP ${response.status}`);
  }

  const items = Array.isArray(body?.items) ? body.items : [];
  return items
    .slice(0, MAX_KITCHEN_ROWS)
    .map((row) => ({
      _id: String(row?._id || "").trim(),
      job_id: cleanText(row?.job_id) || null,
      product_name: cleanText(row?.product_name) || null,
      brand: cleanText(row?.brand) || null,
      variant: cleanText(row?.variant) || null,
      category: cleanText(row?.category) || null,
      barcode: cleanText(row?.barcode) || null,
      product_image_url: cleanText(row?.product_image_url) || null,
      _createdDate: row?._createdDate || null,
    }))
    .filter((row) => row._id);
}

function toSimilarityInput(item) {
  return {
    product_name: cleanText(item?.check_in_label) || cleanText(item?.item_name) || null,
    brand: cleanText(item?.brand) || null,
    variant: cleanText(item?.variant) || null,
    category: cleanText(item?.category) || null,
    barcode: null,
    product_description: cleanText(item?.short_description) || null,
  };
}

async function annotateBulkItemsWithKitchenSimilarity({ owner, items }) {
  if (!owner || !Array.isArray(items) || items.length === 0) {
    return {
      items,
      debug: {
        enabled: false,
        reason: !owner ? "missing_owner" : "no_items",
      },
    };
  }

  const candidateRows = await fetchKitchenCandidateRows(owner);
  const preparedItems = items.map((item, itemIndex) => {
    const groceryItem = toSimilarityInput(item);
    const scoredCandidates = scoreCandidates(groceryItem, candidateRows).filter((row) => row.score > 0);
    const shortlist = scoredCandidates.slice(0, OPENAI_SHORTLIST_LIMIT);
    const localFallback = buildFastSwapSuggestionsFromRows(groceryItem, candidateRows);
    return {
      itemIndex,
      groceryItem,
      shortlist,
      localFallback,
    };
  });

  const deepSuggestions = new Map();
  const rerankableItems = preparedItems.filter((entry) => entry.shortlist.length > 0).map((entry) => ({
    item_index: entry.itemIndex,
    grocery_item: entry.groceryItem,
    shortlist: entry.shortlist,
  }));

  let usedFallback = false;
  // Cost control: the deep OpenAI swap rerank produces `swaps`, which the app
  // never renders (only the derived `has_similar_kitchen_items` flag is used,
  // and the local fallback below already computes it). Skip the LLM batch calls
  // when disabled and let every item use its localFallback ranking. Env-toggle so
  // it's instantly reversible: DISABLE_DEEP_SWAP_RERANK=true to skip.
  const deepSwapRerankDisabled = process.env.DISABLE_DEEP_SWAP_RERANK === "true";
  if (deepSwapRerankDisabled) {
    usedFallback = true;
  } else {
    try {
      const batches = chunkArray(rerankableItems, OPENAI_BATCH_MAX_ITEMS);
      for (const batch of batches) {
        const batchResults = await rerankDeepSwapSuggestionsBatch(batch);
        for (const [itemIndex, suggestion] of batchResults.entries()) {
          deepSuggestions.set(itemIndex, suggestion);
        }
      }
    } catch (error) {
      usedFallback = true;
      console.warn("[bulk-kitchen-similarity] OpenAI rerank failed, falling back to local ranking:", error);
    }
  }

  const annotatedItems = items.map((item, itemIndex) => {
    const prepared = preparedItems[itemIndex];
    const swaps = deepSuggestions.get(itemIndex) || prepared.localFallback;
    return {
      ...item,
      has_similar_kitchen_items: swaps.candidate_ids.length > 0,
      swaps,
    };
  });

  return {
    items: annotatedItems,
    debug: {
      enabled: true,
      candidate_pool_size: candidateRows.length,
      matched_item_count: annotatedItems.filter((item) => item.has_similar_kitchen_items).length,
      generated_by: usedFallback ? "fast_local" : "deep_openai",
      reranked_item_count: deepSuggestions.size,
      fallback_used: usedFallback,
    },
  };
}

module.exports = {
  annotateBulkItemsWithKitchenSimilarity,
};
