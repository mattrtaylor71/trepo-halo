const fetch = require("node-fetch");
const { analyzeProduct } = require("./dist/services/analyzeProduct");
const { estimateStorageGuidance } = require("./dist/utils/estimateStorageGuidance");
const { assignEmojiToItem } = require("./dist/openai/assignEmoji");
const { storageZoneToLocation } = require("./storageGuidance");

// Fields the kitchen API stores in JSON columns. The kitchen API's UPDATE
// (PATCH) path passes raw body values straight to pymysql, which cannot
// serialize a dict/list parameter ("sequence item 0: expected str instance,
// dict found"). We pre-stringify these fields so pymysql receives strings;
// the kitchen API's _json_column_value() passes strings through unchanged.
const JSON_PATCH_FIELDS = new Set([
  "ingredients",
  "harmful_ingredients",
  "similar_items",
  "alternatives",
  "healthier_alternatives",
  "store_availability",
  "storage_guidance",
  "swaps",
  "provisional_payload",
]);

const KITCHEN_API_BASE_URL =
  (process.env.KITCHEN_API_BASE_URL || "https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com").replace(/\/$/, "");

function cleanNullable(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function buildSyntheticGroceryItem(item) {
  return {
    brand: cleanNullable(item.brand),
    product_name: cleanNullable(item.product_name) || cleanNullable(item.check_in_label) || cleanNullable(item.item_name),
    variant: cleanNullable(item.variant),
    category: cleanNullable(item.category),
    estimated_price: null,
    ingredients: [],
    nutrition_summary: null,
    upf: "no",
    harmful_ingredients: [],
    similar_items: [],
    alternatives: [],
    healthier_alternatives: [],
    confidence: typeof item.confidence === "number" ? item.confidence : 0.6,
    explanation: cleanNullable(item.explanation) || cleanNullable(item.short_description) || cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    barcode: null,
    country_guess: null,
    product_description: cleanNullable(item.product_description) || cleanNullable(item.short_description) || cleanNullable(item.remaining_quantity) || cleanNullable(item.estimated_state),
    producer: null,
    size_text: cleanNullable(item.variant),
  };
}

function serializeJsonPatchFields(payload) {
  const out = {};
  for (const [key, value] of Object.entries(payload || {})) {
    if (JSON_PATCH_FIELDS.has(key) && value != null && typeof value === "object") {
      out[key] = JSON.stringify(value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

async function patchKitchenItem(owner, itemId, payload) {
  const response = await fetch(`${KITCHEN_API_BASE_URL}/kitchen/${encodeURIComponent(owner)}/${encodeURIComponent(itemId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(serializeJsonPatchFields(payload)),
    timeout: 30000,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.details || `Kitchen API patch failed with HTTP ${response.status}`);
  }
  return data;
}

function buildPatchPayload(preliminaryItem, initialPayload, analysis, storageGuidance) {
  // Keep the original confirmed product identity (product_name, brand, variant, category)
  // from the bulk/receipt analysis — stock image re-analysis can misidentify the item.
  // Only take enrichment fields (ingredients, nutrition, etc.) from the re-analysis.
  // Derive a storage_location guess from the zone but ONLY fill it if the item has
  // none yet (server-side COALESCE via `storage_location_if_empty`) — never clobber
  // a user's explicit choice.
  const storageLocationGuess = storageZoneToLocation(storageGuidance && storageGuidance.storage_zone);
  return {
    ...(storageLocationGuess ? { storage_location_if_empty: storageLocationGuess } : {}),
    product_name: initialPayload.product_name,
    brand: initialPayload.brand,
    variant: initialPayload.variant,
    category: initialPayload.category,
    confidence: initialPayload.confidence,
    explanation: initialPayload.explanation,
    product_description: initialPayload.product_description,
    barcode: analysis?.groceryItem?.barcode || null,
    country_guess: analysis?.groceryItem?.country_guess || null,
    estimated_price: analysis?.groceryItem?.estimated_price || null,
    ingredients: analysis?.groceryItem?.ingredients || [],
    nutrition_summary: analysis?.groceryItem?.nutrition_summary || null,
    upf: analysis?.groceryItem?.upf || null,
    harmful_ingredients: analysis?.groceryItem?.harmful_ingredients || [],
    similar_items: analysis?.groceryItem?.similar_items || [],
    alternatives: analysis?.groceryItem?.alternatives || [],
    healthier_alternatives: analysis?.groceryItem?.healthier_alternatives || [],
    store_availability: analysis?.store_availability || [],
    product_image_url: initialPayload.product_image_url || null,
    storage_guidance: storageGuidance || initialPayload.storage_guidance || null,
    analysis_stage: "final",
    analysis_status: "ready",
    analysis_source: "grocery_identifier_bulk_enrichment",
    needs_review: Boolean(preliminaryItem?.needs_review),
    provisional_payload: {
      ...(initialPayload?.provisional_payload || {}),
      enrichment_status: "enriched",
      enrich_debug: analysis?.debug || null,
    },
  };
}

// Find the original captured product image for a preliminary item, if one
// exists. The bulk writer stores the captured scene/receipt image in `images`
// and uses an `emoji:` placeholder for `product_image_url`. We only accept a
// real http(s) image URL (never an `emoji:` placeholder).
function resolveOriginalImageUrl(preliminaryItem, initialPayload) {
  const candidates = [
    initialPayload?.image_url,
    preliminaryItem?.image_url,
    initialPayload?.product_image_url,
    preliminaryItem?.product_image_url,
    initialPayload?.images,
    preliminaryItem?.images,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string") {
      const trimmed = candidate.trim();
      if (/^https?:\/\//i.test(trimmed)) {
        return trimmed;
      }
    }
  }
  return null;
}

async function maybeEnrichPersistedBulkItem({ owner, itemId, preliminaryItem, initialPayload }) {
  if (!process.env.BULK_ENRICH_ON_PERSIST || process.env.BULK_ENRICH_ON_PERSIST === "false") {
    return { attempted: false, status: "disabled" };
  }

  const seed = buildSyntheticGroceryItem(preliminaryItem);
  if (!seed.product_name) {
    return { attempted: true, status: "skipped_missing_name" };
  }

  // Storage guidance is always derived from the confirmed product identity (seed),
  // never from a re-identify, so it stays accurate even when re-identify is skipped.
  const storageGuidance = await estimateStorageGuidance(seed);

  try {
    const originalImageUrl = resolveOriginalImageUrl(preliminaryItem, initialPayload);

    let analysis = null;
    if (originalImageUrl) {
      // Deep re-analyze the ORIGINAL captured image to enrich (ingredients,
      // nutrition, etc.). Stock-image resolution is disabled inside
      // analyzeProduct, so this is just a deep identify on the real image.
      analysis = await analyzeProduct({ imageUrl: originalImageUrl }, { stockImageMode: "deep" });
    }

    // buildPatchPayload keeps the confirmed identity (product_name/brand/etc.)
    // from initialPayload and only layers enrichment fields from `analysis`
    // (null analysis -> empty enrichment fields, finalized from the seed identity).
    const patch = buildPatchPayload(preliminaryItem, initialPayload, analysis, storageGuidance);
    // Text-added items arrive with no photo (product_image_url null). Give them an
    // emoji placeholder — the same `emoji:<x>` convention scanned/voice items use —
    // so they render an icon instead of a blank tile. Only fills when empty, so items
    // that already have a real image or emoji are never clobbered.
    if (!patch.product_image_url) {
      const emoji = await assignEmojiToItem(seed);
      patch.product_image_url = `emoji:${emoji}`;
      patch.resized_image_url = `emoji:${emoji}`;
    }
    await patchKitchenItem(owner, itemId, patch);
    return { attempted: true, status: analysis ? "enriched" : "finalized_from_seed" };
  } catch (error) {
    const enrichmentError = error instanceof Error ? error.message : "Unknown enrichment error";
    console.error(`[BulkEnrichment] Enrichment failed for owner=${owner} item=${itemId}:`, enrichmentError);

    // Failure path MUST still mark the item final (terminal) so it can never
    // remain stuck at analysis_stage='preliminary'. Log any patch failure
    // loudly instead of swallowing it.
    try {
      await patchKitchenItem(owner, itemId, {
        analysis_stage: "final",
        analysis_status: "ready",
        analysis_source: "grocery_identifier_bulk_enrichment",
        storage_guidance: storageGuidance,
        ...(storageZoneToLocation(storageGuidance && storageGuidance.storage_zone)
          ? { storage_location_if_empty: storageZoneToLocation(storageGuidance.storage_zone) }
          : {}),
        provisional_payload: {
          ...(initialPayload?.provisional_payload || {}),
          enrichment_status: "failed",
          enrichment_error: enrichmentError,
        },
      });
    } catch (patchError) {
      const patchMessage = patchError instanceof Error ? patchError.message : "Unknown patch error";
      console.error(`[BulkEnrichment] Failure-path patch ALSO failed for owner=${owner} item=${itemId}:`, patchMessage);
      // Truly-stuck item (enrichment failed AND the finalize patch failed) — the user's
      // item may never enrich. Emit a backend_error so it surfaces in the errors dashboard.
      // Only this terminal case is marked; the recoverable OpenAI-retry dribble is not.
      try {
        console.error(JSON.stringify({
          evt: 'backend_error', service: 'enrich', op: 'enrich_kitchen_item',
          code: 'failed_and_unpatched', owner_id: owner, job_id: itemId,
          error: String(enrichmentError || patchMessage || 'enrich failed_and_unpatched').slice(0, 300),
        }));
      } catch (_) { /* marker best-effort */ }
      return {
        attempted: true,
        status: "failed_and_unpatched",
        message: enrichmentError,
        patch_error: patchMessage,
      };
    }

    return {
      attempted: true,
      status: "failed",
      message: enrichmentError,
    };
  }
}

module.exports = {
  maybeEnrichPersistedBulkItem,
};
