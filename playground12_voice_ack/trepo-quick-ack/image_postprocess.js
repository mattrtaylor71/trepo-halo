import { jsonResponse } from "./lib/http.mjs";
import {
  getDishById,
  getKitchenItemById,
  setDishGeneratedImage,
  setKitchenItemProductImage
} from "./lib/data-access.mjs";
import { lookupUserContextByOwnerId } from "./lib/user-context.mjs";
import { buildDishImageAsset, buildKitchenProductImageAsset } from "./lib/voice-image-assets.mjs";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Emits a structured marker for silent/degraded backend failures so a
// CloudWatch metric filter on "backend_error" can alert. Never throws.
function reportBackendError({ op, ownerId, code, err }) {
  try {
    console.error(JSON.stringify({
      evt: "backend_error",
      service: "voice",
      op,
      owner_id: ownerId || null,
      code: code || "error",
      error: String((err && (err.message || err)) || op).slice(0, 500),
      job_id: null
    }));
  } catch (_) { /* never let logging throw */ }
}

async function loadDishForImage(context, rowId, options = {}) {
  const attempts = Number.isFinite(Number(options.attempts)) ? Number(options.attempts) : 3;
  const waitMs = Number.isFinite(Number(options.waitMs)) ? Number(options.waitMs) : 2500;
  let dish = null;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    dish = await getDishById(context, rowId, options);
    if (!dish || dish.analysis_status !== "pending") {
      return dish;
    }
    if (attempt < attempts - 1) {
      await sleep(waitMs);
    }
  }

  return dish;
}

async function processDishImageJob(userContext, rowId, env = process.env) {
  const dish = await loadDishForImage(userContext, rowId, { env });
  if (!dish || dish.action === "OUT") {
    return {
      ok: true,
      skipped: true,
      reason: "dish_missing_or_consumed"
    };
  }

  const asset = await buildDishImageAsset({
    dish,
    rowId,
    userId: dish.user_id || userContext.userId || userContext.tableOwnerId,
    bucketName: String(env.ASSET_BUCKET_NAME || "").trim(),
    env
  });

  if (!asset?.url) {
    return {
      ok: true,
      skipped: true,
      reason: "dish_asset_not_generated"
    };
  }

  const updatedDish = await setDishGeneratedImage(userContext, rowId, {
    dish_image_url: asset.url,
    dish_image_key: asset.key || null
  }, { env });

  return {
    ok: true,
    assetType: "dish",
    rowId,
    source: asset.source || "generated",
    dish: updatedDish
  };
}

async function processKitchenImageJob(userContext, rowId, env = process.env) {
  const item = await getKitchenItemById(userContext, rowId, { env });
  if (!item || item.action === "OUT") {
    return {
      ok: true,
      skipped: true,
      reason: "kitchen_item_missing_or_out"
    };
  }

  if (item.product_image_url) {
    return {
      ok: true,
      skipped: true,
      reason: "already_has_product_image",
      rowId
    };
  }

  const asset = await buildKitchenProductImageAsset({
    item,
    rowId,
    userId: item.user_id || userContext.userId || userContext.tableOwnerId,
    bucketName: String(env.ASSET_BUCKET_NAME || "").trim(),
    env
  });

  if (!asset?.url) {
    return {
      ok: true,
      skipped: true,
      reason: "kitchen_asset_not_generated"
    };
  }

  const updatedItem = await setKitchenItemProductImage(userContext, rowId, {
    product_image_url: asset.url,
    product_image_key: asset.key || null
  }, { env });

  return {
    ok: true,
    assetType: "kitchen",
    rowId,
    source: asset.source || "generated_icon",
    item: updatedItem
  };
}

export async function handler(event = {}) {
  try {
    const ownerId = String(event.ownerId || "").trim();
    const assetType = String(event.assetType || "").trim().toLowerCase();
    const rowId = String(event.rowId || "").trim();

    if (!ownerId || !assetType || !rowId) {
      return jsonResponse(400, {
        ok: false,
        error: "ownerId, assetType, and rowId are required"
      });
    }

    const userContext = await lookupUserContextByOwnerId(ownerId, { env: process.env });
    const result = assetType === "dish"
      ? await processDishImageJob(userContext, rowId, process.env)
      : assetType === "kitchen"
        ? await processKitchenImageJob(userContext, rowId, process.env)
        : {
            ok: false,
            error: `Unsupported assetType: ${assetType}`
          };

    console.log("[DEBUG] voice image postprocess result:", JSON.stringify({
      ownerId,
      assetType,
      rowId,
      sourceTool: event.sourceTool || null,
      result
    }));

    return jsonResponse(result.ok === false ? 400 : 200, result);
  } catch (error) {
    console.error("[ERROR] voice image postprocess failed:", error);
    reportBackendError({
      op: "image_postprocess",
      ownerId: (event && event.ownerId) ? String(event.ownerId).trim() : null,
      code: (event && event.assetType) ? `postprocess_failed:${String(event.assetType).trim().toLowerCase()}` : "postprocess_failed",
      err: error
    });
    return jsonResponse(error.statusCode || 500, {
      ok: false,
      error: error.message || "Voice image postprocess failed"
    });
  }
}
