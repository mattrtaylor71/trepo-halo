/**
 * Lambda handler for POST /dishes/{owner}/{item_id}/recharacterize
 * Re-runs dish identification and nutrition with optional user correction text,
 * updates the row, and returns the updated dish for the frontend.
 */
const AWS = require('aws-sdk');
const mysql = require('mysql2/promise');
const { identifyDish } = require('./dist/openai/identifyDish');
const { extractNutrition } = require('./dist/openai/extractNutrition');
const { getDishById, updateDishRow } = require('./dist/utils/mysqlDishWriter');
const { getHouseholdMemberIds } = require('./dist/utils/householdSync');
const { analyzeTextDish } = require('./textDishAnalyzer');
const { writeMasterFeedEvent } = require('./masterFeedWriter');
const {
  isDishReuseEnabled,
  getDishReuseWindowDays,
  getDishReuseScanLimit,
  lookupRecentDishNutrition,
  decideDishReuse,
} = require('./dishNutritionMemory');

const s3 = new AWS.S3();
const THUMBNAIL_MAX_SIZE = 256;

async function resizeToThumbnail(buffer) {
  try {
    const { Jimp } = require('jimp');
    const image = await Jimp.read(buffer);
    image.scaleToFit({ w: THUMBNAIL_MAX_SIZE, h: THUMBNAIL_MAX_SIZE });
    return await image.getBuffer('image/jpeg', { quality: 82 });
  } catch (e) {
    console.warn('[recharacterize] resize failed, using original:', e.message);
    return null;
  }
}

function tableName(owner) {
  const safe = (owner || '').replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safe) throw new Error('Invalid owner');
  return `${safe}_dishes`;
}

function corsHeaders() {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

function jsonResponse(statusCode, body) {
  return {
    statusCode,
    headers: corsHeaders(),
    body: JSON.stringify(body),
  };
}

function parseJsonArray(value) {
  if (value == null) return [];
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map((item) => String(item || '').trim()).filter(Boolean) : [value.trim()].filter(Boolean);
    } catch (_) {
      return [value.trim()].filter(Boolean);
    }
  }
  return [];
}

function normalizeIngredientName(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\b(and|with|some|a|an|the|of)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeIngredientList(value) {
  const seen = new Set();
  const result = [];
  for (const entry of parseJsonArray(value)) {
    const key = normalizeIngredientName(entry);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    result.push(entry);
  }
  return result;
}

function inferDishNameFromIngredients(ingredients) {
  const normalized = normalizeIngredientList(ingredients);
  if (normalized.length === 0) return null;
  if (normalized.length === 1) return normalized[0];
  if (normalized.length === 2) return `${normalized[0]} with ${normalized[1]}`;
  return `${normalized[0]} with ${normalized.slice(1, -1).join(', ')} and ${normalized[normalized.length - 1]}`;
}

function mergeIngredientLists(existingIngredients, addIngredients, removeIngredients) {
  const current = normalizeIngredientList(existingIngredients);
  const additions = normalizeIngredientList(addIngredients);
  const removals = new Set(normalizeIngredientList(removeIngredients).map(normalizeIngredientName));
  const next = current.filter((ingredient) => !removals.has(normalizeIngredientName(ingredient)));
  for (const ingredient of additions) {
    if (!next.some((currentIngredient) => normalizeIngredientName(currentIngredient) === normalizeIngredientName(ingredient))) {
      next.push(ingredient);
    }
  }
  return next;
}

function buildTextEditInput(row, body, userCorrection) {
  const existingIngredients = parseJsonArray(row.ingredients);
  const replacementIngredients = normalizeIngredientList(body.ingredients);
  const nextIngredients = replacementIngredients.length > 0
    ? replacementIngredients
    : mergeIngredientLists(existingIngredients, body.add_ingredients, body.remove_ingredients);
  const requestedDishName = body.dish_name != null ? String(body.dish_name).trim() : '';
  const servingSize = body.serving_size != null ? String(body.serving_size).trim() : '';
  const dishName = requestedDishName
    || inferDishNameFromIngredients(nextIngredients)
    || row.dish_name
    || 'Voice meal';

  return {
    dish_name: dishName,
    serving_size: servingSize || row.serving_size || null,
    ingredients: nextIngredients.length > 0 ? nextIngredients : existingIngredients,
    user_correction: userCorrection || null,
    existing_context: {
      dish_name: row.dish_name || null,
      serving_size: row.serving_size || null,
      explanation: row.explanation || null,
      ingredients: existingIngredients,
      allergens: parseJsonArray(row.allergens),
    },
  };
}

function numberOrFallback(value, fallback = null) {
  if (value == null || value === '') return fallback;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function buildTextUpdateFields(row, analysis) {
  const ingredients = normalizeIngredientList(analysis.ingredients);
  const allergens = normalizeIngredientList(analysis.allergens);
  return {
    dish_name: analysis.dish_name || row.dish_name || inferDishNameFromIngredients(ingredients) || 'Voice meal',
    confidence: numberOrFallback(analysis.confidence, numberOrFallback(row.confidence, null)),
    explanation: analysis.explanation || row.explanation || 'Voice meal',
    serving_size: analysis.serving_size != null ? analysis.serving_size : (row.serving_size || null),
    calories: analysis.calories != null ? analysis.calories : numberOrFallback(row.calories, null),
    total_fat: analysis.total_fat != null ? analysis.total_fat : numberOrFallback(row.total_fat, null),
    total_carbohydrates: analysis.total_carbohydrates != null ? analysis.total_carbohydrates : numberOrFallback(row.total_carbohydrates, null),
    protein: analysis.protein != null ? analysis.protein : numberOrFallback(row.protein, null),
    ingredients: JSON.stringify(ingredients),
    allergens: JSON.stringify(allergens),
    dish_image_url: row.dish_image_url || null,
    dish_image_key: row.dish_image_key || null,
  };
}

// Bug F-006: build the dish update fields by SNAPPING to a remembered prior
// nutrition result (same item, same portion) instead of a fresh LLM estimate.
function buildReusedUpdateFields(row, textEditInput, prior) {
  const ingredients = (prior.ingredients && prior.ingredients.length > 0)
    ? normalizeIngredientList(prior.ingredients)
    : normalizeIngredientList(textEditInput.ingredients);
  const allergens = normalizeIngredientList(prior.allergens || []);
  return {
    dish_name: textEditInput.dish_name || prior.dish_name || row.dish_name || 'Voice meal',
    confidence: prior.confidence != null ? prior.confidence : numberOrFallback(row.confidence, null),
    explanation: prior.explanation || row.explanation || 'Voice meal',
    serving_size: prior.serving_size != null ? prior.serving_size : (row.serving_size || null),
    calories: prior.calories != null ? prior.calories : numberOrFallback(row.calories, null),
    total_fat: prior.total_fat != null ? prior.total_fat : numberOrFallback(row.total_fat, null),
    total_carbohydrates: prior.total_carbohydrates != null ? prior.total_carbohydrates : numberOrFallback(row.total_carbohydrates, null),
    protein: prior.protein != null ? prior.protein : numberOrFallback(row.protein, null),
    ingredients: JSON.stringify(ingredients),
    allergens: JSON.stringify(allergens),
    dish_image_url: row.dish_image_url || null,
    dish_image_key: row.dish_image_key || null,
  };
}

async function tryUpdateAnalysisState(connection, table, itemId, status, errorMessage) {
  try {
    await connection.execute(
      `UPDATE \`${table}\`
       SET analysis_status = ?, analysis_error = ?, _updatedDate = NOW()
       WHERE _id = ?`,
      [status, errorMessage || null, itemId]
    );
  } catch (error) {
    if (error?.code !== 'ER_BAD_FIELD_ERROR') {
      throw error;
    }
  }
}

async function updateAnalysisStateAcrossHousehold(connection, owner, itemId, status, errorMessage) {
  // Dishes are user-specific — only update the acting user's table
  await tryUpdateAnalysisState(connection, tableName(owner), itemId, status, errorMessage);
}

exports.handler = async (event) => {
  const { requireOwner } = require('./trepo_auth');
  const _denied = requireOwner(event);
  if (_denied !== null) {
    return _denied;
  }
  const pathParams = event.pathParameters || {};
  const owner = pathParams.owner;
  const itemId = pathParams.item_id;
  let body = {};
  try {
    body = typeof event.body === 'string' ? JSON.parse(event.body || '{}') : event.body || {};
  } catch (e) {
    return jsonResponse(400, { error: 'Invalid JSON body' });
  }
  const userCorrection = body.user_correction != null ? String(body.user_correction).trim() : '';
  const requestPath = String(event.rawPath || event.requestContext?.http?.path || '');
  const isTextEditPath = requestPath.endsWith('/edit');

  if (!owner || !itemId) {
    return jsonResponse(400, { error: 'Missing owner or item_id' });
  }

  const dbConfig = {
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT || '3306', 10),
    user: process.env.DB_USER,
    password: process.env.DB_PASS,
    database: process.env.DB_NAME,
    charset: 'utf8mb4',
  };
  if (!dbConfig.host || !dbConfig.user || !dbConfig.password || !dbConfig.database) {
    console.error('[recharacterize] Missing DB env vars');
    return jsonResponse(500, { error: 'Server configuration error' });
  }

  let connection;
  try {
    connection = await mysql.createConnection(dbConfig);
  } catch (err) {
    console.error('[recharacterize] DB connection failed:', err);
    return jsonResponse(500, { error: 'Database unavailable' });
  }

  try {
    const table = tableName(owner);
    const row = await getDishById(connection, table, itemId);
    if (!row) {
      return jsonResponse(404, { error: `Dish with id ${itemId} not found` });
    }

    const s3Key = row.s3_key || row.images;
    const hasStructuredTextEdit = (
      Array.isArray(body.ingredients)
      || Array.isArray(body.add_ingredients)
      || Array.isArray(body.remove_ingredients)
      || typeof body.dish_name === 'string'
      || typeof body.serving_size === 'string'
    );
    const useTextEdit = isTextEditPath || hasStructuredTextEdit || !s3Key || typeof s3Key !== 'string';

    let updateFields;
    let didReuse = false;
    if (useTextEdit) {
      if (!userCorrection && !hasStructuredTextEdit) {
        return jsonResponse(400, { error: 'Text dish edits require user_correction or ingredient/name updates' });
      }
      const textEditInput = buildTextEditInput(row, body, userCorrection);

      // Bug F-006: per-user dish nutrition memory. On a fresh-log fill (the
      // preliminary row has no nutrition yet), snap a repeat log of the SAME
      // item to the previously-resolved portion/macros so calories stop
      // drifting (150 vs 210) between identical logs. Only re-estimate when the
      // user signals a different portion. Gated behind DISH_NUTRITION_REUSE.
      let prior = null;
      if (isDishReuseEnabled(process.env)) {
        try {
          prior = await lookupRecentDishNutrition(connection, table, textEditInput.dish_name, {
            excludeId: itemId,
            windowDays: getDishReuseWindowDays(process.env),
            scanLimit: getDishReuseScanLimit(process.env),
          });
        } catch (memoryError) {
          console.warn('[recharacterize] dish nutrition lookup failed (non-fatal):', memoryError.message || memoryError);
        }
      }
      const reuseDecision = isDishReuseEnabled(process.env)
        ? decideDishReuse({ row, body, userCorrection, prior })
        : { reuse: false, reason: 'disabled' };

      if (reuseDecision.reuse && prior) {
        didReuse = true;
        updateFields = buildReusedUpdateFields(row, textEditInput, prior);
        console.log('[recharacterize] Reused prior nutrition for', JSON.stringify(textEditInput.dish_name), '→', prior.calories, 'cal (skipped LLM); reason:', reuseDecision.reason);
      } else {
        console.log('[recharacterize] Running text dish edit with user_correction:', userCorrection || '(none)', 'path:', requestPath || '(unknown)', 'reuse:', reuseDecision.reason);
        const analysis = await analyzeTextDish(textEditInput, { owner });
        updateFields = buildTextUpdateFields(row, analysis);
      }
    } else {
      const bucketName = process.env.BUCKET_NAME;
      if (!bucketName) {
        console.error('[recharacterize] BUCKET_NAME not set');
        return jsonResponse(500, { error: 'Server configuration error' });
      }
      // If s3_key looks like a URL, extract key or use images URL for fetch
      let imageBuffer;
      if (s3Key.startsWith('http')) {
        const https = require('https');
        const url = s3Key;
        imageBuffer = await new Promise((resolve, reject) => {
          https.get(url, (res) => {
            const chunks = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', () => resolve(Buffer.concat(chunks)));
            res.on('error', reject);
          }).on('error', reject);
        });
      } else {
        const obj = await s3.getObject({ Bucket: bucketName, Key: s3Key }).promise();
        imageBuffer = obj.Body;
      }
      if (!imageBuffer || imageBuffer.length === 0) {
        return jsonResponse(500, { error: 'Could not load original image' });
      }

      const existingDishContext = (userCorrection && (row.dish_name || row.explanation)) ? {
        dish_name: row.dish_name || null,
        explanation: row.explanation || null,
      } : null;
      const existingNutritionContext = (userCorrection && (row.dish_name || row.serving_size || row.ingredients || row.allergens)) ? {
        dish_name: row.dish_name || null,
        explanation: row.explanation || null,
        serving_size: row.serving_size || null,
        ingredients: parseJsonArray(row.ingredients) || (row.ingredients ? [row.ingredients] : null),
        allergens: parseJsonArray(row.allergens) || (row.allergens ? [row.allergens] : null),
      } : null;

      console.log('[recharacterize] Running identifyDish and extractNutrition (parallel) with user_correction:', userCorrection || '(none)', 'existingContext:', !!existingDishContext);
      const [dish, nutritionData] = await Promise.all([
        identifyDish(imageBuffer, userCorrection || null, existingDishContext),
        extractNutrition(imageBuffer, userCorrection || null, existingNutritionContext),
      ]);

      const confidence = dish.confidence != null
        ? Math.max(0, Math.min(1, dish.confidence))
        : (nutritionData.confidence != null ? Math.max(0, Math.min(1, nutritionData.confidence)) : null);
      const dishName = dish.dish_name || nutritionData.dish_name || null;
      const explanation = dish.explanation || nutritionData.explanation || null;
      const ingredientsJson = JSON.stringify(nutritionData.ingredients || []);
      const allergensJson = JSON.stringify(nutritionData.allergens || []);

      updateFields = {
        dish_name: dishName,
        confidence,
        explanation,
        serving_size: nutritionData.serving_size || null,
        calories: nutritionData.calories != null ? nutritionData.calories : null,
        total_fat: nutritionData.total_fat != null ? nutritionData.total_fat : null,
        saturated_fat: nutritionData.saturated_fat != null ? nutritionData.saturated_fat : null,
        trans_fat: nutritionData.trans_fat != null ? nutritionData.trans_fat : null,
        cholesterol: nutritionData.cholesterol != null ? nutritionData.cholesterol : null,
        sodium: nutritionData.sodium != null ? nutritionData.sodium : null,
        total_carbohydrates: nutritionData.total_carbohydrates != null ? nutritionData.total_carbohydrates : null,
        dietary_fiber: nutritionData.dietary_fiber != null ? nutritionData.dietary_fiber : null,
        sugars: nutritionData.sugars != null ? nutritionData.sugars : null,
        protein: nutritionData.protein != null ? nutritionData.protein : null,
        vitamin_a: nutritionData.vitamin_a != null ? nutritionData.vitamin_a : null,
        vitamin_c: nutritionData.vitamin_c != null ? nutritionData.vitamin_c : null,
        calcium: nutritionData.calcium != null ? nutritionData.calcium : null,
        iron: nutritionData.iron != null ? nutritionData.iron : null,
        ingredients: ingredientsJson,
        allergens: allergensJson,
        dish_image_url: row.dish_image_url || null,
        dish_image_key: row.dish_image_key || null,
      };
    }

    // Dishes are user-specific — only update the acting user's table
    const ownerTable = tableName(owner);
    await updateDishRow(connection, ownerTable, itemId, updateFields);
    await tryUpdateAnalysisState(connection, ownerTable, itemId, 'complete', null);

    const updated = await getDishById(connection, tableName(owner), itemId);
    const item = updated ? sanitizeRow(updated) : { _id: itemId, ...row, dish_name: dishName, explanation, dish_image_url: dishImageUrl, dish_image_key: dishImageKey };

    try {
      await writeMasterFeedEvent({
        owner,
        device_id: item?._device || row?._device || null,
        user_id: item?.user_id || row?.user_id || null,
        job_id: item?.job_id || row?.job_id || null,
        item_id: itemId,
        event_type: 'dish_update',
        entity_type: 'dish',
        action: item?.action || row?.action || null,
        title: item?.dish_name || row?.dish_name || 'Dish',
        source_table: `${owner.replace(/[^a-zA-Z0-9_-]/g, '')}_dishes`,
        source_path: requestPath || 'analyze_dish_on_upload_nodejs/recharacterize.js',
        source_system: isTextEditPath ? 'dish_edit' : 'dish_recharacterize',
        primary_image_url: item?.dish_image_url || row?.dish_image_url || item?.images || row?.images || null,
        secondary_image_url: item?.images || row?.images || null,
        event_key: `dish-update:${itemId}:${item?._updatedDate || new Date().toISOString()}`,
        metadata: {
          mode: useTextEdit ? 'text_edit' : 'recharacterize',
          changed_fields: Object.keys(updateFields || {}).sort(),
          user_correction: userCorrection || null,
          reused_prior_nutrition: didReuse,
          source_type: didReuse ? 'reused_prior' : undefined,
        },
      });
    } catch (masterFeedError) {
      console.error('[master-feed] Failed to record dish update (non-fatal):', masterFeedError);
    }

    return jsonResponse(200, {
      message: useTextEdit ? 'Dish updated successfully' : 'Dish recharacterized successfully',
      item,
    });
  } catch (err) {
    console.error('[recharacterize] Error:', err);
    if (connection && owner && itemId) {
      try {
        await updateAnalysisStateAcrossHousehold(connection, owner, itemId, 'failed', err.message || 'Recharacterize failed');
      } catch (statusError) {
        console.error('[recharacterize] Failed to persist analysis error state:', statusError);
      }
    }
    return jsonResponse(500, { error: err.message || 'Recharacterize failed' });
  } finally {
    if (connection) await connection.end();
  }
};

function sanitizeRow(row) {
  const out = { ...row };
  for (const k of Object.keys(out)) {
    if (out[k] instanceof Date) out[k] = out[k].toISOString();
    else if (typeof out[k] === 'object' && out[k] !== null && typeof out[k].toString === 'function' && out[k].constructor.name === 'Date') {
      out[k] = out[k].toISOString();
    }
  }
  return out;
}
