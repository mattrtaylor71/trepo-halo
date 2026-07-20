/**
 * Per-user dish nutrition memory (bug F-006).
 *
 * The SAME food logged repeatedly (e.g. "log a chocolate chip cookie") used to
 * re-estimate calories/macros from scratch every time, so a repeat log returned
 * different numbers (150 vs 210 cal). This helper lets the fresh-log nutrition
 * filler SNAP a repeat log to the previously-resolved portion/macros for the
 * SAME item, only re-estimating when the user signals a DIFFERENT portion.
 *
 * Consistency > perfect accuracy for repeat logs.
 *
 * All behavior is gated behind DISH_NUTRITION_REUSE (default ON) so it is
 * instantly reversible.
 */

// Mirror of recharacterize.js normalizeIngredientName() — lowercase, strip
// punctuation, drop a few stopwords. Kept local so this module is standalone.
function normalizeDishName(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\b(and|with|some|a|an|the|of)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Tokens that indicate an explicit quantity / portion / size. Used to detect
// when a request is asking for a DIFFERENT portion than the remembered one.
const AMOUNT_RE = new RegExp(
  '\\b(' +
    '\\d+(?:\\.\\d+)?|' + // any number: "3", "1.5"
    'half|quarter|third|double|triple|' +
    'large|small|extra|big|tiny|jumbo|mini|huge|whole|full|' +
    'slice|slices|cup|cups|tbsp|tsp|oz|ounce|ounces|' +
    'gram|grams|kg|ml|liter|litre|' +
    'serving|servings|piece|pieces|scoop|scoops|handful|' +
    'bowl|bowls|plate|plates|bar|bars|glass|glasses|can|cans|bottle|bottles' +
  ')\\b'
);

function mentionsAmount(text) {
  const normalized = normalizeDishName(text);
  return normalized.length > 0 && AMOUNT_RE.test(normalized);
}

function isDishReuseEnabled(env = process.env) {
  // Default ON. Set DISH_NUTRITION_REUSE=0 (or "false") to disable.
  const raw = String(env.DISH_NUTRITION_REUSE ?? '1').trim().toLowerCase();
  return !(raw === '0' || raw === 'false' || raw === 'off' || raw === 'no');
}

function getDishReuseWindowDays(env = process.env) {
  const parsed = parseInt(env.DISH_NUTRITION_REUSE_DAYS ?? '90', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 90;
}

function getDishReuseScanLimit(env = process.env) {
  const parsed = parseInt(env.DISH_NUTRITION_REUSE_SCAN ?? '250', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 250;
}

// A dish row "has nutrition" if calories is a real number. Preliminary rows
// (fresh voice/text logs awaiting enrichment) have NULL calories.
function rowHasNutrition(row) {
  const cal = row == null ? null : row.calories;
  return cal != null && cal !== '' && Number.isFinite(Number(cal));
}

function parseJsonArray(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value.map((v) => String(v || '').trim()).filter(Boolean);
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map((v) => String(v || '').trim()).filter(Boolean) : [];
    } catch (_) {
      return [];
    }
  }
  return [];
}

function toNumber(value) {
  if (value == null || value === '') return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

/**
 * Find the MOST RECENT prior dish (for the same table/owner) whose normalized
 * dish_name EXACTLY matches `dishName`, has non-null calories, and was created
 * within `windowDays`. EXACT normalized-name match only — never fuzzy, so
 * "chicken soup" can never snap to "chicken salad".
 *
 * Returns { dish_name, serving_size, calories, protein, total_fat,
 *           total_carbohydrates, confidence, ingredients[], allergens[],
 *           explanation } or null.
 */
async function lookupRecentDishNutrition(connection, tableName, dishName, opts = {}) {
  const target = normalizeDishName(dishName);
  if (!target) return null;

  // windowDays / scanLimit are inlined (as validated integers) rather than
  // bound: MySQL prepared statements reject `LIMIT ?` / `INTERVAL ? DAY`.
  const windowDays = Math.max(1, Math.floor(Number(opts.windowDays) || 90));
  const scanLimit = Math.max(1, Math.floor(Number(opts.scanLimit) || 250));
  const excludeId = opts.excludeId || null;

  let rows;
  try {
    const [result] = await connection.execute(
      `SELECT _id, dish_name, serving_size, calories, protein, total_fat,
              total_carbohydrates, confidence, ingredients, allergens, explanation
         FROM \`${tableName}\`
        WHERE calories IS NOT NULL
          AND dish_name IS NOT NULL
          AND _createdDate >= DATE_SUB(NOW(), INTERVAL ${windowDays} DAY)
        ORDER BY _createdDate DESC
        LIMIT ${scanLimit}`
    );
    rows = result;
  } catch (error) {
    // Missing table/column etc. — never block the estimate on a memory miss.
    console.warn('[dishNutritionMemory] lookup failed (non-fatal):', error && error.message ? error.message : error);
    return null;
  }

  if (!Array.isArray(rows)) return null;

  for (const row of rows) {
    if (excludeId && String(row._id) === String(excludeId)) continue;
    if (normalizeDishName(row.dish_name) !== target) continue;
    const calories = toNumber(row.calories);
    if (calories == null) continue;
    return {
      dish_name: row.dish_name || null,
      serving_size: row.serving_size || null,
      calories,
      protein: toNumber(row.protein),
      total_fat: toNumber(row.total_fat),
      total_carbohydrates: toNumber(row.total_carbohydrates),
      confidence: toNumber(row.confidence),
      ingredients: parseJsonArray(row.ingredients),
      allergens: parseJsonArray(row.allergens),
      explanation: row.explanation || null,
    };
  }
  return null;
}

/**
 * Decide whether a fresh-log fill should REUSE remembered nutrition.
 *
 * Reuse only when it's effectively the same item with NO portion signal:
 *  - the row being filled has no nutrition yet (a fresh preliminary log), and
 *  - the request carries no explicit quantity/amount correction, and
 *  - no add/remove ingredient edit, and
 *  - any explicit serving_size matches the remembered one.
 *
 * `prior` is the result of lookupRecentDishNutrition (or null).
 * Returns { reuse: boolean, reason: string }.
 */
function decideDishReuse({ row, body = {}, userCorrection = '', prior }) {
  if (rowHasNutrition(row)) {
    // row already has nutrition → this is a genuine edit, honor it, never reuse.
    return { reuse: false, reason: 'row_already_has_nutrition' };
  }
  if (!prior) return { reuse: false, reason: 'no_prior_match' };

  if (mentionsAmount(userCorrection)) {
    return { reuse: false, reason: 'correction_mentions_amount' };
  }

  const hasIngredientEdit =
    (Array.isArray(body.add_ingredients) && body.add_ingredients.length > 0) ||
    (Array.isArray(body.remove_ingredients) && body.remove_ingredients.length > 0);
  if (hasIngredientEdit) {
    return { reuse: false, reason: 'ingredient_edit' };
  }

  const explicitServing = body.serving_size != null ? String(body.serving_size).trim() : '';
  if (explicitServing) {
    const priorServing = prior.serving_size != null ? String(prior.serving_size).trim() : '';
    if (normalizeDishName(explicitServing) !== normalizeDishName(priorServing)) {
      return { reuse: false, reason: 'serving_size_changed' };
    }
  }

  return { reuse: true, reason: 'same_item_no_portion_signal' };
}

module.exports = {
  normalizeDishName,
  mentionsAmount,
  isDishReuseEnabled,
  getDishReuseWindowDays,
  getDishReuseScanLimit,
  rowHasNutrition,
  lookupRecentDishNutrition,
  decideDishReuse,
};
