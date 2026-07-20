/**
 * Per-user dish nutrition memory (bug F-006) — quick-ack SYNC path.
 *
 * Prod runs DISH_ENRICHMENT_MODE=sync, so a fresh voice/text log ("log a
 * chocolate chip cookie") gets its nutrition from analyzeDishPayload() ->
 * analyzeDishFromText() on EVERY log, which re-estimates from scratch and
 * returns different calories each time (the 150-vs-210 bug).
 *
 * This module lets analyzeDishPayload SNAP a repeat log of the SAME item to the
 * previously-resolved portion/macros, only re-estimating when the user signals
 * a DIFFERENT portion. Consistency > perfect accuracy for repeat logs.
 *
 * ESM, self-contained (can't cross-require the analyze_dish_on_upload_nodejs
 * CJS copy). Gated behind DISH_NUTRITION_REUSE (default ON) for instant revert.
 */

// Mirror of data-access.mjs normalizeIngredientName() — lowercase, strip
// punctuation, drop a few stopwords.
export function normalizeDishName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\b(and|with|some|a|an|the|of)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const AMOUNT_RE = new RegExp(
  "\\b(" +
    "\\d+(?:\\.\\d+)?|" +
    "half|quarter|third|double|triple|" +
    "large|small|extra|big|tiny|jumbo|mini|huge|whole|full|" +
    "slice|slices|cup|cups|tbsp|tsp|oz|ounce|ounces|" +
    "gram|grams|kg|ml|liter|litre|" +
    "serving|servings|piece|pieces|scoop|scoops|handful|" +
    "bowl|bowls|plate|plates|bar|bars|glass|glasses|can|cans|bottle|bottles" +
  ")\\b"
);

export function mentionsAmount(text) {
  const normalized = normalizeDishName(text);
  return normalized.length > 0 && AMOUNT_RE.test(normalized);
}

export function isDishReuseEnabled(env = process.env) {
  const raw = String(env.DISH_NUTRITION_REUSE ?? "1").trim().toLowerCase();
  return !(raw === "0" || raw === "false" || raw === "off" || raw === "no");
}

export function getDishReuseWindowDays(env = process.env) {
  const parsed = parseInt(env.DISH_NUTRITION_REUSE_DAYS ?? "90", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 90;
}

export function getDishReuseScanLimit(env = process.env) {
  const parsed = parseInt(env.DISH_NUTRITION_REUSE_SCAN ?? "250", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 250;
}

export function rowHasNutrition(row) {
  const cal = row == null ? null : row.calories;
  return cal != null && cal !== "" && Number.isFinite(Number(cal));
}

function parseJsonArray(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value.map((v) => String(v || "").trim()).filter(Boolean);
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map((v) => String(v || "").trim()).filter(Boolean) : [];
    } catch (_) {
      return [];
    }
  }
  return [];
}

function parseJsonRaw(value, fallback) {
  if (value == null) return fallback;
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch (_) {
    return fallback;
  }
}

function toNumber(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

/**
 * Most RECENT prior dish for this table whose normalized dish_name EXACTLY
 * matches `dishName`, has non-null calories, created within `windowDays`.
 * EXACT normalized-name match only — never fuzzy.
 * Returns prior nutrition object or null.
 */
export async function lookupRecentDishNutrition(connection, tableName, dishName, opts = {}) {
  const target = normalizeDishName(dishName);
  if (!target || !connection || !tableName) return null;

  // Inlined (validated ints) — MySQL prepared stmts reject `LIMIT ?`/`INTERVAL ? DAY`.
  const windowDays = Math.max(1, Math.floor(Number(opts.windowDays) || 90));
  const scanLimit = Math.max(1, Math.floor(Number(opts.scanLimit) || 250));
  const excludeId = opts.excludeId || null;

  let rows;
  try {
    const [result] = await connection.execute(
      `SELECT _id, dish_name, serving_size, calories, protein, total_fat,
              total_carbohydrates, confidence, ingredients, components, allergens, explanation
         FROM \`${tableName}\`
        WHERE calories IS NOT NULL
          AND dish_name IS NOT NULL
          AND _createdDate >= DATE_SUB(NOW(), INTERVAL ${windowDays} DAY)
        ORDER BY _createdDate DESC
        LIMIT ${scanLimit}`
    );
    rows = result;
  } catch (error) {
    console.warn("[dishNutritionMemory] lookup failed (non-fatal):", error && error.message ? error.message : error);
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
      components: parseJsonRaw(row.components, []),
      allergens: parseJsonArray(row.allergens),
      explanation: row.explanation || null,
    };
  }
  return null;
}

/**
 * Decide whether a fresh-log fill should REUSE remembered nutrition.
 * `updates` is the voice tool payload. Reuse only when it's effectively the
 * same item with no portion / nutrition / ingredient-edit signal.
 * Returns { reuse: boolean, reason: string }.
 */
export function decideDishReuse({ existingRow, updates = {}, prior }) {
  if (rowHasNutrition(existingRow)) {
    return { reuse: false, reason: "row_already_has_nutrition" };
  }
  if (!prior) return { reuse: false, reason: "no_prior_match" };

  // Caller supplied explicit nutrition → honor it, don't reuse.
  if (updates.calories !== undefined || updates.fat_g !== undefined ||
      updates.carbs_g !== undefined || updates.protein_g !== undefined) {
    return { reuse: false, reason: "explicit_nutrition_provided" };
  }

  const hasIngredientEdit =
    (Array.isArray(updates.add_ingredients) && updates.add_ingredients.length > 0) ||
    (Array.isArray(updates.remove_ingredients) && updates.remove_ingredients.length > 0);
  if (hasIngredientEdit) {
    return { reuse: false, reason: "ingredient_edit" };
  }

  if (mentionsAmount(updates.user_correction)) {
    return { reuse: false, reason: "correction_mentions_amount" };
  }

  if (updates.serving_size !== undefined && updates.serving_size !== null && String(updates.serving_size).trim()) {
    const priorServing = prior.serving_size != null ? String(prior.serving_size).trim() : "";
    if (normalizeDishName(String(updates.serving_size)) !== normalizeDishName(priorServing)) {
      return { reuse: false, reason: "serving_size_changed" };
    }
  }

  return { reuse: true, reason: "same_item_no_portion_signal" };
}

/**
 * Build an analyzer-shaped result (as analyzeDishFromText would return) from a
 * remembered prior, so it can flow through mergeDishAnalysisResult unchanged.
 */
export function buildReusedAnalysisResult(prior, analysisRequest = {}) {
  return {
    dish_name: prior.dish_name || analysisRequest.dish_name || null,
    serving_size: prior.serving_size != null ? prior.serving_size : (analysisRequest.serving_size || null),
    calories: prior.calories,
    total_fat: prior.total_fat,
    total_carbohydrates: prior.total_carbohydrates,
    protein: prior.protein,
    confidence: prior.confidence != null ? prior.confidence : 0.6,
    explanation: prior.explanation || null,
    ingredients: (prior.ingredients && prior.ingredients.length > 0)
      ? prior.ingredients
      : (analysisRequest.ingredients || []),
    components: Array.isArray(prior.components) ? prior.components : [],
    allergens: prior.allergens || [],
    source_type: "reused_prior",
    evidence_urls: [],
  };
}
