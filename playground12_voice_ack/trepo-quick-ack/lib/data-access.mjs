// DO NOT deploy this stack from a stale bundle. template.yaml uses CodeUri:
// .sam-src, so `sam build/deploy` ships .sam-src/…/lib/*.mjs — NOT this repo lib/.
// Keep .sam-src/ (and any .aws-sam/build/) copies of the lib/ files IN SYNC with
// these, or a deploy will silently revert the WRITE_SHARED_ONLY dual-write fixes
// (voice dishes/discards/shopping go invisible again — regressions 07-02 / 07-06).
// The trepo-voice-dualwrite-canary + alarm VoiceDualWriteMiss will catch a revert.
import crypto from "node:crypto";
import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";
import {
  ensureColumn,
  getTableColumns,
  parseJsonColumn,
  sanitizeIdentifier,
  tableExists,
  withDbConnection
} from "./mysql.mjs";
import { analyzeDishFromText } from "../../../shared/voice-assistant/text-dish-analyzer.mjs";
import { dispatchDishEnrichmentJob, shouldUseAsyncDishEnrichment } from "./dish-enrichment-dispatcher.mjs";

const WRITE_SHARED_ONLY = (process.env.WRITE_SHARED_ONLY || 'false').toLowerCase() === 'true';

const DEFAULT_HOUSEHOLD_API_BASE_URL = "https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com";
const RECIPE_SEARCH_USER_AGENT = "Mozilla/5.0 (compatible; TrepoRecipeResolver/1.0)";
const BLOCKED_RECIPE_SEARCH_HOSTS = new Set([
  "facebook.com",
  "www.facebook.com",
  "instagram.com",
  "www.instagram.com",
  "m.instagram.com",
  "tiktok.com",
  "www.tiktok.com",
  "m.tiktok.com",
  "pinterest.com",
  "www.pinterest.com",
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "youtu.be",
  "x.com",
  "twitter.com",
  "www.twitter.com"
]);
let lambdaClient = null;

function getLambdaClient(env = process.env) {
  if (!lambdaClient) {
    lambdaClient = new LambdaClient({
      region: env.AWS_REGION || env.AWS_DEFAULT_REGION || "us-east-1"
    });
  }

  return lambdaClient;
}

function normalizeName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function toTitleCaseWords(value) {
  return String(value || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

function formatUserFacingItemName(value) {
  return toTitleCaseWords(value);
}

function buildLikePattern(value) {
  return `%${normalizeName(value).replace(/\s+/g, "%")}%`;
}

function clampLimit(value, fallback = 5, max = 10) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback;
  }

  return Math.min(Math.floor(numeric), max);
}

function uniqueStrings(values = []) {
  const seen = new Set();
  const items = [];
  for (const value of values) {
    const text = String(value || "").trim();
    if (!text) {
      continue;
    }
    const key = normalizeName(text);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push(text);
  }
  return items;
}

function intersectNormalizedStrings(left = [], right = []) {
  const rightSet = new Set((right || []).map((value) => normalizeName(value)).filter(Boolean));
  return uniqueStrings((left || []).filter((value) => rightSet.has(normalizeName(value))));
}

function decodeHtmlEntities(value) {
  return String(value || "")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function stripHtmlTags(value) {
  return decodeHtmlEntities(String(value || "").replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

function tokenizeSearchText(value) {
  return normalizeName(value)
    .split(/\s+/)
    .filter((token) => token && token.length >= 3);
}

function scoreTokenOverlap(left, right) {
  const leftTokens = new Set(tokenizeSearchText(left));
  const rightTokens = new Set(tokenizeSearchText(right));
  let score = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      score += token.length >= 6 ? 18 : 10;
    }
  }
  return score;
}

function buildRecipeSearchQueries(recipe) {
  const title = String(recipe?.title || "").trim();
  const ingredients = Array.isArray(recipe?.ingredients) ? recipe.ingredients.slice(0, 4) : [];
  const queries = [];
  if (title) {
    queries.push(`${title} recipe`);
  }
  if (title && ingredients.length > 0) {
    queries.push(`${title} ${ingredients.join(" ")} recipe`);
  }
  if (!title && ingredients.length > 0) {
    queries.push(`${ingredients.join(" ")} recipe`);
  }
  return uniqueStrings(queries).slice(0, 2);
}

function unwrapDuckDuckGoHref(rawHref) {
  const href = decodeHtmlEntities(rawHref);
  if (!href) {
    return null;
  }
  try {
    const parsed = new URL(href, "https://html.duckduckgo.com");
    const uddg = parsed.searchParams.get("uddg");
    return uddg ? decodeURIComponent(uddg) : parsed.toString();
  } catch {
    return null;
  }
}

function isLikelyRecipeUrl(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    if (!["http:", "https:"].includes(parsed.protocol)) {
      return false;
    }
    if (BLOCKED_RECIPE_SEARCH_HOSTS.has(host)) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

async function fetchRecipeSearchResults(query) {
  const response = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, {
    headers: {
      "User-Agent": RECIPE_SEARCH_USER_AGENT,
      "Accept-Language": "en-US,en;q=0.9"
    }
  });
  const html = await response.text();
  if (!response.ok) {
    throw new Error(`Recipe search failed with ${response.status}`);
  }

  const matches = [...html.matchAll(/<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi)];
  const results = [];
  const seen = new Set();
  for (const match of matches) {
    const url = unwrapDuckDuckGoHref(match[1]);
    if (!url || !isLikelyRecipeUrl(url)) {
      continue;
    }
    if (seen.has(url)) {
      continue;
    }
    seen.add(url);
    results.push({
      url,
      title: stripHtmlTags(match[2])
    });
    if (results.length >= 6) {
      break;
    }
  }
  return results;
}

function parseRecipeTextSections(recipeText) {
  const lines = String(recipeText || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const parsed = {
    title: "",
    ingredients: [],
    instructions: [],
    notes: []
  };
  let section = null;
  for (const line of lines) {
    if (/^Title:?$/i.test(line)) {
      section = "title";
      continue;
    }
    if (/^Ingredients:?$/i.test(line)) {
      section = "ingredients";
      continue;
    }
    if (/^Instructions:?$/i.test(line)) {
      section = "instructions";
      continue;
    }
    if (/^Notes:?$/i.test(line)) {
      section = "notes";
      continue;
    }
    if (section === "title" && !parsed.title) {
      parsed.title = line;
      continue;
    }
    if (section === "ingredients") {
      parsed.ingredients.push(line.replace(/^-+\s*/, ""));
      continue;
    }
    if (section === "instructions") {
      parsed.instructions.push(line.replace(/^\d+\.\s*/, ""));
      continue;
    }
    if (section === "notes") {
      parsed.notes.push(line.replace(/^-+\s*/, ""));
    }
  }
  return {
    title: String(parsed.title || "").trim(),
    ingredients: uniqueStrings(parsed.ingredients),
    instructions: uniqueStrings(parsed.instructions),
    notes: uniqueStrings(parsed.notes)
  };
}

function stripIngredientLead(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/^\d+([\/.]\d+)?\s*/g, "")
    .replace(/^[a-z]+\)\s*/g, "")
    .replace(/\b(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|lbs|pound|pounds|g|kg|ml|l|pinch|dash|can|cans|jar|jars|package|packages|clove|cloves|slice|slices)\b/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function computeRecipeInventoryCoverage(kitchenItems = [], recipeIngredients = []) {
  const kitchen = (Array.isArray(kitchenItems) ? kitchenItems : [])
    .map((item) => ({
      item,
      key: normalizeName(item?.item_name)
    }))
    .filter((entry) => entry.key);

  const available = [];
  const missing = [];

  for (const ingredient of Array.isArray(recipeIngredients) ? recipeIngredients : []) {
    const ingredientText = String(ingredient || "").trim();
    if (!ingredientText) {
      continue;
    }
    const ingredientKey = stripIngredientLead(ingredientText) || normalizeName(ingredientText);
    const matched = kitchen.some(({ key }) => (
      ingredientKey.includes(key)
      || key.includes(ingredientKey)
      || ingredientKey.split(" ").some((token) => token.length >= 4 && key.includes(token))
    ));
    if (matched) {
      available.push(ingredientText);
    } else {
      missing.push(ingredientText);
    }
  }

  return {
    available_ingredients: uniqueStrings(available),
    missing_ingredients: uniqueStrings(missing)
  };
}

function deriveRecipeSourceName(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./i, "");
    return host
      .split(".")
      .filter(Boolean)
      .slice(0, -1)
      .join(" ")
      .replace(/\b\w/g, (character) => character.toUpperCase()) || host;
  } catch {
    return null;
  }
}

function scoreResolvedRecipeCandidate(seedRecipe, resolvedRecipe, searchTitle = "") {
  const titleScore = scoreTokenOverlap(seedRecipe?.title, resolvedRecipe?.title) + scoreTokenOverlap(searchTitle, resolvedRecipe?.title);
  const ingredientScore = intersectNormalizedStrings(seedRecipe?.ingredients || [], resolvedRecipe?.ingredients || []).length * 10;
  const missingScore = intersectNormalizedStrings(seedRecipe?.missing_ingredients || [], resolvedRecipe?.ingredients || []).length * 8;
  const structureBonus = Math.min((resolvedRecipe?.ingredients || []).length, 10) + Math.min((resolvedRecipe?.steps || []).length, 10);
  return titleScore + ingredientScore + missingScore + structureBonus;
}

async function enrichRecipeDetailFromWeb(context, recipe, options = {}) {
  if (!recipe?.title || recipe?.source_url) {
    return recipe;
  }

  const queries = buildRecipeSearchQueries(recipe);
  if (queries.length === 0) {
    return recipe;
  }

  let searchResults = [];
  for (const query of queries) {
    try {
      const results = await fetchRecipeSearchResults(query);
      searchResults.push(...results);
    } catch (error) {
      console.warn("[WARN] recipe search failed:", error?.message || error);
    }
    if (searchResults.length >= 4) {
      break;
    }
  }
  searchResults = searchResults.slice(0, 4);
  if (searchResults.length === 0) {
    return recipe;
  }

  const kitchenItems = await getKitchenItems(context, options).catch(() => []);
  const analyses = await Promise.all(searchResults.map(async (result) => {
    try {
      const payload = await fetchHouseholdApiJson("/analyze-url", {
        ...options,
        method: "POST",
        body: { url: result.url }
      });
      const parsedRecipe = parseRecipeTextSections(payload.recipe || "");
      const normalized = {
        ...recipe,
        title: parsedRecipe.title || payload.title || recipe.title,
        description: payload.title && payload.title !== parsedRecipe.title ? payload.title : null,
        ingredients: parsedRecipe.ingredients.length > 0 ? parsedRecipe.ingredients : recipe.ingredients,
        steps: parsedRecipe.instructions.length > 0 ? parsedRecipe.instructions : recipe.steps,
        notes: uniqueStrings([
          ...(parsedRecipe.notes || []),
          payload.author_name ? `Author: ${payload.author_name}` : null
        ]),
        image_url: payload.image_url || recipe.image_url || null,
        source_url: payload.resolved_url || payload.url || result.url,
        resolved_url: payload.resolved_url || payload.url || result.url,
        source_name: deriveRecipeSourceName(payload.resolved_url || payload.url || result.url),
        author_name: payload.author_name || null,
        extraction_source: payload.source || null,
        bucket: recipe.bucket
      };
      const coverage = computeRecipeInventoryCoverage(kitchenItems, normalized.ingredients);
      normalized.available_ingredients = coverage.available_ingredients;
      normalized.missing_ingredients = coverage.missing_ingredients.length > 0
        ? coverage.missing_ingredients
        : uniqueStrings(recipe.missing_ingredients || []);
      normalized.bucket = normalized.missing_ingredients.length > 0 ? "need_grocery" : recipe.bucket;
      return {
        recipe: normalized,
        score: scoreResolvedRecipeCandidate(recipe, normalized, result.title)
      };
    } catch (error) {
      console.warn("[WARN] recipe analyze failed:", error?.message || error);
      return null;
    }
  }));

  const best = analyses
    .filter(Boolean)
    .sort((left, right) => right.score - left.score)[0];

  return best?.recipe || recipe;
}

export async function searchWebRecipes(context, args, options = {}) {
  const query = String(args?.query || "").trim();
  if (!query) {
    return { ok: true, recipes: [], count: 0, query: "", error_message: "No search query provided." };
  }

  const maxResults = Math.max(1, Math.min(args?.max_results || 3, 5));
  const searchQuery = /recipe/i.test(query) ? query : `${query} recipe`;

  let searchResults;
  try {
    searchResults = await fetchRecipeSearchResults(searchQuery);
  } catch (error) {
    console.warn("[WARN] web recipe search failed:", error?.message || error);
    return { ok: true, recipes: [], count: 0, query, error_message: "Recipe search is temporarily unavailable. Try again in a moment." };
  }

  searchResults = searchResults.slice(0, maxResults);
  if (searchResults.length === 0) {
    return { ok: true, recipes: [], count: 0, query, error_message: "No recipe results found. Try a different search." };
  }

  const [analyses, kitchenItems] = await Promise.all([
    Promise.all(searchResults.map(async (result) => {
      try {
        const payload = await fetchHouseholdApiJson("/analyze-url", {
          ...options,
          method: "POST",
          body: { url: result.url }
        });
        const parsed = parseRecipeTextSections(payload.recipe || "");
        return {
          title: parsed.title || payload.title || result.title || "Untitled Recipe",
          description: payload.title && payload.title !== parsed.title ? payload.title : null,
          ingredients: parsed.ingredients.length > 0 ? parsed.ingredients : [],
          steps: parsed.instructions.length > 0 ? parsed.instructions : [],
          notes: uniqueStrings([
            ...(parsed.notes || []),
            payload.author_name ? `Author: ${payload.author_name}` : null
          ]),
          image_url: payload.image_url || null,
          source_url: payload.resolved_url || payload.url || result.url,
          source_name: deriveRecipeSourceName(payload.resolved_url || payload.url || result.url),
          author_name: payload.author_name || null
        };
      } catch (error) {
        console.warn("[WARN] recipe URL analyze failed:", result.url, error?.message || error);
        return null;
      }
    })),
    getKitchenItems(context, options).catch(() => [])
  ]);

  const recipes = analyses
    .filter((recipe) => recipe && (recipe.ingredients.length > 0 || recipe.steps.length > 0))
    .map((recipe) => {
      const coverage = computeRecipeInventoryCoverage(kitchenItems, recipe.ingredients);
      return {
        ...recipe,
        available_ingredients: coverage.available_ingredients,
        missing_ingredients: coverage.missing_ingredients,
        bucket: coverage.missing_ingredients.length > 0 ? "need_grocery" : "kitchen_only"
      };
    });

  console.log("[DEBUG] web recipe search:", JSON.stringify({
    query,
    searchQuery,
    urlsFound: searchResults.length,
    recipesReturned: recipes.length
  }));

  return {
    ok: true,
    recipes,
    count: recipes.length,
    query,
    ...(recipes.length === 0 ? { error_message: "Found URLs but could not extract recipe details. Try a more specific search." } : {})
  };
}

function placeholders(values) {
  return values.map(() => "?").join(", ");
}

function serializeDate(value) {
  if (!value) {
    return null;
  }

  return new Date(value).toISOString();
}

function serializeDateOnly(value) {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }

  return date.toISOString().slice(0, 10);
}

function weekdayFromName(value) {
  const normalized = normalizeName(value);
  return new Map([
    ["sunday", 0],
    ["monday", 1],
    ["tuesday", 2],
    ["wednesday", 3],
    ["thursday", 4],
    ["friday", 5],
    ["saturday", 6]
  ]).get(normalized) ?? null;
}

function parseRelativeCount(value) {
  const normalized = normalizeName(value);
  if (/^\d+$/.test(normalized)) {
    return Number.parseInt(normalized, 10);
  }

  return new Map([
    ["a", 1],
    ["an", 1],
    ["one", 1],
    ["two", 2],
    ["three", 3],
    ["four", 4],
    ["five", 5],
    ["six", 6],
    ["seven", 7],
    ["eight", 8],
    ["nine", 9],
    ["ten", 10],
    ["couple", 2]
  ]).get(normalized) ?? null;
}

function resolveUpcomingDayOfMonth(dayOfMonth, today) {
  const day = Number(dayOfMonth);
  if (!Number.isInteger(day) || day < 1 || day > 31) {
    return null;
  }

  for (let monthOffset = 0; monthOffset < 24; monthOffset += 1) {
    const candidate = new Date(today.getFullYear(), today.getMonth() + monthOffset, day);
    if (candidate.getDate() !== day) {
      continue;
    }
    candidate.setHours(0, 0, 0, 0);
    if (candidate.getTime() >= today.getTime()) {
      return candidate;
    }
  }

  return null;
}

function parseKitchenExpirationDate(input) {
  const raw = String(input || "").trim();
  if (!raw) {
    return null;
  }

  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return raw;
  }

  const normalized = normalizeName(raw);
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  if (normalized === "today") {
    return serializeDateOnly(today);
  }
  if (normalized === "tomorrow") {
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    return serializeDateOnly(tomorrow);
  }

  const relativeMatch = normalized.match(/^(?:in )?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|\d+)\s+(day|days|week|weeks|month|months)$/);
  if (relativeMatch) {
    const amount = parseRelativeCount(relativeMatch[1]);
    const unit = relativeMatch[2];
    if (amount != null && amount > 0) {
      const candidate = new Date(today);
      if (unit.startsWith("day")) {
        candidate.setDate(candidate.getDate() + amount);
      } else if (unit.startsWith("week")) {
        candidate.setDate(candidate.getDate() + (amount * 7));
      } else if (unit.startsWith("month")) {
        candidate.setMonth(candidate.getMonth() + amount);
      }
      return serializeDateOnly(candidate);
    }
  }

  const ordinalDayMatch = normalized.match(/^(?:on )?(?:the )?(\d{1,2})(?:st|nd|rd|th)$/);
  if (ordinalDayMatch) {
    const candidate = resolveUpcomingDayOfMonth(Number.parseInt(ordinalDayMatch[1], 10), today);
    if (candidate) {
      return serializeDateOnly(candidate);
    }
  }

  const nextWeekdayMatch = normalized.match(/^next (sunday|monday|tuesday|wednesday|thursday|friday|saturday)$/);
  const plainWeekdayMatch = normalized.match(/^(sunday|monday|tuesday|wednesday|thursday|friday|saturday)$/);
  const weekdayValue = weekdayFromName(nextWeekdayMatch?.[1] || plainWeekdayMatch?.[1] || "");

  if (weekdayValue != null) {
    const candidate = new Date(today);
    let delta = (weekdayValue - candidate.getDay() + 7) % 7;
    if (delta === 0 || nextWeekdayMatch) {
      delta += 7;
    }
    candidate.setDate(candidate.getDate() + delta);
    return serializeDateOnly(candidate);
  }

  const parsed = new Date(raw);
  if (!Number.isNaN(parsed.getTime())) {
    return serializeDateOnly(parsed);
  }

  return null;
}

function kitchenTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_prod_kitchen`;
}

function discardTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_discards`;
}

function dishTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_dishes`;
}

function metricsTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}-metrics`;
}

function shoppingListTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_new_list`;
}

const SHARED_KITCHEN_TABLE = 'shared_kitchen';
const SHARED_ARCHIVE_KITCHEN_TABLE = 'shared_archive_kitchen';
const SHARED_DISCARDS_TABLE = 'shared_discards';
const SHARED_DISHES_TABLE = 'shared_dishes';
const SHARED_SHOPPING_TABLE = 'shared_shopping_list';

let _sharedTablesCreated = false;
async function ensureSharedTables(connection) {
  if (_sharedTablesCreated) return;
  const ddl = [
    `CREATE TABLE IF NOT EXISTS \`${SHARED_KITCHEN_TABLE}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY, \`owner_id\` VARCHAR(36) NOT NULL, \`_owner\` VARCHAR(36) NULL,
      \`_device\` VARCHAR(255) NULL, \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL, \`product_name\` VARCHAR(500) NULL, \`brand\` VARCHAR(255) NULL,
      \`variant\` VARCHAR(255) NULL, \`category\` VARCHAR(255) NULL, \`confidence\` DECIMAL(3,2) NULL,
      \`explanation\` TEXT NULL, \`product_description\` VARCHAR(500) NULL, \`barcode\` VARCHAR(100) NULL,
      \`country_guess\` VARCHAR(100) NULL, \`estimated_price\` VARCHAR(50) NULL, \`ingredients\` JSON NULL,
      \`nutrition_summary\` TEXT NULL, \`upf\` ENUM('yes','no') NULL, \`harmful_ingredients\` JSON NULL,
      \`similar_items\` JSON NULL, \`alternatives\` JSON NULL, \`healthier_alternatives\` JSON NULL,
      \`store_availability\` JSON NULL, \`images\` VARCHAR(1000) NULL, \`s3_key\` VARCHAR(500) NULL,
      \`action\` ENUM('IN','OUT') NOT NULL DEFAULT 'IN', \`product_expiration\` DATE NULL,
      \`product_image_url\` VARCHAR(1000) NULL, \`product_image_key\` VARCHAR(500) NULL,
      \`job_id\` VARCHAR(100) NULL, \`user_id\` VARCHAR(255) NULL, \`storage_location\` VARCHAR(255) NULL,
      \`is_opened\` TINYINT(1) NOT NULL DEFAULT 0, \`remaining_quantity\` VARCHAR(255) NULL,
      \`quantity_value\` DECIMAL(10,2) NULL, \`quantity_unit\` VARCHAR(64) NULL,
      \`fill_percent\` TINYINT UNSIGNED NULL, \`storage_guidance\` JSON NULL,
      INDEX idx_owner_id (owner_id), INDEX idx_action (action), INDEX idx_created (_createdDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,
    `CREATE TABLE IF NOT EXISTS \`${SHARED_ARCHIVE_KITCHEN_TABLE}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY, \`owner_id\` VARCHAR(36) NOT NULL, \`_owner\` VARCHAR(36) NULL,
      \`_device\` VARCHAR(255) NULL, \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL, \`product_name\` VARCHAR(500) NULL, \`brand\` VARCHAR(255) NULL,
      \`category\` VARCHAR(255) NULL, \`action\` ENUM('IN','OUT') NULL, \`archived_at\` DATETIME NULL,
      \`archived_reason\` VARCHAR(255) NULL, \`archived_from_table\` VARCHAR(255) NULL,
      INDEX idx_owner_id (owner_id), INDEX idx_created (_createdDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,
    `CREATE TABLE IF NOT EXISTS \`${SHARED_DISCARDS_TABLE}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY, \`owner_id\` VARCHAR(36) NOT NULL, \`_owner\` VARCHAR(36) NULL,
      \`_device\` VARCHAR(255) NULL, \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL, \`product_name\` VARCHAR(500) NULL, \`brand\` VARCHAR(255) NULL,
      \`category\` VARCHAR(255) NULL, \`images\` VARCHAR(1000) NULL,
      \`action\` ENUM('IN','OUT') NOT NULL DEFAULT 'IN', \`product_expiration\` DATE NULL,
      \`job_id\` VARCHAR(100) NULL, \`user_id\` VARCHAR(255) NULL, \`source_kitchen_id\` VARCHAR(36) NULL,
      \`discard_reason\` VARCHAR(255) NULL,
      INDEX idx_owner_id (owner_id), INDEX idx_created (_createdDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,
    `CREATE TABLE IF NOT EXISTS \`${SHARED_DISHES_TABLE}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY, \`owner_id\` VARCHAR(36) NOT NULL, \`_owner\` VARCHAR(36) NULL,
      \`_device\` VARCHAR(255) NULL, \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL, \`dish_name\` VARCHAR(500) NULL, \`confidence\` DECIMAL(3,2) NULL,
      \`explanation\` TEXT NULL, \`serving_size\` VARCHAR(100) NULL, \`calories\` DECIMAL(10,2) NULL,
      \`total_fat\` DECIMAL(10,2) NULL, \`saturated_fat\` DECIMAL(10,2) NULL, \`trans_fat\` DECIMAL(10,2) NULL,
      \`cholesterol\` DECIMAL(10,2) NULL, \`sodium\` DECIMAL(10,2) NULL,
      \`total_carbohydrates\` DECIMAL(10,2) NULL, \`dietary_fiber\` DECIMAL(10,2) NULL,
      \`sugars\` DECIMAL(10,2) NULL, \`protein\` DECIMAL(10,2) NULL, \`vitamin_a\` DECIMAL(10,2) NULL,
      \`vitamin_c\` DECIMAL(10,2) NULL, \`calcium\` DECIMAL(10,2) NULL, \`iron\` DECIMAL(10,2) NULL,
      \`ingredients\` JSON NULL, \`components\` JSON NULL, \`allergens\` JSON NULL,
      \`images\` VARCHAR(1000) NULL, \`s3_key\` VARCHAR(500) NULL,
      \`action\` ENUM('IN','OUT') NOT NULL DEFAULT 'IN', \`dish_image_url\` VARCHAR(1000) NULL,
      \`dish_image_key\` VARCHAR(500) NULL, \`job_id\` VARCHAR(100) NULL, \`user_id\` VARCHAR(255) NULL,
      \`analysis_status\` VARCHAR(32) NULL, \`analysis_error\` TEXT NULL,
      INDEX idx_owner_id (owner_id), INDEX idx_action (action), INDEX idx_created (_createdDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,
    `CREATE TABLE IF NOT EXISTS \`${SHARED_SHOPPING_TABLE}\` (
      \`_id\` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, \`owner_id\` VARCHAR(36) NOT NULL,
      \`_owner\` CHAR(36) NOT NULL, \`_device\` VARCHAR(64) NOT NULL DEFAULT 'voice',
      \`product_name\` VARCHAR(255) NOT NULL, \`quantity\` VARCHAR(100) DEFAULT NULL,
      \`product_brand\` VARCHAR(255) DEFAULT NULL, \`images\` TEXT, \`product_barcode\` VARCHAR(64) DEFAULT NULL,
      \`store\` VARCHAR(100) DEFAULT NULL, \`action\` VARCHAR(32) NOT NULL DEFAULT '1',
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`created_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`updated_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      \`household_item_uuid\` CHAR(36) DEFAULT NULL,
      PRIMARY KEY (_id), INDEX idx_owner_id (owner_id), KEY idx_created (_createdDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`
  ];
  for (const stmt of ddl) {
    try { await connection.execute(stmt); } catch (e) { console.warn('[WARN] shared table DDL failed (may already exist):', e?.message); }
  }
  _sharedTablesCreated = true;
}

function resolveTableOwnerId(context) {
  return context.tableOwnerId || context.userId || context.ownerId;
}

function resolveShoppingOwnerId(context) {
  return context.userId || context.tableOwnerId || context.ownerId;
}

function resolveApiOwnerId(context) {
  return context.ownerId || context.userId || context.tableOwnerId;
}

function normalizeHouseholdMemberIds(values = []) {
  return Array.from(
    new Set(
      (values || [])
        .map((value) => String(value || "").trim())
        .filter(Boolean)
    )
  );
}

function getTableHouseholdMemberIds(context) {
  return normalizeHouseholdMemberIds([
    resolveTableOwnerId(context),
    context?.userId,
    ...(context?.householdMemberIds || [])
  ]);
}

function logHouseholdWriteTargets(label, context, memberIds, details = {}) {
  console.log(`[DEBUG] ${label}:`, JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    tableOwnerId: context?.tableOwnerId || null,
    householdMemberIds: memberIds,
    householdSize: memberIds.length,
    ...details
  }));
}

async function triggerKitchenDependentGeneration(context, options = {}) {
  const env = options?.env || process.env;
  const ownerIds = getTableHouseholdMemberIds(context);
  if (ownerIds.length === 0) {
    return;
  }

  const generatorTargets = [
    {
      label: "kitchen-analysis",
      functionName: String(env.KITCHEN_ANALYSIS_GENERATOR_ARN || "").trim()
    },
    {
      label: "meal-plan",
      functionName: String(env.MEAL_PLAN_GENERATOR_ARN || "").trim()
    },
    {
      label: "recipes",
      functionName: String(env.RECIPES_GENERATOR_ARN || "").trim()
    }
  ].filter((target) => target.functionName);

  if (generatorTargets.length === 0) {
    return;
  }

  logHouseholdWriteTargets("kitchen dependent generation dispatch", context, ownerIds, {
    mode: "delegated",
    delegate: "lambda_generators",
    generatorTargets: generatorTargets.map((target) => target.label)
  });

  await Promise.all(generatorTargets.flatMap(({ label, functionName }) => ownerIds.map(async (ownerId) => {
    try {
      await getLambdaClient(env).send(new InvokeCommand({
        FunctionName: functionName,
        InvocationType: "Event",
        Payload: Buffer.from(JSON.stringify({ owner: ownerId }))
      }));
    } catch (error) {
      console.warn(`[${label}] generator invoke failed for ${ownerId}:`, error?.message || error);
    }
  })));
}

function getHouseholdApiBaseUrl(options = {}) {
  return String(options?.env?.HOUSEHOLD_API_BASE_URL || process.env.HOUSEHOLD_API_BASE_URL || DEFAULT_HOUSEHOLD_API_BASE_URL)
    .trim()
    .replace(/\/+$/, "");
}

async function fetchHouseholdApiJson(path, options = {}) {
  const baseUrl = getHouseholdApiBaseUrl(options);
  const response = await fetch(`${baseUrl}${path}`, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {})
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const error = new Error(payload?.error || payload?.message || `Request failed with ${response.status}.`);
    error.statusCode = response.status;
    error.details = payload || text || null;
    throw error;
  }

  return payload || {};
}

function normalizeRecipeBucket(value) {
  const normalized = normalizeName(value);
  if (!normalized) {
    return null;
  }
  if (["kitchen_only", "kitchen only", "make now", "ready now", "available now"].includes(normalized)) {
    return "kitchen_only";
  }
  if (["need_grocery", "need grocery", "need groceries", "buy", "buy next"].includes(normalized)) {
    return "need_grocery";
  }
  return normalized;
}

function normalizeRecipeItem(recipe, bucket) {
  if (!recipe || typeof recipe !== "object") {
    return null;
  }
  return {
    title: String(recipe.title || "").trim() || "Untitled recipe",
    description: recipe.description || null,
    ingredients: Array.isArray(recipe.ingredients) ? recipe.ingredients.filter(Boolean) : [],
    steps: Array.isArray(recipe.steps) ? recipe.steps.filter(Boolean) : [],
    notes: Array.isArray(recipe.notes) ? recipe.notes.filter(Boolean) : [],
    image_url: recipe.image_url || null,
    available_ingredients: Array.isArray(recipe.available_ingredients) ? recipe.available_ingredients.filter(Boolean) : [],
    missing_ingredients: Array.isArray(recipe.missing_ingredients) ? recipe.missing_ingredients.filter(Boolean) : [],
    source_url: recipe.source_url || null,
    resolved_url: recipe.resolved_url || null,
    source_name: recipe.source_name || null,
    author_name: recipe.author_name || null,
    extraction_source: recipe.extraction_source || null,
    bucket
  };
}

function normalizeRecipePayload(payload, limit = null) {
  const kitchenOnly = (Array.isArray(payload?.kitchen_only) ? payload.kitchen_only : [])
    .map((recipe) => normalizeRecipeItem(recipe, "kitchen_only"))
    .filter(Boolean);
  const needGrocery = (Array.isArray(payload?.need_grocery) ? payload.need_grocery : [])
    .map((recipe) => normalizeRecipeItem(recipe, "need_grocery"))
    .filter(Boolean);
  const safeLimit = clampLimit(limit, 10, 20);

  return {
    owner: payload?.owner || null,
    status: payload?.status || "empty",
    kitchen_only: limit != null ? kitchenOnly.slice(0, safeLimit) : kitchenOnly,
    need_grocery: limit != null ? needGrocery.slice(0, safeLimit) : needGrocery,
    error_message: payload?.error_message || null,
    _createdDate: payload?._createdDate || null,
    _updatedDate: payload?._updatedDate || null
  };
}

function flattenRecipeSuggestions(payload) {
  return [
    ...(payload?.kitchen_only || []),
    ...(payload?.need_grocery || [])
  ];
}

function resolveRecipeReference(payload, reference = {}) {
  const bucket = normalizeRecipeBucket(reference.recipe_bucket);
  const candidates = flattenRecipeSuggestions(payload)
    .filter((recipe) => !bucket || recipe.bucket === bucket);
  const title = String(reference.recipe_title || "").trim();

  if (!title) {
    if (candidates.length === 1) {
      return candidates[0];
    }
    const error = new Error("Which recipe did you mean?");
    error.statusCode = 409;
    throw error;
  }

  const normalizedTitle = normalizeName(title);
  const scored = candidates
    .map((recipe) => {
      const recipeTitle = normalizeName(recipe.title);
      let score = 0;
      if (recipeTitle === normalizedTitle) {
        score = 400;
      } else if (recipeTitle.startsWith(normalizedTitle) || normalizedTitle.startsWith(recipeTitle)) {
        score = 300;
      } else if (recipeTitle.includes(normalizedTitle) || normalizedTitle.includes(recipeTitle)) {
        score = 200;
      }
      return { recipe, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.recipe.title.localeCompare(right.recipe.title));

  if (scored.length === 0) {
    const error = new Error(`Couldn't find the recipe ${title}.`);
    error.statusCode = 404;
    throw error;
  }

  if (scored.length > 1 && scored[0].score === scored[1].score) {
    const error = new Error(`There are multiple recipes that sound like ${title}.`);
    error.statusCode = 409;
    throw error;
  }

  return scored[0].recipe;
}

function normalizeSavedRecipeItem(recipe) {
  if (!recipe || typeof recipe !== "object") {
    return null;
  }
  return {
    id: String(recipe.id || recipe._id || "").trim() || null,
    title: String(recipe.title || "").trim() || "Untitled recipe",
    source_url: recipe.source_url || null,
    resolved_url: recipe.resolved_url || null,
    image_url: recipe.image_url || null,
    image_urls: Array.isArray(recipe.image_urls) ? recipe.image_urls.filter(Boolean) : [],
    author_name: recipe.author_name || null,
    ingredients: Array.isArray(recipe.ingredients) ? recipe.ingredients.filter(Boolean) : [],
    instructions: Array.isArray(recipe.instructions) ? recipe.instructions.filter(Boolean) : [],
    steps: Array.isArray(recipe.instructions) ? recipe.instructions.filter(Boolean) : [],
    notes: Array.isArray(recipe.notes) ? recipe.notes.filter(Boolean) : [],
    content: recipe.content || recipe.caption || null,
    caption: recipe.caption || recipe.content || null,
    source_type: recipe.source_type || recipe.platform || "tiktok",
    platform: recipe.platform || recipe.source_type || "tiktok",
    extraction_source: recipe.extraction_source || null,
    caption_field: recipe.caption_field || null,
    status: recipe.status || "ready",
    _createdDate: recipe._createdDate || null,
    _updatedDate: recipe._updatedDate || null
  };
}

function normalizeSavedRecipesPayload(payload, limit = null) {
  const recipes = (Array.isArray(payload?.recipes) ? payload.recipes : [])
    .map(normalizeSavedRecipeItem)
    .filter(Boolean);
  const safeLimit = clampLimit(limit, 10, 50);
  return {
    owner: payload?.owner || null,
    recipes: limit != null ? recipes.slice(0, safeLimit) : recipes,
    count: Number(payload?.count || recipes.length || 0)
  };
}

function resolveSavedRecipeReference(payload, reference = {}) {
  const title = String(reference.recipe_title || "").trim();
  const candidates = Array.isArray(payload?.recipes) ? payload.recipes : [];

  if (!title) {
    if (candidates.length === 1) {
      return candidates[0];
    }
    const error = new Error("Which saved recipe did you mean?");
    error.statusCode = 409;
    throw error;
  }

  const normalizedTitle = normalizeName(title);
  const scored = candidates
    .map((recipe) => {
      const recipeTitle = normalizeName(recipe.title);
      let score = 0;
      if (recipeTitle === normalizedTitle) {
        score = 400;
      } else if (recipeTitle.startsWith(normalizedTitle) || normalizedTitle.startsWith(recipeTitle)) {
        score = 300;
      } else if (recipeTitle.includes(normalizedTitle) || normalizedTitle.includes(recipeTitle)) {
        score = 200;
      }
      return { recipe, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.recipe.title.localeCompare(right.recipe.title));

  if (scored.length === 0) {
    const error = new Error(`Couldn't find the saved recipe ${title}.`);
    error.statusCode = 404;
    throw error;
  }

  if (scored.length > 1 && scored[0].score === scored[1].score) {
    const error = new Error(`There are multiple saved recipes that sound like ${title}.`);
    error.statusCode = 409;
    throw error;
  }

  return scored[0].recipe;
}

function normalizeMealPlanSlots(plan) {
  const rawPlan = typeof plan === "string" ? parseJsonColumn(plan, []) : (Array.isArray(plan) ? plan : []);
  return rawPlan
    .map((slot) => ({
      day_index: Number.isFinite(Number(slot?.day_index)) ? Number(slot.day_index) : null,
      day_label: slot?.day_label || null,
      meal_type: slot?.meal_type || null,
      recipe: slot?.recipe && typeof slot.recipe === "object"
        ? {
            title: slot.recipe.title || "Open slot",
            ingredients: Array.isArray(slot.recipe.ingredients) ? slot.recipe.ingredients.filter(Boolean) : [],
            steps: Array.isArray(slot.recipe.steps) ? slot.recipe.steps.filter(Boolean) : [],
            image_url: slot.recipe.image_url || null
          }
        : null
    }))
    .filter((slot) => slot.day_label || slot.meal_type || slot.recipe);
}

const NUMBER_WORDS = new Map([
  ["zero", 0],
  ["one", 1],
  ["two", 2],
  ["three", 3],
  ["four", 4],
  ["five", 5],
  ["six", 6],
  ["seven", 7],
  ["eight", 8],
  ["nine", 9],
  ["ten", 10],
  ["half", 0.5],
  ["quarter", 0.25]
]);

function parseWordNumber(value) {
  const normalized = normalizeName(value);
  return NUMBER_WORDS.has(normalized) ? NUMBER_WORDS.get(normalized) : null;
}

function normalizeUnit(value) {
  const normalized = normalizeName(value);
  if (!normalized) {
    return null;
  }

  const aliases = new Map([
    ["serving", "serving"],
    ["servings", "serving"],
    ["piece", "piece"],
    ["pieces", "piece"],
    ["item", "item"],
    ["items", "item"],
    ["bottle", "bottle"],
    ["bottles", "bottle"],
    ["can", "can"],
    ["cans", "can"],
    ["carton", "carton"],
    ["cartons", "carton"],
    ["box", "box"],
    ["boxes", "box"],
    ["bag", "bag"],
    ["bags", "bag"],
    ["pack", "pack"],
    ["packs", "pack"],
    ["packet", "packet"],
    ["packets", "packet"],
    ["pouch", "pouch"],
    ["pouches", "pouch"],
    ["container", "container"],
    ["containers", "container"],
    ["jar", "jar"],
    ["jars", "jar"],
    ["tub", "tub"],
    ["tubs", "tub"],
    ["loaf", "loaf"],
    ["loaves", "loaf"],
    ["roll", "roll"],
    ["rolls", "roll"],
    ["slice", "slice"],
    ["slices", "slice"],
    ["cup", "cup"],
    ["cups", "cup"],
    ["ounce", "oz"],
    ["ounces", "oz"],
    ["oz", "oz"],
    ["pound", "lb"],
    ["pounds", "lb"],
    ["lb", "lb"],
    ["lbs", "lb"]
  ]);

  return aliases.get(normalized) || normalized;
}

function roundQuantity(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }

  return Math.round(numeric * 100) / 100;
}

function clampPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }

  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function parseStructuredState(input = {}) {
  const rawText = String(
    input.remaining_quantity
      || input.quantity
      || input.quantity_description
      || ""
  ).trim();

  const state = {
    remaining_quantity: rawText || null,
    quantity_value: roundQuantity(input.quantity_value),
    quantity_unit: normalizeUnit(input.quantity_unit),
    fill_percent: clampPercent(input.fill_percent)
  };

  if (!rawText) {
    return state;
  }

  const normalized = normalizeName(rawText);

  if (state.fill_percent == null) {
    const percentMatch = normalized.match(/(\d{1,3})\s*percent|\b(\d{1,3})\s*%\b/);
    if (percentMatch) {
      state.fill_percent = clampPercent(percentMatch[1] || percentMatch[2]);
    } else if (/\babout half\b|\baround half\b|\bhalfway\b/.test(normalized)) {
      state.fill_percent = 50;
    } else if (/\bthree quarters?\b|\b3\/4\b/.test(normalized)) {
      state.fill_percent = 75;
    } else if (/\btwo thirds?\b|\b2\/3\b/.test(normalized)) {
      state.fill_percent = 67;
    } else if (/\bone third\b|\b1\/3\b/.test(normalized)) {
      state.fill_percent = 33;
    } else if (/\bhalf\b|\b1\/2\b/.test(normalized)) {
      state.fill_percent = 50;
    } else if (/\bquarter\b|\b1\/4\b/.test(normalized)) {
      state.fill_percent = 25;
    } else if (/\bmostly empty\b/.test(normalized)) {
      state.fill_percent = 25;
    } else if (/\balmost full\b|\bnearly full\b/.test(normalized)) {
      state.fill_percent = 90;
    } else if (/\bmostly full\b/.test(normalized)) {
      state.fill_percent = 75;
    } else if (/\brunning low\b|\blow\b|\bnot much left\b|\ba little left\b/.test(normalized)) {
      state.fill_percent = 15;
    } else if (/\balmost empty\b|\bnearly empty\b/.test(normalized)) {
      state.fill_percent = 10;
    } else if (/\bnearly gone\b|\balmost gone\b/.test(normalized)) {
      state.fill_percent = 5;
    } else if (/\bempty\b|\bused up\b|\bfinished\b/.test(normalized)) {
      state.fill_percent = 0;
    } else if (/\bfull\b/.test(normalized)) {
      state.fill_percent = 100;
    }
  }

  if (state.quantity_value == null) {
    const numericMatch = normalized.match(/(\d+(?:\.\d+)?)\s*(servings?|pieces?|items?|bottles?|cans?|cartons?|boxes?|bags?|packs?|packets?|pouches?|containers?|jars?|tubs?|loaves?|rolls?|slices?|cups?|ounces?|oz|pounds?|lbs?|lb)\b/);
    if (numericMatch) {
      state.quantity_value = roundQuantity(numericMatch[1]);
      state.quantity_unit = state.quantity_unit || normalizeUnit(numericMatch[2]);
    } else {
      const wordMatch = normalized.match(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(servings?|pieces?|items?|bottles?|cans?|cartons?|boxes?|bags?|packs?|packets?|pouches?|containers?|jars?|tubs?|loaves?|rolls?|slices?|cups?)\b/);
      if (wordMatch) {
        state.quantity_value = roundQuantity(parseWordNumber(wordMatch[1]));
        state.quantity_unit = state.quantity_unit || normalizeUnit(wordMatch[2]);
      } else if (/\b(last one|only one|just one)\b/.test(normalized)) {
        state.quantity_value = 1;
        state.quantity_unit = state.quantity_unit || "item";
      } else if (/\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(left|remaining|remain)\b/.test(normalized)) {
        const leftMatch = normalized.match(/\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(left|remaining|remain)\b/);
        state.quantity_value = roundQuantity(parseWordNumber(leftMatch?.[1]));
        state.quantity_unit = state.quantity_unit || "item";
      } else if (/\b(?:only|just)?\s*(\d+(?:\.\d+)?)\s+(left|remaining|remain)\b/.test(normalized)) {
        const leftMatch = normalized.match(/\b(?:only|just)?\s*(\d+(?:\.\d+)?)\s+(left|remaining|remain)\b/);
        state.quantity_value = roundQuantity(leftMatch?.[1]);
        state.quantity_unit = state.quantity_unit || "item";
      } else if (/\bdown to\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b/.test(normalized)) {
        const downToMatch = normalized.match(/\bdown to\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b/);
        state.quantity_value = roundQuantity(parseWordNumber(downToMatch?.[1]));
        state.quantity_unit = state.quantity_unit || "item";
      } else if (/\bdown to\s+(\d+(?:\.\d+)?)\b/.test(normalized)) {
        const downToMatch = normalized.match(/\bdown to\s+(\d+(?:\.\d+)?)\b/);
        state.quantity_value = roundQuantity(downToMatch?.[1]);
        state.quantity_unit = state.quantity_unit || "item";
      }
    }
  }

  return state;
}

function formatStructuredState(item) {
  const parts = [];

  if (item.remaining_quantity) {
    parts.push(item.remaining_quantity);
  }
  if (item.fill_percent != null) {
    parts.push(`${item.fill_percent}%`);
  }
  if (item.quantity_value != null) {
    parts.push(`${item.quantity_value}${item.quantity_unit ? ` ${item.quantity_unit}` : ""}`);
  }

  return Array.from(new Set(parts)).filter(Boolean).join(" • ");
}

function mapKitchenRow(row) {
  return {
    id: row._id,
    item_name: row.product_name || row.item_name || "Unknown item",
    brand: row.brand || null,
    variant: row.variant || null,
    category: row.category || null,
    confidence: row.confidence != null ? Number(row.confidence) : null,
    explanation: row.explanation || null,
    description: row.product_description || null,
    barcode: row.barcode || null,
    country_guess: row.country_guess || null,
    estimated_price: row.estimated_price || null,
    ingredients: parseJsonColumn(row.ingredients, []),
    nutrition_summary: row.nutrition_summary || null,
    upf: row.upf || null,
    harmful_ingredients: parseJsonColumn(row.harmful_ingredients, []),
    similar_items: parseJsonColumn(row.similar_items, []),
    alternatives: parseJsonColumn(row.alternatives, []),
    healthier_alternatives: parseJsonColumn(row.healthier_alternatives, []),
    storage_guidance: parseJsonColumn(row.storage_guidance, null),
    store_availability: parseJsonColumn(row.store_availability, null),
    image_url: row.product_image_url || row.images || null,
    product_image_url: row.product_image_url || null,
    product_image_key: row.product_image_key || null,
    images: row.images || null,
    s3_key: row.s3_key || null,
    expiration_date: row.product_expiration || null,
    remaining_quantity: row.remaining_quantity || null,
    quantity_value: row.quantity_value != null ? Number(row.quantity_value) : null,
    quantity_unit: row.quantity_unit || null,
    fill_percent: row.fill_percent != null ? Number(row.fill_percent) : null,
    state_label: formatStructuredState({
      remaining_quantity: row.remaining_quantity || null,
      quantity_value: row.quantity_value != null ? Number(row.quantity_value) : null,
      quantity_unit: row.quantity_unit || null,
      fill_percent: row.fill_percent != null ? Number(row.fill_percent) : null
    }) || null,
    is_opened: Boolean(row.is_opened),
    storage_location: row.storage_location || null,
    user_id: row.user_id || null,
    job_id: row.job_id || null,
    created_at: serializeDate(row._createdDate),
    updated_at: serializeDate(row._updatedDate),
    action: row.action || "IN"
  };
}

function mapDiscardRow(row) {
  return {
    id: row._id,
    item_name: row.product_name || "Unknown item",
    brand: row.brand || null,
    category: row.category || null,
    discard_reason: row.discard_reason || null,
    created_at: serializeDate(row._createdDate),
    action: row.action || "IN"
  };
}

function mapDishRow(row) {
  return {
    id: row._id,
    dish_name: row.dish_name || "Unknown dish",
    serving_size: row.serving_size || null,
    calories: row.calories != null ? Number(row.calories) : null,
    total_fat: row.total_fat != null ? Number(row.total_fat) : null,
    total_carbohydrates: row.total_carbohydrates != null ? Number(row.total_carbohydrates) : null,
    protein: row.protein != null ? Number(row.protein) : null,
    confidence: row.confidence != null ? Number(row.confidence) : null,
    explanation: row.explanation || null,
    ingredients: parseJsonColumn(row.ingredients, []),
    components: parseJsonColumn(row.components, []),
    allergens: parseJsonColumn(row.allergens, []),
    image_url: row.images || null,
    dish_image_url: row.dish_image_url || null,
    created_at: serializeDate(row._createdDate),
    updated_at: serializeDate(row._updatedDate || row._createdDate),
    user_id: row.user_id || null,
    action: row.action || "IN",
    analysis_status: row.analysis_status || "complete",
    analysis_error: row.analysis_error || null
  };
}

function normalizeIngredientName(value) {
  return normalizeName(value)
    .replace(/\b(and|with|some|a|an|the|of)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeIngredientList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set();
  const items = [];

  for (const entry of value) {
    const trimmed = String(entry || "").trim();
    if (!trimmed) {
      continue;
    }
    const key = normalizeIngredientName(trimmed);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push(trimmed);
  }

  return items;
}

function normalizeDishComponents(value, fallback = {}) {
  const seen = new Set();
  const normalized = [];
  const entries = Array.isArray(value) ? value : [];

  for (const entry of entries) {
    const name = String(entry?.name || "").trim();
    const ingredients = normalizeIngredientList(entry?.ingredients);
    const resolvedName = toTitleCaseWords(name || ingredients[0] || null);
    if (!resolvedName) {
      continue;
    }
    const key = `${normalizeIngredientName(resolvedName)}::${ingredients.map(normalizeIngredientName).join("|")}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push({
      name: resolvedName,
      ingredients
    });
  }

  if (normalized.length > 0) {
    return normalized;
  }

  const fallbackIngredients = normalizeIngredientList(fallback.ingredients);
  const fallbackName = toTitleCaseWords(String(fallback.dish_name || "").trim())
    || inferDishNameFromIngredients(fallbackIngredients)
    || toTitleCaseWords(fallbackIngredients[0])
    || null;
  if (!fallbackName) {
    return [];
  }

  return [{
    name: fallbackName,
    ingredients: fallbackIngredients
  }];
}

function inferDishNameFromIngredients(ingredients) {
  const normalized = normalizeIngredientList(ingredients);
  if (normalized.length === 0) {
    return null;
  }
  if (normalized.length === 1) {
    return toTitleCaseWords(normalized[0]);
  }
  if (normalized.length === 2) {
    return toTitleCaseWords(`${normalized[0]} with ${normalized[1]}`);
  }
  return toTitleCaseWords(`${normalized[0]} with ${normalized.slice(1, -1).join(", ")} and ${normalized[normalized.length - 1]}`);
}

function inferDishNameFromComponents(components) {
  const names = normalizeIngredientList((components || []).map((component) => component?.name).filter(Boolean));
  if (names.length === 0) {
    return null;
  }
  if (names.length === 1) {
    return toTitleCaseWords(names[0]);
  }
  if (names.length === 2) {
    return toTitleCaseWords(`${names[0]} with ${names[1]}`);
  }
  return toTitleCaseWords(`${names[0]} with ${names.slice(1, -1).join(", ")} and ${names[names.length - 1]}`);
}

function buildDishExplanationFromComponents(components, servingSize = null) {
  const names = normalizeIngredientList((components || []).map((component) => component?.name).filter(Boolean));
  if (names.length === 0) {
    return null;
  }
  const base = names.length === 1
    ? names[0]
    : `Meal components: ${names.join(", ")}`;
  return servingSize ? `${base}. Logged as ${servingSize}.` : `${base}.`;
}

function buildFallbackDishComponents(existingRow, ingredients, analysisRequest = {}) {
  const normalizedIngredients = normalizeIngredientList(ingredients);
  const existingComponents = existingRow
    ? normalizeDishComponents(parseJsonColumn(existingRow.components, []), {
      dish_name: existingRow.dish_name,
      ingredients: parseJsonColumn(existingRow.ingredients, [])
    })
    : [];

  if (existingComponents.length === 0) {
    return normalizeDishComponents([], {
      dish_name: analysisRequest?.dish_name || existingRow?.dish_name || null,
      ingredients: normalizedIngredients
    });
  }

  const covered = new Set();
  for (const component of existingComponents) {
    covered.add(normalizeIngredientName(component.name));
    for (const ingredient of component.ingredients || []) {
      covered.add(normalizeIngredientName(ingredient));
    }
  }

  const additions = normalizedIngredients
    .filter((ingredient) => !covered.has(normalizeIngredientName(ingredient)))
    .map((ingredient) => ({
      name: ingredient,
      ingredients: [ingredient]
    }));

  return normalizeDishComponents(
    [...existingComponents, ...additions],
    {
      dish_name: analysisRequest?.dish_name || existingRow?.dish_name || null,
      ingredients: normalizedIngredients
    }
  );
}

function mergeDishIngredientLists(existingIngredients, addIngredients = [], removeIngredients = []) {
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

function deriveIngredientsFromDishName(dishName) {
  const normalized = String(dishName || "")
    .replace(/\b(i|just|ate|had|some|a|an|the)\b/gi, " ")
    .replace(/\b\d+(\.\d+)?\b/g, " ")
    .trim();
  if (!normalized) {
    return [];
  }

  const roughParts = normalized
    .split(/\bwith\b|,|&|\band\b/gi)
    .map((part) => part.trim())
    .filter(Boolean);

  return normalizeIngredientList(roughParts.length > 0 ? roughParts : [normalized]);
}

function resolveDishIngredientState(existingIngredients, updates = {}) {
  if (Array.isArray(updates.ingredients) && updates.ingredients.length > 0) {
    return normalizeIngredientList(updates.ingredients);
  }

  return mergeDishIngredientLists(
    existingIngredients,
    updates.add_ingredients || [],
    updates.remove_ingredients || []
  );
}

function coerceDishNumber(value) {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function buildDishAnalysisRequest(existingRow = null, updates = {}) {
  const existingIngredients = existingRow ? parseJsonColumn(existingRow.ingredients, []) : [];
  const existingComponents = existingRow
    ? normalizeDishComponents(parseJsonColumn(existingRow.components, []), {
      dish_name: existingRow.dish_name,
      ingredients: existingIngredients
    })
    : [];
  const nextIngredients = resolveDishIngredientState(existingIngredients, updates);
  const requestedDishName = typeof updates.dish_name === "string" && updates.dish_name.trim()
    ? updates.dish_name.trim()
    : null;
  const fallbackDishName = requestedDishName
    || inferDishNameFromIngredients(nextIngredients)
    || existingRow?.dish_name
    || "Voice meal";
  const servingSize = updates.serving_size !== undefined
    ? (updates.serving_size || null)
    : existingRow?.serving_size || null;
  const ingredients = nextIngredients.length > 0
    ? nextIngredients
    : deriveIngredientsFromDishName(fallbackDishName);

  return {
    dish_name: fallbackDishName,
    serving_size: servingSize,
    ingredients,
    existing_context: existingRow ? {
      dish_name: existingRow.dish_name || null,
      serving_size: existingRow.serving_size || null,
      explanation: existingRow.explanation || null,
      ingredients: existingIngredients,
      components: existingComponents,
      allergens: parseJsonColumn(existingRow.allergens, [])
    } : null
  };
}

function hasDishDerivedRefresh(updates = {}, existingRow = null) {
  if (!existingRow) {
    return true;
  }

  return Boolean(
    updates.dish_name !== undefined
    || updates.serving_size !== undefined
    || (Array.isArray(updates.ingredients) && updates.ingredients.length > 0)
    || (Array.isArray(updates.add_ingredients) && updates.add_ingredients.length > 0)
    || (Array.isArray(updates.remove_ingredients) && updates.remove_ingredients.length > 0)
  );
}

function mergeDishAnalysisResult(analysisRequest, analysisResult, updates = {}, existingRow = null) {
  const existingIngredients = existingRow ? parseJsonColumn(existingRow.ingredients, []) : [];
  const requestedIngredients = normalizeIngredientList(analysisRequest.ingredients);
  const resolvedIngredients = normalizeIngredientList(
    analysisResult.ingredients?.length ? analysisResult.ingredients : requestedIngredients
  );
  const resolvedServingSize = updates.serving_size !== undefined
    ? (updates.serving_size || null)
    : analysisResult.serving_size || analysisRequest.serving_size || existingRow?.serving_size || null;
  const analyzerComponents = normalizeDishComponents(analysisResult.components);
  const components = analyzerComponents.length > 0
    ? normalizeDishComponents(analyzerComponents, {
      dish_name: analysisResult.dish_name || analysisRequest.dish_name || existingRow?.dish_name || null,
      ingredients: resolvedIngredients
    })
    : buildFallbackDishComponents(existingRow, resolvedIngredients, analysisRequest);
  const canonicalDishName = inferDishNameFromComponents(components)
    || analysisResult.dish_name
    || analysisRequest.dish_name
    || inferDishNameFromIngredients(resolvedIngredients)
    || existingRow?.dish_name
    || "Voice meal";

  return {
    dish_name: canonicalDishName,
    serving_size: resolvedServingSize,
    calories: updates.calories !== undefined
      ? coerceDishNumber(updates.calories)
      : (analysisResult.calories ?? coerceDishNumber(existingRow?.calories)),
    total_fat: updates.fat_g !== undefined
      ? coerceDishNumber(updates.fat_g)
      : (analysisResult.total_fat ?? coerceDishNumber(existingRow?.total_fat)),
    total_carbohydrates: updates.carbs_g !== undefined
      ? coerceDishNumber(updates.carbs_g)
      : (analysisResult.total_carbohydrates ?? coerceDishNumber(existingRow?.total_carbohydrates)),
    protein: updates.protein_g !== undefined
      ? coerceDishNumber(updates.protein_g)
      : (analysisResult.protein ?? coerceDishNumber(existingRow?.protein)),
    confidence: coerceDishNumber(analysisResult.confidence) ?? coerceDishNumber(existingRow?.confidence),
    explanation: buildDishExplanationFromComponents(components, resolvedServingSize)
      || analysisResult.explanation
      || canonicalDishName
      || existingRow?.explanation
      || "Voice meal",
    ingredients: resolvedIngredients.length > 0
      ? resolvedIngredients
      : normalizeIngredientList(existingIngredients),
    components,
    allergens: normalizeIngredientList(analysisResult.allergens)
  };
}

async function analyzeDishPayload(existingRow, updates = {}, options = {}) {
  const analysisRequest = buildDishAnalysisRequest(existingRow, updates);
  const analysisResult = await analyzeDishFromText(analysisRequest, options?.env || process.env);
  const merged = mergeDishAnalysisResult(analysisRequest, analysisResult, updates, existingRow);

  console.log("[DEBUG] dish analysis result:", JSON.stringify({
    ownerId: options?.context?.ownerId || null,
    userId: options?.context?.userId || null,
    requested_dish_name: analysisRequest.dish_name,
    resolved_dish_name: merged.dish_name,
    ingredient_count: merged.ingredients.length,
    confidence: merged.confidence,
    source_type: analysisResult?.source_type || null,
    evidence_count: Array.isArray(analysisResult?.evidence_urls) ? analysisResult.evidence_urls.length : 0
  }));

  return merged;
}

function buildDishPersistedFieldsFromAnalysis(analysis, existingRow = null) {
  const ingredients = normalizeIngredientList(analysis.ingredients);
  const components = normalizeDishComponents(analysis.components, {
    dish_name: analysis.dish_name || existingRow?.dish_name || null,
    ingredients
  });
  const allergens = normalizeIngredientList(analysis.allergens);
  const canonicalDishName = inferDishNameFromComponents(components)
    || analysis.dish_name
    || existingRow?.dish_name
    || inferDishNameFromIngredients(ingredients)
    || "Voice meal";
  const canonicalExplanation = buildDishExplanationFromComponents(
    components,
    analysis.serving_size ?? existingRow?.serving_size ?? null
  ) || analysis.explanation || existingRow?.explanation || canonicalDishName;

  return {
    dish_name: canonicalDishName,
    confidence: analysis.confidence ?? coerceDishNumber(existingRow?.confidence),
    explanation: canonicalExplanation,
    serving_size: analysis.serving_size ?? existingRow?.serving_size ?? null,
    calories: analysis.calories ?? coerceDishNumber(existingRow?.calories),
    total_fat: analysis.total_fat ?? coerceDishNumber(existingRow?.total_fat),
    total_carbohydrates: analysis.total_carbohydrates ?? coerceDishNumber(existingRow?.total_carbohydrates),
    protein: analysis.protein ?? coerceDishNumber(existingRow?.protein),
    ingredients: JSON.stringify(ingredients),
    components: JSON.stringify(components),
    allergens: JSON.stringify(allergens),
    analysis_status: "complete",
    analysis_error: null
  };
}

function buildPendingDishFields(existingRow, updates = {}) {
  const analysisRequest = buildDishAnalysisRequest(existingRow, updates);
  const ingredients = normalizeIngredientList(analysisRequest.ingredients);
  const components = buildFallbackDishComponents(existingRow, ingredients, analysisRequest);
  const shouldRefresh = hasDishDerivedRefresh(updates, existingRow);
  const servingSize = analysisRequest.serving_size ?? existingRow?.serving_size ?? null;
  const dishName = inferDishNameFromComponents(components)
    || analysisRequest.dish_name
    || existingRow?.dish_name
    || inferDishNameFromIngredients(ingredients)
    || "Voice meal";

  return {
    persisted_fields: {
      dish_name: dishName,
      confidence: shouldRefresh ? null : coerceDishNumber(existingRow?.confidence),
      explanation: buildDishExplanationFromComponents(components, servingSize)
        || (existingRow
          ? "Dish updated. Nutrition is being recalculated in the background."
          : "Dish logged. Nutrition is being calculated in the background."),
      serving_size: servingSize,
      calories: updates.calories !== undefined
        ? coerceDishNumber(updates.calories)
        : (shouldRefresh ? null : coerceDishNumber(existingRow?.calories)),
      total_fat: updates.fat_g !== undefined
        ? coerceDishNumber(updates.fat_g)
        : (shouldRefresh ? null : coerceDishNumber(existingRow?.total_fat)),
      total_carbohydrates: updates.carbs_g !== undefined
        ? coerceDishNumber(updates.carbs_g)
        : (shouldRefresh ? null : coerceDishNumber(existingRow?.total_carbohydrates)),
      protein: updates.protein_g !== undefined
        ? coerceDishNumber(updates.protein_g)
        : (shouldRefresh ? null : coerceDishNumber(existingRow?.protein)),
      ingredients: JSON.stringify(ingredients),
      components: JSON.stringify(components),
      allergens: JSON.stringify(shouldRefresh ? [] : normalizeIngredientList(parseJsonColumn(existingRow?.allergens, []))),
      analysis_status: "pending",
      analysis_error: null
    },
    async_payload: {
      dish_name: analysisRequest.dish_name,
      serving_size: analysisRequest.serving_size,
      ingredients,
      components
    }
  };
}

function resolveAsyncDishOwnerId(context) {
  return String(context?.tableOwnerId || context?.userId || context?.ownerId || "").trim();
}

async function queueAsyncDishEnrichment(context, rowId, payload, options = {}) {
  logHouseholdWriteTargets("dish enrichment dispatch", context, getTableHouseholdMemberIds(context), {
    mode: "delegated",
    delegate: "dish_edit_lambda",
    dispatch_owner_id: resolveAsyncDishOwnerId(context),
    row_id: rowId
  });
  const dispatch = await dispatchDishEnrichmentJob({
    ownerId: resolveAsyncDishOwnerId(context),
    dishId: rowId,
    payload,
    env: options?.env || process.env
  });

  return {
    pending: true,
    status: "pending",
    dish_id: rowId,
    dispatch
  };
}

async function fallbackToSynchronousDishEnrichment(connection, context, rowId, existingRow, updates, options = {}) {
  const finalFields = await buildDishUpdateFieldsFromArgs(existingRow, updates, { ...options, context });
  await updateDishRowAcrossHousehold(connection, context, rowId, finalFields);
  return {
    pending: false,
    status: "complete",
    dish_id: rowId,
    resolved_synchronously: true
  };
}

function createDishNotFoundError(reference = "that dish") {
  const error = new Error(`Couldn't find ${reference} in recent dishes.`);
  error.statusCode = 404;
  return error;
}

function createDishAmbiguityError(reference, rows) {
  const names = (rows || []).slice(0, 3).map((row) => row.dish_name || "Unknown dish");
  const error = new Error(`I found a few recent dishes for ${reference}: ${names.join(", ")}. Which one did you mean?`);
  error.statusCode = 409;
  error.details = {
    type: "ambiguous_dish",
    reference,
    candidates: (rows || []).slice(0, 5).map((row) => ({
      id: row._id,
      dish_name: row.dish_name || "Unknown dish",
      created_at: serializeDate(row._createdDate),
      action: row.action || "IN"
    }))
  };
  return error;
}

function createRecentDishContextError() {
  const error = new Error("I’m not sure which recent dish you mean. Which meal should I update?");
  error.statusCode = 409;
  error.details = {
    type: "recent_dish_context_missing"
  };
  return error;
}

function feedTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_feed_events`;
}

function legacyFeedTableName(ownerId) {
  return `${sanitizeIdentifier(ownerId, "owner")}_new_feed`;
}

function mapFeedEventRow(row) {
  return {
    id: row._id,
    event_type: row.event_type || null,
    title: row.title || null,
    brand: row.brand || null,
    action: row.action || null,
    created_at: serializeDate(row._createdDate),
    image_url: row.image_url || null,
    product_image_url: row.product_image_url || null,
    source_table: row.source_table || null,
    metadata: parseJsonColumn(row.metadata, null)
  };
}

function mapLegacyFeedRow(row) {
  return {
    id: row._id != null ? String(row._id) : null,
    event_type: row.action ? normalizeName(row.action).replace(/\s+/g, "_") : "legacy",
    title: row.product_name || null,
    brand: row.product_brand || null,
    action: row.action || null,
    created_at: serializeDate(row._createdDate || row.created_at || row.updated_at),
    image_url: row.images || null,
    product_image_url: null,
    source_table: "legacy_new_feed",
    metadata: row.product_barcode ? { product_barcode: row.product_barcode } : null
  };
}

async function ensureKitchenTable(connection, tableName) {
  await connection.execute(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY,
      \`_owner\` VARCHAR(36) NOT NULL,
      \`_device\` VARCHAR(255) NOT NULL,
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL,
      \`product_name\` VARCHAR(500) NULL,
      \`brand\` VARCHAR(255) NULL,
      \`variant\` VARCHAR(255) NULL,
      \`category\` VARCHAR(255) NULL,
      \`confidence\` DECIMAL(3,2) NULL,
      \`explanation\` TEXT NULL,
      \`product_description\` VARCHAR(500) NULL,
      \`barcode\` VARCHAR(100) NULL,
      \`country_guess\` VARCHAR(100) NULL,
      \`estimated_price\` VARCHAR(50) NULL,
      \`ingredients\` JSON NULL,
      \`nutrition_summary\` TEXT NULL,
      \`upf\` ENUM('yes', 'no') NULL,
      \`harmful_ingredients\` JSON NULL,
      \`similar_items\` JSON NULL,
      \`alternatives\` JSON NULL,
      \`healthier_alternatives\` JSON NULL,
      \`store_availability\` JSON NULL,
      \`images\` VARCHAR(1000) NULL,
      \`s3_key\` VARCHAR(500) NULL,
      \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN',
      \`product_expiration\` DATE NULL,
      \`product_image_url\` VARCHAR(1000) NULL,
      \`product_image_key\` VARCHAR(500) NULL,
      \`job_id\` VARCHAR(100) NULL,
      \`user_id\` VARCHAR(255) NULL,
      \`storage_location\` VARCHAR(255) NULL,
      \`is_opened\` TINYINT(1) NOT NULL DEFAULT 0,
      \`remaining_quantity\` VARCHAR(255) NULL,
      \`quantity_value\` DECIMAL(10,2) NULL,
      \`quantity_unit\` VARCHAR(64) NULL,
      \`fill_percent\` TINYINT UNSIGNED NULL,
      INDEX \`idx_owner\` (\`_owner\`),
      INDEX \`idx_product_name\` (\`product_name\`),
      INDEX \`idx_action\` (\`action\`),
      INDEX \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await ensureKitchenVoiceColumns(connection, tableName);
}

async function ensureKitchenVoiceColumns(connection, tableName) {
  await ensureColumn(connection, tableName, "_updatedDate", "`_updatedDate` DATETIME NULL AFTER `_createdDate`");
  await ensureColumn(connection, tableName, "storage_location", "`storage_location` VARCHAR(255) NULL AFTER `user_id`");
  await ensureColumn(connection, tableName, "is_opened", "`is_opened` TINYINT(1) NOT NULL DEFAULT 0 AFTER `storage_location`");
  await ensureColumn(connection, tableName, "remaining_quantity", "`remaining_quantity` VARCHAR(255) NULL AFTER `is_opened`");
  await ensureColumn(connection, tableName, "quantity_value", "`quantity_value` DECIMAL(10,2) NULL AFTER `remaining_quantity`");
  await ensureColumn(connection, tableName, "quantity_unit", "`quantity_unit` VARCHAR(64) NULL AFTER `quantity_value`");
  await ensureColumn(connection, tableName, "fill_percent", "`fill_percent` TINYINT UNSIGNED NULL AFTER `quantity_unit`");
  await ensureColumn(connection, tableName, "storage_guidance", "`storage_guidance` JSON NULL AFTER `fill_percent`");
}

async function ensureDiscardTable(connection, tableName) {
  await connection.execute(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY,
      \`_owner\` VARCHAR(36) NOT NULL,
      \`_device\` VARCHAR(255) NOT NULL,
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL,
      \`product_name\` VARCHAR(500) NULL,
      \`brand\` VARCHAR(255) NULL,
      \`category\` VARCHAR(255) NULL,
      \`images\` VARCHAR(1000) NULL,
      \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN',
      \`product_expiration\` DATE NULL,
      \`job_id\` VARCHAR(100) NULL,
      \`user_id\` VARCHAR(255) NULL,
      \`source_kitchen_id\` VARCHAR(36) NULL,
      \`discard_reason\` VARCHAR(255) NULL,
      INDEX \`idx_owner\` (\`_owner\`),
      INDEX \`idx_product_name\` (\`product_name\`),
      INDEX \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await ensureDiscardColumns(connection, tableName);
}

async function ensureDiscardColumns(connection, tableName) {
  await ensureColumn(connection, tableName, "_updatedDate", "`_updatedDate` DATETIME NULL AFTER `_createdDate`");
  await ensureColumn(connection, tableName, "source_kitchen_id", "`source_kitchen_id` VARCHAR(36) NULL AFTER `user_id`");
  await ensureColumn(connection, tableName, "discard_reason", "`discard_reason` VARCHAR(255) NULL AFTER `source_kitchen_id`");
}

async function ensureDishTable(connection, tableName) {
  await connection.execute(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY,
      \`_owner\` VARCHAR(36) NOT NULL,
      \`_device\` VARCHAR(255) NOT NULL,
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`_updatedDate\` DATETIME NULL,
      \`dish_name\` VARCHAR(500) NULL,
      \`confidence\` DECIMAL(3,2) NULL,
      \`explanation\` TEXT NULL,
      \`serving_size\` VARCHAR(100) NULL,
      \`calories\` DECIMAL(10,2) NULL,
      \`total_fat\` DECIMAL(10,2) NULL,
      \`saturated_fat\` DECIMAL(10,2) NULL,
      \`trans_fat\` DECIMAL(10,2) NULL,
      \`cholesterol\` DECIMAL(10,2) NULL,
      \`sodium\` DECIMAL(10,2) NULL,
      \`total_carbohydrates\` DECIMAL(10,2) NULL,
      \`dietary_fiber\` DECIMAL(10,2) NULL,
      \`sugars\` DECIMAL(10,2) NULL,
      \`protein\` DECIMAL(10,2) NULL,
      \`vitamin_a\` DECIMAL(10,2) NULL,
      \`vitamin_c\` DECIMAL(10,2) NULL,
      \`calcium\` DECIMAL(10,2) NULL,
      \`iron\` DECIMAL(10,2) NULL,
      \`ingredients\` JSON NULL,
      \`components\` JSON NULL,
      \`allergens\` JSON NULL,
      \`images\` VARCHAR(1000) NULL,
      \`s3_key\` VARCHAR(500) NULL,
      \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN',
      \`dish_image_url\` VARCHAR(1000) NULL,
      \`dish_image_key\` VARCHAR(500) NULL,
      \`job_id\` VARCHAR(100) NULL,
      \`user_id\` VARCHAR(255) NULL,
      \`analysis_status\` VARCHAR(32) NULL,
      \`analysis_error\` TEXT NULL,
      INDEX \`idx_owner\` (\`_owner\`),
      INDEX \`idx_dish_name\` (\`dish_name\`),
      INDEX \`idx_action\` (\`action\`),
      INDEX \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await ensureDishVoiceColumns(connection, tableName);
}

async function ensureDishVoiceColumns(connection, tableName) {
  await ensureColumn(connection, tableName, "_updatedDate", "`_updatedDate` DATETIME NULL AFTER `_createdDate`");
  await ensureColumn(connection, tableName, "dish_name", "`dish_name` VARCHAR(500) NULL AFTER `_updatedDate`");
  await ensureColumn(connection, tableName, "confidence", "`confidence` DECIMAL(3,2) NULL AFTER `dish_name`");
  await ensureColumn(connection, tableName, "explanation", "`explanation` TEXT NULL AFTER `confidence`");
  await ensureColumn(connection, tableName, "serving_size", "`serving_size` VARCHAR(100) NULL AFTER `explanation`");
  await ensureColumn(connection, tableName, "calories", "`calories` DECIMAL(10,2) NULL AFTER `serving_size`");
  await ensureColumn(connection, tableName, "total_fat", "`total_fat` DECIMAL(10,2) NULL AFTER `calories`");
  await ensureColumn(connection, tableName, "total_carbohydrates", "`total_carbohydrates` DECIMAL(10,2) NULL AFTER `total_fat`");
  await ensureColumn(connection, tableName, "protein", "`protein` DECIMAL(10,2) NULL AFTER `total_carbohydrates`");
  await ensureColumn(connection, tableName, "ingredients", "`ingredients` JSON NULL AFTER `protein`");
  await ensureColumn(connection, tableName, "components", "`components` JSON NULL AFTER `ingredients`");
  await ensureColumn(connection, tableName, "allergens", "`allergens` JSON NULL AFTER `components`");
  await ensureColumn(connection, tableName, "images", "`images` VARCHAR(1000) NULL AFTER `allergens`");
  await ensureColumn(connection, tableName, "s3_key", "`s3_key` VARCHAR(500) NULL AFTER `images`");
  await ensureColumn(connection, tableName, "dish_image_url", "`dish_image_url` VARCHAR(1000) NULL AFTER `action`");
  await ensureColumn(connection, tableName, "dish_image_key", "`dish_image_key` VARCHAR(500) NULL AFTER `dish_image_url`");
  await ensureColumn(connection, tableName, "job_id", "`job_id` VARCHAR(100) NULL AFTER `dish_image_key`");
  await ensureColumn(connection, tableName, "user_id", "`user_id` VARCHAR(255) NULL AFTER `job_id`");
  await ensureColumn(connection, tableName, "analysis_status", "`analysis_status` VARCHAR(32) NULL AFTER `user_id`");
  await ensureColumn(connection, tableName, "analysis_error", "`analysis_error` TEXT NULL AFTER `analysis_status`");
}

async function ensureShoppingTable(connection, tableName) {
  await connection.execute(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      \`_owner\` CHAR(36) NOT NULL,
      \`_device\` VARCHAR(64) NOT NULL,
      \`product_name\` VARCHAR(255) NOT NULL,
      \`product_brand\` VARCHAR(255) DEFAULT NULL,
      \`images\` TEXT,
      \`product_barcode\` VARCHAR(64) DEFAULT NULL,
      \`store\` VARCHAR(100) DEFAULT NULL,
      \`action\` VARCHAR(32) NOT NULL,
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`created_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`updated_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      \`household_item_uuid\` CHAR(36) DEFAULT NULL,
      PRIMARY KEY (\`_id\`),
      KEY \`idx_owner_device\` (\`_owner\`, \`_device\`),
      KEY \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await ensureColumn(connection, tableName, "_createdDate", "`_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER `action`");
  await ensureColumn(connection, tableName, "created_at", "`created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER `_createdDate`");
  await ensureColumn(connection, tableName, "updated_at", "`updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER `created_at`");
  await ensureColumn(connection, tableName, "household_item_uuid", "`household_item_uuid` CHAR(36) DEFAULT NULL AFTER `updated_at`");
  await ensureColumn(connection, tableName, "product_brand", "`product_brand` VARCHAR(255) DEFAULT NULL AFTER `product_name`");
  await ensureColumn(connection, tableName, "images", "`images` TEXT AFTER `product_brand`");
  await ensureColumn(connection, tableName, "product_barcode", "`product_barcode` VARCHAR(64) DEFAULT NULL AFTER `images`");
  await ensureColumn(connection, tableName, "store", "`store` VARCHAR(100) DEFAULT NULL AFTER `product_barcode`");
  await ensureColumn(connection, tableName, "quantity", "`quantity` VARCHAR(100) DEFAULT NULL AFTER `product_name`");
}

async function getKitchenRows(connection, ownerId, limit = 25) {
  if (WRITE_SHARED_ONLY) {
    await ensureSharedTables(connection);
    const safeLimit = clampLimit(limit, 25, 100);
    const [rows] = await connection.execute(
      `SELECT * FROM \`${SHARED_KITCHEN_TABLE}\` WHERE \`owner_id\` = ? AND \`action\` = 'IN' ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${safeLimit}`,
      [ownerId]
    );
    return rows || [];
  }

  const tableName = kitchenTableName(ownerId);
  if (!(await tableExists(connection, tableName))) {
    return [];
  }

  await ensureKitchenVoiceColumns(connection, tableName);
  const safeLimit = clampLimit(limit, 25, 100);
  const [rows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE action = 'IN'
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${safeLimit}`
  );

  return rows || [];
}

function summarizeKitchenRowForDisambiguation(row) {
  const parts = [row.product_name || "Unknown item"];
  if (row.storage_location) {
    parts.push(`in ${row.storage_location}`);
  }
  if (row.remaining_quantity) {
    parts.push(row.remaining_quantity);
  }
  if (row.product_expiration) {
    parts.push(`expires ${serializeDateOnly(row.product_expiration)}`);
  }
  return {
    item_id: row._id || null,
    item_name: row.product_name || "Unknown item",
    location: row.storage_location || null,
    remaining_quantity: row.remaining_quantity || null,
    expiration_date: serializeDateOnly(row.product_expiration),
    created_at: serializeDate(row._createdDate),
    summary: parts.join(" - ")
  };
}

function createKitchenAmbiguityError(itemName, rows) {
  const error = new Error(
    `I found multiple kitchen items matching ${itemName}. Which one did you mean?`
  );
  error.statusCode = 409;
  const candidates = (rows || []).slice(0, 5).map((row) => summarizeKitchenRowForDisambiguation(row));
  error.details = {
    type: "ambiguous_kitchen_item",
    candidates,
    suggested_selection_hints: candidates.length > 1
      ? ["most_recent", "oldest", ...(candidates.length === 2 ? ["other"] : [])]
      : ["most_recent"]
  };
  return error;
}

function createKitchenNotFoundError(itemName) {
  const error = new Error(`Couldn't find ${itemName} in the kitchen.`);
  error.statusCode = 404;
  return error;
}

async function findKitchenRowsByName(connection, ownerId, itemName, limit = 5) {
  if (WRITE_SHARED_ONLY) {
    const exactName = normalizeName(itemName);
    const [exactRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_KITCHEN_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' AND LOWER(TRIM(product_name)) = ? ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${limit}`,
      [ownerId, exactName]
    );
    if (exactRows?.length > 0) {
      return exactRows;
    }
    const [likeRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_KITCHEN_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' AND LOWER(product_name) LIKE ? ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${limit}`,
      [ownerId, buildLikePattern(itemName)]
    );
    return likeRows || [];
  }

  const tableName = kitchenTableName(ownerId);
  if (!(await tableExists(connection, tableName))) {
    return [];
  }

  await ensureKitchenVoiceColumns(connection, tableName);
  const exactName = normalizeName(itemName);
  const [exactRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE action = 'IN'
       AND LOWER(TRIM(product_name)) = ?
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${limit}`,
    [exactName]
  );

  if (exactRows?.length > 0) {
    return exactRows;
  }

  const [likeRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE action = 'IN'
       AND LOWER(product_name) LIKE ?
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${limit}`,
    [buildLikePattern(itemName)]
  );

  return likeRows || [];
}

async function findKitchenRowById(connection, ownerId, rowId) {
  if (WRITE_SHARED_ONLY) {
    const [rows] = await connection.execute(
      `SELECT * FROM \`${SHARED_KITCHEN_TABLE}\` WHERE \`_id\` = ? LIMIT 1`,
      [rowId]
    );
    return rows?.[0] || null;
  }

  const tableName = kitchenTableName(ownerId);
  if (!(await tableExists(connection, tableName))) {
    return null;
  }

  await ensureKitchenVoiceColumns(connection, tableName);
  const [rows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE _id = ?
     LIMIT 1`,
    [rowId]
  );

  return rows?.[0] || null;
}

function normalizeKitchenReference(reference) {
  if (typeof reference === "string") {
    const trimmed = reference.trim();
    return {
      item_name: trimmed || null,
      item_id: null,
      selection_hint: null
    };
  }

  const value = reference && typeof reference === "object" ? reference : {};
  const itemName = String(value.item_name || "").trim();
  const itemId = String(value.item_id || "").trim();
  const selectionHint = String(value.selection_hint || "").trim().toLowerCase();
  return {
    item_name: itemName || null,
    item_id: itemId || null,
    selection_hint: selectionHint || null
  };
}

function normalizeKitchenSelectionHint(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (["most_recent", "latest", "newest", "recent", "first", "top"].includes(normalized)) {
    return "most_recent";
  }
  if (["oldest", "earliest", "least_recent", "last", "bottom"].includes(normalized)) {
    return "oldest";
  }
  if (["other", "other_one", "second", "second_one"].includes(normalized)) {
    return "other";
  }
  return normalized;
}

function pickKitchenRowBySelectionHint(rows, selectionHint) {
  const normalizedHint = normalizeKitchenSelectionHint(selectionHint);
  if (!Array.isArray(rows) || rows.length === 0 || !normalizedHint) {
    return null;
  }
  if (normalizedHint === "most_recent") {
    return rows[0] || null;
  }
  if (normalizedHint === "oldest") {
    return rows[rows.length - 1] || null;
  }
  if (normalizedHint === "other" && rows.length === 2) {
    return rows[1] || null;
  }
  return null;
}

function isHaloSurface(context) {
  return String(context?.responseSurface || "").trim().toLowerCase() === "halo";
}

// True when a quantity target effectively means "none left" — i.e. a removal.
// Used so Halo can apply a set-to-zero across all matching rows like a discard.
function isZeroQuantityTarget(quantityInput) {
  const zeroText = /^\s*(0|0+\.0+|zero|none|nothing|empty|gone|all\s*gone|used\s*up|finished)\s*$/i;
  if (quantityInput == null) {
    return false;
  }
  if (typeof quantityInput === "number") {
    return quantityInput === 0;
  }
  if (typeof quantityInput === "string") {
    return zeroText.test(quantityInput);
  }
  if (typeof quantityInput === "object") {
    if (quantityInput.quantity_value === 0 || String(quantityInput.quantity_value).trim() === "0") {
      return true;
    }
    if (quantityInput.fill_percent === 0 || String(quantityInput.fill_percent).trim() === "0") {
      return true;
    }
    const txt = String(
      quantityInput.remaining_quantity
        || quantityInput.quantity
        || quantityInput.quantity_description
        || ""
    ).trim();
    return zeroText.test(txt);
  }
  return false;
}

// Resolve a spoken kitchen reference to the row(s) an action should target.
// Returns an array (possibly empty). On a one-way Halo surface an ambiguous
// multi-match resolves to action instead of an unanswerable clarifying question:
// the caller decides whether to act on all matches (removals) or just the newest (edits).
// App and any non-halo surface keep the existing behavior: throw to ask for clarification.
async function resolveKitchenRowsByReference(connection, context, reference) {
  const resolved = normalizeKitchenReference(reference);
  const tableOwnerId = resolveTableOwnerId(context);

  if (resolved.item_id) {
    const rowById = await findKitchenRowById(connection, tableOwnerId, resolved.item_id);
    if (rowById?.action === "IN") {
      return [rowById];
    }
  }

  if (!resolved.item_name) {
    return [];
  }

  const rows = await findKitchenRowsByName(connection, tableOwnerId, resolved.item_name, 5);
  console.log("[DEBUG] kitchen item resolution:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    item_name: resolved.item_name,
    item_id: resolved.item_id || null,
    selection_hint: resolved.selection_hint || null,
    responseSurface: context?.responseSurface || null,
    matchCount: rows.length,
    candidates: rows.slice(0, 5).map((row) => summarizeKitchenRowForDisambiguation(row))
  }));

  if (rows.length === 0) {
    return [];
  }
  if (rows.length === 1) {
    return [rows[0]];
  }

  const hintedRow = pickKitchenRowBySelectionHint(rows, resolved.selection_hint);
  if (hintedRow) {
    return [hintedRow];
  }

  if (isHaloSurface(context)) {
    // One-way device: never block on a question the user cannot answer.
    // Return every match; the caller narrows to newest (edits) or acts on all (removals).
    return rows;
  }

  throw createKitchenAmbiguityError(resolved.item_name, rows);
}

// Single-row resolution used by edits and reads. On Halo, an ambiguous multi-match
// resolves to the most recent row (rows are ordered newest-first) instead of throwing.
async function resolveKitchenRowByReference(connection, context, reference) {
  const rows = await resolveKitchenRowsByReference(connection, context, reference);
  return rows.length > 0 ? rows[0] : null;
}

async function updateKitchenRowAcrossHousehold(connection, context, rowId, fields) {
  if (WRITE_SHARED_ONLY) {
    const assignments = [];
    const values = [];
    for (const [key, value] of Object.entries(fields)) {
      assignments.push(`\`${key}\` = ?`);
      values.push(value);
    }
    assignments.push("`_updatedDate` = NOW()");
    values.push(rowId);
    const [updateResult] = await connection.execute(
      `UPDATE \`${SHARED_KITCHEN_TABLE}\` SET ${assignments.join(", ")} WHERE _id = ?`,
      values
    );
    // Phantom-write guard: `_updatedDate = NOW()` is always in the SET, so a
    // matched row ALWAYS reports affectedRows >= 1. Zero rows means the target
    // wasn't there — fail loudly instead of letting a no-op masquerade as success
    // (the removal-path lesson: a silent 0-row flip read back as "removed").
    if ((updateResult?.affectedRows ?? 0) < 1) {
      throw new Error(`kitchen row update matched 0 rows (id=${rowId}) — item not found`);
    }
    return;
  }

  const memberIds = getTableHouseholdMemberIds(context);
  const assignments = [];
  const values = [];

  for (const [key, value] of Object.entries(fields)) {
    assignments.push(`\`${key}\` = ?`);
    values.push(value);
  }

  assignments.push("`_updatedDate` = NOW()");
  values.push(rowId);

  logHouseholdWriteTargets("kitchen row propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId,
    fields: Object.keys(fields)
  });

  for (const memberId of memberIds) {
    const tableName = kitchenTableName(memberId);
    if (!(await tableExists(connection, tableName))) {
      continue;
    }

    await ensureKitchenVoiceColumns(connection, tableName);
    await connection.execute(
      `UPDATE \`${tableName}\`
       SET ${assignments.join(", ")}
       WHERE _id = ?`,
      values
    );
  }
}

function estimateHeuristicStorageGuidance(payload) {
  const haystack = [
    payload?.item_name,
    payload?.brand,
    payload?.category,
    payload?.description
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (!haystack) {
    return null;
  }

  const containsAny = (terms) => terms.some((t) => haystack.includes(t));

  if (containsAny(["berry", "berries", "spinach", "lettuce", "salad", "greens", "herb", "cilantro", "parsley", "mushroom"])) {
    return { summary: "Fresh delicate produce is usually best within about 3-7 days refrigerated.", min_days: 3, max_days: 7, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.72, source: "heuristic" };
  }
  if (containsAny(["apple", "orange", "lemon", "lime", "grape", "cabbage", "carrot"])) {
    return { summary: "Whole produce like this often keeps around 14-30 days when refrigerated.", min_days: 14, max_days: 30, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.7, source: "heuristic" };
  }
  if (containsAny(["banana", "avocado", "tomato", "peach", "pear", "plum"])) {
    return { summary: "Best within a few days at room temperature; refrigerate only to slow further ripening.", min_days: 2, max_days: 7, timing_start: "from_check_in", storage_zone: "counter", confidence: 0.68, source: "heuristic" };
  }
  if (containsAny(["potato", "onion", "garlic", "squash"])) {
    return { summary: "Whole produce like this usually keeps for weeks in a cool pantry area.", min_days: 14, max_days: 45, timing_start: "from_check_in", storage_zone: "pantry", confidence: 0.74, source: "heuristic" };
  }
  if (containsAny(["produce", "fruit", "vegetable", "veggie", "cucumber", "pepper", "broccoli", "cauliflower", "celery"])) {
    return { summary: "Fresh produce is typically best within about 5-14 days when refrigerated.", min_days: 5, max_days: 14, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.64, source: "heuristic" };
  }
  if (containsAny(["milk", "half and half", "cream", "creamer"])) {
    return { summary: "Once opened, dairy like this is usually best within about 5-7 days refrigerated.", min_days: 5, max_days: 7, timing_start: "after_opening", storage_zone: "refrigerated", confidence: 0.76, source: "heuristic" };
  }
  if (containsAny(["yogurt", "yoghurt", "cottage cheese", "sour cream", "dip"])) {
    return { summary: "Once opened, this is usually best within about 5-10 days refrigerated.", min_days: 5, max_days: 10, timing_start: "after_opening", storage_zone: "refrigerated", confidence: 0.72, source: "heuristic" };
  }
  if (containsAny(["egg", "eggs"])) {
    return { summary: "Eggs typically keep around 21-35 days refrigerated from check-in.", min_days: 21, max_days: 35, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.75, source: "heuristic" };
  }
  if (containsAny(["cheese", "butter"])) {
    return { summary: "Once opened, this is usually best within about 7-21 days refrigerated.", min_days: 7, max_days: 21, timing_start: "after_opening", storage_zone: "refrigerated", confidence: 0.66, source: "heuristic" };
  }
  if (containsAny(["chicken", "beef", "pork", "turkey", "fish", "salmon", "shrimp", "meat", "seafood"])) {
    return { summary: "Fresh meat or seafood is usually best within about 1-3 days refrigerated.", min_days: 1, max_days: 3, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.8, source: "heuristic" };
  }
  if (containsAny(["deli", "ham", "bacon", "sausage", "prosciutto", "salami"])) {
    return { summary: "Opened deli meats are usually best within about 3-7 days refrigerated.", min_days: 3, max_days: 7, timing_start: "after_opening", storage_zone: "refrigerated", confidence: 0.72, source: "heuristic" };
  }
  if (containsAny(["chips", "cracker", "cookie", "cookies", "pretzel", "popcorn", "granola bar", "snack"])) {
    return { summary: "Once opened, this is usually best within about 7-21 days in a sealed pantry container.", min_days: 7, max_days: 21, timing_start: "after_opening", storage_zone: "pantry", confidence: 0.7, source: "heuristic" };
  }
  if (containsAny(["chocolate", "candy", "sweet", "dessert"])) {
    return { summary: "This type of sweet snack often keeps around 30-180 days in a cool pantry.", min_days: 30, max_days: 180, timing_start: "after_opening", storage_zone: "pantry", confidence: 0.58, source: "heuristic" };
  }
  if (containsAny(["soda", "sparkling water", "juice", "tea", "coffee", "energy drink", "kombucha", "drink", "beverage"])) {
    return { summary: "Once opened, beverages like this are usually best within about 1-7 days refrigerated.", min_days: 1, max_days: 7, timing_start: "after_opening", storage_zone: "refrigerated", confidence: 0.62, source: "heuristic" };
  }
  if (containsAny(["sauce", "salsa", "dressing", "ketchup", "mustard", "mayo", "mayonnaise", "broth", "stock"])) {
    return { summary: "Once opened, condiments or sauces like this are usually best within about 7-60 days.", min_days: 7, max_days: 60, timing_start: "after_opening", storage_zone: "mixed", confidence: 0.56, source: "heuristic" };
  }
  if (containsAny(["pasta", "rice", "bean", "lentil", "flour", "oat", "granola", "cereal", "oil", "vinegar", "spice", "seasoning", "canned"])) {
    return { summary: "Shelf-stable pantry staples like this can keep for months; once opened, best within about 30-180 days.", min_days: 30, max_days: 180, timing_start: "after_opening", storage_zone: "pantry", confidence: 0.55, source: "heuristic" };
  }
  if (containsAny(["frozen", "pizza", "burrito", "entree", "meal", "prepared", "leftover", "takeout", "soup"])) {
    return { summary: "Prepared foods are usually best within about 3-5 days refrigerated, or longer frozen.", min_days: 3, max_days: 5, timing_start: "from_check_in", storage_zone: "refrigerated", confidence: 0.68, source: "heuristic" };
  }
  if (containsAny(["bread", "bagel", "tortilla", "wrap", "bun", "roll", "muffin", "croissant", "pastry"])) {
    return { summary: "Bread products are usually best within about 3-7 days at room temperature.", min_days: 3, max_days: 7, timing_start: "from_check_in", storage_zone: "counter", confidence: 0.72, source: "heuristic" };
  }

  return { summary: "A typical home-kitchen window for this item is about 5-14 days once opened or properly stored.", min_days: 5, max_days: 14, timing_start: "after_opening", storage_zone: "mixed", confidence: 0.35, source: "heuristic" };
}

// The iOS app filters its kitchen sections by an exact lowercase snake_case
// category enum. Thyme's free-text category guess ("Condiment", "Produce/Dip")
// or a null guess otherwise leaves the row uncategorized/invisible. Normalize
// every voice check-in onto the enum: map the guess, fall back to storage
// location, and default to prepared_other — NEVER write a null/free-text value.
const KITCHEN_CATEGORY_ENUM = new Set([
  "leftovers", "produce", "dairy_eggs", "meat_seafood",
  "pantry", "snacks_sweets", "beverages", "prepared_other",
]);
const KITCHEN_CATEGORY_KEYWORDS = [
  [["leftover"], "leftovers"],
  [["meat", "seafood", "fish", "poultry", "beef", "pork", "bacon", "sausage", "chicken", "deli"], "meat_seafood"],
  [["dairy", "creamer", "cheese", "milk", "yogurt", "butter", "egg"], "dairy_eggs"],
  [["beverage", "drink", "juice", "soda", "coffee", "tea", "water"], "beverages"],
  [["snack", "sweet", "dessert", "candy", "chocolate", "chip", "cookie", "cracker"], "snacks_sweets"],
  [["produce", "fruit", "vegetable", "veggie", "veg", "dip", "herb"], "produce"],
  [["pantry", "condiment", "baking", "spice", "season", "oil", "sauce", "canned", "grain", "pasta", "rice", "nut", "dry", "vinegar", "syrup", "honey", "bouillon", "flour", "sugar"], "pantry"],
  [["prepared", "leftover", "meal", "other", "misc"], "prepared_other"],
];
const KITCHEN_STORAGE_CATEGORY_FALLBACK = {
  pantry: "pantry", produce: "produce", snacks: "snacks_sweets",
};

// Map a raw category guess (+ storage location) onto the app's category enum.
// Never returns null/empty — defaults to 'prepared_other'.
export function normalizeKitchenCategory(rawCategory, storageLocation) {
  const raw = (rawCategory == null ? "" : String(rawCategory)).trim().toLowerCase();
  if (KITCHEN_CATEGORY_ENUM.has(raw)) return raw;
  if (raw) {
    const cleaned = raw.replace(/[^a-z]+/g, " ");
    for (const [keys, value] of KITCHEN_CATEGORY_KEYWORDS) {
      if (keys.some((k) => cleaned.includes(k))) return value;
    }
  }
  const storage = (storageLocation == null ? "" : String(storageLocation)).trim().toLowerCase();
  if (KITCHEN_STORAGE_CATEGORY_FALLBACK[storage]) return KITCHEN_STORAGE_CATEGORY_FALLBACK[storage];
  return "prepared_other";
}

async function insertKitchenRowAcrossHousehold(connection, context, payload) {
  const rowId = crypto.randomUUID();
  const jobId = `voice-${rowId}`;
  const structuredState = parseStructuredState(payload);
  const itemName = formatUserFacingItemName(payload?.item_name);
  const storageGuidance = estimateHeuristicStorageGuidance(payload);
  // Normalize onto the app category enum once; never persist null/free-text.
  const normalizedCategory = normalizeKitchenCategory(payload.category, payload.location);

  if (WRITE_SHARED_ONLY) {
    const tableOwnerId = resolveTableOwnerId(context);
    // Voice check-ins have NO capture image, so bulkEnrichmentEngine (image-path)
    // never promotes them past analysis_stage='preliminary' — and the shared_kitchen
    // column DEFAULT is 'preliminary'. That left every voice item showing the iOS
    // "ANALYZING" badge forever. Write analysis_stage='final'/status='ready' explicitly
    // so voice adds are terminal from the start. (Image path keeps preliminary->final.)
    await connection.execute(
      `INSERT INTO \`${SHARED_KITCHEN_TABLE}\` (
        _id, _owner, _device, _createdDate, _updatedDate,
        product_name, brand, category, product_description,
        images, s3_key, action, product_expiration,
        job_id, user_id, storage_location, is_opened,
        remaining_quantity, quantity_value, quantity_unit,
        fill_percent, storage_guidance, owner_id,
        analysis_stage, analysis_status
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'final', 'ready')`,
      [
        rowId,
        tableOwnerId,
        "voice-assistant",
        itemName,
        payload.brand || null,
        normalizedCategory,
        payload.description || null,
        null,
        null,
        payload.expiration_date || null,
        jobId,
        context.userId,
        payload.location || null,
        payload.is_opened ? 1 : 0,
        structuredState.remaining_quantity,
        structuredState.quantity_value,
        structuredState.quantity_unit,
        structuredState.fill_percent,
        storageGuidance ? JSON.stringify(storageGuidance) : null,
        tableOwnerId
      ]
    );
    return rowId;
  }

  const memberIds = getTableHouseholdMemberIds(context);

  logHouseholdWriteTargets("kitchen row insert propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId,
    item_name: itemName || null
  });

  for (const memberId of memberIds) {
    const tableName = kitchenTableName(memberId);
    await ensureKitchenTable(connection, tableName);
    await connection.execute(
      `INSERT INTO \`${tableName}\` (
        _id,
        _owner,
        _device,
        _createdDate,
        _updatedDate,
        product_name,
        brand,
        category,
        product_description,
        images,
        s3_key,
        action,
        product_expiration,
        job_id,
        user_id,
        storage_location,
        is_opened,
        remaining_quantity,
        quantity_value,
        quantity_unit,
        fill_percent,
        storage_guidance
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        rowId,
        memberId,
        "voice-assistant",
        itemName,
        payload.brand || null,
        normalizedCategory,
        payload.description || null,
        null,
        null,
        payload.expiration_date || null,
        jobId,
        context.userId,
        payload.location || null,
        payload.is_opened ? 1 : 0,
        structuredState.remaining_quantity,
        structuredState.quantity_value,
        structuredState.quantity_unit,
        structuredState.fill_percent,
        storageGuidance ? JSON.stringify(storageGuidance) : null
      ]
    );
  }

  return rowId;
}

async function insertDiscardRowAcrossHousehold(connection, context, kitchenRow, reason) {
  const discardId = crypto.randomUUID();

  if (WRITE_SHARED_ONLY) {
    const tableOwnerId = resolveTableOwnerId(context);
    await connection.execute(
      `INSERT INTO \`${SHARED_DISCARDS_TABLE}\` (
        \`_id\`, \`_owner\`, \`_device\`, \`_createdDate\`, \`_updatedDate\`,
        \`product_name\`, \`brand\`, \`category\`, \`images\`,
        \`action\`, \`product_expiration\`, \`job_id\`, \`user_id\`,
        \`source_kitchen_id\`, \`discard_reason\`, \`owner_id\`
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?)`,
      [
        discardId,
        tableOwnerId,
        "voice-assistant",
        kitchenRow.product_name || null,
        kitchenRow.brand || null,
        kitchenRow.category || null,
        kitchenRow.images || null,
        kitchenRow.product_expiration || null,
        kitchenRow.job_id || `voice-discard-${discardId}`,
        context.userId,
        kitchenRow._id || null,
        reason || null,
        tableOwnerId
      ]
    );

    // ALSO write every household member's per-user {member}_discards table. The
    // Discard log (discards_api) reads ONLY {user}_discards and iterates household
    // members (app.py:436/:498) — without this, voice discards are invisible in the
    // app. Discards are HOUSEHOLD-VISIBLE, so fan out to all members (like the
    // shopping fix); per-user tables have no owner_id column.
    // Thin wrapper over the helper: discards are schema-tolerant (dynamic optional
    // columns), so buildStatement inspects each member table's columns per-member.
    await dualWriteAcrossHousehold(connection, {
      memberIds: getTableHouseholdMemberIds(context), context, op: "insert", reportEvt: "discard_peruser_write_miss",
      tableFn: discardTableName, ensureFn: ensureDiscardTable,
      buildStatement: async (tableName, memberId) => {
        const columns = await getTableColumns(connection, tableName);
        const fieldNames = ["_id", "_owner", "_device", "_createdDate", "product_name", "brand", "category", "images", "action", "product_expiration", "job_id", "user_id"];
        const values = [discardId, memberId, "voice-assistant", kitchenRow.product_name || null, kitchenRow.brand || null, kitchenRow.category || null, kitchenRow.images || null, "IN", kitchenRow.product_expiration || null, kitchenRow.job_id || `voice-discard-${discardId}`, context.userId];
        if (columns.has("_updatedDate")) fieldNames.push("_updatedDate");
        if (columns.has("source_kitchen_id")) { fieldNames.push("source_kitchen_id"); values.push(kitchenRow._id); }
        if (columns.has("discard_reason")) { fieldNames.push("discard_reason"); values.push(reason || null); }
        const sqlFields = fieldNames.map((field) => `\`${field}\``).join(", ");
        const sqlValues = fieldNames.map((field) => (field === "_createdDate" || field === "_updatedDate" ? "NOW()" : "?")).join(", ");
        return { sql: `INSERT INTO \`${tableName}\` (${sqlFields}) VALUES (${sqlValues})`, values };
      },
    });
    return discardId;
  }

  const memberIds = getTableHouseholdMemberIds(context);

  logHouseholdWriteTargets("discard row insert propagation", context, memberIds, {
    mode: "direct",
    discard_id: discardId,
    item_name: kitchenRow?.product_name || kitchenRow?.item_name || null
  });

  for (const memberId of memberIds) {
    const tableName = discardTableName(memberId);
    await ensureDiscardTable(connection, tableName);
    const columns = await getTableColumns(connection, tableName);
    const fieldNames = [
      "_id",
      "_owner",
      "_device",
      "_createdDate",
      "product_name",
      "brand",
      "category",
      "images",
      "action",
      "product_expiration",
      "job_id",
      "user_id"
    ];
    const values = [
      discardId,
      memberId,
      "voice-assistant",
      kitchenRow.product_name || null,
      kitchenRow.brand || null,
      kitchenRow.category || null,
      kitchenRow.images || null,
      "IN",
      kitchenRow.product_expiration || null,
      kitchenRow.job_id || `voice-discard-${discardId}`,
      context.userId
    ];

    if (columns.has("_updatedDate")) {
      fieldNames.push("_updatedDate");
    }
    if (columns.has("source_kitchen_id")) {
      fieldNames.push("source_kitchen_id");
      values.push(kitchenRow._id);
    }
    if (columns.has("discard_reason")) {
      fieldNames.push("discard_reason");
      values.push(reason || null);
    }

    const sqlFields = fieldNames.map((field) => `\`${field}\``).join(", ");
    const sqlValues = fieldNames.map((field) => (field === "_createdDate" || field === "_updatedDate" ? "NOW()" : "?")).join(", ");

    await connection.execute(
      `INSERT INTO \`${tableName}\` (${sqlFields}) VALUES (${sqlValues})`,
      values
    );
  }

  return discardId;
}

function mapShoppingRow(row) {
  return {
    shopping_id: row._id != null ? String(row._id) : null,
    household_item_uuid: row.household_item_uuid || null,
    item_name: row.product_name || row.item || "Unknown item",
    brand: row.product_brand || row.brand || null,
    quantity: row.quantity || null,
    store: row.store || null,
    source: "new_list",
    action: row.action || null,
    created_at: serializeDate(row.created_at || row._createdDate),
    updated_at: serializeDate(row.updated_at || row._createdDate)
  };
}

function storeKeys(value) {
  const normalized = normalizeName(value);
  const compact = normalized.replace(/\s+/g, "");
  const initials = normalized
    .split(" ")
    .map((token) => token[0] || "")
    .join("");
  const keys = new Set([compact, initials].filter(Boolean));

  if (compact === "wholefoods" || compact === "wholefoodsmarket") {
    keys.add("wf");
  }
  if (compact === "wf") {
    keys.add("wholefoods");
    keys.add("wholefoodsmarket");
  }
  if (compact === "costcowholesale") {
    keys.add("costco");
  }

  return keys;
}

function storesReferToSamePlace(left, right) {
  const leftKeys = storeKeys(left);
  const rightKeys = storeKeys(right);

  for (const key of leftKeys) {
    if (rightKeys.has(key)) {
      return true;
    }
    for (const otherKey of rightKeys) {
      if (key.length >= 3 && otherKey.includes(key)) {
        return true;
      }
      if (otherKey.length >= 3 && key.includes(otherKey)) {
        return true;
      }
    }
  }

  return false;
}

function getShoppingHouseholdMemberIds(context) {
  return normalizeHouseholdMemberIds([
    resolveShoppingOwnerId(context),
    ...getTableHouseholdMemberIds(context)
  ]);
}

async function getShoppingRows(connection, context) {
  if (WRITE_SHARED_ONLY) {
    const ownerId = resolveShoppingOwnerId(context);
    const [rows] = await connection.execute(
      `SELECT _id, household_item_uuid, product_name, product_brand, store, action, _createdDate, created_at, updated_at FROM \`${SHARED_SHOPPING_TABLE}\` WHERE \`owner_id\` = ? ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC`,
      [ownerId]
    );
    return rows || [];
  }

  const tableName = shoppingListTableName(resolveShoppingOwnerId(context));
  await ensureShoppingTable(connection, tableName);

  const [rows] = await connection.execute(
    `SELECT _id, household_item_uuid, product_name, product_brand, store, action, _createdDate, created_at, updated_at
     FROM \`${tableName}\`
     ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC`
  );

  return rows || [];
}

async function hydrateShoppingRows(connection, context, shoppingRows) {
  return shoppingRows.map(mapShoppingRow);
}

async function resolveShoppingStore(connection, context, requestedStore) {
  const normalizedRequestedStore = String(requestedStore || "").trim();
  if (!normalizedRequestedStore) {
    return null;
  }

  const existingStores = await listResolvedStoreTabs(connection, context);
  const matchedStore = existingStores.find((store) => storesReferToSamePlace(normalizedRequestedStore, store));
  console.log("[DEBUG] shopping store resolution:", JSON.stringify({
    requestedStore: normalizedRequestedStore,
    resolvedStore: matchedStore || normalizedRequestedStore,
    existingStores
  }));
  return matchedStore || normalizedRequestedStore;
}

async function listResolvedStoreTabs(connection, context) {
  if (WRITE_SHARED_ONLY) {
    const ownerId = resolveShoppingOwnerId(context);
    const [rows] = await connection.execute(
      `SELECT DISTINCT store FROM \`${SHARED_SHOPPING_TABLE}\` WHERE \`owner_id\` = ? AND store IS NOT NULL AND TRIM(store) <> '' ORDER BY store`,
      [ownerId]
    );
    return (rows || []).map((row) => String(row.store || "").trim()).filter(Boolean);
  }

  const tableName = shoppingListTableName(resolveShoppingOwnerId(context));
  await ensureShoppingTable(connection, tableName);

  const [rows] = await connection.execute(
    `SELECT DISTINCT store
     FROM \`${tableName}\`
     WHERE store IS NOT NULL
       AND TRIM(store) <> ''
     ORDER BY store`
  );

  return (rows || [])
    .map((row) => String(row.store || "").trim())
    .filter(Boolean);
}

function singularizeShoppingToken(token) {
  if (token.endsWith("ies") && token.length > 3) {
    return `${token.slice(0, -3)}y`;
  }
  if (token.endsWith("ses") && token.length > 3) {
    return token.slice(0, -2);
  }
  if (token.endsWith("s") && !token.endsWith("ss") && token.length > 3) {
    return token.slice(0, -1);
  }
  return token;
}

function tokenizeShoppingName(value) {
  return normalizeName(value)
    .split(" ")
    .map((token) => singularizeShoppingToken(token.trim()))
    .filter(Boolean);
}

function compactShoppingName(value) {
  return normalizeName(value).replace(/\s+/g, "");
}

function levenshteinDistance(left, right) {
  if (left === right) {
    return 0;
  }
  if (!left.length) {
    return right.length;
  }
  if (!right.length) {
    return left.length;
  }

  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  const current = new Array(right.length + 1);

  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    current[0] = leftIndex + 1;
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      const substitutionCost = left[leftIndex] === right[rightIndex] ? 0 : 1;
      current[rightIndex + 1] = Math.min(
        current[rightIndex] + 1,
        previous[rightIndex + 1] + 1,
        previous[rightIndex] + substitutionCost
      );
    }

    for (let index = 0; index < previous.length; index += 1) {
      previous[index] = current[index];
    }
  }

  return previous[right.length];
}

function scoreShoppingItemMatch(requestedName, candidateName) {
  const normalizedRequested = normalizeName(requestedName);
  const normalizedCandidate = normalizeName(candidateName);
  if (!normalizedRequested || !normalizedCandidate) {
    return 0;
  }
  if (normalizedRequested === normalizedCandidate) {
    return 1;
  }

  const compactRequested = compactShoppingName(requestedName);
  const compactCandidate = compactShoppingName(candidateName);
  if (compactRequested && compactRequested === compactCandidate) {
    return 0.99;
  }

  let score = 0;
  if (
    compactRequested.length >= 3
    && compactCandidate.length >= 3
    && (compactCandidate.includes(compactRequested) || compactRequested.includes(compactCandidate))
  ) {
    score = Math.max(score, 0.9);
  }

  const requestedTokens = tokenizeShoppingName(requestedName);
  const candidateTokens = tokenizeShoppingName(candidateName);
  if (requestedTokens.length > 0 && candidateTokens.length > 0) {
    const candidateTokenSet = new Set(candidateTokens);
    const sharedCount = requestedTokens.filter((token) => candidateTokenSet.has(token)).length;
    const overlap = sharedCount / Math.max(requestedTokens.length, candidateTokens.length);
    score = Math.max(score, overlap * 0.78);

    if (overlap >= 0.5 && compactRequested && compactCandidate) {
      const distance = levenshteinDistance(compactRequested, compactCandidate);
      const similarity = 1 - (distance / Math.max(compactRequested.length, compactCandidate.length, 1));
      if (similarity >= 0.7) {
        score = Math.max(score, 0.88 + ((similarity - 0.7) * 0.1));
      }
    }
  }

  if (compactRequested && compactCandidate) {
    const distance = levenshteinDistance(compactRequested, compactCandidate);
    const similarity = 1 - (distance / Math.max(compactRequested.length, compactCandidate.length, 1));
    score = Math.max(score, similarity * 0.85);
  }

  return Number(score.toFixed(4));
}

function rankShoppingItemMatches(items, itemName) {
  return (items || [])
    .map((item) => ({
      item,
      score: scoreShoppingItemMatch(itemName, item.item_name)
    }))
    .filter((match) => match.score > 0)
    .sort((left, right) => right.score - left.score);
}

function createShoppingNotFoundError(itemName) {
  const error = new Error(`Couldn't find ${itemName} on the shopping list.`);
  error.statusCode = 404;
  return error;
}

function createShoppingAmbiguityError(itemName, matches) {
  const labels = matches.map((match) => match.item.item_name).filter(Boolean).slice(0, 3);
  const error = new Error(`I found a few possible shopping-list matches for ${itemName}: ${labels.join(", ")}. Which one did you mean?`);
  error.statusCode = 409;
  error.details = {
    requested_item_name: itemName,
    matches: matches.slice(0, 5).map((match) => ({
      item_name: match.item.item_name,
      store: match.item.store || null,
      score: match.score
    }))
  };
  return error;
}

function resolveShoppingItemMatch(items, itemName, options = {}) {
  const threshold = Number.isFinite(Number(options.threshold)) ? Number(options.threshold) : 0.82;
  const ambiguityDelta = Number.isFinite(Number(options.ambiguityDelta)) ? Number(options.ambiguityDelta) : 0.05;
  const rankedMatches = rankShoppingItemMatches(items, itemName);
  const topMatches = rankedMatches.slice(0, 3);
  const bestMatch = rankedMatches[0] || null;
  const secondMatch = rankedMatches[1] || null;

  console.log("[DEBUG] shopping item resolution:", JSON.stringify({
    requestedItemName: itemName,
    threshold,
    topMatches: topMatches.map((match) => ({
      item_name: match.item.item_name,
      store: match.item.store || null,
      score: match.score
    }))
  }));

  if (!bestMatch || bestMatch.score < threshold) {
    return null;
  }

  if (
    secondMatch
    && secondMatch.score >= threshold
    && (bestMatch.score - secondMatch.score) < ambiguityDelta
    && bestMatch.item.shopping_id !== secondMatch.item.shopping_id
  ) {
    throw createShoppingAmbiguityError(itemName, rankedMatches);
  }

  return bestMatch.item;
}

function findShoppingItemMatch(items, itemName) {
  return resolveShoppingItemMatch(items, itemName);
}

function maybeCollapseSplitShoppingBatchItems(items, batchItems) {
  const collapsedItems = [];

  for (let index = 0; index < (batchItems || []).length; index += 1) {
    const currentItem = batchItems[index];
    const nextItem = batchItems[index + 1];

    if (
      currentItem
      && nextItem
      && normalizeName(currentItem.store) === normalizeName(nextItem.store)
    ) {
      const combinedName = `${currentItem.item_name} ${nextItem.item_name}`.trim();
      const combinedMatch = rankShoppingItemMatches(items, combinedName)[0] || null;
      const currentMatch = rankShoppingItemMatches(items, currentItem.item_name)[0] || null;
      const nextMatch = rankShoppingItemMatches(items, nextItem.item_name)[0] || null;
      const currentStrong = currentMatch?.score >= 0.82;
      const nextStrong = nextMatch?.score >= 0.82;
      const strongestPieceScore = Math.max(currentMatch?.score || 0, nextMatch?.score || 0);

      // Speech transcripts sometimes split one item into two fragments; collapse those before writing.
      if (
        combinedMatch?.score >= 0.84
        && combinedMatch.score >= strongestPieceScore + 0.2
        && (!currentStrong || !nextStrong)
      ) {
        console.log("[DEBUG] shopping batch item collapse:", JSON.stringify({
          combinedName,
          originalItems: [currentItem.item_name, nextItem.item_name],
          matchedItemName: combinedMatch.item.item_name,
          score: combinedMatch.score
        }));
        collapsedItems.push({
          item_name: combinedName,
          store: currentItem.store
        });
        index += 1;
        continue;
      }
    }

    collapsedItems.push(currentItem);
  }

  return collapsedItems;
}

async function updateShoppingStoreAcrossHousehold(connection, context, target, storeValue) {
  await updateShoppingFieldsAcrossHousehold(connection, context, target, { store: storeValue });
}

async function updateShoppingFieldsAcrossHousehold(connection, context, target, fields) {
  if (WRITE_SHARED_ONLY) {
    const assignments = Object.keys(fields)
      .map((field) => `\`${field}\` = ?`)
      .concat("updated_at = NOW()");
    const fieldValues = Object.values(fields);
    if (target.household_item_uuid) {
      await connection.execute(
        `UPDATE \`${SHARED_SHOPPING_TABLE}\` SET ${assignments.join(", ")} WHERE household_item_uuid = ?`,
        [...fieldValues, target.household_item_uuid]
      );
    } else if (target.shopping_id) {
      await connection.execute(
        `UPDATE \`${SHARED_SHOPPING_TABLE}\` SET ${assignments.join(", ")} WHERE _id = ? LIMIT 1`,
        [...fieldValues, target.shopping_id]
      );
    } else {
      const [rows] = await connection.execute(
        `SELECT _id FROM \`${SHARED_SHOPPING_TABLE}\` WHERE \`owner_id\` = ? AND LOWER(TRIM(product_name)) = ? ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC LIMIT 1`,
        [resolveShoppingOwnerId(context), normalizeName(target.item_name)]
      );
      const rowId = rows?.[0]?._id;
      if (rowId) {
        await connection.execute(
          `UPDATE \`${SHARED_SHOPPING_TABLE}\` SET ${assignments.join(", ")} WHERE _id = ? LIMIT 1`,
          [...fieldValues, rowId]
        );
      }
    }
    // Fall through (no early return): ALSO update the per-user {member}_new_list
    // tables the app/device read (check/uncheck, store, rename).
  }

  const shoppingOwnerId = resolveShoppingOwnerId(context);
  const memberIds = getShoppingHouseholdMemberIds(context);
  const assignments = Object.keys(fields)
    .map((field) => `\`${field}\` = ?`)
    .concat("updated_at = NOW()");
  const fieldValues = Object.values(fields);

  logHouseholdWriteTargets("shopping field propagation", context, memberIds, {
    mode: "direct",
    rowScope: target?.household_item_uuid || target?.shopping_id || target?.item_name || null,
    fields: Object.keys(fields)
  });

  await dualWriteAcrossHousehold(connection, {
    memberIds, context, op: "update", reportEvt: "shopping_peruser_write_miss",
    tableFn: shoppingListTableName, ensureFn: ensureShoppingTable,
    buildStatement: async (tableName, memberId, conn) => {
      if (target.household_item_uuid) {
        return {
          sql: `UPDATE \`${tableName}\`
         SET ${assignments.join(", ")}
         WHERE household_item_uuid = ?`,
          values: [...fieldValues, target.household_item_uuid],
        };
      }
      if (memberId === shoppingOwnerId && target.shopping_id) {
        return {
          sql: `UPDATE \`${tableName}\`
         SET ${assignments.join(", ")}
         WHERE _id = ?
         LIMIT 1`,
          values: [...fieldValues, target.shopping_id],
        };
      }
      const [rows] = await conn.execute(
        `SELECT _id
       FROM \`${tableName}\`
       WHERE LOWER(TRIM(product_name)) = ?
       ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC
       LIMIT 1`,
        [normalizeName(target.item_name)]
      );
      const rowId = rows?.[0]?._id;
      if (!rowId) return null;
      return {
        sql: `UPDATE \`${tableName}\`
       SET ${assignments.join(", ")}
       WHERE _id = ?
       LIMIT 1`,
        values: [...fieldValues, rowId],
      };
    },
  });
}

function sortUseUpCandidates(items) {
  const now = Date.now();
  return [...items].sort((left, right) => scoreUseUp(right, now) - scoreUseUp(left, now));
}

function scoreUseUp(item, now) {
  let score = 0;

  if (item.is_opened) {
    score += 35;
  }

  if (item.fill_percent != null) {
    if (item.fill_percent <= 10) {
      score += 40;
    } else if (item.fill_percent <= 25) {
      score += 30;
    } else if (item.fill_percent <= 50) {
      score += 20;
    }
  }

  const remaining = normalizeName(item.remaining_quantity);
  if (remaining.includes("half") || remaining.includes("left") || remaining.includes("remaining")) {
    score += 25;
  }

  if (item.expiration_date) {
    const expiresAt = new Date(item.expiration_date).getTime();
    const daysUntilExpiry = Math.round((expiresAt - now) / (1000 * 60 * 60 * 24));
    if (daysUntilExpiry <= 3) {
      score += 50;
    } else if (daysUntilExpiry <= 7) {
      score += 30;
    }
  }

  if (item.created_at) {
    const ageDays = Math.round((now - new Date(item.created_at).getTime()) / (1000 * 60 * 60 * 24));
    score += Math.min(Math.max(ageDays, 0), 15);
  }

  return score;
}

export async function getKitchenItems(context, options = {}) {
  return withDbConnection(async (connection) => {
    const rows = await getKitchenRows(connection, resolveTableOwnerId(context), options.limit);
    const items = rows.map(mapKitchenRow);
    console.log("[DEBUG] kitchen items loaded:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: items.length
    }));
    return items;
  }, options);
}

export async function getKitchenOverview(context, options = {}) {
  const items = await getKitchenItems(context, options);
  const householdSummary = await buildHouseholdSummary(context, options);
  const summary = {
    household_size: householdSummary.household_size || context?.householdSize || 0,
    kitchen_count: householdSummary.kitchen_count || items.length,
    shopping_count: householdSummary.shopping_count || 0,
    discard_count: householdSummary.discard_count || 0,
    recent_dish_count: householdSummary.recent_dish_count || 0,
    latest_metrics: householdSummary.latest_metrics || null
  };

  console.log("[DEBUG] kitchen overview built:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    count: items.length,
    summary
  }));

  return {
    items,
    count: items.length,
    summary
  };
}

export async function getKitchenItemById(context, itemId, options = {}) {
  return withDbConnection(async (connection) => {
    const row = await findKitchenRowById(connection, resolveTableOwnerId(context), itemId);
    return row ? mapKitchenRow(row) : null;
  }, options);
}

export async function searchKitchenItem(context, itemReference, options = {}) {
  return withDbConnection(async (connection) => {
    const row = await resolveKitchenRowByReference(connection, context, itemReference);
    const item = row ? mapKitchenRow(row) : null;
    console.log("[DEBUG] kitchen item searched:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: normalizeKitchenReference(itemReference).item_name,
      found: Boolean(item)
    }));
    return {
      found: Boolean(item),
      item
    };
  }, options);
}

export async function setKitchenItemProductImage(context, itemId, fields = {}, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const row = await findKitchenRowById(connection, resolveTableOwnerId(context), itemId);
    if (!row) {
      throw createKitchenNotFoundError(itemId);
    }

    const updates = {};
    if (fields.product_image_url !== undefined) {
      updates.product_image_url = fields.product_image_url || null;
    }
    if (fields.product_image_key !== undefined) {
      updates.product_image_key = fields.product_image_key || null;
    }
    if (Object.keys(updates).length === 0) {
      return mapKitchenRow(row);
    }

    await updateKitchenRowAcrossHousehold(connection, context, row._id, updates);
    const updatedRow = await findKitchenRowById(connection, resolveTableOwnerId(context), itemId);
    console.log("[DEBUG] kitchen product image updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: itemId,
      product_image_url: updates.product_image_url || null
    }));
    return mapKitchenRow(updatedRow || { ...row, ...updates });
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function checkInKitchenItem(context, payload, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const parsedExpirationDate = payload?.expiration_date
      ? parseKitchenExpirationDate(payload.expiration_date)
      : null;
    const resolvedPayload = {
      ...payload,
      item_name: formatUserFacingItemName(payload?.item_name),
      ...(payload?.expiration_date ? { expiration_date: parsedExpirationDate } : {})
    };
    const rowId = await insertKitchenRowAcrossHousehold(connection, context, resolvedPayload);
    const insertedRow = await findKitchenRowById(connection, resolveTableOwnerId(context), rowId);
    const result = {
      id: rowId,
      item: insertedRow ? mapKitchenRow(insertedRow) : null
    };
    console.log("[DEBUG] kitchen item checked in:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: resolvedPayload.item_name,
      row_id: rowId,
      location: resolvedPayload.location || null
    }));
    return result;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function checkInManyKitchenItems(context, items, options = {}) {
  const results = [];
  for (const item of items || []) {
    results.push(await checkInKitchenItem(context, item, {
      ...options,
      skipKitchenDependentGeneration: true
    }));
  }

  console.log("[DEBUG] kitchen items checked in batch:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    count: results.length,
    item_names: results.map((result) => result.item?.item_name || null).filter(Boolean)
  }));

  if (results.length > 0) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return {
    items: results,
    count: results.length
  };
}

export async function markKitchenItemOpened(context, itemReference, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);
    const row = await resolveKitchenRowByReference(connection, context, resolvedReference);
    if (!row) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    await updateKitchenRowAcrossHousehold(connection, context, row._id, { is_opened: 1 });
    const item = mapKitchenRow({
      ...row,
      is_opened: 1
    });
    console.log("[DEBUG] kitchen item opened:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: item.item_name,
      row_id: row._id
    }));
    return item;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function updateKitchenItemQuantity(context, itemReference, quantityInput, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);

    const structuredState = typeof quantityInput === "object" && quantityInput !== null
      ? parseStructuredState(quantityInput)
      : parseStructuredState({ remaining_quantity: quantityInput });

    // Setting an item to zero is a removal: on Halo an ambiguous match resolves to every
    // matching row so we zero all of them, mirroring discard. Non-zero edits stay single-row
    // (newest on Halo). App/non-halo throws on a multi-match as before to ask for clarification.
    let rows;
    if (isZeroQuantityTarget(quantityInput)) {
      rows = await resolveKitchenRowsByReference(connection, context, resolvedReference);
    } else {
      const single = await resolveKitchenRowByReference(connection, context, resolvedReference);
      rows = single ? [single] : [];
    }
    if (rows.length === 0) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    const updates = {
      remaining_quantity: structuredState.remaining_quantity,
      quantity_value: structuredState.quantity_value,
      quantity_unit: structuredState.quantity_unit,
      fill_percent: structuredState.fill_percent
    };
    const updatedItems = [];
    for (const row of rows) {
      await updateKitchenRowAcrossHousehold(connection, context, row._id, updates);
      updatedItems.push({ row_id: row._id, item: mapKitchenRow({ ...row, ...updates }) });
    }

    const primaryItem = updatedItems[0].item;
    console.log("[DEBUG] kitchen item quantity updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: primaryItem.item_name,
      affected_count: updatedItems.length,
      row_ids: updatedItems.map((entry) => entry.row_id),
      remaining_quantity: primaryItem.remaining_quantity,
      fill_percent: primaryItem.fill_percent
    }));
    return updatedItems.length > 1
      ? { ...primaryItem, affected_count: updatedItems.length }
      : primaryItem;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function updateKitchenItemExpiration(context, itemReference, expirationInput, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);
    const row = await resolveKitchenRowByReference(connection, context, resolvedReference);
    if (!row) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    const expirationDate = parseKitchenExpirationDate(expirationInput);
    if (!expirationDate) {
      const error = new Error("I couldn't understand that expiration date. What date should I use?");
      error.statusCode = 400;
      throw error;
    }

    await updateKitchenRowAcrossHousehold(connection, context, row._id, {
      product_expiration: expirationDate
    });

    const item = mapKitchenRow({
      ...row,
      product_expiration: expirationDate
    });
    console.log("[DEBUG] kitchen item expiration updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: item.item_name,
      row_id: row._id,
      expiration_date: expirationDate
    }));
    return item;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function updateKitchenItemLocation(context, itemReference, location, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);
    const row = await resolveKitchenRowByReference(connection, context, resolvedReference);
    if (!row) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    await updateKitchenRowAcrossHousehold(connection, context, row._id, {
      storage_location: location
    });

    const item = mapKitchenRow({
      ...row,
      storage_location: location
    });
    console.log("[DEBUG] kitchen item location updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: item.item_name,
      row_id: row._id,
      storage_location: location
    }));
    return item;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function updateKitchenItemDetails(context, itemReference, updates, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);
    const row = await resolveKitchenRowByReference(connection, context, resolvedReference);
    if (!row) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    const kitchenFields = {};
    const nextRow = { ...row };

    if (
      updates?.remaining_quantity
      || updates?.quantity
      || updates?.quantity_value != null
      || updates?.quantity_unit
      || updates?.fill_percent != null
    ) {
      const structuredState = parseStructuredState(updates);
      kitchenFields.remaining_quantity = structuredState.remaining_quantity;
      kitchenFields.quantity_value = structuredState.quantity_value;
      kitchenFields.quantity_unit = structuredState.quantity_unit;
      kitchenFields.fill_percent = structuredState.fill_percent;
      nextRow.remaining_quantity = structuredState.remaining_quantity;
      nextRow.quantity_value = structuredState.quantity_value;
      nextRow.quantity_unit = structuredState.quantity_unit;
      nextRow.fill_percent = structuredState.fill_percent;
    }

    if (updates?.location) {
      kitchenFields.storage_location = updates.location;
      nextRow.storage_location = updates.location;
    }
    if (updates?.expiration_date) {
      const expirationDate = parseKitchenExpirationDate(updates.expiration_date);
      if (!expirationDate) {
        const error = new Error("I couldn't understand that expiration date. What date should I use?");
        error.statusCode = 400;
        throw error;
      }
      kitchenFields.product_expiration = expirationDate;
      nextRow.product_expiration = expirationDate;
    }
    if (typeof updates?.is_opened === "boolean") {
      kitchenFields.is_opened = updates.is_opened ? 1 : 0;
      nextRow.is_opened = updates.is_opened ? 1 : 0;
    }
    if (updates?.brand) {
      kitchenFields.brand = updates.brand;
      nextRow.brand = updates.brand;
    }
    if (updates?.category) {
      kitchenFields.category = updates.category;
      nextRow.category = updates.category;
    }

    await updateKitchenRowAcrossHousehold(connection, context, row._id, kitchenFields);
    const item = mapKitchenRow(nextRow);
    console.log("[DEBUG] kitchen item details updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: item.item_name,
      row_id: row._id,
      updated_fields: Object.keys(kitchenFields)
    }));
    return item;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function discardKitchenItem(context, itemReference, reason, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const resolvedReference = normalizeKitchenReference(itemReference);
    const rows = await resolveKitchenRowsByReference(connection, context, resolvedReference);
    if (rows.length === 0) {
      throw createKitchenNotFoundError(resolvedReference.item_name || "that kitchen item");
    }

    // On Halo an ambiguous removal resolves to every matching row — discard all of them.
    // App/non-halo never reaches here with multiple rows (it throws to ask for clarification).
    const discarded = [];
    for (const row of rows) {
      await updateKitchenRowAcrossHousehold(connection, context, row._id, { action: "OUT" });
      const discardId = await insertDiscardRowAcrossHousehold(connection, context, row, reason);
      discarded.push({ row_id: row._id, discard_id: discardId, item: mapKitchenRow({ ...row, action: "OUT" }) });
    }

    const primary = discarded[0];
    const result = {
      discard_id: primary.discard_id,
      item: primary.item,
      ...(discarded.length > 1
        ? { affected_count: discarded.length, affected_items: discarded.map((entry) => entry.item) }
        : {})
    };
    console.log("[DEBUG] kitchen item discarded:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: primary.item.item_name,
      affected_count: discarded.length,
      row_ids: discarded.map((entry) => entry.row_id),
      discard_id: primary.discard_id,
      reason: reason || null
    }));
    return result;
  }, options);

  if (!options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function clearKitchenInventory(context, options = {}) {
  const result = await withDbConnection(async (connection) => {
    const ownerId = resolveTableOwnerId(context);

    if (WRITE_SHARED_ONLY) {
      const [rows] = await connection.execute(
        `SELECT * FROM \`${SHARED_KITCHEN_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' ORDER BY COALESCE(_updatedDate, _createdDate) DESC`,
        [ownerId]
      );
      const activeRows = rows || [];
      const items = activeRows.map(mapKitchenRow);
      if (activeRows.length > 0) {
        await connection.execute(
          `UPDATE \`${SHARED_KITCHEN_TABLE}\` SET action = 'OUT', _updatedDate = NOW() WHERE \`owner_id\` = ? AND action = 'IN'`,
          [ownerId]
        );
      }
      console.log("[DEBUG] kitchen inventory cleared (shared):", JSON.stringify({
        ownerId: context?.ownerId || null,
        userId: context?.userId || null,
        count: activeRows.length,
        row_ids: activeRows.slice(0, 25).map((row) => row._id)
      }));
      return { items, count: activeRows.length };
    }

    const tableName = kitchenTableName(ownerId);
    if (!(await tableExists(connection, tableName))) {
      return {
        items: [],
        count: 0
      };
    }

    await ensureKitchenVoiceColumns(connection, tableName);
    const [rows] = await connection.execute(
      `SELECT *
       FROM \`${tableName}\`
       WHERE action = 'IN'
       ORDER BY COALESCE(_updatedDate, _createdDate) DESC`
    );
    const activeRows = rows || [];
    const items = activeRows.map(mapKitchenRow);

    logHouseholdWriteTargets("kitchen inventory clear propagation", context, getTableHouseholdMemberIds(context), {
      mode: "direct",
      count: activeRows.length
    });

    for (const row of activeRows) {
      await updateKitchenRowAcrossHousehold(connection, context, row._id, { action: "OUT" });
    }

    console.log("[DEBUG] kitchen inventory cleared:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: activeRows.length,
      row_ids: activeRows.slice(0, 25).map((row) => row._id)
    }));

    return {
      items,
      count: activeRows.length
    };
  }, options);

  if (result.count > 0 && !options?.skipKitchenDependentGeneration) {
    await triggerKitchenDependentGeneration(context, options);
  }

  return result;
}

export async function getRecentDiscards(context, options = {}) {
  return withDbConnection(async (connection) => {
    if (WRITE_SHARED_ONLY) {
      const ownerId = resolveTableOwnerId(context);
      const safeLimit = clampLimit(options.limit, 5, 20);
      const [rows] = await connection.execute(
        `SELECT * FROM \`${SHARED_DISCARDS_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' ORDER BY _createdDate DESC LIMIT ${safeLimit}`,
        [ownerId]
      );
      return (rows || []).map(mapDiscardRow);
    }

    const tableName = discardTableName(resolveTableOwnerId(context));
    if (!(await tableExists(connection, tableName))) {
      return [];
    }

    await ensureDiscardColumns(connection, tableName);
    const safeLimit = clampLimit(options.limit, 5, 20);
    const [rows] = await connection.execute(
      `SELECT *
       FROM \`${tableName}\`
       WHERE action = 'IN'
       ORDER BY _createdDate DESC
       LIMIT ${safeLimit}`
    );

    return (rows || []).map(mapDiscardRow);
  }, options);
}

function summarizeDiscardRowForDisambiguation(row) {
  const parts = [row.product_name || "Unknown item"];
  if (row.discard_reason) {
    parts.push(row.discard_reason);
  }
  if (row._createdDate) {
    parts.push(`discarded ${serializeDate(row._createdDate)}`);
  }
  return parts.join(" - ");
}

function createDiscardAmbiguityError(itemName, rows) {
  const error = new Error(`I found multiple recent discards matching ${itemName}. Which one did you mean?`);
  error.statusCode = 409;
  error.details = {
    type: "ambiguous_discard_item",
    candidates: (rows || []).slice(0, 5).map((row) => summarizeDiscardRowForDisambiguation(row))
  };
  return error;
}

function createDiscardNotFoundError(itemName) {
  const error = new Error(`Couldn't find ${itemName} in recent discards.`);
  error.statusCode = 404;
  return error;
}

async function findDiscardRowsByName(connection, context, itemName, limit = 5) {
  if (WRITE_SHARED_ONLY) {
    const ownerId = resolveTableOwnerId(context);
    const exactName = normalizeName(itemName);
    const [exactRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISCARDS_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' AND LOWER(TRIM(product_name)) = ? ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${limit}`,
      [ownerId, exactName]
    );
    if (exactRows?.length > 0) {
      return exactRows;
    }
    const [likeRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISCARDS_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' AND LOWER(product_name) LIKE ? ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${limit}`,
      [ownerId, buildLikePattern(itemName)]
    );
    return likeRows || [];
  }

  const tableName = discardTableName(resolveTableOwnerId(context));
  if (!(await tableExists(connection, tableName))) {
    return [];
  }

  await ensureDiscardColumns(connection, tableName);
  const exactName = normalizeName(itemName);
  const [exactRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE action = 'IN'
       AND LOWER(TRIM(product_name)) = ?
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${limit}`,
    [exactName]
  );

  if (exactRows?.length > 0) {
    return exactRows;
  }

  const [likeRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE action = 'IN'
       AND LOWER(product_name) LIKE ?
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${limit}`,
    [buildLikePattern(itemName)]
  );

  return likeRows || [];
}

async function resolveDiscardRowByName(connection, context, itemName) {
  const rows = await findDiscardRowsByName(connection, context, itemName, 5);
  console.log("[DEBUG] discard item resolution:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    item_name: itemName,
    matchCount: rows.length,
    candidates: rows.slice(0, 5).map((row) => summarizeDiscardRowForDisambiguation(row))
  }));

  if (rows.length === 0) {
    return null;
  }
  if (rows.length > 1) {
    throw createDiscardAmbiguityError(itemName, rows);
  }

  return rows[0];
}

async function updateDiscardRowAcrossHousehold(connection, context, rowId, fields) {
  if (WRITE_SHARED_ONLY) {
    const assignments = [];
    const values = [];
    for (const [key, value] of Object.entries(fields)) {
      assignments.push(`\`${key}\` = ?`);
      values.push(value);
    }
    assignments.push("`_updatedDate` = NOW()");
    values.push(rowId);
    await connection.execute(
      `UPDATE \`${SHARED_DISCARDS_TABLE}\` SET ${assignments.join(", ")} WHERE _id = ?`,
      values
    );

    // ALSO apply to every household member's {member}_discards table — the app's
    // Discard log reads ONLY those. Household-visible → fan out to all members;
    // tolerate missing table/row (pre-fix rows live only in shared). Covers
    // deleteRecentDiscard, which routes through here with action=OUT.
    await dualWriteAcrossHousehold(connection, {
      memberIds: getTableHouseholdMemberIds(context), context, op: "update", reportEvt: "discard_peruser_write_miss",
      tableFn: discardTableName, ensureFn: ensureDiscardColumns,
      buildStatement: (tableName) => ({
        sql: `UPDATE \`${tableName}\` SET ${assignments.join(", ")} WHERE _id = ?`,
        values,
      }),
    });
    return;
  }

  const memberIds = getTableHouseholdMemberIds(context);
  const assignments = [];
  const values = [];

  for (const [key, value] of Object.entries(fields)) {
    assignments.push(`\`${key}\` = ?`);
    values.push(value);
  }

  assignments.push("`_updatedDate` = NOW()");
  values.push(rowId);

  logHouseholdWriteTargets("discard row propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId,
    fields: Object.keys(fields)
  });

  for (const memberId of memberIds) {
    const tableName = discardTableName(memberId);
    if (!(await tableExists(connection, tableName))) {
      continue;
    }

    await ensureDiscardColumns(connection, tableName);
    await connection.execute(
      `UPDATE \`${tableName}\`
       SET ${assignments.join(", ")}
       WHERE _id = ?`,
      values
    );
  }
}

export async function deleteRecentDiscard(context, itemName, options = {}) {
  return withDbConnection(async (connection) => {
    const row = await resolveDiscardRowByName(connection, context, itemName);
    if (!row) {
      throw createDiscardNotFoundError(itemName);
    }

    await updateDiscardRowAcrossHousehold(connection, context, row._id, { action: "OUT" });
    const discard = mapDiscardRow({
      ...row,
      action: "OUT"
    });

    console.log("[DEBUG] discard removed from history:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: discard.item_name,
      discard_id: row._id
    }));

    return { discard };
  }, options);
}

export async function clearRecentDiscards(context, options = {}) {
  return withDbConnection(async (connection) => {
    if (WRITE_SHARED_ONLY) {
      const ownerId = resolveTableOwnerId(context);
      const [rows] = await connection.execute(
        `SELECT * FROM \`${SHARED_DISCARDS_TABLE}\` WHERE \`owner_id\` = ? AND action = 'IN' ORDER BY COALESCE(_updatedDate, _createdDate) DESC`,
        [ownerId]
      );
      const activeRows = rows || [];
      const items = activeRows.map(mapDiscardRow);
      if (activeRows.length > 0) {
        await connection.execute(
          `UPDATE \`${SHARED_DISCARDS_TABLE}\` SET action = 'OUT', _updatedDate = NOW() WHERE \`owner_id\` = ? AND action = 'IN'`,
          [ownerId]
        );
        // ALSO clear every household member's {member}_discards — the app's Discard
        // log reads ONLY those. Household-visible → fan out; tolerate missing table.
        await dualWriteAcrossHousehold(connection, {
          memberIds: getTableHouseholdMemberIds(context), context, op: "update", reportEvt: "discard_peruser_write_miss",
          tableFn: discardTableName, ensureFn: ensureDiscardColumns,
          buildStatement: (tableName) => ({
            sql: `UPDATE \`${tableName}\` SET action = 'OUT', _updatedDate = NOW() WHERE action = 'IN'`,
            values: [],
          }),
        });
      }
      console.log("[DEBUG] discard history cleared (shared):", JSON.stringify({
        ownerId: context?.ownerId || null,
        userId: context?.userId || null,
        count: activeRows.length
      }));
      return { items, count: activeRows.length };
    }

    const tableName = discardTableName(resolveTableOwnerId(context));
    if (!(await tableExists(connection, tableName))) {
      return {
        items: [],
        count: 0
      };
    }

    await ensureDiscardColumns(connection, tableName);
    const [rows] = await connection.execute(
      `SELECT *
       FROM \`${tableName}\`
       WHERE action = 'IN'
       ORDER BY COALESCE(_updatedDate, _createdDate) DESC`
    );
    const activeRows = rows || [];
    const items = activeRows.map(mapDiscardRow);
    const memberIds = getTableHouseholdMemberIds(context);

    logHouseholdWriteTargets("discard history clear propagation", context, memberIds, {
      mode: "direct",
      count: activeRows.length
    });

    for (const row of activeRows) {
      await updateDiscardRowAcrossHousehold(connection, context, row._id, { action: "OUT" });
    }

    console.log("[DEBUG] discard history cleared:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: activeRows.length
    }));

    return {
      items,
      count: activeRows.length
    };
  }, options);
}

async function getDishRows(connection, context, options = {}) {
  if (WRITE_SHARED_ONLY) {
    const ownerId = resolveTableOwnerId(context);
    const safeLimit = clampLimit(options.limit, 5, 50);
    const [rows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISHES_TABLE}\` WHERE \`owner_id\` = ? ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${safeLimit}`,
      [ownerId]
    );
    return rows || [];
  }

  const tableName = dishTableName(resolveTableOwnerId(context));
  if (!(await tableExists(connection, tableName))) {
    return [];
  }

  await ensureDishVoiceColumns(connection, tableName);
  const safeLimit = clampLimit(options.limit, 5, 50);
  const [rows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${safeLimit}`
  );

  return rows || [];
}

async function findDishRowById(connection, context, dishId) {
  if (WRITE_SHARED_ONLY) {
    const [rows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISHES_TABLE}\` WHERE \`_id\` = ? LIMIT 1`,
      [dishId]
    );
    return rows?.[0] || null;
  }

  const tableName = dishTableName(resolveTableOwnerId(context));
  if (!(await tableExists(connection, tableName))) {
    return null;
  }

  await ensureDishVoiceColumns(connection, tableName);
  const [rows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE _id = ?
     LIMIT 1`,
    [dishId]
  );

  return rows?.[0] || null;
}

async function findDishRowsByName(connection, context, dishName, options = {}) {
  if (WRITE_SHARED_ONLY) {
    const ownerId = resolveTableOwnerId(context);
    const safeLimit = clampLimit(options.limit, 5, 10);
    const activeOnlyClause = options.activeOnly ? "AND action = 'IN'" : "";
    const exactName = normalizeName(dishName);
    const [exactRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISHES_TABLE}\` WHERE \`owner_id\` = ? AND LOWER(TRIM(dish_name)) = ? ${activeOnlyClause} ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${safeLimit}`,
      [ownerId, exactName]
    );
    if (exactRows?.length > 0) {
      return exactRows;
    }
    const [likeRows] = await connection.execute(
      `SELECT * FROM \`${SHARED_DISHES_TABLE}\` WHERE \`owner_id\` = ? AND LOWER(dish_name) LIKE ? ${activeOnlyClause} ORDER BY COALESCE(_updatedDate, _createdDate) DESC LIMIT ${safeLimit}`,
      [ownerId, buildLikePattern(dishName)]
    );
    return likeRows || [];
  }

  const tableName = dishTableName(resolveTableOwnerId(context));
  if (!(await tableExists(connection, tableName))) {
    return [];
  }

  await ensureDishVoiceColumns(connection, tableName);
  const safeLimit = clampLimit(options.limit, 5, 10);
  const activeOnlyClause = options.activeOnly ? "AND action = 'IN'" : "";
  const exactName = normalizeName(dishName);
  const [exactRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE LOWER(TRIM(dish_name)) = ?
       ${activeOnlyClause}
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${safeLimit}`,
    [exactName]
  );

  if (exactRows?.length > 0) {
    return exactRows;
  }

  const [likeRows] = await connection.execute(
    `SELECT *
     FROM \`${tableName}\`
     WHERE LOWER(dish_name) LIKE ?
       ${activeOnlyClause}
     ORDER BY COALESCE(_updatedDate, _createdDate) DESC
     LIMIT ${safeLimit}`,
    [buildLikePattern(dishName)]
  );

  return likeRows || [];
}

async function resolveDishByName(connection, context, dishName, options = {}) {
  const rows = await findDishRowsByName(connection, context, dishName, options);
  console.log("[DEBUG] dish name resolution:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    dish_name: dishName,
    matchCount: rows.length,
    candidates: rows.slice(0, 5).map((row) => ({
      id: row._id,
      dish_name: row.dish_name || "Unknown dish",
      created_at: serializeDate(row._createdDate),
      action: row.action || "IN"
    }))
  }));

  if (rows.length === 0) {
    return null;
  }
  if (rows.length === 1) {
    return rows[0];
  }

  const newestAt = new Date(rows[0]._createdDate || rows[0]._updatedDate || 0).getTime();
  const secondAt = new Date(rows[1]._createdDate || rows[1]._updatedDate || 0).getTime();
  if (Math.abs(newestAt - secondAt) <= 4 * 60 * 60 * 1000) {
    throw createDishAmbiguityError(dishName, rows);
  }

  return rows[0];
}

async function resolveRecentDishContext(connection, context, options = {}) {
  const rows = await getDishRows(connection, context, { limit: 10 });
  const now = Date.now();
  const recencyWindowMs = Number(options.recencyWindowMs || (90 * 60 * 1000));
  const ambiguityWindowMs = Number(options.ambiguityWindowMs || (15 * 60 * 1000));
  const decisiveGapMs = Number(options.decisiveGapMs || 15 * 1000);
  const activeOnly = options.activeOnly !== false;
  const preferredRows = rows
    .filter((row) => {
      if (activeOnly && row.action === "OUT") {
        return false;
      }
      const rowTime = new Date(row._updatedDate || row._createdDate || 0).getTime();
      if (!rowTime || (now - rowTime) > recencyWindowMs) {
        return false;
      }
      return !context?.userId || !row.user_id || row.user_id === context.userId;
    });
  const fallbackRows = preferredRows.length > 0 ? preferredRows : rows.filter((row) => {
    if (activeOnly && row.action === "OUT") {
      return false;
    }
    const rowTime = new Date(row._updatedDate || row._createdDate || 0).getTime();
    return Boolean(rowTime) && (now - rowTime) <= recencyWindowMs;
  });

  console.log("[DEBUG] recent dish context resolution:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    recencyWindowMs,
    activeOnly,
    candidateCount: fallbackRows.length,
    candidates: fallbackRows.slice(0, 5).map((row) => ({
      id: row._id,
      dish_name: row.dish_name || "Unknown dish",
      user_id: row.user_id || null,
      created_at: serializeDate(row._createdDate),
      action: row.action || "IN"
    }))
  }));

  if (fallbackRows.length === 0) {
    throw createRecentDishContextError();
  }
  if (fallbackRows.length === 1) {
    return {
      row: fallbackRows[0],
      recent_context: {
        resolved_by: "recent_single_candidate",
        recency_window_minutes: Math.round(recencyWindowMs / 60000)
      }
    };
  }

  const newestTime = new Date(fallbackRows[0]._updatedDate || fallbackRows[0]._createdDate || 0).getTime();
  const secondTime = new Date(fallbackRows[1]._updatedDate || fallbackRows[1]._createdDate || 0).getTime();
  const newestGapMs = Math.abs(newestTime - secondTime);
  if (newestGapMs >= decisiveGapMs) {
    return {
      row: fallbackRows[0],
      recent_context: {
        resolved_by: "recent_immediate_followup",
        recency_window_minutes: Math.round(recencyWindowMs / 60000)
      }
    };
  }
  if (Math.abs(newestTime - secondTime) <= ambiguityWindowMs) {
    throw createDishAmbiguityError("that recent dish", fallbackRows);
  }

  return {
    row: fallbackRows[0],
    recent_context: {
      resolved_by: "recent_newest_candidate",
      recency_window_minutes: Math.round(recencyWindowMs / 60000)
    }
  };
}

async function resolveDishReference(connection, context, reference = {}, options = {}) {
  if (reference?.dish_id) {
    const row = await findDishRowById(connection, context, reference.dish_id);
    if (!row) {
      throw createDishNotFoundError(reference.dish_id);
    }
    return {
      row,
      recent_context: {
        resolved_by: "dish_id"
      }
    };
  }

  if (reference?.dish_name) {
    const row = await resolveDishByName(connection, context, reference.dish_name, options);
    if (!row) {
      throw createDishNotFoundError(reference.dish_name);
    }
    return {
      row,
      recent_context: {
        resolved_by: "dish_name"
      }
    };
  }

  return resolveRecentDishContext(connection, context, options);
}

async function updateDishRowAcrossHousehold(connection, context, rowId, fields) {
  if (WRITE_SHARED_ONLY) {
    const assignments = [];
    const values = [];
    for (const [key, value] of Object.entries(fields)) {
      assignments.push(`\`${key}\` = ?`);
      values.push(value);
    }
    assignments.push("`_updatedDate` = NOW()");
    values.push(rowId);
    await connection.execute(
      `UPDATE \`${SHARED_DISHES_TABLE}\` SET ${assignments.join(", ")} WHERE _id = ?`,
      values
    );

    // ALSO apply to the acting user's per-user dish table. The Dish Log (dishes_api)
    // reads ONLY {user}_dishes — without this, edits to a voice dish (including the
    // async image patch via setDishGeneratedImage -> here, and markDishConsumed) never
    // reach the app. User-private: acting user only. Tolerate missing table/row
    // (pre-fix rows exist only in shared).
    const perUserOwner = context.userId || resolveTableOwnerId(context);
    const userDishTable = dishTableName(perUserOwner);
    if (await tableExists(connection, userDishTable)) {
      await ensureDishVoiceColumns(connection, userDishTable);
      await connection.execute(
        `UPDATE \`${userDishTable}\` SET ${assignments.join(", ")} WHERE _id = ?`,
        values
      );
    }
    return;
  }

  const memberIds = getTableHouseholdMemberIds(context);
  const assignments = [];
  const values = [];

  for (const [key, value] of Object.entries(fields)) {
    assignments.push(`\`${key}\` = ?`);
    values.push(value);
  }

  assignments.push("`_updatedDate` = NOW()");
  values.push(rowId);

  logHouseholdWriteTargets("dish row propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId,
    fields: Object.keys(fields)
  });

  for (const memberId of memberIds) {
    const tableName = dishTableName(memberId);
    if (!(await tableExists(connection, tableName))) {
      continue;
    }
    await ensureDishVoiceColumns(connection, tableName);
    await connection.execute(
      `UPDATE \`${tableName}\`
       SET ${assignments.join(", ")}
       WHERE _id = ?`,
      values
    );
  }
}

async function deleteDishRowAcrossHousehold(connection, context, rowId) {
  if (WRITE_SHARED_ONLY) {
    await connection.execute(`DELETE FROM \`${SHARED_DISHES_TABLE}\` WHERE _id = ?`, [rowId]);

    // ALSO delete from the acting user's per-user dish table. The Dish Log (dishes_api)
    // reads ONLY {user}_dishes — without this, a voice-deleted dish stays visible in the
    // app. User-private: acting user only. Tolerate missing table/row.
    const perUserOwner = context.userId || resolveTableOwnerId(context);
    const userDishTable = dishTableName(perUserOwner);
    if (await tableExists(connection, userDishTable)) {
      await connection.execute(`DELETE FROM \`${userDishTable}\` WHERE _id = ?`, [rowId]);
    }
    return;
  }

  const memberIds = getTableHouseholdMemberIds(context);
  logHouseholdWriteTargets("dish row delete propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId
  });

  for (const memberId of memberIds) {
    const tableName = dishTableName(memberId);
    if (!(await tableExists(connection, tableName))) {
      continue;
    }
    await ensureDishVoiceColumns(connection, tableName);
    await connection.execute(`DELETE FROM \`${tableName}\` WHERE _id = ?`, [rowId]);
  }
}

async function insertDishRowAcrossHousehold(connection, context, payload) {
  const rowId = crypto.randomUUID();
  const jobId = `voice-dish-${rowId}`;
  const dishName = payload.dish_name || inferDishNameFromIngredients(payload.ingredients) || "Voice meal";
  const ingredients = normalizeIngredientList(payload.ingredients?.length ? payload.ingredients : deriveIngredientsFromDishName(dishName));
  const components = normalizeDishComponents(payload.components, { dish_name: dishName, ingredients });
  const allergens = normalizeIngredientList(payload.allergens);

  if (WRITE_SHARED_ONLY) {
    const tableOwnerId = resolveTableOwnerId(context);
    await connection.execute(
      `INSERT INTO \`${SHARED_DISHES_TABLE}\` (
        _id, _owner, _device, _createdDate, _updatedDate,
        dish_name, confidence, explanation, serving_size,
        calories, total_fat, total_carbohydrates, protein,
        ingredients, components, allergens, images, s3_key, action, dish_image_url, dish_image_key, job_id, user_id, analysis_status, analysis_error, owner_id
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?, ?)`,
      [
        rowId,
        tableOwnerId,
        "voice-assistant",
        dishName,
        payload.confidence ?? null,
        payload.explanation || "Logged from voice assistant",
        payload.serving_size || null,
        payload.calories ?? null,
        payload.total_fat ?? null,
        payload.total_carbohydrates ?? null,
        payload.protein ?? null,
        JSON.stringify(ingredients),
        JSON.stringify(components),
        JSON.stringify(allergens),
        null,
        null,
        null,
        null,
        jobId,
        context.userId || null,
        payload.analysis_status || "complete",
        payload.analysis_error || null,
        tableOwnerId
      ]
    );

    // ALSO write the acting user's per-user dish table. The Dish Log (dishes_api)
    // reads ONLY {user}_dishes (dishes are USER-PRIVATE per dishes_api app.py:203-205)
    // — without this, voice-logged dishes are invisible in the app (same early-return
    // class as the shopping dual-write bug). Target the acting user only, NOT the whole
    // household; the per-user table has NO owner_id column.
    const perUserOwner = context.userId || tableOwnerId;
    const userDishTable = dishTableName(perUserOwner);
    await ensureDishTable(connection, userDishTable);
    await connection.execute(
      `INSERT INTO \`${userDishTable}\` (
        _id, _owner, _device, _createdDate, _updatedDate,
        dish_name, confidence, explanation, serving_size,
        calories, total_fat, total_carbohydrates, protein,
        ingredients, components, allergens, images, s3_key, action, dish_image_url, dish_image_key, job_id, user_id, analysis_status, analysis_error
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?)`,
      [
        rowId,
        perUserOwner,
        "voice-assistant",
        dishName,
        payload.confidence ?? null,
        payload.explanation || "Logged from voice assistant",
        payload.serving_size || null,
        payload.calories ?? null,
        payload.total_fat ?? null,
        payload.total_carbohydrates ?? null,
        payload.protein ?? null,
        JSON.stringify(ingredients),
        JSON.stringify(components),
        JSON.stringify(allergens),
        null,
        null,
        null,
        null,
        jobId,
        context.userId || null,
        payload.analysis_status || "complete",
        payload.analysis_error || null
      ]
    );
    return rowId;
  }

  const memberIds = getTableHouseholdMemberIds(context);

  logHouseholdWriteTargets("dish row insert propagation", context, memberIds, {
    mode: "direct",
    row_id: rowId,
    dish_name: dishName
  });

  for (const memberId of memberIds) {
    const tableName = dishTableName(memberId);
    await ensureDishTable(connection, tableName);
    await connection.execute(
      `INSERT INTO \`${tableName}\` (
        _id, _owner, _device, _createdDate, _updatedDate,
        dish_name, confidence, explanation, serving_size,
        calories, total_fat, total_carbohydrates, protein,
        ingredients, components, allergens, images, s3_key, action, dish_image_url, dish_image_key, job_id, user_id, analysis_status, analysis_error
      ) VALUES (?, ?, ?, NOW(), NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IN', ?, ?, ?, ?, ?, ?)`,
      [
        rowId,
        memberId,
        "voice-assistant",
        dishName,
        payload.confidence ?? null,
        payload.explanation || "Logged from voice assistant",
        payload.serving_size || null,
        payload.calories ?? null,
        payload.total_fat ?? null,
        payload.total_carbohydrates ?? null,
        payload.protein ?? null,
        JSON.stringify(ingredients),
        JSON.stringify(components),
        JSON.stringify(allergens),
        null,
        null,
        null,
        null,
        jobId,
        context.userId || null,
        payload.analysis_status || "complete",
        payload.analysis_error || null
      ]
    );
  }

  return rowId;
}

async function buildDishUpdateFieldsFromArgs(existingRow, updates = {}, options = {}) {
  const analysis = await analyzeDishPayload(existingRow, updates, options);
  return buildDishPersistedFieldsFromAnalysis(analysis, existingRow);
}

export async function logDishFromVoice(context, payload, options = {}) {
  return withDbConnection(async (connection) => {
    if (shouldUseAsyncDishEnrichment(options?.env || process.env)) {
      const pending = buildPendingDishFields(null, payload);
      const rowId = await insertDishRowAcrossHousehold(connection, context, {
        ...pending.persisted_fields,
        ingredients: pending.async_payload.ingredients,
        allergens: []
      });
      let async_update;
      try {
        async_update = await queueAsyncDishEnrichment(context, rowId, pending.async_payload, options);
      } catch (error) {
        console.warn("[WARN] async dish enrichment dispatch failed; falling back to sync:", error?.message || error);
        async_update = await fallbackToSynchronousDishEnrichment(connection, context, rowId, null, payload, options);
      }
      const insertedRow = await findDishRowById(connection, context, rowId);
      const dish = insertedRow ? mapDishRow(insertedRow) : { id: rowId, dish_name: pending.persisted_fields.dish_name || "Voice meal", action: "IN", analysis_status: "pending" };
      return {
        message: "Dish logged successfully.",
        dish,
        recent_context: {
          resolved_by: "new_voice_dish"
        },
        async_update
      };
    }

    const analyzedPayload = await analyzeDishPayload(null, payload, { ...options, context });
    const rowId = await insertDishRowAcrossHousehold(connection, context, analyzedPayload);
    const insertedRow = await findDishRowById(connection, context, rowId);
    const dish = insertedRow ? mapDishRow(insertedRow) : { id: rowId, dish_name: analyzedPayload.dish_name || "Voice meal", action: "IN" };
    console.log("[DEBUG] dish logged from voice:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: rowId,
      dish_name: dish.dish_name,
      ingredients: dish.ingredients?.length || 0,
      calories: dish.calories ?? null,
      protein: dish.protein ?? null,
      carbs: dish.total_carbohydrates ?? null,
      fat: dish.total_fat ?? null
    }));
    return {
      message: "Dish logged successfully.",
      dish,
      recent_context: {
        resolved_by: "new_voice_dish"
      }
    };
  }, options);
}

export async function logDishIngredients(context, payload, options = {}) {
  const resolvedPayload = {
    ...payload,
    dish_name: payload.dish_name || inferDishNameFromIngredients(payload.ingredients)
  };
  return logDishFromVoice(context, resolvedPayload, options);
}

export async function appendToRecentDish(context, payload, options = {}) {
  return withDbConnection(async (connection) => {
    const { row, recent_context } = await resolveRecentDishContext(connection, context, { activeOnly: true });
    if (shouldUseAsyncDishEnrichment(options?.env || process.env)) {
      const updates = {
        add_ingredients: payload.ingredients,
        serving_size: payload.serving_size
      };
      const pending = buildPendingDishFields(row, updates);
      await updateDishRowAcrossHousehold(connection, context, row._id, pending.persisted_fields);
      let async_update;
      try {
        async_update = await queueAsyncDishEnrichment(context, row._id, pending.async_payload, options);
      } catch (error) {
        console.warn("[WARN] async dish enrichment dispatch failed; falling back to sync:", error?.message || error);
        async_update = await fallbackToSynchronousDishEnrichment(connection, context, row._id, row, updates, options);
      }
      const updatedRow = await findDishRowById(connection, context, row._id);
      return {
        message: "Recent dish updated.",
        dish: mapDishRow(updatedRow || { ...row, ...pending.persisted_fields, ingredients: pending.persisted_fields.ingredients }),
        recent_context,
        async_update
      };
    }

    const updateFields = await buildDishUpdateFieldsFromArgs(row, {
      add_ingredients: payload.ingredients,
      serving_size: payload.serving_size
    }, { ...options, context });
    await updateDishRowAcrossHousehold(connection, context, row._id, updateFields);
    const updatedRow = await findDishRowById(connection, context, row._id);
    return {
      message: "Recent dish updated.",
      dish: mapDishRow(updatedRow || { ...row, ...updateFields, ingredients: updateFields.ingredients }),
      recent_context
    };
  }, options);
}

export async function updateRecentDish(context, payload, options = {}) {
  return withDbConnection(async (connection) => {
    const { row, recent_context } = await resolveRecentDishContext(connection, context, { activeOnly: false });
    if (shouldUseAsyncDishEnrichment(options?.env || process.env)) {
      const pending = buildPendingDishFields(row, payload);
      await updateDishRowAcrossHousehold(connection, context, row._id, pending.persisted_fields);
      let async_update;
      try {
        async_update = await queueAsyncDishEnrichment(context, row._id, pending.async_payload, options);
      } catch (error) {
        console.warn("[WARN] async dish enrichment dispatch failed; falling back to sync:", error?.message || error);
        async_update = await fallbackToSynchronousDishEnrichment(connection, context, row._id, row, payload, options);
      }
      const updatedRow = await findDishRowById(connection, context, row._id);
      return {
        message: "Recent dish updated.",
        dish: mapDishRow(updatedRow || { ...row, ...pending.persisted_fields, ingredients: pending.persisted_fields.ingredients }),
        recent_context,
        async_update
      };
    }

    const updateFields = await buildDishUpdateFieldsFromArgs(row, payload, { ...options, context });
    await updateDishRowAcrossHousehold(connection, context, row._id, updateFields);
    const updatedRow = await findDishRowById(connection, context, row._id);
    console.log("[DEBUG] recent dish updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: row._id,
      recent_context
    }));
    return {
      message: "Recent dish updated.",
      dish: mapDishRow(updatedRow || { ...row, ...updateFields, ingredients: updateFields.ingredients }),
      recent_context
    };
  }, options);
}

export async function getRecentDishes(context, options = {}) {
  return withDbConnection(async (connection) => {
    const rows = await getDishRows(connection, context, options);
    const items = rows.map(mapDishRow);
    console.log("[DEBUG] recent dishes loaded:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: items.length
    }));
    return {
      items,
      count: items.length
    };
  }, options);
}

export async function getDishDetail(context, reference = {}, options = {}) {
  return withDbConnection(async (connection) => {
    const { row, recent_context } = await resolveDishReference(connection, context, reference, { activeOnly: false });
    const dish = mapDishRow(row);
    console.log("[DEBUG] dish detail loaded:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: dish.id,
      dish_name: dish.dish_name,
      recent_context
    }));
    return {
      found: true,
      dish,
      recent_context
    };
  }, options);
}

export async function getDishById(context, dishId, options = {}) {
  return withDbConnection(async (connection) => {
    const row = await findDishRowById(connection, context, dishId);
    return row ? mapDishRow(row) : null;
  }, options);
}

export async function setDishGeneratedImage(context, dishId, fields = {}, options = {}) {
  return withDbConnection(async (connection) => {
    const row = await findDishRowById(connection, context, dishId);
    if (!row) {
      throw createDishNotFoundError(dishId);
    }

    const updates = {};
    if (fields.dish_image_url !== undefined) {
      updates.dish_image_url = fields.dish_image_url || null;
    }
    if (fields.dish_image_key !== undefined) {
      updates.dish_image_key = fields.dish_image_key || null;
    }
    if (Object.keys(updates).length === 0) {
      return mapDishRow(row);
    }

    await updateDishRowAcrossHousehold(connection, context, dishId, updates);
    const updatedRow = await findDishRowById(connection, context, dishId);
    console.log("[DEBUG] dish image updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: dishId,
      dish_image_url: updates.dish_image_url || null
    }));
    return mapDishRow(updatedRow || { ...row, ...updates });
  }, options);
}

export async function markDishConsumed(context, reference = {}, options = {}) {
  return withDbConnection(async (connection) => {
    const { row, recent_context } = await resolveDishReference(connection, context, reference, { activeOnly: true });
    await updateDishRowAcrossHousehold(connection, context, row._id, { action: "OUT" });
    const updatedRow = await findDishRowById(connection, context, row._id);
    const dish = mapDishRow(updatedRow || { ...row, action: "OUT" });
    console.log("[DEBUG] dish marked consumed:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: row._id,
      dish_name: dish.dish_name,
      recent_context
    }));
    return {
      message: "Dish marked consumed.",
      dish,
      recent_context
    };
  }, options);
}

export async function deleteDishLog(context, reference = {}, options = {}) {
  return withDbConnection(async (connection) => {
    const { row, recent_context } = await resolveDishReference(connection, context, reference, { activeOnly: false });
    const dish = mapDishRow(row);
    await deleteDishRowAcrossHousehold(connection, context, row._id);
    console.log("[DEBUG] dish log deleted:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      row_id: row._id,
      dish_name: dish.dish_name,
      recent_context
    }));
    return {
      message: "Dish deleted.",
      dish,
      recent_context
    };
  }, options);
}

export async function addDishIngredientsToShoppingList(context, reference = {}, options = {}) {
  let ingredients = normalizeIngredientList(reference.ingredients);
  let recent_context = null;
  let dish = null;

  if (ingredients.length === 0) {
    const detail = await getDishDetail(context, reference, options);
    recent_context = detail.recent_context || null;
    dish = detail.dish || null;
    ingredients = normalizeIngredientList(dish?.ingredients || []);
  }

  if (ingredients.length === 0) {
    const error = new Error("I couldn't find any dish ingredients to add to the shopping list.");
    error.statusCode = 400;
    throw error;
  }

  const added = await addManyShoppingItems(
    context,
    ingredients.map((ingredient) => ({
      item_name: ingredient,
      store: reference.store || null
    })),
    options
  );

  console.log("[DEBUG] dish ingredients added to shopping list:", JSON.stringify({
    ownerId: context?.ownerId || null,
    userId: context?.userId || null,
    count: added.count,
    dish_name: dish?.dish_name || reference.dish_name || null,
    recent_context
  }));

  return {
    ...added,
    dish,
    recent_context
  };
}

export async function getShoppingItems(context, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const items = await hydrateShoppingRows(connection, context, shoppingRows);
    const limitedItems = items.slice(0, clampLimit(options.limit, 25, 100));
    console.log("[DEBUG] shopping list loaded:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: limitedItems.length
    }));
    return limitedItems;
  }, options);
}

// Fan a per-member write across a household (or the single acting user for
// user-private domains) on ONE connection. Owns the loop mechanics: ensure-table on
// insert / tableExists-tolerate on update+delete, execute the caller-built statement,
// and — generalizing the batch-shopping silent-miss lesson to EVERY site — verify an
// INSERT actually landed, emitting a self-report event if it didn't. buildStatement
// returns { sql, values } (or null to skip a member); onResult(memberId, result) lets
// a caller capture insertId / affectedRows.
async function dualWriteAcrossHousehold(connection, { memberIds, tableFn, ensureFn = null, buildStatement, op = "insert", reportEvt = "household_write_miss", context = null, onResult = null }) {
  for (const memberId of memberIds) {
    const tableName = tableFn(memberId);
    if (op === "insert") {
      if (ensureFn) await ensureFn(connection, tableName);
    } else {
      if (!(await tableExists(connection, tableName))) continue; // update/delete tolerate a missing member table
      if (ensureFn) await ensureFn(connection, tableName);
    }
    const built = await buildStatement(tableName, memberId, connection);
    if (!built) continue;
    const [result] = await connection.execute(built.sql, built.values || []);
    if (op === "insert" && (result?.affectedRows ?? 0) < 1) {
      console.warn(JSON.stringify({ evt: reportEvt, op, table: tableName, member: memberId, owner: context?.userId || context?.ownerId || null }));
    }
    if (onResult) onResult(memberId, result);
  }
}

export async function addShoppingItem(context, itemName, requestedStore = null, quantity = null, options = {}) {
  return withDbConnection(async (connection) => {
    const householdItemUuid = crypto.randomUUID();
    const shoppingOwnerId = resolveShoppingOwnerId(context);
    const store = await resolveShoppingStore(connection, context, requestedStore);
    const displayItemName = formatUserFacingItemName(itemName);

    if (WRITE_SHARED_ONLY) {
      const [result] = await connection.execute(
        `INSERT INTO \`${SHARED_SHOPPING_TABLE}\` (
          _owner, _device, product_name, quantity, product_brand,
          images, product_barcode, store, action,
          _createdDate, created_at, updated_at, household_item_uuid, owner_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), NOW(), ?, ?)`,
        [shoppingOwnerId, "voice-assistant", displayItemName, quantity || null, null, null, null, store, "ADDED", householdItemUuid, shoppingOwnerId]
      );
      const primaryInsertId = result?.insertId != null ? String(result.insertId) : null;
      const shoppingResult = {
        shopping_id: primaryInsertId,
        household_item_uuid: householdItemUuid,
        item_name: displayItemName,
        quantity: quantity || null,
        store,
        source: "new_list",
        action: "ADDED"
      };
      console.log("[DEBUG] shopping item added (shared):", JSON.stringify({
        ownerId: context?.ownerId || null,
        userId: context?.userId || null,
        item_name: displayItemName,
        store,
        household_item_uuid: householdItemUuid,
        shopping_id: primaryInsertId
      }));
      // Fall through (no early return): ALSO write the per-user {member}_new_list
      // tables below. Those are the canonical store the iOS app + HALO device list
      // read from — without this, voice adds land only in shared_shopping_list and
      // never appear anywhere the user looks.
    }

    const memberIds = getShoppingHouseholdMemberIds(context);
    let primaryInsertId = null;

    logHouseholdWriteTargets("shopping item insert propagation", context, memberIds, {
      mode: "direct",
      household_item_uuid: householdItemUuid,
      item_name: displayItemName,
      store
    });

    await dualWriteAcrossHousehold(connection, {
      memberIds, context, op: "insert", reportEvt: "shopping_peruser_write_miss",
      tableFn: shoppingListTableName, ensureFn: ensureShoppingTable,
      buildStatement: (tableName, memberId) => ({
        sql: `INSERT INTO \`${tableName}\` (
          _owner,
          _device,
          product_name,
          quantity,
          product_brand,
          images,
          product_barcode,
          store,
          action,
          _createdDate,
          created_at,
          updated_at,
          household_item_uuid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), NOW(), ?)`,
        values: [memberId, "voice-assistant", displayItemName, quantity || null, null, null, null, store, "ADDED", householdItemUuid],
      }),
      onResult: (memberId, result) => {
        if (memberId === shoppingOwnerId) {
          primaryInsertId = result?.insertId != null ? String(result.insertId) : null;
        }
      },
    });

    const result = {
      shopping_id: primaryInsertId,
      household_item_uuid: householdItemUuid,
      item_name: displayItemName,
      quantity: quantity || null,
      store,
      source: "new_list",
      action: "ADDED"
    };
    console.log("[DEBUG] shopping item added:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: displayItemName,
      store,
      household_item_uuid: householdItemUuid,
      shopping_id: primaryInsertId
    }));
    return result;
  }, options);
}

export async function addManyShoppingItems(context, batchItems, options = {}) {
  // Run the WHOLE batch on ONE pooled connection with the same shared + household
  // fall-through as addShoppingItem (WRITE_SHARED_ONLY dual-write). This previously
  // delegated to addShoppingItem per item, acquiring/releasing a fresh pool
  // connection in a tight 20-item loop; on a 2+ member household that intermittently
  // left per-user {member}_new_list copies missing while the shared_shopping_list
  // row landed (canary trepo-capture-voice-dualwrite-miss; 7/20 for owner 7d7df434).
  // One connection + a per-insert affectedRows check makes the dual-write
  // deterministic and self-reporting.
  return withDbConnection(async (connection) => {
    const memberIds = getShoppingHouseholdMemberIds(context);
    const shoppingOwnerId = resolveShoppingOwnerId(context);
    const items = [];

    for (const batchItem of batchItems || []) {
      const householdItemUuid = crypto.randomUUID();
      const store = await resolveShoppingStore(connection, context, batchItem.store || null);
      const displayItemName = formatUserFacingItemName(batchItem.item_name);
      const quantity = batchItem.quantity || null;

      if (WRITE_SHARED_ONLY) {
        await connection.execute(
          `INSERT INTO \`${SHARED_SHOPPING_TABLE}\` (
            _owner, _device, product_name, quantity, product_brand,
            images, product_barcode, store, action,
            _createdDate, created_at, updated_at, household_item_uuid, owner_id
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), NOW(), ?, ?)`,
          [shoppingOwnerId, "voice-assistant", displayItemName, quantity, null, null, null, store, "ADDED", householdItemUuid, shoppingOwnerId]
        );
      }

      // ALSO write each household member's per-user {member}_new_list — the store
      // the iOS app + HALO device read from.
      let primaryInsertId = null;
      await dualWriteAcrossHousehold(connection, {
        memberIds, context, op: "insert", reportEvt: "shopping_peruser_write_miss",
        tableFn: shoppingListTableName, ensureFn: ensureShoppingTable,
        buildStatement: (tableName, memberId) => ({
          sql: `INSERT INTO \`${tableName}\` (
            _owner, _device, product_name, quantity, product_brand,
            images, product_barcode, store, action,
            _createdDate, created_at, updated_at, household_item_uuid
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), NOW(), ?)`,
          values: [memberId, "voice-assistant", displayItemName, quantity, null, null, null, store, "ADDED", householdItemUuid],
        }),
        onResult: (memberId, result) => {
          if (memberId === shoppingOwnerId) {
            primaryInsertId = result?.insertId != null ? String(result.insertId) : null;
          }
        },
      });

      items.push({
        shopping_id: primaryInsertId,
        household_item_uuid: householdItemUuid,
        item_name: displayItemName,
        quantity,
        store,
        source: "new_list",
        action: "ADDED"
      });
    }

    console.log("[DEBUG] shopping items added in batch:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: items.length,
      item_names: items.map((item) => item.item_name)
    }));

    return { items, count: items.length };
  }, options);
}

export async function removeShoppingItem(context, itemName, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const items = await hydrateShoppingRows(connection, context, shoppingRows);
    const target = findShoppingItemMatch(items, itemName);

    if (!target?.shopping_id) {
      throw createShoppingNotFoundError(itemName);
    }

    if (WRITE_SHARED_ONLY) {
      if (target.household_item_uuid) {
        await connection.execute(
          `DELETE FROM \`${SHARED_SHOPPING_TABLE}\` WHERE household_item_uuid = ?`,
          [target.household_item_uuid]
        );
      } else if (target.shopping_id) {
        await connection.execute(
          `DELETE FROM \`${SHARED_SHOPPING_TABLE}\` WHERE _id = ? LIMIT 1`,
          [target.shopping_id]
        );
      } else {
        await connection.execute(
          `DELETE FROM \`${SHARED_SHOPPING_TABLE}\` WHERE \`owner_id\` = ? AND LOWER(TRIM(product_name)) = ? ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC LIMIT 1`,
          [resolveShoppingOwnerId(context), normalizeName(target.item_name)]
        );
      }
      console.log("[DEBUG] shopping item removed (shared):", JSON.stringify({
        ownerId: context?.ownerId || null,
        userId: context?.userId || null,
        item_name: target.item_name,
        shopping_id: target.shopping_id,
        household_item_uuid: target.household_item_uuid || null
      }));
      // Fall through (no early return): ALSO delete from the per-user
      // {member}_new_list tables the app/device read, so a voice removal is
      // reflected everywhere.
    }

    const shoppingOwnerId = resolveShoppingOwnerId(context);
    const memberIds = getShoppingHouseholdMemberIds(context);
    logHouseholdWriteTargets("shopping item delete propagation", context, memberIds, {
      mode: "direct",
      household_item_uuid: target?.household_item_uuid || null,
      shopping_id: target?.shopping_id || null,
      item_name: target?.item_name || itemName
    });

    await dualWriteAcrossHousehold(connection, {
      memberIds, context, op: "delete", reportEvt: "shopping_peruser_write_miss",
      tableFn: shoppingListTableName, ensureFn: ensureShoppingTable,
      buildStatement: (tableName, memberId) => {
        if (target.household_item_uuid) {
          return {
            sql: `DELETE FROM \`${tableName}\` WHERE household_item_uuid = ?`,
            values: [target.household_item_uuid],
          };
        }
        if (memberId === shoppingOwnerId && target.shopping_id) {
          return {
            sql: `DELETE FROM \`${tableName}\` WHERE _id = ? LIMIT 1`,
            values: [target.shopping_id],
          };
        }
        return {
          sql: `DELETE FROM \`${tableName}\`
         WHERE LOWER(TRIM(product_name)) = ?
         ORDER BY COALESCE(updated_at, created_at, _createdDate) DESC
         LIMIT 1`,
          values: [normalizeName(target.item_name)],
        };
      },
    });

    console.log("[DEBUG] shopping item removed:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: target.item_name,
      shopping_id: target.shopping_id,
      household_item_uuid: target.household_item_uuid || null
    }));
    return target;
  }, options);
}

export async function clearShoppingList(context, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const items = await hydrateShoppingRows(connection, context, shoppingRows);

    if (WRITE_SHARED_ONLY) {
      const ownerId = resolveShoppingOwnerId(context);
      await connection.execute(
        `DELETE FROM \`${SHARED_SHOPPING_TABLE}\` WHERE \`owner_id\` = ?`,
        [ownerId]
      );
      console.log("[DEBUG] shopping list cleared (shared):", JSON.stringify({
        ownerId: context?.ownerId || null,
        userId: context?.userId || null,
        count: items.length,
        item_names: items.slice(0, 25).map((item) => item.item_name)
      }));
      // Fall through (no early return): ALSO clear the per-user {member}_new_list
      // tables the app/device read.
    }

    const memberIds = getShoppingHouseholdMemberIds(context);

    logHouseholdWriteTargets("shopping list clear propagation", context, memberIds, {
      mode: "direct",
      count: items.length
    });

    await dualWriteAcrossHousehold(connection, {
      memberIds, context, op: "delete", reportEvt: "shopping_peruser_write_miss",
      tableFn: shoppingListTableName, ensureFn: ensureShoppingTable,
      buildStatement: (tableName) => ({
        sql: `DELETE FROM \`${tableName}\``,
        values: [],
      }),
    });

    console.log("[DEBUG] shopping list cleared:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: items.length,
      item_names: items.slice(0, 25).map((item) => item.item_name)
    }));

    return {
      items,
      count: items.length
    };
  }, options);
}

export async function updateShoppingItemStore(context, itemName, requestedStore, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const items = await hydrateShoppingRows(connection, context, shoppingRows);
    const target = findShoppingItemMatch(items, itemName);

    if (!target?.shopping_id) {
      throw createShoppingNotFoundError(itemName);
    }

    const store = await resolveShoppingStore(connection, context, requestedStore);
    if (!store) {
      throw new Error("A store name is required.");
    }

    await updateShoppingStoreAcrossHousehold(connection, context, target, store);

    const result = {
      ...target,
      store
    };
    console.log("[DEBUG] shopping item store updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: target.item_name,
      shopping_id: target.shopping_id,
      household_item_uuid: target.household_item_uuid || null,
      store
    }));
    return result;
  }, options);
}

export async function updateManyShoppingItemStores(context, batchItems, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const existingItems = await hydrateShoppingRows(connection, context, shoppingRows);
    const collapsedBatchItems = maybeCollapseSplitShoppingBatchItems(existingItems, batchItems || []);
    const resolvedUpdates = [];

    for (const batchItem of collapsedBatchItems) {
      const target = resolveShoppingItemMatch(existingItems, batchItem.item_name);
      if (!target?.shopping_id) {
        throw createShoppingNotFoundError(batchItem.item_name);
      }

      const store = await resolveShoppingStore(connection, context, batchItem.store);
      if (!store) {
        throw new Error("A store name is required.");
      }

      resolvedUpdates.push({
        requested_item_name: batchItem.item_name,
        target,
        store
      });
    }

    console.log("[DEBUG] shopping batch preflight resolved:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: resolvedUpdates.length,
      items: resolvedUpdates.map((update) => ({
        requested_item_name: update.requested_item_name,
        resolved_item_name: update.target.item_name,
        store: update.store
      }))
    }));

    const items = [];
    for (const update of resolvedUpdates) {
      await updateShoppingStoreAcrossHousehold(connection, context, update.target, update.store);
      items.push({
        ...update.target,
        store: update.store
      });
    }

    console.log("[DEBUG] shopping item stores updated in batch:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: items.length,
      item_names: items.map((item) => item.item_name)
    }));

    return {
      items,
      count: items.length
    };
  }, options);
}

async function setShoppingItemAction(context, itemName, action, options = {}) {
  return withDbConnection(async (connection) => {
    const shoppingRows = await getShoppingRows(connection, context);
    const items = await hydrateShoppingRows(connection, context, shoppingRows);
    const target = findShoppingItemMatch(items, itemName);

    if (!target?.shopping_id) {
      throw createShoppingNotFoundError(itemName);
    }

    await updateShoppingFieldsAcrossHousehold(connection, context, target, { action });
    const result = {
      ...target,
      action
    };
    console.log("[DEBUG] shopping item action updated:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      item_name: target.item_name,
      shopping_id: target.shopping_id,
      household_item_uuid: target.household_item_uuid || null,
      action
    }));
    return result;
  }, options);
}

export async function markShoppingItemBought(context, itemName, options = {}) {
  return setShoppingItemAction(context, itemName, "2", options);
}

export async function markShoppingItemUnbought(context, itemName, options = {}) {
  return setShoppingItemAction(context, itemName, "1", options);
}

export async function listStoreTabs(context, options = {}) {
  return withDbConnection(async (connection) => {
    const stores = await listResolvedStoreTabs(connection, context);
    console.log("[DEBUG] shopping store tabs listed:", JSON.stringify({
      ownerId: context?.ownerId || null,
      userId: context?.userId || null,
      count: stores.length
    }));
    return {
      stores,
      count: stores.length
    };
  }, options);
}

export async function buildHouseholdSummary(context, options = {}) {
  return withDbConnection(async (connection) => {
    const kitchen = await getKitchenRows(connection, resolveTableOwnerId(context), 100);
    const discards = await getRecentDiscards(context, { ...options, limit: 10 });
    const dishes = await getRecentDishes(context, { ...options, limit: 10 });
    const shopping = await getShoppingItems(context, { ...options, limit: 50 });
    const metricsTable = metricsTableName(resolveTableOwnerId(context));

    let latestMetrics = null;
    if (await tableExists(connection, metricsTable)) {
      const [rows] = await connection.execute(
        `SELECT *
         FROM \`${metricsTable}\`
         ORDER BY _createdDate DESC
         LIMIT 1`
      );
      latestMetrics = rows?.[0] || null;
    }

    return {
      household_size: context.householdSize,
      kitchen_count: kitchen.length,
      shopping_count: shopping.length,
      discard_count: discards.length,
      recent_dish_count: dishes.count || 0,
      latest_metrics: latestMetrics
        ? {
            iq: latestMetrics.IQ ?? null,
            points: latestMetrics.Points ?? null,
            upf: latestMetrics.UPF ?? null,
            harmful_ingredients: latestMetrics.harmful_ingredients ?? null,
            suggestions: parseJsonColumn(latestMetrics.IQ_suggestions, [])
          }
        : null
    };
  }, options);
}

export async function getHealthMetrics(context, options = {}) {
  const ownerId = resolveTableOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/metrics/${encodeURIComponent(ownerId)}`, options);
  return {
    owner: payload.owner || ownerId,
    count: Number(payload.count || 0),
    metrics: payload.metrics || null
  };
}

export async function getRecipeSuggestions(context, options = {}) {
  const ownerId = resolveTableOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/recipes/${encodeURIComponent(ownerId)}`, options);
  return normalizeRecipePayload(payload, options.limit ?? null);
}

export async function getRecipeDetail(context, reference = {}, options = {}) {
  const ownerId = resolveTableOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/recipes/${encodeURIComponent(ownerId)}`, options);
  const normalized = normalizeRecipePayload(payload);

  if (flattenRecipeSuggestions(normalized).length === 0 && normalized.status !== "ready") {
    return {
      owner: normalized.owner || ownerId,
      status: normalized.status,
      recipe: null,
      error_message: normalized.error_message || null
    };
  }

  const recipe = await enrichRecipeDetailFromWeb(context, resolveRecipeReference(normalized, reference), options);
  return {
    owner: normalized.owner || ownerId,
    status: normalized.status,
    recipe,
    error_message: normalized.error_message || null
  };
}

export async function addRecipeIngredientsToShoppingList(context, reference = {}, options = {}) {
  const detail = await getRecipeDetail(context, reference, options);
  if (!detail?.recipe) {
    const error = new Error(detail?.status === "regenerating"
      ? "Recipes are still updating right now."
      : "That recipe is not ready yet.");
    error.statusCode = detail?.status === "regenerating" ? 409 : 404;
    throw error;
  }

  const ingredients = reference.only_missing === false
    ? detail.recipe.ingredients
    : (detail.recipe.missing_ingredients.length > 0 ? detail.recipe.missing_ingredients : []);

  if (ingredients.length === 0) {
    return {
      recipe: detail.recipe,
      items: [],
      count: 0
    };
  }

  const added = await addManyShoppingItems(
    context,
    ingredients.map((item_name) => ({
      item_name,
      ...(reference.store ? { store: reference.store } : {})
    })),
    options
  );

  return {
    recipe: detail.recipe,
    items: added.items,
    count: added.count
  };
}

export async function getSavedRecipes(context, options = {}) {
  const ownerId = resolveShoppingOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/saved-recipes/${encodeURIComponent(ownerId)}`, options);
  return normalizeSavedRecipesPayload(payload, options.limit ?? null);
}

export async function getSavedRecipeDetail(context, reference = {}, options = {}) {
  const ownerId = resolveShoppingOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/saved-recipes/${encodeURIComponent(ownerId)}`, options);
  const normalized = normalizeSavedRecipesPayload(payload);
  const recipe = resolveSavedRecipeReference(normalized, reference);
  return {
    owner: normalized.owner || ownerId,
    recipe
  };
}

export async function saveRecipeFromTikTok(context, args = {}, options = {}) {
  const ownerId = resolveShoppingOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/saved-recipes/${encodeURIComponent(ownerId)}`, {
    ...options,
    method: "POST",
    body: {
      url: args.url
    }
  });
  return {
    owner: payload.owner || ownerId,
    recipe: normalizeSavedRecipeItem(payload.recipe || null),
    deduped: Boolean(payload.deduped)
  };
}

export async function saveGeneratedRecipe(context, args = {}, options = {}) {
  const ownerId = resolveShoppingOwnerId(context);
  const title = String(args.title || "").trim();
  const ingredients = (Array.isArray(args.ingredients) ? args.ingredients : [])
    .map((line) => String(line || "").trim())
    .filter(Boolean);
  const steps = (Array.isArray(args.steps) ? args.steps : [])
    .map((line) => String(line || "").trim())
    .filter(Boolean);
  if (!title || ingredients.length === 0 || steps.length === 0) {
    const error = new Error("A generated recipe needs a title, ingredients, and steps to save.");
    error.statusCode = 400;
    throw error;
  }
  // Mirror the iOS recipe-card save: post the formatted recipe text to the saved-recipes
  // endpoint, which parses + refines it asynchronously (same path as the card's Save button).
  const content = `${title}\n\nIngredients:\n`
    + ingredients.map((line) => `- ${line}`).join("\n")
    + `\n\nSteps:\n`
    + steps.map((line, idx) => `${idx + 1}. ${line}`).join("\n");
  const payload = await fetchHouseholdApiJson(`/saved-recipes/${encodeURIComponent(ownerId)}`, {
    ...options,
    method: "POST",
    body: { content }
  });
  return {
    owner: payload.owner || ownerId,
    recipe: normalizeSavedRecipeItem(payload.recipe || { title }),
    title
  };
}

export async function removeSavedRecipe(context, reference = {}, options = {}) {
  const ownerId = resolveShoppingOwnerId(context);
  const detail = await getSavedRecipeDetail(context, reference, options);
  if (!detail?.recipe?.id) {
    const error = new Error("That saved recipe could not be found.");
    error.statusCode = 404;
    throw error;
  }

  const payload = await fetchHouseholdApiJson(
    `/saved-recipes/${encodeURIComponent(ownerId)}/${encodeURIComponent(detail.recipe.id)}`,
    {
      ...options,
      method: "DELETE"
    }
  );

  return {
    owner: payload.owner || ownerId,
    recipe: normalizeSavedRecipeItem(payload.recipe || detail.recipe)
  };
}

export async function addSavedRecipeIngredientsToShoppingList(context, reference = {}, options = {}) {
  const detail = await getSavedRecipeDetail(context, reference, options);
  if (!detail?.recipe) {
    const error = new Error("That saved recipe could not be found.");
    error.statusCode = 404;
    throw error;
  }

  const ingredients = Array.isArray(detail.recipe.ingredients) ? detail.recipe.ingredients.filter(Boolean) : [];
  if (ingredients.length === 0) {
    return {
      recipe: detail.recipe,
      items: [],
      count: 0
    };
  }

  const added = await addManyShoppingItems(
    context,
    ingredients.map((item_name) => ({
      item_name,
      ...(reference.store ? { store: reference.store } : {})
    })),
    options
  );

  return {
    recipe: detail.recipe,
    items: added.items,
    count: added.count
  };
}

export async function getMealPlan(context, options = {}) {
  const ownerId = resolveTableOwnerId(context);
  const payload = await fetchHouseholdApiJson(`/meal-plan/${encodeURIComponent(ownerId)}`, options);
  return {
    owner: payload.owner || ownerId,
    status: payload.status || "empty",
    focus: payload.focus || null,
    explanation_title: payload.explanation_title || null,
    explanation_paragraph: payload.explanation_paragraph || null,
    plan: normalizeMealPlanSlots(payload.plan),
    error_message: payload.error_message || null,
    _createdDate: payload._createdDate || null,
    _updatedDate: payload._updatedDate || null
  };
}

export async function refreshMealPlan(context, args = {}, options = {}) {
  const ownerId = resolveTableOwnerId(context);
  const memberIds = getTableHouseholdMemberIds(context);
  logHouseholdWriteTargets("meal plan refresh propagation", context, memberIds, {
    mode: "delegated",
    delegate: "household_api",
    focus: args?.focus || null
  });

  const results = await Promise.all(memberIds.map(async (memberId) => ({
    ownerId: memberId,
    payload: await fetchHouseholdApiJson(`/meal-plan/${encodeURIComponent(memberId)}`, {
      ...options,
      method: "POST",
      body: args.focus ? { focus: args.focus } : {}
    })
  })));
  const primary = results.find((result) => result.ownerId === ownerId) || results[0] || { ownerId, payload: {} };
  const payload = primary.payload || {};

  return {
    owner: payload.owner || ownerId,
    status: payload.status || "regenerating",
    focus: payload.focus || args.focus || null,
    message: payload.message || "Meal plan regeneration started.",
    async_update: {
      pending: true,
      status: payload.status || "regenerating",
      type: "meal_plan",
      focus: payload.focus || args.focus || null
    }
  };
}

export async function recommendItemsToUseUp(context, options = {}) {
  const kitchenItems = await getKitchenItems(context, { ...options, limit: 100 });
  return sortUseUpCandidates(kitchenItems)
    .slice(0, clampLimit(options.limit, 5, 10))
    .map((item) => ({
      item_name: item.item_name,
      reason: item.expiration_date
        ? `Use soon because it expires on ${item.expiration_date}.`
        : item.is_opened
          ? "Use soon because it is already opened."
          : item.state_label
            ? `Use soon because only ${item.state_label} is left.`
            : "Use soon because it has been in the kitchen for a while."
    }));
}

export async function recommendItemsToBuy(context, options = {}) {
  const limit = clampLimit(options.limit, 5, 10);
  const shoppingItems = await getShoppingItems(context, { ...options, limit: 50 });
  const activeShoppingItems = shoppingItems.filter((item) => item.action !== "2");

  if (activeShoppingItems.length > 0) {
    return activeShoppingItems.slice(0, limit).map((item) => ({
      item_name: item.item_name,
      reason: "It is already on the shopping list."
    }));
  }

  const kitchenItems = await getKitchenItems(context, { ...options, limit: 100 });
  const lowStockItems = kitchenItems.filter((item) => {
    const remaining = normalizeName(item.remaining_quantity);
    return (item.fill_percent != null && item.fill_percent <= 50)
      || remaining.includes("half")
      || remaining.includes("left")
      || remaining.includes("low")
      || remaining.includes("one");
  });

  if (lowStockItems.length > 0) {
    return lowStockItems.slice(0, limit).map((item) => ({
      item_name: item.item_name,
      reason: `You marked it as low stock (${item.state_label || item.remaining_quantity || "limited quantity"}).`
    }));
  }

  const discards = await getRecentDiscards(context, { ...options, limit: 25 });
  const kitchenNames = new Set(kitchenItems.map((item) => normalizeName(item.item_name)));

  return discards
    .filter((item) => !kitchenNames.has(normalizeName(item.item_name)))
    .slice(0, limit)
    .map((item) => ({
      item_name: item.item_name,
      reason: "It was discarded recently and is not currently in the kitchen."
    }));
}

export async function getHouseholdFeed(context, options = {}) {
  return withDbConnection(async (connection) => {
    const ownerId = resolveTableOwnerId(context);
    const modernTable = feedTableName(ownerId);
    const legacyTable = legacyFeedTableName(ownerId);
    const safeLimit = clampLimit(options.limit, 10, 50);
    const eventType = normalizeName(options.event_type).replace(/\s+/g, "_");
    const items = [];
    let source = null;

    if (await tableExists(connection, modernTable)) {
      const params = [];
      let whereClause = "";
      if (eventType) {
        whereClause = "WHERE LOWER(event_type) = ?";
        params.push(eventType);
      }
      const [rows] = await connection.execute(
        `SELECT *
         FROM \`${modernTable}\`
         ${whereClause}
         ORDER BY _createdDate DESC
         LIMIT ${safeLimit}`,
        params
      );
      items.push(...(rows || []).map(mapFeedEventRow));
      if (items.length > 0) {
        source = "feed_events";
      }
    }

    if (items.length < safeLimit && await tableExists(connection, legacyTable)) {
      const [rows] = await connection.execute(
        `SELECT *
         FROM \`${legacyTable}\`
         ORDER BY COALESCE(_createdDate, created_at, updated_at) DESC
         LIMIT ${safeLimit}`
      );
      const legacyItems = (rows || [])
        .map(mapLegacyFeedRow)
        .filter((row) => !eventType || row.event_type === eventType)
        .filter((row) => !items.some((existing) => existing.id === row.id));
      items.push(...legacyItems);
      if (legacyItems.length > 0) {
        source = source ? "mixed" : "new_feed";
      }
    }

    const sortedItems = items
      .sort((left, right) => new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime())
      .slice(0, safeLimit);

    return {
      items: sortedItems,
      count: sortedItems.length,
      source: source || "feed_events"
    };
  }, options);
}

export { parseJsonColumn };
