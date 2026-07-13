"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.lookupPackagedNutrition = lookupPackagedNutrition;
const node_fetch_1 = __importDefault(require("node-fetch"));
const openai_1 = __importDefault(require("openai"));
const webLookupNutritionJsonSchema = {
    type: "object",
    additionalProperties: false,
    required: [
        "dish_name",
        "serving_size",
        "calories",
        "total_fat",
        "saturated_fat",
        "trans_fat",
        "cholesterol",
        "sodium",
        "total_carbohydrates",
        "dietary_fiber",
        "sugars",
        "protein",
        "vitamin_a",
        "vitamin_c",
        "calcium",
        "iron",
        "confidence",
        "explanation",
        "ingredients",
        "allergens",
        "evidence_urls",
        "source_type",
    ],
    properties: {
        dish_name: { type: ["string", "null"] },
        serving_size: { type: ["string", "null"] },
        calories: { type: ["number", "null"] },
        total_fat: { type: ["number", "null"] },
        saturated_fat: { type: ["number", "null"] },
        trans_fat: { type: ["number", "null"] },
        cholesterol: { type: ["number", "null"] },
        sodium: { type: ["number", "null"] },
        total_carbohydrates: { type: ["number", "null"] },
        dietary_fiber: { type: ["number", "null"] },
        sugars: { type: ["number", "null"] },
        protein: { type: ["number", "null"] },
        vitamin_a: { type: ["number", "null"] },
        vitamin_c: { type: ["number", "null"] },
        calcium: { type: ["number", "null"] },
        iron: { type: ["number", "null"] },
        confidence: { type: "number", minimum: 0, maximum: 1 },
        explanation: { type: ["string", "null"] },
        ingredients: { type: "array", items: { type: "string" } },
        allergens: { type: "array", items: { type: "string" } },
        evidence_urls: { type: "array", items: { type: "string" } },
        source_type: { type: ["string", "null"] },
    },
};
let client = null;
function getClient() {
    if (!client) {
        if (!process.env.OPENAI_API_KEY) {
            throw new Error("OPENAI_API_KEY environment variable is required");
        }
        client = new openai_1.default({
            apiKey: process.env.OPENAI_API_KEY,
        });
    }
    return client;
}
function getDefaultModel() {
    return process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
}
function supportsReasoning(model) {
    return /^(o[1-9]|gpt-5)/.test(model);
}
function parseJsonResponse(response) {
    const outputText = response?.output_text;
    if (typeof outputText !== "string" || !outputText.trim()) {
        throw new Error("No structured output returned from OpenAI");
    }
    return JSON.parse(outputText);
}
function normalizeText(value) {
    return String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .trim();
}
function tokenize(value) {
    const normalized = normalizeText(value);
    return normalized ? normalized.split(/\s+/).filter(Boolean) : [];
}
function sanitizeBarcode(value) {
    if (!value)
        return null;
    const digits = value.replace(/\D/g, "");
    return digits.length >= 8 ? digits : null;
}
function parseServingGrams(value) {
    if (!value)
        return null;
    const match = String(value).match(/(\d+(?:\.\d+)?)\s*g\b/i);
    if (!match)
        return null;
    const grams = Number(match[1]);
    return Number.isFinite(grams) && grams > 0 ? grams : null;
}
function normalizeStringArray(values) {
    if (!Array.isArray(values))
        return [];
    return values
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .slice(0, 10);
}
async function fetchJson(url) {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        const res = await (0, node_fetch_1.default)(url, {
            signal: controller.signal,
            headers: {
                "user-agent": "trepo-dish-logger/1.0",
            },
        });
        clearTimeout(timeout);
        if (!res.ok)
            return null;
        return (await res.json());
    }
    catch {
        return null;
    }
}
function pickString(...values) {
    for (const value of values) {
        if (typeof value === "string" && value.trim()) {
            return value.trim();
        }
    }
    return null;
}
function parseIngredients(product) {
    const raw = pickString(product.ingredients_text_en, product.ingredients_text);
    if (!raw)
        return [];
    return raw
        .split(/[,;]\s*/)
        .map((item) => item.trim())
        .filter(Boolean)
        .slice(0, 100);
}
function parseAllergens(product) {
    const raw = [
        ...(Array.isArray(product.allergens_tags) ? product.allergens_tags : []),
        ...(Array.isArray(product.allergens_hierarchy) ? product.allergens_hierarchy : []),
    ];
    const seen = new Set();
    const allergens = [];
    for (const item of raw) {
        const cleaned = String(item || "").replace(/^[a-z]{2}:/i, "").trim().toLowerCase();
        if (cleaned && !seen.has(cleaned)) {
            seen.add(cleaned);
            allergens.push(cleaned);
        }
    }
    return allergens;
}
function toNumber(value) {
    if (value == null || value === "")
        return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
}
function pickNutriment(nutriments, keys) {
    if (!nutriments)
        return { value: null, source: null };
    for (const key of keys) {
        const servingValue = toNumber(nutriments[`${key}_serving`]);
        if (servingValue != null)
            return { value: servingValue, source: "serving" };
        const hundredGValue = toNumber(nutriments[`${key}_100g`]);
        if (hundredGValue != null)
            return { value: hundredGValue, source: "100g" };
        const baseValue = toNumber(nutriments[key]);
        if (baseValue != null)
            return { value: baseValue, source: "base" };
    }
    return { value: null, source: null };
}
function sodiumFromNutriments(nutriments) {
    const sodium = pickNutriment(nutriments, ["sodium"]);
    if (sodium.value != null) {
        return {
            value: sodium.value * 1000,
            source: sodium.source,
        };
    }
    const salt = pickNutriment(nutriments, ["salt"]);
    if (salt.value != null) {
        return {
            value: salt.value * 400,
            source: salt.source,
        };
    }
    return { value: null, source: null };
}
function caloriesFromNutriments(nutriments) {
    const kcal = pickNutriment(nutriments, ["energy-kcal"]);
    if (kcal.value != null)
        return kcal;
    const kj = pickNutriment(nutriments, ["energy-kj", "energy"]);
    if (kj.value != null) {
        return {
            value: Math.round((kj.value / 4.184) * 10) / 10,
            source: kj.source,
        };
    }
    return { value: null, source: null };
}
function scoreCandidate(product, item) {
    let score = 0;
    const candidateName = normalizeText(pickString(product.product_name, product.product_name_en));
    const candidateBrand = normalizeText(product.brands);
    const itemName = normalizeText(item.product_name);
    const itemBrand = normalizeText(item.brand);
    const itemVariant = normalizeText(item.variant);
    if (itemBrand && candidateBrand) {
        if (candidateBrand.includes(itemBrand) || itemBrand.includes(candidateBrand))
            score += 30;
    }
    if (itemName && candidateName) {
        if (candidateName === itemName) {
            score += 60;
        }
        else if (candidateName.includes(itemName) || itemName.includes(candidateName)) {
            score += 40;
        }
        else {
            const itemTokens = tokenize(itemName);
            const candidateTokens = new Set(tokenize(candidateName));
            const overlap = itemTokens.filter((token) => candidateTokens.has(token)).length;
            score += overlap * 8;
        }
    }
    if (itemVariant) {
        const combined = `${candidateName} ${candidateBrand}`;
        if (combined.includes(itemVariant))
            score += 10;
    }
    if (product.serving_size)
        score += 3;
    if (product.nutriments)
        score += 5;
    return score;
}
async function lookupByBarcode(barcode) {
    const url = `https://world.openfoodfacts.org/api/v2/product/${encodeURIComponent(barcode)}.json?fields=code,product_name,product_name_en,brands,brands_tags,serving_size,nutriments,ingredients_text,ingredients_text_en,allergens_tags,allergens_hierarchy`;
    const payload = await fetchJson(url);
    if (!payload?.product)
        return null;
    return payload.product;
}
async function searchByName(item) {
    const query = [item.brand, item.product_name, item.variant].filter(Boolean).join(" ").trim();
    if (!query)
        return null;
    const url = `https://world.openfoodfacts.org/cgi/search.pl?search_terms=${encodeURIComponent(query)}` +
        "&search_simple=1&action=process&json=1&page_size=10" +
        "&fields=code,product_name,product_name_en,brands,brands_tags,serving_size,nutriments,ingredients_text,ingredients_text_en,allergens_tags,allergens_hierarchy";
    const payload = await fetchJson(url);
    const products = Array.isArray(payload?.products) ? payload.products : [];
    if (products.length === 0)
        return null;
    const sorted = products
        .map((product) => ({ product, score: scoreCandidate(product, item) }))
        .sort((a, b) => b.score - a.score);
    return sorted[0]?.score >= 35 ? sorted[0].product : null;
}
function buildDishName(product, item) {
    const productName = pickString(product.product_name, product.product_name_en, item.product_name);
    const brand = pickString(product.brands, item.brand);
    if (brand && productName)
        return `${brand} ${productName}`.trim();
    return productName;
}
function buildServingSize(product, sourceHints, item) {
    const explicitServing = pickString(product.serving_size, item.serving_size);
    if (sourceHints.includes("serving") && explicitServing)
        return explicitServing;
    if (sourceHints.includes("100g"))
        return "100 g";
    return explicitServing || "1 serving";
}
function convertFrom100g(value, source, servingGrams) {
    if (value == null)
        return null;
    if (source !== "100g")
        return value;
    if (servingGrams == null)
        return null;
    return Math.round(((value * servingGrams) / 100) * 10) / 10;
}
async function lookupViaWebSearch(item) {
    if (!item.product_name)
        return null;
    const openai = getClient();
    const model = process.env.FULL_DISH_MODEL || getDefaultModel();
    const query = [item.brand, item.product_name, item.variant].filter(Boolean).join(" ").trim();
    const response = await openai.responses.create({
        model,
        ...(supportsReasoning(model) ? { reasoning: { effort: "low" } } : {}),
        max_output_tokens: 1400,
        tools: [
            {
                type: "web_search_preview",
                search_context_size: "medium",
            },
        ],
        tool_choice: { type: "web_search_preview" },
        include: ["web_search_call.action.sources"],
        text: {
            format: {
                type: "json_schema",
                name: "web_packaged_nutrition",
                strict: true,
                schema: webLookupNutritionJsonSchema,
            },
        },
        input: [
            {
                role: "system",
                content: [{
                        type: "input_text",
                        text: `You reconcile nutrition for a packaged commercial food or beverage by using web search results from actual product pages.

Return STRICT JSON only with this exact structure:
{
  "dish_name": string | null,
  "serving_size": string | null,
  "calories": number | null,
  "total_fat": number | null,
  "saturated_fat": number | null,
  "trans_fat": number | null,
  "cholesterol": number | null,
  "sodium": number | null,
  "total_carbohydrates": number | null,
  "dietary_fiber": number | null,
  "sugars": number | null,
  "protein": number | null,
  "vitamin_a": number | null,
  "vitamin_c": number | null,
  "calcium": number | null,
  "iron": number | null,
  "confidence": number,
  "explanation": string | null,
  "ingredients": string[],
  "allergens": string[],
  "evidence_urls": string[],
  "source_type": string | null
}

CRITICAL RULES:
1. Prioritize exact product pages and nutrition panels from the manufacturer and major retailer product pages.
2. Do NOT rely on crowd-edited databases unless nothing better exists.
3. Only return nutrition when the searched result clearly matches the same brand/product/variant.
4. Prefer per-serving values. If you only find per-100g values and no trustworthy serving size, return nutrition fields as null and confidence <= 0.4.
5. If multiple sources disagree materially, only return values if one source is clearly the authoritative product page; otherwise return null nutrition fields and confidence <= 0.4.
6. evidence_urls must contain 1-3 actual pages used, if any were found.
7. confidence must reflect both source quality and product-match confidence.
8. Return ONLY JSON.`,
                    }],
            },
            {
                role: "user",
                content: [{
                        type: "input_text",
                        text: `Find nutrition for this packaged edible product using web results from actual product pages.

Product details:
- Query: ${query}
- Brand: ${item.brand || "Unknown"}
- Product name: ${item.product_name}
- Variant: ${item.variant || "Unknown"}
- Category: ${item.category || "Unknown"}
- Barcode: ${item.barcode || "Unknown"}
- Visible serving size hint: ${item.serving_size || "Unknown"}

Prioritize manufacturer and retailer product pages. Return per-serving nutrition only when the exact product match is clear.`,
                    }],
            },
        ],
    });
    const parsed = parseJsonResponse(response);
    return {
        ...parsed,
        evidence_urls: normalizeStringArray(parsed.evidence_urls),
        ingredients: normalizeStringArray(parsed.ingredients),
        allergens: normalizeStringArray(parsed.allergens),
    };
}
async function lookupPackagedNutrition(item) {
    if (!item.product_name)
        return null;
    try {
        const webResult = await lookupViaWebSearch(item);
        const webHasServing = typeof webResult?.serving_size === "string" && webResult.serving_size.trim().length > 0;
        const webHasMacros = webResult?.calories != null ||
            webResult?.total_fat != null ||
            webResult?.total_carbohydrates != null ||
            webResult?.protein != null;
        const webHasEvidence = Array.isArray(webResult?.evidence_urls) && webResult.evidence_urls.length > 0;
        if (webResult && webHasServing && webHasMacros && webHasEvidence && (webResult.confidence || 0) >= 0.75) {
            return {
                dish_name: webResult.dish_name,
                serving_size: webResult.serving_size,
                calories: webResult.calories,
                total_fat: webResult.total_fat,
                saturated_fat: webResult.saturated_fat,
                trans_fat: webResult.trans_fat,
                cholesterol: webResult.cholesterol,
                sodium: webResult.sodium,
                total_carbohydrates: webResult.total_carbohydrates,
                dietary_fiber: webResult.dietary_fiber,
                sugars: webResult.sugars,
                protein: webResult.protein,
                vitamin_a: webResult.vitamin_a,
                vitamin_c: webResult.vitamin_c,
                calcium: webResult.calcium,
                iron: webResult.iron,
                confidence: webResult.confidence,
                explanation: `Nutrition sourced from web product pages${webResult.source_type ? ` (${webResult.source_type})` : ""}${webResult.evidence_urls?.length ? `: ${webResult.evidence_urls.slice(0, 2).join(", ")}` : ""}.`,
                ingredients: normalizeStringArray(webResult.ingredients),
                allergens: normalizeStringArray(webResult.allergens),
            };
        }
    }
    catch {
        // Fall through to Open Food Facts fallback.
    }
    const barcode = sanitizeBarcode(item.barcode);
    const product = (barcode ? await lookupByBarcode(barcode) : null) || (await searchByName(item));
    if (!product?.nutriments)
        return null;
    const calories = caloriesFromNutriments(product.nutriments);
    const fat = pickNutriment(product.nutriments, ["fat"]);
    const saturatedFat = pickNutriment(product.nutriments, ["saturated-fat"]);
    const transFat = pickNutriment(product.nutriments, ["trans-fat"]);
    const cholesterol = pickNutriment(product.nutriments, ["cholesterol"]);
    const sodium = sodiumFromNutriments(product.nutriments);
    const carbs = pickNutriment(product.nutriments, ["carbohydrates"]);
    const fiber = pickNutriment(product.nutriments, ["fiber"]);
    const sugars = pickNutriment(product.nutriments, ["sugars"]);
    const protein = pickNutriment(product.nutriments, ["proteins", "protein"]);
    const calcium = pickNutriment(product.nutriments, ["calcium"]);
    const iron = pickNutriment(product.nutriments, ["iron"]);
    const sourceHints = [
        calories.source,
        fat.source,
        carbs.source,
        protein.source,
    ];
    const explicitServing = pickString(product.serving_size, item.serving_size);
    const servingGrams = parseServingGrams(explicitServing);
    const hasReliableServingData = sourceHints.includes("serving") || servingGrams != null;
    const hasMacros = calories.value != null ||
        fat.value != null ||
        carbs.value != null ||
        protein.value != null;
    if (!hasMacros || !hasReliableServingData)
        return null;
    const servingSize = buildServingSize(product, sourceHints, item);
    const ingredients = parseIngredients(product);
    const allergens = parseAllergens(product);
    const usedBarcode = Boolean(barcode && product.code && sanitizeBarcode(product.code) === barcode);
    const dishName = buildDishName(product, item);
    const confidence = usedBarcode ? 0.98 : 0.9;
    return {
        dish_name: dishName,
        serving_size: servingSize,
        calories: convertFrom100g(calories.value, calories.source, servingGrams),
        total_fat: convertFrom100g(fat.value, fat.source, servingGrams),
        saturated_fat: convertFrom100g(saturatedFat.value, saturatedFat.source, servingGrams),
        trans_fat: convertFrom100g(transFat.value, transFat.source, servingGrams),
        cholesterol: convertFrom100g(cholesterol.value, cholesterol.source, servingGrams),
        sodium: convertFrom100g(sodium.value, sodium.source, servingGrams),
        total_carbohydrates: convertFrom100g(carbs.value, carbs.source, servingGrams),
        dietary_fiber: convertFrom100g(fiber.value, fiber.source, servingGrams),
        sugars: convertFrom100g(sugars.value, sugars.source, servingGrams),
        protein: convertFrom100g(protein.value, protein.source, servingGrams),
        vitamin_a: null,
        vitamin_c: null,
        calcium: convertFrom100g(calcium.value, calcium.source, servingGrams),
        iron: convertFrom100g(iron.value, iron.source, servingGrams),
        confidence,
        explanation: `Nutrition sourced from Open Food Facts${usedBarcode ? " via barcode match" : ""}${servingSize ? ` for ${servingSize}` : ""}.`,
        ingredients,
        allergens,
    };
}
//# sourceMappingURL=packagedNutritionLookup.js.map