"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.NutritionSchema = void 0;
exports.extractNutrition = extractNutrition;
const openai_1 = __importDefault(require("openai"));
const zod_1 = require("zod");
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
exports.NutritionSchema = zod_1.z.object({
    dish_name: zod_1.z.string().nullable().optional(),
    serving_size: zod_1.z.string().nullable().optional(),
    calories: zod_1.z.number().nullable().optional(),
    total_fat: zod_1.z.number().nullable().optional(),
    saturated_fat: zod_1.z.number().nullable().optional(),
    trans_fat: zod_1.z.number().nullable().optional(),
    cholesterol: zod_1.z.number().nullable().optional(),
    sodium: zod_1.z.number().nullable().optional(),
    total_carbohydrates: zod_1.z.number().nullable().optional(),
    dietary_fiber: zod_1.z.number().nullable().optional(),
    sugars: zod_1.z.number().nullable().optional(),
    protein: zod_1.z.number().nullable().optional(),
    vitamin_a: zod_1.z.number().nullable().optional(),
    vitamin_c: zod_1.z.number().nullable().optional(),
    calcium: zod_1.z.number().nullable().optional(),
    iron: zod_1.z.number().nullable().optional(),
    confidence: zod_1.z.union([
        zod_1.z.number().min(0).max(1),
        zod_1.z.string().transform((s) => {
            const num = parseFloat(s);
            return isNaN(num) ? 0.5 : Math.max(0, Math.min(1, num));
        }),
        zod_1.z.object({}).passthrough().transform(() => 0.5),
        zod_1.z.any().transform(() => 0.5),
    ]),
    explanation: zod_1.z.string().nullable().optional(),
    ingredients: zod_1.z.array(zod_1.z.string()).optional().default([]),
    allergens: zod_1.z.array(zod_1.z.string()).optional().default([]),
});
const nutritionJsonSchema = {
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
        ingredients: {
            type: "array",
            items: { type: "string" },
        },
        allergens: {
            type: "array",
            items: { type: "string" },
        },
    },
};
async function extractNutrition(imageBuffer, userCorrection, existingContext) {
    const openai = getClient();
    const model = process.env.FULL_DISH_MODEL || getDefaultModel();
    const maxTokensRaw = process.env.FULL_DISH_MAX_TOKENS || "2000";
    const temperatureRaw = process.env.FULL_DISH_TEMPERATURE || "0.3";
    const imageDetail = (process.env.FULL_DISH_IMAGE_DETAIL || "high").toLowerCase();
    const maxTokens = Math.max(512, parseInt(maxTokensRaw, 10) || 2000);
    const temperature = Math.min(1, Math.max(0, parseFloat(temperatureRaw) || 0.3));
    const detail = imageDetail === "low" ? "low" : imageDetail === "high" ? "high" : undefined;
    let userText = `Analyze this image for an edible food or drink item the user may be consuming and ESTIMATE the complete nutritional information if one is visible.
- Valid items include prepared dishes, snacks, desserts, drinks, and edible packaged products such as yogurt cups, canned drinks, chip bags, snack bars, and similar consumer items.
- If no edible food or drink item is visible, return null for all nutrition fields and "No dish detected." as explanation.
- If a nutrition label is visible, use those exact values.
- If no label is visible, ESTIMATE based on visible ingredients, portion size, and standard nutritional values.
- Provide estimates for calories, macros (protein, carbs, fats), and micronutrients.
- Estimate serving size based on what's visible.
- If the visible item is likely a single-serving package/container (e.g. one can, one bottle, one yogurt cup, one candy bar, one small snack bag), estimate nutrition for the whole visible item and set serving_size accordingly.
- If the visible item is likely a bulk or multi-serving package (e.g. large yogurt tub, family-size chip bag, large carton, multi-serve bottle), estimate nutrition for ONE serving, not the whole package, and make that explicit in serving_size.
- List all visible ingredients.
- Identify allergens.
- Set confidence based on how certain you are (higher if label visible, lower if estimating).`;
    if (existingContext && (existingContext.dish_name || existingContext.serving_size || existingContext.ingredients || existingContext.allergens)) {
        const ing = existingContext.ingredients;
        const ingStr = Array.isArray(ing) ? ing.join(', ') : (typeof ing === 'string' ? (ing.startsWith('[') ? ing : ing) : '');
        const all = existingContext.allergens;
        const allStr = Array.isArray(all) ? all.join(', ') : (typeof all === 'string' ? all : '');
        const parts = [
            existingContext.dish_name ? `dish_name: ${existingContext.dish_name}` : '',
            existingContext.serving_size ? `serving_size: ${existingContext.serving_size}` : '',
            ingStr ? `ingredients: ${ingStr}` : '',
            allStr ? `allergens: ${allStr}` : '',
        ].filter(Boolean);
        if (parts.length) {
            userText += `\n\nCURRENT NUTRITION CHARACTERIZATION (from previous user corrections—you MUST preserve these and only add/update per the new correction): ${parts.join('; ')}.`;
        }
    }
    if (userCorrection && userCorrection.trim()) {
        userText += `\n\nUSER'S NEW CORRECTION: "${userCorrection.trim()}". Update nutrition to incorporate this new correction while keeping all previous characterizations (e.g. if current is turkey salad with coffee and they say "I also had 2 cookies", include turkey salad, coffee, AND cookies—do not revert to chicken or drop coffee).`;
    }
    const response = await openai.responses.create({
        model,
        ...(supportsReasoning(model) ? { reasoning: { effort: "low" } } : {}),
        max_output_tokens: maxTokens,
        text: {
            format: {
                type: "json_schema",
                name: "nutrition_estimate",
                strict: true,
                schema: nutritionJsonSchema,
            },
        },
        input: [
            {
                role: "system",
                content: [{
                        type: "input_text",
                        text: `You are a nutrition analysis expert. Analyze images of meals, snacks, drinks, and edible packaged items, and ESTIMATE nutritional information based on visible ingredients, packaging context, and portion sizes.

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
  "confidence": number (0.0 to 1.0),
  "explanation": string | null,
  "ingredients": string[],
  "allergens": string[]
}

CRITICAL RULES:
1. confidence MUST be a NUMBER between 0.0 and 1.0, NOT an object or string.
2. Treat prepared dishes AND edible packaged items/drinks a user may consume as valid loggable items.
3. If no edible food or drink item is visible, set dish_name and all nutrition fields to null, ingredients/allergens to empty arrays, confidence <= 0.2, and explanation "No dish detected." Do NOT guess.
4. If an edible item is visible but no nutrition label is visible, estimate based on:
   - Visible ingredients and their typical nutritional values
   - Estimated portion size (use standard serving sizes)
   - Common preparation methods
5. For packaged items, infer whether the visible package is single-serving or multi-serving:
   - Single-serving examples: one can/bottle intended for one sitting, one yogurt cup, one snack-size chip bag, one bar, one pastry
   - Multi-serving examples: large yogurt tub, family-size chip bag, large cereal box, large juice bottle, bulk snack tub
   - If likely single-serving, estimate/store nutrition for the whole visible item
   - If likely multi-serving, estimate/store nutrition for ONE serving and reflect that in serving_size
   - If a visible label provides per-serving values and the package is multi-serving, use one serving rather than the whole package
6. Use your knowledge of food nutrition to estimate macros:
   - Estimate calories based on visible ingredients and portion
   - Estimate protein, carbs, fats based on ingredient composition
   - Estimate vitamins/minerals based on visible vegetables/fruits
7. Set confidence based on certainty:
   - 0.9+ if nutrition label is clearly visible
   - 0.7-0.8 if ingredients are clear and portion is estimable
   - 0.5-0.6 if ingredients are partially visible
   - Lower if very uncertain
8. Include ALL visible or clearly inferable ingredients in the ingredients array
9. Include allergens (dairy, nuts, gluten, etc.) if visible or clearly inferable from the product/dish
10. Provide serving_size estimate (e.g., "1 bowl", "1 plate", "1 can", "1 yogurt cup", "1 serving (about 28g)", "1 serving (about 170g)")
11. explanation: A short, user-facing description of the meal/item (e.g. "Creamy salad with chicken, celery and almonds" or "Single can of energy drink"). Write as a direct description—do NOT start with "The image shows", "This is", or "This dish is". Start with the item itself. Optionally note if values are estimated vs from label at the end.

IMPORTANT: Provide estimates for calories, protein, carbs, and fats when an edible item is visible. If no edible item is visible, follow rule 3.`,
                    }],
            },
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: userText,
                    },
                    {
                        type: "input_image",
                        image_url: `data:image/jpeg;base64,${imageBuffer.toString("base64")}`,
                        ...(detail ? { detail } : {}),
                    },
                ],
            },
        ],
    });
    const parsed = parseJsonResponse(response);
    try {
        return exports.NutritionSchema.parse(parsed);
    }
    catch (error) {
        console.error('[ExtractNutrition] Schema validation error:', error);
        console.error('[ExtractNutrition] Received data:', JSON.stringify(parsed, null, 2));
        // Try to fix and retry
        if (parsed.confidence && typeof parsed.confidence !== 'number') {
            parsed.confidence = 0.5;
            try {
                return exports.NutritionSchema.parse(parsed);
            }
            catch (retryError) {
                console.error('[ExtractNutrition] Retry also failed:', retryError);
                throw error;
            }
        }
        throw error;
    }
}
//# sourceMappingURL=extractNutrition.js.map