"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DishSchema = void 0;
exports.identifyDish = identifyDish;
const openai_1 = __importDefault(require("openai"));
const zod_1 = require("zod");
let client = null;
function getClient() {
    if (!client) {
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('OPENAI_API_KEY environment variable is required');
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
    // reasoning.effort is only supported by o-series and gpt-5+ models
    return /^(o[1-9]|gpt-5)/.test(model);
}
function parseJsonResponse(response) {
    const outputText = response?.output_text;
    if (typeof outputText !== "string" || !outputText.trim()) {
        throw new Error("No structured output returned from OpenAI");
    }
    return JSON.parse(outputText);
}
exports.DishSchema = zod_1.z.object({
    dish_name: zod_1.z.string().nullable(),
    category: zod_1.z.string().nullable().optional(),
    cuisine_type: zod_1.z.string().nullable().optional(),
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
});
const dishJsonSchema = {
    type: "object",
    additionalProperties: false,
    required: ["dish_name", "category", "cuisine_type", "confidence", "explanation"],
    properties: {
        dish_name: { type: ["string", "null"] },
        category: { type: ["string", "null"] },
        cuisine_type: { type: ["string", "null"] },
        confidence: { type: "number", minimum: 0, maximum: 1 },
        explanation: { type: ["string", "null"] },
    },
};
async function identifyDish(imageBuffer, userCorrection, existingContext) {
    const openai = getClient();
    const model = process.env.FULL_DISH_MODEL || getDefaultModel();
    const maxTokensRaw = process.env.FULL_DISH_MAX_TOKENS || "1200";
    const temperatureRaw = process.env.FULL_DISH_TEMPERATURE || "0.3";
    const imageDetail = (process.env.FULL_DISH_IMAGE_DETAIL || "high").toLowerCase();
    const maxTokens = Math.max(256, parseInt(maxTokensRaw, 10) || 1200);
    const temperature = Math.min(1, Math.max(0, parseFloat(temperatureRaw) || 0.3));
    const detail = imageDetail === "low" ? "low" : imageDetail === "high" ? "high" : undefined;
    const systemPrompt = `You are an expert food intake identification system. Analyze images of meals, snacks, drinks, and edible packaged items a user might want to log, and provide comprehensive information about the primary item being consumed.

Return STRICT JSON only with this exact structure:
{
  "dish_name": string | null,
  "category": string | null (e.g., "Main Course", "Dessert", "Appetizer", "Salad", "Soup", "Breakfast", "Snack", "Beverage"),
  "cuisine_type": string | null (e.g., "Italian", "Asian", "Mexican", "American", "Mediterranean"),
  "confidence": number (0.0 to 1.0),
  "explanation": string | null
}

CRITICAL RULES:
1. confidence MUST be a NUMBER between 0.0 and 1.0
2. Identify the specific food/drink item being consumed. This can be a prepared dish OR a directly edible packaged item/beverage the user would reasonably log (e.g., "Spaghetti Carbonara", "Caesar Salad", "Red Bull Energy Drink", "Strawberry yogurt cup", "Small bag of Doritos").
3. Provide category and cuisine type if identifiable
4. Be specific and accurate - use your knowledge of global cuisines and common packaged food/drink products
5. If multiple edible items are visible, identify the main/primary thing being consumed
6. Treat prepared meals, snacks, desserts, canned/bottled drinks, yogurt cups, protein bars, chip bags, and similar edible consumer items as valid loggable items
7. If an item appears edible but is clearly a large multi-serving package, still identify the product itself rather than returning null
8. If no edible food or drink item is visible, set dish_name, category, cuisine_type to null, confidence <= 0.2, and explanation "No dish detected." Do NOT guess.
9. explanation: A short, user-facing description of the meal/item for the app (e.g. "Creamy salad with chicken, celery and almonds" or "Single can of energy drink"). Write as a direct description—do NOT start with "The image shows", "This is", or "This dish is". Start with the item itself.`;
    let userPrompt = `Identify the primary edible food or drink item from this image. Provide:
- Specific dish/product name
- Category (Main Course, Dessert, Appetizer, Snack, Beverage, etc.)
- Cuisine type if identifiable
- Confidence level
- explanation: One short sentence describing the meal/item for the user (direct description, e.g. "Creamy salad with chicken and almonds" or "Single can of energy drink"—never start with "The image shows" or "This is")

Valid loggable items include prepared dishes plus edible packaged items or drinks a user may be consuming.

If no edible food or drink item is visible, return null for dish_name/category/cuisine_type, confidence <= 0.2, and explanation "No dish detected."

Be thorough and accurate. Return ONLY JSON.`;
    if (existingContext && (existingContext.dish_name || existingContext.explanation)) {
        const current = [
            existingContext.dish_name ? `dish_name: ${existingContext.dish_name}` : '',
            existingContext.explanation ? `explanation: ${existingContext.explanation}` : '',
        ].filter(Boolean).join('; ');
        userPrompt += `\n\nCURRENT CHARACTERIZATION (from previous user corrections—you MUST preserve this and only add/update per the new correction): ${current}.`;
    }
    if (userCorrection && userCorrection.trim()) {
        userPrompt += `\n\nUSER'S NEW CORRECTION: "${userCorrection.trim()}". Update the dish characterization to incorporate this new correction while keeping all previous characterizations (e.g. if current is "turkey salad with coffee" and they say "I also had 2 cookies", return turkey salad with coffee and cookies—do not revert to chicken or drop the coffee). Return ONLY JSON.`;
    }
    try {
        const response = await openai.responses.create({
            model,
            ...(supportsReasoning(model) ? { reasoning: { effort: "low" } } : {}),
            max_output_tokens: maxTokens,
            text: {
                format: {
                    type: "json_schema",
                    name: "dish_identification",
                    strict: true,
                    schema: dishJsonSchema,
                },
            },
            input: [
                {
                    role: "system",
                    content: [{ type: "input_text", text: systemPrompt }],
                },
                {
                    role: "user",
                    content: [
                        { type: "input_text", text: userPrompt },
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
        return exports.DishSchema.parse(parsed);
    }
    catch (error) {
        console.error('[IdentifyDish] Error identifying dish:', error);
        throw error;
    }
}
//# sourceMappingURL=identifyDish.js.map