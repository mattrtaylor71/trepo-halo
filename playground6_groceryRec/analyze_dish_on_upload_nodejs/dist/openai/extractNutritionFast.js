"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FastNutritionSchema = void 0;
exports.extractNutritionFast = extractNutritionFast;
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
exports.FastNutritionSchema = zod_1.z.object({
    dish_name: zod_1.z.string().nullable(),
    calories: zod_1.z.number().nullable(),
    protein_g: zod_1.z.number().nullable(),
    carbs_g: zod_1.z.number().nullable(),
    fat_g: zod_1.z.number().nullable(),
    confidence: zod_1.z.union([
        zod_1.z.number().min(0).max(1),
        zod_1.z.string().transform((s) => {
            const num = parseFloat(s);
            return isNaN(num) ? 0.5 : Math.max(0, Math.min(1, num));
        }),
        zod_1.z.object({}).passthrough().transform(() => 0.5),
        zod_1.z.any().transform(() => 0.5),
    ]),
    summary: zod_1.z.string().nullable(),
});
const fastNutritionJsonSchema = {
    type: "object",
    additionalProperties: false,
    required: ["dish_name", "calories", "protein_g", "carbs_g", "fat_g", "confidence", "summary"],
    properties: {
        dish_name: { type: ["string", "null"] },
        calories: { type: ["number", "null"] },
        protein_g: { type: ["number", "null"] },
        carbs_g: { type: ["number", "null"] },
        fat_g: { type: ["number", "null"] },
        confidence: { type: "number", minimum: 0, maximum: 1 },
        summary: { type: ["string", "null"] },
    },
};
function getFastSettings() {
    const model = process.env.FAST_DISH_MODEL || getDefaultModel();
    const maxTokensRaw = process.env.FAST_DISH_MAX_TOKENS || "350";
    const temperatureRaw = process.env.FAST_DISH_TEMPERATURE || "0.2";
    const imageDetail = (process.env.FAST_DISH_IMAGE_DETAIL || "low").toLowerCase();
    const maxTokens = Math.max(32, parseInt(maxTokensRaw, 10) || 350);
    const temperature = Math.min(1, Math.max(0, parseFloat(temperatureRaw) || 0.2));
    const detail = imageDetail === "low" ? "low" : imageDetail === "high" ? "high" : undefined;
    return { model, maxTokens, temperature, detail };
}
async function extractNutritionFast(imageBuffer) {
    const openai = getClient();
    const { model, maxTokens, temperature, detail } = getFastSettings();
    const systemPrompt = `You are a fast food nutrition estimator. Analyze meals, snacks, drinks, and edible packaged items a user may be consuming. Return STRICT JSON only with this structure:
{
  "dish_name": string | null,
  "calories": number | null,
  "protein_g": number | null,
  "carbs_g": number | null,
  "fat_g": number | null,
  "confidence": number (0.0 to 1.0),
  "summary": string | null
}
Rules:
1) Provide quick estimates; prioritize speed over detail.
2) Valid items include prepared dishes plus edible packaged items/drinks like yogurt cups, canned drinks, snack bags, bars, and desserts.
3) If the visible item is likely single-serving, estimate macros for the whole visible item.
4) If the visible item is likely multi-serving/bulk, estimate macros for one serving rather than the whole package.
5) If no edible food or drink item is visible, return null for dish_name and all macros, set confidence <= 0.2, and summary "No dish detected."
6) If unsure whether an edible item is present, treat it as no dish (use rule 5).
7) summary: one short, direct description (no "The image shows").`;
    const userPrompt = `Estimate the primary edible item name and macros (calories, protein_g, carbs_g, fat_g).
This can be a prepared dish or an edible packaged item/drink the user may be consuming.
If no edible food or drink item is visible, return null for dish_name and all macros, with summary "No dish detected."
Return ONLY JSON.`;
    const response = await openai.responses.create({
        model,
        ...(supportsReasoning(model) ? { reasoning: { effort: "low" } } : {}),
        max_output_tokens: maxTokens,
        text: {
            format: {
                type: "json_schema",
                name: "fast_nutrition_estimate",
                strict: true,
                schema: fastNutritionJsonSchema,
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
    return exports.FastNutritionSchema.parse(parsed);
}
//# sourceMappingURL=extractNutritionFast.js.map