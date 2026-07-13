"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PackagedItemSchema = void 0;
exports.identifyPackagedItem = identifyPackagedItem;
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
exports.PackagedItemSchema = zod_1.z.object({
    is_packaged_item: zod_1.z.boolean(),
    brand: zod_1.z.string().nullable().optional(),
    product_name: zod_1.z.string().nullable().optional(),
    variant: zod_1.z.string().nullable().optional(),
    category: zod_1.z.string().nullable().optional(),
    barcode: zod_1.z.string().nullable().optional(),
    serving_size: zod_1.z.string().nullable().optional(),
    confidence: zod_1.z.number().min(0).max(1),
    explanation: zod_1.z.string().nullable().optional(),
});
const packagedItemJsonSchema = {
    type: "object",
    additionalProperties: false,
    required: [
        "is_packaged_item",
        "brand",
        "product_name",
        "variant",
        "category",
        "barcode",
        "serving_size",
        "confidence",
        "explanation",
    ],
    properties: {
        is_packaged_item: { type: "boolean" },
        brand: { type: ["string", "null"] },
        product_name: { type: ["string", "null"] },
        variant: { type: ["string", "null"] },
        category: { type: ["string", "null"] },
        barcode: { type: ["string", "null"] },
        serving_size: { type: ["string", "null"] },
        confidence: { type: "number", minimum: 0, maximum: 1 },
        explanation: { type: ["string", "null"] },
    },
};
function normalizeBarcode(value) {
    if (!value)
        return null;
    const digits = value.replace(/\D/g, "");
    return digits.length >= 8 ? digits : null;
}
async function identifyPackagedItem(imageBuffer) {
    const openai = getClient();
    const model = process.env.FULL_DISH_MODEL || getDefaultModel();
    const response = await openai.responses.create({
        model,
        ...(supportsReasoning(model) ? { reasoning: { effort: "low" } } : {}),
        max_output_tokens: 900,
        text: {
            format: {
                type: "json_schema",
                name: "packaged_item_identification",
                strict: true,
                schema: packagedItemJsonSchema,
            },
        },
        input: [
            {
                role: "system",
                content: [{
                        type: "input_text",
                        text: `You identify commercial edible packaged products in food log images.

Return STRICT JSON only with this exact structure:
{
  "is_packaged_item": boolean,
  "brand": string | null,
  "product_name": string | null,
  "variant": string | null,
  "category": string | null,
  "barcode": string | null,
  "serving_size": string | null,
  "confidence": number,
  "explanation": string | null
}

CRITICAL RULES:
1. Set is_packaged_item=true only when the image likely shows a branded commercial edible item or beverage the user may be consuming.
2. Examples: soda can, gum pack, yogurt cup, protein bar, candy bag, chips bag, bottled drink, snack cup.
3. For plated meals, homemade dishes, restaurant dishes, or loose unpackaged foods, set is_packaged_item=false.
4. barcode must only be returned when enough digits are actually visible to be useful. Do not guess barcode digits.
5. product_name should be the product itself, not a generic class, when identifiable.
6. confidence must be between 0.0 and 1.0.
7. Return ONLY JSON.`,
                    }],
            },
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: `Analyze this image and determine whether it is a commercial packaged edible item or beverage.

If yes, extract:
- brand
- product_name
- variant
- category
- barcode if visible
- serving_size if visible on packaging
- explanation

If no, return is_packaged_item=false with null fields and a short explanation.

Return ONLY JSON.`,
                    },
                    {
                        type: "input_image",
                        image_url: `data:image/jpeg;base64,${imageBuffer.toString("base64")}`,
                        detail: "high",
                    },
                ],
            },
        ],
    });
    const parsed = parseJsonResponse(response);
    const result = exports.PackagedItemSchema.parse(parsed);
    return {
        ...result,
        barcode: normalizeBarcode(result.barcode),
    };
}
//# sourceMappingURL=identifyPackagedItem.js.map