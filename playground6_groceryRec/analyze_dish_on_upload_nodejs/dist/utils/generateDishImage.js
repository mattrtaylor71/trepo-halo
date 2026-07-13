"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateDishImage = generateDishImage;
const openai_1 = __importDefault(require("openai"));
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
function inferServingVessel(dish, nutritionData) {
    const combinedText = [
        dish.dish_name,
        dish.category,
        dish.explanation,
        nutritionData.dish_name,
        nutritionData.explanation,
        ...(nutritionData.ingredients || []),
    ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
    if (dish.category?.toLowerCase() === 'beverage' ||
        /\b(coffee|tea|latte|espresso|cappuccino|smoothie|juice|soda|drink|beverage)\b/.test(combinedText)) {
        return 'Serve it in a simple white mug or clear glass on the same table instead of a plate.';
    }
    if (/\b(soup|stew|curry|ramen|noodles|oatmeal|porridge)\b/.test(combinedText)) {
        return 'Serve it in a simple white ceramic bowl on the same table.';
    }
    if (/\b(cookie|brownie|cake|pie|pastry|muffin|cupcake|dessert)\b/.test(combinedText)) {
        return 'Serve it on a simple white dessert plate on the same table.';
    }
    return 'Serve it on a simple white ceramic dinner plate on the same table.';
}
/**
 * Generates a realistic standardized food image using gpt-image-1
 * Uses all available dish context for accurate, detailed images
 * gpt-image-1 is OpenAI's best image generation model with highest quality and prompt adherence
 * Returns the image as a Buffer
 */
async function generateDishImage(dish, nutritionData) {
    const openai = getClient();
    // Build comprehensive dish description using all available context
    const parts = [];
    // Dish name
    if (dish.dish_name) {
        parts.push(dish.dish_name);
    }
    // Category and cuisine type for context
    if (dish.category) {
        parts.push(`(${dish.category})`);
    }
    if (dish.cuisine_type) {
        parts.push(`- ${dish.cuisine_type} cuisine`);
    }
    const dishDescription = parts.join(' ') || 'food dish';
    // Build additional context hints from ingredients/nutrition
    const contextHints = [];
    if (nutritionData.ingredients && nutritionData.ingredients.length > 0) {
        // Extract key visual characteristics from ingredients
        const ingredientsStr = nutritionData.ingredients.join(', ').toLowerCase();
        if (ingredientsStr.includes('chicken'))
            contextHints.push('chicken');
        if (ingredientsStr.includes('beef') || ingredientsStr.includes('steak'))
            contextHints.push('beef');
        if (ingredientsStr.includes('fish') || ingredientsStr.includes('salmon') || ingredientsStr.includes('tuna'))
            contextHints.push('seafood');
        if (ingredientsStr.includes('pasta') || ingredientsStr.includes('spaghetti') || ingredientsStr.includes('noodles'))
            contextHints.push('pasta');
        if (ingredientsStr.includes('rice'))
            contextHints.push('rice');
        if (ingredientsStr.includes('cheese'))
            contextHints.push('cheese');
        if (ingredientsStr.includes('tomato') || ingredientsStr.includes('tomatoes'))
            contextHints.push('tomato');
        if (ingredientsStr.includes('lettuce') || ingredientsStr.includes('salad'))
            contextHints.push('fresh greens');
        if (ingredientsStr.includes('chocolate'))
            contextHints.push('chocolate');
        if (ingredientsStr.includes('berry') || ingredientsStr.includes('strawberry'))
            contextHints.push('berries');
    }
    const vesselInstruction = inferServingVessel(dish, nutritionData);
    // Build detailed prompt for realistic, consistent food photography
    let prompt = `Create a realistic food photograph of ${dishDescription}`;
    // Add visual context hints
    if (contextHints.length > 0) {
        prompt += ` featuring ${contextHints.join(', ')}`;
    }
    prompt += `. 
  
Style: Photorealistic food photography. Real textures, real ingredients, natural colors, realistic plating.
Do NOT make it look like an illustration, cartoon, comic, animation frame, painting, or 3D render.
Use the same standardized setting for every image: a clean neutral tabletop, soft natural lighting, minimal styling, and a slightly angled close food-photo composition.
${vesselInstruction}
Keep the item centered and clearly visible as the main focus.
Accurately match the described meal/item so it looks like the actual food or drink being named.
Include only natural visual elements that support the identified item; do not add unrelated side dishes or decorations.
NO TEXT. NO WORDS. NO LABELS. NO WRITING OF ANY KIND.
High detail, professional quality, suitable for use in a premium food tracking app.`;
    try {
        console.log('[generateDishImage] Generating realistic dish image for:', dishDescription);
        if (contextHints.length > 0) {
            console.log('[generateDishImage] Context hints:', contextHints.join(', '));
        }
        // Use gpt-image-1 - OpenAI's best image generation model
        // Optimized for app performance: JPEG format with compression, medium quality
        const response = await openai.images.generate({
            model: 'gpt-image-1',
            prompt: prompt,
            size: '1024x1024', // Smallest available size for gpt-image-1
            quality: 'medium', // Medium quality - good balance of quality and file size
            output_format: 'jpeg', // JPEG is much smaller than PNG and loads faster
            output_compression: 75, // 75% compression - good quality with smaller file size
            n: 1,
        });
        // gpt-image-1 response structure: { created, data: [{ b64_json, ... }] }
        let imageBase64;
        if (response.data && Array.isArray(response.data) && response.data.length > 0) {
            imageBase64 = response.data[0]?.b64_json;
        }
        if (!imageBase64) {
            console.error('[generateDishImage] No base64 image data returned. Response structure:', JSON.stringify({
                hasData: !!response.data,
                dataLength: Array.isArray(response.data) ? response.data.length : 'not array',
                dataType: typeof response.data,
                keys: Object.keys(response || {}),
                firstDataItem: Array.isArray(response.data) && response.data.length > 0 ? Object.keys(response.data[0] || {}) : 'no items'
            }, null, 2));
            return null;
        }
        console.log('[generateDishImage] Image generated, decoding base64 data...');
        // Decode the base64 image data
        // Image is already optimized: JPEG format with 75% compression, medium quality
        const imageBuffer = Buffer.from(imageBase64, 'base64');
        console.log('[generateDishImage] Optimized JPEG image size:', imageBuffer.length, 'bytes');
        return imageBuffer;
    }
    catch (error) {
        console.error('[generateDishImage] Error generating dish image:', error);
        // Return null on error - non-fatal, we can continue without the image
        return null;
    }
}
//# sourceMappingURL=generateDishImage.js.map