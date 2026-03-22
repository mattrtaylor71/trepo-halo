import OpenAI from "openai";
import { z } from "zod";

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY environment variable is required");
    }
    client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
  return client;
}

function getOpenAIModel(): string {
  return process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
}

function parseJsonResponse<T>(response: any): T {
  const outputText = response?.output_text;
  if (typeof outputText !== "string" || !outputText.trim()) {
    throw new Error("No structured output returned from OpenAI");
  }
  return JSON.parse(outputText) as T;
}

export const GroceryItemSchema = z.object({
  brand: z.string().nullable().optional(),
  product_name: z.string().nullable().optional(),
  variant: z.string().nullable().optional(),
  category: z.string().nullable().optional(),
  estimated_price: z.string().nullable().optional(),
  ingredients: z.array(z.string()).optional().default([]),
  nutrition_summary: z.string().nullable().optional(),
  upf: z.enum(['yes', 'no']).nullable().optional(),
  harmful_ingredients: z.array(z.string()).optional().default([]),
  similar_items: z.array(z.object({
    name: z.string(),
    brand: z.string().nullable().optional(),
    reason: z.string(),
  })).optional().default([]),
  alternatives: z.array(z.object({
    name: z.string(),
    brand: z.string().nullable().optional(),
    reason: z.string(),
  })).optional().default([]),
  healthier_alternatives: z.array(z.object({
    name: z.string(),
    brand: z.string().nullable().optional(),
    why_healthier: z.string(),
    trade_offs: z.string().nullable().optional(),
  })).optional().default([]),
  confidence: z.number().min(0).max(1),
  explanation: z.string().nullable().optional(),
  barcode: z.string().nullable().optional(),
  country_guess: z.string().nullable().optional(),
  /** One sentence: what this product actually is (e.g. to avoid confusing "Ice Cubes Gum" with ice). */
  product_description: z.string().nullable().optional(),
});

export type GroceryItem = z.infer<typeof GroceryItemSchema>;

const IngredientLookupSchema = z.object({
  ingredients: z.array(z.string()).optional().default([]),
  upf: z.enum(['yes', 'no']),
  harmful_ingredients: z.array(z.string()).optional().default([]),
});

type IngredientLookup = z.infer<typeof IngredientLookupSchema>;

const ingredientLookupJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["ingredients", "upf", "harmful_ingredients"],
  properties: {
    ingredients: {
      type: "array",
      items: { type: "string" },
    },
    upf: {
      type: "string",
      enum: ["yes", "no"],
    },
    harmful_ingredients: {
      type: "array",
      items: { type: "string" },
    },
  },
};

const groceryItemJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "brand",
    "product_name",
    "variant",
    "category",
    "estimated_price",
    "ingredients",
    "nutrition_summary",
    "upf",
    "harmful_ingredients",
    "similar_items",
    "alternatives",
    "healthier_alternatives",
    "confidence",
    "explanation",
    "barcode",
    "country_guess",
    "product_description",
  ],
  properties: {
    brand: { type: ["string", "null"] },
    product_name: { type: ["string", "null"] },
    variant: { type: ["string", "null"] },
    category: { type: ["string", "null"] },
    estimated_price: { type: ["string", "null"] },
    ingredients: {
      type: "array",
      items: { type: "string" },
    },
    nutrition_summary: { type: ["string", "null"] },
    upf: {
      type: ["string", "null"],
      enum: ["yes", "no", null],
    },
    harmful_ingredients: {
      type: "array",
      items: { type: "string" },
    },
    similar_items: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "brand", "reason"],
        properties: {
          name: { type: "string" },
          brand: { type: ["string", "null"] },
          reason: { type: "string" },
        },
      },
    },
    alternatives: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "brand", "reason"],
        properties: {
          name: { type: "string" },
          brand: { type: ["string", "null"] },
          reason: { type: "string" },
        },
      },
    },
    healthier_alternatives: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "brand", "why_healthier", "trade_offs"],
        properties: {
          name: { type: "string" },
          brand: { type: ["string", "null"] },
          why_healthier: { type: "string" },
          trade_offs: { type: ["string", "null"] },
        },
      },
    },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    explanation: { type: ["string", "null"] },
    barcode: { type: ["string", "null"] },
    country_guess: { type: ["string", "null"] },
    product_description: { type: ["string", "null"] },
  },
};

const HARMFUL_INGREDIENT_RULES: { name: string; patterns: string[] }[] = [
  { name: 'aspartame', patterns: ['aspartame'] },
  { name: 'acesulfame potassium', patterns: ['acesulfame potassium', 'acesulfame k', 'acesulfame-k'] },
  { name: 'sucralose', patterns: ['sucralose'] },
  { name: 'saccharin', patterns: ['saccharin'] },
  { name: 'neotame', patterns: ['neotame'] },
  { name: 'bht', patterns: ['bht', 'butylated hydroxytoluene'] },
  { name: 'bha', patterns: ['bha', 'butylated hydroxyanisole'] },
  { name: 'tbqh', patterns: ['tbqh', 'tb hq', 'tert-butylhydroquinone', 'tbhq'] },
  { name: 'red 40', patterns: ['red 40', 'red#40', 'red no. 40', 'red 40 lake'] },
  { name: 'red 3', patterns: ['red 3', 'red#3', 'red no. 3', 'red 3 lake'] },
  { name: 'yellow 5', patterns: ['yellow 5', 'yellow#5', 'yellow no. 5', 'yellow 5 lake'] },
  { name: 'yellow 6', patterns: ['yellow 6', 'yellow#6', 'yellow no. 6', 'yellow 6 lake'] },
  { name: 'blue 1', patterns: ['blue 1', 'blue#1', 'blue no. 1', 'blue 1 lake'] },
  { name: 'blue 2', patterns: ['blue 2', 'blue#2', 'blue no. 2', 'blue 2 lake'] },
  { name: 'green 3', patterns: ['green 3', 'green#3', 'green no. 3', 'green 3'] },
  { name: 'titanium dioxide', patterns: ['titanium dioxide'] },
  { name: 'sodium nitrite', patterns: ['sodium nitrite'] },
  { name: 'sodium nitrate', patterns: ['sodium nitrate'] },
  { name: 'potassium nitrate', patterns: ['potassium nitrate'] },
  { name: 'propylene glycol', patterns: ['propylene glycol'] },
  { name: 'polysorbate 80', patterns: ['polysorbate 80'] },
  { name: 'carrageenan', patterns: ['carrageenan'] },
  { name: 'monosodium glutamate', patterns: ['monosodium glutamate', 'msg'] },
  { name: 'sodium benzoate', patterns: ['sodium benzoate'] },
  { name: 'potassium benzoate', patterns: ['potassium benzoate'] },
  { name: 'high fructose corn syrup', patterns: ['high fructose corn syrup'] },
];

function normalizeUpf(value: unknown): 'yes' | 'no' {
  if (typeof value === 'string') {
    const upf = value.trim().toLowerCase();
    if (['yes', 'y', 'true', '1'].includes(upf)) return 'yes';
    if (['no', 'n', 'false', '0'].includes(upf)) return 'no';
  }
  return 'no';
}

function normalizeStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item: unknown) => (item == null ? '' : String(item).trim()))
    .filter((item: string) => item.length > 0);
}

function normalizeForMatch(value: string): string {
  return value
    .toLowerCase()
    .replace(/[#.]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function deriveHarmfulIngredients(ingredients: string[]): string[] {
  const found = new Set<string>();
  const normalizedIngredients = ingredients.map((item) => normalizeForMatch(item));

  for (const rule of HARMFUL_INGREDIENT_RULES) {
    const patterns = rule.patterns.map((pattern) => normalizeForMatch(pattern));
    const matched = normalizedIngredients.some((ingredient) =>
      patterns.some((pattern) => ingredient.includes(pattern))
    );
    if (matched) {
      found.add(rule.name);
    }
  }

  return Array.from(found);
}

function mergeUniqueList(primary: string[], secondary: string[]): string[] {
  const seen = new Set(primary.map(item => item.toLowerCase()));
  const merged = [...primary];
  for (const item of secondary) {
    const key = item.toLowerCase();
    if (!seen.has(key)) {
      merged.push(item);
      seen.add(key);
    }
  }
  return merged;
}

async function lookupIngredientsByName(
  openai: OpenAI,
  details: Pick<GroceryItem, 'product_name' | 'brand' | 'variant' | 'category'>
): Promise<IngredientLookup | null> {
  if (!details.product_name) return null;

  const systemPrompt = `You are a grocery ingredient lookup system. Given a product name, brand, and category, return the FULL ingredient list as it appears on packaging.

Return STRICT JSON only with this exact structure:
{
  "ingredients": string[],
  "upf": "yes" | "no",
  "harmful_ingredients": string[]
}

CRITICAL RULES:
1. ingredients must be the full list (no summaries, no omissions). If unsure, return your best complete list.
2. upf must be EXACTLY "yes" or "no" (lowercase only).
3. harmful_ingredients MUST be derived from the ingredients list and must be a subset of it (e.g., red 40, aspartame). If none, return an empty array.
4. Return ONLY JSON with no extra keys or text.`;

  const userPrompt = `Find the full ingredient list for this grocery product:
- Product: ${details.product_name}
- Brand: ${details.brand || 'Unknown'}
- Variant: ${details.variant || 'Unknown'}
- Category: ${details.category || 'Unknown'}

Return ONLY JSON.`;

  const response = await openai.responses.create({
    model: getOpenAIModel(),
    reasoning: { effort: "low" },
    max_output_tokens: 1000,
    text: {
      format: {
        type: "json_schema",
        name: "ingredient_lookup",
        strict: true,
        schema: ingredientLookupJsonSchema,
      },
    } as any,
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }],
      },
      {
        role: "user",
        content: [{ type: "input_text", text: userPrompt }],
      },
    ],
  } as any);

  let parsed: any;
  try {
    parsed = parseJsonResponse<IngredientLookup>(response);
  } catch {
    return null;
  }

  parsed.ingredients = normalizeStringList(parsed.ingredients);
  parsed.harmful_ingredients = normalizeStringList(parsed.harmful_ingredients);
  parsed.upf = normalizeUpf(parsed.upf);

  return IngredientLookupSchema.parse(parsed);
}

export async function identifyGroceryItem(imageBuffer: Buffer): Promise<GroceryItem> {
  const openai = getClient();
  
  const systemPrompt = `You are an expert grocery product identification and analysis system. Analyze food product images and provide comprehensive information.

Return STRICT JSON only with this exact structure:
{
  "brand": string | null,
  "product_name": string | null,
  "variant": string | null,
  "category": string | null,
  "estimated_price": string | null (e.g., "$3.99", "$5.50-$7.00"),
  "ingredients": string[],
  "nutrition_summary": string | null,
  "upf": "yes" | "no",
  "harmful_ingredients": string[],
  "similar_items": [
    {
      "name": string,
      "brand": string | null,
      "reason": string (why it's similar)
    }
  ],
  "alternatives": [
    {
      "name": string,
      "brand": string | null,
      "reason": string (why it's a good alternative)
    }
  ],
  "healthier_alternatives": [
    {
      "name": string,
      "brand": string | null,
      "why_healthier": string,
      "trade_offs": string | null (what you might miss)
    }
  ],
  "confidence": number (0.0 to 1.0),
  "explanation": string | null,
  "barcode": string | null,
  "country_guess": string | null,
  "product_description": string | null (ONE short sentence: what this product actually is—e.g. "Chewing gum shaped like ice cubes, not edible ice" for "Ice Cubes Gum", or "Canned black beans" for a bean can. Clarify anything ambiguous so recipe/meal-plan AIs do not misuse the item.)
}

CRITICAL RULES:
1. confidence MUST be a NUMBER between 0.0 and 1.0
2. Provide at least 2-3 similar items (same category, similar use case)
3. Provide at least 2-3 alternatives (different brands or variants)
4. Provide at least 2-3 healthier alternatives with specific health benefits
5. For healthier alternatives, explain what makes them healthier (lower sugar, more protein, organic, etc.)
6. Include trade-offs for healthier alternatives (taste, price, availability)
7. Estimate price based on typical retail prices for the product
8. List ALL visible ingredients from the label
9. Provide a brief nutrition summary (calories, key macros)
10. Determine if the product is ultra-processed (UPF) and return "yes" or "no" (lowercase only)
11. List harmful ingredients/additives found in the ingredient list (empty array if none). The list MUST be derived from the ingredient list.
12. If ingredients are not clearly visible, infer the full list from the product name/brand/category
13. Be specific and helpful - users want actionable recommendations
14. product_description: ONE short sentence stating what this product actually is, so recipe AIs do not confuse it (e.g. "Ice Cubes Gum" → "Chewing gum shaped like ice cubes, not edible ice"; "Chicken breast" → "Raw chicken breast". Clarify if name could mean something else.)`;

  const userPrompt = `Identify this grocery product from the image. Provide:
- Brand and product name
- Estimated price range
- All visible ingredients
- Similar items (same category)
- Alternative products (different brands/variants)
- Healthier alternatives with specific health benefits and trade-offs
- Nutrition summary
- UPF ("yes" or "no", lowercase only)
- Harmful ingredients list
- Confidence level

Include product_description: one short sentence describing what this product actually is (to avoid confusion in recipes—e.g. gum vs ice). Return ONLY JSON.`;

  try {
    const response = await openai.responses.create({
      model: getOpenAIModel(),
      reasoning: { effort: "low" },
      max_output_tokens: 1800,
      text: {
        format: {
          type: "json_schema",
          name: "grocery_item",
          strict: true,
          schema: groceryItemJsonSchema,
        },
      } as any,
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
              detail: "original",
            },
          ],
        },
      ],
    } as any);

    const parsed = parseJsonResponse<GroceryItem>(response) as GroceryItem & {
      harmful_ingredients?: unknown;
      upf?: unknown;
    };

    parsed.upf = normalizeUpf(parsed.upf);
    parsed.harmful_ingredients = normalizeStringList(parsed.harmful_ingredients);

    let result = GroceryItemSchema.parse(parsed);

    const shouldLookupIngredients = result.ingredients.length < 5 && !!result.product_name;
    if (shouldLookupIngredients) {
      try {
        const lookup = await lookupIngredientsByName(openai, {
          product_name: result.product_name,
          brand: result.brand,
          variant: result.variant,
          category: result.category,
        });
        if (lookup) {
          const preferredIngredients = lookup.ingredients.length >= result.ingredients.length
            ? lookup.ingredients
            : result.ingredients;
          const secondaryIngredients = lookup.ingredients.length >= result.ingredients.length
            ? result.ingredients
            : lookup.ingredients;
          result.ingredients = mergeUniqueList(preferredIngredients, secondaryIngredients);
          result.upf = lookup.upf;
        }
      } catch (lookupError) {
        console.warn('[IdentifyGrocery] Ingredient lookup failed (non-fatal):', lookupError);
      }
    }

    result.harmful_ingredients = deriveHarmfulIngredients(result.ingredients);
    return result;
  } catch (error) {
    console.error("[IdentifyGrocery] Error during OpenAI call or parsing:", error);
    if (error instanceof z.ZodError) {
      console.error("[IdentifyGrocery] Zod validation error details:", JSON.stringify(error.errors, null, 2));
    }
    throw error;
  }
}
