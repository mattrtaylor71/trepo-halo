import type OpenAI from "openai";
import { z } from "zod";
import { getOpenAIClient, getOpenAIModel, parseJsonResponse } from "./client";
import { extractLabelEvidence, LabelEvidence } from "./extractLabelEvidence";
import { withGeminiFallback } from "./geminiFallback";

// ── AI-driven category (Task 3): constrain the identify `category` field to the 8
// app enums directly, so the model does the smart categorization and the keyword
// normalizer (kitchen_api/category_normalizer.py + quick-ack) becomes a SAFETY NET,
// not the decision-maker. FLAG-GATED for instant rollback (env
// CATEGORY_ENUM_PROMPT_VERSION): off/unset => current freetext behavior (inert).
// Scope is category-ONLY: no other identify field is touched. detected_category
// (label evidence) is intentionally left freetext — the main `category` enum wins
// chooseCategory() (the 8 values are not "generic"), and constraining evidence would
// be a broader change to inferProductType(). Zod stays permissive (z.string) so a
// schema-escape can never crash identify — the normalizer clamps it.
export const CATEGORY_ENUM = [
  "leftovers", "produce", "dairy_eggs", "meat_seafood",
  "pantry", "spices", "snacks_sweets", "beverages", "prepared_other",
] as const;
export const CATEGORY_ENUM_ENABLED = ["v1", "enum", "true", "on"].includes(
  String(process.env.CATEGORY_ENUM_PROMPT_VERSION || "").toLowerCase()
);
const CATEGORY_ENUM_GUIDANCE =
  "\n\nCATEGORY — choose EXACTLY ONE of these 9 values, by what the product fundamentally IS, NOT by incidental words in its name: leftovers, produce, dairy_eggs, meat_seafood, pantry, spices, snacks_sweets, beverages, prepared_other." +
  "\nRules + hard examples: Potato Bread / Blueberry Bread = pantry (it IS bread, shelf-stable). Blueberry Muffins / cakes / cookies / pastries = snacks_sweets. Green Onion Pancakes and other frozen/prepared foods = prepared_other. A bag of potatoes / loose bananas / a bunch of celery or herbs = produce. Strawberry yogurt = dairy_eggs. Chicken broth = pantry. Fresh raw meat/poultry/fish = meat_seafood. Any drink = beverages. Home leftover food = leftovers. Spices, seasonings, spice blends, spice rubs, and dried/ground herbs (cinnamon, cumin, paprika, oregano, chili powder, garlic powder, dried basil, etc.) = spices, NOT pantry. When a produce word appears in a processed product's name (potato bread, blueberry muffin, green onion pancake), pick the PROCESSED category, not produce.";

export const GroceryItemSchema = z.object({
  brand: z.string().nullable().optional(),
  product_name: z.string().nullable().optional(),
  variant: z.string().nullable().optional(),
  category: z.string().nullable().optional(),
  product_type: z.string().nullable().optional(),
  producer: z.string().nullable().optional(),
  vintage: z.string().nullable().optional(),
  region_or_appellation: z.string().nullable().optional(),
  varietal_or_blend: z.string().nullable().optional(),
  abv: z.string().nullable().optional(),
  size_text: z.string().nullable().optional(),
  estimated_price: z.string().nullable().optional(),
  ingredients: z.array(z.string()).optional().default([]),
  nutrition_summary: z.string().nullable().optional(),
  upf: z.enum(['yes', 'no']).nullable().optional(),
  harmful_ingredients: z.array(z.string()).optional().default([]),
  similar_items: z
    .array(
      z.object({
        name: z.string(),
        brand: z.string().nullable().optional(),
        reason: z.string(),
      })
    )
    .optional()
    .default([]),
  alternatives: z
    .array(
      z.object({
        name: z.string(),
        brand: z.string().nullable().optional(),
        reason: z.string(),
      })
    )
    .optional()
    .default([]),
  healthier_alternatives: z
    .array(
      z.object({
        name: z.string(),
        brand: z.string().nullable().optional(),
        why_healthier: z.string(),
        trade_offs: z.string().nullable().optional(),
      })
    )
    .optional()
    .default([]),
  confidence: z.number().min(0).max(1),
  explanation: z.string().nullable().optional(),
  barcode: z.string().nullable().optional(),
  country_guess: z.string().nullable().optional(),
  product_description: z.string().nullable().optional(),
  visible_text_lines: z.array(z.string()).optional().default([]),
});

export type GroceryItem = z.infer<typeof GroceryItemSchema>;

export type IdentifyImageInput = {
  imageBuffer?: Buffer;
  imageUrl?: string;
};

function buildImageContent(input: IdentifyImageInput): { type: "input_image"; image_url: string; detail: "original" } {
  if (input.imageUrl) {
    return {
      type: "input_image",
      image_url: input.imageUrl,
      detail: "original",
    };
  }

  if (!input.imageBuffer) {
    throw new Error("Either imageBuffer or imageUrl is required");
  }

  return {
    type: "input_image",
    image_url: `data:image/jpeg;base64,${input.imageBuffer.toString("base64")}`,
    detail: "original",
  };
}

const groceryItemJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "brand",
    "product_name",
    "variant",
    "category",
    "product_type",
    "producer",
    "vintage",
    "region_or_appellation",
    "varietal_or_blend",
    "abv",
    "size_text",
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
    "visible_text_lines",
  ],
  properties: {
    brand: { type: ["string", "null"] },
    product_name: { type: ["string", "null"] },
    variant: { type: ["string", "null"] },
    category: CATEGORY_ENUM_ENABLED
      ? { type: ["string", "null"], enum: [...CATEGORY_ENUM, null] }
      : { type: ["string", "null"] },
    product_type: { type: ["string", "null"] },
    producer: { type: ["string", "null"] },
    vintage: { type: ["string", "null"] },
    region_or_appellation: { type: ["string", "null"] },
    varietal_or_blend: { type: ["string", "null"] },
    abv: { type: ["string", "null"] },
    size_text: { type: ["string", "null"] },
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
    visible_text_lines: {
      type: "array",
      items: { type: "string" },
    },
  },
};

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

  const openaiRequest: any = {
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
    },
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
  };

  let parsed: any;
  try {
    // OpenAI primary -> Gemini fallback (flag-gated) on quota/5xx/malformed-output.
    parsed = await withGeminiFallback<IngredientLookup>(
      "enrich_by_name",
      async () => parseJsonResponse<IngredientLookup>(await openai.responses.create(openaiRequest)),
      () => ({ systemPrompt, userPrompt, schema: ingredientLookupJsonSchema })
    );
  } catch {
    return null;
  }

  parsed.ingredients = normalizeStringList(parsed.ingredients);
  parsed.harmful_ingredients = normalizeStringList(parsed.harmful_ingredients);
  parsed.upf = normalizeUpf(parsed.upf);

  return IngredientLookupSchema.parse(parsed);
}

/**
 * Text-only enrichment entry point: given a product identity (name/brand/variant/
 * category), return {ingredients, upf, harmful_ingredients} with no image required.
 * OpenAI primary -> Gemini fallback (flag-gated), same as the in-identify lookup.
 * Exposed for the voice-item enrich-by-name backfill; NOT wired into the voice
 * write path yet (that's the durable follow-up, Option B).
 */
export async function enrichByName(
  details: Pick<GroceryItem, 'product_name' | 'brand' | 'variant' | 'category'>
): Promise<IngredientLookup | null> {
  return lookupIngredientsByName(getOpenAIClient(), details);
}

function cleanNullable(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return Array.from(
    new Set(
      values
        .map((value) => cleanNullable(value))
        .filter((value): value is string => Boolean(value))
    )
  );
}

function normalizeText(value: string | null | undefined): string {
  return cleanNullable(value)?.toLowerCase() || "";
}

function containsAnyToken(value: string | null | undefined, tokens: string[]): boolean {
  const normalized = normalizeText(value);
  const haystack = ` ${normalized} `;
  return tokens.some((token) => {
    const needle = normalizeText(token);
    return needle ? haystack.includes(` ${needle} `) : false;
  });
}

function isAlcoholCategory(value: string | null | undefined): boolean {
  const normalized = normalizeText(value);
  return [
    "wine",
    "beer",
    "spirit",
    "spirits",
    "liquor",
    "alcohol",
    "champagne",
    "whiskey",
    "whisky",
    "vodka",
    "tequila",
    "rum",
    "gin",
    "cider",
    "seltzer",
  ].some((token) => normalized.includes(token));
}

function isGenericCategory(value: string | null | undefined): boolean {
  const normalized = normalizeText(value);
  return normalized === "" || ["grocery", "product", "packaged product", "beverage", "alcohol", "drink"].includes(normalized);
}

function chooseCategory(primary: string | null | undefined, secondary: string | null | undefined): string | null {
  const first = cleanNullable(primary);
  const second = cleanNullable(secondary);

  if (!first) return second;
  if (!second) return first;
  if (isGenericCategory(first) && !isGenericCategory(second)) {
    return second;
  }

  return first;
}

function inferProductType(category: string | null | undefined, evidence: LabelEvidence | null): string | null {
  const existing = cleanNullable(evidence?.product_type);
  if (existing) {
    return existing;
  }

  const normalized = normalizeText(category || evidence?.detected_category);
  if (!normalized) {
    return null;
  }
  if (normalized.includes("wine")) return "wine";
  if (normalized.includes("beer")) return "beer";
  if (
    normalized.includes("spirit") ||
    normalized.includes("liquor") ||
    normalized.includes("vodka") ||
    normalized.includes("whiskey") ||
    normalized.includes("whisky") ||
    normalized.includes("tequila") ||
    normalized.includes("rum") ||
    normalized.includes("gin")
  ) {
    return "spirits";
  }
  if (normalized.includes("produce") || normalized.includes("fruit") || normalized.includes("vegetable")) {
    return "produce";
  }
  return normalized;
}

function buildEvidencePrompt(evidence: LabelEvidence | null): string {
  if (!evidence) {
    return "No first-pass label evidence was available.";
  }

  return [
    "First-pass label evidence extracted from the image:",
    JSON.stringify(
      {
        detected_category: evidence.detected_category,
        product_type: evidence.product_type,
        producer: evidence.producer,
        product_name: evidence.product_name,
        variant: evidence.variant,
        vintage: evidence.vintage,
        region_or_appellation: evidence.region_or_appellation,
        varietal_or_blend: evidence.varietal_or_blend,
        abv: evidence.abv,
        size_text: evidence.size_text,
        barcode: evidence.barcode,
        visible_text_lines: evidence.visible_text_lines,
        confidence: evidence.confidence,
        evidence_summary: evidence.evidence_summary,
      },
      null,
      2
    ),
    "Prefer exact visible label text from this evidence when it is specific and plausible.",
  ].join("\n");
}

function buildFallbackDescription(item: GroceryItem): string | null {
  const category = cleanNullable(item.category);
  const descriptionParts = uniqueStrings([
    item.brand,
    item.producer,
    item.product_name,
    item.variant,
    item.vintage,
    item.varietal_or_blend,
    category,
    item.size_text,
  ]);

  if (descriptionParts.length === 0) {
    return null;
  }

  return descriptionParts.join(" ");
}

function mergeLabelEvidenceIntoItem(item: GroceryItem, evidence: LabelEvidence | null): GroceryItem {
  const mergedCategory = chooseCategory(item.category, evidence?.detected_category);
  const alcoholLike = isAlcoholCategory(mergedCategory) || isAlcoholCategory(evidence?.product_type);
  const producer = cleanNullable(item.producer) || cleanNullable(evidence?.producer);
  const brand = cleanNullable(item.brand) || (alcoholLike ? producer : null);
  const visible_text_lines = uniqueStrings([...(item.visible_text_lines || []), ...(evidence?.visible_text_lines || [])]);

  const merged: GroceryItem = {
    ...item,
    brand,
    product_name: cleanNullable(item.product_name) || cleanNullable(evidence?.product_name),
    variant: cleanNullable(item.variant) || cleanNullable(evidence?.variant),
    category: mergedCategory,
    product_type: cleanNullable(item.product_type) || inferProductType(mergedCategory, evidence),
    producer: producer || (alcoholLike ? brand : null),
    vintage: cleanNullable(item.vintage) || cleanNullable(evidence?.vintage),
    region_or_appellation:
      cleanNullable(item.region_or_appellation) || cleanNullable(evidence?.region_or_appellation),
    varietal_or_blend: cleanNullable(item.varietal_or_blend) || cleanNullable(evidence?.varietal_or_blend),
    abv: cleanNullable(item.abv) || cleanNullable(evidence?.abv),
    size_text: cleanNullable(item.size_text) || cleanNullable(evidence?.size_text),
    barcode: cleanNullable(item.barcode) || cleanNullable(evidence?.barcode),
    explanation: cleanNullable(item.explanation) || cleanNullable(evidence?.evidence_summary),
    product_description: cleanNullable(item.product_description),
    visible_text_lines,
  };

  if (!merged.product_description) {
    merged.product_description = buildFallbackDescription(merged);
  }

  return GroceryItemSchema.parse(merged);
}

const NON_GROCERY_TOKENS = [
  "book",
  "cookbook",
  "magazine",
  "fitness equipment",
  "exercise equipment",
  "vehicle",
  "model vehicle",
  "display model",
  "robot",
  "toy",
  "furniture",
  "chair",
  "bag",
  "backpack",
  "duffel",
  "delivery vehicle",
  "autonomous delivery vehicle",
  "peloton",
  "nuro",
];

const GROCERY_TOKENS = [
  "grocery",
  "pantry",
  "packaged",
  "boxed",
  "jarred",
  "canned",
  "produce",
  "fruit",
  "vegetable",
  "condiment",
  "sauce",
  "dressing",
  "dip",
  "juice",
  "beverage",
  "drink",
  "spirits",
  "liquor",
  "wine",
  "beer",
  "milk",
  "yogurt",
  "cheese",
  "snack",
  "gum",
  "chewing gum",
  "mint",
  "mints",
  "candy",
  "chocolate",
  "crackers",
  "cookies",
  "cereal",
  "oats",
  "pasta",
  "rice",
  "beans",
  "broth",
  "tea",
  "coffee",
  "peanut butter",
  "banana",
  "apple",
  "orange",
  "miso",
  "mayo",
  "hot sauce",
  "syrup",
];

export function isLikelyNonGroceryItem(item: Partial<GroceryItem> | null | undefined): boolean {
  if (!item) return true;

  const combinedText = [
    item.brand,
    item.product_name,
    item.variant,
    item.category,
    item.product_type,
    item.product_description,
    ...(Array.isArray(item.visible_text_lines) ? item.visible_text_lines.slice(0, 6) : []),
  ]
    .filter(Boolean)
    .join(" ");

  const hasGrocerySignals =
    containsAnyToken(item.category, GROCERY_TOKENS) ||
    containsAnyToken(item.product_type, GROCERY_TOKENS) ||
    containsAnyToken(item.product_name, GROCERY_TOKENS) ||
    containsAnyToken(item.product_description, GROCERY_TOKENS) ||
    containsAnyToken(combinedText, GROCERY_TOKENS);

  const hasNonGrocerySignals =
    containsAnyToken(item.category, NON_GROCERY_TOKENS) ||
    containsAnyToken(item.product_type, NON_GROCERY_TOKENS) ||
    containsAnyToken(item.product_name, NON_GROCERY_TOKENS) ||
    containsAnyToken(item.product_description, NON_GROCERY_TOKENS) ||
    containsAnyToken(item.brand, NON_GROCERY_TOKENS) ||
    containsAnyToken(combinedText, NON_GROCERY_TOKENS);

  if (!cleanNullable(item.product_name) && !cleanNullable(item.brand) && !cleanNullable(item.category)) {
    return true;
  }

  if (hasNonGrocerySignals && !hasGrocerySignals) {
    return true;
  }

  return false;
}

export async function identifyGroceryItem(input: IdentifyImageInput, options: { userHint?: string | null; leftovers?: boolean } = {}): Promise<GroceryItem> {
  const openai = getOpenAIClient();
  const userHint = options.userHint;
  const leftovers = options.leftovers;

  try {
    let labelEvidence: LabelEvidence | null = null;
    try {
      labelEvidence = await extractLabelEvidence(input);
    } catch (error) {
      console.warn("[IdentifyGrocery] Label evidence extraction failed, falling back to direct identification:", error);
    }

    let systemText = "You identify grocery and beverage products from images. Return only the requested JSON fields. Use null when a field cannot be determined. Prefer exact visible label text over broad guesses, especially for bottles and small labels. Only include ingredients that are visible or very strongly implied by recognizable packaging. Confidence must reflect how certain you are that this exact product was identified. For alcohol products, producer, vintage, region/appellation, ABV, and bottle size are important identity fields. Product description must be one short sentence describing what the item is in plain language.";
    if (userHint) {
      systemText += `\n\nCRITICAL OVERRIDE: The user manually described this item as: "${userHint}". This note tells you what the item ACTUALLY IS, regardless of what the image shows. The user may be photographing food inside a container, a reusable bottle, a Tupperware, or a bag — the container is NOT the product. Use the user's description as the product_name. For example, if the image shows a Hydro Flask but the user says "smoothie", the product is a smoothie, NOT a Hydro Flask. The user's note always takes priority over visible branding or packaging.`;
    }
    else if (leftovers) {
      systemText += `\n\nLEFTOVERS MODE: This is a photo of leftover FOOD or DRINK the user is saving — NOT a packaged grocery product. Name the item by what the food/drink actually IS, as specifically as the image allows (e.g. 'Black Coffee', 'Chicken Fried Rice', 'Half a Burrito'). A confident contextual guess beats a generic label — e.g. a Starbucks cup is 'Black Coffee' (or 'Iced Coffee' etc), not 'Prepared Food'. NEVER use 'Prepared Food', 'Leftovers', or 'Prepared Food Leftovers' as the product_name — the item is already tagged as a leftover elsewhere. Put container details (cup, tupperware) in variant, not the name. Only if the contents are truly unidentifiable, use a best-effort descriptive name like 'Mixed Leftover Meal'.`;
    }

    // AI-driven category guidance (flag-gated; category-only, appended last so it
    // applies in all modes). Pairs with the enum schema constraint above.
    if (CATEGORY_ENUM_ENABLED) {
      systemText += CATEGORY_ENUM_GUIDANCE;
    }

    const model = getOpenAIModel();
    // Fast in-handler retry: a single "No structured output" / ZodError blip
    // otherwise fails the invocation (the Merlot-saga class). Retry once immediately.
    let result!: GroceryItem;
    let _identifyLastErr: unknown = null;
    for (let _identifyAttempt = 0; _identifyAttempt < 2; _identifyAttempt += 1) {
      const _aiStart = Date.now();
      try {
    const response = await openai.responses.create({
      model,
      ...((/^(o[1-9]|gpt-5)/.test(model)) ? { reasoning: { effort: "medium" } } : {}),
      // Headroom so reasoning tokens (drawn from this budget on gpt-5.x) don't starve the
      // structured output → "No structured output". 2200 occasionally ran dry under medium.
      max_output_tokens: 4000,
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
          content: [
            {
              type: "input_text",
              text: systemText,
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                `Analyze this grocery or beverage product image and extract the product identity, label details, likely price range, and a few concise product alternatives. Use the image plus the first-pass label evidence below. Do not invent store links or pretend to browse.\n\n${buildEvidencePrompt(
                  labelEvidence
                )}`,
            },
            buildImageContent(input),
          ],
        },
      ],
    } as any);

    const parsed = parseJsonResponse<GroceryItem>(response);
    result = mergeLabelEvidenceIntoItem(GroceryItemSchema.parse(parsed), labelEvidence);
    try {
      const _u: any = (response as any)?.usage || {};
      console.log(JSON.stringify({
        evt: "ai_op", service: "grocery_identify", op: "identify_deep", model,
        status: "success", latency_ms: Date.now() - _aiStart,
        tokens_in: _u.input_tokens ?? _u.prompt_tokens ?? null,
        tokens_out: _u.output_tokens ?? _u.completion_tokens ?? null,
      }));
    } catch (_) { /* telemetry must never throw */ }
    if (_identifyAttempt > 0) {
      console.log(JSON.stringify({ evt: "openai_fast_retry_saved", service: "grocery_identify", op: "identify_deep" }));
    }
    break;
      } catch (_identifyErr) {
        _identifyLastErr = _identifyErr;
        if (_identifyAttempt === 0) { await new Promise((r) => setTimeout(r, 2500)); continue; }
        try {
          console.log(JSON.stringify({
            evt: "ai_op", service: "grocery_identify", op: "identify_deep", model,
            status: "error", latency_ms: Date.now() - _aiStart,
            error: String((_identifyErr as any)?.message || _identifyErr).slice(0, 500),
          }));
        } catch (_) { /* telemetry must never throw */ }
        throw _identifyLastErr;
      }
    }

    // Normalize UPF flag (model may emit "Yes"/null/etc.)
    result.upf = normalizeUpf(result.upf);

    // If the vision pass produced few ingredients, augment with a text-based
    // ingredient lookup from the identified product name/brand/category.
    const shouldLookupIngredients = (result.ingredients?.length || 0) < 5 && !!result.product_name;
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

    // Harmful ingredients are always derived deterministically from the final
    // ingredient list (never trusted from the model directly).
    result.harmful_ingredients = deriveHarmfulIngredients(result.ingredients);

    return result;
  } catch (error) {
    console.error("[IdentifyGrocery] Error during GPT-5.4 Responses call:", error);
    if (error instanceof z.ZodError) {
      console.error(
        "[IdentifyGrocery] Zod validation error details:",
        JSON.stringify(error.errors, null, 2)
      );
    }
    throw error;
  }
}

