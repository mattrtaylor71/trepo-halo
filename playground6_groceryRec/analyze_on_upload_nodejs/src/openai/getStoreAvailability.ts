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

export const StoreAvailabilitySchema = z.object({
  store_availability: z.array(z.object({
    store_name: z.string(),
    price: z.string().nullable().optional(),
    availability: z.string().nullable().optional(),
    store_url: z.string().nullable().optional(),
  })),
});

export type StoreAvailability = z.infer<typeof StoreAvailabilitySchema>;

const storeAvailabilityJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["store_availability"],
  properties: {
    store_availability: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["store_name", "price", "availability", "store_url"],
        properties: {
          store_name: { type: "string" },
          price: { type: ["string", "null"] },
          availability: { type: ["string", "null"] },
          store_url: { type: ["string", "null"] },
        },
      },
    },
  },
};

export async function getStoreAvailability(
  productName: string,
  brand: string | null | undefined,
  category: string | null | undefined
): Promise<StoreAvailability> {
  const openai = getClient();
  
  const systemPrompt = `You are an expert at finding where grocery products are sold and their prices at different retailers. You have comprehensive knowledge of all grocery stores, specialty markets, regional chains, and online retailers.

Return STRICT JSON only with this exact structure:
{
  "store_availability": [
    {
      "store_name": string (store name - can be any retailer),
      "price": string | null (e.g., "$4.99", "$5.50", "On sale: $3.99"),
      "availability": string | null (e.g., "In stock", "Available online", "Limited availability", "Not sold"),
      "store_url": string | null (product page URL if known)
    }
  ]
}

CRITICAL RULES:
1. Find stores where this SPECIFIC product is actually sold - don't limit to a fixed list
2. Consider ALL types of retailers:
   - Major chains: Walmart, Target, Kroger, Safeway, Albertsons, Publix, Stop & Shop, Giant, Wegmans, H-E-B, Meijer
   - Specialty/organic: Whole Foods, Trader Joe's, Sprouts, Natural Grocers, Fresh Market, Earth Fare
   - Premium/specialty: Erewhon, Mother's Market, Bristol Farms, Gelson's, Central Market
   - Regional chains: Vons, Ralph's, King Soopers, Fred Meyer, QFC, Harris Teeter, Food Lion, ShopRite, Acme
   - Warehouse clubs: Costco, Sam's Club, BJ's Wholesale
   - Drug stores: CVS, Walgreens, Rite Aid
   - Online: Amazon Fresh, Instacart, Thrive Market, FreshDirect
   - Ethnic markets: H Mart, 99 Ranch, Patel Brothers (if relevant)
   - Local/regional stores relevant to the product
3. Match stores to the product type:
   - Organic/natural products → Whole Foods, Sprouts, Natural Grocers, Erewhon, Mother's Market
   - Private label products → The store that owns the brand (e.g., Trader Joe's products only at Trader Joe's)
   - Premium products → Erewhon, Whole Foods, Bristol Farms, Gelson's
   - Mainstream products → Walmart, Target, Kroger, Safeway
   - Bulk products → Costco, Sam's Club
4. Provide 3-7 stores where the product is MOST LIKELY to be found
5. If it's a private label or exclusive product, clearly state which stores DON'T carry it
6. Prices should be realistic - research typical prices for this type of product at each store type
7. If you don't know the exact price, estimate based on similar products and store pricing tiers
8. Include availability status - be honest if it's not sold at certain stores
9. IMPORTANT: Do NOT include store_url unless you are absolutely certain of a working product page URL. Most store URLs change frequently and cannot be accurately predicted. Set store_url to null if unsure.
10. Prioritize stores where the product is actually available over stores where it's not`;

  const userPrompt = `Find store availability and prices for this grocery product:
- Product: ${productName}
- Brand: ${brand || 'Unknown'}
- Category: ${category || 'Unknown'}

IMPORTANT: Find stores where this SPECIFIC product is actually sold. Consider:
- If it's a private label product, only list the store that owns that brand
- If it's organic/natural, prioritize Whole Foods, Sprouts, Natural Grocers, Erewhon, Mother's Market
- If it's premium, consider Erewhon, Whole Foods, Bristol Farms, Gelson's
- If it's mainstream, consider Walmart, Target, Kroger, Safeway, regional chains
- Don't limit yourself to common stores - include specialty stores, regional chains, and online retailers where relevant
- If the product is NOT sold at certain stores, you can include them but mark as "Not sold" with explanation

Provide 3-7 stores where this product is most likely to be found, including:
- Store name (any retailer, not limited to a fixed list)
- Estimated price (realistic for that store type)
- Availability status (be honest - "Not sold" if it's not available)
- Product page URL: ONLY include if you can construct a reliable, working URL. Most store product URLs are dynamic and change frequently, so set to null if uncertain. Do not guess or make up URLs.

Return ONLY JSON.`;

  try {
    const response = await openai.responses.create({
      model: getOpenAIModel(),
      reasoning: { effort: "low" },
      max_output_tokens: 1000,
      text: {
        format: {
          type: "json_schema",
          name: "store_availability",
          strict: true,
          schema: storeAvailabilityJsonSchema,
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

    const parsed = parseJsonResponse<StoreAvailability>(response);

    return StoreAvailabilitySchema.parse(parsed);
  } catch (error) {
    console.error("[GetStoreAvailability] Error during OpenAI call or parsing:", error);
    if (error instanceof z.ZodError) {
      console.error("[GetStoreAvailability] Zod validation error details:", JSON.stringify(error.errors, null, 2));
    }
    throw error;
  }
}
