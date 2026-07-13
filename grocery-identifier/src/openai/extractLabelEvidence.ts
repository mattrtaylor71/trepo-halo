import { z } from "zod";
import { getOpenAIClient, getOpenAIModel, parseJsonResponse } from "./client";
import type { IdentifyImageInput } from "./identifyGrocery";

export const LabelEvidenceSchema = z.object({
  detected_category: z.string().nullable().optional(),
  product_type: z.string().nullable().optional(),
  producer: z.string().nullable().optional(),
  product_name: z.string().nullable().optional(),
  variant: z.string().nullable().optional(),
  vintage: z.string().nullable().optional(),
  region_or_appellation: z.string().nullable().optional(),
  varietal_or_blend: z.string().nullable().optional(),
  abv: z.string().nullable().optional(),
  size_text: z.string().nullable().optional(),
  barcode: z.string().nullable().optional(),
  visible_text_lines: z.array(z.string()).optional().default([]),
  confidence: z.number().min(0).max(1),
  evidence_summary: z.string().nullable().optional(),
});

export type LabelEvidence = z.infer<typeof LabelEvidenceSchema>;

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

const labelEvidenceJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "detected_category",
    "product_type",
    "producer",
    "product_name",
    "variant",
    "vintage",
    "region_or_appellation",
    "varietal_or_blend",
    "abv",
    "size_text",
    "barcode",
    "visible_text_lines",
    "confidence",
    "evidence_summary",
  ],
  properties: {
    detected_category: { type: ["string", "null"] },
    product_type: { type: ["string", "null"] },
    producer: { type: ["string", "null"] },
    product_name: { type: ["string", "null"] },
    variant: { type: ["string", "null"] },
    vintage: { type: ["string", "null"] },
    region_or_appellation: { type: ["string", "null"] },
    varietal_or_blend: { type: ["string", "null"] },
    abv: { type: ["string", "null"] },
    size_text: { type: ["string", "null"] },
    barcode: { type: ["string", "null"] },
    visible_text_lines: {
      type: "array",
      items: { type: "string" },
    },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    evidence_summary: { type: ["string", "null"] },
  },
};

export async function extractLabelEvidence(input: IdentifyImageInput): Promise<LabelEvidence> {
  const openai = getOpenAIClient();

  const response = await openai.responses.create({
    model: getOpenAIModel(),
    // Label extraction is an OCR-ish read, not a reasoning task. On gpt-5.x, reasoning
    // tokens are drawn FROM max_output_tokens — "medium" effort with a 1200 ceiling ate
    // the whole budget and left nothing for the structured output ("No structured output"
    // → forced the direct-ID fallback thousands of times/day). Low effort + a generous
    // ceiling leaves ample room for the JSON, and is cheaper (fewer reasoning tokens).
    reasoning: { effort: "low" },
    max_output_tokens: 3500,
    text: {
      format: {
        type: "json_schema",
        name: "label_evidence",
        strict: true,
        schema: labelEvidenceJsonSchema,
      },
    } as any,
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text:
              "You extract product label evidence from grocery and beverage product images. Prioritize exact visible label text over inference. Be careful with bottles, curved labels, glare, stylized fonts, alcohol products, vintage years, appellations, ABV, and size markings. Return null when a field is not visible or not strongly supported. visible_text_lines should only contain short snippets that are actually visible on the package.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              "Read this product label carefully and extract structured identity evidence. If this appears to be wine, beer, or spirits, focus on producer, wine name, vintage, region/appellation, varietal/blend, alcohol percentage, and bottle size. Do not browse or guess beyond the image.",
          },
          buildImageContent(input),
        ],
      },
    ],
  } as any);

  const parsed = parseJsonResponse<LabelEvidence>(response);
  return LabelEvidenceSchema.parse(parsed);
}
