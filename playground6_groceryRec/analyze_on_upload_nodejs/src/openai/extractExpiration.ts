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

function parseJsonResponse<T>(response: any): T {
  const outputText = response?.output_text;
  if (typeof outputText !== "string" || !outputText.trim()) {
    throw new Error("No structured output returned from OpenAI");
  }
  return JSON.parse(outputText) as T;
}

const ExpirationSchema = z.object({
  expiration_date: z.string().nullable().optional(),
  confidence: z.union([
    z.number().min(0).max(1),
    z.string().transform((s) => {
      const num = parseFloat(s);
      return isNaN(num) ? 0.5 : Math.max(0, Math.min(1, num));
    }),
    z.object({}).passthrough().transform(() => 0.5),
    z.any().transform(() => 0.5),
  ]).optional(),
  raw_text: z.string().nullable().optional(),
});

const expirationJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["expiration_date", "confidence", "raw_text"],
  properties: {
    expiration_date: { type: ["string", "null"] },
    confidence: { type: ["number", "null"] },
    raw_text: { type: ["string", "null"] },
  },
};

function pad2(value: number): string {
  return value < 10 ? `0${value}` : String(value);
}

function toIsoDate(year: number, month: number, day: number): string | null {
  if (year < 2000 || year > 2100) return null;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > 31) return null;
  return `${year}-${pad2(month)}-${pad2(day)}`;
}

function normalizeExpirationDate(raw: unknown): string | null {
  if (!raw) return null;
  const text = String(raw).trim();
  if (!text) return null;

  const compact = text.match(/\b(\d{4})(\d{2})(\d{2})\b/);
  if (compact) {
    const year = Number(compact[1]);
    const month = Number(compact[2]);
    const day = Number(compact[3]);
    return toIsoDate(year, month, day);
  }

  const iso = text.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (iso) {
    const year = Number(iso[1]);
    const month = Number(iso[2]);
    const day = Number(iso[3]);
    return toIsoDate(year, month, day);
  }

  const ymd = text.match(/(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})/);
  if (ymd) {
    const year = Number(ymd[1]);
    const month = Number(ymd[2]);
    const day = Number(ymd[3]);
    return toIsoDate(year, month, day);
  }

  const mdy = text.match(/(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})/);
  if (mdy) {
    const month = Number(mdy[1]);
    const day = Number(mdy[2]);
    let year = Number(mdy[3]);
    if (year < 100) {
      year = 2000 + year;
    }
    return toIsoDate(year, month, day);
  }

  const md = text.match(/\b(\d{1,2})[\/\-.](\d{1,2})\b/);
  if (md) {
    const month = Number(md[1]);
    const day = Number(md[2]);
    const now = new Date();
    const currentYear = now.getUTCFullYear();
    const candidate = new Date(Date.UTC(currentYear, month - 1, day));
    const today = new Date(Date.UTC(currentYear, now.getUTCMonth(), now.getUTCDate()));
    const year = candidate < today ? currentYear + 1 : currentYear;
    return toIsoDate(year, month, day);
  }

  const monthNames: Record<string, number> = {
    jan: 1, january: 1,
    feb: 2, february: 2,
    mar: 3, march: 3,
    apr: 4, april: 4,
    may: 5,
    jun: 6, june: 6,
    jul: 7, july: 7,
    aug: 8, august: 8,
    sep: 9, sept: 9, september: 9,
    oct: 10, october: 10,
    nov: 11, november: 11,
    dec: 12, december: 12,
  };
  const normalizedText = text.toLowerCase();
  const monthMatch = normalizedText.match(/\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b/);
  if (monthMatch) {
    const month = monthNames[monthMatch[1]];
    const dayMatch = normalizedText.match(/\b(\d{1,2})(?:st|nd|rd|th)?\b/);
    const yearMatch = normalizedText.match(/\b(20\d{2})\b/);
    if (month && dayMatch && yearMatch) {
      const day = Number(dayMatch[1]);
      const year = Number(yearMatch[1]);
      return toIsoDate(year, month, day);
    }
    if (month && dayMatch && !yearMatch) {
      const day = Number(dayMatch[1]);
      const now = new Date();
      const currentYear = now.getUTCFullYear();
      const candidate = new Date(Date.UTC(currentYear, month - 1, day));
      const today = new Date(Date.UTC(currentYear, now.getUTCMonth(), now.getUTCDate()));
      const year = candidate < today ? currentYear + 1 : currentYear;
      return toIsoDate(year, month, day);
    }
  }

  return null;
}

function getExpirationSettings() {
  const model = process.env.EXPIRATION_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
  const maxTokensRaw = process.env.EXPIRATION_MAX_TOKENS || "300";
  const temperatureRaw = process.env.EXPIRATION_TEMPERATURE || "0.0";
  const imageDetail = (process.env.EXPIRATION_IMAGE_DETAIL || "high").toLowerCase();
  const maxTokens = Math.max(64, parseInt(maxTokensRaw, 10) || 300);
  const temperature = Math.min(1, Math.max(0, parseFloat(temperatureRaw) || 0.0));
  const detail = imageDetail === "low" ? "low" : imageDetail === "high" ? "high" : undefined;
  return { model, maxTokens, temperature, detail };
}

function resolveMimeType(mimeType?: string | null): string {
  if (mimeType) {
    const lowered = mimeType.toLowerCase();
    if (lowered.includes("png")) return "image/png";
    if (lowered.includes("jpeg") || lowered.includes("jpg")) return "image/jpeg";
  }
  return "image/jpeg";
}

export interface ExpirationResult {
  expirationDate: string | null;
  rawText: string | null;
  confidence: number | null;
}

export async function extractExpirationDate(
  imageBuffer: Buffer,
  mimeType?: string | null
): Promise<ExpirationResult> {
  const openai = getClient();
  const { model, maxTokens, temperature, detail } = getExpirationSettings();
  const dataMime = resolveMimeType(mimeType);

  const systemPrompt = `You are an OCR system that extracts expiration dates from food packaging images.

Return STRICT JSON only with this exact structure:
{
  "expiration_date": string | null,
  "confidence": number | null,
  "raw_text": string | null
}

Rules:
1. expiration_date must be in YYYY-MM-DD format or null if missing/ambiguous.
2. If multiple dates exist, choose the explicit expiration date (EXP, USE BY, BEST BY).
3. If the year is missing but month/day is present, assume the next occurrence of that date (this year if upcoming, otherwise next year).
4. Return ONLY JSON with no extra keys or text.`;

  const userPrompt = `Extract the expiration date from this close-up label image. Return ONLY JSON.`;

  const response = await openai.responses.create({
    model,
    reasoning: { effort: "low" },
    max_output_tokens: maxTokens,
    text: {
      format: {
        type: "json_schema",
        name: "expiration_result",
        strict: true,
        schema: expirationJsonSchema,
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
            image_url: `data:${dataMime};base64,${imageBuffer.toString("base64")}`,
            ...(detail ? { detail } : {}),
          },
        ],
      },
    ],
  } as any);

  const parsed = parseJsonResponse<{
    expiration_date?: string | null;
    confidence?: number | null;
    raw_text?: string | null;
  }>(response);

  const normalized = normalizeExpirationDate(parsed.expiration_date) || normalizeExpirationDate(parsed.raw_text);
  const validated = ExpirationSchema.parse({
    expiration_date: normalized,
    confidence: parsed.confidence,
    raw_text: parsed.raw_text,
  });

  const confidence = typeof validated.confidence === "number"
    ? validated.confidence
    : validated.confidence == null
      ? null
      : Number(validated.confidence);

  return {
    expirationDate: validated.expiration_date || null,
    rawText: validated.raw_text || null,
    confidence: Number.isFinite(confidence as number) ? (confidence as number) : null,
  };
}
