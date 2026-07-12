import fetch from "node-fetch";

/**
 * Flag-gated Gemini fallback for the grocery-identifier OpenAI surfaces
 * (identify + enrich-by-name). OpenAI stays the primary provider. Gemini engages
 * ONLY when BOTH:
 *   1. AI_FALLBACK_ENABLED === "true", AND
 *   2. the OpenAI failure is one of the trigger classes below.
 *
 * With the flag off this module is behaviorally inert: withGeminiFallback()
 * re-throws the original OpenAI error unchanged.
 *
 * Trigger classes (all three must engage the fallback — the malformed-output
 * class is the shape that bled during the OpenAI structured-output degradation):
 *   - quota:           HTTP 429 / insufficient_quota / rate-limit
 *   - server_5xx:      HTTP 5xx (raised AFTER the caller's own OpenAI retries)
 *   - malformed_output: parse failure / schema (Zod) failure / "No structured output"
 */

export function fallbackEnabled(): boolean {
  return String(process.env.AI_FALLBACK_ENABLED || "").toLowerCase() === "true";
}

const GEMINI_TIMEOUT_MS = Number(process.env.AI_FALLBACK_TIMEOUT_MS || 60000);
const GEMINI_ATTEMPTS = Number(process.env.AI_FALLBACK_ATTEMPTS || 2);

function log(evt: string, fields: Record<string, unknown>): void {
  try {
    console.log(JSON.stringify({ evt, ...fields }));
  } catch {
    /* logging must never throw */
  }
}

/** Classify an OpenAI error into a fallback trigger (or not). */
export function classifyFallbackTrigger(error: any): { trigger: boolean; reason: string } {
  const status = error?.status ?? error?.statusCode ?? error?.response?.status;
  const code = String(error?.code || error?.error?.code || "").toLowerCase();
  const msg = String(error?.message || error || "");
  const low = msg.toLowerCase();

  if (
    status === 429 ||
    code === "insufficient_quota" ||
    code === "rate_limit_exceeded" ||
    low.includes("insufficient_quota") ||
    low.includes("rate limit") ||
    low.includes("quota")
  ) {
    return { trigger: true, reason: "quota" };
  }
  if (typeof status === "number" && status >= 500 && status < 600) {
    return { trigger: true, reason: `server_${status}` };
  }
  // Malformed / unparseable / schema-invalid structured output.
  if (
    error?.name === "ZodError" ||
    low.includes("no structured output") ||
    low.includes("failed to parse structured") ||
    low.includes("could not parse") ||
    low.includes("unexpected token") ||
    low.includes("schema")
  ) {
    return { trigger: true, reason: "malformed_output" };
  }
  return { trigger: false, reason: "" };
}

/**
 * Translate an OpenAI json_schema (strict dialect) into a Gemini responseSchema.
 * Gemini v1beta rejects `additionalProperties`/`strict`/`$schema`; keep only
 * type / properties / required / items / enum / description / nullable.
 */
export function toGeminiSchema(schema: any): any {
  if (Array.isArray(schema)) return schema.map(toGeminiSchema);
  if (!schema || typeof schema !== "object") return schema;
  const out: any = {};
  for (const [k, v] of Object.entries(schema)) {
    if (k === "additionalProperties" || k === "strict" || k === "$schema") continue;
    if (k === "properties" && v && typeof v === "object") {
      out.properties = {};
      for (const [pk, pv] of Object.entries(v as any)) out.properties[pk] = toGeminiSchema(pv);
    } else if (k === "items") {
      out.items = toGeminiSchema(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

export interface GeminiCallSpec {
  surface: string; // e.g. "enrich_by_name", "identify_deep"
  systemPrompt: string;
  userPrompt: string;
  schema: any; // OpenAI json_schema.schema object (strict dialect)
  image?: { mimeType: string; base64: string };
  models?: string[]; // primary -> fallback gemini models (override)
}

function resolveModels(spec: GeminiCallSpec): string[] {
  if (spec.models && spec.models.length) return [...new Set(spec.models.filter(Boolean))];
  const primary =
    process.env.AI_FALLBACK_MODEL || process.env.GEMINI_MODEL || "gemini-3.1-pro-preview";
  const secondary =
    process.env.AI_FALLBACK_FLASH_MODEL ||
    process.env.GEMINI_RECEIPT_FALLBACK_MODEL ||
    "gemini-2.5-flash";
  return [...new Set([primary, secondary].filter(Boolean))];
}

/** Call Gemini for a structured JSON response, mirroring the receipt scanner's
 * primary->fallback model loop. Returns parsed JSON of type T. */
export async function callGeminiStructured<T>(spec: GeminiCallSpec, reason: string): Promise<T> {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required for fallback");
  }
  const models = resolveModels(spec);

  const parts: any[] = [{ text: spec.userPrompt }];
  if (spec.image) {
    parts.push({ inline_data: { mime_type: spec.image.mimeType, data: spec.image.base64 } });
  }

  const body = {
    systemInstruction: { parts: [{ text: spec.systemPrompt }] },
    contents: [{ role: "user", parts }],
    generationConfig: {
      temperature: 0.1,
      topP: 0.95,
      responseMimeType: "application/json",
      responseSchema: toGeminiSchema(spec.schema),
    },
  };

  let lastError: any = null;
  for (const model of models) {
    for (let attempt = 0; attempt < GEMINI_ATTEMPTS; attempt += 1) {
      try {
        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            timeout: GEMINI_TIMEOUT_MS,
          } as any
        );
        if ((response as any).ok) {
          const payload: any = await response.json();
          const text =
            payload?.candidates?.[0]?.content?.parts
              ?.map((p: any) => p.text)
              .filter(Boolean)
              .join("\n") || "";
          if (!text.trim()) throw new Error("Gemini returned empty structured output");
          const parsed = JSON.parse(text) as T;
          log("ai_fallback_used", {
            surface: spec.surface,
            reason,
            provider: "gemini",
            model,
            ok: true,
          });
          return parsed;
        }
        const status = (response as any).status;
        lastError = new Error(`Gemini HTTP ${status} on ${model}`);
        if (status === 429) break; // quota — try next model
        if (status >= 500 && status < 600) {
          if (attempt < GEMINI_ATTEMPTS - 1) continue; // transient — retry same model
          break;
        }
        break; // other 4xx — try next model
      } catch (err) {
        lastError = err;
        if (attempt < GEMINI_ATTEMPTS - 1) {
          await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
          continue;
        }
      }
    }
  }
  log("ai_fallback_failed", {
    surface: spec.surface,
    reason,
    provider: "gemini",
    error: String(lastError?.message || lastError),
  });
  throw lastError || new Error("Gemini fallback failed");
}

/**
 * Run an OpenAI call (which must INCLUDE its own response parsing so parse/schema
 * failures surface as throws) and, on a trigger-class failure when the flag is on,
 * retry the equivalent request via Gemini. `geminiSpec` is built lazily so the
 * (potentially expensive) image resolution only runs when a fallback is needed.
 */
export async function withGeminiFallback<T>(
  surface: string,
  openaiCall: () => Promise<T>,
  geminiSpec: () => Omit<GeminiCallSpec, "surface"> | Promise<Omit<GeminiCallSpec, "surface">>
): Promise<T> {
  try {
    return await openaiCall();
  } catch (err) {
    const { trigger, reason } = classifyFallbackTrigger(err);
    if (!trigger || !fallbackEnabled()) throw err;
    log("ai_fallback_triggered", { surface, reason, provider: "gemini" });
    const spec = await geminiSpec();
    return await callGeminiStructured<T>({ surface, ...spec }, reason);
  }
}
