const OPENAI_API_BASE = "https://api.openai.com/v1";

const ANALYZER_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "dish_name",
    "serving_size",
    "calories",
    "total_fat",
    "total_carbohydrates",
    "protein",
    "confidence",
    "explanation",
    "ingredients",
    "components",
    "allergens",
    "source_type",
    "evidence_urls"
  ],
  properties: {
    dish_name: { type: ["string", "null"] },
    serving_size: { type: ["string", "null"] },
    calories: { type: ["number", "null"] },
    total_fat: { type: ["number", "null"] },
    total_carbohydrates: { type: ["number", "null"] },
    protein: { type: ["number", "null"] },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    explanation: { type: ["string", "null"] },
    ingredients: {
      type: "array",
      items: { type: "string" }
    },
    components: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "ingredients"],
        properties: {
          name: { type: "string" },
          ingredients: {
            type: "array",
            items: { type: "string" }
          }
        }
      }
    },
    allergens: {
      type: "array",
      items: { type: "string" }
    },
    source_type: { type: ["string", "null"] },
    evidence_urls: {
      type: "array",
      items: { type: "string" }
    }
  }
};

function trimString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function toNumber(value) {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeStringList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set();
  const result = [];
  for (const entry of value) {
    const trimmed = trimString(entry);
    if (!trimmed) {
      continue;
    }
    const key = trimmed.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(trimmed);
  }
  return result;
}

function normalizeComponents(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set();
  const result = [];
  for (const entry of value) {
    const name = trimString(entry?.name);
    const ingredients = normalizeStringList(entry?.ingredients);
    const fallbackName = name || ingredients[0] || null;
    if (!fallbackName) {
      continue;
    }
    const key = `${fallbackName.toLowerCase()}::${ingredients.join("|").toLowerCase()}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push({
      name: fallbackName,
      ingredients
    });
  }
  return result;
}

function hasNutrition(result) {
  return Boolean(
    toNumber(result?.calories) != null
    || toNumber(result?.total_fat) != null
    || toNumber(result?.total_carbohydrates) != null
    || toNumber(result?.protein) != null
  );
}

// All four macro fields present.
function hasAllMacros(result) {
  return Boolean(
    toNumber(result?.calories) != null
    && toNumber(result?.total_fat) != null
    && toNumber(result?.total_carbohydrates) != null
    && toNumber(result?.protein) != null
  );
}

// Calories present but at least one macro missing — a partial set we must not ship.
function isPartialMacros(result) {
  return Boolean(toNumber(result?.calories) != null) && !hasAllMacros(result);
}

function parseJsonResponse(json) {
  const directOutput = typeof json?.output_text === "string" ? json.output_text.trim() : "";
  if (directOutput) {
    return JSON.parse(directOutput);
  }

  const messageText = (json?.output || [])
    .flatMap((item) => Array.isArray(item?.content) ? item.content : [])
    .find((item) => item?.type === "output_text" && typeof item?.text === "string")
    ?.text?.trim();
  if (!messageText) {
    throw new Error("No structured analyzer output returned.");
  }
  return JSON.parse(messageText);
}

function normalizeAnalysisResult(result = {}) {
  return {
    dish_name: trimString(result.dish_name),
    serving_size: trimString(result.serving_size),
    calories: toNumber(result.calories),
    total_fat: toNumber(result.total_fat),
    total_carbohydrates: toNumber(result.total_carbohydrates),
    protein: toNumber(result.protein),
    confidence: Math.max(0, Math.min(1, toNumber(result.confidence) ?? 0.5)),
    explanation: trimString(result.explanation),
    ingredients: normalizeStringList(result.ingredients),
    components: normalizeComponents(result.components),
    allergens: normalizeStringList(result.allergens),
    source_type: trimString(result.source_type),
    evidence_urls: normalizeStringList(result.evidence_urls)
  };
}

function buildUserPrompt(input = {}) {
  const lines = [
    "Analyze this spoken or typed meal description and return one canonical dish record.",
    `Dish text: ${trimString(input.dish_name) || "Unknown"}`,
    `Serving hint: ${trimString(input.serving_size) || "Unknown"}`
  ];

  const ingredients = normalizeStringList(input.ingredients);
  if (ingredients.length > 0) {
    lines.push(`Ingredients: ${ingredients.join(", ")}`);
  }

  const components = normalizeComponents(input.components);
  if (components.length > 0) {
    lines.push("Meal components:");
    for (const component of components) {
      lines.push(`- ${component.name}: ${component.ingredients.join(", ") || component.name}`);
    }
  }

  const correction = trimString(input.user_correction);
  if (correction) {
    lines.push(`User correction: ${correction}`);
  }

  const existing = input.existing_context || null;
  if (existing && (
    trimString(existing.dish_name)
    || trimString(existing.serving_size)
    || trimString(existing.explanation)
    || normalizeStringList(existing.ingredients).length > 0
  )) {
    lines.push("Existing dish context:");
    if (trimString(existing.dish_name)) {
      lines.push(`- Current title: ${trimString(existing.dish_name)}`);
    }
    if (trimString(existing.serving_size)) {
      lines.push(`- Current serving size: ${trimString(existing.serving_size)}`);
    }
    if (trimString(existing.explanation)) {
      lines.push(`- Current summary: ${trimString(existing.explanation)}`);
    }
    if (normalizeStringList(existing.ingredients).length > 0) {
      lines.push(`- Current ingredients: ${normalizeStringList(existing.ingredients).join(", ")}`);
    }
  }

  return lines.join("\n");
}

async function callResponsesApi(requestBody, env) {
  if (!env?.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for dish analysis.");
  }

  const response = await fetch(`${OPENAI_API_BASE}/responses`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Dish analysis failed with ${response.status}: ${text}`);
  }

  return response.json();
}

async function runAnalyzerPass(input, env, { allowWebSearch }) {
  const systemPrompt = allowWebSearch
    ? `You analyze spoken meals into nutrition records. Use web search when it helps anchor nutrition facts for common foods, branded products, or standard servings.

Return STRICT JSON only with this shape:
{
  "dish_name": string | null,
  "serving_size": string | null,
  "calories": number | null,
  "total_fat": number | null,
  "total_carbohydrates": number | null,
  "protein": number | null,
  "confidence": number,
  "explanation": string | null,
  "ingredients": string[],
  "components": [{"name": string, "ingredients": string[]}],
  "allergens": string[],
  "source_type": string | null,
  "evidence_urls": string[]
}

Rules:
1. Return one canonical dish title and one short user-facing explanation.
1a. Return a components array for distinct meal parts when the meal clearly has multiple parts, like juice + wrap + banana.
2. When clear nutrition facts exist on the web for the exact food or a standard serving, use them.
3. If the query is a custom mixed meal, reconcile from standard ingredient nutrition and the stated quantity.
4. If exact external data is weak, still return a best-effort estimate instead of leaving obvious foods blank.
4a. If you can estimate calories for a meal, you can and MUST estimate all four values — calories, protein, total_fat, and total_carbohydrates. Provide best-effort NUMERIC estimates for ALL of them; null is acceptable ONLY when the food is truly unidentifiable, in which case ALL four should be null, not a partial set.
5. Keep titles grounded in the latest ingredient set. If milk gets added, the title/summary should reflect that updated meal.
5a. When there are multiple obvious meal parts, make the title and explanation mention the important parts rather than only the first one.
6. Confidence should be higher when grounded by authoritative product or standard nutrition sources, and lower for rough estimates.
7. source_type should be "web_search" when web results materially informed the answer, otherwise "llm_estimate".`
    : `You analyze spoken meals into nutrition records using food knowledge and standard serving estimates when external lookup is unavailable.

Return STRICT JSON only with this shape:
{
  "dish_name": string | null,
  "serving_size": string | null,
  "calories": number | null,
  "total_fat": number | null,
  "total_carbohydrates": number | null,
  "protein": number | null,
  "confidence": number,
  "explanation": string | null,
  "ingredients": string[],
  "components": [{"name": string, "ingredients": string[]}],
  "allergens": string[],
  "source_type": string | null,
  "evidence_urls": string[]
}

Rules:
1. Estimate nutrition for the full spoken serving, not per 100g.
2. Return one canonical dish title and one short user-facing explanation.
2a. Return a components array for distinct meal parts when the meal clearly has multiple parts.
3. Keep titles grounded in the latest ingredient set.
3a. When there are multiple obvious meal parts, mention the important parts instead of collapsing to only the first part.
4. For simple foods like apples, bananas, yogurt cups, milk, eggs, toast, or sandwiches, return practical nutrition estimates rather than nulls.
4a. If you can estimate calories for a meal, you can and MUST estimate all four values — calories, protein, total_fat, and total_carbohydrates. Provide best-effort NUMERIC estimates for ALL of them; null is acceptable ONLY when the food is truly unidentifiable, in which case ALL four should be null, not a partial set.
5. source_type should be "llm_estimate".`;

  const dishModel = env.FULL_DISH_MODEL || env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
  const useReasoning = /^(o[1-9]|gpt-5)/.test(dishModel);
  const requestBody = {
    model: dishModel,
    ...(useReasoning ? { reasoning: { effort: "low" } } : {}),
    max_output_tokens: 1400,
    text: {
      format: {
        type: "json_schema",
        name: allowWebSearch ? "text_dish_nutrition_web" : "text_dish_nutrition_estimate",
        strict: true,
        schema: ANALYZER_SCHEMA
      }
    },
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }]
      },
      {
        role: "user",
        content: [{ type: "input_text", text: buildUserPrompt(input) }]
      }
    ]
  };

  if (allowWebSearch) {
    requestBody.tools = [{ type: "web_search_preview", search_context_size: "medium" }];
    requestBody.tool_choice = { type: "web_search_preview" };
    requestBody.include = ["web_search_call.action.sources"];
  }

  const json = await callResponsesApi(requestBody, env);
  return normalizeAnalysisResult(parseJsonResponse(json));
}

// If the chosen result has calories but a missing macro, do ONE corrective
// round-trip to fill the rest so we never log a calories-only dish (Matt hit a
// "974.5 cal, protein/fat/carbs NULL" row). If the retry still can't complete
// the set, accept the partial (don't block the log) but emit a visible marker.
async function ensureCompleteMacros(result, input, env) {
  if (!result || !isPartialMacros(result)) {
    return result;
  }
  try {
    const retryInput = {
      ...input,
      existing_context: {
        ...(input.existing_context || {}),
        dish_name: result.dish_name,
        explanation: result.explanation,
        ingredients: result.ingredients
      },
      user_correction: `${trimString(input.user_correction) ? input.user_correction + ". " : ""}Your previous analysis returned calories=${result.calories} but left one or more of protein/total_fat/total_carbohydrates as null. Provide best-effort NUMERIC estimates for ALL FOUR macros (calories, protein, total_fat, total_carbohydrates) for this exact meal — do NOT return null for any macro.`
    };
    const retry = await runAnalyzerPass(retryInput, env, { allowWebSearch: false });
    if (hasAllMacros(retry)) {
      // Take the retry's full macro set (self-consistent), keep the richer base fields.
      return {
        ...result,
        calories: retry.calories,
        total_fat: retry.total_fat,
        total_carbohydrates: retry.total_carbohydrates,
        protein: retry.protein,
        confidence: Math.min(result.confidence, retry.confidence)
      };
    }
  } catch (error) {
    console.warn("[WARN] dish macro completion retry failed:", error?.message || error);
  }
  console.log(JSON.stringify({
    evt: "dish_macros_partial",
    dish_name: result.dish_name,
    calories: result.calories,
    protein: result.protein,
    total_fat: result.total_fat,
    total_carbohydrates: result.total_carbohydrates
  }));
  return result;
}

export async function analyzeDishFromText(input = {}, env = process.env) {
  const webResult = await runAnalyzerPass(input, env, { allowWebSearch: true }).catch((error) => {
    console.warn("[WARN] text dish web analysis failed:", error?.message || error);
    return null;
  });

  const webLooksUsable = webResult
    && hasNutrition(webResult)
    && (webResult.confidence >= 0.7 || (webResult.evidence_urls?.length || 0) > 0);
  if (webLooksUsable) {
    return ensureCompleteMacros(webResult, input, env);
  }

  const fallbackResult = await runAnalyzerPass(input, env, { allowWebSearch: false });
  if (webResult && hasNutrition(webResult) && !hasNutrition(fallbackResult)) {
    return ensureCompleteMacros(webResult, input, env);
  }
  return ensureCompleteMacros(fallbackResult, input, env);
}
