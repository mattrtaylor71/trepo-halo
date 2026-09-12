import { formatRecipeResponse } from "./recipe-response.mjs";

function cleanText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

const READ_TOOL_NAMES = new Set([
  "get_shopping_list",
  "list_store_tabs",
  "get_kitchen_overview",
  "search_kitchen_item",
  "get_recent_discards",
  "get_recent_dishes",
  "get_dish_detail",
  "get_health_metrics",
  "get_recipe_suggestions",
  "get_recipe_detail",
  "get_saved_recipes",
  "get_saved_recipe_detail",
  "get_meal_plan",
  "recommend_items_to_buy",
  "recommend_items_to_use_up"
]);

function normalizeDisplayWhitespace(value) {
  return String(value || "")
    .replace(/[•]/g, "-")
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function stripSimpleMarkdown(value) {
  return String(value || "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1");
}

function splitTrailingFollowUp(value) {
  const match = String(value || "").match(/^(.*?)(\s+(Want me|Should I|Do you want|Would you like|I can also)\b.*)$/i);
  if (!match) {
    return { item: String(value || "").trim(), followUp: "" };
  }
  return {
    item: String(match[1] || "").trim(),
    followUp: String(match[2] || "").trim()
  };
}

function formatInlineDashParagraph(paragraph) {
  const normalized = cleanText(stripSimpleMarkdown(paragraph));
  const dashMatches = normalized.match(/\s-\s/g) || [];
  if (dashMatches.length < 2) {
    return normalized;
  }

  const parts = normalized.split(/\s-\s+/).map((part) => part.trim()).filter(Boolean);
  if (parts.length < 3) {
    return normalized;
  }

  const intro = parts.shift();
  const bullets = [];
  let trailingFollowUp = "";

  for (const part of parts) {
    const split = splitTrailingFollowUp(part);
    if (split.item) {
      bullets.push(split.item);
    }
    if (!trailingFollowUp && split.followUp) {
      trailingFollowUp = split.followUp;
    }
  }

  if (bullets.length < 2) {
    return normalized;
  }

  return [
    intro,
    "",
    ...bullets.map((item) => `- ${item}`),
    ...(trailingFollowUp ? ["", trailingFollowUp] : [])
  ].join("\n");
}

function formatBulletRunParagraph(paragraph) {
  const normalized = cleanText(stripSimpleMarkdown(paragraph));
  if (!normalized.startsWith("- ") || !normalized.includes(" - ")) {
    return normalized;
  }

  const bullets = normalized
    .split(/\s+-\s+/)
    .map((part, index) => (index === 0 ? part.replace(/^-+\s*/, "") : part).trim())
    .filter(Boolean);

  if (bullets.length < 2) {
    return normalized;
  }

  return bullets.map((item) => `- ${item}`).join("\n");
}

function expandInlineListLines(value) {
  const output = [];

  for (const rawLine of String(value || "").split("\n")) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      if (output.length > 0 && output[output.length - 1] !== "") {
        output.push("");
      }
      continue;
    }

    const normalized = cleanText(trimmed);
    const parts = normalized.split(/\s+-\s+/).map((part, index) => (
      index === 0 ? part.replace(/^-+\s*/, "").trim() : part.trim()
    )).filter(Boolean);

    if (normalized.includes(": - ") && parts.length >= 2) {
      output.push(parts[0]);
      output.push("");
      for (const item of parts.slice(1)) {
        output.push(`- ${item}`);
      }
      continue;
    }

    if (normalized.startsWith("- ") && parts.length >= 2) {
      for (const item of parts) {
        output.push(`- ${item}`);
      }
      continue;
    }

    output.push(normalized);
  }

  return output.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function formatMessageText(value) {
  const normalized = normalizeDisplayWhitespace(stripSimpleMarkdown(value));
  if (!normalized) {
    return "";
  }

  const paragraphs = normalized
    .split(/\n\s*\n/)
    .map((paragraph) => formatInlineDashParagraph(paragraph))
    .map((paragraph) => formatBulletRunParagraph(paragraph))
    .filter(Boolean);

  const expanded = expandInlineListLines(
    paragraphs
      .join("\n\n")
      .replace(/:\s+- /g, ":\n\n- ")
      .replace(/\n- ([^\n]+)\s+- (?=[^\n]+)/g, "\n- $1\n- ")
  );

  return expanded.replace(/(\n- [^\n]+)\n{2,}(?=- )/g, "$1\n");
}

function unwrapEntity(value) {
  return value?.item && typeof value.item === "object" ? value.item : value;
}

function summarizeShoppingItem(item) {
  const entity = unwrapEntity(item);
  return {
    id: entity?.shopping_id || entity?.household_item_uuid || null,
    name: entity?.item_name || "Unknown item",
    store: entity?.store || null,
    status: entity?.action || null
  };
}

function summarizeKitchenItem(item) {
  const entity = unwrapEntity(item);
  return {
    id: entity?.id || null,
    name: entity?.item_name || "Unknown item",
    brand: entity?.brand || null,
    variant: entity?.variant || null,
    category: entity?.category || null,
    confidence: entity?.confidence ?? null,
    explanation: entity?.explanation || null,
    description: entity?.description || null,
    barcode: entity?.barcode || null,
    country_guess: entity?.country_guess || null,
    estimated_price: entity?.estimated_price || null,
    location: entity?.storage_location || null,
    state: entity?.state_label || null,
    remaining_quantity: entity?.remaining_quantity || null,
    quantity_value: entity?.quantity_value ?? null,
    quantity_unit: entity?.quantity_unit || null,
    fill_percent: entity?.fill_percent ?? null,
    expiration_date: entity?.expiration_date || null,
    is_opened: Boolean(entity?.is_opened),
    upf: entity?.upf || null,
    harmful_ingredients: Array.isArray(entity?.harmful_ingredients) ? entity.harmful_ingredients : [],
    similar_items: Array.isArray(entity?.similar_items) ? entity.similar_items : [],
    alternatives: Array.isArray(entity?.alternatives) ? entity.alternatives : [],
    healthier_alternatives: Array.isArray(entity?.healthier_alternatives) ? entity.healthier_alternatives : [],
    store_availability: entity?.store_availability ?? null,
    ingredients: Array.isArray(entity?.ingredients) ? entity.ingredients : [],
    nutrition_summary: entity?.nutrition_summary || null,
    image_url: entity?.image_url || null,
    product_image_url: entity?.product_image_url || null,
    images: entity?.images || null,
    action: entity?.action || null,
    created_at: entity?.created_at || null,
    updated_at: entity?.updated_at || null
  };
}

function summarizeDiscard(item) {
  const entity = unwrapEntity(item);
  return {
    id: entity?.id || null,
    name: entity?.item_name || "Unknown item",
    brand: entity?.brand || null,
    category: entity?.category || null,
    discard_reason: entity?.discard_reason || null,
    created_at: entity?.created_at || null
  };
}

function summarizeDish(dish) {
  const entity = unwrapEntity(dish);
  return {
    id: entity?.id || null,
    name: entity?.dish_name || "Unknown dish",
    explanation: entity?.explanation || null,
    serving_size: entity?.serving_size || null,
    calories: entity?.calories ?? null,
    protein: entity?.protein ?? null,
    carbs: entity?.total_carbohydrates ?? null,
    fat: entity?.total_fat ?? null,
    ingredients: Array.isArray(entity?.ingredients) ? entity.ingredients : [],
    components: Array.isArray(entity?.components) ? entity.components.map((component) => ({
      name: component?.name || null,
      ingredients: Array.isArray(component?.ingredients) ? component.ingredients : []
    })) : [],
    status: entity?.action || null,
    analysis_status: entity?.analysis_status || null,
    analysis_error: entity?.analysis_error || null,
    created_at: entity?.created_at || null
  };
}

function summarizeRecipe(recipe) {
  const entity = unwrapEntity(recipe);
  return {
    title: entity?.title || "Unknown recipe",
    bucket: entity?.bucket || null,
    description: entity?.description || null,
    ingredients: Array.isArray(entity?.ingredients) ? entity.ingredients : [],
    available_ingredients: Array.isArray(entity?.available_ingredients) ? entity.available_ingredients : [],
    missing_ingredients: Array.isArray(entity?.missing_ingredients) ? entity.missing_ingredients : [],
    steps: Array.isArray(entity?.steps) ? entity.steps : [],
    notes: Array.isArray(entity?.notes) ? entity.notes : [],
    image_url: entity?.image_url || null,
    source_url: entity?.source_url || null,
    resolved_url: entity?.resolved_url || null,
    source_name: entity?.source_name || null,
    author_name: entity?.author_name || null
  };
}

function summarizeSavedRecipe(recipe) {
  const entity = unwrapEntity(recipe);
  return {
    id: entity?.id || null,
    title: entity?.title || "Unknown recipe",
    source_url: entity?.source_url || null,
    resolved_url: entity?.resolved_url || null,
    author_name: entity?.author_name || null,
    ingredients: Array.isArray(entity?.ingredients) ? entity.ingredients : [],
    instructions: Array.isArray(entity?.instructions) ? entity.instructions : [],
    notes: Array.isArray(entity?.notes) ? entity.notes : [],
    source_type: entity?.source_type || "tiktok",
    extraction_source: entity?.extraction_source || null
  };
}

function summarizeMealPlanSlot(slot) {
  return {
    day_index: slot?.day_index ?? null,
    day_label: slot?.day_label || null,
    meal_type: slot?.meal_type || null,
    recipe: slot?.recipe ? summarizeRecipe(slot.recipe) : null
  };
}

function buildActions(toolTrace = []) {
  return toolTrace.map((action) => ({
    tool_name: action.toolName,
    ok: Boolean(action.ok),
    status_code: action.statusCode || 200,
    summary: action.actionSummary || null,
    error: action.error || null,
    args: action.args || {}
  }));
}

function buildActionResultCard(primaryEvent, text) {
  const toolResult = primaryEvent?.result?.toolResult || {};
  const singleItem = unwrapEntity(toolResult.item || toolResult.dish || toolResult.discard || null);
  const listItems = Array.isArray(toolResult.items) ? toolResult.items : [];

  return {
    type: "action_result",
    title: primaryEvent?.result?.actionSummary || "Update complete",
    body: formatMessageText(text),
    ...(singleItem ? {
      entity: singleItem.shopping_id || singleItem.household_item_uuid
        ? summarizeShoppingItem(singleItem)
        : singleItem.dish_name
          ? summarizeDish(singleItem)
          : (singleItem.discard_reason != null || primaryEvent?.toolName === "delete_recent_discard")
            ? summarizeDiscard(singleItem)
          : summarizeKitchenItem(singleItem)
    } : {}),
    ...(listItems.length > 0 ? {
      items: listItems
        .slice(0, 25)
        .map((item) => {
          const entity = unwrapEntity(item);
          if (entity.shopping_id || entity.household_item_uuid) {
            return summarizeShoppingItem(entity);
          }
          if (entity.dish_name) {
            return summarizeDish(entity);
          }
          if (entity.discard_reason != null) {
            return summarizeDiscard(entity);
          }
          return summarizeKitchenItem(entity);
        })
    } : {})
  };
}

function buildShoppingListCard(toolResult = {}) {
  const items = Array.isArray(toolResult.items) ? toolResult.items : [];
  return {
    type: "shopping_list",
    title: "Shopping List",
    total_count: Number(toolResult.count || items.length || 0),
    items: items.slice(0, 50).map(summarizeShoppingItem)
  };
}

function buildStoreTabsCard(toolResult = {}) {
  const stores = Array.isArray(toolResult.stores) ? toolResult.stores : [];
  return {
    type: "store_tabs",
    title: "Store Tabs",
    total_count: Number(toolResult.count || stores.length || 0),
    stores: stores.slice(0, 50).map((store) => cleanText(store)).filter(Boolean)
  };
}

function buildKitchenOverviewCard(toolResult = {}) {
  const items = Array.isArray(toolResult.items) ? toolResult.items : [];
  const additionalCompact = Array.isArray(toolResult.additional_items_compact)
    ? toolResult.additional_items_compact
    : [];
  return {
    type: "kitchen_overview",
    title: "Kitchen",
    total_count: Number(toolResult.count || items.length || 0),
    summary: toolResult.summary || null,
    items: items.slice(0, 50).map(summarizeKitchenItem),
    // Compact names for items beyond the detailed head so the card reflects the
    // whole kitchen, not just the most-recent detailed slice.
    additional_items: additionalCompact
  };
}

function buildKitchenItemCard(toolResult = {}) {
  if (!toolResult.item) {
    return {
      type: "message",
      title: "Kitchen Item",
      body: "I couldn't find that item in the kitchen."
    };
  }

  return {
    type: "kitchen_item",
    title: toolResult.item?.item_name || "Kitchen Item",
    item: summarizeKitchenItem(toolResult.item || {})
  };
}

function buildDishListCard(toolResult = {}) {
  const items = Array.isArray(toolResult.items) ? toolResult.items : [];
  return {
    type: "recent_dishes",
    title: "Recent Dishes",
    total_count: Number(toolResult.count || items.length || 0),
    items: items.slice(0, 25).map(summarizeDish)
  };
}

function buildDiscardListCard(toolResult = {}) {
  const items = Array.isArray(toolResult.items) ? toolResult.items : [];
  return {
    type: "recent_discards",
    title: "Recent Discards",
    total_count: Number(toolResult.count || items.length || 0),
    items: items.slice(0, 25).map(summarizeDiscard)
  };
}

function buildDishDetailCard(toolResult = {}) {
  return {
    type: "dish_detail",
    title: toolResult.dish?.dish_name || "Dish",
    dish: summarizeDish(toolResult.dish || {})
  };
}

function buildHealthMetricsCard(toolResult = {}) {
  return {
    type: "health_metrics",
    title: "Health Metrics",
    metrics: toolResult.metrics || null
  };
}

function buildRecipeSuggestionsCard(toolResult = {}) {
  return {
    type: "recipe_suggestions",
    title: "Recipes",
    status: toolResult.status || "empty",
    kitchen_only: Array.isArray(toolResult.kitchen_only) ? toolResult.kitchen_only.slice(0, 10).map(summarizeRecipe) : [],
    need_grocery: Array.isArray(toolResult.need_grocery) ? toolResult.need_grocery.slice(0, 10).map(summarizeRecipe) : [],
    error_message: toolResult.error_message || null
  };
}

function buildRecipeDetailCard(toolResult = {}) {
  return {
    type: "recipe_detail",
    title: toolResult.recipe?.title || "Recipe",
    status: toolResult.status || "empty",
    recipe: toolResult.recipe ? summarizeRecipe(toolResult.recipe) : null,
    error_message: toolResult.error_message || null
  };
}

function buildSavedRecipesCard(toolResult = {}) {
  const recipes = Array.isArray(toolResult.recipes) ? toolResult.recipes : [];
  return {
    type: "saved_recipes",
    title: "Saved Recipes",
    total_count: Number(toolResult.count || recipes.length || 0),
    recipes: recipes.slice(0, 25).map(summarizeSavedRecipe)
  };
}

function buildSavedRecipeDetailCard(toolResult = {}) {
  return {
    type: "saved_recipe_detail",
    title: toolResult.recipe?.title || "Saved Recipe",
    recipe: toolResult.recipe ? summarizeSavedRecipe(toolResult.recipe) : null
  };
}

function buildMealPlanCard(toolResult = {}) {
  return {
    type: "meal_plan",
    title: "Meal Plan",
    status: toolResult.status || "empty",
    focus: toolResult.focus || null,
    explanation_title: toolResult.explanation_title || null,
    explanation_paragraph: toolResult.explanation_paragraph || null,
    plan: Array.isArray(toolResult.plan) ? toolResult.plan.slice(0, 12).map(summarizeMealPlanSlot) : [],
    error_message: toolResult.error_message || null
  };
}

function buildRecommendationCard(toolResult = {}, title) {
  return {
    type: "recommendations",
    title,
    total_count: Number(toolResult.count || (Array.isArray(toolResult.items) ? toolResult.items.length : 0)),
    items: Array.isArray(toolResult.items) ? toolResult.items.slice(0, 25) : []
  };
}

function buildFallbackCard(text, responseType, error) {
  return {
    type: responseType === "clarification_needed"
      ? "clarification"
      : responseType === "error" || error
        ? "error"
        : "message",
    title: responseType === "clarification_needed"
      ? "Question"
      : responseType === "error" || error
        ? "Error"
        : "Assistant",
    body: formatMessageText(text)
  };
}

function buildCards(text, responseType, toolEvents = [], error = null) {
  const okEvents = [...toolEvents].filter((event) => event?.result?.ok);
  if (okEvents.length > 1 && okEvents.every((event) => READ_TOOL_NAMES.has(event.toolName))) {
    // For plain conversational replies, message.text is the canonical app bubble.
    // Avoid duplicating the exact same content in a fallback card.
    return [];
  }

  const primaryEvent = [...okEvents].reverse().find(Boolean);
  if (!primaryEvent) {
    return [];
  }

  const toolName = primaryEvent.toolName;
  const toolResult = primaryEvent.result?.toolResult || {};

  switch (toolName) {
    case "get_shopping_list":
      return [buildShoppingListCard(toolResult)];
    case "list_store_tabs":
      return [buildStoreTabsCard(toolResult)];
    case "get_kitchen_overview":
      return [buildKitchenOverviewCard(toolResult)];
    case "search_kitchen_item":
      return [buildKitchenItemCard(toolResult)];
    case "get_recent_discards":
      return [buildDiscardListCard(toolResult)];
    case "get_recent_dishes":
      return [buildDishListCard(toolResult)];
    case "get_dish_detail":
      return [buildDishDetailCard(toolResult)];
    case "get_health_metrics":
      return [buildHealthMetricsCard(toolResult)];
    case "get_recipe_suggestions":
      return [buildRecipeSuggestionsCard(toolResult)];
    case "get_recipe_detail":
      return [buildRecipeDetailCard(toolResult)];
    case "get_saved_recipes":
      return [buildSavedRecipesCard(toolResult)];
    case "get_saved_recipe_detail":
    case "save_recipe_from_tiktok":
    case "add_saved_recipe_ingredients_to_shopping_list":
      return [buildSavedRecipeDetailCard(toolResult)];
    case "get_meal_plan":
      return [buildMealPlanCard(toolResult)];
    case "recommend_items_to_buy":
      return [buildRecommendationCard(toolResult, "Buy Next")];
    case "recommend_items_to_use_up":
      return [buildRecommendationCard(toolResult, "Use Soon")];
    default:
      return [buildActionResultCard(primaryEvent, text)];
  }
}

export function buildAppOutput({
  text,
  transcript = null,
  quickItems = [],
  toolTrace = [],
  toolEvents = [],
  type = "info_answer",
  error = null,
  speechDetected = null,
  durationMs = null,
  sessionId = null,
  memoryUsed = false,
  responseSurface = "halo"
}) {
  const normalizedResponseSurface = String(responseSurface || "halo").trim().toLowerCase() === "app" ? "app" : "halo";
  const recipeText = normalizedResponseSurface === "app" ? formatRecipeResponse(text) : null;
  const normalizedText = recipeText || formatMessageText(text) || "Okay.";
  const responseType = error ? "error" : type;
  const primaryEvent = [...toolEvents].reverse().find((event) => event?.result?.ok);
  const asyncUpdate = primaryEvent?.result?.toolResult?.async_update || null;

  return {
    version: "1",
    presentation: "chat",
    status: error ? "error" : "ok",
    response_type: responseType,
    message: {
      role: "assistant",
      text: normalizedText
    },
    transcript: transcript || null,
    quick_items: Array.isArray(quickItems) ? quickItems.filter(Boolean) : [],
    actions: buildActions(toolTrace),
    cards: buildCards(normalizedText, responseType, toolEvents, error),
    meta: {
      response_surface: normalizedResponseSurface,
      speech_detected: speechDetected,
      duration_ms: durationMs,
      async_update: asyncUpdate,
      ...(sessionId ? {
        session_id: sessionId,
        memory_used: Boolean(memoryUsed)
      } : {})
    }
  };
}
