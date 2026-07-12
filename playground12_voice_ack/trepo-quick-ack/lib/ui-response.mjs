const MAX_SCREENS = 6;
const MAX_TITLE_LENGTH = 16;
const MAX_SUBTITLE_LENGTH = 28;
const MAX_BODY_LENGTH = 72;
const MAX_ITEMS_PER_SCREEN = 3;
const MAX_ITEM_LENGTH = 28;
const MAX_FOOTER_LENGTH = 20;
const ALLOWED_STYLES = new Set(["default", "info", "warning", "success", "error", "neutral"]);
const READ_TOOL_NAMES = new Set([
  "get_shopping_list",
  "list_store_tabs",
  "get_kitchen_overview",
  "search_kitchen_item",
  "get_recent_discards",
  "get_recent_dishes",
  "get_dish_detail",
  "search_dish",
  "get_health_metrics",
  "get_rewards_catalog",
  "get_active_redemptions",
  "get_recipes",
  "get_recipe_suggestions",
  "get_recipe_detail",
  "get_meal_plan",
  "get_household_feed",
  "get_household_summary",
  "recommend_items_to_buy",
  "recommend_items_to_use_up"
]);

function cleanId(value, fallback = "screen") {
  const normalized = String(value || "")
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || fallback;
}

function cleanText(value) {
  return String(value || "")
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, "\"")
    .replace(/[–—]/g, "-")
    .replace(/…/g, "...")
    .replace(/[•]/g, "-")
    .replace(/[^A-Za-z0-9 .,;!?'"/&():%-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function truncateText(value, maxLength) {
  const text = cleanText(value);
  if (text.length <= maxLength) {
    return text;
  }

  if (maxLength <= 3) {
    return text.slice(0, maxLength);
  }

  return `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

function chunkParagraph(text, maxLength = MAX_BODY_LENGTH) {
  const normalized = cleanText(text);
  if (!normalized) {
    return [];
  }

  const chunks = [];
  let remaining = normalized;
  const minPreferredFill = Math.floor(maxLength * 0.72);

  while (remaining.length > maxLength && chunks.length < MAX_SCREENS) {
    let splitIndex = remaining.lastIndexOf(". ", maxLength);
    if (splitIndex < minPreferredFill) {
      splitIndex = remaining.lastIndexOf("; ", maxLength);
    }
    if (splitIndex < minPreferredFill) {
      splitIndex = remaining.lastIndexOf(", ", maxLength);
    }
    if (splitIndex < minPreferredFill) {
      splitIndex = remaining.lastIndexOf(" ", maxLength);
    }

    if (splitIndex <= 0) {
      splitIndex = maxLength;
    } else {
      splitIndex += 1;
    }

    chunks.push(remaining.slice(0, splitIndex).trim());
    remaining = remaining.slice(splitIndex).trim();
  }

  if (remaining && chunks.length < MAX_SCREENS) {
    chunks.push(remaining);
  }

  return chunks.slice(0, MAX_SCREENS);
}

function getItemPageSize(items) {
  return (items || []).every((item) => cleanText(item).length <= 14) ? 3 : 2;
}

function chunkItems(items, pageSize = null) {
  const cleaned = (items || [])
    .map((item) => truncateText(item, MAX_ITEM_LENGTH))
    .filter(Boolean);

  const resolvedPageSize = Math.max(1, pageSize || getItemPageSize(cleaned));
  const pages = [];
  for (let index = 0; index < cleaned.length; index += resolvedPageSize) {
    pages.push(cleaned.slice(index, index + resolvedPageSize));
  }

  return pages.slice(0, MAX_SCREENS);
}

function buildNavigation(screenCount) {
  if (screenCount <= 1) {
    return undefined;
  }

  return {
    mode: "knob_paged",
    wrap: false,
    showPageDots: true,
    showPageCount: true
  };
}

function normalizeBlock(block) {
  const nextBlock = { ...block };
  if (typeof nextBlock.text === "string") {
    nextBlock.text = cleanText(nextBlock.text);
  }
  if (Array.isArray(nextBlock.items)) {
    nextBlock.items = nextBlock.items.map((item) => {
      if (typeof item === "string") {
        return truncateText(item, MAX_ITEM_LENGTH);
      }
      if (item && typeof item === "object") {
        return {
          ...item,
          ...(typeof item.label === "string" ? { label: truncateText(item.label, MAX_ITEM_LENGTH) } : {}),
          ...(typeof item.value === "string" ? { value: truncateText(item.value, MAX_ITEM_LENGTH) } : {})
        };
      }
      return item;
    });
  }
  return nextBlock;
}

function normalizeScreen(screen, index) {
  const normalized = {
    id: cleanId(screen.id, `screen_${index + 1}`),
    template: screen.template || "title_body",
    style: ALLOWED_STYLES.has(screen.style) ? screen.style : "default"
  };

  if (screen.title) {
    normalized.title = truncateText(screen.title, MAX_TITLE_LENGTH);
  }
  if (screen.subtitle) {
    normalized.subtitle = truncateText(screen.subtitle, MAX_SUBTITLE_LENGTH);
  }
  if (screen.body) {
    normalized.body = truncateText(screen.body, MAX_BODY_LENGTH);
  }
  if (Array.isArray(screen.items) && screen.items.length > 0) {
    normalized.items = screen.items
      .slice(0, MAX_ITEMS_PER_SCREEN)
      .map((item) => truncateText(item, MAX_ITEM_LENGTH))
      .filter(Boolean);
  }
  if (screen.footer) {
    normalized.footer = truncateText(screen.footer, MAX_FOOTER_LENGTH);
  }
  if (screen.image) {
    normalized.image = screen.image;
  }
  if (Array.isArray(screen.blocks) && screen.blocks.length > 0) {
    normalized.blocks = screen.blocks.map(normalizeBlock);
  }

  return normalized;
}

function finalizeUi({ presentation, title, screens }) {
  const normalizedScreens = (screens || [])
    .slice(0, MAX_SCREENS)
    .map((screen, index) => normalizeScreen(screen, index))
    .filter((screen) => Boolean(screen.body || screen.items?.length || screen.blocks?.length || screen.title));

  if (normalizedScreens.length === 0) {
    return null;
  }

  return {
    presentation: presentation || (normalizedScreens.length > 1 ? "paged" : "single_screen"),
    title: truncateText(title || normalizedScreens[0]?.title || "What I Found", MAX_TITLE_LENGTH),
    ...(normalizedScreens.length > 1 ? { navigation: buildNavigation(normalizedScreens.length) } : {}),
    screens: normalizedScreens
  };
}

function buildSingleScreen({ id, template = "title_body", style = "default", title, subtitle, body, items, blocks, footer }) {
  return finalizeUi({
    presentation: "single_screen",
    title,
    screens: [{ id, template, style, title, subtitle, body, items, blocks, footer }]
  });
}

function normalizeAnswerText(value) {
  return String(value || "")
    .replace(/\r\n/g, "\n")
    .replace(/[•]/g, "-")
    .trim();
}

function formatAnswerEntry(value) {
  const cleaned = cleanText(String(value || "").replace(/^[-*]\s*/, ""));
  if (!cleaned) {
    return null;
  }
  if (cleaned.endsWith(":") || /[.!?]$/.test(cleaned)) {
    return cleaned;
  }
  return `${cleaned}.`;
}

function extractAnswerEntries(text) {
  const normalized = normalizeAnswerText(text);
  if (!normalized) {
    return [];
  }

  const blocks = normalized
    .split(/\n\s*\n/)
    .map((block) => block.split("\n").map((line) => line.trim()).filter(Boolean))
    .filter((lines) => lines.length > 0);

  const entries = [];
  for (const lines of blocks) {
    for (const line of lines) {
      const entry = formatAnswerEntry(line);
      if (entry) {
        entries.push(entry);
      }
    }
  }

  return entries;
}

function buildPagedAnswerUi({ title, subtitle, text, style = "default" }) {
  const entries = extractAnswerEntries(text);
  if (entries.length === 0) {
    return buildSingleScreen({
      id: "summary",
      template: "title_body",
      style,
      title,
      subtitle,
      body: cleanText(text)
    });
  }

  const screens = paginateBodyEntries({
    title,
    subtitle,
    style,
    entries
  });

  return finalizeUi({
    presentation: screens.length > 1 ? "paged" : "single_screen",
    title,
    screens
  });
}

function inferFallbackTitle(text, inferredType) {
  const normalized = cleanText(text).toLowerCase();

  if (inferredType === "clarification_needed") {
    return "Question";
  }
  if (inferredType === "error") {
    return "Error";
  }
  if (normalized.includes("shopping list")) {
    return "Shopping List";
  }
  if (normalized.includes("recommend") || normalized.includes("suggest")) {
    return "Recommendation";
  }
  if (normalized.includes("expire") || normalized.includes("expiring")) {
    return "Use Soon";
  }
  if (normalized.includes("discard")) {
    return "Discards";
  }
  if (normalized.includes("kitchen summary")) {
    return "Kitchen Summary";
  }
  if (normalized.includes("kitchen")) {
    return "What I Found";
  }

  return "What I Found";
}

const WRITE_TOOL_NAMES = new Set([
  "add_to_shopping_list",
  "add_many_to_shopping_list",
  "update_shopping_item_store",
  "update_many_shopping_item_stores",
  "mark_shopping_item_bought",
  "mark_shopping_item_unbought",
  "remove_from_shopping_list",
  "clear_shopping_list",
  "check_in_item",
  "check_in_many_items",
  "update_item_quantity",
  "mark_item_opened",
  "update_item_expiration",
  "update_item_location",
  "update_item_details",
  "discard_item",
  "delete_recent_discard",
  "clear_recent_discards",
  "clear_kitchen_inventory",
  "log_dish_from_voice",
  "log_dish_ingredients",
  "append_to_recent_dish",
  "update_recent_dish",
  "mark_dish_consumed",
  "delete_dish_log",
  "add_dish_ingredients_to_shopping_list",
  "add_recipe_ingredients_to_shopping_list",
  "add_saved_recipe_ingredients_to_shopping_list",
  "refresh_meal_plan",
  "redeem_reward",
  "update_redemption_status",
  "regenerate_meal_plan"
]);

function inferToolTitle(toolEvents, fallbackTitle) {
  const primaryEvent = [...(toolEvents || [])].reverse().find((event) => event?.result?.ok);
  if (primaryEvent?.toolName && WRITE_TOOL_NAMES.has(primaryEvent.toolName)) {
    return "Updated";
  }
  switch (primaryEvent?.toolName) {
    case "get_shopping_list":
      return "Shopping List";
    case "list_store_tabs":
      return "Store Tabs";
    case "get_kitchen_overview":
    case "search_kitchen_item":
      return "Kitchen";
    case "get_recent_discards":
      return "Discards";
    case "get_recent_dishes":
    case "get_dish_detail":
    case "search_dish":
      return "Recent Dishes";
    case "get_health_metrics":
      return "Health Metrics";
    case "get_recipes":
    case "get_recipe_suggestions":
    case "get_recipe_detail":
      return "Recipes";
    case "get_meal_plan":
      return "Meal Plan";
    case "get_household_feed":
      return "Recent Activity";
    case "get_household_summary":
      return "Household";
    case "recommend_items_to_buy":
      return "Buy Next";
    case "recommend_items_to_use_up":
      return "Use Soon";
    default:
      return fallbackTitle;
  }
}

function inferTypeFromToolEvents(toolEvents, fallbackType) {
  const lastWrite = [...(toolEvents || [])].reverse().find((event) => event?.result?.ok && WRITE_TOOL_NAMES.has(event.toolName));
  return lastWrite ? "action_ack" : fallbackType;
}

function getPagedContinuationTitle(baseTitle, index) {
  if (index === 0) {
    return baseTitle;
  }

  if (baseTitle === "Shopping List" || baseTitle === "What I Found") {
    return "More Items";
  }
  if (baseTitle === "Recommendation") {
    return "More Ideas";
  }
  if (baseTitle === "Kitchen Summary" || baseTitle === "Use Soon" || baseTitle === "Discards") {
    return "More Details";
  }

  return "More Details";
}

function buildPagedTextUi({ title, subtitle, body, style = "default", template = "title_body" }) {
  const chunks = chunkParagraph(body);
  if (chunks.length <= 1) {
    return buildSingleScreen({
      id: "summary",
      template,
      style,
      title,
      subtitle,
      body: chunks[0] || body
    });
  }

  const screens = chunks.map((chunk, index) => ({
    id: `page_${index + 1}`,
    template,
    style,
    title: getPagedContinuationTitle(title, index),
    ...(index === 0 && subtitle ? { subtitle } : {}),
    body: chunk
  }));

  return finalizeUi({
    presentation: "paged",
    title,
    screens
  });
}

function paginateBodyEntries({ title, subtitle, style, intro = null, entries = [] }) {
  const pages = [];
  let currentBody = intro ? cleanText(intro) : "";

  const pushCurrent = () => {
    if (!currentBody) {
      return;
    }
    pages.push({
      id: `page_${pages.length + 1}`,
      template: "title_body",
      style,
      title: getPagedContinuationTitle(title, pages.length),
      ...(pages.length === 0 && subtitle ? { subtitle } : {}),
      body: currentBody
    });
    currentBody = "";
  };

  entries.forEach((entry) => {
    const text = cleanText(entry);
    if (!text) {
      return;
    }

    const candidate = currentBody ? `${currentBody} ${text}` : text;
    if (candidate.length <= MAX_BODY_LENGTH) {
      currentBody = candidate;
      return;
    }

    pushCurrent();
    currentBody = text;
  });

  pushCurrent();
  return pages.slice(0, MAX_SCREENS);
}

function buildTextListUi({ title, subtitle, style, items, emptyBody, intro = null, typeWhenPopulated = "paged_answer" }) {
  if (!items || items.length === 0) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: `${cleanId(title, "list")}_empty`,
        template: "title_body",
        style: "neutral",
        title,
        body: emptyBody
      })
    };
  }

  const entries = (items || [])
    .map((item, index) => {
      const text = truncateText(item, MAX_BODY_LENGTH - 8);
      return text ? `${index + 1}. ${text}.` : null;
    })
    .filter(Boolean);
  const screens = paginateBodyEntries({
    title,
    subtitle,
    style,
    intro,
    entries
  });
  const ui = finalizeUi({
    presentation: screens.length > 1 ? "paged" : "single_screen",
    title,
    screens
  });

  return {
    type: (ui?.screens?.length || 0) > 1 ? typeWhenPopulated : "info_answer",
    ui
  };
}

function formatCountLabel(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function buildActionAckUi(text) {
  return buildPagedTextUi({
    title: "Updated",
    body: text,
    style: "success",
    template: "title_body"
  });
}

function buildShoppingListUi(toolEvent) {
  const items = toolEvent?.result?.toolResult?.items || [];
  const count = Number(toolEvent?.result?.toolResult?.count || items.length || 0);

  if (count === 0) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "shopping_empty",
        template: "title_body",
        style: "neutral",
        title: "Shopping List",
        body: "Your shopping list is empty right now."
      })
    };
  }

  return buildTextListUi({
    title: "Shopping List",
    subtitle: `${formatCountLabel(count, "item", "items")} total`,
    style: "default",
    items: items.map((item) => item.item_name || "Unknown item"),
    emptyBody: "Your shopping list is empty right now.",
    intro: `You have ${formatCountLabel(count, "item", "items")} on your shopping list.`
  });
}

function buildKitchenOverviewUi(toolEvent) {
  const items = toolEvent?.result?.toolResult?.items || [];
  const summary = toolEvent?.result?.toolResult?.summary || {};

  if (items.length === 0) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "kitchen_empty",
        template: "title_body",
        style: "neutral",
        title: "Kitchen",
        body: "Your kitchen looks empty right now."
      })
    };
  }

  const summaryBody = [
    `${formatCountLabel(summary.kitchen_count || items.length, "item", "items")} in kitchen.`,
    `${formatCountLabel(summary.shopping_count || 0, "item", "items")} on list.`,
    `${formatCountLabel(summary.discard_count || 0, "discard", "discards")}.`
  ].join(" ");

  return buildTextListUi({
    title: "Kitchen Summary",
    subtitle: "Current items",
    style: "info",
    items: items.map((item) => {
      const state = item.state_label ? ` - ${item.state_label}` : "";
      return `${item.item_name}${state}`;
    }),
    emptyBody: "Your kitchen looks empty right now.",
    intro: summaryBody
  });
}

function buildKitchenDetailUi(toolEvent) {
  const item = toolEvent?.result?.toolResult?.item || null;
  if (!item) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "item_missing",
        template: "title_body",
        style: "neutral",
        title: "Item Not Found",
        body: "I couldn't find that item in your kitchen."
      })
    };
  }

  const detailText = [
    ...(item.expiration_date ? [`Expires ${item.expiration_date}.`] : []),
    ...(item.state_label ? [`State: ${item.state_label}.`] : []),
    ...(item.storage_location ? [`Location: ${item.storage_location}.`] : []),
    `Opened: ${item.is_opened ? "Yes" : "No"}.`
  ].join(" ");

  return {
    type: "info_answer",
    ui: buildPagedTextUi({
      title: truncateText(item.item_name || "Kitchen Item", MAX_TITLE_LENGTH),
      body: detailText,
      style: "info",
      template: "title_body"
    })
  };
}

function buildDishDetailUi(toolEvent) {
  const dish = toolEvent?.result?.toolResult?.dish || toolEvent?.result?.toolResult?.item || null;
  if (!dish) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "dish_missing",
        template: "title_body",
        style: "neutral",
        title: "Dish Not Found",
        body: "I couldn't find that dish in recent meals."
      })
    };
  }

  const detailText = [
    ...(dish.analysis_status === "pending" ? ["Nutrition update pending."] : []),
    ...(dish.analysis_status === "failed" && dish.analysis_error ? [`Nutrition update failed: ${dish.analysis_error}.`] : []),
    ...(dish.explanation ? [`${dish.explanation}.`] : []),
    ...(dish.serving_size ? [`Serving: ${dish.serving_size}.`] : []),
    ...(dish.calories != null ? [`Calories: ${dish.calories}.`] : []),
    ...((dish.protein != null || dish.total_carbohydrates != null || dish.total_fat != null)
      ? [`Macros: P ${dish.protein ?? "-"}, C ${dish.total_carbohydrates ?? "-"}, F ${dish.total_fat ?? "-"}.`]
      : []),
    ...(dish.action ? [`Status: ${dish.action}.`] : []),
    ...(Array.isArray(dish.ingredients) && dish.ingredients.length > 0
      ? [`Ingredients: ${dish.ingredients.slice(0, 4).join(", ")}.`]
      : [])
  ].join(" ");

  return {
    type: "info_answer",
    ui: buildPagedTextUi({
      title: truncateText(dish.dish_name || "Dish", MAX_TITLE_LENGTH),
      body: detailText || "I found the dish, but there were not many details on it.",
      style: "info",
      template: "title_body"
    })
  };
}

function buildMetricsUi(toolEvent) {
  const metrics = toolEvent?.result?.toolResult?.metrics || null;
  if (!metrics) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "metrics_empty",
        template: "title_body",
        style: "neutral",
        title: "Health Metrics",
        body: "No health metrics were available yet."
      })
    };
  }

  const body = [
    `Kitchen IQ: ${metrics.IQ ?? 0}.`,
    `Points: ${metrics.Points ?? 0}.`,
    `UPF: ${metrics.UPF ?? 0}.`,
    `Harmful ingredients: ${metrics.harmful_ingredients ?? 0}.`,
    metrics.UPF_what ? `UPF: ${metrics.UPF_what}` : null,
    metrics.IQ_what ? `Kitchen IQ: ${metrics.IQ_what}` : null
  ].filter(Boolean).join(" ");

  return {
    type: "info_answer",
    ui: buildPagedTextUi({
      title: "Health Metrics",
      subtitle: "Latest snapshot",
      body,
      style: "info",
      template: "title_body"
    })
  };
}

function buildRecipesUi(toolEvent) {
  const payload = toolEvent?.result?.toolResult || {};
  const status = payload.status || "empty";
  const kitchenOnly = payload.kitchen_only || [];
  const needGrocery = payload.need_grocery || [];

  if (status !== "ready") {
    return {
      type: status === "failed" ? "error" : "info_answer",
      ui: buildSingleScreen({
        id: "recipes_status",
        template: "title_body",
        style: status === "failed" ? "error" : "info",
        title: "Recipes",
        body: status === "regenerating"
          ? "Recipes are updating from the latest kitchen changes."
          : status === "failed"
            ? (payload.error_message || "Recipe generation failed.")
            : "Add items to your kitchen to get recipes."
      })
    };
  }

  const body = [
    `${kitchenOnly.length} ready now. ${needGrocery.length} need grocery items.`,
    kitchenOnly.length > 0
      ? `Make now: ${kitchenOnly.slice(0, 5).map((recipe) => recipe.title || "Recipe").join(", ")}.`
      : null,
    needGrocery.length > 0
      ? `Need grocery: ${needGrocery.slice(0, 5).map((recipe) => recipe.title || "Recipe").join(", ")}.`
      : null,
    kitchenOnly.length > 0 || needGrocery.length > 0
      ? "Want me to show the recipe for any of these?"
      : null
  ].filter(Boolean).join("\n\n");

  return {
    type: "paged_answer",
    ui: buildPagedTextUi({
      title: "Recipes",
      body,
      style: "info",
      template: "title_body"
    })
  };
}

function buildRecipeDetailUi(toolEvent) {
  const payload = toolEvent?.result?.toolResult || {};
  if (payload.status && payload.status !== "ready" && !payload.recipe) {
    return {
      type: payload.status === "failed" ? "error" : "info_answer",
      ui: buildSingleScreen({
        id: "recipe_detail_status",
        template: "title_body",
        style: payload.status === "failed" ? "error" : "info",
        title: "Recipe",
        body: payload.status === "regenerating"
          ? "Recipes are updating from the latest kitchen changes."
          : payload.status === "failed"
            ? (payload.error_message || "Recipe generation failed.")
            : "No recipe details are ready yet."
      })
    };
  }

  const recipe = payload.recipe || null;
  if (!recipe) {
    return {
      type: "empty_state",
      ui: buildSingleScreen({
        id: "recipe_detail_empty",
        template: "title_body",
        style: "neutral",
        title: "Recipe",
        body: "I couldn't find that recipe."
      })
    };
  }

  const title = truncateText(recipe.title || "Recipe", MAX_TITLE_LENGTH);
  const subtitle = recipe.source_url
    ? `Recipe details${recipe.source_name ? ` · ${truncateText(recipe.source_name, 28)}` : ""}`
    : (recipe.bucket === "need_grocery" ? "Needs groceries" : "Recipe details");
  const summaryBody = [
    recipe.description || null,
    recipe.bucket === "need_grocery" && recipe.missing_ingredients?.length > 0
      ? `Need to buy: ${recipe.missing_ingredients.slice(0, 6).join(", ")}.`
      : "Ready now.",
    Array.isArray(recipe.available_ingredients) && recipe.available_ingredients.length > 0
      ? `You have: ${recipe.available_ingredients.slice(0, 5).join(", ")}.`
      : null,
    Array.isArray(recipe.ingredients) && recipe.ingredients.length > 0
      ? `${recipe.ingredients.length} ingredients.`
      : null,
    Array.isArray(recipe.steps) && recipe.steps.length > 0
      ? `${recipe.steps.length} steps.`
      : null,
    recipe.source_url ? "Source link available." : null
  ].filter(Boolean).join(" ");

  const haveEntries = Array.isArray(recipe.available_ingredients)
    ? recipe.available_ingredients.slice(0, 8).map((ingredient, index) => `Have ${index + 1}: ${ingredient}`)
    : [];
  const needEntries = Array.isArray(recipe.missing_ingredients)
    ? recipe.missing_ingredients.slice(0, 10).map((ingredient, index) => `Need ${index + 1}: ${ingredient}`)
    : [];
  const ingredientEntries = Array.isArray(recipe.ingredients)
    ? recipe.ingredients.slice(0, 10).map((ingredient, index) => `Ingredient ${index + 1}: ${ingredient}`)
    : [];
  const stepEntries = Array.isArray(recipe.steps)
    ? recipe.steps.slice(0, 8).map((step, index) => `Step ${index + 1}: ${step}`)
    : [];
  const noteEntries = Array.isArray(recipe.notes)
    ? recipe.notes.slice(0, 6).map((note, index) => `Note ${index + 1}: ${note}`)
    : [];
  const sourceEntries = [
    recipe.source_name ? `Source: ${recipe.source_name}` : null,
    recipe.author_name ? `Author: ${recipe.author_name}` : null,
    recipe.source_url ? `URL: ${recipe.source_url}` : null
  ].filter(Boolean);

  const screens = [
    {
      id: "recipe_summary",
      template: "title_body",
      style: "info",
      title,
      subtitle,
      body: summaryBody || "Recipe details are available."
    },
    ...paginateBodyEntries({
      title,
      subtitle: "You Have",
      style: "info",
      entries: haveEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_have_${index + 1}`,
      ...(index === 0 ? { title } : {})
    })),
    ...paginateBodyEntries({
      title,
      subtitle: "You'll Need",
      style: "info",
      entries: needEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_need_${index + 1}`,
      ...(index === 0 ? { title } : {})
    })),
    ...paginateBodyEntries({
      title,
      subtitle: "Ingredients",
      style: "info",
      entries: ingredientEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_ingredients_${index + 1}`,
      ...(index === 0 ? { title } : {})
    })),
    ...paginateBodyEntries({
      title,
      subtitle: "Steps",
      style: "info",
      entries: stepEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_steps_${index + 1}`,
      ...(index === 0 ? { title } : {})
    })),
    ...paginateBodyEntries({
      title,
      subtitle: "Notes",
      style: "info",
      entries: noteEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_notes_${index + 1}`,
      ...(index === 0 ? { title } : {})
    })),
    ...paginateBodyEntries({
      title,
      subtitle: "Source",
      style: "info",
      entries: sourceEntries
    }).map((screen, index) => ({
      ...screen,
      id: `recipe_source_${index + 1}`,
      ...(index === 0 ? { title } : {})
    }))
  ];

  return {
    type: screens.length > 1 ? "paged_answer" : "info_answer",
    ui: finalizeUi({
      presentation: screens.length > 1 ? "paged" : "single_screen",
      title,
      screens
    })
  };
}

function buildMealPlanUi(toolEvent) {
  const payload = toolEvent?.result?.toolResult || {};
  const status = payload.status || "empty";
  const plan = payload.plan || [];

  if (status !== "ready") {
    return {
      type: status === "failed" ? "error" : "info_answer",
      ui: buildSingleScreen({
        id: "meal_plan_status",
        template: "title_body",
        style: status === "failed" ? "error" : "info",
        title: "Meal Plan",
        body: status === "regenerating"
          ? "Your meal plan is regenerating now."
          : status === "failed"
            ? (payload.error_message || "Meal plan generation failed.")
            : "No meal plan is ready yet."
      })
    };
  }

  const planLines = plan
    .slice(0, 8)
    .map((slot) => `${slot.day_label} ${slot.meal_type}: ${slot.recipe?.title || "Open slot"}.`);
  const body = [
    payload.focus
      ? `Focus: ${payload.focus}. ${plan.length} meals are planned.`
      : `${plan.length} meals are planned.`,
    planLines.length > 0 ? `Planned meals: ${planLines.join(" ")}` : "No planned meals are ready yet."
  ].join("\n\n");

  return {
    type: "paged_answer",
    ui: buildPagedTextUi({
      title: "Meal Plan",
      body,
      style: "info",
      template: "title_body"
    })
  };
}

function buildHouseholdSummaryUi(toolEvent) {
  const summary = toolEvent?.result?.toolResult || {};
  return {
    type: "info_answer",
    ui: buildSingleScreen({
      id: "household_summary",
      template: "title_body",
      style: "info",
      title: "Household",
      body: `${formatCountLabel(summary.kitchen_count || 0, "kitchen item", "kitchen items")}. ${formatCountLabel(summary.shopping_count || 0, "shopping item", "shopping items")}. ${formatCountLabel(summary.discard_count || 0, "discard", "discards")}. ${formatCountLabel(summary.recent_dish_count || 0, "recent dish", "recent dishes")}.`
    })
  };
}

function buildNamedListUi({ title, subtitle, style, items, emptyBody, typeWhenPopulated = "paged_answer" }) {
  return buildTextListUi({
    title,
    subtitle,
    style,
    items,
    emptyBody,
    intro: items?.length ? `${formatCountLabel(items.length, "item", "items")}.` : null,
    typeWhenPopulated
  });
}

function inferTypeFromText(text, error) {
  const normalized = cleanText(text);
  if (error) {
    return "error";
  }
  if (!normalized) {
    return "empty_state";
  }
  if (normalized.endsWith("?")) {
    return "clarification_needed";
  }
  if (normalized.length > MAX_BODY_LENGTH) {
    return "paged_answer";
  }

  return "info_answer";
}

export function buildVoiceUiResponse({ text, quickItems = [], toolEvents = [], error = null, responseSurface = "halo" }) {
  const normalizedText = cleanText(text) || "Okay.";
  const inferredType = quickItems.length > 0
    ? "action_ack"
    : inferTypeFromText(normalizedText, error);
  const resolvedType = inferTypeFromToolEvents(toolEvents, inferredType);
  const fallbackTitle = inferFallbackTitle(normalizedText, resolvedType);
  const resolvedTitle = inferToolTitle(toolEvents, fallbackTitle);

  const fallbackUi = buildPagedAnswerUi({
    title: resolvedTitle,
    text: text || normalizedText,
    style: resolvedType === "error"
      ? "error"
      : resolvedType === "clarification_needed"
        ? "warning"
        : resolvedType === "action_ack"
          ? "success"
          : "default"
  });

  return {
    version: "2",
    type: resolvedType,
    ui: fallbackUi
  };
}
