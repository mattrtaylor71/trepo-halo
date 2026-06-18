import { SUPPORTED_TOOLS } from "./tool-definitions.mjs";

const LIST_TOOLS = new Set([
  "get_shopping_list",
  "list_store_tabs",
  "get_kitchen_overview",
  "get_recent_discards",
  "get_recent_dishes",
  "get_saved_recipes"
]);

export function toTitleCase(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function humanizePrimaryEntity(args) {
  const value = String(args?.item_name || args?.dish_name || args?.recipe_title || "the item").trim();
  return value ? toTitleCase(value) : "the item";
}

function trimString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function normalizeBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  const normalized = trimString(value)?.toLowerCase();
  if (!normalized) {
    return null;
  }
  if (["true", "yes", "opened", "open", "1"].includes(normalized)) {
    return true;
  }
  if (["false", "no", "closed", "unopened", "0"].includes(normalized)) {
    return false;
  }
  return null;
}

function normalizeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeStringList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => trimString(entry))
    .filter(Boolean);
}

const FORBIDDEN_SCOPE_OVERRIDE_KEYS = new Set([
  "owner",
  "ownerid",
  "userid",
  "user",
  "tableownerid",
  "householdmemberids",
  "shoppingnamespace"
]);

function findForbiddenScopeOverride(value, path = "args") {
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      const match = findForbiddenScopeOverride(value[index], `${path}[${index}]`);
      if (match) {
        return match;
      }
    }
    return null;
  }

  if (!value || typeof value !== "object") {
    return null;
  }

  for (const [key, nestedValue] of Object.entries(value)) {
    const normalizedKey = String(key || "").replace(/[^a-z0-9]+/gi, "").toLowerCase();
    if (FORBIDDEN_SCOPE_OVERRIDE_KEYS.has(normalizedKey)) {
      return {
        key,
        path: `${path}.${key}`
      };
    }
    const nestedMatch = findForbiddenScopeOverride(nestedValue, `${path}.${key}`);
    if (nestedMatch) {
      return nestedMatch;
    }
  }

  return null;
}

function normalizeShoppingBatchItems(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => (item && typeof item === "object" ? item : null))
    .filter(Boolean)
    .map((item) => ({
      ...(trimString(item.item_name) ? { item_name: item.item_name.trim() } : {}),
      ...(trimString(item.store) ? { store: item.store.trim() } : {}),
      ...(trimString(item.quantity) ? { quantity: item.quantity.trim() } : {})
    }))
    .filter((item) => item.item_name);
}

function normalizeKitchenBatchItems(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => normalizeKitchenArgs(item).args)
    .filter((item) => item?.item_name);
}

function normalizeKitchenArgs(args) {
  const normalizedArgs = typeof args === "object" && args !== null ? args : {};
  const itemName = trimString(normalizedArgs.item_name);
  const itemId = trimString(normalizedArgs.item_id);
  const selectionHint = trimString(normalizedArgs.selection_hint);
  const quantityValue = normalizeNumber(normalizedArgs.quantity_value);
  const fillPercent = normalizeNumber(normalizedArgs.fill_percent);
  const isOpened = normalizeBoolean(normalizedArgs.is_opened);

  return {
    args: {
      ...(itemName ? { item_name: itemName } : {}),
      ...(itemId ? { item_id: itemId } : {}),
      ...(selectionHint ? { selection_hint: selectionHint } : {}),
      ...(trimString(normalizedArgs.quantity) ? { quantity: normalizedArgs.quantity.trim() } : {}),
      ...(trimString(normalizedArgs.remaining_quantity) ? { remaining_quantity: normalizedArgs.remaining_quantity.trim() } : {}),
      ...(quantityValue != null ? { quantity_value: quantityValue } : {}),
      ...(trimString(normalizedArgs.quantity_unit) ? { quantity_unit: normalizedArgs.quantity_unit.trim() } : {}),
      ...(fillPercent != null ? { fill_percent: fillPercent } : {}),
      ...(trimString(normalizedArgs.location) ? { location: normalizedArgs.location.trim() } : {}),
      ...(trimString(normalizedArgs.expiration_date) ? { expiration_date: normalizedArgs.expiration_date.trim() } : {}),
      ...(isOpened != null ? { is_opened: isOpened } : {}),
      ...(trimString(normalizedArgs.brand) ? { brand: normalizedArgs.brand.trim() } : {}),
      ...(trimString(normalizedArgs.category) ? { category: normalizedArgs.category.trim() } : {}),
      ...(trimString(normalizedArgs.reason) ? { reason: normalizedArgs.reason.trim() } : {})
    }
  };
}

function normalizeDishArgs(args) {
  const normalizedArgs = typeof args === "object" && args !== null ? args : {};
  const dishName = trimString(normalizedArgs.dish_name);
  const dishId = trimString(normalizedArgs.dish_id);

  return {
    args: {
      ...(dishName ? { dish_name: dishName } : {}),
      ...(dishId ? { dish_id: dishId } : {}),
      ...(trimString(normalizedArgs.serving_size) ? { serving_size: normalizedArgs.serving_size.trim() } : {}),
      ...(normalizeStringList(normalizedArgs.ingredients).length > 0 ? { ingredients: normalizeStringList(normalizedArgs.ingredients) } : {}),
      ...(normalizeStringList(normalizedArgs.add_ingredients).length > 0 ? { add_ingredients: normalizeStringList(normalizedArgs.add_ingredients) } : {}),
      ...(normalizeStringList(normalizedArgs.remove_ingredients).length > 0 ? { remove_ingredients: normalizeStringList(normalizedArgs.remove_ingredients) } : {}),
      ...(normalizeNumber(normalizedArgs.calories) != null ? { calories: normalizeNumber(normalizedArgs.calories) } : {}),
      ...(normalizeNumber(normalizedArgs.protein_g) != null ? { protein_g: normalizeNumber(normalizedArgs.protein_g) } : {}),
      ...(normalizeNumber(normalizedArgs.carbs_g) != null ? { carbs_g: normalizeNumber(normalizedArgs.carbs_g) } : {}),
      ...(normalizeNumber(normalizedArgs.fat_g) != null ? { fat_g: normalizeNumber(normalizedArgs.fat_g) } : {}),
      ...(trimString(normalizedArgs.store) ? { store: normalizedArgs.store.trim() } : {})
    }
  };
}

function normalizeBatchLimit(value) {
  const numeric = normalizeNumber(value);
  if (numeric == null) {
    return null;
  }
  return Math.max(1, Math.min(Math.floor(numeric), 25));
}

function summarizeItemList(items, formatter, fallback = "No items found.") {
  if (!Array.isArray(items) || items.length === 0) {
    return fallback;
  }

  return items.slice(0, 5).map(formatter).join("\n");
}

function hasKitchenQuantityUpdate(args) {
  return Boolean(
    trimString(args?.remaining_quantity)
    || trimString(args?.quantity)
    || normalizeNumber(args?.quantity_value) != null
    || trimString(args?.quantity_unit)
    || normalizeNumber(args?.fill_percent) != null
  );
}

function hasKitchenDetailUpdate(args) {
  return Boolean(
    hasKitchenQuantityUpdate(args)
    || trimString(args?.location)
    || trimString(args?.expiration_date)
    || trimString(args?.brand)
    || trimString(args?.category)
    || normalizeBoolean(args?.is_opened) != null
  );
}

function hasDishUpdate(args) {
  return Boolean(
    trimString(args?.dish_name)
    || trimString(args?.serving_size)
    || (Array.isArray(args?.ingredients) && args.ingredients.length > 0)
    || (Array.isArray(args?.add_ingredients) && args.add_ingredients.length > 0)
    || (Array.isArray(args?.remove_ingredients) && args.remove_ingredients.length > 0)
    || normalizeNumber(args?.calories) != null
    || normalizeNumber(args?.protein_g) != null
    || normalizeNumber(args?.carbs_g) != null
    || normalizeNumber(args?.fat_g) != null
  );
}

export function validateToolCall(toolName, args) {
  const normalizedArgs = typeof args === "object" && args !== null ? args : {};
  const forbiddenScopeOverride = findForbiddenScopeOverride(normalizedArgs);
  const itemName = trimString(normalizedArgs.item_name);
  const itemId = trimString(normalizedArgs.item_id);
  const dishName = trimString(normalizedArgs.dish_name);
  const recipeTitle = trimString(normalizedArgs.recipe_title);
  const url = trimString(normalizedArgs.url);
  const limit = normalizeNumber(normalizedArgs.limit);
  const shoppingBatchItems = normalizeShoppingBatchItems(normalizedArgs.items);
  const kitchenArgs = normalizeKitchenArgs(normalizedArgs).args;
  const kitchenBatchItems = normalizeKitchenBatchItems(normalizedArgs.items);
  const dishArgs = normalizeDishArgs(normalizedArgs).args;

  if (!SUPPORTED_TOOLS.has(toolName)) {
    return {
      ok: false,
      error: `Unsupported tool: ${toolName}`
    };
  }

  if (forbiddenScopeOverride) {
    return {
      ok: false,
      error: `Tool calls cannot include owner or user scope overrides (${forbiddenScopeOverride.key}). The assistant can only operate on the current request household.`
    };
  }

  const requiresItemName = new Set([
    "add_to_shopping_list",
    "update_shopping_item_store",
    "remove_from_shopping_list",
    "mark_shopping_item_bought",
    "mark_shopping_item_unbought",
    "search_kitchen_item",
    "check_in_item",
    "update_item_quantity",
    "mark_item_opened",
    "update_item_expiration",
    "update_item_location",
    "update_item_details",
    "discard_item"
    ,
    "delete_recent_discard"
  ]);
  const requiresDishName = new Set([
    "log_dish_from_voice"
  ]);
  const requiresRecipeTitle = new Set([
    "get_recipe_detail",
    "add_recipe_ingredients_to_shopping_list",
    "get_saved_recipe_detail",
    "remove_saved_recipe"
  ]);

  const kitchenResolvableTools = new Set([
    "search_kitchen_item",
    "update_item_quantity",
    "mark_item_opened",
    "update_item_expiration",
    "update_item_location",
    "update_item_details",
    "discard_item"
  ]);

  if (requiresItemName.has(toolName) && !itemName && !(kitchenResolvableTools.has(toolName) && itemId)) {
    return {
      ok: false,
      error: "The tool call is missing a required item_name."
    };
  }

  if (requiresDishName.has(toolName) && !dishName) {
    return {
      ok: false,
      error: "The tool call is missing a required dish_name."
    };
  }

  if (requiresRecipeTitle.has(toolName) && !recipeTitle) {
    return {
      ok: false,
      error: "The tool call is missing a required recipe_title."
    };
  }

  if (toolName === "save_recipe_from_tiktok" && !url) {
    return {
      ok: false,
      error: "The tool call is missing a required url."
    };
  }

  if (toolName === "search_web_recipes" && !trimString(normalizedArgs.query)) {
    return {
      ok: false,
      error: "The tool call is missing a required search query.",
      statusCode: 400
    };
  }

  if (toolName === "update_shopping_item_store" && !trimString(normalizedArgs.store)) {
    return {
      ok: false,
      error: "The tool call is missing a required store."
    };
  }

  if (toolName === "add_many_to_shopping_list" && shoppingBatchItems.length === 0) {
    return {
      ok: false,
      error: "The tool call is missing valid items to add."
    };
  }

  if (
    toolName === "update_many_shopping_item_stores"
    && (shoppingBatchItems.length === 0 || shoppingBatchItems.some((item) => !item.store))
  ) {
    return {
      ok: false,
      error: "The tool call is missing valid item_name/store pairs."
    };
  }

  if (toolName === "check_in_many_items" && kitchenBatchItems.length === 0) {
    return {
      ok: false,
      error: "The tool call is missing valid kitchen items to check in."
    };
  }

  if (toolName === "update_item_quantity" && !hasKitchenQuantityUpdate(kitchenArgs)) {
    return {
      ok: false,
      error: "The tool call is missing a quantity update."
    };
  }

  if (toolName === "update_item_expiration" && !trimString(normalizedArgs.expiration_date)) {
    return {
      ok: false,
      error: "The tool call is missing an expiration_date."
    };
  }

  if (toolName === "update_item_location" && !trimString(normalizedArgs.location)) {
    return {
      ok: false,
      error: "The tool call is missing a location."
    };
  }

  if (toolName === "update_item_details" && !hasKitchenDetailUpdate(kitchenArgs)) {
    return {
      ok: false,
      error: "The tool call is missing kitchen details to update."
    };
  }

  if (toolName === "log_dish_ingredients" && (!Array.isArray(dishArgs.ingredients) || dishArgs.ingredients.length === 0)) {
    return {
      ok: false,
      error: "The tool call is missing ingredients to log."
    };
  }

  if (toolName === "append_to_recent_dish" && (!Array.isArray(dishArgs.ingredients) || dishArgs.ingredients.length === 0)) {
    return {
      ok: false,
      error: "The tool call is missing ingredients to append."
    };
  }

  if (toolName === "update_recent_dish" && !hasDishUpdate(dishArgs)) {
    return {
      ok: false,
      error: "The tool call is missing dish updates."
    };
  }

  const sharedArgs = {
    ...normalizedArgs,
    ...(itemName ? { item_name: itemName } : {}),
    ...(dishName ? { dish_name: dishName } : {}),
    ...(recipeTitle ? { recipe_title: recipeTitle } : {}),
    ...(trimString(normalizedArgs.recipe_bucket) ? { recipe_bucket: normalizedArgs.recipe_bucket.trim() } : {}),
    ...(trimString(normalizedArgs.focus) ? { focus: normalizedArgs.focus.trim() } : {}),
    ...(trimString(normalizedArgs.metric_name) ? { metric_name: normalizedArgs.metric_name.trim() } : {}),
    ...(url ? { url } : {}),
    ...(normalizeBoolean(normalizedArgs.only_missing) != null ? { only_missing: normalizeBoolean(normalizedArgs.only_missing) } : {}),
    ...(limit != null ? { limit: normalizeBatchLimit(limit) ?? limit } : {})
  };

  if (toolName.startsWith("check_in_") || toolName.startsWith("update_item_") || toolName === "mark_item_opened" || toolName === "discard_item" || toolName === "search_kitchen_item") {
    return {
      ok: true,
      args: {
        ...sharedArgs,
        ...kitchenArgs,
        ...(toolName === "check_in_many_items" ? { items: kitchenBatchItems } : {})
      }
    };
  }

  if (
    toolName.startsWith("log_dish_")
    || toolName === "append_to_recent_dish"
    || toolName === "update_recent_dish"
    || toolName === "get_dish_detail"
    || toolName === "mark_dish_consumed"
    || toolName === "delete_dish_log"
    || toolName === "add_dish_ingredients_to_shopping_list"
  ) {
    return {
      ok: true,
      args: {
        ...sharedArgs,
        ...dishArgs
      }
    };
  }

  return {
    ok: true,
    args: {
      ...sharedArgs,
      ...(trimString(normalizedArgs.quantity) ? { quantity: normalizedArgs.quantity.trim() } : {}),
      ...(trimString(normalizedArgs.store) ? { store: normalizedArgs.store.trim() } : {}),
      ...(shoppingBatchItems.length > 0 ? { items: shoppingBatchItems } : {})
    }
  };
}

function buildMockKitchenItem(args = {}) {
  return {
    id: "mock-kitchen-id",
    item_name: args.item_name || "Mock item",
    brand: args.brand || null,
    category: args.category || null,
    expiration_date: args.expiration_date || null,
    remaining_quantity: args.remaining_quantity || args.quantity || null,
    quantity_value: args.quantity_value ?? null,
    quantity_unit: args.quantity_unit || null,
    fill_percent: args.fill_percent ?? null,
    state_label: args.remaining_quantity || args.quantity || (args.fill_percent != null ? `${args.fill_percent}%` : null),
    is_opened: Boolean(args.is_opened),
    storage_location: args.location || null,
    created_at: null,
    updated_at: null,
    action: "IN"
  };
}

function buildMockDish(args = {}) {
  const ingredients = args.ingredients || args.add_ingredients || [];
  return {
    id: args.dish_id || "mock-dish-id",
    dish_name: args.dish_name || (Array.isArray(ingredients) && ingredients.length > 0 ? ingredients.join(", ") : "Mock dish"),
    serving_size: args.serving_size || null,
    calories: args.calories ?? null,
    protein: args.protein_g ?? null,
    total_carbohydrates: args.carbs_g ?? null,
    total_fat: args.fat_g ?? null,
    explanation: Array.isArray(ingredients) && ingredients.length > 0 ? `${ingredients.join(", ")}.` : "Mock dish.",
    ingredients,
    allergens: [],
    created_at: null,
    updated_at: null,
    action: "IN"
  };
}

export function buildMockToolResult(toolName, args) {
  const batchCount = Array.isArray(args?.items) ? args.items.length : 0;

  switch (toolName) {
    case "get_shopping_list":
      return { ok: true, mocked: true, items: [], count: 0, message: "Mocked get_shopping_list successfully." };
    case "list_store_tabs":
      return { ok: true, mocked: true, stores: [], count: 0, message: "Mocked list_store_tabs successfully." };
    case "clear_shopping_list":
      return { ok: true, mocked: true, items: [], count: 0, message: "Mocked clear_shopping_list successfully." };
    case "clear_kitchen_inventory":
      return { ok: true, mocked: true, items: [], count: 0, message: "Mocked clear_kitchen_inventory successfully." };
    case "add_many_to_shopping_list":
    case "update_many_shopping_item_stores":
      return { ok: true, mocked: true, items: args.items || [], count: batchCount, message: `Mocked ${toolName} successfully.` };
    case "get_kitchen_overview":
      return {
        ok: true,
        mocked: true,
        items: [],
        count: 0,
        summary: {
          household_size: 0,
          kitchen_count: 0,
          shopping_count: 0,
          discard_count: 0,
          recent_dish_count: 0,
          latest_metrics: null
        },
        message: "Mocked get_kitchen_overview successfully."
      };
    case "search_kitchen_item":
      return {
        ok: true,
        mocked: true,
        found: false,
        item: null,
        message: "Mocked search_kitchen_item successfully."
      };
    case "check_in_item":
      return {
        ok: true,
        mocked: true,
        id: "mock-kitchen-id",
        item: buildMockKitchenItem(args),
        message: "Mocked check_in_item successfully."
      };
    case "check_in_many_items":
      return {
        ok: true,
        mocked: true,
        items: (args.items || []).map((item, index) => ({
          id: `mock-kitchen-id-${index + 1}`,
          item: buildMockKitchenItem(item)
        })),
        count: batchCount,
        message: "Mocked check_in_many_items successfully."
      };
    case "update_item_quantity":
    case "mark_item_opened":
    case "update_item_expiration":
    case "update_item_location":
    case "update_item_details":
      return {
        ok: true,
        mocked: true,
        item: buildMockKitchenItem(args),
        message: `Mocked ${toolName} successfully.`
      };
    case "discard_item":
      return {
        ok: true,
        mocked: true,
        discard_id: "mock-discard-id",
        item: buildMockKitchenItem(args),
        message: "Mocked discard_item successfully."
      };
    case "get_recent_discards":
      return {
        ok: true,
        mocked: true,
        items: [],
        count: 0,
        message: "Mocked get_recent_discards successfully."
      };
    case "delete_recent_discard":
      return {
        ok: true,
        mocked: true,
        discard: {
          id: "mock-discard-id",
          item_name: args.item_name || "Mock discard",
          discard_reason: args.reason || null,
          created_at: null
        },
        message: "Mocked delete_recent_discard successfully."
      };
    case "clear_recent_discards":
      return {
        ok: true,
        mocked: true,
        items: [],
        count: 0,
        message: "Mocked clear_recent_discards successfully."
      };
    case "log_dish_from_voice":
    case "log_dish_ingredients":
    case "append_to_recent_dish":
    case "update_recent_dish":
    case "get_dish_detail":
    case "mark_dish_consumed":
    case "delete_dish_log":
      return {
        ok: true,
        mocked: true,
        dish: buildMockDish(args),
        recent_context: {
          resolved_by: "mock",
          confidence: "high"
        },
        message: `Mocked ${toolName} successfully.`
      };
    case "get_recent_dishes":
      return {
        ok: true,
        mocked: true,
        items: [],
        count: 0,
        message: "Mocked get_recent_dishes successfully."
      };
    case "add_dish_ingredients_to_shopping_list":
      return {
        ok: true,
        mocked: true,
        items: (args.ingredients || []).map((ingredient) => ({ item_name: ingredient, store: args.store || null })),
        count: Array.isArray(args.ingredients) ? args.ingredients.length : 0,
        message: "Mocked add_dish_ingredients_to_shopping_list successfully."
      };
    case "get_health_metrics":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        count: 1,
        metrics: {
          IQ: 75,
          Points: 420,
          UPF: 36.5,
          harmful_ingredients: 14,
          IQ_what: "IQ reflects scan frequency and consistency over time.",
          IQ_suggestions: ["Scan every meal"],
          UPF_what: "UPF shows what percent of recent groceries are ultra-processed.",
          UPF_suggestions: ["Choose whole foods"],
          harmful_ingredients_what: "This counts harmful additives in recent items.",
          harmful_ingredients_suggestions: ["Pick shorter ingredient lists"]
        },
        message: "Mocked get_health_metrics successfully."
      };
    case "get_recipe_suggestions":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        status: "ready",
        kitchen_only: [{
          title: "Mock Omelet",
          ingredients: ["eggs", "cheese"],
          steps: ["Whisk eggs.", "Cook in pan."],
          image_url: null,
          bucket: "kitchen_only"
        }],
        need_grocery: [{
          title: "Mock Smoothie",
          ingredients: ["milk", "berries", "banana"],
          steps: ["Blend all ingredients."],
          image_url: null,
          missing_ingredients: ["berries"],
          bucket: "need_grocery"
        }],
        error_message: null,
        message: "Mocked get_recipe_suggestions successfully."
      };
    case "get_recipe_detail":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        status: "ready",
        recipe: {
          title: args.recipe_title || "Mock Recipe",
          ingredients: ["milk", "berries", "banana"],
          steps: ["Blend all ingredients."],
          image_url: null,
          missing_ingredients: ["berries"],
          bucket: args.recipe_bucket || "need_grocery"
        },
        message: "Mocked get_recipe_detail successfully."
      };
    case "add_recipe_ingredients_to_shopping_list":
      return {
        ok: true,
        mocked: true,
        recipe: {
          title: args.recipe_title || "Mock Recipe"
        },
        items: [{ item_name: "berries", store: args.store || null }],
        count: 1,
        message: "Mocked add_recipe_ingredients_to_shopping_list successfully."
      };
    case "add_saved_recipe_ingredients_to_shopping_list":
      return {
        ok: true,
        mocked: true,
        recipe: {
          title: args.recipe_title || "Mock TikTok Pasta"
        },
        items: [{ item_name: "pasta", store: args.store || null }],
        count: 1,
        message: "Mocked add_saved_recipe_ingredients_to_shopping_list successfully."
      };
    case "get_saved_recipes":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        recipes: [{
          id: "mock-saved-recipe-id",
          title: "Mock TikTok Pasta",
          source_url: "https://www.tiktok.com/@creator/video/123",
          resolved_url: "https://www.tiktok.com/@creator/video/123",
          author_name: "Mock Creator",
          ingredients: ["pasta", "butter", "parmesan"],
          instructions: ["Boil pasta.", "Mix with butter and parmesan."],
          notes: ["Save extra pasta water."],
          source_type: "tiktok",
          extraction_source: "oembed",
          status: "ready"
        }],
        count: 1,
        message: "Mocked get_saved_recipes successfully."
      };
    case "get_saved_recipe_detail":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        recipe: {
          id: "mock-saved-recipe-id",
          title: args.recipe_title || "Mock TikTok Pasta",
          source_url: "https://www.tiktok.com/@creator/video/123",
          resolved_url: "https://www.tiktok.com/@creator/video/123",
          author_name: "Mock Creator",
          ingredients: ["pasta", "butter", "parmesan"],
          instructions: ["Boil pasta.", "Mix with butter and parmesan."],
          notes: ["Save extra pasta water."],
          source_type: "tiktok",
          extraction_source: "oembed",
          status: "ready"
        },
        message: "Mocked get_saved_recipe_detail successfully."
      };
    case "save_recipe_from_tiktok":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        recipe: {
          id: "mock-saved-recipe-id",
          title: "Mock TikTok Pasta",
          source_url: args.url || "https://www.tiktok.com/@creator/video/123",
          resolved_url: args.url || "https://www.tiktok.com/@creator/video/123",
          author_name: "Mock Creator",
          ingredients: ["pasta", "butter", "parmesan"],
          instructions: ["Boil pasta.", "Mix with butter and parmesan."],
          notes: ["Save extra pasta water."],
          source_type: "tiktok",
          extraction_source: "oembed",
          status: "ready"
        },
        deduped: false,
        message: "Mocked save_recipe_from_tiktok successfully."
      };
    case "save_generated_recipe":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        title: args.title || "Mock Generated Recipe",
        recipe: {
          id: "mock-saved-recipe-id",
          title: args.title || "Mock Generated Recipe",
          ingredients: Array.isArray(args.ingredients) ? args.ingredients : ["ingredient"],
          instructions: Array.isArray(args.steps) ? args.steps : ["step"],
          source_type: "generated",
          status: "ready"
        },
        message: "Mocked save_generated_recipe successfully."
      };
    case "search_web_recipes":
      return {
        ok: true,
        mocked: true,
        recipes: [{
          title: "Mock Apple Crumble",
          ingredients: ["4 apples", "1 cup flour", "1/2 cup butter", "3/4 cup sugar", "1 tsp cinnamon"],
          steps: ["Peel and slice apples.", "Mix flour, butter, sugar, cinnamon.", "Top apples with crumble.", "Bake at 350°F for 40 minutes."],
          notes: [],
          image_url: null,
          source_url: "https://example.com/apple-crumble",
          source_name: "Example",
          author_name: null,
          available_ingredients: ["4 apples"],
          missing_ingredients: ["1 cup flour", "1/2 cup butter", "3/4 cup sugar", "1 tsp cinnamon"],
          bucket: "need_grocery"
        }],
        count: 1,
        query: args?.query || "apple crumble"
      };
    case "remove_saved_recipe":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        recipe: {
          id: "mock-saved-recipe-id",
          title: args.recipe_title || "Mock TikTok Pasta"
        },
        message: "Mocked remove_saved_recipe successfully."
      };
    case "get_meal_plan":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        status: "ready",
        focus: "higher protein",
        explanation_title: "Higher protein meal plan",
        explanation_paragraph: "This plan increases protein across the next three days.",
        plan: [{
          day_index: 0,
          day_label: "Today",
          meal_type: "Breakfast",
          recipe: {
            title: "Greek Yogurt Bowl",
            ingredients: ["yogurt", "berries"],
            steps: ["Assemble and serve."],
            image_url: null
          }
        }],
        message: "Mocked get_meal_plan successfully."
      };
    case "refresh_meal_plan":
      return {
        ok: true,
        mocked: true,
        owner: "mock-owner",
        status: "regenerating",
        focus: args.focus || "great overall health",
        async_update: {
          pending: true,
          status: "regenerating",
          type: "meal_plan"
        },
        message: "Mocked refresh_meal_plan successfully."
      };
    case "recommend_items_to_buy":
      return {
        ok: true,
        mocked: true,
        items: [{ item_name: "eggs", reason: "It is already on the shopping list." }],
        count: 1,
        message: "Mocked recommend_items_to_buy successfully."
      };
    case "recommend_items_to_use_up":
      return {
        ok: true,
        mocked: true,
        items: [{ item_name: "spinach", reason: "Use soon because it expires tomorrow." }],
        count: 1,
        message: "Mocked recommend_items_to_use_up successfully."
      };
    default:
      return {
        ok: true,
        mocked: true,
        tool: toolName,
        received: args,
        message: `Mocked ${toolName} successfully.`
      };
  }
}

export function formatActionSummary(name, args, toolResult = {}) {
  const itemName = humanizePrimaryEntity({
    ...args,
    item_name: toolResult?.item?.item_name || args?.item_name,
    dish_name: toolResult?.dish?.dish_name || args?.dish_name,
    recipe_title: toolResult?.recipe?.title || args?.recipe_title
  });

  switch (name) {
    case "add_to_shopping_list":
      return toolResult?.item?.store
        ? `Added ${itemName} to your shopping list for ${toolResult.item.store}.`
        : `Added ${itemName} to your shopping list.`;
    case "add_many_to_shopping_list":
      return `Added ${toolResult?.count || 0} shopping list item${toolResult?.count === 1 ? "" : "s"}.`;
    case "update_shopping_item_store":
      return `Updated ${itemName} to ${toolResult?.item?.store || args?.store || "the requested store"}.`;
    case "update_many_shopping_item_stores":
      return `Updated ${toolResult?.count || 0} shopping store assignment${toolResult?.count === 1 ? "" : "s"}.`;
    case "mark_shopping_item_bought":
      return `Marked ${itemName} as bought.`;
    case "mark_shopping_item_unbought":
      return `Marked ${itemName} as still needed.`;
    case "remove_from_shopping_list":
      return `Removed ${itemName} from your shopping list.`;
    case "clear_shopping_list":
      return `Cleared ${toolResult?.count || 0} shopping list item${toolResult?.count === 1 ? "" : "s"}.`;
    case "get_shopping_list":
      return `Loaded ${toolResult?.count || 0} shopping list item${toolResult?.count === 1 ? "" : "s"}.`;
    case "list_store_tabs":
      return `Loaded ${toolResult?.count || 0} shopping store tab${toolResult?.count === 1 ? "" : "s"}.`;
    case "get_kitchen_overview":
      return `Loaded ${toolResult?.count || 0} kitchen item${toolResult?.count === 1 ? "" : "s"}.`;
    case "search_kitchen_item":
      return toolResult?.found
        ? `Found ${itemName} in your kitchen.`
        : `Couldn't find ${itemName} in your kitchen.`;
    case "check_in_item":
      return `Checked in ${itemName} to your kitchen.`;
    case "check_in_many_items":
      return `Checked in ${toolResult?.count || 0} kitchen item${toolResult?.count === 1 ? "" : "s"}.`;
    case "update_item_quantity":
      if (toolResult?.item?.discarded) {
        return `Discarded ${itemName} from your kitchen.`;
      }
      return `Updated ${itemName} to ${toolResult?.item?.state_label || args?.remaining_quantity || args?.quantity || "the new amount"}.`;
    case "mark_item_opened":
      return `Marked ${itemName} as opened.`;
    case "update_item_expiration":
      return `Updated ${itemName}'s expiration date.`;
    case "update_item_location":
      return `Moved ${itemName} to ${toolResult?.item?.storage_location || args?.location || "the new location"}.`;
    case "update_item_details":
      if (toolResult?.item?.discarded) {
        return `Discarded ${itemName} from your kitchen.`;
      }
      return `Updated details for ${itemName}.`;
    case "discard_item":
      return `Discarded ${itemName} from your kitchen.`;
    case "clear_kitchen_inventory":
      return `Cleared ${toolResult?.count || 0} kitchen item${toolResult?.count === 1 ? "" : "s"}.`;
    case "get_recent_discards":
      return `Loaded ${toolResult?.count || 0} recent discard${toolResult?.count === 1 ? "" : "s"}.`;
    case "delete_recent_discard":
      return `Removed ${itemName} from discard history.`;
    case "clear_recent_discards":
      return `Cleared ${toolResult?.count || 0} discard${toolResult?.count === 1 ? "" : "s"}.`;
    case "log_dish_from_voice":
      return toolResult?.dish?.analysis_status === "pending"
        ? `Logged ${toolResult?.dish?.dish_name || itemName || "your meal"}. Nutrition is updating in the background.`
        : `Logged ${toolResult?.dish?.dish_name || itemName || "your meal"}.`;
    case "log_dish_ingredients":
      return toolResult?.dish?.analysis_status === "pending"
        ? `Logged a dish with ${toolResult?.dish?.ingredients?.length || args?.ingredients?.length || 0} ingredient${(toolResult?.dish?.ingredients?.length || args?.ingredients?.length || 0) === 1 ? "" : "s"}. Nutrition is updating in the background.`
        : `Logged a dish with ${toolResult?.dish?.ingredients?.length || args?.ingredients?.length || 0} ingredient${(toolResult?.dish?.ingredients?.length || args?.ingredients?.length || 0) === 1 ? "" : "s"}.`;
    case "append_to_recent_dish":
      return toolResult?.dish?.analysis_status === "pending"
        ? `Updated your recent dish with ${args?.ingredients?.join(", ") || "the new ingredients"}. Nutrition is updating in the background.`
        : `Updated your recent dish with ${args?.ingredients?.join(", ") || "the new ingredients"}.`;
    case "update_recent_dish":
      return toolResult?.dish?.analysis_status === "pending"
        ? "Updated your recent dish. Nutrition is updating in the background."
        : "Updated your recent dish.";
    case "get_recent_dishes":
      return `Loaded ${toolResult?.count || 0} recent dish${toolResult?.count === 1 ? "" : "es"}.`;
    case "get_dish_detail":
      return toolResult?.dish?.dish_name
        ? `Loaded details for ${toolResult.dish.dish_name}.`
        : "Loaded dish details.";
    case "mark_dish_consumed":
      return `Marked ${toolResult?.dish?.dish_name || args?.dish_name || "that dish"} as consumed.`;
    case "delete_dish_log":
      return `Deleted ${toolResult?.dish?.dish_name || args?.dish_name || "that dish"} from recent dishes.`;
    case "add_dish_ingredients_to_shopping_list":
      return `Added ${toolResult?.count || 0} dish ingredient${toolResult?.count === 1 ? "" : "s"} to your shopping list.`;
    case "get_health_metrics":
      return "Loaded your latest health metrics.";
    case "get_recipe_suggestions":
      return `Loaded ${(toolResult?.kitchen_only?.length || 0) + (toolResult?.need_grocery?.length || 0)} recipe suggestion${((toolResult?.kitchen_only?.length || 0) + (toolResult?.need_grocery?.length || 0)) === 1 ? "" : "s"}.`;
    case "get_recipe_detail":
      return toolResult?.recipe?.title
        ? `Loaded details for ${toolResult.recipe.title}.`
        : "Loaded recipe details.";
    case "add_recipe_ingredients_to_shopping_list":
      return `Added ${toolResult?.count || 0} recipe ingredient${toolResult?.count === 1 ? "" : "s"} to your shopping list.`;
    case "add_saved_recipe_ingredients_to_shopping_list":
      return `Added ${toolResult?.count || 0} saved recipe ingredient${toolResult?.count === 1 ? "" : "s"} to your shopping list.`;
    case "get_saved_recipes":
      return `Loaded ${toolResult?.count || 0} saved recipe${toolResult?.count === 1 ? "" : "s"}.`;
    case "search_web_recipes":
      return `Found ${toolResult?.count || 0} web recipe${toolResult?.count === 1 ? "" : "s"} for "${trimString(args?.query) || "your search"}".`;
    case "get_saved_recipe_detail":
      return toolResult?.recipe?.title
        ? `Loaded saved recipe ${toolResult.recipe.title}.`
        : "Loaded saved recipe details.";
    case "save_recipe_from_tiktok":
      return toolResult?.recipe?.title
        ? `${toolResult?.deduped ? "Loaded" : "Saved"} ${toolResult.recipe.title} from that recipe URL.`
        : "Saved that recipe from the URL.";
    case "save_generated_recipe":
      return toolResult?.recipe?.title || toolResult?.title || args?.title
        ? `Saved ${toolResult?.recipe?.title || toolResult?.title || args?.title} to your recipes.`
        : "Saved that recipe to your recipes.";
    case "remove_saved_recipe":
      return `Removed ${itemName} from your saved recipes.`;
    case "get_meal_plan":
      return `Loaded your meal plan${toolResult?.status ? ` (${toolResult.status})` : ""}.`;
    case "refresh_meal_plan":
      return toolResult?.status === "regenerating"
        ? `Started regenerating your meal plan${args?.focus ? ` for ${args.focus}` : ""}.`
        : "Updated your meal plan request.";
    case "recommend_items_to_buy":
      return `Loaded ${toolResult?.count || 0} buy recommendation${toolResult?.count === 1 ? "" : "s"}.`;
    case "recommend_items_to_use_up":
      return `Loaded ${toolResult?.count || 0} use-up recommendation${toolResult?.count === 1 ? "" : "s"}.`;
    default:
      return `Completed ${toTitleCase(name)}.`;
  }
}

export function formatActionDetails(name, args, toolResult = {}) {
  const lines = [];

  if (args?.item_name) {
    lines.push(`Item: ${args.item_name}`);
  }
  if (args?.dish_name) {
    lines.push(`Dish: ${args.dish_name}`);
  }
  if (args?.dish_id) {
    lines.push(`Dish ID: ${args.dish_id}`);
  }
  if (args?.recipe_title) {
    lines.push(`Recipe: ${args.recipe_title}`);
  }
  if (args?.url) {
    lines.push(`URL: ${args.url}`);
  }
  if (args?.recipe_bucket) {
    lines.push(`Recipe Group: ${args.recipe_bucket}`);
  }
  if (args?.quantity) {
    lines.push(`Quantity: ${args.quantity}`);
  }
  if (args?.serving_size) {
    lines.push(`Serving: ${args.serving_size}`);
  }
  if (args?.remaining_quantity) {
    lines.push(`Remaining: ${args.remaining_quantity}`);
  }
  if (args?.quantity_value != null) {
    lines.push(`Quantity Value: ${args.quantity_value}`);
  }
  if (args?.quantity_unit) {
    lines.push(`Quantity Unit: ${args.quantity_unit}`);
  }
  if (args?.fill_percent != null) {
    lines.push(`Fill: ${args.fill_percent}%`);
  }
  if (args?.store) {
    lines.push(`Store: ${args.store}`);
  }
  if (args?.location) {
    lines.push(`Location: ${args.location}`);
  }
  if (args?.expiration_date) {
    lines.push(`Expires: ${args.expiration_date}`);
  }
  if (args?.is_opened != null) {
    lines.push(`Opened: ${args.is_opened ? "Yes" : "No"}`);
  }
  if (args?.reason) {
    lines.push(`Reason: ${args.reason}`);
  }
  if (args?.focus) {
    lines.push(`Focus: ${args.focus}`);
  }
  if (args?.metric_name) {
    lines.push(`Metric: ${args.metric_name}`);
  }
  if (args?.only_missing != null) {
    lines.push(`Only Missing: ${args.only_missing ? "Yes" : "No"}`);
  }
  if (Array.isArray(args?.ingredients) && args.ingredients.length > 0) {
    lines.push(`Ingredients: ${args.ingredients.join(", ")}`);
  }
  if (Array.isArray(args?.add_ingredients) && args.add_ingredients.length > 0) {
    lines.push(`Add: ${args.add_ingredients.join(", ")}`);
  }
  if (Array.isArray(args?.remove_ingredients) && args.remove_ingredients.length > 0) {
    lines.push(`Remove: ${args.remove_ingredients.join(", ")}`);
  }
  if (args?.calories != null) {
    lines.push(`Calories: ${args.calories}`);
  }
  if (args?.protein_g != null || args?.carbs_g != null || args?.fat_g != null) {
    lines.push(`Macros: P ${args?.protein_g ?? "-"} / C ${args?.carbs_g ?? "-"} / F ${args?.fat_g ?? "-"}`);
  }
  if (Array.isArray(args?.items) && args.items.length > 0) {
    lines.push(`Items: ${args.items.map((item) => item.item_name).filter(Boolean).join(", ")}`);
  }

  if (name === "get_shopping_list") {
    lines.push(
      summarizeItemList(
        toolResult.items,
        (item) => `${item.item_name}${item.store ? ` @ ${item.store}` : ""}`,
        "Shopping list is empty."
      )
    );
  }

  if (name === "clear_shopping_list") {
    lines.push(`Removed Count: ${toolResult?.count || 0}`);
    if (Array.isArray(toolResult?.items) && toolResult.items.length > 0) {
      lines.push(
        summarizeItemList(
          toolResult.items,
          (item) => `${item.item_name}${item.store ? ` @ ${item.store}` : ""}`,
          "Shopping list is empty."
        )
      );
    }
  }

  if (name === "clear_kitchen_inventory") {
    lines.push(`Removed Count: ${toolResult?.count || 0}`);
    if (Array.isArray(toolResult?.items) && toolResult.items.length > 0) {
      lines.push(
        summarizeItemList(
          toolResult.items,
          (item) => `${item.item_name}${item.storage_location ? ` @ ${item.storage_location}` : ""}`,
          "Kitchen is empty."
        )
      );
    }
  }

  if (name === "list_store_tabs") {
    lines.push(
      summarizeItemList(
        toolResult.stores,
        (store) => store,
        "No shopping store tabs were found."
      )
    );
  }

  if (name === "get_kitchen_overview") {
    lines.push(
      summarizeItemList(
        toolResult.items,
        (item) => `${item.item_name}${item.state_label ? ` - ${item.state_label}` : ""}`,
        "Kitchen looks empty."
      )
    );
  }

  if (name === "get_recent_discards") {
    lines.push(
      summarizeItemList(
        toolResult.items,
        (item) => item.discard_reason ? `${item.item_name}: ${item.discard_reason}` : item.item_name,
        "No recent discards were found."
      )
    );
  }

  if (name === "delete_recent_discard" && toolResult?.discard) {
    if (toolResult.discard.discard_reason) {
      lines.push(`Removed: ${toolResult.discard.item_name} (${toolResult.discard.discard_reason})`);
    } else {
      lines.push(`Removed: ${toolResult.discard.item_name}`);
    }
  }

  if (name === "clear_recent_discards") {
    lines.push(`Removed Count: ${toolResult?.count || 0}`);
    if (Array.isArray(toolResult?.items) && toolResult.items.length > 0) {
      lines.push(
        summarizeItemList(
          toolResult.items,
          (item) => item.discard_reason ? `${item.item_name}: ${item.discard_reason}` : item.item_name,
          "Discard history is empty."
        )
      );
    }
  }

  if (name === "search_kitchen_item" && toolResult?.item) {
    if (toolResult.item.brand) {
      lines.push(`Brand: ${toolResult.item.brand}`);
    }
    if (toolResult.item.variant) {
      lines.push(`Variant: ${toolResult.item.variant}`);
    }
    if (toolResult.item.category) {
      lines.push(`Category: ${toolResult.item.category}`);
    }
    if (toolResult.item.state_label) {
      lines.push(`State: ${toolResult.item.state_label}`);
    }
    if (toolResult.item.quantity_value != null) {
      lines.push(`Quantity Value: ${toolResult.item.quantity_value}`);
    }
    if (toolResult.item.quantity_unit) {
      lines.push(`Quantity Unit: ${toolResult.item.quantity_unit}`);
    }
    if (toolResult.item.fill_percent != null) {
      lines.push(`Fill: ${toolResult.item.fill_percent}%`);
    }
    if (toolResult.item.storage_location) {
      lines.push(`Location: ${toolResult.item.storage_location}`);
    }
    if (toolResult.item.expiration_date) {
      lines.push(`Expires: ${toolResult.item.expiration_date}`);
    }
    if (toolResult.item.estimated_price) {
      lines.push(`Estimated Price: ${toolResult.item.estimated_price}`);
    }
    if (toolResult.item.upf) {
      lines.push(`UPF: ${toolResult.item.upf}`);
    }
    if (Array.isArray(toolResult.item.harmful_ingredients) && toolResult.item.harmful_ingredients.length > 0) {
      lines.push(`Harmful Ingredients: ${toolResult.item.harmful_ingredients.slice(0, 6).join(", ")}`);
    }
    if (Array.isArray(toolResult.item.ingredients) && toolResult.item.ingredients.length > 0) {
      lines.push(`Ingredients: ${toolResult.item.ingredients.slice(0, 10).join(", ")}`);
    }
    if (toolResult.item.nutrition_summary) {
      lines.push(`Nutrition Summary: ${toolResult.item.nutrition_summary}`);
    }
    lines.push(`Opened: ${toolResult.item.is_opened ? "Yes" : "No"}`);
  }

  if (name === "get_recent_dishes") {
    lines.push(
      summarizeItemList(
        toolResult.items,
        (dish) => dish.calories != null ? `${dish.dish_name}: ${dish.calories} cal` : dish.dish_name,
        "No recent dishes were found."
      )
    );
  }

  if (name === "get_dish_detail" && toolResult?.dish) {
    if (toolResult.dish.serving_size) {
      lines.push(`Serving: ${toolResult.dish.serving_size}`);
    }
    if (toolResult.dish.calories != null) {
      lines.push(`Calories: ${toolResult.dish.calories}`);
    }
    if (Array.isArray(toolResult.dish.ingredients) && toolResult.dish.ingredients.length > 0) {
      lines.push(`Ingredients: ${toolResult.dish.ingredients.join(", ")}`);
    }
  }

  if (name === "get_health_metrics" && toolResult?.metrics) {
    lines.push(`Kitchen IQ: ${toolResult.metrics.IQ ?? "-"}`);
    lines.push(`Points: ${toolResult.metrics.Points ?? "-"}`);
    lines.push(`UPF: ${toolResult.metrics.UPF ?? "-"}`);
    lines.push(`Harmful Ingredients: ${toolResult.metrics.harmful_ingredients ?? "-"}`);
  }

  if (name === "get_recipe_suggestions") {
    const readyNow = Array.isArray(toolResult.kitchen_only) ? toolResult.kitchen_only.length : 0;
    const needGrocery = Array.isArray(toolResult.need_grocery) ? toolResult.need_grocery.length : 0;
    lines.push(`Status: ${toolResult.status || "unknown"}`);
    lines.push(`Ready Now: ${readyNow}`);
    lines.push(`Need Grocery: ${needGrocery}`);
  }

  if (name === "get_recipe_detail" && toolResult?.recipe) {
    if (toolResult.recipe.bucket) {
      lines.push(`Bucket: ${toolResult.recipe.bucket}`);
    }
    if (toolResult.recipe.source_url) {
      lines.push(`Source URL: ${toolResult.recipe.source_url}`);
    }
    if (Array.isArray(toolResult.recipe.ingredients) && toolResult.recipe.ingredients.length > 0) {
      lines.push(`Ingredients: ${toolResult.recipe.ingredients.join(", ")}`);
    }
    if (Array.isArray(toolResult.recipe.available_ingredients) && toolResult.recipe.available_ingredients.length > 0) {
      lines.push(`You Have: ${toolResult.recipe.available_ingredients.join(", ")}`);
    }
    if (Array.isArray(toolResult.recipe.missing_ingredients) && toolResult.recipe.missing_ingredients.length > 0) {
      lines.push(`Missing: ${toolResult.recipe.missing_ingredients.join(", ")}`);
    }
    if (Array.isArray(toolResult.recipe.notes) && toolResult.recipe.notes.length > 0) {
      lines.push(`Notes: ${toolResult.recipe.notes.join(", ")}`);
    }
  }

  if (name === "search_web_recipes" && Array.isArray(toolResult?.recipes)) {
    if (args?.query) {
      lines.push(`Query: ${args.query}`);
    }
    for (const recipe of toolResult.recipes.slice(0, 5)) {
      const parts = [recipe.title];
      if (recipe.source_name) parts.push(`from ${recipe.source_name}`);
      if (recipe.missing_ingredients?.length > 0) parts.push(`need: ${recipe.missing_ingredients.slice(0, 3).join(", ")}${recipe.missing_ingredients.length > 3 ? "..." : ""}`);
      lines.push(`- ${parts.join(" — ")}`);
    }
  }

  if (name === "get_saved_recipes") {
    lines.push(
      summarizeItemList(
        toolResult.recipes,
        (recipe) => recipe.source_url ? `${recipe.title} - ${recipe.source_url}` : recipe.title,
        "No saved recipes were found."
      )
    );
  }

  if ((name === "get_saved_recipe_detail" || name === "save_recipe_from_tiktok" || name === "save_generated_recipe" || name === "remove_saved_recipe" || name === "add_saved_recipe_ingredients_to_shopping_list") && toolResult?.recipe) {
    if (toolResult.recipe.source_url) {
      lines.push(`Source URL: ${toolResult.recipe.source_url}`);
    }
    if (Array.isArray(toolResult.recipe.ingredients) && toolResult.recipe.ingredients.length > 0) {
      lines.push(`Ingredients: ${toolResult.recipe.ingredients.join(", ")}`);
    }
    if (Array.isArray(toolResult.recipe.notes) && toolResult.recipe.notes.length > 0) {
      lines.push(`Notes: ${toolResult.recipe.notes.join(", ")}`);
    }
  }

  if (name === "get_meal_plan") {
    lines.push(`Status: ${toolResult.status || "unknown"}`);
    if (toolResult.focus) {
      lines.push(`Focus: ${toolResult.focus}`);
    }
    if (Array.isArray(toolResult.plan) && toolResult.plan.length > 0) {
      lines.push(`Meals: ${toolResult.plan.length}`);
    }
  }

  if (name === "recommend_items_to_buy" || name === "recommend_items_to_use_up") {
    lines.push(
      summarizeItemList(
        toolResult.items,
        (item) => `${item.item_name}: ${item.reason}`,
        "No recommendations were found."
      )
    );
  }

  if (toolResult?.async_update?.pending) {
    lines.push("Async Update: pending");
  }

  if (lines.length === 0) {
    lines.push(`Action type: ${toTitleCase(name)}`);
  }

  return lines.join("\n");
}

function summarizeToolResultForLog(toolResult) {
  if (!toolResult || typeof toolResult !== "object") {
    return null;
  }

  return {
    count: Number.isFinite(Number(toolResult.count)) ? Number(toolResult.count) : undefined,
    itemName: toolResult.item?.item_name || toolResult.dish?.dish_name || toolResult.item_name || undefined,
    itemId: toolResult.item?.id || toolResult.dish?.id || toolResult.id || undefined,
    shoppingId: toolResult.item?.shopping_id || toolResult.shopping_id || undefined,
    store: toolResult.item?.store || toolResult.store || undefined,
    location: toolResult.item?.storage_location || undefined,
    found: typeof toolResult.found === "boolean" ? toolResult.found : undefined,
    stores: Array.isArray(toolResult.stores) ? toolResult.stores.length : undefined,
    dishIngredients: Array.isArray(toolResult.dish?.ingredients) ? toolResult.dish.ingredients.length : undefined,
    recipeTitle: toolResult.recipe?.title || undefined,
    recentContext: toolResult.recent_context?.resolved_by || undefined
    ,
    status: toolResult.status || undefined,
    focus: toolResult.focus || undefined
  };
}

async function invokeRealAction(toolName, args, userContext, options, actions) {
  // Thread the response surface (halo/app) into the household context so the
  // data-access layer can apply surface-specific resolution (e.g. Halo, a one-way
  // device, acts on ambiguous kitchen items instead of asking an unanswerable question).
  if (options?.responseSurface && userContext && !userContext.responseSurface) {
    userContext = { ...userContext, responseSurface: options.responseSurface };
  }
  switch (toolName) {
    case "add_to_shopping_list": {
      const item = await actions.addShoppingItem(userContext, args.item_name, args.store || null, args.quantity || null, options);
      return { item };
    }
    case "add_many_to_shopping_list":
      return actions.addManyShoppingItems(userContext, args.items || [], options);
    case "update_shopping_item_store": {
      const item = await actions.updateShoppingItemStore(userContext, args.item_name, args.store, options);
      return { item };
    }
    case "update_many_shopping_item_stores":
      return actions.updateManyShoppingItemStores(userContext, args.items || [], options);
    case "mark_shopping_item_bought": {
      const item = await actions.markShoppingItemBought(userContext, args.item_name, options);
      return { item };
    }
    case "mark_shopping_item_unbought": {
      const item = await actions.markShoppingItemUnbought(userContext, args.item_name, options);
      return { item };
    }
    case "remove_from_shopping_list": {
      const item = await actions.removeShoppingItem(userContext, args.item_name, options);
      return { item };
    }
    case "clear_shopping_list":
      return actions.clearShoppingList(userContext, options);
    case "get_shopping_list": {
      const items = await actions.getShoppingItems(userContext, { ...options, limit: args.limit });
      return { items, count: items.length };
    }
    case "list_store_tabs":
      return actions.listStoreTabs(userContext, options);
    case "get_kitchen_overview":
      return actions.getKitchenOverview(userContext, { ...options, limit: args.limit });
    case "search_kitchen_item":
      return actions.searchKitchenItem(userContext, args, options);
    case "check_in_item":
      return actions.checkInKitchenItem(userContext, args, options);
    case "check_in_many_items":
      return actions.checkInManyKitchenItems(userContext, args.items || [], options);
    case "update_item_quantity": {
      const item = await actions.updateKitchenItemQuantity(userContext, args, args, options);
      return { item };
    }
    case "mark_item_opened": {
      const item = await actions.markKitchenItemOpened(userContext, args, options);
      return { item };
    }
    case "update_item_expiration": {
      const item = await actions.updateKitchenItemExpiration(userContext, args, args.expiration_date, options);
      return { item };
    }
    case "update_item_location": {
      const item = await actions.updateKitchenItemLocation(userContext, args, args.location, options);
      return { item };
    }
    case "update_item_details": {
      const item = await actions.updateKitchenItemDetails(userContext, args, args, options);
      return { item };
    }
    case "discard_item":
      return actions.discardKitchenItem(userContext, args, args.reason || null, options);
    case "clear_kitchen_inventory":
      return actions.clearKitchenInventory(userContext, options);
    case "get_recent_discards": {
      const items = await actions.getRecentDiscards(userContext, { ...options, limit: args.limit });
      return { items, count: items.length };
    }
    case "delete_recent_discard":
      return actions.deleteRecentDiscard(userContext, args.item_name, options);
    case "clear_recent_discards":
      return actions.clearRecentDiscards(userContext, options);
    case "log_dish_from_voice":
      return actions.logDishFromVoice(userContext, args, options);
    case "log_dish_ingredients":
      return actions.logDishIngredients(userContext, args, options);
    case "append_to_recent_dish":
      return actions.appendToRecentDish(userContext, args, options);
    case "update_recent_dish":
      return actions.updateRecentDish(userContext, args, options);
    case "get_recent_dishes":
      return actions.getRecentDishes(userContext, { ...options, limit: args.limit });
    case "get_dish_detail":
      return actions.getDishDetail(userContext, args, options);
    case "mark_dish_consumed":
      return actions.markDishConsumed(userContext, args, options);
    case "delete_dish_log":
      return actions.deleteDishLog(userContext, args, options);
    case "add_dish_ingredients_to_shopping_list":
      return actions.addDishIngredientsToShoppingList(userContext, args, options);
    case "get_health_metrics":
      return actions.getHealthMetrics(userContext, args, options);
    case "get_recipe_suggestions":
      return actions.getRecipeSuggestions(userContext, args, options);
    case "get_recipe_detail":
      return actions.getRecipeDetail(userContext, args, options);
    case "add_recipe_ingredients_to_shopping_list":
      return actions.addRecipeIngredientsToShoppingList(userContext, args, options);
    case "add_saved_recipe_ingredients_to_shopping_list":
      return actions.addSavedRecipeIngredientsToShoppingList(userContext, args, options);
    case "search_web_recipes":
      return actions.searchWebRecipes(userContext, args, options);
    case "get_saved_recipes":
      return actions.getSavedRecipes(userContext, args, options);
    case "get_saved_recipe_detail":
      return actions.getSavedRecipeDetail(userContext, args, options);
    case "save_recipe_from_tiktok":
      return actions.saveRecipeFromTikTok(userContext, args, options);
    case "save_generated_recipe":
      return actions.saveGeneratedRecipe(userContext, args, options);
    case "remove_saved_recipe":
      return actions.removeSavedRecipe(userContext, args, options);
    case "get_meal_plan":
      return actions.getMealPlan(userContext, args, options);
    case "refresh_meal_plan":
      return actions.refreshMealPlan(userContext, args, options);
    case "recommend_items_to_buy": {
      const items = await actions.recommendItemsToBuy(userContext, { ...options, limit: args.limit });
      return { items, count: items.length };
    }
    case "recommend_items_to_use_up": {
      const items = await actions.recommendItemsToUseUp(userContext, { ...options, limit: args.limit });
      return { items, count: items.length };
    }
    default:
      throw new Error(`Unsupported tool: ${toolName}`);
  }
}

export function createToolActionExecutor(actions) {
  const requiredActions = [
    "addShoppingItem",
    "addManyShoppingItems",
    "updateShoppingItemStore",
    "updateManyShoppingItemStores",
    "markShoppingItemBought",
    "markShoppingItemUnbought",
    "removeShoppingItem",
    "clearShoppingList",
    "getShoppingItems",
    "listStoreTabs",
    "getKitchenOverview",
    "searchKitchenItem",
    "checkInKitchenItem",
    "checkInManyKitchenItems",
    "updateKitchenItemQuantity",
    "markKitchenItemOpened",
    "updateKitchenItemExpiration",
    "updateKitchenItemLocation",
    "updateKitchenItemDetails",
    "discardKitchenItem",
    "clearKitchenInventory",
    "getRecentDiscards",
    "deleteRecentDiscard",
    "clearRecentDiscards",
    "logDishFromVoice",
    "logDishIngredients",
    "appendToRecentDish",
    "updateRecentDish",
    "getRecentDishes",
    "getDishDetail",
    "markDishConsumed",
    "deleteDishLog",
    "addDishIngredientsToShoppingList",
    "getHealthMetrics",
    "getRecipeSuggestions",
    "getRecipeDetail",
    "addRecipeIngredientsToShoppingList",
    "addSavedRecipeIngredientsToShoppingList",
    "searchWebRecipes",
    "getSavedRecipes",
    "getSavedRecipeDetail",
    "saveRecipeFromTikTok",
    "saveGeneratedRecipe",
    "removeSavedRecipe",
    "getMealPlan",
    "refreshMealPlan",
    "recommendItemsToBuy",
    "recommendItemsToUseUp"
  ];

  for (const actionName of requiredActions) {
    if (typeof actions?.[actionName] !== "function") {
      throw new Error(`Missing tool action dependency: ${actionName}`);
    }
  }

  return async function executeToolAction({ toolName, args, env, userContext, responseSurface }) {
    const validation = validateToolCall(toolName, args);
    if (!validation.ok) {
      return {
        ok: false,
        statusCode: 400,
        error: validation.error
      };
    }

    if (!userContext?.ownerId) {
      return {
        ok: false,
        statusCode: 400,
        error: "Connect a household before asking the assistant to use data tools."
      };
    }

    const normalizedArgs = validation.args;
    const actionMode = env?.ACTION_MODE === "real" ? "real" : "mock";

    console.log("[DEBUG] tool action start:", JSON.stringify({
      toolName,
      actionMode,
      args: normalizedArgs,
      context: {
        ownerId: userContext.ownerId,
        userId: userContext.userId,
        tableOwnerId: userContext.tableOwnerId,
        shoppingNamespace: userContext.shoppingNamespace,
        householdSize: userContext.householdSize
      }
    }));

    try {
      const toolResult = actionMode === "real"
        ? await invokeRealAction(toolName, normalizedArgs, userContext, { env, responseSurface }, actions)
        : buildMockToolResult(toolName, normalizedArgs);
      const actionSummary = formatActionSummary(toolName, normalizedArgs, toolResult);
      const actionDetails = formatActionDetails(toolName, normalizedArgs, toolResult);
      const response = {
        ok: true,
        statusCode: 200,
        toolName,
        args: normalizedArgs,
        actionMode,
        actionSummary,
        actionDetails,
        toolResult
      };

      console.log("[DEBUG] tool action result:", JSON.stringify({
        toolName,
        ok: true,
        actionMode,
        statusCode: response.statusCode,
        actionSummary,
        summary: summarizeToolResultForLog(toolResult)
      }));

      return response;
    } catch (error) {
      const response = {
        ok: false,
        statusCode: error.statusCode || 500,
        error: error.message || "Unexpected tool action error.",
        details: error.details || null,
        toolName,
        args: normalizedArgs
      };

      console.error("[ERROR] tool action failed:", JSON.stringify({
        toolName,
        actionMode,
        statusCode: response.statusCode,
        error: response.error,
        details: response.details
      }));

      return response;
    }
  };
}
