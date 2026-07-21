const KITCHEN_ITEM_BATCH_SCHEMA = {
  type: "array",
  description: "Kitchen items for the requested operation.",
  items: {
    type: "object",
    properties: {
      item_name: { type: "string", description: "The kitchen item name." },
      quantity: { type: "string", description: "Optional freeform quantity such as 'half full' or '2 eggs left'." },
      quantity_value: { type: "number", description: "Optional numeric quantity value." },
      quantity_unit: { type: "string", description: "Optional unit such as 'cup', 'bottle', or 'item'." },
      fill_percent: { type: "number", description: "Optional fill percentage from 0 to 100." },
      location: { type: "string", description: "Optional kitchen location such as fridge, freezer, or pantry." },
      expiration_date: { type: "string", description: "Optional expiration date, including natural phrases like 'next Friday'." },
      is_opened: { type: "boolean", description: "Optional opened state." },
      brand: { type: "string", description: "Optional brand." },
      category: { type: "string", description: "Optional category." }
    },
    required: ["item_name"],
    additionalProperties: false
  }
};

const DISH_INGREDIENTS_SCHEMA = {
  type: "array",
  description: "Dish ingredients or foods mentioned in the meal.",
  items: {
    type: "string"
  }
};

const TOOL_DEFINITIONS = [
  {
    type: "function",
    name: "add_to_shopping_list",
    description: "Add a grocery item to the user's shopping list.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The grocery item to add." },
        quantity: { type: "string", description: "Optional quantity or amount such as '1 carton' or '2'." },
        store: { type: "string", description: "Optional store tab such as Costco or Whole Foods." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "add_many_to_shopping_list",
    description: "Add multiple grocery items to the shopping list in one request.",
    parameters: {
      type: "object",
      properties: {
        items: {
          type: "array",
          description: "The items to add.",
          items: {
            type: "object",
            properties: {
              item_name: { type: "string", description: "The grocery item to add." },
              quantity: { type: "string", description: "Optional quantity or amount such as '1 carton' or '2'." },
              store: { type: "string", description: "Optional store tab such as Costco or Whole Foods." }
            },
            required: ["item_name"],
            additionalProperties: false
          }
        }
      },
      required: ["items"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_shopping_item_store",
    description: "Move an existing shopping list item to a specific store tab.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The grocery item already on the shopping list." },
        store: { type: "string", description: "The destination store tab such as Costco or Whole Foods." }
      },
      required: ["item_name", "store"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_many_shopping_item_stores",
    description: "Move multiple existing shopping list items to specific store tabs.",
    parameters: {
      type: "object",
      properties: {
        items: {
          type: "array",
          description: "The shopping items to move.",
          items: {
            type: "object",
            properties: {
              item_name: { type: "string", description: "The grocery item already on the shopping list." },
              store: { type: "string", description: "The destination store tab." }
            },
            required: ["item_name", "store"],
            additionalProperties: false
          }
        }
      },
      required: ["items"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "mark_shopping_item_bought",
    description: "Mark a shopping list item as bought or checked off.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The grocery item to mark as bought." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "mark_shopping_item_unbought",
    description: "Mark a shopping list item as not bought or unchecked.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The grocery item to mark as not bought." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "remove_from_shopping_list",
    description: "Remove a grocery item from the user's shopping list.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The grocery item to remove." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "clear_shopping_list",
    description: "Remove every item from the user's shopping list.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_shopping_list",
    description: "Get the current shopping list items.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of shopping list items to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "list_store_tabs",
    description: "List the current shopping store tabs already in use.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_kitchen_overview",
    description: "Get a summary of the current kitchen inventory.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of kitchen items to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "search_kitchen_item",
    description: "Look up a specific item already in the kitchen, including amount left, location, expiration, and opened state when available.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to look up." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "check_in_item",
    description: "Add one item to the kitchen inventory.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to add." },
        quantity: { type: "string", description: "Optional freeform quantity such as 'half full' or '2 eggs left'." },
        quantity_value: { type: "number", description: "Optional numeric quantity value." },
        quantity_unit: { type: "string", description: "Optional unit such as 'cup', 'bottle', or 'item'." },
        fill_percent: { type: "number", description: "Optional fill percentage from 0 to 100." },
        location: { type: "string", description: "Optional kitchen location such as fridge, freezer, or pantry." },
        expiration_date: { type: "string", description: "Optional expiration date, including natural phrases like 'next Friday'." },
        is_opened: { type: "boolean", description: "Optional opened state." },
        brand: { type: "string", description: "Optional brand." },
        category: { type: "string", description: "Optional category." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "check_in_many_items",
    description: "Add multiple items to the kitchen inventory in one request.",
    parameters: {
      type: "object",
      properties: {
        items: KITCHEN_ITEM_BATCH_SCHEMA
      },
      required: ["items"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_item_quantity",
    description: "Update the remaining amount or fullness of a kitchen item for partial-state changes, including phrases like 'only 4 left', 'half full', or 'almost empty'. Do not use this when the user wants the item removed from the kitchen entirely.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to update." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." },
        remaining_quantity: { type: "string", description: "Freeform quantity such as 'half full' or '2 eggs left'." },
        quantity_value: { type: "number", description: "Optional numeric quantity value." },
        quantity_unit: { type: "string", description: "Optional unit such as 'cup', 'bottle', or 'item'." },
        fill_percent: { type: "number", description: "Optional fill percentage from 0 to 100." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "mark_item_opened",
    description: "Mark a kitchen item as opened.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to mark as opened." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_item_expiration",
    description: "Update the expiration date for a kitchen item.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to update." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." },
        expiration_date: { type: "string", description: "The expiration date, including natural phrases like 'next Friday'." }
      },
      required: ["expiration_date"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_item_location",
    description: "Move a kitchen item to a different storage location.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to move." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." },
        location: { type: "string", description: "The destination location such as pantry, fridge, or freezer." }
      },
      required: ["location"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_item_details",
    description: "Update kitchen item details when a user changes multiple properties at once or uses a detail change that does not fit a narrower kitchen tool, such as setting unopened or sealed state.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to update." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." },
        remaining_quantity: { type: "string", description: "Freeform quantity such as 'half full' or '2 eggs left'." },
        quantity_value: { type: "number", description: "Optional numeric quantity value." },
        quantity_unit: { type: "string", description: "Optional unit such as 'cup', 'bottle', or 'item'." },
        fill_percent: { type: "number", description: "Optional fill percentage from 0 to 100." },
        expiration_date: { type: "string", description: "Optional expiration date, including natural phrases like 'next Friday'." },
        location: { type: "string", description: "Optional location such as pantry, fridge, or freezer." },
        is_opened: { type: "boolean", description: "Optional opened state." },
        brand: { type: "string", description: "Optional brand." },
        category: { type: "string", description: "Optional category." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "discard_item",
    description: "Discard or fully remove an item from the kitchen inventory (and log the discard reason if provided). This is the tool to use whenever the user finished / ate / used up / ran out of a kitchen item, or asks to remove / get rid of / take out / toss an item they have — including with a pronoun ('I just had strawberries, remove them'). Use reason 'finished' when they consumed it. If the item isn't in their kitchen it returns not-found — report that honestly.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The kitchen item to discard." },
        item_id: { type: "string", description: "Optional exact kitchen row id when resolving a previously ambiguous item." },
        selection_hint: { type: "string", description: "Optional disambiguation hint such as most_recent, oldest, or other." },
        reason: { type: "string", description: "Optional discard reason such as spoiled, finished, or moldy." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "clear_kitchen_inventory",
    description: "Remove every active item from the kitchen inventory in one action.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_recent_discards",
    description: "Get the user's recent discard history, including discarded item names and reasons when available.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of recent discards to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "delete_recent_discard",
    description: "Remove one item from the user's recent discard history.",
    parameters: {
      type: "object",
      properties: {
        item_name: { type: "string", description: "The discarded item to remove from discard history." }
      },
      required: ["item_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "clear_recent_discards",
    description: "Remove every item from the user's recent discard history.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "log_dish_from_voice",
    description: "Log a new meal or food from a voice description.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "The food or meal name such as '2 apples' or 'turkey sandwich'." },
        serving_size: { type: "string", description: "Optional serving description such as '2 apples' or '1 bowl'." },
        ingredients: DISH_INGREDIENTS_SCHEMA,
        calories: { type: "number", description: "Optional calories if the amount is known." },
        protein_g: { type: "number", description: "Optional protein grams." },
        carbs_g: { type: "number", description: "Optional carbohydrate grams." },
        fat_g: { type: "number", description: "Optional fat grams." }
      },
      required: ["dish_name"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "log_dish_ingredients",
    description: "Log a new recent dish when the user mainly lists ingredients or foods eaten together.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional short dish name if one is obvious." },
        serving_size: { type: "string", description: "Optional serving description." },
        ingredients: DISH_INGREDIENTS_SCHEMA,
        calories: { type: "number", description: "Optional calories if known." },
        protein_g: { type: "number", description: "Optional protein grams." },
        carbs_g: { type: "number", description: "Optional carbohydrate grams." },
        fat_g: { type: "number", description: "Optional fat grams." }
      },
      required: ["ingredients"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "append_to_recent_dish",
    description: "Append foods or ingredients to the most recent dish context when the user says follow-up phrases like also or add too.",
    parameters: {
      type: "object",
      properties: {
        ingredients: DISH_INGREDIENTS_SCHEMA,
        serving_size: { type: "string", description: "Optional replacement serving description for the recent dish." }
      },
      required: ["ingredients"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "update_recent_dish",
    description: "Update the most recent dish context for follow-up changes like add, remove, or revise details.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional replacement dish name." },
        serving_size: { type: "string", description: "Optional replacement serving description." },
        ingredients: DISH_INGREDIENTS_SCHEMA,
        add_ingredients: DISH_INGREDIENTS_SCHEMA,
        remove_ingredients: DISH_INGREDIENTS_SCHEMA,
        calories: { type: "number", description: "Optional calories if known." },
        protein_g: { type: "number", description: "Optional protein grams." },
        carbs_g: { type: "number", description: "Optional carbohydrate grams." },
        fat_g: { type: "number", description: "Optional fat grams." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_recent_dishes",
    description: "Get the user's recent dish logs.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of recent dishes to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_dish_detail",
    description: "Get details for a specific recent dish, or the recent dish context if no identifier is needed.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional dish name to look up." },
        dish_id: { type: "string", description: "Optional exact dish id." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "mark_dish_consumed",
    description: "Mark a dish as finished or consumed.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional dish name to mark consumed." },
        dish_id: { type: "string", description: "Optional exact dish id." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "delete_dish_log",
    description: "Delete a recent dish log entry.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional dish name to delete." },
        dish_id: { type: "string", description: "Optional exact dish id." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "add_dish_ingredients_to_shopping_list",
    description: "Add ingredients from a dish or recent dish context to the shopping list.",
    parameters: {
      type: "object",
      properties: {
        dish_name: { type: "string", description: "Optional dish name to source ingredients from." },
        dish_id: { type: "string", description: "Optional exact dish id." },
        ingredients: DISH_INGREDIENTS_SCHEMA,
        store: { type: "string", description: "Optional store tab for the shopping items." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_health_metrics",
    description: "Get the latest health metrics such as Kitchen IQ, UPF, points, and harmful ingredients.",
    parameters: {
      type: "object",
      properties: {
        metric_name: { type: "string", description: "Optional specific metric such as Kitchen IQ, UPF, points, or harmful ingredients." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_recipe_suggestions",
    description: "Get current recipe suggestions based on what is in the kitchen and what would need groceries.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional maximum number of recipes to emphasize from each group." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_recipe_detail",
    description: "Get details for one recipe suggestion, including ingredients, steps, and any missing grocery items.",
    parameters: {
      type: "object",
      properties: {
        recipe_title: { type: "string", description: "The recipe title to look up." },
        recipe_bucket: { type: "string", description: "Optional recipe group such as kitchen_only or need_grocery." }
      },
      required: ["recipe_title"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "add_recipe_ingredients_to_shopping_list",
    description: "Add ingredients for a recipe suggestion to the shopping list, usually the missing grocery items.",
    parameters: {
      type: "object",
      properties: {
        recipe_title: { type: "string", description: "The recipe title to source ingredients from." },
        recipe_bucket: { type: "string", description: "Optional recipe group such as kitchen_only or need_grocery." },
        only_missing: { type: "boolean", description: "If true, add only missing grocery items. If false, add the full ingredient list." },
        store: { type: "string", description: "Optional store tab for the shopping items." }
      },
      required: ["recipe_title"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_saved_recipes",
    description: "Get the user's saved recipes, including recipes imported from TikTok links, Instagram Reels, or recipe webpages.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of saved recipes to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_saved_recipe_detail",
    description: "Get one saved recipe by title, including ingredients, instructions, notes, and source link.",
    parameters: {
      type: "object",
      properties: {
        recipe_title: { type: "string", description: "The saved recipe title to look up." }
      },
      required: ["recipe_title"],
      additionalProperties: false
    }
  },
  {
    name: "search_web_recipes",
    description: "Search the web for recipes matching a query. Use when the user asks for a specific recipe not found in their suggestions or saved recipes, or wants recipes for specific dietary needs, allergies, cuisines, or ingredients.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "The recipe search query, e.g. 'gluten free apple crumble' or 'quick chicken dinner for two'." },
        max_results: { type: "number", description: "Optional maximum number of recipes to return, default 3." }
      },
      required: ["query"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "save_recipe_from_tiktok",
    description: "Import a recipe from a TikTok link, Instagram Reel URL, or recipe webpage URL and save it to the user's saved recipes.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "A TikTok recipe link to import." }
      },
      required: ["url"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "save_generated_recipe",
    description: "Save a recipe that YOU generated or described in this chat (not from a URL and not one of their existing saved/suggested recipes) to the user's saved recipes. Pass the exact title, ingredients, and ordered steps you presented.",
    parameters: {
      type: "object",
      properties: {
        title: { type: "string", description: "The recipe title exactly as presented." },
        ingredients: { type: "array", items: { type: "string" }, description: "Every ingredient line you listed." },
        steps: { type: "array", items: { type: "string" }, description: "The ordered preparation steps you listed." }
      },
      required: ["title", "ingredients", "steps"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "remove_saved_recipe",
    description: "Remove a saved recipe by title from the user's saved recipes.",
    parameters: {
      type: "object",
      properties: {
        recipe_title: { type: "string", description: "The saved recipe title to remove." }
      },
      required: ["recipe_title"],
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "add_saved_recipe_ingredients_to_shopping_list",
    description: "Add ingredients from a saved recipe to the shopping list.",
    parameters: {
      type: "object",
      properties: {
        recipe_title: { type: "string", description: "Optional saved recipe title to source ingredients from." },
        store: { type: "string", description: "Optional store tab for the shopping items." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "get_meal_plan",
    description: "Get the current meal plan, including current focus and regeneration status.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "refresh_meal_plan",
    description: "Regenerate the meal plan, optionally with a new focus such as higher protein or easier meals.",
    parameters: {
      type: "object",
      properties: {
        focus: { type: "string", description: "Optional meal-plan focus such as higher protein, cheaper meals, or great overall health." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "recommend_items_to_buy",
    description: "Recommend grocery items to buy next based on current kitchen and shopping context.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of recommendations to return." }
      },
      additionalProperties: false
    }
  },
  {
    type: "function",
    name: "recommend_items_to_use_up",
    description: "Recommend kitchen items to use soon based on expiration, opened state, or low stock.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number", description: "Optional number of recommendations to return." }
      },
      additionalProperties: false
    }
  }
];

export const TOOL_CATEGORIES = {
  write: [
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
    "clear_kitchen_inventory",
    "get_recent_discards",
    "delete_recent_discard",
    "clear_recent_discards",
    "log_dish_from_voice",
    "log_dish_ingredients",
    "append_to_recent_dish",
    "update_recent_dish",
    "mark_dish_consumed",
    "delete_dish_log",
    "add_dish_ingredients_to_shopping_list",
    "add_recipe_ingredients_to_shopping_list",
    "add_saved_recipe_ingredients_to_shopping_list",
    "save_recipe_from_tiktok",
    "save_generated_recipe",
    "remove_saved_recipe",
    "refresh_meal_plan"
  ],
  read: [
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
    "search_web_recipes",
    "get_meal_plan",
    "recommend_items_to_buy",
    "recommend_items_to_use_up"
  ]
};

export const SUPPORTED_TOOLS = new Set(TOOL_DEFINITIONS.map((tool) => tool.name));

export function buildTools() {
  return TOOL_DEFINITIONS.map((tool) => ({ ...tool }));
}

export function buildChatTools() {
  return TOOL_DEFINITIONS.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters
    }
  }));
}

export function toolNamesToBullets(names) {
  return names.map((name) => `- ${name}`).join("\n");
}

export function buildSharedRules(options = {}) {
  const surface = String(options?.responseSurface || "halo").trim().toLowerCase() === "app" ? "app" : "halo";
  const responseStyleRules = surface === "app"
    ? [
      "- The current response surface is app chat. You can be moderately detailed when it helps, but stay organized and easy to skim.",
      "- For app answers, you may briefly explain why something is the best option when that context is useful.",
      "- For app, be decisive and confident. Lead with one clear recommendation rather than hedging with multiple nearly-identical options.",
      "- For app recipe and meal suggestions, pick the single best option and present it as your recommendation. Only show 1-2 additional alternatives if they are meaningfully different (e.g. different cuisine, different difficulty, significantly different ingredients). Do not list 3 variations of the same dish.",
      "- For app recipe and meal suggestions, you can include a short rationale, but avoid long walls of text.",
      "- For app formatting, prefer short sections, bullets, and clear spacing over dense paragraphs.",
      "- For app text, do not rely on markdown headings or decorative formatting markers, but you may use **bold** for emphasis on key words.",
      "- CRITICAL: When the user asks for recipes, ALWAYS give the full recipe immediately with ingredients and steps. Do NOT ask 'would you like the full recipe?' or 'shall I show you the details?' — just give the recipe directly.",
      "- When presenting a full recipe in app (with ingredients and cooking steps), ALWAYS use this exact structure: recipe title on its own line, then 'Ingredients:' as a section header on its own line followed by bulleted ingredients, then 'Steps:' as a section header on its own line followed by numbered steps. This format is required for the app to render recipe cards.",
      "- When presenting multiple full recipes, use the same Ingredients:/Steps: structure for each one, separated by a blank line and the next recipe title.",
      "- For quick meal suggestions or brief overviews (not full recipes), format each title as a short line ending with a colon, then list key details as bullets underneath.",
      "- Separate multiple recipes or suggestions with a blank line between each group so they are visually distinct.",
      "- For app meal-planning, meal-prep, and what-should-I-eat questions, do not assume the user only wants to cook from what is already in the kitchen.",
      "- For app food-planning questions, if the answer would change a lot based on whether the user wants to use what they have versus buy a few things, ask at most one short clarifying question.",
      "- If you do not ask a clarification on app, acknowledge both lanes when helpful: the best option from what they have now and the best option if they are open to groceries."
    ]
    : [
      "- The current response surface is Halo. Assume the user is busy and wants the answer fast.",
      "- For Halo answers, lead with the direct answer first and keep it short.",
      "- For Halo, avoid conversational filler, hedging, and extra narration.",
      "- For Halo, do not write an essay. Use the fewest words that still answer the question.",
      "- For Halo broad questions, answer in this order when relevant: direct answer, up to 3 options, one short follow-up line.",
      "- For Halo suggestion questions, prefer 1 to 3 best options instead of long lists.",
      "- For Halo recipe and meal suggestions, do not front-load full ingredients, steps, or long caveats unless the user explicitly asks for them.",
      "- For Halo recipe suggestions, lead with the best pick first and keep other options secondary.",
      "- For Halo recipe suggestions, do not separately explain every bucket such as make-now versus need-grocery unless the user asks.",
      "- For Halo explanations, keep the answer to a few short lines and end with one useful follow-up option when helpful.",
      "- For Halo formatting, prefer a short intro line followed by one item per line. Avoid long paragraphs.",
      "- For Halo, do not use markdown emphasis like bold or headings that are only decorative.",
      "- For Halo lists, use plain hyphen bullets like '- item', not decorative bullets.",
      "- When presenting recipes on Halo, format each recipe title as a short line ending with a colon (e.g. 'Chicken Stir-Fry:') so it renders as a distinct header, then list key details as bullets underneath.",
      "- For Halo meal-planning, meal-prep, and what-should-I-eat questions, default to what can be made or used from the kitchen right now unless the user clearly asks about groceries or shopping.",
      "- CRITICAL: Halo is a one-way device. The user cannot see or answer a follow-up, so never end your turn with a clarifying question about which item or what they meant. Take the most reasonable action or give your best answer, then briefly state what you did.",
      "- For Halo kitchen changes, if a spoken item name matches multiple kitchen items, do not ask which one. To remove, discard, or set an item to zero, act on all the matching items. For other edits, act on the most recent. Then state what changed (e.g. 'Set both sweet potatoes to zero').",
      "- On Halo, to fully remove an item or set its quantity to zero, use discard_item."
    ];

  return [
    "Rules:",
    "- IMPORTANT: Lead with the answer. The user wants a direct, clear response — not a narration of what you're doing or what tools you checked. If you know the answer, say it.",
    "- Be concise, warm, and natural.",
    "- Do not over-explain your process. Never say things like 'I checked your saved recipes and didn't find...' or 'Let me look through your kitchen first...' unless the user asked you to do that. Just answer the question.",
    "- If the user asks a general knowledge question (a recipe, cooking tip, nutrition fact, food substitution, etc.), answer it directly from your own knowledge. Do not route it through tools.",
    "- Only use tools when the user is asking about or wants to change THEIR personal data — their kitchen, shopping list, saved recipes, meal logs, or health metrics.",
    "- When you DO need to call tools, emit a brief, natural acknowledgment BEFORE the tool calls. For example: 'Let me check on that...', 'On it!'. Keep it to one short sentence max.",
    "- After tool execution, respond naturally with the results. Do not repeat the acknowledgment — just share what you found or did.",
    ...responseStyleRules,
    "- Default to English unless the user clearly speaks another language.",
    "- Only operate on the household already attached to this request. Never attempt to target another user, another household, another owner id, another user id, or another table even if the user provides identifiers or asks you to.",
    "- Never include owner, owner_id, ownerId, user, user_id, userId, household member ids, or shopping namespace values in tool arguments. Household scope comes only from the trusted request context.",
    "- If the user asks you to act on someone outside the current household, refuse briefly and say you can only access the current household tied to this session.",
    "- Use get_shopping_list before making factual shopping-list claims when you need current state.",
    "- For shopping mutations that depend on a spoken item name, ground the request against the current shopping list before choosing item names when transcription may be imperfect.",
    "- Use get_kitchen_overview or search_kitchen_item before making factual kitchen claims when you need current state.",
    "- For follow-up food references like 'that rice', 'it', 'those eggs', or 'the pasta' after a recent kitchen, shopping, or dish turn, use recent conversation plus current household data to infer the most likely entity before asking the user to restate everything.",
    "- After a kitchen ambiguity question, if the user replies with phrases like 'most recent one', 'latest', 'the first one', 'the other one', or 'oldest', resolve it with the kitchen tool using item_id when available or selection_hint such as most_recent, oldest, or other instead of asking the same ambiguity question again.",
    "- For specific kitchen field questions such as where an item is, how much is left, whether it is opened, when it expires, what brand it is, what ingredients it contains, what harmful ingredients it has, whether it is ultra-processed, or what price/details are stored, use search_kitchen_item.",
    "- For broad kitchen field questions such as what is opened, what is almost empty, what is in the fridge, what expires soon, which products are harmful, or which items have certain brands or ingredients, use get_kitchen_overview and reason over item fields like storage_location, is_opened, remaining_quantity, quantity_value, quantity_unit, fill_percent, expiration_date, brand, variant, ingredients, nutrition_summary, harmful_ingredients, upf, and estimated_price.",
    "- If one strong kitchen match exists for a broad food reference like 'rice' or 'that rice', go ahead and use it.",
    "- On app, if several similar items match a broad food reference, ask one short clarification question and name the most likely candidates from current data. On Halo, do not ask — act on the best match (newest), or on all matches for a removal.",
    "- Use the narrowest matching shopping tool for changes.",
    "- Reuse an existing shopping store tab when the requested store clearly refers to it with different spelling, capitalization, or shorthand.",
    "- Use add_many_to_shopping_list when the user clearly names multiple items in a single add request.",
    "- Use update_many_shopping_item_stores when the user clearly gives multiple item-to-store moves in one request.",
    "- Do not split a spoken shopping phrase into multiple items unless the current shopping list strongly supports that interpretation.",
    "- Use mark_shopping_item_bought when the user says an item was bought, checked off, or already picked up.",
    "- Use mark_shopping_item_unbought when the user says an item should go back on the list or be unchecked.",
    "- Use update_shopping_item_store only when the user wants an existing shopping item moved to another store tab.",
    "- Use clear_shopping_list when the user clearly wants the whole shopping list cleared, emptied, wiped, or all remaining shopping items removed.",
    "- If duplicates exist but the user's intent is clearly to remove everything from shopping, do not block on duplicate item clarification. Use clear_shopping_list.",
    "- Use list_store_tabs when the user asks which shopping store tabs already exist.",
    "- Use check_in_many_items when the user clearly names multiple items being added to the kitchen in one request.",
    "- Use update_item_quantity for partial remaining-amount changes like half full, almost empty, nearly empty, 2 left, only 4 left, down to 2, one remaining, 50 percent remaining, or 3/4 full.",
    "- Use mark_item_opened only for opened-state changes that clearly mean opened or open now.",
    "- Use update_item_expiration only for expiration-date changes.",
    "- Use update_item_location only for storage-location changes.",
    "- Use update_item_details when the user changes multiple kitchen properties at once or when a specific detail does not fit a narrower kitchen tool, especially for sealed, unopened, closed, brand, or category updates.",
    "- Use discard_item when the user throws away, discards, tosses, removes, takes out, or fully deducts an item from the kitchen so it lands in discard history/feed.",
    "- Treat phrases like 'only 4 left', 'just 2 left', 'down to one', 'last one', 'half full', 'quarter left', '75 percent full', 'nearly empty', 'still sealed', 'unopened', 'move it to the pantry', and 'it expires Friday' as kitchen field updates when they are grounded to a specific item.",
    "- Use clear_kitchen_inventory when the user clearly wants the whole kitchen or all active kitchen inventory cleared, emptied, wiped, or removed in one action.",
    "- Use get_recent_discards when the user asks what they discarded, threw away, wasted, or wants to review recent discards for missed restocks.",
    "- Use delete_recent_discard when the user wants one item removed from discard history.",
    "- Use clear_recent_discards when the user clearly wants the whole discard history cleared, emptied, wiped, or all recent discards removed.",
    "- Use log_dish_from_voice when the user logs a new meal, snack, or food from speech as one dish entry.",
    "- Use log_dish_ingredients when the user mainly names foods or ingredients they ate together.",
    "- Use append_to_recent_dish for short follow-up dish additions like 'also had' or 'add peanut butter too' when recent dish context is clear.",
    "- Use update_recent_dish for recent-dish follow-ups that remove ingredients, replace the ingredient list, or otherwise revise the last logged meal.",
    "- Use get_recent_dishes for recent meal history and get_dish_detail for one meal's details.",
    "- When the user asks broad household state questions about what they have eaten, discarded, or still have in the kitchen, use the relevant read tools such as get_recent_dishes, get_recent_discards, get_kitchen_overview, and get_shopping_list before answering.",
    "- Use mark_dish_consumed when the user says a meal was finished, done, or consumed.",
    "- Use delete_dish_log when the user wants a recent meal log removed.",
    "- Use add_dish_ingredients_to_shopping_list when the user wants ingredients from a meal added to shopping.",
    "- Use get_health_metrics for overall health metrics such as Kitchen IQ, UPF, points, harmful ingredient counts, or why a household metric is high or low.",
    "- For questions about which specific kitchen products may be harmful, which items have bad ingredients, or what harmful ingredients are in products, use get_kitchen_overview or search_kitchen_item and inspect item fields like harmful_ingredients, upf, ingredients, and nutrition_summary when available.",
    "- For broad harmful-item questions, you may use both get_health_metrics and get_kitchen_overview when the user wants both the big picture and the specific products.",
    "- Do not say you cannot see the specific products or ingredients if kitchen item data already includes harmful_ingredients or related product details.",
    "- Use get_recipe_suggestions when the user asks what they can make from what they have, what they should eat based on their kitchen, or wants ideas from their current inventory. Do NOT use get_recipe_suggestions when the user asks for a specific recipe by name — just answer directly.",
    "- For broad recipe-idea answers, lead with your top 1-2 picks and invite a follow-up. Do not dump a long list of similar options — be decisive and recommend confidently.",
    "- For broad recipe-idea answers on Halo, prioritize the single best option first, then at most 2 backup options.",
    "- For broad recipe, meal-prep, or what-should-I-eat questions on app, do not assume kitchen-only. If needed, ask one short clarification such as whether they want to use what they have or are open to buying ingredients.",
    "- For broad recipe, meal-prep, or what-should-I-eat questions on Halo, stay kitchen-first and avoid clarifying unless the answer would otherwise be misleading.",
    "- When get_recipe_suggestions returns both kitchen_only and need_grocery options, Halo should usually emphasize kitchen_only first, while app can mention or clarify around both paths.",
    "- Apply the same surface-aware thinking to broader eating and meal-planning questions, including use-up ideas, expiring-food meal ideas, and cleaner-ingredient meal suggestions.",
    "- Use get_recipe_detail only when the user is asking about a recipe from their suggestions list (from get_recipe_suggestions). Do not use it for general recipe requests.",
    "- If the user names a recipe title from their suggestions and wants details, use get_recipe_detail.",
    "- When the user asks for a general recipe like stir fry, tacos, pasta, curry, or soup, answer directly from your knowledge. Only use get_recipe_detail if the user is referencing a specific suggestion from their kitchen-based recommendations.",
    "- Use add_recipe_ingredients_to_shopping_list when the user wants ingredients from a recipe suggestion (from get_recipe_suggestions or get_recipe_detail) added to shopping. Default to missing grocery items unless the user clearly asks for every ingredient.",
    "- When the user wants to add ingredients from a web-searched recipe (from search_web_recipes) to shopping, use add_many_to_shopping_list with each item's store set to the recipe name as the tab. Do NOT use add_recipe_ingredients_to_shopping_list for web-searched recipes since they are not saved or suggested recipes.",
    "- Questions like 'what do I need to buy', 'what ingredients do I need', 'what is missing', or 'which groceries would I need' are read-only by default. Answer first with the ingredients or missing groceries and do not add anything to shopping unless the user explicitly asks you to add it.",
    "- Use get_saved_recipes when the user asks what recipes they have saved, imported, bookmarked, or kept for later.",
    "- Use get_saved_recipe_detail when the user asks for one specific saved recipe or wants to open a saved recipe.",
    "- Use save_recipe_from_tiktok when the user shares a TikTok link, Instagram Reel URL, or recipe webpage URL and wants it saved or imported.",
    "- Use save_generated_recipe when the user wants to save a recipe YOU just generated or described in this chat (one that did NOT come from a URL or their existing saved/suggested recipes). Pass the exact title, the full ingredient list, and the ordered steps you presented. Do NOT say you are unable to save recipes you generated — you can, with this tool.",
    "- Use remove_saved_recipe when the user wants a saved recipe deleted or removed from saved recipes.",
    "- IMPORTANT: When the user asks for a specific recipe by name (e.g. 'give me a lemon ninja creami recipe', 'how do I make beef Wellington'), just answer directly from your own knowledge. You are a knowledgeable cooking assistant — give them the recipe. Do not search saved recipes or suggestions first unless the user specifically references their saved or suggested recipes.",
    "- Use search_web_recipes when the user wants a sourced recipe from a specific website or blog, or when you want to supplement your answer with a web source.",
    "- Do NOT use get_saved_recipes or get_recipe_suggestions as a prerequisite before answering recipe questions. Only use those tools when the user explicitly asks about THEIR saved recipes or suggestions (e.g. 'what recipes do I have saved', 'what can I make with what I have').",
    "- When building the search_web_recipes query, incorporate any dietary preferences, allergies, or cuisine types the user mentioned, e.g. 'gluten free apple crumble' or 'dairy free chocolate cake'.",
    "- After search_web_recipes returns results, pick the single best match and recommend it confidently. Only show additional options if they are meaningfully different. Present each recipe title as a line ending with colon, with source and ingredient details as bullets underneath.",
    "- After presenting web recipe search results, offer to save the recipe using save_recipe_from_tiktok with the source_url, or to add missing ingredients to the shopping list under a tab named after the recipe.",
    "- If search_web_recipes returns no results, suggest the user try different search terms or a broader query.",
    "- Use add_saved_recipe_ingredients_to_shopping_list when the user wants ingredients from a saved recipe, imported recipe, or a just-saved recipe added to shopping.",
    "- Use get_meal_plan for today's plan, current meal-plan focus, or the current regeneration state.",
    "- Use refresh_meal_plan when the user wants a new meal plan or wants to change the plan focus, such as higher protein or easier meals.",
    "- Use recommend_items_to_buy for buy-next suggestions and recommend_items_to_use_up for use-soon suggestions.",
    "- Before saying you are not sure which kitchen item, shopping item, dish, or saved recipe the user means, use the current household context to infer the most likely match.",
    "- If one likely kitchen/shopping/dish match clearly stands out, prefer a confident confirmation like 'If you mean X...' instead of sounding confused.",
    "- On app, only ask a clarifying question when multiple plausible matches remain after checking available context. On Halo, do not ask — act on the most reasonable match.",
    "- IMPORTANT: Do not apply this matching logic to general recipe requests. If the user asks for a recipe (e.g. 'give me a chicken tikka masala recipe'), just give them the recipe. Do not try to match it against saved recipes or kitchen items.",
    "- For grocery-store or restock questions like what they need, what they missed, or whether discarded items should be re-bought, compare the shopping list against recent discards, recent dishes, and kitchen state when helpful instead of relying on the shopping list alone.",
    "- Treat follow-up dish updates conservatively. If recent dish context is not clear, ask a short clarifying question.",
    "- When a user corrects a recent dish by naming the new full ingredient set, pass that full list with update_recent_dish instead of only add/remove deltas.",
    "- On app, if a kitchen item name is ambiguous, ask a short clarifying question instead of guessing. On Halo, do not ask — act on the most reasonable match (newest for edits, all matches for a removal).",
    "- If the assistant just asked for confirmation about clearing the shopping list or removing all remaining shopping items, short follow-ups like 'remove', 'yes', 'do it', 'clear it', or 'everything' should usually be treated as confirmation to use clear_shopping_list unless the user narrows the scope.",
    "- If the assistant just asked for confirmation about clearing the kitchen or the user just clarified 'kitchen' after a shopping-vs-kitchen question, short follow-ups like 'kitchen', 'remove', 'yes', 'do it', 'clear it', or 'everything' should usually be treated as confirmation to use clear_kitchen_inventory unless the user narrows the scope.",
    "- If the assistant just asked for confirmation about clearing discard history or removing all recent discards, short follow-ups like 'remove', 'yes', 'do it', 'clear it', or 'everything' should usually be treated as confirmation to use clear_recent_discards unless the user narrows the scope.",
    "- On app, ask a short clarifying question if the request is ambiguous. On Halo, make your best reasonable interpretation and act, since the user cannot answer a follow-up.",
    "- CRITICAL: Never claim you performed an action unless you actually called a tool to do it. If you did not call a tool, you did not do the action.",
    "- CRITICAL: If the user asks you to do something you do not have a tool for — such as setting a timer, playing music, sending a message, making a phone call, opening an app, controlling smart home devices, setting reminders, or anything else not in your tool list — be honest and say you cannot do that. Then suggest something related that you CAN do. For example: 'I can't set a timer, but I can tell you the cooking time so you can set one yourself.' or 'I can't send a message, but I can add those ingredients to your shopping list.'",
    "- Never invent actions outside the provided tools.",
    "- After tool calls, respond with a short grounded answer based on the results.",
    "- When the answer naturally contains multiple items, suggestions, or next steps, format it as a short intro followed by one item per line.",
    "- Prefer real bullet-style lines over inline dash-separated lists.",
    "- When using bullets, prefer plain hyphen bullets.",
    "- Use blank lines between a short intro paragraph and a bullet list when space allows.",
    "- If the answer already clearly solves the request, stop. Do not add extra optional commentary.",
    "- Do not mention internal tools, tables, JSON, or backend processing."
  ].join("\n");
}
