import {
  addShoppingItem,
  addManyShoppingItems,
  addDishIngredientsToShoppingList,
  addRecipeIngredientsToShoppingList,
  addSavedRecipeIngredientsToShoppingList,
  appendToRecentDish,
  listRecipeCategories,
  moveRecipeToCategory,
  moveManyRecipesToCategory,
  createRecipeCategory,
  getMealCalendar,
  addRecipeToMealCalendar,
  addManyToMealCalendar,
  moveMealCalendarEntry,
  removeMealCalendarEntry,
  checkInKitchenItem,
  checkInManyKitchenItems,
  clearKitchenInventory,
  clearRecentDiscards,
  clearShoppingList,
  deleteRecentDiscard,
  deleteDishLog,
  discardKitchenItem,
  getHealthMetrics,
  getDishDetail,
  getKitchenOverview,
  getMealPlan,
  getRecentDiscards,
  getRecentDishes,
  getRecipeDetail,
  getRecipeSuggestions,
  getSavedRecipeDetail,
  getSavedRecipes,
  getShoppingItems,
  listStoreTabs,
  logDishFromVoice,
  logDishIngredients,
  markDishConsumed,
  markKitchenItemOpened,
  markShoppingItemBought,
  markShoppingItemUnbought,
  removeShoppingItem,
  recommendItemsToBuy,
  recommendItemsToUseUp,
  removeSavedRecipe,
  refreshMealPlan,
  searchKitchenItem,
  searchWebRecipes,
  saveRecipeFromTikTok,
  saveGeneratedRecipe,
  updateKitchenItemDetails,
  updateKitchenItemExpiration,
  updateKitchenItemLocation,
  updateKitchenItemQuantity,
  updateRecentDish,
  updateManyShoppingItemStores,
  updateShoppingItemStore
} from "./data-access.mjs";
import {
  buildMockToolResult,
  createToolActionExecutor,
  formatActionDetails,
  formatActionSummary,
  humanizePrimaryEntity,
  toTitleCase,
  validateToolCall
} from "../../../shared/voice-assistant/tool-executor.mjs";

export { buildMockToolResult, formatActionDetails, formatActionSummary, toTitleCase, validateToolCall };

export function humanizeItemName(args) {
  return humanizePrimaryEntity(args);
}

const executeSharedToolAction = createToolActionExecutor({
  addShoppingItem,
  addManyShoppingItems,
  updateShoppingItemStore,
  updateManyShoppingItemStores,
  markShoppingItemBought,
  markShoppingItemUnbought,
  removeShoppingItem,
  clearShoppingList,
  getShoppingItems,
  listStoreTabs,
  getKitchenOverview,
  getRecentDiscards,
  searchKitchenItem,
  searchWebRecipes,
  checkInKitchenItem,
  checkInManyKitchenItems,
  updateKitchenItemQuantity,
  markKitchenItemOpened,
  updateKitchenItemExpiration,
  updateKitchenItemLocation,
  updateKitchenItemDetails,
  discardKitchenItem,
  clearKitchenInventory,
  deleteRecentDiscard,
  clearRecentDiscards,
  logDishFromVoice,
  logDishIngredients,
  appendToRecentDish,
  updateRecentDish,
  getRecentDishes,
  getDishDetail,
  markDishConsumed,
  deleteDishLog,
  addDishIngredientsToShoppingList,
  addRecipeIngredientsToShoppingList,
  addSavedRecipeIngredientsToShoppingList,
  getHealthMetrics,
  getRecipeSuggestions,
  getRecipeDetail,
  getSavedRecipes,
  getSavedRecipeDetail,
  saveRecipeFromTikTok,
  saveGeneratedRecipe,
  removeSavedRecipe,
  getMealPlan,
  refreshMealPlan,
  recommendItemsToBuy,
  recommendItemsToUseUp
});

// ── Local-only tools: recipe categories + meal calendar ──────────────────
// CONTAINMENT: these tools are NOT in the shared tool-definitions.mjs (and must
// not be — other services consume that module and would advertise tools their
// executors can't implement). They are advertised locally by quick-ack's
// buildChatTools() and executed here by wrapping the shared executor: unknown
// (to shared) local tools are handled here, everything else delegates unchanged.
// Filing a recipe into a category AND planning a meal onto the calendar are each
// NEITHER a dish log NOR a recipe save, so they emit none of the guarded tool
// names and cannot trip the recipe-save-vs-dishlog guard.
const LOCAL_CATEGORY_TOOLS = new Set([
  "list_recipe_categories",
  "move_recipe_to_category",
  "create_recipe_category"
]);
const LOCAL_MEAL_CALENDAR_TOOLS = new Set([
  "get_meal_calendar",
  "add_recipe_to_meal_calendar",
  "add_many_to_meal_calendar",
  "move_meal_calendar_entry",
  "remove_meal_calendar_entry"
]);
const LOCAL_TOOLS = new Set([...LOCAL_CATEGORY_TOOLS, ...LOCAL_MEAL_CALENDAR_TOOLS]);

// Honest-error → HTTP status mapping. A resolution miss must surface as non-ok so
// the model narrates truthfully instead of confirming an unperformed action.
const LOCAL_ERROR_STATUS = {
  recipe_not_found: 404,
  recipe_ambiguous: 409,
  entry_not_found: 404,
  entry_ambiguous: 409,
  invalid_date: 400,
  invalid_meal_slot: 400,
  no_recipes: 400,
  no_entries: 400
};

function slotLabel(slot) {
  const s = String(slot || "").trim();
  return s || "a slot";
}

// Compute a human action summary for a successful local tool result. Returns null
// when there is nothing to confirm (so the model never fabricates a confirmation).
function summarizeLocalResult(toolName, toolResult, args) {
  switch (toolName) {
    case "list_recipe_categories": {
      const names = (toolResult.categories || []).map((c) => c.name);
      return names.length
        ? `Found ${names.length} recipe categor${names.length === 1 ? "y" : "ies"}: ${names.join(", ")}.`
        : "No custom recipe categories yet.";
    }
    case "create_recipe_category":
      return toolResult.created
        ? `Created the "${toolResult.name}" category.`
        : `The "${toolResult.name}" category already exists.`;
    case "move_recipe_to_category": {
      if (Array.isArray(toolResult.results)) {
        const okOnes = toolResult.results.filter((r) => r.ok).map((r) => r.recipe_name);
        const failed = toolResult.results.filter((r) => !r.ok).map((r) => r.recipe_name);
        let s = okOnes.length ? `Filed ${okOnes.join(", ")} under ${toolResult.category_name}.` : "";
        if (failed.length) s += ` Couldn't find: ${failed.join(", ")}.`;
        return s.trim() || null;
      }
      return `Filed ${toolResult.recipe_title || "the recipe"} under ${toolResult.category_name}.`;
    }
    case "get_meal_calendar": {
      const n = (toolResult.entries || []).length;
      return n
        ? `Found ${n} planned meal${n === 1 ? "" : "s"}.`
        : "Nothing planned on the calendar for that range.";
    }
    case "add_recipe_to_meal_calendar":
      return `Scheduled ${toolResult.title || "the recipe"} for ${toolResult.plan_date} ${slotLabel(toolResult.meal_slot)}.`;
    case "add_many_to_meal_calendar": {
      const okOnes = (toolResult.results || []).filter((r) => r.ok);
      const failed = (toolResult.results || []).filter((r) => !r.ok);
      let s = okOnes.length
        ? `Added ${okOnes.length} meal${okOnes.length === 1 ? "" : "s"} to the calendar.`
        : "";
      if (failed.length) s += ` ${failed.length} couldn't be added.`;
      return s.trim() || null;
    }
    case "move_meal_calendar_entry":
      return `Moved ${toolResult.title || "the entry"} to ${toolResult.new_date}${toolResult.new_meal_slot ? ` ${toolResult.new_meal_slot}` : ""}.`;
    case "remove_meal_calendar_entry":
      return `Removed ${toolResult.title || "the entry"} from the calendar.`;
    default:
      return null;
  }
}

async function runLocalToolReal(toolName, args, userContext, options) {
  switch (toolName) {
    case "list_recipe_categories":
      return listRecipeCategories(userContext, options);
    case "create_recipe_category": {
      // createRecipeCategory is idempotent on the backend; determine created-ness
      // by checking whether the name pre-existed BEFORE creating.
      const norm = String(args.category_name || "").trim().toLowerCase();
      let existedBefore = false;
      try {
        const { categories } = await listRecipeCategories(userContext, options);
        existedBefore = categories.some((c) => String(c.name || "").trim().toLowerCase() === norm);
      } catch {
        existedBefore = false;
      }
      const category = await createRecipeCategory(userContext, args.category_name, options);
      return { ok: true, id: category.id, name: category.name, created: !existedBefore };
    }
    case "move_recipe_to_category":
      return Array.isArray(args.recipe_names) && args.recipe_names.length
        ? moveManyRecipesToCategory(userContext, args, options)
        : moveRecipeToCategory(userContext, args, options);
    case "get_meal_calendar":
      return getMealCalendar(userContext, args, options);
    case "add_recipe_to_meal_calendar":
      return addRecipeToMealCalendar(userContext, args, options);
    case "add_many_to_meal_calendar":
      return addManyToMealCalendar(userContext, args, options);
    case "move_meal_calendar_entry":
      return moveMealCalendarEntry(userContext, args, options);
    case "remove_meal_calendar_entry":
      return removeMealCalendarEntry(userContext, args, options);
    default:
      throw new Error(`Unknown local tool: ${toolName}`);
  }
}

function mockLocalToolResult(toolName, args) {
  switch (toolName) {
    case "list_recipe_categories":
      return { owner: "mock", categories: [], assignments: {} };
    case "create_recipe_category":
      return { ok: true, id: "mock-cat", name: String(args.category_name || "").trim(), created: true };
    case "move_recipe_to_category":
      return Array.isArray(args.recipe_names) && args.recipe_names.length
        ? { ok: true, category_name: String(args.category_name || "").trim(), created_category: false, moved: args.recipe_names.length, total: args.recipe_names.length, results: args.recipe_names.map((n) => ({ recipe_name: String(n || "").trim(), ok: true, already_filed: false })) }
        : { ok: true, recipe_title: String(args.recipe_name || "").trim() || null, category_name: String(args.category_name || "").trim(), created_category: false, already_filed: false };
    case "get_meal_calendar":
      return { owner: "mock", start: null, end: null, entries: [] };
    case "add_recipe_to_meal_calendar":
      return { ok: true, entry: { id: "mock-entry" }, plan_date: args.plan_date, meal_slot: args.meal_slot, title: String(args.recipe_name || "").trim() || null };
    case "add_many_to_meal_calendar":
      return { ok: true, added: (args.entries || []).length, total: (args.entries || []).length, results: (args.entries || []).map((e) => ({ recipe_name: e?.recipe_name || null, plan_date: e?.plan_date || null, meal_slot: e?.meal_slot || null, ok: true, entry_id: "mock-entry" })) };
    case "move_meal_calendar_entry":
      return { ok: true, entry: { id: "mock-entry" }, new_date: args.new_date, new_meal_slot: args.new_meal_slot || null, title: args.title || null };
    case "remove_meal_calendar_entry":
      return { ok: true, removed_id: "mock-entry", title: args.title || null, plan_date: args.plan_date || null, meal_slot: args.meal_slot || null };
    default:
      return { ok: true };
  }
}

async function executeLocalTool({ toolName, args, env, userContext }) {
  const normalizedArgs = typeof args === "object" && args !== null ? args : {};

  if (!userContext?.ownerId) {
    return {
      ok: false,
      statusCode: 400,
      error: "Connect a household before asking the assistant to use data tools."
    };
  }

  const actionMode = env?.ACTION_MODE === "real" ? "real" : "mock";
  const options = { env };

  console.log("[DEBUG] tool action start:", JSON.stringify({
    toolName,
    actionMode,
    args: normalizedArgs,
    context: {
      ownerId: userContext.ownerId,
      userId: userContext.userId,
      tableOwnerId: userContext.tableOwnerId
    }
  }));

  try {
    const toolResult = actionMode === "real"
      ? await runLocalToolReal(toolName, normalizedArgs, userContext, options)
      : mockLocalToolResult(toolName, normalizedArgs);

    // Surface a not-found / ambiguous / invalid resolution as a non-ok result so
    // the model narrates honestly (never confirm an unperformed action).
    if (toolResult && toolResult.ok === false) {
      const error = toolResult.error || "not_found";
      const response = {
        ok: false,
        statusCode: LOCAL_ERROR_STATUS[error] || 404,
        error,
        toolName,
        args: normalizedArgs,
        actionMode,
        actionSummary: null,
        toolResult
      };
      console.log("[DEBUG] tool action result:", JSON.stringify({
        toolName, ok: false, actionMode, statusCode: response.statusCode, error: response.error
      }));
      return response;
    }

    const actionSummary = summarizeLocalResult(toolName, toolResult || {}, normalizedArgs);
    const response = {
      ok: true,
      statusCode: 200,
      toolName,
      args: normalizedArgs,
      actionMode,
      actionSummary,
      actionDetails: null,
      toolResult
    };
    console.log("[DEBUG] tool action result:", JSON.stringify({
      toolName, ok: true, actionMode, statusCode: response.statusCode, actionSummary
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
      toolName, actionMode, statusCode: response.statusCode, error: response.error
    }));
    return response;
  }
}

export async function executeToolAction({ toolName, args, env, userContext, responseSurface }) {
  if (LOCAL_TOOLS.has(toolName)) {
    return executeLocalTool({ toolName, args, env, userContext, responseSurface });
  }
  return executeSharedToolAction({ toolName, args, env, userContext, responseSurface });
}
