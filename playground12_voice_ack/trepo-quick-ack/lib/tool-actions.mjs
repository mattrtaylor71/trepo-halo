import {
  addShoppingItem,
  addManyShoppingItems,
  addDishIngredientsToShoppingList,
  addRecipeIngredientsToShoppingList,
  addSavedRecipeIngredientsToShoppingList,
  appendToRecentDish,
  listRecipeCategories,
  moveRecipeToCategory,
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

// ── Local-only recipe-category tools ─────────────────────────────────────
// CONTAINMENT: these two tools are NOT in the shared tool-definitions.mjs (and
// must not be — other services consume that module and would advertise tools
// their executors can't implement). They are advertised locally by quick-ack's
// buildChatTools() and executed here by wrapping the shared executor: unknown
// (to shared) category tools are handled locally, everything else delegates
// unchanged. Filing a recipe into a category is NEITHER a dish log NOR a recipe
// save, so it emits none of the guarded tool names and cannot trip the
// recipe-save-vs-dishlog guard.
const LOCAL_CATEGORY_TOOLS = new Set(["list_recipe_categories", "move_recipe_to_category"]);

async function executeLocalCategoryTool({ toolName, args, env, userContext }) {
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
    let toolResult;
    let actionSummary = null;

    if (toolName === "list_recipe_categories") {
      toolResult = actionMode === "real"
        ? await listRecipeCategories(userContext, options)
        : { owner: userContext.ownerId, categories: [], assignments: {} };
      const names = (toolResult.categories || []).map((category) => category.name);
      actionSummary = names.length
        ? `Found ${names.length} recipe categor${names.length === 1 ? "y" : "ies"}: ${names.join(", ")}.`
        : "No custom recipe categories yet.";
    } else {
      // move_recipe_to_category
      toolResult = actionMode === "real"
        ? await moveRecipeToCategory(userContext, normalizedArgs, options)
        : {
          ok: true,
          recipe_title: String(normalizedArgs.recipe_name || "").trim() || null,
          category_name: String(normalizedArgs.category_name || "").trim(),
          created_category: false,
          already_filed: false
        };
      if (toolResult?.ok) {
        actionSummary = `Filed ${toolResult.recipe_title || "the recipe"} under ${toolResult.category_name}.`;
      } else {
        actionSummary = null;
      }
    }

    // Surface a not-found / ambiguous resolution as a non-ok result so the model
    // narrates honestly (never confirm an unperformed action).
    if (toolResult && toolResult.ok === false) {
      const response = {
        ok: false,
        statusCode: toolResult.error === "recipe_ambiguous" ? 409 : 404,
        error: toolResult.error || "recipe_not_found",
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
  if (LOCAL_CATEGORY_TOOLS.has(toolName)) {
    return executeLocalCategoryTool({ toolName, args, env, userContext, responseSurface });
  }
  return executeSharedToolAction({ toolName, args, env, userContext, responseSurface });
}
