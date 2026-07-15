// Custom recipe-category tools (Thyme): list_recipe_categories + move_recipe_to_category.
// These are LOCAL-only tools (not in shared tool-definitions.mjs) — advertised by
// quick-ack's buildChatTools() and executed by quick-ack's executeToolAction wrapper.
// Filing a recipe into a category is NEITHER a dish log NOR a recipe save, so it must
// never trip the recipe-save-vs-dishlog guard and must never fabricate a confirmation.
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { executeToolAction } from "../lib/tool-actions.mjs";
import { buildChatTools } from "../lib/realtime-config.mjs";

const OWNER = "1f4db6b6-2f62-4558-aa40-c8e82527dc74";
const BASE = "https://categories.test";
const ENV = { ACTION_MODE: "real", HOUSEHOLD_API_BASE_URL: BASE };
const CTX = { ownerId: OWNER, userId: OWNER, tableOwnerId: OWNER };

// ── Stateful in-memory mock of the grocery category API ──────────────────
function installMockApi({ recipes = [], categories = [], assignments = {} } = {}) {
  const state = {
    recipes: recipes.map((r) => ({ ...r })),
    categories: categories.map((c) => ({ ...c })),
    assignments: JSON.parse(JSON.stringify(assignments)),
    puts: []
  };
  let idSeq = 1;

  globalThis.fetch = async (url, options = {}) => {
    const method = (options.method || "GET").toUpperCase();
    const path = String(url).slice(BASE.length);
    const body = options.body ? JSON.parse(options.body) : null;

    const json = (payload, ok = true, status = 200) => ({
      ok,
      status,
      text: async () => JSON.stringify(payload)
    });

    // GET /saved-recipes/{owner}
    if (method === "GET" && /^\/saved-recipes\/[^/]+$/.test(path)) {
      return json({ owner: OWNER, recipes: state.recipes, count: state.recipes.length });
    }
    // PUT /saved-recipes/{owner}/{item_id}/categories
    const putMatch = path.match(/^\/saved-recipes\/[^/]+\/([^/]+)\/categories$/);
    if (method === "PUT" && putMatch) {
      const recipeId = decodeURIComponent(putMatch[1]);
      state.assignments[recipeId] = [...(body?.category_ids || [])];
      state.puts.push({ recipeId, category_ids: state.assignments[recipeId] });
      return json({ owner: OWNER, recipe_id: recipeId, category_ids: state.assignments[recipeId] });
    }
    // GET /recipe-categories/{owner}
    if (method === "GET" && /^\/recipe-categories\/[^/]+$/.test(path)) {
      return json({ owner: OWNER, categories: state.categories, assignments: state.assignments });
    }
    // POST /recipe-categories/{owner}
    if (method === "POST" && /^\/recipe-categories\/[^/]+$/.test(path)) {
      const name = String(body?.name || "").trim();
      const existing = state.categories.find((c) => c.name.toLowerCase() === name.toLowerCase());
      if (existing) return json({ category: existing }); // idempotent
      const created = { id: `cat-${idSeq++}`, name, color: body?.color || null, sort_order: state.categories.length };
      state.categories.push(created);
      return json({ category: created });
    }
    throw new Error(`Unexpected mock request: ${method} ${path}`);
  };

  return state;
}

let savedFetch;
beforeEach(() => { savedFetch = globalThis.fetch; });
afterEach(() => { globalThis.fetch = savedFetch; });

test("buildChatTools advertises the two local category tools (and keeps shared tools)", () => {
  const tools = buildChatTools();
  const names = tools.map((t) => t.function?.name);
  assert.ok(names.includes("list_recipe_categories"));
  assert.ok(names.includes("move_recipe_to_category"));
  // Shared tools are still present (not clobbered).
  assert.ok(names.includes("save_generated_recipe"));
  assert.ok(names.includes("get_saved_recipes"));
  // move schema requires only category_name.
  const move = tools.find((t) => t.function?.name === "move_recipe_to_category");
  assert.deepEqual(move.function.parameters.required, ["category_name"]);
});

test("list_recipe_categories returns the user's category names + ids", async () => {
  installMockApi({ categories: [{ id: "c1", name: "Desserts" }, { id: "c2", name: "Weeknight" }] });
  const res = await executeToolAction({ toolName: "list_recipe_categories", args: {}, env: ENV, userContext: CTX });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.categories.length, 2);
  assert.deepEqual(res.toolResult.categories.map((c) => c.name), ["Desserts", "Weeknight"]);
});

test("move existing recipe → existing category (union-adds, does not clobber)", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Banana Bread" }],
    categories: [{ id: "c1", name: "Desserts" }],
    assignments: { r1: ["c-existing"] } // already tagged with something else
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_name: "Banana Bread", category_name: "Desserts" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.recipe_title, "Banana Bread");
  assert.equal(res.toolResult.category_name, "Desserts");
  assert.equal(res.toolResult.created_category, false);
  // Existing tag preserved, new tag added (add-not-replace).
  assert.deepEqual(state.assignments.r1.sort(), ["c-existing", "c1"].sort());
});

test("move → NEW category auto-creates it", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Zoodle Stir Fry" }],
    categories: []
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_name: "Zoodle Stir Fry", category_name: "Gluten-Free" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.created_category, true);
  assert.equal(res.toolResult.category_name, "Gluten-Free");
  assert.equal(state.categories.length, 1);
  assert.equal(state.categories[0].name, "Gluten-Free");
  assert.deepEqual(state.assignments.r1, [state.categories[0].id]);
});

test("save-then-categorize: resolve by the just-saved TITLE", async () => {
  // Recipe already exists (as it would right after a save); file it by title.
  const state = installMockApi({
    recipes: [{ id: "r-new", title: "Turkey Fried Rice" }],
    categories: [{ id: "c1", name: "Weeknight Dinners" }]
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_name: "Turkey Fried Rice", category_name: "Weeknight Dinners" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.recipe_id, "r-new");
  assert.deepEqual(state.assignments["r-new"], ["c1"]);
});

test("move by category name is case-insensitive (no duplicate category created)", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Apple Crumble" }],
    categories: [{ id: "c1", name: "Desserts" }]
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_name: "Apple Crumble", category_name: "desserts" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.created_category, false);
  assert.equal(res.toolResult.category_id, "c1");
  assert.equal(state.categories.length, 1);
});

test("CONTROL: move when recipe not found → honest not-found (no fabrication)", async () => {
  installMockApi({ recipes: [{ id: "r1", title: "Banana Bread" }], categories: [] });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_name: "Nonexistent Lasagna", category_name: "Dinner" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "recipe_not_found");
  assert.equal(res.actionSummary, null); // nothing to confirm
});

test("CONTROL: plain save (shared tool) still routes through the shared executor unchanged", async () => {
  // Non-category tools must delegate to the shared executor with no category side effects.
  // Mock mode → shared buildMockToolResult; proves the wrapper does not intercept saves.
  const res = await executeToolAction({
    toolName: "save_generated_recipe",
    args: { title: "Test Soup", ingredients: ["water"], steps: ["boil"] },
    env: { ACTION_MODE: "mock" }, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolName, "save_generated_recipe");
  // No fabricated category on a plain save.
  assert.ok(!("category_name" in res));
});

test("guard containment: neither category tool is a recipe-save or dish-log tool name", async () => {
  const { batchHasRecipeSaveTool, shouldSuppressDishLog } = await import("../lib/device-assistant.mjs");
  assert.equal(batchHasRecipeSaveTool(["move_recipe_to_category"]), false);
  assert.equal(batchHasRecipeSaveTool(["list_recipe_categories"]), false);
  // Category tools are never suppressed even when a save is in the same turn.
  assert.equal(shouldSuppressDishLog("move_recipe_to_category", { batchHasRecipeSave: true }), false);
  assert.equal(shouldSuppressDishLog("list_recipe_categories", { batchHasRecipeSave: true }), false);
});
