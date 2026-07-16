// Thyme local tools: create_recipe_category, batch move_recipe_to_category,
// and the meal-calendar suite (get/add/add_many/move/remove).
// These are LOCAL-only tools (not in shared tool-definitions.mjs) — advertised by
// quick-ack's buildChatTools() and executed by quick-ack's executeToolAction wrapper.
// Planning a meal onto the calendar is NEITHER a dish log NOR a recipe save: it must
// never trip the recipe-save-vs-dishlog guard, must never fabricate a confirmation,
// and a calendar confirmation must resolve to the meal_calendar claim domain (backed
// by a calendar write) rather than the dishes domain.
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { executeToolAction } from "../lib/tool-actions.mjs";
import { buildChatTools } from "../lib/realtime-config.mjs";

const OWNER = "1f4db6b6-2f62-4558-aa40-c8e82527dc74";
const BASE = "https://calendar.test";
const ENV = { ACTION_MODE: "real", HOUSEHOLD_API_BASE_URL: BASE };
const CTX = { ownerId: OWNER, userId: OWNER, tableOwnerId: OWNER };

// ── Stateful in-memory mock of the grocery + meal-calendar API ───────────
function installMockApi({ recipes = [], categories = [], assignments = {}, entries = [] } = {}) {
  const state = {
    recipes: recipes.map((r) => ({ ...r })),
    categories: categories.map((c) => ({ ...c })),
    assignments: JSON.parse(JSON.stringify(assignments)),
    entries: entries.map((e) => ({ ...e })),
    posts: [],
    puts: [],
    deletes: []
  };
  let idSeq = 1;

  globalThis.fetch = async (url, options = {}) => {
    const method = (options.method || "GET").toUpperCase();
    const full = String(url).slice(BASE.length);
    const path = full.split("?")[0];
    const body = options.body ? JSON.parse(options.body) : null;
    const json = (payload) => ({ ok: true, status: 200, text: async () => JSON.stringify(payload) });
    const fail = (status, error) => ({ ok: false, status, text: async () => JSON.stringify({ error }) });

    // GET /saved-recipes/{owner}
    if (method === "GET" && /^\/saved-recipes\/[^/]+$/.test(path)) {
      return json({ owner: OWNER, recipes: state.recipes, count: state.recipes.length });
    }
    // PUT /saved-recipes/{owner}/{id}/categories
    const putCat = path.match(/^\/saved-recipes\/[^/]+\/([^/]+)\/categories$/);
    if (method === "PUT" && putCat) {
      const rid = decodeURIComponent(putCat[1]);
      state.assignments[rid] = [...(body?.category_ids || [])];
      return json({ owner: OWNER, recipe_id: rid, category_ids: state.assignments[rid] });
    }
    // GET /recipe-categories/{owner}
    if (method === "GET" && /^\/recipe-categories\/[^/]+$/.test(path)) {
      return json({ owner: OWNER, categories: state.categories, assignments: state.assignments });
    }
    // POST /recipe-categories/{owner}
    if (method === "POST" && /^\/recipe-categories\/[^/]+$/.test(path)) {
      const name = String(body?.name || "").trim();
      const existing = state.categories.find((c) => c.name.toLowerCase() === name.toLowerCase());
      if (existing) return json({ category: existing });
      const created = { id: `cat-${idSeq++}`, name, color: null, sort_order: state.categories.length };
      state.categories.push(created);
      return json({ category: created });
    }
    // GET /meal-calendar/{owner} — return ALL entries (ignore range for determinism)
    if (method === "GET" && /^\/meal-calendar\/[^/]+$/.test(path)) {
      return json({ owner: OWNER, start: null, end: null, entries: state.entries });
    }
    // POST /meal-calendar/{owner}
    if (method === "POST" && /^\/meal-calendar\/[^/]+$/.test(path)) {
      if (!body?.title) return fail(400, "title is required");
      const entry = { _id: `entry-${idSeq++}`, _owner: OWNER, ...body };
      state.entries.push(entry);
      state.posts.push(entry);
      return json({ message: "Added to meal calendar", entry });
    }
    // PUT /meal-calendar/{owner}/{id}
    const putEntry = path.match(/^\/meal-calendar\/[^/]+\/([^/]+)$/);
    if (method === "PUT" && putEntry) {
      const id = decodeURIComponent(putEntry[1]);
      const entry = state.entries.find((e) => e._id === id);
      if (!entry) return fail(404, `Entry ${id} not found`);
      Object.assign(entry, body);
      state.puts.push({ id, body });
      return json({ message: "Entry updated", entry });
    }
    // DELETE /meal-calendar/{owner}/{id}
    const delEntry = path.match(/^\/meal-calendar\/[^/]+\/([^/]+)$/);
    if (method === "DELETE" && delEntry) {
      const id = decodeURIComponent(delEntry[1]);
      state.entries = state.entries.filter((e) => e._id !== id);
      state.deletes.push(id);
      return json({ message: "Entry deleted", item_id: id });
    }
    throw new Error(`Unexpected mock request: ${method} ${path}`);
  };
  return state;
}

// A plan_date near "today" so the move/remove resolver's window includes it.
function isoOffset(days) {
  const now = new Date();
  const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + days));
  return d.toISOString().slice(0, 10);
}

let savedFetch;
beforeEach(() => { savedFetch = globalThis.fetch; });
afterEach(() => { globalThis.fetch = savedFetch; });

// ── buildChatTools advertises the new local tools ─────────────────────────
test("buildChatTools advertises category + meal-calendar tools (keeps shared)", () => {
  const names = buildChatTools().map((t) => t.function?.name);
  for (const n of [
    "list_recipe_categories", "move_recipe_to_category", "create_recipe_category",
    "get_meal_calendar", "add_recipe_to_meal_calendar", "add_many_to_meal_calendar",
    "move_meal_calendar_entry", "remove_meal_calendar_entry"
  ]) {
    assert.ok(names.includes(n), `missing tool ${n}`);
  }
  assert.ok(names.includes("save_generated_recipe")); // shared not clobbered
  const move = buildChatTools().find((t) => t.function?.name === "move_recipe_to_category");
  assert.deepEqual(move.function.parameters.required, ["category_name"]);
  assert.ok(move.function.parameters.properties.recipe_names, "batch recipe_names advertised");
});

// ── create_recipe_category ────────────────────────────────────────────────
test("create_recipe_category creates a new tag (created:true), idempotent second time", async () => {
  const state = installMockApi({ categories: [] });
  const first = await executeToolAction({ toolName: "create_recipe_category", args: { category_name: "Seafood" }, env: ENV, userContext: CTX });
  assert.equal(first.ok, true);
  assert.equal(first.toolResult.created, true);
  assert.equal(first.toolResult.name, "Seafood");
  assert.equal(state.categories.length, 1);
  const second = await executeToolAction({ toolName: "create_recipe_category", args: { category_name: "seafood" }, env: ENV, userContext: CTX });
  assert.equal(second.ok, true);
  assert.equal(second.toolResult.created, false, "existing category → created:false");
  assert.equal(state.categories.length, 1, "no duplicate created");
});

// ── batch move_recipe_to_category ─────────────────────────────────────────
test("batch move (recipe_names): per-recipe results, one not-found is honest, category made once", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Salmon Teriyaki" }, { id: "r2", title: "Shrimp Tacos" }],
    categories: []
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_names: ["Salmon Teriyaki", "Shrimp Tacos", "Unicorn Stew"], category_name: "Seafood" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true, "partial success is ok");
  assert.equal(res.toolResult.created_category, true);
  assert.equal(res.toolResult.moved, 2);
  const byName = Object.fromEntries(res.toolResult.results.map((r) => [r.recipe_name, r]));
  assert.equal(byName["Salmon Teriyaki"].ok, true);
  assert.equal(byName["Shrimp Tacos"].ok, true);
  assert.equal(byName["Unicorn Stew"].ok, false);
  assert.equal(byName["Unicorn Stew"].error, "recipe_not_found");
  assert.equal(state.categories.length, 1, "category created exactly once for the batch");
});

test("batch move: a recipe already in the category still reports ok (already_filed)", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Salmon Teriyaki" }],
    categories: [{ id: "c1", name: "Seafood" }],
    assignments: { r1: ["c1"] }
  });
  const res = await executeToolAction({
    toolName: "move_recipe_to_category",
    args: { recipe_names: ["Salmon Teriyaki"], category_name: "Seafood" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.results[0].ok, true);
  assert.equal(res.toolResult.results[0].already_filed, true);
});

// ── get_meal_calendar ─────────────────────────────────────────────────────
test("get_meal_calendar returns entries (empty → honest empty)", async () => {
  installMockApi({ entries: [] });
  const res = await executeToolAction({ toolName: "get_meal_calendar", args: {}, env: ENV, userContext: CTX });
  assert.equal(res.ok, true);
  assert.deepEqual(res.toolResult.entries, []);
});

// ── add_recipe_to_meal_calendar ───────────────────────────────────────────
test("add_recipe_to_meal_calendar: schedules a saved recipe (denormalized snapshot)", async () => {
  const state = installMockApi({
    recipes: [{ id: "r1", title: "Salmon Teriyaki", image_url: "img", ingredients: ["salmon"], instructions: ["cook"], meal_category: "dinner" }]
  });
  const res = await executeToolAction({
    toolName: "add_recipe_to_meal_calendar",
    args: { recipe_name: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(state.entries.length, 1);
  assert.equal(state.posts[0].source_type, "saved");
  assert.equal(state.posts[0].source_id, "r1");
  assert.equal(state.posts[0].title, "Salmon Teriyaki");
  assert.deepEqual(state.posts[0].ingredients, ["salmon"]);
  assert.equal(res.toolResult.meal_slot, "dinner");
});

test("add to a DIFFERENT slot than usual (dinner recipe → breakfast) is allowed", async () => {
  const state = installMockApi({ recipes: [{ id: "r1", title: "Veggie Pasta" }] });
  const res = await executeToolAction({
    toolName: "add_recipe_to_meal_calendar",
    args: { recipe_name: "Veggie Pasta", plan_date: isoOffset(1), meal_slot: "breakfast" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(state.posts[0].meal_slot, "breakfast");
});

test("CONTROL: add recipe not found → honest recipe_not_found (NO write)", async () => {
  const state = installMockApi({ recipes: [{ id: "r1", title: "Salmon Teriyaki" }] });
  const res = await executeToolAction({
    toolName: "add_recipe_to_meal_calendar",
    args: { recipe_name: "Unicorn Stew", plan_date: isoOffset(1), meal_slot: "dinner" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "recipe_not_found");
  assert.equal(res.actionSummary, null);
  assert.equal(state.entries.length, 0, "no calendar write on not-found");
});

test("CONTROL: invalid date (2026-02-30) → invalid_date, no write", async () => {
  const state = installMockApi({ recipes: [{ id: "r1", title: "Salmon Teriyaki" }] });
  const res = await executeToolAction({
    toolName: "add_recipe_to_meal_calendar",
    args: { recipe_name: "Salmon Teriyaki", plan_date: "2026-02-30", meal_slot: "dinner" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "invalid_date");
  assert.equal(state.entries.length, 0);
});

test("CONTROL: invalid meal_slot → invalid_meal_slot, no write", async () => {
  const state = installMockApi({ recipes: [{ id: "r1", title: "Salmon Teriyaki" }] });
  const res = await executeToolAction({
    toolName: "add_recipe_to_meal_calendar",
    args: { recipe_name: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "brunch" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "invalid_meal_slot");
  assert.equal(state.entries.length, 0);
});

// ── add_many_to_meal_calendar ─────────────────────────────────────────────
test("add_many: partial success reported per entry", async () => {
  const state = installMockApi({ recipes: [{ id: "r1", title: "Salmon Teriyaki" }, { id: "r2", title: "Shrimp Tacos" }] });
  const res = await executeToolAction({
    toolName: "add_many_to_meal_calendar",
    args: { entries: [
      { recipe_name: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner" },
      { recipe_name: "Shrimp Tacos", plan_date: isoOffset(2), meal_slot: "dinner" },
      { recipe_name: "Unicorn Stew", plan_date: isoOffset(3), meal_slot: "dinner" }
    ] },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(res.toolResult.added, 2);
  assert.equal(res.toolResult.total, 3);
  assert.equal(res.toolResult.results.filter((r) => r.ok).length, 2);
  assert.equal(res.toolResult.results.find((r) => r.recipe_name === "Unicorn Stew").error, "recipe_not_found");
  assert.equal(state.entries.length, 2);
});

// ── move_meal_calendar_entry ──────────────────────────────────────────────
test("move by title → updates plan_date/slot", async () => {
  const state = installMockApi({
    entries: [{ _id: "e1", title: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner", source_type: "saved" }]
  });
  const res = await executeToolAction({
    toolName: "move_meal_calendar_entry",
    args: { title: "Salmon Teriyaki", new_date: isoOffset(5), new_meal_slot: "lunch" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(state.entries[0].plan_date, isoOffset(5));
  assert.equal(state.entries[0].meal_slot, "lunch");
});

test("CONTROL: move ambiguous title → entry_ambiguous (no write)", async () => {
  const state = installMockApi({
    entries: [
      { _id: "e1", title: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner" },
      { _id: "e2", title: "Salmon Teriyaki", plan_date: isoOffset(2), meal_slot: "dinner" }
    ]
  });
  const res = await executeToolAction({
    toolName: "move_meal_calendar_entry",
    args: { title: "Salmon Teriyaki", new_date: isoOffset(5) },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "entry_ambiguous");
  assert.equal(res.statusCode, 409);
});

test("CONTROL: move not-found title → entry_not_found", async () => {
  installMockApi({ entries: [{ _id: "e1", title: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner" }] });
  const res = await executeToolAction({
    toolName: "move_meal_calendar_entry",
    args: { title: "Unicorn Stew", new_date: isoOffset(5) },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "entry_not_found");
});

test("CONTROL: move with invalid new_date → invalid_date", async () => {
  installMockApi({ entries: [{ _id: "e1", title: "Salmon Teriyaki", plan_date: isoOffset(1), meal_slot: "dinner" }] });
  const res = await executeToolAction({
    toolName: "move_meal_calendar_entry",
    args: { title: "Salmon Teriyaki", new_date: "2026-13-01" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "invalid_date");
});

// ── remove_meal_calendar_entry ────────────────────────────────────────────
test("remove by title → deletes the entry", async () => {
  const state = installMockApi({
    entries: [{ _id: "e1", title: "Veggie Pasta", plan_date: isoOffset(6), meal_slot: "dinner" }]
  });
  const res = await executeToolAction({
    toolName: "remove_meal_calendar_entry",
    args: { title: "Veggie Pasta", plan_date: isoOffset(6), meal_slot: "dinner" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, true);
  assert.equal(state.entries.length, 0);
  assert.deepEqual(state.deletes, ["e1"]);
});

test("CONTROL: remove not-found → entry_not_found (no delete)", async () => {
  const state = installMockApi({ entries: [{ _id: "e1", title: "Veggie Pasta", plan_date: isoOffset(6), meal_slot: "dinner" }] });
  const res = await executeToolAction({
    toolName: "remove_meal_calendar_entry",
    args: { title: "Unicorn Stew" },
    env: ENV, userContext: CTX
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, "entry_not_found");
  assert.deepEqual(state.deletes, []);
});

// ── Guard containment ─────────────────────────────────────────────────────
test("guard containment: calendar tools are neither recipe-save nor dish-log", async () => {
  const { batchHasRecipeSaveTool, shouldSuppressDishLog, detectActionClaim } = await import("../lib/device-assistant.mjs");
  const calendarTools = ["get_meal_calendar", "add_recipe_to_meal_calendar", "add_many_to_meal_calendar", "move_meal_calendar_entry", "remove_meal_calendar_entry", "create_recipe_category"];
  assert.equal(batchHasRecipeSaveTool(calendarTools), false);
  for (const t of calendarTools) {
    assert.equal(shouldSuppressDishLog(t, { batchHasRecipeSave: true }), false, `${t} must never be suppressed`);
  }

  // A calendar confirmation resolves to the meal_calendar claim domain (so a
  // calendar write backs it), NOT the dishes domain.
  assert.equal(detectActionClaim("I added Salmon Teriyaki to your calendar for Friday dinner")?.domain, "meal_calendar");
  assert.equal(detectActionClaim("Added it to your meal calendar.")?.domain, "meal_calendar");
  assert.equal(detectActionClaim("Scheduled Shrimp Tacos for dinner on Saturday.")?.domain, "meal_calendar");
  assert.equal(detectActionClaim("Moved your dinner to Saturday.")?.domain, "meal_calendar");
  assert.equal(detectActionClaim("Removed Veggie Pasta from your calendar.")?.domain, "meal_calendar");

  // Capability questions/answers must NOT look like a completed action.
  assert.equal(detectActionClaim("Yes, I can add recipes to your meal calendar and plan your week for you."), null);
  assert.equal(detectActionClaim("I can help you plan meals and put them on your calendar."), null);

  // A real dish log still resolves to the dishes domain (not stolen by calendar).
  assert.equal(detectActionClaim("Logged your lunch.")?.domain, "dishes");
});
