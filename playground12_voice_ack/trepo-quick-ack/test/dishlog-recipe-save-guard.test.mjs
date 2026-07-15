// Mutual-exclusion guard: "save a recipe (to cook later)" and "log a dish (that
// I ate)" are different actions. On a save-recipe intent the model sometimes
// emits BOTH save_generated_recipe AND log_dish_ingredients in one turn — the
// recipe save is correct, the dish-log write fabricates a phantom "you ate this"
// row (07-14: "Save turkey fried rice"). The guard suppresses the dish-log write
// whenever a recipe-save tool is present in the same request, and must NOT
// over-suppress a legitimate "I ate X" turn (no save tool → dish log allowed).
import { test } from "node:test";
import assert from "node:assert/strict";
import { batchHasRecipeSaveTool, shouldSuppressDishLog } from "../lib/device-assistant.mjs";

const DISH_LOG_TOOLS = [
  "log_dish_ingredients",
  "log_dish_from_voice",
  "update_recent_dish",
  "append_to_recent_dish",
  "mark_dish_consumed",
];

test("batchHasRecipeSaveTool: detects both recipe-save tools", () => {
  assert.equal(batchHasRecipeSaveTool(["save_generated_recipe", "log_dish_ingredients"]), true);
  assert.equal(batchHasRecipeSaveTool(["save_recipe_from_tiktok"]), true);
  assert.equal(batchHasRecipeSaveTool(["log_dish_ingredients"]), false);
  assert.equal(batchHasRecipeSaveTool([]), false);
  assert.equal(batchHasRecipeSaveTool(undefined), false);
});

test("BUG repro: save + log_dish_ingredients in one turn → dish log suppressed, save proceeds", () => {
  // The exact incident shape: model calls both in the same turn.
  const batchHasRecipeSave = batchHasRecipeSaveTool(["save_generated_recipe", "log_dish_ingredients"]);
  assert.equal(batchHasRecipeSave, true);
  // The phantom dish-log write is suppressed…
  assert.equal(shouldSuppressDishLog("log_dish_ingredients", { batchHasRecipeSave }), true);
  // …while the recipe save itself is NOT suppressed (it proceeds).
  assert.equal(shouldSuppressDishLog("save_generated_recipe", { batchHasRecipeSave }), false);
});

test("every dish-log write tool is suppressed when a recipe-save is in the turn", () => {
  for (const tool of DISH_LOG_TOOLS) {
    assert.equal(shouldSuppressDishLog(tool, { batchHasRecipeSave: true }), true, `should suppress: ${tool}`);
  }
});

test("recipe-save in an EARLIER turn still suppresses a later dish-log write", () => {
  // batchHasRecipeSave is false this turn, but recipeSaveIssued carries from before.
  assert.equal(shouldSuppressDishLog("log_dish_ingredients", { batchHasRecipeSave: false, recipeSaveIssued: true }), true);
});

test("CONTROL: 'I ate turkey fried rice' (no save tool) → dish log allowed", () => {
  const batchHasRecipeSave = batchHasRecipeSaveTool(["log_dish_ingredients"]);
  assert.equal(batchHasRecipeSave, false);
  assert.equal(shouldSuppressDishLog("log_dish_ingredients", { batchHasRecipeSave, recipeSaveIssued: false }), false);
  for (const tool of DISH_LOG_TOOLS) {
    assert.equal(shouldSuppressDishLog(tool, { batchHasRecipeSave: false, recipeSaveIssued: false }), false, `should allow: ${tool}`);
  }
});

test("guard does not touch non-dish, non-save tools (e.g. delete_dish_log, shopping)", () => {
  // delete_dish_log is deliberately NOT a suppressed dish-log write (suppressing a
  // delete could strand a bad row).
  assert.equal(shouldSuppressDishLog("delete_dish_log", { batchHasRecipeSave: true }), false);
  assert.equal(shouldSuppressDishLog("add_to_shopping_list", { batchHasRecipeSave: true }), false);
});
