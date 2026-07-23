// Mutual-exclusion guard: "save a recipe (to cook later)" and "log a dish (that
// I ate)" are different actions. On a save-recipe intent the model sometimes
// emits BOTH save_generated_recipe AND log_dish_ingredients in one turn — the
// recipe save is correct, the dish-log write fabricates a phantom "you ate this"
// row (07-14: "Save turkey fried rice"). The guard suppresses the dish-log write
// whenever a recipe-save tool is present in the same request, and must NOT
// over-suppress a legitimate "I ate X" turn (no save tool → dish log allowed).
import { test, mock, before } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

// device-assistant is imported DYNAMICALLY (in the before() hook) AFTER the module
// mocks are registered, so the mocked data-access / tool-actions bindings actually
// take effect for the end-to-end save-all + F-041 tests. A static top-level import
// would eagerly cache the real deps and defeat mock.module. The pure-function
// exports (batchHasRecipeSaveTool / shouldSuppressDishLog / scrubGuardLeakFromNarration)
// are read off the same dynamically-imported module (da) inside test bodies.
const LIB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "lib");
const DEVICE_ASSISTANT = path.join(LIB, "device-assistant.mjs");
const DATA_ACCESS = path.join(LIB, "data-access.mjs");
const TOOL_ACTIONS = path.join(LIB, "tool-actions.mjs");

let batchHasRecipeSaveTool, shouldSuppressDishLog, scrubGuardLeakFromNarration;

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

// ── F-106: guard-leak scrub on the FINAL narration ─────────────────────
// The dish-log suppression guard is silent-correct (it skips a phantom dish log
// but the real kitchen add / list add / discard still succeeds). The bug was the
// model quoting the guard's internal reasoning at the user — tool names and
// "I can't honestly …" — even though the requested action worked. The
// deterministic scrub replaces that leaked narration with a clean confirmation.
const LEAKY_KITCHEN_REPLY = `I can't honestly call log_dish_ingredients for that message, because the user asked to add items to the kitchen, not to log something they ate.\n\nWhat I did confirm correctly was the kitchen check-in, and that action succeeded.`;

function assertCleanNarration(text) {
  const lower = text.toLowerCase();
  for (const tok of [
    "log_dish_ingredients", "log_dish_from_voice", "check_in_item", "check_in_many_items",
    "save_generated_recipe", "discard_item", "dish_log_suppressed", "skip_dishlog",
  ]) {
    assert.ok(!lower.includes(tok), `narration must not contain tool token "${tok}": ${text}`);
  }
  assert.ok(!/honestly/i.test(text), `narration must not say "honestly": ${text}`);
  assert.ok(!/can'?t\b/i.test(text) || /couldn'?t find/i.test(text), `narration must not read as a refusal: ${text}`);
}

test("F-106 scrub: leaked kitchen-add reply is replaced with a clean confirmation naming the items", () => {
  // "add frozen salmon and an onion to my kitchen" — dish log suppressed, kitchen add succeeded.
  const toolTrace = [
    { toolName: "check_in_many_items", ok: true, args: { items: [{ item_name: "Frozen Salmon" }, { item_name: "Onion" }] } },
    { toolName: "log_dish_ingredients", ok: false, suppressed: true, args: {} },
  ];
  const out = scrubGuardLeakFromNarration(LEAKY_KITCHEN_REPLY, toolTrace);
  assert.notEqual(out, LEAKY_KITCHEN_REPLY, "leaked reply should be rewritten");
  assert.match(out, /Frozen Salmon/);
  assert.match(out, /Onion/);
  assert.match(out, /Added to your kitchen/i);
  assertCleanNarration(out);
});

test("F-106 scrub: single check_in_item leak names the one item", () => {
  const toolTrace = [{ toolName: "check_in_item", ok: true, args: { item_name: "Cranberry Juice" } }];
  const out = scrubGuardLeakFromNarration("I can't honestly confirm a dish log here. I already completed the kitchen action.", toolTrace);
  assert.match(out, /Cranberry Juice/);
  assertCleanNarration(out);
});

test("F-106 scrub: a discard leak confirms the removal cleanly", () => {
  const toolTrace = [{ toolName: "discard_item", ok: true, args: { item_name: "Old Broccoli" } }];
  const out = scrubGuardLeakFromNarration("I can't honestly log a dish from that message. I did already complete the kitchen action.", toolTrace);
  assert.match(out, /Removed from your kitchen/i);
  assert.match(out, /Old Broccoli/);
  assertCleanNarration(out);
});

test("F-106 scrub: NO successful primary write → leave text alone (never invent a success)", () => {
  // If nothing primary landed, the false-confirm / honest-failure paths own this.
  const toolTrace = [{ toolName: "log_dish_ingredients", ok: false, suppressed: true, args: {} }];
  const leak = "I can't honestly log a dish from that message.";
  assert.equal(scrubGuardLeakFromNarration(leak, toolTrace), leak);
});

test("F-106 scrub: a clean reply with no leak is passed through untouched", () => {
  const clean = "Added to your kitchen:\n- Pesto\n- Butter";
  const toolTrace = [{ toolName: "check_in_many_items", ok: true, args: { items: [{ item_name: "Pesto" }, { item_name: "Butter" }] } }];
  assert.equal(scrubGuardLeakFromNarration(clean, toolTrace), clean);
});

// ── End-to-end: multi-recipe "save all N" + guard integrity ─────────────
// Drives the real non-streaming assistant loop with a stubbed OpenAI (globalThis.fetch)
// and a spy executeToolAction. Grounding builders short-circuit on ACTION_MODE!=="real",
// and getDietaryPreferences is stubbed, so no DB is touched.
let da;
let TOOL_CALLS = [];
let COMPLETION_QUEUE = [];

function toolCallMsg(calls) {
  // calls: [{ name, args }] → an OpenAI assistant message with tool_calls.
  return {
    choices: [{
      message: {
        role: "assistant",
        content: "",
        tool_calls: calls.map((c, i) => ({
          id: `call_${i}`,
          type: "function",
          function: { name: c.name, arguments: JSON.stringify(c.args || {}) },
        })),
      },
    }],
  };
}
function textMsg(text) {
  return { choices: [{ message: { role: "assistant", content: text, tool_calls: [] } }] };
}
function fakeFetchResponse(json) {
  return { ok: true, status: 200, json: async () => json, text: async () => JSON.stringify(json) };
}

before(async () => {
  const realData = await import(DATA_ACCESS);
  mock.module(DATA_ACCESS, {
    namedExports: {
      ...realData,
      getDietaryPreferences: async () => ({ allergies: [], diets: [], religious: [], health: [], custom: [] }),
    },
  });
  mock.module(TOOL_ACTIONS, {
    namedExports: {
      executeToolAction: async ({ toolName, args }) => {
        TOOL_CALLS.push({ toolName, args });
        // Every tool "succeeds" and echoes its args, so the trace carries item/title names.
        return { ok: true, statusCode: 200, toolName, args, actionSummary: `did ${toolName}`, toolResult: { ok: true } };
      },
    },
  });
  // Stub OpenAI HTTP: each call pops the next canned completion off the queue.
  globalThis.fetch = async () => {
    if (COMPLETION_QUEUE.length === 0) return fakeFetchResponse(textMsg("Okay."));
    return fakeFetchResponse(COMPLETION_QUEUE.shift());
  };
  da = await import(DEVICE_ASSISTANT);
  // Bind the pure-function exports off the mocked module so the unit tests use it too.
  ({ batchHasRecipeSaveTool, shouldSuppressDishLog, scrubGuardLeakFromNarration } = da);
});

const OFFERED = [
  "Turkey Burger Roll Melt",
  "Pesto Broccoli Rice Bowl",
  "Turkey Sausage Breakfast Wraps",
  "Garlic Caesar Chickpea Wraps",
  "Cheesy Broccoli Soup",
];

test("F-106 multi-recipe save: 'All five' issues 5 save_generated_recipe calls, no dish log, per-title confirm", async () => {
  TOOL_CALLS = [];
  // Turn 1: the model saves all five in one batch. Turn 2 (after tool results): confirmation text.
  COMPLETION_QUEUE = [
    toolCallMsg(OFFERED.map((title) => ({
      name: "save_generated_recipe",
      args: { title, ingredients: ["a", "b"], steps: ["1", "2"] },
    }))),
    textMsg(`Saved to your recipes:\n${OFFERED.map((t) => `- ${t}`).join("\n")}`),
  ];
  const result = await da.runDeviceAssistant({
    transcript: "All five",
    userContext: { ownerId: "ownerA", userId: "userA" },
    env: { ACTION_MODE: "mock", OPENAI_API_KEY: "test" },
    sessionMessages: [
      { role: "assistant", content: `Which one should I save? Reply with a title, or say "save all five" and I'll save them.` },
    ],
    responseSurface: "app",
  });
  const saves = TOOL_CALLS.filter((c) => c.toolName === "save_generated_recipe");
  assert.equal(saves.length, 5, `expected 5 saves, got ${saves.length}`);
  const savedTitles = saves.map((c) => c.args.title);
  for (const t of OFFERED) assert.ok(savedTitles.includes(t), `missing save for "${t}"`);
  // No dish-log tool fired on a save-all turn.
  assert.equal(TOOL_CALLS.filter((c) => c.toolName.startsWith("log_dish")).length, 0);
  // Per-title confirmation.
  for (const t of OFFERED) assert.match(result.text, new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
});

test("F-041 regression: a genuine save-recipe turn STILL fires the suppression marker (guard not broken)", async () => {
  TOOL_CALLS = [];
  // The incident shape: the model emits BOTH save_generated_recipe AND log_dish_ingredients
  // in one batch. The guard must SUPPRESS the dish log (not execute it) and keep the save.
  const logs = [];
  const origLog = console.log;
  console.log = (...a) => { logs.push(a.join(" ")); };
  try {
    COMPLETION_QUEUE = [
      toolCallMsg([
        { name: "save_generated_recipe", args: { title: "Turkey Fried Rice", ingredients: ["rice"], steps: ["cook"] } },
        { name: "log_dish_ingredients", args: { dish_name: "Turkey Fried Rice", ingredients: ["rice"] } },
      ]),
      textMsg("Saved Turkey Fried Rice to your recipes."),
    ];
    await da.runDeviceAssistant({
      transcript: "Save turkey fried rice to my recipes",
      userContext: { ownerId: "ownerA", userId: "userA" },
      env: { ACTION_MODE: "mock", OPENAI_API_KEY: "test" },
      responseSurface: "app",
    });
  } finally {
    console.log = origLog;
  }
  // The suppression marker MUST still fire (F-041 stays dead).
  assert.ok(
    logs.some((l) => l.includes("assistant_dishlog_suppressed_on_save")),
    "expected assistant_dishlog_suppressed_on_save marker to fire",
  );
  // The save executed; the dish log did NOT (it was suppressed before executeToolAction).
  assert.equal(TOOL_CALLS.filter((c) => c.toolName === "save_generated_recipe").length, 1);
  assert.equal(TOOL_CALLS.filter((c) => c.toolName === "log_dish_ingredients").length, 0);
});
