import test from "node:test";
import assert from "node:assert/strict";
import { scope, key, hash, Fault } from "../src/core.mjs";
import { MemoryStore } from "../src/store.mjs";
import { Service } from "../src/service.mjs";
import { Runner } from "../src/runner.mjs";
import { ToolGateway, toolDefinitions } from "../src/tools.mjs";
import { createRecipe, checkRecipe } from "../src/recipes.mjs";
import { recipeFromText } from "../src/recipe-text.mjs";
import { verifyChange } from "../src/verification.mjs";
import { FixtureGateway, ACTOR } from "./fixtures.mjs";
const pk = scope(ACTOR),
  recipeArgs = {
    title: "Eggs",
    servings: 1,
    ingredients: ["2 eggs"],
    steps: ["Cook until set."],
  };
async function setup() {
  const store = new MemoryStore(),
    gateway = new FixtureGateway(),
    definitions = await toolDefinitions(gateway),
    tools = new ToolGateway({ store, gateway, definitions }),
    service = new Service({ store, gateway });
  return { store, gateway, definitions, tools, service };
}
const message = (r) =>
  r.service.message(ACTOR, { requestId: "first-request", text: "Make eggs" });
function fakeProvider() {
  let sent = false;
  return {
    create: async () => ({ id: "provider-session" }),
    send: async () => {
      sent = true;
    },
    turns: async () =>
      sent
        ? [
            { id: "old-turn", status: "completed" },
            { id: "new-turn", status: "completed" },
          ]
        : [{ id: "old-turn", status: "completed" }],
    session: async () => ({ required_actions: [] }),
    items: async (s, t) => [
      {
        id: "answer-" + t,
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: t }],
      },
    ],
  };
}
test("every turn includes authoritative review state even when no proposal exists", async () => {
  for (const pending of [false, true]) {
    const r = await setup(), s = await message(r);
    if (pending) await r.store.put(pk, key(s.id, "P", "p1"), {
      id: "p1", type: "proposal", action: "save_generated_recipe", status: "pending", title: "Save recipe", validUntil: 123,
    });
    let received;
    const provider = fakeProvider();
    provider.create = async ({ input }) => { received = JSON.parse(input); return { id: "provider-session" }; };
    await new Runner({ ...r, provider }).run(pk, "first-request");
    assert.equal(received.current_review_requests.complete, true);
    assert.equal(received.current_review_requests.items.length, pending ? 1 : 0);
    if (pending) assert.equal(received.current_review_requests.items[0].status, "pending");
    assert.equal(r.gateway.writes, 0);
  }
});
for (const [allergy, line] of [
  ["peanut", "peanut butter mixed with almond milk"],
  ["soy", "1 cup soy milk"],
  ["tree nuts", "almond milk"],
  ["dairy", "milk mixed with soy milk"],
])
  test(`allergen cannot hide in plant milk: ${allergy}`, () => {
    assert.throws(
      () =>
        checkRecipe(createRecipe({ ...recipeArgs, ingredients: [line] }, "r"), {
          allergies: [allergy],
        }),
      { code: "dietary_conflict" },
    );
  });
test("plant milk alone is not dairy", () =>
  assert.doesNotThrow(() =>
    checkRecipe(
      createRecipe({ ...recipeArgs, ingredients: ["1 cup oat milk"] }, "r"),
      { allergies: ["dairy"] },
    ),
  ));
test("new request never accepts an older completed turn", async () => {
  const r = await setup(),
    s = await message(r),
    meta = await r.store.get(pk, key(s.id));
  await r.store.put(
    pk,
    meta.sk,
    { ...meta, providerId: "provider-session", lastTurnId: "older-turn" },
    meta.version,
  );
  await new Runner({ ...r, provider: fakeProvider() }).run(pk, "first-request");
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(out.status, "completed");
  assert.equal(out.messages.at(-1).text, "new-turn");
});
test("revoked enrollment is terminal and cannot poison stream", async () => {
  const r = await setup(),
    s = await message(r);
  r.gateway.actorValue = { ...ACTOR, actor: "revoked" };
  await new Runner({ ...r, provider: {} }).run(pk, "first-request");
  assert.equal((await r.store.get(pk, "Q#first-request")).status, "failed");
  assert.equal((await r.store.get(pk, key(s.id))).activeRequest, null);
});
test("changed household is terminal without tool execution", async () => {
  const r = await setup();
  await message(r);
  r.gateway.actorValue = { ...ACTOR, household: "new-home" };
  await new Runner({ ...r, provider: {} }).run(pk, "first-request");
  assert.equal((await r.store.get(pk, "Q#first-request")).status, "failed");
  assert.equal(r.gateway.writes, 0);
});
test("recipe is hidden immediately after dietary settings change", async () => {
  const r = await setup(),
    s = await message(r);
  await r.tools.call({
    actor: ACTOR,
    session: s,
    action: { name: "create_recipe", arguments: recipeArgs, call_id: "recipe" },
  });
  assert.equal((await r.service.session(ACTOR, s.id)).recipes.length, 1);
  r.gateway.data.preferences.dietary.allergies.push("egg");
  assert.equal((await r.service.session(ACTOR, s.id)).recipes.length, 0);
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "get_conversation_recipes",
      arguments: {},
      call_id: "read-new",
    },
  });
  assert.equal(out.recipes.length, 0);
});
async function approved(r) {
  const s = await message(r),
    meta = await r.store.get(pk, key(s.id));
  await r.store.put(
    pk,
    meta.sk,
    { ...meta, activeRequest: null, status: "completed" },
    meta.version,
  );
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "request_add_to_shopping_list",
      arguments: { item_name: "Milk", quantity: "2" },
      call_id: "add",
    },
  });
  await r.service.decide(ACTOR, {
    sessionId: s.id,
    proposalId: out.proposal_id,
    decision: "approve",
    requestId: "approve-request",
  });
  return { s, p: await r.store.get(pk, key(s.id, "P", out.proposal_id)) };
}
test("crash after successful write resumes verification without another write", async () => {
  const r = await setup(),
    { s, p } = await approved(r);
  const result = await r.gateway.mutate(ACTOR, p.action, p.args, p.operationId);
  await r.store.put(pk, p.sk, { ...p, status: "verifying", result }, p.version);
  await new Runner({ ...r, provider: {} }).run(pk, "approve-request");
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(out.status, "completed");
  assert.equal(out.proposals[0].status, "applied");
  assert.equal(r.gateway.writes, 1);
});
test("readback outage recovers through Check result without replay", async () => {
  const r = await setup(),
    { s } = await approved(r);
  const original = r.gateway.mutate.bind(r.gateway);
  r.gateway.mutate = async (...args) => {
    const out = await original(...args);
    r.gateway.failRead = "shopping";
    return out;
  };
  const runner = new Runner({ ...r, provider: {} });
  await runner.run(pk, "approve-request");
  assert.equal(
    (await r.store.get(pk, key(s.id))).status,
    "needs_reconciliation",
  );
  r.gateway.failRead = null;
  const p = (await r.service.session(ACTOR, s.id)).proposals[0];
  await r.service.resume(ACTOR, { sessionId: s.id, proposalId: p.id });
  await runner.run(pk, "approve-request");
  assert.equal(
    (await r.service.session(ACTOR, s.id)).proposals[0].status,
    "applied",
  );
  assert.equal(r.gateway.writes, 1);
});
test("partial duplicate-name additions cannot be falsely confirmed", () => {
  assert.equal(
    verifyChange(
      "add_many_to_shopping_list",
      {
        items: [
          { item_name: "Milk", quantity: "2" },
          { item_name: "Milk", quantity: "3" },
        ],
      },
      [],
      [{ id: "milk", item_name: "Milk", quantity: "1" }],
      {},
    ).verified,
    false,
  );
});
test("exact duplicate-name additions verify with distinct identities and quantities", () => {
  assert.equal(
    verifyChange(
      "add_many_to_shopping_list",
      {
        items: [
          { item_name: "Milk", quantity: "2" },
          { item_name: "Milk", quantity: "3" },
        ],
      },
      [],
      [
        { shopping_id: "a", item_name: "Milk", quantity: "2" },
        { shopping_id: "b", item_name: "Milk", quantity: "3" },
      ],
      {},
    ).verified,
    true,
  );
});
test("actual legacy bought/unbought states verify", () => {
  for (const [action, state] of [
    ["mark_shopping_item_bought", "2"],
    ["mark_shopping_item_unbought", "1"],
  ])
    assert.equal(
      verifyChange(
        action,
        { item_name: "Milk" },
        [],
        [{ shopping_id: "a", item_name: "Milk", action: state }],
        {},
      ).verified,
      true,
    );
});
test("removed indirect calendar action cannot report success", () =>
  assert.equal(
    verifyChange(
      "add_many_to_meal_calendar",
      {},
      [],
      [{ id: "a", plan_date: "2026-10-02" }],
      { toolResult: { results: [{ entry_id: "a", ok: true }] } },
    ).verified,
    false,
  ));
test("clear all is unavailable until target fencing is implemented", async () => {
  const r = await setup();
  assert.equal(
    r.definitions.some(
      (t) =>
        t.name === "request_clear_kitchen_inventory" ||
        t.name === "request_clear_shopping_list",
    ),
    false,
  );
});
test("timeout exhaustion preserves resume identity and blocks new messages", async () => {
  const r = await setup(),
    s = await message(r),
    req = await r.store.get(pk, "Q#first-request");
  await r.store.put(pk, req.sk, { ...req, attempts: 2 }, req.version);
  const provider = {
    create: async () => {
      throw new Fault("provider_error", "Connection lost.", 502);
    },
  };
  await new Runner({ ...r, provider }).run(pk, req.id);
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(out.status, "interrupted");
  await assert.rejects(
    r.service.message(ACTOR, {
      sessionId: s.id,
      requestId: "new-request",
      text: "Another",
    }),
    { code: "busy" },
  );
  await r.service.resume(ACTOR, { sessionId: s.id });
  assert.equal((await r.store.get(pk, req.sk)).status, "queued");
  assert.equal((await r.store.get(pk, key(s.id))).activeRequest, req.id);
});

test("check-in verification rejects wrong opened state and storage", () => {
  const args = { item_name: "Milk", location: "fridge", is_opened: false };
  assert.equal(
    verifyChange(
      "check_in_item",
      args,
      [],
      [
        {
          id: "milk",
          item_name: "Milk",
          storage_location: "pantry",
          is_opened: true,
        },
      ],
      {},
    ).verified,
    false,
  );
  assert.equal(
    verifyChange(
      "check_in_item",
      args,
      [],
      [
        {
          id: "milk",
          item_name: "Milk",
          storage_location: "fridge",
          is_opened: false,
        },
      ],
      {},
    ).verified,
    true,
  );
});
test("calendar move checks requested slot too", () => {
  const args = {
    entry_id: "a",
    new_date: "2026-10-02",
    new_meal_slot: "lunch",
  };
  const res = { toolResult: { entry: { id: "a" } } };
  assert.equal(
    verifyChange(
      "move_meal_calendar_entry",
      args,
      [],
      [{ id: "a", plan_date: args.new_date, meal_slot: "dinner" }],
      res,
    ).verified,
    false,
  );
  assert.equal(
    verifyChange(
      "move_meal_calendar_entry",
      args,
      [],
      [{ id: "a", plan_date: args.new_date, meal_slot: "lunch" }],
      res,
    ).verified,
    true,
  );
});
test("inline calendar checks every ingredient and distinct entry", () => {
  const entry = {
    title: "Eggs",
    plan_date: "2026-10-02",
    meal_slot: "breakfast",
    ingredients: ["2 eggs"],
    instructions: ["Cook eggs"],
    notes: [],
  };
  const row = { ...entry, id: "a" };
  assert.equal(
    verifyChange(
      "add_generated_recipes_to_meal_calendar",
      { entries: [entry] },
      [],
      [row],
      {},
    ).verified,
    true,
  );
  assert.equal(
    verifyChange(
      "add_generated_recipes_to_meal_calendar",
      { entries: [entry] },
      [],
      [{ ...row, ingredients: ["1 egg"] }],
      {},
    ).verified,
    false,
  );
  assert.equal(
    verifyChange(
      "add_generated_recipes_to_meal_calendar",
      { entries: [entry, entry] },
      [],
      [row],
      {},
    ).verified,
    false,
  );
});
test("unscoped legacy dish tools and mutable source shortcuts are not exposed", async () => {
  const r = await setup();
  for (const name of [
    "mark_dish_consumed",
    "delete_dish_log",
    "append_to_recent_dish",
    "add_recipe_to_meal_calendar",
    "add_many_to_meal_calendar",
    "add_saved_recipe_ingredients_to_shopping_list",
    "refresh_meal_plan",
  ])
    assert.equal(
      r.definitions.some((t) => t.name === "request_" + name),
      false,
    );
  assert.equal(
    verifyChange(
      "mark_dish_consumed",
      { dish_id: "a" },
      [{ id: "a", action: "IN" }],
      [{ id: "a", action: "IN", updated_at: 1 }],
      { toolResult: { dish: { id: "a" } } },
    ).verified,
    false,
  );
});
test("old reconciliation cannot replace active work or reset its lease", async () => {
  const r = await setup(),
    s = await message(r);
  await r.store.put(pk, key(s.id, "P", "p1"), {
    id: "p1",
    status: "needs_reconciliation",
    operationId: "old-request",
  });
  await r.store.put(pk, "Q#old-request", {
    id: "old-request",
    sessionId: s.id,
    status: "needs_reconciliation",
    leaseUntil: 50,
  });
  await assert.rejects(
    r.service.resume(ACTOR, { sessionId: s.id, proposalId: "p1" }),
    { code: "busy" },
  );
  assert.equal(
    (await r.store.get(pk, key(s.id))).activeRequest,
    "first-request",
  );
  const old = await r.store.get(pk, key(s.id));
  await r.store.put(
    pk,
    key(s.id),
    { ...old, activeRequest: "old-request" },
    old.version,
  );
  const q = await r.store.get(pk, "Q#old-request");
  await r.store.put(
    pk,
    q.sk,
    { ...q, status: "running", leaseUntil: 9999 },
    q.version,
  );
  await r.service.resume(ACTOR, { sessionId: s.id, proposalId: "p1" });
  assert.equal((await r.store.get(pk, q.sk)).leaseUntil, 9999);
});
test("proposal freezes exact owned kitchen ID without mutating tool-call identity", async () => {
  const r = await setup(),
    s = await message(r),
    action = {
      call_id: "qty",
      name: "request_update_item_quantity",
      arguments: { item_name: "Eggs", quantity_value: 4 },
    };
  await r.tools.call({ actor: ACTOR, session: s, action });
  const proposals = await r.store.list(pk, key(s.id, "P") + "#");
  assert.equal(proposals[0].args.item_id, "eggs");
  assert.equal(action.arguments.item_id, undefined);
  assert.equal(
    (await r.tools.call({ actor: ACTOR, session: s, action })).ok,
    true,
  );
  const foreign = {
    ...action,
    call_id: "foreign",
    arguments: {
      item_name: "Eggs",
      item_id: "foreign-household-row",
      quantity_value: 4,
    },
  };
  assert.equal(
    (await r.tools.call({ actor: ACTOR, session: s, action: foreign })).error,
    "ambiguous_item",
  );
});
test("daily budget is durable and does not count idempotent retries twice", async () => {
  const r = await setup(),
    s = await message(r);
  await message(r);
  const budgets = await r.store.list(pk, "B#");
  assert.equal(budgets[0].count, 1);
  const b = budgets[0];
  await r.store.put(pk, b.sk, { ...b, count: 100 }, b.version);
  await assert.rejects(
    r.service.message(ACTOR, {
      requestId: "over-budget",
      text: "another question",
    }),
    { code: "daily_limit" },
  );
});

test("kitchen freeform quantities verify through the actual legacy field", () => {
  assert.equal(
    verifyChange(
      "check_in_item",
      { item_name: "Milk", quantity: "2 cartons" },
      [],
      [{ id: "a", item_name: "Milk", remaining_quantity: "2 cartons" }],
      {},
    ).verified,
    true,
  );
  assert.equal(
    verifyChange(
      "check_in_item",
      { item_name: "Milk", quantity: "2 cartons" },
      [],
      [{ id: "a", item_name: "Milk", remaining_quantity: "1 carton" }],
      {},
    ).verified,
    false,
  );
});

test("shopping proposals reject fuzzy and duplicate targets", async () => {
  const r = await setup(),
    s = await message(r);
  r.gateway.data.shopping = [{ id: "oat", item_name: "Oat Milk", action: "1" }];
  const action = {
    call_id: "fuzzy",
    name: "request_mark_shopping_item_bought",
    arguments: { item_name: "milk" },
  };
  assert.equal(
    (await r.tools.call({ actor: ACTOR, session: s, action })).error,
    "ambiguous_item",
  );
  action.call_id = "exact";
  action.arguments.item_name = "Oat Milk";
  assert.equal(
    (await r.tools.call({ actor: ACTOR, session: s, action })).ok,
    true,
  );
  r.gateway.data.shopping.push({
    id: "second",
    item_name: "Oat Milk",
    action: "1",
  });
  action.call_id = "duplicate";
  assert.equal(
    (await r.tools.call({ actor: ACTOR, session: s, action })).error,
    "ambiguous_item",
  );
});

const plainRecipe =
  "## Eggs (serves 2)\n\nIngredients\n- 4 eggs\n- 1 tsp oil\n\nMethod\n1. Heat oil.\n2. Cook eggs until set.\n\nMissing: check oil.";
test("chat-only recipe becomes an exact canonical card with portions", async () => {
  const r = await setup(),
    s = await message(r),
    runner = new Runner(r);
  await runner.saveMessages(
    pk,
    s,
    { createdAt: 1 },
    [
      {
        id: "plain",
        type: "message",
        role: "assistant",
        phase: "final_answer",
        content: [{ type: "output_text", text: plainRecipe }],
      },
    ],
    r.gateway.data.preferences.dietary,
  );
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(out.recipes.length, 1);
  assert.equal(out.recipes[0].servings, 2);
  assert.deepEqual(
    out.recipes[0].ingredients.map((x) => x.text),
    ["4 eggs", "1 tsp oil"],
  );
  assert.ok(out.messages.filter(x=>x.role==="assistant").at(-1).text.includes("Missing: check oil."));
  assert.equal(r.gateway.writes, 0);
});
test("later untracked recipe rewrite cannot contradict the canonical card", async () => {
  const r = await setup(),
    s = await message(r),
    runner = new Runner(r);
  const item = (text) => ({
    id: hash(text),
    type: "message",
    role: "assistant",
    phase: "final_answer",
    content: [{ type: "output_text", text }],
  });
  await runner.saveMessages(
    pk,
    s,
    { createdAt: 1 },
    [item(plainRecipe)],
    r.gateway.data.preferences.dietary,
  );
  await runner.saveMessages(
    pk,
    s,
    { createdAt: 2 },
    [item(plainRecipe.replace("4 eggs", "9 eggs"))],
    r.gateway.data.preferences.dietary,
  );
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(out.recipes[0].ingredients[0].text, "4 eggs");
  assert.ok(
    out.messages.filter(x=>x.role==="assistant").at(-1).text.includes("kept the recipe card unchanged"),
  );
});
test("recipe text rescue never guesses servings or missing method", () => {
  assert.equal(recipeFromText(plainRecipe.replace(" (serves 2)", "")), null);
  assert.equal(
    recipeFromText(
      plainRecipe.replace(
        "1. Heat oil.\n2. Cook eggs until set.",
        "Just cook.",
      ),
    ),
    null,
  );
});

test("recipe text rescue refuses to truncate wrapped method lines",()=>{
 assert.equal(recipeFromText(plainRecipe.replace("1. Heat oil.","1. Heat oil.\n   Then lower the heat.")),null);
});
