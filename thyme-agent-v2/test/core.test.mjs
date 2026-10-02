import test from "node:test";
import assert from "node:assert/strict";
import {
  hash,
  scope,
  key,
  signRequest,
  verifyRequest,
  legacyResponse,
} from "../src/core.mjs";
import { MemoryStore } from "../src/store.mjs";
import { Service } from "../src/service.mjs";
import { Runner } from "../src/runner.mjs";
import { ToolGateway, toolDefinitions } from "../src/tools.mjs";
import { createRecipe, patchRecipe, checkRecipe } from "../src/recipes.mjs";
import { verifyChange } from "../src/verification.mjs";
import { FixtureGateway, ACTOR } from "./fixtures.mjs";
import { AgentProvider } from "../src/provider.mjs";
const recipeArgs = {
  title: "Spinach eggs",
  servings: 2,
  ingredients: ["4 eggs", "100 g spinach", "1 tsp olive oil"],
  steps: ["Wilt spinach for 2 minutes.", "Add eggs and cook until set."],
};
async function setup() {
  const store = new MemoryStore(),
    gateway = new FixtureGateway(),
    definitions = await toolDefinitions(gateway),
    tools = new ToolGateway({ store, gateway, definitions }),
    service = new Service({ store, gateway });
  return { store, gateway, definitions, tools, service };
}
const message = (service, input = {}) =>
  service.message(ACTOR, {
    requestId: "request-12345",
    text: "Hello",
    ...input,
  });
async function proposal(
  r,
  action = "request_add_to_shopping_list",
  args = { item_name: "Milk" },
) {
  let s = await message(r.service);
  let meta = await r.store.get(scope(ACTOR), key(s.id));
  await r.store.put(
    scope(ACTOR),
    key(s.id),
    { ...meta, activeRequest: null, status: "completed" },
    meta.version,
  );
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: { name: action, arguments: args, call_id: "call-12345" },
  });
  return { s, pid: out.proposal_id };
}
test("HMAC binds method, path, timestamp, nonce and exact body", () => {
  const secret = "private-test-secret",
    body = '{"op":"bootstrap"}',
    timestamp = String(Date.now()),
    nonce = "nonce-123456789012345";
  const signature = signRequest({ body, timestamp, nonce }, secret);
  const e = {
    body,
    headers: {
      "x-thyme-time": timestamp,
      "x-thyme-nonce": nonce,
      "x-thyme-signature": signature,
    },
  };
  assert.equal(verifyRequest(e, secret).data.op, "bootstrap");
  for (const change of [
    { body: "{}" },
    { rawPath: "/other" },
    { requestContext: { http: { method: "GET" } } },
  ])
    assert.throws(() => verifyRequest({ ...e, ...change }, secret));
  assert.throws(() => verifyRequest(e, secret, Date.now() + 120000));
});
test("conditional transaction is atomic", async () => {
  const s = new MemoryStore();
  await s.put("p", "a", { n: 1 });
  await assert.rejects(
    s.transaction([
      { item: { pk: "p", sk: "b" }, expected: null },
      { item: { pk: "p", sk: "a" }, expected: null },
    ]),
  );
  assert.equal(await s.get("p", "b"), null);
});
test("request deduplication returns same session and rejects changed intent", async () => {
  const { service } = await setup();
  const first = await message(service),
    again = await message(service);
  assert.equal(first.id, again.id);
  await assert.rejects(message(service, { text: "Different" }), {
    code: "request_conflict",
  });
});
test("one active request per conversation", async () => {
  const { service } = await setup();
  const s = await message(service);
  await assert.rejects(
    message(service, { sessionId: s.id, requestId: "different-request" }),
    { code: "busy" },
  );
});
test("cross-account and cross-household sessions cannot be retrieved", async () => {
  const { service } = await setup(),
    s = await message(service);
  for (const actor of [
    { ...ACTOR, actor: "someone-else" },
    { ...ACTOR, household: "other-home" },
  ])
    await assert.rejects(service.session(actor, s.id), { code: "not_found" });
});
test("model-selected unknown properties cannot cross the tool boundary", async () => {
  const r = await setup(),
    s = await message(r.service);
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "request_add_to_shopping_list",
      arguments: { item_name: "Milk", owner: "other" },
      call_id: "bad",
    },
  });
  assert.equal(out.ok, false);
  assert.equal(r.gateway.writes, 0);
});
test("write tools only create reviewable proposals", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  assert.equal(r.gateway.writes, 0);
  const p = (await r.service.session(ACTOR, s.id)).proposals[0];
  assert.equal(p.id, pid);
  assert.equal(p.status, "pending");
  assert.equal(p.args.item_name, "Milk");
});
test("duplicate tool calls reuse the receipt and proposal", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "request_add_to_shopping_list",
      arguments: { item_name: "Milk" },
      call_id: "call-12345",
    },
  });
  assert.equal(out.proposal_id, pid);
  assert.equal((await r.service.session(ACTOR, s.id)).proposals.length, 1);
});
test("tool call IDs cannot change meaning", async () => {
  const r = await setup(),
    { s } = await proposal(r);
  await assert.rejects(
    r.tools.call({
      actor: ACTOR,
      session: s,
      action: {
        name: "request_add_to_shopping_list",
        arguments: { item_name: "Bread" },
        call_id: "call-12345",
      },
    }),
    { code: "tool_conflict" },
  );
});
test("reject does not mutate data", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  const out = await r.service.decide(ACTOR, {
    sessionId: s.id,
    proposalId: pid,
    decision: "reject",
    requestId: "reject-123",
  });
  assert.equal(out.proposals[0].status, "rejected");
  assert.equal(r.gateway.writes, 0);
});
test("stale proposals cannot be approved", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  r.gateway.data.shopping.push({ id: "new", item_name: "Bread" });
  await assert.rejects(
    r.service.decide(ACTOR, {
      sessionId: s.id,
      proposalId: pid,
      decision: "approve",
      requestId: "approve-123",
    }),
    { code: "stale_approval" },
  );
  assert.equal(r.gateway.writes, 0);
});
test("expired proposals cannot be approved", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  const p = await r.store.get(scope(ACTOR), key(s.id, "P", pid));
  await r.store.put(p.pk, p.sk, { ...p, validUntil: 0 }, p.version);
  await assert.rejects(
    r.service.decide(ACTOR, {
      sessionId: s.id,
      proposalId: pid,
      decision: "approve",
      requestId: "approve-123",
    }),
    { code: "approval_expired" },
  );
});
test("approved mutation verifies and duplicate delivery never repeats it", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  await r.service.decide(ACTOR, {
    sessionId: s.id,
    proposalId: pid,
    decision: "approve",
    requestId: "approve-123",
  });
  const runner = new Runner({ ...r, provider: {} });
  await runner.run(scope(ACTOR), "approve-123");
  await runner.run(scope(ACTOR), "approve-123");
  const out = await r.service.session(ACTOR, s.id);
  assert.equal(r.gateway.writes, 1);
  assert.equal(out.status, "completed");
  assert.equal(out.proposals[0].status, "applied");
  assert.equal(r.gateway.data.shopping.at(-1).item_name, "Milk");
});
test("uncertain mutation is never automatically repeated", async () => {
  const r = await setup(),
    { s, pid } = await proposal(r);
  r.gateway.failWrite = true;
  await r.service.decide(ACTOR, {
    sessionId: s.id,
    proposalId: pid,
    decision: "approve",
    requestId: "approve-123",
  });
  const runner = new Runner({ ...r, provider: {} });
  await runner.run(scope(ACTOR), "approve-123");
  await runner.run(scope(ACTOR), "approve-123");
  assert.equal(r.gateway.writes, 1);
  assert.equal(
    (await r.service.session(ACTOR, s.id)).status,
    "needs_reconciliation",
  );
});
test("unrelated change is not verification of a requested mutation", () => {
  const before = [{ id: "a", item_name: "Eggs", quantity_value: 6 }],
    after = [{ ...before[0], storage_location: "pantry" }];
  assert.equal(
    verifyChange(
      "update_item_quantity",
      { item_name: "Eggs", quantity_value: 2 },
      before,
      after,
      { ok: true },
    ).verified,
    false,
  );
});
test("adding another item does not prove milk was added", () => {
  assert.equal(
    verifyChange(
      "add_to_shopping_list",
      { item_name: "Milk" },
      [],
      [{ id: "x", item_name: "Bread" }],
      { ok: true },
    ).verified,
    false,
  );
});
test("recipe line edits preserve all unrelated content", () => {
  const r = createRecipe(recipeArgs, "recipe1"),
    changed = patchRecipe(r, {
      expected_revision: 1,
      changes: [
        { kind: "replace_ingredient", id: "ingredient-2", text: "100 g kale" },
      ],
    });
  assert.equal(changed.servings, 2);
  assert.deepEqual(changed.steps, r.steps);
  assert.equal(changed.ingredients[0].text, r.ingredients[0].text);
  assert.equal(changed.ingredients[2].text, r.ingredients[2].text);
  assert.equal(changed.revision, 2);
});
test("stale recipe revision fails", () => {
  assert.throws(
    () =>
      patchRecipe(createRecipe(recipeArgs, "r"), {
        expected_revision: 0,
        changes: [{ kind: "title", text: "Other" }],
      }),
    { code: "recipe_conflict" },
  );
});
test("unknown recipe line fails without mutating original", () => {
  const r = createRecipe(recipeArgs, "r");
  assert.throws(() =>
    patchRecipe(r, {
      expected_revision: 1,
      changes: [{ kind: "remove_ingredient", id: "wrong" }],
    }),
  );
  assert.equal(r.ingredients.length, 3);
});
test("servings cannot silently change while quantities stay unchanged", () => {
  assert.throws(() =>
    patchRecipe(createRecipe(recipeArgs, "r"), {
      expected_revision: 1,
      changes: [{ kind: "servings", servings: 4 }],
    }),
  );
});
test("allergy conflict is caught before rendering a recipe", () => {
  const r = createRecipe(
    { ...recipeArgs, ingredients: ["2 tbsp peanut butter"] },
    "r",
  );
  assert.throws(() => checkRecipe(r, { allergies: ["peanut"] }), {
    code: "dietary_conflict",
  });
});
test("recipe edits never call inventory mutations", async () => {
  const r = await setup(),
    s = await message(r.service),
    out = await r.tools.call({
      actor: ACTOR,
      session: s,
      action: {
        name: "create_recipe",
        arguments: recipeArgs,
        call_id: "create",
      },
    });
  await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "edit_recipe",
      arguments: {
        recipe_id: out.recipe.id,
        expected_revision: 1,
        changes: [{ kind: "remove_ingredient", id: "ingredient-2" }],
      },
      call_id: "edit",
    },
  });
  assert.equal(r.gateway.writes, 0);
  assert.equal(r.gateway.data.kitchen.length, 3);
});
test("save cannot silently regenerate recipe content", async () => {
  const r = await setup(),
    s = await message(r.service);
  await r.tools.call({
    actor: ACTOR,
    session: s,
    action: { name: "create_recipe", arguments: recipeArgs, call_id: "create" },
  });
  const out = await r.tools.call({
    actor: ACTOR,
    session: s,
    action: {
      name: "request_save_generated_recipe",
      arguments: {
        title: recipeArgs.title,
        ingredients: ["100 eggs"],
        steps: recipeArgs.steps,
      },
      call_id: "save",
    },
  });
  assert.equal(out.error, "recipe_changed");
});
test("provider handles empty successful tool-result responses", async () => {
  const p = new AgentProvider("test", {
    fetcher: async () => new Response(null, { status: 204 }),
  });
  assert.deepEqual(
    await p.result("s", { turn_id: "t", call_id: "c" }, { ok: true }),
    {},
  );
});
test("provider paginates all saved items", async () => {
  let calls = 0;
  const p = new AgentProvider("test", {
    fetcher: async () =>
      new Response(
        JSON.stringify(
          ++calls === 1
            ? { data: [{ id: "a" }], has_more: true, last_id: "a" }
            : { data: [{ id: "b" }], has_more: false },
        ),
      ),
  });
  assert.deepEqual(await p.items("s"), [{ id: "a" }, { id: "b" }]);
  assert.equal(calls, 2);
});
test("malformed pagination fails rather than dropping output", async () => {
  const p = new AgentProvider("test", {
    fetcher: async () =>
      new Response(JSON.stringify({ data: [], has_more: true })),
  });
  await assert.rejects(p.items("s"), { code: "pagination" });
});
test("provider turn filter is enforced locally across every page", async () => {
  let calls = 0;
  const p = new AgentProvider("test", {
    fetcher: async () => new Response(JSON.stringify(++calls === 1
      ? {data: [{id:"old", turn_id:"older-turn"}], has_more:true, last_id:"old"}
      : {data: [{id:"new", turn_id:"current-turn"}, {id:"unknown"}], has_more:false})),
  });
  assert.deepEqual(await p.items("s", "current-turn"), [{id:"new",turn_id:"current-turn"}]);
  assert.equal(calls, 2);
});
test("old response envelope retains text and recipes", () => {
  const s = {
    id: "s",
    status: "completed",
    messages: [{ role: "assistant", text: "Ready." }],
    recipes: [{ title: "Eggs" }],
  };
  assert.equal(legacyResponse(s).response, "Ready.");
  assert.equal(legacyResponse(s).recipes[0].title, "Eggs");
});
