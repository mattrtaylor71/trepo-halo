import fs from "node:fs";
import { MemoryStore } from "../src/store.mjs";
import { Service } from "../src/service.mjs";
import { Runner } from "../src/runner.mjs";
import { ToolGateway, toolDefinitions } from "../src/tools.mjs";
import { AgentProvider } from "../src/provider.mjs";
import { scope, id } from "../src/core.mjs";
import { FixtureGateway, ACTOR } from "../test/fixtures.mjs";
const cfg = JSON.parse(fs.readFileSync(process.env.THYME_PRIVATE_CONFIG));
const apiKey = cfg.Environment.Variables.OPENAI_API_KEY;
const models = process.argv.slice(2).length
  ? process.argv.slice(2)
  : ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna"];
const output = process.env.THYME_EVIDENCE;
const all = [];
for (const model of models) {
  const store = new MemoryStore(),
    gateway = new FixtureGateway(),
    definitions = await toolDefinitions(gateway),
    tools = new ToolGateway({ store, gateway, definitions }),
    provider = new AgentProvider(apiKey),
    service = new Service({ store, gateway }),
    runner = new Runner({ store, gateway, definitions, tools, provider });
  let sid;
  for (const [label, text] of [
    [
      "grounded_recipe",
      "Make me a quick breakfast using what I actually have. Give me one complete recipe for two people. Do not add anything to my shopping list.",
    ],
    [
      "targeted_edit",
      "For that same recipe, swap the spinach for kale, and leave everything else exactly the same. I know kale is missing from my kitchen; that is fine.",
    ],
    [
      "recipe_question",
      "How long should I cook the eggs? Just explain; do not change the recipe.",
    ],
    ["list_proposal", "Add milk to my shopping list."],
  ]) {
    const rid = id(),
      started = Date.now();
    try {
      const s = await service.message(ACTOR, {
        requestId: rid,
        text,
        model,
        sessionId: sid,
      });
      sid = s.id;
      await runner.run(scope(ACTOR), rid);
      const result = await service.session(ACTOR, sid);
      const records = await store.list(scope(ACTOR), "S#" + sid + "#T#");
      const record = {
        model,
        label,
        seconds: (Date.now() - started) / 1000,
        status: result.status,
        error: result.error,
        metrics: result.metrics,
        answer: result.messages
          .filter((m) => m.role === "assistant" && m.phase !== "commentary")
          .at(-1)?.text,
        recipes: result.recipes,
        proposals: result.proposals,
        tools: records.map((r) => ({ intent: r.intent, result: r.result })),
        writes: gateway.writes,
      };
      all.push(record);
      fs.writeFileSync(output, JSON.stringify(all, null, 2));
      console.log(
        JSON.stringify({
          model,
          label,
          seconds: record.seconds,
          status: record.status,
          recipes: result.recipes.length,
          proposals: result.proposals.length,
          writes: gateway.writes,
          error: result.error,
        }),
      );
      if (result.status === "failed") break;
    } catch (e) {
      all.push({
        model,
        label,
        error: e.message,
        providerStatus: e.providerStatus,
        providerDetail: e.providerDetail,
      });
      fs.writeFileSync(output, JSON.stringify(all, null, 2));
      console.log(
        JSON.stringify({
          model,
          label,
          error: e.message,
          providerStatus: e.providerStatus,
          providerDetail: e.providerDetail,
        }),
      );
      break;
    }
  }
}
