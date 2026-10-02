import { DynamoStore } from "./store.mjs";
import { TrepoGateway } from "./gateway.mjs";
import { AgentProvider } from "./provider.mjs";
import { ToolGateway, toolDefinitions } from "./tools.mjs";
import { Service } from "./service.mjs";
import { Runner } from "./runner.mjs";
import { Fault, verifyRequest, publicError, hash, scope } from "./core.mjs";
let runtime;
async function getRuntime() {
  if (runtime) return runtime;
  const store = new DynamoStore(process.env.STATE_TABLE),
    gateway = new TrepoGateway();
  await gateway.init();
  const definitions = await toolDefinitions(gateway),
    tools = new ToolGateway({ store, gateway, definitions }),
    provider = new AgentProvider(process.env.OPENAI_API_KEY),
    service = new Service({ store, gateway, provider }),
    runner = new Runner({ store, gateway, provider, definitions, tools });
  return (runtime = { store, gateway, service, runner });
}
export async function handler(event) {
  if (event.Records?.[0]?.eventSource === "aws:dynamodb") {
    const rt = await getRuntime(),
      failures = [];
    for (const r of event.Records) {
      const x = r.dynamodb?.NewImage;
      if (x?.type?.S !== "request" || x.status?.S !== "queued") continue;
      try {
        await rt.runner.run(x.pk.S, x.id.S);
      } catch (e) {
        console.warn(
          JSON.stringify({
            event: "thyme_worker_retry",
            code: e.code || e.name,
          }),
        );
        failures.push({ itemIdentifier: r.dynamodb.SequenceNumber });
        break;
      }
    }
    return { batchItemFailures: failures };
  }
  try {
    const { data, nonce } = verifyRequest(event, process.env.BRIDGE_SECRET),
      rt = await getRuntime();
    const map = JSON.parse(process.env.PRINCIPAL_MAP || "{}"),
      actor = map[data.principal];
    if (!actor)
      throw new Fault(
        "not_enrolled",
        "This ChatGPT account is not connected to the pilot.",
        403,
      );
    if (
      Object.hasOwn(data, "actor") ||
      Object.hasOwn(data, "owner") ||
      Object.hasOwn(data, "household")
    )
      throw new Fault(
        "scope_override",
        "The household is chosen by your authenticated account.",
        403,
      );
    if (
      data.principal === "qualification-read-only" &&
      ["decide", "resume"].includes(data.op)
    )
      throw new Fault(
        "read_only",
        "Qualification cannot apply account changes.",
        403,
      );
    const a = await rt.gateway.actor(actor);
    try {
      await rt.store.put("NONCE", nonce, {
        type: "nonce",
        expires: Math.floor(Date.now() / 1000) + 180,
      });
    } catch (e) {
      if (e.code === "conflict")
        throw new Fault(
          "replayed",
          "This connection request was already used.",
          409,
        );
      throw e;
    }
    const result = await rt.service.dispatch(a, data);
    return response(200, result);
  } catch (e) {
    console.warn(
      JSON.stringify({ event: "thyme_api_error", code: e.code || e.name }),
    );
    return response(e.status || 503, publicError(e));
  }
}
function response(statusCode, body) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
    body: JSON.stringify(body),
  };
}
