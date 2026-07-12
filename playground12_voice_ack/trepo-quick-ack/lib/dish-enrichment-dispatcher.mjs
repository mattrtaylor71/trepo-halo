import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";

let lambdaClient = null;

function getLambdaClient(env = process.env) {
  if (!lambdaClient) {
    lambdaClient = new LambdaClient({
      region: env.AWS_REGION || env.AWS_DEFAULT_REGION || "us-east-1"
    });
  }

  return lambdaClient;
}

export function shouldUseAsyncDishEnrichment(env = process.env) {
  return env?.DISH_ENRICHMENT_MODE === "hybrid" && typeof env?.DISH_EDIT_FUNCTION_NAME === "string" && env.DISH_EDIT_FUNCTION_NAME.trim().length > 0;
}

export async function dispatchDishEnrichmentJob({ ownerId, dishId, payload, env = process.env }) {
  if (!shouldUseAsyncDishEnrichment(env)) {
    return {
      queued: false
    };
  }

  const safeOwnerId = String(ownerId || "").trim();
  const safeDishId = String(dishId || "").trim();
  if (!safeOwnerId || !safeDishId) {
    throw new Error("Missing ownerId or dishId for async dish enrichment.");
  }

  const eventPayload = {
    rawPath: `/dishes/${safeOwnerId}/${safeDishId}/edit`,
    pathParameters: {
      owner: safeOwnerId,
      item_id: safeDishId
    },
    requestContext: {
      http: {
        method: "POST",
        path: `/dishes/${safeOwnerId}/${safeDishId}/edit`
      }
    },
    body: JSON.stringify(payload || {})
  };

  await getLambdaClient(env).send(new InvokeCommand({
    FunctionName: env.DISH_EDIT_FUNCTION_NAME,
    InvocationType: "Event",
    Payload: Buffer.from(JSON.stringify(eventPayload))
  }));

  return {
    queued: true,
    function_name: env.DISH_EDIT_FUNCTION_NAME,
    owner_id: safeOwnerId,
    dish_id: safeDishId
  };
}
