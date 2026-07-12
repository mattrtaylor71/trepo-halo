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

function pushUniqueJob(jobs, seen, job) {
  const key = `${job.assetType}:${job.rowId}`;
  if (!job.rowId || seen.has(key)) {
    return;
  }
  seen.add(key);
  jobs.push(job);
}

function getKitchenRowId(item) {
  return String(item?.item_id || item?.id || "").trim() || null;
}

export function collectVoiceAssetJobs(toolEvents = []) {
  const jobs = [];
  const seen = new Set();

  for (const event of toolEvents || []) {
    const result = event?.result || {};
    if (!result?.ok) {
      continue;
    }
    const toolName = String(event?.toolName || "");
    const toolResult = result.toolResult || {};

    if (toolName === "check_in_item" && getKitchenRowId(toolResult?.item)) {
      pushUniqueJob(jobs, seen, {
        assetType: "kitchen",
        rowId: getKitchenRowId(toolResult.item),
        sourceTool: toolName
      });
      continue;
    }

    if (toolName === "check_in_many_items" && Array.isArray(toolResult?.items)) {
      for (const entry of toolResult.items) {
        const rowId = getKitchenRowId(entry?.item);
        if (rowId) {
          pushUniqueJob(jobs, seen, {
            assetType: "kitchen",
            rowId,
            sourceTool: toolName
          });
        }
      }
      continue;
    }

    if (
      [
        "log_dish_from_voice",
        "log_dish_ingredients",
        "append_to_recent_dish",
        "update_recent_dish"
      ].includes(toolName)
      && toolResult?.dish?.id
    ) {
      pushUniqueJob(jobs, seen, {
        assetType: "dish",
        rowId: toolResult.dish.id,
        sourceTool: toolName
      });
    }
  }

  return jobs;
}

export async function dispatchVoiceAssetJobs({ toolEvents = [], userContext, env = process.env }) {
  const functionName = String(env.VOICE_IMAGE_POSTPROCESS_FUNCTION_NAME || "").trim();
  if (!functionName) {
    return {
      queued: false,
      count: 0,
      jobs: []
    };
  }

  const ownerId = String(
    userContext?.tableOwnerId
    || userContext?.userId
    || userContext?.ownerId
    || ""
  ).trim();
  if (!ownerId) {
    return {
      queued: false,
      count: 0,
      jobs: []
    };
  }

  const jobs = collectVoiceAssetJobs(toolEvents);
  if (jobs.length === 0) {
    return {
      queued: false,
      count: 0,
      jobs: []
    };
  }

  await Promise.all(jobs.map((job) => getLambdaClient(env).send(new InvokeCommand({
    FunctionName: functionName,
    InvocationType: "Event",
    Payload: Buffer.from(JSON.stringify({
      ownerId,
      assetType: job.assetType,
      rowId: job.rowId,
      sourceTool: job.sourceTool
    }))
  }))));

  console.log("[DEBUG] voice asset jobs dispatched:", JSON.stringify({
    ownerId,
    functionName,
    jobs
  }));

  return {
    queued: true,
    count: jobs.length,
    jobs,
    functionName
  };
}
