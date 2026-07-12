import crypto from "node:crypto";

import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";
import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } from "@aws-sdk/lib-dynamodb";
import { syncVoiceTriage } from "./triage.mjs";

const s3Clients = new Map();
const lambdaClients = new Map();
const ddbDocClients = new Map();

function getRegion(env) {
  return env?.AWS_REGION || process.env.AWS_REGION || "us-east-1";
}

function getS3Client(env) {
  const region = getRegion(env);
  if (!s3Clients.has(region)) {
    s3Clients.set(region, new S3Client({ region }));
  }
  return s3Clients.get(region);
}

function getLambdaClient(env) {
  const region = getRegion(env);
  if (!lambdaClients.has(region)) {
    lambdaClients.set(region, new LambdaClient({ region }));
  }
  return lambdaClients.get(region);
}

function getDdbDocClient(env) {
  const region = getRegion(env);
  if (!ddbDocClients.has(region)) {
    const client = new DynamoDBClient({ region });
    ddbDocClients.set(region, DynamoDBDocumentClient.from(client));
  }
  return ddbDocClients.get(region);
}

export function getHeader(headers, name) {
  return headers?.[name] || headers?.[name.toLowerCase()] || headers?.[name.toUpperCase()] || null;
}

export function getContentType(headers) {
  return String(getHeader(headers, "content-type") || "")
    .split(";")[0]
    .trim()
    .toLowerCase();
}

export function normalizeResponseSurface(value) {
  return String(value || "").trim().toLowerCase() === "app" ? "app" : "halo";
}

export function normalizeAudioSampleRate(value, fallback = 24000) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback;
  }
  if (numeric === 16000 || numeric === 24000) {
    return numeric;
  }
  const error = new Error("Unsupported audio sample rate. Supported values are 16000 or 24000.");
  error.statusCode = 400;
  throw error;
}

export function normalizeAudioFormat(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized || "pcm_s16le_mono";
}

export function decodeAudioBody(event) {
  if (!event?.body) {
    const error = new Error("No body (audio) in request");
    error.statusCode = 400;
    throw error;
  }

  if (event.isBase64Encoded) {
    return Buffer.from(event.body, "base64");
  }

  return Buffer.from(event.body, "binary");
}

function sha256Hex(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

export function buildVoiceAsyncJobMetadata({
  ownerId,
  deviceId,
  sessionId = null,
  responseSurface = "halo",
  requestId = null,
  contentType = "audio/pcm",
  audioBuffer,
  audioSampleRate = 24000,
  audioFormat = "pcm_s16le_mono"
}) {
  const audioSha256 = sha256Hex(audioBuffer);
  const dedupeSeed = requestId
    ? `request:${ownerId}|${deviceId || ""}|${sessionId || ""}|${requestId}`
    : `v1|${ownerId}|${deviceId || ""}|${sessionId || ""}|${responseSurface}|${contentType}|${audioSampleRate}|${audioFormat}|${audioSha256}`;
  const jobId = sha256Hex(dedupeSeed);
  return {
    jobId,
    audioSha256,
    objectKey: `voice-jobs/${jobId}.pcm`,
    dedupeSeed
  };
}

function getRequiredEnv(name, env) {
  const value = String(env?.[name] || "").trim();
  if (!value) {
    throw new Error(`${name} is not configured`);
  }
  return value;
}

async function syncVoiceJobTriageSafe(env, job) {
  if (!job) return;
  try {
    await syncVoiceTriage(env, job);
  } catch (error) {
    console.warn("[triage] Failed to sync voice triage:", error?.message || error);
  }
}

export async function putVoiceAudioObject(env, key, audioBuffer, metadata = {}) {
  const bucket = getRequiredEnv("VOICE_ASYNC_BUCKET_NAME", env);
  await getS3Client(env).send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: audioBuffer,
    ContentType: "audio/pcm",
    Metadata: Object.fromEntries(
      Object.entries(metadata)
        .map(([entryKey, value]) => [String(entryKey).toLowerCase(), String(value || "").slice(0, 500)])
        .filter(([, value]) => value)
    )
  }));
  return bucket;
}

export async function getVoiceAudioObject(env, key) {
  const bucket = getRequiredEnv("VOICE_ASYNC_BUCKET_NAME", env);
  const response = await getS3Client(env).send(new GetObjectCommand({
    Bucket: bucket,
    Key: key
  }));
  return streamToBuffer(response.Body);
}

async function streamToBuffer(body) {
  if (!body) {
    return Buffer.alloc(0);
  }
  if (Buffer.isBuffer(body)) {
    return body;
  }

  const chunks = [];
  for await (const chunk of body) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

export async function getVoiceJob(env, jobId) {
  const tableName = getRequiredEnv("VOICE_ASYNC_JOB_TABLE_NAME", env);
  const result = await getDdbDocClient(env).send(new GetCommand({
    TableName: tableName,
    Key: { job_id: jobId }
  }));
  return result.Item || null;
}

export async function createAcceptedVoiceJob(env, item) {
  const tableName = getRequiredEnv("VOICE_ASYNC_JOB_TABLE_NAME", env);
  await getDdbDocClient(env).send(new PutCommand({
    TableName: tableName,
    Item: item,
    ConditionExpression: "attribute_not_exists(job_id)"
  }));
  await syncVoiceJobTriageSafe(env, item);
}

export async function markVoiceJobEnqueued(env, jobId, enqueuePayload = {}) {
  const tableName = getRequiredEnv("VOICE_ASYNC_JOB_TABLE_NAME", env);
  const names = {
    "#status": "status",
    "#updatedAt": "updated_at",
    "#enqueueCount": "enqueue_count"
  };
  const values = {
    ":status": "enqueued",
    ":updatedAt": new Date().toISOString(),
    ":zero": 0,
    ":one": 1
  };
  const updates = [
    "#status = :status",
    "#updatedAt = :updatedAt",
    "#enqueueCount = if_not_exists(#enqueueCount, :zero) + :one"
  ];

  for (const [key, value] of Object.entries(enqueuePayload)) {
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  }

  const result = await getDdbDocClient(env).send(new UpdateCommand({
    TableName: tableName,
    Key: { job_id: jobId },
    UpdateExpression: `SET ${updates.join(", ")}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
    ReturnValues: "ALL_NEW"
  }));
  await syncVoiceJobTriageSafe(env, result.Attributes || null);
}

export async function invokeVoiceWorkerAsync(env, payload) {
  const functionName = getRequiredEnv("VOICE_ASYNC_WORKER_FUNCTION_NAME", env);
  await getLambdaClient(env).send(new InvokeCommand({
    FunctionName: functionName,
    InvocationType: "Event",
    Payload: Buffer.from(JSON.stringify(payload))
  }));
}

export async function claimVoiceJobForProcessing(env, jobId) {
  const tableName = getRequiredEnv("VOICE_ASYNC_JOB_TABLE_NAME", env);
  try {
    const result = await getDdbDocClient(env).send(new UpdateCommand({
      TableName: tableName,
      Key: { job_id: jobId },
      UpdateExpression: "SET #status = :processing, #updatedAt = :updatedAt, #processingStartedAt = :processingStartedAt ADD #workerAttempts :one",
      ConditionExpression: "attribute_exists(job_id) AND (#status = :accepted OR #status = :enqueued OR #status = :failed)",
      ExpressionAttributeNames: {
        "#status": "status",
        "#updatedAt": "updated_at",
        "#processingStartedAt": "processing_started_at",
        "#workerAttempts": "worker_attempts"
      },
      ExpressionAttributeValues: {
        ":accepted": "accepted",
        ":enqueued": "enqueued",
        ":failed": "failed",
        ":processing": "processing",
        ":updatedAt": new Date().toISOString(),
        ":processingStartedAt": new Date().toISOString(),
        ":one": 1
      },
      ReturnValues: "ALL_NEW"
    }));
    await syncVoiceJobTriageSafe(env, result.Attributes || null);
    return true;
  } catch (error) {
    if (error?.name === "ConditionalCheckFailedException") {
      return false;
    }
    throw error;
  }
}

export async function markVoiceJobCompleted(env, jobId, payload = {}) {
  await updateVoiceJobState(env, jobId, "completed", payload);
}

export async function markVoiceJobFailed(env, jobId, payload = {}) {
  await updateVoiceJobState(env, jobId, "failed", payload);
}

async function updateVoiceJobState(env, jobId, status, payload = {}) {
  const tableName = getRequiredEnv("VOICE_ASYNC_JOB_TABLE_NAME", env);
  const names = {
    "#status": "status",
    "#updatedAt": "updated_at"
  };
  const values = {
    ":status": status,
    ":updatedAt": new Date().toISOString()
  };
  const updates = [
    "#status = :status",
    "#updatedAt = :updatedAt"
  ];

  for (const [key, value] of Object.entries(payload)) {
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  }

  const result = await getDdbDocClient(env).send(new UpdateCommand({
    TableName: tableName,
    Key: { job_id: jobId },
    UpdateExpression: `SET ${updates.join(", ")}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
    ReturnValues: "ALL_NEW"
  }));
  await syncVoiceJobTriageSafe(env, result.Attributes || null);
}
