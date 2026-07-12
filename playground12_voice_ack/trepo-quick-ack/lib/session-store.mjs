import crypto from "node:crypto";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DeleteCommand, DynamoDBDocumentClient, GetCommand, PutCommand, QueryCommand } from "@aws-sdk/lib-dynamodb";

let dynamoClient;
let documentClient;

function getDynamoDocumentClient() {
  if (!dynamoClient) {
    dynamoClient = new DynamoDBClient({});
  }
  if (!documentClient) {
    documentClient = DynamoDBDocumentClient.from(dynamoClient, {
      marshallOptions: {
        removeUndefinedValues: true
      }
    });
  }
  return documentClient;
}

function getSessionTableName(env) {
  return String(env?.SESSION_TABLE_NAME || "").trim();
}

function getSessionTtlDays(env) {
  const value = Number(env?.SESSION_MEMORY_TTL_DAYS || 30);
  return Number.isFinite(value) && value > 0 ? value : 30;
}

export function getSessionMessageLimit(env) {
  const value = Number(env?.SESSION_MEMORY_MAX_MESSAGES || 12);
  return Number.isFinite(value) && value > 0 ? Math.min(Math.floor(value), 24) : 12;
}

function buildSessionPrefix(sessionId) {
  return `session#${sessionId}#msg#`;
}

function buildRequestEntryId(sessionId, transcript) {
  const normalizedTranscript = String(transcript || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
  const transcriptHash = crypto.createHash("sha256").update(normalizedTranscript).digest("hex");
  return `request#${sessionId}#${transcriptHash}`;
}

function buildSessionEntryId(sessionId, createdAt, index) {
  return `${buildSessionPrefix(sessionId)}${createdAt}#${String(index).padStart(4, "0")}#${crypto.randomUUID()}`;
}

function normalizeSessionMessage(message) {
  const role = ["system", "assistant", "user"].includes(message?.role) ? message.role : null;
  const content = typeof message?.content === "string" ? message.content.trim() : "";
  if (!role || !content) {
    return null;
  }
  const kind = typeof message?.kind === "string" ? message.kind.trim() : "";
  return {
    role,
    content,
    ...(kind ? { kind } : {})
  };
}

function normalizeOwnerWriteContext(ownerContext) {
  if (ownerContext && typeof ownerContext === "object") {
    const householdOwnerId = String(
      ownerContext.householdOwnerId
      || ownerContext.ownerId
      || ownerContext.household_owner_id
      || ""
    ).trim();
    const requestOwnerId = String(
      ownerContext.requestOwnerId
      || ownerContext.request_owner_id
      || ""
    ).trim();
    const tableOwnerId = String(
      ownerContext.tableOwnerId
      || ownerContext.table_owner_id
      || ""
    ).trim();
    const userId = String(
      ownerContext.userId
      || ownerContext.user_id
      || tableOwnerId
      || requestOwnerId
      || ""
    ).trim();
    return {
      householdOwnerId,
      requestOwnerId,
      tableOwnerId,
      userId
    };
  }

  const householdOwnerId = String(ownerContext || "").trim();
  return {
    householdOwnerId,
    requestOwnerId: "",
    tableOwnerId: "",
    userId: ""
  };
}

export async function loadRecentSessionMessages(ownerId, sessionId, limit, env) {
  const tableName = getSessionTableName(env);
  const normalizedOwnerId = String(ownerId || "").trim();
  const normalizedSessionId = String(sessionId || "").trim();
  if (!tableName || !normalizedOwnerId || !normalizedSessionId) {
    return [];
  }

  const maxItems = Number.isFinite(Number(limit)) && Number(limit) > 0
    ? Math.min(Math.floor(Number(limit)), 20)
    : getSessionMessageLimit(env);

  const response = await getDynamoDocumentClient().send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: "#owner_id = :ownerId AND begins_with(#session_entry_id, :sessionPrefix)",
    ExpressionAttributeNames: {
      "#owner_id": "owner_id",
      "#session_entry_id": "session_entry_id"
    },
    ExpressionAttributeValues: {
      ":ownerId": normalizedOwnerId,
      ":sessionPrefix": buildSessionPrefix(normalizedSessionId)
    },
    ScanIndexForward: false,
    Limit: maxItems
  }));

  return (response.Items || [])
    .map((item) => normalizeSessionMessage(item))
    .filter(Boolean)
    .reverse();
}

export async function appendSessionMessages(ownerId, sessionId, messages, env) {
  const tableName = getSessionTableName(env);
  const ownerContext = normalizeOwnerWriteContext(ownerId);
  const normalizedOwnerId = ownerContext.householdOwnerId;
  const normalizedSessionId = String(sessionId || "").trim();
  if (!tableName || !normalizedOwnerId || !normalizedSessionId) {
    return { written: 0 };
  }

  const normalizedMessages = Array.isArray(messages)
    ? messages.map((message) => normalizeSessionMessage(message)).filter(Boolean)
    : [];
  if (normalizedMessages.length === 0) {
    return { written: 0 };
  }

  const ttlDays = getSessionTtlDays(env);
  const nowSeconds = Math.floor(Date.now() / 1000);
  const ttl = nowSeconds + (ttlDays * 24 * 60 * 60);

  for (const [index, message] of normalizedMessages.entries()) {
    const createdAt = new Date().toISOString();
    await getDynamoDocumentClient().send(new PutCommand({
      TableName: tableName,
      Item: {
        owner_id: normalizedOwnerId,
        household_owner_id: normalizedOwnerId,
        ...(ownerContext.requestOwnerId ? { request_owner_id: ownerContext.requestOwnerId } : {}),
        ...(ownerContext.tableOwnerId ? { table_owner_id: ownerContext.tableOwnerId } : {}),
        ...(ownerContext.userId ? { user_id: ownerContext.userId } : {}),
        session_entry_id: buildSessionEntryId(normalizedSessionId, createdAt, index),
        session_id: normalizedSessionId,
        role: message.role,
        content: message.content,
        message_kind: message.kind || "conversation",
        created_at: createdAt,
        ttl
      }
    }));
  }

  return { written: normalizedMessages.length };
}

export async function claimRecentSessionRequest(ownerId, sessionId, transcript, env) {
  const tableName = getSessionTableName(env);
  const ownerContext = normalizeOwnerWriteContext(ownerId);
  const normalizedOwnerId = ownerContext.householdOwnerId;
  const normalizedSessionId = String(sessionId || "").trim();
  const normalizedTranscript = String(transcript || "").trim();
  if (!tableName || !normalizedOwnerId || !normalizedSessionId || !normalizedTranscript) {
    return { claimed: false, reason: "missing_identity" };
  }

  const entryId = buildRequestEntryId(normalizedSessionId, normalizedTranscript);
  const createdAt = new Date().toISOString();
  const ttl = Math.floor(Date.now() / 1000) + 180;

  try {
    await getDynamoDocumentClient().send(new PutCommand({
      TableName: tableName,
      Item: {
        owner_id: normalizedOwnerId,
        household_owner_id: normalizedOwnerId,
        ...(ownerContext.requestOwnerId ? { request_owner_id: ownerContext.requestOwnerId } : {}),
        ...(ownerContext.tableOwnerId ? { table_owner_id: ownerContext.tableOwnerId } : {}),
        ...(ownerContext.userId ? { user_id: ownerContext.userId } : {}),
        session_entry_id: entryId,
        session_id: normalizedSessionId,
        request_status: "processing",
        transcript: normalizedTranscript,
        created_at: createdAt,
        ttl
      },
      ConditionExpression: "attribute_not_exists(owner_id) AND attribute_not_exists(session_entry_id)"
    }));
    return {
      claimed: true,
      entryId
    };
  } catch (error) {
    if (error?.name !== "ConditionalCheckFailedException") {
      throw error;
    }

    const existing = await getDynamoDocumentClient().send(new GetCommand({
      TableName: tableName,
      Key: {
        owner_id: normalizedOwnerId,
        session_entry_id: entryId
      }
    }));

    return {
      claimed: false,
      entryId,
      existing: existing.Item || null
    };
  }
}

export async function completeRecentSessionRequest(ownerId, entryId, responseBody, env) {
  const tableName = getSessionTableName(env);
  const normalizedOwnerId = String(ownerId || "").trim();
  const normalizedEntryId = String(entryId || "").trim();
  if (!tableName || !normalizedOwnerId || !normalizedEntryId) {
    return;
  }

  const createdAt = new Date().toISOString();
  const ttl = Math.floor(Date.now() / 1000) + 20;
  await getDynamoDocumentClient().send(new PutCommand({
    TableName: tableName,
    Item: {
      owner_id: normalizedOwnerId,
      session_entry_id: normalizedEntryId,
      request_status: "completed",
      response_body: responseBody,
      created_at: createdAt,
      ttl
    }
  }));
}

export async function clearRecentSessionRequest(ownerId, entryId, env) {
  const tableName = getSessionTableName(env);
  const normalizedOwnerId = String(ownerId || "").trim();
  const normalizedEntryId = String(entryId || "").trim();
  if (!tableName || !normalizedOwnerId || !normalizedEntryId) {
    return;
  }

  await getDynamoDocumentClient().send(new DeleteCommand({
    TableName: tableName,
    Key: {
      owner_id: normalizedOwnerId,
      session_entry_id: normalizedEntryId
    }
  }));
}
