import { formatRecipeResponse } from "./lib/recipe-response.mjs";
import { buildAppOutput } from "./lib/app-response.mjs";
import { dispatchVoiceAssetJobs } from "./lib/voice-asset-dispatcher.mjs";
import { runDeviceAssistant, transcribeAudio } from "./lib/device-assistant.mjs";
import { jsonResponse } from "./lib/http.mjs";
import {
  appendSessionMessages,
  claimRecentSessionRequest,
  clearRecentSessionRequest,
  completeRecentSessionRequest,
  getSessionMessageLimit,
  loadRecentSessionMessages
} from "./lib/session-store.mjs";
import { buildVoiceUiResponse } from "./lib/ui-response.mjs";
import { lookupUserContextByOwnerId } from "./lib/user-context.mjs";

function getHeader(headers, name) {
  return headers?.[name] || headers?.[name.toLowerCase()] || headers?.[name.toUpperCase()] || null;
}

function getContentType(headers) {
  return String(getHeader(headers, "content-type") || "")
    .split(";")[0]
    .trim()
    .toLowerCase();
}

function normalizeResponseSurface(value) {
  return String(value || "").trim().toLowerCase() === "app" ? "app" : "halo";
}

function normalizeAudioSampleRate(value, fallback = 24000) {
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

function normalizeAudioFormat(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized || "pcm_s16le_mono";
}

function decodeEventBodyText(event) {
  if (!event?.body) {
    return "";
  }
  if (event.isBase64Encoded) {
    return Buffer.from(event.body, "base64").toString("utf8");
  }
  return String(event.body);
}

function decodeAudioBody(event) {
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

function decodeJsonRequest(event) {
  const rawBody = decodeEventBodyText(event);
  let payload;
  try {
    payload = JSON.parse(rawBody || "{}");
  } catch {
    const error = new Error("Request body must be valid JSON.");
    error.statusCode = 400;
    throw error;
  }

  const transcript = typeof payload.transcript === "string"
    ? payload.transcript.trim()
    : typeof payload.message === "string"
      ? payload.message.trim()
      : typeof payload.text === "string"
        ? payload.text.trim()
        : "";
  const audioBase64 = typeof payload.audio_base64 === "string"
    ? payload.audio_base64.trim()
    : typeof payload.audioBase64 === "string"
      ? payload.audioBase64.trim()
      : "";
  const bodySessionId = typeof payload.session_id === "string" ? payload.session_id.trim() : "";
  const headerSessionId = String(getHeader(event?.headers || {}, "x-session-id") || "").trim();
  const sessionId = bodySessionId || headerSessionId;
  const responseSurface = normalizeResponseSurface(getHeader(event?.headers || {}, "x-client-surface"));
  const audioSampleRate = normalizeAudioSampleRate(
    payload.audio_sample_rate
      ?? payload.audioSampleRate
      ?? getHeader(event?.headers || {}, "x-audio-sample-rate"),
    24000
  );
  const audioFormat = normalizeAudioFormat(
    payload.audio_format
      ?? payload.audioFormat
      ?? getHeader(event?.headers || {}, "x-audio-format")
  );

  if (transcript) {
    return {
      transcript,
      requestMode: "json_transcript",
      requestMeta: payload,
      sessionId: sessionId || null,
      responseSurface,
      audioSampleRate,
      audioFormat
    };
  }

  if (audioBase64) {
    return {
      audioBuffer: Buffer.from(audioBase64, "base64"),
      requestMode: "json_audio_base64",
      requestMeta: payload,
      sessionId: sessionId || null,
      responseSurface,
      audioSampleRate,
      audioFormat
    };
  }

  const error = new Error("JSON requests must include transcript, message, text, or audio_base64.");
  error.statusCode = 400;
  throw error;
}

function decodeAssistantInput(event) {
  const contentType = getContentType(event?.headers || {});
  if (contentType === "application/json") {
    return decodeJsonRequest(event);
  }

  return {
    audioBuffer: decodeAudioBody(event),
    requestMode: "binary_audio",
    requestMeta: null,
    sessionId: String(getHeader(event?.headers || {}, "x-session-id") || "").trim() || null,
    responseSurface: normalizeResponseSurface(getHeader(event?.headers || {}, "x-client-surface")),
    audioSampleRate: normalizeAudioSampleRate(getHeader(event?.headers || {}, "x-audio-sample-rate"), 24000),
    audioFormat: normalizeAudioFormat(getHeader(event?.headers || {}, "x-audio-format"))
  };
}

const READ_ONLY_TOOL_NAMES = new Set([
  "get_shopping_list",
  "list_store_tabs",
  "get_kitchen_overview",
  "search_kitchen_item",
  "get_recent_dishes",
  "get_dish_detail"
]);

function cleanSessionSummaryText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

function summarizeRecentEntityReferences(event) {
  const result = event?.result || {};
  const toolResult = result?.toolResult || {};
  const names = new Set();
  const details = [];

  const pushName = (value) => {
    const cleaned = cleanSessionSummaryText(value);
    if (cleaned) {
      names.add(cleaned);
    }
  };

  pushName(result?.args?.item_name);
  pushName(result?.args?.dish_name);
  pushName(result?.args?.recipe_title);
  pushName(toolResult?.item?.item_name);
  pushName(toolResult?.dish?.dish_name);
  pushName(toolResult?.recipe?.title);

  if (Array.isArray(toolResult?.items)) {
    for (const item of toolResult.items.slice(0, 5)) {
      pushName(item?.item_name || item?.dish_name || item?.title || item?.item?.item_name || item?.dish?.dish_name);
    }
  }

  if (toolResult?.item?.storage_location) {
    details.push(`Location: ${cleanSessionSummaryText(toolResult.item.storage_location)}`);
  }
  if (toolResult?.item?.state_label) {
    details.push(`State: ${cleanSessionSummaryText(toolResult.item.state_label)}`);
  }

  const recentEntities = Array.from(names).slice(0, 5);
  return {
    recentEntities,
    details
  };
}

function summarizeAmbiguityCandidates(toolEvents = []) {
  const lastEvent = [...toolEvents].reverse().find((event) => event?.result?.details?.type);
  const details = lastEvent?.result?.details || null;
  if (!details || details.type !== "ambiguous_kitchen_item" || !Array.isArray(details.candidates)) {
    return null;
  }

  const candidates = details.candidates
    .slice(0, 5)
    .map((candidate) => {
      const parts = [];
      if (candidate?.item_name) parts.push(candidate.item_name);
      if (candidate?.location) parts.push(`in ${candidate.location}`);
      if (candidate?.remaining_quantity) parts.push(candidate.remaining_quantity);
      if (candidate?.expiration_date) parts.push(`expires ${candidate.expiration_date}`);
      return {
        itemId: candidate?.item_id || null,
        summary: parts.join(" - ") || candidate?.summary || null
      };
    })
    .filter((candidate) => candidate.itemId || candidate.summary);

  if (candidates.length === 0) {
    return null;
  }

  return {
    type: details.type,
    candidates
  };
}

function buildHiddenSessionRecap({ transcript, resultText, responseType, toolEvents = [] }) {
  const lastSuccessfulWrite = [...toolEvents]
    .reverse()
    .find((event) => event?.result?.ok && !READ_ONLY_TOOL_NAMES.has(event.toolName));
  const lastSuccessfulEvent = [...toolEvents].reverse().find((event) => event?.result?.ok);

  if (lastSuccessfulWrite) {
    const summary = cleanSessionSummaryText(lastSuccessfulWrite?.result?.actionSummary || resultText);
    const toolName = cleanSessionSummaryText(lastSuccessfulWrite.toolName);
    const entitySummary = summarizeRecentEntityReferences(lastSuccessfulWrite);
    return [
      "Session recap:",
      `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`,
      `The assistant just completed a household update using ${toolName}.`,
      `Result summary: ${summary || "Completed successfully."}`,
      ...(entitySummary.recentEntities.length > 0
        ? [`Recent entity references: ${entitySummary.recentEntities.join(", ")}`]
        : []),
      ...entitySummary.details,
      ...(entitySummary.recentEntities.length > 0
        ? ["If the next user message says 'that', 'it', or a broad food name, prefer these recent entities when current household data supports the match."]
        : [])
    ].join("\n");
  }

  if (responseType === "clarification_needed") {
    const ambiguity = summarizeAmbiguityCandidates(toolEvents);
    return [
      "Session recap:",
      `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`,
      `The assistant asked a clarification question: "${cleanSessionSummaryText(resultText)}"`,
      ...(ambiguity
        ? [
            `Ambiguous kitchen candidates: ${ambiguity.candidates.map((candidate) => `${candidate.summary || "candidate"}${candidate.itemId ? ` [item_id: ${candidate.itemId}]` : ""}`).join("; ")}`,
            "If the next user says 'most recent one', 'oldest', or 'the other one', resolve it against these candidates instead of asking the same question again."
          ]
        : [])
    ].join("\n");
  }

  if (lastSuccessfulEvent) {
    const summary = cleanSessionSummaryText(lastSuccessfulEvent?.result?.actionSummary || resultText);
    const toolName = cleanSessionSummaryText(lastSuccessfulEvent.toolName);
    return [
      "Session recap:",
      `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`,
      `The assistant most recently answered using ${toolName}.`,
      `Result summary: ${summary || cleanSessionSummaryText(resultText) || "Answered successfully."}`
    ].join("\n");
  }

  return [
    "Session recap:",
    `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`,
    `The assistant replied: "${cleanSessionSummaryText(resultText)}"`
  ].join("\n");
}

function buildSessionPersistenceMessages({ transcript, resultText, responseType, toolEvents = [] }) {
  const messages = [
    { role: "user", content: transcript, kind: "conversation" },
    { role: "assistant", content: resultText || "Okay.", kind: "conversation" }
  ];
  const hiddenRecap = buildHiddenSessionRecap({
    transcript,
    resultText,
    responseType,
    toolEvents
  });

  if (hiddenRecap) {
    messages.push({
      role: "system",
      content: hiddenRecap,
      kind: "recap"
    });
  }

  return messages;
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyEntityCapitalization(text, toolEvents = []) {
  let nextText = String(text || "");
  if (!nextText) {
    return nextText;
  }

  const entityNames = [];
  const protectedPhrases = [];
  for (const event of toolEvents || []) {
    if (!event?.result?.ok) {
      continue;
    }
    const toolResult = event?.result?.toolResult || {};
    const itemName = String(toolResult?.item?.item_name || toolResult?.dish?.dish_name || "").trim();
    if (itemName) {
      entityNames.push(itemName);
    }
    const servingSize = String(toolResult?.dish?.serving_size || "").trim();
    if (servingSize) {
      protectedPhrases.push(servingSize);
    }
    if (Array.isArray(toolResult?.items)) {
      for (const entry of toolResult.items) {
        const batchName = String(entry?.item?.item_name || entry?.dish?.dish_name || entry?.item_name || "").trim();
        if (batchName) {
          entityNames.push(batchName);
        }
        const batchServingSize = String(entry?.dish?.serving_size || "").trim();
        if (batchServingSize) {
          protectedPhrases.push(batchServingSize);
        }
      }
    }
  }

  const placeholders = [];
  const uniqueProtectedPhrases = Array.from(new Set(protectedPhrases.filter(Boolean)));
  uniqueProtectedPhrases.sort((left, right) => right.length - left.length);
  for (const phrase of uniqueProtectedPhrases) {
    const token = `__PHRASE_${placeholders.length}__`;
    const pattern = new RegExp(`\\b${escapeRegExp(phrase)}\\b`, "gi");
    nextText = nextText.replace(pattern, token);
    placeholders.push({ token, phrase });
  }

  const uniqueNames = Array.from(new Set(entityNames.filter(Boolean)));
  uniqueNames.sort((left, right) => right.length - left.length);

  for (const name of uniqueNames) {
    const pattern = new RegExp(`\\b${escapeRegExp(name)}\\b`, "gi");
    nextText = nextText.replace(pattern, name);
  }

  for (const { token, phrase } of placeholders) {
    nextText = nextText.replaceAll(token, phrase);
  }

  return nextText;
}

function buildFirmwareBody({
  text,
  transcript = null,
  quickItems = [],
  version = "2",
  type = "info_answer",
  ui = null,
  ownerId = null,
  deviceId = null,
  speechDetected = null,
  durationMs = null,
  toolTrace = [],
  error = null,
  appOutput = null
}) {
  return {
    text,
    transcript,
    quickItems,
    version,
    type,
    ui,
    rawTranscript: transcript,
    ownerId,
    deviceId,
    speechDetected,
    durationMs,
    toolTrace,
    error,
    app_output: appOutput
  };
}

function buildAppOutputPayload({
  text,
  transcript = null,
  quickItems = [],
  toolTrace = [],
  toolEvents = [],
  type = "info_answer",
  error = null,
  speechDetected = null,
  durationMs = null,
  sessionId = null,
  memoryUsed = false,
  responseSurface = "halo"
}) {
  return buildAppOutput({
    text,
    transcript,
    quickItems,
    toolTrace,
    toolEvents,
    type,
    error,
    speechDetected,
    durationMs,
    sessionId,
    memoryUsed,
    responseSurface
  });
}

async function runBackfillSharedTables() {
  const mysql = (await import("mysql2/promise")).default;
  const conn = await mysql.createConnection({
    host: process.env.DB_HOST, port: Number(process.env.DB_PORT || 3306),
    user: process.env.DB_USER, password: process.env.DB_PASS,
    database: process.env.DB_NAME, charset: "utf8mb4", connectTimeout: 10000,
  });
  const suffixes = [
    { suffix: "_prod_kitchen", shared: "shared_kitchen" },
    { suffix: "_discards", shared: "shared_discards" },
    { suffix: "_dishes", shared: "shared_dishes" },
    { suffix: "_new_list", shared: "shared_shopping_list" },
  ];
  try {
    const [tables] = await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()");
    const allTables = tables.map(r => r.table_name || r.TABLE_NAME);
    const [users] = await conn.execute("SELECT DISTINCT user_id FROM new_users WHERE user_id IS NOT NULL");
    const userIds = users.map(r => r.user_id).filter(Boolean);
    const log = [`Found ${userIds.length} users`];
    let total = 0;
    for (const { suffix, shared } of suffixes) {
      let copied = 0;
      for (const uid of userIds) {
        const safe = uid.replace(/[^a-zA-Z0-9_-]/g, "");
        const src = `${safe}${suffix}`;
        if (!allTables.includes(src)) continue;
        try {
          const [srcCols] = await conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = ?", [src]);
          const [dstCols] = await conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = ?", [shared]);
          const srcSet = new Set(srcCols.map(r => (r.column_name || r.COLUMN_NAME).toLowerCase()));
          const dstSet = new Set(dstCols.map(r => (r.column_name || r.COLUMN_NAME).toLowerCase()));
          const common = [...srcSet].filter(c => dstSet.has(c) && c !== "owner_id" && (suffix !== "_new_list" || c !== "_id"));
          if (!common.length) continue;
          const cols = common.map(c => `\`${c}\``).join(", ");
          const [res] = await conn.execute(`INSERT IGNORE INTO \`${shared}\` (\`owner_id\`, ${cols}) SELECT ?, ${cols} FROM \`${src}\``, [safe]);
          if (res.affectedRows > 0) { copied += res.affectedRows; log.push(`  ${src}: ${res.affectedRows} rows`); }
        } catch (e) { log.push(`  ${src}: ERROR ${e.message}`); }
      }
      log.push(`${shared}: ${copied} total`);
      total += copied;
    }
    log.push(`TOTAL: ${total} rows backfilled`);
    return log.join("\n");
  } finally { await conn.end(); }
}

export async function handler(event, context) {
  const startedAt = Date.now();
  const lambdaDeadline = context?.getRemainingTimeInMillis
    ? Date.now() + context.getRemainingTimeInMillis() - 5000
    : Date.now() + 140000;
  console.log("[DEBUG] event keys:", Object.keys(event || {}));

  // Admin action: backfill shared tables (invoke directly, not via API Gateway)
  if (event?.admin_action === "backfill_shared_tables") {
    try {
      const result = await runBackfillSharedTables();
      return { statusCode: 200, body: result };
    } catch (e) {
      return { statusCode: 500, body: `Backfill failed: ${e.message}` };
    }
  }

  if (!process.env.OPENAI_API_KEY) {
    console.error("OPENAI_API_KEY is not set");
    return jsonResponse(500, { error: "Server misconfigured" });
  }

  const headers = event?.headers || {};
  const ownerId = getHeader(headers, "x-owner-id");
  const deviceId = getHeader(headers, "x-device-id");

  console.log("[DEBUG] ownerId:", ownerId, "deviceId:", deviceId);

  let requestInput;
  let audioBuffer;
  let transcript = null;
  let speechDetected = false;
  let sessionMessages = [];
  let memoryUsed = false;
  let requestClaim = null;
  try {
    requestInput = decodeAssistantInput(event);
    audioBuffer = requestInput.audioBuffer || null;
    transcript = requestInput.transcript || null;
    speechDetected = Boolean(transcript || audioBuffer?.length);
  } catch (error) {
    console.error("[ERROR] failed to decode audio body:", error);
    return jsonResponse(error.statusCode || 400, { error: error.message || "Invalid audio body" });
  }

  console.log("[DEBUG] request mode:", requestInput.requestMode, "audio bytes =", audioBuffer?.length || 0);
  console.log("[DEBUG] response surface:", requestInput.responseSurface);
  let toolTrace = [];
  let toolEvents = [];

  if (!transcript) {
    try {
      transcript = await transcribeAudio(audioBuffer, process.env, {
        sampleRate: requestInput?.audioSampleRate,
        audioFormat: requestInput?.audioFormat
      });
      console.log("[DEBUG] transcript:", transcript);
    } catch (error) {
      console.error("[ERROR] transcription failed:", error);
      const uiResponse = buildVoiceUiResponse({
        text: "I couldn't understand that audio.",
        error: "transcription_failed"
      });
      const durationMs = Date.now() - startedAt;
      return jsonResponse(200, buildFirmwareBody({
        text: "I couldn't understand that audio.",
        transcript: null,
        quickItems: [],
        version: uiResponse.version,
        type: uiResponse.type,
        ui: uiResponse.ui,
        ownerId,
        deviceId,
        speechDetected: false,
        durationMs,
        toolTrace: [],
        error: "transcription_failed",
        appOutput: buildAppOutputPayload({
          text: "I couldn't understand that audio.",
          transcript: null,
          quickItems: [],
          toolTrace: [],
          toolEvents: [],
          type: uiResponse.type,
          error: "transcription_failed",
          speechDetected: false,
          durationMs,
          sessionId: requestInput?.sessionId || null,
          memoryUsed,
          responseSurface: requestInput?.responseSurface || "halo"
        })
      }));
    }
  }

  if (!transcript) {
    const uiResponse = buildVoiceUiResponse({
      text: "No speech detected."
    });
    const durationMs = Date.now() - startedAt;
    return jsonResponse(200, buildFirmwareBody({
      text: "No speech detected.",
      transcript: null,
      quickItems: [],
      version: uiResponse.version,
      type: uiResponse.type,
      ui: uiResponse.ui,
      ownerId,
      deviceId,
      speechDetected: false,
      durationMs,
      toolTrace: [],
      appOutput: buildAppOutputPayload({
        text: "No speech detected.",
        transcript: null,
        quickItems: [],
        toolTrace: [],
        toolEvents: [],
        type: uiResponse.type,
        speechDetected: false,
        durationMs,
        sessionId: requestInput?.sessionId || null,
        memoryUsed,
        responseSurface: requestInput?.responseSurface || "halo"
      })
    }));
  }

  if (!ownerId) {
    const uiResponse = buildVoiceUiResponse({
      text: "This device is not linked to a household yet.",
      error: "missing_owner_id"
    });
    const durationMs = Date.now() - startedAt;
    return jsonResponse(200, buildFirmwareBody({
      text: "This device is not linked to a household yet.",
      transcript,
      quickItems: [],
      version: uiResponse.version,
      type: uiResponse.type,
      ui: uiResponse.ui,
      ownerId: null,
      deviceId,
      speechDetected: true,
      durationMs,
      toolTrace: [],
      error: "missing_owner_id",
      appOutput: buildAppOutputPayload({
        text: "This device is not linked to a household yet.",
        transcript,
        quickItems: [],
        toolTrace: [],
        toolEvents: [],
        type: uiResponse.type,
        error: "missing_owner_id",
        speechDetected: true,
        durationMs,
        sessionId: requestInput?.sessionId || null,
        memoryUsed,
        responseSurface: requestInput?.responseSurface || "halo"
      })
    }));
  }

  let userContext;
  try {
    userContext = await lookupUserContextByOwnerId(ownerId, { env: process.env });
    console.log("[DEBUG] resolved user context:", JSON.stringify({
      ownerId: userContext.ownerId,
      userId: userContext.userId,
      tableOwnerId: userContext.tableOwnerId,
      householdSize: userContext.householdSize,
      shoppingNamespace: userContext.shoppingNamespace,
      hasShoppingNamespace: userContext.hasShoppingNamespace,
      isFallbackContext: userContext.isFallbackContext
    }));
  } catch (error) {
    console.error("[ERROR] context lookup failed:", error);
    const uiResponse = buildVoiceUiResponse({
      text: "I couldn't load this household right now.",
      error: "context_lookup_failed"
    });
    const durationMs = Date.now() - startedAt;
    return jsonResponse(200, buildFirmwareBody({
      text: "I couldn't load this household right now.",
      transcript,
      quickItems: [],
      version: uiResponse.version,
      type: uiResponse.type,
      ui: uiResponse.ui,
      ownerId,
      deviceId,
      speechDetected: true,
      durationMs,
      toolTrace: [],
      error: "context_lookup_failed",
      appOutput: buildAppOutputPayload({
        text: "I couldn't load this household right now.",
        transcript,
        quickItems: [],
        toolTrace: [],
        toolEvents: [],
        type: uiResponse.type,
        error: "context_lookup_failed",
        speechDetected: true,
        durationMs,
        sessionId: requestInput?.sessionId || null,
        memoryUsed,
        responseSurface: requestInput?.responseSurface || "halo"
      })
    }));
  }

  if (requestInput?.sessionId) {
    try {
      sessionMessages = await loadRecentSessionMessages(
        userContext.ownerId,
        requestInput.sessionId,
        getSessionMessageLimit(process.env),
        process.env
      );
      memoryUsed = sessionMessages.length > 0;
      console.log("[DEBUG] loaded session history:", JSON.stringify({
        ownerId: userContext.ownerId,
        sessionId: requestInput.sessionId,
        messageCount: sessionMessages.length
      }));
    } catch (error) {
      console.warn("[WARN] failed to load session history:", error?.message || error);
      sessionMessages = [];
      memoryUsed = false;
    }
  }

  const shouldDeduplicateAppRequest = requestInput?.responseSurface === "app"
    && requestInput?.requestMode === "json_transcript"
    && Boolean(requestInput?.sessionId)
    && Boolean(transcript);

  if (shouldDeduplicateAppRequest) {
    try {
      requestClaim = await claimRecentSessionRequest(
        {
          householdOwnerId: userContext.ownerId,
          requestOwnerId: ownerId,
          tableOwnerId: userContext.tableOwnerId,
          userId: userContext.userId
        },
        requestInput.sessionId,
        transcript,
        process.env
      );

      if (!requestClaim?.claimed) {
        if (requestClaim?.existing?.request_status === "completed" && requestClaim?.existing?.response_body) {
          console.log("[DEBUG] duplicate app request served from recent cache:", JSON.stringify({
            ownerId: userContext.ownerId,
            sessionId: requestInput.sessionId
          }));
          return jsonResponse(200, requestClaim.existing.response_body);
        }

        const duplicateText = "I'm still finishing your last update. Give me a moment.";
        const durationMs = Date.now() - startedAt;
        return jsonResponse(200, buildFirmwareBody({
          text: duplicateText,
          transcript,
          quickItems: [],
          version: "2",
          type: "clarification_needed",
          ui: null,
          ownerId,
          deviceId,
          speechDetected: true,
          durationMs,
          toolTrace: [],
          appOutput: buildAppOutputPayload({
            text: duplicateText,
            transcript,
            quickItems: [],
            toolTrace: [],
            toolEvents: [],
            type: "clarification_needed",
            speechDetected: true,
            durationMs,
            sessionId: requestInput?.sessionId || null,
            memoryUsed,
            responseSurface: requestInput?.responseSurface || "halo"
          })
        }));
      }
    } catch (error) {
      console.warn("[WARN] request dedupe claim failed:", error?.message || error);
      requestClaim = null;
    }
  }

  try {
    const result = await runDeviceAssistant({
      transcript,
      userContext,
      env: process.env,
      sessionMessages,
      responseSurface: requestInput?.responseSurface || "halo",
      lambdaDeadline
    });

    toolTrace = result.toolTrace || [];
    toolEvents = result.toolEvents || [];
    const originalResponseText = applyEntityCapitalization(result.text || "Okay.", toolEvents);
    const responseText = requestInput?.responseSurface === "app"
      ? (formatRecipeResponse(originalResponseText) || originalResponseText) : originalResponseText;
    if (requestInput?.sessionId) {
      try {
        await appendSessionMessages(
          {
            householdOwnerId: userContext.ownerId,
            requestOwnerId: ownerId,
            tableOwnerId: userContext.tableOwnerId,
            userId: userContext.userId
          },
          requestInput.sessionId,
          buildSessionPersistenceMessages({
            transcript,
            resultText: responseText,
            responseType: result.type || "info_answer",
            toolEvents
          }),
          process.env
        );
      } catch (error) {
        console.warn("[WARN] failed to persist session history:", error?.message || error);
      }
    }
    console.log("[DEBUG] firmware response summary:", JSON.stringify({
      transcript,
      text: responseText,
      type: result.type || "info_answer",
      quickItems: result.quickItems || [],
      toolCount: toolTrace.length,
      sessionId: requestInput?.sessionId || null,
      memoryUsed,
      responseSurface: requestInput?.responseSurface || "halo"
    }));

    const durationMs = Date.now() - startedAt;
    const responseBody = buildFirmwareBody({
      text: responseText,
      transcript,
      quickItems: result.quickItems || [],
      version: result.version || "2",
      type: result.type || "info_answer",
      ui: result.ui || null,
      ownerId,
      deviceId,
      speechDetected: true,
      durationMs,
      toolTrace,
      appOutput: buildAppOutputPayload({
        text: responseText,
        transcript,
        quickItems: result.quickItems || [],
        toolTrace,
        toolEvents,
        type: result.type || "info_answer",
        speechDetected: true,
        durationMs,
        sessionId: requestInput?.sessionId || null,
        memoryUsed,
        responseSurface: requestInput?.responseSurface || "halo"
      })
    });
    if (requestClaim?.claimed && requestClaim?.entryId) {
      try {
        await completeRecentSessionRequest(userContext.ownerId, requestClaim.entryId, responseBody, process.env);
      } catch (error) {
        console.warn("[WARN] failed to persist duplicate response cache:", error?.message || error);
      }
    }
    try {
      await dispatchVoiceAssetJobs({
        toolEvents,
        userContext,
        env: process.env
      });
    } catch (error) {
      console.warn("[WARN] failed to dispatch voice asset jobs:", error?.message || error);
    }
    return jsonResponse(200, responseBody);
  } catch (error) {
    if (requestClaim?.claimed && requestClaim?.entryId) {
      try {
        await clearRecentSessionRequest(userContext.ownerId, requestClaim.entryId, process.env);
      } catch (clearError) {
        console.warn("[WARN] failed to clear request dedupe entry:", clearError?.message || clearError);
      }
    }
    console.error("[ERROR] assistant execution failed:", error);
    const uiResponse = buildVoiceUiResponse({
      text: "I heard you, but I couldn't finish that request.",
      error: "assistant_failed"
    });
    const durationMs = Date.now() - startedAt;
    return jsonResponse(200, buildFirmwareBody({
      text: "I heard you, but I couldn't finish that request.",
      transcript,
      quickItems: [],
      version: uiResponse.version,
      type: uiResponse.type,
      ui: uiResponse.ui,
      ownerId,
      deviceId,
      speechDetected: true,
      durationMs,
      toolTrace,
      error: "assistant_failed",
      appOutput: buildAppOutputPayload({
        text: "I heard you, but I couldn't finish that request.",
        transcript,
        quickItems: [],
        toolTrace,
        toolEvents,
        type: uiResponse.type,
        error: "assistant_failed",
        speechDetected: true,
        durationMs,
        sessionId: requestInput?.sessionId || null,
        memoryUsed,
        responseSurface: requestInput?.responseSurface || "halo"
      })
    }));
  }
}

