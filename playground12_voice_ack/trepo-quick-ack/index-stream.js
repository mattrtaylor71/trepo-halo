import { formatRecipeResponse } from "./lib/recipe-response.mjs";
/**
 * Streaming voice assistant handler for iOS app.
 *
 * Uses Lambda Function URL with RESPONSE_STREAM invoke mode.
 * Writes newline-delimited JSON (NDJSON) events to the client:
 *   meta, text_delta, tool_start, tool_end, done, error
 *
 * Reuses all decode, auth, session, dedup, and tool logic from the
 * synchronous handler (index.js).
 */

import { buildAppOutput } from "./lib/app-response.mjs";
import { dispatchVoiceAssetJobs } from "./lib/voice-asset-dispatcher.mjs";
import { runDeviceAssistantStreaming, transcribeAudio } from "./lib/device-assistant.mjs";
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

// ---------------------------------------------------------------------------
// Helpers copied from index.js (lightweight, no external deps)
// ---------------------------------------------------------------------------

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

function decodeEventBodyText(event) {
  if (!event?.body) return "";
  if (event.isBase64Encoded) return Buffer.from(event.body, "base64").toString("utf8");
  return String(event.body);
}

function decodeAudioBody(event) {
  if (!event?.body) {
    const error = new Error("No body (audio) in request");
    error.statusCode = 400;
    throw error;
  }
  return event.isBase64Encoded
    ? Buffer.from(event.body, "base64")
    : Buffer.from(event.body, "binary");
}

function decodeJsonRequest(event) {
  const rawBody = decodeEventBodyText(event);
  let payload;
  try { payload = JSON.parse(rawBody || "{}"); } catch {
    const error = new Error("Request body must be valid JSON.");
    error.statusCode = 400;
    throw error;
  }

  const transcript = typeof payload.transcript === "string" ? payload.transcript.trim()
    : typeof payload.message === "string" ? payload.message.trim()
    : typeof payload.text === "string" ? payload.text.trim()
    : "";
  const audioBase64 = typeof payload.audio_base64 === "string" ? payload.audio_base64.trim()
    : typeof payload.audioBase64 === "string" ? payload.audioBase64.trim()
    : "";
  const bodySessionId = typeof payload.session_id === "string" ? payload.session_id.trim() : "";
  const headerSessionId = String(getHeader(event?.headers || {}, "x-session-id") || "").trim();
  const sessionId = bodySessionId || headerSessionId;

  if (transcript) {
    return { transcript, requestMode: "json_transcript", sessionId: sessionId || null, responseSurface: normalizeResponseSurface(getHeader(event?.headers || {}, "x-client-surface")) };
  }
  if (audioBase64) {
    return { audioBuffer: Buffer.from(audioBase64, "base64"), requestMode: "json_audio_base64", sessionId: sessionId || null, responseSurface: normalizeResponseSurface(getHeader(event?.headers || {}, "x-client-surface")) };
  }
  const error = new Error("JSON requests must include transcript, message, text, or audio_base64.");
  error.statusCode = 400;
  throw error;
}

function decodeAssistantInput(event) {
  const contentType = getContentType(event?.headers || {});
  if (contentType === "application/json") return decodeJsonRequest(event);
  return {
    audioBuffer: decodeAudioBody(event),
    requestMode: "binary_audio",
    sessionId: String(getHeader(event?.headers || {}, "x-session-id") || "").trim() || null,
    responseSurface: normalizeResponseSurface(getHeader(event?.headers || {}, "x-client-surface"))
  };
}

// Entity capitalization (same as index.js)
function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyEntityCapitalization(text, toolEvents = []) {
  let nextText = String(text || "");
  if (!nextText) return nextText;

  const entityNames = [];
  const protectedPhrases = [];
  for (const event of toolEvents || []) {
    if (!event?.result?.ok) continue;
    const toolResult = event?.result?.toolResult || {};
    const itemName = String(toolResult?.item?.item_name || toolResult?.dish?.dish_name || "").trim();
    if (itemName) entityNames.push(itemName);
    const servingSize = String(toolResult?.dish?.serving_size || "").trim();
    if (servingSize) protectedPhrases.push(servingSize);
    if (Array.isArray(toolResult?.items)) {
      for (const entry of toolResult.items) {
        const batchName = String(entry?.item?.item_name || entry?.dish?.dish_name || entry?.item_name || "").trim();
        if (batchName) entityNames.push(batchName);
        const batchServingSize = String(entry?.dish?.serving_size || "").trim();
        if (batchServingSize) protectedPhrases.push(batchServingSize);
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

// Session persistence helpers (same as index.js)
const READ_ONLY_TOOL_NAMES = new Set([
  "get_shopping_list", "list_store_tabs", "get_kitchen_overview",
  "search_kitchen_item", "get_recent_dishes", "get_dish_detail"
]);

function cleanSessionSummaryText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function summarizeRecentEntityReferences(event) {
  const result = event?.result || {};
  const toolResult = result?.toolResult || {};
  const names = new Set();
  const details = [];
  const pushName = (v) => { const c = cleanSessionSummaryText(v); if (c) names.add(c); };
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
  if (toolResult?.item?.storage_location) details.push(`Location: ${cleanSessionSummaryText(toolResult.item.storage_location)}`);
  if (toolResult?.item?.state_label) details.push(`State: ${cleanSessionSummaryText(toolResult.item.state_label)}`);
  return { recentEntities: Array.from(names).slice(0, 5), details };
}

function summarizeAmbiguityCandidates(toolEvents = []) {
  const lastEvent = [...toolEvents].reverse().find((e) => e?.result?.details?.type);
  const details = lastEvent?.result?.details || null;
  if (!details || details.type !== "ambiguous_kitchen_item" || !Array.isArray(details.candidates)) return null;
  const candidates = details.candidates.slice(0, 5).map((c) => {
    const parts = [];
    if (c?.item_name) parts.push(c.item_name);
    if (c?.location) parts.push(`in ${c.location}`);
    if (c?.remaining_quantity) parts.push(c.remaining_quantity);
    if (c?.expiration_date) parts.push(`expires ${c.expiration_date}`);
    return { itemId: c?.item_id || null, summary: parts.join(" - ") || c?.summary || null };
  }).filter((c) => c.itemId || c.summary);
  return candidates.length === 0 ? null : { type: details.type, candidates };
}

function buildHiddenSessionRecap({ transcript, resultText, responseType, toolEvents = [] }) {
  const lastWrite = [...toolEvents].reverse().find((e) => e?.result?.ok && !READ_ONLY_TOOL_NAMES.has(e.toolName));
  const lastSuccess = [...toolEvents].reverse().find((e) => e?.result?.ok);
  if (lastWrite) {
    const summary = cleanSessionSummaryText(lastWrite?.result?.actionSummary || resultText);
    const toolName = cleanSessionSummaryText(lastWrite.toolName);
    const entity = summarizeRecentEntityReferences(lastWrite);
    return ["Session recap:", `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`, `The assistant just completed a household update using ${toolName}.`, `Result summary: ${summary || "Completed successfully."}`, ...(entity.recentEntities.length > 0 ? [`Recent entity references: ${entity.recentEntities.join(", ")}`] : []), ...entity.details, ...(entity.recentEntities.length > 0 ? ["If the next user message says 'that', 'it', or a broad food name, prefer these recent entities when current household data supports the match."] : [])].join("\n");
  }
  if (responseType === "clarification_needed") {
    const ambiguity = summarizeAmbiguityCandidates(toolEvents);
    return ["Session recap:", `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`, `The assistant asked a clarification question: "${cleanSessionSummaryText(resultText)}"`, ...(ambiguity ? [`Ambiguous kitchen candidates: ${ambiguity.candidates.map((c) => `${c.summary || "candidate"}${c.itemId ? ` [item_id: ${c.itemId}]` : ""}`).join("; ")}`, "If the next user says 'most recent one', 'oldest', or 'the other one', resolve it against these candidates instead of asking the same question again."] : [])].join("\n");
  }
  if (lastSuccess) {
    const summary = cleanSessionSummaryText(lastSuccess?.result?.actionSummary || resultText);
    const toolName = cleanSessionSummaryText(lastSuccess.toolName);
    return ["Session recap:", `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`, `The assistant most recently answered using ${toolName}.`, `Result summary: ${summary || cleanSessionSummaryText(resultText) || "Answered successfully."}`].join("\n");
  }
  return ["Session recap:", `The user most recently asked: "${cleanSessionSummaryText(transcript)}"`, `The assistant replied: "${cleanSessionSummaryText(resultText)}"`].join("\n");
}

function buildSessionPersistenceMessages({ transcript, resultText, responseType, toolEvents = [] }) {
  const messages = [
    { role: "user", content: transcript, kind: "conversation" },
    { role: "assistant", content: resultText || "Okay.", kind: "conversation" }
  ];
  const recap = buildHiddenSessionRecap({ transcript, resultText, responseType, toolEvents });
  if (recap) messages.push({ role: "system", content: recap, kind: "recap" });
  return messages;
}

// ---------------------------------------------------------------------------
// Streaming handler
// ---------------------------------------------------------------------------

function writeLine(stream, obj) {
  stream.write(JSON.stringify(obj) + "\n");
}

async function streamHandler(event, responseStream, context) {
  // Don't let a dangling handle (open DB socket / keep-alive connection) from the
  // post-stream cleanup hold the invocation open to the 150s timeout after the
  // handler logic is done — return as soon as the handler promise resolves.
  if (context) context.callbackWaitsForEmptyEventLoop = false;
  const startedAt = Date.now();
  const lambdaDeadline = context?.getRemainingTimeInMillis
    ? Date.now() + context.getRemainingTimeInMillis() - 5000
    : Date.now() + 140000;
  const headers = event?.headers || {};
  const ownerId = getHeader(headers, "x-owner-id");
  const deviceId = getHeader(headers, "x-device-id");

  // Set content type for NDJSON
  responseStream = awslambda.HttpResponseStream.from(responseStream, {
    statusCode: 200,
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-store",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "content-type, x-owner-id, x-device-id, x-session-id, x-client-surface"
    }
  });

  let requestInput;
  let transcript = null;
  let sessionMessages = [];
  let memoryUsed = false;
  let requestClaim = null;

  try {
    requestInput = decodeAssistantInput(event);
    transcript = requestInput.transcript || null;
  } catch (error) {
    writeLine(responseStream, { type: "error", message: error.message || "Invalid request" });
    responseStream.end();
    return;
  }

  // Transcribe audio if needed
  if (!transcript && requestInput.audioBuffer) {
    try {
      transcript = await transcribeAudio(requestInput.audioBuffer, process.env, {
        sampleRate: requestInput?.audioSampleRate,
        audioFormat: requestInput?.audioFormat
      });
    } catch {
      writeLine(responseStream, { type: "error", message: "I couldn't understand that audio." });
      responseStream.end();
      return;
    }
  }

  if (!transcript) {
    writeLine(responseStream, { type: "error", message: "No speech detected." });
    responseStream.end();
    return;
  }

  if (!ownerId) {
    writeLine(responseStream, { type: "error", message: "This device is not linked to a household yet." });
    responseStream.end();
    return;
  }

  // Resolve user context
  let userContext;
  try {
    userContext = await lookupUserContextByOwnerId(ownerId, { env: process.env });
  } catch {
    writeLine(responseStream, { type: "error", message: "I couldn't load this household right now." });
    responseStream.end();
    return;
  }

  // Emit meta event immediately — client knows request is acknowledged
  writeLine(responseStream, { type: "meta", transcript, session_id: requestInput?.sessionId || null });

  // Load session history
  if (requestInput?.sessionId) {
    try {
      sessionMessages = await loadRecentSessionMessages(
        userContext.ownerId,
        requestInput.sessionId,
        getSessionMessageLimit(process.env),
        process.env
      );
      memoryUsed = sessionMessages.length > 0;
    } catch {
      sessionMessages = [];
    }
  }

  // Request dedup (same as sync handler)
  const shouldDedup = requestInput?.responseSurface === "app"
    && requestInput?.requestMode === "json_transcript"
    && Boolean(requestInput?.sessionId)
    && Boolean(transcript);

  if (shouldDedup) {
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
          // Serve cached response as a single done event
          const cached = requestClaim.existing.response_body;
          const text = cached?.app_output?.message?.text || cached?.text || "Done.";
          writeLine(responseStream, { type: "text_delta", delta: text });
          writeLine(responseStream, { type: "done", app_output: cached?.app_output || null });
          responseStream.end();
          return;
        }
        writeLine(responseStream, { type: "text_delta", delta: "I'm still finishing your last update. Give me a moment." });
        writeLine(responseStream, { type: "done", app_output: null });
        responseStream.end();
        return;
      }
    } catch {
      requestClaim = null;
    }
  }

  // Run the streaming assistant
  let result;
  try {
    const generator = runDeviceAssistantStreaming({
      transcript,
      userContext,
      env: process.env,
      sessionMessages,
      responseSurface: requestInput?.responseSurface || "halo",
      lambdaDeadline
    });

    // Forward events from the generator to the response stream
    let iterResult;
    do {
      iterResult = await generator.next();
      if (iterResult.value && !iterResult.done) {
        writeLine(responseStream, iterResult.value);
      }
      if (iterResult.done && iterResult.value) {
        result = iterResult.value;
      }
    } while (!iterResult.done);

  } catch (error) {
    console.error("[ERROR] streaming assistant failed:", error);
    if (requestClaim?.claimed && requestClaim?.entryId) {
      try { await clearRecentSessionRequest(userContext.ownerId, requestClaim.entryId, process.env); } catch {}
    }
    writeLine(responseStream, { type: "error", message: "I heard you, but I couldn't finish that request." });
    responseStream.end();
    return;
  }

  // Post-stream: entity capitalization, app_output, session persistence
  const toolEvents = result?.toolEvents || [];
  const originalResponseText = applyEntityCapitalization(result?.text || "Okay.", toolEvents);
    const responseText = requestInput?.responseSurface === "app"
      ? (formatRecipeResponse(originalResponseText) || originalResponseText) : originalResponseText;
  const durationMs = Date.now() - startedAt;

  const appOutput = buildAppOutput({
    text: responseText,
    transcript,
    quickItems: result?.quickItems || [],
    toolTrace: result?.toolTrace || [],
    toolEvents,
    type: result?.type || "info_answer",
    speechDetected: true,
    durationMs,
    sessionId: requestInput?.sessionId || null,
    memoryUsed,
    responseSurface: requestInput?.responseSurface || "halo"
  });

  // Write the final done event with the complete app_output
  writeLine(responseStream, {
    type: "done",
    final_text: responseText,
    app_output: appOutput,
    quick_items: result?.quickItems || [],
    duration_ms: durationMs
  });

  responseStream.end();

  // Fire-and-forget: session persistence, dedup completion, voice assets
  const postStreamTasks = [];

  if (requestInput?.sessionId) {
    postStreamTasks.push(
      appendSessionMessages(
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
          responseType: result?.type || "info_answer",
          toolEvents
        }),
        process.env
      ).catch((e) => console.warn("[WARN] session persist failed:", e?.message || e))
    );
  }

  if (requestClaim?.claimed && requestClaim?.entryId) {
    // Build the full response body for caching (matches sync handler format)
    const responseBody = {
      text: responseText,
      transcript,
      app_output: appOutput
    };
    postStreamTasks.push(
      completeRecentSessionRequest(userContext.ownerId, requestClaim.entryId, responseBody, process.env)
        .catch((e) => console.warn("[WARN] dedup complete failed:", e?.message || e))
    );
  }

  postStreamTasks.push(
    dispatchVoiceAssetJobs({ toolEvents, userContext, env: process.env })
      .catch((e) => console.warn("[WARN] voice asset dispatch failed:", e?.message || e))
  );

  // The response is already delivered (responseStream.end above). These are
  // fire-and-forget — cap how long we wait so a hung/locked session-store write
  // can never hold the invocation open to the 150s Lambda timeout (observed:
  // owner 30411 voice requests stuck at exactly 150000ms).
  let postStreamTimer;
  await Promise.race([
    Promise.allSettled(postStreamTasks),
    new Promise((resolve) => { postStreamTimer = setTimeout(resolve, 8000); }),
  ]);
  clearTimeout(postStreamTimer);
}

export const handler = awslambda.streamifyResponse(streamHandler);
