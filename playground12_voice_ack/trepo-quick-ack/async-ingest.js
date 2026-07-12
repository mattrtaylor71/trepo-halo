import { jsonResponse } from "./lib/http.mjs";
import {
  buildVoiceAsyncJobMetadata,
  createAcceptedVoiceJob,
  decodeAudioBody,
  getContentType,
  getHeader,
  getVoiceJob,
  invokeVoiceWorkerAsync,
  markVoiceJobEnqueued,
  normalizeAudioFormat,
  normalizeAudioSampleRate,
  normalizeResponseSurface,
  putVoiceAudioObject,
} from "./lib/voice-async.mjs";

const ACCEPTED_AUDIO_TYPES = new Set([
  "audio/pcm",
  "application/octet-stream",
  "audio/raw"
]);

function ttlDaysFromEnv(env) {
  const numeric = Number(env?.VOICE_ASYNC_JOB_TTL_DAYS || 14);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 14;
}

function buildAcceptedResponse(jobId, extra = {}) {
  return jsonResponse(202, {
    accepted: true,
    async: true,
    jobId,
    ...extra
  });
}

export async function handler(event) {
  const startedAt = Date.now();
  const headers = event?.headers || {};
  const ownerId = String(getHeader(headers, "x-owner-id") || "").trim();
  const deviceId = String(getHeader(headers, "x-device-id") || "").trim();
  const sessionId = String(getHeader(headers, "x-session-id") || "").trim() || null;
  const requestId = String(getHeader(headers, "x-request-id") || "").trim() || null;
  const responseSurface = normalizeResponseSurface(getHeader(headers, "x-client-surface"));
  const contentType = getContentType(headers);
  const audioSampleRate = normalizeAudioSampleRate(getHeader(headers, "x-audio-sample-rate"), 24000);
  const audioFormat = normalizeAudioFormat(getHeader(headers, "x-audio-format"));

  console.log("[DEBUG] async voice ingest received:", JSON.stringify({
    ownerId: ownerId || null,
    deviceId: deviceId || null,
    sessionId,
    requestId,
    responseSurface,
    contentType,
    audioSampleRate,
    audioFormat
  }));

  if (!ownerId) {
    return jsonResponse(400, { accepted: false, async: true, error: "missing_owner_id" });
  }

  if (!deviceId) {
    return jsonResponse(400, { accepted: false, async: true, error: "missing_device_id" });
  }

  if (!ACCEPTED_AUDIO_TYPES.has(contentType)) {
    return jsonResponse(400, {
      accepted: false,
      async: true,
      error: "unsupported_content_type",
      accepted_content_types: Array.from(ACCEPTED_AUDIO_TYPES)
    });
  }

  let audioBuffer;
  try {
    audioBuffer = decodeAudioBody(event);
  } catch (error) {
    console.error("[ERROR] async voice ingest decode failed:", error);
    return jsonResponse(error.statusCode || 400, { accepted: false, async: true, error: error.message || "invalid_audio_body" });
  }

  const { jobId, audioSha256, objectKey } = buildVoiceAsyncJobMetadata({
    ownerId,
    deviceId,
    sessionId,
    responseSurface,
    requestId,
    contentType,
    audioBuffer,
    audioSampleRate,
    audioFormat
  });

  const existingJob = await getVoiceJob(process.env, jobId);
  if (existingJob && ["accepted", "enqueued", "processing", "completed"].includes(existingJob.status)) {
    console.log("[DEBUG] async voice ingest duplicate accepted:", JSON.stringify({
      jobId,
      status: existingJob.status
    }));
    return buildAcceptedResponse(jobId, {
      duplicate: true,
      status: existingJob.status
    });
  }

  try {
    const bucketName = await putVoiceAudioObject(process.env, objectKey, audioBuffer, {
      owner_id: ownerId,
      device_id: deviceId,
      session_id: sessionId || "",
      response_surface: responseSurface,
      request_id: requestId || "",
      audio_sample_rate: audioSampleRate,
      audio_format: audioFormat
    });

    const nowIso = new Date().toISOString();
    const ttl = Math.floor(Date.now() / 1000) + (ttlDaysFromEnv(process.env) * 24 * 60 * 60);
    if (!existingJob) {
      try {
        await createAcceptedVoiceJob(process.env, {
          job_id: jobId,
          owner_id: ownerId,
          device_id: deviceId,
          session_id: sessionId,
          response_surface: responseSurface,
          content_type: contentType,
          request_id: requestId,
          audio_sample_rate: audioSampleRate,
          audio_format: audioFormat,
          audio_sha256: audioSha256,
          audio_bytes: audioBuffer.length,
          s3_bucket: bucketName,
          s3_key: objectKey,
          status: "accepted",
          accepted_at: nowIso,
          updated_at: nowIso,
          ttl
        });
      } catch (error) {
        if (error?.name !== "ConditionalCheckFailedException") {
          throw error;
        }

        const racedJob = await getVoiceJob(process.env, jobId);
        console.log("[DEBUG] async voice ingest duplicate after race:", JSON.stringify({
          jobId,
          status: racedJob?.status || null
        }));
        return buildAcceptedResponse(jobId, {
          duplicate: true,
          status: racedJob?.status || "accepted"
        });
      }
    }

    await invokeVoiceWorkerAsync(process.env, {
      jobId,
      ownerId,
      deviceId,
      sessionId,
      responseSurface,
      audioSampleRate,
      audioFormat
    });

    await markVoiceJobEnqueued(process.env, jobId, {
      enqueued_at: new Date().toISOString(),
      last_error: null
    });

    console.log("[DEBUG] async voice ingest accepted:", JSON.stringify({
      jobId,
      ownerId,
      deviceId,
      responseSurface,
      audioSampleRate,
      bytes: audioBuffer.length,
      elapsedMs: Date.now() - startedAt
    }));

    return buildAcceptedResponse(jobId, {
      duplicate: false
    });
  } catch (error) {
    console.error("[ERROR] async voice ingest failed:", error);
    return jsonResponse(500, {
      accepted: false,
      async: true,
      error: "enqueue_failed"
    });
  }
}
