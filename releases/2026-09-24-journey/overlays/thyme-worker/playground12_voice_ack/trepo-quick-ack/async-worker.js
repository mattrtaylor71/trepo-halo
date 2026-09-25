import { handler as syncVoiceHandler } from "./index.js";
import {
  claimVoiceJobForProcessing,
  assertVoiceJobAudio,
  getVoiceAudioObject,
  getVoiceJob,
  markVoiceJobCompleted,
  markVoiceJobFailed
} from "./lib/voice-async.mjs";

// Emits a structured marker for silent/degraded backend failures so a
// CloudWatch metric filter on "backend_error" can alert. Never throws.
function reportBackendError({ op, ownerId, code, err, jobId }) {
  try {
    console.error(JSON.stringify({
      evt: "backend_error",
      service: "voice",
      op,
      owner_id: ownerId || null,
      code: code || "error",
      error: String((err && (err.message || err)) || op).slice(0, 500),
      job_id: jobId || null
    }));
  } catch (_) { /* never let logging throw */ }
}

function buildSyntheticVoiceEvent(job) {
  return {
    version: "2.0",
    routeKey: "POST /voice-ack",
    rawPath: "/voice-ack",
    rawQueryString: "",
    headers: {
      "content-type": job.content_type || "audio/pcm",
      "x-owner-id": job.owner_id,
      "x-operation-id": "voice-job:" + job.job_id,
      "x-device-id": job.device_id || "unknown-device",
      "x-client-surface": job.response_surface || "halo",
      ...(job.audio_sample_rate ? { "x-audio-sample-rate": String(job.audio_sample_rate) } : {}),
      ...(job.audio_format ? { "x-audio-format": String(job.audio_format) } : {}),
      ...(job.session_id ? { "x-session-id": job.session_id } : {}),
      ...(job.request_id ? { "x-request-id": job.request_id } : {})
    },
    requestContext: {
      http: {
        method: "POST",
        path: "/voice-ack"
      }
    },
    body: null,
    isBase64Encoded: true
  };
}

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

async function processVoiceJob(jobId) {
  const job = await getVoiceJob(process.env, jobId);
  if (!job) {
    console.warn("[WARN] async voice worker missing job:", jobId);
    reportBackendError({ op: "process_voice_job", code: "job_missing", err: "voice job not found", jobId });
    return;
  }

  const claimed = await claimVoiceJobForProcessing(process.env, jobId);
  if (!claimed) {
    console.log("[DEBUG] async voice worker skipped job:", JSON.stringify({
      jobId,
      status: job.status || null
    }));
    return;
  }

  console.log("[DEBUG] async voice worker started:", JSON.stringify({
    jobId,
    ownerId: job.owner_id,
    deviceId: job.device_id,
    sessionId: job.session_id || null,
    responseSurface: job.response_surface || "halo",
    audioSampleRate: job.audio_sample_rate || null
  }));

  try {
    const audioBuffer = await getVoiceAudioObject(process.env, job.s3_key);
    assertVoiceJobAudio(job, audioBuffer);
    const event = buildSyntheticVoiceEvent(job);
    event.body = audioBuffer.toString("base64");
    const response = await syncVoiceHandler(event);
    const responseBody = safeJsonParse(response?.body);
    const hasError = Boolean(responseBody?.error);

    if (hasError) {
      await markVoiceJobFailed(process.env, jobId, {
        failed_at: new Date().toISOString(),
        last_error: String(responseBody.error || "unknown_error"),
        response_type: responseBody?.type || null,
        transcript: responseBody?.transcript || null
      });
      console.error("[ERROR] async voice worker completed with response error:", JSON.stringify({
        jobId,
        error: responseBody?.error || null
      }));
      reportBackendError({
        op: "process_voice_job",
        ownerId: job.owner_id,
        code: responseBody?.type ? `response_error:${responseBody.type}` : "response_error",
        err: responseBody?.error || "unknown_error",
        jobId
      });
      return;
    }

    await markVoiceJobCompleted(process.env, jobId, {
      completed_at: new Date().toISOString(),
      response_type: responseBody?.type || null,
      transcript: responseBody?.transcript || null,
      response_text: responseBody?.text ? String(responseBody.text).slice(0, 2000) : null,
      outcome: responseBody?.outcome || null,
      operation_id: responseBody?.operation_id || "voice-job:" + job.job_id,
      last_error: null
    });

    console.log("[DEBUG] async voice worker completed:", JSON.stringify({
      jobId,
      type: responseBody?.type || null
    }));
  } catch (error) {
    await markVoiceJobFailed(process.env, jobId, {
      failed_at: new Date().toISOString(),
      last_error: String(error?.message || error || "worker_failed")
    });
    console.error("[ERROR] async voice worker failed:", JSON.stringify({
      jobId,
      error: error?.message || error
    }));
    reportBackendError({ op: "process_voice_job", ownerId: job.owner_id, code: "worker_exception", err: error, jobId });
    throw error;
  }
}

export async function handler(event) {
  const jobId = String(event?.jobId || "").trim();
  if (!jobId) {
    console.warn("[WARN] async voice worker skipped invoke without jobId");
    return;
  }
  await processVoiceJob(jobId);
}
