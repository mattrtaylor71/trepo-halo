// trepo-alarm-triage — Tier-1 real-time alarm triage.
//
// Subscribed to SNS topic trepo-capture-alerts. For each CloudWatch alarm that
// fires (NewStateValue=ALARM), pull the relevant logs, ask Claude for a
// diagnosis, and publish an ENRICHED report to SNS topic trepo-triage-reports
// (matt@trepo.ai subscribed). Wraps EVERYTHING and always exits 0 — a throwing
// triage fn would itself feed the account error alarm.

import {
  CloudWatchLogsClient,
  FilterLogEventsCommand,
  DescribeMetricFiltersCommand,
} from "@aws-sdk/client-cloudwatch-logs";
import { SNSClient, PublishCommand } from "@aws-sdk/client-sns";
import Anthropic from "@anthropic-ai/sdk";

const REGION = process.env.AWS_REGION || "us-east-1";
const TRIAGE_TOPIC_ARN = process.env.TRIAGE_TOPIC_ARN;
const MODEL = process.env.TRIAGE_MODEL || "claude-opus-4-8";

const logs = new CloudWatchLogsClient({ region: REGION });
const sns = new SNSClient({ region: REGION });

const LOOKBACK_MS = 15 * 60 * 1000; // ~15 min evidence window
const MAX_LOG_BYTES = 15000; // cap total log text at ~15KB
const MAX_HITS_PER_GROUP = 20;
const CONTEXT_WINDOW_MS = 30 * 1000; // ±30s around the first hit for surrounding lines

// Error-ish terms we grep the logs for (CloudWatch `?term` = OR match).
const ERROR_TERMS = [
  "ERROR", "errorMessage", "backend_error", "Status: timeout", "insufficient_quota",
  "Traceback", "FetchError", "request_failed", "web_fetch_blocked", "instagram_url_rejected",
  "Task timed out", "Runtime.", "No structured output", "Deadlock",
];
const ERROR_FILTER_PATTERN = ERROR_TERMS.map((t) => `?"${t}"`).join(" ");

// Known Trepo/Capture metric -> log group(s). This is a fast-path/fallback;
// the authoritative resolution is DescribeMetricFilters at runtime (below).
const STATIC_METRIC_LOG_GROUPS = {
  RecipesBackendError: ["/aws/lambda/trepo-grocery-backend-dev-RecipesGeneratorFunction-ycoZeX7W0vum", "/aws/lambda/trepo-grocery-backend-dev-RecipesApiFunction-zjwZ1RP9bUAQ"],
  BulkIdentifyAnalysisFailed: ["/aws/lambda/grocery-identifier-dev-identify-async"],
  BulkCommitFailed: ["/aws/lambda/grocery-identifier-dev-bulk-commit"],
  KitchenBackendError: ["/aws/lambda/trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k"],
  KitchenItemsDropped: ["/aws/lambda/grocery-identifier-dev-bulk-commit", "/aws/lambda/trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k"],
  DishAnalysisFailed: ["/aws/lambda/trepo-grocery-backend-dev-AnalyzeDishOnUpload-1wGSk6GmnxLv"],
  DiscardAnalysisFailed: ["/aws/lambda/trepo-grocery-backend-dev-AnalyzeDiscardOnUpload-BnUyRpHHwaOx"],
  GroceryAnalysisFailed: ["/aws/lambda/trepo-grocery-backend-dev-AnalyzeOnUpload-bpWcKEif3Gq7"],
  AnalyticsIngestFailed: ["/aws/lambda/trepo-analytics-IngestFunction-HjM0blGGSDKn"],
  // Emitted by PutMetricData from the canary (no metric filter -> DescribeMetricFilters
  // can't find it), so it MUST be mapped statically to the canary's own log group.
  VoiceDualWriteMiss: ["/aws/lambda/trepo-voice-dualwrite-canary"],
  AssistantFalseConfirm: [
    "/aws/lambda/trepo-quick-ack-stream-dev",
    "/aws/lambda/trepo-quick-ack-async-worker-dev",
    "/aws/lambda/trepo-quick-ack-sam-dev",
  ],
  // OpenAIQuotaExceeded is emitted by many services -> sweep the OpenAI-dependent set.
  OpenAIQuotaExceeded: [
    "/aws/lambda/grocery-identifier-dev-identify-async",
    "/aws/lambda/grocery-identifier-dev-enrich-kitchen-item",
    "/aws/lambda/trepo-grocery-backend-dev-RecipesGeneratorFunction-ycoZeX7W0vum",
    "/aws/lambda/trepo-grocery-backend-dev-SavedRecipesApiFunction-a7aWuWlJvEF2",
    "/aws/lambda/trepo-grocery-backend-dev-AnalyzeOnUpload-bpWcKEif3Gq7",
  ],
};

// Core fn log groups swept for account-wide / 5xx / unmapped alarms.
const CORE_LOG_GROUPS = [
  "/aws/lambda/trepo-quick-ack-stream-dev",
  "/aws/lambda/trepo-quick-ack-async-worker-dev",
  "/aws/lambda/trepo-quick-ack-sam-dev",
  "/aws/lambda/trepo-quick-ack",
  "/aws/lambda/trepo-list-handler",
  "/aws/lambda/trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k",
  "/aws/lambda/trepo-grocery-backend-dev-SavedRecipesApiFunction-a7aWuWlJvEF2",
  "/aws/lambda/trepo-grocery-backend-dev-RecipesGeneratorFunction-ycoZeX7W0vum",
  "/aws/lambda/trepo-grocery-backend-dev-RecipesApiFunction-zjwZ1RP9bUAQ",
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeDishOnUpload-1wGSk6GmnxLv",
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeDiscardOnUpload-BnUyRpHHwaOx",
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeOnUpload-bpWcKEif3Gq7",
  "/aws/lambda/grocery-identifier-dev-identify-async",
  "/aws/lambda/grocery-identifier-dev-bulk-commit",
  "/aws/lambda/grocery-identifier-dev-enrich-kitchen-item",
  "/aws/lambda/trepo-analytics-SummaryFunction-r16bAHjy96zm",
  // Slow fn behind most 5xx-main-grocery 504s (25-29s -> API GW 30s cap).
  "/aws/lambda/trepo-grocery-backend-dev-HomeSuggestionsFunction-eq77AzDop23s",
  // Non-capture fns that also feed the account-wide error alarm (were coming back
  // 'unattributable').
  "/aws/lambda/halo-live-dashboard-updater",
];

const DIAGNOSIS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "severity", "root_cause_hypothesis", "affected_scope", "self_healed",
    "evidence_summary", "recommended_action", "confidence",
  ],
  properties: {
    severity: { type: "string", enum: ["critical", "degraded", "self_healed", "noise"] },
    root_cause_hypothesis: { type: "string" },
    affected_scope: { type: "string" },
    self_healed: { type: "boolean" },
    evidence_summary: { type: "string" },
    recommended_action: { type: "string" },
    confidence: { type: "string", enum: ["high", "medium", "low"] },
  },
};

const SYSTEM_PROMPT = `You are the on-call triage engineer for Trepo, a consumer food/kitchen app (iOS + serverless AWS backend). A CloudWatch alarm fired; you receive the alarm JSON plus recent error logs from the implicated service(s). Diagnose it. Be honest — "noise / self-healed, no action" is a valid and COMMON verdict.

SYSTEM MAP
- 4 API gateways front the backend: main-grocery, grocery-identifier, quick-ack (capture ingest), analytics. Plus a realtime-voice ELB and gemini-live service.
- Core lambdas: quick-ack stream/async-worker/sam/ingest (capture upload + async fan-out); KitchenApi (kitchen CRUD, /kitchen, text-add); SavedRecipesApi (recipe URL/IG/TikTok saves); RecipesGenerator + RecipesApi (recipe generation from kitchen); AnalyzeOnUpload / AnalyzeDishOnUpload / AnalyzeDiscardOnUpload (photo analysis via Gemini); grocery-identifier identify-async + bulk-commit + enrich-kitchen-item (bulk grocery/receipt scan -> kitchen); trepo-list-handler; analytics Summary/Ingest.
- Data stores: MySQL (shared_kitchen, shared_shopping_list, {owner}_prod_kitchen/*_archive tables) on RDS database-1; grocery-identifier-dev-jobs DynamoDB (async job state).
- AI deps: OpenAI (ONE shared key) powers text parsing, recipe generation, storage-guidance, enrichment — an OpenAI quota outage takes ALL OpenAI-dependent features down at once. Gemini 3.1-pro does photo identify (occasional network-timeout hangs on identify jobs). Apify carries ALL Instagram extraction — if the Apify cap is hit, IG recipe saves soft-fail 422.

KNOWN FAILURE CLASSES (match the logs to these)
- OpenAI insufficient_quota -> AI features down until quota resets/tops up. Broad blast radius. Usually CRITICAL/degraded, not self-healed.
- Gemini "network timeout" on an identify job -> the specific capture job hangs; it is re-drivable from the persisted S3 source image. Usually degraded + self_healed-able (retry).
- MySQL metadata-lock "Status: timeout" / Deadlock -> transient DB contention; often self-heals on retry.
- Recipes generator JSON-parse backend_error (evt=backend_error) -> a single generation blipped; the recipes refresh flag re-drives it, so it typically SELF-HEALS. Often noise.
- SavedRecipes 502 web_fetch_blocked -> a recipe site bot-walled our fetch. Graceful, per-URL, not systemic. noise/degraded.
- SavedRecipes instagram_url_rejected -> user shared a non-content IG URL (profile/login). Gate working as intended. noise.
- enrich "No structured output" -> transient OpenAI response; Lambda async retries usually heal it. self_healed/noise.
- trepo-capture-voice-dualwrite-miss (VoiceDualWriteMiss metric, from trepo-voice-dualwrite-canary): fires when a voice write's per-user app-visible copy is missing. BUT any row/_id/household_item_uuid prefixed 'canary-fault-test-' is a SYNTHETIC fault-injection self-test, NOT a real miss — if the canary logs / missing rows are only these prefixed ids, the verdict is noise/validation (the canary is testing itself) and the recommended_action is "none — synthetic self-test". NEVER recommend a data backfill for canary-fault-test- rows. Only genuine (non-prefixed) missing rows are a real dual-write regression worth acting on.
- 5xx-main-grocery / account-error alarms are OFTEN API Gateway 504s from a SLOW function, NOT an error log. API Gateway hard-times-out any integration at 30s. So a "SLOW INVOCATIONS" evidence block with a fn whose Duration ≈ 25000-30000ms (or a "Task timed out") IS the root cause — that fn tipped over the 30s cap and returned 504. The usual culprit is HomeSuggestions. In that case: severity degraded (not noise), root cause = "<fn> exceeds the 30s API Gateway timeout", recommended_action names the slow fn. Do NOT call a 504 "noise" just because there are no ERROR lines — a timeout leaves no error line, only a slow REPORT.

INSTRUCTIONS
- Use the RECURRENCE CHECK line: only call something self_healed when it says no matching errors in the last 10 min. If it says the error is STILL appearing, it has NOT self-healed — set self_healed=false and severity degraded/critical.
- Decide whether the incident ALREADY self-healed (async Lambda retries, recipe refresh flags, job re-drive) — set self_healed accordingly, but never over the RECURRENCE CHECK.
- Name the affected scope concretely from log fields: which owners/user_ids, job_ids, or features are hit, and roughly how many.
- Give exactly ONE concrete recommended_action (or "none — self-healed/noise" when that's the truth).
- severity: critical = active user-facing outage; degraded = partial/one feature or subset of users; self_healed = fired but already recovered; noise = expected/benign/single transient.
- confidence reflects how strongly the logs support your hypothesis. If logs are empty or unrelated, say so and lower confidence.`;

// ---------------------------------------------------------------------------

function clampSubject(s) {
  // SNS subject: ASCII printable, no newlines, <=100 chars.
  let out = (s || "").replace(/[\r\n\t]+/g, " ").replace(/[^\x20-\x7E]+/g, "").trim();
  if (out.length > 100) out = out.slice(0, 99).trimEnd() + "…"; // will be re-sanitized below
  out = out.replace(/[^\x20-\x7E]+/g, "");
  return out.slice(0, 100) || "Trepo alarm";
}

// Extract the underlying metric(s) from an alarm Trigger. Handles BOTH a simple
// alarm (top-level Namespace/MetricName) AND a metric-math/expression alarm, where
// the real metrics live in Trigger.Metrics[] (each with MetricStat.Metric or an
// Expression). Without this, math alarms render "undefined/undefined" and resolve
// to no log group.
function describeMetrics(trigger) {
  trigger = trigger || {};
  const dimStr = (dims) => (dims || []).map((d) => d.value || d.Value).filter(Boolean).join(",");
  if (trigger.MetricName) {
    return {
      label: `${trigger.Namespace || "?"}/${trigger.MetricName}${(trigger.Dimensions || []).length ? `[${dimStr(trigger.Dimensions)}]` : ""}`,
      metrics: [{ namespace: trigger.Namespace, metricName: trigger.MetricName, dimensions: trigger.Dimensions || [] }],
    };
  }
  const arr = Array.isArray(trigger.Metrics) ? trigger.Metrics : [];
  const metrics = [];
  const parts = [];
  for (const m of arr) {
    const met = m && m.MetricStat && m.MetricStat.Metric;
    if (met) {
      metrics.push({ namespace: met.Namespace, metricName: met.MetricName, dimensions: met.Dimensions || [] });
      parts.push(`${met.Namespace || "?"}/${met.MetricName || "?"}${(met.Dimensions || []).length ? `[${dimStr(met.Dimensions)}]` : ""}`);
    } else if (m && m.Expression) {
      parts.push(`expr(${m.Id || "?"})=${m.Expression}`);
    }
  }
  return { label: parts.length ? parts.join(" ; ") : "(no parseable metric)", metrics };
}

async function resolveLogGroups(alarm) {
  const trigger = alarm.Trigger || {};
  const dims = trigger.Dimensions || [];
  const fnDim = dims.find((d) => d.name === "FunctionName" || d.Name === "FunctionName");
  const fnName = fnDim && (fnDim.value || fnDim.Value);
  if (fnName) return { groups: [`/aws/lambda/${fnName}`], mode: "function_dimension" };

  // Works for simple AND metric-math alarms (Metrics[] may hold several metrics).
  const { metrics } = describeMetrics(trigger);
  const capMetrics = metrics.filter((m) => m.namespace === "Trepo/Capture" && m.metricName);
  if (capMetrics.length) {
    const groups = new Set();
    for (const m of capMetrics) {
      if (STATIC_METRIC_LOG_GROUPS[m.metricName]) {
        STATIC_METRIC_LOG_GROUPS[m.metricName].forEach((g) => groups.add(g));
        continue;
      }
      // Authoritative: which log group's metric filter emits this metric?
      try {
        const r = await logs.send(new DescribeMetricFiltersCommand({ metricName: m.metricName, metricNamespace: m.namespace }));
        (r.metricFilters || []).map((f) => f.logGroupName).filter(Boolean).forEach((g) => groups.add(g));
      } catch (e) {
        console.warn("[triage] DescribeMetricFilters failed:", e && e.message);
      }
    }
    if (groups.size) return { groups: [...groups], mode: capMetrics.length > 1 ? "math_metric_map" : "metric_map" };
  }
  // Account-wide / 5xx / unmapped -> sweep the core fleet for error lines.
  return { groups: CORE_LOG_GROUPS, mode: "core_sweep" };
}

async function filterOnce(params) {
  try {
    const r = await logs.send(new FilterLogEventsCommand(params));
    return r.events || [];
  } catch (e) {
    console.warn(`[triage] FilterLogEvents ${params.logGroupName} failed:`, e && e.message);
    return [];
  }
}

async function gatherLogs(groups, endMs) {
  const startMs = endMs - LOOKBACK_MS;
  const chunks = [];
  let total = 0;
  for (const lg of groups) {
    if (total >= MAX_LOG_BYTES) break;
    const hits = await filterOnce({
      logGroupName: lg, startTime: startMs, endTime: endMs,
      filterPattern: ERROR_FILTER_PATTERN, limit: MAX_HITS_PER_GROUP, interleaved: true,
    });
    if (!hits.length) continue;
    const lines = [];
    // Surrounding context for the FIRST hit (±30s, same stream, no pattern).
    const first = hits[0];
    if (first.logStreamName) {
      const ctx = await filterOnce({
        logGroupName: lg, logStreamNames: [first.logStreamName],
        startTime: first.timestamp - CONTEXT_WINDOW_MS, endTime: first.timestamp + CONTEXT_WINDOW_MS,
        limit: 40, interleaved: true,
      });
      for (const e of ctx) lines.push(e.message);
    }
    for (const e of hits) lines.push(e.message);
    const seen = new Set();
    const uniq = lines.map((m) => (m || "").trimEnd()).filter((m) => m && !seen.has(m) && seen.add(m));
    let block = `### ${lg} (${hits.length} error hit(s))\n` + uniq.join("\n");
    if (total + block.length > MAX_LOG_BYTES) block = block.slice(0, Math.max(0, MAX_LOG_BYTES - total));
    chunks.push(block);
    total += block.length;
  }
  return chunks.join("\n\n").slice(0, MAX_LOG_BYTES);
}

// Slow-invocation evidence for 5xx / latency alarms. A Lambda REPORT line whose
// Duration is near API Gateway's 30s hard cap (or a "Task timed out") is the fn that
// caused the 504 — usually HomeSuggestions. This is the log-side proxy for the
// ApiGateway IntegrationLatency=30000ms metric, without a metrics SDK dependency.
async function gatherSlowReports(groups, endMs) {
  const startMs = endMs - LOOKBACK_MS;
  const slow = [];
  for (const lg of groups) {
    const reports = await filterOnce({
      logGroupName: lg, startTime: startMs, endTime: endMs,
      filterPattern: '?"REPORT RequestId" ?"Task timed out"', limit: 30, interleaved: true,
    });
    for (const e of reports) {
      const m = /Billed Duration:\s*([\d.]+)\s*ms/.exec(e.message || "") || /Duration:\s*([\d.]+)\s*ms/.exec(e.message || "");
      const dur = m ? parseFloat(m[1]) : (/Task timed out/.test(e.message || "") ? 999999 : 0);
      if (dur >= 15000) slow.push({ lg, dur, msg: (e.message || "").trim() });
    }
  }
  if (!slow.length) return "";
  slow.sort((a, b) => b.dur - a.dur);
  const lines = slow.slice(0, 8).map((s) => `${s.lg.split("/").pop()}  Duration≈${Math.round(s.dur)}ms  ${s.msg.slice(0, 110)}`);
  return "### SLOW INVOCATIONS (Duration ≥15s — a fn behind API Gateway near ~30000ms IS the 504 culprit):\n" + lines.join("\n");
}

// Stale self-heal guard: is the same error pattern STILL firing right up to now?
async function recentRecurrenceNote(groups, nowMs) {
  const startMs = nowMs - 10 * 60 * 1000;
  let hits = 0;
  for (const lg of groups) {
    const r = await filterOnce({ logGroupName: lg, startTime: startMs, endTime: nowMs, filterPattern: ERROR_FILTER_PATTERN, limit: 5, interleaved: true });
    hits += r.length;
    if (hits >= 3) break;
  }
  return hits > 0
    ? `RECURRENCE CHECK: the error pattern is STILL appearing — ${hits}+ matching line(s) in the last 10 min. Do NOT declare this self_healed.`
    : "RECURRENCE CHECK: no matching error lines in the last 10 min (consistent with self_healed).";
}

async function diagnose(alarm, evidence, resolveMode) {
  const anthropic = new Anthropic(); // ANTHROPIC_API_KEY from env
  const userContent =
    `ALARM:\n${JSON.stringify({
      AlarmName: alarm.AlarmName,
      AlarmDescription: alarm.AlarmDescription,
      NewStateValue: alarm.NewStateValue,
      NewStateReason: alarm.NewStateReason,
      StateChangeTime: alarm.StateChangeTime,
      Trigger: alarm.Trigger,
    }, null, 2)}\n\n` +
    `LOG-GROUP RESOLUTION: ${resolveMode}\n\n` +
    `RECENT ERROR LOGS (~15 min window, may be empty):\n` +
    (evidence ? evidence : "(no matching error lines found in the implicated log group(s))");

  const resp = await anthropic.messages.create({
    model: MODEL,
    max_tokens: 8000,
    thinking: { type: "adaptive" },
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userContent }],
    output_config: { format: { type: "json_schema", schema: DIAGNOSIS_SCHEMA } },
  });
  const textBlock = (resp.content || []).find((b) => b.type === "text");
  if (!textBlock || !textBlock.text) throw new Error("Claude returned no text block");
  return JSON.parse(textBlock.text);
}

const SEV_EMOJI = { critical: "🔴", degraded: "🟠", self_healed: "🟢", noise: "⚪" };

function buildReport(alarm, diag, evidence, resolveMode) {
  const alarmName = alarm.AlarmName || "unknown-alarm";
  const link = `https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#alarmsV2:alarm/${encodeURIComponent(alarmName)}`;
  const emoji = SEV_EMOJI[diag.severity] || "🔍";
  const evExcerpt = (evidence || "(none)").split("\n").slice(0, 25).join("\n").slice(0, 3500);
  const body =
`${emoji} TRIAGE — ${alarmName}
severity: ${diag.severity}   self-healed: ${diag.self_healed ? "yes" : "no"}   confidence: ${diag.confidence}

WHAT FIRED
${alarm.AlarmDescription || "(no description)"}
reason: ${alarm.NewStateReason || "(none)"}
at: ${alarm.StateChangeTime || "(unknown)"}   metric: ${describeMetrics(alarm.Trigger).label}

DIAGNOSIS
${diag.root_cause_hypothesis}

AFFECTED SCOPE
${diag.affected_scope}

EVIDENCE
${diag.evidence_summary}

RECOMMENDED ACTION
${diag.recommended_action}

--- log excerpt (resolution: ${resolveMode}) ---
${evExcerpt}

alarm: ${link}`;
  const subject = clampSubject(`${emoji} [${diag.severity}] ${alarmName}: ${diag.root_cause_hypothesis}`);
  return { subject, body };
}

async function publish(subject, message) {
  await sns.send(new PublishCommand({ TopicArn: TRIAGE_TOPIC_ARN, Subject: subject, Message: message }));
}

async function publishFallback(alarm, err, raw) {
  try {
    const name = (alarm && alarm.AlarmName) || "unknown-alarm";
    const subject = clampSubject(`⚠️ triage failed: ${name}`);
    const message =
`⚠️ TRIAGE FAILED for ${name}
error: ${err && (err.stack || err.message || String(err))}

raw alarm attached:
${raw || (alarm ? JSON.stringify(alarm, null, 2) : "(unparseable SNS message)")}`;
    await publish(subject, message);
  } catch (e) {
    console.error("[triage] fallback publish ALSO failed:", e && e.message);
  }
}

async function handleRecord(record) {
  const raw = record && record.Sns && record.Sns.Message;
  let alarm;
  try {
    alarm = JSON.parse(raw);
  } catch (e) {
    await publishFallback(null, new Error("SNS message is not JSON: " + (e && e.message)), raw);
    return;
  }

  // Self-loop guard + only process real ALARM transitions.
  if ((alarm.AlarmName || "").toLowerCase().includes("triage")) {
    console.log("[triage] skipping self/triage alarm:", alarm.AlarmName);
    return;
  }
  if (alarm.NewStateValue !== "ALARM") {
    console.log(`[triage] skipping ${alarm.AlarmName} state=${alarm.NewStateValue}`);
    return;
  }

  try {
    const endMs = alarm.StateChangeTime ? Date.parse(alarm.StateChangeTime) || Date.now() : Date.now();
    const { groups, mode } = await resolveLogGroups(alarm);
    const baseEvidence = await gatherLogs(groups, endMs);
    // For 5xx/account sweeps, add slow-invocation evidence (which fn tipped over the
    // API GW 30s cap). Always add a recurrence check so a stale error isn't called
    // self-healed.
    const [slowReports, recurrence] = await Promise.all([
      mode === "core_sweep" ? gatherSlowReports(groups, endMs) : Promise.resolve(""),
      recentRecurrenceNote(groups, Date.now()),
    ]);
    const evidence = [baseEvidence, slowReports, recurrence].filter(Boolean).join("\n\n");
    const diag = await diagnose(alarm, evidence, mode);
    const { subject, body } = buildReport(alarm, diag, evidence, mode);
    console.log("[triage] report:\n" + subject + "\n" + body); // corpus in our own logs
    await publish(subject, body);
  } catch (e) {
    console.error("[triage] triage error for", alarm.AlarmName, e && (e.stack || e.message));
    await publishFallback(alarm, e, raw);
  }
}

export const handler = async (event) => {
  const records = (event && event.Records) || [];
  for (const record of records) {
    try {
      await handleRecord(record);
    } catch (e) {
      console.error("[triage] unexpected record error:", e && (e.stack || e.message));
      try { await publishFallback(null, e, record && record.Sns && record.Sns.Message); } catch {}
    }
  }
  return { ok: true, processed: records.length };
};
