# trepo-alarm-triage

Tier-1 real-time CloudWatch alarm triage. When any alarm fires on SNS topic
`trepo-capture-alerts`, this Lambda pulls the relevant error logs, asks Claude
(`claude-opus-4-8`, adaptive thinking, structured output) for a diagnosis, and
publishes an **enriched** report to SNS topic `trepo-triage-reports`
(matt@trepo.ai subscribed) instead of the raw AWS alarm.

## Flow
1. SNS (`trepo-capture-alerts`) → this Lambda, one record per alarm.
2. Skip unless `NewStateValue == "ALARM"`; hard-skip any alarm name containing
   `triage` (self-loop guard).
3. Resolve the implicated log group(s):
   - `FunctionName` dimension → that fn's `/aws/lambda/...` group.
   - `Trepo/Capture` namespace → static metric→log-group map, else
     `logs:DescribeMetricFilters` (authoritative) at runtime.
   - account-wide / 5xx / unmapped → sweep the core fleet (`CORE_LOG_GROUPS`).
4. `FilterLogEvents` ~15-min window for error-ish lines (+ ±30s context around
   the first hit), capped at ~15KB.
5. Claude → structured diagnosis `{severity, root_cause_hypothesis,
   affected_scope, self_healed, evidence_summary, recommended_action,
   confidence}`.
6. Publish a plain-text report to `trepo-triage-reports` with a deep link to the
   alarm. Also `console.log`'d to this fn's own logs (searchable corpus).
7. **Failure-safe:** everything is wrapped; on any error it publishes a fallback
   ("triage failed: …; raw alarm attached") and exits 0 — a throwing triage fn
   would itself feed the account error alarm.

## Deployed resources (account 566667681926, us-east-1)
- Lambda `trepo-alarm-triage` — nodejs22.x, 512MB, 60s timeout, reserved
  concurrency 2. Env: `ANTHROPIC_API_KEY` (copied from
  `trepo-analytics-ai-overview`), `TRIAGE_TOPIC_ARN`.
- IAM role `trepo-alarm-triage-role`, inline policy `triage-least-privilege`:
  `logs:FilterLogEvents`+`DescribeMetricFilters` on `log-group:*`;
  `cloudwatch:GetMetricStatistics`+`DescribeAlarms`; `sns:Publish` on the
  triage-reports topic only; own-log-group write.
- SNS topic `trepo-triage-reports` + email subscription `matt@trepo.ai`.
- SNS subscription `trepo-capture-alerts` → this Lambda (+ invoke permission).

Created via `deploy.sh` (raw CLI, not a SAM stack, to stay surgical and
independent of the large backend stacks).

## Redeploy (code only)
```
cd obs/alarm-triage
npm install --omit=dev
bash deploy.sh   # zips + update-function-code (creates infra on first run)
```

## Notes
- `CORE_LOG_GROUPS` / `STATIC_METRIC_LOG_GROUPS` hold physical log-group names.
  If a stack recreates a function (new random suffix), update those constants;
  the runtime `DescribeMetricFilters` path self-heals Trepo/Capture metrics.
- The email subscription requires Matt to click the confirmation link before
  reports are delivered.
