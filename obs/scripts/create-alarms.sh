#!/usr/bin/env bash
#
# create-alarms.sh — Baseline Trepo observability CloudWatch alarms (WP1)
#
# Idempotent: put-metric-alarm is upsert-by-name, so re-running is safe.
# All alarms: Sum >= 1 over 300s (1 eval period), TreatMissingData=notBreaching,
# both alarm-actions AND ok-actions -> trepo-capture-alerts SNS topic.
# (Exceptions: gemini-live-tasks uses Minimum/LessThan/breaching — see below.)
#
# Account: 566667681926   Region: us-east-1
# Naming prefix: trepo-obs-*   (does NOT touch trepo-capture-*/trepo-list-* alarms)
#
# Usage:  AWS_PROFILE=trepo-dev AWS_REGION=us-east-1 ./create-alarms.sh
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-trepo-dev}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
SNS="arn:aws:sns:us-east-1:566667681926:trepo-capture-alerts"

# Helper for the standard "Sum>=1 / 300s / 1 period / notBreaching" alarms.
put_std() {
  # $1=name  $2=namespace  $3=metric  $4=description  then dimension args...
  local name="$1" ns="$2" metric="$3" desc="$4"; shift 4
  aws cloudwatch put-metric-alarm \
    --alarm-name "$name" \
    --alarm-description "$desc" \
    --namespace "$ns" \
    --metric-name "$metric" \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$SNS" \
    --ok-actions "$SNS" \
    "$@"
  echo "  ok: $name"
}

echo "== 1. API Gateway 5xx alarms (all HTTP/v2 APIs -> AWS/ApiGateway metric '5xx', dim ApiId) =="
# short-name : api-id
declare -a APIS=(
  "main-grocery:7tn3gvwvh7"
  "grocery-identifier:6m9t6wosh9"
  "twilio-auth:1zc0nh8x48"
  "analytics:9s5i5b7lfi"
  "notifications:nua4yt5q26"
  "quick-ack:ivwu7ls6p8"
  "quick-ack-2:qq5tn5i3t3"
  "feedback:hbricl6rka"
  "explore:pzpx2y2qph"
  "realtime-voice:fo4cjcmqs7"
)
for entry in "${APIS[@]}"; do
  sn="${entry%%:*}"; id="${entry##*:}"
  put_std "trepo-obs-5xx-${sn}" "AWS/ApiGateway" "5xx" \
    "HTTP API ${sn} (ApiId ${id}) returned a 5xx in a 5-min window" \
    --dimensions Name=ApiId,Value="${id}"
done

echo "== 2. Account-wide Lambda Errors catch-all =="
put_std "trepo-obs-lambda-errors-account" "AWS/Lambda" "Errors" \
  "Account-wide Lambda Errors catch-all (NO dimensions). May be NOISY from experimental/one-off functions; use as a coarse tripwire, not a page."

echo "== 3. Lambda Function URL 5xx alarms (AWS/Lambda 'Url5xxCount', dim FunctionName) =="
# short-suffix : full function name
declare -a URLFNS=(
  "quick-ack-stream:trepo-quick-ack-stream-dev"
  "device-claim:device_claim"
  "internal-chatbot:trepo-analytics-InternalChatbotFunction-h9Nx6KVAVTmi"
  "shopping-assistant-stream:trepo-analytics-ShoppingAssistantStreamFunction-NFhlQmKCkewz"
  "tiktok-assistant:tiktok-assistant-TikTokAssistantFunction-0d2O9Knw5Qfy"
)
for entry in "${URLFNS[@]}"; do
  sn="${entry%%:*}"; fn="${entry##*:}"
  put_std "trepo-obs-urlerrors-${sn}" "AWS/Lambda" "Url5xxCount" \
    "Function URL 5xx for ${fn}" \
    --dimensions Name=FunctionName,Value="${fn}"
done

echo "== 4. Gemini Live ECS/ALB =="
GEMINI_LB="app/trepo-gemini-live-backend-alb/99a336e3a71c085c"
GEMINI_TG="targetgroup/tg-trepo-gemini-live-backend-ecs/90b119d98f469d73"

# 4a. ALB target 5xx (standard Sum>=1/300/notBreaching)
put_std "trepo-obs-gemini-live-5xx" "AWS/ApplicationELB" "HTTPCode_Target_5XX_Count" \
  "Gemini Live backend ALB returned a target 5xx in a 5-min window" \
  --dimensions Name=LoadBalancer,Value="${GEMINI_LB}" Name=TargetGroup,Value="${GEMINI_TG}"

# 4b. ECS running-task proxy. ContainerInsights is NOT enabled (no ECS/ContainerInsights
#     RunningTaskCount metric), so we alarm on ALB HealthyHostCount < 1 (no healthy
#     backend task registered). NOTE: Minimum / LessThanThreshold / breaching — differs
#     from the standard alarm shape on purpose.
aws cloudwatch put-metric-alarm \
  --alarm-name "trepo-obs-gemini-live-tasks" \
  --alarm-description "Gemini Live backend has 0 healthy tasks/hosts registered to the target group (proxy for RunningTaskCount; ContainerInsights not enabled)." \
  --namespace "AWS/ApplicationELB" \
  --metric-name "HealthyHostCount" \
  --statistic Minimum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --dimensions Name=LoadBalancer,Value="${GEMINI_LB}" Name=TargetGroup,Value="${GEMINI_TG}" \
  --alarm-actions "$SNS" \
  --ok-actions "$SNS"
echo "  ok: trepo-obs-gemini-live-tasks"

echo "Done. Verify: aws cloudwatch describe-alarms --alarm-name-prefix trepo-obs"
