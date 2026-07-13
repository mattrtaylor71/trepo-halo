#!/usr/bin/env bash
# trepo-obs-forwarder — CloudWatch Logs subscription filters
# Recorded by WP2. These are managed via CLI (not the SAM template) because the
# target log groups are owned by other, cross-stack CloudFormation stacks.
#
# Re-run this script to (re)create the subscription filters. Idempotent:
# put-subscription-filter overwrites a filter of the same name in place.
#
# NOTE: CloudWatch allows a maximum of 2 subscription filters per log group.
# All 20 targets had 0 existing filters at creation time (2026-07-02), so none
# were skipped and nothing was clobbered.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-trepo-dev}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_PAGER=""

DEST="arn:aws:lambda:us-east-1:566667681926:function:trepo-obs-error-forwarder"
FILTERNAME="trepo-obs-forwarder"
# Match JSON error markers (evt=analysis_failed / evt=backend_error) OR the
# legacy free-text "Background processing error" marker. '?' prefix = OR.
PATTERN='?analysis_failed ?backend_error ?ai_op ?"Background processing error"'

LOG_GROUPS=(
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeOnUpload-bpWcKEif3Gq7"
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeDishOnUpload-1wGSk6GmnxLv"
  "/aws/lambda/trepo-grocery-backend-dev-AnalyzeDiscardOnUpload-BnUyRpHHwaOx"
  "/aws/lambda/grocery-identifier-dev-identify-async"
  "/aws/lambda/grocery-identifier-dev-bulk-commit"
  "/aws/lambda/grocery-identifier-dev-enrich-kitchen-item"
  "/aws/lambda/trepo-list-handler"
  "/aws/lambda/twilioAuth"
  "/aws/lambda/trepo-quick-ack-async-worker-dev"
  "/aws/lambda/trepo-quick-ack-image-postprocess-dev"
  "/aws/lambda/trepo-quick-ack-sam-dev"
  "/aws/lambda/trepo-quick-ack-stream-dev"
  "/aws/lambda/trepo-notifications-SendFunction-mbwOf9XTzTmu"
  "/aws/lambda/trepo-notifications-RegisterFunction-J8du0ifBgrpo"
  "/aws/lambda/trepo-analytics-IngestFunction-HjM0blGGSDKn"
  "/aws/lambda/trepo-grocery-backend-dev-SavedRecipesApiFunction-a7aWuWlJvEF2"
  "/aws/lambda/trepo-grocery-backend-dev-RecipesGeneratorFunction-ycoZeX7W0vum"
  "/aws/lambda/trepo-grocery-backend-dev-MealPlanGeneratorFunctio-4rNstnlDTS1Z"
  "/aws/lambda/trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k"
  "/aws/lambda/trepo-grocery-backend-dev-DishesApiFunction-nf0KMidcpnmW"
  "/aws/lambda/trepo-grocery-backend-dev-DiscardsApiFunction-ySas82v8qBkQ"
)

for g in "${LOG_GROUPS[@]}"; do
  # Safety: abort if the group already has 2 filters and none is ours.
  existing=$(aws logs describe-subscription-filters --log-group-name "$g" \
    --query 'subscriptionFilters[].filterName' --output text 2>/dev/null || true)
  count=$(printf '%s\n' $existing | grep -c . || true)
  if [ "$count" -ge 2 ] && ! printf '%s\n' $existing | grep -qx "$FILTERNAME"; then
    echo "SKIP (2-filter limit) $g :: [$existing]" >&2
    continue
  fi
  aws logs put-subscription-filter \
    --log-group-name "$g" \
    --filter-name "$FILTERNAME" \
    --filter-pattern "$PATTERN" \
    --destination-arn "$DEST"
  echo "OK $g"
done
