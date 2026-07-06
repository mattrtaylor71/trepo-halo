#!/usr/bin/env bash
# Deploy/redeploy trepo-alarm-triage. Idempotent-ish: updates code if the fn
# exists; on first run, create the infra blocks below (commented) once.
set -euo pipefail
PROFILE="${AWS_PROFILE:-trepo-dev}"; REGION="us-east-1"
FN=trepo-alarm-triage
cd "$(dirname "$0")"
[ -d node_modules ] || npm install --omit=dev --no-audit --no-fund
rm -f /tmp/${FN}.zip
zip -qr /tmp/${FN}.zip index.mjs package.json node_modules
aws lambda update-function-code --profile "$PROFILE" --region "$REGION" \
  --function-name "$FN" --zip-file fileb:///tmp/${FN}.zip \
  --query '{Status:LastUpdateStatus,Sha:CodeSha256}' --output json
aws lambda wait function-updated --profile "$PROFILE" --region "$REGION" --function-name "$FN"
echo "deployed $FN"

# --- First-time infra (run once; see README). Kept here for reproducibility: ---
# ROLE=$(aws iam create-role --role-name trepo-alarm-triage-role \
#   --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
#   --query Role.Arn --output text)
# aws iam put-role-policy --role-name trepo-alarm-triage-role --policy-name triage-least-privilege --policy-document file://iam-policy.json
# TOPIC=$(aws sns create-topic --name trepo-triage-reports --query TopicArn --output text)
# aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint matt@trepo.ai
# KEY=$(aws lambda get-function-configuration --function-name trepo-analytics-ai-overview --query 'Environment.Variables.ANTHROPIC_API_KEY' --output text)
# aws lambda create-function --function-name trepo-alarm-triage --runtime nodejs22.x --handler index.handler \
#   --role "$ROLE" --zip-file fileb:///tmp/trepo-alarm-triage.zip --timeout 60 --memory-size 512 \
#   --environment "Variables={ANTHROPIC_API_KEY=$KEY,TRIAGE_TOPIC_ARN=$TOPIC}"
# aws lambda put-function-concurrency --function-name trepo-alarm-triage --reserved-concurrent-executions 2
# aws lambda add-permission --function-name trepo-alarm-triage --statement-id sns-capture-alerts \
#   --action lambda:InvokeFunction --principal sns.amazonaws.com \
#   --source-arn arn:aws:sns:us-east-1:566667681926:trepo-capture-alerts
# aws sns subscribe --topic-arn arn:aws:sns:us-east-1:566667681926:trepo-capture-alerts \
#   --protocol lambda --notification-endpoint <fn-arn>
