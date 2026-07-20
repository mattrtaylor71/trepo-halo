#!/usr/bin/env bash
#
# deploy.sh — package + deploy trepo-recall-deep-dive (Python 3.11, pymysql vendored).
#
# The DEEP/SLOW sibling of trepo-recall-check. It runs a GPT-5 + web_search investigation
# PER kitchen item (engine.py), so it is async: POST self-invokes an internal {"action":
# "run"} worker and returns 202; the client polls GET until status==done.
#
# Idempotent-ish: updates code + config if the fn exists. FIRST-TIME infra (IAM role, DDB
# table, env vars, HTTP wiring) lives in the commented block below — the LEAD runs those
# ONCE. This script only touches the Lambda's own code/config thereafter.
#
# Usage:  AWS_PROFILE=trepo-dev AWS_REGION=us-east-1 bash playground6_groceryRec/recall_deep_dive/deploy.sh
set -euo pipefail
PROFILE="${AWS_PROFILE:-trepo-dev}"; REGION="${AWS_REGION:-us-east-1}"
FN=trepo-recall-deep-dive
cd "$(dirname "$0")"

BUILD=/tmp/${FN}-build
rm -rf "$BUILD" /tmp/${FN}.zip
mkdir -p "$BUILD"
cp app.py engine.py "$BUILD/"
# Vendor pymysql (pure-python). boto3 is in the runtime; OpenAI is called over raw HTTPS
# (urllib) so there is NO openai SDK to vendor — the package stays tiny.
python3 -m pip install --quiet --target "$BUILD" "pymysql==1.1.0"
( cd "$BUILD" && zip -qr /tmp/${FN}.zip . )

aws lambda update-function-code --profile "$PROFILE" --region "$REGION" \
  --function-name "$FN" --zip-file fileb:///tmp/${FN}.zip \
  --query '{Status:LastUpdateStatus,Sha:CodeSha256}' --output json
aws lambda wait function-updated --profile "$PROFILE" --region "$REGION" --function-name "$FN"
echo "deployed $FN"

# ============================================================================
# FIRST-TIME INFRA (run once; kept here for reproducibility). Uncomment + run the pieces
# you need, then use the update-function-code path above thereafter.
# ============================================================================
# ACCOUNT=566667681926
#
# # 1) DynamoDB job table. PK owner_id (S). On-demand billing (traffic is a rare user tap).
# #    TTL attribute 'ttl' auto-expires stale records (~30d) so the table self-cleans.
# aws dynamodb create-table --profile "$PROFILE" --region "$REGION" \
#   --table-name trepo-recall-deep-dive \
#   --attribute-definitions AttributeName=owner_id,AttributeType=S \
#   --key-schema AttributeName=owner_id,KeyType=HASH \
#   --billing-mode PAY_PER_REQUEST
# aws dynamodb wait table-exists --profile "$PROFILE" --region "$REGION" --table-name trepo-recall-deep-dive
# aws dynamodb update-time-to-live --profile "$PROFILE" --region "$REGION" \
#   --table-name trepo-recall-deep-dive \
#   --time-to-live-specification "Enabled=true,AttributeName=ttl"
#
# # 2) IAM role (least-priv: RW the job table, InvokeFunction on ITSELF for the async
# #    self-invoke worker, logs). See iam-policy.json.
# ROLE_ARN=$(aws iam create-role --role-name trepo-recall-deep-dive-role \
#   --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
#   --query Role.Arn --output text)
# aws iam put-role-policy --role-name trepo-recall-deep-dive-role \
#   --policy-name deep-dive-least-privilege --policy-document file://iam-policy.json
# sleep 10  # let the role propagate
#
# # 3) Mirror DB creds off the RecipesGenerator fn (source of truth), same as ux-failure-monitor,
# #    and reuse the SAME OpenAI key the instant recall-check uses.
# GEN=trepo-grocery-backend-dev-RecipesGeneratorFunction-ycoZeX7W0vum
# DB_HOST=$(aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" --function-name "$GEN" --query 'Environment.Variables.DB_HOST' --output text)
# DB_USER=$(aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" --function-name "$GEN" --query 'Environment.Variables.DB_USER' --output text)
# DB_PASS=$(aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" --function-name "$GEN" --query 'Environment.Variables.DB_PASS' --output text)
# DB_NAME=$(aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" --function-name "$GEN" --query 'Environment.Variables.DB_NAME' --output text)
# OPENAI_API_KEY=$(aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" --function-name trepo-recall-check --query 'Environment.Variables.OPENAI_API_KEY' --output text)
#
# # 4) Create the function. RDS (database-1) is PUBLICLY reachable so NO VpcConfig is needed;
# #    default (no-VPC) Lambda has internet egress to reach RDS, DynamoDB, AND api.openai.com
# #    (web search runs SERVER-SIDE at OpenAI — no crawler egress from us).
# #    Timeout 840s (14 min) — the async worker runs up to ~11 min (DEEP_RUN_DEADLINE_S=660
# #    is the internal hard stop, well under this). Memory 512MB is plenty (I/O-bound threads).
# aws lambda create-function --profile "$PROFILE" --region "$REGION" \
#   --function-name trepo-recall-deep-dive \
#   --runtime python3.11 --handler app.lambda_handler --role "$ROLE_ARN" \
#   --zip-file fileb:///tmp/trepo-recall-deep-dive.zip --timeout 840 --memory-size 512 \
#   --environment "Variables={DB_HOST=$DB_HOST,DB_USER=$DB_USER,DB_PASS=$DB_PASS,DB_NAME=$DB_NAME,DB_PORT=3306,OPENAI_API_KEY=$OPENAI_API_KEY,SELF_FUNCTION_NAME=trepo-recall-deep-dive,DEEP_DIVE_TABLE=trepo-recall-deep-dive}"
#
# # 5) Reserved concurrency. The HTTP path is trivial; the heavy work is the async worker.
# #    A cap of 5 bounds the worst case (5 concurrent users × up to 24 in-flight OpenAI
# #    calls) and protects both cost and OpenAI rate limits. Raise as usage grows.
# aws lambda put-function-concurrency --profile "$PROFILE" --region "$REGION" \
#   --function-name trepo-recall-deep-dive --reserved-concurrent-executions 5
#
# # 6) HTTP WIRING — pick ONE:
# #
# #  (A) RECOMMENDED — add two routes to the EXISTING grocery HttpApi so iOS uses the same
# #      {base} + the same no-token owner-in-path auth as every other route. Both POST and
# #      GET return in well under the HttpApi 30s integration limit (the long work is async),
# #      so the 30s cap is a non-issue. Wire POST+GET /deep-dive/{owner} -> this fn, and add
# #      the lambda:InvokeFunction permission for apigateway. (The lead owns the HttpApi, so
# #      add the two routes + integration there and grant:)
# #   aws lambda add-permission --profile "$PROFILE" --region "$REGION" \
# #     --function-name trepo-recall-deep-dive --statement-id apigw-invoke \
# #     --action lambda:InvokeFunction --principal apigateway.amazonaws.com \
# #     --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT}:<API_ID>/*/*/deep-dive/*"
# #
# #  (B) ALTERNATIVE — a dedicated Function URL (fastest to stand up, but a DIFFERENT base
# #      host than the grocery API, so iOS would special-case it). AuthType NONE mirrors the
# #      instant recall-check's public owner-in-path model.
# #   aws lambda create-function-url-config --profile "$PROFILE" --region "$REGION" \
# #     --function-name trepo-recall-deep-dive --auth-type NONE \
# #     --cors '{"AllowOrigins":["*"],"AllowMethods":["GET","POST"]}'
# #   aws lambda add-permission --profile "$PROFILE" --region "$REGION" \
# #     --function-name trepo-recall-deep-dive --statement-id fnurl \
# #     --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE
# #   # Tradeoff: (A) keeps one base + one auth model for iOS; (B) is faster but adds a 2nd host.
#
# # 7) (OPTIONAL) keep-warm: an EventBridge rate(5 min) rule sending {"action":"ping"} kills
# #    the cold-start on the first user tap, same idea as recall-check's warm ping.
