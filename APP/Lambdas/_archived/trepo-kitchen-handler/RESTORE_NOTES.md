# Archived Lambda: `trepo-kitchen-handler` (shared-table migration — straggler #1)

**Deleted:** 2026-07-12, by backend-deploy under team-lead GO option (a).
**Why:** Dead legacy straggler. **0 invocations in 90 days**, CloudWatch log group never created (never invoked). Wrote `{owner}_new_kitchen` — an already-migrated, near-dead family (30 rows / 2 tables). The go-forward app uses grocery-backend `kitchen_api` (→ `shared_kitchen`), not this legacy `POST /v1/kitchen` route. Superseded.

**This archive makes deletion literally reversible** — everything needed to redeploy is here.

## Artifacts in this dir
- `trepo-kitchen-handler.code.zip` — the exact DEPLOYED code artifact (nodejs22.x, single-file `kitchenHandler` + node_modules).
- `config.redacted.json` — full function config. **Env var VALUES redacted** (DB_HOST/DB_NAME/DB_PASSWORD/DB_USER = standard grocery-backend RDS creds; copy from any sibling fn's env, e.g. `SavedRecipesApiFunction`). No unique secrets.
- `apigw_integration_kitchen.json` — the API GW v2 integration (AWS_PROXY, payload 2.0, POST).

## Deployed config (non-secret)
- Runtime `nodejs22.x`, Handler `kitchenHandler.handler` (confirm from zip), Memory 128 MB, Timeout 3s, arch per config.
- Role: `arn:aws:iam::566667681926:role/service-role/trepo-kitchen-handler-role-2s79ajds`

## API Gateway (HTTP API v2, api `1zc0nh8x48`) — ORPHANED, not deleted
The Lambda is deleted but its route/integration are **left in place** (per team-lead — bundle route removal with the `/v1/feed` route deletion at Matt's `_new_feed` archive decision; same api, one change set, one review):
- Route `vzugold` = `POST /v1/kitchen` → integration `s2sqvhp`
- Integration `s2sqvhp` = AWS_PROXY, PayloadFormatVersion 2.0, Method POST, IntegrationUri `arn:aws:lambda:us-east-1:566667681926:function:trepo-kitchen-handler`
- (Sibling for the bundled cleanup: feed route `f6164wk` = `POST /v1/feed` → integration `xia60do`.)
> The orphaned route will 5xx if ever hit (0/90d says it won't) until removed.

## Restore procedure
1. `aws lambda create-function --function-name trepo-kitchen-handler --runtime nodejs22.x --handler kitchenHandler.handler --role <role above> --memory-size 128 --timeout 3 --zip-file fileb://trepo-kitchen-handler.code.zip` (+ env vars from a sibling fn).
2. Re-add the invoke permission (deleted with the function): `aws lambda add-permission --function-name trepo-kitchen-handler --statement-id apigw --action lambda:InvokeFunction --principal apigateway.amazonaws.com --source-arn 'arn:aws:execute-api:us-east-1:566667681926:1zc0nh8x48/*/*/v1/kitchen'`.
3. The existing integration `s2sqvhp` targets the function by ARN, so it resolves again automatically once the function (same name/ARN) exists. No route/integration change needed if they were left orphaned.
