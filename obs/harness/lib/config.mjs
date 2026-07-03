// lib/config.mjs — static configuration for the observability verification harness.
//
// AWS: run with `export AWS_PROFILE=trepo-dev AWS_REGION=us-east-1` before invoking.
// The SDK clients below read credentials/region from the ambient environment.

export const REGION = process.env.AWS_REGION || 'us-east-1';
export const PROFILE_NOTE = 'Run with AWS_PROFILE=trepo-dev AWS_REGION=us-east-1';

// ── API base URLs (from the iOS app + team-lead brief) ──────────────────────
export const API = {
  main:          'https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com',
  identifier:    'https://6m9t6wosh9.execute-api.us-east-1.amazonaws.com',
  auth:          'https://1zc0nh8x48.execute-api.us-east-1.amazonaws.com/v1',
  analytics:     'https://9s5i5b7lfi.execute-api.us-east-1.amazonaws.com/v1',
  notifications: 'https://nua4yt5q26.execute-api.us-east-1.amazonaws.com/v1',
  feedback:      'https://hbricl6rka.execute-api.us-east-1.amazonaws.com',
  // Voice quick-ack (POST /voice-ack). Base URL IS the full endpoint (iOS VoiceAssistantService).
  voiceAck:      'https://ivwu7ls6p8.execute-api.us-east-1.amazonaws.com/voice-ack',
};

// ── TEST_OWNER ──────────────────────────────────────────────────────────────
// The ONLY owner_id permitted for mutations. This is the known-safe test user.
//
// Resolution (2026-07-02): the brief said "user_id starting 1f4db6b6, zip 94041".
// That user is NOT currently present in MySQL `new_users` (316 rows) — it was
// purged during account-deletion testing (see memory: account-deletion-test-user).
// Its household owner_id is `94041`, but ALL of its analytics data in DynamoDB
// TrepoAnalyticsEvents is keyed by the full user_id UUID (214 rows found under the
// UUID; 0 under "94041"). Because dynamoEvent() polling must match the analytics
// partition key, TEST_OWNER MUST be the UUID. Confirmed empirically:
//   - TrepoAnalyticsEvents owner_id = '1f4db6b6-2f62-4558-aa40-c8e82527dc74' -> 214 rows
//   - TrepoAnalyticsEvents owner_id = '94041'                                -> 0 rows
//   - shared_kitchen owner_id (both ids)                                     -> 0 rows (clean)
export const TEST_OWNER = '1f4db6b6-2f62-4558-aa40-c8e82527dc74';
// Household owner_id for the same identity (shared_* tables). Not used for
// analytics matching; recorded for reference only.
export const TEST_HOUSEHOLD_OWNER = '94041';

// ── DynamoDB ────────────────────────────────────────────────────────────────
export const ANALYTICS_TABLE = 'TrepoAnalyticsEvents';   // PK owner_id, SK ts_id, GSI EventNameIndex

// ── CloudWatch metric namespace ─────────────────────────────────────────────
export const CAPTURE_NS = 'Trepo/Capture';

// ── Log groups (resolved 2026-07-02) ────────────────────────────────────────
export const LOG_GROUPS = {
  grocery:       '/aws/lambda/trepo-grocery-backend-dev-AnalyzeOnUpload-bpWcKEif3Gq7',
  dish:          '/aws/lambda/trepo-grocery-backend-dev-AnalyzeDishOnUpload-1wGSk6GmnxLv',
  discard:       '/aws/lambda/trepo-grocery-backend-dev-AnalyzeDiscardOnUpload-BnUyRpHHwaOx',
  bulkIdentify:  '/aws/lambda/grocery-identifier-dev-identify-async',
  auth:          '/aws/lambda/twilioAuth',
  list:          '/aws/lambda/trepo-list-handler',
  voiceWorker:   '/aws/lambda/trepo-quick-ack-async-worker-dev',
  savedRecipes:  '/aws/lambda/trepo-grocery-backend-dev-SavedRecipesApiFunction-a7aWuWlJvEF2',
  notifications: '/aws/lambda/trepo-notifications-SendFunction-mbwOf9XTzTmu',
};

// API Gateway ids (for AWS/ApiGateway 5xx metrics).
export const APIGW = {
  mainId: '7tn3gvwvh7',
};

// Marker for all harness-created data. A fresh run stamps its own run id in ctx.
export const HARNESS_TAG = 'obs-harness';
export const HARNESS_ITEM_NAME = 'obs-harness-item';
