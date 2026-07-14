// Lambda handler for analyzing uploaded dish images using dish identification and nutrition-extractor
const AWS = require('aws-sdk');
const crypto = require('crypto');
const fetch = require('node-fetch');
const mysql = require('mysql2/promise');
const { identifyDish } = require('./dist/openai/identifyDish');
const { identifyPackagedItem } = require('./dist/openai/identifyPackagedItem');
const { extractNutrition } = require('./dist/openai/extractNutrition');
const { extractNutritionFast } = require('./dist/openai/extractNutritionFast');
const { writeToDishesTable } = require('./dist/utils/mysqlDishWriter');
const { writeFeedEvent } = require('./dist/utils/mysqlFeedWriter');
const { getHouseholdMemberIds, stableHouseholdRowId } = require('./dist/utils/householdSync');
const { lookupPackagedNutrition } = require('./dist/utils/packagedNutritionLookup');
const { findStockImage } = require('./dist/utils/stockImageSearch');

const { syncImageTriage } = require('./triage');
const { writeMasterFeedEvent } = require('./masterFeedWriter');

const s3 = new AWS.S3();

const THUMBNAIL_MAX_SIZE = 256;
const NOTIFICATIONS_API_URL = 'https://nua4yt5q26.execute-api.us-east-1.amazonaws.com/v1/notifications/send';

// Per-LLM-call telemetry marker (evt=ai_op). One line of JSON per LLM HTTP call,
// success and error. Best-effort: a logging failure must never break the op.
function logAiOp(rec) {
  try {
    const out = {
      evt: 'ai_op',
      service: rec.service,
      op: rec.op,
      owner_id: rec.owner_id != null ? String(rec.owner_id) : null,
      model: rec.model != null ? String(rec.model) : null,
      latency_ms: Number.isFinite(rec.latency_ms) ? Math.trunc(rec.latency_ms) : null,
      status: rec.status,
      input: String(rec.input == null ? '' : rec.input).slice(0, 400),
      output: String(rec.output == null ? '' : rec.output).slice(0, 1200),
    };
    if (rec.error != null) out.error = String(rec.error).slice(0, 500);
    if (rec.job_id != null) out.job_id = String(rec.job_id);
    console.log(JSON.stringify(out));
  } catch (_) { /* never let telemetry throw */ }
}

// Model resolvers mirror the dist/openai/* wrappers so the marker records the
// model actually used (env-overridable, same defaults as the wrappers).
const OPENAI_MODEL_DEFAULT = 'gpt-5.4-2026-03-05';
function fullDishModel() { return process.env.FULL_DISH_MODEL || process.env.OPENAI_MODEL || OPENAI_MODEL_DEFAULT; }
function fastDishModel() { return process.env.FAST_DISH_MODEL || process.env.OPENAI_MODEL || OPENAI_MODEL_DEFAULT; }

async function sendJobCompletePush(ownerId, productName) {
  if (!ownerId) return;
  try {
    const res = await fetch(NOTIFICATIONS_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner_id: ownerId,
        title: 'Dish Analyzed',
        body: productName
          ? `${productName} has been analyzed.`
          : 'Your dish has been analyzed.',
        data: { route: 'bulk_session_review' },
      }),
      timeout: 5000,
    });
    console.log('[push] Notification sent for', ownerId, 'status:', res.status);
  } catch (err) {
    console.warn('[push] Failed to send notification (non-fatal):', err.message);
  }
}

async function resizeToThumbnail(buffer) {
  try {
    const { Jimp } = require('jimp');
    const image = await Jimp.read(buffer);
    image.scaleToFit({ w: THUMBNAIL_MAX_SIZE, h: THUMBNAIL_MAX_SIZE });
    return await image.getBuffer('image/jpeg', { quality: 82 });
  } catch (e) {
    console.warn('[dishImage] resize failed, using original:', e.message);
    return null;
  }
}

const APP_IMAGE_MAX_DIMENSION = Math.max(256, Number(process.env.APP_IMAGE_MAX_DIMENSION || 1024));
const APP_IMAGE_QUALITY = Math.max(40, Math.min(95, Number(process.env.APP_IMAGE_QUALITY || 72)));

async function resizeForAppImage(buffer) {
  try {
    const sharp = require('sharp');
    return await sharp(buffer)
      .rotate()
      .resize({
        width: APP_IMAGE_MAX_DIMENSION,
        height: APP_IMAGE_MAX_DIMENSION,
        fit: 'inside',
        withoutEnlargement: true,
      })
      .jpeg({
        quality: APP_IMAGE_QUALITY,
        progressive: true,
        mozjpeg: true,
      })
      .toBuffer();
  } catch (sharpError) {
    console.warn('[resized-image] sharp resize failed, falling back to jimp:', sharpError.message);
    try {
      const { Jimp } = require('jimp');
      const image = await Jimp.read(buffer);
      image.scaleToFit({ w: APP_IMAGE_MAX_DIMENSION, h: APP_IMAGE_MAX_DIMENSION });
      return await image.getBuffer('image/jpeg', { quality: APP_IMAGE_QUALITY });
    } catch (jimpError) {
      console.error('[resized-image] Jimp resize failed:', jimpError.message);
      return null;
    }
  }
}

async function uploadResizedOriginalImage(buffer, userId, deviceId, jobId) {
  const resizedBuffer = await resizeForAppImage(buffer);
  if (!resizedBuffer) {
    return { url: null, key: null };
  }

  const resizedKey = `resized-images/${userId}/${deviceId}/${jobId}.jpg`;
  await s3.putObject({
    Bucket: BUCKET_NAME,
    Key: resizedKey,
    Body: resizedBuffer,
    ContentType: 'image/jpeg',
  }).promise();

  return {
    url: `https://${BUCKET_NAME}.s3.amazonaws.com/${resizedKey}`,
    key: resizedKey,
  };
}
const dynamodb = new AWS.DynamoDB.DocumentClient();
const iot = new AWS.IotData({ endpoint: process.env.IOT_ENDPOINT });

const BUCKET_NAME = process.env.BUCKET_NAME;
const JOBS_TABLE = process.env.JOBS_TABLE;
const RESULTS_TABLE = process.env.RESULTS_TABLE || null;
const KEY_PREFIX = process.env.KEY_PREFIX || 'images/';
const TOPIC_TEMPLATE = process.env.RESULT_TOPIC_TEMPLATE || 'trepo/{user_id}/{device_id}/jobs/{job_id}/result';
const DEFAULT_STOCK_IMAGE_TIMEOUT_MS = 120000;
const DEFAULT_PENDING_POLL_AFTER_MS = 1000;
const DEFAULT_FAST_POLL_AFTER_MS = 1500;
const DEFAULT_RESULT_RETENTION_SECONDS = 24 * 60 * 60;
const METRICS_WINDOW_DAYS = 14;
const DEFAULT_IQ = 75;
const DISH_POINTS_MIN = 7;
const DISH_POINTS_MAX = 10;
const PACKAGED_LOOKUP_KEYWORDS = [
  'bar',
  'beverage',
  'bottle',
  'can',
  'candy',
  'chips',
  'cookie',
  'cracker',
  'cup',
  'drink',
  'energy',
  'gum',
  'jerky',
  'pack',
  'package',
  'packaged',
  'protein',
  'snack',
  'soda',
  'yogurt',
];

function containsPackagedKeyword(value) {
  const text = String(value || '').toLowerCase();
  return PACKAGED_LOOKUP_KEYWORDS.some((keyword) => text.includes(keyword));
}

function shouldAttemptPackagedLookup(dish, nutritionData) {
  const category = String(dish?.category || '').toLowerCase();
  if (category === 'beverage' || category === 'snack') {
    return true;
  }

  return [
    dish?.dish_name,
    dish?.explanation,
    nutritionData?.dish_name,
    nutritionData?.explanation,
    nutritionData?.serving_size,
  ].some(containsPackagedKeyword);
}

function mergeNutritionData(base, override) {
  if (!override) return base;
  return {
    ...base,
    ...override,
    dish_name: override.dish_name || base.dish_name || null,
    serving_size: override.serving_size || base.serving_size || null,
    explanation: override.explanation || base.explanation || null,
    ingredients: Array.isArray(override.ingredients) && override.ingredients.length > 0
      ? override.ingredients
      : (Array.isArray(base.ingredients) ? base.ingredients : []),
    allergens: Array.isArray(override.allergens) && override.allergens.length > 0
      ? override.allergens
      : (Array.isArray(base.allergens) ? base.allergens : []),
    confidence: typeof override.confidence === 'number'
      ? override.confidence
      : (typeof base.confidence === 'number' ? base.confidence : 0),
  };
}

function mergeDishWithPackagedItem(dish, packagedItem, lookedUpNutrition) {
  if (!packagedItem?.is_packaged_item) return dish;

  const originalDishName =
    typeof dish?.dish_name === 'string' && dish.dish_name.trim()
      ? dish.dish_name.trim()
      : null;
  const originalExplanation =
    typeof dish?.explanation === 'string' && dish.explanation.trim()
      ? dish.explanation.trim()
      : null;
  const packagedName =
    originalDishName ||
    packagedItem.product_name ||
    lookedUpNutrition?.dish_name ||
    null;

  return {
    ...dish,
    dish_name: packagedName,
    category: packagedItem.category || dish?.category || null,
    confidence: Math.max(
      typeof dish?.confidence === 'number' ? dish.confidence : 0,
      typeof packagedItem?.confidence === 'number' ? packagedItem.confidence : 0
    ),
    explanation: originalExplanation || packagedItem.explanation || lookedUpNutrition?.explanation || null,
  };
}

function sanitizeOwner(owner) {
  return String(owner || '').replace(/[^a-zA-Z0-9_-]/g, '');
}

function metricsTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) throw new Error('Invalid owner');
  return `${safe}-metrics`;
}

function prodKitchenTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) throw new Error('Invalid owner');
  return `${safe}_prod_kitchen`;
}

function newKitchenTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) throw new Error('Invalid owner');
  return `${safe}_new_kitchen`;
}

function discardsTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) throw new Error('Invalid owner');
  return `${safe}_discards`;
}

async function getDbConnection() {
  const { DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME } = process.env;
  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing MySQL environment variables for metrics snapshot');
  }
  const conn = await mysql.createConnection({
    host: DB_HOST,
    port: DB_PORT ? parseInt(DB_PORT, 10) : 3306,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    connectTimeout: 10000,
  });
  // Bound metadata-lock waits (default is 1 year) so metrics-snapshot DDL can't ride the
  // 300s Lambda ceiling on a lock held by a concurrent writer. Best-effort.
  try {
    await conn.query('SET SESSION lock_wait_timeout = 15, innodb_lock_wait_timeout = 15');
  } catch (e) {
    console.warn('[getDbConnection] could not set lock timeouts (non-fatal):', e && e.message ? e.message : e);
  }
  return conn;
}

async function tableExists(conn, tableName) {
  const [rows] = await conn.execute(
    `SELECT COUNT(*) AS count
     FROM information_schema.tables
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName],
  );
  return Number(rows?.[0]?.count || 0) > 0;
}

async function getTableColumns(conn, tableName) {
  const [rows] = await conn.execute(
    `SELECT column_name
     FROM information_schema.columns
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName],
  );
  return new Set((rows || []).map((row) => row.column_name || row.COLUMN_NAME).filter(Boolean));
}

async function ensureMetricsTable(conn, tableName) {
  await conn.query(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
      \`_owner\` VARCHAR(36) NOT NULL COMMENT 'Owner UUID',
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the metrics snapshot was created',
      \`IQ\` INT NOT NULL DEFAULT ${DEFAULT_IQ} COMMENT 'IQ score out of 100',
      \`Points\` BIGINT NOT NULL DEFAULT 0 COMMENT 'Points total',
      \`UPF\` DECIMAL(5,2) NOT NULL DEFAULT 0 COMMENT 'UPF percentage (last 2 weeks)',
      \`harmful_ingredients\` INT NOT NULL DEFAULT 0 COMMENT 'Harmful ingredient count (last 2 weeks)',
      \`IQ_what\` TEXT COMMENT 'What this IQ score means',
      \`IQ_suggestions\` JSON COMMENT 'Suggestions to improve IQ',
      \`UPF_what\` TEXT COMMENT 'What this UPF score means',
      \`UPF_suggestions\` JSON COMMENT 'Suggestions to improve UPF score',
      \`harmful_ingredients_what\` TEXT COMMENT 'What this harmful ingredient count means',
      \`harmful_ingredients_suggestions\` JSON COMMENT 'Suggestions to reduce harmful ingredients',
      \`kitchen_analysis_status\` VARCHAR(32) DEFAULT NULL COMMENT 'Latest kitchen analysis job status',
      \`kitchen_analysis_content\` MEDIUMTEXT COMMENT 'Formatted AI summary of the kitchen',
      \`kitchen_analysis_generated_at\` DATETIME NULL COMMENT 'When the kitchen analysis was generated',
      \`kitchen_analysis_error\` TEXT COMMENT 'Latest kitchen analysis error, if any',
      INDEX \`idx_owner\` (\`_owner\`),
      INDEX \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User metrics snapshots'
  `);
}

async function ensureMetricsColumns(conn, tableName) {
  if (!(await tableExists(conn, tableName))) return;
  const columns = await getTableColumns(conn, tableName);
  const alterParts = [];
  if (!columns.has('IQ_what')) alterParts.push("ADD COLUMN `IQ_what` TEXT COMMENT 'What this IQ score means' AFTER `harmful_ingredients`");
  if (!columns.has('IQ_suggestions')) alterParts.push("ADD COLUMN `IQ_suggestions` JSON COMMENT 'Suggestions to improve IQ' AFTER `IQ_what`");
  if (!columns.has('UPF_what')) alterParts.push("ADD COLUMN `UPF_what` TEXT COMMENT 'What this UPF score means' AFTER `IQ_suggestions`");
  if (!columns.has('UPF_suggestions')) alterParts.push("ADD COLUMN `UPF_suggestions` JSON COMMENT 'Suggestions to improve UPF score' AFTER `UPF_what`");
  if (!columns.has('harmful_ingredients_what')) alterParts.push("ADD COLUMN `harmful_ingredients_what` TEXT COMMENT 'What this harmful ingredient count means' AFTER `UPF_suggestions`");
  if (!columns.has('harmful_ingredients_suggestions')) alterParts.push("ADD COLUMN `harmful_ingredients_suggestions` JSON COMMENT 'Suggestions to reduce harmful ingredients' AFTER `harmful_ingredients_what`");
  if (!columns.has('kitchen_analysis_status')) alterParts.push("ADD COLUMN `kitchen_analysis_status` VARCHAR(32) DEFAULT NULL COMMENT 'Latest kitchen analysis job status' AFTER `harmful_ingredients_suggestions`");
  if (!columns.has('kitchen_analysis_content')) alterParts.push("ADD COLUMN `kitchen_analysis_content` MEDIUMTEXT COMMENT 'Formatted AI summary of the kitchen' AFTER `kitchen_analysis_status`");
  if (!columns.has('kitchen_analysis_generated_at')) alterParts.push("ADD COLUMN `kitchen_analysis_generated_at` DATETIME NULL COMMENT 'When the kitchen analysis was generated' AFTER `kitchen_analysis_content`");
  if (!columns.has('kitchen_analysis_error')) alterParts.push("ADD COLUMN `kitchen_analysis_error` TEXT COMMENT 'Latest kitchen analysis error, if any' AFTER `kitchen_analysis_generated_at`");
  if (alterParts.length > 0) {
    await conn.query(`ALTER TABLE \`${tableName}\` ${alterParts.join(', ')}`);
  }
}

async function ensureGroceryColumns(conn, tableName) {
  if (!(await tableExists(conn, tableName))) return;
  const existingColumns = await getTableColumns(conn, tableName);
  const upfPosition = existingColumns.has('nutrition_summary') ? ' AFTER `nutrition_summary`' : '';
  const [rows] = await conn.execute(
    `SELECT column_name, column_type
     FROM information_schema.columns
     WHERE table_schema = DATABASE() AND table_name = ?
       AND LOWER(column_name) IN ('upf', 'harmful_ingredients')`,
    [tableName],
  );
  const columnInfo = new Map();
  for (const row of rows || []) {
    const name = String(row.column_name || row.COLUMN_NAME || '').toLowerCase();
    if (name) columnInfo.set(name, row.column_type || row.COLUMN_TYPE || '');
  }

  if (!columnInfo.has('upf')) {
    await conn.query(
      `ALTER TABLE \`${tableName}\` ADD COLUMN \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'${upfPosition}`,
    );
  } else {
    const normalizedType = String(columnInfo.get('upf') || '').replace(/\s+/g, '');
    if (normalizedType !== "enum('yes','no')") {
      await conn.query(
        `ALTER TABLE \`${tableName}\` MODIFY COLUMN \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'`,
      );
      await conn.query(`UPDATE \`${tableName}\` SET \`upf\` = LOWER(\`upf\`) WHERE \`upf\` IS NOT NULL`);
    }
  }

  if (!columnInfo.has('harmful_ingredients')) {
    await conn.query(
      `ALTER TABLE \`${tableName}\` ADD COLUMN \`harmful_ingredients\` JSON COMMENT 'Array of harmful ingredient strings' AFTER \`upf\``,
    );
  }
}

function recentWhereClause(columns) {
  const clauses = [`\`_createdDate\` >= DATE_SUB(NOW(), INTERVAL ${METRICS_WINDOW_DAYS} DAY)`];
  if (columns.has('action')) {
    clauses.push("`action` = 'IN'");
  }
  return clauses.join(' AND ');
}

async function getUpfCounts(conn, tableName) {
  if (!(await tableExists(conn, tableName))) return { total: 0, upfCount: 0 };
  const columns = await getTableColumns(conn, tableName);
  const whereClause = recentWhereClause(columns);
  if (columns.has('upf')) {
    const [rows] = await conn.query(
      `SELECT COUNT(*) AS total,
              SUM(CASE WHEN LOWER(\`upf\`) = 'yes' THEN 1 ELSE 0 END) AS upf_count
       FROM \`${tableName}\`
       WHERE ${whereClause}`,
    );
    return {
      total: Number(rows?.[0]?.total || 0),
      upfCount: Number(rows?.[0]?.upf_count || 0),
    };
  }
  const [rows] = await conn.query(
    `SELECT COUNT(*) AS total
     FROM \`${tableName}\`
     WHERE ${whereClause}`,
  );
  return { total: Number(rows?.[0]?.total || 0), upfCount: 0 };
}

async function getHarmfulCount(conn, tableName) {
  if (!(await tableExists(conn, tableName))) return 0;
  const columns = await getTableColumns(conn, tableName);
  if (!columns.has('harmful_ingredients')) return 0;
  const whereClause = recentWhereClause(columns);
  const [rows] = await conn.query(
    `SELECT SUM(COALESCE(JSON_LENGTH(\`harmful_ingredients\`), 0)) AS harmful_count
     FROM \`${tableName}\`
     WHERE ${whereClause}`,
  );
  return Number(rows?.[0]?.harmful_count || 0);
}

function safeInt(value, defaultValue = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : defaultValue;
}

function getScanPointsDelta() {
  return DISH_POINTS_MIN + Math.floor(Math.random() * (DISH_POINTS_MAX - DISH_POINTS_MIN + 1));
}

function coerceJsonList(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return value.trim() ? [value] : [];
    }
  }
  return [];
}

async function calculateMetrics(conn, owner) {
  const prodKitchenTable = prodKitchenTableName(owner);
  const newKitchenTable = newKitchenTableName(owner);
  const discardsTable = discardsTableName(owner);
  await ensureGroceryColumns(conn, prodKitchenTable);
  await ensureGroceryColumns(conn, newKitchenTable);
  await ensureGroceryColumns(conn, discardsTable);

  const [prodKitchen, newKitchen, discards, harmfulCounts] = await Promise.all([
    getUpfCounts(conn, prodKitchenTable),
    getUpfCounts(conn, newKitchenTable),
    getUpfCounts(conn, discardsTable),
    Promise.all([
      getHarmfulCount(conn, prodKitchenTable),
      getHarmfulCount(conn, newKitchenTable),
      getHarmfulCount(conn, discardsTable),
    ]),
  ]);

  const total = prodKitchen.total + newKitchen.total + discards.total;
  const upfCount = prodKitchen.upfCount + newKitchen.upfCount + discards.upfCount;
  const harmfulIngredients = Math.max(0, Math.min(1000, harmfulCounts.reduce((sum, count) => sum + count, 0)));

  return {
    UPF: total > 0 ? Number(((upfCount / total) * 100).toFixed(2)) : 0,
    harmful_ingredients: harmfulIngredients,
  };
}

// Shared-table migration: mirror each per-owner metrics snapshot into shared_metrics
// (keyed by (owner_id,_id) after the 2026-07-13 PK change). Best-effort, gated by
// DUAL_WRITE_METRICS; never blocks the capture write. Mirrors the Python
// _dual_write_metrics_to_shared in redemptions_api/app.py.
const DUAL_WRITE_METRICS = String(process.env.DUAL_WRITE_METRICS || 'false').toLowerCase() === 'true';
const SHARED_METRICS_COLS = [
  '_id', '_owner', '_createdDate', 'IQ', 'Points', 'UPF', 'harmful_ingredients',
  'IQ_what', 'IQ_suggestions', 'UPF_what', 'UPF_suggestions',
  'harmful_ingredients_what', 'harmful_ingredients_suggestions',
  'kitchen_analysis_status', 'kitchen_analysis_content', 'kitchen_analysis_generated_at',
  'kitchen_analysis_error',
];
async function dualWriteMetricsToShared(conn, ownerId, memberTable, entryId) {
  if (!DUAL_WRITE_METRICS) return;
  try {
    const cols = SHARED_METRICS_COLS.map((c) => `\`${c}\``).join(', ');
    const src = SHARED_METRICS_COLS.map((c) => `s.\`${c}\``).join(', ');
    const upd = SHARED_METRICS_COLS.filter((c) => c !== '_id').map((c) => `\`${c}\`=VALUES(\`${c}\`)`).join(', ');
    await conn.execute(
      `INSERT INTO \`shared_metrics\` (\`owner_id\`, ${cols}) `
      + `SELECT ?, ${src} FROM \`${memberTable}\` s WHERE s.\`_id\` = ? `
      + `ON DUPLICATE KEY UPDATE \`owner_id\`=VALUES(\`owner_id\`), ${upd}`,
      [ownerId, entryId],
    );
  } catch (err) {
    try {
      console.error(JSON.stringify({
        evt: 'dual_write_miss', family: 'metrics',
        owner_id: String(ownerId || ''), metrics_id: String(entryId || ''),
        error: String(err && err.message ? err.message : err).slice(0, 500),
      }));
    } catch (_) { /* marker best-effort */ }
  }
}

async function appendMetricsSnapshot(owner, pointsDelta = getScanPointsDelta()) {
  const conn = await getDbConnection();
  try {
    const memberIds = await getHouseholdMemberIds(conn, owner);
    const computed = await calculateMetrics(conn, owner);
    const entryId = stableHouseholdRowId('metrics', Date.now(), owner, safeInt(pointsDelta, 0));

    for (const memberId of memberIds) {
      const tableName = metricsTableName(memberId);
      await ensureMetricsTable(conn, tableName);
      await ensureMetricsColumns(conn, tableName);
      const [latestRows] = await conn.query(`SELECT * FROM \`${tableName}\` ORDER BY \`_createdDate\` DESC LIMIT 1`);
      const latest = latestRows?.[0] || {};

      await conn.execute(
        `INSERT INTO \`${tableName}\`
         (\`_id\`, \`_owner\`, \`_createdDate\`, \`IQ\`, \`Points\`, \`UPF\`, \`harmful_ingredients\`,
          \`IQ_what\`, \`IQ_suggestions\`, \`UPF_what\`, \`UPF_suggestions\`,
          \`harmful_ingredients_what\`, \`harmful_ingredients_suggestions\`,
          \`kitchen_analysis_status\`, \`kitchen_analysis_content\`, \`kitchen_analysis_generated_at\`,
          \`kitchen_analysis_error\`)
         VALUES (?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          entryId,
          memberId,
          DEFAULT_IQ,
          Math.max(0, safeInt(latest.Points, 0) + safeInt(pointsDelta, 0)),
          computed.UPF,
          computed.harmful_ingredients,
          latest.IQ_what || null,
          JSON.stringify(coerceJsonList(latest.IQ_suggestions)),
          latest.UPF_what || null,
          JSON.stringify(coerceJsonList(latest.UPF_suggestions)),
          latest.harmful_ingredients_what || null,
          JSON.stringify(coerceJsonList(latest.harmful_ingredients_suggestions)),
          latest.kitchen_analysis_status || null,
          latest.kitchen_analysis_content || null,
          latest.kitchen_analysis_generated_at || null,
          latest.kitchen_analysis_error || null,
        ],
      );
      await dualWriteMetricsToShared(conn, memberId, tableName, entryId);
    }
  } finally {
    await conn.end();
  }
}

function isoNow() {
  return new Date().toISOString();
}

function getResultRetentionSeconds() {
  const raw = process.env.DISH_RESULT_TTL_SECONDS;
  const parsed = parseInt(raw || '', 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_RESULT_RETENTION_SECONDS;
  }
  return parsed;
}

function getResultExpiresAt(createdAt) {
  const createdMs = Date.parse(createdAt);
  const baseMs = Number.isFinite(createdMs) ? createdMs : Date.now();
  return new Date(baseMs + (getResultRetentionSeconds() * 1000)).toISOString();
}

function toNullableNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function buildFinalSummary(action, dishName, confidence) {
  const hasDishName = typeof dishName === 'string' && dishName.trim();
  if (!hasDishName) {
    return 'No dish detected.';
  }
  return `${action} ${dishName} (${(confidence * 100).toFixed(0)}%)`;
}

function buildFastResultPayload(jobMeta, fastPayload, createdAt) {
  return {
    job_id: jobMeta.job_id,
    mode: 'dish',
    state: 'ready',
    phase: 'fast',
    is_terminal: false,
    action: jobMeta.action,
    dish_name: typeof fastPayload.dish_name === 'string' && fastPayload.dish_name.trim() ? fastPayload.dish_name.trim() : null,
    meal_summary: fastPayload.summary || null,
    confidence: toNullableNumber(fastPayload.confidence),
    calories: toNullableNumber(fastPayload.calories),
    protein_g: toNullableNumber(fastPayload.protein_g),
    carbs_g: toNullableNumber(fastPayload.carbs_g),
    fat_g: toNullableNumber(fastPayload.fat_g),
    health_score: 0,
    recommendation: null,
    created_at: createdAt,
    updated_at: createdAt,
    poll_after_ms: DEFAULT_FAST_POLL_AFTER_MS,
  };
}

function buildFinalResultPayload(jobMeta, dishName, confidence, nutritionData, createdAt) {
  return {
    job_id: jobMeta.job_id,
    mode: 'dish',
    state: 'ready',
    phase: 'final',
    is_terminal: true,
    action: jobMeta.action,
    dish_name: dishName,
    meal_summary: buildFinalSummary(jobMeta.action, dishName, confidence),
    confidence: toNullableNumber(confidence),
    calories: toNullableNumber(nutritionData.calories),
    protein_g: toNullableNumber(nutritionData.protein),
    carbs_g: toNullableNumber(nutritionData.total_carbohydrates),
    fat_g: toNullableNumber(nutritionData.total_fat),
    health_score: 0,
    recommendation: null,
    serving_size: nutritionData.serving_size || null,
    explanation: nutritionData.explanation || null,
    saturated_fat_g: toNullableNumber(nutritionData.saturated_fat),
    trans_fat_g: toNullableNumber(nutritionData.trans_fat),
    cholesterol_mg: toNullableNumber(nutritionData.cholesterol),
    sodium_mg: toNullableNumber(nutritionData.sodium),
    dietary_fiber_g: toNullableNumber(nutritionData.dietary_fiber),
    sugars_g: toNullableNumber(nutritionData.sugars),
    vitamin_a: toNullableNumber(nutritionData.vitamin_a),
    vitamin_c: toNullableNumber(nutritionData.vitamin_c),
    calcium: toNullableNumber(nutritionData.calcium),
    iron: toNullableNumber(nutritionData.iron),
    ingredients: Array.isArray(nutritionData.ingredients) ? nutritionData.ingredients : [],
    allergens: Array.isArray(nutritionData.allergens) ? nutritionData.allergens : [],
    created_at: createdAt,
    updated_at: createdAt,
  };
}

async function writeDishResultSnapshot(jobMeta, phase, payload) {
  if (!RESULTS_TABLE) {
    return;
  }

  const existing = await dynamodb.get({
    TableName: RESULTS_TABLE,
    Key: { job_id: jobMeta.job_id },
  }).promise();

  const current = existing.Item || {};
  const createdAt = current.created_at || jobMeta.created_at || payload.created_at || isoNow();
  const item = {
    ...current,
    job_id: jobMeta.job_id,
    mode: 'dish',
    state: 'ready',
    latest_phase: phase,
    user_id: jobMeta.user_id,
    device_id: jobMeta.device_id,
    owner: jobMeta.owner || current.owner || null,
    action: jobMeta.action,
    created_at: createdAt,
    updated_at: payload.updated_at || payload.created_at || isoNow(),
    expires_at: current.expires_at || getResultExpiresAt(createdAt),
  };

  if (phase === 'fast') {
    item.fast_result = payload;
  } else if (phase === 'final') {
    item.final_result = payload;
  }

  await dynamodb.put({
    TableName: RESULTS_TABLE,
    Item: item,
  }).promise();
}

function getStockImageTimeoutMs() {
  const raw = process.env.STOCK_IMAGE_TIMEOUT_MS;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_STOCK_IMAGE_TIMEOUT_MS;
  const parsed = parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_STOCK_IMAGE_TIMEOUT_MS;
  return parsed;
}

function withTimeout(promise, ms) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      const error = new Error(`stock image search timed out after ${ms}ms`);
      error.code = 'STOCK_IMAGE_TIMEOUT';
      reject(error);
    }, ms);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutId));
}

function getFastDishMinConfidence() {
  const raw = process.env.FAST_DISH_MIN_CONFIDENCE || '0.35';
  const parsed = parseFloat(raw);
  return Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : 0.35;
}

function normalizeFastPayload(fastPayload) {
  const minConfidence = getFastDishMinConfidence();
  const dishName = typeof fastPayload.dish_name === 'string' ? fastPayload.dish_name.trim() : '';
  const confidence = typeof fastPayload.confidence === 'number' ? fastPayload.confidence : 0;
  const shouldNullMacros = !dishName || confidence < minConfidence;

  if (!shouldNullMacros) {
    return fastPayload;
  }

  return {
    ...fastPayload,
    dish_name: null,
    calories: null,
    protein_g: null,
    carbs_g: null,
    fat_g: null,
    summary: 'No dish detected.',
    confidence: Math.min(confidence, minConfidence),
  };
}

function extractS3FromEvent(event) {
  // Support EventBridge S3 events (detail.bucket.name / detail.object.key)
  if (event['detail-type'] === 'Object Created' && event.detail) {
    try {
      const bucket = event.detail.bucket.name;
      const key = decodeURIComponent(event.detail.object.key);
      return { bucket, key };
    } catch (e) {
      console.error('[extract] EventBridge parse failed:', e);
    }
  }

  // Support native S3 notifications (Records[].s3.bucket.name / .object.key)
  if (event.Records && Array.isArray(event.Records)) {
    try {
      const rec = event.Records[0];
      if (rec.eventSource && rec.eventSource.startsWith('aws:s3')) {
        const bucket = rec.s3.bucket.name;
        const key = decodeURIComponent(rec.s3.object.key);
        return { bucket, key };
      }
    } catch (e) {
      console.error('[extract] Records parse failed:', e);
    }
  }

  console.error('[extract] Unrecognized event shape');
  return { bucket: null, key: null };
}

function extractIdsFromKey(key) {
  // Pattern: images/<user_id>/<device_id>/YYYY/MM/DD/<job_id>.jpg
  const pattern = new RegExp(`^${KEY_PREFIX.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^/]+)/([^/]+)/\\d{4}/\\d{2}/\\d{2}/([^.]+)\\.(jpg|png)$`);
  const match = key.match(pattern);
  if (!match) {
    return null;
  }
  return {
    user_id: match[1],
    device_id: match[2],
    job_id: match[3],
  };
}

function getStockImageTargetUrl(job) {
  const candidates = [
    job?.stock_image_target_url,
    job?.target_image_url,
    job?.reference_image_url,
    job?.product_image_target_url,
    process.env.STOCK_IMAGE_TARGET_URL,
  ];
  for (const value of candidates) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function shouldRequireTargetMatch(targetImageUrl) {
  if (!targetImageUrl) return false;
  const raw = process.env.STOCK_IMAGE_REQUIRE_TARGET_MATCH;
  if (raw === undefined || raw === null || raw === '') {
    return true;
  }
  return String(raw).toLowerCase() === 'true';
}

async function uploadTargetImageIfNeeded(targetImageUrl, userId, jobId) {
  if (!targetImageUrl || typeof targetImageUrl !== 'string') return targetImageUrl;
  if (!targetImageUrl.startsWith('data:image/')) return targetImageUrl;
  if (!BUCKET_NAME) return targetImageUrl;

  const match = targetImageUrl.match(/^data:image\/([a-z0-9+.-]+);base64,(.+)$/i);
  if (!match) return targetImageUrl;

  const ext = match[1].toLowerCase();
  const base64 = match[2];
  const buffer = Buffer.from(base64, 'base64');
  const key = `stock-image-targets/${userId}/${jobId}.${ext}`;
  const contentType = `image/${ext}`;

  try {
    await s3.putObject({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: buffer,
      ContentType: contentType,
    }).promise();

    const signedUrl = s3.getSignedUrl('getObject', {
      Bucket: BUCKET_NAME,
      Key: key,
      Expires: 900,
    });
    return signedUrl;
  } catch (error) {
    console.warn('[stock-image] Failed to upload target image (non-fatal):', error);
    return targetImageUrl;
  }
}

async function updateJob(jobId, attrs) {
  const nextAttrs = {
    ...attrs,
    updated_at: new Date().toISOString(),
  };
  const updateExpr = Object.keys(nextAttrs)
    .map((k, i) => `#${k} = :${k}`)
    .join(', ');
  const names = {};
  const values = {};
  Object.keys(nextAttrs).forEach(k => {
    names[`#${k}`] = k;
    values[`:${k}`] = nextAttrs[k];
  });

  const result = await dynamodb.update({
    TableName: JOBS_TABLE,
    Key: { job_id: jobId },
    UpdateExpression: `SET ${updateExpr}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
    ReturnValues: 'ALL_NEW',
  }).promise();
  const updatedJob = result.Attributes || null;
  if (updatedJob) {
    try {
      await syncImageTriage(updatedJob);
    } catch (triageError) {
      console.warn('[triage] Failed to sync dish triage:', triageError.message || triageError);
    }
  }
  return updatedJob;
}

async function publishResult(userId, deviceId, jobId, dishName, confidence, action) {
  const topic = TOPIC_TEMPLATE
    .replace('{user_id}', userId)
    .replace('{device_id}', deviceId)
    .replace('{job_id}', jobId);

  const hasDishName = typeof dishName === 'string' && dishName.trim();
  const summary = buildFinalSummary(action, dishName, confidence);

  const payload = {
    job_id: jobId,
    mode: 'dish',
    action: action,
    dish_name: hasDishName ? dishName : null,
    confidence: confidence,
    summary,
    created_at: new Date().toISOString(),
  };

  await iot.publish({
    topic: topic,
    qos: 1,
    payload: JSON.stringify(payload),
  }).promise();
}

async function publishFastResult(userId, deviceId, jobId, fastPayload, action, latencyMs) {
  const topic = TOPIC_TEMPLATE
    .replace('{user_id}', userId)
    .replace('{device_id}', deviceId)
    .replace('{job_id}', jobId);

  const payload = {
    job_id: jobId,
    mode: 'dish',
    phase: 'fast',
    action: action,
    dish_name: fastPayload.dish_name,
    confidence: fastPayload.confidence,
    calories: fastPayload.calories,
    protein_g: fastPayload.protein_g,
    carbs_g: fastPayload.carbs_g,
    fat_g: fastPayload.fat_g,
    summary: fastPayload.summary,
    latency_ms: latencyMs,
    created_at: new Date().toISOString(),
  };

  await iot.publish({
    topic: topic,
    qos: 1,
    payload: JSON.stringify(payload),
  }).promise();
}

// Emits a structured marker for silent (non-throwing) analysis failures so a
// CloudWatch metric filter on "analysis_failed" can alert. Hard failures rethrow
// and are caught by the Lambda Errors alarm; this covers the swallowed/degraded
// paths that return 200 (e.g. the preliminary Dish-Log write failing).
function reportAnalysisFailed(kind, stage, jobId, err) {
  try {
    console.error(JSON.stringify({
      evt: 'analysis_failed',
      kind,
      stage,
      job_id: jobId || null,
      error: (err && (err.message || String(err))) || stage,
    }));
  } catch (_) { /* never let logging throw */ }
}

exports.handler = async (event) => {
  console.log('[handler] Event received:', JSON.stringify(event).substring(0, 500));

  const { bucket, key } = extractS3FromEvent(event);
  if (!bucket || !key) {
    console.log('[handler] No bucket/key; ignoring.');
    return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true }) };
  }

  console.log('[handler] Processing S3 object:', { bucket, key });

  if (KEY_PREFIX && !key.startsWith(KEY_PREFIX)) {
    console.log('[handler] Key not under prefix; skipping.', { prefix: KEY_PREFIX });
    return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true }) };
  }

  let jobId = null;
  try {
    const ids = extractIdsFromKey(key);
    if (!ids) {
      console.log('[handler] Key does not match dish upload pattern; skipping.', { key });
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'unexpected_key_pattern' }) };
    }
    const { user_id, device_id } = ids;
    jobId = ids.job_id;
    console.log('[handler] Parsed IDs:', ids);

    // Mark job as PROCESSING
    await updateJob(jobId, {
      status: 'PROCESSING',
      t_start_proc: new Date().toISOString(),
    });

    // Load job to get owner and action
    const jobResult = await dynamodb.get({
      TableName: JOBS_TABLE,
      Key: { job_id: jobId },
    }).promise();

    const job = jobResult.Item || { ...ids, job_id: jobId };
    const owner = job.owner;
    const action = job.action || 'IN';
    const jobMeta = {
      job_id: jobId,
      user_id,
      device_id,
      owner,
      action,
      created_at: job.created_at || null,
    };

    if (!owner) {
      console.error('[error] owner parameter required in job');
      await updateJob(jobId, {
        status: 'FAILED',
        error_msg: 'owner parameter required',
      });
      reportAnalysisFailed('dish', 'owner_missing', jobId, 'owner parameter required');
      return {
        statusCode: 400,
        body: JSON.stringify({ ok: false, error: 'owner parameter required' }),
      };
    }

    // Only process dish type jobs
    if (job.type && job.type !== 'dish') {
      console.log('[handler] Job type is not dish, skipping:', job.type);
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'not a dish job' }) };
    }

    console.log('[dynamo] Loaded job:', {
      has_action: !!job.action,
      action: action,
      owner: owner,
      type: job.type,
    });

    // Download image from S3
    console.log('[s3] Downloading image...');
    const s3Object = await s3.getObject({
      Bucket: BUCKET_NAME,
      Key: key,
    }).promise();

    const imageBuffer = Buffer.from(s3Object.Body);
    console.log('[s3] Image downloaded, size:', imageBuffer.length, 'bytes');

    // Construct the S3 URL up front (key + BUCKET_NAME are already in scope) so the
    // preliminary Dish-Log write in the fast path below can reference it. These were
    // previously declared AFTER the fast path, which threw a temporal-dead-zone
    // ReferenceError and silently broke the "appears immediately" write.
    // resizedImageUrl/Key are filled in later after the resize step.
    const imageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${key}`;
    let resizedImageUrl = null;
    let resizedImageKey = null;

    // Fast dish estimation for immediate MQTT feedback
    try {
      console.log('[fast] Starting fast dish nutrition estimate...');
      const fastStart = Date.now();
      let fastData;
      try {
        fastData = await extractNutritionFast(imageBuffer);
        logAiOp({
          service: 'dish', op: 'extract_nutrition_fast', owner_id: owner, model: fastDishModel(),
          latency_ms: Date.now() - fastStart, status: 'success', job_id: jobId,
          input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
          output: `dish_name=${fastData && fastData.dish_name} confidence=${fastData && fastData.confidence} calories=${fastData && fastData.calories}`,
        });
      } catch (fastLlmError) {
        logAiOp({
          service: 'dish', op: 'extract_nutrition_fast', owner_id: owner, model: fastDishModel(),
          latency_ms: Date.now() - fastStart, status: 'error', job_id: jobId,
          input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
          error: fastLlmError && fastLlmError.message ? fastLlmError.message : String(fastLlmError),
        });
        throw fastLlmError;
      }
      const normalizedFastData = normalizeFastPayload(fastData);
      const fastCreatedAt = isoNow();
      const fastResultPayload = buildFastResultPayload(jobMeta, normalizedFastData, fastCreatedAt);
      const fastTime = Date.now() - fastStart;
      console.log('[fast] Fast estimate complete in', fastTime, 'ms');
      await writeDishResultSnapshot(jobMeta, 'fast', fastResultPayload);
      console.log('[fast] Durable fast result written');
      await publishFastResult(user_id, device_id, jobId, normalizedFastData, action, fastTime);
      console.log('[fast] MQTT fast result published');

      // Write preliminary dish to MySQL so it appears in Dish Log immediately
      try {
        await writeToDishesTable({
          owner,
          device_id,
          user_id,
          job_id: jobId,
          action,
          s3_key: key,
          image_url: imageUrl,
          resized_image_url: resizedImageUrl,
          resized_image_key: resizedImageKey,
          dish_image_url: null,
          dish_image_key: null,
          dish: { dish_name: normalizedFastData.dish_name, confidence: normalizedFastData.confidence || 0.5, explanation: normalizedFastData.meal_summary || null },
          nutritionData: normalizedFastData,
        });
        console.log('[fast] Preliminary dish written to MySQL dishes table');
      } catch (fastWriteError) {
        console.error('[fast] Failed to write preliminary dish to MySQL (non-fatal):', fastWriteError);
        reportAnalysisFailed('dish', 'preliminary_write', jobId, fastWriteError);
      }
    } catch (fastError) {
      console.error('[fast] Failed fast estimate/publish (non-fatal):', fastError);
    }

    // Identify dish
    console.log('[identify] Starting dish identification...');
    const identifyStart = Date.now();
    let dish;
    try {
      dish = await identifyDish(imageBuffer);
      logAiOp({
        service: 'dish', op: 'identify_dish', owner_id: owner, model: fullDishModel(),
        latency_ms: Date.now() - identifyStart, status: 'success', job_id: jobId,
        input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
        output: `dish_name=${dish && dish.dish_name} confidence=${dish && dish.confidence} category=${dish && dish.category}`,
      });
    } catch (identifyLlmError) {
      logAiOp({
        service: 'dish', op: 'identify_dish', owner_id: owner, model: fullDishModel(),
        latency_ms: Date.now() - identifyStart, status: 'error', job_id: jobId,
        input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
        error: identifyLlmError && identifyLlmError.message ? identifyLlmError.message : String(identifyLlmError),
      });
      throw identifyLlmError;
    }
    const identifyTime = Date.now() - identifyStart;
    console.log('[identify] Dish identification complete in', identifyTime, 'ms');
    console.log('[identify] Dish:', dish.dish_name, 'Confidence:', dish.confidence);

    // Extract nutrition data
    console.log('[nutrition] Starting nutrition extraction...');
    const nutritionStart = Date.now();
    let nutritionData;
    try {
      nutritionData = await extractNutrition(imageBuffer);
      logAiOp({
        service: 'dish', op: 'extract_nutrition', owner_id: owner, model: fullDishModel(),
        latency_ms: Date.now() - nutritionStart, status: 'success', job_id: jobId,
        input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
        output: `calories=${nutritionData && nutritionData.calories} protein=${nutritionData && nutritionData.protein} serving_size=${nutritionData && nutritionData.serving_size}`,
      });
    } catch (nutritionLlmError) {
      logAiOp({
        service: 'dish', op: 'extract_nutrition', owner_id: owner, model: fullDishModel(),
        latency_ms: Date.now() - nutritionStart, status: 'error', job_id: jobId,
        input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
        error: nutritionLlmError && nutritionLlmError.message ? nutritionLlmError.message : String(nutritionLlmError),
      });
      throw nutritionLlmError;
    }
    const nutritionTime = Date.now() - nutritionStart;
    console.log('[nutrition] Nutrition extraction complete in', nutritionTime, 'ms');
    console.log('[nutrition] Calories:', nutritionData.calories, 'Protein:', nutritionData.protein);

    let packagedItem = null;
    let finalDish = dish;
    let finalNutritionData = nutritionData;

    if (shouldAttemptPackagedLookup(dish, nutritionData)) {
      try {
        console.log('[packaged] Attempting packaged item identification...');
        const packagedStart = Date.now();
        try {
          packagedItem = await identifyPackagedItem(imageBuffer);
          logAiOp({
            service: 'dish', op: 'identify_packaged', owner_id: owner, model: fullDishModel(),
            latency_ms: Date.now() - packagedStart, status: 'success', job_id: jobId,
            input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
            output: `is_packaged=${packagedItem && packagedItem.is_packaged_item} product_name=${packagedItem && packagedItem.product_name} confidence=${packagedItem && packagedItem.confidence}`,
          });
        } catch (packagedLlmError) {
          logAiOp({
            service: 'dish', op: 'identify_packaged', owner_id: owner, model: fullDishModel(),
            latency_ms: Date.now() - packagedStart, status: 'error', job_id: jobId,
            input: `job_id=${jobId} image_bytes=${imageBuffer.length}`,
            error: packagedLlmError && packagedLlmError.message ? packagedLlmError.message : String(packagedLlmError),
          });
          throw packagedLlmError;
        }
        console.log('[packaged] Result:', {
          is_packaged_item: packagedItem.is_packaged_item,
          brand: packagedItem.brand,
          product_name: packagedItem.product_name,
          barcode: packagedItem.barcode,
          confidence: packagedItem.confidence,
        });

        if (
          packagedItem.is_packaged_item &&
          packagedItem.confidence >= 0.7 &&
          packagedItem.product_name
        ) {
          finalDish = mergeDishWithPackagedItem(dish, packagedItem, null);
          const lookedUpNutrition = await lookupPackagedNutrition(packagedItem);
          if (lookedUpNutrition) {
            finalNutritionData = mergeNutritionData(nutritionData, lookedUpNutrition);
            finalDish = mergeDishWithPackagedItem(finalDish, packagedItem, lookedUpNutrition);
            console.log('[packaged] Using Open Food Facts nutrition override:', {
              dish_name: finalNutritionData.dish_name,
              calories: finalNutritionData.calories,
              carbs: finalNutritionData.total_carbohydrates,
              protein: finalNutritionData.protein,
              serving_size: finalNutritionData.serving_size,
            });
          } else {
            console.log('[packaged] No confident nutrition match found, keeping model estimate.');
          }
        }
      } catch (packagedError) {
        console.error('[packaged] Packaged lookup failed (non-fatal):', packagedError);
      }
    }

    // imageUrl / resizedImageUrl / resizedImageKey are declared above (before the
    // fast path) so the preliminary write can use imageUrl without a TDZ error.
    try {
      const resizedUpload = await uploadResizedOriginalImage(imageBuffer, user_id, device_id, jobId);
      resizedImageUrl = resizedUpload.url;
      resizedImageKey = resizedUpload.key;
      console.log('[resized-image] Uploaded resized original image:', {
        resized_image_url: resizedImageUrl,
        resized_image_key: resizedImageKey,
      });
    } catch (resizeUploadError) {
      console.error('[resized-image] Error uploading resized original image (non-fatal):', resizeUploadError);
    }

    // Find a stock dish image first, then fallback to dish image generation
    const dishSearchItem = {
      product_name: finalDish.dish_name || finalNutritionData.dish_name || 'dish',
      brand: packagedItem?.brand || null,
      category: packagedItem?.category || finalDish.category || 'dish',
      variant: packagedItem?.variant || finalDish.cuisine_type || null,
      ingredients: Array.isArray(finalNutritionData.ingredients) ? finalNutritionData.ingredients : [],
      nutrition: null,
      confidence: finalDish.confidence || 0,
    };
    let targetImageUrl = getStockImageTargetUrl(job);
    targetImageUrl = await uploadTargetImageIfNeeded(targetImageUrl, user_id, jobId);
    const requireTargetMatch = shouldRequireTargetMatch(targetImageUrl);
    if (targetImageUrl) {
      console.log('[stock-image] Using target image for verification:', {
        target_image_url: targetImageUrl,
        require_target_match: requireTargetMatch,
      });
    }
    const stockImageTimeoutMs = getStockImageTimeoutMs();
    console.log('[stock-image] Searching for dish stock image...', { timeout_ms: stockImageTimeoutMs });
    let dishImageUrl = null;
    let dishImageKey = null;
    try {
      const stockImage = await withTimeout(findStockImage(dishSearchItem, {
        targetImageUrl: targetImageUrl,
        requireTargetMatch: requireTargetMatch,
      }), stockImageTimeoutMs);
      if (stockImage?.url) {
        dishImageUrl = stockImage.url;
        console.log('[stock-image] Dish stock image found from', stockImage.source, ':', dishImageUrl);
      } else {
        console.log('[stock-image] No dish stock image found.');
      }
    } catch (stockImageError) {
      if (stockImageError?.code === 'STOCK_IMAGE_TIMEOUT') {
        console.warn('[stock-image] Search timed out (non-fatal):', stockImageError.message);
      } else {
        console.error('[stock-image] Error finding stock image (non-fatal):', stockImageError);
      }
    }

    // Write to MySQL
    console.log('[mysql] Writing to dishes table...');
    await writeToDishesTable({
      owner: owner,
      device_id: device_id,
      user_id: user_id,
      job_id: jobId,
      action: action,
      s3_key: key,
      image_url: imageUrl,
      resized_image_url: resizedImageUrl,
      resized_image_key: resizedImageKey,
      dish_image_url: dishImageUrl,
      dish_image_key: dishImageKey,
      dish: finalDish,
      nutritionData: finalNutritionData,
    });
    console.log('[mysql] Write complete');

    const dishName =
      (typeof finalDish.dish_name === 'string' && finalDish.dish_name.trim()) ||
      (typeof finalNutritionData.dish_name === 'string' && finalNutritionData.dish_name.trim()) ||
      null;
    const confidence = finalDish.confidence != null ? finalDish.confidence : (finalNutritionData.confidence || 0);
    const finalCreatedAt = isoNow();
    const finalResultPayload = buildFinalResultPayload(jobMeta, dishName, confidence, finalNutritionData, finalCreatedAt);
    finalResultPayload.dish_id = stableHouseholdRowId('dishes', jobId);

    try {
      const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
      await writeFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        event_type: 'dish',
        action,
        title: dishName || 'No dish detected',
        image_url: dishImageUrl || imageUrl,
        product_image_url: dishImageUrl || null,
        source_table: `${escapedOwner}_dishes`,
        metadata: {
          confidence,
          calories: finalNutritionData.calories || null,
          cuisine_type: finalDish.cuisine_type || null,
        },
      });
      await writeMasterFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        item_id: stableHouseholdRowId('dishes', jobId),
        event_type: 'dish_add',
        entity_type: 'dish',
        action,
        title: dishName || 'No dish detected',
        source_table: `${escapedOwner}_dishes`,
        source_path: 'analyze_dish_on_upload_nodejs/app.js',
        source_system: 'analyze_dish_on_upload',
        primary_image_url: dishImageUrl || imageUrl,
        secondary_image_url: imageUrl,
        event_key: `upload:${jobId}:dish`,
        metadata: {
          confidence,
          calories: finalNutritionData.calories || null,
          cuisine_type: finalDish.cuisine_type || null,
          serving_size: finalNutritionData.serving_size || null,
        },
      });
    } catch (feedError) {
      console.error('[feed] Error (non-fatal):', feedError);
    }

    await appendMetricsSnapshot(owner);
    console.log('[metrics] Snapshot appended');

    await writeDishResultSnapshot(jobMeta, 'final', finalResultPayload);
    console.log('[dynamo] Durable final result written');

    // Update job status to DONE
    await updateJob(jobId, {
      status: 'DONE',
      t_done: new Date().toISOString(),
      last_dish: dishName,
      confidence: confidence,
    });
    console.log('[dynamo] Job marked as DONE');
    await sendJobCompletePush(owner, dishName);

    // Publish result via IoT
    await publishResult(
      user_id,
      device_id,
      jobId,
      dishName,
      confidence,
      action
    );
    console.log('[mqtt] Result published');

    return {
      statusCode: 200,
      body: JSON.stringify({ ok: true, job_id: jobId }),
    };
  } catch (error) {
    console.error('[error] Exception:', error);
    if (jobId) {
      try {
        await updateJob(jobId, {
          status: 'FAILED',
          error_msg: error.message || String(error),
        });
      } catch (e2) {
        console.error('[error] Failed to mark job as FAILED:', e2);
      }
    }
    throw error;
  }
};
