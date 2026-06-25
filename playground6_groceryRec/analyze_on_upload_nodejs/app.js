// Lambda handler for analyzing uploaded grocery images using grocery-identifier
const AWS = require('aws-sdk');
const crypto = require('crypto');
const fetch = require('node-fetch');
const mysql = require('mysql2/promise');
const { identifyGroceryItem: identifyLegacyGroceryItem } = require('./dist/openai/identifyGrocery');
const { extractExpirationDate } = require('./dist/openai/extractExpiration');
const { generateProductIcon } = require('./dist/utils/generateProductImage');
const { assignEmojiToItem } = require('./dist/utils/assignEmoji');
const { refineProductTitle } = require('./dist/utils/refineProductTitle');
const { standardizeKitchenCategory } = require('./dist/utils/standardizeKitchenCategory');
const { estimateStorageGuidance } = require('./vendor/grocery-identifier/dist/utils/estimateStorageGuidance');
const {
  insertProvisionalKitchenRow,
  finalizeKitchenRow,
  finalizeKitchenRowAsIs,
  writeToArchiveKitchenTable,
  updateKitchenExpiration,
  markKitchenRowsFailed,
  deleteKitchenRowsByJobId,
} = require('./dist/utils/mysqlWriter');
const { writeFeedEvent } = require('./dist/utils/mysqlFeedWriter');
const { getHouseholdMemberIds, stableHouseholdRowId } = require('./dist/utils/householdSync');
const { suggestKitchenSwapsFast, suggestKitchenSwapsDeep } = require('./dist/utils/suggestKitchenSwaps');
const { syncSwapReviewPrompts, clearPendingSwapReviewPrompts } = require('./dist/utils/swapReviewWriter');
const { analyzeProduct } = require('./vendor/grocery-identifier/dist/services/analyzeProduct');
const { identifyItemFastLegacy } = require('./vendor/grocery-identifier/quickIdentify/service');
const { isLikelyNonGroceryItem } = require('./vendor/grocery-identifier/dist/openai/identifyGrocery');
const { syncImageTriage } = require('./triage');
const { writeMasterFeedEvent } = require('./masterFeedWriter');

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();
const iot = new AWS.IotData({ endpoint: process.env.IOT_ENDPOINT });
const lambda = new AWS.Lambda();

const BUCKET_NAME = process.env.BUCKET_NAME;
const JOBS_TABLE = process.env.JOBS_TABLE;
const KEY_PREFIX = process.env.KEY_PREFIX || 'images/';
const TOPIC_TEMPLATE = process.env.RESULT_TOPIC_TEMPLATE || 'trepo/{user_id}/{device_id}/jobs/{job_id}/result';
const DEFAULT_STOCK_IMAGE_TIMEOUT_MS = 120000;
const METRICS_WINDOW_DAYS = 14;
const DEFAULT_IQ = 75;
const CHECKIN_POINTS_MIN = 4;
const CHECKIN_POINTS_MAX = 7;
const THUMBNAIL_MAX_SIZE = 256;
const APP_IMAGE_MAX_DIMENSION = Math.max(256, Number(process.env.APP_IMAGE_MAX_DIMENSION || 1024));
const APP_IMAGE_QUALITY = Math.max(40, Math.min(95, Number(process.env.APP_IMAGE_QUALITY || 72)));
const UNKNOWN_ITEM_NAME = 'Unknown item';
const UNKNOWN_ITEM_EXPLANATION = 'Item could not be confidently identified from the image.';
const UNKNOWN_ITEM_CATEGORY = 'prepared_other';
const DB_CONNECT_TIMEOUT_MS = Number(process.env.DB_CONNECT_TIMEOUT_MS || 5000);
const UNKNOWN_ITEM_TEXT_PATTERNS = [
  'unknown item',
  'unidentified',
  'too blurry',
  'too blurred',
  'too unclear',
  'too dark',
  'too distant',
  'cannot identify',
  "can't identify",
  'unable to identify',
  'not clearly visible',
  'not clearly show',
  'not clearly display',
  'not clearly identifiable',
  'not enough detail',
  'exact product',
];
const CONSUMABLE_SIGNAL_PATTERNS = [
  'gum', 'candy', 'chocolate', 'cookie', 'cracker', 'chip', 'chips', 'popcorn', 'snack', 'dessert',
  'drink', 'beverage', 'juice', 'soda', 'water', 'coffee', 'tea', 'energy', 'smoothie', 'kombucha',
  'produce', 'fruit', 'vegetable', 'salad', 'dairy', 'egg', 'eggs', 'milk', 'cheese', 'yogurt',
  'meat', 'seafood', 'turkey', 'chicken', 'beef', 'pork', 'ham', 'fish',
  'pantry', 'pasta', 'rice', 'sauce', 'cereal', 'oat', 'oats', 'bread', 'frozen', 'prepared', 'soup',
  'meal', 'protein bar', 'nut butter', 'peanut butter',
];

const NOTIFICATIONS_API_URL = 'https://nua4yt5q26.execute-api.us-east-1.amazonaws.com/v1/notifications/send';

async function sendJobCompletePush(ownerId, productName, analysisMode, job) {
  if (!ownerId) return;
  const isReceipt = analysisMode && analysisMode.includes('receipt');

  // If this receipt is part of a multi-image batch, only notify when ALL siblings are DONE
  if (isReceipt && job && job.receipt_batch_id && job.receipt_batch_total > 1) {
    try {
      // Scan for all jobs in this batch
      const batchResult = await dynamodb.scan({
        TableName: JOBS_TABLE,
        FilterExpression: 'receipt_batch_id = :batchId',
        ExpressionAttributeValues: { ':batchId': job.receipt_batch_id },
      }).promise();
      const siblings = batchResult.Items || [];
      const doneCount = siblings.filter(s => s.status === 'DONE').length;
      const total = job.receipt_batch_total;
      console.log(`[push] Receipt batch ${job.receipt_batch_id}: ${doneCount}/${total} done (${siblings.length} found)`);
      if (doneCount < total) {
        console.log('[push] Skipping notification — batch not yet complete');
        return;
      }
    } catch (batchErr) {
      console.warn('[push] Batch check failed (sending notification anyway):', batchErr.message);
    }
  }

  try {
    const res = await fetch(NOTIFICATIONS_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner_id: ownerId,
        title: isReceipt ? 'Receipt Ready' : 'Item Analyzed',
        body: isReceipt
          ? 'Your receipt analysis is ready to review.'
          : productName
            ? `${productName} has been analyzed.`
            : 'Your item has been analyzed.',
        data: { route: isReceipt ? 'receipt_analysis_review' : 'bulk_session_review' },
      }),
      timeout: 5000,
    });
    console.log('[push] Notification sent for', ownerId, 'status:', res.status, 'mode:', analysisMode || 'default');
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
    console.warn('[product-icon] resize failed, using original:', e.message);
    return null;
  }
}

async function resizeForAppImage(buffer) {
  try {
    const { Jimp } = require('jimp');
    const image = await Jimp.read(buffer);
    image.scaleToFit({ w: APP_IMAGE_MAX_DIMENSION, h: APP_IMAGE_MAX_DIMENSION });
    return await image.getBuffer('image/jpeg', { quality: APP_IMAGE_QUALITY });
  } catch (resizeError) {
    console.error('[resized-image] App image resize failed:', resizeError.message);
    return null;
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

function sanitizeOwner(owner) {
  return String(owner || '').replace(/[^a-zA-Z0-9_-]/g, '');
}

function cleanText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeText(value) {
  return cleanText(value).toLowerCase();
}

function containsUnknownPattern(value) {
  const normalized = normalizeText(value);
  return normalized
    ? UNKNOWN_ITEM_TEXT_PATTERNS.some((pattern) => normalized.includes(pattern))
    : false;
}

function hasResolvedConsumableSignals(groceryItem) {
  const productName = cleanText(groceryItem?.product_name);
  const brand = cleanText(groceryItem?.brand);
  const category = cleanText(groceryItem?.category);
  const productDescription = cleanText(groceryItem?.product_description);

  const hasNamedSignal = Boolean(
    (productName && !containsUnknownPattern(productName)) ||
    (brand && !containsUnknownPattern(brand)) ||
    (productDescription && !containsUnknownPattern(productDescription))
  );
  const hasCategorySignal = Boolean(category && !containsUnknownPattern(category));

  if (hasNamedSignal && hasCategorySignal) {
    return true;
  }

  const combinedSignals = [productName, brand, category, productDescription]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();

  return Boolean(
    combinedSignals &&
    CONSUMABLE_SIGNAL_PATTERNS.some((pattern) => combinedSignals.includes(pattern))
  );
}

function shouldStandardizeUnknownItem(groceryItem) {
  if (!groceryItem || typeof groceryItem !== 'object') return false;
  if (isLikelyNonGroceryItem(groceryItem)) return false;

  const productName = cleanText(groceryItem.product_name);
  const brand = cleanText(groceryItem.brand);
  const category = cleanText(groceryItem.category);
  const explanation = cleanText(groceryItem.explanation);
  const productDescription = cleanText(groceryItem.product_description);

  if (hasResolvedConsumableSignals(groceryItem)) {
    return false;
  }

  if (!productName && !brand && !category && !productDescription) {
    return true;
  }

  return containsUnknownPattern(productName) || containsUnknownPattern(explanation);
}

function applyUnknownItemPlaceholder(groceryItem) {
  return {
    ...groceryItem,
    product_name: UNKNOWN_ITEM_NAME,
    brand: null,
    variant: null,
    category: UNKNOWN_ITEM_CATEGORY,
    estimated_price: null,
    ingredients: [],
    nutrition_summary: null,
    upf: null,
    harmful_ingredients: [],
    similar_items: [],
    alternatives: [],
    healthier_alternatives: [],
    explanation: UNKNOWN_ITEM_EXPLANATION,
    barcode: null,
    country_guess: null,
    product_description: null,
  };
}

function isUnknownItemPlaceholder(groceryItem) {
  return normalizeText(groceryItem?.product_name) === normalizeText(UNKNOWN_ITEM_NAME);
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
  return mysql.createConnection({
    host: DB_HOST,
    port: DB_PORT ? parseInt(DB_PORT, 10) : 3306,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    connectTimeout: DB_CONNECT_TIMEOUT_MS,
  });
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
  if (columns.has('analysis_stage')) {
    clauses.push("(`analysis_stage` = 'final' OR `analysis_stage` IS NULL)");
  }
  if (columns.has('analysis_status')) {
    clauses.push("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)");
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
  return CHECKIN_POINTS_MIN + Math.floor(Math.random() * (CHECKIN_POINTS_MAX - CHECKIN_POINTS_MIN + 1));
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
    }
  } finally {
    await conn.end();
  }
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
  const baseId = match[3];
  const isExpiration = baseId.endsWith('_exp');
  return {
    user_id: match[1],
    device_id: match[2],
    job_id: isExpiration ? baseId.slice(0, -4) : baseId,
    is_expiration: isExpiration,
  };
}

function normalizeQuantity(value) {
  if (value === undefined || value === null) return 1;
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return 1;
  return parsed;
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

function choosePreferredValue(primary, secondary) {
  if (primary === undefined || primary === null) return secondary ?? null;
  if (typeof primary === 'string' && !primary.trim()) return secondary ?? null;
  if (Array.isArray(primary) && primary.length === 0) return Array.isArray(secondary) ? secondary : [];
  return primary;
}

function mergeUniqueStrings(primary, secondary) {
  const values = [];
  const seen = new Set();
  for (const source of [primary, secondary]) {
    for (const item of Array.isArray(source) ? source : []) {
      const text = item == null ? '' : String(item).trim();
      if (!text) continue;
      const key = text.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      values.push(text);
    }
  }
  return values;
}

function normalizeFastCategory(fastItem) {
  const explicitCategory = cleanText(fastItem?.category);
  if (explicitCategory) return explicitCategory;
  if (fastItem?.item_type === 'produce') return 'produce_fresh';
  if (fastItem?.item_type === 'packaged') return 'packaged_other';
  return UNKNOWN_ITEM_CATEGORY;
}

function fastItemExplanation(fastItem) {
  const pieces = [];
  const shortDescription = cleanText(fastItem?.short_description);
  const visibleText = Array.isArray(fastItem?.visible_text) ? fastItem.visible_text.map(cleanText).filter(Boolean) : [];
  if (shortDescription) pieces.push(shortDescription);
  if (visibleText.length > 0) pieces.push(`Visible text: ${visibleText.join(', ')}`);
  return pieces.join(' ') || 'Quick provisional identification.';
}

function toProvisionalGroceryItem(fastItem) {
  const description = cleanText(fastItem?.short_description);
  return {
    product_name: cleanText(fastItem?.item_name) || UNKNOWN_ITEM_NAME,
    brand: cleanText(fastItem?.brand) || null,
    variant: null,
    category: normalizeFastCategory(fastItem),
    confidence: typeof fastItem?.confidence === 'number' ? fastItem.confidence : 0,
    explanation: fastItemExplanation(fastItem),
    product_description: description || null,
    barcode: null,
    country_guess: null,
    estimated_price: null,
    ingredients: [],
    nutrition_summary: null,
    upf: 'no',
    harmful_ingredients: [],
    similar_items: [],
    alternatives: [],
    healthier_alternatives: [],
  };
}

const FAST_GENERIC_CONTAINER_TOKENS = ['bag', 'box', 'package', 'pack', 'container', 'carton', 'pouch'];

function fastCandidateScore(item) {
  const itemName = cleanText(item?.item_name).toLowerCase();
  const visibleTextCount = Array.isArray(item?.visible_text) ? item.visible_text.map(cleanText).filter(Boolean).length : 0;
  const hasGenericContainerName = FAST_GENERIC_CONTAINER_TOKENS.some((token) => (
    itemName === token || itemName.endsWith(` ${token}`)
  ));

  let score = Number(item?.confidence || 0) * 100;
  if (!item?.needs_review) score += 15;
  if (cleanText(item?.brand)) score += 8;
  if (cleanText(item?.category)) score += 4;
  score += Math.min(visibleTextCount, 4) * 6;
  if (itemName && !hasGenericContainerName) score += 10;
  if (hasGenericContainerName && visibleTextCount === 0) score -= 12;
  return score;
}

function selectBestFastItem(fastResult) {
  const items = Array.isArray(fastResult?.items) ? fastResult.items : [];
  const ranked = items
    .filter((item) => cleanText(item?.item_name))
    .sort((a, b) => fastCandidateScore(b) - fastCandidateScore(a));
  return ranked[0] || null;
}

function mergeGroceryItems(modernItem, legacyItem) {
  if (!legacyItem) {
    return modernItem;
  }

  return {
    ...legacyItem,
    ...modernItem,
    brand: choosePreferredValue(modernItem.brand, legacyItem.brand),
    product_name: choosePreferredValue(modernItem.product_name, legacyItem.product_name),
    variant: choosePreferredValue(modernItem.variant, legacyItem.variant),
    category: choosePreferredValue(modernItem.category, legacyItem.category),
    estimated_price: choosePreferredValue(modernItem.estimated_price, legacyItem.estimated_price),
    ingredients: mergeUniqueStrings(modernItem.ingredients, legacyItem.ingredients),
    nutrition_summary: choosePreferredValue(modernItem.nutrition_summary, legacyItem.nutrition_summary),
    upf: choosePreferredValue(legacyItem.upf, modernItem.upf) || 'no',
    harmful_ingredients: mergeUniqueStrings(legacyItem.harmful_ingredients, modernItem.harmful_ingredients),
    similar_items: Array.isArray(modernItem.similar_items) && modernItem.similar_items.length > 0
      ? modernItem.similar_items
      : (legacyItem.similar_items || []),
    alternatives: Array.isArray(modernItem.alternatives) && modernItem.alternatives.length > 0
      ? modernItem.alternatives
      : (legacyItem.alternatives || []),
    healthier_alternatives: Array.isArray(modernItem.healthier_alternatives) && modernItem.healthier_alternatives.length > 0
      ? modernItem.healthier_alternatives
      : (legacyItem.healthier_alternatives || []),
    confidence: Math.max(Number(modernItem.confidence || 0), Number(legacyItem.confidence || 0)),
    explanation: choosePreferredValue(modernItem.explanation, legacyItem.explanation),
    barcode: choosePreferredValue(modernItem.barcode, legacyItem.barcode),
    country_guess: choosePreferredValue(modernItem.country_guess, legacyItem.country_guess),
    product_description: choosePreferredValue(modernItem.product_description, legacyItem.product_description),
  };
}

function toLegacyStoreAvailability(modernAnalysis) {
  const options = Array.isArray(modernAnalysis?.store_availability) ? modernAnalysis.store_availability : [];
  return {
    store_availability: options.map((option) => ({
      store_name: option.store_name,
      price: option.price || null,
      availability: option.availability || null,
      store_url: option.store_url || null,
      source: option.source || null,
      domain: option.domain || null,
      match_confidence: option.match_confidence ?? null,
      match_reason: option.match_reason || null,
      price_source: option.price_source || null,
      price_evidence: option.price_evidence || null,
    })),
  };
}

function inferImageExtension(contentType, imageUrl) {
  const normalizedType = String(contentType || '').toLowerCase();
  if (normalizedType.includes('png')) return 'png';
  if (normalizedType.includes('webp')) return 'webp';
  if (normalizedType.includes('gif')) return 'gif';
  if (normalizedType.includes('jpeg') || normalizedType.includes('jpg')) return 'jpg';

  try {
    const pathname = new URL(imageUrl).pathname.toLowerCase();
    if (pathname.endsWith('.png')) return 'png';
    if (pathname.endsWith('.webp')) return 'webp';
    if (pathname.endsWith('.gif')) return 'gif';
  } catch (_) {
    // ignore URL parsing failure
  }

  return 'jpg';
}

async function mirrorStockImageToS3(stockImageUrl, userId, deviceId, jobId) {
  if (!stockImageUrl) {
    return { url: null, key: null };
  }

  if (!BUCKET_NAME) {
    return { url: stockImageUrl, key: null };
  }

  try {
    const response = await fetch(stockImageUrl, {
      headers: { 'user-agent': 'Mozilla/5.0' },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const contentType = response.headers.get('content-type') || 'image/jpeg';
    if (!contentType.startsWith('image/')) {
      throw new Error(`Unexpected content type: ${contentType}`);
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    const ext = inferImageExtension(contentType, stockImageUrl);
    const key = `product-images/${userId}/${deviceId}/${jobId}.${ext}`;
    await s3.putObject({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: buffer,
      ContentType: contentType,
    }).promise();

    return {
      url: `https://${BUCKET_NAME}.s3.amazonaws.com/${key}`,
      key,
    };
  } catch (error) {
    console.warn('[stock-image] Failed to mirror stock image to S3 (non-fatal):', error);
    return {
      url: stockImageUrl,
      key: null,
    };
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
      console.warn('[triage] Failed to sync image triage:', triageError.message || triageError);
    }
  }
  return updatedJob;
}

async function getJob(jobId) {
  const jobResult = await dynamodb.get({
    TableName: JOBS_TABLE,
    Key: { job_id: jobId },
  }).promise();
  return jobResult.Item || null;
}

async function invokeKitchenAnalysisGenerator(owner, pointsDelta = 0) {
  const functionName = process.env.KITCHEN_ANALYSIS_GENERATOR_ARN;
  if (!functionName || !owner) return;
  try {
    await lambda.invoke({
      FunctionName: functionName,
      InvocationType: 'Event',
      Payload: JSON.stringify({ owner, points_delta: safeInt(pointsDelta, 0) }),
    }).promise();
  } catch (error) {
    console.warn('[kitchen-analysis] invoke failed', error.message || error);
  }
}

async function publishResult(userId, deviceId, jobId, productName, confidence, action) {
  const topic = TOPIC_TEMPLATE
    .replace('{user_id}', userId)
    .replace('{device_id}', deviceId)
    .replace('{job_id}', jobId);
  
  const payload = {
    job_id: jobId,
    mode: 'grocery',
    action: action,
    product_name: productName,
    confidence: confidence,
    summary: `${action} ${productName} (${(confidence * 100).toFixed(0)}%)`,
    created_at: new Date().toISOString(),
  };

  await iot.publish({
    topic: topic,
    qos: 1,
    payload: JSON.stringify(payload),
  }).promise();
}

function shouldSkipKitchenInsert(groceryItem) {
  if (!groceryItem || typeof groceryItem !== 'object') return true;
  if (isLikelyNonGroceryItem(groceryItem)) return true;
  return false;
}

// Emits a structured marker for silent (non-throwing) analysis failures so a
// CloudWatch metric filter on "analysis_failed" can alert. Hard failures rethrow
// and are caught by the Lambda Errors alarm; this covers the swallowed/degraded
// paths that return 200 (e.g. the preliminary kitchen-item write failing).
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

exports.handler = async (event, context) => {
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
  let provisionalKitchenInserted = false;
  let provisionalKitchenResolved = false;
  let provisionalKitchenOwner = null;
  // Function-scoped so the top-level catch can reference it for cleanup
  // (clearPendingSwapReviewPrompts / markKitchenRowsFailed). Previously `let
  // quantity` was declared inside the try, so a deep-analysis failure threw
  // `ReferenceError: quantity is not defined` in the catch, aborting the
  // provisional-row + job-FAILED cleanup.
  let quantity = 1;
  try {
    const ids = extractIdsFromKey(key);
    if (!ids) {
      console.log('[handler] Key does not match analyzer upload pattern; skipping.', { key });
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'unexpected_key_pattern' }) };
    }
    const { user_id, device_id } = ids;
    jobId = ids.job_id;
    const isExpirationImage = !!ids.is_expiration;
    console.log('[handler] Parsed IDs:', ids);

    // Mark job as PROCESSING
    const startUpdate = isExpirationImage
      ? { status: 'PROCESSING', expiration_status: 'PROCESSING', t_expiration_start: new Date().toISOString() }
      : {
        status: 'PROCESSING',
        grocery_status: 'PROCESSING',
        fast_status: 'PROCESSING',
        deep_status: 'PROCESSING',
        t_start_proc: new Date().toISOString(),
      };
    await updateJob(jobId, startUpdate);

    // Load job to get owner, action, and product_expiration
    const job = (await getJob(jobId)) || { ...ids, job_id: jobId };
    const owner = job.owner;
    provisionalKitchenOwner = owner || null;
    const action = job.action || 'IN';
    let productExpiration = job.product_expiration || null;
    const expirationExpected = !!(job.expiration_expected || job.expiration_s3_key);
    quantity = normalizeQuantity(job.quantity);

    if (!owner) {
      console.error('[error] owner parameter required in job');
      await updateJob(jobId, {
        status: 'FAILED',
        error_msg: 'owner parameter required',
      });
      reportAnalysisFailed('grocery', 'owner_missing', jobId, 'owner parameter required');
      return {
        statusCode: 400,
        body: JSON.stringify({ ok: false, error: 'owner parameter required' }),
      };
    }

    // Only process grocery type jobs (skip dish and discard jobs)
    if (job.type && job.type === 'dish') {
      console.log('[handler] Job type is dish, skipping (handled by dish analyzer):', job.type);
      await updateJob(jobId, { deep_status: 'SKIPPED', fast_status: 'SKIPPED' });
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'dish job handled by dish analyzer' }) };
    }
    if (job.type && job.type === 'discard') {
      console.log('[handler] Job type is discard, skipping (handled by discard analyzer):', job.type);
      await updateJob(jobId, { deep_status: 'SKIPPED', fast_status: 'SKIPPED' });
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'discard job handled by discard analyzer' }) };
    }

    // Extract user note and capture mode from camera_meta (for leftovers/manual hints)
    const cameraMeta = job.camera_meta || {};
    const userNote = (typeof cameraMeta === 'object' ? cameraMeta.user_note : null) || null;
    const captureMode = (typeof cameraMeta === 'object' ? cameraMeta.capture_mode : null) || null;
    const isLeftovers = captureMode === 'leftovers';

    console.log('[dynamo] Loaded job:', {
      has_action: !!job.action,
      action: action,
      owner: owner,
      user_note: userNote || '(none)',
      capture_mode: captureMode || '(none)',
    });

    // Download image from S3
    console.log('[s3] Downloading image...');
    const s3Object = await s3.getObject({
      Bucket: BUCKET_NAME,
      Key: key,
    }).promise();

    const imageBuffer = Buffer.from(s3Object.Body);
    const imageContentType = s3Object.ContentType || (key.endsWith('.png') ? 'image/png' : 'image/jpeg');
    console.log('[s3] Image downloaded, size:', imageBuffer.length, 'bytes');

    if (isExpirationImage) {
      console.log('[expiration] Extracting expiration date from image...');
      let expirationDate = null;
      let expirationRawText = null;
      let expirationConfidence = null;
      let expirationStatus = 'FAILED';
      try {
        const expirationResult = await extractExpirationDate(imageBuffer, imageContentType);
        expirationDate = expirationResult.expirationDate;
        expirationRawText = expirationResult.rawText;
        expirationConfidence = expirationResult.confidence;
        expirationStatus = expirationDate ? 'DONE' : 'FAILED';
      } catch (expirationError) {
        console.error('[expiration] OCR failed:', expirationError);
        expirationStatus = 'FAILED';
      }

      await updateJob(jobId, {
        expiration_status: expirationStatus,
        product_expiration: expirationDate,
        expiration_raw_text: expirationRawText,
        expiration_confidence: expirationConfidence,
        t_expiration_done: new Date().toISOString(),
      });

      try {
        await updateKitchenExpiration({
          owner,
          job_id: jobId,
          product_expiration: expirationDate,
        });
        await invokeKitchenAnalysisGenerator(owner, 0);
      } catch (updateError) {
        console.warn('[expiration] MySQL update failed (non-fatal):', updateError);
      }

      const latestJob = await getJob(jobId);
      const groceryStatus = latestJob?.grocery_status || latestJob?.status;
      const expirationComplete = !expirationExpected || ['DONE', 'FAILED'].includes(expirationStatus);
      if (groceryStatus === 'DONE' && expirationComplete) {
        await updateJob(jobId, {
          status: 'DONE',
          t_done: new Date().toISOString(),
        });
        console.log('[dynamo] Job marked as DONE (grocery + expiration complete)');
        const finalJob = await getJob(jobId);
        await sendJobCompletePush(owner, latestJob?.last_product, job.analysis_mode, finalJob);
      }

      return {
        statusCode: 200,
        body: JSON.stringify({ ok: true, job_id: jobId, product_expiration: expirationDate }),
      };
    }

    const imageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${key}`;
    const analysisImageUrl = s3.getSignedUrl('getObject', {
      Bucket: BUCKET_NAME,
      Key: key,
      Expires: 900,
    });
    let resizedImageUrl = null;
    let resizedImageKey = null;

    // --- Preliminary kitchen presence: insert placeholder BEFORE any AI analysis ---
    try {
      console.log('[preliminary] Creating preliminary kitchen item for immediate presence...');
      const preliminaryGroceryItem = {
        product_name: 'Analyzing...',
        brand: null,
        variant: null,
        category: null,
        confidence: 0,
        explanation: 'Item is being analyzed.',
        product_description: null,
        barcode: null,
        country_guess: null,
        estimated_price: null,
        ingredients: [],
        nutrition_summary: null,
        upf: 'no',
        harmful_ingredients: [],
        similar_items: [],
        alternatives: [],
        healthier_alternatives: [],
      };
      await insertProvisionalKitchenRow({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        action,
        quantity,
        product_expiration: productExpiration,
        storage_guidance: null,
        s3_key: key,
        image_url: imageUrl,
        resized_image_url: null,
        resized_image_key: null,
        product_image_url: null,
        product_image_key: null,
        swaps: null,
        groceryItem: preliminaryGroceryItem,
        storeAvailability: undefined,
        analysis_source: 'preliminary',
        analysis_stage_override: 'preliminary',
        needs_review: false,
        provisional_payload: null,
      });
      provisionalKitchenInserted = true;
      console.log('[preliminary] Preliminary kitchen item created successfully.');
      await updateJob(jobId, {
        preliminary_status: 'DONE',
        t_preliminary_done: new Date().toISOString(),
      });
    } catch (preliminaryError) {
      console.error('[preliminary] Failed to create preliminary kitchen item (non-fatal):', preliminaryError);
      reportAnalysisFailed('grocery', 'preliminary_write', jobId, preliminaryError);
      // Non-fatal: fast/deep analysis will still create the kitchen row
    }

    // --- Resize image early so the fast scan sends a smaller payload to OpenAI ---
    let fastImageBuffer = imageBuffer;
    try {
      const resizedBuffer = await resizeForAppImage(imageBuffer);
      if (resizedBuffer) {
        console.log('[fast-identify] Resized image for fast scan:', {
          original_bytes: imageBuffer.length,
          resized_bytes: resizedBuffer.length,
        });
        fastImageBuffer = resizedBuffer;
      }
    } catch (resizeErr) {
      console.warn('[fast-identify] Pre-resize failed, using original buffer:', resizeErr.message);
    }

    try {
      console.log('[fast-identify] Starting provisional legacy fast scan...', userNote ? `(user note: ${userNote})` : '');
      const fastScan = await identifyItemFastLegacy({ imageBuffer: fastImageBuffer, userHint: userNote });
      const fastCandidate = selectBestFastItem(fastScan);
      if (fastCandidate) {
        if (isLeftovers) fastCandidate.category = 'leftovers';
        const provisionalGroceryItem = toProvisionalGroceryItem(fastCandidate);
        let fastSwaps = {
          candidate_ids: [],
          generated_by: 'fast_local',
          generated_at: new Date().toISOString(),
          reason_summary: 'Fast swap ranking failed before provisional write.',
        };
        try {
          fastSwaps = await suggestKitchenSwapsFast({
            owner,
            jobId,
            groceryItem: provisionalGroceryItem,
          });
        } catch (swapError) {
          console.warn('[swap-suggestions] Fast ranking failed (non-fatal):', swapError);
        }
        await insertProvisionalKitchenRow({
          owner,
          device_id,
          user_id,
          job_id: jobId,
          action,
          quantity,
          product_expiration: productExpiration,
          storage_guidance: null,
          s3_key: key,
          image_url: imageUrl,
          resized_image_url: resizedImageUrl,
          resized_image_key: resizedImageKey,
          product_image_url: null,
          product_image_key: null,
          swaps: fastSwaps,
          groceryItem: provisionalGroceryItem,
          storeAvailability: undefined,
          analysis_source: 'quick_identify',
          needs_review: !!fastCandidate.needs_review,
          provisional_payload: {
            fast_mode: 'fast_legacy',
            selected_item: fastCandidate,
            items: Array.isArray(fastScan?.items) ? fastScan.items : [],
            debug: fastScan?.debug || null,
          },
        });
        await syncSwapReviewPrompts({
          owner,
          job_id: jobId,
          quantity,
          swaps: fastSwaps,
        });
        provisionalKitchenInserted = true;
        console.log('[fast-identify] Provisional kitchen rows inserted:', {
          item_name: fastCandidate.item_name || null,
          confidence: fastCandidate.confidence || null,
          needs_review: !!fastCandidate.needs_review,
        });
        await updateJob(jobId, {
          fast_status: 'DONE',
          t_fast_done: new Date().toISOString(),
          fast_last_product: fastCandidate.item_name || null,
          fast_confidence: fastCandidate.confidence || null,
          fast_needs_review: !!fastCandidate.needs_review,
          fast_mode: 'fast_legacy',
        });

        // Trigger recipe + meal plan generation immediately after fast analysis (don't wait for enrichment)
        const recipesArnFast = process.env.RECIPES_GENERATOR_ARN;
        if (recipesArnFast && owner) {
          console.log('[recipes] Triggering recipe gen immediately after fast analysis for', owner);
          lambda.invoke({
            FunctionName: recipesArnFast,
            InvocationType: 'Event',
            Payload: JSON.stringify({ owner }),
          }).promise().catch(err => console.warn('[recipes] fast-trigger invoke failed', err.message));
        }
        const mealPlanArnFast = process.env.MEAL_PLAN_GENERATOR_ARN;
        if (mealPlanArnFast && owner) {
          console.log('[meal-plan] Triggering meal plan gen immediately after fast analysis for', owner);
          lambda.invoke({
            FunctionName: mealPlanArnFast,
            InvocationType: 'Event',
            Payload: JSON.stringify({ owner }),
          }).promise().catch(err => console.warn('[meal-plan] fast-trigger invoke failed', err.message));
        }
      } else {
        console.log('[fast-identify] No provisional fast candidate found.');
        await updateJob(jobId, {
          fast_status: 'SKIPPED',
          t_fast_done: new Date().toISOString(),
          fast_error: 'No provisional fast candidate found',
        });
      }
    } catch (fastError) {
      console.warn('[fast-identify] Fast scan failed (non-fatal):', fastError);
      await updateJob(jobId, {
        fast_status: 'FAILED',
        t_fast_done: new Date().toISOString(),
        fast_error: fastError.message || String(fastError),
      });
    }

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

    // Check remaining Lambda time before starting deep analysis (~120s needed)
    const remainingMs = context?.getRemainingTimeInMillis?.() || 300000;
    const remainingSec = Math.round(remainingMs / 1000);
    console.log(`[timing] Remaining Lambda time before deep analysis: ${remainingSec}s`);
    if (remainingMs < 120000) {
      console.warn(`[timing] Only ${remainingSec}s remaining — skipping deep analysis, finalizing with fast results`);
      // Finalize the provisional row as-is (fast analysis already wrote it)
      await updateJob(jobId, {
        status: 'DONE',
        grocery_status: 'DONE',
        deep_status: 'TIMEOUT_SKIPPED',
        t_done: new Date().toISOString(),
        error_msg: `Deep analysis skipped: only ${remainingSec}s remaining (need 120s)`,
      });
      if (provisionalKitchenInserted) {
        // Mark the fast-analysis rows as final so they don't show as "in progress"
        try {
          await finalizeKitchenRowAsIs(owner, jobId);
          provisionalKitchenResolved = true;
        } catch (finalizeErr) {
          console.error('[timing] Failed to finalize fast rows:', finalizeErr);
        }
      }
      return {
        statusCode: 200,
        body: JSON.stringify({ ok: true, job_id: jobId, deep_skipped: true }),
      };
    }

    console.log('[identify] Starting modern grocery analysis...', userNote ? `(user note: ${userNote})` : '');
    const modernAnalysis = await analyzeProduct(
      { imageUrl: analysisImageUrl },
      { stockImageMode: 'deep', userHint: userNote }
    );
    console.log('[identify] Modern analysis complete:', {
      product_name: modernAnalysis.groceryItem?.product_name || null,
      brand: modernAnalysis.groceryItem?.brand || null,
      category: modernAnalysis.groceryItem?.category || null,
      identify_ms: modernAnalysis.debug?.identify_ms || null,
      retailer_ms: modernAnalysis.debug?.retailer_search_ms || null,
      stock_image_status: modernAnalysis.stock_image?.status || null,
    });

    if (shouldSkipKitchenInsert(modernAnalysis.groceryItem)) {
      console.log('[identify] Non-grocery or unidentified scene detected; archiving unknown placeholder instead of skipping.');
      const archivedUnknownItem = applyUnknownItemPlaceholder(modernAnalysis.groceryItem || {});
      if (provisionalKitchenInserted) {
        try {
          await deleteKitchenRowsByJobId(owner, jobId);
          await clearPendingSwapReviewPrompts({
            owner,
            job_id: jobId,
            quantity,
          });
          provisionalKitchenResolved = true;
          console.log('[mysql] Removed provisional kitchen rows before archiving unknown result.');
        } catch (deleteError) {
          console.error('[mysql] Failed to remove provisional kitchen rows before archiving unknown result:', deleteError);
          throw deleteError;
        }
      }
      try {
        await writeToArchiveKitchenTable({
          owner: owner,
          device_id: device_id,
          user_id: user_id,
          job_id: jobId,
          action: action,
          quantity: quantity,
          product_expiration: productExpiration,
          s3_key: key,
          image_url: imageUrl,
          product_image_url: null,
          product_image_key: null,
          groceryItem: archivedUnknownItem,
          storeAvailability: undefined,
          archived_reason: 'unidentified_item',
        });
        console.log('[mysql] Unknown placeholder archived from skip branch.');
      } catch (archiveError) {
        console.error('[mysql] Failed to archive skipped unidentified item:', archiveError);
        throw archiveError;
      }

      try {
        const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
        await writeFeedEvent({
          owner,
          device_id,
          user_id,
          job_id: jobId,
          event_type: action === 'OUT' ? 'checkout' : 'checkin',
          action,
          title: archivedUnknownItem.product_name,
          brand: null,
          image_url: null,
          product_image_url: null,
          source_table: `${escapedOwner}_archive_kitchen`,
          metadata: {
            confidence: archivedUnknownItem.confidence || modernAnalysis.groceryItem?.confidence || null,
            category: archivedUnknownItem.category || null,
            variant: null,
            barcode: null,
            product_expiration: productExpiration || null,
            quantity: quantity || 1,
            archived_immediately: true,
          },
        });
        await writeMasterFeedEvent({
          owner,
          device_id,
          user_id,
          job_id: jobId,
          item_id: stableHouseholdRowId('prod_kitchen', jobId, 0),
          event_type: action === 'OUT' ? 'kitchen_remove' : 'kitchen_add',
          entity_type: 'kitchen_item',
          action,
          title: archivedUnknownItem.product_name,
          source_table: `${escapedOwner}_archive_kitchen`,
          source_path: 'analyze_on_upload_nodejs/app.js',
          source_system: 'analyze_on_upload',
          event_key: `upload:${jobId}:kitchen:${action}:archived_unknown`,
          metadata: {
            confidence: archivedUnknownItem.confidence || modernAnalysis.groceryItem?.confidence || null,
            category: archivedUnknownItem.category || null,
            product_expiration: productExpiration || null,
            quantity: quantity || 1,
            archived_immediately: true,
            archived_reason: 'unidentified_item',
          },
        });
      } catch (feedError) {
        console.error('[feed] Error writing archived unknown feed event (non-fatal):', feedError);
      }

      await updateJob(jobId, {
        status: 'DONE',
        grocery_status: 'ARCHIVED_UNIDENTIFIED',
        deep_status: 'DONE',
        last_product: archivedUnknownItem.product_name,
        confidence: archivedUnknownItem.confidence || modernAnalysis.groceryItem?.confidence || 0,
        error_msg: 'Archived unidentified item',
        t_done: new Date().toISOString(),
      });
      await sendJobCompletePush(owner, archivedUnknownItem.product_name, job.analysis_mode, job);
      await publishResult(
        user_id,
        device_id,
        jobId,
        archivedUnknownItem.product_name,
        archivedUnknownItem.confidence || modernAnalysis.groceryItem?.confidence || 0,
        action
      );
      return {
        statusCode: 200,
        body: JSON.stringify({ ok: true, job_id: jobId, archived_unidentified: true }),
      };
    }

    let groceryItem = modernAnalysis.groceryItem;
    let storeAvailability = { store_availability: [] }; // retail search disabled

    console.log('[identify] Running legacy enrichment for reverse-compatible fields...');
    try {
      const legacyGroceryItem = await identifyLegacyGroceryItem(imageBuffer);
      groceryItem = mergeGroceryItems(modernAnalysis.groceryItem, legacyGroceryItem);
      console.log('[identify] Legacy enrichment merged:', {
        upf: groceryItem.upf || null,
        harmful_ingredients: Array.isArray(groceryItem.harmful_ingredients) ? groceryItem.harmful_ingredients.length : 0,
        ingredients: Array.isArray(groceryItem.ingredients) ? groceryItem.ingredients.length : 0,
      });
    } catch (legacyIdentifyError) {
      console.warn('[identify] Legacy enrichment failed (non-fatal):', legacyIdentifyError);
    }

    let storageGuidance = null;

    if (shouldStandardizeUnknownItem(groceryItem)) {
      groceryItem = applyUnknownItemPlaceholder(groceryItem);
      storeAvailability = [];
      console.log('[identify] Standardized unresolved grocery item to unknown placeholder.');
    } else {
      try {
        groceryItem = await refineProductTitle(groceryItem);
        console.log('[title-refine] Final product title:', {
          product_name: groceryItem.product_name || null,
          brand: groceryItem.brand || null,
          variant: groceryItem.variant || null,
        });
      } catch (titleError) {
        console.warn('[title-refine] Failed (non-fatal):', titleError);
      }

      try {
        groceryItem = await standardizeKitchenCategory(groceryItem);
        console.log('[category-standardize] Final category:', {
          product_name: groceryItem.product_name || null,
          category: groceryItem.category || null,
        });
      } catch (categoryError) {
        console.warn('[category-standardize] Failed (non-fatal):', categoryError);
      }

      // Force category to "leftovers" when capture_mode indicates leftovers
      if (isLeftovers) {
        groceryItem.category = 'leftovers';
        console.log('[category-override] Forced category to leftovers (capture_mode=leftovers)');
      }

      try {
        storageGuidance = await estimateStorageGuidance(groceryItem);
        console.log('[storage-guidance] Final guidance:', storageGuidance);
      } catch (storageGuidanceError) {
        console.warn('[storage-guidance] Failed (non-fatal):', storageGuidanceError);
      }
    }

    let productImageUrl = null;
    let productImageKey = null;
    if (isUnknownItemPlaceholder(groceryItem)) {
      console.log('[assign-emoji] Unknown item placeholder; leaving product image empty.');
    } else {
      try {
        const emoji = await assignEmojiToItem(groceryItem);
        productImageUrl = `emoji:${emoji}`;
        resizedImageUrl = `emoji:${emoji}`;
        console.log('[assign-emoji] Assigned emoji:', { product_name: groceryItem.product_name, emoji, productImageUrl });
      } catch (emojiError) {
        console.error('[assign-emoji] Failed (non-fatal):', emojiError);
        productImageUrl = 'emoji:🍽️';
        resizedImageUrl = 'emoji:🍽️';
        console.log('[assign-emoji] Using fallback emoji: 🍽️');
      }
    }

    // Refresh expiration date if OCR finished before insert
    const latestJobForInsert = await getJob(jobId);
    if (latestJobForInsert && latestJobForInsert.product_expiration) {
      productExpiration = latestJobForInsert.product_expiration;
    }

    const isUnknownArchivedItem = isUnknownItemPlaceholder(groceryItem);

    if (isUnknownArchivedItem) {
      if (provisionalKitchenInserted) {
        await deleteKitchenRowsByJobId(owner, jobId);
        await clearPendingSwapReviewPrompts({
          owner,
          job_id: jobId,
          quantity,
        });
        provisionalKitchenResolved = true;
        console.log('[mysql] Removed provisional kitchen rows before archiving finalized unknown result.');
      }
      console.log('[mysql] Writing unknown placeholder directly to archive kitchen table...');
      await writeToArchiveKitchenTable({
        owner: owner,
        device_id: device_id,
        user_id: user_id,
        job_id: jobId,
        action: action,
        quantity: quantity,
        product_expiration: productExpiration,
        storage_guidance: null,
        s3_key: key,
        image_url: imageUrl,
        product_image_url: null,
        product_image_key: null,
        groceryItem: groceryItem,
        storeAvailability: undefined,
        archived_reason: 'unidentified_item',
      });
    } else {
      let deepSwaps = {
        candidate_ids: [],
        generated_by: 'fast_local',
        generated_at: new Date().toISOString(),
        reason_summary: 'Deep swap ranking failed before final write.',
      };
      try {
        deepSwaps = await suggestKitchenSwapsDeep({
          owner,
          jobId,
          groceryItem,
        });
      } catch (swapError) {
        console.warn('[swap-suggestions] Deep ranking failed (non-fatal):', swapError);
      }
      console.log('[mysql] Finalizing kitchen table row(s)...');
      await finalizeKitchenRow({
        owner: owner,
        device_id: device_id,
        user_id: user_id,
        job_id: jobId,
        action: action,
        quantity: quantity,
        product_expiration: productExpiration,
        storage_guidance: storageGuidance,
        s3_key: key,
        image_url: imageUrl,
        resized_image_url: resizedImageUrl,
        resized_image_key: resizedImageKey,
        product_image_url: productImageUrl,
        product_image_key: productImageKey,
        swaps: deepSwaps,
        groceryItem: groceryItem,
        storeAvailability: storeAvailability || undefined,
      });
      await syncSwapReviewPrompts({
        owner,
        job_id: jobId,
        quantity,
        swaps: deepSwaps,
      });
      provisionalKitchenResolved = true;
    }
    console.log('[mysql] Write complete');

    try {
      const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
      const eventType = action === 'OUT' ? 'checkout' : 'checkin';
      await writeFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        event_type: eventType,
        action,
        title: groceryItem.product_name || 'unknown',
        brand: groceryItem.brand || null,
        image_url: isUnknownArchivedItem ? null : (productImageUrl || imageUrl),
        product_image_url: isUnknownArchivedItem ? null : (productImageUrl || null),
        source_table: isUnknownArchivedItem ? `${escapedOwner}_archive_kitchen` : `${escapedOwner}_prod_kitchen`,
        metadata: {
          confidence: groceryItem.confidence || null,
          category: groceryItem.category || null,
          variant: groceryItem.variant || null,
          barcode: groceryItem.barcode || null,
          product_expiration: productExpiration || null,
          storage_guidance: storageGuidance || null,
          quantity: quantity || 1,
          archived_immediately: isUnknownArchivedItem,
        },
      });
      await writeMasterFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        item_id: stableHouseholdRowId('prod_kitchen', jobId, 0),
        event_type: action === 'OUT' ? 'kitchen_remove' : 'kitchen_add',
        entity_type: 'kitchen_item',
        action,
        title: groceryItem.product_name || 'unknown',
        brand: groceryItem.brand || null,
        source_table: isUnknownArchivedItem ? `${escapedOwner}_archive_kitchen` : `${escapedOwner}_prod_kitchen`,
        source_path: 'analyze_on_upload_nodejs/app.js',
        source_system: 'analyze_on_upload',
        primary_image_url: isUnknownArchivedItem ? null : (productImageUrl || imageUrl),
        secondary_image_url: isUnknownArchivedItem ? null : imageUrl,
        event_key: `upload:${jobId}:kitchen:${action}:${isUnknownArchivedItem ? 'archived' : 'live'}`,
        metadata: {
          confidence: groceryItem.confidence || null,
          category: groceryItem.category || null,
          variant: groceryItem.variant || null,
          barcode: groceryItem.barcode || null,
          product_expiration: productExpiration || null,
          storage_guidance: storageGuidance || null,
          quantity: quantity || 1,
          archived_immediately: isUnknownArchivedItem,
        },
      });
    } catch (feedError) {
      console.error('[feed] Error (non-fatal):', feedError);
    }

    if (!isUnknownArchivedItem) {
      await invokeKitchenAnalysisGenerator(owner, getScanPointsDelta());
      console.log('[kitchen-analysis] Regeneration requested');
    } else {
      console.log('[metrics] Unknown placeholder archived immediately; metrics unchanged.');
    }

    // Recipe and meal plan generation for single items is triggered at fast analysis stage (above).
    // Bulk/receipt items are handled by bulkKitchenWriter after commit.
    // No additional trigger needed here after finalize.

    const latestJob = await getJob(jobId);
    const expirationComplete = !expirationExpected || ['DONE', 'FAILED'].includes(latestJob?.expiration_status);
    const groceryUpdate = {
      status: expirationComplete ? 'DONE' : 'PROCESSING',
      grocery_status: 'DONE',
      deep_status: 'DONE',
      last_product: groceryItem.product_name || null,
      confidence: groceryItem.confidence || 0,
    };
    if (expirationComplete) {
      groceryUpdate.t_done = new Date().toISOString();
    }
    await updateJob(jobId, groceryUpdate);
    console.log('[dynamo] Job marked as DONE:', expirationComplete);

    if (expirationComplete) {
      await sendJobCompletePush(owner, groceryItem.product_name, job.analysis_mode, job);
    }

    // Publish result via IoT
    await publishResult(
      user_id,
      device_id,
      jobId,
      groceryItem.product_name || 'unknown',
      groceryItem.confidence || 0,
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
        if (provisionalKitchenInserted && !provisionalKitchenResolved && provisionalKitchenOwner) {
          await clearPendingSwapReviewPrompts({
            owner: provisionalKitchenOwner,
            job_id: jobId,
            quantity,
          });
          await markKitchenRowsFailed({
            owner: provisionalKitchenOwner,
            job_id: jobId,
            error_message: error.message || String(error),
          });
        }
        await updateJob(jobId, {
          status: 'FAILED',
          deep_status: 'FAILED',
          error_msg: error.message || String(error),
        });
      } catch (e2) {
        console.error('[error] Failed to mark job as FAILED:', e2);
      }
    }
    throw error;
  }
};
