// Lambda handler for analyzing uploaded discarded grocery images (same flow as grocery, writes to {owner}_discards)
const AWS = require('aws-sdk');
const crypto = require('crypto');
const mysql = require('mysql2/promise');
const fetch = require('node-fetch');
const { identifyGroceryItem } = require('./dist/openai/identifyGrocery');
const { getStoreAvailability } = require('./dist/openai/getStoreAvailability');
const { findStockImage } = require('./dist/utils/stockImageSearch');
const { generateProductIcon } = require('./dist/utils/generateProductImage');
const { refineProductTitle } = require('./dist/utils/refineProductTitle');
const {
  insertProvisionalDiscardRow,
  finalizeDiscardRow,
  markDiscardRowFailed,
} = require('./dist/utils/mysqlDiscardWriter');
const { matchAndRemoveFromKitchen } = require('./dist/utils/matchAndRemoveFromKitchen');
const { writeFeedEvent } = require('./dist/utils/mysqlFeedWriter');
const { getHouseholdMemberIds, stableHouseholdRowId } = require('./dist/utils/householdSync');
const { identifyItemFastLegacy, identifyItemQuickGlance } = require('./vendor/grocery-identifier/quickIdentify/service');
const { syncImageTriage } = require('./triage');
const { writeMasterFeedEvent } = require('./masterFeedWriter');

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();
const iot = new AWS.IotData({ endpoint: process.env.IOT_ENDPOINT });
const lambda = new AWS.Lambda();

const BUCKET_NAME = process.env.BUCKET_NAME;
const PRODUCT_IMAGES_BUCKET = process.env.PRODUCT_IMAGES_BUCKET || BUCKET_NAME;
const JOBS_TABLE = process.env.JOBS_TABLE;
const KEY_PREFIX = process.env.KEY_PREFIX || 'images/';
const TOPIC_TEMPLATE = process.env.RESULT_TOPIC_TEMPLATE || 'trepo/{user_id}/{device_id}/jobs/{job_id}/result';
const METRICS_WINDOW_DAYS = 14;
const DEFAULT_IQ = 75;
const DISCARD_POINTS_MIN = 9;
const DISCARD_POINTS_MAX = 12;
const APP_IMAGE_MAX_DIMENSION = Math.max(256, Number(process.env.APP_IMAGE_MAX_DIMENSION || 1024));
const APP_IMAGE_QUALITY = Math.max(40, Math.min(95, Number(process.env.APP_IMAGE_QUALITY || 72)));
const UNKNOWN_ITEM_NAME = 'Unknown item';
const UNKNOWN_ITEM_EXPLANATION = 'Item could not be confidently identified from the image.';
const UNKNOWN_ITEM_CATEGORY = 'prepared_other';
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

function isResizedDerivativeKey(key) {
  return typeof key === 'string' && key.startsWith(`${KEY_PREFIX}resized/`);
}

function isFastTimeoutError(error) {
  const message = error instanceof Error ? error.message : String(error || '');
  const normalized = message.toLowerCase();
  return (
    normalized.includes('request timed out') ||
    normalized.includes('timeout') ||
    normalized.includes('apiconnectiontimeouterror')
  );
}

async function runDiscardFastScan(imageBuffer) {
  try {
    return {
      fastScan: await identifyItemFastLegacy({ imageBuffer }),
      fastMode: 'fast_legacy',
      fallbackUsed: false,
    };
  } catch (fastError) {
    if (!isFastTimeoutError(fastError)) {
      throw fastError;
    }

    console.warn('[fast-identify] Legacy fast scan timed out; retrying with quick glance fallback.');
    return {
      fastScan: await identifyItemQuickGlance({ imageBuffer }),
      fastMode: 'quick_glance',
      fallbackUsed: true,
    };
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

function shouldStandardizeUnknownItem(groceryItem) {
  if (!groceryItem || typeof groceryItem !== 'object') return true;

  const productName = cleanText(groceryItem.product_name);
  const brand = cleanText(groceryItem.brand);
  const category = cleanText(groceryItem.category);
  const explanation = cleanText(groceryItem.explanation);

  if (!productName && !brand && !category) {
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

function normalizeOptionalBoolean(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'n', 'off', ''].includes(normalized)) return false;
  }
  if (value == null) return false;
  return Boolean(value);
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
  return {
    product_name: cleanText(fastItem?.item_name) || UNKNOWN_ITEM_NAME,
    brand: cleanText(fastItem?.brand) || null,
    variant: null,
    category: normalizeFastCategory(fastItem),
    confidence: typeof fastItem?.confidence === 'number' ? fastItem.confidence : 0,
    explanation: fastItemExplanation(fastItem),
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
    product_description: cleanText(fastItem?.short_description) || null,
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

function shoppingListTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) throw new Error('Invalid owner');
  return `${safe}_new_list`;
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

async function ensureShoppingListTable(conn, tableName) {
  await conn.query(`
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      \`_owner\` CHAR(36) NOT NULL,
      \`_device\` VARCHAR(64) NOT NULL,
      \`product_name\` VARCHAR(255) NOT NULL,
      \`product_brand\` VARCHAR(255) DEFAULT NULL,
      \`images\` TEXT,
      \`product_barcode\` VARCHAR(64) DEFAULT NULL,
      \`store\` VARCHAR(100) DEFAULT NULL,
      \`action\` VARCHAR(32) NOT NULL,
      \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`created_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`updated_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      \`household_item_uuid\` CHAR(36) DEFAULT NULL,
      PRIMARY KEY (\`_id\`),
      KEY \`idx_owner_device\` (\`_owner\`, \`_device\`),
      KEY \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  const columns = await getTableColumns(conn, tableName);
  const alterParts = [];
  if (!columns.has('_createdDate')) alterParts.push("ADD COLUMN `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER `action`");
  if (!columns.has('created_at')) alterParts.push("ADD COLUMN `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER `_createdDate`");
  if (!columns.has('updated_at')) alterParts.push("ADD COLUMN `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER `created_at`");
  if (!columns.has('household_item_uuid')) alterParts.push("ADD COLUMN `household_item_uuid` CHAR(36) DEFAULT NULL AFTER `updated_at`");
  if (!columns.has('product_brand')) alterParts.push("ADD COLUMN `product_brand` VARCHAR(255) DEFAULT NULL AFTER `product_name`");
  if (!columns.has('images')) alterParts.push("ADD COLUMN `images` TEXT AFTER `product_brand`");
  if (!columns.has('product_barcode')) alterParts.push("ADD COLUMN `product_barcode` VARCHAR(64) DEFAULT NULL AFTER `images`");
  if (!columns.has('store')) alterParts.push("ADD COLUMN `store` VARCHAR(100) DEFAULT NULL AFTER `product_barcode`");
  if (alterParts.length > 0) {
    await conn.query(`ALTER TABLE \`${tableName}\` ${alterParts.join(', ')}`);
  }
}

async function addDiscardToShoppingList({ owner, user_id, device_id, groceryItem, imageUrl }) {
  const itemName = String(groceryItem?.product_name || '').trim();
  if (!itemName) {
    throw new Error('Cannot add discard to shopping list without a recognized product_name');
  }

  const conn = await getDbConnection();
  try {
    const memberIds = await getHouseholdMemberIds(conn, owner);
    const householdItemUuid = crypto.randomUUID();
    const brand = String(groceryItem?.brand || '').trim() || null;
    const barcode = String(groceryItem?.barcode || '').trim() || null;
    const storedImageUrl = imageUrl || null;

    for (const memberId of memberIds) {
      const tableName = shoppingListTableName(memberId);
      await ensureShoppingListTable(conn, tableName);
      await conn.execute(
        `INSERT INTO \`${tableName}\` (
          _owner,
          _device,
          product_name,
          product_brand,
          images,
          product_barcode,
          store,
          action,
          _createdDate,
          created_at,
          updated_at,
          household_item_uuid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), NOW(), ?)`,
        [
          memberId,
          device_id || 'discard-analyzer',
          itemName,
          brand,
          storedImageUrl,
          barcode,
          null,
          'ADDED',
          householdItemUuid,
        ],
      );
    }

    return {
      household_item_uuid: householdItemUuid,
      item_name: itemName,
      user_id,
    };
  } finally {
    await conn.end();
  }
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
  return DISCARD_POINTS_MIN + Math.floor(Math.random() * (DISCARD_POINTS_MAX - DISCARD_POINTS_MIN + 1));
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

function extractS3FromEvent(event) {
  if (event['detail-type'] === 'Object Created' && event.detail) {
    try {
      const bucket = event.detail.bucket.name;
      const key = decodeURIComponent(event.detail.object.key);
      return { bucket, key };
    } catch (e) {
      console.error('[extract] EventBridge parse failed:', e);
    }
  }
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

  if (!PRODUCT_IMAGES_BUCKET) {
    return { url: stockImageUrl, key: null };
  }

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
    Bucket: PRODUCT_IMAGES_BUCKET,
    Key: key,
    Body: buffer,
    ContentType: contentType,
  }).promise();

  return {
    url: `https://${PRODUCT_IMAGES_BUCKET}.s3.amazonaws.com/${key}`,
    key,
  };
}

async function uploadGeneratedImageToS3(buffer, userId, deviceId, jobId) {
  if (!buffer || !PRODUCT_IMAGES_BUCKET) {
    return { url: null, key: null };
  }

  const key = `product-images/${userId}/${deviceId}/${jobId}.jpg`;
  await s3.putObject({
    Bucket: PRODUCT_IMAGES_BUCKET,
    Key: key,
    Body: buffer,
    ContentType: 'image/jpeg',
  }).promise();

  return {
    url: `https://${PRODUCT_IMAGES_BUCKET}.s3.amazonaws.com/${key}`,
    key,
  };
}

async function updateJob(jobId, attrs) {
  const nextAttrs = {
    ...attrs,
    updated_at: new Date().toISOString(),
  };
  const updateExpr = Object.keys(nextAttrs)
    .map((k) => `#${k} = :${k}`)
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
      console.warn('[triage] Failed to sync discard triage:', triageError.message || triageError);
    }
  }
  return updatedJob;
}

async function publishResult(userId, deviceId, jobId, productName, confidence, action) {
  const topic = TOPIC_TEMPLATE
    .replace('{user_id}', userId)
    .replace('{device_id}', deviceId)
    .replace('{job_id}', jobId);
  const payload = {
    job_id: jobId,
    mode: 'discard',
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

// Emits a structured marker for silent (non-throwing) analysis failures so a
// CloudWatch metric filter on "analysis_failed" can alert. Hard failures rethrow
// and are caught by the Lambda Errors alarm; this covers the swallowed/degraded
// paths that return 200.
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

  if (KEY_PREFIX && !key.startsWith(KEY_PREFIX)) {
    console.log('[handler] Key not under prefix; skipping.', { prefix: KEY_PREFIX });
    return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true }) };
  }

  if (isResizedDerivativeKey(key)) {
    console.log('[handler] Ignoring resized discard derivative event.', { key });
    return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'resized_derivative' }) };
  }

  let jobId = null;
  let provisionalDiscardInserted = false;
  let provisionalDiscardResolved = false;
  let provisionalDiscardOwner = null;
  try {
    const ids = extractIdsFromKey(key);
    if (!ids) {
      console.log('[handler] Key does not match discard upload pattern; skipping.', { key });
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'unexpected_key_pattern' }) };
    }
    const { user_id, device_id } = ids;
    jobId = ids.job_id;

    await updateJob(jobId, {
      status: 'PROCESSING',
      discard_status: 'PROCESSING',
      fast_status: 'PROCESSING',
      deep_status: 'PROCESSING',
      t_start_proc: new Date().toISOString(),
    });

    const jobResult = await dynamodb.get({
      TableName: JOBS_TABLE,
      Key: { job_id: jobId },
    }).promise();

    const job = jobResult.Item || { ...ids, job_id: jobId };
    const owner = job.owner;
    provisionalDiscardOwner = owner || null;
    const action = job.action || 'IN';
    const productExpiration = job.product_expiration || null;
    const addToShoppingList = normalizeOptionalBoolean(job.add_to_shopping_list);

    if (!owner) {
      console.error('[error] owner parameter required in job');
      await updateJob(jobId, { status: 'FAILED', error_msg: 'owner parameter required' });
      reportAnalysisFailed('discard', 'owner_missing', jobId, 'owner parameter required');
      return { statusCode: 400, body: JSON.stringify({ ok: false, error: 'owner parameter required' }) };
    }

    // Only process discard type jobs (skip grocery and dish)
    if (!job.type || job.type !== 'discard') {
      console.log('[handler] Job type is not discard, skipping:', job.type);
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'not a discard job' }) };
    }

    // Download image from S3
    const s3Object = await s3.getObject({ Bucket: BUCKET_NAME, Key: key }).promise();
    const imageBuffer = Buffer.from(s3Object.Body);
    const imageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${key}`;
    let resizedImageUrl = null;
    let resizedImageKey = null;

    try {
      console.log('[fast-identify] Starting provisional discard legacy fast scan...');
      const { fastScan, fastMode, fallbackUsed } = await runDiscardFastScan(imageBuffer);
      const fastCandidate = selectBestFastItem(fastScan);
      if (fastCandidate) {
        await insertProvisionalDiscardRow({
          owner,
          device_id,
          user_id,
          job_id: jobId,
          action,
          product_expiration: productExpiration,
          s3_key: key,
          image_url: imageUrl,
          resized_image_url: resizedImageUrl,
          resized_image_key: resizedImageKey,
          product_image_url: null,
          product_image_key: null,
          groceryItem: toProvisionalGroceryItem(fastCandidate),
          storeAvailability: undefined,
          analysis_source: 'quick_identify',
          needs_review: !!fastCandidate.needs_review,
          provisional_payload: {
            fast_mode: fastMode,
            selected_item: fastCandidate,
            items: Array.isArray(fastScan?.items) ? fastScan.items : [],
            debug: fastScan?.debug || null,
            fallback_used: fallbackUsed,
          },
        });
        provisionalDiscardInserted = true;
        await updateJob(jobId, {
          fast_status: 'DONE',
          t_fast_done: new Date().toISOString(),
          fast_last_product: fastCandidate.item_name || null,
          fast_confidence: fastCandidate.confidence || null,
          fast_needs_review: !!fastCandidate.needs_review,
          fast_mode: fastMode,
        });
      } else {
        await updateJob(jobId, {
          fast_status: 'SKIPPED',
          t_fast_done: new Date().toISOString(),
          fast_error: 'No provisional fast candidate found',
        });
      }
    } catch (fastError) {
      console.warn('[fast-identify] Discard fast scan failed (non-fatal):', fastError);
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

    let groceryItem = await identifyGroceryItem(imageBuffer);
    let storeAvailability = null;
    if (shouldStandardizeUnknownItem(groceryItem)) {
      groceryItem = applyUnknownItemPlaceholder(groceryItem);
      console.log('[discard] Standardized unresolved discard item to unknown placeholder.');
    } else {
      try {
        const refinedItem = await refineProductTitle(groceryItem);
        groceryItem.product_name = refinedItem.product_name;
        console.log('[title-refine] Final discard title:', {
          product_name: groceryItem.product_name || null,
          brand: groceryItem.brand || null,
          variant: groceryItem.variant || null,
        });
      } catch (titleError) {
        console.warn('[title-refine] Failed (non-fatal):', titleError);
      }
      try {
        storeAvailability = await getStoreAvailability(
          groceryItem.product_name || '',
          groceryItem.brand || null,
          groceryItem.category || null
        );
      } catch (storeError) {
        console.error('[stores] Error (non-fatal):', storeError);
      }
    }
    let targetImageUrl = getStockImageTargetUrl(job);
    targetImageUrl = await uploadTargetImageIfNeeded(targetImageUrl, user_id, jobId);
    const requireTargetMatch = shouldRequireTargetMatch(targetImageUrl);
    if (targetImageUrl) {
      console.log('[stock-image] Using target image for verification:', {
        target_image_url: targetImageUrl,
        require_target_match: requireTargetMatch,
      });
    }
    const isUnknownItem = isUnknownItemPlaceholder(groceryItem);
    let matchedKitchenRow = null;
    if (!isUnknownItem) {
      try {
        matchedKitchenRow = await matchAndRemoveFromKitchen(owner, groceryItem, { remove: false });
        if (matchedKitchenRow?.matched) {
          console.log('[matchKitchen] Preview match found:', {
            _id: matchedKitchenRow._id || null,
            product_image_url: matchedKitchenRow.product_image_url || null,
          });
        } else {
          console.log(`[matchKitchen] Preview found no match: ${matchedKitchenRow?.reason || 'no_match'}`);
        }
      } catch (matchPreviewError) {
        console.error('[matchKitchen] Preview error (non-fatal):', matchPreviewError);
      }
    }

    let productImageUrl = null;
    let productImageKey = null;

    if (isUnknownItem) {
      console.log('[discard-image] Unknown item placeholder; leaving product image empty.');
    } else if (matchedKitchenRow?.product_image_url) {
      productImageUrl = matchedKitchenRow.product_image_url;
      productImageKey = matchedKitchenRow.product_image_key || null;
      console.log('[discard-image] Reusing matched kitchen stock image:', {
        product_image_url: productImageUrl,
        product_image_key: productImageKey,
      });
    } else {
      try {
        const stockImage = await findStockImage(groceryItem, {
          targetImageUrl,
          requireTargetMatch,
        });
        if (stockImage?.url) {
          const mirrored = await mirrorStockImageToS3(stockImage.url, user_id, device_id, jobId);
          productImageUrl = mirrored.url;
          productImageKey = mirrored.key;
          console.log('[discard-image] Found discard stock image:', {
            product_image_url: productImageUrl,
            product_image_key: productImageKey,
            source: stockImage.source || null,
          });
        }
      } catch (stockImageError) {
        console.error('[discard-image] Stock image lookup failed (non-fatal):', stockImageError);
      }
    }

    if (!productImageUrl && !isUnknownItem) {
      try {
        const iconBuffer = await generateProductIcon(groceryItem);
        if (iconBuffer) {
          const uploaded = await uploadGeneratedImageToS3(iconBuffer, user_id, device_id, jobId);
          productImageUrl = uploaded.url;
          productImageKey = uploaded.key;
          console.log('[discard-image] Generated icon fallback uploaded:', {
            product_image_url: productImageUrl,
            product_image_key: productImageKey,
          });
        }
      } catch (iconError) {
        console.error('[discard-image] Icon fallback failed (non-fatal):', iconError);
      }
    }

    await finalizeDiscardRow({
      owner,
      device_id,
      user_id,
      job_id: jobId,
      action,
      product_expiration: productExpiration,
      s3_key: key,
      image_url: imageUrl,
      resized_image_url: resizedImageUrl,
      resized_image_key: resizedImageKey,
      product_image_url: productImageUrl,
      product_image_key: productImageKey,
      groceryItem,
      storeAvailability: storeAvailability || undefined,
    });
    provisionalDiscardResolved = true;

    if (addToShoppingList && !isUnknownItem) {
      try {
        const shoppingItem = await addDiscardToShoppingList({
          owner,
          user_id,
          device_id,
          groceryItem,
          imageUrl: productImageUrl || imageUrl,
        });
        await updateJob(jobId, {
          shopping_list_status: 'DONE',
          shopping_list_item_name: shoppingItem.item_name,
          shopping_list_household_item_uuid: shoppingItem.household_item_uuid,
        });
        console.log('[shopping-list] Added recognized discard to shopping list:', shoppingItem.item_name);
      } catch (shoppingError) {
        console.error('[shopping-list] Error adding discard to shopping list (non-fatal):', shoppingError);
        reportAnalysisFailed('discard', 'shopping_list_add', jobId, shoppingError);
        try {
          await updateJob(jobId, {
            shopping_list_status: 'FAILED',
            shopping_list_error: shoppingError.message || String(shoppingError),
          });
        } catch (shoppingStatusError) {
          console.error('[shopping-list] Failed to update shopping list status:', shoppingStatusError);
        }
      }
    } else if (addToShoppingList && isUnknownItem) {
      try {
        await updateJob(jobId, {
          shopping_list_status: 'SKIPPED_UNKNOWN_ITEM',
        });
      } catch (shoppingStatusError) {
        console.error('[shopping-list] Failed to update skipped status:', shoppingStatusError);
      }
    }

    try {
      const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
      await writeFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        event_type: 'discard',
        action,
        title: groceryItem.product_name || 'unknown',
        brand: groceryItem.brand || null,
        image_url: productImageUrl || imageUrl,
        product_image_url: productImageUrl || null,
        source_table: `${escapedOwner}_discards`,
        metadata: {
          confidence: groceryItem.confidence || null,
          category: groceryItem.category || null,
          variant: groceryItem.variant || null,
          barcode: groceryItem.barcode || null,
          product_expiration: productExpiration || null,
        },
      });
      await writeMasterFeedEvent({
        owner,
        device_id,
        user_id,
        job_id: jobId,
        item_id: stableHouseholdRowId('discards', jobId),
        event_type: 'discard_add',
        entity_type: 'discard_item',
        action,
        title: groceryItem.product_name || 'unknown',
        brand: groceryItem.brand || null,
        source_table: `${escapedOwner}_discards`,
        source_path: 'analyze_discard_on_upload_nodejs/app.js',
        source_system: 'analyze_discard_on_upload',
        primary_image_url: productImageUrl || imageUrl,
        secondary_image_url: imageUrl,
        event_key: `upload:${jobId}:discard`,
        metadata: {
          confidence: groceryItem.confidence || null,
          category: groceryItem.category || null,
          variant: groceryItem.variant || null,
          barcode: groceryItem.barcode || null,
          product_expiration: productExpiration || null,
          add_to_shopping_list: addToShoppingList,
          unknown_item: isUnknownItem,
        },
      });
    } catch (feedError) {
      console.error('[feed] Error (non-fatal):', feedError);
    }

    if (!isUnknownItem) {
      let removedFromKitchen = false;
      try {
        const matchResult = matchedKitchenRow?._id
          ? await matchAndRemoveFromKitchen(owner, groceryItem, { preferredId: matchedKitchenRow._id })
          : await matchAndRemoveFromKitchen(owner, groceryItem);
        removedFromKitchen = !!matchResult.removed;
        if (matchResult.removed) {
          console.log(`[matchKitchen] Removed from prod kitchen _id=${matchResult._id}`);
          try {
            const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
            await writeMasterFeedEvent({
              owner,
              device_id,
              user_id,
              job_id: jobId,
              item_id: matchResult._id || null,
              event_type: 'kitchen_remove',
              entity_type: 'kitchen_item',
              action: 'OUT',
              title: groceryItem.product_name || 'unknown',
              brand: groceryItem.brand || null,
              source_table: `${escapedOwner}_archive_kitchen`,
              source_path: 'analyze_discard_on_upload_nodejs/app.js',
              source_system: 'analyze_discard_on_upload',
              primary_image_url: matchResult.product_image_url || productImageUrl || imageUrl,
              secondary_image_url: imageUrl,
              event_key: `discard-match:${jobId}:${matchResult._id || 'unknown'}`,
              metadata: {
                archived_reason: 'discard_match',
                removed_from_table: `${escapedOwner}_prod_kitchen`,
                discard_job_id: jobId,
                matched_reason: matchResult.reason || null,
              },
            });
          } catch (masterFeedError) {
            console.error('[master-feed] Failed to record discard-linked kitchen removal (non-fatal):', masterFeedError);
          }
        } else {
          console.log(`[matchKitchen] No removal: ${matchResult.reason || 'no_match'}`);
        }
      } catch (matchError) {
        console.error('[matchKitchen] Error (non-fatal):', matchError);
      }
      if (removedFromKitchen) {
        await invokeKitchenAnalysisGenerator(owner, getScanPointsDelta());
        console.log('[kitchen-analysis] Regeneration requested after kitchen removal');
      } else {
        await appendMetricsSnapshot(owner);
        console.log('[metrics] Snapshot appended');
      }
    } else {
      console.log('[matchKitchen] Unknown item placeholder; skipping kitchen removal.');
      await appendMetricsSnapshot(owner);
      console.log('[metrics] Snapshot appended');
    }

    await updateJob(jobId, {
      status: 'DONE',
      discard_status: 'DONE',
      deep_status: 'DONE',
      t_done: new Date().toISOString(),
      last_product: groceryItem.product_name || null,
      confidence: groceryItem.confidence || 0,
    });

    await publishResult(
      user_id,
      device_id,
      jobId,
      groceryItem.product_name || 'unknown',
      groceryItem.confidence || 0,
      action
    );

    return { statusCode: 200, body: JSON.stringify({ ok: true, job_id: jobId }) };
  } catch (error) {
    console.error('[error] Exception:', error);
    if (jobId) {
      try {
        if (provisionalDiscardInserted && !provisionalDiscardResolved && provisionalDiscardOwner) {
          await markDiscardRowFailed({
            owner: provisionalDiscardOwner,
            job_id: jobId,
            error_message: error.message || String(error),
          });
        }
        await updateJob(jobId, {
          status: 'FAILED',
          discard_status: 'FAILED',
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
