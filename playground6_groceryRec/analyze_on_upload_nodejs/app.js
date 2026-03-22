// Lambda handler for analyzing uploaded grocery images using grocery-identifier
const AWS = require('aws-sdk');
const crypto = require('crypto');
const mysql = require('mysql2/promise');
const { identifyGroceryItem } = require('./dist/openai/identifyGrocery');
const { extractExpirationDate } = require('./dist/openai/extractExpiration');
const { getStoreAvailability } = require('./dist/openai/getStoreAvailability');
const { writeToKitchenTable, updateKitchenExpiration } = require('./dist/utils/mysqlWriter');
const { writeFeedEvent } = require('./dist/utils/mysqlFeedWriter');
const { findStockImage } = require('./dist/utils/stockImageSearch');
const { generateProductIcon } = require('./dist/utils/generateProductImage');
const { getHouseholdMemberIds, stableHouseholdRowId } = require('./dist/utils/householdSync');

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
          \`harmful_ingredients_what\`, \`harmful_ingredients_suggestions\`)
         VALUES (?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
    throw new Error(`Key does not match expected pattern: ${key}`);
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

async function updateJob(jobId, attrs) {
  const updateExpr = Object.keys(attrs)
    .map((k, i) => `#${k} = :${k}`)
    .join(', ');
  const names = {};
  const values = {};
  Object.keys(attrs).forEach(k => {
    names[`#${k}`] = k;
    values[`:${k}`] = attrs[k];
  });

  await dynamodb.update({
    TableName: JOBS_TABLE,
    Key: { job_id: jobId },
    UpdateExpression: `SET ${updateExpr}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
  }).promise();
}

async function getJob(jobId) {
  const jobResult = await dynamodb.get({
    TableName: JOBS_TABLE,
    Key: { job_id: jobId },
  }).promise();
  return jobResult.Item || null;
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
    const { user_id, device_id } = ids;
    jobId = ids.job_id;
    const isExpirationImage = !!ids.is_expiration;
    console.log('[handler] Parsed IDs:', ids);

    // Mark job as PROCESSING
    const startUpdate = isExpirationImage
      ? { status: 'PROCESSING', expiration_status: 'PROCESSING', t_expiration_start: new Date().toISOString() }
      : { status: 'PROCESSING', grocery_status: 'PROCESSING', t_start_proc: new Date().toISOString() };
    await updateJob(jobId, startUpdate);

    // Load job to get owner, action, and product_expiration
    const job = (await getJob(jobId)) || { ...ids, job_id: jobId };
    const owner = job.owner;
    const action = job.action || 'IN';
    let productExpiration = job.product_expiration || null;
    const expirationExpected = !!(job.expiration_expected || job.expiration_s3_key);
    const quantity = normalizeQuantity(job.quantity);

    if (!owner) {
      console.error('[error] owner parameter required in job');
      await updateJob(jobId, {
        status: 'FAILED',
        error_msg: 'owner parameter required',
      });
      return {
        statusCode: 400,
        body: JSON.stringify({ ok: false, error: 'owner parameter required' }),
      };
    }

    // Only process grocery type jobs (skip dish and discard jobs)
    if (job.type && job.type === 'dish') {
      console.log('[handler] Job type is dish, skipping (handled by dish analyzer):', job.type);
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'dish job handled by dish analyzer' }) };
    }
    if (job.type && job.type === 'discard') {
      console.log('[handler] Job type is discard, skipping (handled by discard analyzer):', job.type);
      return { statusCode: 200, body: JSON.stringify({ ok: true, ignored: true, reason: 'discard job handled by discard analyzer' }) };
    }

    console.log('[dynamo] Loaded job:', {
      has_action: !!job.action,
      action: action,
      owner: owner,
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
      }

      return {
        statusCode: 200,
        body: JSON.stringify({ ok: true, job_id: jobId, product_expiration: expirationDate }),
      };
    }

    // Identify grocery item using grocery-identifier
    console.log('[identify] Starting grocery identification...');
    const identifyStart = Date.now();
    const groceryItem = await identifyGroceryItem(imageBuffer);
    const identifyTime = Date.now() - identifyStart;
    console.log('[identify] Identification complete in', identifyTime, 'ms');
    console.log('[identify] Product:', groceryItem.product_name, 'Brand:', groceryItem.brand);

    // Get store availability
    console.log('[stores] Fetching store availability...');
    let storeAvailability = null;
    try {
      const storesStart = Date.now();
      storeAvailability = await getStoreAvailability(
        groceryItem.product_name || '',
        groceryItem.brand || null,
        groceryItem.category || null
      );
      const storesTime = Date.now() - storesStart;
      console.log('[stores] Found', storeAvailability.store_availability.length, 'stores in', storesTime, 'ms');
    } catch (storeError) {
      console.error('[stores] Error fetching store availability (non-fatal):', storeError);
      // Continue without store data
    }

    // Construct S3 URL for uploaded image
    const imageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${key}`;

    // Find a stock product image first, then fallback to icon generation
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
    console.log('[stock-image] Searching for stock product image...', { timeout_ms: stockImageTimeoutMs });
    let productImageUrl = null;
    let productImageKey = null;
    try {
      const stockImage = await withTimeout(findStockImage(groceryItem, {
        targetImageUrl: targetImageUrl,
        requireTargetMatch: requireTargetMatch,
      }), stockImageTimeoutMs);
      if (stockImage?.url) {
        productImageUrl = stockImage.url;
        console.log('[stock-image] Stock image found from', stockImage.source, ':', productImageUrl);
      } else {
        console.log('[stock-image] No stock image found.');
      }
    } catch (stockImageError) {
      if (stockImageError?.code === 'STOCK_IMAGE_TIMEOUT') {
        console.warn('[stock-image] Search timed out (non-fatal):', stockImageError.message);
      } else {
        console.error('[stock-image] Error finding stock image (non-fatal):', stockImageError);
      }
    }

    if (!productImageUrl) {
      console.log('[icon] Generating icon fallback...');
      try {
        const iconBuffer = await generateProductIcon(groceryItem);
        if (iconBuffer) {
          productImageKey = `product-images/${user_id}/${device_id}/${jobId}.jpg`;
          await s3.putObject({
            Bucket: BUCKET_NAME,
            Key: productImageKey,
            Body: iconBuffer,
            ContentType: 'image/jpeg',
          }).promise();
          productImageUrl = `https://${BUCKET_NAME}.s3.amazonaws.com/${productImageKey}`;
          console.log('[icon] Icon image uploaded to S3:', productImageUrl);
        } else {
          console.log('[icon] Icon generation skipped or failed (non-fatal)');
        }
      } catch (iconError) {
        console.error('[icon] Error generating/uploading icon (non-fatal):', iconError);
      }
    }

    // Refresh expiration date if OCR finished before insert
    const latestJobForInsert = await getJob(jobId);
    if (latestJobForInsert && latestJobForInsert.product_expiration) {
      productExpiration = latestJobForInsert.product_expiration;
    }

    // Write to MySQL
    console.log('[mysql] Writing to kitchen table...');
    await writeToKitchenTable({
      owner: owner,
      device_id: device_id,
      user_id: user_id,
      job_id: jobId,
      action: action,
      quantity: quantity,
      product_expiration: productExpiration,
      s3_key: key,
      image_url: imageUrl,
      product_image_url: productImageUrl,
      product_image_key: productImageKey,
      groceryItem: groceryItem,
      storeAvailability: storeAvailability || undefined,
    });
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
        image_url: productImageUrl || imageUrl,
        product_image_url: productImageUrl || null,
        source_table: `${escapedOwner}_prod_kitchen`,
        metadata: {
          confidence: groceryItem.confidence || null,
          category: groceryItem.category || null,
          variant: groceryItem.variant || null,
          barcode: groceryItem.barcode || null,
          product_expiration: productExpiration || null,
          quantity: quantity || 1,
        },
      });
    } catch (feedError) {
      console.error('[feed] Error (non-fatal):', feedError);
    }

    await appendMetricsSnapshot(owner);
    console.log('[metrics] Snapshot appended');

    // Trigger meal plan and recipes regeneration (async)
    const mealPlanArn = process.env.MEAL_PLAN_GENERATOR_ARN;
    if (mealPlanArn) {
      lambda.invoke({
        FunctionName: mealPlanArn,
        InvocationType: 'Event',
        Payload: JSON.stringify({ owner }),
      }).promise().catch((err) => console.warn('[meal-plan] invoke failed', err.message));
    }
    const recipesArn = process.env.RECIPES_GENERATOR_ARN;
    if (recipesArn) {
      lambda.invoke({
        FunctionName: recipesArn,
        InvocationType: 'Event',
        Payload: JSON.stringify({ owner }),
      }).promise().catch((err) => console.warn('[recipes] invoke failed', err.message));
    }

    const latestJob = await getJob(jobId);
    const expirationComplete = !expirationExpected || ['DONE', 'FAILED'].includes(latestJob?.expiration_status);
    const groceryUpdate = {
      status: expirationComplete ? 'DONE' : 'PROCESSING',
      grocery_status: 'DONE',
      last_product: groceryItem.product_name || null,
      confidence: groceryItem.confidence || 0,
    };
    if (expirationComplete) {
      groceryUpdate.t_done = new Date().toISOString();
    }
    await updateJob(jobId, groceryUpdate);
    console.log('[dynamo] Job marked as DONE:', expirationComplete);

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
