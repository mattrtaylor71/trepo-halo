// index.mjs (ES module)
import axios from 'axios';
import mysql from 'mysql2/promise';
import crypto from 'crypto';
import * as jose from 'jose';

const DEBUG_TWILIO = false; // set true only when debugging Twilio

const {
  TWILIO_ACCOUNT_SID,
  TWILIO_AUTH_TOKEN,
  TWILIO_VERIFY_SERVICE_SID,
  DB_HOST,
  DB_USER,
  DB_PASSWORD,
  DB_NAME,
  VERIFY_ALLOWED_PHONE_PREFIXES = '+1',
  VERIFY_CLIENT_SECRET,
  VERIFY_REQUIRE_CLIENT_SECRET = 'false',
  VERIFY_SEND_RATE_LIMIT_WINDOW_SECONDS = '900',
  VERIFY_SEND_LIMIT_PER_IP = '10',
  VERIFY_SEND_LIMIT_PER_PHONE = '3',
  VERIFY_CHECK_RATE_LIMIT_WINDOW_SECONDS = '600',
  VERIFY_CHECK_LIMIT_PER_IP = '25',
  VERIFY_CHECK_LIMIT_PER_PHONE = '8',
  REVIEW_TEST_PHONE = '',
  REVIEW_TEST_CODE = '',
  APPLE_BUNDLE_ID = 'com.mattaylor.trepo',
} = process.env;

// ── Blocked owner/user IDs ──────────────────────────────────────────
// Users with these IDs are denied auth token generation and API access.
const BLOCKED_USER_IDS = new Set([
  'deadbeef-cafe-babe-feed-1234567890ab',
]);

// ── Invite code system ─────────────────────────────────────────────
// Codes stored as JSON env var: {"CODE":{"max_uses":50},...}
// max_uses=0 means unlimited. Remove a code to revoke it.
const INVITE_CODES_RAW = process.env.INVITE_CODES || '{}';
let INVITE_CODES = {};
try { INVITE_CODES = JSON.parse(INVITE_CODES_RAW); } catch (e) { console.error('[InviteCode] Bad INVITE_CODES JSON:', e.message); }
const REQUIRE_INVITE_CODE = (process.env.REQUIRE_INVITE_CODE || 'true').toLowerCase() === 'true';

let pool;
let verificationTablesReadyPromise;

/** Get a MySQL connection pool. Recreates each cold start. */
function getPool() {
  if (!pool) {
    pool = mysql.createPool({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      database: DB_NAME,
      waitForConnections: true,
      connectionLimit: 2,
      queueLimit: 0,
      connectTimeout: 5000,
      maxIdle: 0,
    });
  }
  return pool;
}

/** Get a fresh direct connection (no pool). Use for transactional work. */
async function getDirectConnection() {
  return mysql.createConnection({
    host: DB_HOST,
    user: DB_USER,
    password: DB_PASSWORD,
    database: DB_NAME,
    connectTimeout: 5000,
  });
}

async function ensureVerificationTables() {
  if (!verificationTablesReadyPromise) {
    verificationTablesReadyPromise = (async () => {
      const db = getPool();
      await db.query(`
        CREATE TABLE IF NOT EXISTS phone_verification_attempts (
          id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
          action VARCHAR(32) NOT NULL,
          phone_number VARCHAR(32) NOT NULL,
          ip_address VARCHAR(64) NOT NULL,
          outcome VARCHAR(32) NOT NULL,
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          KEY idx_action_created (action, created_at),
          KEY idx_phone_action_created (phone_number, action, created_at),
          KEY idx_ip_action_created (ip_address, action, created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
      `);
    })().catch((error) => {
      verificationTablesReadyPromise = null;
      throw error;
    });
  }

  return verificationTablesReadyPromise;
}

async function recordVerificationAttempt({ action, phoneNumber, ipAddress, outcome }) {
  await ensureVerificationTables();
  const db = getPool();
  await db.query(
    `INSERT INTO phone_verification_attempts (action, phone_number, ip_address, outcome)
     VALUES (?, ?, ?, ?)`,
    [action, phoneNumber, ipAddress, outcome]
  );
}

async function recordVerificationAttemptSafely(details) {
  try {
    await recordVerificationAttempt(details);
  } catch (error) {
    console.error('Failed to record verification attempt:', error);
  }
}

async function getRecentAttemptCounts({ action, phoneNumber, ipAddress, windowSeconds }) {
  await ensureVerificationTables();
  const db = getPool();
  const cutoff = mysqlDateTime(new Date(Date.now() - windowSeconds * 1000));

  const [rows] = await db.query(
    `SELECT
       COALESCE(SUM(CASE WHEN ip_address = ? THEN 1 ELSE 0 END), 0) AS ip_count,
       COALESCE(SUM(CASE WHEN phone_number = ? THEN 1 ELSE 0 END), 0) AS phone_count,
       UNIX_TIMESTAMP(MIN(created_at)) AS oldest_attempt_ts
     FROM phone_verification_attempts
     WHERE action = ? AND created_at >= ?`,
    [ipAddress, phoneNumber, action, cutoff]
  );

  const row = rows[0] || {};
  return {
    ipCount: Number(row.ip_count || 0),
    phoneCount: Number(row.phone_count || 0),
    oldestAttemptTs: row.oldest_attempt_ts ? Number(row.oldest_attempt_ts) : null,
  };
}

function buildRateLimitResponse(message, retryAfterSeconds) {
  return {
    statusCode: 429,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Verify-Client-Secret',
      'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE',
      'Retry-After': String(Math.max(1, retryAfterSeconds)),
    },
    body: JSON.stringify({ message }),
  };
}

async function enforceRateLimit({ action, phoneNumber, ipAddress }) {
  const config = action === 'send_code' ? RATE_LIMITS.send : RATE_LIMITS.check;
  const counts = await getRecentAttemptCounts({
    action,
    phoneNumber,
    ipAddress,
    windowSeconds: config.windowSeconds,
  });

  if (counts.ipCount >= config.perIp || counts.phoneCount >= config.perPhone) {
    const retryAfterSeconds = counts.oldestAttemptTs
      ? Math.max(
          1,
          config.windowSeconds -
            Math.floor(Date.now() / 1000 - counts.oldestAttemptTs)
        )
      : config.windowSeconds;

    return buildRateLimitResponse('Too many verification attempts', retryAfterSeconds);
  }

  return null;
}

function buildResponse(statusCode, bodyObj) {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Verify-Client-Secret',
      'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE',
    },
    body: JSON.stringify(bodyObj),
  };
}

function parseBooleanEnv(value) {
  return String(value).trim().toLowerCase() === 'true';
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseAllowedPrefixes(value) {
  return String(value)
    .split(',')
    .map((prefix) => prefix.trim())
    .filter(Boolean);
}

const REQUIRE_CLIENT_SECRET = parseBooleanEnv(VERIFY_REQUIRE_CLIENT_SECRET);
const ALLOWED_PHONE_PREFIXES = parseAllowedPrefixes(VERIFY_ALLOWED_PHONE_PREFIXES);
const RATE_LIMITS = {
  send: {
    windowSeconds: parsePositiveInt(VERIFY_SEND_RATE_LIMIT_WINDOW_SECONDS, 900),
    perIp: parsePositiveInt(VERIFY_SEND_LIMIT_PER_IP, 10),
    perPhone: parsePositiveInt(VERIFY_SEND_LIMIT_PER_PHONE, 3),
  },
  check: {
    windowSeconds: parsePositiveInt(VERIFY_CHECK_RATE_LIMIT_WINDOW_SECONDS, 600),
    perIp: parsePositiveInt(VERIFY_CHECK_LIMIT_PER_IP, 25),
    perPhone: parsePositiveInt(VERIFY_CHECK_LIMIT_PER_PHONE, 8),
  },
};

function getHeader(headers = {}, headerName) {
  const direct = headers[headerName];
  if (direct !== undefined) {
    return direct;
  }

  const lowerKey = Object.keys(headers).find(
    (key) => key.toLowerCase() === headerName.toLowerCase()
  );
  return lowerKey ? headers[lowerKey] : undefined;
}

function getPath(event) {
  const rawPath = event.rawPath || event.path || '';
  return rawPath.replace(/\/+$/, '') || '/';
}

function getClientIp(event) {
  const forwardedFor = getHeader(event.headers, 'x-forwarded-for');
  if (forwardedFor) {
    return forwardedFor.split(',')[0].trim().slice(0, 64);
  }

  return (
    event.requestContext?.http?.sourceIp ||
    event.requestContext?.identity?.sourceIp ||
    'unknown'
  ).slice(0, 64);
}

function normalizePhoneNumber(value) {
  if (typeof value !== 'string') {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const digitsOnly = trimmed.replace(/\D/g, '');
  if (digitsOnly.length === 10) {
    return `+1${digitsOnly}`;
  }
  if (digitsOnly.length === 11 && digitsOnly.startsWith('1')) {
    return `+${digitsOnly}`;
  }

  let normalized = trimmed.replace(/[^\d+]/g, '');
  if (normalized.startsWith('00')) {
    normalized = `+${normalized.slice(2)}`;
  }
  if (!normalized.startsWith('+')) {
    normalized = `+${normalized.replace(/\D/g, '')}`;
  } else {
    normalized = `+${normalized.slice(1).replace(/\D/g, '')}`;
  }

  return /^\+[1-9]\d{7,14}$/.test(normalized) ? normalized : null;
}

function normalizeVerificationCode(value) {
  if (typeof value !== 'string' && typeof value !== 'number') {
    return null;
  }
  const normalized = String(value).trim();
  return /^\d{4,10}$/.test(normalized) ? normalized : null;
}

function isAllowedPhoneNumber(phoneNumber) {
  if (ALLOWED_PHONE_PREFIXES.length === 0) {
    return true;
  }
  return ALLOWED_PHONE_PREFIXES.some((prefix) => phoneNumber.startsWith(prefix));
}

function extractClientSecret(headers = {}) {
  const authorization = getHeader(headers, 'authorization');
  if (authorization?.startsWith('Bearer ')) {
    return authorization.slice('Bearer '.length).trim();
  }

  const secretHeader = getHeader(headers, 'x-verify-client-secret');
  return typeof secretHeader === 'string' ? secretHeader.trim() : '';
}

function validateClientSecret(headers = {}) {
  if (!REQUIRE_CLIENT_SECRET) {
    return null;
  }

  const providedSecret = extractClientSecret(headers);
  if (!VERIFY_CLIENT_SECRET || providedSecret !== VERIFY_CLIENT_SECRET) {
    return buildResponse(401, { message: 'Unauthorized' });
  }

  return null;
}

function mysqlDateTime(date) {
  return date.toISOString().slice(0, 19).replace('T', ' ');
}

function isTruthyFlag(value) {
  return value === true || value === 'true' || value === 1 || value === '1';
}

function normalizeOptionalEmail(value) {
  if (typeof value !== 'string') {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  return trimmed;
}

function isValidBasicEmail(email) {
  if (!email || email.length > 254) {
    return false;
  }

  const atIndex = email.indexOf('@');
  if (atIndex <= 0 || atIndex !== email.lastIndexOf('@')) {
    return false;
  }

  const domain = email.slice(atIndex + 1);
  const dotIndex = domain.indexOf('.');
  return dotIndex > 0 && dotIndex < domain.length - 1;
}

/**
 * SEC-001 Phase A: issue signed HS256 JWTs when TOKEN_SIGNING_SECRET is set
 * and ISSUE_SIGNED_TOKENS !== 'false'. Otherwise fall back to the legacy
 * base64 JSON blob so issuance never breaks (kill-switch / missing secret).
 */
async function generateToken({ ownerId, userId, phoneNumber }) {
  const payload = {
    owner_id: ownerId,
    user_id: userId,
    phone_number: phoneNumber,
  };

  if (process.env.TOKEN_SIGNING_SECRET && process.env.ISSUE_SIGNED_TOKENS !== 'false') {
    const secret = new TextEncoder().encode(process.env.TOKEN_SIGNING_SECRET);
    return await new jose.SignJWT({ owner_id: ownerId, user_id: userId, phone_number: phoneNumber })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setIssuer('trepo-auth')
      .setExpirationTime('180d')
      .sign(secret);
  }

  return Buffer.from(JSON.stringify(payload)).toString('base64');
}

/**
 * SEC-001 Phase A: accept-both decode. Returns the token payload object, or
 * null if the token can't be decoded/verified.
 *   - 3 dot-separated segments -> verify as HS256 JWT (issuer trepo-auth)
 *   - otherwise -> legacy base64(JSON) decode
 */
async function decodeToken(rawToken) {
  const token = (rawToken || '').trim();
  if (token.split('.').length === 3) {
    try {
      const { payload } = await jose.jwtVerify(
        token,
        new TextEncoder().encode(process.env.TOKEN_SIGNING_SECRET),
        { issuer: 'trepo-auth' }
      );
      return payload;
    } catch (e) {
      return null;
    }
  }
  try {
    return JSON.parse(Buffer.from(token, 'base64').toString('utf-8'));
  } catch (e) {
    return null;
  }
}

// ── Invite code helpers ────────────────────────────────────────────

let _inviteTableEnsured = false;

async function ensureInviteCodeTable(conn) {
  if (_inviteTableEnsured) return;
  await conn.query(`
    CREATE TABLE IF NOT EXISTS invite_code_usage (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      code VARCHAR(64) NOT NULL,
      user_id CHAR(36) NOT NULL,
      redeemed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      KEY idx_code (code),
      KEY idx_user_id (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);
  _inviteTableEnsured = true;
}

async function validateInviteCode(code) {
  if (!REQUIRE_INVITE_CODE) return { valid: true, code: '' };
  if (!code || typeof code !== 'string' || !code.trim()) {
    return { valid: false, reason: 'Invite code is required' };
  }
  const upper = code.trim().toUpperCase();
  const config = INVITE_CODES[upper];
  if (!config) return { valid: false, reason: 'Invalid invite code' };

  if (config.max_uses > 0) {
    const conn = await mysql.createConnection({
      host: DB_HOST, user: DB_USER, password: DB_PASSWORD, database: DB_NAME,
      connectTimeout: 5000,
    });
    try {
      await ensureInviteCodeTable(conn);
      const [rows] = await conn.query(
        'SELECT COUNT(*) AS cnt FROM invite_code_usage WHERE code = ?', [upper]
      );
      if (Number(rows[0].cnt) >= config.max_uses) {
        return { valid: false, reason: 'This invite code has been fully redeemed' };
      }
    } finally {
      await conn.end();
    }
  }
  return { valid: true, code: upper };
}

async function recordInviteCodeUsage(code, userId) {
  if (!code) return;
  try {
    const conn = await mysql.createConnection({
      host: DB_HOST, user: DB_USER, password: DB_PASSWORD, database: DB_NAME,
      connectTimeout: 5000,
    });
    try {
      await ensureInviteCodeTable(conn);
      await conn.query(
        'INSERT INTO invite_code_usage (code, user_id) VALUES (?, ?)',
        [code.trim().toUpperCase(), userId]
      );
      console.log(`[InviteCode] Recorded usage: code=${code}, user=${userId}`);
    } finally {
      await conn.end();
    }
  } catch (err) {
    console.error('[InviteCode] Failed to record usage:', err.message);
  }
}

/**
 * Ensure all per-user tables exist for this ownerId.
 * Uses the same schemas as existing production tables.
 */
async function ensureUserTables(conn, ownerId) {
  const listTable = `\`${ownerId}_new_list\``;
  const kitchenTable = `\`${ownerId}_new_kitchen\``;
  const feedTable = `\`${ownerId}_new_feed\``;
  const prodKitchenTable = `\`${ownerId}_prod_kitchen\``;
  const discardsTable = `\`${ownerId}_discards\``;
  const dishesTable = `\`${ownerId}_dishes\``;
  const recipesTable = `\`${ownerId}_recipes\``;
  const savedRecipesTable = `\`${ownerId}_saved_recipes\``;
  const mealPlanTable = `\`${ownerId}_meal_plan\``;
  const metricsTable = `\`${ownerId}-metrics\``;

  const statements = [
    `
      CREATE TABLE IF NOT EXISTS ${listTable} (
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
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${kitchenTable} (
        \`_id\` CHAR(36) NOT NULL,
        \`_owner\` CHAR(36) NOT NULL,
        \`_device\` VARCHAR(64) NOT NULL,
        \`product_name\` VARCHAR(255) NOT NULL,
        \`product_brand\` VARCHAR(255) DEFAULT NULL,
        \`product_expiration\` DATE DEFAULT NULL,
        \`images\` TEXT,
        \`product_barcode\` VARCHAR(64) DEFAULT NULL,
        \`action\` VARCHAR(32) NOT NULL,
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner_time\` (\`_owner\`, \`_createdDate\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${feedTable} (
        \`_id\` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        \`_owner\` CHAR(36) NOT NULL,
        \`_device\` VARCHAR(64) NOT NULL,
        \`product_name\` VARCHAR(255) NOT NULL,
        \`product_brand\` VARCHAR(255) DEFAULT NULL,
        \`images\` TEXT,
        \`product_barcode\` VARCHAR(64) DEFAULT NULL,
        \`action\` VARCHAR(32) NOT NULL,
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \`created_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \`updated_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner_device\` (\`_owner\`, \`_device\`),
        KEY \`idx_created\` (\`_createdDate\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${prodKitchenTable} (
        \`_id\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`product_name\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product name identified by AI',
        \`brand\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Brand name if visible',
        \`variant\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product variant (e.g., "Large", "Organic")',
        \`category\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product category',
        \`confidence\` DECIMAL(3,2) DEFAULT NULL COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'AI explanation of identification',
        \`product_description\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'One sentence: what this product actually is (for recipe/meal-plan AI context)',
        \`barcode\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Barcode if detected',
        \`country_guess\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Country of origin guess',
        \`estimated_price\` VARCHAR(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Estimated price (e.g., "$3.99", "$5.50-$7.00")',
        \`ingredients\` JSON DEFAULT NULL COMMENT 'Array of ingredient strings',
        \`nutrition_summary\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'Nutrition information summary',
        \`upf\` ENUM('yes','no') COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Ultra-processed food flag',
        \`harmful_ingredients\` JSON DEFAULT NULL COMMENT 'Array of harmful ingredient strings',
        \`similar_items\` JSON DEFAULT NULL COMMENT 'Array of similar items with name, brand, reason',
        \`alternatives\` JSON DEFAULT NULL COMMENT 'Array of alternative products with name, brand, reason',
        \`healthier_alternatives\` JSON DEFAULT NULL COMMENT 'Array of healthier alternatives with name, brand, why_healthier, trade_offs',
        \`store_availability\` JSON DEFAULT NULL COMMENT 'Array of stores with name, price, availability, store_url',
        \`images\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to the uploaded image',
        \`s3_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the image',
        \`action\` ENUM('IN','OUT') COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'IN' COMMENT 'IN = adding to kitchen, OUT = removing from kitchen',
        \`product_expiration\` DATE DEFAULT NULL COMMENT 'Product expiration date (YYYY-MM-DD)',
        \`product_image_url\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to generated wireframe/product image',
        \`product_image_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the generated product image',
        \`job_id\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'User ID from the upload request',
        \`_updatedDate\` DATETIME DEFAULT NULL COMMENT 'Last update timestamp',
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner\` (\`_owner\`),
        KEY \`idx_device\` (\`_device\`),
        KEY \`idx_product_name\` (\`product_name\`),
        KEY \`idx_category\` (\`category\`),
        KEY \`idx_action\` (\`action\`),
        KEY \`idx_created\` (\`_createdDate\`),
        KEY \`idx_expiration\` (\`product_expiration\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Comprehensive grocery kitchen inventory with full product details';
    `,
    `
      CREATE TABLE IF NOT EXISTS ${discardsTable} (
        \`_id\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`product_name\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product name identified by AI',
        \`brand\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Brand name if visible',
        \`variant\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product variant',
        \`category\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Product category',
        \`confidence\` DECIMAL(3,2) DEFAULT NULL COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'AI explanation of identification',
        \`barcode\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Barcode if detected',
        \`country_guess\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Country of origin guess',
        \`estimated_price\` VARCHAR(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Estimated price',
        \`ingredients\` JSON DEFAULT NULL COMMENT 'Array of ingredient strings',
        \`nutrition_summary\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'Nutrition information summary',
        \`upf\` ENUM('yes','no') COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Ultra-processed food flag',
        \`harmful_ingredients\` JSON DEFAULT NULL COMMENT 'Array of harmful ingredient strings',
        \`similar_items\` JSON DEFAULT NULL COMMENT 'Array of similar items',
        \`alternatives\` JSON DEFAULT NULL COMMENT 'Array of alternative products',
        \`healthier_alternatives\` JSON DEFAULT NULL COMMENT 'Array of healthier alternatives',
        \`store_availability\` JSON DEFAULT NULL COMMENT 'Array of stores',
        \`images\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to the uploaded image',
        \`s3_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the image',
        \`action\` ENUM('IN','OUT') COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'IN' COMMENT 'IN = adding to discards, OUT = removing',
        \`product_expiration\` DATE DEFAULT NULL COMMENT 'Product expiration date (YYYY-MM-DD)',
        \`product_image_url\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to generated product image',
        \`product_image_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the generated product image',
        \`job_id\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'User ID from the upload request',
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner\` (\`_owner\`),
        KEY \`idx_device\` (\`_device\`),
        KEY \`idx_product_name\` (\`product_name\`),
        KEY \`idx_category\` (\`category\`),
        KEY \`idx_action\` (\`action\`),
        KEY \`idx_created\` (\`_createdDate\`),
        KEY \`idx_expiration\` (\`product_expiration\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Discarded groceries - same structure as prod_kitchen';
    `,
    `
      CREATE TABLE IF NOT EXISTS ${dishesTable} (
        \`_id\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`_updatedDate\` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT 'When the record was last updated',
        \`dish_name\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Dish name identified by AI',
        \`confidence\` DECIMAL(3,2) DEFAULT NULL COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'AI explanation of dish identification',
        \`serving_size\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Estimated serving size (e.g., "1 bowl", "1 plate", "200g")',
        \`calories\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Calories per serving',
        \`total_fat\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Total fat in grams',
        \`saturated_fat\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Saturated fat in grams',
        \`trans_fat\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Trans fat in grams',
        \`cholesterol\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Cholesterol in milligrams',
        \`sodium\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Sodium in milligrams',
        \`total_carbohydrates\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Total carbohydrates in grams',
        \`dietary_fiber\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Dietary fiber in grams',
        \`sugars\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Sugars in grams',
        \`protein\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Protein in grams',
        \`vitamin_a\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Vitamin A in IU or mcg',
        \`vitamin_c\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Vitamin C in milligrams',
        \`calcium\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Calcium in milligrams',
        \`iron\` DECIMAL(10,2) DEFAULT NULL COMMENT 'Iron in milligrams',
        \`ingredients\` JSON DEFAULT NULL COMMENT 'Array of ingredient strings',
        \`allergens\` JSON DEFAULT NULL COMMENT 'Array of allergen strings (e.g., ["dairy", "nuts", "gluten"])',
        \`images\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to the uploaded dish image',
        \`s3_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the uploaded image',
        \`action\` ENUM('IN','OUT') COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'IN' COMMENT 'IN = adding dish, OUT = removing/consumed',
        \`dish_image_url\` VARCHAR(1000) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 URL to generated dish image',
        \`dish_image_key\` VARCHAR(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3 object key for the generated dish image',
        \`job_id\` VARCHAR(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'User ID from the upload request',
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner\` (\`_owner\`),
        KEY \`idx_device\` (\`_device\`),
        KEY \`idx_dish_name\` (\`dish_name\`),
        KEY \`idx_action\` (\`action\`),
        KEY \`idx_created\` (\`_createdDate\`),
        KEY \`idx_calories\` (\`calories\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Comprehensive dish tracking with nutrition analysis';
    `,
    `
      CREATE TABLE IF NOT EXISTS ${recipesTable} (
        \`_id\` VARCHAR(36) NOT NULL,
        \`_owner\` VARCHAR(36) NOT NULL,
        \`status\` ENUM('ready','regenerating','failed','empty') NOT NULL DEFAULT 'empty',
        \`kitchen_only\` JSON DEFAULT NULL,
        \`need_grocery\` JSON DEFAULT NULL,
        \`error_message\` TEXT,
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \`_updatedDate\` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (\`_id\`),
        KEY \`idx_status\` (\`status\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${savedRecipesTable} (
        \`_id\` VARCHAR(36) NOT NULL,
        \`_owner\` VARCHAR(36) NOT NULL,
        \`source_type\` VARCHAR(32) NOT NULL DEFAULT 'tiktok',
        \`source_url\` VARCHAR(1000) NOT NULL,
        \`resolved_url\` VARCHAR(1000) NOT NULL,
        \`resolved_url_hash\` CHAR(64) NOT NULL,
        \`title\` VARCHAR(255) NOT NULL,
        \`image_url\` VARCHAR(1000) DEFAULT NULL,
        \`image_urls\` JSON DEFAULT NULL,
        \`ingredients\` JSON DEFAULT NULL,
        \`instructions\` JSON DEFAULT NULL,
        \`notes\` JSON DEFAULT NULL,
        \`raw_caption\` TEXT,
        \`raw_content\` TEXT,
        \`extraction_source\` VARCHAR(32) DEFAULT NULL,
        \`author_name\` VARCHAR(255) DEFAULT NULL,
        \`caption_field\` VARCHAR(64) DEFAULT NULL,
        \`status\` ENUM('ready','failed') NOT NULL DEFAULT 'ready',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \`_updatedDate\` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (\`_id\`),
        UNIQUE KEY \`uniq_resolved_url_hash\` (\`resolved_url_hash\`),
        KEY \`idx_created\` (\`_createdDate\`),
        KEY \`idx_title\` (\`title\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${mealPlanTable} (
        \`_id\` VARCHAR(36) NOT NULL,
        \`_owner\` VARCHAR(36) NOT NULL,
        \`status\` ENUM('ready','regenerating','failed','empty') NOT NULL DEFAULT 'empty',
        \`focus\` TEXT,
        \`explanation_title\` VARCHAR(255) DEFAULT NULL,
        \`explanation_paragraph\` TEXT,
        \`plan\` JSON DEFAULT NULL,
        \`error_message\` TEXT,
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \`_updatedDate\` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (\`_id\`),
        KEY \`idx_status\` (\`status\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    `,
    `
      CREATE TABLE IF NOT EXISTS ${metricsTable} (
        \`_id\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Owner UUID',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the metrics snapshot was created',
        \`IQ\` INT NOT NULL DEFAULT 75 COMMENT 'IQ score out of 100',
        \`Points\` BIGINT NOT NULL DEFAULT 0 COMMENT 'Points total',
        \`UPF\` DECIMAL(5,2) NOT NULL DEFAULT 0.00 COMMENT 'UPF percentage (last 2 weeks)',
        \`harmful_ingredients\` INT NOT NULL DEFAULT 0 COMMENT 'Harmful ingredient count (last 2 weeks)',
        \`IQ_what\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'What this IQ score means',
        \`IQ_suggestions\` JSON DEFAULT NULL COMMENT 'Suggestions to improve IQ',
        \`UPF_what\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'What this UPF score means',
        \`UPF_suggestions\` JSON DEFAULT NULL COMMENT 'Suggestions to improve UPF score',
        \`harmful_ingredients_what\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'What this harmful ingredient count means',
        \`harmful_ingredients_suggestions\` JSON DEFAULT NULL COMMENT 'Suggestions to reduce harmful ingredients',
        \`kitchen_analysis_status\` VARCHAR(32) DEFAULT NULL COMMENT 'Latest kitchen analysis job status',
        \`kitchen_analysis_content\` MEDIUMTEXT COLLATE utf8mb4_unicode_ci COMMENT 'Formatted AI summary of the kitchen',
        \`kitchen_analysis_generated_at\` DATETIME NULL COMMENT 'When the kitchen analysis was generated',
        \`kitchen_analysis_error\` TEXT COLLATE utf8mb4_unicode_ci COMMENT 'Latest kitchen analysis error, if any',
        PRIMARY KEY (\`_id\`),
        KEY \`idx_owner\` (\`_owner\`),
        KEY \`idx_created\` (\`_createdDate\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User metrics snapshots';
    `,
  ];

  for (const sql of statements) {
    await conn.query(sql);
  }

  // Ensure _createdDate exists on list/feed for older tables.
  const alterClauses = [
    { table: listTable, column: '_createdDate', ddl: 'ADD COLUMN `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP' },
    { table: feedTable, column: '_createdDate', ddl: 'ADD COLUMN `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP' },
  ];
  for (const { table, ddl } of alterClauses) {
    try {
      await conn.query(`ALTER TABLE ${table} ${ddl}`);
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') {
        throw err;
      }
    }
  }
}

/**
 * Generate a unique 5-digit household ID string.
 * Stored in new_users.owner_id so household members can share it.
 */
async function generateUniqueHouseholdId(conn) {
  const maxAttempts = 10;
  for (let i = 0; i < maxAttempts; i++) {
    const code = String(Math.floor(10000 + Math.random() * 90000));
    const [rows] = await conn.query(
      'SELECT owner_id FROM new_users WHERE owner_id = ? LIMIT 1',
      [code]
    );
    if (rows.length === 0) {
      return code;
    }
  }
  const error = new Error('FAILED_TO_GENERATE_HOUSEHOLD_ID');
  error.code = 'FAILED_TO_GENERATE_HOUSEHOLD_ID';
  throw error;
}

/**
 * Ensure a user exists, optionally joining an existing household.
 * Returns { userId, ownerId, firstName, isNewUser, createdNewHousehold }.
 *
 * Rules:
 * - If user (by phone) exists:
 *     - If join_household_id provided and valid: update owner_id to that.
 *     - Else: keep existing owner_id.
 * - If user doesn't exist:
 *     - If join_household_id provided and valid: owner_id = join_household_id.
 *     - Else: owner_id = user_id (user becomes household owner).
 */
async function findOrCreateUser({ phoneNumber, firstName, lastName, zipCode, email, joinHouseholdId }) {
  console.log('[findOrCreateUser] connecting...');
  const conn = await mysql.createConnection({
    host: DB_HOST, user: DB_USER, password: DB_PASSWORD, database: DB_NAME,
    connectTimeout: 5000,
  });
  console.log('[findOrCreateUser] connected, ensuring columns...');
  try {
    await ensureUserColumns(conn);
    console.log('[findOrCreateUser] columns ensured, beginning txn...');
    await conn.beginTransaction();

    const [rows] = await conn.query(
      'SELECT user_id, owner_id, first_name, last_name, zip_code, email FROM new_users WHERE phone_number = ? LIMIT 1',
      [phoneNumber]
    );

    let userId;
    let ownerId;
    let finalFirstName = firstName || null;
    let finalLastName = lastName || null;
    let finalZipCode = zipCode || null;
    let finalEmail = email || null;
    let isNewUser = false;
    let createdNewHousehold = false;
    let templateUserId = null;

    if (rows.length > 0) {
      // Existing user
      const existing = rows[0];
      userId = existing.user_id;
      ownerId = existing.owner_id;
      if (!finalFirstName) {
        finalFirstName = existing.first_name;
      }
      if (!finalLastName) {
        finalLastName = existing.last_name;
      }
      if (!finalZipCode) {
        finalZipCode = existing.zip_code;
      }
      if (!finalEmail) {
        finalEmail = existing.email;
      }

      const updateFields = [];
      const updateValues = [];
      if (firstName) {
        updateFields.push('first_name = ?');
        updateValues.push(firstName);
      }
      if (lastName) {
        updateFields.push('last_name = ?');
        updateValues.push(lastName);
      }
      if (zipCode) {
        updateFields.push('zip_code = ?');
        updateValues.push(zipCode);
      }
      if (email) {
        updateFields.push('email = ?');
        updateValues.push(email);
      }
      if (updateFields.length > 0) {
        updateValues.push(userId);
        await conn.query(
          `UPDATE new_users SET ${updateFields.join(', ')}, updated_at = NOW() WHERE user_id = ?`,
          updateValues
        );
      }

      if (joinHouseholdId && joinHouseholdId !== ownerId) {
        templateUserId = await findTemplateUserId(conn, joinHouseholdId, userId);
        if (!templateUserId) {
          const error = new Error('INVALID_HOUSEHOLD');
          error.code = 'INVALID_HOUSEHOLD';
          throw error;
        }

        ownerId = joinHouseholdId;
        await conn.query(
          'UPDATE new_users SET owner_id = ?, updated_at = NOW() WHERE user_id = ?',
          [ownerId, userId]
        );
      }
    } else {
      // New user
      isNewUser = true;
      userId = crypto.randomUUID();

      if (joinHouseholdId) {
        // Joining an existing household
        templateUserId = await findTemplateUserId(conn, joinHouseholdId);
        if (!templateUserId) {
          const error = new Error('INVALID_HOUSEHOLD');
          error.code = 'INVALID_HOUSEHOLD';
          throw error;
        }
        ownerId = joinHouseholdId;
      } else {
        // Creating a brand new household (generate a shareable household ID)
        ownerId = await generateUniqueHouseholdId(conn);
        createdNewHousehold = true;
      }

      await conn.query(
        `INSERT INTO new_users (user_id, owner_id, phone_number, first_name, last_name, zip_code, email, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, NOW(), NOW())`,
        [userId, ownerId, phoneNumber, finalFirstName, finalLastName, finalZipCode, finalEmail]
      );

    }

    // Ensure all per-user tables exist for this user
    await ensureUserTables(conn, userId);
    if (templateUserId) {
      await mirrorHouseholdTables(conn, templateUserId, userId);
    }

    await conn.commit();

    return {
      userId,
      ownerId,
      firstName: finalFirstName,
      email: finalEmail,
      isNewUser,
      createdNewHousehold,
    };
  } catch (err) {
    await conn.rollback();
    if (err.code === 'INVALID_HOUSEHOLD') {
      throw err;
    }
    console.error('DB error in findOrCreateUser:', err);
    throw err;
  } finally {
    try { await conn.end(); } catch (_) {}
  }
}

// Columns and indexes are already in place — no ALTER TABLE needed at runtime.
// Running ALTER TABLE on every request caused metadata lock contention that blocked
// all queries on new_users. These schema changes only need to run once during
// initial setup, not on every Lambda invocation.
async function ensureUserColumns(conn) {
  // No-op — all columns (last_name, zip_code, email, apple_user_id) and
  // indexes (idx_apple_user_id) already exist. phone_number is already nullable.
}

// ── Apple Sign In ──────────────────────────────────────────────────────────

const APPLE_JWKS = jose.createRemoteJWKSet(
  new URL('https://appleid.apple.com/auth/keys')
);

async function verifyAppleIdentityToken(identityToken) {
  // Accept both bundle IDs (App Store + TestFlight)
  const validAudiences = [APPLE_BUNDLE_ID, 'com.mattaylor.trepo', 'com.mattaylor.trepo-v0'];
  const uniqueAudiences = [...new Set(validAudiences)];

  let lastErr;
  for (const aud of uniqueAudiences) {
    try {
      const { payload } = await jose.jwtVerify(identityToken, APPLE_JWKS, {
        issuer: 'https://appleid.apple.com',
        audience: aud,
      });
      return {
        appleUserId: payload.sub,
        email: payload.email || null,
        emailVerified: payload.email_verified === 'true' || payload.email_verified === true,
      };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr;
}

async function findOrCreateUserByApple({ appleUserId, email, firstName, lastName, zipCode, joinHouseholdId }) {
  const conn = await mysql.createConnection({
    host: DB_HOST, user: DB_USER, password: DB_PASSWORD, database: DB_NAME,
    connectTimeout: 5000,
  });
  try {
    await ensureUserColumns(conn);
    await conn.beginTransaction();

    const [rows] = await conn.query(
      'SELECT user_id, owner_id, first_name, last_name, zip_code, email FROM new_users WHERE apple_user_id = ? LIMIT 1',
      [appleUserId]
    );

    let userId;
    let ownerId;
    let finalFirstName = firstName || null;
    let finalLastName = lastName || null;
    let finalZipCode = zipCode || null;
    let finalEmail = email || null;
    let isNewUser = false;
    let createdNewHousehold = false;
    let templateUserId = null;

    if (rows.length > 0) {
      // Returning Apple user
      const existing = rows[0];
      userId = existing.user_id;
      ownerId = existing.owner_id;
      if (!finalFirstName) finalFirstName = existing.first_name;
      if (!finalLastName) finalLastName = existing.last_name;
      if (!finalZipCode) finalZipCode = existing.zip_code;
      if (!finalEmail) finalEmail = existing.email;

      const updateFields = [];
      const updateValues = [];
      if (firstName) { updateFields.push('first_name = ?'); updateValues.push(firstName); }
      if (lastName) { updateFields.push('last_name = ?'); updateValues.push(lastName); }
      if (zipCode) { updateFields.push('zip_code = ?'); updateValues.push(zipCode); }
      if (email) { updateFields.push('email = ?'); updateValues.push(email); }
      if (updateFields.length > 0) {
        updateValues.push(userId);
        await conn.query(
          `UPDATE new_users SET ${updateFields.join(', ')}, updated_at = NOW() WHERE user_id = ?`,
          updateValues
        );
      }

      if (joinHouseholdId && joinHouseholdId !== ownerId) {
        templateUserId = await findTemplateUserId(conn, joinHouseholdId, userId);
        if (!templateUserId) {
          const error = new Error('INVALID_HOUSEHOLD');
          error.code = 'INVALID_HOUSEHOLD';
          throw error;
        }
        ownerId = joinHouseholdId;
        await conn.query(
          'UPDATE new_users SET owner_id = ?, updated_at = NOW() WHERE user_id = ?',
          [ownerId, userId]
        );
      }
    } else {
      // New Apple user
      isNewUser = true;
      userId = crypto.randomUUID();

      if (joinHouseholdId) {
        templateUserId = await findTemplateUserId(conn, joinHouseholdId);
        if (!templateUserId) {
          const error = new Error('INVALID_HOUSEHOLD');
          error.code = 'INVALID_HOUSEHOLD';
          throw error;
        }
        ownerId = joinHouseholdId;
      } else {
        ownerId = await generateUniqueHouseholdId(conn);
        createdNewHousehold = true;
      }

      await conn.query(
        `INSERT INTO new_users (user_id, owner_id, phone_number, apple_user_id, first_name, last_name, zip_code, email, created_at, updated_at)
         VALUES (?, ?, NULL, ?, ?, ?, ?, ?, NOW(), NOW())`,
        [userId, ownerId, appleUserId, finalFirstName, finalLastName, finalZipCode, finalEmail]
      );
    }

    await ensureUserTables(conn, userId);
    if (templateUserId) {
      await mirrorHouseholdTables(conn, templateUserId, userId);
    }

    await conn.commit();

    return {
      userId,
      ownerId,
      firstName: finalFirstName,
      email: finalEmail,
      isNewUser,
      createdNewHousehold,
    };
  } catch (err) {
    await conn.rollback();
    if (err.code === 'INVALID_HOUSEHOLD') throw err;
    console.error('DB error in findOrCreateUserByApple:', err);
    throw err;
  } finally {
    try { await conn.end(); } catch (_) {}
  }
}

// ── End Apple Sign In ──────────────────────────────────────────────────────

const MIRRORED_TABLES = [
  { suffix: '_new_list', rewriteOwner: true, skipColumns: ['_id', 'created_at', 'updated_at'] },
  { suffix: '_new_kitchen', rewriteOwner: true },
  { suffix: '_new_feed', rewriteOwner: true, skipColumns: ['_id', 'created_at', 'updated_at'] },
  { suffix: '_prod_kitchen', rewriteOwner: true },
  { suffix: '_discards', rewriteOwner: true },
  { suffix: '_dishes', rewriteOwner: true },
  { suffix: '_recipes', rewriteOwner: true },
  { suffix: '_meal_plan', rewriteOwner: true },
  { suffix: '-metrics', rewriteOwner: true },
  { suffix: '_redemptions', rewriteOwner: true },
];

function quoteId(identifier) {
  return `\`${String(identifier).replace(/`/g, '``')}\``;
}

function getMirroredTablesForUser(userId) {
  return MIRRORED_TABLES.map(({ suffix, rewriteOwner, skipColumns = [] }) => ({
    tableName: `${userId}${suffix}`,
    rewriteOwner,
    skipColumns,
  }));
}

async function tableExists(conn, tableName) {
  const [rows] = await conn.query(
    `SELECT COUNT(*) AS count
     FROM information_schema.tables
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName]
  );
  return Number(rows?.[0]?.count || 0) > 0;
}

async function getTableColumns(conn, tableName) {
  const [rows] = await conn.query(
    `SELECT column_name
     FROM information_schema.columns
     WHERE table_schema = DATABASE() AND table_name = ?
     ORDER BY ordinal_position`,
    [tableName]
  );
  return rows
    .map((row) => {
      if (Array.isArray(row)) {
        return row[0] || null;
      }
      if (row && typeof row === 'object') {
        return row.column_name || row.COLUMN_NAME || null;
      }
      return null;
    })
    .filter(Boolean);
}

async function ensureDestinationTableLikeSource(conn, sourceTableName, destinationTableName) {
  if (!(await tableExists(conn, sourceTableName))) {
    return false;
  }
  if (await tableExists(conn, destinationTableName)) {
    return true;
  }
  await conn.query(`CREATE TABLE ${quoteId(destinationTableName)} LIKE ${quoteId(sourceTableName)}`);
  return true;
}

async function clearTableIfExists(conn, tableName) {
  if (!(await tableExists(conn, tableName))) {
    return;
  }
  await conn.query(`DELETE FROM ${quoteId(tableName)}`);
}

async function copyTableContents(conn, { sourceTableName, destinationTableName, targetOwnerId, rewriteOwner, skipColumns = [] }) {
  if (!(await tableExists(conn, sourceTableName)) || !(await tableExists(conn, destinationTableName))) {
    return;
  }

  const sourceColumns = await getTableColumns(conn, sourceTableName);
  const destinationColumns = await getTableColumns(conn, destinationTableName);
  const sourceColumnSet = new Set(sourceColumns);
  const skipColumnSet = new Set(skipColumns);
  const insertColumns = [];
  const selectExpressions = [];
  const values = [];

  for (const column of destinationColumns) {
    if (skipColumnSet.has(column)) {
      continue;
    }

    if (rewriteOwner && column === '_owner') {
      insertColumns.push(quoteId(column));
      selectExpressions.push(`? AS ${quoteId(column)}`);
      values.push(targetOwnerId);
      continue;
    }

    if (!sourceColumnSet.has(column)) {
      continue;
    }

    insertColumns.push(quoteId(column));
    selectExpressions.push(quoteId(column));
  }

  if (insertColumns.length === 0) {
    return;
  }

  await conn.query(
    `INSERT INTO ${quoteId(destinationTableName)} (${insertColumns.join(', ')})
     SELECT ${selectExpressions.join(', ')}
     FROM ${quoteId(sourceTableName)}`,
    values
  );
}

async function mirrorHouseholdTables(conn, templateUserId, targetUserId) {
  if (!templateUserId || !targetUserId || templateUserId === targetUserId) {
    return;
  }

  const sourceTables = getMirroredTablesForUser(templateUserId);
  const destinationTables = getMirroredTablesForUser(targetUserId);

  for (let i = 0; i < sourceTables.length; i += 1) {
    await ensureDestinationTableLikeSource(
      conn,
      sourceTables[i].tableName,
      destinationTables[i].tableName
    );
  }

  for (const { tableName } of destinationTables) {
    await clearTableIfExists(conn, tableName);
  }

  for (let i = 0; i < sourceTables.length; i += 1) {
    await copyTableContents(conn, {
      sourceTableName: sourceTables[i].tableName,
      destinationTableName: destinationTables[i].tableName,
      targetOwnerId: targetUserId,
      rewriteOwner: sourceTables[i].rewriteOwner,
      skipColumns: sourceTables[i].skipColumns,
    });
  }
}

async function findTemplateUserId(conn, householdId, excludeUserId = null) {
  const params = [householdId];
  let sql = 'SELECT user_id FROM new_users WHERE owner_id = ?';
  if (excludeUserId) {
    sql += ' AND user_id <> ?';
    params.push(excludeUserId);
  }
  sql += ' ORDER BY created_at ASC LIMIT 1';

  const [rows] = await conn.query(sql, params);
  return rows[0]?.user_id || null;
}

/**
 * Check if a user exists by phone number.
 */
async function userExistsByPhone(phoneNumber) {
  const pool = getPool();
  const [rows] = await pool.query(
    'SELECT user_id FROM new_users WHERE phone_number = ? LIMIT 1',
    [phoneNumber]
  );
  return rows.length > 0;
}

// Emits a structured marker for backend failures (500-returning paths) so a
// CloudWatch metric filter on "backend_error" can alert. Never throws.
function reportBackendError({ op, ownerId, code, err }) {
  try {
    console.error(JSON.stringify({
      evt: 'backend_error',
      service: 'auth',
      op,
      owner_id: ownerId || null,
      code: code || 'error',
      error: String((err && (err.message || err)) || op).slice(0, 500),
      job_id: null,
    }));
  } catch (_) { /* never let logging throw */ }
}

export const handler = async (event, context) => {
  // Don't wait for the mysql2 pool's idle sockets to drain before returning —
  // otherwise the response can be held open until API Gateway times out (503).
  if (context) context.callbackWaitsForEmptyEventLoop = false;

  const path = getPath(event);
  const method = event.requestContext?.http?.method || event.httpMethod || 'GET';

  // CORS preflight
  if (method === 'OPTIONS') {
    return buildResponse(200, { ok: true });
  }

  let body = {};
  try {
    if (event.body) {
      body = JSON.parse(event.body);
    }
  } catch (e) {
    console.error('Bad JSON body', e);
    return buildResponse(400, { message: 'Invalid JSON body' });
  }

  const {
    phone_number,
    code,
    first_name,
    last_name,
    zip_code,
    email,
    join_household_id,
    is_signup,
  } = body || {};
  const clientIp = getClientIp(event);
  const normalizedPhoneNumber = normalizePhoneNumber(phone_number);
  const normalizedCode = normalizeVerificationCode(code);
  const isSignupRequest = isTruthyFlag(is_signup);

  // 1) Send verification code
  if (path.endsWith('/send-code')) {
    if (!normalizedPhoneNumber) {
      return buildResponse(400, { message: 'phone_number is required' });
    }

    if (!isAllowedPhoneNumber(normalizedPhoneNumber)) {
      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'blocked_country',
      });
      return buildResponse(403, { message: 'Phone number country is not supported' });
    }

    const authError = validateClientSecret(event.headers);
    if (authError) {
      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'unauthorized',
      });
      return authError;
    }

    const rateLimitResponse = await enforceRateLimit({
      action: 'send_code',
      phoneNumber: normalizedPhoneNumber,
      ipAddress: clientIp,
    });
    if (rateLimitResponse) {
      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'rate_limited',
      });
      return rateLimitResponse;
    }

    // If this is a login attempt, block unknown phone numbers
    if (!isSignupRequest) {
      try {
        const exists = await userExistsByPhone(normalizedPhoneNumber);
        if (!exists) {
          await recordVerificationAttemptSafely({
            action: 'send_code',
            phoneNumber: normalizedPhoneNumber,
            ipAddress: clientIp,
            outcome: 'new_user_login_blocked',
          });
          return buildResponse(404, {
            message: 'NEW_USER',
            code: 'NEW_USER',
          });
        }
      } catch (err) {
        console.error('DB error in userExistsByPhone:', err);
        reportBackendError({ op: 'send_code', code: 'db_error', err });
        return buildResponse(500, { message: 'Failed to check user' });
      }
    }

    // App Store review test account — skip Twilio entirely
    if (REVIEW_TEST_PHONE && normalizedPhoneNumber === normalizePhoneNumber(REVIEW_TEST_PHONE)) {
      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'test_account',
      });
      return buildResponse(200, { message: 'Code sent successfully' });
    }

    try {
      await axios.post(
        `https://verify.twilio.com/v2/Services/${TWILIO_VERIFY_SERVICE_SID}/Verifications`,
        new URLSearchParams({
          To: normalizedPhoneNumber,
          Channel: 'sms',
        }),
        {
          auth: {
            username: TWILIO_ACCOUNT_SID,
            password: TWILIO_AUTH_TOKEN,
          },
        }
      );

      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'sent',
      });

      return buildResponse(200, { message: 'Code sent successfully' });
    } catch (error) {
      console.error('Error sending verification code:');
      console.error('Message:', error.message);
      console.error('Response data:', error.response?.data);
      console.error('Status:', error.response?.status);

      await recordVerificationAttemptSafely({
        action: 'send_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'twilio_error',
      });

      reportBackendError({ op: 'send_code', code: 'twilio_error', err: error });

      return buildResponse(500, {
        message: 'Failed to send code',
        ...(DEBUG_TWILIO && {
          twilio_error: {
            message: error.message,
            status: error.response?.status,
            data: error.response?.data,
          },
        }),
      });
    }
  }

  // 2) Verify code + signup / login
  if (path.endsWith('/verify-code')) {
    if (!normalizedPhoneNumber || !normalizedCode) {
      return buildResponse(400, { message: 'phone_number and code are required' });
    }

    if (!isAllowedPhoneNumber(normalizedPhoneNumber)) {
      await recordVerificationAttemptSafely({
        action: 'verify_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'blocked_country',
      });
      return buildResponse(403, { message: 'Phone number country is not supported' });
    }

    const rateLimitResponse = await enforceRateLimit({
      action: 'verify_code',
      phoneNumber: normalizedPhoneNumber,
      ipAddress: clientIp,
    });
    if (rateLimitResponse) {
      await recordVerificationAttemptSafely({
        action: 'verify_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'rate_limited',
      });
      return rateLimitResponse;
    }

    const normalizedEmail = normalizeOptionalEmail(email);
    if (normalizedEmail && !isValidBasicEmail(normalizedEmail)) {
      return buildResponse(400, { message: 'Invalid email format' });
    }

    try {
      // App Store review test account — skip Twilio verification entirely
      const isTestAccount = REVIEW_TEST_PHONE && REVIEW_TEST_CODE
        && normalizedPhoneNumber === normalizePhoneNumber(REVIEW_TEST_PHONE)
        && normalizedCode === REVIEW_TEST_CODE;

      if (!isTestAccount) {
        // Step 1: verify code with Twilio
        const twilioResp = await axios.post(
          `https://verify.twilio.com/v2/Services/${TWILIO_VERIFY_SERVICE_SID}/VerificationCheck`,
          new URLSearchParams({
            To: normalizedPhoneNumber,
            Code: normalizedCode,
          }),
          {
            auth: {
              username: TWILIO_ACCOUNT_SID,
              password: TWILIO_AUTH_TOKEN,
            },
          }
        );

        if (twilioResp.data.status !== 'approved') {
          await recordVerificationAttemptSafely({
            action: 'verify_code',
            phoneNumber: normalizedPhoneNumber,
            ipAddress: clientIp,
            outcome: 'invalid_code',
          });
          return buildResponse(401, { message: 'Invalid or expired code' });
        }
      }

      // If this isn't a signup request and user doesn't exist, reject and prompt signup
      if (!isSignupRequest) {
        let exists = false;
        try {
          exists = await userExistsByPhone(normalizedPhoneNumber);
        } catch (err) {
          console.error('DB error in userExistsByPhone:', err);
          reportBackendError({ op: 'verify_code', code: 'db_error', err });
          return buildResponse(500, { message: 'Failed to check user' });
        }

        if (!exists) {
          return buildResponse(404, {
            message: 'NEW_USER',
            code: 'NEW_USER',
          });
        }
      }

      // Validate invite code for signup requests (before creating user)
      if (isSignupRequest) {
        const inviteResult = await validateInviteCode(body.invite_code);
        if (!inviteResult.valid) {
          return buildResponse(403, { message: inviteResult.reason, code: 'INVALID_INVITE_CODE' });
        }
      }

      // Step 2: ensure user exists in MySQL and handle household logic / table creation
      let userRecord;
      try {
        userRecord = await findOrCreateUser({
          phoneNumber: normalizedPhoneNumber,
          firstName: first_name,
          lastName: last_name,
          zipCode: zip_code,
          email: normalizedEmail,
          joinHouseholdId: join_household_id,
        });
      } catch (err) {
        if (err.code === 'INVALID_HOUSEHOLD') {
          return buildResponse(400, { message: 'Invalid household ID' });
        }
        reportBackendError({ op: 'verify_code', ownerId: join_household_id || null, code: 'create_user_failed', err });
        return buildResponse(500, { message: 'Failed to create or fetch user' });
      }

      // Block denied users before issuing a token
      if (BLOCKED_USER_IDS.has(userRecord.userId)) {
        console.warn(`[SECURITY] Blocked user denied login: ${userRecord.userId}`);
        return buildResponse(403, { message: 'Account suspended' });
      }

      const token = await generateToken({
        ownerId: userRecord.ownerId,
        userId: userRecord.userId,
        phoneNumber: normalizedPhoneNumber,
      });

      // Record invite code usage for new signups
      if (userRecord.isNewUser && isSignupRequest && body.invite_code) {
        await recordInviteCodeUsage(body.invite_code, userRecord.userId);
      }

      await recordVerificationAttemptSafely({
        action: 'verify_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'approved',
      });

      return buildResponse(200, {
        token,
        phone_number: normalizedPhoneNumber,
        first_name: userRecord.firstName,
        email: userRecord.email,
        user_id: userRecord.userId,
        owner_id: userRecord.ownerId, // Household ID
        is_new_user: userRecord.isNewUser,
        created_new_household: userRecord.createdNewHousehold,
        message: 'Verification successful',
      });
    } catch (error) {
      console.error('Error verifying code or creating user:');
      console.error('Message:', error.message);
      console.error('Response data:', error.response?.data);
      console.error('Status:', error.response?.status);

      await recordVerificationAttemptSafely({
        action: 'verify_code',
        phoneNumber: normalizedPhoneNumber,
        ipAddress: clientIp,
        outcome: 'verify_failed',
      });

      return buildResponse(401, {
        message: 'Invalid code or failed signup',
        ...(DEBUG_TWILIO && {
          twilio_error: {
            message: error.message,
            status: error.response?.status,
            data: error.response?.data,
          },
        }),
      });
    }
  }

  // 3) Apple Sign In
  if (path.endsWith('/auth/apple') && method === 'POST') {
    const { identity_token, first_name: appleFirstName, last_name: appleLastName, email: appleEmail, zip_code: appleZipCode, join_household_id: appleJoinHouseholdId } = body || {};

    if (!identity_token) {
      return buildResponse(400, { message: 'identity_token is required' });
    }

    let applePayload;
    try {
      applePayload = await verifyAppleIdentityToken(identity_token);
    } catch (err) {
      console.error('Apple token verification failed:', err.message);
      return buildResponse(401, { message: 'Invalid Apple identity token' });
    }

    const { appleUserId, email: tokenEmail } = applePayload;
    const normalizedAppleEmail = normalizeOptionalEmail(appleEmail || tokenEmail);

    try {
      // Check if this Apple user already exists (before creating anything)
      const preCheckConn = await mysql.createConnection({
        host: DB_HOST, user: DB_USER, password: DB_PASSWORD, database: DB_NAME,
        connectTimeout: 5000,
      });
      let appleUserExists;
      try {
        const [existingRows] = await preCheckConn.query(
          'SELECT user_id FROM new_users WHERE apple_user_id = ? LIMIT 1', [appleUserId]
        );
        appleUserExists = existingRows.length > 0;
      } finally {
        await preCheckConn.end();
      }

      // New Apple user without a name → don't create yet, ask for profile + invite code
      if (!appleUserExists && !appleFirstName) {
        return buildResponse(200, {
          needs_profile: true,
          apple_user_id: appleUserId,
          message: 'Profile completion required',
        });
      }

      // New Apple user with profile (second call) → validate invite code first
      if (!appleUserExists) {
        const inviteResult = await validateInviteCode(body.invite_code);
        if (!inviteResult.valid) {
          return buildResponse(403, { message: inviteResult.reason, code: 'INVALID_INVITE_CODE' });
        }
      }

      const userRecord = await findOrCreateUserByApple({
        appleUserId,
        email: normalizedAppleEmail,
        firstName: appleFirstName || null,
        lastName: appleLastName || null,
        zipCode: appleZipCode || null,
        joinHouseholdId: appleJoinHouseholdId || null,
      });

      // Block denied users before issuing a token
      if (BLOCKED_USER_IDS.has(userRecord.userId)) {
        console.warn(`[SECURITY] Blocked user denied Apple login: ${userRecord.userId}`);
        return buildResponse(403, { message: 'Account suspended' });
      }

      // Record invite code usage for newly created users
      if (userRecord.isNewUser && body.invite_code) {
        await recordInviteCodeUsage(body.invite_code, userRecord.userId);
      }

      const token = await generateToken({
        ownerId: userRecord.ownerId,
        userId: userRecord.userId,
        phoneNumber: null,
      });

      return buildResponse(200, {
        token,
        phone_number: null,
        first_name: userRecord.firstName,
        email: userRecord.email,
        user_id: userRecord.userId,
        owner_id: userRecord.ownerId,
        is_new_user: userRecord.isNewUser,
        created_new_household: userRecord.createdNewHousehold,
        message: 'Apple sign in successful',
      });
    } catch (err) {
      if (err.code === 'INVALID_HOUSEHOLD') {
        return buildResponse(400, { message: 'Invalid household ID' });
      }
      console.error('Error in Apple sign in:', err);
      reportBackendError({ op: 'apple', ownerId: appleJoinHouseholdId || null, code: 'apple_signin_failed', err });
      return buildResponse(500, { message: 'Failed to sign in with Apple' });
    }
  }

  // 4) Delete account
  if (path.endsWith('/account') && method === 'DELETE') {
    const authorization = getHeader(event.headers, 'authorization');
    if (!authorization || !authorization.startsWith('Bearer ')) {
      return buildResponse(401, { message: 'Authorization required' });
    }

    const tokenPayload = await decodeToken(authorization.slice(7).trim());
    if (!tokenPayload) {
      return buildResponse(401, { message: 'Invalid token' });
    }

    const { user_id: tokenUserId } = tokenPayload;
    if (!tokenUserId) {
      return buildResponse(401, { message: 'Invalid token: missing user_id' });
    }

    const db = getPool();
    const conn = await db.getConnection();
    try {
      // 1) Resolve the user and their household owner_id.
      //    NOTE: owner_id is NOT the same as user_id. Solo users get
      //    owner_id == user_id, but anyone who joined/was assigned a
      //    household has a separate owner_id (e.g. a short numeric id).
      //    All current data lives in shared_* tables keyed by owner_id
      //    (WRITE_SHARED_ONLY), so deletion MUST act on owner_id, not just
      //    the user_id token value.
      const [userRows] = await conn.query(
        'SELECT user_id, owner_id, phone_number FROM new_users WHERE user_id = ? LIMIT 1',
        [tokenUserId]
      );
      if (userRows.length === 0) {
        return buildResponse(404, { message: 'User not found' });
      }
      const ownerId = userRows[0].owner_id != null ? String(userRows[0].owner_id) : null;

      // 2) Only purge household-shared data if this is the last member of
      //    the household. Otherwise we'd wipe other members' data.
      let soleMember = true;
      if (ownerId && ownerId !== String(tokenUserId)) {
        const [memberRows] = await conn.query(
          'SELECT COUNT(*) AS c FROM new_users WHERE owner_id = ?',
          [ownerId]
        );
        soleMember = Number(memberRows?.[0]?.c || 0) <= 1;
      }

      // Identities this user's data may be keyed by. The user_id (UUID) is
      // always personal; the owner_id is only safe to purge when sole member.
      const ownerScopedIds = [String(tokenUserId)];
      if (ownerId && soleMember && ownerId !== String(tokenUserId)) {
        ownerScopedIds.push(ownerId);
      }

      // 3) Purge rows from every shared_* table (keyed by owner_id).
      //    This is where all live kitchen/dish/list/discard data now lives.
      if (soleMember) {
        const [sharedTables] = await conn.query(
          `SELECT t.table_name AS tn
             FROM information_schema.tables t
             JOIN information_schema.columns c
               ON c.table_schema = t.table_schema AND c.table_name = t.table_name
            WHERE t.table_schema = DATABASE()
              AND t.table_name LIKE 'shared\\_%'
              AND c.column_name = 'owner_id'`
        );
        const idPlaceholders = ownerScopedIds.map(() => '?').join(',');
        for (const row of sharedTables) {
          const tn = row.tn || row.TABLE_NAME || row.table_name;
          if (!tn) continue;
          await conn.query(
            `DELETE FROM ${quoteId(tn)} WHERE owner_id IN (${idPlaceholders})`,
            ownerScopedIds
          );
        }
      }

      // 4) Drop legacy per-user tables (pre-migration data + caches).
      //    These are prefixed by the id + a '_' or '-' separator. We match
      //    by prefix so the suffix list never goes stale.
      const escapeLike = (s) => String(s).replace(/[\\%_]/g, (m) => '\\' + m);
      for (const prefix of ownerScopedIds) {
        const [legacyTables] = await conn.query(
          `SELECT table_name AS tn FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND (table_name LIKE ? OR table_name LIKE ?)`,
          [`${escapeLike(prefix)}\\_%`, `${escapeLike(prefix)}-%`]
        );
        for (const row of legacyTables) {
          const tn = row.tn || row.TABLE_NAME || row.table_name;
          if (!tn) continue;
          // Defensive: never drop shared_* tables, and require the exact prefix.
          if (tn.startsWith('shared_')) continue;
          if (!(tn.startsWith(`${prefix}_`) || tn.startsWith(`${prefix}-`))) continue;
          await conn.query(`DROP TABLE IF EXISTS ${quoteId(tn)}`);
        }
      }

      // 5) Remove the auth row. Done last so a mid-purge failure leaves the
      //    account recoverable rather than orphaning live data.
      await conn.query('DELETE FROM new_users WHERE user_id = ?', [tokenUserId]);

      console.log(`[DELETE_ACCOUNT] Deleted user ${tokenUserId} (owner_id=${ownerId}, soleMember=${soleMember})`);
      return buildResponse(200, { message: 'Account deleted successfully' });
    } catch (err) {
      console.error('Error deleting account:', err);
      reportBackendError({ op: 'delete_account', ownerId: (typeof ownerId !== 'undefined' ? ownerId : null), code: 'db_error', err });
      return buildResponse(500, { message: 'Failed to delete account' });
    } finally {
      // Return the connection to the pool. Must be release() (sync) — awaiting
      // conn.end() on a POOLED connection hangs ~30-60s, which previously held
      // the handler promise open until API Gateway timed out (503) even though
      // the deletion itself had already succeeded in <1s.
      try { conn.release(); } catch (_) {}
    }
  }

  return buildResponse(404, { message: 'Not found' });
};

