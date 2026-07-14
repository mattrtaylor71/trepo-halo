import mysql from 'mysql2/promise';
import { GroceryItem } from '../openai/identifyGrocery';
import { StoreAvailability } from '../openai/getStoreAvailability';
import { getHouseholdMemberIds, stableHouseholdRowId } from './householdSync';

// Shared-table migration: mirror each per-member discard row into shared_discards
// (owner_id-keyed, PK (owner_id,_id)). Best-effort, gated by DUAL_WRITE_DISCARDS.
const DUAL_WRITE_DISCARDS = String(process.env.DUAL_WRITE_DISCARDS || 'false').toLowerCase() === 'true';

async function dualWriteDiscardsToShared(
  connection: mysql.Connection,
  ownerId: string,
  memberTable: string,
  entryId: string
): Promise<void> {
  if (!DUAL_WRITE_DISCARDS) return;
  try {
    const [srcRows] = await connection.execute<mysql.RowDataPacket[]>(
      "SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=?",
      [memberTable]
    );
    const [shdRows] = await connection.execute<mysql.RowDataPacket[]>(
      "SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='shared_discards'"
    );
    const src = new Set(srcRows.map((r) => r.cn || r.COLUMN_NAME));
    const cols = shdRows.map((r) => r.cn || r.COLUMN_NAME).filter((c) => src.has(c) && c !== 'owner_id');
    if (!cols.length) return;
    const collist = cols.map((c) => `\`${c}\``).join(', ');
    const s = cols.map((c) => `s.\`${c}\``).join(', ');
    const upd = cols.filter((c) => c !== '_id').map((c) => `\`${c}\`=VALUES(\`${c}\`)`).join(', ');
    await connection.execute(
      `INSERT INTO \`shared_discards\` (\`owner_id\`, ${collist}) SELECT ?, ${s} FROM \`${memberTable}\` s WHERE s.\`_id\` = ? ON DUPLICATE KEY UPDATE \`owner_id\`=VALUES(\`owner_id\`), ${upd}`,
      [ownerId, entryId]
    );
  } catch (err) {
    try {
      console.error(JSON.stringify({
        evt: 'dual_write_miss',
        family: 'discards',
        owner_id: String(ownerId || ''),
        item_id: String(entryId || ''),
        error: String(err instanceof Error ? err.message : err).slice(0, 500),
      }));
    } catch (_) { /* never block the primary write */ }
  }
}

type AnalysisStage = 'fast' | 'final';
type AnalysisStatus = 'processing' | 'ready' | 'failed';

interface DiscardRecord {
  owner: string;
  device_id: string;
  user_id: string;
  job_id: string;
  action: 'IN' | 'OUT';
  product_expiration?: string | null;
  s3_key: string;
  image_url: string;
  resized_image_url?: string | null;
  resized_image_key?: string | null;
  product_image_url?: string | null;
  product_image_key?: string | null;
  groceryItem: GroceryItem;
  storeAvailability?: StoreAvailability;
}

interface ProvisionalDiscardRecord extends DiscardRecord {
  analysis_source?: string | null;
  needs_review?: boolean;
  provisional_payload?: unknown;
}

interface DiscardFailureUpdate {
  owner: string;
  job_id: string;
  error_message?: string | null;
}

interface AnalysisMetadata {
  analysis_stage: AnalysisStage;
  analysis_status: AnalysisStatus;
  analysis_source?: string | null;
  needs_review?: boolean;
  provisional_payload?: unknown;
}

interface DiscardRowPayload {
  product_name: string | null;
  brand: string | null;
  variant: string | null;
  category: string | null;
  confidence: number | null;
  explanation: string | null;
  barcode: string | null;
  country_guess: string | null;
  estimated_price: string | null;
  ingredients: string;
  nutrition_summary: string | null;
  upf: 'yes' | 'no';
  harmful_ingredients: string;
  similar_items: string;
  alternatives: string;
  healthier_alternatives: string;
  store_availability: string;
  images: string;
  s3_key: string;
  resized_image_url: string | null;
  resized_image_key: string | null;
  action: 'IN' | 'OUT';
  product_expiration: string | null;
  product_image_url: string | null;
  product_image_key: string | null;
  job_id: string;
  user_id: string;
  analysis_stage: AnalysisStage;
  analysis_status: AnalysisStatus;
  analysis_source: string | null;
  analysis_updated_at: Date;
  needs_review: number;
  provisional_payload: string | null;
}

const DISCARD_INSERT_COLUMNS = [
  '_id', '_owner', '_device',
  'product_name', 'brand', 'variant', 'category', 'confidence', 'explanation', 'barcode', 'country_guess',
  'estimated_price',
  'ingredients', 'nutrition_summary', 'upf', 'harmful_ingredients',
  'similar_items', 'alternatives', 'healthier_alternatives',
  'store_availability',
  'images', 's3_key', 'action', 'product_expiration',
  'resized_image_url', 'resized_image_key',
  'product_image_url', 'product_image_key',
  'job_id', 'user_id',
  'analysis_stage', 'analysis_status', 'analysis_source', 'analysis_updated_at', 'needs_review', 'provisional_payload',
] as const;

const DISCARD_UPDATE_COLUMNS = [
  'product_name', 'brand', 'variant', 'category', 'confidence', 'explanation', 'barcode', 'country_guess',
  'estimated_price',
  'ingredients', 'nutrition_summary', 'upf', 'harmful_ingredients',
  'similar_items', 'alternatives', 'healthier_alternatives',
  'store_availability',
  'images', 's3_key', 'action', 'product_expiration',
  'resized_image_url', 'resized_image_key',
  'product_image_url', 'product_image_key',
  'job_id', 'user_id',
  'analysis_stage', 'analysis_status', 'analysis_source', 'analysis_updated_at', 'needs_review', 'provisional_payload',
] as const;

function normalizeExpirationDate(value: string | null | undefined): string | null {
  return value && value.trim() ? value.trim() : null;
}

function buildDiscardPayload(record: DiscardRecord, analysis: AnalysisMetadata): DiscardRowPayload {
  const ingredientsJson = JSON.stringify(record.groceryItem.ingredients || []);
  const harmfulIngredientsJson = JSON.stringify(record.groceryItem.harmful_ingredients || []);
  const similarItemsJson = JSON.stringify(record.groceryItem.similar_items || []);
  const alternativesJson = JSON.stringify(record.groceryItem.alternatives || []);
  const healthierAlternativesJson = JSON.stringify(record.groceryItem.healthier_alternatives || []);
  const storeAvailabilityJson = JSON.stringify(record.storeAvailability?.store_availability || []);
  const upfValue = record.groceryItem.upf === 'yes' ? 'yes' : 'no';
  const confidence = record.groceryItem.confidence != null
    ? Math.max(0, Math.min(1, record.groceryItem.confidence))
    : null;

  return {
    product_name: record.groceryItem.product_name || null,
    brand: record.groceryItem.brand || null,
    variant: record.groceryItem.variant || null,
    category: record.groceryItem.category || null,
    confidence,
    explanation: record.groceryItem.explanation || null,
    barcode: record.groceryItem.barcode || null,
    country_guess: record.groceryItem.country_guess || null,
    estimated_price: record.groceryItem.estimated_price || null,
    ingredients: ingredientsJson,
    nutrition_summary: record.groceryItem.nutrition_summary || null,
    upf: upfValue,
    harmful_ingredients: harmfulIngredientsJson,
    similar_items: similarItemsJson,
    alternatives: alternativesJson,
    healthier_alternatives: healthierAlternativesJson,
    store_availability: storeAvailabilityJson,
    images: record.image_url,
    s3_key: record.s3_key,
    resized_image_url: record.resized_image_url || null,
    resized_image_key: record.resized_image_key || null,
    action: record.action,
    product_expiration: normalizeExpirationDate(record.product_expiration),
    product_image_url: record.product_image_url || null,
    product_image_key: record.product_image_key || null,
    job_id: record.job_id,
    user_id: record.user_id,
    analysis_stage: analysis.analysis_stage,
    analysis_status: analysis.analysis_status,
    analysis_source: analysis.analysis_source || null,
    analysis_updated_at: new Date(),
    needs_review: analysis.needs_review ? 1 : 0,
    provisional_payload: analysis.provisional_payload == null ? null : JSON.stringify(analysis.provisional_payload),
  };
}

function sqlForDiscardInsert(tableName: string): string {
  return `
      INSERT INTO \`${tableName}\`
      (${DISCARD_INSERT_COLUMNS.map((column) => `\`${column}\``).join(', ')})
      VALUES (${DISCARD_INSERT_COLUMNS.map(() => '?').join(', ')})
    `;
}

function sqlForDiscardUpsert(tableName: string): string {
  return `
      ${sqlForDiscardInsert(tableName)}
      ON DUPLICATE KEY UPDATE
      ${DISCARD_UPDATE_COLUMNS.map((column) => `\`${column}\` = VALUES(\`${column}\`)`).join(',\n      ')}
    `;
}

function buildDiscardValues(entryId: string, targetOwnerId: string, record: DiscardRecord, payload: DiscardRowPayload) {
  return [
    entryId,
    targetOwnerId,
    record.device_id,
    payload.product_name,
    payload.brand,
    payload.variant,
    payload.category,
    payload.confidence,
    payload.explanation,
    payload.barcode,
    payload.country_guess,
    payload.estimated_price,
    payload.ingredients,
    payload.nutrition_summary,
    payload.upf,
    payload.harmful_ingredients,
    payload.similar_items,
    payload.alternatives,
    payload.healthier_alternatives,
    payload.store_availability,
    payload.images,
    payload.s3_key,
    payload.action,
    payload.product_expiration,
    payload.resized_image_url,
    payload.resized_image_key,
    payload.product_image_url,
    payload.product_image_key,
    payload.job_id,
    payload.user_id,
    payload.analysis_stage,
    payload.analysis_status,
    payload.analysis_source,
    payload.analysis_updated_at,
    payload.needs_review,
    payload.provisional_payload,
  ];
}

export async function insertProvisionalDiscardRow(record: ProvisionalDiscardRecord): Promise<void> {
  const {
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASS,
    DB_NAME,
  } = process.env;

  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing required database environment variables');
  }

  const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;

  const connection = await mysql.createConnection({
    host: DB_HOST,
    port: port,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    charset: 'utf8mb4',
  });

  try {
    const memberIds = await getHouseholdMemberIds(connection, record.owner);
    const entryId = stableHouseholdRowId('discards', record.job_id);
    const payload = buildDiscardPayload(record, {
      analysis_stage: 'fast',
      analysis_status: 'ready',
      analysis_source: record.analysis_source || 'quick_identify',
      needs_review: record.needs_review,
      provisional_payload: record.provisional_payload,
    });

    for (const memberId of memberIds) {
      const tableName = `${memberId.replace(/[^a-zA-Z0-9_-]/g, '')}_discards`;
      await ensureTableExists(connection, tableName);
      const [existingRows] = await connection.execute<mysql.RowDataPacket[]>(
        `SELECT _id FROM \`${tableName}\` WHERE _id = ? LIMIT 1`,
        [entryId]
      );
      if (Array.isArray(existingRows) && existingRows.length > 0) {
        continue;
      }

      await connection.execute(sqlForDiscardInsert(tableName), buildDiscardValues(entryId, memberId, record, payload));
      console.log(`[MySQL] Successfully inserted provisional discard into ${tableName}, entry_id: ${entryId}`);
      await dualWriteDiscardsToShared(connection, memberId, tableName, entryId);
    }
  } catch (error) {
    console.error('[MySQL] Error writing provisional household discards:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

export async function finalizeDiscardRow(record: DiscardRecord): Promise<void> {
  const {
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASS,
    DB_NAME,
  } = process.env;

  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing required database environment variables');
  }

  const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;

  const connection = await mysql.createConnection({
    host: DB_HOST,
    port: port,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    charset: 'utf8mb4',
  });

  try {
    const memberIds = await getHouseholdMemberIds(connection, record.owner);
    const entryId = stableHouseholdRowId('discards', record.job_id);
    const payload = buildDiscardPayload(record, {
      analysis_stage: 'final',
      analysis_status: 'ready',
      analysis_source: 'deep_identify',
      needs_review: false,
      provisional_payload: null,
    });

    for (const memberId of memberIds) {
      const tableName = `${memberId.replace(/[^a-zA-Z0-9_-]/g, '')}_discards`;
      await ensureTableExists(connection, tableName);
      await connection.execute(sqlForDiscardUpsert(tableName), buildDiscardValues(entryId, memberId, record, payload));
      console.log(`[MySQL] Successfully inserted discard into ${tableName}, entry_id: ${entryId}`);
      await dualWriteDiscardsToShared(connection, memberId, tableName, entryId);
    }
  } catch (error) {
    console.error('[MySQL] Error writing household discards:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

export async function writeToDiscardsTable(record: DiscardRecord): Promise<void> {
  await finalizeDiscardRow(record);
}

export async function markDiscardRowFailed(update: DiscardFailureUpdate): Promise<boolean> {
  const {
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASS,
    DB_NAME,
  } = process.env;

  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing required database environment variables');
  }

  const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;
  const connection = await mysql.createConnection({
    host: DB_HOST,
    port,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    charset: 'utf8mb4',
  });

  try {
    const memberIds = await getHouseholdMemberIds(connection, update.owner);
    let updatedAny = false;
    for (const memberId of memberIds) {
      const tableName = `${memberId.replace(/[^a-zA-Z0-9_-]/g, '')}_discards`;
      await ensureTableExists(connection, tableName);
      const [result] = await connection.execute<mysql.ResultSetHeader>(
        `UPDATE \`${tableName}\`
         SET analysis_status = 'failed',
             analysis_source = 'deep_identify',
             analysis_updated_at = NOW(),
             needs_review = 1,
             explanation = COALESCE(?, explanation)
         WHERE job_id = ? AND _owner = ?`,
        [update.error_message || null, update.job_id, memberId]
      );
      if (result.affectedRows && result.affectedRows > 0) {
        updatedAny = true;
        const [idRows] = await connection.execute<mysql.RowDataPacket[]>(
          `SELECT _id FROM \`${tableName}\` WHERE job_id = ? AND _owner = ?`,
          [update.job_id, memberId]
        );
        for (const row of idRows) {
          await dualWriteDiscardsToShared(connection, memberId, tableName, String(row._id));
        }
      }
    }
    return updatedAny;
  } finally {
    await connection.end();
  }
}

async function ensureTableExists(connection: mysql.Connection, tableName: string): Promise<void> {
  const [tables] = await connection.execute(
    `SELECT COUNT(*) as count FROM information_schema.tables 
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName]
  ) as any[];

  if (tables[0].count === 0) {
    const createTableSql = `
      CREATE TABLE IF NOT EXISTS \`${tableName}\` (
        \`_id\` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`product_name\` VARCHAR(500) COMMENT 'Product name identified by AI',
        \`brand\` VARCHAR(255) COMMENT 'Brand name if visible',
        \`variant\` VARCHAR(255) COMMENT 'Product variant',
        \`category\` VARCHAR(255) COMMENT 'Product category',
        \`confidence\` DECIMAL(3,2) COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COMMENT 'AI explanation of identification',
        \`barcode\` VARCHAR(100) COMMENT 'Barcode if detected',
        \`country_guess\` VARCHAR(100) COMMENT 'Country of origin guess',
        \`estimated_price\` VARCHAR(50) COMMENT 'Estimated price',
        \`ingredients\` JSON COMMENT 'Array of ingredient strings',
        \`nutrition_summary\` TEXT COMMENT 'Nutrition information summary',
        \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag',
        \`harmful_ingredients\` JSON COMMENT 'Array of harmful ingredient strings',
        \`similar_items\` JSON COMMENT 'Array of similar items',
        \`alternatives\` JSON COMMENT 'Array of alternative products',
        \`healthier_alternatives\` JSON COMMENT 'Array of healthier alternatives',
        \`store_availability\` JSON COMMENT 'Array of stores',
        \`images\` VARCHAR(1000) COMMENT 'S3 URL to the uploaded image',
        \`s3_key\` VARCHAR(500) COMMENT 'S3 object key for the image',
        \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN' COMMENT 'IN = adding to discards, OUT = removing',
        \`product_expiration\` DATE COMMENT 'Product expiration date (YYYY-MM-DD)',
        \`resized_image_url\` VARCHAR(1000) COMMENT 'S3 URL to resized app-friendly original image',
        \`resized_image_key\` VARCHAR(500) COMMENT 'S3 object key for resized app-friendly original image',
        \`product_image_url\` VARCHAR(1000) COMMENT 'S3 URL to generated product image',
        \`product_image_key\` VARCHAR(500) COMMENT 'S3 object key for the generated product image',
        \`job_id\` VARCHAR(100) COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COMMENT 'User ID from the upload request',
        \`analysis_stage\` ENUM('fast', 'final') DEFAULT 'final' COMMENT 'Whether this row is provisional fast output or final deep output',
        \`analysis_status\` ENUM('processing', 'ready', 'failed') DEFAULT 'ready' COMMENT 'Availability of the row analysis output',
        \`analysis_source\` VARCHAR(64) DEFAULT NULL COMMENT 'Which analyzer last wrote this row',
        \`analysis_updated_at\` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'When analysis fields were last updated',
        \`needs_review\` TINYINT(1) NOT NULL DEFAULT 0 COMMENT 'Whether the row should be reviewed by the user',
        \`provisional_payload\` JSON COMMENT 'Raw provisional fast-scan payload for debugging and refinement',
        INDEX \`idx_owner\` (\`_owner\`),
        INDEX \`idx_device\` (\`_device\`),
        INDEX \`idx_product_name\` (\`product_name\`),
        INDEX \`idx_category\` (\`category\`),
        INDEX \`idx_action\` (\`action\`),
        INDEX \`idx_created\` (\`_createdDate\`),
        INDEX \`idx_expiration\` (\`product_expiration\`),
        INDEX \`idx_analysis_stage_status\` (\`analysis_stage\`, \`analysis_status\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Discarded groceries - same structure as prod_kitchen'
    `;

    await connection.execute(createTableSql);
    console.log(`[MySQL] Created table ${tableName}`);
  } else {
    await ensureColumnsExist(connection, tableName);
  }
}

async function ensureColumnsExist(connection: mysql.Connection, tableName: string): Promise<void> {
  const [columns] = await connection.execute(
    `SELECT column_name, column_type FROM information_schema.columns 
     WHERE table_schema = DATABASE() AND table_name = ? AND column_name IN (
       'variant',
       'confidence',
       'explanation',
       'barcode',
       'country_guess',
       'estimated_price',
       'ingredients',
       'nutrition_summary',
       'upf',
       'harmful_ingredients',
       'similar_items',
       'alternatives',
       'healthier_alternatives',
       'store_availability',
       's3_key',
       'resized_image_url',
       'resized_image_key',
       'product_image_url',
       'product_image_key',
       'analysis_stage',
       'analysis_status',
       'analysis_source',
       'analysis_updated_at',
       'needs_review',
       'provisional_payload'
     )`,
    [tableName]
  ) as any[];

  const columnMap = new Map(
    (columns || []).map((row: { column_name?: string; column_type?: string; COLUMN_NAME?: string; COLUMN_TYPE?: string }) => [
      row.column_name || row.COLUMN_NAME || '',
      row.column_type || row.COLUMN_TYPE || ''
    ]).filter(([name]: [string, string]) => name)
  );
  const alterParts: string[] = [];

  if (!columnMap.has('variant')) {
    alterParts.push("ADD COLUMN `variant` VARCHAR(255) COMMENT 'Product variant' AFTER `brand`");
  }
  if (!columnMap.has('confidence')) {
    alterParts.push("ADD COLUMN `confidence` DECIMAL(3,2) COMMENT 'AI confidence score (0.00 to 1.00)' AFTER `category`");
  }
  if (!columnMap.has('explanation')) {
    alterParts.push("ADD COLUMN `explanation` TEXT COMMENT 'AI explanation of identification' AFTER `confidence`");
  }
  if (!columnMap.has('barcode')) {
    alterParts.push("ADD COLUMN `barcode` VARCHAR(100) COMMENT 'Barcode if detected' AFTER `explanation`");
  }
  if (!columnMap.has('country_guess')) {
    alterParts.push("ADD COLUMN `country_guess` VARCHAR(100) COMMENT 'Country of origin guess' AFTER `barcode`");
  }
  if (!columnMap.has('estimated_price')) {
    alterParts.push("ADD COLUMN `estimated_price` VARCHAR(50) COMMENT 'Estimated price' AFTER `country_guess`");
  }
  if (!columnMap.has('ingredients')) {
    alterParts.push("ADD COLUMN `ingredients` JSON COMMENT 'Array of ingredient strings' AFTER `estimated_price`");
  }
  if (!columnMap.has('nutrition_summary')) {
    alterParts.push("ADD COLUMN `nutrition_summary` TEXT COMMENT 'Nutrition information summary' AFTER `ingredients`");
  }

  if (!columnMap.has('upf')) {
    alterParts.push("ADD COLUMN `upf` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag' AFTER `nutrition_summary`");
  } else {
    const upfType = String(columnMap.get('upf') || '');
    const normalizedUpfType = upfType.replace(/\s+/g, '');
    if (normalizedUpfType !== "enum('yes','no')") {
      console.log(`[MySQL] Normalizing upf enum values on ${tableName}`);
      await connection.execute(`
        ALTER TABLE \`${tableName}\`
        MODIFY COLUMN \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'
      `);
      await connection.execute(`UPDATE \`${tableName}\` SET \`upf\` = LOWER(\`upf\`) WHERE \`upf\` IS NOT NULL`);
    }
  }

  if (!columnMap.has('harmful_ingredients')) {
    alterParts.push("ADD COLUMN `harmful_ingredients` JSON COMMENT 'Array of harmful ingredient strings' AFTER `upf`");
  }
  if (!columnMap.has('similar_items')) {
    alterParts.push("ADD COLUMN `similar_items` JSON COMMENT 'Array of similar items' AFTER `harmful_ingredients`");
  }
  if (!columnMap.has('alternatives')) {
    alterParts.push("ADD COLUMN `alternatives` JSON COMMENT 'Array of alternative products' AFTER `similar_items`");
  }
  if (!columnMap.has('healthier_alternatives')) {
    alterParts.push("ADD COLUMN `healthier_alternatives` JSON COMMENT 'Array of healthier alternatives' AFTER `alternatives`");
  }
  if (!columnMap.has('store_availability')) {
    alterParts.push("ADD COLUMN `store_availability` JSON COMMENT 'Array of stores' AFTER `healthier_alternatives`");
  }
  if (!columnMap.has('s3_key')) {
    alterParts.push("ADD COLUMN `s3_key` VARCHAR(500) COMMENT 'S3 object key for the image' AFTER `images`");
  }
  if (!columnMap.has('product_image_url')) {
    alterParts.push("ADD COLUMN `product_image_url` VARCHAR(1000) COMMENT 'S3 URL to generated product image' AFTER `product_expiration`");
  }
  if (!columnMap.has('product_image_key')) {
    alterParts.push("ADD COLUMN `product_image_key` VARCHAR(500) COMMENT 'S3 object key for the generated product image' AFTER `product_image_url`");
  }
  if (!columnMap.has('resized_image_url')) {
    alterParts.push("ADD COLUMN `resized_image_url` VARCHAR(1000) COMMENT 'S3 URL to resized app-friendly original image' AFTER `product_expiration`");
  }
  if (!columnMap.has('resized_image_key')) {
    alterParts.push("ADD COLUMN `resized_image_key` VARCHAR(500) COMMENT 'S3 object key for resized app-friendly original image' AFTER `resized_image_url`");
  }
  if (!columnMap.has('analysis_stage')) {
    alterParts.push("ADD COLUMN `analysis_stage` ENUM('fast', 'final') DEFAULT 'final' COMMENT 'Whether this row is provisional fast output or final deep output' AFTER `user_id`");
  }
  if (!columnMap.has('analysis_status')) {
    alterParts.push("ADD COLUMN `analysis_status` ENUM('processing', 'ready', 'failed') DEFAULT 'ready' COMMENT 'Availability of the row analysis output' AFTER `analysis_stage`");
  }
  if (!columnMap.has('analysis_source')) {
    alterParts.push("ADD COLUMN `analysis_source` VARCHAR(64) DEFAULT NULL COMMENT 'Which analyzer last wrote this row' AFTER `analysis_status`");
  }
  if (!columnMap.has('analysis_updated_at')) {
    alterParts.push("ADD COLUMN `analysis_updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'When analysis fields were last updated' AFTER `analysis_source`");
  }
  if (!columnMap.has('needs_review')) {
    alterParts.push("ADD COLUMN `needs_review` TINYINT(1) NOT NULL DEFAULT 0 COMMENT 'Whether the row should be reviewed by the user' AFTER `analysis_updated_at`");
  }
  if (!columnMap.has('provisional_payload')) {
    alterParts.push("ADD COLUMN `provisional_payload` JSON COMMENT 'Raw provisional fast-scan payload for debugging and refinement' AFTER `needs_review`");
  }

  if (alterParts.length > 0) {
    console.log(`[MySQL] Backfilling missing discard columns on ${tableName}:`, alterParts.map((part) => part.split('`')[1]));
    await connection.execute(`
      ALTER TABLE \`${tableName}\`
      ${alterParts.join(',\n      ')}
    `);
    console.log(`[MySQL] Added missing discard columns to ${tableName}`);
  }
}
