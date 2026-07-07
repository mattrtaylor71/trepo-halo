import mysql from 'mysql2/promise';
import { GroceryItem } from '../openai/identifyGrocery';
import { StoreAvailability } from '../openai/getStoreAvailability';
import { stableHouseholdRowId } from './householdSync';
import type { StorageGuidance } from '../../vendor/grocery-identifier/dist/utils/estimateStorageGuidance';

const WRITE_SHARED_ONLY = (process.env.WRITE_SHARED_ONLY || 'false').toLowerCase() === 'true';

async function writeToSharedKitchen(connection: mysql.Connection, owner: string, entryId: string, payload: Record<string, any>): Promise<number> {
  const row: Record<string, any> = { ...payload, owner_id: owner, _id: entryId };
  const cols = Object.keys(row);
  const placeholders = cols.map(() => '?').join(', ');
  const columnsSql = cols.map(c => `\`${c}\``).join(', ');
  const values = cols.map(c => row[c]);

  const updateClauses = cols
    .filter(c => c !== '_id' && c !== 'owner_id')
    .map(c => `\`${c}\` = VALUES(\`${c}\`)`)
    .join(', ');
  const [result] = await connection.execute<mysql.ResultSetHeader>(
    `INSERT INTO \`shared_kitchen\` (${columnsSql}) VALUES (${placeholders})
     ON DUPLICATE KEY UPDATE ${updateClauses}, \`_updatedDate\` = NOW()`,
    values
  );
  return result?.affectedRows ?? 0;
}

type KitchenAnalysisStage = 'preliminary' | 'fast' | 'final';
type KitchenAnalysisStatus = 'processing' | 'ready' | 'failed';

interface KitchenRecord {
  owner: string;
  device_id: string;
  user_id: string;
  job_id: string;
  action: 'IN' | 'OUT';
  quantity?: number;
  remaining_quantity?: string | null;
  quantity_value?: number | null;
  quantity_unit?: string | null;
  is_opened?: boolean | null;
  fill_percent?: number | null;
  product_expiration?: string | null;
  storage_guidance?: StorageGuidance | null;
  s3_key: string;
  image_url: string;
  resized_image_url?: string | null;
  resized_image_key?: string | null;
  product_image_url?: string | null;
  product_image_key?: string | null;
  swaps?: unknown;
  groceryItem: GroceryItem;
  storeAvailability?: StoreAvailability;
}

interface ArchiveKitchenRecord extends KitchenRecord {
  archived_reason: string;
}

interface ExpirationUpdate {
  owner: string;
  job_id: string;
  product_expiration: string | null;
}

interface KitchenAnalysisMetadata {
  analysis_stage: KitchenAnalysisStage;
  analysis_status: KitchenAnalysisStatus;
  analysis_source?: string | null;
  needs_review?: boolean;
  provisional_payload?: unknown;
}

interface ProvisionalKitchenRecord extends KitchenRecord {
  analysis_source?: string | null;
  analysis_stage_override?: KitchenAnalysisStage;
  needs_review?: boolean;
  provisional_payload?: unknown;
}

interface KitchenFailureUpdate {
  owner: string;
  job_id: string;
  error_message?: string | null;
}

interface KitchenRowPayload {
  product_name: string | null;
  brand: string | null;
  variant: string | null;
  category: string | null;
  remaining_quantity: string | null;
  quantity_value: number | null;
  quantity_unit: string | null;
  is_opened: number | null;
  fill_percent: number | null;
  confidence: number | null;
  explanation: string | null;
  product_description: string | null;
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
  storage_guidance: string | null;
  product_image_url: string | null;
  product_image_key: string | null;
  job_id: string;
  user_id: string;
  analysis_stage: KitchenAnalysisStage;
  analysis_status: KitchenAnalysisStatus;
  analysis_source: string | null;
  analysis_updated_at: Date;
  needs_review: number;
  swaps: string | null;
  provisional_payload: string | null;
}

function normalizeQuantity(value: number | undefined): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value as number)) : 1;
}

function buildEntryIds(jobId: string, _quantity: number): string[] {
  // Always create a single row — quantity is stored as quantity_value on the row itself.
  return [stableHouseholdRowId('prod_kitchen', jobId, 0)];
}

/**
 * Check if any of the given entry IDs have been manually deleted (archived).
 * Returns the set of IDs that were archived with reason 'manual_delete'.
 */
async function getManuallyDeletedIds(
  connection: mysql.Connection,
  ownerPrefix: string,
  entryIds: string[],
): Promise<Set<string>> {
  const archiveTableName = 'shared_archive_kitchen';
  try {
    const placeholders = entryIds.map(() => '?').join(', ');
    const [rows] = await connection.execute<mysql.RowDataPacket[]>(
      // Both reasons are user deletes and must tombstone the job so a late/duplicate
      // delivery can't resurrect a deleted item (phantom-checkin guard): kitchen_api
      // DELETE /kitchen archives as 'deleted'; the manual-delete path uses 'manual_delete'.
      `SELECT _id FROM \`${archiveTableName}\` WHERE _id IN (${placeholders}) AND archived_reason IN ('manual_delete', 'deleted')`,
      entryIds,
    );
    return new Set((rows || []).map((r: any) => String(r._id)));
  } catch {
    // Archive table may not exist yet — no deletions possible
    return new Set();
  }
}

function normalizeExpirationDate(value: string | null | undefined): string | null {
  return value && value.trim() ? value.trim() : null;
}

function buildKitchenRowPayload(
  record: KitchenRecord,
  analysis: KitchenAnalysisMetadata,
): KitchenRowPayload {
  const ingredientsJson = JSON.stringify(record.groceryItem.ingredients || []);
  const harmfulIngredientsJson = JSON.stringify(record.groceryItem.harmful_ingredients || []);
  const similarItemsJson = JSON.stringify(record.groceryItem.similar_items || []);
  const alternativesJson = JSON.stringify(record.groceryItem.alternatives || []);
  const healthierAlternativesJson = JSON.stringify(record.groceryItem.healthier_alternatives || []);
  const storeAvailabilityJson = JSON.stringify(record.storeAvailability?.store_availability || []);
  const storageGuidanceJson = record.storage_guidance ? JSON.stringify(record.storage_guidance) : null;
  const upfValue = record.groceryItem.upf === 'yes' ? 'yes' : 'no';
  const confidence = record.groceryItem.confidence != null
    ? Math.max(0, Math.min(1, record.groceryItem.confidence))
    : null;

  return {
    product_name: record.groceryItem.product_name || null,
    brand: record.groceryItem.brand || null,
    variant: record.groceryItem.variant || null,
    category: record.groceryItem.category || null,
    remaining_quantity: record.remaining_quantity || null,
    quantity_value: record.quantity_value ?? (record.quantity && record.quantity > 1 ? record.quantity : null),
    quantity_unit: record.quantity_unit || (record.quantity && record.quantity > 1 ? 'count' : null),
    is_opened: typeof record.is_opened === 'boolean' ? (record.is_opened ? 1 : 0) : null,
    fill_percent: record.fill_percent ?? null,
    confidence,
    explanation: record.groceryItem.explanation || null,
    product_description: record.groceryItem.product_description || null,
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
    storage_guidance: storageGuidanceJson,
    product_image_url: record.product_image_url || null,
    product_image_key: record.product_image_key || null,
    job_id: record.job_id,
    user_id: record.user_id,
    analysis_stage: analysis.analysis_stage,
    analysis_status: analysis.analysis_status,
    analysis_source: analysis.analysis_source || null,
    analysis_updated_at: new Date(),
    needs_review: analysis.needs_review ? 1 : 0,
    swaps: record.swaps == null ? null : JSON.stringify(record.swaps),
    provisional_payload: analysis.provisional_payload == null ? null : JSON.stringify(analysis.provisional_payload),
  };
}

export async function insertProvisionalKitchenRow(record: ProvisionalKitchenRecord): Promise<void> {
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
    const analysisStage = record.analysis_stage_override || 'fast';
    const targetQuantity = normalizeQuantity(record.quantity);
    const entryIds = buildEntryIds(record.job_id, targetQuantity);
    const payload = buildKitchenRowPayload(record, {
      analysis_stage: analysisStage,
      analysis_status: 'ready',
      analysis_source: record.analysis_source || 'quick_identify',
      needs_review: record.needs_review,
      provisional_payload: record.provisional_payload,
    });

    const deletedIds = await getManuallyDeletedIds(connection, record.owner, entryIds);
    const liveEntryIds = entryIds.filter((id) => !deletedIds.has(id));
    if (liveEntryIds.length === 0) {
      console.log(`[MySQL] Skipping provisional upsert into shared_kitchen — all entries were manually deleted (job_id=${record.job_id})`);
      return;
    }
    for (const entryId of liveEntryIds) {
      await writeToSharedKitchen(connection, record.owner, entryId, payload);
    }
    console.log(`[MySQL] Upserted ${liveEntryIds.length} provisional kitchen record(s) (stage=${analysisStage}) into shared_kitchen for job_id=${record.job_id}`);
  } catch (error) {
    console.error('[MySQL] Error writing provisional household prod_kitchen rows:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

export async function finalizeKitchenRow(record: KitchenRecord): Promise<boolean> {
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
  // Create connection
  const connection = await mysql.createConnection({
    host: DB_HOST,
    port: port,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    charset: 'utf8mb4',
  });

  try {
    const targetQuantity = normalizeQuantity(record.quantity);
    const entryIds = buildEntryIds(record.job_id, targetQuantity);
    const payload = buildKitchenRowPayload(record, {
      analysis_stage: 'final',
      analysis_status: 'ready',
      analysis_source: 'deep_identify',
      needs_review: false,
      provisional_payload: null,
    });

    const deletedIds = await getManuallyDeletedIds(connection, record.owner, entryIds);
    const liveEntryIds = entryIds.filter((id) => !deletedIds.has(id));
    if (liveEntryIds.length === 0) {
      // User deleted this item while analysis was still running/retrying. A late or
      // duplicate delivery must NOT resurrect it. Return false so the caller skips
      // the feed check-in (phantom-checkin guard).
      console.log(`[MySQL] Skipping finalize into shared_kitchen — item was deleted by the user (job_id=${record.job_id})`);
      return false;
    }
    let totalAffected = 0;
    for (const entryId of liveEntryIds) {
      totalAffected += await writeToSharedKitchen(connection, record.owner, entryId, payload);
    }
    if (totalAffected === 0) {
      console.warn(`[MySQL] finalizeKitchenRow affected 0 rows for job_id=${record.job_id} despite ${liveEntryIds.length} live entr(ies) — concurrent delete? Not confirming finalize.`);
      return false;
    }
    console.log(`[MySQL] Finalized ${liveEntryIds.length} kitchen record(s) in shared_kitchen for job_id=${record.job_id}`);
    return true;
  } catch (error) {
    console.error('[MySQL] Error finalizing household prod_kitchen rows:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

export async function writeToKitchenTable(record: KitchenRecord): Promise<void> {
  await finalizeKitchenRow(record);
}

export async function markKitchenRowsFailed(update: KitchenFailureUpdate): Promise<boolean> {
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
    const [result] = await connection.execute<mysql.ResultSetHeader>(
      `UPDATE shared_kitchen
       SET analysis_stage = 'final',
           analysis_status = 'failed',
           analysis_source = 'deep_identify',
           analysis_updated_at = NOW(),
           needs_review = 1,
           explanation = COALESCE(?, explanation)
       WHERE job_id = ? AND owner_id = ?`,
      [update.error_message || null, update.job_id, update.owner]
    );
    console.log(`[MySQL] Marked failed in shared_kitchen for job_id=${update.job_id}`);
    return (result.affectedRows ?? 0) > 0;
  } finally {
    await connection.end();
  }
}

/**
 * Promote existing fast-analysis rows to analysis_stage='final' without
 * changing product data. Used when deep analysis is skipped (e.g. Lambda
 * timeout) so the item doesn't stay in "deeper dive in progress" forever.
 */
export async function finalizeKitchenRowAsIs(owner: string, jobId: string): Promise<boolean> {
  const { DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME } = process.env;
  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing required database environment variables');
  }
  const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;
  const connection = await mysql.createConnection({
    host: DB_HOST, port, user: DB_USER, password: DB_PASS, database: DB_NAME, charset: 'utf8mb4',
  });
  try {
    const [result] = await connection.execute<mysql.ResultSetHeader>(
      `UPDATE shared_kitchen
       SET analysis_stage = 'final',
           analysis_status = 'ready',
           analysis_source = 'fast_only',
           analysis_updated_at = NOW()
       WHERE job_id = ? AND owner_id = ? AND analysis_stage != 'final'`,
      [jobId, owner]
    );
    console.log(`[MySQL] Finalized ${result.affectedRows ?? 0} fast-only row(s) in shared_kitchen for job_id=${jobId}`);
    return (result.affectedRows ?? 0) > 0;
  } finally {
    await connection.end();
  }
}

export async function deleteKitchenRowsByJobId(owner: string, jobId: string): Promise<boolean> {
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
    const [result] = await connection.execute<mysql.ResultSetHeader>(
      `DELETE FROM shared_kitchen WHERE job_id = ? AND owner_id = ?`,
      [jobId, owner]
    );
    console.log(`[MySQL] Deleted from shared_kitchen for job_id=${jobId}`);
    return (result.affectedRows ?? 0) > 0;
  } finally {
    await connection.end();
  }
}

export async function writeToArchiveKitchenTable(record: ArchiveKitchenRecord): Promise<void> {
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
    const targetQuantity = Number.isFinite(record.quantity)
      ? Math.max(1, Math.floor(record.quantity as number))
      : 1;
    const entryIds = Array.from(
      { length: targetQuantity },
      (_, index) => stableHouseholdRowId('prod_kitchen', record.job_id, index)
    );

    const {
      groceryItem,
      storeAvailability,
      device_id,
      user_id,
      job_id,
      action,
      product_expiration,
      storage_guidance,
      s3_key,
      image_url,
      product_image_url,
      product_image_key,
      archived_reason,
    } = record;

    const ingredientsJson = JSON.stringify(groceryItem.ingredients || []);
    const harmfulIngredientsJson = JSON.stringify(groceryItem.harmful_ingredients || []);
    const similarItemsJson = JSON.stringify(groceryItem.similar_items || []);
    const alternativesJson = JSON.stringify(groceryItem.alternatives || []);
    const healthierAlternativesJson = JSON.stringify(groceryItem.healthier_alternatives || []);
    const storeAvailabilityJson = JSON.stringify(storeAvailability?.store_availability || []);
    const storageGuidanceJson = storage_guidance ? JSON.stringify(storage_guidance) : null;
    const upfValue = groceryItem.upf === 'yes' ? 'yes' : 'no';
    const confidence = groceryItem.confidence != null
      ? Math.max(0, Math.min(1, groceryItem.confidence))
      : null;
    const expirationDate = product_expiration && product_expiration.trim()
      ? product_expiration.trim()
      : null;

    const sqlForTable = (tableName: string) => `
      INSERT INTO \`${tableName}\`
      (
        _id, _owner, _device, _createdDate,
        product_name, brand, variant, category, confidence, explanation, product_description, barcode, country_guess,
        estimated_price,
        ingredients, nutrition_summary, upf, harmful_ingredients,
        similar_items, alternatives, healthier_alternatives,
        store_availability,
        images, s3_key, action, product_expiration, storage_guidance,
        product_image_url, product_image_key,
        job_id, user_id,
        archived_at, archived_reason, archived_from_table
      )
      VALUES (?, ?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), ?, ?)
      ON DUPLICATE KEY UPDATE
        archived_at = VALUES(archived_at),
        archived_reason = VALUES(archived_reason),
        archived_from_table = VALUES(archived_from_table)
    `;

    const buildValues = (entryId: string, targetOwnerId: string) => ([
      entryId,
      targetOwnerId,
      device_id,
      groceryItem.product_name || null,
      groceryItem.brand || null,
      groceryItem.variant || null,
      groceryItem.category || null,
      confidence,
      groceryItem.explanation || null,
      groceryItem.product_description || null,
      groceryItem.barcode || null,
      groceryItem.country_guess || null,
      groceryItem.estimated_price || null,
      ingredientsJson,
      groceryItem.nutrition_summary || null,
      upfValue,
      harmfulIngredientsJson,
      similarItemsJson,
      alternativesJson,
      healthierAlternativesJson,
      storeAvailabilityJson,
      image_url,
      s3_key,
      action,
      expirationDate,
      storageGuidanceJson,
      product_image_url || null,
      product_image_key || null,
      job_id,
      user_id,
      archived_reason,
      `${targetOwnerId}_prod_kitchen`,
    ]);

    const archiveTableName = 'shared_archive_kitchen';
    const [existingRows] = await connection.execute<mysql.RowDataPacket[]>(
      `SELECT _id FROM \`${archiveTableName}\` WHERE job_id = ? AND owner_id = ?`,
      [record.job_id, record.owner]
    );
    const existingCount = Array.isArray(existingRows) ? existingRows.length : 0;
    if (existingCount < targetQuantity) {
      const existingIdSet = new Set(
        (existingRows || []).map((row: any) => String(row._id || ''))
      );
      const missingEntryIds = entryIds.filter((entryId) => !existingIdSet.has(entryId));
      for (const entryId of missingEntryIds) {
        await connection.execute(sqlForTable(archiveTableName), buildValues(entryId, record.owner));
      }
      console.log(`[MySQL] Inserted ${missingEntryIds.length} archived record(s) into ${archiveTableName} for job_id=${job_id}`);
    }
  } catch (error) {
    console.error('[MySQL] Error writing to household archive kitchen tables:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

export async function updateKitchenExpiration(update: ExpirationUpdate): Promise<boolean> {
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
    const expirationDate = update.product_expiration && update.product_expiration.trim()
      ? update.product_expiration.trim()
      : null;
    const [result] = await connection.execute<mysql.ResultSetHeader>(
      `UPDATE shared_kitchen SET product_expiration = ? WHERE job_id = ? AND owner_id = ?`,
      [expirationDate, update.job_id, update.owner]
    );
    console.log(`[MySQL] Updated expiration in shared_kitchen for job_id=${update.job_id}`);
    const updatedAny = (result.affectedRows ?? 0) > 0;

    if (updatedAny) {
      console.log(`[MySQL] Updated expiration for household job_id=${update.job_id}`);
      return true;
    }

    console.log(`[MySQL] No existing household rows for job_id=${update.job_id}, expiration update skipped`);
    return false;
  } catch (error) {
    console.error('[MySQL] Error updating household expiration:', error);
    throw error;
  } finally {
    await connection.end();
  }
}

