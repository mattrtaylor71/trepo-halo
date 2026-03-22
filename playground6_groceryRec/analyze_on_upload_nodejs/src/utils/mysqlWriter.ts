import mysql from 'mysql2/promise';
import { GroceryItem } from '../openai/identifyGrocery';
import { StoreAvailability } from '../openai/getStoreAvailability';
import { getHouseholdMemberIds, stableHouseholdRowId } from './householdSync';

interface KitchenRecord {
  owner: string;
  device_id: string;
  user_id: string;
  job_id: string;
  action: 'IN' | 'OUT';
  quantity?: number;
  product_expiration?: string | null;
  s3_key: string;
  image_url: string;
  product_image_url?: string | null;
  product_image_key?: string | null;
  groceryItem: GroceryItem;
  storeAvailability?: StoreAvailability;
}

interface ExpirationUpdate {
  owner: string;
  job_id: string;
  product_expiration: string | null;
}

export async function writeToKitchenTable(record: KitchenRecord): Promise<void> {
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
    const targetQuantity = Number.isFinite(record.quantity)
      ? Math.max(1, Math.floor(record.quantity as number))
      : 1;
    const memberIds = await getHouseholdMemberIds(connection, record.owner);
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
      s3_key,
      image_url,
      product_image_url,
      product_image_key,
    } = record;

    // Convert arrays/objects to JSON strings for MySQL
    const ingredientsJson = JSON.stringify(groceryItem.ingredients || []);
    const harmfulIngredientsJson = JSON.stringify(groceryItem.harmful_ingredients || []);
    const similarItemsJson = JSON.stringify(groceryItem.similar_items || []);
    const alternativesJson = JSON.stringify(groceryItem.alternatives || []);
    const healthierAlternativesJson = JSON.stringify(groceryItem.healthier_alternatives || []);
    const storeAvailabilityJson = JSON.stringify(storeAvailability?.store_availability || []);
    const upfValue = groceryItem.upf === 'yes' ? 'yes' : 'no';

    // Convert confidence to decimal (0.00 to 1.00)
    const confidence = groceryItem.confidence != null 
      ? Math.max(0, Math.min(1, groceryItem.confidence))
      : null;

    // Prepare expiration date (YYYY-MM-DD or NULL)
    const expirationDate = product_expiration && product_expiration.trim() 
      ? product_expiration.trim() 
      : null;

    // Insert query
    const sqlForTable = (tableName: string) => `
      INSERT INTO \`${tableName}\`
      (
        _id, _owner, _device, _createdDate,
        product_name, brand, variant, category, confidence, explanation, product_description, barcode, country_guess,
        estimated_price,
        ingredients, nutrition_summary, upf, harmful_ingredients,
        similar_items, alternatives, healthier_alternatives,
        store_availability,
        images, s3_key, action, product_expiration,
        product_image_url, product_image_key,
        job_id, user_id
      )
      VALUES (?, ?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
      product_image_url || null,
      product_image_key || null,
      job_id,
      user_id,
    ]);

    for (const memberId of memberIds) {
      const escapedOwner = memberId.replace(/[^a-zA-Z0-9_-]/g, '');
      const tableName = `${escapedOwner}_prod_kitchen`;
      await ensureTableExists(connection, tableName);

      const [existingRows] = await connection.execute<mysql.RowDataPacket[]>(
        `SELECT _id FROM \`${tableName}\` WHERE job_id = ? AND _owner = ?`,
        [record.job_id, memberId]
      );
      const existingCount = Array.isArray(existingRows) ? existingRows.length : 0;
      if (existingCount >= targetQuantity) {
        continue;
      }
      const existingIdSet = new Set(
        (existingRows || []).map((row: any) => String(row._id || ''))
      );
      const missingEntryIds = entryIds.filter((entryId) => !existingIdSet.has(entryId));
      for (const entryId of missingEntryIds) {
        await connection.execute(sqlForTable(tableName), buildValues(entryId, memberId));
      }
      console.log(`[MySQL] Successfully inserted ${missingEntryIds.length} record(s) into ${tableName} for job_id=${job_id}`);
    }
  } catch (error) {
    console.error('[MySQL] Error writing to household prod_kitchen tables:', error);
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
    const memberIds = await getHouseholdMemberIds(connection, update.owner);
    let updatedAny = false;
    for (const memberId of memberIds) {
      const escapedOwner = memberId.replace(/[^a-zA-Z0-9_-]/g, '');
      const tableName = `${escapedOwner}_prod_kitchen`;
      await ensureTableExists(connection, tableName);
      const [result] = await connection.execute<mysql.ResultSetHeader>(
        `UPDATE \`${tableName}\` SET product_expiration = ? WHERE job_id = ? AND _owner = ?`,
        [expirationDate, update.job_id, memberId]
      );
      if (result.affectedRows && result.affectedRows > 0) {
        updatedAny = true;
      }
    }

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

async function ensureTableExists(connection: mysql.Connection, tableName: string): Promise<void> {
  // Check if table exists
  const [tables] = await connection.execute(
    `SELECT COUNT(*) as count FROM information_schema.tables 
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName]
  ) as any[];

  if (tables[0].count === 0) {
    // Table doesn't exist - create it
    // Extract owner from table name (remove _prod_kitchen suffix)
    const owner = tableName.replace(/_prod_kitchen$/, '');
    
    // Create table using the schema
    const createTableSql = `
      CREATE TABLE IF NOT EXISTS \`${tableName}\` (
        -- Primary identification
        \`_id\` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        
        -- Product identification (from grocery-identifier)
        \`product_name\` VARCHAR(500) COMMENT 'Product name identified by AI',
        \`brand\` VARCHAR(255) COMMENT 'Brand name if visible',
        \`variant\` VARCHAR(255) COMMENT 'Product variant (e.g., "Large", "Organic")',
        \`category\` VARCHAR(255) COMMENT 'Product category',
        \`confidence\` DECIMAL(3,2) COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COMMENT 'AI explanation of identification',
        \`product_description\` VARCHAR(500) COMMENT 'One sentence: what this product actually is (for recipe/meal-plan AI context)',
        \`barcode\` VARCHAR(100) COMMENT 'Barcode if detected',
        \`country_guess\` VARCHAR(100) COMMENT 'Country of origin guess',
        
        -- Pricing
        \`estimated_price\` VARCHAR(50) COMMENT 'Estimated price (e.g., "$3.99", "$5.50-$7.00")',
        
        -- Ingredients and nutrition
        \`ingredients\` JSON COMMENT 'Array of ingredient strings',
        \`nutrition_summary\` TEXT COMMENT 'Nutrition information summary',
        \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag',
        \`harmful_ingredients\` JSON COMMENT 'Array of harmful ingredient strings',
        
        -- Similar items and alternatives (stored as JSON)
        \`similar_items\` JSON COMMENT 'Array of similar items with name, brand, reason',
        \`alternatives\` JSON COMMENT 'Array of alternative products with name, brand, reason',
        \`healthier_alternatives\` JSON COMMENT 'Array of healthier alternatives with name, brand, why_healthier, trade_offs',
        
        -- Store availability and pricing (from getStoreAvailability)
        \`store_availability\` JSON COMMENT 'Array of stores with name, price, availability, store_url',
        
        -- Image and action tracking
        \`images\` VARCHAR(1000) COMMENT 'S3 URL to the uploaded image',
        \`s3_key\` VARCHAR(500) COMMENT 'S3 object key for the image',
        \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN' COMMENT 'IN = adding to kitchen, OUT = removing from kitchen',
        \`product_expiration\` DATE COMMENT 'Product expiration date (YYYY-MM-DD)',
        \`product_image_url\` VARCHAR(1000) COMMENT 'S3 URL to generated wireframe/product image',
        \`product_image_key\` VARCHAR(500) COMMENT 'S3 object key for the generated product image',
        
        -- Job tracking
        \`job_id\` VARCHAR(100) COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COMMENT 'User ID from the upload request',
        
        -- Indexes for common queries
        INDEX \`idx_owner\` (\`_owner\`),
        INDEX \`idx_device\` (\`_device\`),
        INDEX \`idx_product_name\` (\`product_name\`),
        INDEX \`idx_category\` (\`category\`),
        INDEX \`idx_action\` (\`action\`),
        INDEX \`idx_created\` (\`_createdDate\`),
        INDEX \`idx_expiration\` (\`product_expiration\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Comprehensive grocery kitchen inventory with full product details'
    `;

    await connection.execute(createTableSql);
    console.log(`[MySQL] Created table ${tableName}`);
  } else {
    // Table exists - check if it has the new columns, add them if missing
    await ensureColumnsExist(connection, tableName);
  }
}

async function ensureColumnsExist(connection: mysql.Connection, tableName: string): Promise<void> {
  const [columns] = await connection.execute(
    `SELECT column_name, column_type FROM information_schema.columns 
     WHERE table_schema = DATABASE() AND table_name = ? AND column_name IN ('product_image_url','upf','harmful_ingredients','product_description')`,
    [tableName]
  ) as any[];

  const columnMap = new Map(
    (columns || []).map((row: { column_name?: string; column_type?: string; COLUMN_NAME?: string; COLUMN_TYPE?: string }) => [
      row.column_name || row.COLUMN_NAME || '',
      row.column_type || row.COLUMN_TYPE || ''
    ]).filter(([name]: [string, string]) => name)
  );
  if (!columnMap.has('product_image_url')) {
    console.log(`[MySQL] Adding product_image_url and product_image_key columns to ${tableName}`);
    await connection.execute(`
      ALTER TABLE \`${tableName}\`
      ADD COLUMN \`product_image_url\` VARCHAR(1000) COMMENT 'S3 URL to generated wireframe/product image' AFTER \`product_expiration\`,
      ADD COLUMN \`product_image_key\` VARCHAR(500) COMMENT 'S3 object key for the generated product image' AFTER \`product_image_url\`
    `);
    console.log(`[MySQL] Added product_image columns to ${tableName}`);
  }

  if (!columnMap.has('upf')) {
    console.log(`[MySQL] Adding upf column to ${tableName}`);
    await connection.execute(`
      ALTER TABLE \`${tableName}\`
      ADD COLUMN \`upf\` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag' AFTER \`nutrition_summary\`
    `);
    console.log(`[MySQL] Added upf column to ${tableName}`);
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
    console.log(`[MySQL] Adding harmful_ingredients column to ${tableName}`);
    await connection.execute(`
      ALTER TABLE \`${tableName}\`
      ADD COLUMN \`harmful_ingredients\` JSON COMMENT 'Array of harmful ingredient strings' AFTER \`upf\`
    `);
    console.log(`[MySQL] Added harmful_ingredients column to ${tableName}`);
  }

  if (!columnMap.has('product_description')) {
    console.log(`[MySQL] Adding product_description column to ${tableName}`);
    await connection.execute(`
      ALTER TABLE \`${tableName}\`
      ADD COLUMN \`product_description\` VARCHAR(500) COMMENT 'One sentence: what this product actually is (for recipe/meal-plan AI context)' AFTER \`explanation\`
    `);
    console.log(`[MySQL] Added product_description column to ${tableName}`);
  }
}
