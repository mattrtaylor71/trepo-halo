"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.writeToDishesTable = writeToDishesTable;
exports.getDishById = getDishById;
exports.updateDishRow = updateDishRow;
const promise_1 = __importDefault(require("mysql2/promise"));
const householdSync_1 = require("./householdSync");
const DUAL_WRITE_ENABLED = (process.env.DUAL_WRITE_ENABLED || 'false').toLowerCase() === 'true';
async function dualWriteSharedDishes(connection, owner, entryId, record, dishName, confidence, ingredientsJson, allergensJson) {
    if (!DUAL_WRITE_ENABLED)
        return;
    try {
        const { dish, nutritionData, device_id, user_id, job_id, action, s3_key, image_url, resized_image_url, resized_image_key, dish_image_url, dish_image_key } = record;
        await connection.execute(`INSERT INTO \`shared_dishes\` (
        _id, owner_id, _owner, _device, dish_name, confidence, explanation,
        serving_size, calories, total_fat, saturated_fat, trans_fat, cholesterol, sodium,
        total_carbohydrates, dietary_fiber, sugars, protein,
        vitamin_a, vitamin_c, calcium, iron,
        ingredients, allergens, images, s3_key, action,
        resized_image_url, resized_image_key, dish_image_url, dish_image_key,
        job_id, user_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON DUPLICATE KEY UPDATE
        dish_name = COALESCE(VALUES(dish_name), dish_name),
        confidence = COALESCE(VALUES(confidence), confidence),
        calories = COALESCE(VALUES(calories), calories),
        protein = COALESCE(VALUES(protein), protein),
        _updatedDate = NOW()`, [
            entryId, owner, owner, device_id,
            dishName, confidence,
            dish.explanation || nutritionData.explanation || null,
            nutritionData.serving_size || null,
            nutritionData.calories || null, nutritionData.total_fat || null,
            nutritionData.saturated_fat || null, nutritionData.trans_fat || null,
            nutritionData.cholesterol || null, nutritionData.sodium || null,
            nutritionData.total_carbohydrates || null, nutritionData.dietary_fiber || null,
            nutritionData.sugars || null, nutritionData.protein || null,
            nutritionData.vitamin_a || null, nutritionData.vitamin_c || null,
            nutritionData.calcium || null, nutritionData.iron || null,
            ingredientsJson, allergensJson,
            image_url, s3_key, action,
            resized_image_url || null, resized_image_key || null,
            dish_image_url || null, dish_image_key || null,
            job_id, user_id,
        ]);
    }
    catch (err) {
        console.error('[DUAL-WRITE] shared_dishes write failed (non-fatal):', err?.message || err);
    }
}
async function writeToDishesTable(record) {
    const { DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME, } = process.env;
    if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
        throw new Error('Missing required database environment variables');
    }
    const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;
    // Create connection
    const connection = await promise_1.default.createConnection({
        host: DB_HOST,
        port: port,
        user: DB_USER,
        password: DB_PASS,
        database: DB_NAME,
        charset: 'utf8mb4',
        connectTimeout: 10000,
    });
    try {
        // Bound metadata-lock waits. MySQL's default lock_wait_timeout is 1 year, so a
        // CREATE TABLE / ALTER TABLE (schema ensure) that contends with a concurrent dish
        // write's DDL blocks until this Lambda hits its 300s ceiling and is killed with no
        // error — a silent hang that never marks the job failed. Capping it here converts
        // that into a fast, logged, retryable error. (Best-effort: ignore if it can't be set.)
        try {
            await connection.query('SET SESSION lock_wait_timeout = 15, innodb_lock_wait_timeout = 15');
        }
        catch (e) {
            console.warn('[mysqlDishWriter] could not set lock timeouts (non-fatal):', e?.message || e);
        }
        // Dishes are user-specific — no household sync
        const memberIds = [record.owner];
        const entryId = (0, householdSync_1.stableHouseholdRowId)('dishes', record.job_id);
        const { dish, nutritionData, device_id, user_id, job_id, action, s3_key, image_url, resized_image_url, resized_image_key, dish_image_url, dish_image_key, } = record;
        // Convert arrays/objects to JSON strings for MySQL
        const ingredientsJson = JSON.stringify(nutritionData.ingredients || []);
        const allergensJson = JSON.stringify(nutritionData.allergens || []);
        // Use dish confidence or nutrition confidence (prefer dish confidence)
        const confidence = dish.confidence != null
            ? Math.max(0, Math.min(1, dish.confidence))
            : (nutritionData.confidence != null
                ? Math.max(0, Math.min(1, nutritionData.confidence))
                : null);
        // Combine dish name from dish identification or nutrition data
        const dishName = dish.dish_name || nutritionData.dish_name || null;
        // Insert query
        const sqlForTable = (tableName) => `
      INSERT INTO \`${tableName}\`
      (
        _id, _owner, _device, _createdDate,
        dish_name, confidence, explanation,
        serving_size,
        calories, total_fat, saturated_fat, trans_fat, cholesterol, sodium,
        total_carbohydrates, dietary_fiber, sugars, protein,
        vitamin_a, vitamin_c, calcium, iron,
        ingredients, allergens,
        images, s3_key, action,
        resized_image_url, resized_image_key,
        dish_image_url, dish_image_key,
        job_id, user_id
      )
      VALUES (?, ?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
        for (const memberId of memberIds) {
            const tableName = `${memberId.replace(/[^a-zA-Z0-9_-]/g, '')}_dishes`;
            await ensureTableExists(connection, tableName);
            const [existingRows] = await connection.execute(`SELECT _id FROM \`${tableName}\` WHERE _id = ? LIMIT 1`, [entryId]);
            if (Array.isArray(existingRows) && existingRows.length > 0) {
                // Update existing preliminary record with final deep-dive data
                await connection.execute(`UPDATE \`${tableName}\` SET
            dish_name = COALESCE(?, dish_name),
            confidence = COALESCE(?, confidence),
            explanation = COALESCE(?, explanation),
            serving_size = COALESCE(?, serving_size),
            calories = COALESCE(?, calories),
            total_fat = COALESCE(?, total_fat),
            saturated_fat = COALESCE(?, saturated_fat),
            trans_fat = COALESCE(?, trans_fat),
            cholesterol = COALESCE(?, cholesterol),
            sodium = COALESCE(?, sodium),
            total_carbohydrates = COALESCE(?, total_carbohydrates),
            dietary_fiber = COALESCE(?, dietary_fiber),
            sugars = COALESCE(?, sugars),
            protein = COALESCE(?, protein),
            ingredients = COALESCE(?, ingredients),
            allergens = COALESCE(?, allergens),
            resized_image_url = COALESCE(?, resized_image_url),
            resized_image_key = COALESCE(?, resized_image_key),
            dish_image_url = COALESCE(?, dish_image_url),
            dish_image_key = COALESCE(?, dish_image_key)
          WHERE _id = ?`, [
                    dishName, confidence,
                    dish.explanation || nutritionData.explanation || null,
                    nutritionData.serving_size || null,
                    nutritionData.calories || null, nutritionData.total_fat || null,
                    nutritionData.saturated_fat || null, nutritionData.trans_fat || null,
                    nutritionData.cholesterol || null, nutritionData.sodium || null,
                    nutritionData.total_carbohydrates || null, nutritionData.dietary_fiber || null,
                    nutritionData.sugars || null, nutritionData.protein || null,
                    ingredientsJson, allergensJson,
                    resized_image_url || null, resized_image_key || null,
                    dish_image_url || null, dish_image_key || null,
                    entryId,
                ]);
                console.log(`[MySQL] Updated existing dish record in ${tableName}, entry_id: ${entryId}`);
                continue;
            }
            const values = [
                entryId,
                memberId,
                device_id,
                dishName,
                confidence,
                dish.explanation || nutritionData.explanation || null,
                nutritionData.serving_size || null,
                nutritionData.calories || null,
                nutritionData.total_fat || null,
                nutritionData.saturated_fat || null,
                nutritionData.trans_fat || null,
                nutritionData.cholesterol || null,
                nutritionData.sodium || null,
                nutritionData.total_carbohydrates || null,
                nutritionData.dietary_fiber || null,
                nutritionData.sugars || null,
                nutritionData.protein || null,
                nutritionData.vitamin_a || null,
                nutritionData.vitamin_c || null,
                nutritionData.calcium || null,
                nutritionData.iron || null,
                ingredientsJson,
                allergensJson,
                image_url,
                s3_key,
                action,
                resized_image_url || null,
                resized_image_key || null,
                dish_image_url || null,
                dish_image_key || null,
                job_id,
                user_id,
            ];
            await connection.execute(sqlForTable(tableName), values);
            console.log(`[MySQL] Successfully inserted dish record into ${tableName}, entry_id: ${entryId}`);
        }
        // Dual-write to shared table
        await dualWriteSharedDishes(connection, record.owner, entryId, record, dishName, confidence, ingredientsJson, allergensJson);
    }
    catch (error) {
        console.error('[MySQL] Error writing household dishes:', error);
        throw error;
    }
    finally {
        await connection.end();
    }
}
async function ensureTableExists(connection, tableName) {
    // Check if table exists
    const [tables] = await connection.execute(`SELECT COUNT(*) as count FROM information_schema.tables 
     WHERE table_schema = DATABASE() AND table_name = ?`, [tableName]);
    if (tables[0].count === 0) {
        // Table doesn't exist - create it
        // Extract owner from table name (remove _dishes suffix)
        const owner = tableName.replace(/_dishes$/, '');
        // Create table using the schema
        const createTableSql = `
      CREATE TABLE IF NOT EXISTS \`${tableName}\` (
        -- Primary identification
        \`_id\` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) NOT NULL COMMENT 'Owner UUID (matches table name prefix)',
        \`_device\` VARCHAR(255) NOT NULL COMMENT 'Device ID that captured the image',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`_updatedDate\` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT 'When the record was last updated',
        
        -- Dish identification (from dish recognition)
        \`dish_name\` VARCHAR(500) COMMENT 'Dish name identified by AI',
        \`confidence\` DECIMAL(3,2) COMMENT 'AI confidence score (0.00 to 1.00)',
        \`explanation\` TEXT COMMENT 'AI explanation of dish identification',
        
        -- Serving information
        \`serving_size\` VARCHAR(100) COMMENT 'Estimated serving size (e.g., "1 bowl", "1 plate", "200g")',
        
        -- Nutrition data (from nutrition-extractor)
        \`calories\` DECIMAL(10,2) COMMENT 'Calories per serving',
        \`total_fat\` DECIMAL(10,2) COMMENT 'Total fat in grams',
        \`saturated_fat\` DECIMAL(10,2) COMMENT 'Saturated fat in grams',
        \`trans_fat\` DECIMAL(10,2) COMMENT 'Trans fat in grams',
        \`cholesterol\` DECIMAL(10,2) COMMENT 'Cholesterol in milligrams',
        \`sodium\` DECIMAL(10,2) COMMENT 'Sodium in milligrams',
        \`total_carbohydrates\` DECIMAL(10,2) COMMENT 'Total carbohydrates in grams',
        \`dietary_fiber\` DECIMAL(10,2) COMMENT 'Dietary fiber in grams',
        \`sugars\` DECIMAL(10,2) COMMENT 'Sugars in grams',
        \`protein\` DECIMAL(10,2) COMMENT 'Protein in grams',
        
        -- Vitamins and minerals
        \`vitamin_a\` DECIMAL(10,2) COMMENT 'Vitamin A in IU or mcg',
        \`vitamin_c\` DECIMAL(10,2) COMMENT 'Vitamin C in milligrams',
        \`calcium\` DECIMAL(10,2) COMMENT 'Calcium in milligrams',
        \`iron\` DECIMAL(10,2) COMMENT 'Iron in milligrams',
        
        -- Ingredients and allergens
        \`ingredients\` JSON COMMENT 'Array of ingredient strings',
        \`allergens\` JSON COMMENT 'Array of allergen strings (e.g., ["dairy", "nuts", "gluten"])',
        
        -- Image and action tracking
        \`images\` VARCHAR(1000) COMMENT 'S3 URL to the uploaded dish image',
        \`s3_key\` VARCHAR(500) COMMENT 'S3 object key for the uploaded image',
        \`action\` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN' COMMENT 'IN = adding dish, OUT = removing/consumed',
        \`resized_image_url\` VARCHAR(1000) COMMENT 'S3 URL to resized app-friendly original image',
        \`resized_image_key\` VARCHAR(500) COMMENT 'S3 object key for resized app-friendly original image',
        \`dish_image_url\` VARCHAR(1000) COMMENT 'S3 URL to generated dish image',
        \`dish_image_key\` VARCHAR(500) COMMENT 'S3 object key for the generated dish image',
        
        -- Job tracking
        \`job_id\` VARCHAR(100) COMMENT 'Job ID from DynamoDB for tracking',
        \`user_id\` VARCHAR(255) COMMENT 'User ID from the upload request',
        
        -- Indexes for common queries
        UNIQUE KEY \`uq_job_owner\` (\`job_id\`, \`_owner\`) COMMENT 'One MySQL row per upload (job_id)',
        INDEX \`idx_owner\` (\`_owner\`),
        INDEX \`idx_device\` (\`_device\`),
        INDEX \`idx_dish_name\` (\`dish_name\`),
        INDEX \`idx_action\` (\`action\`),
        INDEX \`idx_created\` (\`_createdDate\`),
        INDEX \`idx_calories\` (\`calories\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Comprehensive dish tracking with nutrition analysis'
    `;
        await connection.execute(createTableSql);
        console.log(`[MySQL] Created table ${tableName}`);
    }
    else {
        // Table exists - check if it has the new columns, add them if missing
        await ensureColumnsExist(connection, tableName);
    }
}
async function ensureColumnsExist(connection, tableName) {
    const [columns] = await connection.execute(`SELECT column_name FROM information_schema.columns 
     WHERE table_schema = DATABASE() AND table_name = ? AND column_name IN ('dish_image_url', 'dish_image_key', 'resized_image_url', 'resized_image_key')`, [tableName]);
    const columnSet = new Set((columns || []).map((row) => row.column_name || row.COLUMN_NAME || ''));
    // Add each missing column as its OWN statement, tolerating ER_DUP_FIELDNAME.
    // Two dish writes for the same owner can race: both read information_schema,
    // both see a column missing, both build the ALTER — one wins, the other used to
    // throw "Duplicate column name" and fail the whole dish write. A duplicate just
    // means a concurrent write already added it (the desired end state), so we skip
    // it. Per-column (not one combined ALTER) so a duplicate on one column never
    // blocks adding the others.
    const columnDefs = [
        ['resized_image_url', "ADD COLUMN `resized_image_url` VARCHAR(1000) COMMENT 'S3 URL to resized app-friendly original image' AFTER `action`"],
        ['resized_image_key', "ADD COLUMN `resized_image_key` VARCHAR(500) COMMENT 'S3 object key for resized app-friendly original image' AFTER `resized_image_url`"],
        ['dish_image_url', "ADD COLUMN `dish_image_url` VARCHAR(1000) COMMENT 'S3 URL to generated dish image' AFTER `resized_image_key`"],
        ['dish_image_key', "ADD COLUMN `dish_image_key` VARCHAR(500) COMMENT 'S3 object key for the generated dish image' AFTER `dish_image_url`"],
    ];
    for (const [name, addClause] of columnDefs) {
        if (columnSet.has(name))
            continue;
        try {
            await connection.execute(`ALTER TABLE \`${tableName}\` ${addClause}`);
        }
        catch (err) {
            if (err && (err.errno === 1060 || err.code === 'ER_DUP_FIELDNAME')) {
                continue; // concurrent writer already added it — fine
            }
            throw err;
        }
    }
}
async function getDishById(connection, tableName, itemId) {
    const [rows] = await connection.execute(`SELECT * FROM \`${tableName}\` WHERE _id = ? LIMIT 1`, [itemId]);
    if (Array.isArray(rows) && rows.length > 0) {
        return rows[0];
    }
    return null;
}
async function updateDishRow(connection, tableName, itemId, fields) {
    const updates = [];
    const values = [];
    if (fields.dish_name !== undefined) {
        updates.push('dish_name = ?');
        values.push(fields.dish_name);
    }
    if (fields.confidence !== undefined) {
        updates.push('confidence = ?');
        values.push(fields.confidence);
    }
    if (fields.explanation !== undefined) {
        updates.push('explanation = ?');
        values.push(fields.explanation);
    }
    if (fields.serving_size !== undefined) {
        updates.push('serving_size = ?');
        values.push(fields.serving_size);
    }
    if (fields.calories !== undefined) {
        updates.push('calories = ?');
        values.push(fields.calories);
    }
    if (fields.total_fat !== undefined) {
        updates.push('total_fat = ?');
        values.push(fields.total_fat);
    }
    if (fields.saturated_fat !== undefined) {
        updates.push('saturated_fat = ?');
        values.push(fields.saturated_fat);
    }
    if (fields.trans_fat !== undefined) {
        updates.push('trans_fat = ?');
        values.push(fields.trans_fat);
    }
    if (fields.cholesterol !== undefined) {
        updates.push('cholesterol = ?');
        values.push(fields.cholesterol);
    }
    if (fields.sodium !== undefined) {
        updates.push('sodium = ?');
        values.push(fields.sodium);
    }
    if (fields.total_carbohydrates !== undefined) {
        updates.push('total_carbohydrates = ?');
        values.push(fields.total_carbohydrates);
    }
    if (fields.dietary_fiber !== undefined) {
        updates.push('dietary_fiber = ?');
        values.push(fields.dietary_fiber);
    }
    if (fields.sugars !== undefined) {
        updates.push('sugars = ?');
        values.push(fields.sugars);
    }
    if (fields.protein !== undefined) {
        updates.push('protein = ?');
        values.push(fields.protein);
    }
    if (fields.vitamin_a !== undefined) {
        updates.push('vitamin_a = ?');
        values.push(fields.vitamin_a);
    }
    if (fields.vitamin_c !== undefined) {
        updates.push('vitamin_c = ?');
        values.push(fields.vitamin_c);
    }
    if (fields.calcium !== undefined) {
        updates.push('calcium = ?');
        values.push(fields.calcium);
    }
    if (fields.iron !== undefined) {
        updates.push('iron = ?');
        values.push(fields.iron);
    }
    if (fields.ingredients !== undefined) {
        updates.push('ingredients = ?');
        values.push(fields.ingredients);
    }
    if (fields.allergens !== undefined) {
        updates.push('allergens = ?');
        values.push(fields.allergens);
    }
    if (fields.resized_image_url !== undefined) {
        updates.push('resized_image_url = ?');
        values.push(fields.resized_image_url);
    }
    if (fields.resized_image_key !== undefined) {
        updates.push('resized_image_key = ?');
        values.push(fields.resized_image_key);
    }
    if (fields.dish_image_url !== undefined) {
        updates.push('dish_image_url = ?');
        values.push(fields.dish_image_url);
    }
    if (fields.dish_image_key !== undefined) {
        updates.push('dish_image_key = ?');
        values.push(fields.dish_image_key);
    }
    if (updates.length === 0)
        return;
    updates.push('_updatedDate = NOW()');
    values.push(itemId);
    const sql = `UPDATE \`${tableName}\` SET ${updates.join(', ')} WHERE _id = ?`;
    await connection.execute(sql, values);
    console.log(`[MySQL] Updated dish ${itemId} in ${tableName}`);
}
//# sourceMappingURL=mysqlDishWriter.js.map