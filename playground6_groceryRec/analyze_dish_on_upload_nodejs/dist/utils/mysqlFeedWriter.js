"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.writeFeedEvent = writeFeedEvent;
const promise_1 = __importDefault(require("mysql2/promise"));
const householdSync_1 = require("./householdSync");
async function writeFeedEvent(record) {
    const { DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME } = process.env;
    if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
        throw new Error("Missing required database environment variables");
    }
    const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;
    const connection = await promise_1.default.createConnection({
        host: DB_HOST,
        port,
        user: DB_USER,
        password: DB_PASS,
        database: DB_NAME,
        charset: "utf8mb4",
    });
    try {
        // Feed events for dishes are user-specific — no household sync
        const memberIds = [record.owner];
        const entryId = (0, householdSync_1.stableHouseholdRowId)("feed_events", record.job_id || "", record.event_type || "");
        const metadataJson = record.metadata ? JSON.stringify(record.metadata) : null;
        const sqlForTable = (tableName) => `
      INSERT INTO \`${tableName}\`
      (
        _id, _owner, _device, _createdDate,
        event_type, action, title, brand,
        image_url, product_image_url,
        job_id, user_id, source_table, metadata
      )
      VALUES (?, ?, ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
        for (const memberId of memberIds) {
            const tableName = `${memberId.replace(/[^a-zA-Z0-9_-]/g, "")}_feed_events`;
            await ensureTableExists(connection, tableName);
            const [existingRows] = await connection.execute(`SELECT _id FROM \`${tableName}\` WHERE _id = ? LIMIT 1`, [entryId]);
            if (Array.isArray(existingRows) && existingRows.length > 0) {
                continue;
            }
            const values = [
                entryId,
                memberId,
                record.device_id,
                record.event_type,
                record.action || null,
                record.title || null,
                record.brand || null,
                record.image_url || null,
                record.product_image_url || null,
                record.job_id || null,
                record.user_id || null,
                record.source_table || null,
                metadataJson,
            ];
            await connection.execute(sqlForTable(tableName), values);
            console.log(`[MySQL] Inserted feed event job_id=${record.job_id} event_type=${record.event_type} into ${tableName}`);
        }
    }
    catch (error) {
        console.error(`[MySQL] Error writing household feed event:`, error);
        throw error;
    }
    finally {
        await connection.end();
    }
}
async function ensureTableExists(connection, tableName) {
    const [tables] = (await connection.execute(`SELECT COUNT(*) as count FROM information_schema.tables
     WHERE table_schema = DATABASE() AND table_name = ?`, [tableName]));
    if (tables[0].count === 0) {
        const createSql = `
      CREATE TABLE IF NOT EXISTS \`${tableName}\` (
        \`_id\` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
        \`_owner\` VARCHAR(36) NOT NULL COMMENT 'Owner UUID',
        \`_device\` VARCHAR(255) NOT NULL COMMENT 'Device ID',
        \`_createdDate\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the record was created',
        \`event_type\` VARCHAR(32) NOT NULL COMMENT 'checkin | checkout | discard | dish',
        \`action\` VARCHAR(8) NULL COMMENT 'IN/OUT or null',
        \`title\` VARCHAR(500) COMMENT 'Product or dish name',
        \`brand\` VARCHAR(255) COMMENT 'Brand name',
        \`image_url\` VARCHAR(1000) COMMENT 'Primary image url',
        \`product_image_url\` VARCHAR(1000) COMMENT 'Generated/product image url',
        \`job_id\` VARCHAR(100) COMMENT 'Job ID from DynamoDB',
        \`user_id\` VARCHAR(255) COMMENT 'User ID from the upload request',
        \`source_table\` VARCHAR(255) COMMENT 'Source table name',
        \`metadata\` JSON COMMENT 'Extra metadata',
        INDEX \`idx_owner\` (\`_owner\`),
        INDEX \`idx_created\` (\`_createdDate\`),
        INDEX \`idx_job\` (\`job_id\`),
        INDEX \`idx_event\` (\`event_type\`)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Unified activity feed events (per owner)';
    `;
        await connection.execute(createSql);
        console.log(`[MySQL] Created table ${tableName}`);
    }
}
//# sourceMappingURL=mysqlFeedWriter.js.map