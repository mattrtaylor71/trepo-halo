// household_membership.mjs (ES module)
import mysql from 'mysql2/promise';

const {
  DB_HOST,
  DB_USER,
  DB_PASSWORD,
  DB_NAME,
} = process.env;

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

// Shared-table (owner_id-keyed) mirror for the families that HAVE one. The per-owner
// copy above heals a joiner's own tables; without this the joiner's shared_* rows
// never appear, so the shared-migration parity monitor flags every household joiner.
// EXCLUDED on purpose: _prod_kitchen (legacy — shared_kitchen is owned by live
// check-in fan-out; mirroring a legacy copy risks dupes), _redemptions / _new_kitchen
// / _new_feed (no shared mirror). Copy is INSERT-if-absent and columns are carried
// AS-IS (action/status verbatim — never normalized), honoring the ghost-resurrection
// rule: never update or resurrect an existing shared row.
// Only families whose per-owner copy keeps `_id` STABLE are safe to mirror by an
// INSERT-if-absent keyed on the shared PK: re-copying then reproduces the same _ids,
// so a re-join is a no-op. shared_recipes/meal_plan are PK owner_id (1 row); metrics/
// dishes/discards are PK (owner_id,_id) and copyTableContents carries their _id AS-IS.
// EXCLUDED: _new_list -> shared_shopping_list. Its copy SKIPS _id (rows get fresh
// auto-increment ids on every copy) while shared_shopping_list is PK _id and dedupes
// logically by household_item_uuid — so INSERT-if-absent by _id would ACCUMULATE
// duplicate shopping rows on a re-join (a ghost-resurrection foot-gun). Shopping also
// self-heals to shared_shopping_list via listHandler's dual-write on the joiner's next
// list mutation. It needs a household_item_uuid-keyed mirror — handled separately.
const SHARED_MIRRORS = [
  { suffix: '_recipes', sharedTable: 'shared_recipes' },
  { suffix: '_meal_plan', sharedTable: 'shared_meal_plan' },
  { suffix: '-metrics', sharedTable: 'shared_metrics' },
  { suffix: '_dishes', sharedTable: 'shared_dishes' },
  { suffix: '_discards', sharedTable: 'shared_discards' },
];

let pool;

function getPool() {
  if (!pool) {
    pool = mysql.createPool({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      database: DB_NAME,
      waitForConnections: true,
      connectionLimit: 5,
      queueLimit: 0,
    });
  }
  return pool;
}

function buildResponse(statusCode, bodyObj) {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,Authorization',
      'Access-Control-Allow-Methods': 'OPTIONS,POST',
    },
    body: JSON.stringify(bodyObj),
  };
}

function quoteId(identifier) {
  return `\`${String(identifier).replace(/`/g, '``')}\``;
}

function validateUserId(userId) {
  return Boolean(userId && /^[0-9a-f-]{36}$/i.test(userId));
}

function decodeAuthFromEvent(event) {
  const headers = event.headers || {};
  const authHeader = headers.Authorization || headers.authorization || '';
  if (!authHeader) {
    return {
      ownerId: null,
      userId: null,
      householdId: null,
      phoneNumber: null,
    };
  }

  const token = authHeader.startsWith('Bearer ')
    ? authHeader.slice(7).trim()
    : authHeader.trim();
  if (!token) {
    return {
      ownerId: null,
      userId: null,
      householdId: null,
      phoneNumber: null,
    };
  }

  try {
    const payload = JSON.parse(Buffer.from(token, 'base64').toString('utf8'));
    const ownerId = payload.owner_id || payload.ownerId || null;
    const userId = payload.user_id || payload.userId || null;
    const phoneNumber = payload.phone_number || payload.phoneNumber || null;
    return {
      ownerId,
      userId,
      householdId: ownerId,
      phoneNumber,
    };
  } catch (err) {
    console.error('Failed to decode auth token:', err);
    return {
      ownerId: null,
      userId: null,
      householdId: null,
      phoneNumber: null,
    };
  }
}

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

  throw new Error('FAILED_TO_GENERATE_HOUSEHOLD_ID');
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

// Mirror a joiner's just-copied per-owner table into its owner_id-keyed shared table.
// Column-generic: inserts owner_id=targetUserId plus every shared column that also
// exists on the per-owner table, copied AS-IS (never normalizes action/status).
// INSERT-IF-ABSENT ONLY (ON DUPLICATE KEY UPDATE owner_id=owner_id is a no-op) so an
// existing shared row is never updated or resurrected (ghost-resurrection rule).
async function mirrorToSharedTable(conn, targetUserId, perOwnerTableName, sharedTableName) {
  if (!(await tableExists(conn, perOwnerTableName)) || !(await tableExists(conn, sharedTableName))) {
    return 0;
  }
  const srcColumns = new Set(await getTableColumns(conn, perOwnerTableName));
  const sharedColumns = await getTableColumns(conn, sharedTableName);
  const insertColumns = ['owner_id'];
  const selectExpressions = ['? AS `owner_id`'];
  const values = [targetUserId];
  for (const column of sharedColumns) {
    if (column === 'owner_id') continue;
    if (!srcColumns.has(column)) continue; // shared-only/computed column — leave to its default
    insertColumns.push(quoteId(column));
    selectExpressions.push(quoteId(column));
  }
  if (insertColumns.length <= 1) {
    return 0;
  }
  const [res] = await conn.query(
    `INSERT INTO ${quoteId(sharedTableName)} (${insertColumns.join(', ')})
     SELECT ${selectExpressions.join(', ')}
     FROM ${quoteId(perOwnerTableName)}
     ON DUPLICATE KEY UPDATE \`owner_id\` = \`owner_id\``,
    values
  );
  return res?.affectedRows || 0;
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

  // Heal the owner_id-keyed shared mirrors for the joiner so it doesn't trip the
  // shared-migration parity monitor. Each is best-effort within the join transaction;
  // a mirror failure for one family must not abort the join or the other mirrors.
  for (const { suffix, sharedTable } of SHARED_MIRRORS) {
    const perOwnerTableName = `${targetUserId}${suffix}`;
    try {
      const rows = await mirrorToSharedTable(conn, targetUserId, perOwnerTableName, sharedTable);
      console.log(JSON.stringify({
        evt: 'household_join_shared_mirror', target: targetUserId,
        family: suffix, shared_table: sharedTable, rows,
      }));
    } catch (mirrorErr) {
      console.error(JSON.stringify({
        evt: 'household_join_shared_mirror_error', target: targetUserId,
        family: suffix, shared_table: sharedTable, error: String(mirrorErr?.message || mirrorErr).slice(0, 300),
      }));
    }
  }
}

async function clearMirroredTables(conn, userId) {
  for (const { tableName } of getMirroredTablesForUser(userId)) {
    await clearTableIfExists(conn, tableName);
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

async function handleJoinHousehold({ userId, targetHouseholdId }) {
  if (!validateUserId(userId)) {
    return buildResponse(400, { message: 'Invalid user_id', code: 'INVALID_USER_ID' });
  }

  if (!targetHouseholdId || !/^\d{4,6}$/.test(targetHouseholdId)) {
    return buildResponse(400, { message: 'Invalid household ID', code: 'INVALID_HOUSEHOLD' });
  }

  const pool = getPool();
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const [userRows] = await conn.query(
      'SELECT user_id, owner_id FROM new_users WHERE user_id = ? LIMIT 1',
      [userId]
    );
    if (userRows.length === 0) {
      const err = new Error('USER_NOT_FOUND');
      err.code = 'USER_NOT_FOUND';
      throw err;
    }

    const currentHouseholdId = userRows[0].owner_id;
    let templateUserId = await findTemplateUserId(conn, targetHouseholdId, userId);

    if (!templateUserId) {
      if (currentHouseholdId === targetHouseholdId) {
        await conn.commit();
        return buildResponse(200, {
          message: 'Already in household',
          user_id: userId,
          previous_household_id: currentHouseholdId,
          new_household_id: targetHouseholdId,
          copied_from_user_id: null,
        });
      }
      const err = new Error('INVALID_HOUSEHOLD');
      err.code = 'INVALID_HOUSEHOLD';
      throw err;
    }

    await conn.query(
      'UPDATE new_users SET owner_id = ?, updated_at = NOW() WHERE user_id = ?',
      [targetHouseholdId, userId]
    );

    await mirrorHouseholdTables(conn, templateUserId, userId);

    await conn.commit();

    return buildResponse(200, {
      message: 'Joined household successfully',
      user_id: userId,
      previous_household_id: currentHouseholdId,
      new_household_id: targetHouseholdId,
      copied_from_user_id: templateUserId,
    });
  } catch (err) {
    await conn.rollback();
    console.error('handleJoinHousehold error:', err);
    if (err.code === 'INVALID_HOUSEHOLD') {
      return buildResponse(400, { message: 'Invalid household ID', code: 'INVALID_HOUSEHOLD' });
    }
    if (err.code === 'USER_NOT_FOUND') {
      return buildResponse(404, { message: 'User not found', code: 'USER_NOT_FOUND' });
    }
    if (err.code === 'INVALID_USER_ID') {
      return buildResponse(400, { message: 'Invalid user_id', code: 'INVALID_USER_ID' });
    }
    return buildResponse(500, { message: 'Internal error', code: 'INTERNAL_ERROR' });
  } finally {
    conn.release();
  }
}

async function handleLeaveHousehold({ userId }) {
  if (!validateUserId(userId)) {
    return buildResponse(400, { message: 'Invalid user_id', code: 'INVALID_USER_ID' });
  }

  const pool = getPool();
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const [userRows] = await conn.query(
      'SELECT user_id, owner_id FROM new_users WHERE user_id = ? LIMIT 1',
      [userId]
    );
    if (userRows.length === 0) {
      const err = new Error('USER_NOT_FOUND');
      err.code = 'USER_NOT_FOUND';
      throw err;
    }

    const previousHouseholdId = userRows[0].owner_id;
    const newHouseholdId = await generateUniqueHouseholdId(conn);

    await conn.query(
      'UPDATE new_users SET owner_id = ?, updated_at = NOW() WHERE user_id = ?',
      [newHouseholdId, userId]
    );

    await clearMirroredTables(conn, userId);

    await conn.commit();

    return buildResponse(200, {
      message: 'Left household successfully',
      user_id: userId,
      previous_household_id: previousHouseholdId,
      new_household_id: newHouseholdId,
    });
  } catch (err) {
    await conn.rollback();
    console.error('handleLeaveHousehold error:', err);
    if (err.code === 'USER_NOT_FOUND') {
      return buildResponse(404, { message: 'User not found', code: 'USER_NOT_FOUND' });
    }
    if (err.message === 'FAILED_TO_GENERATE_HOUSEHOLD_ID') {
      return buildResponse(500, { message: 'Could not generate household ID' });
    }
    if (err.code === 'INVALID_USER_ID') {
      return buildResponse(400, { message: 'Invalid user_id', code: 'INVALID_USER_ID' });
    }
    return buildResponse(500, { message: 'Internal error', code: 'INTERNAL_ERROR' });
  } finally {
    conn.release();
  }
}

export const handler = async (event) => {
  const method = event.requestContext?.http?.method || event.httpMethod || 'GET';

  if (method === 'OPTIONS') {
    return buildResponse(200, { ok: true });
  }

  if (method !== 'POST') {
    return buildResponse(405, { message: 'Method not allowed' });
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

  const { ownerId, userId } = decodeAuthFromEvent(event);
  const resolvedUserId = userId || body.owner_id || body.user_id || ownerId;
  const operation = body.operation;

  if (operation === 'join') {
    return handleJoinHousehold({
      userId: resolvedUserId,
      targetHouseholdId: body.target_household_id,
    });
  }

  if (operation === 'leave') {
    return handleLeaveHousehold({ userId: resolvedUserId });
  }

  return buildResponse(400, {
    message: 'Invalid operation (use "join" or "leave")',
  });
};

