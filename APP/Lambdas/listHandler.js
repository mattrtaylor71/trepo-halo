const mysql = require('mysql2/promise');
const crypto = require('crypto');
const { getHouseholdMemberIds } = require('./householdSync');

const {
  DB_HOST,
  DB_USER,
  DB_PASSWORD,
  DB_NAME,
  DUAL_WRITE_ENABLED,
  LIST_UNION_READ,
  LIST_UNION_READ_OWNERS,
  LIST_STRICT_UUID_MATCH,
} = process.env;

const isDualWriteEnabled = (DUAL_WRITE_ENABLED || 'false').toLowerCase() === 'true';

// Stage 1: union read. When enabled, `view` merges every household member's
// `_new_list` (deduped by household_item_uuid) instead of reading only the
// caller's own table — so a single missed write fan-out is invisible to users.
const isUnionReadEnabled = (LIST_UNION_READ || 'false').toLowerCase() === 'true';
// Optional gradual-rollout allowlist: comma-separated ownerIds (any member id of
// a household). Empty => apply to all households when the flag is on.
const unionReadOwnerAllowlist = (LIST_UNION_READ_OWNERS || '')
  .split(',').map((s) => s.trim()).filter(Boolean);

// Stage 2: when enabled, mutations match ONLY by household_item_uuid (the shared
// id) and drop the per-table `_id` fallback (which can land a delete/check on only
// the actor's copy). Off by default; union read covers any uuid-less rows meanwhile.
const isStrictUuidMatch = (LIST_STRICT_UUID_MATCH || 'false').toLowerCase() === 'true';

function unionReadActiveFor(ownerId, memberIds) {
  if (!isUnionReadEnabled) return false;
  if (!Array.isArray(memberIds) || memberIds.length <= 1) return false; // solo: nothing to union
  if (unionReadOwnerAllowlist.length === 0) return true; // global
  return memberIds.some((id) => unionReadOwnerAllowlist.includes(id))
    || unionReadOwnerAllowlist.includes(ownerId);
}

// Single source of truth for the iOS client item shape (used by both the union
// path and the legacy single-table path so they never drift).
function mapListRow(r) {
  return {
    id: String(r._id),
    product_name: r.product_name,
    action: r.action,
    product_brand: r.product_brand || null,
    product_barcode: r.product_barcode || null,
    store: r.store || null,
    quantity: null,
    itemUUID: r.household_item_uuid || String(r._id),
    sortOrder: r.sort_order != null ? Number(r.sort_order) : null,
  };
}

// Merge-key for union dedup: prefer the shared id; fall back to normalized name
// so uuid-less rows (legacy) still collapse to one logical item across members.
function unionMergeKey(r) {
  if (r.household_item_uuid) return `U:${r.household_item_uuid}`;
  return `N:${String(r.product_name || '').trim().toLowerCase().replace(/\s+/g, ' ')}`;
}

function rowRecency(r) {
  const t = r.updated_at || r._createdDate;
  return t ? new Date(t).getTime() : 0;
}

let pool;

function getPool() {
  if (!pool) {
    pool = mysql.createPool({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      database: DB_NAME,
      waitForConnections: true,
      // Fix #16: cap each warm Lambda container's footprint on database-1. With
      // high invocation volume Lambda spins up many concurrent containers, each
      // holding its own pool — connectionLimit*containers is what pinned RDS at
      // its 636 ceiling. 5 active / 1 idle keeps the per-container footprint low.
      connectionLimit: 5,
      queueLimit: 20,
      // Fail fast instead of hanging on the default (~10s) connect when the RDS
      // instance is at max_connections — a hung connect during a storm pins the
      // 3s Lambda and piles up more half-open sockets. Short connect timeout +
      // idle reaping keeps this function's footprint on database-1 bounded.
      connectTimeout: 5000,
      enableKeepAlive: true,
      keepAliveInitialDelay: 10000,
      maxIdle: 1,
      idleTimeout: 30000,
    });
  }
  return pool;
}

function parseEvent(event) {
  if (event.body) {
    try {
      return JSON.parse(event.body);
    } catch {
      // ignore
    }
  }
  return event;
}

function validateOwner(ownerId) {
  // Accept UUID format (36 chars) or numeric household IDs
  if (!ownerId || typeof ownerId !== 'string' || ownerId.length === 0 || ownerId.length > 36) {
    console.error(`[validateOwner] Rejected ownerId: '${ownerId}' (type: ${typeof ownerId}, length: ${ownerId ? ownerId.length : 0})`);
    throw new Error('Invalid ownerId');
  }
  // Sanitize: only allow alphanumeric + hyphens
  if (!/^[0-9a-zA-Z-]+$/.test(ownerId)) {
    console.error(`[validateOwner] Rejected ownerId with bad chars: '${ownerId}'`);
    throw new Error('Invalid ownerId');
  }
}

function response(statusCode, body) {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,Authorization',
      'Access-Control-Allow-Methods': 'POST,DELETE,OPTIONS',
    },
    body: JSON.stringify(body),
  };
}

exports.handler = async (event) => {
  try {
    const body = parseEvent(event);
    const {
      operation,
      ownerId,
      device,
      product_name,
      product_brand,
      images,
      product_barcode,
      action,   // e.g. "CHECKED"
      id,
      itemUUID,
      store,
    } = body;

    validateOwner(ownerId);
    const pool = getPool();
    const memberIds = await getHouseholdMemberIds(pool, ownerId);

    if (operation === 'view') {
      const tableName = `${ownerId}_new_list`;

      // Stage 1: union read across the household (flagged). Read-only; on any
      // unexpected error it falls through to the legacy single-table path below.
      if (unionReadActiveFor(ownerId, memberIds)) {
        try {
          const merged = new Map();
          for (const memberId of memberIds) {
            const mt = `${memberId}_new_list`;
            let mrows;
            try {
              const conn = await pool.getConnection();
              await conn.execute('SET autocommit = 1');
              [mrows] = await conn.execute(`SELECT * FROM \`${mt}\``);
              conn.release();
            } catch (e) {
              if (e.code === 'ER_NO_SUCH_TABLE') continue; // member never created a list yet
              throw e;
            }
            for (const r of mrows) {
              const key = unionMergeKey(r);
              const existing = merged.get(key);
              // newest write wins (respects the latest action/store/sort toggle)
              if (!existing || rowRecency(existing) < rowRecency(r)) merged.set(key, r);
            }
          }
          const rows = [...merged.values()].sort((a, b) => {
            const sa = a.sort_order, sb = b.sort_order;
            if (sa != null && sb != null && sa !== sb) return sa - sb;
            if (sa != null && sb == null) return -1;
            if (sa == null && sb != null) return 1;
            return new Date(b._createdDate) - new Date(a._createdDate);
          });
          const items = rows.map(mapListRow);
          return response(200, { message: 'List items fetched', items, count: items.length, union: true });
        } catch (err) {
          console.error('[UNION-READ] failed, falling back to single-table:', err.message);
          // fall through to legacy path
        }
      }

      try {
        // Use a dedicated connection with autocommit to avoid stale REPEATABLE READ snapshots
        const conn = await pool.getConnection();
        await conn.execute('SET autocommit = 1');
        let [rows] = await conn.execute(
          `SELECT * FROM \`${tableName}\` ORDER BY \`_createdDate\` DESC`
        );
        conn.release();

        // Lazy backfill: if this user's list is empty but a household member has items, copy them
        if (rows.length === 0 && memberIds.length > 1) {
          for (const memberId of memberIds) {
            if (memberId === ownerId) continue;
            const srcTable = `${memberId}_new_list`;
            try {
              const [srcRows] = await pool.execute(
                `SELECT product_name, product_brand, images, product_barcode, action, store, household_item_uuid, _device, sort_order
                 FROM \`${srcTable}\` ORDER BY \`_createdDate\` ASC`
              );
              if (srcRows.length === 0) continue;
              for (const sr of srcRows) {
                const uuid = sr.household_item_uuid || crypto.randomUUID();
                await pool.execute(
                  `INSERT INTO \`${tableName}\`
                    (_owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid, sort_order)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                  [ownerId, sr._device || 'backfill', sr.product_name, sr.product_brand || null,
                   sr.images || null, sr.product_barcode || null, sr.action || 'ADDED',
                   sr.store || null, uuid, sr.sort_order != null ? sr.sort_order : null]
                );
                // Backfill the source row's household_item_uuid if it was missing
                if (!sr.household_item_uuid) {
                  await pool.execute(
                    `UPDATE \`${srcTable}\` SET household_item_uuid = ? WHERE product_name = ? AND household_item_uuid IS NULL LIMIT 1`,
                    [uuid, sr.product_name]
                  );
                }
              }
              // Re-read after backfill
              [rows] = await pool.execute(
                `SELECT * FROM \`${tableName}\` ORDER BY \`_createdDate\` DESC`
              );
              break; // only backfill from first member with items
            } catch (srcErr) {
              if (srcErr.code !== 'ER_NO_SUCH_TABLE') {
                console.error(`[BACKFILL] Failed to read ${srcTable}:`, srcErr.message);
              }
            }
          }
        }

        // Map _id to id for iOS client compatibility
        const items = rows.map(mapListRow);
        return response(200, {
          message: 'List items fetched',
          items,
          count: items.length,
        });
      } catch (err) {
        if (err.code === 'ER_NO_SUCH_TABLE') {
          return response(200, { message: 'List items fetched', items: [], count: 0 });
        }
        throw err;
      }
    }

    if (operation === 'add') {
      if (!device || !product_name) {
        return response(400, { message: 'Missing required fields for add' });
      }
      const effectiveAction = action || 'ADDED';
      const sharedUUID = crypto.randomUUID();

      const sql = `
        INSERT INTO \`${'${tableName}'}\`
          (_owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `;
      let result;
      for (const memberId of memberIds) {
        const tableName = `${memberId}_new_list`;
        try {
          const [insertResult] = await pool.execute(sql.replace('${tableName}', tableName), [
            memberId,
            device,
            product_name,
            product_brand || null,
            images || null,
            product_barcode || null,
            effectiveAction,
            store || null,
            sharedUUID,
          ]);
          if (memberId === ownerId) result = insertResult;
        } catch (e) {
          // A sibling member's write failing must never abort the others (this is
          // exactly how items got orphaned). Log it loudly so misses stop being
          // silent; only re-throw if the caller's OWN write failed.
          console.error(JSON.stringify({
            evt: 'list_fanout_miss', op: 'add', ownerId, memberId,
            product_name, error: e.code || e.message,
          }));
          if (memberId === ownerId) throw e;
        }
      }

      // Dual-write to shared_list
      if (isDualWriteEnabled) {
        try {
          await pool.execute(
            `INSERT INTO shared_list
              (owner_id, _owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid, _createdDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())`,
            [ownerId, ownerId, device, product_name, product_brand || null, images || null, product_barcode || null, effectiveAction, store || null, sharedUUID]
          );
        } catch (e) {
          console.error('[DUAL-WRITE] shared_list INSERT failed (non-fatal):', e.message);
        }
      }

      // Read back the created item so the iOS client can update optimistically
      let createdItem = null;
      {
        try {
          const tableName = `${ownerId}_new_list`;
          const [rows] = await pool.execute(
            `SELECT * FROM \`${tableName}\` WHERE product_name = ? AND _device = ? ORDER BY _createdDate DESC LIMIT 1`,
            [product_name, device]
          );
          if (rows.length) {
            createdItem = mapListRow(rows[0]);
          }
        } catch (e) {
          console.error('Read-back after add failed (non-fatal):', e.message);
        }
      }

      return response(200, {
        message: 'List item added',
        insertId: result.insertId ?? null,
        items: createdItem ? [createdItem] : [],
      });
    }

    if (operation === 'batch_add') {
      const { items: batchItems } = body;
      if (!Array.isArray(batchItems) || batchItems.length === 0) {
        return response(400, { message: 'Missing items array for batch_add' });
      }

      const addedItems = [];
      for (const batchItem of batchItems) {
        const itemName = batchItem.product_name;
        if (!itemName) continue;
        const itemDevice = batchItem.device || device || 'voice';
        const itemAction = batchItem.action || 'ADDED';
        const itemStore = batchItem.store || null;
        const itemBrand = batchItem.product_brand || null;
        const itemBarcode = batchItem.product_barcode || null;
        const itemImages = batchItem.images || null;
        const sharedUUID = crypto.randomUUID();

        for (const memberId of memberIds) {
          const tableName = `${memberId}_new_list`;
          try {
            await pool.execute(
              `INSERT INTO \`${tableName}\`
                (_owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
              [memberId, itemDevice, itemName, itemBrand, itemImages, itemBarcode, itemAction, itemStore, sharedUUID]
            );
          } catch (err) {
            if (err.code !== 'ER_NO_SUCH_TABLE') {
              // Resilient fan-out: log the miss, keep writing the other members.
              console.error(JSON.stringify({
                evt: 'list_fanout_miss', op: 'batch_add', ownerId, memberId,
                product_name: itemName, error: err.code || err.message,
              }));
            }
          }
        }

        if (isDualWriteEnabled) {
          try {
            await pool.execute(
              `INSERT INTO shared_list
                (owner_id, _owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid, _createdDate)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())`,
              [ownerId, ownerId, batchItems[0]?.device || device || 'voice', itemName, itemBrand, itemImages, itemBarcode, itemAction, itemStore, sharedUUID]
            );
          } catch (e) {
            console.error('[DUAL-WRITE] shared_list batch INSERT failed (non-fatal):', e.message);
          }
        }

        addedItems.push({ product_name: itemName, store: itemStore, itemUUID: sharedUUID });
      }

      return response(200, {
        message: `Added ${addedItems.length} items`,
        count: addedItems.length,
        items: addedItems,
      });
    }

    if (operation === 'remove') {
      // Priority: itemUUID > id > device+barcode
      const removeKey = itemUUID || id;

      if (removeKey) {
        let affectedRows = 0;
        for (const memberId of memberIds) {
          const tableName = `${memberId}_new_list`;
          try {
            // Match by the shared id first. The `_id` fallback is per-table and
            // meaningless across mirrors — under strict mode we skip it so a delete
            // can never land on only one member's copy.
            let [result] = await pool.execute(
              `DELETE FROM \`${tableName}\` WHERE household_item_uuid = ? LIMIT 1`, [removeKey]);
            if (!isStrictUuidMatch && !result.affectedRows) {
              [result] = await pool.execute(
                `DELETE FROM \`${tableName}\` WHERE _id = ? LIMIT 1`, [removeKey]);
            }
            affectedRows += result.affectedRows || 0;
          } catch (err) {
            if (err.code !== 'ER_NO_SUCH_TABLE') {
              console.error(JSON.stringify({
                evt: 'list_fanout_miss', op: 'remove', ownerId, memberId,
                removeKey, error: err.code || err.message,
              }));
            }
          }
        }
        if (isDualWriteEnabled) {
          try {
            await pool.execute('DELETE FROM shared_list WHERE household_item_uuid = ? LIMIT 1', [removeKey]);
            await pool.execute('DELETE FROM shared_list WHERE _id = ? LIMIT 1', [removeKey]);
          } catch (e) {
            console.error('[DUAL-WRITE] shared_list DELETE failed (non-fatal):', e.message);
          }
        }
        return response(200, {
          message: 'List item removed',
          affectedRows,
        });
      } else {
        if (!device || !product_barcode) {
          return response(400, {
            message: 'Provide itemUUID, id, or (device + product_barcode) to remove',
          });
        }
        let affectedRows = 0;
        for (const memberId of memberIds) {
          const tableName = `${memberId}_new_list`;
          const [result] = await pool.execute(
            `
              DELETE FROM \`${tableName}\`
              WHERE _owner = ?
                AND _device = ?
                AND product_barcode = ?
              LIMIT 1
            `,
            [memberId, device, product_barcode]
          );
          affectedRows += result.affectedRows || 0;
        }
        // Dual-write: delete from shared_list
        if (isDualWriteEnabled) {
          try {
            await pool.execute(
              'DELETE FROM shared_list WHERE owner_id = ? AND _device = ? AND product_barcode = ? LIMIT 1',
              [ownerId, device, product_barcode]
            );
          } catch (e) {
            console.error('[DUAL-WRITE] shared_list DELETE by barcode failed (non-fatal):', e.message);
          }
        }
        return response(200, {
          message: 'List item removed',
          affectedRows,
        });
      }
    }

    if (operation === 'set_action') {
      const { checked, itemUUID } = body;
      if (!itemUUID) {
        return response(400, { message: 'Missing itemUUID for set_action' });
      }
      const newAction = checked ? 'CHECKED' : 'ADDED';
      let affectedRows = 0;
      for (const memberId of memberIds) {
        const tableName = `${memberId}_new_list`;
        try {
          // Match by the shared id first; skip the per-table _id fallback in
          // strict mode so a check/uncheck never applies to only one copy.
          let [result] = await pool.execute(
            `UPDATE \`${tableName}\` SET action = ?, updated_at = NOW() WHERE household_item_uuid = ?`,
            [newAction, itemUUID]
          );
          if (!isStrictUuidMatch && !result.affectedRows) {
            [result] = await pool.execute(
              `UPDATE \`${tableName}\` SET action = ?, updated_at = NOW() WHERE _id = ?`,
              [newAction, itemUUID]
            );
          }
          affectedRows += result.affectedRows || 0;
        } catch (err) {
          if (err.code !== 'ER_NO_SUCH_TABLE') throw err;
        }
      }
      if (isDualWriteEnabled) {
        try {
          await pool.execute(
            'UPDATE shared_list SET action = ?, updated_at = NOW() WHERE household_item_uuid = ?',
            [newAction, itemUUID]
          );
        } catch (e) {
          console.error('[DUAL-WRITE] shared_list set_action failed (non-fatal):', e.message);
        }
      }
      return response(affectedRows > 0 ? 200 : 404, {
        message: affectedRows > 0 ? 'Item updated' : 'Item not found',
        affectedRows,
      });
    }

    if (operation === 'update_item') {
      const { itemUUID, product_name: newName, checked: newChecked, store: newStore } = body;
      if (!itemUUID) {
        return response(400, { message: 'Missing itemUUID for update_item' });
      }
      const updates = [];
      const params = [];
      if (newName !== undefined) { updates.push('product_name = ?'); params.push(newName); }
      if (newChecked !== undefined) { updates.push('action = ?'); params.push(newChecked ? 'CHECKED' : 'ADDED'); }
      if (newStore !== undefined) { updates.push('store = ?'); params.push(newStore || null); }
      if (updates.length === 0) {
        return response(200, { message: 'Nothing to update' });
      }
      updates.push('updated_at = NOW()');
      const setClause = updates.join(', ');
      let affectedRows = 0;
      for (const memberId of memberIds) {
        const tableName = `${memberId}_new_list`;
        try {
          let [result] = await pool.execute(
            `UPDATE \`${tableName}\` SET ${setClause} WHERE household_item_uuid = ?`,
            [...params, itemUUID]
          );
          if (!isStrictUuidMatch && !result.affectedRows) {
            [result] = await pool.execute(
              `UPDATE \`${tableName}\` SET ${setClause} WHERE _id = ?`,
              [...params, itemUUID]
            );
          }
          affectedRows += result.affectedRows || 0;
        } catch (err) {
          if (err.code !== 'ER_NO_SUCH_TABLE') throw err;
        }
      }
      if (isDualWriteEnabled) {
        try {
          await pool.execute(
            `UPDATE shared_list SET ${setClause} WHERE household_item_uuid = ?`,
            [...params, itemUUID]
          );
        } catch (e) {
          console.error('[DUAL-WRITE] shared_list update_item failed (non-fatal):', e.message);
        }
      }
      return response(affectedRows > 0 ? 200 : 404, {
        message: affectedRows > 0 ? 'Item updated' : 'Item not found',
        affectedRows,
      });
    }

    if (operation === 'reorder') {
      const { orderedItems } = body;
      if (!Array.isArray(orderedItems)) {
        return response(400, { message: 'Missing orderedItems array for reorder' });
      }
      for (const memberId of memberIds) {
        const tableName = `${memberId}_new_list`;
        for (const { itemUUID, sortOrder } of orderedItems) {
          if (!itemUUID) continue;
          try {
            // Match by the shared id first; skip the per-table _id fallback in
            // strict mode so reorder stays consistent across all copies.
            let [result] = await pool.execute(
              `UPDATE \`${tableName}\` SET sort_order = ?, updated_at = NOW() WHERE household_item_uuid = ?`,
              [sortOrder, itemUUID]
            );
            if (!isStrictUuidMatch && !result.affectedRows) {
              await pool.execute(
                `UPDATE \`${tableName}\` SET sort_order = ?, updated_at = NOW() WHERE _id = ?`,
                [sortOrder, itemUUID]
              );
            }
          } catch (err) {
            if (err.code !== 'ER_NO_SUCH_TABLE') throw err;
          }
        }
      }
      return response(200, { message: 'Reorder applied', count: orderedItems.length });
    }

    return response(400, { message: 'Invalid operation (use view/add/remove/set_action/update_item/reorder)' });
  } catch (err) {
    console.error(err);
    return response(500, { message: 'Internal server error', error: err.message });
  }
};
