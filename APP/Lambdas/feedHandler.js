const mysql = require('mysql2/promise');

const {
  DB_HOST,
  DB_USER,
  DB_PASSWORD,
  DB_NAME,
} = process.env;

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

function parseEvent(event) {
  if (event.body) {
    try {
      return JSON.parse(event.body);
    } catch {
      // fall through
    }
  }
  return event; // direct invoke
}

function validateOwner(ownerId) {
  if (!ownerId || !/^[0-9a-f-]{36}$/i.test(ownerId)) {
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
      action,
      id,
    } = body;

    validateOwner(ownerId);
    const tableName = `${ownerId}_new_feed`;
    const pool = getPool();

    // -------- VIEW --------
    if (operation === 'view') {
      if (!device) {
        return response(400, { message: 'Missing device for view' });
      }

      const sql = `
        SELECT
          _id AS id,
          _owner,
          _device,
          product_name,
          product_brand,
          images,
          product_barcode,
          action
        FROM \`${tableName}\`
        WHERE _owner = ? AND _device = ?
        ORDER BY _id DESC
      `;

      const [rows] = await pool.execute(sql, [ownerId, device]);

      return response(200, {
        message: 'Feed items fetched',
        items: rows,
      });
    }

    // -------- ADD --------
    if (operation === 'add') {
      if (!device || !product_name || !action) {
        return response(400, { message: 'Missing required fields for add' });
      }

      // WRITE-STOP (shared-table migration): `_new_feed` is a legacy write-only table
      // with zero readers, being retired. Neuter the write (no INSERT) while keeping
      // the response contract byte-identical, so stale-app-version clients still hitting
      // POST /v1/feed get a graceful 200 no-op. The go-forward iOS app does not call
      // this route. Table archive/drop is a separate (Matt-owned) destructive step.
      console.log(JSON.stringify({ evt: 'feed_write_neutered', op: 'add', ownerId }));

      return response(200, {
        message: 'Feed item added',
        insertId: null,
      });
    }

    // -------- REMOVE --------
    if (operation === 'remove') {
      // Preserve the input-validation contract (400 when neither id nor
      // device+product_barcode is provided) …
      if (!id && (!device || !product_barcode)) {
        return response(400, {
          message: 'Provide either id or (device + product_barcode) to remove',
        });
      }

      // … but neuter the DELETE write to `_new_feed` (see ADD note). Byte-identical
      // 200 shape; affectedRows: 0 since nothing is mutated.
      console.log(JSON.stringify({ evt: 'feed_write_neutered', op: 'remove', ownerId }));
      return response(200, {
        message: 'Feed item removed',
        affectedRows: 0,
      });
    }

    return response(400, { message: 'Invalid operation (use add/remove/view)' });
  } catch (err) {
    console.error(err);
    return response(500, { message: 'Internal server error', error: err.message });
  }
};
