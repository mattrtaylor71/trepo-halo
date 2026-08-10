const mysql = require('mysql2/promise');
const crypto = require('crypto');
const { getHouseholdMemberIds } = require('./householdSync');
// AWS SDK v3 is bundled in the nodejs22.x runtime — no need to ship it in the zip.
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');

// ---- Aisle categorization (shopping-list grouping) ----------------------------
// Canonical, EXACT lowercase enum. Consumers group by exact string match, so we
// normalize case-insensitively on the way in and always store lowercase. Anything
// not in this set clamps to 'other'.
const AISLE_CATEGORIES = [
  'produce', 'meat_seafood', 'dairy_eggs', 'bakery', 'frozen', 'canned_goods',
  'pantry', 'snacks', 'beverages', 'condiments_sauces', 'spices_baking',
  'household', 'personal_care', 'other',
];
const AISLE_SET = new Set(AISLE_CATEGORIES);
function clampAisle(v) {
  if (typeof v !== 'string') return 'other';
  const norm = v.trim().toLowerCase();
  return AISLE_SET.has(norm) ? norm : 'other';
}

// Lambda self-invoke client (async fire-and-forget categorization).
let lambdaClient;
function getLambdaClient() {
  if (!lambdaClient) lambdaClient = new LambdaClient({});
  return lambdaClient;
}
// Lambda sets AWS_LAMBDA_FUNCTION_NAME automatically; fall back for local runs.
const SELF_FUNCTION_NAME = process.env.AWS_LAMBDA_FUNCTION_NAME || 'trepo-list-handler';

// Fire-and-forget self-invoke to categorize any pending (uncategorized) items for
// this owner's household. Never awaited on the request path — mirrors the feed
// async-invoke pattern. Swallows all errors so it can never break a mutation/read.
function dispatchCategorize(owner) {
  try {
    getLambdaClient()
      .send(new InvokeCommand({
        FunctionName: SELF_FUNCTION_NAME,
        InvocationType: 'Event',
        Payload: Buffer.from(JSON.stringify({ action: 'categorize_pending', owner })),
      }))
      .catch((e) => console.error(JSON.stringify({
        evt: 'list_categorize_dispatch_error', owner, error: e.code || e.message,
      })));
  } catch (e) {
    console.error(JSON.stringify({
      evt: 'list_categorize_dispatch_error', owner, error: e.code || e.message,
    }));
  }
}

// True if any *live* (ADDED) item is still missing an aisle_category — the exact
// condition categorize_pending fixes, so gating on it avoids infinite re-dispatch
// (CHECKED/REMOVED rows are never categorized and must not keep re-triggering).
function needsCategorize(items) {
  return Array.isArray(items)
    && items.some((it) => it && it.action === 'ADDED' && !it.aisle_category);
}

// Idempotent one-time ALTER to add the aisle_category column. Guarded against the
// concurrent-ALTER race (ER_DUP_FIELDNAME) and missing member tables.
async function ensureAisleColumn(table) {
  if (!/^[0-9a-zA-Z_-]{1,80}$/.test(table)) return;
  try {
    await getPool().query(`ALTER TABLE \`${table}\` ADD COLUMN aisle_category VARCHAR(40) NULL`);
  } catch (e) {
    if (e.code === 'ER_DUP_FIELDNAME') return; // already added (idempotent / concurrent ALTER)
    if (e.code === 'ER_NO_SUCH_TABLE') return; // member never created a list yet
    throw e;
  }
}

// One OpenAI call: map a deduped array of grocery item names -> aisle enum.
// gpt-4.1-mini, json_object, temperature 0, 6s AbortSignal budget.
async function categorizeNames(names) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not configured');
  const controller = new AbortController();
  // 15s: the 6s budget TimeoutError'd in prod on normal batches (DOMException
  // code 20). The categorize branch is always an async self-invoke (20s fn
  // timeout), so nothing user-facing waits on this.
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const system = `You categorize grocery items into supermarket aisles. `
      + `Return a JSON object of the form {"categories": {"<item_name>": "<category>"}} `
      + `where every key is an input item name (verbatim) and every value is EXACTLY one of: `
      + `${AISLE_CATEGORIES.join(', ')}. Use "other" when unsure. Do not invent categories.`;
    const resp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: 'gpt-4.1-mini',
        temperature: 0,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `Categorize these grocery item names: ${JSON.stringify(names)}` },
        ],
      }),
      signal: controller.signal,
    });
    if (!resp.ok) {
      const t = await resp.text().catch(() => '');
      throw new Error(`openai ${resp.status}: ${String(t).slice(0, 200)}`);
    }
    const data = await resp.json();
    const content = data && data.choices && data.choices[0]
      && data.choices[0].message && data.choices[0].message.content;
    const parsed = JSON.parse(content || '{}');
    return parsed.categories || {};
  } finally {
    clearTimeout(timer);
  }
}

// Internal async action `categorize_pending`: resolve household members, ensure the
// column exists in every store, collect ADDED+uncategorized rows across per-user
// tables AND shared_shopping_list, one LLM call, then write categories back by _id
// in BOTH stores. Never throws to the caller (invoked fire-and-forget).
async function categorizePending(owner) {
  try {
    if (!owner || !/^[0-9a-zA-Z-]{1,36}$/.test(owner)) {
      console.error(JSON.stringify({ evt: 'list_categorize_error', owner, error: 'invalid owner' }));
      return { ok: false };
    }
    const pool = getPool();
    const memberIds = await getHouseholdMemberIds(pool, owner);
    if (!memberIds.length) return { ok: true, updated: 0 };

    // b. one-time idempotent ALTER on each store.
    for (const memberId of memberIds) {
      await ensureAisleColumn(`${memberId}_new_list`);
    }
    await ensureAisleColumn('shared_shopping_list');

    // c. collect ADDED rows still missing a category (carry ids for exact write-back).
    const perUser = []; // { table, id, name }
    for (const memberId of memberIds) {
      const t = `${memberId}_new_list`;
      try {
        const [rows] = await pool.execute(
          `SELECT _id, product_name FROM \`${t}\` WHERE action = 'ADDED' AND aisle_category IS NULL`
        );
        for (const r of rows) perUser.push({ table: t, id: r._id, name: r.product_name });
      } catch (e) {
        if (e.code !== 'ER_NO_SUCH_TABLE') {
          console.error(JSON.stringify({ evt: 'list_categorize_error', owner, table: t, error: e.code || e.message }));
        }
      }
    }
    const shared = []; // { id, name }
    try {
      const ph = memberIds.map(() => '?').join(',');
      const [rows] = await pool.execute(
        `SELECT _id, product_name FROM shared_shopping_list
         WHERE action = 'ADDED' AND aisle_category IS NULL AND owner_id IN (${ph})`,
        memberIds
      );
      for (const r of rows) shared.push({ id: r._id, name: r.product_name });
    } catch (e) {
      if (e.code !== 'ER_NO_SUCH_TABLE') {
        console.error(JSON.stringify({ evt: 'list_categorize_error', owner, table: 'shared_shopping_list', error: e.code || e.message }));
      }
    }

    const names = [...new Set(
      [...perUser, ...shared].map((x) => String(x.name || '').trim()).filter(Boolean)
    )];
    if (names.length === 0) {
      console.log(JSON.stringify({ evt: 'list_categorized', owner, updated: 0 }));
      return { ok: true, updated: 0 };
    }

    // d. single LLM call, then normalize keys for case-insensitive lookup.
    const raw = await categorizeNames(names);
    const lookup = {};
    for (const [k, v] of Object.entries(raw)) {
      lookup[String(k).trim().toLowerCase()] = clampAisle(v);
    }
    const catFor = (name) => lookup[String(name || '').trim().toLowerCase()] || 'other';

    // e. write back by _id in BOTH stores (no fuzzy matching).
    let updated = 0;
    for (const item of perUser) {
      try {
        const [res] = await pool.execute(
          `UPDATE \`${item.table}\` SET aisle_category = ? WHERE _id = ?`,
          [catFor(item.name), item.id]
        );
        updated += res.affectedRows || 0;
      } catch (e) {
        console.error(JSON.stringify({ evt: 'list_categorize_error', owner, table: item.table, id: item.id, error: e.code || e.message }));
      }
    }
    for (const item of shared) {
      try {
        const [res] = await pool.execute(
          'UPDATE shared_shopping_list SET aisle_category = ? WHERE _id = ?',
          [catFor(item.name), item.id]
        );
        updated += res.affectedRows || 0;
      } catch (e) {
        console.error(JSON.stringify({ evt: 'list_categorize_error', owner, table: 'shared_shopping_list', id: item.id, error: e.code || e.message }));
      }
    }

    console.log(JSON.stringify({ evt: 'list_categorized', owner, updated }));
    return { ok: true, updated };
  } catch (err) {
    console.error(JSON.stringify({ evt: 'list_categorize_error', owner, error: err.name === 'TimeoutError' ? 'llm_timeout' : (err.message || err.code || err.name) }));
    return { ok: false };
  }
}

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
    aisle_category: r.aisle_category || null,
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

// Self-heal: recreate a caller's missing `_new_list` table (purged/legacy
// accounts hit ER_NO_SUCH_TABLE on their OWN write, which used to 500).
// DDL mirrors the twilioAuth signup DDL plus `sort_order` (which signup
// historically omitted — schema-drift bug backfilled 2026-07-03).
async function ensureOwnListTable(ownerId) {
  // ownerId is a UUID from the auth path, but never interpolate unvalidated.
  if (!/^[0-9a-zA-Z-]{1,64}$/.test(ownerId)) throw new Error('invalid ownerId for table create');
  await getPool().query(`
    CREATE TABLE IF NOT EXISTS \`${ownerId}_new_list\` (
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
      \`sort_order\` INT DEFAULT NULL,
      PRIMARY KEY (\`_id\`),
      KEY \`idx_owner_device\` (\`_owner\`, \`_device\`),
      KEY \`idx_created\` (\`_createdDate\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
  `);
  console.error(JSON.stringify({
    evt: 'backend_error', service: 'list', op: 'own_table_recreated',
    owner_id: ownerId, code: 'missing_new_list_table',
    error: 'caller _new_list table was missing; recreated (purged/legacy account)',
  }));
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

    // Internal async action (self-invoked, InvocationType Event): categorize any
    // pending list items for this household via one LLM call. Routed before the
    // normal owner validation because its payload carries `owner`, not `ownerId`.
    if (body && body.action === 'categorize_pending') {
      const result = await categorizePending(body.owner);
      return response(200, { message: 'categorize_pending done', ...result });
    }

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
      aisle_category,   // optional: caller placed the item in a specific aisle
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
          // Fire-and-forget: backfill missing aisle categories without blocking.
          if (needsCategorize(items)) dispatchCategorize(ownerId);
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
        // Fire-and-forget: backfill missing aisle categories without blocking.
        if (needsCategorize(items)) dispatchCategorize(ownerId);
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

      // Caller-chosen aisle (user tapped "+" on an aisle section). Optional.
      //
      // Setting it here is what makes the choice STICK: the categorizer only ever
      // touches rows WHERE aisle_category IS NULL, so a pre-set value is skipped
      // forever. No "user set this" flag is needed.
      //
      // The column is added by a LAZY ALTER that until now only ran in the categorize
      // path — so a table that has never been categorized may not have it, and
      // `shared_list` never gets it at all. Two consequences, both handled:
      //   1. Only widen the INSERT when an aisle was actually supplied, so every
      //      existing caller executes byte-identical SQL to before.
      //   2. Ensure the column on the tables we are about to name, first.
      const chosenAisle = (aisle_category != null && String(aisle_category).trim() !== '')
        ? clampAisle(aisle_category)
        : null;
      if (chosenAisle) {
        for (const memberId of memberIds) await ensureAisleColumn(`${memberId}_new_list`);
        await ensureAisleColumn('shared_shopping_list');
      }
      const aisleCol = chosenAisle ? ', aisle_category' : '';
      const aisleVal = chosenAisle ? ', ?' : '';
      const aisleParam = chosenAisle ? [chosenAisle] : [];

      const sql = `
        INSERT INTO \`${'${tableName}'}\`
          (_owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid${aisleCol})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?${aisleVal})
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
            ...aisleParam,
          ]);
          if (memberId === ownerId) result = insertResult;
        } catch (e) {
          // Caller's own table missing (purged/legacy account): recreate and
          // retry once instead of 500ing.
          if (memberId === ownerId && e.code === 'ER_NO_SUCH_TABLE') {
            await ensureOwnListTable(ownerId);
            // ensureOwnListTable creates the ORIGINAL schema, which has no
            // aisle_category — the column only ever arrives via the lazy ALTER. Without
            // this the retry would fail on the very account the retry exists to rescue.
            if (chosenAisle) await ensureAisleColumn(tableName);
            const [retryResult] = await pool.execute(sql.replace('${tableName}', tableName), [
              memberId, device, product_name, product_brand || null, images || null,
              product_barcode || null, effectiveAction, store || null, sharedUUID,
              ...aisleParam,
            ]);
            result = retryResult;
            continue;
          }
          // A sibling member's write failing must never abort the others (this is
          // exactly how items got orphaned). Log it loudly so misses stop being
          // silent; only re-throw if the caller's OWN write failed.
          console.error(JSON.stringify({
            evt: 'list_fanout_miss', op: 'add', ownerId, memberId,
            product_name, error: e.code || e.message,
          }));
          if (memberId === ownerId) throw e;
        }
        // Mirror into the VOICE store (shared_shopping_list) per member so items
        // added in the app are visible to HALO voice. Voice reads owner-scoped,
        // and the remove path marks these rows 'REMOVED' — this is the missing add
        // direction. Ungated + non-fatal, matching the remove-side mark.
        try {
          await pool.execute(
            `INSERT INTO shared_shopping_list
              (owner_id, _owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid${aisleCol})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?${aisleVal})`,
            [memberId, memberId, device, product_name, product_brand || null, images || null, product_barcode || null, effectiveAction, store || null, sharedUUID, ...aisleParam]
          );
        } catch (e) {
          console.error(JSON.stringify({ evt: 'shared_shopping_add_miss', op: 'add', ownerId, memberId, error: e.code || e.message }));
        }
      }

      // Dual-write to shared_list.
      // Deliberately NOT carrying aisle_category: `shared_list` is a migration mirror and
      // is the one table ensureAisleColumn never touches, so naming the column here would
      // throw on every add. The miss is self-healing — if reads ever flip to this table the
      // categorizer fills the NULL — and this INSERT is already non-fatal, which would have
      // turned a hard schema error into a silently dead dual-write.
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

      console.log(JSON.stringify({ evt: 'list_op', op: 'add', ownerId, uuid: sharedUUID, insertId: result.insertId ?? null }));
      // Fire-and-forget: categorize the just-added item (and any pending ones).
      try { dispatchCategorize(ownerId); } catch (e) { /* never block the add */ }
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
            if (err.code === 'ER_NO_SUCH_TABLE') {
              // Caller's OWN table missing: recreate + retry so the item isn't
              // silently dropped while the response still claims success.
              if (memberId === ownerId) {
                await ensureOwnListTable(ownerId);
                await pool.execute(
                  `INSERT INTO \`${tableName}\`
                    (_owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                  [memberId, itemDevice, itemName, itemBrand, itemImages, itemBarcode, itemAction, itemStore, sharedUUID]
                );
              }
              // sibling member never created a list: skip, same as before
            } else {
              // Resilient fan-out: log the miss, keep writing the other members.
              console.error(JSON.stringify({
                evt: 'list_fanout_miss', op: 'batch_add', ownerId, memberId,
                product_name: itemName, error: err.code || err.message,
              }));
            }
          }
          // Mirror into the VOICE store (shared_shopping_list) per member — the
          // missing add direction that made app-added items invisible to HALO voice.
          try {
            await pool.execute(
              `INSERT INTO shared_shopping_list
                (owner_id, _owner, _device, product_name, product_brand, images, product_barcode, action, store, household_item_uuid)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
              [memberId, memberId, itemDevice, itemName, itemBrand, itemImages, itemBarcode, itemAction, itemStore, sharedUUID]
            );
          } catch (e) {
            console.error(JSON.stringify({ evt: 'shared_shopping_add_miss', op: 'batch_add', ownerId, memberId, error: e.code || e.message }));
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

      console.log(JSON.stringify({ evt: 'list_op', op: 'add_batch', ownerId, count: addedItems.length }));
      // Fire-and-forget: categorize the just-added items (and any pending ones).
      try { dispatchCategorize(ownerId); } catch (e) { /* never block the batch add */ }
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
        // Mark the VOICE shared table (shared_shopping_list) REMOVED so the dual-write
        // canary stops treating the item as live and RESURRECTING it hourly. UPDATE
        // (not DELETE) preserves audit history; the canary ignores non-ADDED rows.
        // Always runs (independent of the legacy shared_list dual-write flag).
        try {
          await pool.execute(
            "UPDATE shared_shopping_list SET action = 'REMOVED', updated_at = NOW() WHERE (household_item_uuid = ? OR _id = ?) AND action = 'ADDED'",
            [removeKey, removeKey]
          );
        } catch (e) {
          console.error('[DUAL-WRITE] shared_shopping_list REMOVED-mark failed (non-fatal):', e.message);
        }
        console.log(JSON.stringify({ evt: 'list_op', op: 'remove', ownerId, uuid: removeKey, affectedRows }));
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
        // Mark the VOICE shared table REMOVED by barcode across the household so the
        // canary doesn't resurrect it. (Voice rows are owner-scoped per member.)
        try {
          if (memberIds.length) {
            const ph = memberIds.map(() => '?').join(',');
            await pool.execute(
              `UPDATE shared_shopping_list SET action = 'REMOVED', updated_at = NOW() WHERE product_barcode = ? AND owner_id IN (${ph}) AND action = 'ADDED'`,
              [product_barcode, ...memberIds]
            );
          }
        } catch (e) {
          console.error('[DUAL-WRITE] shared_shopping_list REMOVED-mark (barcode) failed (non-fatal):', e.message);
        }
        console.log(JSON.stringify({ evt: 'list_op', op: 'remove_barcode', ownerId, barcode: product_barcode, affectedRows }));
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
      console.log(JSON.stringify({ evt: 'list_op', op: 'reorder', ownerId, count: orderedItems.length }));
      return response(200, { message: 'Reorder applied', count: orderedItems.length });
    }

    return response(400, { message: 'Invalid operation (use view/add/remove/set_action/update_item/reorder)' });
  } catch (err) {
    console.error(err);
    return response(500, { message: 'Internal server error', error: err.message });
  }
};
