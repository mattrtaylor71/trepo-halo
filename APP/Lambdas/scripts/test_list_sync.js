// Throwaway integration test for listHandler.js sync behavior.
// Creates two synthetic household members + scratch _new_list tables, exercises
// every op, asserts against direct DB reads, then tears everything down.
// NEVER touches real user rows (uses TEST_ prefixed owner_id + random uuids).
const mysql = require('mysql2/promise');
const crypto = require('crypto');

const DB = {
  host: 'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',
  user: 'admin', password: 'Nbmqyq17!', database: 'mysqlTutorial', connectTimeout: 15000,
};
process.env.DB_HOST = DB.host; process.env.DB_USER = DB.user;
process.env.DB_PASSWORD = DB.password; process.env.DB_NAME = DB.database;

const A = '00000000-test-4aaa-8000-' + crypto.randomBytes(6).toString('hex'); // member A
const B = '00000000-test-4bbb-8000-' + crypto.randomBytes(6).toString('hex'); // member B
const HH = 'TESTHH' + crypto.randomBytes(3).toString('hex');
const TPL = '7d7df434-d942-4037-b054-2d3005ea6abc_new_list'; // schema template

let pass = 0, fail = 0;
function check(name, cond) { if (cond) { pass++; console.log('  ✓', name); } else { fail++; console.log('  ✗ FAIL:', name); } }

function loadHandler(flags) {
  for (const k of ['LIST_UNION_READ','LIST_UNION_READ_OWNERS','LIST_STRICT_UUID_MATCH']) delete process.env[k];
  Object.assign(process.env, flags || {});
  delete require.cache[require.resolve('../listHandler.js')];
  delete require.cache[require.resolve('../householdSync.js')];
  return require('../listHandler.js');
}

(async () => {
  const conn = await mysql.createConnection(DB);
  const tableA = `${A}_new_list`, tableB = `${B}_new_list`;
  async function rowsOf(t) { const [r] = await conn.execute(`SELECT product_name, action, household_item_uuid, sort_order FROM \`${t}\``); return r; }

  try {
    // --- setup ---
    await conn.execute('INSERT INTO new_users (user_id, owner_id, first_name, created_at) VALUES (?,?,?,NOW())', [A, HH, 'TestA']);
    await conn.execute('INSERT INTO new_users (user_id, owner_id, first_name, created_at) VALUES (?,?,?,NOW())', [B, HH, 'TestB']);
    await conn.execute(`CREATE TABLE \`${tableA}\` LIKE \`${TPL}\``);
    await conn.execute(`CREATE TABLE \`${tableB}\` LIKE \`${TPL}\``);
    console.log('Setup: members', A.slice(0,13), '/', B.slice(0,13), 'household', HH);

    // === TEST 1: add fans out to BOTH members ===
    console.log('\n[1] add fan-out');
    let h = loadHandler({ LIST_STRICT_UUID_MATCH: 'true' });
    await h.handler({ operation: 'add', ownerId: A, device: 'test', product_name: 'milk' });
    let ra = await rowsOf(tableA), rb = await rowsOf(tableB);
    check('A has milk', ra.some(r => r.product_name === 'milk'));
    check('B has milk (fanned out)', rb.some(r => r.product_name === 'milk'));
    const milkUuidA = ra.find(r => r.product_name === 'milk').household_item_uuid;
    const milkUuidB = rb.find(r => r.product_name === 'milk').household_item_uuid;
    check('milk shares the SAME uuid across both', milkUuidA && milkUuidA === milkUuidB);

    // === TEST 2: simulate an orphan (item only on B, like tallow fries) ===
    console.log('\n[2] orphan + union read heals it');
    const orphanUuid = crypto.randomUUID();
    await conn.execute(`INSERT INTO \`${tableB}\` (_owner,_device,product_name,action,household_item_uuid) VALUES (?,?,?,?,?)`,
      [B, 'test', 'eggs', 'ADDED', orphanUuid]);
    // union OFF: A sees only its own table => 1 item (milk), NOT eggs
    h = loadHandler({ LIST_UNION_READ: 'false' });
    let res = await h.handler({ operation: 'view', ownerId: A });
    let body = JSON.parse(res.body);
    check('union OFF: A sees 1 item (own table only)', body.count === 1 && body.items[0].product_name === 'milk');
    // union ON: A sees BOTH milk + eggs (orphan healed at read)
    h = loadHandler({ LIST_UNION_READ: 'true' });
    res = await h.handler({ operation: 'view', ownerId: A });
    body = JSON.parse(res.body);
    const names = body.items.map(i => i.product_name).sort();
    check('union ON: A sees 2 items (milk+eggs)', body.count === 2 && names.join(',') === 'eggs,milk');
    check('union response flagged union:true', body.union === true);

    // === TEST 3: union dedups shared item (no doubles) ===
    console.log('\n[3] union dedup');
    // milk is on BOTH with same uuid -> must appear ONCE
    const milkCount = body.items.filter(i => i.product_name === 'milk').length;
    check('milk appears exactly once in union', milkCount === 1);

    // === TEST 4: set_action (check) propagates to BOTH (strict, by uuid) ===
    console.log('\n[4] set_action propagation (strict uuid)');
    h = loadHandler({ LIST_STRICT_UUID_MATCH: 'true' });
    await h.handler({ operation: 'set_action', ownerId: A, itemUUID: milkUuidA, checked: true });
    ra = await rowsOf(tableA); rb = await rowsOf(tableB);
    check('A milk CHECKED', ra.find(r => r.product_name === 'milk').action === 'CHECKED');
    check('B milk CHECKED (propagated)', rb.find(r => r.product_name === 'milk').action === 'CHECKED');

    // === TEST 5: strict mode does NOT match by _id (no single-copy mutation) ===
    console.log('\n[5] strict mode ignores _id fallback');
    const aMilkId = (await conn.execute(`SELECT _id FROM \`${tableA}\` WHERE product_name='milk'`))[0][0]._id;
    let delRes = await h.handler({ operation: 'remove', ownerId: A, id: aMilkId }); // passing local _id, not uuid
    check('strict: remove by _id removes nothing', JSON.parse(delRes.body).affectedRows === 0);
    check('A still has milk after _id remove', (await rowsOf(tableA)).some(r => r.product_name === 'milk'));

    // === TEST 6: remove by uuid clears BOTH ===
    console.log('\n[6] remove by uuid clears both');
    await h.handler({ operation: 'remove', ownerId: A, itemUUID: milkUuidA });
    ra = await rowsOf(tableA); rb = await rowsOf(tableB);
    check('A milk gone', !ra.some(r => r.product_name === 'milk'));
    check('B milk gone (propagated)', !rb.some(r => r.product_name === 'milk'));

    // === TEST 7: add resilient when a member table is missing ===
    console.log('\n[7] add resilient if a member table is dropped');
    await conn.execute(`DROP TABLE \`${tableB}\``);
    h = loadHandler({ LIST_STRICT_UUID_MATCH: 'true' });
    let addRes = await h.handler({ operation: 'add', ownerId: A, device: 'test', product_name: 'bread' });
    check('add still succeeds (200) despite B table missing', addRes.statusCode === 200);
    check('A got bread', (await rowsOf(tableA)).some(r => r.product_name === 'bread'));

  } catch (e) {
    console.error('TEST ERROR:', e);
    fail++;
  } finally {
    // --- teardown ---
    await conn.execute(`DROP TABLE IF EXISTS \`${tableA}\``);
    await conn.execute(`DROP TABLE IF EXISTS \`${tableB}\``);
    await conn.execute('DELETE FROM new_users WHERE owner_id = ?', [HH]);
    console.log('\nTeardown complete.');
    console.log(`\n=== RESULT: ${pass} passed, ${fail} failed ===`);
    await conn.end();
    process.exit(fail ? 1 : 0);
  }
})();
