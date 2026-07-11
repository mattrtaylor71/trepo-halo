// Fault-injection validator for the canary's AUTO-HEAL path.
//
// Unlike the `canary-fault-test-` prefixed rows (which the checks EXCLUDE, so they
// never move prod metrics), this injects a NON-prefixed shared_dishes "voice" row for
// the SAFE test user with NO per-user copy — i.e. a row the canary treats as a REAL
// miss. Running the canary should then: detect it, auto-heal it into {user}_dishes,
// emit VoiceDualWriteReconciled=1 / VoiceDualWriteUnhealed=0, and log
// voice_dualwrite_autohealed. `cleanup` removes both the shared row and the healed copy.
//
//   node faultInject.mjs inject     # create the synthetic miss
//   node faultInject.mjs verify     # is the per-user copy present yet? (post-canary)
//   node faultInject.mjs cleanup    # remove shared row + healed copy
//
// SAFE test user ONLY. Never run against a real owner.
import mysql from "mysql2/promise";

const SAFE_USER = "1f4db6b6-2f62-4558-aa40-c8e82527dc74";
const TEST_ID = "f0e1d2c3-b4a5-4867-9899-healvalid000"; // NON-prefixed, exactly 36 chars (fits _id VARCHAR(36))
const DISH_TABLE = `${SAFE_USER}_dishes`;

const conn = () => mysql.createConnection({
  host: process.env.DB_HOST || "database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com",
  user: process.env.DB_USER || "admin",
  password: process.env.DB_PASS || "Nbmqyq17!",
  database: process.env.DB_NAME || "mysqlTutorial",
  connectTimeout: 15000,
});

const mode = process.argv[2] || "verify";
const c = await conn();
try {
  if (mode === "inject") {
    await c.execute(
      `INSERT INTO shared_dishes (_id, _owner, _device, dish_name, owner_id, user_id, _createdDate)
       VALUES (?, ?, 'voice-assistant', 'CANARY_HEAL_VALIDATION', ?, ?, NOW())
       ON DUPLICATE KEY UPDATE _createdDate = NOW()`,
      [TEST_ID, SAFE_USER, SAFE_USER, SAFE_USER]
    );
    // ensure NO per-user copy exists (so it's a real miss)
    try { await c.execute(`DELETE FROM \`${DISH_TABLE}\` WHERE _id = ?`, [TEST_ID]); } catch { /* table may not exist yet */ }
    const [s] = await c.execute("SELECT _id FROM shared_dishes WHERE _id = ?", [TEST_ID]);
    console.log(JSON.stringify({ mode, injected: !!s.length, shared_id: TEST_ID, per_user_copy: "absent (real miss)" }));
  } else if (mode === "verify") {
    const [pu] = await c.execute(`SELECT _id, dish_name, _createdDate FROM \`${DISH_TABLE}\` WHERE _id = ?`, [TEST_ID]);
    console.log(JSON.stringify({ mode, healed: !!pu.length, per_user_row: pu[0] || null }));
  } else if (mode === "cleanup") {
    const [d1] = await c.execute("DELETE FROM shared_dishes WHERE _id = ?", [TEST_ID]);
    let d2 = { affectedRows: 0 };
    try { [d2] = await c.execute(`DELETE FROM \`${DISH_TABLE}\` WHERE _id = ?`, [TEST_ID]); } catch { /* ignore */ }
    console.log(JSON.stringify({ mode, removed_shared: d1.affectedRows, removed_per_user: d2.affectedRows }));
  } else {
    console.error("usage: node faultInject.mjs inject|verify|cleanup");
    process.exit(2);
  }
} finally {
  await c.end();
}
