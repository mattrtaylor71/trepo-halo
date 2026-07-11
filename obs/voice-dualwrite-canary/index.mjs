// trepo-voice-dualwrite-canary — the "never again" guard for the WRITE_SHARED_ONLY
// dual-write class. Hourly (EventBridge), find voice-written rows in the last 48h
// whose per-user, app-visible copy is MISSING, across three domains:
//   shopping  — shared_shopping_list (_device='voice-assistant') household_item_uuid
//               present in >=1 household member's {member}_new_list
//   dishes    — shared_dishes (_device='voice-assistant') _id in {user_id}_dishes
//   discards  — shared_discards (_device LIKE 'voice%') _id in the owner's {owner}_discards
//
// DETECT + AUTO-HEAL: when a miss is found, the authoritative shared row is copied
// (idempotent, original columns/timestamps) into the missing per-user/member table(s).
// Metrics per Domain:
//   VoiceDualWriteMiss        — detected misses (observability)
//   VoiceDualWriteReconciled  — misses auto-healed this run
//   VoiceDualWriteUnhealed    — misses the heal did NOT fix, OR "stale" misses whose
//                               shared row is older than STALE_THRESHOLD (they survived
//                               a prior hourly heal → real corruption, not the transient
//                               runtime anomaly). The ALARM fires on THIS metric, so Matt
//                               is paged only when self-healing FAILS — not when it works.
// On INTERNAL failure it emits NOTHING and logs loudly — never a 0 (which would mask a
// real miss) — so a broken canary doesn't hide a regression.

import mysql from "mysql2/promise";
import { CloudWatchClient, PutMetricDataCommand } from "@aws-sdk/client-cloudwatch";

const cw = new CloudWatchClient({ region: process.env.AWS_REGION || "us-east-1" });
const LOOKBACK_HOURS = 48;
const MAX_ROWS_LOGGED = 25;
// A miss whose shared row is older than this "should" have been healed by a prior
// hourly run; still missing => reappeared/persistent corruption, not the transient
// anomaly. Counts toward Unhealed (alarm) even if we heal it now. ~2 run cycles.
const STALE_THRESHOLD_HOURS = 2;
// Synthetic fault-injection rows (validation self-tests) use this _id prefix and
// must NEVER count as real misses — otherwise the canary alarms on its own test.
// Keep using this prefix when fault-injecting the PROD-metric path; the prod alarm
// ignores these rows. (To validate the AUTO-HEAL path end-to-end, inject a
// NON-prefixed row for the SAFE test user instead — see obs/.../faultInject.mjs.)
const SELF_TEST_PREFIX = "canary-fault-test-";

async function tableExists(conn, name) {
  const [r] = await conn.execute(
    "SELECT COUNT(*) c FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = ?",
    [name]
  );
  return r[0].c > 0;
}

async function columnsOf(conn, table) {
  const [r] = await conn.execute(
    "SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = ?",
    [table]
  );
  return new Set(r.map((x) => x.column_name || x.COLUMN_NAME));
}

async function householdMembers(conn, userId) {
  const [r] = await conn.execute("SELECT owner_id FROM new_users WHERE user_id = ? LIMIT 1", [userId]);
  const household = r[0]?.owner_id;
  if (!household) return [userId];
  const [m] = await conn.execute("SELECT user_id FROM new_users WHERE owner_id = ?", [household]);
  const ids = m.map((x) => x.user_id).filter(Boolean);
  return ids.length ? [...new Set(ids)] : [userId];
}

const ageHours = (createdDate) => {
  const t = createdDate instanceof Date ? createdDate.getTime() : new Date(createdDate).getTime();
  return Number.isFinite(t) ? (Date.now() - t) / 3.6e6 : 0;
};

// Idempotent copy of the authoritative shared row into ONE per-user/member table.
// Common columns only (preserves original values + timestamps); owner_id (if present)
// is rewritten to the member id. INSERT-if-absent by keyCol. Returns true if the key is
// present in destTable afterward.
async function healInto(conn, { sharedTable, destTable, keyCol, keyVal, memberId }) {
  if (!(await tableExists(conn, destTable))) return false; // cannot heal into a table that doesn't exist
  const [sc, dc] = [await columnsOf(conn, sharedTable), await columnsOf(conn, destTable)];
  const common = [...sc].filter((c) => dc.has(c));
  if (!common.length || !dc.has(keyCol)) return false;
  const colList = common.map((c) => `\`${c}\``).join(", ");
  const selectList = common.map((c) => (c === "owner_id" ? "?" : `\`${c}\``)).join(", ");
  const hasOwner = common.includes("owner_id");
  const sql =
    `INSERT INTO \`${destTable}\` (${colList})\n` +
    `SELECT ${selectList} FROM \`${sharedTable}\` s\n` +
    `WHERE s.\`${keyCol}\` = ?\n` +
    `  AND NOT EXISTS (SELECT 1 FROM \`${destTable}\` d WHERE d.\`${keyCol}\` = ?)\n` +
    `LIMIT 1`;
  const params = hasOwner ? [memberId, keyVal, keyVal] : [keyVal, keyVal];
  await conn.execute(sql, params);
  const [c] = await conn.execute(`SELECT COUNT(*) c FROM \`${destTable}\` WHERE \`${keyCol}\` = ?`, [keyVal]);
  return c[0].c > 0;
}

async function checkDishes(conn) {
  const [rows] = await conn.execute(
    `SELECT _id, user_id, owner_id, _createdDate FROM shared_dishes WHERE _device = 'voice-assistant' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR AND _id NOT LIKE ?`,
    [`${SELF_TEST_PREFIX}%`]
  );
  const misses = [];
  for (const r of rows) {
    const owner = r.user_id || r.owner_id;
    const table = `${owner}_dishes`;
    const has = (await tableExists(conn, table)) &&
      (await conn.execute(`SELECT COUNT(*) c FROM \`${table}\` WHERE _id = ?`, [r._id]))[0][0].c > 0;
    if (!has) misses.push({ key: r._id, owner, ageHours: ageHours(r._createdDate),
      sharedTable: "shared_dishes", keyCol: "_id", keyVal: r._id, targets: [{ table, memberId: owner }] });
  }
  return misses;
}

async function checkDiscards(conn) {
  const [rows] = await conn.execute(
    `SELECT _id, user_id, owner_id, _createdDate FROM shared_discards WHERE _device LIKE 'voice%' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR AND _id NOT LIKE ?`,
    [`${SELF_TEST_PREFIX}%`]
  );
  const misses = [];
  for (const r of rows) {
    const owner = r.owner_id || r.user_id;
    const table = `${owner}_discards`;
    const has = (await tableExists(conn, table)) &&
      (await conn.execute(`SELECT COUNT(*) c FROM \`${table}\` WHERE _id = ?`, [r._id]))[0][0].c > 0;
    if (!has) misses.push({ key: r._id, owner, ageHours: ageHours(r._createdDate),
      sharedTable: "shared_discards", keyCol: "_id", keyVal: r._id, targets: [{ table, memberId: owner }] });
  }
  return misses;
}

async function checkShopping(conn) {
  const [rows] = await conn.execute(
    `SELECT household_item_uuid, owner_id, _owner, _createdDate FROM shared_shopping_list WHERE _device = 'voice-assistant' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR AND _id NOT LIKE ? AND COALESCE(household_item_uuid, '') NOT LIKE ?`,
    [`${SELF_TEST_PREFIX}%`, `${SELF_TEST_PREFIX}%`]
  );
  const misses = [];
  for (const r of rows) {
    if (!r.household_item_uuid) continue;
    const owner = r.owner_id || r._owner;
    const members = await householdMembers(conn, owner);
    let found = false;
    for (const member of members) {
      const table = `${member}_new_list`;
      if (!(await tableExists(conn, table))) continue;
      const [c] = await conn.execute(`SELECT COUNT(*) c FROM \`${table}\` WHERE household_item_uuid = ?`, [r.household_item_uuid]);
      if (c[0].c > 0) { found = true; break; }
    }
    if (!found) misses.push({ key: r.household_item_uuid, owner, ageHours: ageHours(r._createdDate),
      sharedTable: "shared_shopping_list", keyCol: "household_item_uuid", keyVal: r.household_item_uuid,
      targets: members.map((m) => ({ table: `${m}_new_list`, memberId: m })) });
  }
  return misses;
}

// Heal a miss into all its target tables; return {healed, stale}. healed = the row is
// present in >=1 target afterward (matches the detection contract); stale = shared row
// older than STALE_THRESHOLD (survived a prior heal cycle => corruption signal).
async function reconcileMiss(conn, miss) {
  let anyPresent = false;
  for (const t of miss.targets) {
    try {
      const present = await healInto(conn, { sharedTable: miss.sharedTable, destTable: t.table, keyCol: miss.keyCol, keyVal: miss.keyVal, memberId: t.memberId });
      anyPresent = anyPresent || present;
    } catch (e) {
      console.error(JSON.stringify({ evt: "voice_dualwrite_heal_error", table: t.table, key: miss.keyVal, error: String(e && (e.stack || e.message) || e) }));
    }
  }
  return { healed: anyPresent, stale: miss.ageHours > STALE_THRESHOLD_HOURS };
}

export const handler = async () => {
  let conn;
  try {
    conn = await mysql.createConnection({
      host: process.env.DB_HOST, user: process.env.DB_USER,
      password: process.env.DB_PASS, database: process.env.DB_NAME,
      connectTimeout: 15000,
    });
  } catch (e) {
    console.error(JSON.stringify({ evt: "voice_dualwrite_canary_error", phase: "connect", error: String(e && (e.stack || e.message) || e) }));
    return { ok: false };
  }

  let results;
  try {
    // Compute ALL domains before emitting — if any check throws, we emit nothing.
    results = {
      shopping: await checkShopping(conn),
      dishes: await checkDishes(conn),
      discards: await checkDiscards(conn),
    };
  } catch (e) {
    console.error(JSON.stringify({ evt: "voice_dualwrite_canary_error", phase: "check", error: String(e && (e.stack || e.message) || e) }));
    try { await conn.end(); } catch { /* ignore */ }
    return { ok: false };
  }

  // AUTO-HEAL pass (best-effort per row; a heal failure just surfaces as Unhealed).
  const summary = {};
  try {
    for (const [domain, misses] of Object.entries(results)) {
      let reconciled = 0, unhealed = 0;
      const healedRows = [];
      for (const miss of misses) {
        const { healed, stale } = await reconcileMiss(conn, miss);
        if (healed) { reconciled += 1; healedRows.push({ key: miss.keyVal, owner: miss.owner, ageHours: Math.round(miss.ageHours) }); }
        // Unhealed = heal failed, OR a stale row that keeps reappearing (real corruption).
        if (!healed || stale) unhealed += 1;
      }
      if (healedRows.length) {
        console.log(JSON.stringify({ evt: "voice_dualwrite_autohealed", domain, rows: healedRows.slice(0, MAX_ROWS_LOGGED), count: healedRows.length }));
      }
      summary[domain] = { detected: misses.length, reconciled, unhealed };
    }
  } catch (e) {
    console.error(JSON.stringify({ evt: "voice_dualwrite_canary_error", phase: "heal", error: String(e && (e.stack || e.message) || e) }));
    try { await conn.end(); } catch { /* ignore */ }
    return { ok: false };
  }

  try {
    for (const [domain, s] of Object.entries(summary)) {
      console.log(JSON.stringify({ evt: "voice_dualwrite_check", domain, ...s }));
      await cw.send(new PutMetricDataCommand({
        Namespace: "Trepo/Capture",
        MetricData: [
          { MetricName: "VoiceDualWriteMiss", Dimensions: [{ Name: "Domain", Value: domain }], Value: s.detected, Unit: "Count" },
          { MetricName: "VoiceDualWriteReconciled", Dimensions: [{ Name: "Domain", Value: domain }], Value: s.reconciled, Unit: "Count" },
          { MetricName: "VoiceDualWriteUnhealed", Dimensions: [{ Name: "Domain", Value: domain }], Value: s.unhealed, Unit: "Count" },
        ],
      }));
    }
    const totals = Object.values(summary).reduce((a, s) => ({
      detected: a.detected + s.detected, reconciled: a.reconciled + s.reconciled, unhealed: a.unhealed + s.unhealed,
    }), { detected: 0, reconciled: 0, unhealed: 0 });
    console.log(JSON.stringify({ evt: "voice_dualwrite_summary", ...totals, per_domain: summary }));
    return { ok: true, ...totals };
  } catch (e) {
    console.error(JSON.stringify({ evt: "voice_dualwrite_canary_error", phase: "emit", error: String(e && (e.stack || e.message) || e) }));
    return { ok: false };
  } finally {
    try { await conn.end(); } catch { /* ignore */ }
  }
};
