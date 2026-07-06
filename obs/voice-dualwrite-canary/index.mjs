// trepo-voice-dualwrite-canary — the "never again" guard for the WRITE_SHARED_ONLY
// dual-write class. Every 6h (EventBridge), find voice-written rows in the last 48h
// whose per-user, app-visible copy is MISSING, across three domains:
//   shopping  — shared_shopping_list (_device='voice-assistant') household_item_uuid
//               present in >=1 household member's {member}_new_list
//   dishes    — shared_dishes (_device='voice-assistant') _id in {user_id}_dishes
//   discards  — shared_discards (_device LIKE 'voice%') _id in the owner's {owner}_discards
// Emits Trepo/Capture:VoiceDualWriteMiss (Sum, dimension Domain=...) and logs the
// missing rows. On INTERNAL failure it emits NOTHING and logs loudly — never a 0
// (which would mask a real miss) — so a broken canary doesn't hide a regression.

import mysql from "mysql2/promise";
import { CloudWatchClient, PutMetricDataCommand } from "@aws-sdk/client-cloudwatch";

const cw = new CloudWatchClient({ region: process.env.AWS_REGION || "us-east-1" });
const LOOKBACK_HOURS = 48;
const MAX_ROWS_LOGGED = 25;

async function tableExists(conn, name) {
  const [r] = await conn.execute(
    "SELECT COUNT(*) c FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = ?",
    [name]
  );
  return r[0].c > 0;
}

async function householdMembers(conn, userId) {
  const [r] = await conn.execute("SELECT owner_id FROM new_users WHERE user_id = ? LIMIT 1", [userId]);
  const household = r[0]?.owner_id;
  if (!household) return [userId];
  const [m] = await conn.execute("SELECT user_id FROM new_users WHERE owner_id = ?", [household]);
  const ids = m.map((x) => x.user_id).filter(Boolean);
  return ids.length ? [...new Set(ids)] : [userId];
}

async function checkDishes(conn) {
  const [rows] = await conn.execute(
    `SELECT _id, user_id, owner_id FROM shared_dishes WHERE _device = 'voice-assistant' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR`
  );
  const misses = [];
  for (const r of rows) {
    const owner = r.user_id || r.owner_id;
    const table = `${owner}_dishes`;
    if (!(await tableExists(conn, table))) { misses.push({ _id: r._id, owner, reason: "no_per_user_table" }); continue; }
    const [c] = await conn.execute(`SELECT COUNT(*) c FROM \`${table}\` WHERE _id = ?`, [r._id]);
    if (c[0].c === 0) misses.push({ _id: r._id, owner });
  }
  return misses;
}

async function checkDiscards(conn) {
  const [rows] = await conn.execute(
    `SELECT _id, user_id, owner_id FROM shared_discards WHERE _device LIKE 'voice%' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR`
  );
  const misses = [];
  for (const r of rows) {
    const owner = r.owner_id || r.user_id;
    const table = `${owner}_discards`;
    if (!(await tableExists(conn, table))) { misses.push({ _id: r._id, owner, reason: "no_per_user_table" }); continue; }
    const [c] = await conn.execute(`SELECT COUNT(*) c FROM \`${table}\` WHERE _id = ?`, [r._id]);
    if (c[0].c === 0) misses.push({ _id: r._id, owner });
  }
  return misses;
}

async function checkShopping(conn) {
  const [rows] = await conn.execute(
    `SELECT household_item_uuid, owner_id, _owner FROM shared_shopping_list WHERE _device = 'voice-assistant' AND _createdDate > NOW() - INTERVAL ${LOOKBACK_HOURS} HOUR`
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
    if (!found) misses.push({ household_item_uuid: r.household_item_uuid, owner });
  }
  return misses;
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
    // Internal failure — emit NOTHING (do not mask a real miss with a 0), log loudly.
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

  try {
    for (const [domain, misses] of Object.entries(results)) {
      console.log(JSON.stringify({ evt: "voice_dualwrite_check", domain, missing: misses.length, rows: misses.slice(0, MAX_ROWS_LOGGED) }));
      await cw.send(new PutMetricDataCommand({
        Namespace: "Trepo/Capture",
        MetricData: [{ MetricName: "VoiceDualWriteMiss", Dimensions: [{ Name: "Domain", Value: domain }], Value: misses.length, Unit: "Count" }],
      }));
    }
    const total = Object.values(results).reduce((a, m) => a + m.length, 0);
    console.log(JSON.stringify({ evt: "voice_dualwrite_summary", total_missing: total, per_domain: Object.fromEntries(Object.entries(results).map(([d, m]) => [d, m.length])) }));
    return { ok: true, total_missing: total };
  } catch (e) {
    console.error(JSON.stringify({ evt: "voice_dualwrite_canary_error", phase: "emit", error: String(e && (e.stack || e.message) || e) }));
    return { ok: false };
  } finally {
    try { await conn.end(); } catch { /* ignore */ }
  }
};
