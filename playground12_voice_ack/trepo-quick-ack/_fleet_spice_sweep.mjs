// Fleet sweep: find shared_kitchen items currently mislabeled prepared_other/pantry
// whose product_name — run through the NEW normalizer (spice hard-override) — now
// yields 'spices'. READ-ONLY by default: prints COUNT + a ~30-row sample (owner,
// name, old->new) and the exact UPDATE it WOULD run. Nothing is mutated unless you
// pass --apply (which then backs up every touched row first, same as _fleet_repair).
//
//   node _fleet_spice_sweep.mjs            # dry-run report (default, safe)
//   node _fleet_spice_sweep.mjs --apply    # backup + batched UPDATE (only after review)
//
// Env: DB_HOST DB_PORT DB_USER DB_PASS DB_NAME (same as _fleet_repair.mjs).
import mysql from "mysql2/promise";
import fs from "fs";
import { normalizeKitchenCategory } from "./lib/data-access.mjs";

const APPLY = process.argv.includes("--apply");
const conn = await mysql.createConnection({
  host: process.env.DB_HOST, port: +process.env.DB_PORT,
  user: process.env.DB_USER, password: process.env.DB_PASS, database: process.env.DB_NAME,
});

// Only the two enum buckets spices leak into (per the bug report). We re-run the
// normalizer in-process so the sample reflects the EXACT logic the fix ships.
const [rows] = await conn.query(
  `SELECT _id, owner_id, product_name, category, storage_location
     FROM shared_kitchen
    WHERE category IN ('prepared_other','pantry')
      AND action = 'IN'`);

const hits = [];
for (const r of rows) {
  const nc = normalizeKitchenCategory(r.category, r.storage_location, r.product_name);
  if (nc === "spices" && nc !== r.category) {
    hits.push({ id: r._id, owner: r.owner_id, name: r.product_name, old: r.category, nc });
  }
}

console.log(`scanned ${rows.length} prepared_other/pantry rows`);
console.log(`WOULD RE-CATEGORIZE TO 'spices': ${hits.length}`);
// break down old category + owners touched
const byOld = hits.reduce((m, h) => ((m[h.old] = (m[h.old] || 0) + 1), m), {});
console.log("by old category:", JSON.stringify(byOld));
console.log("distinct owners affected:", new Set(hits.map((h) => h.owner)).size);
console.log("\n--- sample (up to 30) ---");
for (const h of hits.slice(0, 30)) {
  console.log(`${(h.owner || "").slice(0, 8)}  ${String(h.name).slice(0, 48).padEnd(48)}  ${h.old} -> ${h.nc}`);
}

// Full candidate list to disk for review before any apply.
fs.writeFileSync("/tmp/fleet_spice_candidates.json", JSON.stringify(hits, null, 1));
console.log("\nfull candidate list -> /tmp/fleet_spice_candidates.json");

// The exact UPDATE this sweep WOULD run (shown, not executed in dry-run). Guarded on
// the old category so a concurrent change can't be clobbered.
console.log("\n--- UPDATE that --apply will run (per row, batched in a txn) ---");
console.log("UPDATE shared_kitchen SET category='spices', _updatedDate=NOW()");
console.log("  WHERE _id = ? AND (category <=> ? /* old */) AND category IN ('prepared_other','pantry') AND action='IN';");

if (!APPLY) {
  console.log("\nDRY RUN — no rows changed. Re-run with --apply after review to execute.");
  await conn.end();
  process.exit(0);
}

// ---- APPLY PATH (only with --apply) — backup first, then batched guarded UPDATE ----
const BK = "zz_spice_recat_backup_20260720";
await conn.query(`CREATE TABLE IF NOT EXISTS \`${BK}\` (
  _id VARCHAR(64), owner_id VARCHAR(64), product_name VARCHAR(500), old_category VARCHAR(255),
  storage_location VARCHAR(255), new_category VARCHAR(64), repaired_at DATETIME, KEY(_id))`);
const [existing] = await conn.query(`SELECT COUNT(*) n FROM \`${BK}\``);
if (existing[0].n > 0) { console.log("backup table already populated — aborting to avoid double-run"); process.exit(1); }
for (let i = 0; i < hits.length; i += 500) {
  await conn.query(
    `INSERT INTO \`${BK}\` (_id,owner_id,product_name,old_category,storage_location,new_category,repaired_at) VALUES ?`,
    [hits.slice(i, i + 500).map((h) => [h.id, h.owner, String(h.name || "").slice(0, 500), h.old, null, h.nc, new Date()])]);
}
console.log("backup rows written:", hits.length, "-> table", BK);
let done = 0;
for (let i = 0; i < hits.length; i += 250) {
  const chunk = hits.slice(i, i + 250);
  await conn.beginTransaction();
  for (const h of chunk) {
    await conn.query(
      "UPDATE shared_kitchen SET category='spices', _updatedDate=NOW() WHERE _id=? AND (category <=> ?) AND category IN ('prepared_other','pantry') AND action='IN'",
      [h.id, h.old]);
  }
  await conn.commit();
  done += chunk.length;
  await new Promise((r) => setTimeout(r, 250)); // polite
}
console.log("UPDATED rows:", done, "| backup table:", BK);
await conn.end();
