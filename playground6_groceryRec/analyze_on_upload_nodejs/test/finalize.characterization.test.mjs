// Characterization tests for the AnalyzeOnUpload finalize/tombstone logic
// (mysqlWriter.js) — the phantom-check-in fix invariants:
//   - getManuallyDeletedIds tombstones BOTH 'manual_delete' AND 'deleted'
//   - finalizeKitchenRow SKIPS (returns false) when the item was user-deleted
//   - finalizeKitchenRow returns false when the upsert affects 0 rows (never a lie)
//   - finalizeKitchenRow returns true on a real write
// No live DB: mysql2/promise.createConnection is module-mocked with a spy connection.
// Run: node --experimental-test-module-mocks --test

import { test, mock, before } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WRITER = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist", "utils", "mysqlWriter.js");

// archivedReason: if set, the archive SELECT reports the entry as tombstoned with
// that reason. insertAffected: affectedRows the shared_kitchen upsert reports.
let SCENARIO = { archivedReason: null, insertAffected: 1, calls: [] };

function makeConn() {
  return {
    async execute(sql, params = []) {
      const norm = String(sql).replace(/\s+/g, " ").trim();
      SCENARIO.calls.push(norm);
      const u = norm.toUpperCase();
      if (u.includes("SHARED_ARCHIVE_KITCHEN")) {
        // getManuallyDeletedIds: echo the queried ids as archived iff the SQL asks
        // for this reason and the scenario has a tombstone.
        const asksDeleted = /'MANUAL_DELETE', 'DELETED'|'DELETED'|'MANUAL_DELETE'/.test(u);
        if (SCENARIO.archivedReason && asksDeleted) {
          return [(params || []).map((p) => ({ _id: p }))];
        }
        return [[]];
      }
      if (u.includes("INFORMATION_SCHEMA")) return [[{ count: 1 }]];
      if (u.startsWith("INSERT")) return [{ insertId: 1, affectedRows: SCENARIO.insertAffected }];
      if (u.startsWith("UPDATE") || u.startsWith("DELETE")) return [{ affectedRows: 1 }];
      return [[]];
    },
    async end() {},
  };
}

let finalizeKitchenRow;
before(async () => {
  for (const k of ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"]) process.env[k] = process.env[k] || "test";
  mock.module("mysql2/promise", { defaultExport: { createConnection: async () => makeConn() } });
  ({ finalizeKitchenRow } = await import(WRITER));
});

function record() {
  return {
    owner: "ownerA", device_id: "d", user_id: "ownerA", job_id: "job-1", action: "IN",
    quantity: 1, s3_key: "k", image_url: "http://img", groceryItem: { product_name: "Merlot Wine" },
  };
}

test("finalizeKitchenRow returns true and upserts shared_kitchen on a real write", async () => {
  SCENARIO = { archivedReason: null, insertAffected: 1, calls: [] };
  const landed = await finalizeKitchenRow(record());
  assert.equal(landed, true);
  assert.ok(SCENARIO.calls.some((c) => /INSERT INTO `shared_kitchen`/i.test(c)), "shared_kitchen upsert missing");
});

test("finalizeKitchenRow SKIPS (false) when the item was user-deleted (archived_reason='deleted')", async () => {
  SCENARIO = { archivedReason: "deleted", insertAffected: 1, calls: [] };
  const landed = await finalizeKitchenRow(record());
  assert.equal(landed, false, "must not resurrect a user-deleted item");
  assert.ok(!SCENARIO.calls.some((c) => /INSERT INTO `shared_kitchen`/i.test(c)), "must NOT upsert when tombstoned");
});

test("finalizeKitchenRow SKIPS (false) for legacy 'manual_delete' tombstone too", async () => {
  SCENARIO = { archivedReason: "manual_delete", insertAffected: 1, calls: [] };
  const landed = await finalizeKitchenRow(record());
  assert.equal(landed, false);
});

test("finalizeKitchenRow returns false when the upsert affects 0 rows (Finalized-N can't be a lie)", async () => {
  SCENARIO = { archivedReason: null, insertAffected: 0, calls: [] };
  const landed = await finalizeKitchenRow(record());
  assert.equal(landed, false);
});
