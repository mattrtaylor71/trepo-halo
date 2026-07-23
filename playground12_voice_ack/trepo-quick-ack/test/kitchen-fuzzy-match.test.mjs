// P4 (audit #2): the kitchen discard/resolve matcher was exact + LIKE only, so
// plurals / variants / brand phrasings ("raspberries", "broccoli", "second ground
// beef", "sour cream", "brie", "bagels") returned a flat "not found". This pins the
// F-040 fuzzy fallback now wired into findKitchenRowsByName (reuses the shopping
// scorer scoreShoppingItemMatch) plus the "did you mean?" candidates on a miss.
//
// No live DB: withDbConnection is module-mocked with a connection whose .execute()
// answers the three kitchen SELECT shapes (exact / LIKE / full-inventory) from a
// canned inventory, so the fuzzy path is exercised end-to-end through the real ops.
// Run: node --experimental-test-module-mocks --test

import { test, mock, before, beforeEach } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const LIB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "lib");
const MYSQL = path.join(LIB, "mysql.mjs");
const DATA_ACCESS = path.join(LIB, "data-access.mjs");

// The canned active kitchen. product_name is what the fuzzy scorer matches against.
function inventory() {
  return [
    { _id: "r-rasp", owner_id: "ownerA", _owner: "ownerA", product_name: "Raspberry", action: "IN", category: "produce", storage_location: "fridge" },
    { _id: "r-broc", owner_id: "ownerA", _owner: "ownerA", product_name: "Broccoli", action: "IN", category: "produce" },
    { _id: "r-beef", owner_id: "ownerA", _owner: "ownerA", product_name: "Ground Beef", action: "IN", category: "meat_seafood" },
    { _id: "r-sour", owner_id: "ownerA", _owner: "ownerA", product_name: "Sour Cream", action: "IN", category: "dairy_eggs" },
    { _id: "r-brie", owner_id: "ownerA", _owner: "ownerA", product_name: "Brie Cheese", action: "IN", category: "dairy_eggs" },
    { _id: "r-bagel", owner_id: "ownerA", _owner: "ownerA", product_name: "Bagel", action: "IN", category: "pantry" },
    { _id: "r-milk", owner_id: "ownerA", _owner: "ownerA", product_name: "Whole Milk", action: "IN", category: "dairy_eggs" },
  ];
}

// Classify a kitchen SELECT: exact (LOWER(TRIM(...)) = ?), like (LIKE ?), or the
// full-inventory read (neither). Returns rows accordingly. UPDATE/INSERT/DDL/count
// probes get benign answers so discard fan-out can run.
function makeConn(rows) {
  const calls = [];
  const conn = {
    calls,
    async execute(sql, params = []) {
      const norm = String(sql).replace(/\s+/g, " ").trim();
      calls.push({ sql: norm, params });
      const u = norm.toUpperCase();
      if (u.includes("INFORMATION_SCHEMA.TABLES")) return [[{ count: 1 }]];
      if (u.includes("INFORMATION_SCHEMA.COLUMNS")) return [[]];
      if (u.startsWith("CREATE") || u.startsWith("ALTER")) return [{}];
      if (u.startsWith("INSERT")) return [{ insertId: 900, affectedRows: 1 }];
      if (u.startsWith("UPDATE") || u.startsWith("DELETE")) return [{ affectedRows: 1 }];
      if (u.startsWith("SELECT")) {
        if (/COUNT\(/i.test(norm) || /SELECT\s+1\b/i.test(norm)) return [[{ count: 1 }]];
        // Only kitchen reads (against shared_kitchen here) return inventory.
        const isKitchen = /SHARED_KITCHEN|`?SHARED_KITCHEN`?/i.test(norm) || /FROM `shared_kitchen`/i.test(norm);
        if (!isKitchen) return [[]];
        const isExact = /LOWER\(TRIM\(PRODUCT_NAME\)\)\s*=\s*\?/i.test(norm);
        const isLike = /LIKE\s*\?/i.test(norm);
        if (isExact) {
          const target = String(params[params.length - 1] || "").toLowerCase().trim();
          return [rows.filter((r) => String(r.product_name).toLowerCase().trim() === target)];
        }
        if (isLike) {
          // Emulate MySQL LIKE with the buildLikePattern (%a%b%) — spaces became %.
          const pat = String(params[params.length - 1] || "").toLowerCase();
          const parts = pat.split("%").filter(Boolean);
          return [rows.filter((r) => {
            const name = String(r.product_name).toLowerCase();
            let idx = 0;
            for (const p of parts) { const at = name.indexOf(p, idx); if (at < 0) return false; idx = at + p.length; }
            return true;
          })];
        }
        // Full-inventory read (getKitchenRowsFull / getKitchenRows).
        return [rows];
      }
      return [[]];
    },
    release() {},
  };
  return conn;
}

let da;
let CONN;
before(async () => {
  process.env.WRITE_SHARED_ONLY = "true";
  process.env.DB_HOST = process.env.DB_HOST || "test";
  process.env.DB_USER = process.env.DB_USER || "test";
  process.env.DB_PASS = process.env.DB_PASS || "test";
  process.env.DB_NAME = process.env.DB_NAME || "test";
  const realMysql = await import(MYSQL);
  mock.module(MYSQL, {
    namedExports: {
      ...realMysql,
      withDbConnection: async (cb) => cb(CONN),
    },
  });
  da = await import(DATA_ACCESS);
});

beforeEach(() => {
  CONN = makeConn(inventory());
});

function ctx() {
  return { ownerId: "ownerA", userId: "ownerA", tableOwnerId: "ownerA", householdMemberIds: ["ownerA"], responseSurface: "halo" };
}

// ── searchKitchenItem: fuzzy resolution of the audit repros ──────────────
test("plural 'raspberries' resolves to the 'Raspberry' kitchen row", async () => {
  const res = await da.searchKitchenItem(ctx(), { item_name: "raspberries" }, { skipKitchenDependentGeneration: true });
  assert.equal(res.found, true, "raspberries should resolve to Raspberry");
  assert.equal(res.item.item_name, "Raspberry");
});

test("'broccoli', 'sour cream', 'brie', 'bagels' all resolve", async () => {
  for (const [name, expected] of [
    ["broccoli", "Broccoli"],
    ["sour cream", "Sour Cream"],
    ["brie", "Brie Cheese"],
    ["bagels", "Bagel"],
  ]) {
    CONN = makeConn(inventory());
    const res = await da.searchKitchenItem(ctx(), { item_name: name }, { skipKitchenDependentGeneration: true });
    assert.equal(res.found, true, `${name} should resolve`);
    assert.equal(res.item.item_name, expected, `${name} -> ${expected}`);
  }
});

test("'second ground beef' resolves to the 'Ground Beef' row", async () => {
  const res = await da.searchKitchenItem(ctx(), { item_name: "second ground beef" }, { skipKitchenDependentGeneration: true });
  assert.equal(res.found, true);
  assert.equal(res.item.item_name, "Ground Beef");
});

// ── discardKitchenItem: the audit's throwaway repros land the write ──────
test("'Throw away one packet of broccoli' discards the Broccoli row", async () => {
  const res = await da.discardKitchenItem(ctx(), { item_name: "broccoli" }, "finished", { skipKitchenDependentGeneration: true });
  assert.equal(res.item.item_name, "Broccoli");
  // The discard flips the kitchen row to action=OUT.
  const flippedToOut = CONN.calls.some((c) => c.sql.toUpperCase().startsWith("UPDATE") && /ACTION/i.test(c.sql));
  assert.ok(flippedToOut, "expected an UPDATE flipping the row to OUT");
});

test("'Mark the sour cream as opened' resolves when a sour-cream row exists", async () => {
  const res = await da.markKitchenItemOpened(ctx(), { item_name: "the sour cream" }, { skipKitchenDependentGeneration: true });
  assert.equal(res.item_name, "Sour Cream");
});

// ── no-match returns candidates (did-you-mean), not a bare not-found ──────
test("a genuine no-match surfaces the closest kitchen items as candidates", async () => {
  // "dragon fruit" is not in the inventory and is far from every row.
  const res = await da.searchKitchenItem(ctx(), { item_name: "dragon fruit" }, { skipKitchenDependentGeneration: true });
  assert.equal(res.found, false, "dragon fruit must NOT falsely resolve");
  assert.ok(Array.isArray(res.candidates) && res.candidates.length > 0, "should offer candidate items");
  assert.ok(res.candidates.every((c) => typeof c.item_name === "string"), "candidates carry item names");
});

test("discard of a missing item throws not-found carrying candidates", async () => {
  await assert.rejects(
    () => da.discardKitchenItem(ctx(), { item_name: "dragon fruit" }, "finished", { skipKitchenDependentGeneration: true }),
    (err) => {
      assert.equal(err.statusCode, 404, "not-found status");
      assert.ok(err.details && Array.isArray(err.details.candidates) && err.details.candidates.length > 0, "not-found error must carry candidates");
      return true;
    }
  );
});

// ── conservative: an obvious mismatch does NOT silently resolve ──────────
test("does not silently substitute across an obvious mismatch (milk !-> unrelated)", async () => {
  // "almond flour" is not present; the only vaguely-similar row is Whole Milk, but
  // the scorer is well below threshold, so it must miss (a wrong discard is worse).
  const res = await da.searchKitchenItem(ctx(), { item_name: "almond flour" }, { skipKitchenDependentGeneration: true });
  assert.equal(res.found, false, "almond flour must not resolve to any dairy row");
});
