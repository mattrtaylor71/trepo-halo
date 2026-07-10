// Characterization tests for the quick-ack data-access dual-write ops.
// These pin the CURRENT behavior (shared table + per-user/member fan-out, with the
// right SCOPE: dishes = user-private, shopping/discards = household) so the item-4
// dualWriteAcrossHousehold() helper extraction can be proven behavior-preserving.
//
// No live DB: withDbConnection is module-mocked to hand the ops a spy connection
// whose .execute() records every SQL and answers information_schema / INSERT /
// UPDATE / DELETE / SELECT shapes. Run: node --experimental-test-module-mocks --test
//
// getShoppingHouseholdMemberIds / getTableHouseholdMemberIds are pure (read the
// context) so household membership is controlled by the test context, not mocked.

import { test, mock, before } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const LIB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "lib");
const MYSQL = path.join(LIB, "mysql.mjs");
const DATA_ACCESS = path.join(LIB, "data-access.mjs");
const ANALYZER = path.resolve(LIB, "..", "..", "..", "shared", "voice-assistant", "text-dish-analyzer.mjs");

// ---- mock connection ---------------------------------------------------------
// selectRows: canned rows returned for SELECTs (other than information_schema),
// used by lookup-then-mutate ops (clear/remove).
function makeConn({ columns = ["_id", "_owner", "_device", "product_name", "quantity", "store", "action", "household_item_uuid", "owner_id", "_createdDate", "_updatedDate", "created_at", "updated_at", "source_kitchen_id", "discard_reason", "dish_name", "user_id"], selectRows = [] } = {}) {
  const calls = [];
  const conn = {
    calls,
    async execute(sql, params = []) {
      const norm = String(sql).replace(/\s+/g, " ").trim();
      calls.push({ sql: norm, params });
      const u = norm.toUpperCase();
      if (u.includes("INFORMATION_SCHEMA.TABLES")) return [[{ count: 1 }]];
      if (u.includes("INFORMATION_SCHEMA.COLUMNS")) return [columns.map((c) => ({ column_name: c }))];
      if (u.startsWith("INSERT")) return [{ insertId: 101, affectedRows: 1 }];
      if (u.startsWith("UPDATE") || u.startsWith("DELETE")) return [{ affectedRows: 1 }];
      if (u.startsWith("SELECT")) return [Array.isArray(selectRows) ? selectRows : []];
      return [[]];
    },
    release() {},
  };
  return conn;
}

const VERB_RE = /^(INSERT INTO|UPDATE|DELETE FROM)\s+`([^`]+)`/i;
function targets(conn, verb) {
  return conn.calls
    .filter((c) => c.sql.toUpperCase().startsWith(verb))
    .map((c) => (c.sql.match(VERB_RE) || [])[2])
    .filter(Boolean);
}

// ---- module mock + import ----------------------------------------------------
let da;
let CONN = makeConn();
before(async () => {
  process.env.WRITE_SHARED_ONLY = "true"; // exercise shared insert + per-user fall-through
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
  // Stub the LLM dish analyzer so dish ops exercise only the DB dual-write path.
  mock.module(ANALYZER, {
    namedExports: {
      analyzeDishFromText: async (req) => ({
        dish_name: req?.dish_name || "Test Dish",
        serving_size: null, calories: 100, total_fat: 1, total_carbohydrates: 10, protein: 5,
        confidence: 0.9, explanation: null, ingredients: req?.ingredients || [], components: [],
        allergens: [], source_type: "llm_estimate", evidence_urls: [],
      }),
    },
  });
  da = await import(DATA_ACCESS);
});

// Household context: owner A + member B. Dishes are user-private (only userId's table).
function householdCtx() {
  return { ownerId: "ownerA", userId: "ownerA", tableOwnerId: "ownerA", householdMemberIds: ["ownerA", "memberB"] };
}

// ---- SHOPPING: household scope ----------------------------------------------
test("addShoppingItem writes shared_shopping_list + BOTH members' _new_list", async () => {
  CONN = makeConn();
  await da.addShoppingItem(householdCtx(), "Bananas", null, 1);
  const inserts = targets(CONN, "INSERT");
  assert.ok(inserts.includes("shared_shopping_list"), "shared insert missing");
  assert.ok(inserts.includes("ownerA_new_list"), "owner per-user insert missing");
  assert.ok(inserts.includes("memberB_new_list"), "member per-user insert missing (household scope)");
});

test("addManyShoppingItems fans out EVERY item to shared + both members", async () => {
  CONN = makeConn();
  await da.addManyShoppingItems(householdCtx(), [{ item_name: "Milk" }, { item_name: "Eggs" }]);
  const inserts = targets(CONN, "INSERT");
  assert.equal(inserts.filter((t) => t === "shared_shopping_list").length, 2, "2 shared inserts");
  assert.equal(inserts.filter((t) => t === "ownerA_new_list").length, 2, "2 owner inserts");
  assert.equal(inserts.filter((t) => t === "memberB_new_list").length, 2, "2 member inserts");
});

test("clearShoppingList clears shared + both members (household scope)", async () => {
  CONN = makeConn({ selectRows: [{ _id: 1, product_name: "Bananas", household_item_uuid: "u1" }] });
  await da.clearShoppingList(householdCtx());
  const mutated = [...targets(CONN, "DELETE"), ...targets(CONN, "UPDATE")];
  assert.ok(mutated.some((t) => t === "shared_shopping_list"), "shared not cleared");
  assert.ok(mutated.some((t) => t === "ownerA_new_list"), "owner not cleared");
  assert.ok(mutated.some((t) => t === "memberB_new_list"), "member not cleared");
});

// ---- DISHES: user-private scope (NOT the other household member) -------------
test("logDishIngredients writes shared_dishes + ONLY the user's _dishes (user-private)", async () => {
  CONN = makeConn();
  await da.logDishIngredients(householdCtx(), { dish_name: "Oatmeal", ingredients: ["oats"] });
  const inserts = targets(CONN, "INSERT");
  assert.ok(inserts.includes("shared_dishes"), "shared_dishes insert missing");
  assert.ok(inserts.includes("ownerA_dishes"), "user _dishes insert missing");
  assert.ok(!inserts.includes("memberB_dishes"), "dishes must NOT fan out to other members (user-private)");
});

// ---- DISCARDS: household scope ----------------------------------------------
test("clearRecentDiscards clears shared + both members (household scope)", async () => {
  CONN = makeConn({ selectRows: [{ _id: "d1", product_name: "Old Milk", action: "IN" }] });
  await da.clearRecentDiscards(householdCtx());
  const mutated = [...targets(CONN, "UPDATE"), ...targets(CONN, "DELETE")];
  assert.ok(mutated.some((t) => t === "shared_discards"), "shared_discards not cleared");
  assert.ok(mutated.some((t) => t === "ownerA_discards"), "owner _discards not cleared");
  assert.ok(mutated.some((t) => t === "memberB_discards"), "member _discards not cleared (household scope)");
});
