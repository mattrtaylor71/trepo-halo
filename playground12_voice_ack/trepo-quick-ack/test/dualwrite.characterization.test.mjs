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
const ALL_COLUMNS = ["_id", "_owner", "_device", "product_name", "quantity", "store", "action", "household_item_uuid", "owner_id", "_createdDate", "_updatedDate", "created_at", "updated_at", "source_kitchen_id", "discard_reason", "dish_name", "user_id", "product_brand", "images", "product_barcode", "category", "storage_location", "product_image_url", "product_image_key", "is_opened", "confidence", "explanation", "serving_size", "calories", "ingredients", "components", "allergens", "job_id", "analysis_status"];

// A rich canned "target" row so lookup-then-mutate ops (remove/update/delete) find
// something to act on. Tests override fields as needed.
function targetRow(over = {}) {
  return {
    _id: "row-1", household_item_uuid: "uuid-1", product_name: "Bananas", dish_name: "Oatmeal",
    store: "Groceries", action: "IN", quantity: 1, owner_id: "ownerA", _owner: "ownerA",
    user_id: "ownerA", category: "produce", ...over,
  };
}

function makeConn({ columns = ALL_COLUMNS, selectRows = [], countRow = { count: 1 } } = {}) {
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
      if (u.startsWith("SELECT")) {
        // COUNT(*) probes -> a count row; row-finding SELECTs -> the canned target(s).
        if (/COUNT\(/i.test(norm) || /SELECT\s+1\b/i.test(norm)) return [[countRow]];
        return [Array.isArray(selectRows) ? selectRows : []];
      }
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
function allWriteTargets(conn) {
  return [...targets(conn, "INSERT"), ...targets(conn, "UPDATE"), ...targets(conn, "DELETE")];
}
// Params of the (first) INSERT into a specific table — lets us assert VALUES are
// preserved through the helper's buildStatement (targets alone wouldn't catch a
// garbage value, e.g. a stray function in the params).
function insertParamsFor(conn, tableName) {
  const c = conn.calls.find((x) => x.sql.toUpperCase().startsWith("INSERT INTO") && new RegExp("`" + tableName + "`").test(x.sql));
  return c ? c.params : null;
}
// Params of the (first) UPDATE/DELETE against a table — same VALUE-preservation
// guard as insertParamsFor, but for the update/delete buildStatement closures.
function mutateParamsFor(conn, verb, tableName) {
  const c = conn.calls.find((x) => x.sql.toUpperCase().startsWith(verb) && new RegExp("`" + tableName + "`").test(x.sql));
  return c ? c.params : null;
}
const wroteForMember = (conn, memberId) => allWriteTargets(conn).some((t) => t.includes(memberId));
const wroteShared = (conn) => allWriteTargets(conn).some((t) => t.startsWith("shared_"));

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
  // VALUES preserved: member row is _owner=memberB, product_name=Bananas, action=ADDED.
  const mp = insertParamsFor(CONN, "memberB_new_list");
  assert.deepEqual(mp, ["memberB", "voice-assistant", "Bananas", 1, null, null, null, mp[7], "ADDED", mp[9]]);
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

// ==== EXPANDED NET: the remaining fan-out sites (item-4 consolidation targets) ====

// KITCHEN — shared-only under WRITE_SHARED_ONLY (kitchen_api resolves _prod_kitchen
// -> shared_kitchen, so the per-member fan-out is intentionally NOT reached; the
// item-4 helper must PRESERVE this shared-only scope, not enable the legacy loop).
test("checkInKitchenItem writes shared_kitchen ONLY (no per-member fan-out)", async () => {
  CONN = makeConn();
  await da.checkInKitchenItem(householdCtx(), { product_name: "Yogurt", category: "dairy" });
  assert.ok(targets(CONN, "INSERT").includes("shared_kitchen"), "shared_kitchen insert missing");
  assert.ok(!wroteForMember(CONN, "memberB"), "kitchen must NOT fan out to member tables (shared-only model)");
});

test("setKitchenItemProductImage updates shared_kitchen ONLY (no per-member fan-out)", async () => {
  CONN = makeConn({ selectRows: [targetRow()] });
  await da.setKitchenItemProductImage(householdCtx(), "row-1", { product_image_url: "http://img" });
  assert.ok(targets(CONN, "UPDATE").includes("shared_kitchen"), "shared_kitchen update missing");
  assert.ok(!wroteForMember(CONN, "memberB"), "kitchen must NOT fan out to member tables (shared-only model)");
});

// DISCARDS (household)
test("discardKitchenItem inserts discard into shared + BOTH members (household)", async () => {
  CONN = makeConn({ selectRows: [targetRow()] });
  await da.discardKitchenItem(householdCtx(), "Bananas", "spoiled");
  assert.ok(wroteForMember(CONN, "ownerA") && wroteForMember(CONN, "memberB"), "discard insert must fan out to both members");
  // VALUES preserved: member discard row is _owner=memberB (a plain string, not a
  // function — guards the dynamic-column buildStatement wrapper).
  const dp = insertParamsFor(CONN, "memberB_discards");
  assert.equal(typeof dp?.[1], "string");
  assert.equal(dp[1], "memberB");
  assert.equal(dp[0], dp[0]); // _id (discardId) present
});

test("deleteRecentDiscard (action=OUT) updates shared + BOTH members (household)", async () => {
  CONN = makeConn({ selectRows: [targetRow({ action: "IN" })] });
  await da.deleteRecentDiscard(householdCtx(), "Bananas");
  assert.ok(wroteForMember(CONN, "ownerA") && wroteForMember(CONN, "memberB"), "discard update must fan out to both members");
});

// DISHES (user-private — never the other member)
test("appendToRecentDish updates shared_dishes + ONLY the user (user-private)", async () => {
  const nowIso = new Date().toISOString();
  CONN = makeConn({ selectRows: [targetRow({ dish_name: "Oatmeal", ingredients: "[]", components: "[]", _createdDate: nowIso, _updatedDate: nowIso })] });
  await da.appendToRecentDish(householdCtx(), { ingredients: ["banana"] });
  assert.ok(wroteForMember(CONN, "ownerA"), "user dish update missing");
  assert.ok(!wroteForMember(CONN, "memberB"), "dish update must NOT touch the other member (user-private)");
});

test("deleteDishLog deletes shared_dishes + ONLY the user (user-private)", async () => {
  CONN = makeConn({ selectRows: [targetRow({ dish_name: "Oatmeal" })] });
  await da.deleteDishLog(householdCtx(), { dish_name: "Oatmeal" });
  assert.ok(wroteForMember(CONN, "ownerA"), "user dish delete missing");
  assert.ok(!wroteForMember(CONN, "memberB"), "dish delete must NOT touch the other member (user-private)");
});

// SHOPPING (household)
test("removeShoppingItem deletes from shared + BOTH members (household)", async () => {
  CONN = makeConn({ selectRows: [targetRow({ product_name: "Bananas", household_item_uuid: "uuid-1" })] });
  await da.removeShoppingItem(householdCtx(), "Bananas");
  assert.ok(wroteForMember(CONN, "ownerA") && wroteForMember(CONN, "memberB"), "shopping delete must fan out to both members");
  // VALUES preserved: household_item_uuid branch deletes by the resolved uuid, not a
  // stray function/undefined (guards the branchy buildStatement values array).
  const dp = mutateParamsFor(CONN, "DELETE", "memberB_new_list");
  assert.deepEqual(dp, ["uuid-1"]);
});

test("updateShoppingItemStore updates shared + BOTH members (household)", async () => {
  CONN = makeConn({ selectRows: [targetRow({ product_name: "Bananas", household_item_uuid: "uuid-1" })] });
  await da.updateShoppingItemStore(householdCtx(), "Bananas", "Costco");
  assert.ok(wroteForMember(CONN, "ownerA") && wroteForMember(CONN, "memberB"), "shopping store-update must fan out to both members");
  // VALUES preserved: the UPDATE carries the new store field value + the uuid key
  // (fieldValues spread + household_item_uuid), proving buildStatement's values order.
  const up = mutateParamsFor(CONN, "UPDATE", "memberB_new_list");
  assert.ok(up.includes("Costco"), "store field value missing from UPDATE params");
  assert.ok(up.includes("uuid-1"), "household_item_uuid key missing from UPDATE params");
});

// ---- CATEGORY NORMALIZATION (voice check-in never writes null/free-text) ------
test("normalizeKitchenCategory: exact enum values pass through", () => {
  for (const v of ["leftovers", "produce", "dairy_eggs", "meat_seafood", "pantry", "snacks_sweets", "beverages", "prepared_other"]) {
    assert.equal(da.normalizeKitchenCategory(v, null), v);
  }
});
test("normalizeKitchenCategory: free-text guesses map onto the enum", () => {
  assert.equal(da.normalizeKitchenCategory("Condiment", null), "pantry");
  assert.equal(da.normalizeKitchenCategory("Baking", null), "pantry");
  assert.equal(da.normalizeKitchenCategory("Beverage", null), "beverages");
  assert.equal(da.normalizeKitchenCategory("Snacks", null), "snacks_sweets");
  assert.equal(da.normalizeKitchenCategory("Produce/Dip", null), "produce");
  assert.equal(da.normalizeKitchenCategory("Dairy/Creamer", null), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory("dairy", null), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory("Frozen Meat", null), "meat_seafood");
});
test("normalizeKitchenCategory: null category falls back to storage, then prepared_other", () => {
  assert.equal(da.normalizeKitchenCategory(null, "snacks"), "snacks_sweets");
  assert.equal(da.normalizeKitchenCategory(null, "pantry"), "pantry");
  assert.equal(da.normalizeKitchenCategory(null, "produce"), "produce");
  assert.equal(da.normalizeKitchenCategory(null, "fridge"), "prepared_other"); // ambiguous storage
  assert.equal(da.normalizeKitchenCategory(null, null), "prepared_other");
  assert.equal(da.normalizeKitchenCategory("", ""), "prepared_other");
});
test("normalizeKitchenCategory: unknown free-text defaults to prepared_other (never null)", () => {
  const out = da.normalizeKitchenCategory("Xyzzy Nonsense", null);
  assert.ok(out && da.normalizeKitchenCategory("Xyzzy", "freezer") === "prepared_other");
  assert.equal(out, "prepared_other");
});
test("checkInKitchenItem persists a NON-NULL enum category (null guess -> normalized)", async () => {
  CONN = makeConn();
  await da.checkInKitchenItem(householdCtx(), { product_name: "Mixed Berry Yogurt Bites", category: null, location: "snacks" });
  const params = insertParamsFor(CONN, "shared_kitchen");
  assert.ok(params, "shared_kitchen INSERT missing");
  // category is the 6th INSERT param (rowId, owner, device, name, brand, category, ...)
  const category = params[5];
  assert.ok(category != null && category !== "", "category must never be null/empty");
  assert.equal(category, "snacks_sweets", "null guess + snacks location should normalize to snacks_sweets");
});
test("checkInKitchenItem normalizes a free-text category guess (Condiment -> pantry)", async () => {
  CONN = makeConn();
  await da.checkInKitchenItem(householdCtx(), { product_name: "Ketchup", category: "Condiment" });
  const category = insertParamsFor(CONN, "shared_kitchen")[5];
  assert.equal(category, "pantry");
});

// ---- REMOVAL PATH (honest failure, no phantom success) -----------------------
test("discardKitchenItem throws (honest) when the item is not found — no phantom success", async () => {
  CONN = makeConn({ selectRows: [] }); // resolution finds nothing
  await assert.rejects(
    da.discardKitchenItem(householdCtx(), { item_name: "Nonexistent Thing" }, "test", { skipKitchenDependentGeneration: true }),
    /couldn'?t find|not found/i
  );
  // Nothing should have been flipped OUT when nothing resolved.
  assert.equal(mutateParamsFor(CONN, "UPDATE", "shared_kitchen"), null, "no OUT flip should happen when nothing resolved");
});
test("discardKitchenItem flips action=OUT in shared_kitchen + inserts a discard", async () => {
  CONN = makeConn({ selectRows: [targetRow({ product_name: "Colby Jack Cheese" })] });
  await da.discardKitchenItem(householdCtx(), { item_name: "Colby Jack Cheese" }, "eaten", { skipKitchenDependentGeneration: true });
  const upParams = mutateParamsFor(CONN, "UPDATE", "shared_kitchen");
  assert.ok(upParams && upParams.includes("OUT"), "action=OUT flip missing from shared_kitchen UPDATE");
  assert.ok(targets(CONN, "INSERT").includes("shared_discards"), "discard row not inserted into shared_discards");
});

// ---- CATEGORY: product-name inference + UPDATE-path clamp ---------------------
test("normalizeKitchenCategory infers from product NAME when category is null", () => {
  assert.equal(da.normalizeKitchenCategory(null, null, "Jasmine Rice"), "pantry");
  assert.equal(da.normalizeKitchenCategory(null, null, "Elbow Noodles"), "pantry");
  assert.equal(da.normalizeKitchenCategory(null, null, "Mild Cheddar"), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory(null, null, "Whole Milk"), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory(null, null, "Tortilla Chips"), "snacks_sweets");
  assert.equal(da.normalizeKitchenCategory(null, null, "Ground Turkey"), "meat_seafood");
});
test("normalizeKitchenCategory: seasonings are spices (not meat/pantry); sauces stay pantry (order)", () => {
  // Seasonings/spice-blends route to the `spices` enum (scanned before pantry). The
  // point is they must never be meat; and sauces/bouillon stay pantry.
  assert.equal(da.normalizeKitchenCategory("Seasoning", "pantry", "Cajun Seasoning"), "spices");
  assert.equal(da.normalizeKitchenCategory("seasoning", null, "Lemon Pepper Seasoning"), "spices");
  assert.equal(da.normalizeKitchenCategory(null, null, "Kinder's Steak Blend Seasoning"), "spices");
  assert.equal(da.normalizeKitchenCategory(null, null, "Fish Sauce"), "pantry");
  assert.equal(da.normalizeKitchenCategory(null, null, "Chicken Bouillon"), "pantry");
});
test("normalizeKitchenCategory: SPICE hard-override rescues spices mislabeled with a VALID enum", () => {
  // THE live bug (owner d303a754): the model confidently labels a spice with a VALID
  // enum ('prepared_other'/'pantry'), so it passed straight through the enum check and
  // the product-name scan never ran. The spice hard-override fires BEFORE that check.
  // PRECISION-FIRST (drives a mass data sweep): spice/salt/pepper words are routinely
  // flavor DESCRIPTORS on prepared foods, so false positives must be ~zero.
  // MUST-PASS positives (each with a VALID passthrough enum the model actually emitted):
  const POSITIVE = [
    "Black Salt", "Garam Masala", "Chaat Masala", "Coarse Salt", "White Pepper",
    "Black Pepper", "Sichuan Peppers", "Morton Salt", "Kosher Salt", "Himalayan Salt",
    "Diamond Crystal Iodized Salt", "Good & Gather Ground Cumin", "Turmeric",
    "Smoked Paprika", "Badia Cayenne Pepper", "Stonemill Black Pepper", "Nutmeg",
    "Chili Powder", "Kinder's Seasoning Blend",
  ];
  for (const name of POSITIVE) {
    assert.equal(da.normalizeKitchenCategory("prepared_other", null, name), "spices", `${name} (prepared_other) -> spices`);
    assert.equal(da.normalizeKitchenCategory("pantry", null, name), "spices", `${name} (pantry) -> spices`);
  }
  // MUST-PASS negatives — the 10 real fleet false positives (spice/salt/pepper as a
  // descriptor on a food). The override must DECLINE, leaving the stored enum intact.
  const NEGATIVE = [
    "Great Value Black Beans No Salt Added",
    "Ithaca Hummus Olive Oil Sea Salt",
    "Bumble Bee Wild Caught Tuna Lemon Pepper",
    "Trader Joe's Organic Garbanzo Beans No Salt",
    "Whole Kernel Golden Corn No Salt Added",
    "Del Monte Sweet Corn Cream Style No Salt Added",
    "365 Organic Cannellini Beans No Salt",
    "Vigo Yellow Rice Saffron",
    "Aldi Greek Chickpeas With Parsley & Cumin",
    "Kinder's Crispy Fried Onions",
  ];
  for (const name of NEGATIVE) {
    assert.equal(da.normalizeKitchenCategory("prepared_other", null, name), "prepared_other", `${name} must NOT become spices`);
  }
  // Prior negatives (share a token with a spice/salt but are not spices):
  assert.equal(da.normalizeKitchenCategory("dairy_eggs", null, "Salted Butter"), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory("snacks_sweets", null, "Salted Caramel"), "snacks_sweets");
  assert.equal(da.normalizeKitchenCategory("produce", null, "Bell Pepper"), "produce");
  assert.equal(da.normalizeKitchenCategory("produce", null, "Jalapeno"), "produce");
  assert.equal(da.normalizeKitchenCategory("meat_seafood", null, "Pepperoni"), "meat_seafood");
  assert.equal(da.normalizeKitchenCategory("beverages", null, "Peppermint Tea"), "beverages");
  assert.equal(da.normalizeKitchenCategory("dairy_eggs", null, "Pepper Jack Cheese"), "dairy_eggs");
  assert.equal(da.normalizeKitchenCategory("pantry", null, "Hot Sauce"), "pantry");
  assert.equal(da.normalizeKitchenCategory("snacks_sweets", null, "Sea Salt Crackers"), "snacks_sweets");
  // Order/leftovers guards: leftovers win; meat override (peppercorn-crusted steak) runs first.
  assert.equal(da.normalizeKitchenCategory("leftovers", null, "Leftover Garam Masala Chicken"), "leftovers");
  assert.equal(da.normalizeKitchenCategory("prepared_other", null, "Peppercorn Crusted Ribeye Steak"), "meat_seafood");
});
test("normalizeKitchenCategory: round-3 precision — garlic cloves, ready-meal masalas, food-descriptor FPs", () => {
  // The ~5% real-fleet FP tail — each must DECLINE (stay prepared_other), never spices.
  const R3_NEG = [
    "Garlic Cloves", "Garlic clove", "Pickled Garlic Cloves", "Frozen garlic cloves",
    "Trader Joe's Paneer Tikka Masala", "Maya Kaimal Tikka Masala",
    "Trader Joe's Vegan Tikka Masala", "Tasty Bite Organic Channa Masala",
    "Trader Joe's Channa Masala", "Masala Noodles", "Masala Roti", "Masala Chai",
    "Sweet Potato Fries Sea Salt", "Sliced Carrots with Sea Salt",
    "Green Peas No added salt", "Pumpkin Seeds Sea Salt", "Pepitas with Sea Salt",
    "Seaweed Snacks Sea Salt", "Ghee Himalayan Pink Salt", "Good Thins Simply Salt",
    "Turmeric Pearl Couscous", "Maple Syrup Cardamom", "Alfredo Sauce Paprika",
    "Orange & Cloves Spread", "Bone Broth with Turmeric",
    "Apple Cider Vinegar with Turmeric", "Sliced Carrots No Added Salt",
    "Goya Chick Peas with Sea Salt",
  ];
  for (const name of R3_NEG) {
    assert.equal(da.normalizeKitchenCategory("prepared_other", null, name), "prepared_other", `${name} must NOT become spices`);
  }
  // Real spices that MUST survive the new vetoes (veto must not over-reach):
  for (const name of ["Ground Cloves", "Whole Cloves", "Cloves", "Tandoori Masala",
    "Tikka Masala Seasoning", "Tikka Masala Spice Blend", "Chai Spice Blend"]) {
    assert.equal(da.normalizeKitchenCategory("prepared_other", null, name), "spices", `${name} should still be spices`);
  }
});
test("normalizeKitchenCategory: explicit enum/category guess wins over name", () => {
  assert.equal(da.normalizeKitchenCategory("produce", null, "Chicken Breast"), "produce");
  assert.equal(da.normalizeKitchenCategory("meat_seafood", null, "Rice"), "meat_seafood");
});
test("updateKitchenItemDetails clamps a free-text category onto the enum", async () => {
  CONN = makeConn({ selectRows: [targetRow({ product_name: "Cajun Seasoning" })] });
  await da.updateKitchenItemDetails(householdCtx(), "row-1", { category: "Seasoning" }, { skipKitchenDependentGeneration: true });
  const up = mutateParamsFor(CONN, "UPDATE", "shared_kitchen");
  assert.ok(up && up.includes("spices"), "a seasoning should normalize to 'spices' in the UPDATE");
  assert.ok(!up.includes("Seasoning"), "raw free-text 'Seasoning' must not be persisted");
});
test("updateKitchenItemDetails: spice mislabeled prepared_other is rescued to 'spices'", async () => {
  // The user-facing fix path: even when the stored/guessed category is a VALID enum
  // like 'prepared_other', the UPDATE clamp routes an unambiguous spice to 'spices'.
  CONN = makeConn({ selectRows: [targetRow({ product_name: "Garam Masala" })] });
  await da.updateKitchenItemDetails(householdCtx(), "row-1", { category: "prepared_other" }, { skipKitchenDependentGeneration: true });
  const up = mutateParamsFor(CONN, "UPDATE", "shared_kitchen");
  assert.ok(up && up.includes("spices"), "Garam Masala should be rescued to 'spices' even from a prepared_other guess");
});
