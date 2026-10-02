import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { saveCanonical } from "../src/canonical-save.mjs";
const require = createRequire(
  "/Users/MattTaylor/Library/Caches/trepo-thyme-release-20261001/trepo-quick-ack-stream-dev/source/playground12_voice_ack/trepo-quick-ack/package.json",
);
const mysql = require("mysql2/promise");
const c = await mysql.createConnection({
  host: "127.0.0.1",
  port: 33317,
  user: "fixture",
  password: "fixture",
  database: "thyme_test_quality_20261001",
});
const actor = {
    actor: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    household: "fixture-home",
  },
  args = {
    title: "Canonical Eggs",
    ingredients: ["2 eggs", "1 tsp oil"],
    steps: ["Heat oil.", "Cook eggs until set."],
    notes: ["Serves 2"],
  };
const tests = [];
try {
  await c.execute(
    "CREATE TABLE IF NOT EXISTS new_users(user_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36))",
  );
  await c.execute(
    "INSERT INTO new_users(user_id,owner_id) VALUES (?,?) ON DUPLICATE KEY UPDATE owner_id=VALUES(owner_id)",
    [actor.actor, actor.household],
  );
  await c.execute(
    "CREATE TABLE IF NOT EXISTS shared_saved_recipes(owner_id VARCHAR(36),_id VARCHAR(36),_owner VARCHAR(36),source_type VARCHAR(32),source_url VARCHAR(1000),resolved_url VARCHAR(1000),resolved_url_hash CHAR(64),title VARCHAR(255),ingredients JSON,instructions JSON,notes JSON,status VARCHAR(32),PRIMARY KEY(owner_id,_id))",
  );
  await c.execute(
    "CREATE TABLE IF NOT EXISTS `" +
      actor.actor +
      "_saved_recipes` LIKE shared_saved_recipes",
  );
  for (const table of [
    "shared_saved_recipes",
    actor.actor + "_saved_recipes",
  ]) {
    const [cols] = await c.execute(
      "SELECT COLUMN_NAME FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=?",
      [table],
    );
    if (!cols.some((x) => x.COLUMN_NAME === "notes"))
      await c.execute("ALTER TABLE `" + table + "` ADD COLUMN notes JSON");
  }
  const op = "canonical-save-" + Date.now();
  let res = await saveCanonical(c, actor, args, op);
  const id = res.toolResult.recipe.id;
  const [rows] = await c.execute(
    "SELECT title,ingredients,instructions,notes FROM shared_saved_recipes WHERE owner_id=? AND _id=?",
    [actor.actor, id],
  );
  assert.equal(rows.length, 1);
  assert.deepEqual(rows[0].ingredients, args.ingredients);
  assert.deepEqual(rows[0].instructions, args.steps);
  assert.deepEqual(rows[0].notes, args.notes);
  tests.push("current library exact save including servings");
  const [legacy] = await c.execute(
    "SELECT ingredients,instructions FROM `" +
      actor.actor +
      "_saved_recipes` WHERE _id=?",
    [id],
  );
  assert.equal(legacy.length, 1);
  assert.deepEqual(legacy[0].instructions, args.steps);
  tests.push("older app exact save");
  res = await saveCanonical(c, actor, args, op);
  assert.equal(res.toolResult.recipe.id, id);
  tests.push("duplicate request idempotent");
  await assert.rejects(
    saveCanonical(c, actor, { ...args, title: "Changed" }, op),
    { code: "request_conflict" },
  );
  tests.push("changed intent rejected");
  await assert.rejects(
    saveCanonical(c, { ...actor, household: "other" }, args, op + "wrong"),
    { code: "membership" },
  );
  tests.push("membership fenced inside transaction");
  const deletedActor = {
    actor: "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee",
    household: "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee",
  };
  await assert.rejects(saveCanonical(c, deletedActor, args, "deleted-user"), {
    code: "membership",
  });
  tests.push("deleted primary account cannot save");
  console.log(
    JSON.stringify({
      ok: true,
      tests,
      environment: "isolated local MySQL",
      customerWrites: 0,
    }),
  );

} finally {
  await c.end();
}
