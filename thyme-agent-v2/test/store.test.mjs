import test from "node:test";
import assert from "node:assert/strict";
import { marshall, unmarshall } from "@aws-sdk/util-dynamodb";
import { record, MemoryStore } from "../src/store.mjs";
import { hash } from "../src/core.mjs";

test("SQL recipe dates survive durable approval snapshots with identical fingerprints", () => {
  const before = {
    recipes: [{ id: "recipe-1", _createdDate: new Date("2026-10-01T12:00:00Z"), notes: [] }],
    count: 1,
  };
  const item = record("actor", "proposal", { before, beforeHash: hash(before) });
  const stored = unmarshall(marshall(item, { removeUndefinedValues: true }));
  assert.equal(stored.before.recipes[0]._createdDate, "2026-10-01T12:00:00.000Z");
  assert.equal(hash(stored.before), hash(before));
  assert.equal(stored.beforeHash, hash(before));
  assert.ok(before.recipes[0]._createdDate instanceof Date);
});

test("fixture persistence uses the same JSON date representation as production", async () => {
  const store = new MemoryStore();
  await store.put("actor", "receipt", { result: [{ nested: { date: new Date("2026-10-01T12:00:00Z") } }] });
  const stored = await store.get("actor", "receipt");
  assert.equal(stored.result[0].nested.date, "2026-10-01T12:00:00.000Z");
  assert.doesNotThrow(() => marshall(stored));
});
