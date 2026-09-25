// Characterization test for bulkCommit idempotency — the money path with a
// burst-503 / resweep history. The atomic claim (jobQueue.claimCommit, a DynamoDB
// ConditionExpression on the source job) must let exactly ONE commit persist; every
// duplicate delivery of the same source_job_id must write NOTHING a second time.
// No live infra: jobQueue and the kitchen writer are module-mocked.
// Run: node --experimental-test-module-mocks --test

import { test, mock, before, beforeEach } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const GI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const APP = path.join(GI, "bulkCommit", "app.js");
const JOBQUEUE = path.join(GI, "dist", "utils", "jobQueue.js");
const WRITER = path.join(GI, "bulkKitchenWriter.js");

const SCENARIO = { sourceJob: null, claim: null, persistCalls: [] };

let handler;
before(async () => {
  process.env.AWS_REGION = process.env.AWS_REGION || "us-east-1";
  mock.module(JOBQUEUE, {
    namedExports: {
      getJob: async () => SCENARIO.sourceJob,
      claimCommit: async () => SCENARIO.claim,
      createJob: async () => ({}),
      finalizeCommit: async () => ({}),
      releaseCommitClaim: async () => ({}),
      updateJobProgress: async () => ({}),
      updateJobStatus: async () => ({}),
    },
  });
  mock.module(WRITER, {
    namedExports: {
      normalizeConfirmedItems: (items, sourceItems) => (items && items.length ? items : (sourceItems || [])),
      persistConfirmedBulkItems: async (arg) => {
        SCENARIO.persistCalls.push(arg);
        const n = (arg?.items || []).length;
        return { persisted_count: n, created_count: n, duplicate_count: 0, errors: [] };
      },
    },
  });
  const mod = await import(APP);
  handler = mod.default?.handler || mod.handler;
});

beforeEach(() => {
  SCENARIO.persistCalls = [];
});

const payload = () => ({ source_job_id: "src-1", owner: "ownerA", user_id: "ownerA", device_id: "d", items: [{ item_name: "Milk" }] });
const completedSource = () => ({ status: "completed", analysis_mode: "bulk_inventory_deep", result: { items: [{ item_name: "Milk" }] } });

test("first commit (claim acquired) persists confirmed items exactly once", async () => {
  SCENARIO.sourceJob = completedSource();
  SCENARIO.claim = { claimed: true };
  await handler({ bulk_commit_internal: true, job_id: "job-1", payload: payload() });
  assert.equal(SCENARIO.persistCalls.length, 1, "the winning claim must persist once");
});

test("duplicate commit (claim lost) writes NOTHING a second time", async () => {
  SCENARIO.sourceJob = completedSource();
  SCENARIO.claim = {
    claimed: false,
    existing: { commit_status: "committed", commit_result: { persisted_count: 1, created_count: 1, duplicate_count: 0 } },
  };
  await handler({ bulk_commit_internal: true, job_id: "job-2", payload: payload() });
  assert.equal(SCENARIO.persistCalls.length, 0, "a lost claim must NOT re-persist (burst-503 / resweep idempotency)");
});
