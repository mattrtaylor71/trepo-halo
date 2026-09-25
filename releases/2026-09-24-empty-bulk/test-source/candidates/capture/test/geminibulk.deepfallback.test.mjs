// Unit test for the cross-model deep fallback DECISION logic (no network).
// Verifies the flag gating, transient-only trigger, and Lambda-budget capping —
// the risky control-flow of the AI_FALLBACK_DEEP path. The actual model swap is
// exercised by a flag-ON live probe (Matt's flip); this pins the decision.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { deepFallbackDecision } = require("../quickIdentify/geminiBulk.js");

const transient = new Error("network timeout at: https://generativelanguage.googleapis.com/...");
const nonTransient = new Error("Gemini bulk scan failed: HTTP 400 bad request");

test("flag OFF => never falls back (inert)", () => {
  const d = deepFallbackDecision(transient, { enabled: false, elapsedMs: 0 });
  assert.equal(d.fallback, false);
  assert.equal(d.reason, "disabled");
});

test("flag ON + transient + budget => fall back to secondary with capped timeout", () => {
  const d = deepFallbackDecision(transient, { enabled: true, elapsedMs: 0 });
  assert.equal(d.fallback, true);
  assert.equal(d.reason, "primary_transient");
  assert.ok(d.timeoutMs > 0 && d.timeoutMs <= 120000, `timeout capped: ${d.timeoutMs}`);
  assert.equal(d.model, "gemini-2.5-flash");
});

test("flag ON + NON-transient error => no fallback (don't retry hard failures)", () => {
  const d = deepFallbackDecision(nonTransient, { enabled: true, elapsedMs: 0 });
  assert.equal(d.fallback, false);
  assert.equal(d.reason, "non_transient");
});

test("flag ON + transient but budget exhausted => no fallback (protect the 900s Lambda budget)", () => {
  // elapsed 830s -> remaining = 900 - 830 - 60(finalize) = 10s < 15s min => skip
  const d = deepFallbackDecision(transient, { enabled: true, elapsedMs: 830000 });
  assert.equal(d.fallback, false);
  assert.equal(d.reason, "no_budget");
});

test("fallback timeout shrinks with elapsed time (budget-aware)", () => {
  // elapsed 780s -> remaining = 900 - 780 - 60 = 60s => timeout capped to 60s (< 120s default)
  const d = deepFallbackDecision(transient, { enabled: true, elapsedMs: 780000 });
  assert.equal(d.fallback, true);
  assert.equal(d.timeoutMs, 60000);
});
