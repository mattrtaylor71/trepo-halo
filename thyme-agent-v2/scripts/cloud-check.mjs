import fs from "node:fs";
import assert from "node:assert/strict";
import { signRequest, id } from "../src/core.mjs";
const root = "/Users/MattTaylor/Library/Caches/trepo-thyme-agent-v2-20261001",
  cfg = JSON.parse(fs.readFileSync(root + "/deployment-private.json")),
  secret = fs.readFileSync(root + "/bridge-secret", "utf8");
async function call(
  data,
  { tamper = false, principal = "qualification-read-only" } = {},
) {
  const timestamp = String(Date.now()),
    nonce = id(),
    body = JSON.stringify({ ...data, principal }),
    signature = signRequest({ body, timestamp, nonce }, secret);
  const r = await fetch(cfg.url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-thyme-time": timestamp,
      "x-thyme-nonce": nonce,
      "x-thyme-signature": tamper ? "a".repeat(64) : signature,
    },
    body,
    signal: AbortSignal.timeout(30000),
  });
  let result;
  const raw = await r.text();
  try {
    result = JSON.parse(raw);
  } catch {
    result = { message: raw.slice(0, 150) };
  }
  return { status: r.status, result };
}
const evidence = [];
function check(name, res, expected) {
  assert.equal(res.status, expected, JSON.stringify(res));
  evidence.push({ name, status: res.status });
}
check(
  "unsigned or forged connection rejected",
  await call({ op: "bootstrap" }, { tamper: true }),
  401,
);
check(
  "unenrolled principal rejected",
  await call({ op: "bootstrap" }, { principal: "other-user" }),
  403,
);
check(
  "client cannot select another household",
  await call({ op: "bootstrap", household: "other" }),
  403,
);
check(
  "read-only canary cannot approve changes",
  await call({ op: "decide" }),
  403,
);
const boot = await call({ op: "bootstrap" });
check("Matt live household read", boot, 200);
evidence.push({
  name: "live context counts",
  kitchen: boot.result.kitchen.length,
  shopping: boot.result.shopping.length,
});
const requestId = id(),
  start = Date.now();
let res = await call({
  op: "message",
  requestId,
  model: "gpt-6-luna",
  text: "Read my actual kitchen and preferences. Name three ingredients I currently have and suggest one simple dinner idea. Do not create, save or change anything.",
});
check("durable message accepted", res, 200);
let s = res.result;
const sid = s.id;
const duplicate = await call({
  op: "message",
  requestId,
  model: "gpt-6-luna",
  text: "Read my actual kitchen and preferences. Name three ingredients I currently have and suggest one simple dinner idea. Do not create, save or change anything.",
});
assert.equal(duplicate.result.id, sid);
evidence.push({
  name: "duplicate request returns same conversation",
  ok: true,
});
for (
  let i = 0;
  i < 90 && ["queued", "running", "verifying"].includes(s.status);
  i++
) {
  await new Promise((r) => setTimeout(r, 2000));
  res = await call({ op: "session", sessionId: sid });
  check("poll " + i, res, 200);
  s = res.result;
}
assert.equal(
  s.status,
  "completed",
  JSON.stringify({ status: s.status, error: s.error }),
);
assert.equal(s.proposals.length, 0);
fs.writeFileSync(root + "/cloud-session-private.json", JSON.stringify(s), {
  mode: 0o600,
});
evidence.push({
  name: "real AWS durable worker + managed agent + Matt data",
  status: s.status,
  seconds: (Date.now() - start) / 1000,
  liveMutations: 0,
  metrics: s.metrics,
});
fs.writeFileSync(
  "/Users/MattTaylor/Documents/Codex/2026-10-01/thyme-agent-pilot/evidence/cloud-check.json",
  JSON.stringify(evidence, null, 2),
);
console.log(
  JSON.stringify(
    evidence.filter((x) => !x.name.startsWith("poll ")),
    null,
    2,
  ),
);
