// Shared source-hash used by the sam-src freshness gate. prepare-sam-src writes the
// hash of the SOURCE inputs it stages (lib/ + entry files + the shared voice files)
// into .sam-src/.source-manifest; assert-sam-src-fresh recomputes it and fails if the
// live source has changed since the last prepare — i.e. .sam-src is stale. This is the
// guard against the 07-02/07-06-class "deployed a stale .sam-src bundle" reverts.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import path from "node:path";

const ENTRY_FILES = [
  "index.js", "index-stream.js", "image_postprocess.js", "async-ingest.js",
  "async-worker.js", "async_ingest.js", "async_worker.js", "package.json", "package-lock.json",
];
const SHARED_FILES = ["tool-definitions.mjs", "tool-executor.mjs", "text-dish-analyzer.mjs"];

function walk(dir, out) {
  if (!existsSync(dir)) return;
  for (const name of readdirSync(dir).sort()) {
    if (name === "node_modules") continue;
    const p = path.join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else out.push(p);
  }
}

export function computeSourceHash(quickAckRoot, repoRoot) {
  const files = [];
  walk(path.join(quickAckRoot, "lib"), files);
  for (const f of ENTRY_FILES) {
    const p = path.join(quickAckRoot, f);
    if (existsSync(p)) files.push(p);
  }
  for (const f of SHARED_FILES) {
    const p = path.join(repoRoot, "shared", "voice-assistant", f);
    if (existsSync(p)) files.push(p);
  }
  const h = createHash("sha256");
  for (const p of files.sort()) {
    // Include the path (relative to quickAckRoot) + content so moves/renames register.
    h.update(path.relative(quickAckRoot, p));
    h.update("\0");
    h.update(readFileSync(p));
    h.update("\0");
  }
  return h.digest("hex");
}
