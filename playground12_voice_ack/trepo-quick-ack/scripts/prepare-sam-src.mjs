import { cpSync, copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const quickAckRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(quickAckRoot, "..", "..");
const stageRoot = path.join(quickAckRoot, ".sam-src");
const stagedQuickAckRoot = path.join(stageRoot, "playground12_voice_ack", "trepo-quick-ack");
const stagedSharedRoot = path.join(stageRoot, "shared", "voice-assistant");

rmSync(stageRoot, { recursive: true, force: true });
mkdirSync(stagedQuickAckRoot, { recursive: true });
mkdirSync(stagedSharedRoot, { recursive: true });

for (const filename of [
  "index.js",
  "index-stream.js",
  "image_postprocess.js",
  "async-ingest.js",
  "async-worker.js",
  "async_ingest.js",
  "async_worker.js",
  "package.json",
  "package-lock.json"
]) {
  copyFileSync(path.join(quickAckRoot, filename), path.join(stagedQuickAckRoot, filename));
}
cpSync(path.join(quickAckRoot, "lib"), path.join(stagedQuickAckRoot, "lib"), { recursive: true });

for (const filename of ["tool-definitions.mjs", "tool-executor.mjs", "text-dish-analyzer.mjs"]) {
  copyFileSync(
    path.join(repoRoot, "shared", "voice-assistant", filename),
    path.join(stagedSharedRoot, filename)
  );
}

if (existsSync(path.join(stagedQuickAckRoot, "node_modules"))) {
  rmSync(path.join(stagedQuickAckRoot, "node_modules"), { recursive: true, force: true });
}

execFileSync("npm", ["ci", "--omit=dev", "--prefix", stagedQuickAckRoot], {
  cwd: quickAckRoot,
  stdio: "inherit"
});

// Freshness sentinel: record the hash of the source inputs we just staged so
// assert-sam-src-fresh can later detect a stale .sam-src (source changed since prepare).
const { computeSourceHash } = await import("./sam-src-hash.mjs");
const { writeFileSync } = await import("node:fs");
writeFileSync(path.join(stageRoot, ".source-manifest"), computeSourceHash(quickAckRoot, repoRoot));

console.log(`Prepared SAM source at ${stageRoot}`);
