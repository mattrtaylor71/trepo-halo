// Freshness gate for the SAM bundle. Fails LOUDLY if .sam-src is missing or stale
// (the live lib/ + shared source changed since the last `npm run sam:prepare`), so a
// bare `sam build`/`sam deploy` can't ship an out-of-date bundle — the exact
// stale-.sam-src footgun behind the 07-02 / 07-06 voice dual-write reverts.
//
// The ONLY blessed deploy is `make deploy` (== npm run sam:deploy), which regenerates
// .sam-src from source first. Never run `sam deploy` directly.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { computeSourceHash } from "./sam-src-hash.mjs";

const quickAckRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(quickAckRoot, "..", "..");
const manifest = path.join(quickAckRoot, ".sam-src", ".source-manifest");

function die(msg) {
  console.error("\n❌  SAM bundle is not deployable.\n" + msg +
    "\n\n✅  Run the ONE blessed command:  make deploy   (== npm run sam:deploy)\n" +
    "    It regenerates .sam-src from lib/+shared, builds, and deploys. NEVER run `sam deploy` directly.\n");
  process.exit(1);
}

if (!existsSync(manifest)) {
  die("   .sam-src/.source-manifest is missing — .sam-src was never prepared (or is stale from an old build).");
}
const stored = readFileSync(manifest, "utf8").trim();
const current = computeSourceHash(quickAckRoot, repoRoot);
if (stored !== current) {
  die("   lib/ or shared/voice-assistant changed since the last `sam:prepare` — .sam-src is STALE.\n" +
    `     manifest=${stored.slice(0, 12)}…  live=${current.slice(0, 12)}…`);
}
console.log("✅  .sam-src is fresh (matches live lib/ + shared source).");
