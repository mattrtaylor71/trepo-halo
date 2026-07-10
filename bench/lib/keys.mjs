// Keys are read at runtime from /tmp/bench_secrets (extracted from Lambda env).
// NEVER printed, NEVER committed.
import { readFileSync } from "node:fs";
const read = (p) => { try { return readFileSync(p, "utf8").trim(); } catch { return ""; } };
export const OPENAI_KEY = read("/tmp/bench_secrets/openai.key");
export const ANTHROPIC_KEY = read("/tmp/bench_secrets/anthropic.key");
if (!OPENAI_KEY) { console.error("Missing /tmp/bench_secrets/openai.key"); process.exit(1); }
