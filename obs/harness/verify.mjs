#!/usr/bin/env node
// verify.mjs — observability verification harness runner.
//
// Usage:
//   export AWS_PROFILE=trepo-dev AWS_REGION=us-east-1
//   node verify.mjs --flow <name>            run one flow
//   node verify.mjs --all                    run every flow
//   node verify.mjs --all --mode synthetic   only synthetic-mode tests
//   node verify.mjs --all --slow             include tests marked slow (e.g. junk upload)
//   node verify.mjs --purge                  delete harness_probe/api_error rows for TEST_OWNER
//
// A result is PASS(e2e) when the real flow ran end-to-end and its signal was
// read; PASS(synthetic) when an injected marker proved the signal path is
// readable; FAIL when the signal could not be read; SKIP when the signal (e.g.
// a metric filter) does not exist yet.

import { readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  DynamoDBClient,
} from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient, QueryCommand, DeleteCommand,
} from '@aws-sdk/lib-dynamodb';
import { REGION, TEST_OWNER, ANALYTICS_TABLE, PROFILE_NOTE } from './lib/config.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({ region: REGION }));

// ── arg parsing ──────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const a = { flow: null, all: false, mode: 'both', slow: false, purge: false };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--all') a.all = true;
    else if (t === '--flow') a.flow = argv[++i];
    else if (t === '--mode') a.mode = argv[++i];
    else if (t === '--slow') a.slow = true;
    else if (t === '--purge') a.purge = true;
  }
  return a;
}

const runId = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(16).slice(2, 8)}`;

async function loadFlows() {
  const files = (await readdir(join(__dirname, 'flows'))).filter((f) => f.endsWith('.mjs'));
  const flows = [];
  for (const f of files.sort()) {
    const mod = await import(join(__dirname, 'flows', f));
    flows.push({ name: mod.name || f.replace('.mjs', ''), tests: mod.tests || [] });
  }
  return flows;
}

// ── purge ────────────────────────────────────────────────────────────────────
async function purge() {
  const targets = ['harness_probe', 'api_error'];
  let deleted = 0;
  const res = await ddb.send(new QueryCommand({
    TableName: ANALYTICS_TABLE,
    KeyConditionExpression: 'owner_id = :o',
    ExpressionAttributeValues: { ':o': TEST_OWNER },
    Limit: 1000,
  }));
  for (const it of res.Items || []) {
    if (targets.includes(it.event_name)) {
      await ddb.send(new DeleteCommand({
        TableName: ANALYTICS_TABLE,
        Key: { owner_id: it.owner_id, ts_id: it.ts_id },
      }));
      deleted += 1;
    }
  }
  console.log(`[purge] deleted ${deleted} harness_probe/api_error row(s) for TEST_OWNER ${TEST_OWNER}`);
}

// ── run ──────────────────────────────────────────────────────────────────────
function wantMode(testMode, cliMode) {
  if (cliMode === 'both') return true;
  return testMode === cliMode;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  console.log(`# Trepo Observability Harness  (${PROFILE_NOTE})`);
  console.log(`# run_id=${runId}  owner=${TEST_OWNER}`);

  if (args.purge) { await purge(); }
  if (!args.all && !args.flow) {
    if (args.purge) return;
    console.error('Specify --flow <name> or --all (or --purge).');
    process.exit(2);
  }

  const allFlows = await loadFlows();
  const flows = args.all ? allFlows : allFlows.filter((f) => f.name === args.flow);
  if (flows.length === 0) {
    console.error(`No flow named "${args.flow}". Available: ${allFlows.map((f) => f.name).join(', ')}`);
    process.exit(2);
  }

  const ctx = { runId, owner: TEST_OWNER, cleanup: [], log: (m) => console.log(`  · ${m}`) };
  const results = [];

  for (const flow of flows) {
    for (const test of flow.tests) {
      if (!wantMode(test.mode, args.mode)) continue;
      if (test.slow && !args.slow) {
        results.push({ flow: flow.name, test: test.label, mode: test.mode, status: 'SKIP', detail: 'slow test skipped (pass --slow to run)' });
        console.log(`SKIP  ${flow.name} / ${test.label}  (slow)`);
        continue;
      }
      const t0 = Date.now();
      let r;
      try {
        r = await test.run(ctx);
      } catch (err) {
        r = { status: 'FAIL', mode: test.mode, detail: `threw: ${err.message}` };
      }
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const rec = { flow: flow.name, test: test.label, mode: r.mode || test.mode, status: r.status, detail: r.detail || '', elapsedS: Number(elapsed) };
      results.push(rec);
      console.log(`${rec.status.padEnd(5)} ${flow.name} / ${test.label}  [${rec.mode}, ${elapsed}s]\n      ${rec.detail}`);
    }
  }

  // ── cleanup ──
  if (ctx.cleanup.length) {
    console.log(`\n# cleanup: running ${ctx.cleanup.length} deferred cleanup task(s)`);
    for (const fn of ctx.cleanup) {
      try { await fn(); } catch (err) { console.log(`  ! cleanup error: ${err.message}`); }
    }
  }

  // ── matrix ──
  printMatrix(results);

  const outPath = join(__dirname, `results-${runId}.json`);
  await writeFile(outPath, JSON.stringify({ runId, owner: TEST_OWNER, generatedAt: new Date().toISOString(), args, results }, null, 2));
  console.log(`\n# wrote ${outPath}`);

  const failed = results.filter((r) => r.status === 'FAIL').length;
  process.exit(failed > 0 ? 1 : 0);
}

function printMatrix(results) {
  console.log('\n' + '='.repeat(88));
  console.log('RESULTS MATRIX');
  console.log('='.repeat(88));
  const label = (r) => (r.status === 'PASS' ? `PASS(${r.mode})` : r.status);
  const wFlow = Math.max(6, ...results.map((r) => r.flow.length));
  const wTest = Math.max(4, ...results.map((r) => r.test.length));
  const head = `${'FLOW'.padEnd(wFlow)}  ${'TEST'.padEnd(wTest)}  RESULT`;
  console.log(head);
  console.log('-'.repeat(head.length + 8));
  for (const r of results) {
    console.log(`${r.flow.padEnd(wFlow)}  ${r.test.padEnd(wTest)}  ${label(r)}`);
  }
  const counts = results.reduce((a, r) => { a[r.status] = (a[r.status] || 0) + 1; return a; }, {});
  console.log('-'.repeat(head.length + 8));
  console.log(`TOTAL: ${results.length}  |  PASS ${counts.PASS || 0}  FAIL ${counts.FAIL || 0}  SKIP ${counts.SKIP || 0}`);
}

main().catch((err) => { console.error('fatal:', err); process.exit(3); });
