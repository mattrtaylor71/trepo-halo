// flows/bulk.mjs — bulk identify pipeline (identifier API 6m9t6wosh9).
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { API, TEST_OWNER, LOG_GROUPS, CAPTURE_NS } from '../lib/config.mjs';
import { httpStatus, jobStatus, logLine, cloudwatchMetric, nowMs } from '../lib/signals.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

export const name = 'bulk';

export const tests = [
  {
    label: 'GET /health -> 200',
    mode: 'e2e',
    async run() {
      const res = await httpStatus({ url: `${API.identifier}/health`, method: 'GET', wantStatuses: [200] });
      return { status: res.ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: res.detail + (res.ok ? '' : ` body=${res.body.slice(0, 120)}`) };
    },
  },
  {
    label: 'GET /job/{bogus} -> 404/error',
    mode: 'e2e',
    async run() {
      const res = await httpStatus({ url: `${API.identifier}/job/obs-harness-nonexistent-job-id`, method: 'GET', wantStatuses: [400, 404, 422] });
      // Some backends return 200 with status=FAILED/NOT_FOUND; accept that too.
      if (!res.ok && res.status === 200) {
        const upper = res.body.toUpperCase();
        if (upper.includes('NOT_FOUND') || upper.includes('FAILED') || upper.includes('ERROR')) {
          return { status: 'PASS', mode: 'e2e', detail: `200 with not-found/error body (${res.body.slice(0, 100)})` };
        }
      }
      return { status: res.ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: res.detail + ` body=${res.body.slice(0, 120)}` };
    },
  },
  {
    label: 'junk image upload -> job FAILED + failure log line + BulkIdentifyAnalysisFailed',
    mode: 'e2e',
    slow: true, // NOT run in the safe subset — orchestrator runs this in the final pass.
    async run() {
      const sinceMs = nowMs();
      const junk = await readFile(join(__dirname, '..', 'fixtures', 'junk.jpg'));
      const b64 = junk.toString('base64');
      const post = await httpStatus({
        url: `${API.identifier}/identify-async`,
        method: 'POST',
        body: {
          owner: TEST_OWNER,
          user_id: TEST_OWNER,
          device_id: 'obs-harness',
          analysis_mode: 'bulk_inventory_deep',
          image: b64,
        },
        wantStatuses: [200, 201, 202],
      });
      if (!post.ok) return { status: 'FAIL', mode: 'e2e', detail: `identify-async POST ${post.status}: ${post.body.slice(0, 160)}` };
      let jobId = null;
      try { const j = JSON.parse(post.body); jobId = j.job_id || j.jobId || j.id; } catch (_) { /* */ }
      if (!jobId) return { status: 'FAIL', mode: 'e2e', detail: `no job_id in response: ${post.body.slice(0, 160)}` };

      const job = await jobStatus({ baseUrl: API.identifier, jobId, want: ['FAILED', 'ERROR'], timeoutMs: 300_000 });
      // Two failure paths exist in identifyAsync: "Background processing error for job X"
      // (dispatch .catch) and "[IdentifyAsync] Job X failed:" (processJob catch). Either counts.
      const log = await logLine({ logGroup: LOG_GROUPS.bulkIdentify, pattern: '?"Background processing error" ?"failed:"', sinceMs, timeoutMs: 120_000 });
      const metric = await cloudwatchMetric({ namespace: CAPTURE_NS, metricName: 'BulkIdentifyAnalysisFailed', sinceMs, timeoutMs: 120_000 });

      const parts = [`job=${job.ok ? 'FAILED' : job.detail}`, `log=${log.ok ? 'seen' : 'MISSING'}`, `metric=${metric.ok ? 'seen' : 'MISSING'}`];
      const ok = job.ok && log.ok && metric.ok;
      return { status: ok ? 'PASS' : 'FAIL', mode: 'e2e', detail: parts.join(', ') };
    },
  },
];
