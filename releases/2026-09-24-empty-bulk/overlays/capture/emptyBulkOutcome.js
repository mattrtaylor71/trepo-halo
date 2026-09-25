// Keep valid empty bulk analysis terminal and nonreviewable for status-based clients.
// Receipt captures have a separate typed recovery contract and are not changed here.
const EMPTY_BULK_CODE = 'no_items_identified';
const EMPTY_BULK_MESSAGE = 'No grocery items were found. Try a clearer photo.';
function isEmptyBulkResult(mode, result) {
  return mode === 'bulk_inventory_deep' && Array.isArray(result?.items) && result.items.length === 0;
}
function emptyBulkResult(result) {
  return {...result, error_code: EMPTY_BULK_CODE, capture_outcome: 'no_items'};
}
function isTerminalEmptyBulkJob(job) {
  return !job?.commit_status && isEmptyBulkResult(job?.analysis_mode, job?.result) &&
    (job.status === 'completed' || (job.status === 'failed' && job.result.error_code === EMPTY_BULK_CODE));
}
// Pure response projection: do not rewrite retained legacy records or their ownership.
function projectEmptyBulkJob(job) {
  if (!isTerminalEmptyBulkJob(job)) return job;
  return {...job, status: 'failed', stage: 'failed', stage_message: EMPTY_BULK_MESSAGE,
    progress: 100, error: EMPTY_BULK_MESSAGE, result: emptyBulkResult(job.result)};
}
module.exports = {EMPTY_BULK_CODE, EMPTY_BULK_MESSAGE, isEmptyBulkResult, emptyBulkResult,
  isTerminalEmptyBulkJob, projectEmptyBulkJob};
