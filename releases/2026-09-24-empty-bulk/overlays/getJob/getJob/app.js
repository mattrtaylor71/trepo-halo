const {isTerminalEmptyBulkJob, projectEmptyBulkJob} = require('../emptyBulkOutcome');
// Lambda handler for getting job status
let jobQueue;
try {
  jobQueue = require('../dist/utils/jobQueue');
} catch (e) {
  console.error('[GetJob] Failed to load jobQueue:', e);
  jobQueue = {
    getJob: async () => { throw new Error('Upload recovery temporarily unavailable'); },
  };
}

exports.handler = async (event) => {
  console.log('[GetJob] Event received');
  
  try {
    const jobId = event.pathParameters?.job_id;
    
    if (!jobId) {
      return {
        statusCode: 400,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          error: 'Missing job_id in path. Use /job/{job_id}',
        }),
      };
    }

    let job = await jobQueue.getJob(jobId);
    
    if (!job) {
      return {
        statusCode: 404,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          error: 'Job not found',
          job_id: jobId,
        }),
      };
    }

    // HTTP200 lets older installed pollers read the terminal failed status. Keep
    // the established HTTP500 contract for provider/processing failures.
    const emptyBulk = isTerminalEmptyBulkJob(job);
    job = projectEmptyBulkJob(job);

    // Return appropriate status code based on job status
    const statusCode = emptyBulk || job.status === 'completed' ? 200 : 
                      job.status === 'failed' ? 500 : 
                      202; // Accepted (pending/processing)

    return {
      statusCode,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({
        job_id: job.job_id,
        status: job.status,
        commit_status: job.commit_status || null,
        review_dismissed_at: job.review_dismissed_at || null,
        commit_completed_at: job.commit_completed_at || null,
        analysis_mode: job.analysis_mode || null,
        created_at: job.created_at,
        updated_at: job.updated_at,
        stage: job.stage || null,
        stage_message: job.stage_message || null,
        progress: typeof job.progress === 'number' ? job.progress : null,
        meta: job.meta || null,
        result: job.result || null,
        error: job.error || null,
      }),
    };
  } catch (error) {
    console.error('[GetJob] Error:', error);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
    };
  }
};

