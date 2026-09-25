const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, UpdateCommand, GetCommand, QueryCommand } = require('@aws-sdk/lib-dynamodb');

const docClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));

function tableName() {
  return process.env.SESSION_TABLE_NAME;
}

async function ensureSession(sessionId, ownerId, jobId, analysisMode) {
  const now = new Date().toISOString();
  const ttl = Math.floor(Date.now() / 1000) + 86400; // 24 hours from now

  const params = {
    TableName: tableName(),
    Key: { session_id: sessionId },
    ConditionExpression: '(attribute_not_exists(owner_id) OR owner_id = :ownerId) AND (attribute_not_exists(job_ids) OR NOT contains(job_ids, :jobId))',
    UpdateExpression: `
      SET owner_id = if_not_exists(owner_id, :ownerId),
          #status = if_not_exists(#status, :processing),
          created_at = if_not_exists(created_at, :now),
          updated_at = :now,
          #ttl = :ttl,
          total_jobs = if_not_exists(total_jobs, :zero) + :one,
          job_ids = list_append(if_not_exists(job_ids, :emptyList), :jobIdList),
          completed_job_ids = if_not_exists(completed_job_ids, :emptyList),
          failed_job_ids = if_not_exists(failed_job_ids, :emptyList)
    `,
    ExpressionAttributeNames: {
      '#status': 'status',
      '#ttl': 'ttl',
    },
    ExpressionAttributeValues: {
      ':ownerId': ownerId,
      ':processing': 'processing',
      ':now': now,
      ':ttl': ttl,
      ':zero': 0,
      ':one': 1,
      ':emptyList': [],
      ':jobIdList': [jobId],
      ':jobId': jobId,
    },
    ReturnValues: 'ALL_NEW',
  };

  // Store analysis_mode on the session (first job wins)
  if (analysisMode) {
    params.UpdateExpression += `, analysis_mode = if_not_exists(analysis_mode, :mode)`;
    params.ExpressionAttributeValues[':mode'] = analysisMode;
  }

  try {
    const result = await docClient.send(new UpdateCommand(params));
    return result.Attributes;
  } catch (error) {
    if (error.name !== 'ConditionalCheckFailedException') throw error;
    const existing = await getSession(sessionId);
    if (!existing || existing.owner_id !== ownerId) throw new Error('Session owner mismatch');
    return existing;
  }
}

async function markJobCompleted(sessionId, jobId) {
  return markJobTerminal(sessionId, jobId, 'completed_job_ids');
}

async function markJobFailed(sessionId, jobId) {
  return markJobTerminal(sessionId, jobId, 'failed_job_ids');
}

// DynamoDB retries and duplicate Lambda deliveries must not advance the count.
// Keep lists on the wire for every shipped client; conditional writes provide
// uniqueness without migrating existing records to DynamoDB sets.
async function markJobTerminal(sessionId, jobId, field) {
  try {
    const result = await docClient.send(new UpdateCommand({
      TableName: tableName(), Key: { session_id: sessionId },
      UpdateExpression: 'SET #terminal = list_append(if_not_exists(#terminal, :empty), :jobs), updated_at = :now',
      ConditionExpression: 'attribute_exists(session_id) AND contains(job_ids, :job) AND (attribute_not_exists(completed_job_ids) OR NOT contains(completed_job_ids, :job)) AND (attribute_not_exists(failed_job_ids) OR NOT contains(failed_job_ids, :job))',
      ExpressionAttributeNames: { '#terminal': field },
      ExpressionAttributeValues: { ':empty': [], ':jobs': [jobId], ':job': jobId, ':now': new Date().toISOString() },
      ReturnValues: 'ALL_NEW',
    }));
    return result.Attributes;
  } catch (error) {
    if (error.name !== 'ConditionalCheckFailedException') throw error;
    return getSession(sessionId);
  }
}

async function getSession(sessionId) {
  const result = await docClient.send(new GetCommand({
    TableName: tableName(),
    Key: { session_id: sessionId },
    ConsistentRead: true,
  }));

  return result.Item || null;
}

async function getActiveSessions(ownerId) {
  const sessions = [];
  let cursor;
  do {
    const result = await docClient.send(new QueryCommand({
      TableName: tableName(),
      IndexName: 'owner-index',
      KeyConditionExpression: 'owner_id = :ownerId',
      FilterExpression: '#status <> :completed',
      ExpressionAttributeNames: {
        '#status': 'status',
      },
      ExpressionAttributeValues: {
        ':ownerId': ownerId,
        ':completed': 'completed',
      },
      ExclusiveStartKey: cursor,
    }));
    sessions.push(...(result.Items || []));
    cursor = result.LastEvaluatedKey;
  } while (cursor);
  return sessions;
}

function isSessionComplete(session) {
  if (!session) return false;
  const terminal = new Set([...(session.completed_job_ids || []), ...(session.failed_job_ids || [])]);
  const jobs = new Set(session.job_ids || []);
  if (jobs.size) return [...jobs].every(id => terminal.has(id));
  const total = session.total_jobs || 0;
  return total > 0 && terminal.size >= total;
}

async function tryMarkSessionNotified(sessionId) {
  try {
    await docClient.send(new UpdateCommand({
      TableName: tableName(),
      Key: { session_id: sessionId },
      UpdateExpression: 'SET notification_sent = :true, #status = :completed',
      ConditionExpression: 'attribute_exists(session_id) AND (attribute_not_exists(notification_sent) OR notification_sent = :false)',
      ExpressionAttributeNames: {
        '#status': 'status',
      },
      ExpressionAttributeValues: {
        ':true': true,
        ':false': false,
        ':completed': 'completed',
      },
    }));
    return true;
  } catch (err) {
    if (err.name === 'ConditionalCheckFailedException') {
      return false;
    }
    throw err;
  }
}

module.exports = {
  ensureSession,
  markJobCompleted,
  markJobFailed,
  getSession,
  getActiveSessions,
  isSessionComplete,
  tryMarkSessionNotified,
};
