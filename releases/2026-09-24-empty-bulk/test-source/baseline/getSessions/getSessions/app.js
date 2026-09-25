// Lambda handler for getting active bulk sessions by owner
let sessionStore;
try {
  sessionStore = require('../sessionStore');
} catch (e) {
  console.error('[GetSessions] Failed to load sessionStore:', e);
  sessionStore = {
    getActiveSessions: async () => { throw new Error('Upload recovery temporarily unavailable'); },
  };
}

exports.handler = async (event) => {
  console.log('[GetSessions] Event received');

  try {
    if (event.queryStringParameters?.review === '1') {
      return await require('../reviewInboxHandler').createHandler()(event);
    }
    const ownerId = event.pathParameters?.owner_id;

    if (!ownerId) {
      return {
        statusCode: 400,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          error: 'Missing owner_id in path. Use /sessions/{owner_id}',
        }),
      };
    }

    const sessions = await sessionStore.getActiveSessions(ownerId);

    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({
        owner_id: ownerId,
        sessions,
      }),
    };
  } catch (error) {
    console.error('[GetSessions] Error:', error);
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
