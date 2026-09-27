import { reconnectRedirect } from './ios-reconnect.mjs';
import { lookupUserContextByOwnerId } from './user-context.mjs';
import { prepareBrainSession, candidateOwner } from './family-brain.mjs';
import { jsonResponse } from './http.mjs';
import { header, operationIdentity, validSessionId, verifiedActor, operationScope, readOperation, acknowledgeOperation } from './operation-recovery.mjs';

export function resolvedOperationScope(identity, proof, context, brainSession, event, env) {
  if (candidateOwner(header(event.headers,'authorization'),env) && !brainSession) {
    throw Object.assign(new Error('recovery_unavailable'),{statusCode:503});
  }
  return operationScope(identity,proof,context,brainSession?.sessionOwner || context.ownerId,env);
}

export async function recoverRequest(event, env) {
  try {
    const proof = await verifiedActor(event,env);
    const rendered = header(event.headers,'x-thyme-event') === 'rendered';
    const method = event.requestContext?.http?.method || event.httpMethod;
    if (rendered && method !== 'POST') return jsonResponse(405,{error:'invalid_method'});
    let payload = event.queryStringParameters || {};
    if (rendered) {
      const raw = event.isBase64Encoded ? Buffer.from(event.body || '', 'base64').toString() : String(event.body || '');
      if (Buffer.byteLength(raw)>2048) return jsonResponse(400,{error:'invalid_acknowledgment'});
      payload=JSON.parse(raw);
    }
    const identity = operationIdentity(payload.operation_id);
    if (!identity || !validSessionId(payload.session_id)) {
      return jsonResponse(400,{error:'invalid_request_identity'});
    }
    const context = await lookupUserContextByOwnerId(proof.actor,{env});
    const brain = await prepareBrainSession({authorization:header(event.headers,'authorization'),owner:proof.owner,surface:'app'},env);
    const scope = resolvedOperationScope({...identity,sessionId:payload.session_id},proof,context,brain,event,env);
    if (rendered) {
      const answer=await acknowledgeOperation(scope);
      return jsonResponse(answer.acknowledged?200:409,answer);
    }
    const answer=await readOperation(scope);
    const body={...(answer.body || {}),operation:{...(answer.body?.operation || {}),id:scope.id,status:answer.status}};
    // Recovery remains read-only: the bridge never runs/repeats the command.
    try {
      const redirect=await reconnectRedirect(event,env,scope,body);
      if (redirect) return redirect;
    } catch { /* Connection setup must not prevent reading an existing result. */ }
    return jsonResponse(200,body);
  } catch (error) {
    return jsonResponse(error instanceof SyntaxError?400:error.statusCode || 503,
      {error:error.code || 'recovery_unavailable'});
  }
}
