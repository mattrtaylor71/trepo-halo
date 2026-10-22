import crypto from 'node:crypto';
import { jwtVerify } from 'jose';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } from '@aws-sdk/lib-dynamodb';

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const now = () => Math.floor(Date.now()/1000);
const fail = (code, statusCode = 409) => Object.assign(new Error(code), {code, statusCode});
export const header = (headers, name) => Object.entries(headers || {}).find(([k]) => k.toLowerCase() === name)?.[1];
let client;
const database = () => client ||= DynamoDBDocumentClient.from(new DynamoDBClient({}), {marshallOptions:{removeUndefinedValues:true}});
const send = (command, db) => (db || database()).send(command, {abortSignal:AbortSignal.timeout(2500)});

export function operationIdentity(value, seconds = now()) {
  if (value == null || value === '') return null;
  if (typeof value !== 'string') throw fail('invalid_operation_id',400);
  const m = /^v1\.(\d{10,11})\.([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$/i.exec(value);
  if (!m) throw fail('invalid_operation_id',400);
  const created = Number(m[1]);
  if (created > seconds+300) throw fail('future_operation_id');
  if (created+86400 <= seconds) throw fail('expired_operation_id');
  return {id:`v1.${created}.${m[2].toLowerCase()}`,expires:created+86400};
}

// Shipped iOS clients use chat_ + UUID().uuidString.prefix(12); Android uses
// the complete UUID. Preserve each exact session key for receipt ownership.
export function validSessionId(value) {
  return typeof value === 'string' && (
    (value.length === 36 && uuid.test(value)) ||
    (value.length === 17 && /^chat_[0-9a-f]{8}-[0-9a-f]{3}$/i.test(value))
  );
}

export function inputIdentity(event, input) {
  const body = input?.requestMeta || {};
  const rawHeader = header(event.headers,'x-operation-id');
  // Older iOS/Halo clients use ordinary correlation UUIDs in this header.
  // Only the explicit v1 receipt protocol opts into durable admission.
  if (![rawHeader,body.operation_id].some(v=>typeof v==='string' && v.startsWith('v1.'))) return null;
  const a = operationIdentity(rawHeader);
  const b = operationIdentity(body.operation_id);
  if (a && b && a.id !== b.id) throw fail('conflicting_operation_id',400);
  const identity = a || b;
  if (!identity) return null;
  if (input.responseSurface !== 'app' || !validSessionId(input.sessionId)) throw fail('invalid_session_id',400);
  const audio = input.audioBuffer;
  const value = audio ? crypto.createHash('sha256').update(audio).digest('hex') : String(input.transcript || '').trim();
  if (!value) throw fail('empty_request',400);
  const fingerprint = crypto.createHash('sha256').update(JSON.stringify([
    input.sessionId, audio?'audio':'text',value,
    audio ? input.audioSampleRate || body.audio_sample_rate || header(event.headers,'x-audio-sample-rate') : null,
    audio ? input.audioFormat || body.audio_format || header(event.headers,'x-audio-format') || header(event.headers,'content-type') : null,
    header(event.headers,'x-client-time-zone') || null,
  ])).digest('hex');
  return {...identity,sessionId:input.sessionId,fingerprint};
}

export async function verifiedActor(event, env) {
  const value = header(event.headers,'authorization');
  if (typeof value !== 'string' || !/^Bearer /i.test(value) || value.length > 8192) throw fail('sign_in_required',401);
  if (!env.TOKEN_SIGNING_SECRET) throw fail('recovery_unavailable',503);
  let payload;
  try {
    ({payload} = await jwtVerify(value.slice(7).trim(), new TextEncoder().encode(env.TOKEN_SIGNING_SECRET),
      {issuer:'trepo-auth',algorithms:['HS256'],requiredClaims:['exp','user_id','owner_id']}));
  } catch { throw fail('sign_in_required',401); }
  // Trepo's signed user ID is a UUID; older household IDs are decimal strings.
  if (!uuid.test(payload.user_id) || typeof payload.owner_id !== 'string' ||
      !/^[a-zA-Z0-9_-]{1,64}$/.test(payload.owner_id)) throw fail('sign_in_required',401);
  // The requested namespace must be the signed user's namespace. Membership is
  // resolved again server-side; a header alone never authorizes transcript access.
  const owner = header(event.headers,'x-owner-id');
  if (owner !== payload.user_id && owner !== payload.owner_id) throw fail('owner_mismatch',403);
  return {actor:payload.user_id,owner};
}

export function operationScope(identity, proof, context, sessionOwner, env) {
  if (!env.SESSION_TABLE_NAME || !identity || !sessionOwner) throw fail('recovery_unavailable',503);
  if (context.isFallbackContext || context.userId !== proof.actor ||
      (proof.owner !== proof.actor && proof.owner !== context.ownerId)) throw fail('owner_mismatch',403);
  return {...identity,actor:proof.actor,owner:proof.owner,table:env.SESSION_TABLE_NAME,
    key:{owner_id:sessionOwner,session_entry_id:`android-operation#v1#${proof.actor}#${identity.id}`}};
}

export async function readOperation(scope, db) {
  const result = await send(new GetCommand({TableName:scope.table,Key:scope.key,ConsistentRead:true}),db);
  const row = result.Item;
  if (!row || row.actor !== scope.actor || row.session_id !== scope.sessionId) return {status:'not_found'};
  if (row.operation_expires_at <= now()) return {status:'expired'};
  if (scope.fingerprint && row.fingerprint !== scope.fingerprint) return {status:'input_conflict'};
  if (row.request_status === 'completed' && row.response_body) return {status:'completed',body:row.response_body};
  return {status:row.request_status === 'unknown' || row.lease_expires_at <= now() ? 'unknown_commit' : 'in_progress'};
}

export async function claimOperation(scope, db) {
  const token = crypto.randomUUID();
  try {
    await send(new PutCommand({TableName:scope.table,Item:{...scope.key,actor:scope.actor,
      session_id:scope.sessionId,operation_id:scope.id,fingerprint:scope.fingerprint,
      request_status:'processing',claim_token:token,lease_expires_at:now()+180,
      operation_expires_at:scope.expires,ttl:scope.expires+7*86400},
      ConditionExpression:'attribute_not_exists(owner_id) AND attribute_not_exists(session_entry_id)'}),db);
    return {claimed:true,token};
  } catch (e) {
    if (e.name !== 'ConditionalCheckFailedException') throw fail('recovery_unavailable',503);
    const prior = await readOperation(scope,db);
    return {claimed:false,...prior};
  }
}

export async function finishOperation(scope, claim, body, db) {
  const saved = {...body,operation:{id:scope.id,status:'completed',render_ack:true}};
  if (Buffer.byteLength(JSON.stringify(saved)) > 300000) throw fail('receipt_too_large',503);
  await send(new UpdateCommand({TableName:scope.table,Key:scope.key,
    UpdateExpression:'SET request_status = :completed, response_body = :body',
    ConditionExpression:'claim_token = :token AND request_status = :processing AND session_id = :session',
    ExpressionAttributeValues:{':completed':'completed',':body':saved,':token':claim.token,':processing':'processing',':session':scope.sessionId}}),db);
  return saved;
}

export async function markOperationUnknown(scope, claim, db) {
  if (!scope || !claim?.claimed) return;
  try {
    await send(new UpdateCommand({TableName:scope.table,Key:scope.key,
      UpdateExpression:'SET request_status = :unknown',
      ConditionExpression:'claim_token = :token AND request_status = :processing',
      ExpressionAttributeValues:{':unknown':'unknown',':token':claim.token,':processing':'processing'}}),db);
  } catch { /* Preserve a processing record on storage outage; never allow reexecution. */ }
}

export async function acknowledgeOperation(scope, db) {
  try {
    await send(new UpdateCommand({TableName:scope.table,Key:scope.key,
      UpdateExpression:'SET client_rendered_at = if_not_exists(client_rendered_at, :date)',
      ConditionExpression:'request_status = :completed AND actor = :actor AND session_id = :session AND operation_expires_at > :now',
      ExpressionAttributeValues:{':date':new Date().toISOString(),':completed':'completed',':actor':scope.actor,':session':scope.sessionId,':now':now()}}),db);
    return {acknowledged:true};
  } catch(e) { if(e.name==='ConditionalCheckFailedException')return {acknowledged:false};throw fail('recovery_unavailable',503); }
}

export function stoppedBody(scope, status) {
  const text = status === 'in_progress' ? 'Your original request is still finishing. Check its result shortly.' :
    status === 'input_conflict' ? 'This request does not match the original. Your original request has not been repeated.' :
    'The result of your original request could not be confirmed. It has not been repeated.';
  return {text,operation:{id:scope.id,status},app_output:null};
}
