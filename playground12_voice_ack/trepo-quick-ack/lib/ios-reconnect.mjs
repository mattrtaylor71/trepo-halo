// Temporary bridge for shipped iOS streams that omit Authorization. Authentication
// comes ONLY from the signed recovery GET. No command is admitted or replayed here.
import crypto from 'node:crypto';
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient,PutCommand,DeleteCommand} from '@aws-sdk/lib-dynamodb';
import {header,verifiedActor} from './operation-recovery.mjs';

export const COOKIE = '__Host-trepo_thyme_reconnect';
export const CALLBACK = '/ios-reconnect';
export const SUNSET = Date.parse('2026-10-11T00:00:00Z') / 1000;
const DEFAULT_ORIGIN = 'https://mkincpjehxmwd44g5ald3yinqu0xtqds.lambda-url.us-east-1.on.aws';
const seconds = () => Math.floor(Date.now()/1000);
const failure = () => Object.assign(new Error('sign_in_required'),{code:'sign_in_required',statusCode:401});
let client;
const database = () => client ||= DynamoDBDocumentClient.from(new DynamoDBClient({}));
const send = (command,db) => (db || database()).send(command,{abortSignal:AbortSignal.timeout(2500)});
function origin(env) {
  const url = new URL(env.THYME_RECONNECT_STREAM_URL || DEFAULT_ORIGIN);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash ||
      (url.port && url.port !== '443') || url.pathname !== '/') throw failure();
  return url.origin;
}
export function enabled(env,at=seconds()) {
  return env.THYME_IOS_RECONNECT_DISABLED !== 'true' && at < SUNSET;
}
function key(env,purpose) {
  if (!env.TOKEN_SIGNING_SECRET) throw failure();
  return crypto.hkdfSync('sha256',Buffer.from(env.TOKEN_SIGNING_SECRET),Buffer.from('trepo-ios-reconnect-v1'),Buffer.from(purpose+'|'+origin(env)),32);
}
function seal(value,env,purpose) {
  const nonce=crypto.randomBytes(12),cipher=crypto.createCipheriv('aes-256-gcm',key(env,purpose),nonce);
  const bytes=Buffer.concat([cipher.update(JSON.stringify(value),'utf8'),cipher.final()]);
  return Buffer.concat([nonce,cipher.getAuthTag(),bytes]).toString('base64url');
}
function unseal(value,env,purpose) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_-]+$/.test(value) || value.length > (purpose==='cookie'?3500:450000)) throw failure();
  try {
    const raw=Buffer.from(value,'base64url');
    const decipher=crypto.createDecipheriv('aes-256-gcm',key(env,purpose),raw.subarray(0,12));
    decipher.setAuthTag(raw.subarray(12,28));
    return JSON.parse(Buffer.concat([decipher.update(raw.subarray(28)),decipher.final()]).toString('utf8'));
  } catch { throw failure(); }
}
const ticketKey = ticket => ({owner_id:'thyme-ios-reconnect-v1',session_entry_id:crypto.createHash('sha256').update(ticket).digest('hex')});
const privateHeaders = {'Content-Type':'application/json','Cache-Control':'no-store, private','Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff'};

// An authenticated, membership-checked result is preserved byte for byte. Redirect
// only to the configured owned stream origin; never to a caller-provided URL.
export async function reconnectRedirect(event,env,scope,answer,db) {
  if (!enabled(env) || !scope.sessionId?.startsWith('chat_') || header(event.headers,'x-client-surface') !== 'app') return null;
  const proof=await verifiedActor(event,env);
  if (proof.actor !== scope.actor || scope.owner !== proof.actor) return null;
  const authorization=header(event.headers,'authorization');
  const claims=JSON.parse(Buffer.from(authorization.slice(7).trim().split('.')[1],'base64url').toString());
  const now=seconds(),expires=Math.min(now+86400,claims.exp,SUNSET);
  if (!Number.isFinite(expires) || expires<=now+60 || !env.SESSION_TABLE_NAME) return null;
  const cookie=seal({actor:proof.actor,authorization,expires},env,'cookie');
  if (cookie.length>3500) return null;
  const ticket=crypto.randomBytes(32).toString('base64url');
  const sealed=seal({cookie,answer,actor:proof.actor,expires},env,'ticket');
  // Only authenticated callers reach this write. No JWT, answer or raw ticket is
  // stored in plaintext; short expiry is enforced independently of DynamoDB TTL.
  await send(new PutCommand({TableName:env.SESSION_TABLE_NAME,Item:{...ticketKey(ticket),sealed,expires:now+60,ttl:now+60},
    ConditionExpression:'attribute_not_exists(owner_id) AND attribute_not_exists(session_entry_id)'}),db);
  return {statusCode:303,headers:{...privateHeaders,Location:origin(env)+CALLBACK+'?ticket='+ticket},body:''};
}

export function isReconnectCallback(event) {
  return (event.rawPath || event.path) === CALLBACK;
}
export async function reconnectCallback(event,env,db) {
  try {
    if (!enabled(env) || (event.requestContext?.http?.method || event.httpMethod) !== 'GET') throw failure();
    const ticket=event.queryStringParameters?.ticket;
    if (typeof ticket !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(ticket)) throw failure();
    const result=await send(new DeleteCommand({TableName:env.SESSION_TABLE_NAME,Key:ticketKey(ticket),ReturnValues:'ALL_OLD',
      ConditionExpression:'expires > :now',ExpressionAttributeValues:{':now':seconds()}}),db);
    const stored=unseal(result.Attributes?.sealed,env,'ticket');
    const credentials=unseal(stored.cookie,env,'cookie');
    if (stored.expires<=seconds() || credentials.expires!==stored.expires || credentials.actor!==stored.actor) throw failure();
    await verifiedActor({headers:{authorization:credentials.authorization,'x-owner-id':credentials.actor}},env);
    return {statusCode:200,headers:{...privateHeaders,'Set-Cookie':`${COOKIE}=${stored.cookie}; Path=/; Max-Age=${Math.floor(stored.expires-seconds())}; Secure; HttpOnly; SameSite=Strict`},body:JSON.stringify(stored.answer)};
  } catch {
    // Fail closed without logging the ticket/cookie/Authorization or model input.
    return {statusCode:401,headers:privateHeaders,body:JSON.stringify({error:'reconnect_expired'})};
  }
}

export async function restoreStreamAuthentication(event,env) {
  // An explicitly supplied invalid token MUST NOT fall back to a cookie.
  if (header(event.headers,'authorization') != null || !enabled(env)) return event;
  if (header(event.headers,'x-client-surface') !== 'app' || header(event.headers,'x-device-id') !== 'ios-app' || header(event.headers,'origin') != null) return event;
  const extract=sources=>sources.flatMap(raw=>String(raw).split(';')).map(v=>v.trim()).filter(v=>v.startsWith(COOKIE+'=')).map(v=>v.slice(COOKIE.length+1));
  const array=extract(Array.isArray(event.cookies)?event.cookies:[]);
  const headers=extract([header(event.headers,'cookie') || '']);
  // Function URLs expose the SAME cookie both ways. Reject actual duplicate or
  // disagreeing values, rather than rejecting the platform's mirrored field.
  if (array.length>1 || headers.length>1 || (array.length && headers.length && array[0]!==headers[0])) throw failure();
  const value=array[0] || headers[0];
  if (!value) return event;
  const credentials=unseal(value,env,'cookie');
  if (credentials.expires<=seconds() || credentials.expires>SUNSET || credentials.actor!==header(event.headers,'x-owner-id')) throw failure();
  const restored={...event,headers:{...event.headers,authorization:credentials.authorization}};
  // JWT signature/issuer/expiry are checked again on every use. The regular
  // handler still resolves current membership and performs durable admission.
  await verifiedActor(restored,env);
  return restored;
}
