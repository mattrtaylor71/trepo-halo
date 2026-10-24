// List-side transport only. No login key, bridge key, JWT parsing or database access.
const crypto = require('node:crypto');
const API_COOKIE = '__Host-trepo_thyme_api';
const STREAM_COOKIE = '__Host-trepo_thyme_auto';
const CALLBACK = '/ios-auto-reconnect';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const header = (e, k) => Object.entries(e.headers || {}).find(([n]) => n.toLowerCase() === k)?.[1];
const method = e => e.requestContext?.http?.method || e.httpMethod;
const privateHeaders = {'Cache-Control':'no-store, private','Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff'};
const fields = ['x-owner-id','x-device-id','x-client-surface','x-session-id','x-operation-id','x-client-time-zone','content-type','x-audio-sample-rate','x-audio-format'];
const bodyBytes = e => Buffer.from(e.body || '', e.isBase64Encoded ? 'base64' : 'utf8');
function allowed(actor, env) {
  if (env.THYME_AUTO_RECONNECT_ENABLED !== 'true' || !UUID.test(actor || '')) return false;
  // Audience is not authentication: Auth and Stream still independently verify
  // the signed credential, actor binding, expiry and one-use request ticket.
  // Keep the old pilot as the default and a reversible rollback destination.
  const audience = env.THYME_AUTO_RECONNECT_AUDIENCE ?? 'pilot';
  if (audience === 'all_valid_sessions') return true;
  if (audience !== 'pilot') return false;
  const entries = (env.THYME_AUTO_RECONNECT_ACTORS || '').split(',').map(x => x.trim()).filter(Boolean);
  // Misconfiguration fails closed; wildcards and partial IDs are never supported.
  return entries.length > 0 && entries.length <= 20 && entries.every(x => UUID.test(x)) && entries.includes(actor);
}
function eligible(e) {
  if (method(e) !== 'POST' || header(e, 'origin') != null || header(e, 'x-device-id') !== 'ios-app' || header(e, 'x-client-surface') !== 'app' || !UUID.test(header(e, 'x-owner-id') || '')) return false;
  const match = /^v1\.(\d{10,11})\.([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$/i.exec(header(e, 'x-operation-id') || '');
  const now = Math.floor(Date.now()/1000);
  return !!match && +match[1] <= now+300 && +match[1]+86400 > now && /^chat_[0-9a-f]{8}-[0-9a-f]{3}$/i.test(header(e, 'x-session-id') || '');
}
function fingerprint(e) {
  const bytes = bodyBytes(e);
  if (!bytes.length || bytes.length > 6000000) throw Error('invalid_request');
  return crypto.createHash('sha256').update(JSON.stringify(fields.map(n => header(e,n) ?? null))).update(bytes).digest('hex');
}
function metadata(e) {
  const headers = {};
  for (const n of [...fields, 'authorization','origin','cookie']) if (header(e,n) != null) headers[n] = header(e,n);
  return {requestContext:{http:{method:method(e)}}, headers, cookies:e.cookies};
}
const unavailable = () => ({statusCode:503,headers:{...privateHeaders,'Content-Type':'application/x-ndjson'},body:JSON.stringify({type:'error',message:'Please reopen Trepo and try again.',code:'reconnect_unavailable'})+'\n'});
let client;
function invokeBudget(context) {
  const remaining = context?.getRemainingTimeInMillis?.() ?? 4500;
  if (!Number.isFinite(remaining) || remaining < 750) throw Error('deadline');
  return Math.min(4000, remaining - 500);
}
async function invoke(payload, env, transport, context) {
  const budget = invokeBudget(context);
  const target = env.THYME_BOOTSTRAP_FUNCTION;
  if (!/^arn:aws:lambda:us-east-1:566667681926:function:(twilioAuth|trepo-thyme-isolated-qa-20260927-auth)$/.test(target || '')) throw Error('invalid_target');
  if (transport) {
    let timer;
    try { return await Promise.race([Promise.resolve().then(()=>transport(payload)), new Promise((_, reject)=>{timer=setTimeout(()=>reject(Error('deadline')),budget);})]); }
    finally { clearTimeout(timer); }
  }
  const {LambdaClient, InvokeCommand} = require('@aws-sdk/client-lambda');
  client ||= new LambdaClient({maxAttempts:1});
  const reply = await client.send(new InvokeCommand({FunctionName:target,InvocationType:'RequestResponse',Payload:Buffer.from(JSON.stringify(payload))}),{abortSignal:AbortSignal.timeout(budget)});
  if (reply.FunctionError || reply.StatusCode !== 200 || !reply.Payload || reply.Payload.length > 16000) throw Error('reconnect_unavailable');
  return JSON.parse(Buffer.from(reply.Payload).toString('utf8'));
}
async function decorateListResponse(e, response, env, transport, context) {
  if (method(e) !== 'POST' || response?.statusCode !== 200 || header(e,'origin') != null) return response;
  try {
    const body = JSON.parse(bodyBytes(e).toString('utf8'));
    if (body.operation !== 'view' || !allowed(body.ownerId,env) || !/^Bearer /i.test(header(e,'authorization') || '')) return response;
    const result = await invoke({thymeBootstrap:'v2',action:'seed',actor:body.ownerId,authorization:header(e,'authorization')},env,transport,context);
    if (result.statusCode !== 200 || !Array.isArray(result.cookies) || result.cookies.length !== 1 || typeof result.cookies[0] !== 'string' || result.cookies[0].length > 4096 || !result.cookies[0].startsWith(API_COOKIE+'=') || /[\r\n]/.test(result.cookies[0])) return response;
    return {...response,headers:{...response.headers,...privateHeaders},cookies:[...(response.cookies || []),result.cookies[0]]};
  } catch { return response; }
}
async function broker(e, env, transport, context) {
  try {
    if (!eligible(e) || !allowed(header(e,'x-owner-id'),env) || header(e,'authorization') != null) return unavailable();
    const request = metadata(e);
    // The auth service never receives the user's text/audio body, only its digest.
    const result = await invoke({thymeBootstrap:'v2',action:'ticket',request,fingerprint:fingerprint(e)},env,transport,context);
    if (result.statusCode === 401) return {statusCode:401,headers:{...privateHeaders,'Content-Type':'application/x-ndjson'},body:'{"type":"error","code":"sign_in_required","message":"Please sign in again."}\n'};
    const expected = new URL(CALLBACK,env.THYME_BOOTSTRAP_STREAM_URL || 'https://mkincpjehxmwd44g5ald3yinqu0xtqds.lambda-url.us-east-1.on.aws/');
    if (expected.protocol !== 'https:' || expected.username || expected.password || (expected.port && expected.port !== '443')) throw Error('invalid_configuration');
    const actual = new URL(result.headers?.Location);
    if (result.statusCode !== 307 || actual.origin !== expected.origin || actual.pathname !== CALLBACK || actual.username || actual.password || actual.hash || !/^[A-Za-z0-9_-]{43}$/.test(actual.searchParams.get('ticket') || '') || [...actual.searchParams.keys()].join(',') !== 'ticket') throw Error('invalid_redirect');
    return {statusCode:307,headers:{...privateHeaders,Location:actual.href},body:'{}\n'};
  } catch { return unavailable(); }
}
module.exports = {API_COOKIE,STREAM_COOKIE,CALLBACK,header,method,privateHeaders,bodyBytes,allowed,eligible,fingerprint,decorateListResponse,broker};
