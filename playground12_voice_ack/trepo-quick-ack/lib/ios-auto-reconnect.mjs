// Compatibility for released iOS streams that omit Authorization. Only a signed
// ordinary List read can establish the first credential. UUIDs are never proof.
import crypto from 'node:crypto';
import protocol from './ios-auto-proxy.cjs';
const {allowed,eligible,fingerprint}=protocol;
import {jwtVerify} from 'jose';
import {DynamoDBClient} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient,PutCommand,DeleteCommand} from '@aws-sdk/lib-dynamodb';

export const API_COOKIE='__Host-trepo_thyme_api';
export const STREAM_COOKIE='__Host-trepo_thyme_auto';
export const CALLBACK='/ios-auto-reconnect';
const API='https://1zc0nh8x48.execute-api.us-east-1.amazonaws.com/v1/list';
const STREAM='https://mkincpjehxmwd44g5ald3yinqu0xtqds.lambda-url.us-east-1.on.aws/';
const now=()=>Math.floor(Date.now()/1000);
const header=(e,k)=>Object.entries(e.headers || {}).find(([n])=>n.toLowerCase()===k)?.[1];
const method=e=>e.requestContext?.http?.method || e.httpMethod;
const failure=(code='sign_in_required')=>Object.assign(new Error(code),{code});
const privateHeaders={'Cache-Control':'no-store, private','Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff'};
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
let database;
const send=(cmd,db)=>(db || (database ||= DynamoDBDocumentClient.from(new DynamoDBClient({})))).send(cmd,{abortSignal:AbortSignal.timeout(2500)});
export const enabled=env=>env.THYME_AUTO_RECONNECT_ENABLED==='true';
function endpoints(env){
 let api,stream;try {api=new URL(env.THYME_BOOTSTRAP_API_URL || API);stream=new URL(env.THYME_BOOTSTRAP_STREAM_URL || STREAM);} catch {throw failure('configuration_error');}
 for(const u of [api,stream])if(u.protocol!=='https:' || u.username || u.password || u.search || u.hash || (u.port && u.port!=='443'))throw failure('configuration_error');
 if(stream.pathname!=='/')throw failure('configuration_error');
 return {api,stream};
}
function key(env,purpose){
 if(!/^[a-f0-9]{64}$/.test(env.THYME_BRIDGE_SECRET || ''))throw failure('configuration_error');
 const {api,stream}=endpoints(env);
 return crypto.hkdfSync('sha256',Buffer.from(env.THYME_BRIDGE_SECRET,'hex'),Buffer.from('trepo-ios-auto-isolated-v2'),Buffer.from(purpose+'|'+api.origin+'|'+stream.origin),32);
}
function seal(value,env,purpose){
 const iv=crypto.randomBytes(12),c=crypto.createCipheriv('aes-256-gcm',key(env,purpose),iv);
 const result=Buffer.concat([c.update(JSON.stringify(value),'utf8'),c.final()]);
 return Buffer.concat([iv,c.getAuthTag(),result]).toString('base64url');
}
function unseal(value,env,purpose){
 if(typeof value!=='string' || value.length>3500 || !/^[A-Za-z0-9_-]+$/.test(value))throw failure();
 try{const b=Buffer.from(value,'base64url'),d=crypto.createDecipheriv('aes-256-gcm',key(env,purpose),b.subarray(0,12));d.setAuthTag(b.subarray(12,28));return JSON.parse(Buffer.concat([d.update(b.subarray(28)),d.final()]).toString('utf8'));}catch(error){if(error.code==='configuration_error')throw error;throw failure();}
}
async function verify(authorization,actor,env){
 if(typeof authorization!=='string' || !/^Bearer /i.test(authorization) || authorization.length>2500 || !uuid.test(actor))throw failure();
 if(!env.TOKEN_SIGNING_SECRET)throw failure('configuration_error');
 let p;try{({payload:p}=await jwtVerify(authorization.slice(7).trim(),new TextEncoder().encode(env.TOKEN_SIGNING_SECRET),{issuer:'trepo-auth',algorithms:['HS256'],requiredClaims:['exp','user_id','owner_id']}));}catch{throw failure();}
 if(p.user_id!==actor || typeof p.owner_id!=='string' || !/^[a-zA-Z0-9_-]{1,64}$/.test(p.owner_id))throw failure();
 return p;
}
function cookieValue(event,name){
 const extract=xs=>xs.flatMap(s=>String(s).split(';')).map(s=>s.trim()).filter(s=>s.startsWith(name+'=')).map(s=>s.slice(name.length+1));
 const a=extract(Array.isArray(event.cookies)?event.cookies:[]),b=extract([header(event,'cookie') || '']);
 if(a.length>1 || b.length>1 || (a.length && b.length && a[0]!==b[0]))throw failure();
 return a[0] || b[0];
}
async function readCookie(e,env,name,purpose){
 const c=unseal(cookieValue(e,name),env,purpose);
 if(c.actor!==header(e,'x-owner-id') || !Number.isFinite(c.expires) || c.expires<=now())throw failure();
 const p=await verify(c.authorization,c.actor,env);
 if(c.expires>p.exp)throw failure();
 return c;
}
const cookieHeader=(name,value,expires)=>`${name}=${value}; Path=/; Max-Age=${Math.floor(expires-now())}; Secure; HttpOnly; SameSite=Strict`;
function bodyBytes(e){return Buffer.from(e.body || '',e.isBase64Encoded?'base64':'utf8');}
const ticketKey=t=>({owner_id:'thyme-ios-auto-isolated-v2',session_entry_id:crypto.createHash('sha256').update(t).digest('hex')});
const responseError=code=>{
 const infrastructure=!['sign_in_required','invalid_callback','reconnect_expired'].includes(code);
 return {statusCode:infrastructure?503:401,headers:{...privateHeaders,'Content-Type':'application/x-ndjson'},body:JSON.stringify({type:'error',message:infrastructure?'Please try again shortly.':'Please reopen Trepo and try again. If this continues, sign in again.',code})+'\n'};
};
export const isBroker=e=>e.queryStringParameters?.thyme_bootstrap==='1';
export const isCallback=e=>(e.rawPath || e.path)===CALLBACK;

// No extra network/database call on a normal read. Failure must never turn a
// working shopping-list response into a failed read. Existing body is untouched.
export async function decorateListResponse(e,response,env){
 if(!enabled(env) || method(e)!=='POST' || response?.statusCode!==200 || header(e,'origin')!=null)return response;
 try{
  const body=JSON.parse(bodyBytes(e).toString('utf8'));
  if(body.operation!=='view' || !allowed(body.ownerId,env))return response;
  const authorization=header(e,'authorization'),p=await verify(authorization,body.ownerId,env);
  const expires=Math.min(now()+86400,p.exp);if(expires<=now()+60)return response;
  const value=seal({actor:body.ownerId,authorization,expires},env,'api');if(value.length>3500)return response;
  return {...response,headers:{...response.headers,...privateHeaders},cookies:[...(response.cookies || []),cookieHeader(API_COOKIE,value,expires)]};
 }catch{return response;}
}
async function issueTicket(e,hash,env,db){
 try{
  if(!allowed(header(e,'x-owner-id'),env) || !eligible(e) || header(e,'authorization')!=null || !/^[a-f0-9]{64}$/.test(hash || ''))throw failure();
  const c=await readCookie(e,env,API_COOKIE,'api');
  if(!env.SESSION_TABLE_NAME)throw failure('configuration_error');
  const ticket=crypto.randomBytes(32).toString('base64url');
  await send(new PutCommand({TableName:env.SESSION_TABLE_NAME,Item:{...ticketKey(ticket),sealed:seal(c,env,'ticket'),fingerprint:hash,expires:now()+45,ttl:now()+45},ConditionExpression:'attribute_not_exists(owner_id) AND attribute_not_exists(session_entry_id)'}),db);
  const url=new URL(CALLBACK,endpoints(env).stream);url.searchParams.set('ticket',ticket);
  return {statusCode:307,headers:{...privateHeaders,Location:url.href},body:'{}\n'};
 }catch(error){return responseError(error.code || 'reconnect_unavailable');}
}

// Called before any transcription, assistant execution, or durable operation claim.
export async function prepareStream(e,env,db){
 if(!allowed(header(e,'x-owner-id'),env))return isCallback(e)?{response:responseError('sign_in_required')}:{event:e};
 if(header(e,'authorization')!=null)return isCallback(e)?{response:responseError('invalid_callback')}:{event:e};
 if(!eligible(e))return isCallback(e)?{response:responseError('invalid_callback')}:{event:e};
 if(isCallback(e)){
  try{
   const t=e.queryStringParameters?.ticket;if(typeof t!=='string' || !/^[A-Za-z0-9_-]{43}$/.test(t))throw failure();
   if(!env.SESSION_TABLE_NAME)throw failure('configuration_error');
   const result=await send(new DeleteCommand({TableName:env.SESSION_TABLE_NAME,Key:ticketKey(t),ReturnValues:'ALL_OLD',ConditionExpression:'expires > :now AND fingerprint = :hash',ExpressionAttributeValues:{':now':now(),':hash':fingerprint(e)}}),db);
   const c=unseal(result.Attributes?.sealed,env,'ticket');if(c.actor!==header(e,'x-owner-id') || c.expires<=now())throw failure();
   const p=await verify(c.authorization,c.actor,env);if(c.expires>p.exp)throw failure();
   const value=seal(c,env,'stream');
   return {event:{...e,headers:{...e.headers,authorization:c.authorization}},headers:{'Set-Cookie':cookieHeader(STREAM_COOKIE,value,c.expires)}};
  }catch(error){return {response:responseError(error.code || (error.name==='ConditionalCheckFailedException' || error.message==='ConditionalCheckFailed'?'reconnect_expired':'reconnect_unavailable'))};}
 }
 try{
  const c=await readCookie(e,env,STREAM_COOKIE,'stream');
  return {event:{...e,headers:{...e.headers,authorization:c.authorization}}};
 }catch{
  // A stale or other-account cookie cannot authenticate; a fresh, signed API-host
  // cookie may replace it through the bounded two-redirect handshake instead.
  try {
   const url=endpoints(env).api;url.searchParams.set('thyme_bootstrap','1');
   return {response:{statusCode:307,headers:{...privateHeaders,Location:url.href},body:'{}\n'}};
  } catch { return {response:{...responseError('reconnect_unavailable'),statusCode:503}}; }
 }
}

// Only a direct IAM invocation can supply this root-level envelope. Public HTTP
// bodies are not unwrapped or trusted as internal invocations. JWT/cookie proof
// and the independently configured cohort remain mandatory even for IAM callers.
export const isInternalBootstrap=e=>e?.thymeBootstrap==='v2';
export async function internalBootstrap(e,env,db){
 try {
  if(!isInternalBootstrap(e) || e.requestContext || e.httpMethod || e.rawPath || e.path || e.headers || e.body || e.cookies)throw failure();
  if(e.action==='seed'){
   if(Object.keys(e).some(k=>!['thymeBootstrap','action','actor','authorization'].includes(k)) || !allowed(e.actor,env))throw failure();
   await verify(e.authorization,e.actor,env);key(env,'api');
   const result=await decorateListResponse({requestContext:{http:{method:'POST'}},headers:{authorization:e.authorization},body:JSON.stringify({operation:'view',ownerId:e.actor})},{statusCode:200},env);
   if(!result.cookies)throw failure();
   return {statusCode:200,cookies:result.cookies};
  }
  if(e.action==='ticket'){
   if(Object.keys(e).some(k=>!['thymeBootstrap','action','request','fingerprint'].includes(k)) || !e.request || Object.keys(e.request).some(k=>!['requestContext','headers','cookies'].includes(k)))throw failure();
   return issueTicket(e.request,e.fingerprint,env,db);
  }
  throw failure();
 } catch(error){return responseError(error.code || 'reconnect_unavailable');}
}
