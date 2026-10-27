// Server gate. An absent record is compatibility for an older installed app,
// never a recorded consent. New clients activate a pending record before UI.
const { DynamoDBClient, GetItemCommand } = require('@aws-sdk/client-dynamodb');
const client = new DynamoDBClient({});
const {AsyncLocalStorage}=require('node:async_hooks');
const actorContext=new AsyncLocalStorage();
const currentActor=()=>actorContext.getStore();
const VERSION = '2026-10-22.v1';
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
async function allowed(owner, db = client) {
  if (typeof owner !== 'string' || !uuid.test(owner)) return false;
  const { Item } = await db.send(new GetItemCommand({TableName:'trepo-current-user-consents',
    Key:{user_id:{S:'ai:'+owner.toLowerCase()}},ConsistentRead:true}));
  if (!Item) return process.env.AI_CONSENT_REQUIRE_ALL !== 'true';
  return Item.policy_version?.S === VERSION && Item.allowed?.BOOL === true;
}
async function requireConsent(owner, db) {
  if (!await allowed(owner, db)) {
    const error = new Error('AI processing is off. Review your choice in Trepo > Profile > AI privacy.');
    error.code = 'ai_consent_required'; error.status = 428; throw error;
  }
}
function response() {
  return {statusCode:428,headers:{'Content-Type':'application/json','Cache-Control':'no-store'},
    body:JSON.stringify({error:'ai_consent_required',message:'Review AI privacy in Trepo before using this feature.'})};
}
function eventOwner(event) {
  let body = event?.body || {};
  if (typeof body === 'string') {
    try { body = JSON.parse(event.isBase64Encoded ? Buffer.from(body, 'base64').toString('utf8') : body); }
    catch { body = {}; }
  }
  const headers = Object.fromEntries(Object.entries(event?.headers || {}).map(([k,v])=>[k.toLowerCase(),v]));
  for (const key of ['user_id','userId','owner','owner_id','ownerId']) {
    for (const obj of [body,event?.payload,event?.pathParameters,event]) {
      if (typeof obj?.[key] === 'string' && uuid.test(obj[key])) return obj[key].toLowerCase();
    }
  }
  for (const key of ['x-owner-id','x-user-id']) if (typeof headers[key] === 'string' && uuid.test(headers[key])) return headers[key].toLowerCase();
  return null;
}
async function requestActor(event) {
  const headers=Object.fromEntries(Object.entries(event?.headers||{}).map(([k,v])=>[k.toLowerCase(),v]));
  if(!headers.authorization)return eventOwner(event);
  const response=await fetch('https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com/v1/consents/ai',{
    method:'POST',redirect:'error',signal:AbortSignal.timeout(5000),
    headers:{Authorization:headers.authorization,'Content-Type':'application/json'},
    body:JSON.stringify({version:VERSION,action:'check'})});
  if(!response.ok)throw Object.assign(new Error('Consent identity unavailable'),{code:response.status===401?'ai_consent_required':'ai_consent_unavailable'});
  const result=await response.json();
  if(typeof result.user_id!=='string'||!uuid.test(result.user_id))throw new Error('Invalid consent identity');
  return result.user_id;
}
function guardHandler(handler) {
  return async (event, context) => {
    if ((event?.requestContext?.http?.method || event?.httpMethod) === 'OPTIONS') return handler(event, context);
    let actor;
    try {
      actor=await requestActor(event); await requireConsent(actor);
      // Carry the authenticated actor in the existing user_id attribution field
      // that identifier admission persists in its jobs and retry payloads.
      if(event?.headers && Object.keys(event.headers).some(k=>k.toLowerCase()==='authorization')){
        if(typeof event.body==='string'){
          const body=JSON.parse(event.isBase64Encoded?Buffer.from(event.body,'base64').toString('utf8'):event.body);
          body.user_id=actor;event={...event,body:JSON.stringify(body),isBase64Encoded:false};
        }else if(event.body&&typeof event.body==='object')event={...event,body:{...event.body,user_id:actor}};
      }
    }
    catch (error) { return error.code === 'ai_consent_required' ? response() : {statusCode:503,body:JSON.stringify({error:'ai_consent_unavailable'})}; }
    return actorContext.run(actor,()=>handler(event, context));
  };
}
async function requireLegacyProvider(event) {
  const actor=await requestActor(event);
  if(typeof actor!=='string'||!uuid.test(actor))throw Object.assign(new Error('Identity required'),{code:'ai_consent_required'});
  const {Item}=await client.send(new GetItemCommand({TableName:'trepo-current-user-consents',Key:{user_id:{S:'ai:'+actor.toLowerCase()}},ConsistentRead:true}));
  // The new disclosure grants OpenAI/Google only. Retired Anthropic routes must
  // never interpret that permission as authorization for another provider.
  if(Item||process.env.AI_CONSENT_REQUIRE_ALL==='true')throw Object.assign(new Error('This legacy AI feature is unavailable'),{code:'ai_consent_required'});
}
module.exports = {actorContext, currentActor, requireLegacyProvider, VERSION, allowed, requireConsent, response, eventOwner, requestActor, guardHandler};

module.exports.bodyOperation = event => {
  try { const body=typeof event?.body==='string'?JSON.parse(event.isBase64Encoded?Buffer.from(event.body,'base64').toString('utf8'):event.body):event?.body;return body?.operation; }
  catch { return undefined; }
};

// Recheck at the actual HTTP attempt, including an SDK's own retry transport.
function providerFetch(fetcher, providerOnly = false) {
  return async (input, init) => {
    const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
    if (providerOnly || ['api.openai.com','generativelanguage.googleapis.com'].includes(url.hostname)) await requireConsent(currentActor());
    return fetcher(input, init);
  };
}
module.exports.providerFetch = providerFetch;
module.exports.providerOptions = (options = {}) => ({...options, fetch:providerFetch(options.fetch || globalThis.fetch, true)});
