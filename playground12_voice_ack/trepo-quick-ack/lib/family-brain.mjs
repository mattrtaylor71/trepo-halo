// Optional founder pilot. The service verifies signed identity; decoded claims only avoid
// network calls for non-pilot traffic. Never trust x-owner-id as personal identity.
import { createHash } from 'node:crypto';
const clients = new Map();

export function candidateOwner(authorization, env = process.env) {
  try {
    const allowed = JSON.parse(env.FAMILY_BRAIN_ACTORS || '[]');
    if (env.HOUSEHOLD_BRAIN_V2 !== '1' && (!Array.isArray(allowed) || allowed.length > 2)) return null;
    const token = String(authorization || '').replace(/^Bearer /i, '');
    if (token.split('.').length !== 3) return null;
    const claims = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString());
    return (env.HOUSEHOLD_BRAIN_V2 === '1' && /^[A-Za-z0-9_-]{1,64}$/.test(claims.user_id || '')) || allowed.includes(claims.user_id) ? claims.user_id : null;
  } catch { return null; }
}

export async function callBrain(request, env = process.env, transport = null) {
  if (!env.FAMILY_BRAIN_FUNCTION) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(env.FAMILY_BRAIN_TIMEOUT_MS || 900));
  try {
    let result;
    if (transport) result = await transport(request, controller.signal);
    else {
      const {LambdaClient, InvokeCommand} = await import('@aws-sdk/client-lambda');
      const region=env.AWS_REGION || 'us-east-1';
      if(!clients.has(region))clients.set(region,new LambdaClient({region,maxAttempts:1}));
      const client=clients.get(region);
      const response = await client.send(new InvokeCommand({FunctionName:env.FAMILY_BRAIN_FUNCTION,
        InvocationType:'RequestResponse', Payload:Buffer.from(JSON.stringify({family_brain_internal:request}))}),
        {abortSignal:controller.signal});
      if (response.FunctionError) return null;
      result = JSON.parse(Buffer.from(response.Payload).toString());
    }
    return result?.statusCode === 200 ? result : null;
  } catch {
    // No token, preference text, prompt, actor ID, or provider error in telemetry.
    console.warn(JSON.stringify({evt:'family_brain',outcome:'fallback',purpose:request.purpose || 'thyme'}));
    return null;
  } finally { clearTimeout(timer); }
}

export async function loadBrain({authorization, ticket, owner, surface='app'}, env, transport=null) {
  const actor = candidateOwner(authorization, env);
  if (surface !== 'app' || (!actor && !ticket)) return null;
  // owner is the requested user namespace, not an arbitrary household member fallback.
  if (actor && owner !== actor) return null;
  return callBrain({authorization, ticket, owner:actor || owner, purpose:'thyme', operation:'context'}, env, transport);
}

export async function prepareBrainSession(credentials, env, transport=null) {
  const brain=await loadBrain(credentials,env,transport);
  if(!brain)return null;
  const version=createHash('sha256').update(JSON.stringify([brain.fence.actor,brain.fence.household,
    brain.fence.states,brain.fence.dietary_hash,
    brain.context])).digest('hex').slice(0,32);
  return {brain,credentials:{authorization:credentials.authorization,ticket:credentials.ticket,owner:credentials.owner},
    sessionOwner:`family-brain:${credentials.owner}:${version}`};
}

export async function validateBrain(brain, credentials, env, transport=null) {
  if (!brain) return true;
  const result = await callBrain({...credentials, operation:'validate', purpose:'thyme', fence:brain.fence}, env, transport);
  return result?.current === true;
}

export const foodMemoryTool = {type:'function',function:{name:'update_food_memory',
  description:'Save an explicitly stated soft food preference, or pause/reset learning, only when the user asks. Never change allergies, dietary restrictions, religion, health, inventory or permissions. Default to personal; household sharing requires an explicit request to share with the household.',
  parameters:{type:'object',properties:{action:{type:'string',enum:['set','forget','pause','resume','reset']},
    key:{type:'string',enum:['cuisines','favorite_foods','avoid_foods','preferred_products','cooking_style','max_prep_minutes']},
    values:{type:'array',items:{type:'string'},maxItems:8},minutes:{type:'integer',minimum:5,maximum:240},
    visibility:{type:'string',enum:['personal','household']}},required:['action'],additionalProperties:false}}};

export async function updateFoodMemory(input, transport=null) {
  if (input.env?.HOUSEHOLD_BRAIN_V2 === '1') return {ok:false,error:'internally_managed'};
  const request = input.env?.FAMILY_BRAIN_REQUEST;
  const text = String(input.sourceTranscript || '');
  if (!request?.brain || !request.credentials.authorization) return {ok:false,error:'memory_unavailable'};
  if (!input.mutationContext?.requestId) return {ok:false,error:'memory_request_identity_required'};
  if (!/\b(remember|prefer|like|love|dislike|hate|forget|pause|resume|reset|stop learning|start learning)\b/i.test(text)
      || /\b(don.t|do not|never) (remember|save|change)\b/i.test(text)) return {ok:false,error:'explicit_preference_request_required'};
  // A model-selected scope cannot override the user's explicit privacy choice.
  // Ambiguous sharing language remains personal until the user asks to share.
  const sharingDenied = /\b(?:not|don['’]?t|never)\b[^.!?\n]{0,100}\b(?:shar(?:e|ed|ing)|disclos(?:e|ed|ing)|tell|told|publish)\b/i.test(text)
    || /\b(?:just|only)\s+for\s+me\b|\bkeep\b[^.!?\n]{0,40}\bprivate\b/i.test(text)
    || /\bnot\s+(?:for|with)\b[^.!?\n]{0,40}\b(?:household|family|everyone)\b/i.test(text);
  const sharingRequested = /\b(?:share|save|remember|use|apply)\b[^.!?\n]{0,80}\b(?:with|for)\b[^.!?\n]{0,30}\b(?:household|family|everyone)\b/i.test(text);
  if (input.args.visibility === 'household' && (sharingDenied || !sharingRequested))
    return {ok:false,error:'explicit_household_sharing_required'};
  const own = request.brain.fence.states.find(x=>x[0]===request.credentials.owner);
  const change = {action:input.args.action, key:input.args.key,
    value:input.args.key==='max_prep_minutes'?input.args.minutes:input.args.values,
    visibility:input.args.visibility || 'personal', expected_revision:own?.[2],
    operation_id:createHash('sha256').update(`${input.mutationContext?.requestId}:${JSON.stringify(input.args)}`).digest('hex')};
  const result = await callBrain({...request.credentials,operation:'control',change},input.env,transport);
  if (!result?.ok) return {ok:false,error:'memory_not_saved',statusCode:409};
  // Refresh after our own successful control; do not reject our answer as stale.
  const fresh = await callBrain({...request.credentials,operation:'context',purpose:'thyme'},input.env,transport);
  if (fresh) request.brain = fresh;
  return {ok:true,statusCode:200,toolName:'update_food_memory',toolResult:{saved:true,revision:result.revision,scope:change.visibility},actionSummary:'Food preference updated.'};
}
