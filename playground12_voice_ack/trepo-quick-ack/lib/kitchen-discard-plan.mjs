import crypto from 'node:crypto';
import identity from '../../../shared/kitchen-discard/identity.cjs';
import {readKitchenEditRequest,retainKitchenEditRequest} from './session-store.mjs';

const reject=(message,statusCode=409)=>Object.assign(new Error(message),{statusCode});
export async function createDiscardIdentity(body, options, actor) {
  const {default:consent}=await import('/var/task/ai_consent_guard.cjs');
  const {default:OpenAI}=await import('openai');
  // Older entrypoints do not establish the provider's AsyncLocalStorage scope.
  // Bind this call to the actor verified by the kitchen access path, never an
  // assistant argument or another concurrent request's inherited context.
  return consent.actorContext.run(actor,async()=>{
    await consent.requireConsent(actor);
    const client=new OpenAI(consent.providerOptions({apiKey:process.env.OPENAI_API_KEY,maxRetries:0,timeout:45000}));
    return client.responses.create(body, options);
  });
}
// A durable frozen target is retained BEFORE any write. A retry cannot resolve
// the same words again and accidentally remove the next duplicate.
export async function prepareDiscardPlan(context,reference,options,snapshot) {
  const mutation=options.mutationContext;
  if (!mutation?.requestId) throw reject('Could not confirm this removal request.',503);
  // Selection comes exclusively from the current transcript. The assistant's
  // optional hint is not a new intent and may vary on an identical retry.
  const intentFor=hint=>JSON.stringify({reference:{...reference,selection_hint:hint},transcript:options.sourceTranscript || null});
  const intent=intentFor(null);
  mutation.discardIntents ||= new Map();
  if (!mutation.discardIntents.has(intent)) mutation.discardIntents.set(intent,mutation.discardIntents.size);
  const operationId=crypto.createHash('sha256').update(`${mutation.requestId}:discard:${mutation.discardIntents.get(intent)}`).digest('hex').replace(/^(.{8})(.{4})(.{4})(.{4})(.{12}).*$/, '$1-$2-$3-$4-$5');
  const actor=context.userId || context.tableOwnerId;
  // Retry identity belongs to the actor + request, not their mutable household.
  // Otherwise moving households could miss the old plan and target fresh stock.
  const scope={household:`discard-actor:${actor}`,actor,operationId,
    intentHash:crypto.createHash('sha256').update(intent).digest('hex')};
  const env=options.env || process.env;
  const readPlan=async()=>{
    // Preserve journals written by the first release, which hashed this unused
    // hint. Only the hint may differ: actor, request, item and transcript must
    // still match exactly. Other changes remain conflicts.
    let conflict;
    for(const hint of new Set([null,reference.selection_hint || null,'oldest','earliest','most_recent','newest','latest'])){
      try{return await readKitchenEditRequest({...scope,intentHash:crypto.createHash('sha256').update(intentFor(hint)).digest('hex')},env);}
      catch(error){if(error.statusCode!==409)throw error;conflict=error;}
    }
    throw conflict;
  };
  let plan=await readPlan();
  if (!plan) {
    const current=await snapshot();
    // IDs produced by a language model never bypass semantic matching of the
    // current words. A missing explicit ID still fails without substituting.
    if (reference.item_id && !current.rows.some(r=>String(r._id)===String(reference.item_id)))
      throw reject('That item is no longer in your kitchen.');
    const selected=reference.item_id && current.rows.find(r=>String(r._id)===String(reference.item_id));
    const target={product_name:reference.item_name || selected?.product_name};
    const result=await identity.matchIdentity({kind:'voice',transcript:options.sourceTranscript,
      target,candidates:current.rows},{env,create:(body,options)=>createDiscardIdentity(body,options,actor)});
    // Explicit IDs may constrain the verified identity, never expand it.
    if (selected && result.matched && !result.matching_ids.includes(String(selected._id)))
      throw reject('The requested item does not match that kitchen entry.');
    try{
      plan=await retainKitchenEditRequest(scope,{...result,operation_id:operationId,kind:'discard_identity',
        household:current.scope.ownerId,members:current.scope.householdMemberIds,actor:scope.actor},env);
    }catch(error){if(error.statusCode!==409)throw error;plan=await readPlan();}
  }
  if (plan?.operation_id!==operationId || plan.kind!=='discard_identity' || plan.actor!==scope.actor || plan.version!==identity.VERSION)
    throw reject('Could not confirm the original removal request.',503);
  if (!plan.matched) throw Object.assign(reject('I could not confidently match that to one kitchen entry. Nothing was removed.'),
    {details:{type:'discard_not_matched',reason:plan.reason,status:plan.status}});
  if (!plan.expected || plan.expected._id!==plan.selected_id || identity.fingerprint(plan.expected)!==plan.fingerprint)
    throw reject('Could not verify the original kitchen entry.',503);
  return plan;
}
export async function readDiscardRows(connection,scope,{lock=false,id=null}={}) {
  const members=scope.householdMemberIds;
  if (!members?.length) throw reject('Kitchen membership unavailable.',403);
  const [rows]=await connection.execute(`SELECT k.*, COALESCE(e.revision,0) AS amount_revision,
    JSON_UNQUOTE(JSON_EXTRACT(e.overrides,'$.__acquired_on')) AS acquired_on
    FROM shared_kitchen k LEFT JOIN kitchen_item_edits e
    ON e.item_id=CONVERT(k._id USING utf8mb4) COLLATE utf8mb4_unicode_ci
    WHERE k.owner_id IN (${members.map(()=>'?').join(',')}) AND k.action='IN'${id?' AND k._id=?':''}${lock?' FOR UPDATE':''}`,
    [...members,...(id?[id]:[])]);
  return rows;
}
export function validateDiscardSnapshot(plan,row,scope) {
  if (plan.household!==scope.ownerId || JSON.stringify(plan.members)!==JSON.stringify(scope.householdMemberIds)
      || !row || identity.fingerprint(row)!==plan.fingerprint)
    throw reject('Your kitchen changed while matching this item. Nothing else was removed.');
}
