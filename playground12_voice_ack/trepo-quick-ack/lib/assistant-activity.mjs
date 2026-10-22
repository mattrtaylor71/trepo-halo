import {actionOutcomes} from "./action-outcomes.mjs";
const physicalDevice = /^halo-[a-f0-9]{4}-[a-f0-9]{4}$/i;
export function assistantDeviceID(device, surface) {
  return surface !== 'app' && typeof device === 'string' && physicalDevice.test(device) ? device : null;
}
const kinds = {shopping_add:'shopping_add',shopping_remove:'shopping_remove',kitchen_add:'kitchen_item',kitchen_remove:'discard',kitchen_update:'kitchen_item',dishes:'dish'};
export function assistantActions(events) {
  if (!Array.isArray(events)) return [];
  const real = events.filter(e => e?.result?.actionMode === 'real' && e?.result?.toolResult?.count !== 0 && e?.result?.toolResult?.changed !== false);
  return actionOutcomes(real).flatMap(outcome => {
    const kind = kinds[outcome.domain];
    if (!kind) return [];
    return outcome.entities.map(entity => ({kind,status:'completed',
      title:`${outcome.label}: ${entity.name}`.slice(0,300),
      item_id:/^[A-Za-z0-9_.:-]{1,200}$/.test(entity.id) ? entity.id : null}));
  }).slice(0,50);
}

export function voiceNotification({owner,device,job,actions}) {
  const action = actions.length === 1 ? actions[0] : null;
  const all = kind => actions.length > 0 && actions.every(a => a.kind === kind);
  // Preserve the precise returned outcomes in the body. A dish update/removal
  // must not be relabeled as a newly logged dish, nor a kitchen edit as a check-in.
  const checkedIn = all('kitchen_item') && actions.every(a => a.title.startsWith('Added to your kitchen:'));
  const title = checkedIn ? (action ? 'Your item was checked in!' : 'Your items were checked in!')
    : all('shopping_add') || all('shopping_remove') ? 'Your shopping list was updated!'
    : all('kitchen_item') ? 'Your kitchen was updated!'
    : all('discard') ? (action ? 'Your item was removed!' : 'Your items were removed!')
    : all('dish') ? 'Your dish log was updated!'
    : 'Here’s what changed';
  return {owner_id:owner,title,body:actions.map(a => a.title).join(' ').slice(0,600),
    data:{route:'assistant_activity',source:'kitchen_assistant',owner_id:owner,device_id:device,
      target_kind:action?.item_id && ['dish','kitchen_item'].includes(action.kind) ? action.kind : 'activity',
      ...(action?.item_id ? {item_id:action.item_id} : {}),job_id:job,event_id:job}};
}
export async function sendVoiceNotification(args, send=fetch) {
  if (!assistantDeviceID(args.device) || !args.actions?.length) return;
  try {
    const response=await send('https://nua4yt5q26.execute-api.us-east-1.amazonaws.com/v1/notifications/send', {
      method:'POST',headers:{'Content-Type':'application/json'},signal:AbortSignal.timeout(4000),body:JSON.stringify(voiceNotification(args))});
    if (!response.ok) console.warn('[assistant-activity] notification not accepted');
  } catch (_) { console.warn('[assistant-activity] notification unavailable'); }
}
export async function recordDirectVoice({env,owner,device,surface,operation,transcript,result,events}) {
  // Async worker owns its durable job and notification. Never create duplicates.
  if (!assistantDeviceID(device,surface) || !operation || operation.startsWith('voice-job:')) return;
  const job='direct:'+operation;
  if (!/^[A-Za-z0-9_.:-]{1,100}$/.test(job)) return;
  const actions=assistantActions(events), now=new Date().toISOString();
  try {
    const {syncVoiceTriage}=await import('./triage.mjs');
    await syncVoiceTriage(env,{job_id:job,owner_id:owner,device_id:device,response_surface:surface,
      status:'completed',accepted_at:now,completed_at:now,updated_at:now,transcript,
      assistant_actions:actions,outcome:result?.outcome || null});
  } catch (_) { console.warn('[assistant-activity] direct voice history unavailable'); }
  await sendVoiceNotification({owner,device,job,actions});
}
