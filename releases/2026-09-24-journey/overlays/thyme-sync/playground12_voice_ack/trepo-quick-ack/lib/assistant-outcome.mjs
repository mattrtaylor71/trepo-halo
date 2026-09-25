import {actionOutcomes} from './action-outcomes.mjs';

// Transport acceptance is not proof that the requested tool action committed.
export function toolOutcome(event) {
  const result=event?.result || {};
  if(result.ok && result.needs_clarification)return 'needs_clarification';
  if(result.suppressed)return 'suppressed';
  const typed=actionOutcomes([event])[0];
  if(typed)return typed.state;
  if(!result.ok || Number(result.statusCode)>=400 || result.toolResult?.ok===false)
    return Number(result.statusCode)>=500 || Number(result.statusCode)===408 ? 'unverified' : 'failed';
  return 'confirmed';
}

export function assistantOutcome({text='',toolEvents=[],recipeOutcome=null,actionOutcome=null,reason=null}={}) {
  const states=toolEvents.map(toolOutcome).filter(x=>x!=='suppressed');
  const writes=actionOutcomes(toolEvents);
  const actions=writes.length===0 ? 'none' : writes.every(x=>x.state==='confirmed') ? 'confirmed'
    : writes.some(x=>x.entities.length) ? 'partial' : writes.every(x=>x.state==='failed') ? 'failed' : 'unconfirmed';
  let answer='complete';
  if(reason || !String(text).trim())answer='incomplete';
  else if(recipeOutcome && recipeOutcome!=='not_recipe_request')answer=recipeOutcome==='complete' ? 'complete' : recipeOutcome;
  if(states.some(x=>['failed','unverified','partial'].includes(x)))
    answer=states.some(x=>x==='confirmed'||x==='partial') ? 'partial' : 'incomplete';
  if(actionOutcome==='unconfirmed')answer='incomplete';
  if(states.includes('needs_clarification') && actions==='none' && !states.some(x=>['failed','unverified','partial'].includes(x)))answer='needs_clarification';
  return {version:'1',answer,actions,contract:recipeOutcome || 'not_applicable',
    reason:reason || (!String(text).trim() ? 'empty_answer' : null),
    delivery:'unknown',client_render:'unknown'};
}

export function outcomeMarkerStatus(outcome) {
  if(outcome.answer==='complete')return 'success';
  if(outcome.answer==='partial')return 'partial';
  if(['needs_clarification','constraints_unmet'].includes(outcome.answer))return outcome.answer;
  return 'error';
}
