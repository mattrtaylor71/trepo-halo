import {detectActionClaims} from './action-claims.mjs';
import {actionOutcomes,actionOutcomeText} from './action-outcomes.mjs';

// This boundary can repair words only. It deliberately has no action executor.
// All positive confirmations are rendered from typed, committed tool results.
export async function reconcileActionNarration({text,toolEvents=[],repair}) {
  const outcomes=actionOutcomes(toolEvents);
  const questions=toolEvents.filter(e=>e.result?.needs_clarification === true && e.result?.toolResult?.inventory_changed === false)
    .map(e=>e.result.toolResult.confirmation_question).filter(q=>typeof q==='string' && q.trim());
  const question=[...new Set(questions)].join('\n');
  const claims=detectActionClaims(text);
  if (outcomes.length) {
    let answer=actionOutcomeText(outcomes);
    if (claims.some(claim=>!outcomes.some(o=>o.domain===claim.domain))) answer+='\n\nI haven’t confirmed any other changes.';
    if (question) answer+='\n\n'+question;
    return {text:answer,outcome:outcomes.every(o=>o.state==='confirmed') ? 'confirmed' : outcomes.some(o=>o.entities.length) ? 'partial' : 'unconfirmed',outcomes};
  }
  if (question) return {text:question,outcome:'needs_clarification',outcomes:[]};
  if (!claims.length) return {text,outcome:'read_only',outcomes:[]};
  if (repair) {
    try {
      const candidate=await repair();
      if (typeof candidate==='string' && candidate.trim() && !detectActionClaims(candidate).length) return {text:candidate.trim(),outcome:'answer_repaired',outcomes:[]};
    } catch { /* Bounded answer repair failed; never retry a mutation. */ }
  }
  return {text:"I haven't made that change. Please check the app before trying again.",outcome:'unconfirmed',outcomes:[]};
}
