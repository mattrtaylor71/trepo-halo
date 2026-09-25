// Completion means analysis finished; it does not mean items were found/committed.
function captureOutcome(result, index=0) {
 const count=Array.isArray(result?.items)?result.items.length:0;
 const type=['receipt','product','scene','unreadable'].includes(result?.capture_type)?result.capture_type:'unknown';
 return {image_index:index,capture_type:type,item_count:count,
   outcome:count?'items_identified':['product','scene'].includes(type)?'wrong_capture_type':'no_items'};
}
function aggregateCaptureOutcomes(results) {
 const outcomes=results.map((r,i)=>captureOutcome(r,i));
 return {capture_outcome:outcomes.some(r=>r.item_count>0)?'items_identified':outcomes.some(r=>r.outcome==='wrong_capture_type')?'wrong_capture_type':'no_items',
   capture_outcomes:outcomes};
}
// Classification is model output; recovery availability is server-owned.
// A verified manifest was attached only after retained source uploads succeeded.
function applyRecoveryAvailability(result, job, {enabled=false, owner, bucket}={}) {
 const unset=value=>value===undefined||value===null||value==='';
 const eligible=enabled===true && typeof owner==='string' && owner.length>0 &&
   job?.meta?.owner===owner && job?.analysis_mode==='receipt_inventory_deep' &&
   ['processing','completed'].includes(job?.status) && job.meta.persist_to_kitchen===false &&
   [undefined,null,false].includes(job.deleted) && unset(job.deleted_at) &&
   unset(job.commit_status) && unset(job.review_dismissed_at);
 const manifest=eligible ? require('./sceneSourceManifest').verifiedSceneSourceManifest(job,{owner,bucket}) : null;
 return {...result,capture_outcomes:(result.capture_outcomes||[]).map(outcome=>({...outcome,
   recovery_available:Boolean(manifest && Number.isInteger(outcome.image_index) &&
     outcome.image_index>=0 && manifest.sources[outcome.image_index] &&
     outcome.item_count===0 && outcome.outcome==='wrong_capture_type')
 }))};
}
module.exports={captureOutcome,aggregateCaptureOutcomes,applyRecoveryAvailability};
