const {createHash}=require('node:crypto');
const {authorizeCommit}=require('./bulkCommit/auth');
const {replayPayload}=require('./dispatchJournal');
const {verifiedSceneSourceManifest}=require('./sceneSourceManifest');
const reply=(statusCode,body)=>({statusCode,headers:{'Content-Type':'application/json','Access-Control-Allow-Origin':'*'},body:JSON.stringify(body)});
const active=job=>job && [undefined,null,false].includes(job.deleted) && [undefined,null,''].includes(job.deleted_at);
function recoveryId(owner,sourceId,index){
 const b=createHash('sha256').update(JSON.stringify(['capture-recovery-v1',owner,sourceId,index])).digest().subarray(0,16);
 b[6]=(b[6]&15)|80;b[8]=(b[8]&63)|128;const h=b.toString('hex');return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20)}`;
}
function createCaptureRecovery({readJob,readImage,createChild,admit,bucket,authenticate=authorizeCommit,enabled=true}){
 return async(event)=>{
  if(!enabled)return reply(503,{code:'capture_recovery_unavailable',error:'Retake this photo using Fridge / pantry.'});
  let body;try{body=JSON.parse(event.body||'{}');}catch{return reply(400,{error:'Invalid request.'});}
  const {owner,source_job_id:sourceId,image_index:index=0}=body;
  if(typeof owner!=='string'||!/^[a-zA-Z0-9_-]{1,64}$/.test(owner)||typeof sourceId!=='string'||!/^[a-f0-9-]{36}$/i.test(sourceId)||!Number.isInteger(index)||index<0||index>=30)
   return reply(400,{error:'Choose the original scan photo.'});
  const auth=authenticate(event,owner);if(auth.status)return reply(auth.status,{error:auth.code});
  if(auth.userId!==owner)return reply(403,{error:'owner_forbidden'});
  try{
   const source=await readJob(sourceId);
   if(!active(source)||source.meta?.owner!==owner)return reply(404,{error:'Scan not found.'});
   if(source.status!=='completed'||source.analysis_mode!=='receipt_inventory_deep'||source.commit_status||source.review_dismissed_at||source.meta.persist_to_kitchen===true)
    return reply(409,{error:'This scan is no longer available for recovery.'});
   const outcomes=source.result?.capture_outcomes;
   const selected=Array.isArray(outcomes)?outcomes.find(r=>r.image_index===index):null;
   if((selected && !['no_items','wrong_capture_type'].includes(selected.outcome)) || (!selected && source.result?.items?.length))
    return reply(409,{error:'Review the identified items before recovering another photo.'});
   let ref;
   if(source.scene_source_manifest){
    const manifest=verifiedSceneSourceManifest(source,{owner,bucket});
    if(!manifest)return reply(409,{error:'Original scan could not be verified.'});
    // The user chose this image explicitly; no item-to-photo attribution is inferred.
    ref=manifest.sources[index];
   }else{
    let payload;try{payload=replayPayload(source.dispatch_payload,{bucket});}catch{return reply(410,{code:'source_unavailable',error:'The original photo is unavailable. Take another photo.'});}
    if(payload.job_id!==sourceId||payload.owner!==owner||payload.analysis_mode!==source.analysis_mode)return reply(409,{error:'Original scan could not be verified.'});
    ref=payload.receipt_s3_keys?.[index] || (index===0 && !payload.receipt_s3_keys ? {bucket:payload.s3_bucket,key:payload.s3_key,sha256:payload.s3_sha256}:null);
   }
   if(!ref)return reply(400,{error:'Choose the original scan photo.'});
   const id=recoveryId(owner,sourceId,index);
   let child=await readJob(id);
   if(child && (!active(child)||child.meta?.owner!==owner||child.recovery_source_id!==sourceId||child.meta.persist_to_kitchen!==false))
    return reply(409,{error:'Recovery could not be verified.'});
   if(!child?.dispatch_digest && (!child || child.status==='pending')){
    let image;try{image=await readImage(ref);}catch(error){if(['NoSuchKey','NotFound'].includes(error.name)||error.statusCode===404)return reply(410,{code:'source_unavailable',error:'The original photo has expired. Take another photo.'});throw error;}
    if(!Buffer.isBuffer(image)||!image.length||image.length>20*1024*1024)throw Error('Invalid source image');
    if(ref.sha256 && createHash('sha256').update(image).digest('hex')!==ref.sha256)throw Error('Source digest mismatch');
    if(!child){
     const now=new Date().toISOString();
     child={job_id:id,status:'pending',analysis_mode:'bulk_inventory_deep',created_at:now,updated_at:now,
       meta:{owner,user_id:auth.userId,persist_to_kitchen:false},recovery_source_id:sourceId,recovery_image_index:index,
       review_owner:owner,review_order:now+'#'+id,stage:'pending',stage_message:'Checking your photo',progress:0};
    }
    // Retry admission must revalidate the parent too; an interrupted child is
    // not permission to ignore a dismissal that raced the retained-photo read.
    await createChild(source,child);
    try { await admit({job_id:id,analysis_mode:'bulk_inventory_deep',owner,user_id:auth.userId,image:image.toString('base64')}); }
    catch(error) {
      const admitted=await readJob(id);
      if(!admitted?.dispatch_digest || admitted.meta?.owner!==owner || admitted.recovery_source_id!==sourceId)throw error;
      child=admitted;
    }
   }
   return reply(202,{job_id:id,status:child?.status||'pending',analysis_mode:'bulk_inventory_deep',source_job_id:sourceId,review_required:true});
  }catch(error){
   console.error(JSON.stringify({event:'capture_recovery_failed',failure_type:error.name||'Error'}));
   return reply(error.name==='TransactionCanceledException'?409:503,{code:'capture_recovery_unavailable',error:'The photo could not be recovered yet. Try again.'});
  }
 };
}
module.exports={createCaptureRecovery,recoveryId};
