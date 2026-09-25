// Internal durable dispatch journal. Not wired into admission until recovery is ready.
const {createHash}=require('node:crypto');
const {GetCommand,UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const {makeSceneSourceManifest}=require('./sceneSourceManifest');
const active=job=>job&&[undefined,null,false].includes(job.deleted)&&[undefined,null,''].includes(job.deleted_at);
const MODES=new Set(['product_analysis','quick_identify_deep','bulk_inventory_deep','receipt_inventory_deep']);
function replayPayload(input,{bucket}) {
 if(!input || typeof input.job_id!=='string' || !/^[a-zA-Z0-9-]{1,80}$/.test(input.job_id) || !MODES.has(input.analysis_mode))throw Error('Invalid replay identity');
 const identity=value=>{if(value==null)return null;if(typeof value!=='string'||!value||value.length>256)throw Error('Invalid replay identity');return value;};
 if(!bucket || input.image || input.image_url)throw Error('Replay requires durable image references');
 function reference(value,index) {
  const stem='bulk-identify-uploads/'+input.job_id+(index===undefined?'':'-img'+index)+'.';
  if(!value || value.bucket!==bucket || typeof value.key!=='string' || !value.key.startsWith(stem) || !(value.sha256 ? /^[a-f0-9]{64}$/.test(value.sha256) && ['jpg','png','webp','gif'].some(ext=>value.key===stem+value.sha256+'.'+ext) : ['jpg','png','webp','gif'].includes(value.key.slice(stem.length))))throw Error('Invalid replay source');
  return {bucket:value.bucket,key:value.key,...(value.sha256?{sha256:value.sha256}:{})};
 }
 let source;
 if(input.receipt_s3_keys!=null) {
  if(input.analysis_mode!=='receipt_inventory_deep'||!Array.isArray(input.receipt_s3_keys)||input.receipt_s3_keys.length<1||input.receipt_s3_keys.length>30||input.s3_bucket||input.s3_key)throw Error('Invalid replay receipt pages');
  source={receipt_s3_keys:input.receipt_s3_keys.map(reference)};
 } else {
  const ref=reference({bucket:input.s3_bucket,key:input.s3_key,sha256:input.s3_sha256});source={s3_bucket:ref.bucket,s3_key:ref.key,...(ref.sha256?{s3_sha256:ref.sha256}:{})};
 }
 // Explicit fields prevent admission headers, raw images and lease tokens entering storage.
 return {deep_async_internal:true,job_id:input.job_id,analysis_mode:input.analysis_mode,owner:identity(input.owner),user_id:identity(input.user_id),device_id:identity(input.device_id),session_id:identity(input.session_id),...source};
}
function createDispatchJournal({client,tableName,bucket,now=()=>Math.floor(Date.now()/1000)}) {
 if(!client||!tableName||!bucket)throw Error('Invalid dispatch journal configuration');
 const read=async jobId=>(await client.send(new GetCommand({TableName:tableName,Key:{job_id:jobId},ConsistentRead:true}))).Item||null;
 async function prepare(input) {
  const payload=replayPayload(input,{bucket}),digest=createHash('sha256').update(JSON.stringify(payload)).digest('hex'),time=now();
  const manifest=makeSceneSourceManifest(payload,{bucket});
  const owner=payload.owner==null?'(attribute_not_exists(meta.#owner) OR meta.#owner=:owner)':'meta.#owner=:owner';
  try {
   await client.send(new UpdateCommand({TableName:tableName,Key:{job_id:payload.job_id},
    UpdateExpression:'SET dispatch_version=:version, dispatch_payload=:payload, dispatch_digest=:digest, dispatch_partition=:partition, dispatch_due_at=:due, dispatch_created_at=:now'+(manifest?', scene_source_manifest=:manifest':'')+' REMOVE #ttl',
    ConditionExpression:`attribute_exists(job_id) AND ${owner} AND analysis_mode=:mode AND #status=:pending AND attribute_not_exists(dispatch_digest) AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at) AND (attribute_not_exists(#deleted) OR #deleted=:false OR attribute_type(#deleted,:nullType)) AND (attribute_not_exists(#deletedAt) OR #deletedAt=:empty OR attribute_type(#deletedAt,:nullType))`+(manifest?' AND attribute_not_exists(scene_source_manifest)':''),
    ExpressionAttributeNames:{'#owner':'owner','#status':'status','#ttl':'ttl','#deleted':'deleted','#deletedAt':'deleted_at'},
    ExpressionAttributeValues:{':false':false,':nullType':'NULL',':empty':'',':owner':payload.owner,':mode':payload.analysis_mode,':pending':'pending',':version':1,':payload':payload,':digest':digest,':partition':'dispatch-v1',':due':time+120,':now':time,...(manifest?{':manifest':manifest}:{})}}));
   return {state:'prepared',payload,digest};
  }catch(error){
   if(error.name!=='ConditionalCheckFailedException')throw error;
   const job=await read(payload.job_id);
   if(active(job) && job.dispatch_digest===digest && job.dispatch_version===1 && (job.meta?.owner??null)===payload.owner && job.analysis_mode===payload.analysis_mode && ['pending','processing'].includes(job.status) && !job.commit_status && !job.review_dismissed_at)return {state:'existing',payload,digest};
   return {state:'conflict'};
  }
 }
 return {prepare,read};
}
module.exports={replayPayload,createDispatchJournal};
