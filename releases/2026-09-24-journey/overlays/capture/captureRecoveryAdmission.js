// Recovery uses the already-serving async worker and its processing lease.
// A saved payload is not acknowledgement: only Lambda 202 (or observed worker
// progress) permits the recovery route to tell the client processing started.
const {createHash,randomUUID}=require('node:crypto');
const {GetCommand,UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const {PutObjectCommand}=require('@aws-sdk/client-s3');
const {InvokeCommand}=require('@aws-sdk/client-lambda');
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function createRecoveryAdmission({db,s3,lambda,TableName,bucket,functionName,now=Date.now,wait=pause}) {
 const send=(client,command)=>client.send(command,{abortSignal:AbortSignal.timeout(5000)});
 const read=async id=>(await send(db,new GetCommand({TableName,Key:{job_id:id},ConsistentRead:true}))).Item;
 const valid=(job,input)=>job && job.meta?.owner===input.owner && job.meta.persist_to_kitchen===false &&
  job.analysis_mode==='bulk_inventory_deep' && job.recovery_source_id && !job.deleted && !job.deleted_at && !job.review_dismissed_at && !job.commit_status;
 return async input=>{
  if(!functionName||!bucket)throw Error('Recovery worker unavailable');
  const image=Buffer.from(input.image||'','base64');
  if(!image.length||image.length>20*1024*1024)throw Error('Invalid recovery image');
  const sha=createHash('sha256').update(image).digest('hex'),token=randomUUID();
  const key=`bulk-identify-uploads/${input.job_id}.${sha}.jpg`;
  let claimed=false;
  for(let attempt=0;attempt<24;attempt++){
   const job=await read(input.job_id);if(!valid(job,input))throw Error('Recovery identity changed');
   if(job.dispatch_digest||['processing','completed','failed'].includes(job.status))return;
   try {
    await send(db,new UpdateCommand({TableName,Key:{job_id:input.job_id},
     UpdateExpression:'SET recovery_dispatch_token=:token, recovery_dispatch_until=:until',
     ConditionExpression:'#status=:pending AND meta.#owner=:owner AND meta.persist_to_kitchen=:false AND attribute_exists(recovery_source_id) AND attribute_not_exists(dispatch_digest) AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at) AND attribute_not_exists(deleted_at) AND (attribute_not_exists(deleted) OR deleted=:false) AND (attribute_not_exists(recovery_dispatch_until) OR recovery_dispatch_until < :now)',
     ExpressionAttributeNames:{'#status':'status','#owner':'owner'},
     ExpressionAttributeValues:{':token':token,':until':now()+20000,':now':now(),':pending':'pending',':owner':input.owner,':false':false}}));
    claimed=true;break;
   }catch(error){if(error.name!=='ConditionalCheckFailedException')throw error;await wait(200);}
  }
  if(!claimed)throw Error('Recovery dispatch is still being confirmed');
  try {
   await send(s3,new PutObjectCommand({Bucket:bucket,Key:key,Body:image,ContentType:'application/octet-stream',ChecksumSHA256:Buffer.from(sha,'hex').toString('base64')}));
   const payload={deep_async_internal:true,job_id:input.job_id,analysis_mode:'bulk_inventory_deep',owner:input.owner,user_id:input.user_id,s3_bucket:bucket,s3_key:key};
   const response=await send(lambda,new InvokeCommand({FunctionName:functionName,InvocationType:'Event',Payload:Buffer.from(JSON.stringify(payload))}));
   if(response.StatusCode!==202)throw Error('Recovery worker did not accept the photo');
   await send(db,new UpdateCommand({TableName,Key:{job_id:input.job_id},
    UpdateExpression:'SET dispatch_digest=:digest REMOVE recovery_dispatch_token, recovery_dispatch_until',
    ConditionExpression:'recovery_dispatch_token=:token',
    ExpressionAttributeValues:{':token':token,':digest':createHash('sha256').update(JSON.stringify(payload)).digest('hex')}}));
  }catch(error){
   const job=await read(input.job_id).catch(()=>null);
   if(valid(job,input)&&(job.dispatch_digest||['processing','completed','failed'].includes(job.status)))return;
   // Unknown acknowledgement remains retryable under this same child identity.
   await send(db,new UpdateCommand({TableName,Key:{job_id:input.job_id},
    UpdateExpression:'REMOVE recovery_dispatch_token, recovery_dispatch_until',
    ConditionExpression:'recovery_dispatch_token=:token',ExpressionAttributeValues:{':token':token}})).catch(()=>{});
   throw error;
  }
 };
}
module.exports={createRecoveryAdmission};
