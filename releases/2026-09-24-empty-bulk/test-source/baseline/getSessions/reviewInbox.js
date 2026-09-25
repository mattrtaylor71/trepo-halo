// Storage operations for the signed review inbox. Legacy session APIs are unchanged.
const {DynamoDBClient}=require('@aws-sdk/client-dynamodb');
const {DynamoDBDocumentClient,QueryCommand,GetCommand,UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const modes=new Set(['receipt_inventory_deep','bulk_inventory_deep']);
const active=job=>job&&[undefined,null,false].includes(job.deleted)&&[undefined,null,''].includes(job.deleted_at);
const deletionFence=' AND (attribute_not_exists(#deleted) OR #deleted=:false OR attribute_type(#deleted,:nullType)) AND (attribute_not_exists(#deletedAt) OR #deletedAt=:empty OR attribute_type(#deletedAt,:nullType))';
const client=DynamoDBDocumentClient.from(new DynamoDBClient({}));
function reviewIndexFields(job) {
 if(!active(job))return {};
 if(typeof job?.job_id!=='string'||!job.job_id||typeof job.created_at!=='string'||!Number.isFinite(Date.parse(job.created_at)))return {};
 const owner=job?.meta?.owner;
 if(job?.meta?.persist_to_kitchen===true||!modes.has(job?.analysis_mode)||typeof owner!=='string'||!owner.trim()||owner==='anonymous')return {};
 return {review_owner:owner,review_order:job.created_at+'#'+job.job_id};
}
function visible(job,owner) {
 return active(job)&&job?.meta?.owner===owner&&job.meta.persist_to_kitchen!==true&&modes.has(job.analysis_mode)&&
 ['pending','processing','completed'].includes(job.status)&&!job.commit_status&&!job.review_dismissed_at&&
 (job.status!=='completed'||(Array.isArray(job.result?.items)&&job.result.items.length>0));
}
function createInbox(doc=client,table=process.env.JOB_TABLE_NAME) {
 return {
  // Internal migration operation. The source predicates fence a stale scan page
  // against concurrent commit/dismiss/ownership changes; never recreate a TTL-deleted row.
  async indexRetained(job) {
   const fields=reviewIndexFields(job),owner=fields.review_owner;
   if(!owner||!visible(job,owner))return 'skipped';
   if(job.review_owner===owner&&job.review_order===fields.review_order&&job.ttl===undefined)return 'unchanged';
   try {
    await doc.send(new UpdateCommand({TableName:table,Key:{job_id:job.job_id},
     UpdateExpression:'SET review_owner = :owner, review_order = :order, review_recovered = :true REMOVE #ttl',
     ConditionExpression:'attribute_exists(job_id) AND #meta.#owner = :owner AND created_at = :created AND analysis_mode = :mode AND #status IN (:pending, :processing, :completed) AND (attribute_not_exists(#meta.persist_to_kitchen) OR #meta.persist_to_kitchen = :false) AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at)'+deletionFence,
     ExpressionAttributeNames:{'#meta':'meta','#owner':'owner','#status':'status','#ttl':'ttl','#deleted':'deleted','#deletedAt':'deleted_at'},
     ExpressionAttributeValues:{':nullType':'NULL',':empty':'',':true':true,':owner':owner,':order':fields.review_order,':created':job.created_at,':mode':job.analysis_mode,':pending':'pending',':processing':'processing',':completed':'completed',':false':false}}));
    return 'indexed';
   } catch(error) {if(error.name==='ConditionalCheckFailedException')return 'conflict';throw error;}
  },
  async list(owner,{cursor,limit=20}={}) {
   if(typeof owner!=='string'||!owner.trim())throw new Error('Invalid owner');
   if(!Number.isInteger(limit)||limit<1||limit>25)throw new Error('Invalid limit');
   let key;
   if(cursor){
    if(typeof cursor!=='string'||cursor.length>2048)throw new Error('Invalid cursor');
    try {key=JSON.parse(Buffer.from(cursor,'base64url').toString());}catch{throw new Error('Invalid cursor');}
    if(!key||key.review_owner!==owner||typeof key.job_id!=='string'||typeof key.review_order!=='string'||Object.keys(key).length!==3)throw new Error('Invalid cursor');
   }
   const page=await doc.send(new QueryCommand({TableName:table,IndexName:'review-owner-index',KeyConditionExpression:'review_owner = :owner',ExpressionAttributeValues:{':owner':owner},ExclusiveStartKey:key,Limit:limit,ScanIndexForward:false}));
   // GSI membership is eventual. Re-read the source row consistently before
   // returning it so a just-committed/dismissed result cannot reappear as reviewable.
   const rows=await Promise.all((page.Items||[]).map(async entry=>(await doc.send(new GetCommand({TableName:table,Key:{job_id:entry.job_id},ConsistentRead:true}))).Item));
   return {jobs:rows.filter(job=>visible(job,owner)).map(job=>({job_id:job.job_id,status:job.status,analysis_mode:job.analysis_mode,recovered:job.review_recovered===true,created_at:job.created_at,updated_at:job.updated_at,stage:job.stage||null,progress:typeof job.progress==='number'?job.progress:null})),next_cursor:page.LastEvaluatedKey?Buffer.from(JSON.stringify(page.LastEvaluatedKey)).toString('base64url'):null};
  },
  async dismiss(owner,jobId) {
   try {
    await doc.send(new UpdateCommand({TableName:table,Key:{job_id:jobId},
     UpdateExpression:'SET review_dismissed_at = :now, #ttl = :ttl REMOVE review_owner, review_order',
     ConditionExpression:'attribute_exists(job_id) AND #meta.#owner = :owner AND (attribute_not_exists(#meta.persist_to_kitchen) OR #meta.persist_to_kitchen = :false) AND #status = :completed AND analysis_mode IN (:receipt, :bulk) AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at)'+deletionFence,
     ExpressionAttributeNames:{'#meta':'meta','#owner':'owner','#status':'status','#ttl':'ttl','#deleted':'deleted','#deletedAt':'deleted_at'},
     ExpressionAttributeValues:{':nullType':'NULL',':empty':'',':owner':owner,':false':false,':completed':'completed',':receipt':'receipt_inventory_deep',':bulk':'bulk_inventory_deep',':now':new Date().toISOString(),':ttl':Math.floor(Date.now()/1000)+7*86400}}));
    return {dismissed:true};
   } catch(error) {
    if(error.name!=='ConditionalCheckFailedException')throw error;
    const job=(await doc.send(new GetCommand({TableName:table,Key:{job_id:jobId},ConsistentRead:true}))).Item;
    if(active(job)&&job?.meta?.owner===owner&&job.review_dismissed_at)return {dismissed:true};
    return {dismissed:false,conflict:true};
   }
  }
 };
}
module.exports={createInbox,reviewIndexFields,isReviewable:visible};
