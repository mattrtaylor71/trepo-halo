const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const {randomUUID,createHash}=require('node:crypto');
const {Readable}=require('node:stream');
const {DynamoDBClient,CreateTableCommand,DeleteTableCommand}=require('@aws-sdk/client-dynamodb');
const {DynamoDBDocumentClient,PutCommand,GetCommand,UpdateCommand,ScanCommand}=require('@aws-sdk/lib-dynamodb');
const {createRuntime}=require('../captureRecoveryRuntime');
const {makeSceneSourceManifest}=require('../sceneSourceManifest');
const raw=new DynamoDBClient({endpoint:'http://127.0.0.1:38001',region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});
const db=DynamoDBDocumentClient.from(raw),TableName='capture_recovery_'+randomUUID().replaceAll('-','');
const owner='fixture-owner',bucket='fixture-bucket',image=Buffer.from('fixture photo'),sha=createHash('sha256').update(image).digest('hex');
before(()=>raw.send(new CreateTableCommand({TableName,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'job_id',KeyType:'HASH'}],AttributeDefinitions:[{AttributeName:'job_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new DeleteTableCommand({TableName}));raw.destroy();});
async function fixture(){
 const id=randomUUID(),payload={job_id:id,owner,analysis_mode:'receipt_inventory_deep',s3_bucket:bucket,s3_key:`bulk-identify-uploads/${id}.${sha}.jpg`,s3_sha256:sha};
 const job={job_id:id,meta:{owner,persist_to_kitchen:false},analysis_mode:payload.analysis_mode,status:'completed',result:{items:[]},scene_source_manifest:makeSceneSourceManifest(payload,{bucket})};
 await db.send(new PutCommand({TableName,Item:job}));return job;
}
const read=async id=>(await db.send(new GetCommand({TableName,Key:{job_id:id},ConsistentRead:true}))).Item;
function runner(extra={}){
 const admitted=new Set();
 const run=createRuntime({db,TableName,bucket,enabled:true,authenticate:()=>({userId:owner}),
  s3:{send:async()=>({ContentLength:image.length,Body:Readable.from([image])})},
  admit:async input=>{
   const child=await read(input.job_id);assert.equal(child.meta.persist_to_kitchen,false);
   admitted.add(input.job_id);
   await db.send(new UpdateCommand({TableName,Key:{job_id:input.job_id},UpdateExpression:'SET dispatch_digest=:d',ExpressionAttributeValues:{':d':'accepted'}}));
  },...extra});
 return {admitted,call:async job=>run({body:JSON.stringify({owner,source_job_id:job.job_id})})};
}
test('actual Dynamo transaction admits one manifest-backed review identity across twenty simultaneous taps',async()=>{
 const job=await fixture(),r=runner();const replies=await Promise.all(Array.from({length:20},()=>r.call(job)));
 assert.ok(replies.every(x=>x.statusCode===202),replies.map(x=>x.body).join('\n'));
 assert.equal(new Set(replies.map(x=>JSON.parse(x.body).job_id)).size,1);assert.equal(r.admitted.size,1);
 const id=JSON.parse(replies[0].body).job_id;const child=await read(id);
 assert.equal(child.recovery_source_id,job.job_id);assert.equal(child.meta.persist_to_kitchen,false);
 assert.equal((await runner().call(job)).statusCode,202);
});
test('source dismissal between read and child admission fails the real conditional transaction',async()=>{
 const job=await fixture();
 const r=runner({s3:{send:async()=>{
  await db.send(new UpdateCommand({TableName,Key:{job_id:job.job_id},UpdateExpression:'SET review_dismissed_at=:t',ExpressionAttributeValues:{':t':'fixture'}}));
  return {Body:Readable.from([image])};
 }}});
 assert.equal((await r.call(job)).statusCode,409);assert.equal(r.admitted.size,0);
 const all=(await db.send(new ScanCommand({TableName}))).Items;
 assert.equal(all.filter(x=>x.recovery_source_id===job.job_id).length,0);
});
test('source mutation race rejects child even when another child already exists',async()=>{
 const job=await fixture(),first=runner({admit:async()=>{throw Error('fixture interrupted');}});
 assert.equal((await first.call(job)).statusCode,503);
 const resumed=runner({s3:{send:async()=>{
  await db.send(new UpdateCommand({TableName,Key:{job_id:job.job_id},UpdateExpression:'SET review_dismissed_at=:t',ExpressionAttributeValues:{':t':'fixture'}}));
  return {Body:Readable.from([image])};
 }}});
 assert.equal((await resumed.call(job)).statusCode,409);assert.equal(resumed.admitted.size,0);
});
test('incorrect source bytes never create a recoverable review job',async()=>{
 const job=await fixture(),r=runner({s3:{send:async()=>({Body:Readable.from([Buffer.from('wrong')])})}});
 assert.equal((await r.call(job)).statusCode,503);assert.equal(r.admitted.size,0);
});
