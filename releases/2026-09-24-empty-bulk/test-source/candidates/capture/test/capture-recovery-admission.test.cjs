const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const {randomUUID}=require('node:crypto');
const {Readable}=require('node:stream');
const {DynamoDBClient,CreateTableCommand,DeleteTableCommand}=require('@aws-sdk/client-dynamodb');
const {DynamoDBDocumentClient,PutCommand,GetCommand,UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const {createSourceRetention}=require('../captureSourceRetention');
const {createRecoveryAdmission}=require('../captureRecoveryAdmission');
const {createRuntime}=require('../captureRecoveryRuntime');
const raw=new DynamoDBClient({endpoint:'http://127.0.0.1:38001',region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});
const db=DynamoDBDocumentClient.from(raw),TableName='capture_admission_'+randomUUID().replaceAll('-','');
const owner='fixture-owner',bucket='fixture-bucket',image=Buffer.from('original product photo');
before(()=>raw.send(new CreateTableCommand({TableName,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'job_id',KeyType:'HASH'}],AttributeDefinitions:[{AttributeName:'job_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new DeleteTableCommand({TableName}));raw.destroy();});
const read=async id=>(await db.send(new GetCommand({TableName,Key:{job_id:id},ConsistentRead:true}))).Item;
async function fixture(invoke){
 const job_id=randomUUID(),objects=new Map(),invocations=[];
 const s3={send:async command=>{
  const p=command.input;assert.equal(p.Bucket,bucket);
  if(command.constructor.name==='PutObjectCommand'){objects.set(p.Key,Buffer.from(p.Body));return {};}
  if(!objects.has(p.Key))throw Object.assign(Error('missing'),{name:'NoSuchKey'});
  return {Body:Readable.from([objects.get(p.Key)])};
 }};
 await db.send(new PutCommand({TableName,Item:{job_id,meta:{owner,persist_to_kitchen:false},status:'pending',analysis_mode:'receipt_inventory_deep'}}));
 const retain=createSourceRetention({db,s3,TableName,bucket});
 await retain({job_id,owner,images:[image.toString('base64')]});
 await db.send(new UpdateCommand({TableName,Key:{job_id},UpdateExpression:'SET #s=:s, #r=:r',ExpressionAttributeNames:{'#s':'status','#r':'result'},ExpressionAttributeValues:{':s':'completed',':r':{items:[],capture_outcomes:[{image_index:0,outcome:'wrong_capture_type'}]}}}));
 const lambda={send:async command=>{
  const p=JSON.parse(Buffer.from(command.input.Payload));invocations.push(p);
  assert.equal(p.analysis_mode,'bulk_inventory_deep');assert.equal(p.owner,owner);
  assert.deepEqual(objects.get(p.s3_key),image);assert.equal((await read(p.job_id)).meta.persist_to_kitchen,false);
  return invoke?invoke(p):{StatusCode:202};
 }};
 const admit=createRecoveryAdmission({db,s3,lambda,TableName,bucket,functionName:'fixture-worker'});
 const run=createRuntime({db,s3,TableName,bucket,enabled:true,admit,authenticate:()=>({userId:owner})});
 return {job_id,invocations,objects,retain,admit,call:()=>run({body:JSON.stringify({owner,source_job_id:job_id})})};
}
test('serving-compatible retained-photo recovery: twenty taps, one accepted worker, same review ID after restart',async()=>{
 const f=await fixture();const results=await Promise.all(Array.from({length:20},()=>f.call()));
 assert.ok(results.every(x=>x.statusCode===202),JSON.stringify(results));
 const id=JSON.parse(results[0].body).job_id;assert.equal(new Set(results.map(x=>JSON.parse(x.body).job_id)).size,1);
 assert.equal(f.invocations.length,1);assert.ok((await read(id)).dispatch_digest);
 assert.equal((await f.call()).statusCode,202);assert.equal(f.invocations.length,1);
 assert.equal((await read(f.job_id)).status,'completed');
});
test('failed acknowledgement is not reported as accepted and retry preserves child identity',async()=>{
 let n=0;const f=await fixture(()=>++n===1?{StatusCode:500}:{StatusCode:202});
 assert.equal((await f.call()).statusCode,503);
 const childId=f.invocations[0].job_id;assert.equal((await read(childId)).dispatch_digest,undefined);
 assert.equal((await f.call()).statusCode,202);assert.equal(f.invocations[1].job_id,childId);
 assert.ok((await read(childId)).dispatch_digest);
});
test('lost acknowledgement after the worker starts is verified from authoritative status',async()=>{
 const f=await fixture(async p=>{
  await db.send(new UpdateCommand({TableName,Key:{job_id:p.job_id},UpdateExpression:'SET #s=:s',ExpressionAttributeNames:{'#s':'status'},ExpressionAttributeValues:{':s':'processing'}}));
  throw Error('acknowledgement lost');
 });
 assert.equal((await f.call()).statusCode,202);assert.equal((await f.call()).statusCode,202);assert.equal(f.invocations.length,1);
});
test('lost acknowledgement before worker start stays retryable',async()=>{
 let n=0;const f=await fixture(()=>{if(!n++)throw Error('offline');return {StatusCode:202};});
 assert.equal((await f.call()).statusCode,503);assert.equal((await f.call()).statusCode,202);
 assert.equal(f.invocations[0].job_id,f.invocations[1].job_id);
});
test('source failure never creates a verified manifest or accepted worker',async()=>{
 const id=randomUUID();await db.send(new PutCommand({TableName,Item:{job_id:id,status:'pending',meta:{owner},analysis_mode:'receipt_inventory_deep'}}));
 const retain=createSourceRetention({db,s3:{send:async()=>{throw Error('storage offline');}},TableName,bucket});
 await assert.rejects(retain({job_id:id,owner,images:[image.toString('base64')]}),/offline/);
 assert.equal((await read(id)).scene_source_manifest,undefined);
});
test('invalid identity cannot write photos; terminated source cannot gain a manifest',async()=>{
 let writes=0;const retain=createSourceRetention({db,s3:{send:async()=>{writes++;}},TableName,bucket});
 await assert.rejects(retain({job_id:'../invalid',owner,images:[image.toString('base64')]}));assert.equal(writes,0);
 const id=randomUUID();await db.send(new PutCommand({TableName,Item:{job_id:id,status:'completed',meta:{owner},analysis_mode:'receipt_inventory_deep'}}));
 await assert.rejects(retain({job_id:id,owner,images:[image.toString('base64')]}),e=>e.name==='ConditionalCheckFailedException');
 assert.equal((await read(id)).scene_source_manifest,undefined);
});
