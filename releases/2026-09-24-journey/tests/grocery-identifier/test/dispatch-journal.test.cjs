const {test,before,after}=require('node:test'),assert=require('node:assert/strict'),{randomUUID}=require('node:crypto');
const {DynamoDBClient,CreateTableCommand,DeleteTableCommand}=require('@aws-sdk/client-dynamodb');
const {DynamoDBDocumentClient,PutCommand,UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const {replayPayload,createDispatchJournal}=require('../dispatchJournal');
const {sceneSourceReference}=require('../sceneSourceManifest');
const {createDispatchRecovery}=require('../dispatchRecovery');
const raw=new DynamoDBClient({endpoint:'http://127.0.0.1:38001',region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1}),client=DynamoDBDocumentClient.from(raw);
const tableName='dispatch_journal_'+randomUUID().replaceAll('-',''),bucket='fixture-bucket';let now=5000;
const journal=createDispatchJournal({client,tableName,bucket,now:()=>now});
before(()=>raw.send(new CreateTableCommand({TableName:tableName,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'job_id',KeyType:'HASH'}],AttributeDefinitions:[{AttributeName:'job_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new DeleteTableCommand({TableName:tableName}));raw.destroy();});
async function fixture(overrides={}){const job={job_id:randomUUID(),status:'pending',analysis_mode:'bulk_inventory_deep',meta:{owner:randomUUID()},ttl:5010,...overrides};await client.send(new PutCommand({TableName:tableName,Item:job}));return {job_id:job.job_id,owner:job.meta.owner,analysis_mode:job.analysis_mode,s3_bucket:bucket,s3_key:'bulk-identify-uploads/'+job.job_id+'.jpg'};}
test('prospective immutable provenance survives terminal dispatch cleanup and replay',async()=>{
 const input=await fixture();input.s3_sha256='a'.repeat(64);input.s3_key=input.s3_key.replace('.jpg','.'+input.s3_sha256+'.jpg');
 const first=await journal.prepare(input);assert.equal(first.state,'prepared');
 const initial=await journal.read(input.job_id);assert.ok(initial.scene_source_manifest);
 assert.equal((await journal.prepare(input)).state,'existing');
 assert.deepEqual((await journal.read(input.job_id)).scene_source_manifest,initial.scene_source_manifest);
 await client.send(new UpdateCommand({TableName:tableName,Key:{job_id:input.job_id},UpdateExpression:'SET #s=:s',ExpressionAttributeNames:{'#s':'status'},ExpressionAttributeValues:{':s':'completed'}}));
 const recovery=createDispatchRecovery({client,tableName,bucket,invoke:async()=>{throw Error('Unexpected dispatch')}});
 assert.equal(await recovery.recover(input.job_id),'terminal');
 const terminal=await journal.read(input.job_id);assert.equal(terminal.dispatch_payload,undefined);
 assert.deepEqual(sceneSourceReference(terminal,{owner:input.owner,bucket}),{bucket,key:input.s3_key,sha256:input.s3_sha256});
});
test('existing legacy dispatch is never backfilled with guessed source manifest',async()=>{
 const input=await fixture();await journal.prepare(input);assert.equal((await journal.read(input.job_id)).scene_source_manifest,undefined);
 assert.equal((await journal.prepare(input)).state,'existing');assert.equal((await journal.read(input.job_id)).scene_source_manifest,undefined);
});
test('concurrent admission journals one immutable replay without losing legacy status or metadata',async()=>{
 const input=await fixture();const answers=await Promise.all(Array.from({length:20},()=>journal.prepare(input)));
 assert.equal(answers.filter(x=>x.state==='prepared').length,1);assert.equal(answers.filter(x=>x.state==='existing').length,19);
 const job=await journal.read(input.job_id);assert.equal(job.status,'pending');assert.equal(job.meta.owner,input.owner);assert.equal(job.ttl,undefined);assert.equal(job.dispatch_due_at,5120);assert.deepEqual(job.dispatch_payload,replayPayload(input,{bucket}));
 now+=7*3600;assert.equal((await journal.prepare(input)).state,'existing');assert.deepEqual((await journal.read(input.job_id)).dispatch_payload,job.dispatch_payload);
});
test('deleted, completed, failed, committed, dismissed and wrong-owner records cannot be journaled',async()=>{
 for(const override of [{status:'completed'},{status:'failed'},{commit_status:'committed'},{review_dismissed_at:'now'}]){const input=await fixture(override),before=await journal.read(input.job_id);assert.equal((await journal.prepare(input)).state,'conflict');assert.deepEqual(await journal.read(input.job_id),before);}
 const input=await fixture();assert.equal((await journal.prepare({...input,owner:randomUUID()})).state,'conflict');assert.equal((await journal.prepare({...input,analysis_mode:'product_analysis'})).state,'conflict');
 const missing={...input,job_id:randomUUID()};missing.s3_key='bulk-identify-uploads/'+missing.job_id+'.jpg';assert.equal((await journal.prepare(missing)).state,'conflict');assert.equal(await journal.read(missing.job_id),null);
});
test('existing journal cannot be changed to another source or revive terminal work',async()=>{
 const input=await fixture();await journal.prepare(input);assert.equal((await journal.prepare({...input,s3_key:input.s3_key.replace('.jpg','.png')})).state,'conflict');
 await client.send(new UpdateCommand({TableName:tableName,Key:{job_id:input.job_id},UpdateExpression:'SET #s=:s',ExpressionAttributeNames:{'#s':'status'},ExpressionAttributeValues:{':s':'completed'}}));assert.equal((await journal.prepare(input)).state,'conflict');
});
test('receipt pages retain ordered durable references and private request fields are excluded',async()=>{
 const input=await fixture({analysis_mode:'receipt_inventory_deep'});delete input.s3_bucket;delete input.s3_key;input.receipt_s3_keys=[0,1,2].map(i=>({bucket,key:'bulk-identify-uploads/'+input.job_id+'-img'+i+'.jpg'}));
 const result=await journal.prepare({...input,Authorization:'secret',processing_handoff_token:'old-token'});assert.equal(result.state,'prepared');assert.deepEqual(result.payload.receipt_s3_keys,input.receipt_s3_keys);assert.equal(JSON.stringify(result).includes('secret'),false);assert.equal(JSON.stringify(result).includes('old-token'),false);
 assert.throws(()=>replayPayload({...input,receipt_s3_keys:[]},{bucket}),/receipt pages/);
});
test('raw images, expiring URLs, wrong buckets and foreign job references fail before writing',async()=>{
 const input=await fixture();
 for(const change of [{image:'base64'},{image_url:'https://fixture.invalid/temporary'},{s3_bucket:'other'},{s3_key:'bulk-identify-uploads/other.jpg'}])await assert.rejects(journal.prepare({...input,...change}),/Replay requires|Invalid replay source/);
 assert.equal((await journal.read(input.job_id)).dispatch_payload,undefined);
});
test('storage outage is not reported as durable acceptance; legacy anonymous owner remains supported',async()=>{
 const input=await fixture({meta:{owner:null}});assert.equal((await journal.prepare(input)).state,'prepared');
 const broken=createDispatchJournal({client:{send:async()=>{throw Error('Storage unavailable');}},tableName,bucket});await assert.rejects(broken.prepare(input),/Storage unavailable/);
});

for(const marker of [{deleted:true},{deleted:'yes'},{deleted_at:'now'},{deleted_at:false}])
 for(const replay of [false,true])test('deletion marker prevents journal mutation/replay '+JSON.stringify(marker)+' replay '+replay,async()=>{
  const input=await fixture();
  if(replay)await journal.prepare(input);
  const before={...await journal.read(input.job_id),...marker,ttl:6000};
  await client.send(new PutCommand({TableName:tableName,Item:before}));
  assert.equal((await journal.prepare(input)).state,'conflict');
  assert.deepEqual(await journal.read(input.job_id),before);
 });
for(const marker of [{deleted:false,deleted_at:''},{deleted:null,deleted_at:null}])test('legacy inactive deletion markers remain compatible '+JSON.stringify(marker),async()=>{
 const input=await fixture(marker);assert.equal((await journal.prepare(input)).state,'prepared');
 assert.equal((await journal.prepare(input)).state,'existing');
});
