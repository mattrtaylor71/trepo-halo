const {test,before,after}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {randomUUID}=require('node:crypto');
const dynamo=require('@aws-sdk/client-dynamodb');
const document=require('@aws-sdk/lib-dynamodb');
const table='async_replay_'+randomUUID().replaceAll('-',''),clients=[];
class LocalClient extends dynamo.DynamoDBClient { constructor(){super({endpoint:'http://127.0.0.1:38001',region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});clients.push(this);} }
const raw=new LocalClient();
before(async()=>raw.send(new dynamo.CreateTableCommand({TableName:table,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'job_id',KeyType:'HASH'}],AttributeDefinitions:[{AttributeName:'job_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new dynamo.DeleteTableCommand({TableName:table}));clients.forEach(c=>c.destroy());});
const queueModule={exports:{}};
vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,'../dist/utils/jobQueue.js'),'utf8'),{
 module:queueModule,exports:queueModule.exports,Error,Date,console:{warn:()=>{}},process:{env:{JOB_TABLE_NAME:table,AWS_LAMBDA_FUNCTION_NAME:'fixture'}},
 require:name=>{
  if(name==='@aws-sdk/client-dynamodb')return {...dynamo,DynamoDBClient:LocalClient};
  if(name==='@aws-sdk/lib-dynamodb')return document;
  if(name==='../../reviewInbox')return {reviewIndexFields:require('../reviewInbox').reviewIndexFields};
  throw Error('Unexpected queue dependency '+name);
 }
},{filename:'dist/utils/jobQueue.js'});
const queue=queueModule.exports;
const processing=require('../processingLeaseRunner').createProcessingLeaseRunner({leases:require('../processingLeases').createProcessingLeases({client:document.DynamoDBDocumentClient.from(raw),tableName:table})});
function handler(jobId,owner,options={}){
 let calls=0,notifications=0,sessionFailures=0;
 const analyze=async()=>{calls++;if(options.analyze)return options.analyze();return {items:[{item_name:'Generated fixture',count:2}]};};
 class Command{constructor(input){this.input=input;}}
 class ExternalClient{async send(command){if(options.invoke && command.input.FunctionName)return options.invoke(command.input);if(options.readImage && command.input.Key)return {Body:{transformToByteArray:async()=>Buffer.from('retained receipt')}};throw Error('Unexpected external request');}}
 const module={exports:{}};
 vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,'../identifyAsync/app.js'),'utf8'),{
 module,exports:module.exports,Buffer,Date,JSON,Number,Map,Set,Promise,Error,
 process:{env:{NOTIFICATIONS_API_BASE_URL:'https://fixture.invalid',COMPLETED_TRANSPORT_RETENTION_ENABLED:'false',PROCESSING_LEASES_ENABLED:options.leasesEnabled===false?'false':'true',...(options.invoke?{AWS_LAMBDA_FUNCTION_NAME:'fixture',IDENTIFY_MAX_REDRIVE:'1'}:{})}},
 console:{log:()=>{},warn:()=>{},error:()=>{}},fetch:async()=>{notifications++;return {ok:true,status:200};},
 require:name=>{
  if(name==='crypto')return require('node:crypto');
  if(name==='@aws-sdk/client-lambda')return {LambdaClient:ExternalClient,InvokeCommand:Command};
  if(name==='@aws-sdk/client-s3')return {S3Client:ExternalClient,PutObjectCommand:Command,GetObjectCommand:Command};
  if(name==='../processingLeaseRuntime')return options.processingOverride || processing;
  if(name==='../sessionStore')return {markJobFailed:async()=>{sessionFailures++;return {};},isSessionComplete:()=>false};
  if(name==='../dist/utils/jobQueue')return queue;
  if(name==='../dist/services/analyzeProduct')return {analyzeProduct:analyze};
  if(name==='../quickIdentify/service')return {identifyItemBulkDeep:analyze,identifyItemDeep:analyze,identifyItemReceiptDeep:analyze};
  if(name==='../bulkKitchenWriter')return {persistBulkKitchenResults:()=>{throw Error('Review must not persist');}};
  if(name==='../bulkKitchenSimilarity')return {annotateBulkItemsWithKitchenSimilarity:async({items})=>({items,debug:{}})};
  if(name==='../emptyBulkOutcome')return require('../emptyBulkOutcome');
  if(name==='../captureOutcome')return require('../captureOutcome');
  throw Error('Unexpected handler dependency '+name);
 }
 },{filename:'identifyAsync/app.js'});
 return {get sessionFailures(){return sessionFailures;},get calls(){return calls;},get notifications(){return notifications;},runEvent:event=>module.exports.handler(event),run:()=>module.exports.handler({deep_async_internal:true,job_id:jobId,image_url:'https://fixture.invalid/photo.png',analysis_mode:'bulk_inventory_deep',owner,session_id:options.sessionId})};
}
async function fixture(status='pending'){
 const id=randomUUID(),owner=randomUUID();
 await queue.createJob(id,{status,analysis_mode:'bulk_inventory_deep',meta:{owner,persist_to_kitchen:false},...(status==='completed'?{result:{items:[{item_name:'Original saved fixture',count:1}]}}:{})});
 return {id,owner,h:handler(id,owner)};
}
test('completed replay preserves the old polling payload without another provider call or notification',async()=>{
 const {id,h}=await fixture('completed');const before=await queue.getJob(id);const result=await h.run();
 assert.equal(result.statusCode,202);assert.deepEqual(JSON.parse(result.body),{status:'processing'});assert.equal((await queue.getJob(id)).result.items[0].item_name,before.result.items[0].item_name);
 assert.equal(h.calls,0,'A rejected completed-job write must not re-drive analysis');
 assert.equal(h.notifications,0,'Completed replays must not send another ready notification');
});
test('only the durable completion winner emits the ready notification',async()=>{
 const {id,h}=await fixture();await Promise.all([h.run(),h.run()]);
 assert.equal((await queue.getJob(id)).status,'completed');assert.equal(h.notifications,1);
});
test('atomic update outcomes distinguish the winner, terminal replay and missing job',async()=>{
 const {id}=await fixture();
 assert.equal(await queue.updateJobStatus(id,'processing'),true);
 assert.equal(await queue.updateJobStatus(id,'completed',{items:[{item_name:'Saved'}]}),true);
 assert.equal(await queue.updateJobStatus(id,'processing'),false);
 assert.equal(await queue.updateJobStatus(randomUUID(),'processing'),false);
});

test('failed and removed legacy jobs do not produce work or replacement records',async()=>{
 const failed=await fixture('failed');
 const missing={id:randomUUID(),owner:randomUUID()};missing.h=handler(missing.id,missing.owner);
 for(const f of [failed,missing]){const result=await f.h.run();assert.equal(result.statusCode,202);assert.equal(f.h.calls,0);assert.equal(f.h.notifications,0);}
 assert.equal((await queue.getJob(failed.id)).status,'failed');assert.equal(await queue.getJob(missing.id),null);
});

test('a losing failed attempt cannot mark a successfully completed session job failed',async()=>{
 const {id,owner}=await fixture();
 const h=handler(id,owner,{sessionId:randomUUID(),analyze:async()=>{
  assert.equal(await queue.updateJobStatus(id,'completed',{items:[{item_name:'Winning result'}]}),true);
  throw Error('Losing provider attempt');
 }});
 assert.equal((await h.run()).statusCode,202);
 assert.equal((await queue.getJob(id)).status,'completed');
 assert.equal((await queue.getJob(id)).result.items[0].item_name,'Winning result');
 assert.equal(h.sessionFailures,0,'Rejected failure transition must not damage the successful session');
});

test('the genuine failure winner still records the error and session failure',async()=>{
 const {id,owner}=await fixture();
 const h=handler(id,owner,{sessionId:randomUUID(),analyze:async()=>{throw Error('Provider unavailable');}});
 assert.equal((await h.run()).statusCode,202);
 const job=await queue.getJob(id);
 assert.equal(job.status,'failed');assert.equal(job.error,'Provider unavailable');
 assert.equal(h.sessionFailures,1);assert.equal(h.notifications,0);
});

test('overlapping active deliveries perform provider work once',{timeout:5000},async()=>{
 const {id,owner}=await fixture();let release,entered,enteredAgain,invocations=0;
 const secondStarted=new Promise(resolve=>enteredAgain=resolve);
 const started=new Promise(resolve=>entered=resolve),hold=new Promise(resolve=>release=resolve);
 const h=handler(id,owner,{analyze:async()=>{if(invocations++===0)entered();else enteredAgain();await hold;return {items:[{item_name:'Single analysis'}]};}});
 const first=h.run();await started;
 const second=h.run();
 // Observe either duplicate provider entry or a coalesced acknowledgement, without timing sleeps.
 await Promise.race([second,secondStarted]);release();await Promise.all([first,second]);
 assert.equal(h.calls,1,'Concurrent deliveries must share the in-flight job');
 assert.equal((await queue.getJob(id)).status,'completed');assert.equal(h.notifications,1);
});

test('retry dispatch can hand off before its acknowledgement without dropping work',{timeout:5000},async()=>{
 const {id,owner}=await fixture();let attempts=0,dispatches=0,h;
 h=handler(id,owner,{analyze:async()=>{if(attempts++===0)throw Error('Transient provider error');return {items:[{item_name:'Retry result'}]};},invoke:async input=>{
  dispatches++;const event=JSON.parse(input.Payload.toString());assert.equal(event.retry_count,1);assert.ok(event.processing_handoff_token);
  assert.equal((await h.runEvent(event)).statusCode,202);return {StatusCode:202};
 }});
 assert.equal((await h.run()).statusCode,202);const job=await queue.getJob(id);
 assert.equal(job.status,'completed');assert.equal(job.result.items[0].item_name,'Retry result');assert.equal(job.processing_token,undefined);
 assert.equal(attempts,2);assert.equal(dispatches,1);assert.equal(h.notifications,1);
});

test('lost retry acknowledgement cannot overwrite the new worker successful result',{timeout:5000},async()=>{
 const {id,owner}=await fixture();let attempts=0,h;
 h=handler(id,owner,{analyze:async()=>{if(attempts++===0)throw Error('Transient provider error');return {items:[{item_name:'Successful retry'}]};},invoke:async input=>{
  await h.runEvent(JSON.parse(input.Payload.toString()));throw Error('Acknowledgement lost');
 }});
 assert.equal((await h.run()).statusCode,202);assert.equal((await queue.getJob(id)).status,'completed');assert.equal(h.notifications,1);assert.equal(h.sessionFailures,0);
});

test('disabled rollout flag preserves the installed-client processing path',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{leasesEnabled:false});
 const result=await h.run();assert.equal(result.statusCode,202);assert.deepEqual(JSON.parse(result.body),{status:'processing'});
 assert.equal(h.calls,1);assert.equal((await queue.getJob(id)).status,'completed');assert.equal((await queue.getJob(id)).processing_attempts,undefined);
});

test('internal coordination outages remain Lambda errors so accepted work can retry',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{processingOverride:{run:async()=>{const error=Error('Coordination unavailable');error.retryableProcessing=true;throw error;}}});
 await assert.rejects(h.run(),error=>error.retryableProcessing===true);
 assert.equal(h.calls,0);assert.equal((await queue.getJob(id)).status,'pending');
 const publicError=await h.runEvent({body:'{}',headers:{}});assert.equal(publicError.statusCode,400);
});

for(const photoCount of [1,2])test('retained receipt '+photoCount+' photo(s) retries the serving worker without a new review identity',async()=>{
 const {id,owner}=await fixture();let attempts=0,dispatches=0,h;
 h=handler(id,owner,{readImage:true,analyze:async()=>{if(attempts++===0)throw Error('Transient provider error');return {items:[{item_name:'Recovered receipt item'}],capture_type:'receipt'};},invoke:async input=>{
  dispatches++;const event=JSON.parse(input.Payload.toString());assert.equal(event.job_id,id);assert.equal(event.retry_count,1);assert.equal(event.receipt_s3_keys.length,photoCount);assert.ok(event.processing_handoff_token);
  await h.runEvent(event);return {StatusCode:202};
 }});
 await h.runEvent({deep_async_internal:true,job_id:id,owner,analysis_mode:'receipt_inventory_deep',receipt_s3_keys:Array.from({length:photoCount},(_,i)=>({bucket:'fixture-bucket',key:'fixture-'+i}))});
 const job=await queue.getJob(id);assert.equal(job.status,'completed');assert.equal(dispatches,1);assert.equal(job.result.items[0].item_name,'Recovered receipt item');assert.equal(h.notifications,1);
});
