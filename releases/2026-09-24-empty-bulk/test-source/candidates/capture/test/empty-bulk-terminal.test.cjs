const {test,before,after}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {randomUUID}=require('node:crypto');
const dynamo=require('@aws-sdk/client-dynamodb');
const document=require('@aws-sdk/lib-dynamodb');
const table='async_replay_'+randomUUID().replaceAll('-',''),clients=[];
class LocalClient extends dynamo.DynamoDBClient { constructor(){super({endpoint:'http://127.0.0.1:38001',region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});clients.push(this);} }
const raw=new LocalClient();
const sessionsTable=table+'_sessions';
before(async()=>raw.send(new dynamo.CreateTableCommand({TableName:sessionsTable,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'session_id',KeyType:'HASH'}],AttributeDefinitions:[{AttributeName:'session_id',AttributeType:'S'}]})));
after(async()=>raw.send(new dynamo.DeleteTableCommand({TableName:sessionsTable})));
const sessionModule={exports:{}};
vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,'../sessionStore.js'),'utf8'),{module:sessionModule,exports:sessionModule.exports,Date,Set,console,process:{env:{SESSION_TABLE_NAME:sessionsTable}},require:name=>{if(name==='@aws-sdk/client-dynamodb')return {...dynamo,DynamoDBClient:LocalClient};if(name==='@aws-sdk/lib-dynamodb')return document;throw Error(name);}});
const realSessionStore=sessionModule.exports;
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
 let calls=0,notifications=0,sessionFailures=0,sessionCompletions=0,sessionSettled=0;
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
  if(name==='../sessionStore')return options.sessionStore || {markJobFailed:async()=>{sessionFailures++;return options.session || {};},markJobCompleted:async()=>{sessionCompletions++;return options.session || {};},isSessionComplete:()=>!!options.session,tryMarkSessionNotified:async()=>{sessionSettled++;return true;}};
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
 return {get sessionSettled(){return sessionSettled;},get sessionCompletions(){return sessionCompletions;},get sessionFailures(){return sessionFailures;},get calls(){return calls;},get notifications(){return notifications;},runEvent:event=>module.exports.handler(event),run:()=>module.exports.handler({deep_async_internal:true,job_id:jobId,image_url:'https://fixture.invalid/photo.png',analysis_mode:options.mode || 'bulk_inventory_deep',owner,session_id:options.sessionId})};
}
async function fixture(status='pending'){
 const id=randomUUID(),owner=randomUUID();
 await queue.createJob(id,{status,analysis_mode:'bulk_inventory_deep',meta:{owner,persist_to_kitchen:false},...(status==='completed'?{result:{items:[{item_name:'Original saved fixture',count:1}]}}:{})});
 return {id,owner,h:handler(id,owner)};
}

test('empty bulk is a terminal failed job with no ready effect and no automatic redrive',async()=>{
 const {id,owner}=await fixture();let dispatches=0;
 const h=handler(id,owner,{analyze:async()=>({items:[],visual_analysis_log:'No groceries'}),invoke:async()=>{dispatches++;throw Error('Must not redrive valid empty analysis');}});
 await h.run();const job=await queue.getJob(id);
 assert.equal(job.status,'failed');assert.equal(job.result.error_code,'no_items_identified');assert.deepEqual(job.result.items,[]);
 assert.equal(job.stage,'failed');assert.match(job.error,/No grocery items/);assert.equal(job.progress,100);
 assert.equal(h.calls,1);assert.equal(h.notifications,0);assert.equal(dispatches,0);
 await h.run();assert.equal(h.calls,1);assert.equal(h.notifications,0);assert.deepEqual(await queue.getJob(id),job);
});
test('empty bulk settles the session as failed without ready notification for an all-empty session',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{sessionId:randomUUID(),session:{completed_job_ids:[],failed_job_ids:[id],total_jobs:1},analyze:async()=>({items:[]})});
 await h.run();assert.equal((await queue.getJob(id)).status,'failed');assert.equal(h.sessionFailures,1);assert.equal(h.sessionCompletions,0);assert.equal(h.sessionSettled,0);assert.equal(h.notifications,0);
});
test('an empty last photo still completes a mixed session with the valid photos reviewable',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{sessionId:randomUUID(),session:{completed_job_ids:[randomUUID()],failed_job_ids:[id],total_jobs:2},analyze:async()=>({items:[]})});
 await h.run();assert.equal((await queue.getJob(id)).status,'failed');assert.equal(h.sessionFailures,1);assert.equal(h.notifications,1);
});
test('nonempty bulk keeps completed and ready semantics',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{sessionId:randomUUID(),session:{completed_job_ids:[id],failed_job_ids:[],total_jobs:1}});
 await h.run();const job=await queue.getJob(id);assert.equal(job.status,'completed');assert.equal(job.result.items.length,1);assert.equal(h.sessionFailures,0);assert.equal(h.sessionCompletions,1);assert.equal(h.notifications,1);
});
test('empty typed receipt is not converted to empty bulk failure',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{mode:'receipt_inventory_deep',analyze:async()=>({items:[],capture_type:'receipt',capture_outcomes:[]})});
 await h.run();const job=await queue.getJob(id);assert.equal(job.status,'completed');assert.notEqual(job.result.error_code,'no_items_identified');
});
test('empty noninventory analysis is not converted to empty bulk failure',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{mode:'product',analyze:async()=>({items:[],product:{name:'Fixture'}})});
 await h.run();assert.equal((await queue.getJob(id)).status,'completed');
});
test('a stale empty worker cannot fail a concurrent nonempty completion or its session',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{leasesEnabled:false,sessionId:randomUUID(),analyze:async()=>{await queue.updateJobStatus(id,'completed',{items:[{item_name:'Winning valid item'}]});return {items:[]};}});
 await h.run();const job=await queue.getJob(id);assert.equal(job.status,'completed');assert.equal(job.result.items.length,1);assert.equal(h.sessionFailures,0);assert.equal(h.notifications,0);
});
test('overlapping empty deliveries analyze once and settle without duplicate effects',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{sessionId:randomUUID(),analyze:async()=>({items:[]})});
 await Promise.all([h.run(),h.run()]);assert.equal(h.calls,1);assert.equal((await queue.getJob(id)).status,'failed');assert.equal(h.sessionFailures,1);assert.equal(h.notifications,0);
});
test('disabled leases rollout still terminalizes an empty bulk result',async()=>{
 const {id,owner}=await fixture();const h=handler(id,owner,{leasesEnabled:false,analyze:async()=>({items:[]})});
 await h.run();assert.equal((await queue.getJob(id)).status,'failed');assert.equal(h.notifications,0);
});

test('actual session store keeps empty jobs terminal without consuming the later valid photo notification',async()=>{
 const {id,owner}=await fixture(),sid=randomUUID();await realSessionStore.ensureSession(sid,owner,id,'bulk_inventory_deep');
 const h=handler(id,owner,{sessionId:sid,sessionStore:realSessionStore,analyze:async()=>({items:[]})});await h.run();const emptySession=await realSessionStore.getSession(sid);
 assert.equal(emptySession.owner_id,owner);assert.deepEqual(Array.from(emptySession.failed_job_ids),[id]);assert.deepEqual(Array.from(emptySession.completed_job_ids),[]);assert.equal(emptySession.notification_sent,undefined);assert.equal(h.notifications,0);assert.equal(realSessionStore.isSessionComplete(emptySession),true);
 await h.run();assert.deepEqual(await realSessionStore.getSession(sid),emptySession);assert.equal(h.calls,1);
 const later=randomUUID();await queue.createJob(later,{status:'pending',analysis_mode:'bulk_inventory_deep',meta:{owner,persist_to_kitchen:false}});await realSessionStore.ensureSession(sid,owner,later,'bulk_inventory_deep');
 const valid=handler(later,owner,{sessionId:sid,sessionStore:realSessionStore});await valid.run();const completed=await realSessionStore.getSession(sid);
 assert.equal(completed.status,'completed');assert.equal(completed.notification_sent,true);assert.equal(valid.notifications,1);assert.deepEqual(Array.from(completed.failed_job_ids),[id]);assert.deepEqual(Array.from(completed.completed_job_ids),[later]);
 await valid.run();assert.equal(valid.notifications,1);assert.deepEqual(await realSessionStore.getSession(sid),completed);
 await assert.rejects(realSessionStore.ensureSession(sid,randomUUID(),randomUUID(),'bulk_inventory_deep'),/owner mismatch/);assert.deepEqual(await realSessionStore.getSession(sid),completed);
});
