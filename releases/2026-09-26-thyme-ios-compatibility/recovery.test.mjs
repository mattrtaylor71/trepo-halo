import {test,before,after,mock} from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire(import.meta.url);
const dynalite=req(process.env.DYNALITE_PATH || 'dynalite');
const root=pathToFileURL(process.env.THYME_TEST_SYNC_ROOT + '/');
const streamRoot=pathToFileURL(process.env.THYME_TEST_STREAM_ROOT + '/');
const {SignJWT}=await import(new URL('node_modules/jose/dist/webapi/index.js',root));
const {DynamoDBClient,CreateTableCommand,DeleteTableCommand}=req(new URL('node_modules/@aws-sdk/client-dynamodb',root).pathname);
const {DynamoDBDocumentClient,GetCommand,UpdateCommand}=req(new URL('node_modules/@aws-sdk/lib-dynamodb',root).pathname);
const op=await import(new URL('lib/operation-recovery.mjs',root));
let server,db,raw,table,executions=0,throwAfterRun=false,sync,stream,legacyCalls=0;
const actor=crypto.randomUUID(),other=crypto.randomUUID(),house=crypto.randomUUID();
const secret='isolated-only-test-secret-not-a-real-credential';
const operation=()=>`v1.${Math.floor(Date.now()/1000)}.${crypto.randomUUID()}`;
const identity=()=>({id:operation(),expires:Math.floor(Date.now()/1000)+86400,sessionId:crypto.randomUUID(),fingerprint:crypto.randomUUID()});
const env={TOKEN_SIGNING_SECRET:secret};
const context={userId:actor,ownerId:house,tableOwnerId:actor,shoppingNamespace:actor,householdSize:1};
const scope=()=>op.operationScope(identity(),{actor,owner:actor},context,house,env);
async function jwt({key=secret,id=actor,owner=id,issuer='trepo-auth',expired=false,alg='HS256'}={}) {
 return new SignJWT({user_id:id,owner_id:owner}).setProtectedHeader({alg}).setIssuer(issuer).setIssuedAt().setExpirationTime(expired?'0s':'10m').sign(new TextEncoder().encode(key));
}
async function event(id=operation(),session=crypto.randomUUID(),text='Tell me one recipe') {
 return {requestContext:{http:{method:'POST'}},headers:{authorization:'Bearer '+await jwt(),'content-type':'application/json','x-owner-id':actor,'x-client-surface':'app','x-operation-id':id,'x-session-id':session},
 body:JSON.stringify({message:text,session_id:session})};
}
const answer=()=>({text:'A simple fruit salad.',type:'info_answer',toolEvents:[],toolTrace:[],quickItems:[],outcome:'answered'});
async function runAssistant(){executions++;if(throwAfterRun)throw Error('fixture interruption after possible work');return answer();}
async function* runStreaming(){executions++;if(throwAfterRun)throw Error('fixture interruption after possible work');yield {type:'text_delta',delta:'A simple fruit salad.'};return answer();}
async function invokeStream(e){const lines=[];await stream(e,{write:v=>{for(const l of String(v).trim().split('\n'))if(l)lines.push(JSON.parse(l));},end:()=>{}},{});return lines;}
async function recover(e,overrides={}) {
 const body=JSON.parse(e.body);
 return sync({...e,requestContext:{http:{method:'GET'}},queryStringParameters:{operation_id:e.headers['x-operation-id'],session_id:body.session_id,...overrides},body:''},{});
}
before(async()=>{
 server=dynalite({createTableMs:0,deleteTableMs:0});await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const endpoint=`http://127.0.0.1:${server.address().port}`;
 Object.assign(process.env,{AWS_ENDPOINT_URL_DYNAMODB:endpoint,AWS_ACCESS_KEY_ID:'isolatedfixture',AWS_SECRET_ACCESS_KEY:'isolatedfixture',AWS_REGION:'us-east-1',TOKEN_SIGNING_SECRET:secret,OPENAI_API_KEY:'fixture-no-network'});
 raw=new DynamoDBClient({endpoint,region:'us-east-1',credentials:{accessKeyId:'isolatedfixture',secretAccessKey:'isolatedfixture'}});db=DynamoDBDocumentClient.from(raw);
 table='android_recovery_'+crypto.randomUUID().replaceAll('-','');process.env.SESSION_TABLE_NAME=table;env.SESSION_TABLE_NAME=table;
 await raw.send(new CreateTableCommand({TableName:table,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'owner_id',KeyType:'HASH'},{AttributeName:'session_entry_id',KeyType:'RANGE'}],AttributeDefinitions:[{AttributeName:'owner_id',AttributeType:'S'},{AttributeName:'session_entry_id',AttributeType:'S'}]}));
 for(const base of [root,streamRoot]) {
  mock.module(new URL('lib/user-context.mjs',base).href,{namedExports:{lookupUserContextByOwnerId:async()=>context}});
  mock.module(new URL('lib/family-brain.mjs',base).href,{namedExports:{prepareBrainSession:async()=>null,candidateOwner:()=>null,validateBrain:async()=>true}});
  mock.module(new URL('lib/device-assistant.mjs',base).href,{namedExports:{runDeviceAssistant:runAssistant,runDeviceAssistantStreaming:runStreaming,transcribeAudio:async()=> 'Audio fixture transcript'}});
  mock.module(new URL('lib/voice-asset-dispatcher.mjs',base).href,{namedExports:{dispatchVoiceAssetJobs:async()=>{}}});
  mock.module(new URL('lib/session-store.mjs',base).href,{namedExports:{appendSessionMessages:async()=>{},claimRecentSessionRequest:async()=>{legacyCalls++;return {claimed:true,entryId:'legacy'};},clearRecentSessionRequest:async()=>{},completeRecentSessionRequest:async()=>{},getSessionMessageLimit:()=>12,loadRecentSessionMessages:async()=>[]}});
 }
 globalThis.awslambda={streamifyResponse:fn=>fn,HttpResponseStream:{from:s=>s}};
 sync=(await import(new URL('index.js',root))).handler;stream=(await import(new URL('index-stream.js',streamRoot))).handler;
});
after(async()=>{await raw?.send(new DeleteTableCommand({TableName:table}));raw?.destroy();await new Promise(r=>server?.close(r));});

test('strict modern identities reject malformed, future and expired operations',()=>{
 const now=1800000000,uuid=crypto.randomUUID();assert.equal(op.operationIdentity(`v1.${now}.${uuid}`,now).id,`v1.${now}.${uuid}`);
 for(const x of ['v1.bad',`v1.${now-86400}.${uuid}`,`v1.${now+301}.${uuid}`,{}])assert.throws(()=>op.operationIdentity(x,now));
 assert.equal(op.operationIdentity(null),null);
});
test('legacy iOS correlation IDs do not opt into the new protocol',()=>{
 assert.equal(op.inputIdentity({headers:{'x-operation-id':crypto.randomUUID().replaceAll('-','')}},{responseSurface:'app'}),null);
});
test('JWT signature, issuer, expiry, algorithm and requested ownership are all verified',async()=>{
 const good={headers:{authorization:'Bearer '+await jwt(),'x-owner-id':actor}};assert.equal((await op.verifiedActor(good,env)).actor,actor);
 for(const options of [{key:'different-secret'},{issuer:'wrong'},{expired:true},{alg:'HS384'}])await assert.rejects(()=>op.verifiedActor({headers:{...good.headers,authorization:'Bearer '+''}},env));
 for(const options of [{key:'different-secret'},{issuer:'wrong'},{expired:true},{alg:'HS384'}]){
  const token=await jwt(options);await assert.rejects(()=>op.verifiedActor({headers:{...good.headers,authorization:'Bearer '+token}},env));
 }
 await assert.rejects(()=>op.verifiedActor({headers:{...good.headers,'x-owner-id':other}},env),/owner_mismatch/);
 await assert.rejects(()=>op.verifiedActor(good,{}),/recovery_unavailable/);
 assert.throws(()=>op.operationScope(identity(),{actor:other},context,house,env),/owner_mismatch/);
});
test('eight concurrent requests admit one executor and return its exact saved answer',async()=>{
 const s=scope();const claims=await Promise.all(Array.from({length:8},()=>op.claimOperation(s,db)));assert.equal(claims.filter(c=>c.claimed).length,1);
 const c=claims.find(c=>c.claimed);const saved=await op.finishOperation(s,c,{text:'Exact original',receipt:{id:'fixed'}},db);
 const again=await op.claimOperation(s,db);assert.equal(again.claimed,false);assert.deepEqual(again.body,saved);assert.equal(again.status,'completed');
 await assert.rejects(()=>op.finishOperation(s,{token:'stale'},{text:'replacement'},db));assert.deepEqual((await op.readOperation(s,db)).body,saved);
});
test('changed input, session, actor and expired lease never admit a second execution',async()=>{
 const s=scope(),c=await op.claimOperation(s,db);
 assert.equal((await op.claimOperation({...s,fingerprint:'changed'},db)).status,'input_conflict');
 assert.equal((await op.readOperation({...s,sessionId:crypto.randomUUID()},db)).status,'not_found');
 assert.equal((await op.readOperation({...s,actor:other},db)).status,'not_found');
 await db.send(new UpdateCommand({TableName:table,Key:s.key,UpdateExpression:'SET lease_expires_at = :old',ExpressionAttributeValues:{':old':1}}));
 assert.equal((await op.claimOperation(s,db)).status,'unknown_commit');
 await op.markOperationUnknown(s,c,db);assert.equal((await op.claimOperation(s,db)).claimed,false);
});
test('acknowledgment is idempotent and cannot create or change an answer',async()=>{
 const s=scope();assert.deepEqual(await op.acknowledgeOperation(s,db),{acknowledged:false});assert.equal((await op.readOperation(s,db)).status,'not_found');
 const c=await op.claimOperation(s,db),body=await op.finishOperation(s,c,{text:'Saved'},db);
 assert.deepEqual(await op.acknowledgeOperation(s,db),{acknowledged:true});const a=(await db.send(new GetCommand({TableName:table,Key:s.key}))).Item;
 await op.acknowledgeOperation(s,db);const b=(await db.send(new GetCommand({TableName:table,Key:s.key}))).Item;assert.deepEqual(a,b);assert.deepEqual(b.response_body,body);
 assert.equal((await op.acknowledgeOperation({...s,actor:other},db)).acknowledged,false);
});
test('text, audio, timezone and header/body mismatch are bound to the same operation',()=>{
 const id=operation(),session=crypto.randomUUID(),base={headers:{'x-operation-id':id}},input={sessionId:session,responseSurface:'app',transcript:'Hello'};
 const a=op.inputIdentity(base,input);assert.notEqual(a.fingerprint,op.inputIdentity(base,{...input,transcript:'Different'}).fingerprint);
 assert.notEqual(a.fingerprint,op.inputIdentity({headers:{...base.headers,'x-client-time-zone':'Asia/Tokyo'}},input).fingerprint);
 assert.notEqual(op.inputIdentity(base,{...input,audioBuffer:Buffer.from('one')}).fingerprint,op.inputIdentity(base,{...input,audioBuffer:Buffer.from('two')}).fingerprint);
 assert.throws(()=>op.inputIdentity(base,{...input,requestMeta:{operation_id:operation()}}));
});
for(const mode of ['sync','stream']) {
 test(mode+' handler saves before delivery, replays across transports and refuses conflicting input',async()=>{
  executions=0;const e=await event();let first=mode==='sync'?JSON.parse((await sync(e,{})).body):(await invokeStream(e)).find(x=>x.type==='done');
  assert.equal(first.operation.status,'completed');assert.equal(executions,1);const found=JSON.parse((await recover(e)).body);assert.equal(found.operation.status,'completed');assert.equal(found.text,'A simple fruit salad.');
  if(mode==='sync')await invokeStream(e);else await sync(e,{});assert.equal(executions,1);
  const conflict={...e,body:JSON.stringify({...JSON.parse(e.body),message:'Different command'})};const conflictBody=JSON.parse((await sync(conflict,{})).body);assert.equal(conflictBody.operation.status,'input_conflict');assert.equal(executions,1);
  const fresh=await event();await sync(fresh,{});assert.equal(executions,2);
 });
 test(mode+' uncertain execution is retained and never silently retried',async()=>{
  executions=0;throwAfterRun=true;const e=await event();try{if(mode==='sync')await sync(e,{});else await invokeStream(e);}finally{throwAfterRun=false;}
  assert.equal(executions,1);assert.equal(JSON.parse((await recover(e)).body).operation.status,'unknown_commit');await sync(e,{});assert.equal(executions,1);
 });
 test(mode+' unsigned modern request cannot reach the assistant',async()=>{
  executions=0;const e=await event();delete e.headers.authorization;if(mode==='sync')assert.equal((await sync(e,{})).statusCode,401);else assert.ok((await invokeStream(e)).some(x=>x.type==='error'));assert.equal(executions,0);
 });
}
test('GET ownership checks happen before any conversation is read and render ack is harmless',async()=>{
 const e=await event();await sync(e,{});const spoof={...e,headers:{...e.headers,'x-owner-id':other}};assert.equal((await recover(spoof)).statusCode,403);
 assert.equal(JSON.parse((await recover(e,{session_id:crypto.randomUUID()})).body).operation.status,'not_found');
 const ack={...e,headers:{...e.headers,'x-thyme-event':'rendered'},body:JSON.stringify({session_id:JSON.parse(e.body).session_id,operation_id:e.headers['x-operation-id']})};const old=executions;
 assert.equal((await sync(ack,{})).statusCode,200);assert.equal(executions,old);
});
test('existing correlation-only clients keep the unchanged legacy path',async()=>{
 executions=0;legacyCalls=0;const e=await event(crypto.randomUUID().replaceAll('-',''));await sync(e,{});await invokeStream(e);assert.equal(executions,2);assert.equal(legacyCalls,2);
});
test('storage outage fails closed before the assistant executes',async()=>{
 executions=0;const saved=process.env.SESSION_TABLE_NAME;process.env.SESSION_TABLE_NAME='';try{
  const e=await event();assert.equal((await sync(e,{})).statusCode,503);assert.ok((await invokeStream(e)).some(x=>x.type==='error'));assert.equal(executions,0);
 }finally{process.env.SESSION_TABLE_NAME=saved;}
});
test('deleted users and stale household ownership cannot retrieve receipts',()=>{
 assert.throws(()=>op.operationScope(identity(),{actor,owner:actor},{...context,isFallbackContext:true},house,env),/owner_mismatch/);
 assert.throws(()=>op.operationScope(identity(),{actor,owner:other},context,house,env),/owner_mismatch/);
});
test('GET requires a verified token even when the owner, session and operation are known',async()=>{
 const e=await event();await sync(e,{});delete e.headers.authorization;assert.equal((await recover(e)).statusCode,401);
});
test('a signed UUID user with a legacy numeric household remains authenticated',async()=>{
 const token=await jwt({owner:'12345'}),event={headers:{authorization:'Bearer '+token,'x-owner-id':actor}};
 assert.equal((await op.verifiedActor(event,env)).actor,actor);
 assert.equal(op.operationScope(identity(),{actor,owner:'12345'},{...context,ownerId:'12345'},'12345',env).actor,actor);
 await assert.rejects(()=>op.verifiedActor({headers:{...event.headers,'x-owner-id':'67890'}},env),/owner_mismatch/);
});

// The shipped SwiftUI client uses UUID().uuidString.prefix(12), NOT a full UUID.
const iosSession=()=>`chat_${crypto.randomUUID().slice(0,12).toUpperCase()}`;
for (const mode of ['sync','stream']) {
 for (const inputMode of ['text','audio']) {
  test(`iOS ${mode} ${inputMode}: completes, recovers, acknowledges and replays exactly once`,async()=>{
   executions=0; const e=await event(operation(),iosSession());
   if(inputMode==='audio') {
    e.headers['content-type']='audio/wav'; e.headers['x-audio-format']='wav';
    e.body=Buffer.from('isolated-audio-fixture').toString('base64'); e.isBase64Encoded=true;
   }
   const first=mode==='sync'?JSON.parse((await sync(e,{})).body):(await invokeStream(e)).find(x=>x.type==='done');
   assert.equal(first?.operation?.status,'completed'); assert.equal(executions,1);
   const getEvent={...e,requestContext:{http:{method:'GET'}},queryStringParameters:{operation_id:e.headers['x-operation-id'],session_id:e.headers['x-session-id']},body:''};
   const recovered=await sync(getEvent,{}); assert.equal(recovered.statusCode,200);
   const answer=JSON.parse(recovered.body); assert.equal(answer.operation.status,'completed'); assert.equal(answer.text,'A simple fruit salad.');
   await sync(e,{}); await invokeStream(e); assert.equal(executions,1);
   const ack={...e,isBase64Encoded:false,headers:{...e.headers,'x-thyme-event':'rendered'},body:JSON.stringify(getEvent.queryStringParameters)};
   assert.equal((await sync(ack,{})).statusCode,200); assert.equal(executions,1);
   const wrong={...getEvent,queryStringParameters:{...getEvent.queryStringParameters,session_id:iosSession()}};
   assert.equal(JSON.parse((await sync(wrong,{})).body).operation.status,'not_found');
   assert.equal((await sync({...getEvent,headers:{...e.headers,'x-owner-id':other}},{})).statusCode,403);
   assert.deepEqual(JSON.parse((await sync(getEvent,{})).body),answer);
  });
 }
 test(`iOS ${mode}: an interrupted write cannot be executed twice`,async()=>{
  executions=0;throwAfterRun=true;const e=await event(operation(),iosSession());
  try {if(mode==='sync')await sync(e,{});else await invokeStream(e);} finally {throwAfterRun=false;}
  assert.equal(executions,1); assert.equal(JSON.parse((await recover(e)).body).operation.status,'unknown_commit');
  await sync(e,{});await invokeStream(e);assert.equal(executions,1);
 });
}
test('admission and recovery accept the same bounded iOS / Android session formats',async()=>{
 for (const session of [iosSession(),iosSession().toLowerCase(),crypto.randomUUID()]) {
  const e=await event(operation(),session);
  assert.ok(op.inputIdentity(e,{responseSurface:'app',sessionId:session,transcript:'fixture'}));
  assert.equal((await recover(e)).statusCode,200);
 }
 for (const session of ['', 'chat_', 'chat_not-a-uuid', 'chat_01234567-89AZ', 'chat_01234567-89', 'a'.repeat(1000), '12345678-1234-1234-1234-123456789012\n', 'chat_01234567-89A\n', {}, null]) {
  const e=await event(operation(),session);
  assert.throws(()=>op.inputIdentity(e,{responseSurface:'app',sessionId:session,transcript:'fixture'}),/invalid_session_id/);
  assert.equal((await recover(e)).statusCode,400);
 }
});
