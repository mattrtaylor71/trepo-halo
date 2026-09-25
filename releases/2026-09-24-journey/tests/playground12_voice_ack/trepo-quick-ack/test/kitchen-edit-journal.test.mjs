import {test,before,after} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {DynamoDBClient,CreateTableCommand,DeleteTableCommand} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient,ScanCommand,PutCommand,DeleteCommand} from '@aws-sdk/lib-dynamodb';
import {prepareKitchenEdit,createKitchenMutationContext,kitchenEditBody} from '../lib/kitchen-edit-client.mjs';
import {readKitchenEditRequest,retainKitchenEditRequest,claimRecentSessionRequest} from '../lib/session-store.mjs';
process.env.AWS_ENDPOINT_URL_DYNAMODB='http://127.0.0.1:38001';
process.env.AWS_ACCESS_KEY_ID='fixture';process.env.AWS_SECRET_ACCESS_KEY='fixture';process.env.AWS_REGION='us-east-1';
const raw=new DynamoDBClient({endpoint:process.env.AWS_ENDPOINT_URL_DYNAMODB,region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});
const client=DynamoDBDocumentClient.from(raw),table='kitchen_replay_'+randomUUID().replaceAll('-',''),env={SESSION_TABLE_NAME:table};
const context={ownerId:'household',userId:'actor'},reference={item_name:'Old label'},updates={new_name:'New label'};
const options=id=>({env,sourceTranscript:'Rename Old label to New label',mutationContext:createKitchenMutationContext({operationId:id,authorization:'Bearer must-never-be-stored'})});
const resolved={row:{_id:'item',amount_revision:3},fields:{product_name:'New label'}};
before(()=>raw.send(new CreateTableCommand({TableName:table,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'owner_id',KeyType:'HASH'},{AttributeName:'session_entry_id',KeyType:'RANGE'}],AttributeDefinitions:[{AttributeName:'owner_id',AttributeType:'S'},{AttributeName:'session_entry_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new DeleteTableCommand({TableName:table}));raw.destroy();});
test('fresh handler reuses original bytes before renamed/deleted target lookup',async()=>{
 const first=options('restart');const a=await prepareKitchenEdit(context,reference,updates,first,async()=>resolved);
 const original=kitchenEditBody(first.mutationContext,a.row,a.fields);
 const next=options('restart');const b=await prepareKitchenEdit(context,reference,updates,next,async()=>assert.fail('must not look up old name or reread newer revision'));
 assert.deepEqual(kitchenEditBody(next.mutationContext,b.row,b.fields),original);
 assert.equal(b.row.amount_revision,3);
});
test('same identity with changed fields, reference, or transcript fails before lookup',async()=>{
 await prepareKitchenEdit(context,reference,updates,options('conflict'),async()=>resolved);
 for(const change of [{updates:{new_name:'Other'}},{reference:{item_name:'Other'}},{sourceTranscript:'different request'}]){
  const opts={...options('conflict'),...(change.sourceTranscript?{sourceTranscript:change.sourceTranscript}:{})};
  await assert.rejects(prepareKitchenEdit(context,change.reference||reference,change.updates||updates,opts,async()=>assert.fail('conflict must precede lookup')),e=>e.statusCode===409);
 }
});
test('actor and household partitions never reuse another callers journal',async()=>{
 await prepareKitchenEdit(context,reference,updates,options('scope'),async()=>resolved);
 for(const other of [{...context,userId:'other'},{...context,ownerId:'other'}]){
  let calls=0;const result=await prepareKitchenEdit(other,reference,updates,options('scope'),async()=>{calls++;return {...resolved,row:{_id:'their-item',amount_revision:9}};});
  assert.equal(calls,1);assert.equal(result.row._id,'their-item');
 }
});
test('concurrent fresh handlers retain one immutable revision before either sends a write',async()=>{
 const results=await Promise.all(Array.from({length:12},(_,i)=>prepareKitchenEdit(context,reference,updates,options('race'),async()=>({...resolved,row:{_id:'item',amount_revision:i}}))));
 assert.equal(new Set(results.map(r=>r.row.amount_revision)).size,1);
});
test('missing or unavailable retry storage fails closed before target lookup',async()=>{
 for(const failEnv of [{},{SESSION_TABLE_NAME:'absent_'+randomUUID()}])
  await assert.rejects(prepareKitchenEdit(context,reference,updates,{...options('missing'),env:failEnv},async()=>assert.fail('must fail before lookup')),e=>e.statusCode===503);
});
test('lost journal write acknowledgment cannot authorize transport and is recovered on retry',async()=>{
 const scope={household:'household',actor:'actor',operationId:'journal-uncertain',intentHash:'fixture'};
 const body={operation_id:scope.operationId,kind:'edit',items:[{item_id:'item',revision:1,fields:{brand:'Brand'}}]};
 const uncertain={send:async(command,opts)=>{await client.send(command,opts);throw Error('ack lost');}};
 await assert.rejects(retainKitchenEditRequest(scope,body,env,{client:uncertain}),/ack lost/);
 assert.deepEqual(await readKitchenEditRequest(scope,env,{client}),body);
 assert.deepEqual(await retainKitchenEditRequest(scope,{...body,items:[{...body.items[0],revision:2}]},env,{client}),body);
});
test('corrupt retained body fails closed rather than falling back to new resolution',async()=>{
 const opts=options('corrupt');await prepareKitchenEdit(context,reference,updates,opts,async()=>resolved);
 const id=[...opts.mutationContext.editIntents.values()][0];
 const {Items}=await client.send(new ScanCommand({TableName:table}));const item=Items.find(i=>i.request_body.operation_id===id);
 item.request_body.items[0].revision='3';await client.send(new PutCommand({TableName:table,Item:item}));
 await assert.rejects(prepareKitchenEdit(context,reference,updates,options('corrupt'),async()=>assert.fail('must not lookup')),e=>e.statusCode===503);
});
test('journal has no expiry, raw transcript, or credentials',async()=>{
 const {Items}=await client.send(new ScanCommand({TableName:table}));assert.ok(Items.length>0);
 for(const item of Items){assert.equal(item.ttl,undefined);assert.equal(item.transcript,undefined);assert.equal(item.authorization,undefined);}
 assert.ok(!JSON.stringify(Items).includes('Bearer must-never-be-stored'));
});

test('separate accepted legacy generations do not replay a prior same-worded command',async()=>{
 const owner={householdOwnerId:'legacy-house',userId:'legacy-actor'};
 const first=await claimRecentSessionRequest(owner,'same-session','Move Milk to the fridge',env);
 assert.equal(first.claimed,true);assert.ok(first.claimToken);
 // DynamoDB TTL eventually deletes the old response-cache record. Emulate only
 // that deletion; keep the durable edit journal to expose accidental reuse.
 const a=options(first.claimToken);const prepared=await prepareKitchenEdit(context,reference,updates,a,async()=>resolved);
 await client.send(new DeleteCommand({TableName:table,Key:{owner_id:owner.householdOwnerId,session_entry_id:first.entryId}}));
 const next=await claimRecentSessionRequest(owner,'same-session','Move Milk to the fridge',env);
 assert.equal(next.entryId,first.entryId);assert.notEqual(next.claimToken,first.claimToken);
 let lookedUp=false;const b=options(next.claimToken);const later=await prepareKitchenEdit(context,reference,updates,b,async()=>{lookedUp=true;return {...resolved,row:{_id:'item',amount_revision:7}};});
 assert.ok(lookedUp);assert.equal(later.row.amount_revision,7);
 assert.notEqual(kitchenEditBody(a.mutationContext,prepared.row,prepared.fields).operation_id,kitchenEditBody(b.mutationContext,later.row,later.fields).operation_id);
});
