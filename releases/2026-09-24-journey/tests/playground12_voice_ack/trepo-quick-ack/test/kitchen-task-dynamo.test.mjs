import {test,before,after} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {DynamoDBClient,CreateTableCommand,DeleteTableCommand} from '@aws-sdk/client-dynamodb';
import {DynamoDBDocumentClient} from '@aws-sdk/lib-dynamodb';
import {loadKitchenTask,storeKitchenTask,prepareKitchenTask,validateTaskHistory,applyKitchenTask,taskOrigin} from '../lib/kitchen-task-context.mjs';
// SDK calls made inside the actual session store remain confined to this local service.
process.env.AWS_ENDPOINT_URL_DYNAMODB='http://127.0.0.1:38001';
process.env.AWS_ACCESS_KEY_ID='fixture';process.env.AWS_SECRET_ACCESS_KEY='fixture';process.env.AWS_REGION='us-east-1';
const {appendSessionMessages,loadRecentSessionMessages}=await import('../lib/session-store.mjs');
const raw=new DynamoDBClient({endpoint:process.env.AWS_ENDPOINT_URL_DYNAMODB,region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'},maxAttempts:1});
const client=DynamoDBDocumentClient.from(raw),table='kitchen_task_'+randomUUID().replaceAll('-',''),env={SESSION_TABLE_NAME:table};
const scope={household:'fixture-house',actor:'fixture-actor',session:'fixture-session'};
const owner={householdOwnerId:scope.household,userId:scope.actor};
const candidates=[{item_id:'old',item_name:'Garlic Oil',revision:2,created_at:'2026-09-01T00:00:00Z'},{item_id:'new',item_name:'Garlic Oil',revision:3,created_at:'2026-09-23T00:00:00Z'}];
const load=(s=scope,now)=>loadKitchenTask(s,env,{client,...(now?{now}:{})});
const history=()=>loadRecentSessionMessages(scope.household,scope.session,12,env,{actorId:scope.actor});
before(()=>raw.send(new CreateTableCommand({TableName:table,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'owner_id',KeyType:'HASH'},{AttributeName:'session_entry_id',KeyType:'RANGE'}],AttributeDefinitions:[{AttributeName:'owner_id',AttributeType:'S'},{AttributeName:'session_entry_id',AttributeType:'S'}]})));
after(async()=>{await raw.send(new DeleteTableCommand({TableName:table}));raw.destroy();});
test('twenty simultaneous task writes admit one revision and scope cannot leak',async()=>{
 const s={...scope,session:'race'};const states=await Promise.all(Array.from({length:20},()=>load(s)));
 const result=await Promise.all(states.map((state,i)=>storeKitchenTask(state,{kind:'rename',new_name:'Name '+i,origin:'fixture',candidates})));
 assert.equal(result.filter(Boolean).length,1);const saved=await load(s);assert.equal(saved.revision,1);
 for(const changes of [{actor:'foreign'},{household:'foreign'},{session:'foreign'}])assert.equal((await load({...s,...changes})).task,null);
 assert.equal((await load(s,Math.floor(Date.now()/1000)+601)).task,null);
});
test('actual stored session messages validate a restart and reject stale task after a failed new task save',async()=>{
 const first='Rename Garlic Oil to Garlic Olive Oil';
 await appendSessionMessages(owner,scope.session,[{role:'user',content:first},{role:'assistant',content:'Which Garlic Oil?'}],env);
 await storeKitchenTask(await load(),{kind:'rename',new_name:'Garlic Olive Oil',origin:taskOrigin(first),operation_id:'origin',candidates});
 let next=await load();validateTaskHistory(next,await history());await prepareKitchenTask(next,'most recent');
 assert.equal(applyKitchenTask(next,'most recent','update_item_details',{}).args.item_id,'new');
 // Simulate a new conversation turn persisted while its task write fails.
 await appendSessionMessages(owner,scope.session,[{role:'user',content:'Rename Garlic Oil to Sesame Oil'},{role:'assistant',content:'I could not confirm that correction.'}],env);
 next=await load();validateTaskHistory(next,await history());await prepareKitchenTask(next,'most recent');
 assert.equal(next.task,null);assert.ok(applyKitchenTask(next,'most recent','update_item_details',{new_name:'Old name'}).error);
});
test('missing history cannot authorize a retained correction; another actor has no messages',async()=>{
 const next=await load();validateTaskHistory(next,[]);await prepareKitchenTask(next,'the oldest one');
 assert.ok(applyKitchenTask(next,'the oldest one','update_item_details',{new_name:'Name'}).error);
 assert.deepEqual(await loadRecentSessionMessages(scope.household,scope.session,12,env,{actorId:'different-actor'}),[]);
});
