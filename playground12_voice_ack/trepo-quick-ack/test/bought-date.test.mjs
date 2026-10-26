import {test,mock,beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeBoughtDate,boughtDateFields} from '../../../shared/voice-assistant/bought-date.mjs';
import {validateToolCall} from '../../../shared/voice-assistant/tool-executor.mjs';
const lib=new URL('../lib/',import.meta.url);
const retained=new Map();
mock.module(new URL('session-store.mjs',lib).href,{namedExports:{readKitchenEditRequest:async s=>retained.get(s.operationId),retainKitchenEditRequest:async(s,b)=>{retained.set(s.operationId,b);return b;}}});
const {createKitchenMutationContext,kitchenEditBody,prepareKitchenEdit,saveKitchenEdit,renameReference}=await import('../lib/kitchen-edit-client.mjs');
const {applyKitchenTask,rememberKitchenTaskResult,explicitKitchenEdit}=await import('../lib/kitchen-task-context.mjs');
const at={now:new Date('2026-10-27T01:00:00Z'),timeZone:'America/Los_Angeles'};
const context={userId:'test',ownerId:'test-family'}, row={_id:'milk',amount_revision:4};
beforeEach(()=>{retained.clear();mock.restoreAll();});
test('calendar handles exact dates and relative phrases in household time',()=>{
 for(const [input,want] of [['today','2026-10-26'],['yesterday','2026-10-25'],['3 days ago','2026-10-23'],['two weeks ago','2026-10-12'],['last Friday','2026-10-23'],['last Monday','2026-10-19'],['October 19, 2026','2026-10-19'],['2024-02-29','2024-02-29'],[null,null]]) assert.equal(normalizeBoughtDate(input,at),want);
 assert.equal(normalizeBoughtDate('today',{...at,timeZone:'Asia/Tokyo'}),'2026-10-27');
 for(const bad of ['last week','10/11/26','2026-02-30','2026-10-27','February 29, 2026','',{},'next Friday']) assert.throws(()=>normalizeBoughtDate(bad,at),{statusCode:400});
});
test('shared action schema accepts date-only and explicit unknown, rejects invalid values',()=>{
 for(const value of ['yesterday',null]) assert.equal(validateToolCall('update_item_details',{item_name:'milk',bought_date:value}).ok,true);
 for(const value of ['',false,123,{}]) assert.equal(validateToolCall('update_item_details',{item_name:'milk',bought_date:value}).ok,false);
});
test('date-only body is separate from expiration, created date and quantity',()=>{
 const mutation=createKitchenMutationContext({operationId:'request'});
 const body=kitchenEditBody(mutation,row,{},undefined,{acquired_on:'2026-10-19'});
 assert.deepEqual(body.items,[{item_id:'milk',revision:4,fields:{},shelf_dates:{acquired_on:'2026-10-19'}}]);
 assert.throws(()=>kitchenEditBody(mutation,row,{}),/Unsupported/);
 assert.throws(()=>kitchenEditBody(mutation,row,{},undefined,{opened_on:'2026-10-19'}),/Unsupported/);
});
test('retained request is reused before re-resolving relative date or revision',async()=>{
 let reads=0;const options={mutationContext:createKitchenMutationContext({operationId:'turn'}),sourceTranscript:'Change bought date for milk to yesterday'};
 const resolve=async()=>{reads++;return {row,fields:{},shelfDates:{acquired_on:'2026-10-25'}};};
 const first=await prepareKitchenEdit(context,{item_name:'milk'},{bought_date:'yesterday'},options,resolve);
 options.mutationContext=createKitchenMutationContext({operationId:'turn'});
 const again=await prepareKitchenEdit(context,{item_name:'milk'},{bought_date:'yesterday'},options,async()=>{throw Error('should never resolve again');});
 assert.equal(reads,1);assert.deepEqual(again,first);
});
test('only matching persisted bought date earns successful receipt, including null',async()=>{
 for(const value of ['2026-10-19',null]) {
  const options={mutationContext:createKitchenMutationContext(),shelfDates:{acquired_on:value}};
  let received;
  mock.method(globalThis,'fetch',async(url,init)=>{
   received=JSON.parse(init.body);return {ok:true,json:async()=>({operation_id:received.operation_id,items:[{item:{_id:'milk',amount_revision:6,acquired_on:value}}]})};
  });
  const saved=await saveKitchenEdit(context,row,{},options);
  assert.equal(saved.mutation_outcome,'applied');assert.equal(saved.acquired_on,value);
  mock.restoreAll();
 }
 for(const result of [{}, {acquired_on:'2026-10-20'}]) {
  mock.method(globalThis,'fetch',async(url,init)=>({ok:true,json:async()=>({operation_id:JSON.parse(init.body).operation_id,items:[{item:{_id:'milk',amount_revision:5,...result}}]})}));
  await assert.rejects(()=>saveKitchenEdit(context,row,{}, {shelfDates:{acquired_on:null}}),/Could not confirm/);mock.restoreAll();
 }
});
test('fresh date correction cannot use a model-selected ID to evade duplicate-item clarification',()=>{
 assert.deepEqual(renameReference({item_name:'milk',item_id:'guess'},{bought_date:'yesterday'},{sourceTranscript:'Change the bought date for milk to yesterday'}),{item_name:'milk',item_id:null,selection_hint:null});
 assert.throws(()=>renameReference({item_name:'eggs',item_id:'guess'},{bought_date:'yesterday'},{sourceTranscript:'Change the bought date for milk to yesterday'}),/repeat the item/);
});
test('duplicate clarification retains server-resolved date and revision',async()=>{
 let saved;const state={table:'fixture',client:{send:async command=>{saved=command.input.Item;}},scope:{household:'home',actor:'test',session:'conversation'},available:true,now:123,transcript:'Change bought date for milk to yesterday'};
 await rememberKitchenTaskResult(state,'update_item_details',{args:{bought_date:'yesterday'},details:{type:'ambiguous_kitchen_item',bought_date:'2026-10-25',candidates:[{item_id:'a',item_name:'Milk',revision:4,created_at:'2026-10-20'},{item_id:'b',item_name:'Milk',revision:7,created_at:'2026-10-21'}]}},'original-request');
 assert.equal(saved.task.kind,'bought_date');state.turn='continue';
 assert.deepEqual(applyKitchenTask(state,'the oldest one','update_item_details',{bought_date:'today'}),{args:{item_id:'a',bought_date:'2026-10-25'},expectedRevision:4,operationId:'original-request'});
 assert.ok(applyKitchenTask(state,'the one','update_item_details',{}).error);
 assert.equal(applyKitchenTask(state,'the older one','update_item_details',{}).args.item_id,'a');
 assert.equal(applyKitchenTask(state,'newer','update_item_details',{}).args.item_id,'b');
 state.task.candidates.push({item_id:'c',item_name:'Milk',revision:1,created_at:'2026-10-22'});
 assert.ok(applyKitchenTask(state,'older','update_item_details',{}).error);
 for(const text of ['I bought the milk yesterday','Can you change bought date for milk to yesterday']) assert.equal(explicitKitchenEdit(text),true);
});
test('reading bought date never turns check-in time into a claimed purchase',()=>{
 assert.deepEqual(boughtDateFields({_createdDate:'2026-10-26'}),{});
 assert.deepEqual(boughtDateFields({acquired_on:'2026-10-19'}),{bought_date:'2026-10-19'});
 assert.deepEqual(boughtDateFields({acquired_on:null}),{bought_date:null});
});

test('correction lookup resolves current household, including partner items',async()=>{
 const {householdCorrectionRows}=await import('../../../shared/voice-assistant/bought-date.mjs');
 let data=[{_id:'partner-eggs',owner_id:'partner',product_name:'Eggs',_correction_actor:'test',_correction_household:'test-family'}];
 const connection={execute:async(sql,params)=>{assert.deepEqual(params,['test']);assert.match(sql,/member.owner_id=actor.owner_id/);assert.match(sql,/k.action='IN'/);return [data];}};
 assert.equal((await householdCorrectionRows(connection,context))[0]._id,'partner-eggs');
 data=[{...data[0],_correction_household:'moved'}];await assert.rejects(()=>householdCorrectionRows(connection,context),{statusCode:403});
 data=[];await assert.rejects(()=>householdCorrectionRows(connection,context),{statusCode:403});
});

test('tool catalog exposes the date on edits, not unsupported check-ins',async()=>{
 const {buildTools}=await import('../../../shared/voice-assistant/tool-definitions.mjs');
 const definitions=buildTools();
 const edit=definitions.find(t=>t.name==='update_item_details');
 assert.deepEqual(edit.parameters.properties.bought_date.type,['string','null']);
 assert.equal(definitions.find(t=>t.name==='check_in_item').parameters.properties.bought_date,undefined);
});
test('visible confirmation shows saved date and allows duplicate/date clarification through',async()=>{
 const {actionOutcomes,actionOutcomeText}=await import('../lib/action-outcomes.mjs');
 const event={toolName:'update_item_details',args:{item_name:'Milk',bought_date:'yesterday'},result:{ok:true,toolResult:{item:{id:'milk',item_name:'Milk',bought_date:'2026-10-25',quantity_value:1,quantity_unit:'count'}}}};
 assert.match(actionOutcomeText(actionOutcomes([event])),/Milk · Bought 2026-10-25/);
 for(const type of ['ambiguous_kitchen_item','invalid_bought_date']) {
  assert.deepEqual(actionOutcomes([{...event,result:{ok:false,details:{type}}}]),[]);
 }
});
