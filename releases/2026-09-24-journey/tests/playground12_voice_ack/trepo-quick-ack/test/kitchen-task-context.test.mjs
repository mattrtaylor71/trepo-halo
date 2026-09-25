import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadKitchenTask,prepareKitchenTask,storeKitchenTask,applyKitchenTask,rememberKitchenTaskResult} from '../lib/kitchen-task-context.mjs';
const scope={household:'house',actor:'actor',session:'session'};
const now=1790280000;
function store(){let item;return {async send(command){const p=command.input;if(command.constructor.name==='GetCommand')return {Item:item};if((item?.revision ?? null)!==(p.ExpressionAttributeValues?.[':revision'] ?? null))throw Error('ConditionalCheckFailed');item=structuredClone(p.Item);return {};},get item(){return item;}};}
async function load(client,s=scope,n=now){return loadKitchenTask(s,{SESSION_TABLE_NAME:'fixture'},{client,now:n});}
const candidates=[{item_id:'old',item_name:'Garlic Oil',revision:2,created_at:'2026-09-01T00:00:00Z'},{item_id:'new',item_name:'Garlic Oil',revision:3,created_at:'2026-09-23T00:00:00Z'}];
test('pending rename survives a restart with target revision and original patch',async()=>{
 const db=store(),state=await load(db);
 await rememberKitchenTaskResult(state,'update_item_details',{ok:false,args:{new_name:'Garlic Olive Oil'},details:{type:'ambiguous_kitchen_item',candidates}},'operation-1');
 const next=await prepareKitchenTask(await load(db),'most recent');
 assert.deepEqual(applyKitchenTask(next,'most recent','update_item_details',{new_name:'Wrong',is_opened:true}),{args:{item_id:'new',new_name:'Garlic Olive Oil'},expectedRevision:3,operationId:'operation-1'});
 assert.ok(applyKitchenTask(next,'Drizzle it','mark_item_opened',{}).error);
 assert.ok(applyKitchenTask(next,'Garlic Oil','update_item_details',{}).error);
});
test('state cannot leak between actors, households, sessions, or beyond ten minutes',async()=>{
 const db=store();await storeKitchenTask(await load(db),{kind:'pantry'});
 for(const s of [{...scope,actor:'other'},{...scope,household:'other'},{...scope,session:'other'}])assert.equal((await load(db,s)).task,null);
 assert.equal((await load(db,scope,now+601)).task,null);
});
test('concurrent writers must compare revisions, failed state write grants no pending action',async()=>{
 const db=store(),a=await load(db),b=await load(db);
 assert.equal(await storeKitchenTask(a,{kind:'pantry'}),true);
 assert.equal(await storeKitchenTask(b,{kind:'pantry'}),false);assert.equal(b.task,null);
});
test('pantry follow-ups cannot log a dish or add shopping, questions do not mutate',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Add beef broth to my pantry');
 assert.equal(state.task.kind,'pantry');
 await prepareKitchenTask(state,'One carton');
 assert.ok(applyKitchenTask(state,'One carton','log_dish_from_voice',{}).error);
 assert.ok(applyKitchenTask(state,'One carton','add_to_shopping_list',{}).error);
 assert.equal(applyKitchenTask(state,'One carton','check_in_item',{item_name:'Beef broth'}).error,'Which product is that amount for?');
 await prepareKitchenTask(state,'How much broth is in a carton?');
 assert.ok(applyKitchenTask(state,'How much broth is in a carton?','check_in_item',{}).error);
 await prepareKitchenTask(state,'Never mind');assert.equal(state.task,null);
});
test('task switch clears old state, explicit new opening is not blocked by old rename',async()=>{
 const db=store(),state=await load(db);await storeKitchenTask(state,{kind:'pantry'});
 await prepareKitchenTask(state,'Add milk to my shopping list');assert.equal(state.task,null);
});
test('missing dates or tied candidates require clarification',()=>{
 const state={turn:'continue',task:{kind:'rename',new_name:'New name',candidates:candidates.map(c=>({...c,created_at:''}))}};
 assert.ok(applyKitchenTask(state,'most recent','update_item_details',{}).error);
});

test('a compound name is a target, not a stocking command',async()=>{
 const db=store(),state=await load(db);
 await storeKitchenTask(state,{kind:'rename',new_name:'Organic Chicken Stock',operation_id:'op',candidates:[{item_id:'stock',item_name:'Chicken Stock',revision:0}]});
 await prepareKitchenTask(state,'Chicken Stock');assert.equal(state.task.kind,'rename');
 assert.equal(applyKitchenTask(state,'Chicken Stock','update_item_details',{}).args.item_id,'stock');
 assert.ok(applyKitchenTask(state,'Chicken Stock','check_in_item',{}).error);
});
test('expired or missing pending state cannot authorize a rename from terse selection',async()=>{
 const state=await prepareKitchenTask(await load(store()),'most recent');
 assert.ok(applyKitchenTask(state,'most recent','update_item_details',{item_id:'old',new_name:'Stale name'}).error);
});
test('negative additional-field constraint does not cancel an explicit primary rename',async()=>{
 const state=await prepareKitchenTask(await load(store()),'Rename Garlic Oil to Garlic Olive Oil, do not change its brand');
 assert.deepEqual(applyKitchenTask(state,state.transcript,'update_item_details',{new_name:'Garlic Olive Oil'}).args,{new_name:'Garlic Olive Oil'});
});

test('an amount-only follow-up adjusts the confirmed item instead of adding it again',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Add beef broth to my pantry');
 await rememberKitchenTaskResult(state,'check_in_item',{ok:true,toolResult:{item:{id:'broth',item_name:'Beef broth',amount_revision:0}}},'first');
 await prepareKitchenTask(state,'One carton');
 assert.deepEqual(applyKitchenTask(state,'One carton','check_in_item',{item_name:'Beef broth'}),{
   toolName:'update_item_details',args:{item_id:'broth',quantity_value:1,quantity_unit:'container'},expectedRevision:0});
});
test('an amount continuation without one confirmed target asks a specific question',async()=>{
 const state=await prepareKitchenTask(await load(store()),'Add beef broth to my pantry');
 await prepareKitchenTask(state,'One carton');
 assert.equal(applyKitchenTask(state,'One carton','check_in_item',{item_name:'Guessed item'}).error,'Which product is that amount for?');
});

const {retainKitchenProposal,validateTaskHistory}=await import('../lib/kitchen-task-context.mjs');
test('proposal survives restart, asks before adding and binds yes to the retained product',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');
 const result=await retainKitchenProposal(state,{item_name:'Tuscan olive oil',location:'pantry'});
 assert.equal(result.needs_clarification,true);assert.equal(result.toolResult.inventory_changed,false);
 assert.ok(applyKitchenTask(state,state.transcript,'check_in_item',{item_name:'Guessed'}).error);
 const next=await load(db);validateTaskHistory(next,[{role:'user',content:'Check in my pantry'}]);await prepareKitchenTask(next,'yes');
 const action=applyKitchenTask(next,'yes','check_in_item',{item_name:'Wrong'});
 assert.deepEqual(action,{toolName:'check_in_item',args:{item_name:'Tuscan olive oil',location:'pantry'},claimConfirmation:true});
 assert.ok(applyKitchenTask(next,'yes','check_in_item',{}).error);
});
test('two confirmations cannot both claim the same pending product',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');
 await retainKitchenProposal(state,{item_name:'Beef broth'});
 const a=await prepareKitchenTask(await load(db),'one carton'),b=await prepareKitchenTask(await load(db),'one carton');
 assert.equal(applyKitchenTask(a,'one carton','check_in_item',{}).args.quantity_unit,'container');
 assert.equal(applyKitchenTask(b,'one carton','check_in_item',{}).claimConfirmation,true);
 assert.equal(await storeKitchenTask(a,{...a.task,pending_claim:'a'}),true);
 assert.equal(await storeKitchenTask(b,{...b.task,pending_claim:'b'}),false);
 const retry=await prepareKitchenTask(await load(db),'yes');
 assert.ok(applyKitchenTask(retry,'yes','check_in_item',{}).error);
});
test('question, cancellation, task switch and failed persistence cannot admit a proposal',async()=>{
 for(const turn of ['How much oil is in a bottle?','Never mind','Add milk to my shopping list']){
   const state=await prepareKitchenTask(await load(store()),'Check in my pantry');await prepareKitchenTask(state,turn);
   assert.equal((await retainKitchenProposal(state,{item_name:'Oil'})).ok,false);
 }
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');state.available=false;
 assert.equal((await retainKitchenProposal(state,{item_name:'Oil'})).ok,false);
});
test('a replayed yes after success or expiration cannot add another item',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');
 await retainKitchenProposal(state,{item_name:'Oil'});
 await rememberKitchenTaskResult(state,'check_in_item',{ok:true,toolResult:{item:{id:'oil',item_name:'Oil',amount_revision:0}}},'first');
 for(const clock of [now,now+601]) {
  const next=await prepareKitchenTask(await load(db,scope,clock),'yes');
  assert.ok(applyKitchenTask(next,'yes','check_in_item',{item_name:'Oil'}).error);
 }
});
test('confirmation variants preserve the proposal and every rejection prevents a write',async()=>{
 for(const reply of ['Correct','yes, please','yes, add it','Yes please!']) {
  const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');await retainKitchenProposal(state,{item_name:'Oil'});
  const next=await prepareKitchenTask(await load(db),reply);
  assert.equal(applyKitchenTask(next,reply,'check_in_item',{item_name:'Wrong'}).claimConfirmation,true,reply);
 }
 for(const reply of ['no','no thanks',"no, don't add it"]) {
  const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');await retainKitchenProposal(state,{item_name:'Oil'});
  const next=await prepareKitchenTask(await load(db),reply);
  assert.ok(applyKitchenTask(next,reply,'check_in_item',{item_name:'Oil'}).error,reply);
 }
});
test('unknown commit claim blocks every later pantry write, until an explicit new task',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');
 await storeKitchenTask(state,{...state.task,pending_item:{item_name:'Oil'},pending_claim:'unknown'});
 for(const text of ['yes, please','yes, add it','Oil','one bottle']) {
  const next=await prepareKitchenTask(await load(db),text);
  assert.ok(applyKitchenTask(next,text,'check_in_item',{item_name:'Oil'}).error,text);
 }
 const next=await prepareKitchenTask(await load(db),'Add milk to my pantry');
 assert.equal(next.task.pending_claim,undefined);
 assert.deepEqual(applyKitchenTask(next,next.transcript,'check_in_item',{item_name:'Milk'}),{args:{item_name:'Milk'}});
});
test('amount-only replies need a live target, including after context expires',()=>{
 assert.ok(applyKitchenTask({task:null,turn:'continue'},'one bottle','check_in_item',{item_name:'Oil'}).error);
});

test('without a kitchen task, recipe, shopping and calendar confirmations keep their own context',()=>{
 for(const tool of ['save_generated_recipe','add_to_shopping_list','add_recipe_to_meal_calendar']) {
  for(const text of ['yes please','two cartons']) {
   const args={fixture:'unmodified'};
   assert.deepEqual(applyKitchenTask({task:null,turn:'continue'},text,tool,args),{args},`${tool}: ${text}`);
  }
 }
});
test('a failed pantry-context read does not disable unrelated recipe confirmation',()=>{
 const args={recipe_id:'saved'};
 assert.deepEqual(applyKitchenTask({task:null,blocked:true,turn:'continue'},'yes please','save_generated_recipe',args),{args});
 assert.ok(applyKitchenTask({task:null,blocked:true,turn:'continue'},'yes please','check_in_item',{}).error);
});
test('an active or consumed kitchen confirmation still blocks cross-domain writes',()=>{
 for(const state of [{task:{kind:'pantry'},turn:'continue'},{task:null,confirmationConsumed:true,turn:'continue'}])
  assert.ok(applyKitchenTask(state,'yes please','save_generated_recipe',{}).error);
});
test('explicit rejection or a question cannot mutate another feature after a task switch',async()=>{
 const {toolIntentDenial}=await import('../lib/device-assistant.mjs');
 for(const withTask of [false,true]) for(const text of ['no thanks','cancel','How many cartons are there?']) {
  const state=await load(store());
  if(withTask)await storeKitchenTask(state,{kind:'pantry'});
  await prepareKitchenTask(state,text);
  for(const tool of ['save_generated_recipe','add_to_shopping_list','add_recipe_to_meal_calendar']) {
   const outer=toolIntentDenial(text,tool);
   const scoped=applyKitchenTask(state,text,tool,{item_name:'Milk'});
   assert.ok(outer||scoped.error,`${text}: ${tool}`);
  }
  assert.deepEqual(applyKitchenTask(state,text,'get_shopping_list',{}),{args:{}});
 }
});
test('successful addition with failed state persistence still blocks later writes in that request',async()=>{
 const db=store(),state=await prepareKitchenTask(await load(db),'Check in my pantry');await retainKitchenProposal(state,{item_name:'Oil'});
 const next=await prepareKitchenTask(await load(db),'one bottle');
 assert.equal(applyKitchenTask(next,'one bottle','check_in_item',{}).claimConfirmation,true);
 assert.equal(await storeKitchenTask(next,{...next.task,pending_claim:'one'}),true);
 next.client={send:async()=>{throw Error('failed state write');}};
 assert.equal(await rememberKitchenTaskResult(next,'check_in_item',{ok:true,toolResult:{item:{id:'oil',item_name:'Oil',amount_revision:0}}},'one'),false);
 assert.equal(next.blocked,true);assert.equal(next.task,null);
 assert.ok(applyKitchenTask(next,'one bottle','check_in_item',{item_name:'Oil'}).error);
});
