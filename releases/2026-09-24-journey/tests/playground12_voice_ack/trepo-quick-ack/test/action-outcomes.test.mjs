import {test} from 'node:test';
import assert from 'node:assert/strict';
import {actionOutcomes,actionOutcomeText} from '../lib/action-outcomes.mjs';
const event=(toolName,toolResult,args={}) => ({toolName,args,result:{toolName,ok:true,statusCode:200,toolResult}});
const describe=events=>actionOutcomeText(actionOutcomes(events));

test('confirmation uses returned entity and quantity, never requested values',()=>{
 const e=event('update_item_quantity',{item:{id:'a',item_name:'Ground beef',remaining_quantity:'12.5 oz'}},{item_id:'a',item_name:'Chicken',remaining_quantity:'25 oz'});
 assert.match(describe([e]),/Ground beef \(12.5 oz\)/);assert.doesNotMatch(describe([e]),/Chicken|\(25 oz\)/);
 assert.equal(actionOutcomes([e])[0].entities[0].id,'a');
});
test('successful result for a different target ID cannot confirm requested mutation',()=>{
 const e=event('update_item_quantity',{item:{id:'b',item_name:'Milk',remaining_quantity:'1 oz'}},{item_id:'a'});
 assert.equal(actionOutcomes([e])[0].state,'unverified');assert.doesNotMatch(describe([e]),/Milk/);
});
test('outer ok cannot hide a failed or mismatched tool result',()=>{
 for(const bad of [{ok:false,error:'not_found'},{count:4},null]) assert.notEqual(actionOutcomes([event('check_in_item',bad)])[0].state,'confirmed');
 const e=event('check_in_item',{item:{id:'a',item_name:'Milk'}});e.result.toolName='discard_item';assert.equal(actionOutcomes([e])[0].state,'unverified');
});
test('partial batch confirms only successful named target IDs',()=>{
 const e=event('add_many_to_meal_calendar',{results:[{ok:true,entry_id:'a',recipe_name:'Pasta',plan_date:'2026-09-18'},{ok:false,recipe_name:'Pizza',error:'not_found'},{ok:true,recipe_name:'Soup'}]});
 const out=actionOutcomes([e])[0];assert.equal(out.state,'partial');assert.equal(out.failed,1);assert.equal(out.unverified,1);assert.equal(out.entities.length,1);
 const text=actionOutcomeText([out]);assert.match(text,/Pasta.*2026-09-18/);assert.doesNotMatch(text,/Pizza|Soup/);
});
test('nested kitchen batches retain IDs and exclude requested-but-missing items',()=>{
 const e=event('check_in_many_items',{items:[{id:'a',item:{id:'a',item_name:'Milk'}},{id:'b',item:{id:'b',item_name:'Eggs'}}]},{items:[{item_name:'Bread'}]});
 assert.match(describe([e]),/Milk/);assert.match(describe([e]),/Eggs/);assert.doesNotMatch(describe([e]),/Bread/);
});
test('404,403 and timeout never become confirmations or instructions to retry automatically',()=>{
 for(const statusCode of [404,403,504]) {
  const e=event('discard_item',{item:{id:'a',item_name:'Milk'}});e.result.statusCode=statusCode;e.result.ok=false;
  assert.equal(actionOutcomes([e])[0].state,statusCode===504?'unverified':'failed');assert.doesNotMatch(describe([e]),/Removed|Milk/);
  if(statusCode===504)assert.match(describe([e]),/Check the relevant list before trying again/);
 }
});
test('read result and unnamed count are not proof of a write',()=>{
 assert.deepEqual(actionOutcomes([event('get_kitchen_overview',{items:[{id:'a',item_name:'Milk'}]})]),[]);
 assert.equal(actionOutcomes([event('clear_kitchen_inventory',{count:20})])[0].state,'unverified');
});
test('blocked unrequested writes do not overwrite a valid informational answer',()=>{
 const e=event('log_dish_from_voice',null);e.result={ok:false,suppressed:true,statusCode:409,error:'write_not_authorized_by_recipe_request'};
 assert.deepEqual(actionOutcomes([e]),[]);
});
test('async recipe placeholder without recipe ID does not say saved',()=>{
 assert.equal(actionOutcomes([event('save_generated_recipe',{recipe:{title:'Pasta'}})])[0].state,'unverified');
 assert.doesNotMatch(describe([event('save_generated_recipe',{recipe:{title:'Pasta'}})]),/Saved/);
 assert.match(describe([event('save_generated_recipe',{recipe:{id:'r',title:'Pasta'}})]),/Saved to your recipes/);
});
