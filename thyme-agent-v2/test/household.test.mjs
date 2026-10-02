import test from 'node:test';
import assert from 'node:assert/strict';
import { TrepoGateway } from '../src/gateway.mjs';
import { compactInventory } from '../src/runner.mjs';
import { verifyChange } from '../src/verification.mjs';

const actor = { actor:'matt', household:'home', ctx:{userId:'matt', tableOwnerId:'matt', ownerId:'home', householdMemberIds:['matt','member']} };
function setup() {
  const gateway = new TrepoGateway({env:{}});
  const state = {members:['matt','member'], calls:[], rows:[{_id:'a', owner_id:'matt', product_name:'Milk'}, {_id:'b',owner_id:'member',product_name:'Eggs'}]};
  gateway.check = async () => structuredClone(actor);
  gateway.data = {mapKitchenRow: r => ({id:r._id,item_name:r.product_name}), mapShoppingRow:r=>({...r}), getKitchenItemsFull:async()=>[{id:'a'}],getShoppingItems:async()=>[{action:'REMOVED'}]};
  gateway.mysql = {withDbConnection: async fn => fn({execute:async (sql,params)=> {
    if (sql.includes('FROM new_users')) return [state.members.map(user_id=>({user_id}))];
    if (sql.includes('FROM shared_kitchen')) {
      assert.deepEqual(params,state.members);
      assert.match(sql,/action='IN'/);
      return [state.rows];
    }
    assert.match(sql,/action IN \('ADDED','CHECKED'\)/);
    assert.deepEqual(params,['matt']);
    return [[{action:'ADDED',item_name:'Bread'},{action:'CHECKED',item_name:'Carrots'}]];
  }})};
  gateway.actions = {executeToolAction:async args=> {state.calls.push(args);return {ok:true}}};
  return {gateway,state};
}
test('kitchen includes every current household member; shopping excludes removed history without duplicating mirrors',async()=>{
  const {gateway}=setup();
  assert.deepEqual((await gateway.read(actor,'kitchen')).map(x=>x.id),['a','b']);
  assert.deepEqual((await gateway.read(actor,'shopping')).map(x=>x.action),['ADDED','CHECKED']);
});
test('membership loss between checks fails closed',async()=>{
  const {gateway,state}=setup();state.members=['member'];
  await assert.rejects(gateway.read(actor,'kitchen'),{code:'membership'});
});
test('kitchen edits retain the acting user and exact reviewed ID/revision',async()=>{
  const {gateway}=setup();let request;
  gateway.appAPI=async(a,path,opts)=>{request={a,path,body:opts.body};return {operation_id:opts.body.operation_id,items:[{item:{_id:'b'}}]};};
  await gateway.mutate(actor,'update_item_details',{item_id:'b',new_name:'Fresh eggs'},'operation',[{id:'b',amount_revision:9}]);
  assert.equal(request.a.actor,'matt');assert.match(request.path,/kitchen\/matt/);
  assert.deepEqual(request.body.items,[{item_id:'b',revision:9,fields:{product_name:'Fresh eggs'}}]);
});
test('missing and ambiguous kitchen targets fail before an API call',async()=>{
  const {gateway}=setup();let calls=0;gateway.appAPI=async()=>{calls++;};
  for(const item_id of [undefined,'foreign','gone']) await assert.rejects(gateway.mutate(actor,'discard_item',{item_id,item_name:'Milk'},'operation',[{id:'a',amount_revision:1}]),{code:'ambiguous_item'});
  await assert.rejects(gateway.mutate(actor,'discard_item',{item_id:'a'},'operation',[{id:'a',amount_revision:1},{id:'a',amount_revision:1}]),{code:'ambiguous_item'});
  assert.equal(calls,0);
});
test('fresh inventory preserves zero, unknown amounts and dates, excludes bulky marketing fields, and never silently truncates',()=>{
  const k=[{id:'a',item_name:'Milk',quantity_value:0,is_opened:false,created_at:'2026-10-01',description:'noise'},{id:'b',item_name:'Eggs',quantity_value:null}];
  const snapshot=compactInventory(k,[], 'today');
  assert.equal(snapshot.complete,true);assert.equal(snapshot.kitchen.length,2);
  assert.equal(snapshot.kitchen[0].quantity_value,0);assert.equal(snapshot.kitchen[0].is_opened,false);
  assert.equal(snapshot.kitchen[0].created_at,'2026-10-01');assert.equal(snapshot.kitchen[0].description,undefined);
  assert.equal(snapshot.kitchen[1].quantity_value,undefined);
  assert.equal(compactInventory([{item_name:'a'.repeat(60001)}],[],'today').complete,false);
});
test('current app ADDED state verifies an unchecked item',()=>{
  const before=[{shopping_id:'1',item_name:'Milk',action:'CHECKED'}];
  assert.equal(verifyChange('mark_shopping_item_unbought',{item_name:'Milk'},before,[{...before[0],action:'ADDED'}]).verified,true);
});
