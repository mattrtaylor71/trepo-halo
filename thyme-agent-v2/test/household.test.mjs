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
test('approved member item uses exact owning row while retaining acting user',async()=>{
  const {gateway,state}=setup();
  await gateway.mutate(actor,'update_item_details',{item_id:'b',new_name:'Fresh eggs'},'operation');
  assert.equal(state.calls[0].userContext.tableOwnerId,'member');
  assert.equal(state.calls[0].userContext.userId,'matt');
  assert.equal(state.calls[0].userContext.requireExactKitchenId,true);
  assert.equal(state.calls[0].mutationContext.actor,'matt');
  assert.equal(actor.ctx.tableOwnerId,'matt');
});
test('missing, ambiguous and outside-household IDs cannot fall back to an item with the same name',async()=>{
  const {gateway,state}=setup();
  for (const item_id of [undefined,'foreign','gone'])
    await assert.rejects(gateway.mutate(actor,'discard_item',{item_id,item_name:'Milk'},'operation'),{code:'item_changed'});
  state.rows.push({...state.rows[0],owner_id:'member'});
  await assert.rejects(gateway.mutate(actor,'discard_item',{item_id:'a'},'operation'),{code:'item_changed'});
  assert.equal(state.calls.length,0);
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
