import test from 'node:test';
import assert from 'node:assert/strict';
import {TrepoGateway} from '../src/gateway.mjs';
import {verifyChange} from '../src/verification.mjs';
import {pageData} from '../src/prepare-actions.mjs';
const actor={actor:'test',household:'family'};
test('pilot normalizes date before approval and sends a date-only canonical edit',async()=>{
 const gateway=new TrepoGateway({env:{}});gateway.data={};let query,request;
 gateway.mysql={withDbConnection:async cb=>cb({execute:async(sql,params)=>{query={sql,params};return [[{time_zone:'America/Los_Angeles'}]];}})};
 const args={item_id:'milk',bought_date:'October 19, 2026'};
 await gateway.normalizeProposal('update_item_details',args,actor);
 assert.equal(args.bought_date,'2026-10-19');assert.deepEqual(query.params,['family']);
 gateway.appAPI=async(a,path,options)=>{request=options.body;return {operation_id:request.operation_id,items:[{item:{_id:'milk',acquired_on:'2026-10-19'}}]};};
 await gateway.kitchenEdit(actor,'update_item_details',args,'test',[{id:'milk',amount_revision:9}]);
 assert.deepEqual(request.items,[{item_id:'milk',revision:9,fields:{},shelf_dates:{acquired_on:'2026-10-19'}}]);
 await assert.rejects(()=>gateway.kitchenEdit(actor,'update_item_details',{...args,quantity_value:0},'other',[{id:'milk',amount_revision:9}]),/separately/);
});
test('pilot verification requires the exact saved bought date, including explicit null',()=>{
 const before=[{id:'milk',item_name:'Milk',bought_date:'2026-10-20'}];
 for(const value of ['2026-10-19',null]) {
  const args={item_id:'milk',bought_date:value};
  assert.equal(verifyChange('update_item_details',args,before,[{...before[0],bought_date:value}]).verified,true);
  assert.equal(verifyChange('update_item_details',args,before,[{id:'milk',item_name:'Milk'}]).verified,false);
  assert.equal(verifyChange('update_item_details',args,before,before).verified,false);
 }
 const items=pageData([{id:'milk',bought_date:null}],{},'kitchen').items;
 assert.equal(Object.hasOwn(items[0],'bought_date'),true);assert.equal(items[0].bought_date,null);
});
