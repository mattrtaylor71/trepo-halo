import test from 'node:test';
import assert from 'node:assert/strict';
import {legacyDateRefreshEvents} from './legacy-date-refresh.mjs';
test('confirmed date edit restores old refresh event without leaking arguments',()=>{
  const result={toolEvents:[{toolName:'update_item_details',args:{bought_date:'2026-10-20',item_name:'private item'},result:{ok:true}}]};
  assert.deepEqual(legacyDateRefreshEvents(result),[
    {type:'tool_start',tool_name:'update_item_details'},
    {type:'tool_end',tool_name:'update_item_details',ok:true}]);
  assert.equal(JSON.stringify(legacyDateRefreshEvents(result)).includes('private item'),false);
  assert.equal(legacyDateRefreshEvents({toolEvents:[...result.toolEvents,...result.toolEvents]}).length,2);
});
test('failed, ambiguous, read-only and unrelated actions do not announce an edit',()=>{
  for (const result of [null,{}, {toolEvents:[{toolName:'update_item_details',args:{bought_date:null},result:{ok:false}}]},
    {toolEvents:[{toolName:'read_kitchen',result:{ok:true}}]},
    {toolEvents:[{toolName:'update_item_details',args:{new_name:'x'},result:{ok:true}}]}]) assert.deepEqual(legacyDateRefreshEvents(result),[]);
});
test('clearing the date also invalidates old cached data',()=>{
  assert.equal(legacyDateRefreshEvents({toolEvents:[{toolName:'update_item_details',result:{ok:true,args:{bought_date:null}}}]}).length,2);
});
