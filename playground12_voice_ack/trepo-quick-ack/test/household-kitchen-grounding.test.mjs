import {test,mock,before} from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
const lib=resolve(import.meta.dirname,'../lib');
let build, items=[], failed=false;
before(async()=>{
 const data=await import(lib+'/data-access.mjs');
 mock.module(lib+'/data-access.mjs',{namedExports:{...data,getKitchenItemsFull:async()=>{if(failed)throw new Error('read failed');return items;}}});
 ({buildKitchenGroundingMessage:build}=await import(lib+'/device-assistant.mjs'));
});
const ctx={userId:'member',ownerId:'household'}, env={ACTION_MODE:'real'};
test('successful empty kitchen is explicit and does not rely on old messages',async()=>{
 items=[];failed=false;assert.match(await build(ctx,env),/successfully checked: 0 active items/);
});
test('failed read cannot be described as empty or complete',async()=>{
 failed=true;const message=await build(ctx,env);assert.match(message,/inventory is unavailable/);assert.match(message,/Do not claim the kitchen is empty/);
});
test('partner items in compact tail remain visible and supersede old incorrect answers',async()=>{
 failed=false;items=Array.from({length:100},(_,i)=>({item_name:i===99?'Dijon Mustard':'Food '+i}));
 const message=await build(ctx,env);assert.match(message,/Dijon Mustard/);assert.match(message,/whole household/);assert.match(message,/supersedes earlier conversation/);
});
