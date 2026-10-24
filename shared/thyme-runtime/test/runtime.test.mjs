import test from 'node:test';
import assert from 'node:assert/strict';
import {planTurn,selectTools,runtimeEnabled} from '../policy.mjs';
import {activity,toolActivity,updateActivities} from '../progress.mjs';
import {ResponsesClient,chatInput} from '../responses.mjs';
test('routing is explicit actor opt-in, short vs durable, with conservative unknown tool access',()=>{
 assert.equal(runtimeEnabled({THYME_RUNTIME_V1:'1',THYME_RUNTIME_ACTORS:'["pilot"]'},'pilot'),true);
 for(const actors of ['["*"]','broken','[]'])assert.equal(runtimeEnabled({THYME_RUNTIME_V1:'1',THYME_RUNTIME_ACTORS:actors},'other'),false);
 assert.equal(planTurn('What can I make for breakfast?').model,'gpt-6.1-sol');
 assert.equal(planTurn('Plan meals for next week').mode,'task');
 const tools=['read_trepo','create_recipe','request_remove_shopping_list','ask_clarifying_question','lookup_household_history'].map(name=>({type:'function',name}));
 assert.deepEqual(selectTools(tools,planTurn('Hello')),tools);
 const names=selectTools(tools,planTurn('breakfast'), 'request_remove_shopping_list').map(t=>t.name);
 assert.ok(names.includes('create_recipe'));assert.ok(names.includes('request_remove_shopping_list'));assert.ok(names.includes('ask_clarifying_question'));
});
test('progress excludes arguments, preserves failed steps and stays bounded',()=>{
 let rows=[];
 rows=updateActivities(rows,toolActivity('request_add_to_shopping_list',{id:'a',args:{secret:'PRIVATE'}}));
 assert.equal(rows[0].label,'Preparing changes for your review');assert.ok(!JSON.stringify(rows).includes('PRIVATE'));
 rows=updateActivities(rows,toolActivity('request_add_to_shopping_list',{id:'a',state:'failed'}));
 for(let i=0;i<7;i++)rows=updateActivities(rows,activity('thinking',{id:String(i)}));
 assert.equal(rows[0].state,'failed');
 rows=updateActivities(rows,activity('validating'));assert.equal(rows.length,8);
});
test('Responses uses explicit tools, no storage, body deadline and safe errors',async()=>{
 let body,signal;
 const c=new ResponsesClient('fixture',{fetcher:async(url,options)=>{body=JSON.parse(options.body);signal=options.signal;return {ok:true,json:async()=>({status:'completed',output:[]})};}});
 await c.complete({input:[],tools:[{type:'function',name:'read',defer_loading:true,parameters:{type:'object'}}],requestId:'op'});
 assert.equal(body.store,false);assert.equal(body.reasoning.effort,'low');assert.equal(body.tools[0].defer_loading,undefined);assert.ok(signal instanceof AbortSignal);
 c.fetcher=async()=>({ok:true,json:async()=>({status:'incomplete',output:[]})});
 await assert.rejects(c.complete({input:[]}),{code:'incomplete_response'});
 await assert.rejects(c.complete({input:[],deadline:0}),{code:'timeout'});
 c.fetcher=async()=>({ok:false,status:429,body:{cancel:async()=>{}}});
 await assert.rejects(c.complete({input:[]}),{code:'provider_error',providerStatus:429});
});
test('chat adapter preserves call IDs and results without creating assistant facts',()=>{
 assert.deepEqual(chatInput([{role:'assistant',tool_calls:[{id:'call',function:{name:'read',arguments:'{}'}}]},{role:'tool',tool_call_id:'call',content:'{"ok":true}'}]),[
 {type:'function_call',call_id:'call',name:'read',arguments:'{}'},{type:'function_call_output',call_id:'call',output:'{"ok":true}'}]);
});
test('legacy tool continuation retains opaque reasoning exactly once',()=>{
 const output=[{type:'reasoning',encrypted_content:'opaque'}, {type:'function_call',call_id:'first',name:'read',arguments:'{}'}, {type:'function_call',call_id:'second',name:'read',arguments:'{}'}];
 const map=new Map([['first',output],['second',output]]);
 const input=chatInput([{role:'assistant',tool_calls:[{id:'first',function:{name:'read',arguments:'{}'}},{id:'second',function:{name:'read',arguments:'{}'}}]},{role:'tool',tool_call_id:'first',content:'ok'}],map);
 assert.deepEqual(input,[...output,{type:'function_call_output',call_id:'first',output:'ok'}]);
});
