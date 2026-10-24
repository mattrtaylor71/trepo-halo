import test from 'node:test';
import assert from 'node:assert/strict';
import {MemoryStore} from '../src/store.mjs';
import {Service} from '../src/service.mjs';
import {Runner} from '../src/runner.mjs';
import {ToolGateway,toolDefinitions} from '../src/tools.mjs';
import {key,scope} from '../src/core.mjs';
import {FixtureGateway,ACTOR} from './fixtures.mjs';
const pk=scope(ACTOR);
const answer=text=>({status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text}]}],usage:{input_tokens:100,output_tokens:20}});
const tool=(name,args)=>({status:'completed',output:[{type:'reasoning',encrypted_content:'opaque'},{type:'function_call',call_id:'call-one',name,arguments:JSON.stringify(args)}],usage:{input_tokens:50,output_tokens:10}});
async function setup(responses){
 const store=new MemoryStore(),gateway=new FixtureGateway();gateway.env={THYME_RUNTIME_V1:'1',THYME_RUNTIME_ACTORS:JSON.stringify([ACTOR.actor])};
 const definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions}),service=new Service({store,gateway});
 const runner=new Runner({store,gateway,definitions,tools,responses});
 return {store,gateway,definitions,tools,service,runner};
}
test('quick route publishes real steps and current household context without a provider polling loop',async()=>{
 let input;
 const r=await setup({complete:async x=>{input=x;return answer('You have eggs and spinach for breakfast.');}});
 const s=await r.service.message(ACTOR,{requestId:'quick-request',text:'What can I make for breakfast?'});
 await r.runner.run(pk,'quick-request');const saved=await r.service.session(ACTOR,s.id);
 assert.equal(input.model,'gpt-6.1-sol');assert.ok(input.input.some(x=>x.content.includes('current_inventory')));
 assert.equal(saved.status,'completed');assert.equal(saved.messages.length,2);assert.ok(saved.activity.every(x=>x.state==='completed'));
 assert.equal(saved.activity.filter(x=>x.stage==='thinking').length,1);
 assert.equal(saved.activity.find(x=>x.stage==='household').label,'Kitchen, list and preferences checked');
 assert.equal(r.gateway.writes,0);
});
test('tool results and encrypted reasoning continue; claims cannot bypass review',async()=>{
 let calls=0,last;
 const r=await setup({complete:async x=>{last=x;return calls++===0?tool('request_add_to_shopping_list',{item_name:'Milk',quantity:'1'}):answer('I added milk to your shopping list.');}});
 const s=await r.service.message(ACTOR,{requestId:'proposal-request',text:'Add milk to my shopping list'});
 await r.runner.run(pk,'proposal-request');const saved=await r.service.session(ACTOR,s.id);
 assert.equal(saved.status,'completed');assert.equal(saved.proposals.length,1);assert.equal(saved.proposals[0].status,'pending');
 assert.match(saved.messages.at(-1).text,/review/);assert.equal(r.gateway.writes,0);
 assert.ok(last.input.some(x=>x.type==='reasoning' && x.encrypted_content==='opaque'));
 assert.ok(last.input.some(x=>x.type==='function_call_output' && x.call_id==='call-one'));
 assert.equal(saved.metrics.inputTokens,150);assert.equal(saved.metrics.outputTokens,30);
});
test('unbacked action claim is never published as success',async()=>{
 const r=await setup({complete:async()=>answer('I removed the eggs from your kitchen.')});
 const s=await r.service.message(ACTOR,{requestId:'false-request',text:'Remove eggs from my kitchen'});
 await r.runner.run(pk,'false-request');const saved=await r.service.session(ACTOR,s.id);
 assert.match(saved.messages.at(-1).text,/could not confirm/);assert.equal(r.gateway.writes,0);
});
for(const [name,modelText] of [
 ['confirms an earlier approved change','I added eggs to your shopping list.'],
 ['never borrows an old receipt to confirm a different item','I added milk to your shopping list.'],
])test(name,async()=>{
 const r=await setup({complete:async()=>answer(modelText)});
 const s=await r.service.message(ACTOR,{requestId:'receipt-request',text:'Please confirm that the shopping-list change I just approved went through.'});
 await r.store.put(pk,key(s.id,'P','old-receipt'),{id:'old-receipt',status:'applied',action:'add_many_to_shopping_list',
  args:{items:[{item_name:'Eggs'}]},summary:'Add Eggs to the shopping list.',appliedAt:new Date(Date.now()-1000).toISOString(),
  afterHash:'verified-data-hash',verification:'Confirmed by reading your updated Trepo data.'});
 await r.store.put(pk,key(s.id,'P','aaa-older-receipt'),{id:'aaa-older-receipt',status:'applied',action:'add_many_to_shopping_list',
  args:{items:[{item_name:'Apples'}]},summary:'Add Apples to the shopping list.',appliedAt:new Date(Date.now()-60000).toISOString(),
  afterHash:'older-verified-data-hash',verification:'Confirmed by reading your updated Trepo data.'});
 await r.runner.run(pk,'receipt-request');const saved=await r.service.session(ACTOR,s.id);
 assert.match(saved.messages.at(-1).text,/earlier approved change was confirmed: Add Eggs/);
 assert.match(saved.messages.at(-1).text,/not applied any new account changes/);
 assert.doesNotMatch(saved.messages.at(-1).text,/milk|Apples|saved data is unchanged/);
 assert.equal(r.gateway.writes,0);
});
test('applied label without readback evidence cannot confirm a change',async()=>{
 const r=await setup({complete:async()=>answer('I added eggs to your shopping list.')});
 const s=await r.service.message(ACTOR,{requestId:'unverified-receipt',text:'Did my shopping-list change go through?'});
 await r.store.put(pk,key(s.id,'P','old-receipt'),{status:'applied',action:'add_many_to_shopping_list',summary:'Add Eggs'});
 await r.runner.run(pk,'unverified-receipt');const saved=await r.service.session(ACTOR,s.id);
 assert.match(saved.messages.at(-1).text,/could not confirm/);assert.equal(r.gateway.writes,0);
});
test('household change during inference suppresses the answer',async()=>{
 let r;r=await setup({complete:async()=>{r.gateway.actorValue={...ACTOR,household:'other'};return answer('Private answer');}});
 const s=await r.service.message(ACTOR,{requestId:'scope-request',text:'What do I have?'});
 await r.runner.run(pk,'scope-request');
 assert.equal((await r.store.list(pk,key(s.id,'M')+'#')).filter(x=>x.role==='assistant').length,0);
});
test('worker restart reuses persisted call IDs and receipts after tool checkpoint failure',async()=>{
 let calls=0;
 const r=await setup({complete:async()=>calls++===0?tool('create_recipe',{title:'Eggs',servings:1,ingredients:['2 eggs'],steps:['Cook eggs until set.']}):answer('Your recipe is ready.')});
 const s=await r.service.message(ACTOR,{requestId:'resume-request',text:'Make a recipe with eggs'});
 const put=r.store.put.bind(r.store);let crash=true;
 r.store.put=async(pk,sk,value,version)=>{if(crash && value.direct?.cursor===1){crash=false;throw new Error('checkpoint unavailable');}return put(pk,sk,value,version);};
 await assert.rejects(r.runner.run(pk,'resume-request'),/checkpoint unavailable/);
 const req=await r.store.get(pk,'Q#resume-request');assert.ok(req.direct.pending);assert.equal(req.direct.cursor,0);
 await put(pk,'Q#resume-request',{...req,status:'queued',leaseUntil:0},req.version);
 const session=await r.store.get(pk,key(s.id));await put(pk,key(s.id),{...session,status:'queued',activeRequest:req.id},session.version);
 await r.runner.run(pk,'resume-request');
 const recipes=await r.store.list(pk,key(s.id,'R')+'#');assert.equal(recipes.length,1);assert.equal(calls,2);assert.equal(r.gateway.writes,0);
});
test('resuming at a prior round budget continues with saved context instead of locking the chat',async()=>{
 let calls=0;const r=await setup({complete:async()=>{calls++;return answer('Your saved task is complete.');}});
 const s=await r.service.message(ACTOR,{requestId:'round-request',text:'What can I cook?'});
 const req=await r.store.get(pk,'Q#round-request');
 await r.store.put(pk,'Q#round-request',{...req,direct:{round:6,toolCalls:6,input:[{role:'user',content:'Saved task'}]}},req.version);
 await r.runner.run(pk,'round-request');
 assert.equal(calls,1);assert.equal((await r.service.session(ACTOR,s.id)).status,'completed');
});
test('rolling back the pilot rebuilds the legacy provider with all prior conversation turns',async()=>{
 const r=await setup({complete:async()=>answer('You have eggs for breakfast.')});
 const s=await r.service.message(ACTOR,{requestId:'first-response',text:'Breakfast with eggs'});await r.runner.run(pk,'first-response');
 r.gateway.env.THYME_RUNTIME_V1='0';let received;
 const provider={create:async x=>{received=JSON.parse(x.input);return {id:'new-provider'};},turns:async()=>[{id:'turn',status:'completed'}],session:async()=>({required_actions:[]}),items:async()=>answer('I kept your earlier breakfast context.').output};
 await r.service.message(ACTOR,{sessionId:s.id,requestId:'fallback-request',text:'Make that for two'});
 r.runner.provider=provider;await r.runner.run(pk,'fallback-request');
 assert.equal(received.conversation_history.length,2);assert.match(received.conversation_history[1].text,/eggs/);
 assert.equal((await r.service.session(ACTOR,s.id)).status,'completed');
});
test('throwing tool never shows a successful completed step during recovery',async()=>{
 const r=await setup({complete:async()=>tool('read_trepo',{resource:'saved_recipes'})});
 const s=await r.service.message(ACTOR,{requestId:'failed-tool-request',text:'Find my saved recipes'});
 r.gateway.failRead='saved_recipes';await assert.rejects(r.runner.run(pk,'failed-tool-request'),/read failure/);
 const saved=await r.service.session(ACTOR,s.id);
 assert.equal(saved.activity.find(x=>x.id==='call-one').state,'failed');
 assert.equal(saved.activity.find(x=>x.id==='call-one').label,'This step needs attention');
});
