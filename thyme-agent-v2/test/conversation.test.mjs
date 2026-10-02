import test from 'node:test';
import assert from 'node:assert/strict';
import {MemoryStore} from '../src/store.mjs';
import {Service} from '../src/service.mjs';
import {Runner} from '../src/runner.mjs';
import {ToolGateway,toolDefinitions} from '../src/tools.mjs';
import {readConversation} from '../src/conversation.mjs';
import {scope,key,hash} from '../src/core.mjs';
import {INSTRUCTIONS} from '../src/instructions.mjs';
import {FixtureGateway,ACTOR} from './fixtures.mjs';
const pk=scope(ACTOR);
async function setup(text='Only what I have, for two, 20 minutes') {
 const store=new MemoryStore(),gateway=new FixtureGateway(),definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions}),service=new Service({store,gateway});
 const s=await service.message(ACTOR,{requestId:'context-request',text});const r=await store.get(pk,'Q#context-request');await store.put(pk,r.sk,{...r,status:'running'},r.version);
 return {store,gateway,definitions,tools,service,s};
}
const remember=(r,updates,call='remember')=>r.tools.call({actor:ACTOR,session:r.s,action:{name:'remember_conversation_preferences',arguments:{updates},call_id:call}});
const update=(field,value,source_quote)=>({field,value,source_quote});
test('explicit chat preferences survive store reconstruction and are actor/household/session scoped',async()=>{
 const r=await setup();const res=await remember(r,[update('inventory_mode','existing_only','Only what I have'),update('servings',2,'for two'),update('max_minutes',20,'20 minutes')]);assert.equal(res.ok,true);
 const restarted=new MemoryStore();restarted.items=structuredClone(r.store.items);
 const c=await readConversation(restarted,ACTOR,r.s.id);assert.equal(c.values.servings.value,2);assert.equal(c.values.max_minutes.value,20);
 for(const [a,sid] of [[{...ACTOR,actor:'other'},r.s.id],[{...ACTOR,household:'other'},r.s.id],[ACTOR,'other-chat']]) assert.deepEqual((await readConversation(restarted,a,sid)).values,{});
 assert.equal(r.gateway.writes,0);assert.deepEqual(r.gateway.data.preferences.dietary.allergies,['peanut']);
});
test('changed choice overrides one field without erasing others; explicit reset clears it',async()=>{
 const r=await setup('Only what I have for two. Actually I can shop. Forget the shopping choice.');
 await remember(r,[update('inventory_mode','existing_only','Only what I have'),update('servings',2,'for two')]);
 await remember(r,[update('inventory_mode','shopping_ok','I can shop')],'change');
 assert.equal((await readConversation(r.store,ACTOR,r.s.id)).values.inventory_mode.value,'shopping_ok');
 await remember(r,[update('inventory_mode',null,'Forget the shopping choice')],'clear');
 const c=await readConversation(r.store,ACTOR,r.s.id);assert.equal(c.values.inventory_mode,undefined);assert.equal(c.values.servings.value,2);
});
test('fabricated evidence, unsupported sensitive fields, mismatched types and duplicate fields are rejected atomically',async()=>{
 for(const updates of [
 [update('inventory_mode','shopping_ok','Invented quotation')],
 [update('allergies','either','Only what I have')],
 [update('servings','shopping_ok','for two')],
 [update('servings',41,'for two')],
 [update('inventory_mode',2,'for two')],
 [update('servings',2,'for two'),update('servings',3,'for two')],
 [update('servings',2,'for two'),update('max_minutes',20,'Not present')],
 ]) {const r=await setup();assert.equal((await remember(r,updates)).ok,false);assert.deepEqual((await readConversation(r.store,ACTOR,r.s.id)).values,{});}
});
test('retry after context write but before receipt does not duplicate or lose the preference',async()=>{
 const r=await setup();const put=r.store.put.bind(r.store);let fail=true;
 r.store.put=async(p,k,v,e)=>{if(k.includes('#T#')&&fail){fail=false;throw Error('simulated crash');}return put(p,k,v,e)};
 await assert.rejects(remember(r,[update('servings',2,'for two')]));
 const before=await r.store.get(pk,key(r.s.id,'C','preferences'));
 assert.equal((await remember(r,[update('servings',2,'for two')])).ok,true);
 assert.equal((await r.store.get(pk,before.sk)).version,before.version);
});
test('asking a question prevents same-turn recipe/proposal and final output stays the single recorded question',async()=>{
 const r=await setup('Give me a recipe');
 const q='Use your kitchen ingredients, or are you happy to shop?';
 assert.equal((await r.tools.call({actor:ACTOR,session:r.s,action:{name:'ask_clarifying_question',arguments:{question:q},call_id:'ask'}})).ok,true);
 for(const [name,args] of [['create_recipe',{title:'Eggs',servings:2,ingredients:['2 eggs'],steps:['Cook eggs until set.']}],['request_add_to_shopping_list',{item_name:'Milk'}]]){
 const out=await r.tools.call({actor:ACTOR,session:r.s,action:{name,arguments:args,call_id:name}});assert.equal(out.error,'awaiting_clarification');}
 await new Runner(r).saveMessages(pk,r.s,await r.store.get(pk,'Q#context-request'),[{type:'message',role:'assistant',id:'bad',content:[{type:'output_text',text:'I guessed a recipe anyway.'}]}],{});
 const s=await r.service.session(ACTOR,r.s.id);assert.equal(s.messages.at(-1).text,q);assert.equal(s.recipes.length,0);assert.equal(s.proposals.length,0);
});
test('session preference changes require a running current message, never an approval or another actor',async()=>{
 const r=await setup();const req=await r.store.get(pk,'Q#context-request');await r.store.put(pk,req.sk,{...req,kind:'approval'},req.version);
 assert.equal((await remember(r,[update('servings',2,'for two')])).error,'conversation_request');
});
test('a same-turn preference write cannot clear the question and reopen action tools',async()=>{
 const r=await setup('Give me a recipe for two');
 await r.tools.call({actor:ACTOR,session:r.s,action:{name:'ask_clarifying_question',arguments:{question:'Use your kitchen ingredients, or are you happy to shop?'},call_id:'ask'}});
 assert.equal((await remember(r,[update('servings',2,'for two')])).ok,true);
 assert.equal((await readConversation(r.store,ACTOR,r.s.id)).pending_question.requestId,'context-request');
 const result=await r.tools.call({actor:ACTOR,session:r.s,action:{name:'request_add_to_shopping_list',arguments:{item_name:'Milk'},call_id:'after-memory'}});
 assert.equal(result.error,'awaiting_clarification');
});
test('old idle provider session upgrades instructions/tools with full saved history and explicit context',async()=>{
 const r=await setup();let s=await r.store.get(pk,key(r.s.id));await r.store.put(pk,s.sk,{...s,providerId:'old-provider'},s.version);
 await r.store.put(pk,key(s.id,'M','old-answer'),{type:'message',id:'old-answer',role:'assistant',text:'Use kitchen only or shop?',order:1});
 await remember(r,[update('inventory_mode','existing_only','Only what I have')]);
 let body;
 const provider={turns:async(id)=>[{id:id==='old-provider'?'old-turn':'new-turn',status:'completed'}],create:async(b)=>{body=b;return{id:'new-provider'}},session:async()=>({required_actions:[]}),items:async()=>[]};
 const req=await r.store.get(pk,'Q#context-request');await r.store.put(pk,req.sk,{...req,status:'queued',leaseUntil:0},req.version);
 await new Runner({...r,provider}).run(pk,req.id);
 const input=JSON.parse(body.input);assert.equal(input.conversation_history[0].text,'Use kitchen only or shop?');assert.equal(input.conversation_preferences.values.inventory_mode.value,'existing_only');
 s=await r.store.get(pk,key(s.id));assert.equal(s.providerId,'new-provider');assert.equal(s.agentVersion,hash([INSTRUCTIONS,r.definitions]));assert.equal(s.lastTurnId,'new-turn');
});
test('a stale worker cannot write preferences after its session moved on',async()=>{
 const r=await setup();const current=await r.store.get(pk,key(r.s.id));await r.store.put(pk,current.sk,{...current,activeRequest:'different-request'},current.version);
 assert.equal((await remember(r,[update('servings',2,'for two')])).error,'conversation_request');
});
test('concurrent preference writes are revision fenced, rather than losing an answer silently',async()=>{
 const r=await setup();
 const results=await Promise.allSettled([remember(r,[update('servings',2,'for two')],'one'),remember(r,[update('max_minutes',20,'20 minutes')],'two')]);
 const succeeded=results.filter(x=>x.status==='fulfilled'&&x.value.ok);assert.equal(succeeded.length,1);
 const failed=results.find(x=>x.status==='fulfilled'&&!x.value.ok);assert.equal(failed.value.error,'conflict');
});
test('old in-flight provider turns are never migrated or silently replayed',async()=>{
 const r=await setup(),s=await r.store.get(pk,key(r.s.id));await r.store.put(pk,s.sk,{...s,providerId:'old-provider'},s.version);
 const req=await r.store.get(pk,'Q#context-request');await r.store.put(pk,req.sk,{...req,status:'queued',leaseUntil:0},req.version);
 let created=false;
 const provider={turns:async()=>[{id:'old-turn',status:'running'}],create:async()=>{created=true}};
 await new Runner({...r,provider}).run(pk,req.id);
 assert.equal(created,false);assert.equal((await r.service.session(ACTOR,s.id)).status,'interrupted');assert.equal((await r.store.get(pk,s.sk)).providerId,'old-provider');
});
test('large migration history is persisted once alongside input, not twice above DynamoDB limit',async()=>{
 const r=await setup(),s=await r.store.get(pk,key(r.s.id));await r.store.put(pk,s.sk,{...s,providerId:'old-provider'},s.version);
 await r.store.put(pk,key(s.id,'M','long-history'),{type:'message',id:'long-history',role:'assistant',text:'x'.repeat(170000),order:1});
 const req=await r.store.get(pk,'Q#context-request');await r.store.put(pk,req.sk,{...req,status:'queued',leaseUntil:0},req.version);
 const provider={turns:async(id)=>[{id:id==='old-provider'?'old-turn':'new-turn',status:'completed'}],create:async()=>({id:'new-provider'}),session:async()=>({required_actions:[]}),items:async()=>[]};
 await new Runner({...r,provider}).run(pk,req.id);
 const saved=await r.store.get(pk,req.sk);assert.equal(saved.status,'completed');assert.equal(saved.upgradeHistory,undefined);assert.equal(JSON.parse(saved.submittedInput).conversation_history[0].text.length,170000);
});
