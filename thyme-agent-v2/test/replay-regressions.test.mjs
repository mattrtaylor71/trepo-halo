import test from 'node:test';
import assert from 'node:assert/strict';
import {pageData} from '../src/prepare-actions.mjs';
import {compactInventory} from '../src/runner.mjs';
import {detectActionClaims} from '../../playground12_voice_ack/trepo-quick-ack/lib/action-claims.mjs';
import {SnapshotGateway} from '../../scripts/thyme/replay-conversations.mjs';
import {FixtureGateway,ACTOR} from './fixtures.mjs';
import {MemoryStore} from '../src/store.mjs';
import {ToolGateway,toolDefinitions} from '../src/tools.mjs';

test('saved-recipe retrieval is not an account-change confirmation',()=>{
 for(const text of ['Yes—of the saved recipes I checked, these are the best fits for your red onion.',
   'I checked your saved recipes and found three dinners.', 'That is an addition, not part of the saved recipe.'])
   assert.deepEqual(detectActionClaims(text,{memoryContext:true}),[],text);
 for(const text of ['I saved your recipe.','Your recipe has been saved.','Saved the recipe to your library.','I checked your kitchen and saved the recipe.'])
   assert.ok(detectActionClaims(text,{memoryContext:true}).length,text);
});

test('explicit no-change narration is not rewritten as a failed mutation',()=>{
 for(const memoryContext of [false,true]) {
   for(const text of ['Nothing has been added to your calendar.',
     'These are ideas rather than scheduled meals—nothing has been added to your lists or calendar.',
     'No recipes have been saved to your library.'])
     assert.deepEqual(detectActionClaims(text,{memoryContext}),[],text);
   assert.ok(detectActionClaims('Nothing has been added to your calendar, but I saved your recipe.',{memoryContext}).length);
   assert.ok(detectActionClaims('Nothing else was needed; I added milk to your shopping list.',{memoryContext}).length);
 }
});

test('large saved libraries have bounded ingredient-searchable pages and exact continuation',()=>{
 const recipes=Array.from({length:180},(_,i)=>({id:String(i),title:'Dinner '+i,ingredients:['1 red onion','2 potatoes'],
   instructions:['Cook until tender.'],caption:'x'.repeat(20000),content:'x'.repeat(20000),
   notes:['Check the source.'],saved_by:'household-member',saved_by_name:'Member',status:'ready'}));
 const seen=[];let offset=0;
 do {
   const page=pageData({recipes},{offset,limit:100,query:'onion'},'saved_recipes');
   assert.equal(page.count,180);assert.ok(Buffer.byteLength(JSON.stringify(page))<37000);
   assert.ok(page.items.every(x=>x.saved_by==='household-member'&&!x.caption&&!x.content&&!x.instructions));
   seen.push(...page.items.map(x=>x.id));offset=page.next_offset;
 }while(offset!==null);
 assert.deepEqual(seen,recipes.map(x=>x.id));
 assert.equal(pageData({recipes},{query:'garlic'},'saved_recipes').count,0);
});

test('large kitchens retain every item without importing storage essays into the prompt',()=>{
 const kitchen=Array.from({length:210},(_,i)=>({id:String(i),item_name:'Food '+i,quantity_value:null,quantity_unit:'count',storage_location:'fridge',
   ingredients:['ingredient '.repeat(100)],storage_guidance:'x'.repeat(20000),expiration_date:'2026-12-01'}));
 const snapshot=compactInventory(kitchen,[],'2026-10-24');
 assert.equal(snapshot.complete,true);assert.equal(snapshot.kitchen.length,210);
 assert.ok(!JSON.stringify(snapshot).includes('storage_guidance'));assert.ok(!('quantity_value' in snapshot.kitchen[0]));
 const page=pageData(kitchen,{limit:100},'kitchen');
 assert.ok(page.items.length<100);assert.equal(page.next_offset,page.items.length);assert.equal(page.count,210);
});

test('bounded summaries still allow explicit product and storage detail reads',()=>{
 const row={id:'milk',item_name:'Milk',storage_guidance:'Keep refrigerated',created_at:'2026-10-24',
   estimated_price:4,ingredients:['Milk'],nutrition_summary:{protein_g:8},s3_key:'internal-upload'};
 assert.equal(pageData([row],{},'kitchen').items[0].storage_guidance,undefined);
 const detailed=pageData([row],{query:'milk',detail:true,limit:1},'kitchen').items[0];
 assert.equal(detailed.storage_guidance,row.storage_guidance);assert.equal(detailed.created_at,row.created_at);
 assert.equal(detailed.estimated_price,4);assert.equal(detailed.s3_key,undefined);
 const recipe={id:'r',title:'Dinner',ingredients:['Rice'],instructions:['Boil the rice.'],source_url:'https://example.com/recipe'};
 assert.deepEqual(pageData([recipe],{detail:true},'saved_recipes').items[0],recipe);
});

test('replay cannot write, cross households, or turn a missing snapshot into empty data',async()=>{
 const fixture=new FixtureGateway(),g=new SnapshotGateway({actor:ACTOR,data:fixture.data});
 await assert.rejects(g.mutate(),/REPLAY_WRITE_FORBIDDEN/);
 await assert.rejects(g.read({...ACTOR,household:'other'},'kitchen'),{code:'membership'});
 await assert.rejects(g.read(ACTOR,'web_recipes'),{code:'snapshot_unavailable'});
 assert.equal((await g.history(ACTOR,{kind:'purchases'})).ok,false);
});

test('history success is recognized only for the current actor and household',async()=>{
 const gateway=new FixtureGateway();gateway.env={HOUSEHOLD_BRAIN_V2:'1'};
 const evidence={schema:'household-history-v1',actor:ACTOR.actor,household:ACTOR.household,records:[{id:'receipt-one'}]};
 let response=evidence;gateway.history=async()=>response;
 const tools=new ToolGateway({store:new MemoryStore(),gateway,definitions:await toolDefinitions(gateway)});
 const call={name:'lookup_household_history',arguments:{kind:'purchases'}};
 const run=()=>tools.perform(ACTOR,{id:'history-session'},call,{revision:1});
 assert.deepEqual(await run(),{...evidence,ok:true});
 for(const bad of [{...evidence,actor:'another-user'},{...evidence,household:'another-home'},
   {...evidence,schema:'unknown'}, {...evidence,records:null}]) {
   response=bad;assert.equal((await run()).error,'history_unavailable');
 }
 response={ok:false,error:'consent_changed'};assert.deepEqual(await run(),response);
 assert.equal((await tools.perform(ACTOR,{id:'history-session'},call)).error,'history_unavailable');
});
