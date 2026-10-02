import fs from 'node:fs';
import assert from 'node:assert/strict';
import {MemoryStore} from '../src/store.mjs';
import {Service} from '../src/service.mjs';
import {Runner} from '../src/runner.mjs';
import {ToolGateway,toolDefinitions} from '../src/tools.mjs';
import {AgentProvider} from '../src/provider.mjs';
import {readConversation} from '../src/conversation.mjs';
import {scope,id} from '../src/core.mjs';
import {FixtureGateway,ACTOR} from '../test/fixtures.mjs';
const cfg=JSON.parse(fs.readFileSync(process.env.THYME_PRIVATE_CONFIG));
const models=process.argv.slice(2).length?process.argv.slice(2):['gpt-6.1-sol','gpt-6-luna'];
const all=[];
const cases=[
 {name:'broad recipe then multi-answer and continuation',turns:[
  ['Can you suggest dinner?', 'ask'],
  ['Only what I have, for two, and no more than 20 minutes.', 'recipe', {inventory_mode:'existing_only',servings:2,max_minutes:20}],
  ['Another one please.', 'recipe', {inventory_mode:'existing_only',servings:2,max_minutes:20}],
  ['Actually shopping is fine now. Give me another dinner.', 'recipe',{inventory_mode:'shopping_ok',servings:2,max_minutes:20}],
 ]},
 {name:'yes is not an answer to an either-or question',turns:[['I want a recipe.','ask'],['Yes','ask'],['Just pick something.','recipe']]},
 {name:'ordinal choice remembered',turns:[['Help me choose dinner.','ask'],['The second one.','recipe',{inventory_mode:'shopping_ok'}]]},
 {name:'acknowledgment still remembers choices',turns:[
  ['Can you suggest dinner?','ask'],
  ['Shopping is fine, for three people, no more than 30 minutes. Just confirm these choices; do not create a recipe or change anything.','answer',{inventory_mode:'shopping_ok',servings:3,max_minutes:30}],
  ['What choices have I given you for this chat? Just remind me; do not create anything.','answer',{inventory_mode:'shopping_ok',servings:3,max_minutes:30}],
 ]},
 {name:'heldout trip to store',turns:[["I'm on my way to the grocery store. Any dinner ideas for four?",'answer',{inventory_mode:'shopping_ok',servings:4}]]},
 {name:'specific named dish is actionable',turns:[['Give me a white rice recipe.','recipe']]},
 {name:'explicit pantry request skips question',turns:[['Make a breakfast using only what I have for two.','recipe',{inventory_mode:'existing_only',servings:2}]]},
 {name:'shopping permission skips question',turns:[['I can shop. Give me a chicken dinner for four.','recipe',{inventory_mode:'shopping_ok',servings:4}]]},
 {name:'clear list action proceeds to review',turns:[['Add milk to my shopping list.','proposal']]},
 {name:'ambiguous destination asks',turns:[['Add milk.','ask']]},
 {name:'isolated voice fragment asks',turns:[['One yogurt and one raspberry.','ask']]},
 {name:'healthier needs a goal',turns:[['Give me a simple egg fried rice recipe.','recipe'],['Make it healthier.','ask']]},
 {name:'technique answers without recipe intake',turns:[['How do I stop rice sticking to the pan?','answer']]},
];
for(const model of models){
 for(const c of cases.filter(c=>!process.env.THYME_CASE_FILTER || c.name.includes(process.env.THYME_CASE_FILTER))){
  const store=new MemoryStore(),gateway=new FixtureGateway(),definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions}),provider=new AgentProvider(cfg.Environment.Variables.OPENAI_API_KEY),service=new Service({store,gateway});
  let sid;
  for(const [text,want,values] of c.turns){
   const previous=sid?await service.session(ACTOR,sid):null;
   const priorRecipes=previous?.recipes||[];
   const priorQuestion=previous?.messages.filter(m=>m.id.startsWith('clarify-')).at(-1);
   const rid=id(),start=Date.now();let entry={model,case:c.name,text,want};
   try{
    const s=await service.message(ACTOR,{requestId:rid,text,model,sessionId:sid});sid=s.id;
    // New runner/service reads durable context every turn, as after worker restart/reload.
    await new Runner({store,gateway,definitions,tools,provider}).run(scope(ACTOR),rid);
    const out=await new Service({store,gateway}).session(ACTOR,sid),context=await readConversation(store,ACTOR,sid);
    const answer=out.messages.filter(x=>x.role==='assistant'&&x.phase!=='commentary').at(-1)?.text;
    Object.assign(entry,{status:out.status,seconds:(Date.now()-start)/1000,answer,recipes:out.recipes,proposals:out.proposals,context,writes:gateway.writes});
    assert.equal(out.status,'completed',out.error);assert.equal(gateway.writes,0);
    if(want==='ask'){assert.equal(context.pending_question?.requestId,rid);assert.ok(answer?.includes('?'));assert.ok(answer.length<=350);assert.equal(out.proposals.length,0);if(c.turns[0][0]===text)assert.equal(out.recipes.length,0);}
    else assert.notEqual(context.pending_question?.requestId,rid,'Unnecessary clarification');
    if(want!=='ask' && priorQuestion) assert.equal(out.messages.filter(m=>m.role==='assistant'&&m.text===priorQuestion.text).length,1,'Earlier question rendered twice');
    if(want==='recipe') { const added=out.recipes.filter(r=>!priorRecipes.some(p=>p.id===r.id)); assert.ok(added.length>0,'No new recipe created'); if(values?.servings) for(const r of added) assert.equal(r.servings,values.servings); }
    if(want==='proposal') assert.equal(out.proposals.at(-1)?.status,'pending');
    if(values)for(const [k,v] of Object.entries(values)) assert.equal(context.values[k]?.value,v,k);
    entry.pass=true;
   }catch(e){entry.pass=false;entry.error=e.message;entry.providerDetail=e.providerDetail;}
   all.push(entry);fs.writeFileSync(process.env.THYME_EVIDENCE,JSON.stringify(all,null,2));console.log(JSON.stringify({model,case:c.name,want,pass:entry.pass,seconds:entry.seconds,answer:entry.answer,error:entry.error}));
   if(!entry.pass)break;
  }
 }
}
if(all.some(x=>!x.pass))process.exitCode=1;
