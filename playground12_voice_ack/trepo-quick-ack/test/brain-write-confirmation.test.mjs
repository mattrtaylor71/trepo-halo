import {buildAppOutput} from '../lib/app-response.mjs';
import {test,before,mock} from 'node:test';
import assert from 'node:assert/strict';
let da,queue,executed,revision,scope,outage,requests,markers,failTool,fullOnly,receiptReads;
const LIB=new URL('../lib/',import.meta.url).href;
const tool=(name,args)=>({id:'call_'+Math.random(),type:'function',function:{name,arguments:JSON.stringify(args)}});
const msg=(calls=[],text='')=>({role:'assistant',content:text,tool_calls:calls});
const dish=name=>tool('log_dish_ingredients',{dish_name:name,ingredients:[name]});
const brain=()=>({prompt:'CURRENT_CONTEXT_'+revision,fence:{schema:'household-brain-v2',actor:'test-actor',household:'test-home',revision}});
const input=()=>({transcript:'Log oatmeal and coffee as two separate dishes',userContext:{ownerId:'test-home',userId:'test-actor'},
  responseSurface:'app',env:{ACTION_MODE:'mock',OPENAI_API_KEY:'test',HOUSEHOLD_BRAIN_V2:'1'},
  kitchenRequest:{operationId:'synthetic-operation',preparedBrain:{brain:brain(),credentials:{owner:'test-actor'}}},
  sessionMessages:[{role:'assistant',content:'OLD_PRIVATE_CONTEXT'}]});
function reset(){queue=[];executed=[];revision=1;scope=true;outage=false;requests=[];markers=[];failTool=false;fullOnly=false;receiptReads=0;}
before(async()=>{
 const data=await import(LIB+'data-access.mjs');mock.module(LIB+'data-access.mjs',{namedExports:{...data,getDietaryPreferences:async()=>({allergies:[],diets:[],religious:[],health:[],custom:[]})}});
 const real=await import(LIB+'family-brain.mjs');mock.module(LIB+'family-brain.mjs',{namedExports:{...real,
   validateBrain:async b=>!outage&&scope&&b.fence.revision===revision,
   callBrain:async req=>{if(outage)return null;if(req.operation==='validate_receipt'){receiptReads++;return {current:scope};} if(req.operation==='refresh_turn')return scope?{current:true,...brain()}:{current:false};throw Error('unexpected op');}}});
 mock.module(LIB+'tool-actions.mjs',{namedExports:{executeToolAction:async({toolName,args})=>{
   executed.push({toolName,args});revision++;
   if(failTool)return {ok:false,statusCode:504,toolName,error:'timeout'};
   return {ok:true,statusCode:200,toolName,args,toolResult:{ok:true,id:'saved-'+executed.length,dish_name:args.dish_name || args.item_name}};
 }}});
 const oldLog=console.log;console.log=(...args)=>{for(const a of args){try{const v=JSON.parse(a);if(v.evt==='ai_op')markers?.push(v);}catch{}}};
 globalThis.fetch=async(url,opt)=>{
   assert.match(String(url),/api.openai.com/);const body=JSON.parse(opt.body);requests.push(body);
   let m=queue.shift() || msg([],'All done.');if(typeof m==='function')m=m();
   if(body.stream){const delta=m.tool_calls.length?{tool_calls:m.tool_calls.map((x,index)=>({...x,index}))}:{content:m.content};
     return new Response('data: '+JSON.stringify({choices:[{delta}]})+'\n\ndata: [DONE]\n\n',{status:200});}
   return new Response(JSON.stringify({choices:[{message:m}]}),{status:200,headers:{'content-type':'application/json'}});
 };
 da=await import(LIB+'device-assistant.mjs');
});
async function run(stream=false,i=input()){
 if(!stream)return da.runDeviceAssistant(i);
 const it=da.runDeviceAssistantStreaming(i),events=[];let s;do{s=await it.next();if(!s.done)events.push(s.value);}while(!s.done);
 assert.equal(events.filter(e=>e.type==='text_delta').map(e=>e.delta).join(''),s.value.text,'only final validated text reaches stream');
 return s.value;
}
for(const streaming of [false,true]){
 test(`${streaming?'stream':'sync'} single saved dish survives its own input revision`,async()=>{
   reset();queue=[msg([dish('Oatmeal')]),msg([],'Logged oatmeal. Also OLD_PRIVATE_CONTEXT')];
   const r=await run(streaming);assert.match(r.text,/Oatmeal/);assert.doesNotMatch(r.text,/preferences changed|OLD_PRIVATE_CONTEXT/);
   assert.equal(r.outcome.actions,'confirmed');assert.equal(r.outcome.answer,'complete');assert.equal(r.toolEvents.length,1);assert.ok(r.ui);assert.equal(executed.length,1);
   const app=buildAppOutput({...r,responseSurface:'app'});
   assert.equal(app.actions.length,1);assert.equal(app.actions[0].ok,true);assert.equal(app.cards.length,1);
   assert.match(app.message.text,/Oatmeal/);assert.doesNotMatch(JSON.stringify(app),/OLD_PRIVATE_CONTEXT/);
   assert.equal(markers.filter(m=>m.op==='chat_app').length,1);assert.equal(markers.at(-1).status,'success');assert.match(markers.at(-1).output,/Oatmeal/);
 });
 test(`${streaming?'stream':'sync'} second dish replans with fresh context without replaying first`,async()=>{
   reset();queue=[msg([dish('Oatmeal'),dish('Coffee')]),msg([dish('Oatmeal'),dish('Coffee')]),msg([],'Logged both.')];
   const r=await run(streaming);assert.deepEqual(executed.map(x=>x.args.dish_name),['Oatmeal','Coffee']);assert.match(r.text,/Oatmeal/);assert.match(r.text,/Coffee/);assert.equal(r.toolEvents.length,2);
   assert.ok(requests[1].messages.some(x=>x.content==='CURRENT_CONTEXT_2'));
   assert.ok(!JSON.stringify(requests[1].messages).includes('OLD_PRIVATE_CONTEXT'));
 });
 test(`${streaming?'stream':'sync'} genuine household or dietary change hides receipts and blocks next write`,async()=>{
   reset();queue=[msg([dish('Oatmeal')]),()=>{scope=false;return msg([dish('Coffee')]);}];
   const r=await run(streaming);assert.equal(executed.length,1);assert.doesNotMatch(r.text,/Oatmeal|Coffee|preferences changed/);assert.equal(r.ui,null);assert.equal(r.outcome.answer,'incomplete');assert.equal(markers.at(-1).status,'error');
 });
 test(`${streaming?'stream':'sync'} validation outage never invites a blind duplicate`,async()=>{
   reset();queue=[msg([dish('Oatmeal')]),()=>{outage=true;return msg([],'Logged oatmeal.');}];
   const r=await run(streaming);assert.equal(executed.length,1);assert.equal(r.outcome.reason,'context_unavailable');assert.match(r.text,/before repeating/);assert.doesNotMatch(r.text,/food preferences changed/);
 });
}
test('ordinary read answer still requires the full content freshness fence',async()=>{
 reset();queue=[()=>{revision++;return msg([],'REVOKED_EVIDENCE');}];const r=await run();assert.doesNotMatch(r.text,/REVOKED/);assert.equal(r.outcome.answer,'incomplete');
});
test('uncertain timeout is not replayed after context replan',async()=>{
 reset();failTool=true;queue=[msg([dish('Oatmeal'),dish('Coffee')]),msg([dish('Oatmeal')]),msg([],'Could not confirm.')];
 const r=await run();assert.equal(executed.length,1);assert.equal(r.outcome.actions,'unconfirmed');
});
test('existing current read response remains available',async()=>{reset();queue=[msg([],'Here are your saved recipes.')];const r=await run();assert.match(r.text,/saved recipes/);assert.equal(receiptReads,0);});

test('shopping list receipt survives the same activity revision change',async()=>{
 reset();queue=[msg([tool('add_to_shopping_list',{item_name:'Bananas'})]),msg([],'Added bananas.')];
 const i=input();i.transcript='Add bananas to my shopping list';const r=await run(false,i);
 assert.match(r.text,/Bananas/);assert.equal(r.outcome.actions,'confirmed');assert.equal(executed.length,1);
});
