import {test,mock,before} from 'node:test';
import assert from 'node:assert/strict';
let assistant,queue,requests,writes,toolResult;
const call=(name,args={})=>({content:null,tool_calls:[{index:0,id:'fixture-call',type:'function',function:{name,arguments:JSON.stringify(args)}}]});
before(async()=>{
 const url=new URL('../lib/data-access.mjs',import.meta.url),data=await import(url);
 mock.module(url,{namedExports:{...data,getDietaryPreferences:async()=>({allergies:[],diets:[],religious:[],health:[],custom:[]})}});
 mock.module(new URL('../lib/tool-actions.mjs',import.meta.url),{namedExports:{executeToolAction:async input=>{writes.push(input);return {...toolResult,args:input.args,toolName:input.toolName};}}});
 globalThis.fetch=async(url,options)=>{
  const request=JSON.parse(options.body);requests.push(request);
  const content=queue.shift();assert.notEqual(content,undefined,'Unexpected provider retry');
  const message=content?.tool_calls?content:{role:'assistant',content};
  return new Response(request.stream?`data: ${JSON.stringify({choices:[{delta:message}]})}\n\ndata: [DONE]\n\n`:JSON.stringify({choices:[{message}]}));
 };
 assistant=await import('../lib/device-assistant.mjs');
});
async function run(surface,transcript,overrides={}) {
 const input={transcript,userContext:{ownerId:'fixture',userId:'fixture'},env:{ACTION_MODE:'mock',OPENAI_API_KEY:'fixture'},responseSurface:'app',...overrides};
 if(surface==='sync')return assistant.runDeviceAssistant(input);
 const stream=assistant.runDeviceAssistantStreaming(input);
 while(true){const next=await stream.next();if(next.done)return next.value;}
}
function reset(completions,result){queue=completions;requests=[];writes=[];toolResult=result;}
for(const surface of ['sync','stream']) {
 test(surface+' retains historical/help answers without writes',async()=>{
  const answer='Items you added to your shopping list appear in List.';reset([answer]);
  const result=await run(surface,'Where do items I already added show up?');
  assert.equal(result.text,answer);assert.equal(writes.length,0);assert.equal(requests.length,1);
 });
 test(surface+' unsupported claim cannot authorize a corrective mutation even for an original add request',async()=>{
  reset(['Added milk to your shopping list.',call('add_to_shopping_list',{item_name:'Milk'})]);
  const result=await run(surface,'Add milk to my shopping list');
  assert.equal(writes.length,0);assert.equal(requests.length,2);assert.equal(requests[1].tool_choice,'none');assert.match(result.text,/haven't made/);
 });
 test(surface+' genuine authorized action executes once and confirms actual returned amount',async()=>{
  reset([call('add_to_shopping_list',{item_name:'Milk',quantity:'1 carton'}),'Added five loaves of bread to your shopping list.'],{ok:true,statusCode:200,toolResult:{item:{shopping_id:'a',item_name:'Milk',quantity:'1 carton'}}});
  const result=await run(surface,'Add one carton of milk to my shopping list');
  assert.equal(writes.length,1);assert.equal(requests.length,2);assert.match(result.text,/Milk \(1 carton\)/);assert.doesNotMatch(result.text,/bread|five/);
  assert.equal(writes[0].sourceTranscript,'Add one carton of milk to my shopping list');
 });
 test(surface+' 404 is not disguised by a successful-sounding answer',async()=>{
  reset([call('discard_item',{item_id:'a',item_name:'Milk'}),'I removed milk from your kitchen.'],{ok:false,statusCode:404,error:'not_found'});
  const result=await run(surface,'Remove milk from my kitchen');
  assert.equal(writes.length,1);assert.equal(requests.length,2);assert.doesNotMatch(result.text,/removed milk/i);assert.match(result.text,/couldn't/);
 });
 test(surface+' false read-only answer gets useful text-only correction',async()=>{
  reset(['Logged your breakfast.','A large egg has about 6 grams of protein.']);
  const result=await run(surface,'How much protein is in an egg?');
  assert.equal(writes.length,0);assert.equal(requests[1].tool_choice,'none');assert.match(result.text,/6 grams/);
 });
}
for(const surface of ['sync','stream']) {
 const aisleArgs={item_id:'item-fixture-123',item_name:'Paper towels',aisle_key:'custom:toiletries'};
 test(surface+' confirms the actual aisle assignment instead of provider-invented names',async()=>{
  reset([call('set_shopping_item_aisle',aisleArgs),'I moved your milk into Produce.'],{ok:true,statusCode:200,toolResult:{item:{id:aisleArgs.item_id,item_name:'Paper towels',aisle_key:aisleArgs.aisle_key,aisle_label:'Toiletries'}}});
  const result=await run(surface,'Move the paper towels into the Toiletries aisle');
  assert.equal(writes.length,1);assert.equal(writes[0].sourceTranscript,'Move the paper towels into the Toiletries aisle');
  assert.match(result.text,/Paper towels.*Toiletries/);assert.doesNotMatch(result.text,/milk|Produce/);
 });
 test(surface+' failed aisle mutation cannot become a successful confirmation',async()=>{
  reset([call('set_shopping_item_aisle',aisleArgs),'I moved Paper towels into Toiletries.'],{ok:false,statusCode:409,error:'stale_revision'});
  const result=await run(surface,'Move the paper towels into the Toiletries aisle');
  assert.equal(writes.length,1);assert.match(result.text,/couldn't|could not|not confirmed/i);assert.doesNotMatch(result.text,/I moved/);
 });
 test(surface+' aisle help remains read-only even if the provider requests a write',async()=>{
  reset([call('set_shopping_item_aisle',aisleArgs),'Open List, then Manage aisles.']);
  const result=await run(surface,'How can I move paper towels into an aisle?');
  assert.equal(writes.length,0);assert.match(result.text,/List/);
 });
}
