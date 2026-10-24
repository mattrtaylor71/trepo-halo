// Synthetic only: no customer reads or writes. Deliberately bounded model probe.
import {ResponsesClient} from '../../shared/thyme-runtime/responses.mjs';
import {QUICK_INSTRUCTIONS} from '../../shared/thyme-runtime/policy.mjs';
const scenarios=[
 {name:'breakfast',text:'What can I make for breakfast?',check:r=>/egg|spinach/i.test(JSON.stringify(r.output))},
 {name:'condiments',text:'What condiments do I have? List only my actual condiments.',check:r=>/mustard/i.test(JSON.stringify(r.output)) && !/chicken soup/i.test(JSON.stringify(r.output))},
 {name:'proposal',text:'Add milk to my shopping list.',check:r=>r.output.some(x=>x.type==='function_call' && x.name==='request_add_to_shopping_list')},
];
const tools=[{type:'function',name:'request_add_to_shopping_list',description:'Prepare a shopping-list change for user review. Does NOT apply the change.',parameters:{type:'object',properties:{item_name:{type:'string'}},required:['item_name'],additionalProperties:false}}];
const models=(process.env.THYME_CANARY_MODELS||'gpt-6-luna,gpt-6.1-sol').split(',');
for(const model of models)for(const scenario of scenarios){
 const start=Date.now();
 try{
  const client=new ResponsesClient(process.env.OPENAI_API_KEY);
  const result=await client.complete({model,deadline:Date.now()+30000,maxOutputTokens:1600,tools,
   input:[{role:'system',content:QUICK_INSTRUCTIONS+'\nA request_* tool prepares review cards; do not claim the account changed.'},
    {role:'user',content:JSON.stringify({current_inventory:{complete:true,kitchen:[{id:'egg',item_name:'Eggs',quantity:6},{id:'spinach',item_name:'Spinach'},{id:'mustard',item_name:'Dijon mustard'},{id:'soup',item_name:'Chicken soup'}],shopping:[]},current_preferences:{allergies:['peanuts']},user_message:scenario.text})}]});
  console.log(JSON.stringify({model,scenario:scenario.name,elapsed_ms:Date.now()-start,pass:scenario.check(result),usage:result.usage,output:result.output}));
 }catch(e){console.log(JSON.stringify({model,scenario:scenario.name,elapsed_ms:Date.now()-start,error:e.code||e.name,status:e.providerStatus}));}
}
