import fs from "node:fs";
import assert from "node:assert/strict";
import {MemoryStore} from "../src/store.mjs";
import {Service} from "../src/service.mjs";
import {Runner} from "../src/runner.mjs";
import {ToolGateway,toolDefinitions} from "../src/tools.mjs";
import {AgentProvider} from "../src/provider.mjs";
import {scope,id} from "../src/core.mjs";
import {FixtureGateway,ACTOR} from "../test/fixtures.mjs";
const cfg=JSON.parse(fs.readFileSync(process.env.THYME_PRIVATE_CONFIG));
const model=process.argv[2]||"gpt-6.1-sol",out=[];
const cases=[
 ["read list","What's on my shopping list? Don't change anything.",[]],
 ["create list","Create a Costco shopping list with milk and apples.",["create_shopping_list","add_many_to_shopping_list"]],
 ["remove list","Remove the Groceries shopping list and its items, leaving other lists alone.",["remove_shopping_list"]],
 ["quantity","Set the eggs in my kitchen to 4 count.",["update_item_quantity","update_item_details"]],
 ["check in","Check 3 count of apples into my kitchen, in the fridge, sealed.",["check_in_item","check_in_many_items"]],
 ["kitchen edit","Rename Spinach in my kitchen to Baby spinach. Keep every other detail the same.",["update_item_details"]],
 ["kitchen remove","Remove the Rice from my kitchen.",["discard_item"]],
 ["saved edit","Change the title of my saved Egg toast recipe to Quick egg toast. Do not create a new copy or change its ingredients or instructions.",["edit_saved_recipe"]],
 ["dish log","Log Egg toast in my Dish Log: 2 eggs, 1 slice bread, one serving. Leave nutrition unknown.",["log_dish_from_voice","log_dish_ingredients"]],
 ["dish edit","Change the serving size of the Yogurt in my Dish Log to 2 servings, leaving everything else unchanged.",["update_dish","update_recent_dish"]],
 ["saved to calendar","Put my saved Egg toast recipe in the meal plan calendar for breakfast on 2026-10-05.",["add_recipe_to_meal_calendar","add_generated_recipes_to_meal_calendar","add_many_to_meal_calendar"]],
 ["recipe groceries","Add all ingredients from my saved Egg toast recipe to my shopping list.",["add_saved_recipe_ingredients_to_shopping_list","add_many_to_shopping_list"]],
 ["full meal plan","Create a two-day dinner plan for two people for 2026-10-06 and 2026-10-07, and store both dinners in my app meal calendar. Shopping is fine. I want the complete recipes in the meal plan.",["add_generated_recipes_to_meal_calendar"]],
];
for(const [label,text,actions] of cases.filter(c=>!process.env.THYME_CASES || process.env.THYME_CASES.split(",").includes(c[0]))) {
 const store=new MemoryStore(),gateway=new FixtureGateway();
 gateway.data.saved_recipes=[{id:"saved-eggs",title:"Egg toast",ingredients:["2 eggs","1 slice bread"],instructions:["Cook eggs until set.","Serve on toasted bread."],notes:["Serves 1"],status:"ready"}];
 gateway.data.dishes=[{id:"dish-yogurt",dish_name:"Yogurt",serving_size:"1 serving",ingredients:["Yogurt"],action:"IN",updated_at:new Date().toISOString()}];
 const definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions}),provider=new AgentProvider(cfg.Environment.Variables.OPENAI_API_KEY),service=new Service({store,gateway});
 const rid=id(),started=Date.now();let record={model,label};
 try {
  const session=await service.message(ACTOR,{requestId:rid,text,model});
  const runner=new Runner({store,gateway,definitions,tools,provider});record.redeliveries=0;
  for(let attempt=0;attempt<3;attempt++) {
    try {await runner.run(scope(ACTOR),rid);break;} catch(e) {
      const durable=await store.get(scope(ACTOR),"Q#"+rid);
      if(durable?.status!=="retrying" || attempt===2) throw e;
      record.redeliveries++;await new Promise(r=>setTimeout(r,2000*(attempt+1)));
    }
  }
  const result=await service.session(ACTOR,session.id);
  const proposals=(await store.list(scope(ACTOR),"S#"+session.id+"#P#"));
  Object.assign(record,{status:result.status,seconds:(Date.now()-started)/1000,proposals:proposals.map(p=>({action:p.action,args:p.args})),answer:result.messages.filter(m=>m.role==="assistant"&&m.phase!=="commentary").at(-1)?.text});
  assert.equal(result.status,"completed",result.error);assert.equal(gateway.writes,0);
  if(actions.length) assert(proposals.some(p=>actions.includes(p.action)),"No matching action proposal");else assert.equal(proposals.length,0);
  if(label==="saved edit") assert.deepEqual(Object.keys(proposals.at(-1).args).sort(),["recipe_id","title"]);
  if(label==="dish log") assert(!["calories","protein_g","carbs_g","fat_g"].some(k=>proposals.at(-1).args[k]!==undefined));
  if(label==="full meal plan") assert.equal(proposals.flatMap(p=>p.args.entries||[]).length,2);
  record.pass=true;
 }catch(e){record.pass=false;record.error=e.message;record.providerDetail=e.providerDetail;}
 out.push(record);fs.writeFileSync(process.env.THYME_EVIDENCE,JSON.stringify(out,null,2));console.log(JSON.stringify({model,label,pass:record.pass,seconds:record.seconds,actions:record.proposals?.map(p=>p.action),error:record.error}));
}
if(out.some(r=>!r.pass))process.exitCode=1;
