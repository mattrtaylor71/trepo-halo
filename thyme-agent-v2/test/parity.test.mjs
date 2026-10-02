import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {catalog,READ_TOOLS} from "../src/capabilities.mjs";
import {ACTIONS,TrepoGateway,resourceFor} from "../src/gateway.mjs";
import {prepareAction} from "../src/prepare-actions.mjs";
import {verifyChange} from "../src/verification.mjs";
import {FixtureGateway,ACTOR} from "./fixtures.mjs";
const legacy=JSON.parse(fs.readFileSync(new URL("../vendor/tool-definitions.json",import.meta.url)));
test("all 57 original tools have a discoverable read or review-gated equivalent",()=>{
 const defs=catalog(legacy,ACTIONS);assert.equal(legacy.length,57);
 for(const t of legacy) assert(defs.some(d=>d.name===(READ_TOOLS[t.name]?t.name:"request_"+t.name)),t.name);
 assert.equal(new Set(defs.map(d=>d.name)).size,defs.length);
});
test("resource routing keeps ingredient additions on shopping and category assignments on categories",()=>{
 for(const n of ["add_dish_ingredients_to_shopping_list","add_saved_recipe_ingredients_to_shopping_list","create_shopping_list"]) assert.equal(resourceFor(n),"shopping");
 assert.equal(resourceFor("move_recipe_to_category"),"recipe_categories");assert.equal(resourceFor("delete_recent_discard"),"discards");
});
test("ID disambiguates duplicate shopping names; names alone and foreign IDs fail",async()=>{
 const g=new FixtureGateway(),before=[{shopping_id:"1",item_name:"Milk"},{shopping_id:"2",item_name:"Milk"}];
 const a={item_id:"2"};await prepareAction(g,ACTOR,"remove_from_shopping_list",a,before);assert.equal(a.item_id,"2");
 for(const args of [{item_name:"Milk"},{item_name:"Mil"},{item_id:"foreign",item_name:"Milk"}]) await assert.rejects(prepareAction(g,ACTOR,"remove_from_shopping_list",args,before),{code:"ambiguous_item"});
});
test("list remove/rename freezes all and only the requested store",async()=>{
 const g=new FixtureGateway(),before=[{shopping_id:"1",item_name:"Milk",store:"Costco"},{shopping_id:"2",item_name:"Eggs",store:"Groceries"}];
 for(const name of ["remove_shopping_list","rename_shopping_list"]) {
  const args={store:"Costco",new_name:"Weekend"};await prepareAction(g,ACTOR,name,args,before);assert.equal(args.targets.length,1);assert.equal(args.targets[0].item_id,"1");
 }
});
test("saved-recipe ingredients and calendar freeze the reviewed recipe, never fetch it during approval",async()=>{
 const g=new FixtureGateway();g.data.saved_recipes=[{id:"recipe",title:"Rice",ingredients:["1 cup rice"],instructions:["Cook rice."],notes:["Serves 2"]}];
 const args={recipe_title:"Rice"};await prepareAction(g,ACTOR,"add_saved_recipe_ingredients_to_shopping_list",args,[]);assert.deepEqual(args.items,[{item_name:"1 cup rice"}]);
 const meal={recipe_id:"recipe",plan_date:"2026-10-05",meal_slot:"dinner"};await prepareAction(g,ACTOR,"add_recipe_to_meal_calendar",meal,[]);
 g.data.saved_recipes[0].ingredients=["9 cups rice"];assert.deepEqual(meal.entries[0].ingredients,["1 cup rice"]);
 assert.equal(meal.entries[0].notes[0],"Serves 2");
});
test("calendar invalid dates, slots and incomplete recipes fail before approval",async()=>{
 const g=new FixtureGateway();
 for(const entry of [{plan_date:"2026-02-30",meal_slot:"dinner",ingredients:["rice"],instructions:["cook"]},{plan_date:"2026-10-05",meal_slot:"whenever",ingredients:["rice"],instructions:["cook"]},{plan_date:"2026-10-05",meal_slot:"dinner",ingredients:[],instructions:[]}]) await assert.rejects(prepareAction(g,ACTOR,"add_generated_recipes_to_meal_calendar",{entries:[entry]},[]));
});
test("dish followups require an unambiguous recent personal entry",async()=>{
 const g=new FixtureGateway(),rows=[{id:"a",ingredients:["Eggs"],updated_at:new Date().toISOString()}];
 const args={ingredients:["Toast"]};await prepareAction(g,ACTOR,"append_to_recent_dish",args,rows);assert.equal(args.dish_id,"a");assert.deepEqual(args.ingredients,["Eggs","Toast"]);
 await assert.rejects(prepareAction(g,ACTOR,"update_recent_dish",{dish_name:"Lunch"},[...rows,{...rows[0],id:"b"}]),{code:"ambiguous_item"});
});
test("recipe and dish edits never verify a changed unrequested field",()=>{
 const before=[{id:"a",title:"Rice",ingredients:["rice"],instructions:["cook"],notes:[]}];
 assert.equal(verifyChange("edit_saved_recipe",{recipe_id:"a",title:"Brown rice"},before,[{...before[0],title:"Brown rice",instructions:["different"]}],{}).verified,false);
 const dishes=[{id:"a",dish_name:"Toast",ingredients:["bread"],action:"IN"}];
 assert.equal(verifyChange("update_dish",{dish_id:"a",dish_name:"Breakfast"},dishes,[{...dishes[0],dish_name:"Breakfast",ingredients:["egg"]}],{}).verified,false);
});
test("imports and generated meal plans cannot earn success from a queued receipt",()=>{
 assert.equal(verifyChange("save_recipe_from_tiktok",{},[],[{id:"a",status:"pending"}],{toolResult:{recipe:{id:"a"}}}).verified,false);
 assert.equal(verifyChange("refresh_meal_plan",{}, {plan:[],_updatedDate:1},{status:"regenerating",plan:[],_updatedDate:2},{}).verified,false);
});
test("social imports cannot become arbitrary internal URL requests",async()=>{
 const g=new FixtureGateway();
 for(const url of ["http://tiktok.com/video/123","https://localhost/admin","https://tiktok.com.evil.example/x","https://name:secret@tiktok.com/v"]) await assert.rejects(prepareAction(g,ACTOR,"save_recipe_from_tiktok",{url},[]),{code:"url"});
 await prepareAction(g,ACTOR,"save_recipe_from_tiktok",{url:"https://www.tiktok.com/@user/video/123"},[]);
});
test("setting a known quantity to zero uses canonical consume and exact reviewed revision",async()=>{
 const g=new TrepoGateway({env:{}});g.check=async()=>ACTOR;let request;
 g.appAPI=async(a,path,{body})=>{request=body;return{operation_id:body.operation_id,items:[{item:{_id:"egg"},removed:true}]};};
 const before=[{id:"egg",quantity_value:2,quantity_unit:"count",amount_revision:7}];
 const result=await g.mutate(ACTOR,"update_item_quantity",{item_id:"egg",quantity_value:0},"op-zero",before);
 assert.equal(request.kind,"consume");assert.equal(request.items[0].revision,7);assert.equal(request.items[0].amount,2);
 assert(verifyChange("update_item_quantity",{item_id:"egg",quantity_value:0},before,[],result).verified);
});
test("clear kitchen uses one canonical atomic batch, not per-name deletes",async()=>{
 const g=new TrepoGateway({env:{}});g.check=async()=>ACTOR;let request;
 g.appAPI=async(a,path,{body})=>{request=body;return{operation_id:body.operation_id};};
 await g.mutate(ACTOR,"clear_kitchen_inventory",{},"op-clear",[{id:"a",amount_revision:2},{id:"b",amount_revision:5}]);
 assert.equal(request.kind,"discard");assert.deepEqual(request.items,[{item_id:"a",revision:2},{item_id:"b",revision:5}]);
});

test("default Groceries list and full discard history freeze only reviewed IDs",async()=>{
 const g=new FixtureGateway();
 for(const action of ["rename_shopping_list","remove_shopping_list"]) {
  const args={store:"Groceries",new_name:"Weekly"};await prepareAction(g,ACTOR,action,args,[{shopping_id:"a",item_name:"Milk",store:null},{shopping_id:"b",store:"Costco"}]);assert.deepEqual(args.targets.map(x=>x.item_id),["a"]);
 }
 const rows=Array.from({length:25},(_,i)=>({id:String(i),item_name:"Item "+i})),args={};await prepareAction(g,ACTOR,"clear_recent_discards",args,rows);assert.equal(args.targets.length,25);
 const one={item_id:"24"};await prepareAction(g,ACTOR,"delete_recent_discard",one,rows);assert.equal(one.targets.length,1);
});
test("suggested recipe groceries default to missing-only and empty means no write",async()=>{
 const g=new FixtureGateway();g.data.suggestions={kitchen_only:[],need_grocery:[{title:"Toast",ingredients:["bread","eggs"],missing_ingredients:["bread"]}]};
 const args={recipe_title:"Toast"};await prepareAction(g,ACTOR,"add_recipe_ingredients_to_shopping_list",args,[]);assert.deepEqual(args.items,[{item_name:"bread"}]);
 g.data.suggestions.need_grocery[0].missing_ingredients=[];await assert.rejects(prepareAction(g,ACTOR,"add_recipe_ingredients_to_shopping_list",{recipe_title:"Toast"},[]),{code:"nothing_to_add"});
});
test("completed automatic meal regeneration can keep the same plan, but needs an acknowledged fresh result",()=>{
 const before={plan:{breakfast:"Toast"},_updatedDate:"2026-10-01T08:00:00Z"},after={...before,status:"ready",focus:"easy",_updatedDate:"2026-10-02T08:00:05Z"};
 const receipt={requestedAt:"2026-10-02T08:00:00Z",toolResult:{async_update:{pending:true}}};
 assert(verifyChange("refresh_meal_plan",{focus:"easy"},before,after,receipt).verified);
 assert(!verifyChange("refresh_meal_plan",{focus:"protein"},before,after,receipt).verified);
 assert(!verifyChange("refresh_meal_plan",{},before,after,{}).verified);
});
test("read pagination is complete and dates preserve the approval window",async()=>{
 const {pageData,proposalReadArgs,proposalSnapshot}=await import("../src/prepare-actions.mjs");
 const rows=Array.from({length:231},(_,i)=>({id:String(i),dish_name:"Dish "+i}));let offset=0,seen=[];
 do {const page=pageData({items:rows},{offset});seen.push(...page.items);offset=page.next_offset;}while(offset!==null);
 assert.equal(seen.length,231);assert.equal(pageData(rows,{query:"Dish 230"}).items[0].id,"230");
 assert.deepEqual(proposalReadArgs("add_generated_recipes_to_meal_calendar",{entries:[{plan_date:"2027-05-01"},{plan_date:"2027-05-07"}]}),{start_date:"2027-05-01",end_date:"2027-05-07"});
 assert.equal(proposalSnapshot("update_dish",{dish_id:"230"},rows).length,1);
});
test("calendar move by ID follows the entry rather than limiting lookup to the destination",async()=>{
 const {proposalReadArgs}=await import('../src/prepare-actions.mjs');
 assert.deepEqual(proposalReadArgs('move_meal_calendar_entry',{entry_id:'meal',new_date:'2027-10-01'}),{entry_id:'meal'});
 const g=new FixtureGateway(),args={entry_id:'meal',new_date:'2027-10-01'};
 await prepareAction(g,ACTOR,'move_meal_calendar_entry',args,[{id:'meal',title:'Toast',plan_date:'2026-10-01',meal_slot:'breakfast'}]);assert.equal(args.entry_id,'meal');
});
test("numeric amount proposals discard derived labels before verification",async()=>{
 const g=new FixtureGateway(),args={item_id:'a',fill_percent:50,remaining_quantity:'half full'};
 await prepareAction(g,ACTOR,'update_item_quantity',args,[{id:'a',quantity_value:10,quantity_unit:'count',reference_amount:10}]);assert(!('remaining_quantity' in args));
 assert(verifyChange('update_item_quantity',args,[{id:'a',fill_percent:100}],[{id:'a',fill_percent:50,quantity_value:5,quantity_unit:'count',remaining_quantity:'5 count'}],{}).verified);
});
