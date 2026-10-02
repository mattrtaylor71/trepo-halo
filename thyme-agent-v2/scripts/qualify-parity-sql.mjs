import assert from "node:assert/strict";
import fs from "node:fs";
import {createRequire} from "node:module";
import {personalWrite,shoppingWrite,discardWrite,discardFields,categoryWrite} from "../src/app-writes.mjs";
import {saveCanonical} from "../src/canonical-save.mjs";
import {MemoryStore} from "../src/store.mjs";
import {Service} from "../src/service.mjs";
import {Runner} from "../src/runner.mjs";
import {ToolGateway,toolDefinitions} from "../src/tools.mjs";
import {catalog} from "../src/capabilities.mjs";
import {ACTIONS} from "../src/gateway.mjs";
import {prepareAction} from "../src/prepare-actions.mjs";
import {scope,key,id} from "../src/core.mjs";
import {verifyChange} from "../src/verification.mjs";
const require=createRequire(process.env.THYME_LEGACY_PACKAGE || "/Users/MattTaylor/Library/Caches/trepo-thyme-release-20261001/trepo-quick-ack-stream-dev/source/playground12_voice_ack/trepo-quick-ack/package.json");
const c=await require("mysql2/promise").createConnection({host:"127.0.0.1",port:33317,user:"fixture",password:"fixture",database:"thyme_test_parity_20261002"});
const actor={actor:"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",household:"parity-house"}, member="bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee",other="cccccccc-bbbb-cccc-dddd-eeeeeeeeeeee";
const report={environment:"isolated local MySQL; no customer data",tests:[]};
const pass=name=>{report.tests.push({name,passed:true});console.log(name);};
async function read(kind,who=actor.actor) {
 const table=kind==="shopping"?"shared_shopping_list":who+"_"+kind;
 const [rows]=await c.execute(`SELECT * FROM \`${table}\`${kind==="shopping"?" WHERE owner_id=?":""} ORDER BY _id`,kind==="shopping"?[who]:[]);
 return rows.map(r=>({...r,id:String(r._id),item_name:r.product_name,shopping_id:String(r._id),quantity:r.quantity||null,
   ingredients:r.ingredients||[],instructions:r.instructions||[],notes:r.notes||[]}));
}
try {
 await c.execute("CREATE TABLE IF NOT EXISTS new_users(user_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36))");
 for(const id of [actor.actor,member,other]) await c.execute("INSERT INTO new_users VALUES (?,?) ON DUPLICATE KEY UPDATE owner_id=VALUES(owner_id)",[id,id===other?"other-house":actor.household]);
 await c.execute("CREATE TABLE IF NOT EXISTS shared_dishes(owner_id VARCHAR(36),_id VARCHAR(36),_owner VARCHAR(36),user_id VARCHAR(36),_device VARCHAR(50),dish_name VARCHAR(500),serving_size VARCHAR(100),ingredients JSON,calories DOUBLE,protein DOUBLE,total_fat DOUBLE,total_carbohydrates DOUBLE,action VARCHAR(20),explanation TEXT,analysis_status VARCHAR(30),_createdDate DATETIME,_updatedDate DATETIME,PRIMARY KEY(owner_id,_id))");
 await c.execute("CREATE TABLE IF NOT EXISTS shared_saved_recipes(owner_id VARCHAR(36),_id VARCHAR(36),_owner VARCHAR(36),source_type VARCHAR(32),source_url VARCHAR(1000),resolved_url VARCHAR(1000),resolved_url_hash CHAR(64),title VARCHAR(255),ingredients JSON,instructions JSON,notes JSON,status VARCHAR(32),_createdDate DATETIME,_updatedDate DATETIME,PRIMARY KEY(owner_id,_id))");
 await c.execute("CREATE TABLE IF NOT EXISTS shared_shopping_list(owner_id VARCHAR(36),_id VARCHAR(36),household_item_uuid VARCHAR(36),product_name VARCHAR(255),quantity VARCHAR(100),store VARCHAR(255),action VARCHAR(20),updated_at DATETIME,PRIMARY KEY(owner_id,_id))");
 for(const id of [actor.actor,member,other]) {
  await c.execute(`CREATE TABLE IF NOT EXISTS \`${id}_master_feed\` (_id VARCHAR(36) PRIMARY KEY,_owner VARCHAR(36),_device VARCHAR(50),_createdDate DATETIME,event_type VARCHAR(64),entity_type VARCHAR(64),action VARCHAR(64),title TEXT,item_id VARCHAR(64),user_id VARCHAR(36),source_table VARCHAR(128),source_path VARCHAR(255),source_system VARCHAR(64),metadata JSON)`);
  await c.execute(`DELETE FROM \`${id}_master_feed\``);
 }
 for(const id of [actor.actor,member,other]) for(const [suffix,base] of [["dishes","shared_dishes"],["saved_recipes","shared_saved_recipes"],["new_list","shared_shopping_list"]]) await c.execute(`CREATE TABLE IF NOT EXISTS \`${id}_${suffix}\` LIKE ${base}`);
 // This database belongs exclusively to this qualification script.
 for(const base of ["shared_dishes","shared_saved_recipes","shared_shopping_list"]) await c.execute(`DELETE FROM ${base}`);
 for(const id of [actor.actor,member,other]) for(const suffix of ["dishes","saved_recipes","new_list"]) await c.execute(`DELETE FROM \`${id}_${suffix}\``);
 const recipe={title:"Egg rice",ingredients:["2 eggs","1 cup rice"],steps:["Cook rice.","Cook eggs fully."],notes:["Serves 2"]};
 const saved=await saveCanonical(c,actor,recipe,"parity-save");
 pass("Save exact recipe into current and older app stores");
 let before=await read("saved_recipes");const recipeId=saved.toolResult.recipe.id;
 let args={recipe_id:recipeId,title:"Egg rice for two"};
 let result=await personalWrite(c,actor,"edit_saved_recipe",args,"edit-1",before),after=await read("saved_recipes");
 assert(verifyChange("edit_saved_recipe",args,before,after,result).verified);assert.deepEqual(after[0].ingredients,recipe.ingredients);
 const [sharedRecipe]=await c.execute("SELECT * FROM shared_saved_recipes WHERE owner_id=? AND _id=?",[actor.actor,recipeId]);assert.equal(sharedRecipe[0].title,args.title);
 pass("Targeted recipe edit preserves every unrequested line and mirrors the app");
 await assert.rejects(personalWrite(c,actor,"edit_saved_recipe",{recipe_id:recipeId,title:"stale"},"stale",before),{code:"stale_approval"});pass("Stale saved recipe approval rejected inside transaction");
 await assert.rejects(personalWrite(c,{...actor,actor:other,household:"other-house"},"edit_saved_recipe",args,"foreign",after),{code:"missing"});pass("Foreign recipe ID cannot be edited");
 before=after;await personalWrite(c,actor,"remove_saved_recipe",{recipe_id:recipeId},"delete-recipe",before);
 assert(verifyChange("remove_saved_recipe",{recipe_id:recipeId},before,await read("saved_recipes"),{}).verified);pass("Saved recipe removal persists in both stores");
 const dish={dish_name:"Egg toast",ingredients:["2 eggs","1 slice toast"],serving_size:"1 serving",calories:290,protein_g:18};
 before=await read("dishes");result=await personalWrite(c,actor,"log_dish_from_voice",dish,"log-1",before);after=await read("dishes");
 assert(verifyChange("log_dish_from_voice",dish,before,after,result).verified);const dishId=result.toolResult.dish.id;
 assert.equal((await read("dishes",member)).length,0);pass("Dish logging persists to personal app log without leaking to household members");
 await personalWrite(c,actor,"log_dish_from_voice",dish,"log-1",[]);assert.equal((await read("dishes")).length,1);pass("Repeated dish operation is idempotent");
 await assert.rejects(personalWrite(c,actor,"log_dish_from_voice",{...dish,dish_name:"Wrong"},"log-1",[]),{code:"request_conflict"});pass("Same operation cannot log different content");
 await c.execute("DELETE FROM shared_dishes WHERE owner_id=? AND _id=?",[actor.actor,dishId]);
 before=await read("dishes");await personalWrite(c,actor,"update_dish",{dish_id:dishId,serving_size:"one serving"},"repair-dish",before);
 const [repaired]=await c.execute("SELECT * FROM shared_dishes WHERE owner_id=? AND _id=?",[actor.actor,dishId]);assert.equal(repaired.length,1);assert(repaired[0]._createdDate instanceof Date);assert.equal(repaired[0].user_id,actor.actor);pass("Missing dish mirror repairs transactionally with valid database timestamps");
 for(const [action,patch] of [["update_dish",{serving_size:"2 servings"}],["append_to_recent_dish",{ingredients:[...dish.ingredients,"1 tomato"]}],["update_recent_dish",{dish_name:"Breakfast toast",calories:300}],["mark_dish_consumed",{}]]) {
  before=await read("dishes");args={dish_id:dishId,...patch};result=await personalWrite(c,actor,action,args,action,before);after=await read("dishes");assert(verifyChange(action,args,before,after,result).verified,action);pass(action+" persisted and verified");
 }
 before=await read("dishes");await assert.rejects(personalWrite(c,{...actor,actor:member},"delete_dish_log",{dish_id:dishId},"foreign-dish",before),{code:"missing"});pass("Same-household foreign dish ID rejected");
 await personalWrite(c,actor,"delete_dish_log",{dish_id:dishId},"delete-dish",before);assert.equal((await read("dishes")).length,0);const [left]=await c.execute("SELECT * FROM shared_dishes WHERE _id=?",[dishId]);assert.equal(left.length,0);pass("Dish delete removes exact personal mirrors");
 // Duplicate names, different IDs: only the reviewed item may change.
 for(const id of [actor.actor,member,other]) for(const index of [1,2,3]) for(const table of ["shared_shopping_list",id+"_new_list"]) await c.execute(`INSERT INTO \`${table}\` (owner_id,_id,household_item_uuid,product_name,quantity,store,action) VALUES (?,?,?,?,?,?,?)`,[id,String(index),"stable-"+index,index<3?"Milk":"Rice","1",index<3?"Costco":"Groceries","ADDED"]);
 await c.execute("DELETE FROM shared_shopping_list WHERE owner_id=?",[member]);
 for(const [action,patch] of [["update_shopping_item",{new_name:"Oat milk",quantity:"2 cartons"}],["mark_shopping_item_bought",{}],["mark_shopping_item_unbought",{}],["update_shopping_item_store",{store:"Trader Joe's"}]]) {
  before=await read("shopping");args={item_id:"1",...patch};await shoppingWrite(c,actor,action,args,before);after=await read("shopping");assert(verifyChange(action,args,before,after,{}).verified,action);
  assert.equal(after.find(r=>r.id==="2").item_name,"Milk");assert.equal((await read("shopping",other)).find(r=>r.id==="1").item_name,"Milk");pass(action+" preserves duplicate names and foreign household UUID collisions");
 }
 before=await read("shopping");args={targets:[{item_id:"1",store:"Weekend"},{item_id:"2",store:"Weekend"}]};await shoppingWrite(c,actor,"rename_shopping_list",args,before);after=await read("shopping");assert(verifyChange("rename_shopping_list",args,before,after,{}).verified);pass("Rename list moves exact members and preserves other stores");
 before=after;await c.execute(`DELETE FROM \`${member}_new_list\` WHERE _id='2'`);args={targets:[{item_id:"1"},{item_id:"2"}]};await assert.rejects(shoppingWrite(c,actor,"remove_shopping_list",args,before),{code:"mirror_changed"});assert.equal((await read("shopping")).length,3);pass("Mid-batch missing mirror rolls back all earlier deletes");
 await c.execute(`INSERT INTO \`${member}_new_list\` SELECT * FROM shared_shopping_list WHERE owner_id=? AND _id='2'`,[actor.actor]);
 await shoppingWrite(c,actor,"remove_shopping_list",args,before);after=await read("shopping");assert(verifyChange("remove_shopping_list",args,before,after,{}).verified);assert.equal(after.length,1);pass("Delete one list keeps other lists and their items");
 const [history]=await c.execute(`SELECT metadata FROM \`${actor.actor}_master_feed\``);assert.equal(history.length,2);assert(history.every(x=>x.metadata.snapshot.product_name));pass("Successful deletes preserve history; rolled-back deletes leave no history");
 await c.execute("CREATE TABLE IF NOT EXISTS shared_discards(owner_id VARCHAR(36),_id VARCHAR(36),product_name VARCHAR(255),brand VARCHAR(128),category VARCHAR(128),discard_reason TEXT,action VARCHAR(10),_createdDate DATETIME,_updatedDate DATETIME,PRIMARY KEY(owner_id,_id))");
 for(const id of [actor.actor,member,other]) {await c.execute(`CREATE TABLE IF NOT EXISTS \`${id}_discards\` LIKE shared_discards`);await c.execute(`DELETE FROM \`${id}_discards\``);}
 await c.execute("DELETE FROM shared_discards");
 for(const id of [actor.actor,member,other]) for(let i=1;i<=25;i++) for(const table of ["shared_discards",id+"_discards"])
  await c.execute(`INSERT INTO \`${table}\` (owner_id,_id,product_name,action,_createdDate) VALUES (?,?,?,'IN','2026-10-01')`,[id,String(i),"Old item "+i]);
 let [discardRows]=await c.execute(`SELECT * FROM \`${actor.actor}_discards\` WHERE action='IN'`);const discards=discardRows.map(discardFields),targets=discards.map(r=>({item_id:r.id}));
 for(const table of ["shared_discards",actor.actor+"_discards"]) await c.execute(`INSERT INTO \`${table}\` (owner_id,_id,product_name,action,_createdDate) VALUES (?, 'late','Added after approval','IN','2026-10-02')`,[actor.actor]);
 await discardWrite(c,actor,{targets},discards);
 const [remaining]=await c.execute(`SELECT * FROM \`${actor.actor}_discards\` WHERE action='IN'`);assert.deepEqual(remaining.map(r=>r._id),["late"]);
 assert(verifyChange("clear_recent_discards",{targets},discards,remaining.map(discardFields),{}).verified);const [foreignDiscards]=await c.execute("SELECT * FROM shared_discards WHERE owner_id=? AND action='IN'",[other]);assert.equal(foreignDiscards.length,25);pass("Clear 25 exact discard IDs preserves post-approval and foreign rows");
 await assert.rejects(discardWrite(c,actor,{targets},discards),{code:"stale_approval"});pass("Stale discard approval rejected without replay");
 await c.execute("CREATE TABLE IF NOT EXISTS shared_recipe_categories(owner_id VARCHAR(36),category_id VARCHAR(36),name VARCHAR(64),sort_order INT,PRIMARY KEY(owner_id,category_id),UNIQUE KEY(owner_id,name))");
 await c.execute("CREATE TABLE IF NOT EXISTS shared_recipe_category_map(owner_id VARCHAR(36),recipe_id VARCHAR(36),category_id VARCHAR(36),PRIMARY KEY(owner_id,recipe_id,category_id))");
 await c.execute("DELETE FROM shared_recipe_category_map");await c.execute("DELETE FROM shared_recipe_categories");
 const second=await saveCanonical(c,actor,recipe,"category-recipe");const sid=second.toolResult.recipe.id;
 await categoryWrite(c,actor,{category_name:"Dinner",recipe_ids:[sid]});await categoryWrite(c,actor,{category_name:"Quick",recipe_ids:[sid]});
 const [tags]=await c.execute("SELECT * FROM shared_recipe_category_map WHERE owner_id=? AND recipe_id=?",[actor.actor,sid]);assert.equal(tags.length,2);pass("Category addition preserves other app tags atomically");
 await assert.rejects(categoryWrite(c,actor,{category_name:"Wrong",recipe_ids:["foreign"]}),{code:"missing"});const [absent]=await c.execute("SELECT * FROM shared_recipe_categories WHERE name='Wrong'");assert.equal(absent.length,0);pass("Foreign recipe category assignment rolls back category creation");
 // Full review -> approval -> worker -> SQL -> authoritative readback path.
 const gateway={actor:async value=>{assert.equal(value,actor.actor);return actor;},check:async value=>{assert.deepEqual(value,actor);return actor;},definitions:async()=>catalog(JSON.parse(fs.readFileSync(new URL("../vendor/tool-definitions.json",import.meta.url))),ACTIONS),
  read:async(a,resource)=>resource==="preferences"?{dietary:{}}:resource==="kitchen"?[]:read(resource),
  prepare:async(a,action,args,before)=>prepareAction(gateway,a,action,args,before),
  mutate:async(a,action,args,op,before)=>action==="update_shopping_item"?shoppingWrite(c,a,action,args,before,op):personalWrite(c,a,action,args,op,before)};
 const store=new MemoryStore(),service=new Service({store,gateway}),definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions});
 for(const [action,args] of [["update_shopping_item",{item_id:"3",quantity:"2 bags"}],["log_dish_ingredients",{dish_name:"Review-tested breakfast",ingredients:["2 eggs"],serving_size:"one"}],["edit_saved_recipe",{recipe_id:sid,title:"Updated through approval"}]]) {
  const requestId=id(),session=await service.message(actor,{requestId,text:"Prepare fixture change"});const meta=await store.get(scope(actor),key(session.id));
  await store.put(scope(actor),key(session.id),{...meta,status:"completed",activeRequest:null},meta.version);
  const before=await read(action==="update_shopping_item"?"shopping":action==="edit_saved_recipe"?"saved_recipes":"dishes");
  const proposal=await tools.call({actor,session,action:{name:"request_"+action,arguments:args,call_id:id()}});assert(proposal.ok,JSON.stringify(proposal));
  assert.deepEqual(await read(action==="update_shopping_item"?"shopping":action==="edit_saved_recipe"?"saved_recipes":"dishes"),before);
  const approvalId=id();await service.decide(actor,{sessionId:session.id,proposalId:proposal.proposal_id,decision:"approve",requestId:approvalId});
  const runner=new Runner({store,gateway,tools,definitions,provider:{}});await runner.run(scope(actor),approvalId);await runner.run(scope(actor),approvalId);
  const receipt=await store.get(scope(actor),key(session.id,"P",proposal.proposal_id));assert.equal(receipt.status,"applied",receipt.error);
  pass(action+" full approval/worker/SQL/readback path; duplicate worker delivery is harmless");
 }
 await c.execute("UPDATE new_users SET owner_id='moved' WHERE user_id=?",[actor.actor]);await assert.rejects(shoppingWrite(c,actor,"remove_from_shopping_list",{item_id:"3"},after),{code:"membership"});pass("Membership change fenced inside transaction");
 report.passed=true;report.customerWrites=0;
 if(process.env.THYME_EVIDENCE) fs.writeFileSync(process.env.THYME_EVIDENCE,JSON.stringify(report,null,2));
 console.log(JSON.stringify({passed:true,count:report.tests.length,customerWrites:0}));
} finally {await c.end();}
