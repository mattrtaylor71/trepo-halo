import {test,mock,before,beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {householdSavedRecipes,householdShoppingRows,householdMealPlans,readHouseholdFoodRows} from '../lib/household-food-data.mjs';
const ctx=(userId='anna')=>({userId,tableOwnerId:userId,ownerId:'home',householdMemberIds:['outside']});
const lib=resolve(import.meta.dirname,'../lib');let data,rows,members,fail=false;
const recipe=(id,owner,title)=>({_id:id,owner_id:owner,_owner:owner,title,ingredients:'["Peas"]',instructions:'["Cook peas"]',notes:'[]',status:'ready'});
const c={execute:async(sql,params=[])=>{
 if(fail)throw new Error('unavailable');
 if(sql.includes('FROM new_users actor')){
  const actor=members.find(r=>r.user_id===params[0]);if(!actor)return [[]];
  const visible=members.filter(r=>r.user_id===actor.user_id||(actor.owner_id&&r.owner_id===actor.owner_id));
  const source=sql.includes('shared_saved_recipes')?rows.recipes:sql.includes('shared_shopping_list')?rows.shopping:rows.plans;
  const found=source.filter(r=>visible.some(m=>m.user_id===r.owner_id));
  return [(found.length?found:[{_id:null}]).map(r=>({...r,_scope_actor:actor.user_id,_scope_household:actor.owner_id||actor.user_id,_member_name:r.owner_id==='anna'?'Anna':'Matt'}))];
 }
 if(sql.includes('shared_shopping_list'))return [rows.shopping.filter(r=>r.owner_id===params[0])];
 return [[]];
}};
before(async()=>{
 process.env.WRITE_SHARED_ONLY='true';
 const mysql=await import(lib+'/mysql.mjs');mock.module(lib+'/mysql.mjs',{namedExports:{...mysql,withDbConnection:async fn=>fn(c)}});
 globalThis.fetch=async url=>({ok:true,text:async()=>JSON.stringify({owner:ctx().userId,recipes:rows.recipes.filter(r=>String(url).includes(r.owner_id)),status:'ready',plan:[]})});
 data=await import(lib+'/data-access.mjs');
});
beforeEach(()=>{fail=false;members=[{user_id:'anna',owner_id:'home'},{user_id:'matt',owner_id:'home'},{user_id:'outside',owner_id:'other'}];rows={recipes:[recipe('a','anna','Anna Peas'),recipe('m','matt','Matt Salad'),recipe('x','outside','Private Dish')],shopping:[],plans:[]};});
test('saved recipes include both people with attribution and full detail',async()=>{
 for(const actor of ['anna','matt']){const result=await data.getSavedRecipes(ctx(actor));assert.equal(result.count,2);assert.equal(result.scope,'household');assert.deepEqual(result.recipes.map(r=>r.saved_by).sort(),['anna','matt']);assert.deepEqual(result.recipes[0].instructions,['Cook peas']);}
});
test('recipe beyond first API page remains discoverable by title and id',async()=>{
 rows.recipes=Array.from({length:135},(_,i)=>recipe(String(i),i%2?'matt':'anna','Recipe '+i));
 const list=await data.getSavedRecipes(ctx(),{limit:8});assert.equal(list.count,135);assert.equal(list.recipes.length,8);assert.equal(list.catalog.length,135);assert.equal(list.has_more,true);
 assert.equal((await data.getSavedRecipeDetail(ctx(),{recipe_title:'Recipe 134'})).recipe.id,'134');
 assert.equal((await data.getSavedRecipeDetail(ctx(),{recipe_id:'133'})).recipe.saved_by,'matt');
});
test('partner recipe may be read but cannot be deleted through the personal endpoint',async()=>{
 await assert.rejects(()=>data.removeSavedRecipe(ctx(),{recipe_title:'Matt Salad'}),/only its owner/);
});
test('separately owned copies keep their IDs and authors even for identical sources',()=>{
 const base={...recipe('a','anna','Peas'),resolved_url_hash:'same',source_owner_id:'anna'};
 const copies=[{...base,_id:'b',source_owner_id:'matt'},base,{...base,_id:'c',source_owner_id:'matt',instructions:'["Steam peas"]'}];
 const result=householdSavedRecipes(copies,'anna');assert.equal(result.length,3);assert.deepEqual(result.map(r=>r.id),['b','a','c']);assert.equal(result[1].saved_by,'anna');
});
test('shopping merges shared items and excludes removed replicas before applying a limit',async()=>{
 rows.shopping=[{_id:'1',owner_id:'matt',product_name:'Peas',household_item_uuid:'p',action:'ADDED',updated_at:'2026-10-18'},
 {_id:'2',owner_id:'anna',product_name:'Peas',household_item_uuid:'p',action:'ADDED',updated_at:'2026-10-18'},
 {_id:'3',owner_id:'anna',product_name:'Old Milk',household_item_uuid:'m',action:'ADDED',updated_at:'2026-10-17'},
 {_id:'4',owner_id:'matt',product_name:'Old Milk',household_item_uuid:'m',action:'REMOVED',updated_at:'2026-10-18'}];
 const result=await data.getShoppingItems(ctx());assert.deepEqual(result.map(r=>r.item_name),['Peas']);
});
test('checked items remain labelled as bought, and removed wins a timestamp tie',()=>{
 const rows=[{_id:'a',owner_id:'anna',action:'CHECKED'},{_id:'b',owner_id:'anna',action:'ADDED'},{_id:'b',owner_id:'matt',action:'REMOVED'}];
 assert.deepEqual(householdShoppingRows(rows,'anna').map(r=>r.action),['CHECKED']);
});
test('new household member can read a partner plan; conflicting plans are not silently lost',async()=>{
 rows.plans=[{_id:'current',owner_id:'matt',status:'ready',plan:'[{"meal_type":"dinner","recipe":{"title":"Peas"}}]'},
 {_id:'current',owner_id:'anna',status:'failed',error_message:'Needs refresh',plan:'[]'}];
 const result=await data.getMealPlan(ctx());assert.equal(result.scope,'household');assert.equal(result.plan[0].recipe.title,'Peas');assert.equal(result.household_plans.length,2);assert.equal(result.household_plans[1].status,'failed');
 assert.equal(householdMealPlans([rows.plans[0],{...rows.plans[0],owner_id:'anna'}]).length,1);
});
test('membership departure immediately removes partner recipes and plans',async()=>{
 members[1].owner_id='other';const result=await data.getSavedRecipes(ctx());assert.equal(result.count,1);assert.equal(result.recipes[0].saved_by,'anna');
});
test('empty household differs from unknown membership or failed reads',async()=>{
 rows.recipes=[];assert.equal((await data.getSavedRecipes(ctx())).count,0);
 await assert.rejects(()=>data.getSavedRecipes(ctx('missing')),/membership/);
 fail=true;await assert.rejects(()=>data.getSavedRecipes(ctx()),/unavailable/);
});
test('unsupported collections cannot interpolate arbitrary SQL',async()=>{
 await assert.rejects(()=>readHouseholdFoodRows(c,ctx(),'shared_health'),/Unsupported/);
});
