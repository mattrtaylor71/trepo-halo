import test from 'node:test';
import assert from 'node:assert/strict';
import {TrepoGateway} from '../src/gateway.mjs';
const a={actor:'anna',household:'home',ctx:{userId:'anna',ownerId:'home'}};
function setup(){
 const g=new TrepoGateway({env:{}});g.check=async()=>a;
 const recipes=Array.from({length:135},(_,i)=>({id:String(i),title:'Recipe '+i,saved_by:i%2?'matt':'anna',saved_by_name:i%2?'Matt':'Anna',ingredients:['peas'],instructions:['Steam peas']}));
 g.data={getSavedRecipes:async(ctx,o)=>{assert.equal(ctx.userId,'anna');assert.equal(o.all,true);return {scope:'household',recipes,count:recipes.length}},getShoppingItems:async(ctx,o)=>{assert.equal(o.all,true);return [{item_name:'Peas',action:'0'}]}};
 return {g,recipes};
}
test('native recipe lookup pages the complete household collection and resolves older partner detail',async()=>{
 const {g}=setup();const first=await g.readTool(a,'get_saved_recipes',{});
 assert.equal(first.count,135);assert.equal(first.items.length,25);assert.equal(first.items[1].saved_by_name,'Matt');
 const last=await g.readTool(a,'get_saved_recipes',{offset:125,limit:25});assert.equal(last.items.length,10);
 const detail=await g.readTool(a,'get_saved_recipe_detail',{recipe_title:'Recipe 133'});assert.equal(detail.recipe.saved_by,'matt');assert.deepEqual(detail.recipe.instructions,['Steam peas']);
});
test('household read access does not allow proposing an edit or deletion of a partner recipe',async()=>{
 const {g,recipes}=setup();for(const name of ['edit_saved_recipe','remove_saved_recipe'])await assert.rejects(g.prepare(a,name,{recipe_id:'133'},{recipes}),{code:'recipe_not_owned'});
});
test('native shopping reads use the reconciled active household list',async()=>{
 const {g}=setup();assert.deepEqual(await g.read(a,'shopping'),[{item_name:'Peas',action:'0'}]);
});
