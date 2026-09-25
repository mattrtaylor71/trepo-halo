import test from 'node:test';
import assert from 'node:assert/strict';
import {detectActionClaims} from '../lib/action-claims.mjs';
import {reconcileActionNarration} from '../lib/action-narration.mjs';
for(const text of ['This recipe uses all fruit raw with no added ingredients.','This recipe has no added sugar.','This recipe uses fruit without added ingredients.','This recipe uses fruit without any added sugar.'])test('ingredient negation is not a save claim: '+text,async()=>{
 assert.deepEqual(detectActionClaims(text),[]);
 let calls=0;const r=await reconcileActionNarration({text,repair:()=>{calls++;return 'bad';}});
 assert.equal(r.text,text);assert.equal(r.outcome,'read_only');assert.equal(calls,0);
});
for(const text of ['No problem, I added milk to your shopping list.','No worries, I saved that recipe.','This recipe has no added sugar, but I saved the recipe.','This recipe has no added sugar and I saved the recipe.','This recipe uses fruit without added sugar. I saved that recipe.'])test('separate actual write claim is still guarded: '+text,()=>{
 assert.equal(detectActionClaims(text).length,1);
});
