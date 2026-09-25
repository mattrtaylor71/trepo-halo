import {test} from 'node:test';
import assert from 'node:assert/strict';
import {reconcileActionNarration} from '../lib/action-narration.mjs';
import {detectActionClaims} from '../lib/action-claims.mjs';
const milk={toolName:'add_to_shopping_list',args:{item_name:'Milk'},result:{ok:true,toolResult:{item:{shopping_id:'milk-1',item_name:'Milk',quantity:'1 carton'}}}};

test('wrong target and amount cannot borrow a successful same-domain tool',async()=>{
 const r=await reconcileActionNarration({text:'Added five loaves of bread to your shopping list.',toolEvents:[milk]});
 assert.equal(r.outcome,'confirmed');assert.match(r.text,/Milk \(1 carton\)/);assert.doesNotMatch(r.text,/five|bread/);
});
test('every claimed domain is checked; one real action cannot validate another',async()=>{
 const text='Added milk to your shopping list and I logged your breakfast.';
 assert.deepEqual(detectActionClaims(text).map(x=>x.domain),['shopping_add','dishes']);
 const r=await reconcileActionNarration({text,toolEvents:[milk]});
 assert.match(r.text,/Milk/);assert.match(r.text,/other changes/);assert.doesNotMatch(r.text,/logged/i);
});
test('negated sentence does not suppress a separate false positive confirmation',()=>{
 assert.deepEqual(detectActionClaims("I haven't logged your breakfast, but I added milk to your shopping list.").map(x=>x.domain),['shopping_add']);
});
test('a historical read stays helpful without a repair or mutation',async()=>{
 let attempts=0;const text='Your breakfast was logged yesterday. Open Dish Log to see it.';
 const r=await reconcileActionNarration({text,repair:()=>{attempts++;throw Error('must not run')}});
 assert.equal(r.text,text);assert.equal(attempts,0);
});
test('false read-answer claim is repaired once with no action executor',async()=>{
 let attempts=0;const r=await reconcileActionNarration({text:'Logged your breakfast.',repair:()=>{attempts++;return 'Eggs contain protein.'}});
 assert.equal(r.text,'Eggs contain protein.');assert.equal(attempts,1);
});
test('a repair repeating an unbacked claim gets an honest terminal answer',async()=>{
 let attempts=0;const r=await reconcileActionNarration({text:'Added milk to your shopping list.',repair:()=>{attempts++;return 'I added it to your shopping list.'}});
 assert.equal(r.outcome,'unconfirmed');assert.equal(attempts,1);assert.match(r.text,/haven't made/);
});
test('repair outage never creates a corrective mutation or repeated model loop',async()=>{
 let attempts=0;const r=await reconcileActionNarration({text:'Logged your breakfast.',repair:()=>{attempts++;throw Error('timeout')}});
 assert.equal(r.outcome,'unconfirmed');assert.equal(attempts,1);
});
test('actual failures survive upbeat text, with successful targets retained',async()=>{
 const failed={toolName:'discard_item',result:{ok:false,statusCode:404,error:'not_found'}};
 const r=await reconcileActionNarration({text:'All done!',toolEvents:[milk,failed]});
 assert.equal(r.outcome,'partial');assert.match(r.text,/Milk/);assert.match(r.text,/couldn't/);assert.doesNotMatch(r.text,/All done/);
});
