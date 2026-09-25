import {test} from 'node:test';
import assert from 'node:assert/strict';
import {kitchenLabelDenial as deny} from '../lib/kitchen-label-intent.mjs';
import {correctionFieldDenial} from '../../../shared/voice-assistant/kitchen-correction-intent.mjs';
test('one flavored oil cannot silently become its separate ingredients',()=>{
 assert.ok(deny('Add garlic olive oil to my pantry','check_in_many_items',{items:[{item_name:'Garlic Oil'},{item_name:'Olive Oil'}]}));
 assert.ok(deny('Add one bottle of garlic olive oil','check_in_item',{item_name:'Olive Oil'}));
 assert.equal(deny('Add fresh crushed cilantro olive oil','check_in_item',{item_name:'Fresh Crushed Cilantro Olive Oil'}),null);
});
test('ambiguous punctuation asks; explicitly separate products and ordinary lists still work',()=>{
 assert.ok(deny('In my pantry, garlic oil, olive oil','check_in_many_items',{}));
 assert.equal(deny('Add two separate products: garlic oil, olive oil','check_in_many_items',{items:[{item_name:'Garlic Oil'},{item_name:'Olive Oil'}]}),null);
 assert.equal(deny('Add garlic olive oil and milk to my pantry','check_in_many_items',{items:[{item_name:'Garlic Olive Oil'},{item_name:'Milk'}]}),null);
 assert.equal(deny('Add milk, eggs, and bread','check_in_many_items',{items:[{item_name:'Milk'},{item_name:'Eggs'},{item_name:'Bread'}]}),null);
});
test('case changes in a name do not authorize opening the product',()=>{
 assert.equal(correctionFieldDenial('update_item_details',{new_name:'Opened Sesame Oil',is_opened:true},'Rename it to OPENED SESAME OIL'),'opened_state_not_requested');
});
