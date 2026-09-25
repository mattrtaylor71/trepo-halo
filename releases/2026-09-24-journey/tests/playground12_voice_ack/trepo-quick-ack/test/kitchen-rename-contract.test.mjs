import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateToolCall } from '../../../shared/voice-assistant/tool-executor.mjs';

test('name-only correction distinguishes lookup name from replacement', () => {
  const result = validateToolCall('update_item_details', {item_name:'Garlic Oil', new_name:' Garlic Olive Oil '});
  assert.equal(result.ok, true);
  assert.deepEqual(result.args, {item_name:'Garlic Oil', new_name:'Garlic Olive Oil'});
});

test('exact-id rename retains only the explicit name when optional model fields are null', () => {
  const result = validateToolCall('update_item_details', {item_id:'bottle', new_name:'Garlic Olive Oil',
    quantity_value:null, quantity_unit:null, brand:null, is_opened:null, location:null});
  assert.equal(result.ok, true);
  assert.deepEqual(result.args, {item_id:'bottle', new_name:'Garlic Olive Oil'});
});

for (const extra of [{product_name:'Wrong'}, {made_up:'value'}, {revision:0}, {owner:'other'}]) {
  test(`unknown mutation field is rejected: ${Object.keys(extra)[0]}`, () => {
    assert.equal(validateToolCall('update_item_details', {item_id:'bottle', brand:'Brand', ...extra}).ok, false);
  });
}

for (const name of ['', ' ', 'x'.repeat(501), 12, {}]) {
  test(`invalid new name cannot be hidden by another valid field: ${typeof name}`, () => {
    assert.equal(validateToolCall('update_item_details', {item_id:'bottle', new_name:name, brand:'Brand'}).ok, false);
  });
}
