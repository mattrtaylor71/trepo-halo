import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectActionClaim } from '../lib/action-claims.mjs';

const ordinaryAnswers = [
  'Items you added to your shopping list appear in List.',
  'Your breakfast was logged yesterday.',
  'If you have already checked items into your kitchen, open Cook.',
  'I have not added milk to your shopping list.',
  "I haven't logged your breakfast.",
  'The phrase “I added milk to your shopping list” is an example confirmation.',
  '> Added milk to your shopping list.\nThat was the previous message, not a new action.',
  'You saved that recipe last week. Open your saved recipes to find it.',
  'Once an item has been removed from your kitchen, it stops appearing there.',
  'Your meal was logged on Monday, according to your history.'
];
for (const text of ordinaryAnswers) test('read-only narration: ' + text, () => assert.equal(detectActionClaim(text), null));

test('a saved recipe confirmation belongs to recipes, never food consumed', () => {
  assert.equal(detectActionClaim('Saved: the stir fry recipe you asked about.')?.domain, 'recipe_save');
});

test('current positive write confirmations remain guarded', () => {
  assert.equal(detectActionClaim('I added milk to your shopping list.')?.domain, 'shopping_add');
  assert.equal(detectActionClaim('Your breakfast has been logged.')?.domain, 'dishes');
  assert.equal(detectActionClaim('I removed the cheese from your kitchen.')?.domain, 'kitchen_remove');
});
