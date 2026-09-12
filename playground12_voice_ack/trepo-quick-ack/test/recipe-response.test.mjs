import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {formatRecipeResponse} from '../lib/recipe-response.mjs';
import {buildAppOutput} from '../lib/app-response.mjs';
const fixtures = JSON.parse(fs.readFileSync(new URL('./recipe-response-fixtures.json', import.meta.url)));
for (const fixture of fixtures) {
  test(fixture.name, () => {
    const formatted = formatRecipeResponse(fixture.text);
    assert.equal(formatted !== null, fixture.expected.length > 0);
    if (!formatted) return;
    assert.equal(formatRecipeResponse(formatted), formatted, 'normalization must be idempotent');
    const output = buildAppOutput({text: fixture.text, responseSurface: 'app', sessionId:'test-session', memoryUsed:true});
    assert.equal(output.message.text, formatted);
    assert.equal(output.version, '1');
    assert.equal(output.status, 'ok');
    assert.equal(output.meta.session_id, 'test-session');
    assert.equal(output.meta.memory_used, true);
    assert.deepEqual(output.actions, []);
    for (const recipe of fixture.expected) {
      assert.ok(formatted.includes(recipe.title+'\nIngredients:\n'));
      for (const ingredient of recipe.ingredients) assert.ok(formatted.includes('- '+ingredient));
      for (const [i, step] of recipe.steps.entries()) assert.ok(formatted.includes(`${i+1}. ${step}`));
    }
  });
}
test('descriptions, followups and allergy notes remain in the response', () => {
  const text='### Apple Yogurt Bites\nQuick and easy.\nIngredients:\n- 1 apple\n- yogurt\nSteps:\n1. Slice the apple.\n2. Top with yogurt.\n\nNotes:\nCheck the yogurt label for allergens.\n\nWould you like more ideas?';
  const result=formatRecipeResponse(text);
  for (const value of ['Quick and easy.', 'Check the yogurt label for allergens.', 'Would you like more ideas?']) assert.ok(result.includes(value));
});
test('non-string and ordinary replies do not become recipes', () => {
  for (const value of [null,undefined,{},'Added milk to your list.','Ingredients are what you cook with. Steps make a method.']) assert.equal(formatRecipeResponse(value),null);
});
