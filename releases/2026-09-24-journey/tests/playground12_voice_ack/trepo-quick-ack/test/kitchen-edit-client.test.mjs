import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createKitchenMutationContext, kitchenEditBody, saveKitchenEdit} from '../lib/kitchen-edit-client.mjs';

const context = {ownerId:'household-1', userId:'actor-1', tableOwnerId:'actor-1'};
const row = {_id:'bottle-1', product_name:'Garlic Oil', amount_revision:3};
const fields = {product_name:'Garlic Olive Oil'};

test('operation identity survives tool-call changes and uses the original revision on retry', () => {
  const first = createKitchenMutationContext({operationId:'request-1'});
  const body = kitchenEditBody(first, row, fields);
  assert.deepEqual(kitchenEditBody(first, {...row,amount_revision:4}, fields), body);
  assert.deepEqual(kitchenEditBody(createKitchenMutationContext({operationId:'request-1'}), row, fields), body);
  const changed = kitchenEditBody(createKitchenMutationContext({operationId:'request-1'}), row, {brand:'Wrong'});
  assert.equal(changed.operation_id, body.operation_id); // Canonical request hash rejects this reuse.
  assert.notEqual(kitchenEditBody(createKitchenMutationContext({operationId:'request-2'}), row, fields).operation_id, body.operation_id);
});

test('canonical request carries the caller credential and only resolved actor plus authorized fields', async t => {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, 'https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com/kitchen/actor-1?amount_operation=v1');
    assert.equal(options.headers.Authorization, 'Bearer fixture-token');
    assert.equal(options.redirect, 'error');
    const body = JSON.parse(options.body);
    assert.deepEqual(body.items, [{item_id:'bottle-1',revision:3,fields}]);
    return Response.json({operation_id:body.operation_id, items:[{removed:false,item:{...row,...fields,amount_revision:4}}]});
  });
  const result = await saveKitchenEdit(context, row, fields, {mutationContext:createKitchenMutationContext({operationId:'req',authorization:'Bearer fixture-token'})});
  assert.equal(result.product_name, fields.product_name);
  assert.equal(result.amount_revision, 4);
});

for (const status of [401,403,404,409,429,500,503]) {
  test(`API rejection ${status} never becomes a success`, async t => {
    t.mock.method(globalThis,'fetch',async () => Response.json({error:'fixture rejected'}, {status}));
    await assert.rejects(saveKitchenEdit(context,row,fields), e => e.statusCode === status);
  });
}

for (const fault of ['name','identity','operation','missing','removed','revision']) {
  test(`an invalid success response is not trusted: ${fault}`, async t => {
    t.mock.method(globalThis,'fetch', async (_, options) => {
      const body=JSON.parse(options.body);
      const payload={operation_id:body.operation_id,items:[{removed:false,item:{...row,...fields,amount_revision:4}}]};
      if (fault === 'name') payload.items[0].item.product_name='Old name';
      if (fault === 'identity') payload.items[0].item._id='foreign-item';
      if (fault === 'operation') payload.operation_id='wrong-request';
      if (fault === 'missing') payload.items=[];
      if (fault === 'removed') payload.items[0].removed=true;
      if (fault === 'revision') delete payload.items[0].item.amount_revision;
      return Response.json(payload);
    });
    await assert.rejects(saveKitchenEdit(context,row,fields), /confirm/);
  });
}

test('missing actor and invalid API configuration stop before any network request', async t => {
  t.mock.method(globalThis, 'fetch', async () => assert.fail('must not request'));
  await assert.rejects(saveKitchenEdit({ownerId:'household'},row,fields), /account/);
  await assert.rejects(saveKitchenEdit(context,row,fields,{env:{KITCHEN_API_BASE_URL:'http://localhost'}}), /unavailable/);
  await assert.rejects(saveKitchenEdit(context,row,{owner_id:'other'}), /Unsupported/);
});

for (const [input,expected] of [['cloves','clove'],['heads','head'],['pieces','count']]) {
  test(`canonical ${input} is confirmed without a false post-commit failure`, async t => {
    t.mock.method(globalThis,'fetch',async (_,options) => {
      const body=JSON.parse(options.body);
      assert.equal(body.items[0].fields.quantity_unit,expected);
      return Response.json({operation_id:body.operation_id,items:[{item:{...row,product_name:'Garlic',quantity_value:4,quantity_unit:expected,amount_revision:4}}]});
    });
    assert.equal((await saveKitchenEdit(context,row,{product_name:'Garlic',quantity_value:4,quantity_unit:input})).quantity_unit,expected);
  });
}

const {renameReference}=await import('../lib/kitchen-edit-client.mjs');
test('a model-selected duplicate ID and selection hint are not a user selection',()=>{
 const ref={item_id:'old',item_name:'Garlic Oil',selection_hint:'oldest'};
 assert.deepEqual(renameReference(ref,{new_name:'Garlic Olive Oil'},{sourceTranscript:'Rename Garlic Oil to Garlic Olive Oil'},{product_name:'Garlic Oil'}),{item_id:null,item_name:'Garlic Oil',selection_hint:null});
 assert.equal(renameReference(ref,{new_name:'Garlic Olive Oil'},{sourceTranscript:'Rename the newest Garlic Oil to Garlic Olive Oil'},null).selection_hint,'most_recent');
 assert.equal(renameReference(ref,{new_name:'Newest Garlic Oil'},{sourceTranscript:'Rename Garlic Oil to Newest Garlic Oil'},null).selection_hint,null);
 assert.equal(renameReference(ref,{new_name:'Garlic Olive Oil'},{sourceTranscript:'most recent',expectedRevision:0}),ref);
 assert.equal(renameReference(ref,{new_name:'Garlic Olive Oil'},{sourceTranscript:'Rename old to Garlic Olive Oil'}),ref);
});
test('an unrelated grounded ID cannot redirect a user-named correction',()=>{
 const ref={item_id:'garlic-id',item_name:'Beef broth'};
 assert.equal(renameReference(ref,{new_name:'Organic Beef Broth'},{sourceTranscript:'Rename Beef broth to Organic Beef Broth'},{product_name:'Garlic Oil'}).item_name,'Beef broth');
 assert.throws(()=>renameReference({item_id:'garlic-id',item_name:'Garlic Oil'},{new_name:'Organic Beef Broth'},{sourceTranscript:'Rename Beef broth to Organic Beef Broth'},{product_name:'Garlic Oil'}),/repeat/);
});
test('shortening an overlapping name preserves its old target; label words are not selectors',()=>{
 assert.equal(renameReference({item_name:'Beef Broth'},{new_name:'Broth'},{sourceTranscript:'Rename Beef Broth to Broth'}).item_name,'Beef Broth');
 assert.equal(renameReference({item_name:'Newest Garlic Oil'},{new_name:'Olive Oil'},{sourceTranscript:'Rename Newest Garlic Oil to Olive Oil'}).selection_hint,null);
});
const {pickRenameByCreation}=await import('../lib/kitchen-edit-client.mjs');
test('newest and oldest use creation, not last edited date, and reject ties',()=>{
 const old={id:'old',_createdDate:'2026-08-01',_updatedDate:'2026-09-24'},recent={id:'recent',_createdDate:'2026-09-01',_updatedDate:'2026-09-01'};
 assert.equal(pickRenameByCreation([old,recent],'most_recent').id,'recent');
 assert.equal(pickRenameByCreation([old,recent],'oldest').id,'old');
 assert.equal(pickRenameByCreation([old,{...old,id:'tie'}],'oldest'),null);
});
