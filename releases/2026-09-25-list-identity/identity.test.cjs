// Exercise the actual mapper without loading network/database dependencies.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../APP/Lambdas/listHandler.js'), 'utf8');
const start = source.indexOf('function mapListRow(');
const end = source.indexOf('\nfunction unionMergeKey(', start);
assert(start >= 0 && end > start);
const map = vm.runInNewContext(source.slice(start, end) + '\nmapListRow');
const row = (id, uuid, name = 'Test grocery') => ({_id:id, household_item_uuid:uuid, product_name:name, action:'ADDED'});

test('different household groceries sharing a table-local number stay distinct', () => {
  const items = [map(row(37,'11111111-1111-4111-8111-111111111111')), map(row(37,'22222222-2222-4222-8222-222222222222'))];
  assert.equal(new Set(items.map(x=>x.id)).size,2);
  const selected = items.find(x=>x.id===items[1].id);
  assert.equal(selected.itemUUID,items[1].itemUUID);
});
test('another household mirror winning does not change grocery identity', () => {
  assert.equal(map(row(37,'shared-uuid')).id,map(row(99,'shared-uuid')).id);
});
test('public identity matches the existing mutation identity', () => {
  const item=map(row(37,'shared-uuid'));assert.equal(item.id,item.itemUUID);
  assert.equal(typeof item.id,'string');
});
for (const missing of [undefined,null,'']) {
  test(`legacy UUID ${String(missing)} retains numeric string identity`,()=>{
    const item=map(row(37,missing));assert.equal(item.id,'37');assert.equal(item.itemUUID,'37');
  });
}
test('other response fields retain their existing wire shape',()=>{
  const item=JSON.parse(JSON.stringify(map({...row(37,'shared-uuid'),product_brand:'Brand',store:'Shop',sort_order:'2',aisle_category:'produce'})));
  assert.deepEqual(item,{id:'shared-uuid',product_name:'Test grocery',action:'ADDED',product_brand:'Brand',product_barcode:null,store:'Shop',quantity:null,itemUUID:'shared-uuid',sortOrder:2,aisle_category:'produce'});
});
