const assert=require('node:assert/strict');
const fs=require('fs');
const f=require('./sql-fixture.cjs');
(async()=>{
 await f.setup();
 const c=await f.connect();
 for(const member of [f.A,f.B,f.C]) await c.query(`ALTER TABLE \`${member}_new_list\` MODIFY _id BIGINT NOT NULL AUTO_INCREMENT`);
 // Mirror schema matches the fields used by the real add path; all local only.
 await c.query(`CREATE TABLE IF NOT EXISTS shared_shopping_list LIKE \`${f.A}_new_list\``);
 await c.query('ALTER TABLE shared_shopping_list ADD COLUMN owner_id VARCHAR(36)');
 for(const [name,type] of [['owner_id','VARCHAR(36)'],['_owner','VARCHAR(36)'],['_device','VARCHAR(64)'],['product_name','VARCHAR(255)'],['product_brand','VARCHAR(255)'],['images','TEXT'],['product_barcode','VARCHAR(64)'],['store','VARCHAR(100)'],['_createdDate','DATETIME']]) await c.query(`ALTER TABLE shared_list ADD COLUMN ${name} ${type}`);
 await c.end();
 const h=f.handler(true);
 const call=async b=>{const r=await h({ownerId:f.A,...b});assert.equal(r.statusCode,200);return JSON.parse(r.body)};
 const input={operation:'add',device:'isolated-release-test',product_name:'Release add regression',aisle_category:'produce'};
 const added=await call(input);
 assert.equal(added.items.length,1);
 const item=added.items[0];assert.equal(item.id,item.itemUUID);assert.notEqual(item.id,String(added.insertId));
 const view=await call({operation:'view'});assert.equal(view.items.find(x=>x.product_name===input.product_name).id,item.id);
 const before=await f.state();assert.equal(before.a.length,3);assert.equal(before.b.length,3);assert.equal(before.foreign.length,1);
 console.log('PASS fresh add returns canonical identity stable across next household view');
 const dedup=await call(input);assert.equal(dedup.deduped,true);assert.equal(dedup.items[0].id,item.id);assert.equal(dedup.items[0].itemUUID,item.itemUUID);
 assert.deepEqual(await f.state(),before);
 console.log('PASS duplicate add echoes canonical identity and creates no extra row');
 fs.writeFileSync(__dirname+'/evidence/add-tests.json',JSON.stringify({passed:2,real_isolated_mysql:true,cloud_dispatch_stubbed:true,customer_mutations:false},null,2));
})().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>f.close());
