const assert=require('node:assert/strict');
const fs=require('fs');
const f=require('./sql-fixture.cjs');
const cases=[];
async function check(name,fn){await fn();cases.push(name);console.log('PASS',name);}
const call=async(h,b)=>{const r=await h({...b,ownerId:b.ownerId||f.A}); return {...r,json:JSON.parse(r.body)};};
(async()=>{
 await f.setup();
 const old=f.handler(false),fixed=f.handler(true);
 await check('serving handler reproduces duplicate id and wrong toggle target',async()=>{
  const r=await call(old,{operation:'view'}); assert.equal(r.statusCode,200);
  assert.deepEqual(r.json.items.map(x=>x.id),['37','37']);
  const target=r.json.items.find(x=>x.itemUUID===f.TARGET);
  const chosen=r.json.items.find(x=>x.id===target.id); // existing native lookup
  assert.equal(chosen.itemUUID,f.ALPHA);
  await call(old,{operation:'set_action',itemUUID:chosen.itemUUID,checked:chosen.action!=='CHECKED'});
  const s=await f.state();assert.equal(s.a.find(x=>x.household_item_uuid===f.TARGET).action,'ADDED');
 });
 await f.reset();
 await check('candidate returns two unique household identities',async()=>{
  const r=await call(fixed,{operation:'view'});assert.equal(r.statusCode,200);
  assert.deepEqual(r.json.items.map(x=>x.id),[f.ALPHA,f.TARGET]);
  fs.writeFileSync(__dirname+'/evidence/candidate-response.json',JSON.stringify(r.json,null,2));
 });
 await check('existing client check persists both mirrors without touching sibling or foreign household',async()=>{
  const items=(await call(fixed,{operation:'view'})).json.items;
  const target=items.find(x=>x.itemUUID===f.TARGET);const chosen=items.find(x=>x.id===target.id);
  assert.equal(chosen.itemUUID,f.TARGET);
  assert.equal((await call(fixed,{operation:'set_action',itemUUID:chosen.itemUUID,checked:true})).statusCode,200);
  const s=await f.state();for(const k of ['a','b'])for(const row of s[k])assert.equal(row.action,'CHECKED');
  assert.equal(s.foreign[0].action,'ADDED');
 });
 await check('refresh and mirror winner change retain row identity',async()=>{
  const c=await f.connect();await c.query(`UPDATE \`${f.B}_new_list\` SET updated_at='2027-01-01' WHERE household_item_uuid=?`,[f.ALPHA]);await c.end();
  const r=await call(fixed,{operation:'view'});assert.deepEqual(r.json.items.map(x=>x.id),[f.ALPHA,f.TARGET]);
 });
 await check('uncheck persists and older UUID request contract remains valid',async()=>{
  const r=await call(fixed,{operation:'set_action',itemUUID:f.TARGET,checked:false});assert.equal(r.statusCode,200);
  const s=await f.state();for(const k of ['a','b'])assert.equal(s[k].find(x=>x.household_item_uuid===f.TARGET).action,'ADDED');
 });
 await check('single-table response preserves same string identity',async()=>{
  const h=f.handler(true,true,false);const r=await call(h,{operation:'view'});assert.equal(r.json.union,undefined);
  assert.equal(r.json.items.find(x=>x.itemUUID===f.TARGET).id,f.TARGET);
 });
 await check('UUID-less legacy single-table numeric ID and non-strict mutation remain compatible',async()=>{
  const c=await f.connect();await c.query(`UPDATE \`${f.C}_new_list\` SET household_item_uuid=NULL`);await c.end();
  const h=f.handler(true,false,false);const r=await call(h,{operation:'view',ownerId:f.C});
  assert.equal(r.json.items[0].id,'37');assert.equal(r.json.items[0].itemUUID,'37');
  assert.equal((await call(h,{operation:'set_action',ownerId:f.C,itemUUID:'37',checked:true})).statusCode,200);
 });
 await check('unknown UUID still returns 404',async()=>{
  assert.equal((await call(fixed,{operation:'set_action',itemUUID:'unknown-fixture',checked:true})).statusCode,404);
 });
 fs.writeFileSync(__dirname+'/evidence/sql-test-summary.json',JSON.stringify({passed:cases.length,cases,sql:'real isolated MySQL',customer_mutations:false},null,2));
 console.log(`${cases.length} SQL integration checks passed`);
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>f.close());
