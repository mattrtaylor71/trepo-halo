const {test}=require('node:test');
const assert=require('node:assert/strict');
const {matchIdentity,resolveDecision,fingerprint}=require('../identity.cjs');
const row=(id,name,date='2026-10-15T12:00:00Z',extra={})=>({_id:id,owner_id:'alice',action:'IN',product_name:name,_createdDate:date,...extra});
const decision=(ids,extra={})=>({status:'matched',matching_ids:ids,reason:'same product',scope:'one_entry',selection:'oldest',authorization_quote:null,...extra});
const resolve=(rows,answer,target={product_name:'Avocado'},extra={})=>resolveDecision({target,candidates:rows,...extra},answer);
test('one oldest unbranded duplicate, not newest or edited time',()=>{
 const rows=[row('new','Avocado','2026-10-20'),row('old','Avocados','2026-10-10',{_updatedDate:'2026-10-25'})];
 assert.equal(resolve(rows,decision(['new','old'])).selected_id,'old');
});
test('recorded bought date precedes capture date',()=>{
 const rows=[row('a','Avocado','2026-10-20',{acquired_on:'2026-10-01'}),row('b','Avocado','2026-10-10')];
 assert.equal(resolve(rows,decision(['a','b'])).selected_id,'a');
});
test('missing duplicate dates fail; equal dates have deterministic ID tie break',()=>{
 assert.equal(resolve([row('a','Avocado','bad'),row('b','Avocado')],decision(['a','b'])).matched,false);
 assert.equal(resolve([row('b','Avocado'),row('a','Avocado')],decision(['b','a'])).selected_id,'a');
});
test('provider abstention or failure never becomes a heuristic match',async()=>{
 const input={target:{product_name:'Avocado'},candidates:[row('a','Avocado')]};
 for(const status of ['no_match','ambiguous','uncertain','not_authorized']) {
  const answer=await matchIdentity(input,{create:async()=>({output_text:JSON.stringify({status,matching_ids:[],reason:status})})});
  assert.equal(answer.matched,false);
 }
 assert.equal((await matchIdentity(input,{create:async()=>{throw Error('timeout');}})).matched,false);
 assert.equal((await matchIdentity(input,{create:async()=>({status:'incomplete',output_text:JSON.stringify(decision(['a']))})})).matched,false);
});
test('vague identities and incomplete candidate coverage do not call model',async()=>{
 for(const input of [{target:{product_name:'Dairy tub'},candidates:[]},{target:{product_name:'Avocado'},candidates:Array(501).fill(row('a','Avocado'))}]) {
  assert.equal((await matchIdentity(input,{create:()=>{throw Error('must not call');}})).matched,false);
 }
});
test('unknown IDs, conflicting brands and barcodes are rejected even if model agrees',()=>{
 assert.equal(resolve([row('a','Avocado')],decision(['foreign'])).matched,false);
 assert.equal(resolve([row('a','Beans',undefined,{brand:'Goya'})],decision(['a']),{product_name:'Beans',brand:'Trader Joes'}).matched,false);
 assert.equal(resolve([row('a','Milk',undefined,{barcode:'041900074518'})],decision(['a']),{product_name:'Milk',barcode:'041900074501'}).matched,false);
});
test('voice requires current authorization; unsupported quantity or selection cannot broaden removal',()=>{
 const rows=[row('a','Avocado')];
 assert.equal(resolve(rows,decision(['a']),undefined,{kind:'voice',transcript:'Check in avocado'}).matched,false);
 assert.equal(resolve(rows,decision(['a'],{authorization_quote:'Discard avocado'}),undefined,{kind:'voice',transcript:'Chicken one avocado which is five days old'}).matched,false);
 for (const scope of ['multiple','partial']) assert.equal(resolve(rows,decision(['a'],{scope})).matched,false);
 assert.equal(resolve(rows,decision(['a'],{selection:'newest'})).matched,false);
});
test('snapshot covers quantities, customer edits, ownership and Dates',()=>{
 const original=row('a','Avocado',new Date('2026-10-20'),{amount_revision:2,quantity_value:1});
 assert.equal(fingerprint(original),fingerprint(JSON.parse(JSON.stringify(original))));
 for(const patch of [{amount_revision:3},{quantity_value:2},{owner_id:'bob'},{product_name:'Hass avocado'}])
  assert.notEqual(fingerprint(original),fingerprint({...original,...patch}));
});
