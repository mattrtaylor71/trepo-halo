import {test,before,beforeEach,mock} from 'node:test';
import assert from 'node:assert/strict';
let da,state,conn,stores,decision,modelCalls,duringModel;
const ctx=()=>({userId:'alice',tableOwnerId:'alice',ownerId:'family',householdMemberIds:['foreign'],responseSurface:'halo'});
const opts=(id='request-1')=>({skipKitchenDependentGeneration:true,sourceTranscript:'Discard the avocado.',mutationContext:{requestId:id}});
const avocado=(id,date)=>({_id:id,owner_id:'alice',action:'IN',product_name:'Avocado',_createdDate:date,amount_revision:0,acquired_on:null});
before(async()=>{
 process.env.WRITE_SHARED_ONLY='true';process.env.DB_HOST='test';process.env.DB_USER='test';process.env.DB_PASS='test';process.env.DB_NAME='test';
 const mysqlPath=new URL('../lib/mysql.mjs',import.meta.url).href, mysql=await import(mysqlPath);
 mock.module(mysqlPath,{namedExports:{...mysql,withDbConnection:async callback=>callback(conn)}});
 const storePath=new URL('../lib/session-store.mjs',import.meta.url).href,store=await import(storePath);
 mock.module(storePath,{namedExports:{...store,
  readKitchenEditRequest:async scope=>{const old=stores.get(scope.household+scope.operationId);if(old && old.hash!==scope.intentHash)throw Error('intent changed');return old?.body || null;},
  retainKitchenEditRequest:async(scope,body)=>{const key=scope.household+scope.operationId;if(!stores.has(key))stores.set(key,{hash:scope.intentHash,body:structuredClone(body)});return stores.get(key).body;}
 }});
 // Replace the actual provider SDK, not the product matcher or data-access code.
 mock.module('openai',{defaultExport:class {responses={create:async()=>{
  modelCalls++;assert.equal(state.inTransaction,false,'no provider work under row locks');if(duringModel)await duringModel();
  if(decision instanceof Error)throw decision;return {status:'completed',output_text:JSON.stringify(decision),model:'test-model'};
 }}}});
 da=await import('../lib/data-access.mjs');
});
beforeEach(()=>{
 stores=new Map();modelCalls=0;duringModel=null;
 decision={status:'matched',matching_ids:['new','old'],scope:'one_entry',selection:'oldest',authorization_quote:'Discard the avocado',reason:'same fruit'};
 state={rows:[avocado('new','2026-10-20'),avocado('old','2026-10-01')],members:['alice','bob'],family:'family',receipts:[],writes:[],inTransaction:false};
 let backup;
 conn={beginTransaction:async()=>{backup=structuredClone({rows:state.rows,receipts:state.receipts});state.inTransaction=true;},commit:async()=>{state.inTransaction=false;},rollback:async()=>{Object.assign(state,backup);state.inTransaction=false;},
 execute:async(raw,args=[])=>{
  const sql=raw.replace(/\s+/g,' ').trim();
  if(sql.startsWith('SELECT user_id,owner_id FROM new_users'))return [[{user_id:'alice',owner_id:state.family}]];
  if(sql.startsWith('SELECT user_id FROM new_users'))return [state.members.map(user_id=>({user_id}))];
  if(sql.includes('information_schema.tables'))return [[{count:1}]];
  if(sql.includes('information_schema.columns'))return [[]];
  if(sql.startsWith('ALTER') || sql.startsWith('CREATE'))return [{}];
  if(sql.startsWith('SELECT k.*'))return [structuredClone(state.rows.filter(r=>r.action==='IN' && state.members.includes(r.owner_id) && (!sql.includes('k._id=?') || r._id===args.at(-1))))];
  if(sql.startsWith('SELECT _id,source_kitchen_id'))return [structuredClone(state.receipts.filter(r=>r._id===args[0]))];
  if(sql.startsWith('UPDATE `shared_kitchen`')){state.writes.push(sql);const row=state.rows.find(r=>r._id===args[2]);if(row)row.action='OUT';return [{affectedRows:row?1:0}];}
  if(sql.startsWith('INSERT INTO `shared_discards`')){state.writes.push(sql);state.receipts.push({_id:args[0],source_kitchen_id:args[10],owner_id:args[12]});return [{affectedRows:1}];}
  if(sql.startsWith('INSERT')){state.writes.push(sql);return [{affectedRows:1}];}
  if(sql.startsWith('SELECT'))return [[]];
  throw Error(sql);
 }};
});
test('real voice pipeline removes oldest once and replays frozen receipt',async()=>{
 const result=await da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',opts());
 assert.equal(result.item.id,'old');assert.equal(state.rows.find(r=>r._id==='new').action,'IN');
 assert.equal(result.affected_count,1);const count=state.writes.length;
 const retry=await da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',opts());
 assert.equal(retry.replayed,true);assert.equal(state.writes.length,count);assert.equal(modelCalls,1);
});
for(const change of ['quantity','revision','deletion','transfer'])test('voice rejects '+change+' during provider request',async()=>{
 duringModel=async()=>{
  if(change==='quantity')state.rows[1].quantity_value=4;
  if(change==='revision')state.rows[1].amount_revision=2;
  if(change==='deletion')state.rows=state.rows.filter(r=>r._id!=='old');
  if(change==='transfer')state.members=['alice'];
 };
 await assert.rejects(()=>da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',opts()));
 assert.equal(state.writes.length,0);assert.equal(state.rows[0].action,'IN');
});
test('no-match, model failure, and missing transcript cannot mutate',async()=>{
 for(const value of [{status:'no_match',matching_ids:[],reason:'different product'},new Error('timeout')]){
  decision=value;await assert.rejects(()=>da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',opts(String(modelCalls))));
  assert.equal(state.writes.length,0);
 }
 await assert.rejects(()=>da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',{...opts('missing'),sourceTranscript:null}));
 assert.equal(state.writes.length,0);
});
test('foreign/stale exact ID cannot substitute a same-name owned row',async()=>{
 await assert.rejects(()=>da.discardKitchenItem(ctx(),{item_id:'foreign',item_name:'avocado'},'finished',opts()));
 assert.equal(modelCalls,0);assert.equal(state.writes.length,0);
});
test('quantity zero goes through the same guard instead of fuzzy fan-out',async()=>{
 decision={status:'no_match',matching_ids:[],reason:'not an avocado'};
 await assert.rejects(()=>da.updateKitchenItemQuantity(ctx(),{item_name:'avocado'},{quantity_value:0},opts()));
 assert.equal(modelCalls,1);assert.equal(state.writes.length,0);
});
test('retry after changing households does not resolve a new target',async()=>{
 await da.discardKitchenItem(ctx(),{item_name:'avocado'},'finished',opts());
 const writes=state.writes.length;
 state.family='other-family';
 await assert.rejects(()=>da.discardKitchenItem({...ctx(),ownerId:'other-family'},{item_name:'avocado'},'finished',opts()));
 assert.equal(modelCalls,1);assert.equal(state.writes.length,writes);
});
