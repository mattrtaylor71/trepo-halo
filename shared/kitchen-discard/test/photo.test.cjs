const {test}=require('node:test');
const assert=require('node:assert/strict');
const {createPhotoMatcher}=require('../../../playground6_groceryRec/analyze_discard_on_upload_nodejs/discard_matcher.cjs');
const {resolveDecision}=require('../identity.cjs');
function fixture() {
 const state={rows:[{_id:'old',owner_id:'alice',action:'IN',product_name:'Avocado',_createdDate:'2026-10-01',amount_revision:0,acquired_on:null},
 {_id:'new',owner_id:'alice',action:'IN',product_name:'Hass Avocados',_createdDate:'2026-10-20',amount_revision:0,acquired_on:null}],plan:null,archive:[],members:['alice'],writes:[],calls:0,failDelete:false};
 let backup;
 const conn={beginTransaction:async()=>{backup=structuredClone({rows:state.rows,archive:state.archive});},
  commit:async()=>{},rollback:async()=>Object.assign(state,backup),end:async()=>{},
  execute:async(sql,args)=>{
   if(sql.startsWith('SELECT user_id'))return [state.members.map(user_id=>({user_id}))];
   if(sql.startsWith('SELECT k.*'))return [structuredClone(sql.includes('k._id=?')?state.rows.filter(r=>r._id===args.at(-1)):state.rows)];
   if(sql.startsWith('SELECT _id'))return [structuredClone(state.archive.filter(r=>r._id===args[0]))];
   state.writes.push(sql);
   if(sql.startsWith('INSERT')) {state.archive.push({...state.rows.find(r=>r._id===args[1]),archived_reason:args[0]});return [{affectedRows:1}];}
   if(sql.startsWith('DELETE')) {if(state.failDelete)throw Error('database disconnected');state.rows=state.rows.filter(r=>r._id!==args[0]);return [{affectedRows:1}];}
   throw Error(sql);
  }};
 const matcher=createPhotoMatcher({connect:async()=>conn,members:async()=>state.members,load:async()=>state.plan,
  retain:async(job,owner,plan)=>state.plan ||= structuredClone(plan),match:async input=>{
   state.calls++;return resolveDecision(input,{status:'matched',matching_ids:['new','old'],scope:'one_entry',selection:'oldest',reason:'same fruit'});
  }});
 const call=remove=>matcher('alice',{product_name:'Avocado'},{jobId:'job-one',remove});
 return {state,call};
}
test('preview is read-only; commit and replay remove only oldest once',async()=>{
 const {state,call}=fixture();assert.equal((await call(false))._id,'old');assert.equal(state.writes.length,0);
 assert.equal((await call(true)).removed,true);assert.deepEqual(state.rows.map(r=>r._id),['new']);
 assert.equal((await call(false))._id,'old');assert.equal((await call(true)).replayed,true);
 assert.equal(state.calls,1);assert.equal(state.writes.filter(s=>s.startsWith('DELETE')).length,1);
});
test('commit without frozen preview never resolves a fresh target',async()=>{
 const {state,call}=fixture();assert.equal((await call(true)).removed,false);assert.equal(state.calls,0);assert.equal(state.writes.length,0);
});
for(const change of ['edit','delete','transfer','bought_date'])test(change+' after preview prevents removal, never substitutes next row',async()=>{
 const {state,call}=fixture();await call(false);
 if(change==='edit')state.rows[0].quantity_value=2;
 if(change==='delete')state.rows.shift();
 if(change==='transfer')state.members=['bob'];
 if(change==='bought_date')state.rows[0].amount_revision=1;
 assert.equal((await call(true)).removed,false);assert.equal(state.writes.length,0);assert.equal(state.calls,1);
});
test('archive + delete roll back together on commit failure',async()=>{
 const {state,call}=fixture();await call(false);state.failDelete=true;
 assert.equal((await call(true)).removed,false);assert.equal(state.rows.length,2);assert.equal(state.archive.length,0);
});
test('retained model no-match is final across retries',async()=>{
 const {state,call}=fixture();await call(false);state.plan={...state.plan,matched:false,status:'no_match',reason:'no_match'};
 assert.equal((await call(true)).removed,false);assert.equal(state.calls,1);assert.equal(state.writes.length,0);
});
