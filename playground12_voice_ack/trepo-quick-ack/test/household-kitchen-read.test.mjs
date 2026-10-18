import {test,mock,before,beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
const lib=resolve(import.meta.dirname,'../lib');
let da, rows, members, failed;
const context=(userId='anna')=>({userId,tableOwnerId:userId,ownerId:'family',householdMemberIds:['outsider']});
const inventory=()=>[
 {_id:'mustard',owner_id:'matt',product_name:'Dijon Mustard',action:'IN',_updatedDate:'2026-10-17'},
 {_id:'peas',owner_id:'anna',product_name:'Green Peas',action:'IN',_updatedDate:'2026-10-16'},
 {_id:'outside',owner_id:'outsider',product_name:'Private Sauce',action:'IN'},
 {_id:'removed',owner_id:'matt',product_name:'Old Ketchup',action:'OUT'},
];
const connection={async execute(sql,params=[]){
 if(failed) throw new Error('database unavailable');
 if(/FROM new_users actor/.test(sql)){
  const actor=members.find(m=>m.user_id===params[0]);if(!actor)return [[]];
  const allowed=members.filter(m=>m.user_id===actor.user_id || (actor.owner_id && m.owner_id===actor.owner_id)).map(m=>m.user_id);
  const found=rows.filter(r=>allowed.includes(r.owner_id)&&r.action==='IN');
  return [(found.length?found:[{_id:null}]).map(r=>({...r,_kitchen_scope_actor:actor.user_id,_kitchen_scope_household:actor.owner_id||actor.user_id}))];
 }
 if(/shared_kitchen/.test(sql)&&/^\s*SELECT/.test(sql))return [rows.filter(r=>r.owner_id===params[0]&&r.action==='IN')];
 if(/information_schema.tables/.test(sql))return [[{count:0}]];
 return [[]];
}};
before(async()=>{
 process.env.WRITE_SHARED_ONLY='true';
 const mysql=await import(lib+'/mysql.mjs');
 mock.module(lib+'/mysql.mjs',{namedExports:{...mysql,withDbConnection:async cb=>cb(connection)}});
 da=await import(lib+'/data-access.mjs');
});
beforeEach(()=>{rows=inventory();members=[{user_id:'anna',owner_id:'family'},{user_id:'matt',owner_id:'family'},{user_id:'outsider',owner_id:'elsewhere'}];failed=false;});
test('both household members see every active item, regardless of who added it',async()=>{
 for(const user of ['anna','matt'])assert.deepEqual((await da.getKitchenItemsFull(context(user))).map(x=>x.item_name),['Dijon Mustard','Green Peas']);
});
test('limited inventory is sliced after household collection',async()=>{
 const items=await da.getKitchenItems(context(),{limit:1});assert.deepEqual(items.map(x=>x.item_name),['Dijon Mustard']);
});
test('follow-up lookup finds a partner item by id and name',async()=>{
 assert.equal((await da.getKitchenItemById(context(),'mustard'))?.item_name,'Dijon Mustard');
 assert.equal((await da.searchKitchenItem(context(),'mustard')).item?.item_name,'Dijon Mustard');
});
test('foreign and removed ids are not readable',async()=>{
 for(const id of ['outside','removed'])assert.equal(await da.getKitchenItemById(context(),id),null);
 assert.equal((await da.searchKitchenItem(context(),'Private Sauce')).found,false);
});
test('current membership overrides a stale member list',async()=>{
 members[1].owner_id='elsewhere';assert.deepEqual((await da.getKitchenItemsFull(context())).map(x=>x.item_name),['Green Peas']);
});
test('missing account is unavailable rather than an empty kitchen',async()=>{
 await assert.rejects(()=>da.getKitchenItemsFull(context('missing')),/unavailable|membership/i);
});
test('empty kitchen and null household do not include other solo accounts',async()=>{
 members[0].owner_id=null;members[1].owner_id=null;rows=[];
 assert.deepEqual(await da.getKitchenItemsFull({...context(),ownerId:'anna'}),[]);
 rows=inventory();assert.deepEqual((await da.getKitchenItemsFull({...context(),ownerId:'anna'})).map(x=>x.item_name),['Green Peas']);
});
test('database failure propagates instead of pretending inventory is empty',async()=>{
 failed=true;await assert.rejects(()=>da.getKitchenItemsFull(context()),/unavailable/);
});
test('overview count equals its complete inventory, including more than 100 items',async()=>{
 rows=Array.from({length:125},(_,i)=>({_id:String(i),owner_id:i%2?'anna':'matt',product_name:'Item '+i,action:'IN'}));
 const result=await da.getKitchenOverview(context());assert.equal(result.count,125);assert.equal(result.summary.kitchen_count,125);assert.equal(result.items.length+result.additional_items_compact.length,125);
});
