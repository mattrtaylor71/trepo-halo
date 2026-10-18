// Real MySQL evaluation with statement-local fixtures: no tables or customer data are changed.
// Run with DB_* configured: node test/household-kitchen.sql-check.mjs
import assert from 'node:assert/strict';
import {getDbPool} from '../lib/mysql.mjs';
import {readSharedHouseholdKitchen} from '../lib/household-kitchen.mjs';
const pool=getDbPool();let members,items;
const prefix=`WITH new_users AS (
 SELECT user_id COLLATE utf8mb4_0900_ai_ci AS user_id,owner_id COLLATE utf8mb4_0900_ai_ci AS owner_id
 FROM JSON_TABLE(?, '$[*]' COLUMNS(user_id VARCHAR(64) PATH '$.user_id',owner_id VARCHAR(64) PATH '$.owner_id')) j
), shared_kitchen AS (
 SELECT _id,owner_id COLLATE utf8mb4_unicode_ci AS owner_id,action,product_name,_updatedDate,_createdDate
 FROM JSON_TABLE(?, '$[*]' COLUMNS(_id VARCHAR(64) PATH '$._id',owner_id VARCHAR(64) PATH '$.owner_id',action VARCHAR(8) PATH '$.action',product_name VARCHAR(255) PATH '$.product_name',_updatedDate DATETIME PATH '$._updatedDate',_createdDate DATETIME PATH '$._createdDate')) j
) `;
const c={execute:(sql,params)=>pool.execute(prefix+sql,[JSON.stringify(members),JSON.stringify(items),...params])};
const ctx=(userId='a',ownerId='home')=>({userId,ownerId,householdMemberIds:['foreign'],tableOwnerId:'foreign'});
try{
 members=[{user_id:'a',owner_id:'home'},{user_id:'b',owner_id:'home'},{user_id:'foreign',owner_id:'other'},{user_id:'solo',owner_id:null},{user_id:'solo2',owner_id:null},{user_id:'empty',owner_id:'empty'}];
 items=[{_id:'1',owner_id:'a',product_name:'Mustard',action:'IN',_updatedDate:'2026-10-16 12:00:00'},
 {_id:'2',owner_id:'b',product_name:'Mustard',action:'IN',_updatedDate:'2026-10-17 12:00:00'},
 {_id:'3',owner_id:'foreign',product_name:'Private',action:'IN'},
 {_id:'4',owner_id:'a',product_name:'Removed',action:'OUT'},
 {_id:'5',owner_id:'solo',product_name:'Solo',action:'IN'},
 {_id:'6',owner_id:'solo2',product_name:'Solo2',action:'IN'}];
 for(const actor of ['a','b'])assert.deepEqual((await readSharedHouseholdKitchen(c,ctx(actor))).map(r=>r._id),['2','1']);
 assert.deepEqual((await readSharedHouseholdKitchen(c,ctx('solo','solo'))).map(r=>r._id),['5']);
 assert.deepEqual(await readSharedHouseholdKitchen(c,ctx('empty','empty')),[]);
 await assert.rejects(()=>readSharedHouseholdKitchen(c,ctx('missing','missing')),/membership/);
 await assert.rejects(()=>readSharedHouseholdKitchen(c,ctx('a','old-home')),/changed/);
 await assert.rejects(()=>readSharedHouseholdKitchen(c,{...ctx(),isFallbackContext:true}),/unavailable/);
 members[1].owner_id='other';assert.deepEqual((await readSharedHouseholdKitchen(c,ctx())).map(r=>r._id),['1']);
 members[0].owner_id='';assert.deepEqual((await readSharedHouseholdKitchen(c,ctx('a','a'))).map(r=>r._id),['1']);
 console.log('PASS: 9 real-SQL assertions: full household, same-name items, ordering, removed/foreign exclusion, solo/empty, unknown/stale/fallback membership, member departure, mixed collations.');
}finally{await pool.end();}
