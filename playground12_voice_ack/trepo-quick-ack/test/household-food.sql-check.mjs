// Actual MySQL semantics with statement-local fixtures; changes no tables/data.
import assert from 'node:assert/strict';
import {getDbPool} from '../lib/mysql.mjs';
import {readHouseholdFoodRows} from '../lib/household-food-data.mjs';
const pool=getDbPool();
let members=[{user_id:'a',owner_id:'home',first_name:'A'},{user_id:'b',owner_id:'home',first_name:'B'},{user_id:'x',owner_id:'elsewhere'},{user_id:'solo',owner_id:null},{user_id:'solo2',owner_id:null},{user_id:'empty',owner_id:'empty'}];
const rows=[{_id:'1',owner_id:'a'},{_id:'2',owner_id:'b'},{_id:'3',owner_id:'x'},{_id:'4',owner_id:'solo'},{_id:'5',owner_id:'solo2'}];
const ctx=(userId='a',ownerId='home')=>({userId,ownerId,householdMemberIds:['x']});
try{
 for(const [resource,table] of [['saved_recipes','shared_saved_recipes'],['shopping','shared_shopping_list'],['meal_plan','shared_meal_plan']]){
  const prefix=`WITH new_users AS (SELECT user_id COLLATE utf8mb4_0900_ai_ci AS user_id,owner_id COLLATE utf8mb4_0900_ai_ci AS owner_id,first_name FROM JSON_TABLE(?,'$[*]' COLUMNS(user_id VARCHAR(64) PATH '$.user_id',owner_id VARCHAR(64) PATH '$.owner_id',first_name VARCHAR(40) PATH '$.first_name')) j), ${table} AS (SELECT _id,owner_id COLLATE utf8mb4_unicode_ci AS owner_id,NULL AS _createdDate,NULL AS _updatedDate,NULL AS updated_at,NULL AS created_at FROM JSON_TABLE(?,'$[*]' COLUMNS(_id VARCHAR(64) PATH '$._id',owner_id VARCHAR(64) PATH '$.owner_id')) j) `;
  const c={execute:(sql,p)=>pool.execute(prefix+sql,[JSON.stringify(members),JSON.stringify(rows),...p])};
  for(const actor of ['a','b'])assert.deepEqual((await readHouseholdFoodRows(c,ctx(actor),resource)).map(r=>r._id),['1','2']);
  assert.deepEqual((await readHouseholdFoodRows(c,ctx('solo','solo'),resource)).map(r=>r._id),['4']);
  assert.deepEqual(await readHouseholdFoodRows(c,ctx('empty','empty'),resource),[]);
  await assert.rejects(()=>readHouseholdFoodRows(c,ctx('missing','missing'),resource),/membership/);
  await assert.rejects(()=>readHouseholdFoodRows(c,ctx('a','old'),resource),/changed/);
  members[1].owner_id='elsewhere';assert.deepEqual((await readHouseholdFoodRows(c,ctx(),resource)).map(r=>r._id),['1']);members[1].owner_id='home';
 }
 console.log('PASS: 21 real-SQL household food assertions, including mixed collations and membership boundaries.');
}finally{await pool.end();}
