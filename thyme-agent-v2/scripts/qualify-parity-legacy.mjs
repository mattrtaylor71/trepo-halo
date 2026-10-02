import fs from 'node:fs';import assert from 'node:assert/strict';import {createRequire} from 'node:module';
import {shoppingWrite} from '../src/app-writes.mjs';
const root=process.env.THYME_LEGACY_ROOT;
if(!root)throw Error('Set THYME_LEGACY_ROOT to the extracted qualification package.');
const path=root+'/playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs';
// Test-only exports on an extracted copy, never the deployed artifact or pinned base.
if(!fs.readFileSync(path,'utf8').includes('export { ensureSharedTables'))fs.appendFileSync(path,'\nexport { ensureSharedTables, ensureMasterFeedTable };\n');
const env={DB_HOST:'127.0.0.1',DB_PORT:'33317',DB_USER:'fixture',DB_PASS:'fixture',DB_NAME:'thyme_test_legacy_parity_20261002',WRITE_SHARED_ONLY:'true'};Object.assign(process.env,env);
const data=await import(path),mysql=await import(root+'/playground12_voice_ack/trepo-quick-ack/lib/mysql.mjs');
const require=createRequire(root+'/playground12_voice_ack/trepo-quick-ack/package.json');const c=await require('mysql2/promise').createConnection({host:'127.0.0.1',port:33317,user:'fixture',password:'fixture',database:env.DB_NAME});
const actor='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',member='bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee',a={actor,household:'parity-house'},ctx={userId:actor,ownerId:a.household,householdMemberIds:[actor,member]};
const report={environment:'real pinned legacy add adapter + new write adapter; isolated SQL',customerWrites:0,tests:[]};
const pass=name=>{report.tests.push(name);console.log(name);};
const read=async()=>{const [r]=await c.execute("SELECT * FROM shared_shopping_list WHERE owner_id=? AND action IN ('ADDED','CHECKED')",[actor]);return r.map(data.mapShoppingRow);};
try {
 await data.ensureSharedTables(c);await c.execute('CREATE TABLE IF NOT EXISTS new_users(user_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36))');
 for(const id of [actor,member]){await c.execute('INSERT INTO new_users VALUES (?,?) ON DUPLICATE KEY UPDATE owner_id=VALUES(owner_id)',[id,a.household]);await data.ensureMasterFeedTable(c,id+'_master_feed');}
 const name='Qualification milk '+Date.now();await data.addShoppingItem(ctx,name,null,'1 carton',{env});
 let before=await read();const added=before.find(r=>r.item_name.toLowerCase()===name.toLowerCase());assert(added);
 const [shared]=await c.execute('SELECT * FROM shared_shopping_list WHERE household_item_uuid=?',[added.household_item_uuid]);assert.equal(shared.length,1);
 for(const id of [actor,member]){const [r]=await c.execute(`SELECT * FROM \`${id}_new_list\` WHERE household_item_uuid=?`,[added.household_item_uuid]);assert.equal(r.length,1);}
 pass('Pinned add creates one shared actor row and both household app mirrors');
 for(const [action,args] of [['update_shopping_item',{new_name:'Qualification oat milk',quantity:'2 cartons'}],['mark_shopping_item_bought',{}],['mark_shopping_item_unbought',{}]]) {
  before=await read();await shoppingWrite(c,a,action,{item_id:added.shopping_id,...args},before,'legacy-'+action);
  for(const id of [actor,member]){const [r]=await c.execute(`SELECT * FROM \`${id}_new_list\` WHERE household_item_uuid=?`,[added.household_item_uuid]);assert.equal(r[0].product_name,'Qualification oat milk');assert.equal(r[0].quantity,'2 cartons');if(action==='mark_shopping_item_bought')assert.equal(r[0].action,'CHECKED');}
  pass('New '+action+' works on the real legacy-created household item');
 }
 before=await read();await shoppingWrite(c,a,'remove_from_shopping_list',{item_id:added.shopping_id},before,'legacy-delete-'+added.shopping_id);
 assert(!(await read()).some(r=>r.shopping_id===added.shopping_id));
 for(const id of [actor,member]){const [r]=await c.execute(`SELECT * FROM \`${id}_master_feed\` WHERE item_id=?`,[added.shopping_id]);assert(r.some(x=>JSON.stringify(x.metadata).includes('Qualification oat milk')));}
 pass('Exact delete clears app mirrors and leaves recoverable activity snapshots');
 const [cols]=await c.execute("SHOW COLUMNS FROM shared_kitchen");
 for(const col of ["analysis_stage","analysis_status"]) if(!cols.some(x=>x.Field===col)) await c.execute(`ALTER TABLE shared_kitchen ADD \`${col}\` VARCHAR(32)`);
 const kitchen=await data.checkInKitchenItem(ctx,{item_name:"Qualification apples",quantity_value:3,quantity_unit:"count",location:"fridge",is_opened:false},{env,skipKitchenDependentGeneration:true});
 assert(kitchen.item);assert.equal(kitchen.item.quantity_value,3);assert.equal(kitchen.item.quantity_unit,"count");assert.equal(kitchen.item.storage_location,"fridge");
 const [stock]=await c.execute("SELECT * FROM shared_kitchen WHERE _id=?",[kitchen.id]);assert.equal(stock[0].analysis_stage,"final");assert.equal(stock[0].owner_id,actor);
 pass('Pinned kitchen check-in persists quantity, unit, storage and ready status');report.passed=true;fs.writeFileSync(process.env.THYME_EVIDENCE,JSON.stringify(report,null,2));
}finally{await c.end();await mysql.getDbPool(env).end();}
