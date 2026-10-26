const {test,before,after,beforeEach}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const mysql=require('node:module').createRequire(path.resolve(__dirname,'../../../playground12_voice_ack/trepo-quick-ack/package.json'))('mysql2/promise');
const {createPhotoMatcher}=require('../../../playground6_groceryRec/analyze_discard_on_upload_nodejs/discard_matcher.cjs');
const {resolveDecision}=require('../identity.cjs');
const enabled=process.env.RELIABILITY_MYSQL_PORT==='33387';
const integration=(name,fn)=>test(name,{skip:!enabled},fn);
const database=`discard_photo_${process.pid}`;
const config={host:'127.0.0.1',port:33387,user:'root',multipleStatements:true,timezone:'Z'};
let conn,matcher,plan,failDelete;
before(async()=>{
 if(!enabled)return;
 conn=await mysql.createConnection(config);
 await conn.query(`CREATE DATABASE \`${database}\`; USE \`${database}\`;
 CREATE TABLE new_users (user_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36));
 INSERT INTO new_users VALUES ('alice','family'),('bob','family'),('outside','other');
 CREATE TABLE shared_kitchen (_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36),product_name TEXT,action VARCHAR(8),_createdDate DATETIME,_updatedDate DATETIME,quantity_value DECIMAL(12,3));
 CREATE TABLE shared_archive_kitchen LIKE shared_kitchen;
 ALTER TABLE shared_archive_kitchen ADD archived_at DATETIME,ADD archived_reason VARCHAR(255),ADD archived_from_table VARCHAR(255);
 CREATE TABLE kitchen_item_edits (item_id VARCHAR(36) PRIMARY KEY,revision BIGINT DEFAULT 0,overrides JSON);`);
 matcher=createPhotoMatcher({connect:async()=>{
   const c=await mysql.createConnection({...config,database});const execute=c.execute.bind(c);
   c.execute=async(sql,args)=>{if(failDelete && sql.startsWith('DELETE'))throw Error('simulated deletion failure');return execute(sql,args);};return c;
  },members:async c=>{const [rows]=await c.execute("SELECT user_id FROM new_users WHERE owner_id='family' ORDER BY user_id");return rows.map(r=>r.user_id);},
  load:async()=>plan,retain:async(j,o,p)=>plan ||= p,
  match:async input=>resolveDecision(input,{status:'matched',matching_ids:['old','new'],scope:'one_entry',selection:'oldest',reason:'same fruit'})});
});
beforeEach(async()=>{
 if(!enabled)return;plan=null;failDelete=false;
 await conn.query(`DELETE FROM shared_kitchen;DELETE FROM shared_archive_kitchen;DELETE FROM kitchen_item_edits;
 UPDATE new_users SET owner_id='family' WHERE user_id IN ('alice','bob');
 INSERT INTO shared_kitchen VALUES ('old','bob','Avocado','IN','2026-10-01','2026-10-20',1),('new','alice','Avocado','IN','2026-10-20','2026-10-20',1),('private','outside','Avocado','IN','2026-09-01','2026-09-01',1);`);
});
after(async()=>{if(conn){await conn.query(`DROP DATABASE \`${database}\``);await conn.end();}});
const call=remove=>matcher('alice',{product_name:'Avocado'},{jobId:'photo-job',remove});
integration('photo MySQL: oldest member item removed, archive and idempotency verified',async()=>{
 assert.equal((await call(false))._id,'old');assert.equal((await call(true)).removed,true);
 assert.equal((await call(true)).replayed,true);
 const [rows]=await conn.query('SELECT _id FROM shared_kitchen ORDER BY _id');assert.deepEqual(rows.map(r=>r._id),['new','private']);
 const [archive]=await conn.query('SELECT _id,archived_reason FROM shared_archive_kitchen');assert.equal(archive.length,1);assert.equal(archive[0].archived_reason,'discard_match:photo-job');
});
integration('photo MySQL: edit revision cancels a frozen match',async()=>{
 await call(false);await conn.query("INSERT INTO kitchen_item_edits VALUES ('old',1,JSON_OBJECT('__acquired_on','2026-10-25'))");
 assert.equal((await call(true)).removed,false);const [rows]=await conn.query('SELECT COUNT(*) n FROM shared_kitchen');assert.equal(rows[0].n,3);
});
integration('photo MySQL: archive insertion rolls back with failed deletion',async()=>{
 await call(false);failDelete=true;assert.equal((await call(true)).removed,false);
 const [rows]=await conn.query('SELECT COUNT(*) n FROM shared_kitchen');assert.equal(rows[0].n,3);
 const [archive]=await conn.query('SELECT COUNT(*) n FROM shared_archive_kitchen');assert.equal(archive[0].n,0);
});
integration('photo MySQL: transferred household cannot remove frozen target',async()=>{
 await call(false);await conn.query("UPDATE new_users SET owner_id='other' WHERE user_id='bob'");
 assert.equal((await call(true)).removed,false);const [rows]=await conn.query('SELECT COUNT(*) n FROM shared_kitchen');assert.equal(rows[0].n,3);
});
