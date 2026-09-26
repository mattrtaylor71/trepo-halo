// Actual serving/candidate handlers against a dedicated, disposable MySQL schema.
// Only connection configuration is redirected. No SQL or handler calls are mocked.
const mysql = require('./serving/node_modules/mysql2/promise');
const path = require('path');
const DB = 'triage_1790374208533179';
const A = '8fcb37c1-0927-4c31-856a-5b761efc226c';
const B = '00000000-0000-4000-8000-000000000002';
const C = '00000000-0000-4000-8000-000000000003';
const ALPHA = '11111111-1111-4111-8111-111111111111';
const TARGET = '22222222-2222-4222-8222-222222222222';
const base = {host:'127.0.0.1',port:19379,user:'root',password:'',database:DB};
const originalPool = mysql.createPool;
const pools=[];
mysql.createPool = options => {
  if (options.host !== base.host || options.database !== DB) throw Error('Non-fixture database refused');
  const pool=originalPool({...options,port:19379}); pools.push(pool); return pool;
};
Object.assign(process.env,{DB_HOST:base.host,DB_USER:'root',DB_PASSWORD:'',DB_NAME:DB,
  LIST_UNION_READ:'true',LIST_STRICT_UUID_MATCH:'true',DUAL_WRITE_ENABLED:'true'});
delete process.env.LIST_UNION_READ_OWNERS;
function handler(candidate=true,strict=true,union=true) {
  process.env.LIST_STRICT_UUID_MATCH=String(strict); process.env.LIST_UNION_READ=String(union);
  const file=path.resolve(__dirname,candidate?'backend/APP/Lambdas/listHandler.js':'serving/listHandler.js');
  delete require.cache[file]; return require(file).handler;
}
async function connect() {return mysql.createConnection(base);}
async function setup() {
  const root=await mysql.createConnection({...base,database:undefined});
  await root.query('CREATE DATABASE IF NOT EXISTS '+DB); await root.end();
  const c=await connect();
  await c.query('CREATE TABLE IF NOT EXISTS new_users (user_id VARCHAR(36) PRIMARY KEY, owner_id VARCHAR(64), created_at DATETIME)');
  await c.query('CREATE TABLE IF NOT EXISTS shared_list (household_item_uuid VARCHAR(36), action VARCHAR(32), updated_at DATETIME)');
  for(const user of [A,B,C]) await c.query(`CREATE TABLE IF NOT EXISTS \`${user}_new_list\` (
    _id BIGINT PRIMARY KEY, _owner VARCHAR(36), _device VARCHAR(64), product_name VARCHAR(255),
    product_brand VARCHAR(255), product_barcode VARCHAR(64), images TEXT, store VARCHAR(100),
    action VARCHAR(32), household_item_uuid VARCHAR(36), _createdDate DATETIME,
    updated_at DATETIME, sort_order INT, aisle_category VARCHAR(40))`);
  await c.end(); await reset();
}
async function reset() {
  const c=await connect();
  try {
    await c.query('DELETE FROM new_users'); await c.query('DELETE FROM shared_list');
    for(const user of [A,B,C]) await c.query(`DELETE FROM \`${user}_new_list\``);
    await c.query("INSERT INTO new_users VALUES (?, 'household-fixture', '2026-01-01'), (?, 'household-fixture', '2026-01-02'), (?, 'foreign-fixture', '2026-01-01')",[A,B,C]);
    for(const [user,id,uuid,name,action,stamp,order] of [
      [A,37,ALPHA,'Triage checked sibling','CHECKED','2026-09-25 12:01:00',0],
      [A,39,TARGET,'Triage collision target','ADDED','2026-09-25 12:00:00',1],
      [B,35,ALPHA,'Triage checked sibling','CHECKED','2026-09-25 12:00:00',0],
      [B,37,TARGET,'Triage collision target','ADDED','2026-09-25 12:02:00',1],
      [C,37,TARGET,'Unrelated household fixture','ADDED','2026-09-25 12:00:00',0]]) {
      await c.execute(`INSERT INTO \`${user}_new_list\` (_id,_owner,_device,product_name,action,household_item_uuid,_createdDate,updated_at,sort_order,store,aisle_category) VALUES (?,?,?,?,?,?,?, ?,?,'Triage test','produce')`,
        [id,user,'triage-fixture',name,action,uuid,'2026-09-25 12:00:00',stamp,order]);
    }
    await c.execute("INSERT INTO shared_list VALUES (?, 'CHECKED', NOW()), (?, 'ADDED', NOW())",[ALPHA,TARGET]);
  } finally {await c.end();}
}
async function state() {
  const c=await connect(); const state={};
  try {for(const [key,user] of [['a',A],['b',B],['foreign',C]]) state[key]=(await c.query(`SELECT _id,household_item_uuid,action,product_name FROM \`${user}_new_list\` ORDER BY _id`))[0];}
  finally {await c.end();} return state;
}
async function close() {await Promise.all(pools.map(p=>p.end()));}
module.exports={A,B,C,ALPHA,TARGET,DB,setup,reset,state,connect,handler,close};
