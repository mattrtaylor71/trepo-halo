import { test, before, after, beforeEach, mock } from 'node:test';
import assert from 'node:assert/strict';
import mysql from 'mysql2/promise';
import { resolve, dirname } from 'node:path';

// Explicitly opt into a disposable local server; never use application DB env.
const enabled = process.env.RELIABILITY_MYSQL_PORT === '33387';
const integration = (name, fn) => test(name, { skip: !enabled }, fn);
let conn, da;
const database = `reliability_household_${process.pid}`;
const context = () => ({ userId:'actor', tableOwnerId:'actor', ownerId:'family', householdMemberIds:['foreign'], responseSurface:'halo' });
const options = { skipKitchenDependentGeneration:true };
before(async () => {
  if (!enabled) return;
  conn = await mysql.createConnection({ host:'127.0.0.1', port:33387, user:'root', multipleStatements:true });
  await conn.query(`CREATE DATABASE \`${database}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`);
  await conn.query(`USE \`${database}\``);
  await conn.query(`CREATE TABLE new_users (user_id VARCHAR(36) PRIMARY KEY, owner_id VARCHAR(36), INDEX(owner_id)) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
    INSERT INTO new_users VALUES ('actor','family'),('partner','family'),('foreign','other');
    CREATE TABLE shared_kitchen (_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36),product_name TEXT,action VARCHAR(8),_createdDate DATETIME,_updatedDate DATETIME,brand TEXT,category TEXT,images TEXT,product_expiration DATETIME,job_id TEXT,quantity_value DECIMAL(12,3),quantity_unit VARCHAR(30),remaining_quantity TEXT,fill_percent INT);
    CREATE TABLE shared_discards (_id VARCHAR(36) PRIMARY KEY,_owner VARCHAR(36),_device TEXT,_createdDate DATETIME,_updatedDate DATETIME,product_name TEXT,brand TEXT,category TEXT,images TEXT,action VARCHAR(8),product_expiration DATETIME,job_id TEXT,user_id VARCHAR(36),source_kitchen_id VARCHAR(36),discard_reason TEXT,owner_id VARCHAR(36));`);
  process.env.WRITE_SHARED_ONLY='true';
  const dataModule=process.env.RELIABILITY_DATA_ACCESS || resolve(import.meta.dirname,'../lib/data-access.mjs');
  const mysqlModule=resolve(dirname(dataModule),'mysql.mjs');
  const original=await import(mysqlModule);
  mock.module(mysqlModule,{namedExports:{...original,withDbConnection:async callback=>callback(conn)}});
  da=await import(dataModule);
});
beforeEach(async () => {
  if (!enabled) return;
  await conn.query(`DELETE FROM shared_kitchen; DELETE FROM shared_discards;
    UPDATE new_users SET owner_id='family' WHERE user_id IN ('actor','partner');
    INSERT INTO shared_kitchen (_id,owner_id,product_name,action,_createdDate,_updatedDate) VALUES
    ('own','actor','Rice','IN',NOW(),NOW()),('partner-yogurt','partner','Mango yogurt','IN',NOW(),NOW()),
    ('other-yogurt','partner','Mango yogurt','IN',NOW(),NOW()),('outside','foreign','Private sauce','IN',NOW(),NOW());`);
});
after(async () => { if (conn) { await conn.query(`DROP DATABASE \`${database}\``); await conn.end(); } });
integration('partner exact item is discarded atomically and copies reach both household members',async()=>{
  const result=await da.discardKitchenItem(context(),{item_id:'partner-yogurt'},'finished',options);
  assert.equal(result.item.item_name,'Mango yogurt');
  const [rows]=await conn.query('SELECT _id,action FROM shared_kitchen ORDER BY _id');
  assert.equal(rows.find(r=>r._id==='partner-yogurt').action,'OUT');
  assert.equal(rows.find(r=>r._id==='other-yogurt').action,'IN');
  for (const table of ['shared_discards','actor_discards','partner_discards']) {
    const [copies]=await conn.query(`SELECT _id FROM \`${table}\` WHERE _id=?`,[result.discard_id]);
    assert.equal(copies.length,1,table);
  }
});
integration('foreign id cannot fall back to an accessible item with the supplied name',async()=>{
  await assert.rejects(()=>da.discardKitchenItem(context(),{item_id:'outside',item_name:'Rice'},'finished',options));
  const [rows]=await conn.query("SELECT COUNT(*) AS n FROM shared_kitchen WHERE action='OUT'");assert.equal(rows[0].n,0);
});
integration('moved household member and cached foreign member never authorize a mutation',async()=>{
  await conn.query("UPDATE new_users SET owner_id='other' WHERE user_id='partner'");
  await assert.rejects(()=>da.discardKitchenItem(context(),{item_id:'partner-yogurt'},'finished',options));
  await assert.rejects(()=>da.discardKitchenItem(context(),{item_id:'outside'},'finished',options));
});
integration('discard sink failure rolls back kitchen removal',async()=>{
  await conn.query("CREATE TRIGGER fail_discard BEFORE INSERT ON shared_discards FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='fixture sink failure'");
  try { await assert.rejects(()=>da.discardKitchenItem(context(),{item_id:'own'},'finished',options),/fixture sink failure/); }
  finally { await conn.query('DROP TRIGGER fail_discard'); }
  const [rows]=await conn.query("SELECT action FROM shared_kitchen WHERE _id='own'");assert.equal(rows[0].action,'IN');
});
integration('partner quantity edit and deletion guard use the same scope',async()=>{
  await da.updateKitchenItemQuantity(context(),{item_id:'partner-yogurt'},{quantity_value:2,quantity_unit:'count'},options);
  const [rows]=await conn.query("SELECT quantity_value FROM shared_kitchen WHERE _id='partner-yogurt'");assert.equal(Number(rows[0].quantity_value),2);
  await conn.query("UPDATE shared_kitchen SET action='OUT' WHERE _id='partner-yogurt'");
  await assert.rejects(()=>da.updateKitchenItemQuantity(context(),{item_id:'partner-yogurt'},{quantity_value:3,quantity_unit:'count'},options));
});
integration('a household transfer cannot commit while a kitchen mutation holds its membership locks',async()=>{
  const { withHouseholdKitchenMutation, updateHouseholdKitchenRow } = await import('../lib/household-kitchen-mutation.mjs');
  const transfer = await mysql.createConnection({host:'127.0.0.1',port:33387,user:'root',database});
  let transferDone=false, pending;
  try {
    await withHouseholdKitchenMutation(conn,context(),async scope=>{
      pending=transfer.query("UPDATE new_users SET owner_id='other' WHERE user_id='partner'").then(()=>{transferDone=true;});
      await new Promise(resolve=>setTimeout(resolve,75));
      assert.equal(transferDone,false);
      await updateHouseholdKitchenRow(conn,scope,'partner-yogurt',{action:'OUT'});
    });
    await pending;
    await assert.rejects(()=>da.discardKitchenItem(context(),{item_id:'other-yogurt'},'finished',options));
  } finally { await transfer.end(); }
});
integration('shopping receipts resolve UUID primary keys instead of insertId zero',async()=>{
  await conn.query(`CREATE TABLE IF NOT EXISTS actor_new_list (_id CHAR(36) PRIMARY KEY DEFAULT (UUID()), _owner VARCHAR(36), _device VARCHAR(64), product_name VARCHAR(255), action VARCHAR(32));
    CREATE TABLE IF NOT EXISTS shared_shopping_list (_id BIGINT PRIMARY KEY AUTO_INCREMENT,_owner VARCHAR(36),owner_id VARCHAR(36),_device VARCHAR(64),product_name TEXT,quantity TEXT,product_brand TEXT,images TEXT,product_barcode TEXT,store TEXT,action VARCHAR(32),_createdDate DATETIME,created_at DATETIME,updated_at DATETIME,household_item_uuid CHAR(36));`);
  const result=await da.addShoppingItem({...context(),householdMemberIds:['actor','partner']},'Yogurt','Groceries',1);
  const [saved]=await conn.query('SELECT _id FROM actor_new_list WHERE household_item_uuid=?',[result.household_item_uuid]);
  assert.equal(saved.length,1);assert.equal(result.shopping_id,saved[0]._id);assert.notEqual(result.shopping_id,'0');
  const batch=await da.addManyShoppingItems({...context(),householdMemberIds:['actor','partner']},[{item_name:'Bananas',store:'Groceries'}]);
  const [items]=await conn.query('SELECT _id FROM actor_new_list WHERE household_item_uuid=?',[batch.items[0].household_item_uuid]);
  assert.equal(batch.items[0].shopping_id,items[0]._id);
});
integration('legacy household root with null owner_id can access its verified partner',async()=>{
  await conn.query("UPDATE new_users SET owner_id=NULL WHERE user_id='actor'");
  await conn.query("UPDATE new_users SET owner_id='actor' WHERE user_id='partner'");
  await da.discardKitchenItem({...context(),ownerId:'actor'},{item_id:'partner-yogurt'},'finished',options);
  const [rows]=await conn.query("SELECT action FROM shared_kitchen WHERE _id='partner-yogurt'");
  assert.equal(rows[0].action,'OUT');
});
