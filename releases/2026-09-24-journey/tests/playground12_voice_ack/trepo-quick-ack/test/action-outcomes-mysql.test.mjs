import {test} from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import mysql from 'mysql2/promise';
import {executeToolAction} from '../lib/tool-actions.mjs';
import {getDbConfig,getDbPool} from '../lib/mysql.mjs';
import {reconcileActionNarration} from '../lib/action-narration.mjs';

test('real local add/remove returns durable IDs; new connections see exactly one change', {skip:!process.env.TREPO_VOICE_IMAGE_MYSQL},async()=>{
 assert.equal(process.env.DB_HOST,'127.0.0.1');assert.match(process.env.DB_NAME,/^thyme_test_/);
 const id=crypto.randomUUID(), userContext={ownerId:id,userId:id,tableOwnerId:id,householdMemberIds:[id]};
 const env={...process.env,ACTION_MODE:'real'};
 const args={item_name:'Fixture milk',quantity:'1 carton'};
 let connection;
 try {
  const result=await executeToolAction({toolName:'add_to_shopping_list',args,env,userContext,responseSurface:'app'});
  assert.equal(result.ok,true,JSON.stringify(result));assert.ok(result.toolResult.item.shopping_id);
  const reply=await reconcileActionNarration({text:'Added three cartons of juice to your shopping list.',toolEvents:[{toolName:'add_to_shopping_list',args,result}]});
  assert.match(reply.text,/Fixture Milk \(1 carton\)/);assert.doesNotMatch(reply.text,/juice|three/);
  connection=await mysql.createConnection(getDbConfig());
  const [rows]=await connection.execute(`SELECT _id,product_name,quantity FROM \`${id}_new_list\``);
  assert.equal(rows.length,1);assert.equal(String(rows[0]._id),result.toolResult.item.shopping_id);assert.equal(rows[0].quantity,'1 carton');
  await connection.end();connection=null;
  const removal=await executeToolAction({toolName:'remove_from_shopping_list',args:{item_name:'Fixture milk'},env,userContext,responseSurface:'app'});
  assert.equal(removal.ok,true,JSON.stringify(removal));assert.equal(removal.toolResult.item.shopping_id,result.toolResult.item.shopping_id);
  const removedReply=await reconcileActionNarration({text:'Removed bread from your shopping list.',toolEvents:[{toolName:'remove_from_shopping_list',result:removal}]});
  assert.match(removedReply.text,/Fixture Milk/);assert.doesNotMatch(removedReply.text,/bread/);
  connection=await mysql.createConnection(getDbConfig());const [after]=await connection.execute(`SELECT _id FROM \`${id}_new_list\``);assert.equal(after.length,0);
  const repeat=await executeToolAction({toolName:'remove_from_shopping_list',args:{item_name:'Fixture milk'},env,userContext,responseSurface:'app'});
  assert.equal(repeat.ok,false);assert.equal(repeat.statusCode,404);
 } finally {if(connection)await connection.end();await getDbPool().end();}
});
