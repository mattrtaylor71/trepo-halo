// Real Node resolver -> actual Python canonical transaction -> authoritative readback.
// Only the transport is substituted; both sides use the isolated MySQL fixture.
import mysql from 'mysql2/promise';
import {DynamoDBClient,CreateTableCommand,DeleteTableCommand} from '@aws-sdk/client-dynamodb';
import {spawn} from 'node:child_process';
import {mock} from 'node:test';
import {createKitchenMutationContext} from '../../lib/kitchen-edit-client.mjs';
import assert from 'node:assert/strict';

const [schema, mode, worker, python] = process.argv.slice(2);
assert.match(schema,/^quantity_fixture_[a-f0-9]+$/);
process.env.AWS_ENDPOINT_URL_DYNAMODB='http://127.0.0.1:38001';
process.env.AWS_ACCESS_KEY_ID='fixture';process.env.AWS_SECRET_ACCESS_KEY='fixture';process.env.AWS_REGION='us-east-1';
const dynamo=new DynamoDBClient({endpoint:process.env.AWS_ENDPOINT_URL_DYNAMODB,region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'}});
await dynamo.send(new CreateTableCommand({TableName:schema,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'owner_id',KeyType:'HASH'},{AttributeName:'session_entry_id',KeyType:'RANGE'}],AttributeDefinitions:[{AttributeName:'owner_id',AttributeType:'S'},{AttributeName:'session_entry_id',AttributeType:'S'}]}));
const connection=await mysql.createConnection({host:'127.0.0.1',port:33317,user:'root',database:schema});
const mysqlModuleURL=new URL('../../lib/mysql.mjs',import.meta.url),mysqlModule=await import(mysqlModuleURL);
mock.module(mysqlModuleURL,{namedExports:{...mysqlModule,withDbConnection:async cb=>cb(connection)}});
const {updateKitchenItemDetails}=await import('../../lib/data-access.mjs');
let requests=0;
globalThis.fetch=async (url, options) => {
  assert.match(url,/\/kitchen\/owner\?amount_operation=v1$/);
  requests++;
  if (mode==='stale') await connection.execute("UPDATE shared_kitchen SET brand='Concurrent edit' WHERE _id='beef'");
  const result=await new Promise((resolve,reject) => {
    const child=spawn(python,[worker,schema],{stdio:['pipe','pipe','pipe'],env:{...process.env,TREPO_SHELF_LIFECYCLE_ENABLED:'false'}});
    let output='',error='';
    child.stdout.on('data',chunk=>output+=chunk);child.stderr.on('data',chunk=>error+=chunk);
    child.on('error',reject);child.on('close',code=>code?reject(Error(error)):resolve(JSON.parse(output)));
    child.stdin.end(options.body);
  });
  if (mode.startsWith('fresh-') && requests===1 && result.status===200) throw Error('fixture lost commit acknowledgment');
  return Response.json(result.body,{status:result.status});
};
const context={ownerId:'household',userId:'owner',tableOwnerId:'owner',householdMemberIds:['owner','member'],responseSurface:'app'};
const options={connection,env:{SESSION_TABLE_NAME:schema},mutationContext:createKitchenMutationContext({operationId:'fixture-request'})};
try {
  let item,error;
  try {
    if (mode.startsWith('fresh-')) {
      options.sourceTranscript='Rename beef to Garlic Olive Oil';
      const ref={item_id:'beef',item_name:'beef'},changes={new_name:'Garlic Olive Oil'};
      await assert.rejects(updateKitchenItemDetails(context,ref,changes,options),/lost commit acknowledgment/);
      if(mode==='fresh-later-edit'){
        await connection.execute("UPDATE kitchen_item_edits SET overrides=JSON_SET(overrides,'$.product_name','Later human name') WHERE item_id='beef'");
        await connection.execute("UPDATE shared_kitchen SET product_name='Later human name' WHERE _id='beef'");
      }
      if(mode==='fresh-deleted') await connection.execute("DELETE FROM shared_kitchen WHERE _id='beef'");
      const retry={...options,mutationContext:createKitchenMutationContext({operationId:'fixture-request'})};
      item=await updateKitchenItemDetails(context,ref,mode==='fresh-conflict'?{new_name:'Other name'}:changes,retry);
      assert.equal(item.item_name,'Garlic Olive Oil');
      const [receipts]=await connection.execute("SELECT COUNT(*) AS n FROM kitchen_amount_operations WHERE actor='owner'");
      assert.equal(receipts[0].n,1);
    } else {
    const fresh=['duplicate','newest','unrelated-id','shortening'].includes(mode);
    if(fresh)options.sourceTranscript=mode==='newest'?'Rename the newest Beef Broth to Organic Beef Broth':mode==='shortening'?'Rename Beef Broth to Broth':'Rename Beef Broth to Organic Beef Broth';
    item=await updateKitchenItemDetails(context,{item_id:mode==='foreign'?'private':mode==='unrelated-id'?'eggs':'beef',...(fresh?{item_name:'Beef Broth'}:{})},{new_name:fresh?(mode==='shortening'?'Broth':'Organic Beef Broth'):'Garlic Olive Oil'},options);
    if (mode==='retry') assert.deepEqual(await updateKitchenItemDetails(context,{item_id:'beef'},{new_name:'Garlic Olive Oil'},options),item);
    }
  } catch (e) { error={status:e.statusCode,message:e.message}; }
  console.log(JSON.stringify({item,error,requests}));
} finally { await connection.end();await dynamo.send(new DeleteTableCommand({TableName:schema}));dynamo.destroy(); }
