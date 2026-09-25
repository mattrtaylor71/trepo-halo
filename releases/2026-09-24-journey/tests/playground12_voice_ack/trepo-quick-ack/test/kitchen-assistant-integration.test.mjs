import {test,mock,before,after} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {DynamoDBClient,CreateTableCommand,DeleteTableCommand} from '@aws-sdk/client-dynamodb';
process.env.AWS_ENDPOINT_URL_DYNAMODB='http://127.0.0.1:38001';
process.env.AWS_ACCESS_KEY_ID='fixture';process.env.AWS_SECRET_ACCESS_KEY='fixture';process.env.AWS_REGION='us-east-1';
const journalTable='assistant_edits_'+randomUUID().replaceAll('-','');
const dynamo=new DynamoDBClient({endpoint:process.env.AWS_ENDPOINT_URL_DYNAMODB,region:'us-east-1',credentials:{accessKeyId:'fixture',secretAccessKey:'fixture'}});
after(async()=>{await dynamo.send(new DeleteTableCommand({TableName:journalTable}));dynamo.destroy();});
let assistant,row,requests,providerTurns;
const base={_id:'milk',owner_id:'actor',_owner:'actor',product_name:'Milk',action:'IN',storage_location:'pantry',is_opened:0,amount_revision:0};
before(async()=>{
 await dynamo.send(new CreateTableCommand({TableName:journalTable,BillingMode:'PAY_PER_REQUEST',KeySchema:[{AttributeName:'owner_id',KeyType:'HASH'},{AttributeName:'session_entry_id',KeyType:'RANGE'}],AttributeDefinitions:[{AttributeName:'owner_id',AttributeType:'S'},{AttributeName:'session_entry_id',AttributeType:'S'}]}));
 process.env.WRITE_SHARED_ONLY='true';
 const mysqlURL=new URL('../lib/mysql.mjs',import.meta.url),mysql=await import(mysqlURL);
 mock.module(mysqlURL,{namedExports:{...mysql,withDbConnection:async cb=>cb({execute:async(sql)=>{
   assert.match(sql,/^\s*SELECT/i,'Thyme must not directly mutate SQL');
   return [[{...row}]];
 }})}});
 const url=new URL('../lib/data-access.mjs',import.meta.url),data=await import(url);
 const overrides=Object.fromEntries(['getKitchenItems','getKitchenItemsFull','getMealPlan','getRecentDiscards','getRecentDishes','getRecipeSuggestions','getSavedRecipes','getShoppingItems'].map(name=>[name,async()=>[]]));
 mock.module(url,{namedExports:{...data,...overrides,getDietaryPreferences:async()=>({allergies:[],diets:[],religious:[],health:[],custom:[]})}});
 mock.module(new URL('../lib/oura-context.mjs',import.meta.url),{namedExports:{buildOuraGroundingMessage:async()=>null}});
 assistant=await import('../lib/device-assistant.mjs');
});
for(const surface of ['sync','stream'])test(`${surface}: ambiguous product returns its question without claiming a write`,async t=>{
 providerTurns=0;
 t.mock.method(globalThis,'fetch',async(url,options)=>{
   assert.ok(!String(url).includes('/kitchen/'),'No kitchen request is allowed');
   const request=JSON.parse(options.body);
   const message=providerTurns++===0?{content:'',tool_calls:[{index:0,id:'ambiguous',type:'function',function:{name:'check_in_many_items',arguments:JSON.stringify({items:[{item_name:'Garlic oil'},{item_name:'Olive oil'}]})}}]}:{content:'Added both oils.'};
   return new Response(request.stream?`data: ${JSON.stringify({choices:[{delta:message}]})}\n\ndata: [DONE]\n\n`:JSON.stringify({choices:[{message}]}));
 });
 const input={transcript:'Garlic oil, olive oil',userContext:{ownerId:'house',userId:'actor'},env:{ACTION_MODE:'real',OPENAI_API_KEY:'fixture',SESSION_TABLE_NAME:journalTable},responseSurface:'app'};
 let result;
 if(surface==='sync')result=await assistant.runDeviceAssistant(input);
 else{const generator=assistant.runDeviceAssistantStreaming(input);while(true){const n=await generator.next();if(n.done){result=n.value;break;}}}
 assert.match(result.text,/one flavored oil, or two separate products\?/);
 assert.equal(result.outcome.actions,'none');
 assert.equal(result.outcome.answer,'needs_clarification');
});
for(const surface of ['sync','stream'])test(`${surface}: actual assistant and wrapper preserve auth/identity and serialize two edits`,async t=>{
 row={...base};requests=[];providerTurns=0;
 t.mock.method(globalThis,'fetch',async(url,options)=>{
   const request=JSON.parse(options.body);
   if(String(url).includes('/kitchen/')){
     assert.equal(options.headers.Authorization,'Bearer fixture-auth');
     assert.equal(request.items[0].revision,row.amount_revision);
     requests.push(request);
     await new Promise(resolve=>setTimeout(resolve,10));
     row={...row,...request.items[0].fields,amount_revision:row.amount_revision+1};
     return Response.json({operation_id:request.operation_id,items:[{item:row}]});
   }
   const tool=(name,args,index)=>({index,id:`call-${index}`,type:'function',function:{name,arguments:JSON.stringify(args)}});
   const message=providerTurns++===0?{content:'',tool_calls:[tool('update_item_location',{item_id:'milk',location:'fridge'},0),tool('mark_item_opened',{item_id:'milk'},1)]}:{content:'Moved Milk to the fridge and marked it opened.'};
   return new Response(request.stream?`data: ${JSON.stringify({choices:[{delta:message}]})}\n\ndata: [DONE]\n\n`:JSON.stringify({choices:[{message}]}));
 });
 const input={transcript:'Move the milk to the fridge and mark it opened',userContext:{ownerId:'house',userId:'actor',tableOwnerId:'actor',householdMemberIds:['actor']},env:{ACTION_MODE:'real',OPENAI_API_KEY:'fixture',SESSION_TABLE_NAME:journalTable},responseSurface:'app',kitchenRequest:{operationId:'request-123-'+surface,authorization:'Bearer fixture-auth'}};
 let result;
 if(surface==='sync')result=await assistant.runDeviceAssistant(input);
 else{const generator=assistant.runDeviceAssistantStreaming(input);while(true){const n=await generator.next();if(n.done){result=n.value;break;}}}
 assert.equal(row.storage_location,'fridge');assert.equal(row.is_opened,true);
 assert.equal(row.amount_revision,2);assert.equal(requests.length,2);
 assert.notEqual(requests[0].operation_id,requests[1].operation_id);
 assert.equal(result.outcome.actions,'confirmed');
});
