const {authorizeCommit}=require('./bulkCommit/auth');
const {createInbox}=require('./reviewInbox');
const response=(statusCode,body)=>({statusCode,headers:{'Content-Type':'application/json','Access-Control-Allow-Origin':'*'},body:JSON.stringify(body)});
function createHandler(inbox=createInbox(),authenticate=authorizeCommit){
 return async (event,operation='list')=>{
  let input;
  try{input=operation==='list'?event.queryStringParameters||{}:JSON.parse(event.body||'{}');}catch{return response(400,{error:'Invalid request'});}
  const owner=operation==='list'?event.pathParameters?.owner_id:input?.owner;
  if(typeof owner!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(owner))return response(400,{error:'Invalid owner'});
  const auth=authenticate(event,owner);
  if(auth.status)return response(auth.status,{error:auth.code});
  if(auth.userId!==owner)return response(403,{error:'owner_forbidden'});
  try{
   if(operation==='dismiss'){
    if(typeof input.job_id!=='string'||!/^[0-9a-f-]{36}$/i.test(input.job_id))return response(400,{error:'Invalid job'});
    const result=await inbox.dismiss(owner,input.job_id);return response(result.dismissed?200:409,result);
   }
   const limit=input.limit===undefined?20:Number(input.limit);
   const result=await inbox.list(owner,{cursor:input.cursor,limit});return response(200,{owner_id:owner,...result});
  }catch(error){
   if(['Invalid cursor','Invalid limit'].includes(error.message))return response(400,{error:error.message});
   console.error(JSON.stringify({event:'review_inbox_unavailable',operation}));return response(503,{error:'Review temporarily unavailable'});
  }
 };
}
module.exports={createHandler};
