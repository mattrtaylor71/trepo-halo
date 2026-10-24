import test from 'node:test';
import assert from 'node:assert/strict';
class Client {}
test.mock.module('@aws-sdk/client-lambda',{namedExports:{LambdaClient:Client,InvokeCommand:Client}});
test.mock.module('@aws-sdk/client-s3',{namedExports:{S3Client:Client,GetObjectCommand:Client,PutObjectCommand:Client}});
const {createHandler,privacyKey}=await import('./index.mjs');
function fixture(overrides={}) {
  const brain={fence:{household:'house',generation:1,states:[]},context:{coverage:{learning_paused:false}}};
  const raw=owner=>({statusCode:200,owner,bundle:{snapshot:{ownerId:owner}},brain,
    context:{owner,status:'ready',scope:'personal',list:[],preferences:[]},dietary:{}});
  const calls=[],saved=[];
  const handler=createHandler({config:()=>({token:'secret-test'}),clock:()=>1700000000000,
    readState:async()=>({entries:Object.fromEntries(['a','b'].map(owner=>[owner,{userId:owner,householdId:'house',anchor:'not-matching'}]))}),
    invoke:async req=>{calls.push(req);return overrides.reply ? overrides.reply(req,raw):req.operation==='validate'?{statusCode:200,current:true}:raw(req.owner)},
    writeCache:async (owner,value)=>saved.push({owner,value}),project:async()=>{throw Error('must not project unmatched state')},...overrides});
  return {handler,calls,saved,raw,brain};
}
test('one unavailable owner retries once without replaying a successful owner',async()=>{
 const f=fixture({reply:(req,raw)=>req.owner==='a'?{statusCode:503,code:'BRAIN_UNAVAILABLE'}:req.operation==='validate'?{statusCode:200,current:true}:raw(req.owner)});
 assert.deepEqual(await f.handler({operation:'refresh'}),{refreshed:1,failed:1});
 assert.equal(f.calls.filter(c=>c.owner==='a').length,2);assert.deepEqual(f.saved.map(s=>s.owner),['b']);
});
test('changed validation starts a new snapshot and only publishes the revalidated one',async()=>{
 let validations=0;
 const f=fixture({readState:async()=>({entries:{a:{userId:'a',householdId:'house',anchor:''}}}),reply:(req,raw)=>req.operation==='validate'?(++validations===1?{statusCode:409,current:false,code:'CONTEXT_CHANGED'}:{statusCode:200,current:true}):raw(req.owner)});
 assert.deepEqual(await f.handler({operation:'refresh'}),{refreshed:1,failed:0});
 assert.deepEqual(f.calls.map(c=>c.operation),['snapshot','validate','snapshot','validate']);assert.equal(f.saved.length,1);
});
for(const code of ['FORBIDDEN','SCOPE_CHANGED']) test(`${code} never retries or writes stale cache`,async()=>{
 const f=fixture({reply:(req,raw)=>code==='FORBIDDEN'?{statusCode:403,code}:({...raw(req.owner),brain:{fence:{household:'other'}}})});
 assert.deepEqual(await f.handler({operation:'refresh'}),{refreshed:0,failed:2});assert.equal(f.calls.length,2);assert.equal(f.saved.length,0);
});
test('deadline preserves a failure outcome before another source call',async()=>{
 const f=fixture();assert.deepEqual(await f.handler({operation:'refresh'},{getRemainingTimeInMillis:()=>20000}),{refreshed:0,failed:2});assert.equal(f.calls.length,0);
});
test('state enumeration failure still fails the invocation, never claims zero failures',async()=>{
 const f=fixture({readState:async()=>{throw Error('connection')}});await assert.rejects(f.handler({operation:'refresh'}),/Context refresh failed/);
});
test('GET cannot serve cached context after current membership validation fails',async()=>{
 const owner='a'.repeat(64);const f=fixture({reply:(req,raw)=>req.operation==='validate'?{statusCode:409,current:false,code:'CONTEXT_CHANGED'}:raw(req.owner),readCache:async()=>({owner,at:1700000000000,anchor:'old',context:{}})});
 const result=await f.handler({headers:{authorization:'Bearer secret-test'},requestContext:{http:{method:'GET'}},rawPath:'/v1/context/'+owner});
 assert.equal(result.statusCode,409);assert.equal(JSON.parse(result.body).code,'CONTEXT_CHANGED');assert.equal(f.saved.length,0);
});
