const test=require('node:test'),assert=require('node:assert/strict');
const {makeSceneSourceManifest,sceneSourceReference}=require('../sceneSourceManifest');
const bucket='fixture-scenes',sha='a'.repeat(64);
const payload=()=>({owner:'owner',job_id:'job',analysis_mode:'bulk_inventory_deep',s3_bucket:bucket,s3_key:'bulk-identify-uploads/job.'+sha+'.jpg',s3_sha256:sha});
function job(p=payload()){return {job_id:p.job_id,meta:{owner:p.owner},status:'completed',analysis_mode:p.analysis_mode,scene_source_manifest:makeSceneSourceManifest(p,{bucket})};}

test('verified single-image manifest survives removal of dispatch payload',()=>{
 const value=job();assert.deepEqual(sceneSourceReference(value,{owner:'owner',bucket}),{bucket,key:payload().s3_key,sha256:sha});
 assert.equal(value.dispatch_payload,undefined);
});
test('single receipt source and one indexed receipt page are both supported',()=>{
 const single={...payload(),analysis_mode:'receipt_inventory_deep'};
 assert.equal(sceneSourceReference(job(single),{owner:'owner',bucket}).key,single.s3_key);
 const page={owner:'owner',job_id:'job',analysis_mode:'receipt_inventory_deep',receipt_s3_keys:[{bucket,key:'bulk-identify-uploads/job-img0.'+sha+'.png',sha256:sha}]};
 assert.equal(sceneSourceReference(job(page),{owner:'owner',bucket}).key,page.receipt_s3_keys[0].key);
});
test('multiple receipt pages retain provenance but no item page is guessed',()=>{
 const p={owner:'owner',job_id:'job',analysis_mode:'receipt_inventory_deep',receipt_s3_keys:[0,1].map(i=>({bucket,key:'bulk-identify-uploads/job-img'+i+'.'+sha+'.jpg',sha256:sha}))};
 const value=job(p);assert.equal(value.scene_source_manifest.sources.length,2);
 assert.equal(sceneSourceReference(value,{owner:'owner',bucket}),null);
});
for(const kind of ['owner','job','mode','bucket','digest','key','status','deleted','dismissed','malformed_deleted'])test('unverified source withheld: '+kind,()=>{
 const j=job(),options={owner:'owner',bucket};
 if(kind==='owner')options.owner='other';
 if(kind==='job')j.job_id='other-job';
 if(kind==='mode')j.analysis_mode='product_analysis';
 if(kind==='bucket')options.bucket='other-bucket';
 if(kind==='digest')j.scene_source_manifest.digest='b'.repeat(64);
 if(kind==='key')j.scene_source_manifest.sources[0].key='other-key';
 if(kind==='status')j.status='pending';
 if(kind==='deleted')j.deleted=true;
 if(kind==='dismissed')j.review_dismissed_at='now';
 if(kind==='malformed_deleted')j.deleted=0;
 assert.equal(sceneSourceReference(j,options),null);
});
test('legacy and anonymous jobs do not acquire invented provenance',()=>{
 assert.equal(makeSceneSourceManifest({...payload(),owner:null},{bucket}),null);
 assert.equal(makeSceneSourceManifest({...payload(),s3_key:'bulk-identify-uploads/job.jpg',s3_sha256:undefined},{bucket}),null);
 assert.equal(sceneSourceReference({...job(),scene_source_manifest:undefined},{owner:'owner',bucket}),null);
});
test('serving writer keeps photo reuse disabled for scanned and manual rows',()=>{
 const fs=require('node:fs'),vm=require('node:vm'),ts=require('typescript');
 const ast=ts.createSourceFile('writer.js',fs.readFileSync(require.resolve('../bulkKitchenWriter'),'utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.JS);
 const scope={estimateStorageGuidance:()=>null,storageZoneToLocation:()=>null,NAME_EMOJI_MAP:{},CATEGORY_EMOJI_MAP:{}};vm.createContext(scope);
 vm.runInContext(ast.statements.filter(ts.isFunctionDeclaration).map(n=>n.getText(ast)).join('\n'),scope);
 const item={product_name:'Milk',source_index:0,manual:false};
 const context={owner:'owner',jobId:'commit',sourceJobId:'job',sourceImageReference:sceneSourceReference(job(),{owner:'owner',bucket})};
 assert.equal(scope.buildKitchenPayload(item,context,0).s3_key,null);
 assert.equal(scope.buildKitchenPayload({...item,manual:true},context,0).s3_key,null);
 assert.equal(scope.buildKitchenPayload({...item,source_index:null},context,0).s3_key,null);
 assert.equal(scope.buildKitchenPayload(item,{...context,sourceImageReference:undefined},0).s3_key,null);
});
test('public job response omits internal source manifest',async()=>{
 const fs=require('node:fs'),vm=require('node:vm'),exports={};
 const scope={exports,console:{log(){},error(){}},require:()=>({getJob:async()=>job()})};
 vm.runInNewContext(fs.readFileSync(require.resolve('../getJob/app'),'utf8'),scope);
 const response=await exports.handler({pathParameters:{job_id:'job'}}),body=JSON.parse(response.body);
 assert.equal(response.statusCode,200);assert.equal(body.scene_source_manifest,undefined);assert.equal(body.status,'completed');
});
