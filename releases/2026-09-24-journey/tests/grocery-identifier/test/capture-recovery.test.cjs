const {test}=require('node:test');const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {createCaptureRecovery}=require('../captureRecovery');
const sourceId='10000000-0000-4000-8000-000000000001',owner='fixture-owner',bucket='fixture-bucket';
const image=Buffer.from('fixture image'),sha=createHash('sha256').update(image).digest('hex');
function harness(overrides={}){
 const source={job_id:sourceId,status:'completed',analysis_mode:'receipt_inventory_deep',meta:{owner,persist_to_kitchen:false},result:{items:[]},dispatch_digest:'fixture',
 dispatch_payload:{job_id:sourceId,analysis_mode:'receipt_inventory_deep',owner,receipt_s3_keys:[{bucket,key:`bulk-identify-uploads/${sourceId}-img0.${sha}.jpg`,sha256:sha}]},...overrides};
 const jobs=new Map([[sourceId,source]]),calls=[];
 const deps={bucket,authenticate:()=>({userId:owner}),readJob:async id=>jobs.get(id),readImage:async()=>image,
 createChild:async(s,c)=>{if(!jobs.has(c.job_id))jobs.set(c.job_id,c);calls.push('create');},
 admit:async input=>{const c=jobs.get(input.job_id);if(!c.dispatch_digest){c.dispatch_digest='admitted';calls.push('analyze');}assert.equal(c.meta.persist_to_kitchen,false);assert.equal(c.analysis_mode,'bulk_inventory_deep');}};
 return {jobs,calls,source,deps,run:async(body={},custom={})=>{const response=await createCaptureRecovery({...deps,...custom})({body:JSON.stringify({operation:'recover_capture',owner,source_job_id:sourceId,...body})});return {...response,json:JSON.parse(response.body)};}};
}
test('double tap/restart admits one child; review is required and original identity survives',async()=>{
 const h=harness();const [a,b]=await Promise.all([h.run(),h.run()]);assert.equal(a.statusCode,202);assert.equal(b.statusCode,202);assert.equal(a.json.job_id,b.json.job_id);
 assert.equal(a.json.source_job_id,sourceId);assert.equal(a.json.review_required,true);assert.equal(h.calls.filter(c=>c==='analyze').length,1);
 const retry=await h.run();assert.equal(retry.json.job_id,a.json.job_id);assert.equal(h.jobs.size,2);
});
for(const [name,patch,status] of [['foreign',{meta:{owner:'foreign'}},404],['deleted',{deleted:true},404],['committed',{commit_status:'completed'},409],['dismissed',{review_dismissed_at:'today'},409],['unfinished',{status:'processing'},409],['real-items',{result:{items:[{item_name:'Milk'}]}},409],['lost-source',{dispatch_payload:null},410]])
 test(name+' cannot reprocess or mutate inventory',async()=>{const h=harness(patch);assert.equal((await h.run()).statusCode,status);assert.equal(h.jobs.size,1);assert.equal(h.calls.length,0);});
test('authentication failure cannot inspect another source',async()=>{
 const h=harness();assert.equal((await h.run({}, {authenticate:()=>({status:401,code:'authentication_required'}),readJob:async()=>assert.fail('no read')})).statusCode,401);
});
test('expired image offers a retake without creating an orphan',async()=>{
 const h=harness();const r=await h.run({}, {readImage:async()=>{throw Object.assign(Error(),{name:'NoSuchKey'});}});
 assert.equal(r.statusCode,410);assert.equal(r.json.code,'source_unavailable');assert.equal(h.jobs.size,1);
});
test('stored image digest and requested index are verified',async()=>{
 const h=harness();assert.equal((await h.run({image_index:1})).statusCode,400);assert.equal((await h.run({}, {readImage:async()=>Buffer.from('different')})).statusCode,503);assert.equal(h.jobs.size,1);
});
test('mixed batch recovers only its empty photo',async()=>{
 const h=harness({result:{items:[{item_name:'Milk'}],capture_outcomes:[{image_index:0,outcome:'wrong_capture_type'}]}});
 assert.equal((await h.run()).statusCode,202);
});
test('crash after child creation retries that child instead of generating a new ID',async()=>{
 const h=harness();const failed=await h.run({}, {admit:async()=>{throw Error('network');}});assert.equal(failed.statusCode,503);assert.equal(h.jobs.size,2);
 assert.equal((await h.run()).statusCode,202);assert.equal(h.jobs.size,2);assert.equal(h.calls.filter(c=>c==='analyze').length,1);
});

test('completed jobs recover from the retained manifest after dispatch cleanup',async()=>{
 const h=harness();h.source.scene_source_manifest=require('../sceneSourceManifest').makeSceneSourceManifest(h.source.dispatch_payload,{bucket});
 delete h.source.dispatch_payload;delete h.source.dispatch_digest;
 assert.equal((await h.run()).statusCode,202);
});
test('a modified manifest cannot fall back to another dispatch source',async()=>{
 const h=harness();h.source.scene_source_manifest=require('../sceneSourceManifest').makeSceneSourceManifest(h.source.dispatch_payload,{bucket});
 h.source.scene_source_manifest.sources[0].sha256='0'.repeat(64);
 assert.equal((await h.run()).statusCode,409);assert.equal(h.jobs.size,1);
});
