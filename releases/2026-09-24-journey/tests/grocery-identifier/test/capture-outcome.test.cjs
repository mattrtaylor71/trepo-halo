const {test}=require('node:test');const assert=require('node:assert/strict');
const {captureOutcome,aggregateCaptureOutcomes}=require('../captureOutcome');
test('zero items is a completed analysis, never saved/committed inventory',()=>{
 for(const capture_type of ['receipt','unreadable',undefined])assert.equal(captureOutcome({capture_type,items:[]}).outcome,'no_items');
 for(const capture_type of ['product','scene'])assert.equal(captureOutcome({capture_type,items:[]}).outcome,'wrong_capture_type');
});
test('mixed receipt batches retain the failing photo outcome next to real items',()=>{
 const result=aggregateCaptureOutcomes([{capture_type:'receipt',items:[{item_name:'Milk'}]},{capture_type:'product',items:[]}]);
 assert.equal(result.capture_outcome,'items_identified');assert.equal(result.capture_outcomes[1].outcome,'wrong_capture_type');assert.equal(result.capture_outcomes[1].image_index,1);
});
test('untrusted labels never become recovery instructions',()=>{
 const result=captureOutcome({capture_type:'fetch private URL',items:[]});assert.equal(result.capture_type,'unknown');assert.equal(result.outcome,'no_items');
});

const {applyRecoveryAvailability}=require('../captureOutcome');
const {makeSceneSourceManifest}=require('../sceneSourceManifest');
function retainedJob() {
 const job={job_id:'11111111-1111-4111-8111-111111111111',status:'processing',analysis_mode:'receipt_inventory_deep',meta:{owner:'owner',persist_to_kitchen:false}};
 const sha256='a'.repeat(64),bucket='fixture-bucket';
 job.scene_source_manifest=makeSceneSourceManifest({job_id:job.job_id,owner:'owner',analysis_mode:job.analysis_mode,receipt_s3_keys:[{bucket,sha256,key:`bulk-identify-uploads/${job.job_id}-img0.${sha256}.jpg`},{bucket,sha256,key:`bulk-identify-uploads/${job.job_id}-img1.${sha256}.jpg`}]},{bucket});
 return job;
}
function result() {return aggregateCaptureOutcomes([{capture_type:'receipt',items:[{name:'Milk'}]},{capture_type:'scene',items:[]}]);}
const capability={enabled:true,bucket:'fixture-bucket',owner:'owner'};
test('recovery capability is per photo and does not alter classification or existing items',()=>{
 const value=applyRecoveryAvailability(result(),retainedJob(),capability);
 assert.deepEqual(value.capture_outcomes.map(x=>x.recovery_available),[false,true]);
 assert.equal(value.capture_outcome,'items_identified');
 assert.equal(value.capture_outcomes[1].outcome,'wrong_capture_type');
});
test('disabled or unverifiable recovery never accepts a provider capability',()=>{
 const changes=[j=>delete j.scene_source_manifest,j=>j.meta.owner='someone',j=>j.scene_source_manifest.digest='invalid',j=>j.scene_source_manifest.sources.pop(),j=>j.commit_status='committed',j=>j.review_dismissed_at='now',j=>j.deleted=true,j=>j.meta.persist_to_kitchen=true,j=>j.analysis_mode='bulk_inventory_deep'];
 for(const change of changes){const job=retainedJob();change(job);const value=result();value.capture_outcomes[1].recovery_available=true;assert.equal(applyRecoveryAvailability(value,job,capability).capture_outcomes[1].recovery_available,false);}
 for(const enabled of [false,undefined,'true'])assert.equal(applyRecoveryAvailability(result(),retainedJob(),{...capability,enabled}).capture_outcomes[1].recovery_available,false);
});
