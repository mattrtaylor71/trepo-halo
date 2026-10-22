import {test} from 'node:test';
import assert from 'node:assert/strict';
import {operationScope,claimOperation,readOperation,finishOperation,STABLE_ADMISSION_SINCE} from '../lib/operation-recovery.mjs';
import {prepareBrainSession} from '../lib/family-brain.mjs';
const actor='00000000-0000-4000-8000-000000000001';
const created=Math.max(Math.floor(Date.now()/1000),STABLE_ADMISSION_SINCE);
const identity={id:'v1.'+created+'.00000000-0000-4000-8000-000000000002',expires:created+86400,sessionId:'test-session',fingerprint:'request'};
const scope=(namespace='scope-before',i=identity)=>operationScope(i,{actor,owner:actor},{userId:actor,ownerId:'home',isFallbackContext:false},namespace,{SESSION_TABLE_NAME:'test-only'});
function database(){let row;return {send:async command=>{const x=command.input;switch(command.constructor.name){
 case 'GetCommand':return {Item:row};
 case 'PutCommand':if(row)throw Object.assign(new Error(),{name:'ConditionalCheckFailedException'});row=x.Item;return {};
 case 'UpdateCommand':row={...row,request_status:'completed',response_body:x.ExpressionAttributeValues[':body']};return {};
 default:throw Error('Unexpected command');
}}};}
test('same operation reuses the saved receipt after activity changes',async()=>{
 const db=database(),s=scope();const first=await claimOperation(s,db);assert.equal(first.claimed,true);
 await finishOperation(s,first,{text:'Logged oatmeal',app_output:{cards:[{entity:{id:'saved-dish'}}]}},db);
 const retry=await claimOperation(scope(),db);assert.equal(retry.claimed,false);assert.equal(retry.status,'completed');assert.equal(retry.body.app_output.cards[0].entity.id,'saved-dish');
});
test('scope changes cannot create a second admission or reveal the old receipt',async()=>{
 const db=database(),s=scope();const first=await claimOperation(s,db);await finishOperation(s,first,{text:'private old answer'},db);
 const changed=scope('changed-household-or-controls');assert.deepEqual(changed.key,s.key);
 const retry=await claimOperation(changed,db);assert.equal(retry.claimed,false);assert.equal(retry.status,'scope_changed');assert.equal(retry.body,undefined);
});
test('pre-upgrade IDs never start again under the new namespace',async()=>{
 if(!STABLE_ADMISSION_SINCE)return;
 const db=database();const old=scope('scope',{...identity,expires:STABLE_ADMISSION_SINCE+86400-1});assert.equal((await claimOperation(old,db)).status,'pre_upgrade_request');
});
test('legacy brief retry namespace survives ordinary input changes, not controls',async()=>{
 const env={HOUSEHOLD_BRAIN_V2:'1',FAMILY_BRAIN_FUNCTION:'test'};
 const credentials={authorization:'Bearer e30.'+Buffer.from(JSON.stringify({user_id:actor})).toString('base64url')+'.sig',owner:actor,surface:'app'};
 const brain=(revision,dietary_hash='diet')=>({statusCode:200,context:{schema:'household-brain-v2',coverage:{},activity:{dishes:revision}},fence:{actor,household:'home',generation:1,revision,members_hash:'members',dietary_hash,states:[]}});
 const a=await prepareBrainSession(credentials,env,async()=>brain(1));const b=await prepareBrainSession(credentials,env,async()=>brain(2));const c=await prepareBrainSession(credentials,env,async()=>brain(2,'changed'));
 assert.equal(a.requestOwner,b.requestOwner);assert.notEqual(b.requestOwner,c.requestOwner);
});
test('reusing an operation ID with different input cannot execute again',async()=>{
 const db=database(),s=scope();await claimOperation(s,db);
 const retry=await claimOperation({...scope(),fingerprint:'different request'},db);assert.equal(retry.claimed,false);assert.equal(retry.status,'input_conflict');
});
test('an in-flight duplicate receives processing status without executing',async()=>{
 const db=database(),s=scope();await claimOperation(s,db);
 const retry=await claimOperation(scope(),db);assert.equal(retry.claimed,false);assert.equal(retry.status,'in_progress');
});
