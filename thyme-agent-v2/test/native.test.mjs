import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeHandler,voiceBytes,nativeSession} from '../src/native.mjs';
import {MemoryStore} from '../src/store.mjs';
import {Service} from '../src/service.mjs';
import {ACTOR,FixtureGateway} from './fixtures.mjs';
import {hash,scope} from '../src/core.mjs';
const context=hash([ACTOR.actor,ACTOR.household]);
function wav(seconds=1){const b=Buffer.alloc(44+seconds*32000);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(16000,24);b.writeUInt32LE(32000,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(b.length-44,40);return b;}
function setup(options={}){
 const store=new MemoryStore(),gateway=new FixtureGateway(),service=new Service({store,gateway});let starts=0,transcriptions=0;
 const handler=createNativeHandler({env:{NATIVE_PILOT_ACTORS:JSON.stringify([ACTOR.actor])},verifyActor:async()=>({actor:ACTOR.actor,owner:ACTOR.actor}),runtime:async()=>{starts++;return {store,gateway,service};},transcribeAudio:async()=>{transcriptions++;return 'Add milk';},...options});
 const call=async(input={},event={})=>{const r=await handler({requestContext:{http:{method:'POST'}},headers:{'content-type':'application/json'},body:JSON.stringify({scope:context,...input}),...event});return {...JSON.parse(r.body),httpStatus:r.statusCode};};
 return {store,gateway,call,counts:()=>({starts,transcriptions})};
}
test('native rejects unsigned, expired and non-enrolled accounts before runtime',async()=>{
 for(const status of [401,403,503]){const r=setup({verifyActor:async()=>{throw {statusCode:status};}});assert.equal((await r.call()).httpStatus,status);assert.equal(r.counts().starts,0);}
 const r=setup({verifyActor:async()=>({actor:'someone-else',owner:'someone-else'})});assert.equal((await r.call()).httpStatus,403);assert.equal(r.counts().starts,0);
});
test('native prevents client owner overrides and mismatched owner header',async()=>{
 const r=setup();for(const key of ['actor','owner','ownerId','household','principal'])assert.equal((await r.call({op:'bootstrap',[key]:'other'})).httpStatus,403);
 assert.equal(r.counts().starts,0);
 const other=setup({verifyActor:async()=>({actor:ACTOR.actor,owner:'other-household'})});assert.equal((await other.call()).httpStatus,403);assert.equal(other.counts().starts,0);
});
test('native validates request framing before account operations',async()=>{
 const r=setup();assert.equal((await r.call({}, {requestContext:{http:{method:'GET'}}})).httpStatus,405);
 assert.equal((await r.call({}, {headers:{}})).httpStatus,415);
 assert.equal((await r.call({}, {isBase64Encoded:true})).httpStatus,413);
 assert.equal((await r.call({}, {body:'x'.repeat(2700001)})).httpStatus,413);
 assert.equal((await r.call({}, {body:'['})).httpStatus,400);assert.equal(r.counts().starts,0);
});
test('native enrollment reads signed membership and requires its scope on subsequent calls',async()=>{
 const r=setup();assert.equal((await r.call({op:'enrollment',scope:undefined})).scope,context);
 for(const invalid of [undefined,'old-household'])assert.equal((await r.call({op:'message',text:'hello',requestId:'request-123456',scope:invalid})).httpStatus,409);
 assert.equal((await r.store.list(scope(ACTOR), "")).length,0);
 r.gateway.actorValue={...ACTOR,household:'moved'};assert.equal((await r.call({op:'bootstrap'})).httpStatus,409);
});
test('native message preserves durable request identity across lost acknowledgement',async()=>{
 const r=setup(),input={op:'message',text:'Hello',requestId:'native-request-12345'};
 const first=await r.call(input),again=await r.call(input);assert.equal(first.httpStatus,200);assert.equal(first.id,again.id);assert.equal(again.messages.length,1);
 assert.equal((await r.call({...input,text:'changed'})).httpStatus,409);
 assert.equal(r.gateway.writes,0);
});
test('native response omits provider IDs, actor, table keys and active request internals',()=>{
 const clean=nativeSession({id:'s',title:'Hi',messages:[],recipes:[],proposals:[],providerId:'secret',pk:'secret',actor:'secret',activeRequest:'secret'});
 assert.deepEqual(Object.keys(clean).sort(),['id','messages','proposals','recipes','title']);
});
test('voice accepts 0.5–60s PCM without regex stack overflow',()=>{for(const n of [.5,1,60])assert.deepEqual(voiceBytes(wav(n).toString('base64')),wav(n));});
test('voice rejects bad framing, duration, channels, sample rate and encoded junk',()=>{
 for(const n of [.25,61])assert.throws(()=>voiceBytes(wav(n).toString('base64')));
 for(const [offset,value] of [[22,2],[24,44100],[28,64000],[32,4],[34,8],[20,3]]){const b=wav();b.writeUInt16LE(value,offset);assert.throws(()=>voiceBytes(b.toString('base64')));}
 for(const s of ['', 'AA==x',wav().toString('base64')+'\n','a'.repeat(2600004)])assert.throws(()=>voiceBytes(s));
 const b=wav();b.writeUInt32LE(999,4);assert.throws(()=>voiceBytes(b.toString('base64')));
});
test('voice is replayable without retranscription and never writes kitchen/list',async()=>{
 const r=setup(),input={op:'transcribe',requestId:'native-audio-12345',audio:wav().toString('base64')};
 assert.equal((await r.call(input)).text,'Add milk');assert.equal((await r.call(input)).text,'Add milk');assert.equal(r.counts().transcriptions,1);assert.equal(r.gateway.writes,0);
 const changed=wav();changed[100]=1;assert.equal((await r.call({...input,audio:changed.toString('base64')})).httpStatus,409);
});
test('voice request validation precedes paid transcription',async()=>{const r=setup();assert.equal((await r.call({op:'transcribe',requestId:'native-audio-12345',audio:'oops'})).httpStatus,400);assert.equal(r.counts().transcriptions,0);});
test('voice quota blocks extra paid work',async()=>{
 const r=setup(),day=new Date().toISOString().slice(0,10);await r.store.put(scope(ACTOR),'AB#'+day,{count:60});
 assert.equal((await r.call({op:'transcribe',requestId:'native-audio-12345',audio:wav().toString('base64')})).httpStatus,429);assert.equal(r.counts().transcriptions,0);
});
test('voice cannot return data after household membership changes mid transcription',async()=>{
 let r;r=setup({transcribeAudio:async()=>{r.gateway.actorValue={...ACTOR,household:'changed'};return 'Test';}});
 assert.equal((await r.call({op:'transcribe',requestId:'native-audio-12345',audio:wav().toString('base64')})).httpStatus,403);
});
test('concurrent same recording starts only one transcription',async()=>{
 const r=setup(),input={op:'transcribe',requestId:'native-audio-12345',audio:wav().toString('base64')};
 const results=await Promise.all([r.call(input),r.call(input)]);assert.ok(results.some(x=>x.httpStatus===200));assert.equal(r.counts().transcriptions,1);
});
