import test from 'node:test';
import assert from 'node:assert/strict';
import {householdBrain,shoppingBrain} from '../../playground12_voice_ack/trepo-quick-ack/lib/household-brain-client.mjs';
import {TrepoGateway} from '../src/gateway.mjs';
import {toolDefinitions,ToolGateway} from '../src/tools.mjs';
import {MemoryStore} from '../src/store.mjs';
import {Service} from '../src/service.mjs';
import {Runner} from '../src/runner.mjs';
import {scope,key} from '../src/core.mjs';
import {FixtureGateway,ACTOR} from './fixtures.mjs';

const env={HOUSEHOLD_BRAIN_V2:'1',HOUSEHOLD_BRAIN_FUNCTION:'private-reader'};
const a={actor:'a',household:'h'};
const response=()=>({statusCode:200,context:{schema:'household-brain-v2',explicit_preferences:{},observations:{}},
    fence:{schema:'household-brain-v2',actor:'a',household:'h',audience:'a',generation:1,revision:1,expires:Date.now()/1000+300}});

test('private adapter refuses disabled, foreign, expired and failed responses',async()=>{
  assert.equal(await householdBrain({actor:'a',household:'h'},{},()=>{throw Error('must not invoke')}),null);
  const request={actor:'a',household:'h',audience:'personal'};
  assert.ok(await householdBrain(request,env,async()=>response()));
  for(const patch of [{actor:'b'},{household:'other'},{audience:null},{expires:1}]) {
    const r=response();Object.assign(r.fence,patch);
    assert.equal(await householdBrain(request,env,async()=>r),null);
  }
  assert.equal(await householdBrain(request,env,async()=>({statusCode:503})),null);
});

test('shopping consumes the same authenticated household context',async()=>{
  let sent;
  const r=await shoppingBrain({userId:'a',householdId:'h'},env,async request=>{sent=request;return response()});
  assert.equal(r.context.schema,'household-brain-v2');
  assert.deepEqual(sent,{operation:'context',actor:'a',household:'h',audience:'personal',purpose:'shopping'});
});

test('Thyme reads the central brain, rejects customer controls and omits its memory write tool',async()=>{
  const gateway=new TrepoGateway({env});gateway.check=async()=>a;gateway.brainTransport=async()=>response();
  gateway.definitions=async()=>[];
  assert.equal((await gateway.memory(a)).enrolled,true);
  assert.equal((await toolDefinitions(gateway)).some(t=>t.name==='request_update_food_memory'),false);
  await assert.rejects(gateway.controlMemory(a,{},'op',{}),{code:'internally_managed'});
  gateway.brainTransport=async()=>null;
  const fallback=await gateway.memory(a);
  assert.equal(fallback.available,false);assert.deepEqual(fallback.observations,{});
});

test('recovered response validates original memory boundary after a reset',async()=>{
  const store=new MemoryStore(),gateway=new FixtureGateway();gateway.env=env;
  const prior={schema:'household-brain-v2',revision:1},fresh={schema:'household-brain-v2',revision:2};
  const read=gateway.read.bind(gateway);
  gateway.read=async(a,resource,args)=>resource==='preferences'?{dietary:{},memory:{fence:fresh}}:read(a,resource,args);
  let validated;
  gateway.validateMemory=async(a,f)=>{validated=f;return false};
  const definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions});
  const service=new Service({store,gateway}),pk=scope(ACTOR);
  const session=await service.message(ACTOR,{requestId:'brain-recovery',text:'What should I cook?'});
  let req=await store.get(pk,'Q#brain-recovery');
  await store.put(pk,'Q#brain-recovery',{...req,memoryFence:prior,submitted:true,baselineTurns:['old'],preferencesHash:null},req.version);
  let s=await store.get(pk,key(session.id));
  await store.put(pk,key(session.id),{...s,providerId:'provider'},s.version);
  const provider={turns:async()=>[{id:'old',status:'completed'},{id:'new',status:'completed'}],
    session:async()=>({required_actions:[]}),items:async()=>{throw Error('must not publish stale answer')}};
  await new Runner({store,gateway,tools,definitions,provider}).run(pk,'brain-recovery');
  assert.deepEqual(validated,prior);
  req=await store.get(pk,'Q#brain-recovery');assert.equal(req.status,'failed');
});
