const test=require('node:test'),assert=require('node:assert/strict');
const {Readable}=require('node:stream'),{createHash}=require('node:crypto');
const {createSceneSourceLoader}=require('../sceneSourceLoader');
const image=Buffer.from('fixture-image'),sha256=createHash('sha256').update(image).digest('hex');
const reference=()=>({bucket:'fixture-scenes',key:'bulk-identify-uploads/job.'+sha256+'.jpg',sha256});
function fixture(response,options={}) {
  const calls=[];
  const load=createSceneSourceLoader({bucket:'fixture-scenes',client:{send:async(command,opts)=>{calls.push({command,opts});return response;}},...options});
  return {load,calls};
}

test('only configured immutable object is requested and digest checked',async()=>{
  const body=Readable.from([image.subarray(0,4),image.subarray(4)]),f=fixture({Body:body,ContentLength:image.length});
  assert.deepEqual(await f.load(reference()),image);
  assert.deepEqual(f.calls[0].command.input,{Bucket:'fixture-scenes',Key:reference().key});
  assert.ok(f.calls[0].opts.abortSignal);assert.equal(body.destroyed,true);
});

for(const kind of ['bucket','url','key','digest','extra'])test('invalid reference rejected before request: '+kind,async()=>{
  const f=fixture({}),source=reference();
  if(kind==='bucket')source.bucket='other-bucket';
  if(kind==='url')source.key='https://fixture.invalid/image.jpg';
  if(kind==='key')source.key='bulk-identify-uploads/job.jpg';
  if(kind==='digest')source.sha256='b'.repeat(64);
  if(kind==='extra')source.url='https://fixture.invalid';
  await assert.rejects(f.load(source),/Invalid admitted/);assert.equal(f.calls.length,0);
});

for(const length of [0,-1,1.2,21*1024*1024,'13'])test('invalid declared size closes stream: '+length,async()=>{
  const body=Readable.from([image]),f=fixture({Body:body,ContentLength:length});
  await assert.rejects(f.load(reference()),/size is invalid/);assert.equal(body.destroyed,true);
});

test('missing length still enforces streaming bound',async()=>{
  const body=Readable.from([image]),f=fixture({Body:body},{maxBytes:image.length-1});
  await assert.rejects(f.load(reference()),/size limit/);assert.equal(body.destroyed,true);
});
test('lying small length cannot bypass streaming bound',async()=>{
  const body=Readable.from([image]),f=fixture({Body:body,ContentLength:1},{maxBytes:image.length-1});
  await assert.rejects(f.load(reference()),/size limit/);assert.equal(body.destroyed,true);
});
test('truncated stream rejected even with matching bytes hash',async()=>{
  const f=fixture({Body:Readable.from([image]),ContentLength:image.length+1});
  await assert.rejects(f.load(reference()),/length mismatch/);
});
test('wrong bytes rejected',async()=>{
  const f=fixture({Body:Readable.from([Buffer.from('wrong')])});
  await assert.rejects(f.load(reference()),/digest mismatch/);
});
test('empty stream rejected',async()=>{
  await assert.rejects(fixture({Body:Readable.from([])}).load(reference()),/length mismatch/);
});
test('stream error propagates and closes body',async()=>{
  const body=new Readable({read(){this.destroy(Error('fixture stream failure'));}});
  await assert.rejects(fixture({Body:body}).load(reference()),/fixture stream failure/);assert.equal(body.destroyed,true);
});
test('aborted caller opens no request',async()=>{
  const c=new AbortController();c.abort();const f=fixture({});
  await assert.rejects(f.load(reference(),{signal:c.signal}),{name:'AbortError'});assert.equal(f.calls.length,0);
});
test('hanging stream times out and is destroyed',async()=>{
  const body=new Readable({read(){}}),f=fixture({Body:body},{timeoutMs:15});
  await assert.rejects(f.load(reference()),{name:'AbortError'});assert.equal(body.destroyed,true);
});
test('parent abort cancels an active stream',async()=>{
  const c=new AbortController(),body=new Readable({read(){}}),f=fixture({Body:body});
  const pending=f.load(reference(),{signal:c.signal});setTimeout(()=>c.abort(),5);
  await assert.rejects(pending,{name:'AbortError'});assert.equal(body.destroyed,true);
});
test('late ignored-abort response body is closed',async()=>{
  let resolve;const body=Readable.from([image]);
  const load=createSceneSourceLoader({bucket:'fixture-scenes',timeoutMs:5,client:{send:()=>new Promise(r=>resolve=r)}});
  await assert.rejects(load(reference()),{name:'AbortError'});
  resolve({Body:body});await new Promise(r=>setImmediate(r));assert.equal(body.destroyed,true);
});
test('missing content length is accepted only after bounded digest verification',async()=>{
  assert.deepEqual(await fixture({Body:Readable.from([image])}).load(reference()),image);
});
test('caller cannot change digest while source read is pending',async()=>{
  let resolve;const source=reference(),wrong=Buffer.from('wrong');
  const load=createSceneSourceLoader({bucket:'fixture-scenes',client:{send:()=>new Promise(r=>resolve=r)}});
  const pending=load(source);source.sha256=createHash('sha256').update(wrong).digest('hex');
  resolve({Body:Readable.from([wrong])});
  await assert.rejects(pending,/digest mismatch/);
});
test('elapsed deadline rejects synchronous late completion even before timer fires',async()=>{
  const body=Readable.from([image]);
  const load=createSceneSourceLoader({bucket:'fixture-scenes',timeoutMs:5,client:{send:async()=>{
    const end=Date.now()+25;while(Date.now()<end){};
    return {Body:body};
  }}});
  await assert.rejects(load(reference()),{name:'AbortError'});assert.equal(body.destroyed,true);
});
