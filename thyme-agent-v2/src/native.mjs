// Separate authenticated native entry point. The public/legacy and Sites routes are unchanged.
import {pathToFileURL} from 'node:url';
import {getRuntime} from './entry.mjs';
import {Fault, hash, scope, checkId, publicError} from './core.mjs';
import {record} from './store.mjs';

const json=(statusCode,body)=>({statusCode,headers:{'content-type':'application/json','cache-control':'no-store','x-content-type-options':'nosniff'},body:JSON.stringify(body)});
const lib=()=>process.env.TREPO_TOOLS_ROOT+'/playground12_voice_ack/trepo-quick-ack/lib/';
const verify=async(e,env)=>(await import(pathToFileURL(lib()+'operation-recovery.mjs'))).verifiedActor(e,env);
const transcribe=async(bytes,env)=>(await import(pathToFileURL(lib()+'device-assistant.mjs'))).transcribeAudio(bytes,{...env,OPENAI_TIMEOUT_MS:'25000',OPENAI_MAX_RETRIES:'0',GEMINI_API_KEY:''},{sampleRate:16000});
const header=(e,name)=>Object.entries(e.headers||{}).find(([k])=>k.toLowerCase()===name)?.[1];

export function voiceBytes(value){
 if(typeof value!=='string'||value.length>2600000||!value.length||value.length%4!==0||!/^[A-Za-z0-9+/]*={0,2}$/.test(value))throw new Fault('audio','Record up to 60 seconds of audio.',400);
 const b=Buffer.from(value,'base64');
 if(b.toString('base64')!==value)throw new Fault('audio','Please record that again.',400);
 if(b.length<44||b.length>1921000||b.toString('ascii',0,4)!=='RIFF'||b.toString('ascii',8,12)!=='WAVE'||b.readUInt32LE(4)+8!==b.length)throw new Fault('audio','Please record that again.',400);
 let format=false,data=false;
 for(let i=12;i+8<=b.length;){const n=b.readUInt32LE(i+4),end=i+8+n;if(end>b.length)throw new Fault('audio','Please record that again.',400);
  const kind=b.toString('ascii',i,i+4);
  if(kind==='fmt '){if(format||n<16||b.readUInt16LE(i+8)!==1||b.readUInt16LE(i+10)!==1||b.readUInt32LE(i+12)!==16000||b.readUInt32LE(i+16)!==32000||b.readUInt16LE(i+20)!==2||b.readUInt16LE(i+22)!==16)throw new Fault('audio','Please record that again.',400);format=true;}
  if(kind==='data'){if(data||n<16000||n>1920000||n%2)throw new Fault('audio','Record between half a second and 60 seconds.',400);data=true;}
  i=end+(n%2);
 }
 if(!format||!data)throw new Fault('audio','Please record that again.',400);return b;
}

export function nativeSession(s){
 return Object.fromEntries(['id','title','status','model','messages','recipes','proposals','progress','progressDetails','error','metrics'].filter(k=>s[k]!==undefined).map(k=>[k,s[k]]));
}

export function createNativeHandler({verifyActor=verify,runtime=getRuntime,transcribeAudio=transcribe,env=process.env,clock=Date.now}={}){
 return async event=>{
  try{
   if((event.requestContext?.http?.method||event.httpMethod)!=='POST')return json(405,{message:'Use POST.'});
   if(!String(header(event,'content-type')||'').toLowerCase().startsWith('application/json'))throw new Fault('content_type','Send JSON.',415);
   if(event.isBase64Encoded||typeof event.body!=='string'||Buffer.byteLength(event.body)>2700000)throw new Fault('body','Request is too large.',413);
   // No runtime initialization, transcription or account lookup before signed identity + enrollment.
   let proof;try{proof=await verifyActor(event,env);}catch(e){throw new Fault(e.code||'sign_in_required',e.statusCode===503?'Thyme is temporarily unavailable.':'Please sign in to Trepo again.',e.statusCode||401);}
   const enrolled=JSON.parse(env.NATIVE_PILOT_ACTORS||'[]');
   if(!Array.isArray(enrolled)||!enrolled.includes(proof.actor))throw new Fault('not_enrolled','This private trial is not enabled for your account.',403);
   if(proof.owner!==proof.actor)throw new Fault('scope','Use your signed-in account.',403);
   let input;try{input=JSON.parse(event.body);}catch{throw new Fault('body','Invalid request.',400);}
   if(!input||Array.isArray(input)||typeof input!=='object'||['actor','owner','ownerId','household','principal'].some(k=>Object.hasOwn(input,k)))throw new Fault('scope','The account comes from your secure sign-in.',403);
   const rt=await runtime(),a=await rt.gateway.actor(proof.actor);await rt.gateway.check(a);
   const context=hash([a.actor,a.household]);
   if(input.op==='enrollment')return json(200,{enabled:true,scope:context});
   if(input.scope!==context)throw new Fault('scope_changed','Your household changed. Start a new conversation.',409);
   if(input.op==='transcribe'){
    const bytes=voiceBytes(input.audio),rid=checkId(input.requestId),pk=scope(a),sk='A#'+rid,intent=hash(bytes.toString('base64'));
    const old=await rt.store.get(pk,sk);
    if(old&&old.intent!==intent)throw new Fault('request_conflict','This recording changed. Record again.',409);
    if(old?.text)return json(200,{text:old.text,scope:context});
    if(old&&old.leaseUntil>clock())throw new Fault('busy','Your recording is still being transcribed. Try again shortly.',409);
    const day=new Date(clock()).toISOString().slice(0,10),qk='AB#'+day,quota=await rt.store.get(pk,qk);
    if((quota?.count||0)>=60)throw new Fault('limit','The private trial has reached today’s voice limit.',429);
    const row=record(pk,sk,{type:'native_audio',intent,leaseUntil:clock()+45000,expires:Math.floor(clock()/1000)+86400},old?.version??null);
    await rt.store.transaction([{item:row,expected:old?.version??null},{item:record(pk,qk,{type:'audio_budget',count:(quota?.count||0)+1},quota?.version??null),expected:quota?.version??null}]);
    const result=await transcribeAudio(bytes,env),text=String(typeof result==='string'?result:result?.text||'').trim();
    if(!text||text.length>12000)throw new Fault('audio','I couldn’t hear that clearly. Please try again.',400);
    await rt.gateway.check(a);
    await rt.store.put(pk,sk,{...row,text,leaseUntil:0},row.version);
    return json(200,{text,scope:context});
   }
   const result=await rt.service.dispatch(a,input);
   return json(200,{...(input.op==='bootstrap'?result:input.op==='feedback'?result:nativeSession(result)),scope:context});
  }catch(e){return json(e.status||503,publicError(e));}
 };
}
export const handler=createNativeHandler();
