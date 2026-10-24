// Opt-in paid inference against a privately captured corpus. Never imports a live
// mutation gateway. Supply an absolute private --corpus and --output path; keep
// customer text, household snapshots, and results outside the repository.
import fs from 'node:fs';
import path from 'node:path';
import {ResponsesClient} from '../../shared/thyme-runtime/responses.mjs';
import {MemoryStore} from '../../thyme-agent-v2/src/store.mjs';
import {Service} from '../../thyme-agent-v2/src/service.mjs';
import {Runner} from '../../thyme-agent-v2/src/runner.mjs';
import {ToolGateway,toolDefinitions} from '../../thyme-agent-v2/src/tools.mjs';
import {FixtureGateway} from '../../thyme-agent-v2/test/fixtures.mjs';
import {scope,key,Fault} from '../../thyme-agent-v2/src/core.mjs';

export class SnapshotGateway extends FixtureGateway {
  constructor(snapshot) {
    super();
    this.actorValue=structuredClone(snapshot.actor);
    this.data=structuredClone(snapshot.data);
    this.historyReads=structuredClone(snapshot.historyReads||[]);
    this.env={THYME_RUNTIME_V1:'1',THYME_RUNTIME_ACTORS:JSON.stringify([snapshot.actor.actor]),HOUSEHOLD_BRAIN_V2:'1'};
    this.reads=[];
  }
  async read(a,resource,args={}) {
    await this.check(a);this.reads.push({resource,args});
    if(!Object.hasOwn(this.data,resource))throw new Fault('snapshot_unavailable','This data source was not captured for this replay. Do not treat it as empty.');
    const data=structuredClone(this.data[resource]);
    if(resource==='calendar' && Array.isArray(data))return data.filter(x=>(!args.start_date||x.plan_date>=args.start_date)&&(!args.end_date||x.plan_date<=args.end_date));
    return data;
  }
  async validateMemory(a,fence){await this.check(a);return JSON.stringify(fence)===JSON.stringify(this.data.preferences?.memory?.fence);}
  async history(a,args){
    await this.check(a);
    const normalize=x=>JSON.stringify([x.kind,x.query||'',x.limit??null,x.before||'',x.cursor||'']);
    const captured=this.historyReads.find(x=>normalize(x.args)===normalize(args));
    return captured ? structuredClone(captured.result) : {ok:false,error:'snapshot_unavailable',message:'Exact historical records were not captured for this replay. Do not guess.'};
  }
  async mutate(){this.writes++;throw new Error('REPLAY_WRITE_FORBIDDEN');}
}

export async function replayOne(test,snapshot,{responses,budgetMs=150000}={}) {
  const store=new MemoryStore(),gateway=new SnapshotGateway(snapshot),a=gateway.actorValue;
  const definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions});
  const service=new Service({store,gateway}),runner=new Runner({store,gateway,definitions,tools,responses,budgetMs});
  // A replay only submits text. It never invokes approve/apply, even when the
  // historical text says "yes". The model must prepare a new reviewable change.
  const start=Date.now(),events=[],calls=[];
  const put=store.put.bind(store);store.put=async(...args)=>{
    const result=await put(...args);
    if(result.type==='session')events.push({ms:Date.now()-start,status:result.status,progress:result.progress,activity:result.activity});
    return result;
  };
  const call=tools.call.bind(tools);tools.call=async args=>{
    const begin=Date.now(),result=await call(args);
    calls.push({name:args.action.name,args:args.action.arguments,result,ms:Date.now()-begin,at:begin-start});return result;
  };
  const session=await service.message(a,{requestId:test.id,text:test.text});
  const req=await store.get(scope(a),'Q#'+test.id);
  for(const [i,m] of (test.history||[]).entries())await store.put(scope(a),key(session.id,'M','history-'+i),{
    type:'message',id:'history-'+i,role:m.role,text:m.text,phase:'final_answer',order:req.createdAt-100+i,
  });
  let thrown;
  try{await runner.run(scope(a),test.id);}catch(e){thrown=e.code||e.name;}
  const result=await service.session(a,session.id),request=await store.get(scope(a),'Q#'+test.id);
  const answer=result.messages.filter(m=>m.role==='assistant'&&!m.id.startsWith('history-')).map(m=>m.text).join('\n\n');
  const invariants={completed:result.status==='completed',no_live_writes:gateway.writes===0,
    answer_present:!!answer.trim(),no_raw_payload:!/^\s*\{\s*"(?:reply|message|tool|recipe)/.test(answer),
    mobile_format:!/^\s*\|.*\|\s*$/m.test(answer),
    no_generic_action_rewrite:!answer.startsWith('I could not confirm an account change from this request.'),
    no_unresolved_template:!/<\/?(?:think|tool_call)>|\[object Object\]/.test(answer),
    recipe_structure:result.recipes.every(r=>r.ingredients.length&&r.steps.length&&r.ingredients.every(x=>x.text.trim())&&r.steps.every(x=>x.text.trim())),
    proposals_unapplied:result.proposals.every(p=>p.status==='pending'),
    no_running_final:!(result.activity||[]).some(x=>x.state==='running')};
  return {id:test.id,sourceDate:test.sourceDate,text:test.text,history:test.history,
    snapshotTime:snapshot.capturedAt,elapsedMs:Date.now()-start,status:result.status,error:result.error||thrown||null,
    answer,recipes:result.recipes,proposals:result.proposals,metrics:result.metrics,
    events,calls,reads:gateway.reads,activity:result.activity,invariants,
    // Provider steps help distinguish tool errors from renderer errors; kept private.
    providerSteps:request.direct?.input.filter(x=>x.type==='function_call'||x.type==='message')};
}

async function main(){
  const arg=name=>process.argv[process.argv.indexOf(name)+1];
  if(!process.argv.includes('--corpus')||!process.argv.includes('--output'))throw Error('Use --corpus /private/corpus.json --output /private/results.jsonl');
  const input=arg('--corpus'),output=arg('--output');
  if(!path.isAbsolute(input)||!path.isAbsolute(output))throw Error('Use absolute private paths');
  const corpus=JSON.parse(fs.readFileSync(input));
  const ids=process.argv.includes('--ids')?new Set(arg('--ids').split(',')):null;
  const cases=corpus.cases.filter(x=>!ids||ids.has(x.id));
  const credentials=process.env.THYME_REPLAY_CONFIG?JSON.parse(fs.readFileSync(process.env.THYME_REPLAY_CONFIG)).Environment.Variables:{};
  const apiKey=process.env.OPENAI_API_KEY||credentials.OPENAI_API_KEY;
  if(!apiKey)throw Error('Missing inference credential');
  // Use the deployed consent adapter at request time and before every HTTP call.
  // The local loader must point at that adapter, not a synthetic test stub.
  const consent=(await import('/var/task/ai_consent_guard.cjs')).default;
  const responses=new ResponsesClient(apiKey,{fetcher:consent.providerFetch(fetch,true)});
  const fd=fs.openSync(output,'wx',0o600);
  let cursor=0;
  async function worker(){while(cursor<cases.length){
    const test=cases[cursor++];let result;
    try{result=await replayOne(test,{...corpus.households[test.actor],capturedAt:corpus.capturedAt},{responses});}
    catch(e){result={id:test.id,status:'harness_error',error:e.code||e.name};}
    fs.writeSync(fd,JSON.stringify(result)+'\n');
    console.log(JSON.stringify({id:result.id,status:result.status,seconds:Math.round((result.elapsedMs||0)/100)/10,tools:result.calls?.map(x=>x.name),failed:Object.entries(result.invariants||{}).filter(([,v])=>!v).map(([k])=>k),error:result.error}));
  }}
  await Promise.all([worker(),worker()]);fs.closeSync(fd);
}
if(process.argv[1]===new URL(import.meta.url).pathname)await main();
