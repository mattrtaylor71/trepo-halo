// Offline assertions over private replay evidence. No inference or customer writes.
// Expectations live alongside the private corpus, not in source control.
import fs from 'node:fs';
import path from 'node:path';

const arg=name=>process.argv[process.argv.indexOf(name)+1];
if(!['--corpus','--results','--output'].every(x=>process.argv.includes(x)))
  throw Error('Use --corpus /private/corpus.json --results /private/run.jsonl[,/private/retry.jsonl] --output /private/assessment.json');
const files=[arg('--corpus'),...arg('--results').split(','),arg('--output')];
if(!files.every(path.isAbsolute))throw Error('Use absolute private paths');
const corpus=JSON.parse(fs.readFileSync(files[0])),latest=new Map();
for(const file of arg('--results').split(','))for(const line of fs.readFileSync(file,'utf8').split('\n').filter(Boolean)) {
  const row=JSON.parse(line);latest.set(row.id,row);
}
const cases=corpus.cases.map(test=>{
  const r=latest.get(test.id),expect=test.expect||{},checks={recorded:!!r};
  if(!r)return {id:test.id,checks};
  Object.assign(checks,r.invariants);
  const calls=r.calls||[],proposals=r.proposals||[];
  if(expect.noProposals)checks.no_proposals=proposals.length===0;
  if(expect.minRecipes!=null)checks.minimum_recipes=r.recipes.length>=expect.minRecipes;
  if(expect.toolsAny)checks.intent_tool_used=calls.some(c=>expect.toolsAny.includes(c.name));
  if(expect.successfulTool)checks.intent_tool_succeeded=calls.some(c=>c.name===expect.successfulTool&&c.result?.ok===true);
  if(expect.forbiddenTools)checks.forbidden_tool_absent=!calls.some(c=>expect.forbiddenTools.includes(c.name));
  // Progress is a record of execution: at no point may a single step have two
  // states, and every emitted tool completion must follow its running event.
  const started=new Set();let ordered=true,unique=true;
  for(const event of r.events||[]) {
    const steps=event.activity||[];
    if(new Set(steps.map(s=>s.id)).size!==steps.length)unique=false;
    for(const step of steps) {
      if(step.stage!=='tool')continue;
      if(step.state==='running')started.add(step.id);
      else if(!started.has(step.id))ordered=false;
    }
  }
  checks.unique_progress_steps=unique;checks.tool_progress_order=ordered;
  const failures=Object.entries(checks).filter(([,v])=>!v).map(([k])=>k);
  return {id:test.id,status:r.status,elapsedMs:r.elapsedMs,checks,failures,
    toolErrors:calls.filter(c=>c.result?.ok===false).map(c=>({name:c.name,error:c.result.error}))};
});
const times=cases.filter(r=>r.status==='completed').map(r=>r.elapsedMs).sort((a,b)=>a-b);
const report={generatedAt:new Date().toISOString(),snapshotTime:corpus.capturedAt,
  cases:cases.length,accounts:new Set(corpus.cases.map(c=>c.actor)).size,
  withHistory:corpus.cases.filter(c=>c.history?.length).length,
  completed:cases.filter(c=>c.status==='completed').length,
  casesWithIntentExpectations:corpus.cases.filter(c=>c.expect).length,
  failed:cases.filter(c=>!c.checks.recorded||c.failures?.length).map(c=>c.id),
  medianMs:times.length?times[Math.floor(times.length/2)]:null,
  p95Ms:times.length?times[Math.ceil(times.length*.95)-1]:null,
  maxMs:times.at(-1)||null,
  notes:['Frozen snapshot reads, not live app latency. Completion and assertions do not establish universal answer accuracy.',
    'Intent expectations complement manual grounding review; they do not replace it.'],results:cases};
fs.writeFileSync(arg('--output'),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
console.log(JSON.stringify({...report,results:undefined}));
if(report.failed.length)process.exitCode=1;
