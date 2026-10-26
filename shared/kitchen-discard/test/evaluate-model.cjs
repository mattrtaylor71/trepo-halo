// Explicit, synthetic-only provider test; never invoked by node --test.
const fs=require('node:fs'),path=require('node:path');
const {matchIdentity}=require('../identity.cjs');
if (!process.argv.includes('--live') || !process.env.OPENAI_API_KEY) {
  console.error('Use --live and OPENAI_API_KEY to evaluate synthetic cases. No customer mutations are available.');
  process.exit(2);
}
const fixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'model-fixtures.json')));
const create=async(body,options)=>{
  const response=await fetch('https://api.openai.com/v1/responses',{method:'POST',signal:options.signal,
    headers:{'Content-Type':'application/json',Authorization:`Bearer ${process.env.OPENAI_API_KEY}`},body:JSON.stringify(body)});
  if(!response.ok)throw Error('Provider status '+response.status);
  return response.json();
};
(async()=>{
 const results=[];
 for(let i=0;i<fixtures.length;i+=3) {
  results.push(...await Promise.all(fixtures.slice(i,i+3).map(async fixture=>{
   const start=Date.now(),result=await matchIdentity(fixture.input,{create});
   return {id:fixture.id,passed:!!result.matched===fixture.expected_match && (!fixture.expected_match ||
     JSON.stringify([...result.matching_ids].sort())===JSON.stringify([...fixture.expected_ids].sort())),elapsed_ms:Date.now()-start,result};
  })));
 }
 const report={total:results.length,passed:results.filter(x=>x.passed).length,results};
 console.log(JSON.stringify(report,null,2));
 if(report.passed!==report.total)process.exitCode=1;
})().catch(error=>{console.error(error.message);process.exitCode=1;});
