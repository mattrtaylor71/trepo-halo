// Opt-in paid inference with a synthetic in-memory household. No live account tools.
import {ResponsesClient} from '../../shared/thyme-runtime/responses.mjs';
import {MemoryStore} from '../../thyme-agent-v2/src/store.mjs';
import {Service} from '../../thyme-agent-v2/src/service.mjs';
import {Runner} from '../../thyme-agent-v2/src/runner.mjs';
import {ToolGateway,toolDefinitions} from '../../thyme-agent-v2/src/tools.mjs';
import {FixtureGateway,ACTOR} from '../../thyme-agent-v2/test/fixtures.mjs';
import {scope} from '../../thyme-agent-v2/src/core.mjs';
for(const [i,text] of ['What condiments do I have?','Use only my kitchen eggs and spinach to make breakfast for two.','Add milk to my shopping list.'].entries()){
 const store=new MemoryStore(),gateway=new FixtureGateway();gateway.env={THYME_RUNTIME_V1:'1',THYME_RUNTIME_ACTORS:JSON.stringify([ACTOR.actor])};
 gateway.data.kitchen.push({id:'mustard',item_name:'Dijon mustard',quantity_value:null},{id:'soup',item_name:'Chicken soup',quantity_value:null});
 const definitions=await toolDefinitions(gateway),tools=new ToolGateway({store,gateway,definitions}),service=new Service({store,gateway});
 const responses=new ResponsesClient(process.env.OPENAI_API_KEY),runner=new Runner({store,gateway,definitions,tools,responses});
 const id='synthetic-'+i,start=Date.now();
 try{
  const session=await service.message(ACTOR,{requestId:id,text});await runner.run(scope(ACTOR),id);
  const result=await service.session(ACTOR,session.id);
  console.log(JSON.stringify({scenario:i,elapsed_ms:Date.now()-start,status:result.status,error:result.error,metrics:result.metrics,activity:result.activity,messages:result.messages,recipes:result.recipes,proposals:result.proposals,writes:gateway.writes}));
 }catch(e){console.log(JSON.stringify({scenario:i,elapsed_ms:Date.now()-start,error:e.code||e.name}));}
}
