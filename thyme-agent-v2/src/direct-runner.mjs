import {detectActionClaims} from '../../playground12_voice_ack/trepo-quick-ack/lib/action-claims.mjs';
import {ResponsesClient} from '../../shared/thyme-runtime/responses.mjs';
import {planTurn,selectTools} from '../../shared/thyme-runtime/policy.mjs';
import {activity,toolActivity,updateActivities} from '../../shared/thyme-runtime/progress.mjs';
import {INSTRUCTIONS} from './instructions.mjs';
import {readConversation} from './conversation.mjs';
import {Fault,hash,key,scope} from './core.mjs';

// Same durable requests, tool receipts, recipe cards and approval path as the
// original runner. Only inference orchestration changes; writes stay in apply().
export async function runDirect(runner,a,initialSession,initialRequest,context,started) {
  let s=initialSession,req=initialRequest;
  const pk=scope(a), env=runner.gateway.env || {};
  const consent=(await import('/var/task/ai_consent_guard.cjs')).default;
  const client=runner.responses || new ResponsesClient(env.OPENAI_API_KEY,{fetcher:consent.providerFetch(fetch,true)});
  const history=(await runner.store.list(pk,key(s.id,'M')+'#')).filter(m=>m.id!==req.id)
    .sort((x,y)=>x.order-y.order).map(m=>({role:m.role,content:m.text}));
  const plan=planTurn(req.text,history);
  const tools=selectTools(runner.definitions,plan);
  const allowed=new Set(tools.map(t=>t.name));
  const deadline=started+Math.min(runner.budgetMs,plan.mode==='quick'?45000:150000);
  const maxRounds=plan.mode==='quick'?6:12;
  const persist=async patch=>{
    const next={...req,...patch};
    if(Buffer.byteLength(JSON.stringify(next))>320000)
      throw new Fault('context_too_large','This request needs a smaller selection. Your saved work is unchanged.',409);
    req=await runner.store.put(pk,'Q#'+req.id,next,req.version);
  };
  const progress=async event=>{
    s=await runner.store.put(pk,key(s.id),{...s,progress:event.label,
      activity:updateActivities(s.activity,event),executionMode:plan.mode},s.version);
  };
  const validate=async()=>{
    await runner.gateway.check(a);
    if(req.memoryFence && !await runner.gateway.validateMemory(a,req.memoryFence))
      throw new Fault('memory_changed','Your household information changed. Check the saved work before continuing.',409);
  };
  if(!req.direct) {
    const proposals=await runner.store.list(pk,key(s.id,'P')+'#');
    const recipes=await runner.store.list(pk,key(s.id,'R')+'#');
    const instructions=INSTRUCTIONS.replaceAll('Use tool_search for', 'Use the supplied tools for')
      .replace('Use tool_search to discover the appropriate deferred request_* tool when needed.', 'Use the supplied request_* tools to prepare changes for review.');
    await persist({direct:{round:0,toolCalls:0,input:[{role:'system',content:instructions},...history,
      {role:'user',content:JSON.stringify({user_message:req.text,current_time:new Date().toISOString(),
        current_preferences:context.preferences.dietary,food_memory:context.preferences.memory,
        confirmed_changes:proposals.filter(p=>p.status==='applied').map(p=>({action:p.action,args:p.args,status:p.status})),
        current_review_requests:{complete:true,items:proposals.map(p=>({id:p.id,action:p.action,status:p.status,title:p.title,validUntil:p.validUntil}))},
        conversation_recipe_versions:recipes.map(r=>({id:r.recipe.id,revision:r.recipe.revision,title:r.recipe.title})),
        current_inventory:context.inventory,
        conversation_preferences:await readConversation(runner.store,a,s.id)})}]}});
  }
  const startingRound=req.direct.round;
  while(req.direct.round-startingRound<maxRounds && runner.clock()<deadline) {
    if(req.direct.round>=30)throw new Fault('task_limit','This task reached its step limit. Your work is saved; start a new request to continue.',409);
    await validate();
    if(!req.direct.pending) {
      await progress(activity('thinking',{requestId:req.id,id:`answer-${req.direct.round}`}));
      const response=await client.complete({model:req.model,input:req.direct.input,tools,deadline,
        requestId:`thyme-direct-${req.id}-${req.direct.round}`,maxOutputTokens:plan.mode==='quick'?3000:6000});
      // Persist provider call IDs BEFORE executing any tools. A retried worker
      // resumes the same calls and their deterministic receipts.
      await persist({direct:{...req.direct,pending:response.output,cursor:0,
        usage:{input_tokens:(req.direct.usage?.input_tokens||0)+(response.usage?.input_tokens||0),
          output_tokens:(req.direct.usage?.output_tokens||0)+(response.usage?.output_tokens||0)},input:[...req.direct.input,...response.output]}});
    }
    const calls=req.direct.pending.filter(x=>x.type==='function_call');
    if(!calls.length) {
      await progress(activity('validating',{requestId:req.id}));
      await validate();
      const latest=await runner.gateway.read(a,'preferences');
      if(hash(latest.dietary)!==req.preferencesHash)
        throw new Fault('preferences_changed','Your food preferences changed. Please ask again using the latest preferences.',409);
      const items=req.direct.pending.filter(x=>x.type==='message').map((x,index)=>({...x,
        id:`direct-${req.id}-${index}`,phase:'final_answer'}));
      if(!items.some(x=>(x.content||[]).some(c=>c.type==='output_text' && c.text?.trim())))
        throw new Fault('empty_answer','Thyme could not finish this answer. Your work is saved.',502);
      // Inference never applies account changes. Historical receipts can explain
      // a prior approval, but can never validate the model's new action or item.
      for(const item of items)for(const content of item.content||[]) {
        const claims=content.type==='output_text' ? detectActionClaims(content.text,{memoryContext:true}) : [];
        if(claims.length) {
          const proposals=await runner.store.list(pk,key(s.id,'P')+'#');
          const receipt=proposals.filter(p=>p.status==='applied' && p.appliedAt && Number.isFinite(new Date(p.appliedAt).getTime()) && p.afterHash && p.verification && p.summary
            && claims.every(c=>c.okTools.has(p.action)))
            .sort((a,b)=>new Date(b.appliedAt).getTime()-new Date(a.appliedAt).getTime())[0];
          content.text=receipt
            ? `Your earlier approved change was confirmed: ${receipt.summary}\n\nThyme has not applied any new account changes in this request.`
            : proposals.some(p=>p.status==='pending')
              ? 'The changes below are ready for your review. Nothing new has been applied in this request.'
              : 'I could not confirm an account change from this request. Thyme has not applied any new account changes.';
        }
      }
      await runner.saveMessages(pk,s,req,items,latest.dietary);
      await runner.finish(pk,s,req,{status:'completed',progress:null,
        activity:(s.activity||[]).map(x=>x.state==='running'?{...x,state:'completed'}:x),
        metrics:{seconds:(runner.clock()-req.createdAt)/1000,model:req.model,toolCalls:req.direct.toolCalls,
          inputTokens:req.direct.usage?.input_tokens??null,outputTokens:req.direct.usage?.output_tokens??null}});
      return;
    }
    for(let i=req.direct.cursor;i<calls.length;i++) {
      await validate();
      if(runner.clock()>=deadline)throw new Fault('timeout','Your completed steps are saved. Resume this request to continue.',504);
      const call=calls[i];
      if(!allowed.has(call.name))throw new Fault('unsupported_action','Thyme requested an unavailable action. No new change was applied.',502);
      let args;try{args=JSON.parse(call.arguments);}catch{throw new Fault('invalid_tool_arguments','Thyme could not prepare that step.',502);}
      const action={...call,arguments:args};
      await progress(toolActivity(call.name,{id:call.call_id,args,requestId:req.id}));
      const result=await runner.tools.call({actor:a,session:s,action,memoryFence:req.memoryFence});
      await persist({direct:{...req.direct,cursor:i+1,toolCalls:req.direct.toolCalls+1,input:[...req.direct.input,
        {type:'function_call_output',call_id:call.call_id,output:JSON.stringify(result)}]}});
      await progress(toolActivity(call.name,{id:call.call_id,state:result.ok?'completed':'failed',requestId:req.id}));
    }
    await persist({direct:{...req.direct,round:req.direct.round+1,pending:null,cursor:0}});
  }
  throw new Fault('timeout','Thyme needs more time. Your request and completed steps are saved.',504);
}
