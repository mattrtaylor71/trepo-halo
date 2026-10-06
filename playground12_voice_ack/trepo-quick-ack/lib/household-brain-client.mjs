// Private service adapter shared by Thyme and the server-side shopping relay.
// The caller obtains actor/household from its authenticated session, never model args.
const clients = new Map();
export async function householdBrain(request, env=process.env, transport=null) {
  if (env.HOUSEHOLD_BRAIN_V2 !== '1' || !env.HOUSEHOLD_BRAIN_FUNCTION) return null;
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),Number(env.HOUSEHOLD_BRAIN_TIMEOUT_MS || 3000));
  try {
    let response;
    if (transport) response=await transport(request,controller.signal);
    else {
      const {LambdaClient,InvokeCommand}=await import('@aws-sdk/client-lambda');
      const region=env.AWS_REGION || 'us-east-1';
      if(!clients.has(region))clients.set(region,new LambdaClient({region,maxAttempts:1}));
      const result=await clients.get(region).send(new InvokeCommand({FunctionName:env.HOUSEHOLD_BRAIN_FUNCTION,
        InvocationType:'RequestResponse',Payload:Buffer.from(JSON.stringify(request))}),{abortSignal:controller.signal});
      if(result.FunctionError)return null;
      response=JSON.parse(Buffer.from(result.Payload).toString());
    }
    if (response?.statusCode!==200)return null;
    if(request.operation==='validate')return {current:response.current===true};
    const f=response.fence;
    if(response.context?.schema!=='household-brain-v2' || f?.actor!==request.actor || f?.household!==request.household
      || f.audience!==(request.audience==='household'?null:request.actor) || !Number.isFinite(f.expires) || f.expires*1000<=Date.now())return null;
    return response;
  } catch {
    console.warn(JSON.stringify({event:'household_brain',outcome:'unavailable'}));
    return null;
  } finally {clearTimeout(timer);}
}

export async function shoppingBrain(verifiedSession, env=process.env, transport=null) {
  // Invoke in the authenticated server relay. A local "trepo-copy-*" ID is not
  // an authority. Never persist this response in a public/browser snapshot.
  return householdBrain({operation:'context',actor:verifiedSession.userId,
    household:verifiedSession.householdId,audience:'personal',purpose:'shopping'},env,transport);
}
