const {DynamoDBClient}=require('@aws-sdk/client-dynamodb');
const {DynamoDBDocumentClient,GetCommand,TransactWriteCommand}=require('@aws-sdk/lib-dynamodb');
const {S3Client,GetObjectCommand}=require('@aws-sdk/client-s3');
const {LambdaClient}=require('@aws-sdk/client-lambda');
const {createRecoveryAdmission}=require('./captureRecoveryAdmission');
const {createCaptureRecovery}=require('./captureRecovery');
const {NodeHttpHandler}=require('@smithy/node-http-handler');
function createRuntime({db,s3,TableName,bucket,enabled,admit,authenticate}) {
  const send=(client,command)=>client.send(command,{abortSignal:AbortSignal.timeout(7000)});
  return createCaptureRecovery({enabled,bucket,...(authenticate?{authenticate}:{}),
   readJob:async id=>(await send(db,new GetCommand({TableName,Key:{job_id:id},ConsistentRead:true}))).Item,
   readImage:async ref=>{
    const response=await send(s3,new GetObjectCommand({Bucket:ref.bucket,Key:ref.key}));
    if(!response.Body||response.ContentLength>20*1024*1024)throw Error('Source unavailable');
    const chunks=[];let length=0;
    const timer=setTimeout(()=>response.Body.destroy?.(Error('Source read timed out')),7000);
    try {
     for await (const chunk of response.Body) {
      length+=chunk.length;if(length>20*1024*1024)throw Error('Source too large');
      chunks.push(chunk);
     }
     return Buffer.concat(chunks);
    } finally {clearTimeout(timer);response.Body.destroy?.();}
   },
   createChild:async(source,child)=>{
    try{await send(db,new TransactWriteCommand({TransactItems:[
     {ConditionCheck:{TableName,Key:{job_id:source.job_id},
      ConditionExpression:'#status = :completed AND meta.#owner = :owner AND analysis_mode = :mode AND '+(source.scene_source_manifest?'scene_source_manifest.digest':'dispatch_digest')+' = :digest AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at) AND attribute_not_exists(deleted_at) AND (attribute_not_exists(deleted) OR deleted = :false)',
      ExpressionAttributeNames:{'#status':'status','#owner':'owner'},
      ExpressionAttributeValues:{':completed':'completed',':owner':child.meta.owner,':mode':'receipt_inventory_deep',':digest':source.scene_source_manifest?.digest||source.dispatch_digest,':false':false}}},
     {Put:{TableName,Item:child,ConditionExpression:'attribute_not_exists(job_id)'}}]}));}
    catch(error){
     if(error.name!=='TransactionCanceledException')throw error;
     // A duplicate child is a retry; a changed/dismissed source is not.
     if(error.CancellationReasons?.[0]?.Code!=='None'||error.CancellationReasons?.[1]?.Code!=='ConditionalCheckFailed')throw error;
     const existing=(await send(db,new GetCommand({TableName,Key:{job_id:child.job_id},ConsistentRead:true}))).Item;
     if(!existing||existing.meta?.owner!==child.meta.owner||existing.meta.persist_to_kitchen!==false||existing.recovery_source_id!==source.job_id
       ||existing.recovery_image_index!==child.recovery_image_index||existing.analysis_mode!=='bulk_inventory_deep'
       ||existing.deleted||existing.deleted_at||existing.review_dismissed_at||existing.commit_status)throw error;
    }
   },
   admit
  });
}
exports.createRuntime=createRuntime;
let handler;
exports.handle=async event=>{
 if(!handler){
  const options={maxAttempts:1,requestHandler:new NodeHttpHandler({connectionTimeout:1000,requestTimeout:5000})};
  const db=DynamoDBDocumentClient.from(new DynamoDBClient(options)),s3=new S3Client(options);
  const TableName=process.env.JOB_TABLE_NAME,bucket=process.env.UPLOADS_BUCKET_NAME;
  handler=createRuntime({db,s3,TableName,bucket,
   enabled:process.env.CAPTURE_RECOVERY_ENABLED==='true' && process.env.PROCESSING_LEASES_ENABLED==='true' && Boolean(process.env.TOKEN_SIGNING_SECRET),
   admit:createRecoveryAdmission({db,s3,lambda:new LambdaClient(options),TableName,bucket,functionName:process.env.AWS_LAMBDA_FUNCTION_NAME})});
 }
 return handler(event);
};
