const {createHash}=require('node:crypto');
const {UpdateCommand}=require('@aws-sdk/lib-dynamodb');
const {PutObjectCommand}=require('@aws-sdk/client-s3');
const {makeSceneSourceManifest}=require('./sceneSourceManifest');
function createSourceRetention({db,s3,TableName,bucket}) {
 return async({job_id,owner,images})=>{
  if(typeof job_id!=='string'||!/^[a-f0-9-]{36}$/i.test(job_id)||typeof owner!=='string'||!/^[A-Za-z0-9_-]{1,64}$/.test(owner))throw Error('Invalid receipt identity');
  if(!Array.isArray(images)||!images.length||images.length>30)throw Error('Invalid receipt photos');
  const signal=AbortSignal.timeout(18000);
  const sources=[];
  for(let i=0;i<images.length;i++){
   const body=Buffer.from(images[i]||'','base64');if(!body.length||body.length>20*1024*1024)throw Error('Invalid receipt photo');
   const sha256=createHash('sha256').update(body).digest('hex');
   const key=`bulk-identify-uploads/${job_id}-img${i}.${sha256}.jpg`;
   await s3.send(new PutObjectCommand({Bucket:bucket,Key:key,Body:body,ContentType:'application/octet-stream',ChecksumSHA256:Buffer.from(sha256,'hex').toString('base64')}),{abortSignal:AbortSignal.any([signal,AbortSignal.timeout(5000)])});
   sources.push({bucket,key,sha256});
  }
  const manifest=makeSceneSourceManifest({job_id,owner,analysis_mode:'receipt_inventory_deep',receipt_s3_keys:sources},{bucket});
  if(!manifest)throw Error('Invalid retained receipt identity');
  await db.send(new UpdateCommand({TableName,Key:{job_id},
   UpdateExpression:'SET scene_source_manifest=:manifest',
   ConditionExpression:'#status=:pending AND meta.#owner=:owner AND analysis_mode=:mode AND attribute_not_exists(scene_source_manifest) AND attribute_not_exists(commit_status) AND attribute_not_exists(review_dismissed_at) AND attribute_not_exists(deleted_at) AND (attribute_not_exists(deleted) OR deleted=:false)',
   ExpressionAttributeNames:{'#status':'status','#owner':'owner'},
   ExpressionAttributeValues:{':manifest':manifest,':pending':'pending',':owner':owner,':mode':'receipt_inventory_deep',':false':false}}),{abortSignal:AbortSignal.any([signal,AbortSignal.timeout(5000)])});
  return sources;
 };
}
module.exports={createSourceRetention};
let retain;
module.exports.retain=async input=>{
 if(!retain){
  const {NodeHttpHandler}=require('@smithy/node-http-handler');
  const {DynamoDBClient}=require('@aws-sdk/client-dynamodb');
  const {DynamoDBDocumentClient}=require('@aws-sdk/lib-dynamodb');
  const {S3Client}=require('@aws-sdk/client-s3');
  const options={maxAttempts:1,requestHandler:new NodeHttpHandler({connectionTimeout:1000,requestTimeout:5000})};
  retain=createSourceRetention({db:DynamoDBDocumentClient.from(new DynamoDBClient(options)),s3:new S3Client(options),TableName:process.env.JOB_TABLE_NAME,bucket:process.env.UPLOADS_BUCKET_NAME});
 }
 return retain(input);
};
