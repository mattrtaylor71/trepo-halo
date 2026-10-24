"""Authenticated, durable recipe-photo admission; the accepted job is its immutable receipt."""
import hashlib
import json
import os
import re
import time
import uuid
from botocore.exceptions import ClientError
import operation_auth
import recipe_batch_checkpoint as checkpoint

PROTOCOL='recipe-import-v1'
UUID=operation_auth.UUID

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def identity(owner,operation):
    return uuid.uuid5(uuid.NAMESPACE_URL,PROTOCOL+':'+owner+':'+operation).hex

def handle(api,event,owner,method,body=None,query=None,request_id=None):
    denied=operation_auth.authenticate(event,owner)
    if denied:return api._error(*denied)
    query=query or {};body=body or {}
    if method=='GET' and query.get('capabilities')==PROTOCOL:
        return api._success({'protocol':PROTOCOL,'owner':owner,'idempotent_admission':True,'ordered_photo_identity':True})
    operation=query.get('operation_id') if method=='GET' else body.get('operation_id')
    if not isinstance(operation,str) or not UUID.fullmatch(operation):
        return api._error(400,'Invalid operation_id')
    job_id=identity(owner,operation);table=api._jobs_table()
    job=table.get_item(Key={'job_id':job_id},ConsistentRead=True).get('Item')
    if job and (job.get('owner')!=owner or job.get('admission_operation_id')!=operation):
        return api._error(409,'OPERATION_IDENTITY_CONFLICT')
    if method=='GET':
        if not job:return api._success({'protocol':PROTOCOL,'owner':owner,'operation_id':operation,'status':'not_found'},status=404)
        return receipt(api,owner,job,event)
    if method!='POST':return api._error(405,'Unsupported admission method')
    try:
        request=validate(body)
        submission=api._build_saved_recipe_input({'images':request['images']})
    except (ValueError,api.ServiceError):
        return api._error(400,'Invalid recipe-photo input')
    digest=hashlib.sha256(canonical(request).encode()).hexdigest()
    if job:
        if job.get('admission_digest')!=digest:return api._error(409,'OPERATION_INPUT_MISMATCH')
        dispatch_if_needed(api,job)
        return receipt(api,owner,job,event)
    for image,manifest in zip(submission['images'],request['input_manifest']):
        image['_expected_sha256']=manifest['sha256'];image['_expected_size']=manifest['size']
    # Competing payloads for one operation have distinct immutable object keys. The winner's
    # conditional job write binds its exact object; a loser cannot overwrite that submission.
    payload_key='recipe-images/admissions-v1/%s/%s/%s.json'%(owner,job_id,digest)
    api.boto3.client('s3').put_object(Bucket=os.environ['BUCKET_NAME'],Key=payload_key,
        Body=json.dumps(submission).encode(),ContentType='application/json',ServerSideEncryption='AES256')
    now=api._utc_now_iso()
    job={'job_id':job_id,'owner':owner,'ai_consent_actor':api._ai_recipe_owner.get(),'type':api._SAVED_RECIPE_BATCH_JOB_TYPE,'status':'PENDING',
        'image_count':len(request['images']),'result_count':0,'recipe_ids':[],'results':[],'partial_errors':[],
        'created_at':now,'updated_at':now,'request_id':request_id,'submission_kind':'images',
        'recovery_deadline':int(time.time())+86400,'payload_s3_key':payload_key,
        'admission_operation_id':operation,'admission_request':request,'admission_digest':digest}
    checkpoint.register(api,job)
    try:
        table.put_item(Item=job,ConditionExpression='attribute_not_exists(job_id)')
    except ClientError as exc:
        if exc.response['Error']['Code']!='ConditionalCheckFailedException':raise
        job=table.get_item(Key={'job_id':job_id},ConsistentRead=True)['Item']
        if job.get('owner')!=owner or job.get('admission_digest')!=digest:return api._error(409,'OPERATION_INPUT_MISMATCH')
    dispatch_if_needed(api,job)
    return receipt(api,owner,job,event)

def validate(body):
    if set(body)!={'protocol','operation_id','images','input_manifest'} or body.get('protocol')!=PROTOCOL:
        raise ValueError('Invalid fields')
    images=body.get('images');manifest=body.get('input_manifest')
    if not isinstance(images,list) or not 1<=len(images)<=8 or not isinstance(manifest,list) or len(manifest)!=len(images):
        raise ValueError('Invalid ordered photos')
    for image,item in zip(images,manifest):
        if not isinstance(image,dict) or set(image)!={'image_url'} or not isinstance(image['image_url'],str) or not image['image_url'].startswith('https://') or len(image['image_url'])>8192:
            raise ValueError('Invalid image reference')
        if not isinstance(item,dict) or set(item)!={'sha256','size','content_type'}:
            raise ValueError('Invalid photo manifest')
        if not re.fullmatch('[0-9a-f]{64}',str(item['sha256'])) or type(item['size']) is not int or not 1<=item['size']<=15000000 or not isinstance(item['content_type'],str) or not item['content_type'].startswith('image/'):
            raise ValueError('Invalid photo identity')
    return body

def dispatch_if_needed(api,job):
    if job.get('status') in ('PENDING','RUNNING') and int(job.get('lease_until',0))<=int(time.time()):
        try:checkpoint.dispatch(api,job)
        except Exception as exc:
            api._log_event(job.get('request_id'),'recipe_admission_dispatch_pending',error_type=type(exc).__name__)
            # register() persisted recovery discovery before admission. The scheduled worker or
            # an exact repeated admission can dispatch this same job after an invocation failure.

def receipt(api,owner,job,event):
    return api._success({'protocol':PROTOCOL,'owner':owner,'operation_id':job['admission_operation_id'],
        'status':'admitted','request':job['admission_request'],'accepted':True,
        'job':api._serialize_saved_recipe_batch_job(job,event=event)},status=202)
