"""Actual HTTP admission/polling on the dedicated review account only."""
from pathlib import Path
import base64, boto3, datetime, json, time, urllib.error, urllib.request, uuid

R=Path(__file__).resolve().parent.parent; D=R/'empty-bulk-live'
O='8fcb37c1-0927-4c31-856a-5b761efc226c'
B='https://6m9t6wosh9.execute-api.us-east-1.amazonaws.com'
K='https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com/kitchen/'+O
F=Path('/Users/MattTaylor/Desktop/Apps/trepo-ios-codex/personas/Fixtures')
release=json.loads((D/'deployment.json').read_text())
assert release['status']=='deployed'
s=boto3.Session(profile_name='trepo-dev',region_name='us-east-1');l=s.client('lambda')
for service, item in release['functions'].items():
 assert l.get_function_configuration(FunctionName=item['function'])['CodeSha256']==item['candidate_code_sha256']
c=l.get_function_configuration(FunctionName=release['functions']['capture']['function']);e=c['Environment']['Variables']
assert e.get('CAPTURE_RECOVERY_ENABLED','false')!='true'
t=s.resource('dynamodb').Table(e['JOB_TABLE_NAME']);s3=s.client('s3')
record={'run':'empty-bulk-'+uuid.uuid4().hex[:10],'owner':O,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'events':[],'jobs':[]}
out=D/'live-canary.json';assert not out.exists(),'Preserve previous evidence before rerunning'
def save():out.write_text(json.dumps(record,indent=2,default=str)+'\n')
def req(method,url,body=None,expected=(200,202)):
 request=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,method=method,headers={'Content-Type':'application/json','X-Correlation-ID':record['run']})
 start=time.monotonic()
 try:
  with urllib.request.urlopen(request,timeout=45) as r:code=r.status;data=json.load(r)
 except urllib.error.HTTPError as err:code=err.code;data=json.load(err)
 record['events'].append({'method':method,'status':code,'seconds':round(time.monotonic()-start,3)});save()
 assert code in expected,('Unexpected HTTP response',code)
 return data,code

def inventory_ids():return {r['_id'] for r in req('GET',K)[0]['items']}
try:
 before=inventory_ids()
 # A synthetic old-format job proves read normalization without modifying any
 # existing job or invoking a provider. It has no source image or inventory write.
 jid=str(uuid.uuid4()); now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 fixture={'job_id':jid,'status':'completed','analysis_mode':'bulk_inventory_deep','meta':{'owner':O,'user_id':O,'persist_to_kitchen':False},'result':{'items':[]},'created_at':now,'updated_at':now,'stage':'completed','progress':100,'stage_message':'No items identified','ttl':int(time.time())+3600}
 t.put_item(Item=fixture,ConditionExpression='attribute_not_exists(job_id)')
 record['jobs'].append(jid);save()
 stored=t.get_item(Key={'job_id':jid},ConsistentRead=True)['Item']
 projected,code=req('GET',B+'/job/'+jid)
 assert code==200 and projected['job_id']==jid and projected['status']=='failed'
 assert projected['result']['items']==[] and projected['result']['error_code']=='no_items_identified'
 assert t.get_item(Key={'job_id':jid},ConsistentRead=True)['Item']==stored
 record['events'].append({'check':'Legacy completed-empty row is nonreviewable through actual HTTP without persisted rewrites','pass':True});save()
 for name,filename,mode,expected_status,expected_outcome in [
  ('empty-bulk','recipe-upload-blank.png','bulk_inventory_deep','failed','no_items'),
  ('valid-fridge','upload-reliability-fridge.png','bulk_inventory_deep','completed',None),
  ('receipt','upload-reliability-receipt.png','receipt_inventory_deep','completed','items_identified'),
  ('empty-receipt','recipe-upload-blank.png','receipt_inventory_deep','completed','no_items')]:
  result,code=req('POST',B+'/identify-async',{'image':base64.b64encode((F/filename).read_bytes()).decode(),'analysis_mode':mode,'owner':O,'user_id':O,'persist_to_kitchen':False},expected=(202,))
  jid=result['job_id'];record['jobs'].append(jid);save()
  deadline=time.monotonic()+180; job=None
  while time.monotonic()<deadline:
   job,code=req('GET',B+'/job/'+jid,expected=(200,202,500))
   if job.get('status') in ('completed','failed'):break
   time.sleep(3)
  assert code==200 and job['status']==expected_status,(name,code,job.get('status'))
  stored=t.get_item(Key={'job_id':jid},ConsistentRead=True)['Item']
  assert stored['meta']['owner']==O and stored['meta']['persist_to_kitchen']==False
  items=(job.get('result') or {}).get('items',[])
  assert bool(items)==(name in ('valid-fridge','receipt'))
  if expected_outcome:assert job['result']['capture_outcome']==expected_outcome
  if name=='empty-bulk':assert stored['status']=='failed' and stored['result']['error_code']=='no_items_identified'
  assert not any(x.get('recovery_available') for x in job['result'].get('capture_outcomes',[]))
  record['events'].append({'check':name,'job_id':jid,'status':job['status'],'item_count':len(items),'capture_outcome':job['result'].get('capture_outcome'),'pass':True});save()
 assert inventory_ids()==before
 record['inventory_unchanged']=True;record['passed']=True;save()
except Exception as error:
 record['passed']=False;record['failure_type']=type(error).__name__;save();raise
finally:
 errors=[]
 for jid in record['jobs']:
  try:
   job=t.get_item(Key={'job_id':jid},ConsistentRead=True).get('Item')
   if not job:continue
   assert job['meta']['owner']==O and job['status'] in ('completed','failed'),'Never delete an active job or another owner'
   bucket=e.get('UPLOADS_BUCKET_NAME');removed=[]
   if bucket:
    prefix='bulk-identify-uploads/'+jid
    objects=s3.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=30)
    assert not objects.get('IsTruncated')
    for obj in objects.get('Contents',[]):
     assert obj['Key'].startswith(prefix)
     s3.delete_object(Bucket=bucket,Key=obj['Key']);removed.append(obj['Key'])
    assert not s3.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=1).get('Contents')
   t.delete_item(Key={'job_id':jid},ConditionExpression='#m.#o=:owner',ExpressionAttributeNames={'#m':'meta','#o':'owner'},ExpressionAttributeValues={':owner':O})
   assert 'Item' not in t.get_item(Key={'job_id':jid},ConsistentRead=True)
   record['events'].append({'cleanup_job':jid,'absence_verified':True,'current_source_objects_removed':len(removed)});save()
  except Exception as error:errors.append({'job_id':jid,'failure_type':type(error).__name__})
 record['cleanup_errors']=errors;record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 assert not errors,'Fixture cleanup requires attention'
print(json.dumps({'passed':record.get('passed'),'checks':[x for x in record['events'] if x.get('check')],'inventory_unchanged':record.get('inventory_unchanged'),'fixture_cleanup_verified':True}))
