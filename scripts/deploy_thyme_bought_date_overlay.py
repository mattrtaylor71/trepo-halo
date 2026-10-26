"""Guarded module-only bought-date rollout over captured serving ZIPs.

prepare validates each exact package; deploy rechecks revision/hash and preserves
all configuration. Private captures and rollback ZIPs stay outside Git.
"""
import copy
import argparse, base64, hashlib, io, json, pathlib, subprocess, tempfile, zipfile, sys
R=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
from verify_node_lambda_package import verify_package
SHARED=['shared/voice-assistant/tool-definitions.mjs','shared/voice-assistant/tool-executor.mjs','shared/thyme-runtime/policy.mjs']
LEGACY=['playground12_voice_ack/trepo-quick-ack/lib/'+n+'.mjs' for n in ['data-access','household-kitchen','kitchen-edit-client','kitchen-task-context','action-outcomes','device-assistant']]
PILOT=['thyme-agent-v2/src/'+n+'.mjs' for n in ['gateway','instructions','prepare-actions','verification','runner']]
NEW='shared/voice-assistant/bought-date.mjs'
FUNCTIONS=['trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k','trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev','trepo-thyme-agent-v2-pilot-api','trepo-thyme-agent-v2-pilot-native','trepo-thyme-agent-v2-pilot-worker']
def sha(data):return base64.b64encode(hashlib.sha256(data).digest()).decode()
def prepare(p,base):
 for name in FUNCTIONS:
  folder=p/name;old=(folder/'before.zip').read_bytes();config=json.loads((folder/'config.private.json').read_text())
  assert sha(old)==config['CodeSha256']
  with zipfile.ZipFile(io.BytesIO(old)) as z:
   names=set(z.namelist());changes={}
   if 'KitchenApi' in name:
    for file in ['amount_operations.py','shelf_dates.py']:
     path='playground6_groceryRec/kitchen_api/'+file
     assert z.read(file)==subprocess.check_output(['git','show',base+':'+path],cwd=R)
     changes[file]=(R/path).read_bytes()
   else:
    paths=[x for x in SHARED+LEGACY+PILOT if x in names and not ('pilot' in name and x.endswith('/device-assistant.mjs'))]
    assert all(x in paths for x in SHARED[:2]+[p for p in LEGACY if not p.endswith(('/household-kitchen.mjs','/action-outcomes.mjs','/device-assistant.mjs'))]),name
    if 'pilot' in name:assert all(x in paths for x in PILOT)
    # Apply only our hunks to each function's own baseline. Preserve service-
    # specific differences; reject any unexpected/preexisting conflict.
    with tempfile.TemporaryDirectory() as tmp:
     dest=pathlib.Path(tmp)
     for path in paths:
      target=dest/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(path))
     special=[p for p in paths if p in ['thyme-agent-v2/src/instructions.mjs','thyme-agent-v2/src/prepare-actions.mjs','thyme-agent-v2/src/runner.mjs']]
     patch=subprocess.check_output(['git','diff','--unified=1',base,'--']+[p for p in paths if p not in special],cwd=R)
     result=subprocess.run(['git','apply','-'],input=patch,capture_output=True,cwd=dest)
     if result.returncode:raise RuntimeError(name+': '+result.stderr.decode())
     # Pilot packages retain older, intentionally different inventory renderers.
     # Add the date without replacing their unrelated fields or prompt changes.
     for path in special:
      target=dest/path;text=target.read_text()
      if path.endswith('/instructions.mjs'):
       rule=next(line for line in (R/path).read_text().splitlines() if line.startswith('- To correct when an existing kitchen item'))
       anchor='- Before editing/deleting a named item,';assert text.count(anchor)==1
       text=text.replace(anchor,rule+'\n'+anchor)
      elif path.endswith('/runner.mjs'):
       anchor='"storage_location", "expiration_date", "created_at"';assert text.count(anchor)==1
       text=text.replace(anchor,'"storage_location", "expiration_date", "bought_date", "created_at"')
      elif 'export function readFields' in text:
       anchor="'storage_location','expiration_date','ingredients'";assert text.count(anchor)==1
       text=text.replace(anchor,"'storage_location','expiration_date','bought_date','ingredients'")
       anchor='fields.filter(k=>row[k]!=null)';assert text.count(anchor)==1
       text=text.replace(anchor,"fields.filter(k=>row[k]!=null || (k==='bought_date' && Object.hasOwn(row,k)))")
      # Older pageData returns full rows already, including the mapped date.
      target.write_text(text)
     for path in paths:
      data=(dest/path).read_bytes()
      if data!=z.read(path):changes[path]=data
    assert NEW not in names
    changes[NEW]=(R/NEW).read_bytes()
   out=io.BytesIO()
   with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as nz:
    for info in z.infolist():nz.writestr(copy.copy(info),changes.get(info.filename,z.read(info.filename)))
    for file in set(changes)-names:nz.writestr(file,changes[file])
   candidate=out.getvalue();(folder/'candidate.zip').write_bytes(candidate)
   with zipfile.ZipFile(io.BytesIO(candidate)) as nz:
    assert {x for x in names if z.read(x)!=nz.read(x)}==set(changes)&names
    assert set(nz.namelist())-names==set(changes)-names
   startup=verify_package(candidate,config['Handler']) if 'nodejs' in config['Runtime'] else {'python_compile':'passed'}
   if 'nodejs' not in config['Runtime']:
    for path,data in changes.items():compile(data,path,'exec')
   manifest={'function':name,'before_sha':sha(old),'after_sha':sha(candidate),'changed':sorted(changes),'startup':startup,'source_base':base}
   (folder/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
def deploy(p,only):
 import boto3
 l=boto3.Session(profile_name='trepo-dev',region_name='us-east-1').client('lambda')
 results=[]
 for name in FUNCTIONS:
  if only and name not in only:continue
  folder=p/name;before=json.loads((folder/'config.private.json').read_text());manifest=json.loads((folder/'candidate-manifest.json').read_text());blob=(folder/'candidate.zip').read_bytes();assert sha(blob)==manifest['after_sha']
  live=l.get_function_configuration(FunctionName=name)
  previous=json.loads((folder/'deployment.json').read_text()) if (folder/'deployment.json').exists() else None
  expected_sha=previous['after_sha'] if previous else before['CodeSha256'];expected_revision=previous['revision'] if previous else before['RevisionId']
  assert live['CodeSha256']==expected_sha and live['RevisionId']==expected_revision, 'Serving code changed: '+name
  result=l.update_function_code(FunctionName=name,ZipFile=blob,RevisionId=live['RevisionId'],Publish=False)
  l.get_waiter('function_updated_v2').wait(FunctionName=name,WaiterConfig={'Delay':2,'MaxAttempts':30})
  after=l.get_function_configuration(FunctionName=name)
  assert after['CodeSha256']==manifest['after_sha'] and after['LastUpdateStatus']=='Successful'
  for key in ['Environment','Role','Handler','Runtime','Layers','VpcConfig','MemorySize','Timeout','Architectures']:assert after.get(key)==before.get(key),key
  receipt={**manifest,'replaced_sha':live['CodeSha256'],'revision':after['RevisionId'],'status':after['LastUpdateStatus']};(folder/'deployment.json').write_text(json.dumps(receipt,indent=2));results.append(receipt);print(json.dumps(receipt),flush=True)
 return results
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','deploy']);a.add_argument('--capture',type=pathlib.Path,required=True);a.add_argument('--base',default='4e43b22');a.add_argument('--only',action='append');args=a.parse_args()
 prepare(args.capture,args.base) if args.mode=='prepare' else deploy(args.capture,args.only)
