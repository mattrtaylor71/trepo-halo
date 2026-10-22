"""Guarded retry-key repair over the tested card rollout; no configuration changes."""
import base64,hashlib,io,json,os,time,urllib.request,zipfile,argparse
from copy import copy
from pathlib import Path
import boto3
os.umask(0o077)
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--apply',action='store_true');p.add_argument('--function',action='append');a=p.parse_args()
root=Path(__file__).resolve().parents[1];base='playground12_voice_ack/trepo-quick-ack/lib/';files=[base+f for f in ['family-brain.mjs','operation-bridge.mjs','operation-recovery.mjs']];changes={f:(root/f).read_bytes() for f in files}
cutoff=int(changes[base+'operation-recovery.mjs'].decode().split('export const STABLE_ADMISSION_SINCE = ')[1].split(';')[0]);assert cutoff>0
l=boto3.Session(profile_name='trepo',region_name='us-east-1').client('lambda')
# Refuse a too-late rollout. The cutover must be AFTER every update; older IDs
# stop without reexecution, including requests from the rolling-deploy interval.
if a.apply:assert time.time()<cutoff-70,'Choose and test a new future cutover'
targets=['trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev']
assert set(a.function or targets)<=set(targets)
for name in a.function or targets:
 selected=[base+'family-brain.mjs'] if name=='trepo-quick-ack-async-worker-dev' else files
 selected_changes={f:changes[f] for f in selected}
 previous=a.evidence/'live-write'/'card-deploy'/name;prior=json.loads((previous/'receipt.json').read_text());d=a.evidence/'live-write'/'retry-deploy'/name;d.mkdir(parents=True,exist_ok=True)
 f=l.get_function(FunctionName=name);c=f['Configuration'];assert c['RevisionId']==prior['revision'] and c['CodeSha256']==prior['after_sha'];before=urllib.request.urlopen(f['Code']['Location']).read();assert base64.b64encode(hashlib.sha256(before).digest()).decode()==c['CodeSha256'];(d/'before.zip').write_bytes(before);(d/'config.private.json').write_text(json.dumps(c,default=str,indent=2));out=io.BytesIO()
 with zipfile.ZipFile(io.BytesIO(before)) as old,zipfile.ZipFile(previous/'after.zip') as expected,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for path in selected:assert old.read(path)==expected.read(path)
  for info in old.infolist():new.writestr(copy(info),selected_changes.get(info.filename,old.read(info.filename)))
 package=out.getvalue()
 with zipfile.ZipFile(io.BytesIO(before)) as old,zipfile.ZipFile(io.BytesIO(package)) as new:assert {f for f in old.namelist() if old.read(f)!=new.read(f)}==set(selected)
 (d/'after.zip').write_bytes(package);receipt={'function':name,'before_sha':c['CodeSha256'],'after_sha':base64.b64encode(hashlib.sha256(package).digest()).decode(),'changed':selected,'cutover':cutoff,'applied':False}
 if a.apply:
  assert time.time()<cutoff-25,'Deployment exceeded cutover buffer; stop and re-plan'
  l.update_function_code(FunctionName=name,ZipFile=package,RevisionId=c['RevisionId']);deadline=time.monotonic()+90
  while time.monotonic()<deadline:
   final=l.get_function_configuration(FunctionName=name)
   if final['LastUpdateStatus']!='InProgress':break
   time.sleep(2)
  assert final['LastUpdateStatus']=='Successful' and final['CodeSha256']==receipt['after_sha']
  for k in ['Environment','Handler','Role','Runtime','Layers','VpcConfig','Timeout','MemorySize']:assert final.get(k)==c.get(k)
  receipt.update(applied=True,revision=final['RevisionId'],modified=final['LastModified']);assert time.time()<cutoff
 (d/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
