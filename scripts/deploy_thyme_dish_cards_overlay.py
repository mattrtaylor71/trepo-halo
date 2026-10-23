"""Apply app-response.mjs and its required formatter to verified rollout packages.
Private evidence and exact previous deploy receipts are required; never changes config.
"""
import argparse,base64,hashlib,io,json,os,time,urllib.request,zipfile
from copy import copy
from pathlib import Path
import boto3
from verify_node_lambda_package import verify_package
os.umask(0o077)
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--apply',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];member='playground12_voice_ack/trepo-quick-ack/lib/app-response.mjs';new_source=(root/member).read_bytes();dependency=member.replace('app-response','recipe-response');dependency_source=(root/dependency).read_bytes();l=boto3.Session(profile_name='trepo',region_name='us-east-1').client('lambda')
for name in ['trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev']:
 previous=json.loads((args.evidence/name/'confirmation-deployed.json').read_text());d=args.evidence/'live-write'/'card-deploy'/name;d.mkdir(parents=True,exist_ok=True)
 current=l.get_function(FunctionName=name);c=current['Configuration'];assert c['CodeSha256']==previous['after_sha'] and c['RevisionId']==previous['revision'],name+' changed since confirmation deployment'
 before=urllib.request.urlopen(current['Code']['Location']).read();assert base64.b64encode(hashlib.sha256(before).digest()).decode()==c['CodeSha256']
 (d/'before.zip').write_bytes(before);(d/'config.private.json').write_text(json.dumps(c,default=str,indent=2));out=io.BytesIO()
 with zipfile.ZipFile(io.BytesIO(before)) as old,zipfile.ZipFile(args.evidence/name/'confirmation-after.zip') as expected,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  assert old.read(member)==expected.read(member) and old.read(member)!=new_source
  if dependency in old.namelist():assert old.read(dependency)==dependency_source,'Formatter differs; inspect before replacing'
  for info in old.infolist():new.writestr(copy(info),new_source if info.filename==member else old.read(info.filename))
  if dependency not in old.namelist():new.writestr(dependency,dependency_source)
 package=out.getvalue()
 with zipfile.ZipFile(io.BytesIO(before)) as old,zipfile.ZipFile(io.BytesIO(package)) as new:
  assert set(new.namelist())-set(old.namelist()) <= {dependency};assert set(old.namelist()) <= set(new.namelist());assert [f for f in old.namelist() if old.read(f)!=new.read(f)]==[member]
  added=sorted(set(new.namelist())-set(old.namelist()))
 startup=verify_package(package,c['Handler'])
 (d/'after.zip').write_bytes(package);receipt={'function':name,'before_sha':c['CodeSha256'],'after_sha':base64.b64encode(hashlib.sha256(package).digest()).decode(),'changed':[member],'added':added,'startup':startup,'applied':False}
 if args.apply:
  l.update_function_code(FunctionName=name,ZipFile=package,RevisionId=c['RevisionId']);deadline=time.monotonic()+110
  while time.monotonic()<deadline:
   latest=l.get_function_configuration(FunctionName=name)
   if latest['LastUpdateStatus']!='InProgress':break
   time.sleep(2)
  assert latest['LastUpdateStatus']=='Successful' and latest['CodeSha256']==receipt['after_sha']
  for k in ['Environment','Handler','Role','Runtime','Layers','VpcConfig','Timeout','MemorySize']:assert latest.get(k)==c.get(k)
  receipt.update(applied=True,revision=latest['RevisionId'],modified=latest['LastModified'])
 (d/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
