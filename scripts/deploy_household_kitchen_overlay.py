"""Prepare/apply the scoped 2026-10-18 Thyme fix over captured live ZIPs.
Capture config.json and before.zip per function into a private --evidence directory.
Checks exact live revision/hash and preserves every unrelated ZIP member and config.
"""
import argparse,base64,hashlib,io,json,time,zipfile
from copy import copy
from pathlib import Path
import boto3
parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--apply',action='store_true');parser.add_argument('--function',action='append');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];rel='playground12_voice_ack/trepo-quick-ack/lib/'
names=args.function or ['trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev']
allowed={'trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev'}
assert set(names)<=allowed
L=boto3.Session(profile_name='trepo-dev',region_name='us-east-1').client('lambda')
with zipfile.ZipFile(args.evidence/'trepo-quick-ack-stream-dev/before.zip') as z:
 old_data=z.read(rel+'data-access.mjs');old_assistant=z.read(rel+'device-assistant.mjs').decode()
def block(text):
 start=text.index('async function buildKitchenGroundingMessage(')
 if text[start-7:start]=='export ':start-=7
 end=text.index('\nasync function buildFoodGroundingMessage(',start)
 return text[start:end]
old_block=block(old_assistant);new_block=block((root/rel/'device-assistant.mjs').read_text())
prepared=[]
for name in names:
 d=args.evidence/name;config=json.loads((d/'config.json').read_text())
 live=L.get_function_configuration(FunctionName=name)
 assert live['RevisionId']==config['RevisionId'] and live['CodeSha256']==config['CodeSha256'],f'{name}: deployment changed; recapture before continuing'
 assert config['Environment']['Variables']['WRITE_SHARED_ONLY']=='true'
 with zipfile.ZipFile(d/'before.zip') as z:
  assert z.read(rel+'data-access.mjs')==old_data
  assistant=z.read(rel+'device-assistant.mjs').decode();assert assistant.count(old_block)==1
  updates={rel+'data-access.mjs':(root/rel/'data-access.mjs').read_bytes(),rel+'device-assistant.mjs':assistant.replace(old_block,new_block).encode(),rel+'household-kitchen.mjs':(root/rel/'household-kitchen.mjs').read_bytes()}
  out=io.BytesIO()
  with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dest:
   for info in z.infolist():dest.writestr(copy(info),updates.get(info.filename,z.read(info.filename)))
   for path,content in updates.items():
    if path not in z.namelist():dest.writestr(path,content)
  blob=out.getvalue();(d/'after.zip').write_bytes(blob)
  with zipfile.ZipFile(io.BytesIO(blob)) as post:
   changed={n for n in set(z.namelist())|set(post.namelist()) if n not in z.namelist() or post.read(n)!=z.read(n)}
   assert changed==set(updates),changed
 record={'function':name,'before_sha':config['CodeSha256'],'after_sha':base64.b64encode(hashlib.sha256(blob).digest()).decode(),'changed':sorted(changed),'published':False}
 prepared.append((d,config,blob,record));print(json.dumps(record),flush=True)
if not args.apply:raise SystemExit(0)
for d,config,blob,record in prepared:
 name=record['function'];response=L.update_function_code(FunctionName=name,ZipFile=blob,RevisionId=config['RevisionId'],Publish=False)
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  current=L.get_function_configuration(FunctionName=name)
  if current['LastUpdateStatus']!='InProgress':break
  time.sleep(2)
 assert current['LastUpdateStatus']=='Successful',current.get('LastUpdateStatusReason')
 assert current['CodeSha256']==record['after_sha']
 for key in ['Environment','Handler','Role','Runtime','Layers','VpcConfig','Timeout','MemorySize']:
  assert current.get(key)==config.get(key),key
 record.update(deployed=True,revision=current['RevisionId'],modified=current['LastModified']);(d/'deployment.json').write_text(json.dumps(record,indent=2))
 print(json.dumps(record),flush=True)
