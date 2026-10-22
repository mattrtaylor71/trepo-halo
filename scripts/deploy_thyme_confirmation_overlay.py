"""Scoped overlay: preserve exact live package/config and reject concurrent deploys.
Requires private evidence/{function}/{before.zip,config.private.json}. No schema changes.
Prepare with --evidence; deploy one verified function with --apply --function NAME.
"""
import argparse,base64,hashlib,io,json,zipfile,time
from pathlib import Path
from copy import copy
import boto3
PARSER=argparse.ArgumentParser();PARSER.add_argument('--evidence',type=Path,required=True);PARSER.add_argument('--apply',action='store_true');PARSER.add_argument('--function',action='append');args=PARSER.parse_args()
ROOT=Path(__file__).resolve().parents[1]
LIB='playground12_voice_ack/trepo-quick-ack/lib/'
HOME='trepo-grocery-backend-dev-HomeSuggestionsFunction-eq77AzDop23s'
TARGETS=[HOME,'trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev']
names=args.function or TARGETS
assert set(names)<=set(TARGETS)
def service_variant(blob, name):
 if name not in {'trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev'}:return blob
 text=blob.decode()
 pairs=[('import {leftoverFirstToolChoice} from "./leftover-check-in-intent.mjs";\n',''),
        ('  const leftover = leftoverFirstToolChoice(transcript);\n  if (leftover) return leftover;\n',''),
        ('const turnToolChoice = forceToolChoiceNext\n      || (turn === 0 && normalizedResponseSurface === "app" ? leftoverFirstToolChoice(transcript) : null)\n      ||', 'const turnToolChoice = forceToolChoiceNext ||')]
 for old,new in pairs:
  assert text.count(old)==1,(name,'leftover routing variant changed')
  text=text.replace(old,new)
 return text.encode()
client=boto3.Session(profile_name='trepo',region_name='us-east-1').client('lambda')
for name in names:
 d=args.evidence/name;before=json.loads((d/'config.private.json').read_text());live=client.get_function_configuration(FunctionName=name)
 assert live['RevisionId']==before['RevisionId'] and live['CodeSha256']==before['CodeSha256'],f'{name}: live changed; recapture and review'
 if name==HOME:changes={f:(ROOT/'playground6_groceryRec'/f).read_bytes() for f in ['family_brain/api.py','household_brain/store.py']}
 else:
  changes={LIB+f:(ROOT/LIB/f).read_bytes() for f in ['device-assistant.mjs','brain-turn.mjs']}
  changes[LIB+'device-assistant.mjs']=service_variant(changes[LIB+'device-assistant.mjs'],name)
 baseline=args.evidence/('trepo-quick-ack-stream-dev' if name!=HOME else HOME)/'before.zip'
 blob=io.BytesIO()
 with zipfile.ZipFile(d/'before.zip') as old,zipfile.ZipFile(baseline) as expected,zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as new:
  for f in changes:
   if f in old.namelist():
    expected_bytes=expected.read(f)
    if f==LIB+'device-assistant.mjs':expected_bytes=service_variant(expected_bytes,name)
    assert old.read(f)==expected_bytes,(name,f,'service-specific baseline differs')
  for info in old.infolist():new.writestr(copy(info),changes.get(info.filename,old.read(info.filename)))
  for f,data in changes.items():
   if f not in old.namelist():new.writestr(f,data)
 package=blob.getvalue()
 with zipfile.ZipFile(d/'before.zip') as old,zipfile.ZipFile(io.BytesIO(package)) as new:
  changed={f for f in set(old.namelist())|set(new.namelist()) if f not in old.namelist() or old.read(f)!=new.read(f)}
  assert changed==set(changes)
 (d/'confirmation-after.zip').write_bytes(package)
 receipt={'function':name,'before_sha':before['CodeSha256'],'after_sha':base64.b64encode(hashlib.sha256(package).digest()).decode(),'changed':sorted(changed),'applied':False}
 if args.apply:
  client.update_function_code(FunctionName=name,ZipFile=package,RevisionId=before['RevisionId'],Publish=False)
  deadline=time.monotonic()+120
  while time.monotonic()<deadline:
   current=client.get_function_configuration(FunctionName=name)
   if current['LastUpdateStatus']!='InProgress':break
   time.sleep(2)
  assert current['LastUpdateStatus']=='Successful'
  assert current['CodeSha256']==receipt['after_sha']
  for k in ['Environment','Handler','Role','Runtime','Layers','VpcConfig','Timeout','MemorySize']:
   assert current.get(k)==before.get(k),k
  receipt.update(applied=True,revision=current['RevisionId'],modified=current['LastModified'])
  (d/'confirmation-deployed.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True)
