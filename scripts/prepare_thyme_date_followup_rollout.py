"""Prepare minimal date-followup patches on captured serving packages; never deploys.

The fixed base is the pre-repair source, not a moving HEAD. Preserve independent
serving changes and refuse ambiguous/missing patch contexts.
"""
import sys,subprocess,json,zipfile,io,copy,base64,hashlib,difflib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from verify_node_lambda_package import verify_package
p=Path(sys.argv[1])
paths=['shared/voice-assistant/bought-date.mjs','shared/voice-assistant/tool-definitions.mjs','shared/thyme-runtime/policy.mjs','playground12_voice_ack/trepo-quick-ack/lib/kitchen-task-context.mjs','playground12_voice_ack/trepo-quick-ack/lib/device-assistant.mjs','playground12_voice_ack/trepo-quick-ack/lib/action-outcomes.mjs','thyme-agent-v2/src/instructions.mjs']
sha=lambda b:base64.b64encode(hashlib.sha256(b).digest()).decode()
def patch_live(actual,base,new):
 a=base.splitlines(True);b=new.splitlines(True)
 for tag,i,j,k,l in reversed(difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes()):
  if tag=='equal':continue
  before=''.join(a[i:j]);after=''.join(b[k:l])
  if not before:
   before=''.join(a[max(0,i-1):i+1]);after=''.join(a[max(0,i-1):i])+after+''.join(a[i:i+1])
  assert actual.count(before)==1,('patch context not unique',before[:120])
  actual=actual.replace(before,after,1)
 return actual
for f in p.glob('*/before.zip'):
 config=json.loads((f.parent/'config.private.json').read_text());original=f.read_bytes();assert sha(original)==config['CodeSha256']
 with zipfile.ZipFile(io.BytesIO(original)) as z:
  changes={}
  for path in paths:
   if path not in z.namelist():continue
   base=subprocess.check_output(['git','show','481e283:'+path],cwd=root).decode();new=(root/path).read_text();actual=z.read(path).decode()
   changed=patch_live(actual,base,new)
   if changed!=actual:changes[path]=changed.encode()
  assert 'shared/voice-assistant/bought-date.mjs' in changes
  output=io.BytesIO()
  with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as dest:
   for info in z.infolist():dest.writestr(copy.copy(info),changes.get(info.filename,z.read(info.filename)))
  blob=output.getvalue()
  with zipfile.ZipFile(io.BytesIO(blob)) as check:
   assert check.namelist()==z.namelist()
   assert {n for n in z.namelist() if z.read(n)!=check.read(n)}==set(changes)
  verification=verify_package(blob,config['Handler']);(f.parent/'candidate.zip').write_bytes(blob)
  manifest={'before_sha':sha(original),'after_sha':sha(blob),'changed':list(changes),'verification':verification}
  (f.parent/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'function':f.parent.name,**manifest}),flush=True)
