"""Build a two-file compatibility overlay on verified serving Lambda archives.
Usage: python3 package.py PRIVATE_EVIDENCE_DIRECTORY
Never copies a whole dirty checkout or deploys a function.
"""
import base64, hashlib, json, pathlib, sys, zipfile
root=pathlib.Path(sys.argv[1]).resolve();repo=pathlib.Path(__file__).resolve().parents[2]
files=['playground12_voice_ack/trepo-quick-ack/lib/'+f for f in ['operation-recovery.mjs','operation-bridge.mjs']]
manifest={}
for name in ['trepo-quick-ack-sam-dev','trepo-quick-ack-stream-dev']:
 p=root/name;identity=json.loads((p/'identity.json').read_text());original=p/'live.zip';candidate=p/'candidate.zip'
 digest=lambda p:base64.b64encode(hashlib.sha256(p.read_bytes()).digest()).decode()
 assert digest(original)==identity['CodeSha256']
 with zipfile.ZipFile(original) as src,zipfile.ZipFile(candidate,'w',zipfile.ZIP_DEFLATED) as dst:
  assert set(files)<=set(src.namelist())
  for item in src.infolist():dst.writestr(item,(repo/item.filename).read_bytes() if item.filename in files else src.read(item.filename))
 with zipfile.ZipFile(original) as a,zipfile.ZipFile(candidate) as b:
  assert a.namelist()==b.namelist()
  changed=[n for n in a.namelist() if a.read(n)!=b.read(n)];assert set(changed)==set(files)
 manifest[name]={'base':identity,'candidateHash':digest(candidate),'changedFiles':changed,'fileHashes':{n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files}}
(root/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
