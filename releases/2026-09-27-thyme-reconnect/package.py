from pathlib import Path
import json,zipfile,hashlib,base64,sys
p=Path(sys.argv[1]);overlay=Path(__file__).parent/'overlay';prefix='playground12_voice_ack/trepo-quick-ack/'
manifest={}
for name in ['trepo-quick-ack-sam-dev','trepo-quick-ack-stream-dev']:
 folder=p/name;ident=json.loads((folder/'identity.json').read_text());files=['lib/ios-reconnect.mjs','index-stream.js' if 'stream' in name else 'lib/operation-bridge.mjs'];mapping={prefix+f:(overlay/f).read_bytes() for f in files}
 digest=lambda path:base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()
 assert digest(folder/'live.zip')==ident['CodeSha256']
 with zipfile.ZipFile(folder/'live.zip') as src,zipfile.ZipFile(folder/'candidate.zip','w',zipfile.ZIP_DEFLATED) as dst:
  for item in src.infolist():dst.writestr(item,mapping.get(item.filename,src.read(item)))
  for name2,data in mapping.items():
   if name2 not in src.namelist():dst.writestr(name2,data)
 with zipfile.ZipFile(folder/'live.zip') as a,zipfile.ZipFile(folder/'candidate.zip') as b:
  assert set(b.namelist())==set(a.namelist())|set(mapping)
  changed=[n for n in b.namelist() if n not in a.namelist() or a.read(n)!=b.read(n)];assert set(changed)==set(mapping)
  b.extractall(folder/'qualified-runtime')
 manifest[name]={'base':ident,'candidateHash':digest(folder/'candidate.zip'),'changedFiles':changed,'fileHashes':{n:hashlib.sha256(d).hexdigest() for n,d in mapping.items()}}
(p/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:{'candidateHash':v['candidateHash'],'changedFiles':v['changedFiles']} for k,v in manifest.items()},indent=2))
