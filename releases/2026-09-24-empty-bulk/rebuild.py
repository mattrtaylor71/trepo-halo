#!/usr/bin/env python3
"""Rebuild a single reviewed ZIP from its checksum-pinned original serving ZIP.
No cloud calls or deployment. Python stdlib only. Byte hash depends on zlib version;
the tool never silently accepts a different artifact.
"""
import argparse,base64,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('service',choices=['capture','getJob']);p.add_argument('base_zip',type=Path);p.add_argument('output_zip',type=Path);a=p.parse_args()
r=Path(__file__).resolve().parent;m=json.loads((r/'manifest.json').read_text())[a.service];meta=json.loads((r/'new-entry-metadata.json').read_text())[a.service]
assert a.base_zip.resolve()!=a.output_zip.resolve(),'Never overwrite a rollback ZIP'
assert not a.output_zip.exists(),'Choose a new output file'
assert base64.b64encode(hashlib.sha256(a.base_zip.read_bytes()).digest()).decode()==m['base']['CodeSha256'],'Wrong original serving ZIP'
changes={v['path']:v for v in m['changes']}
with zipfile.ZipFile(a.base_zip) as src,zipfile.ZipFile(a.output_zip,'w',zipfile.ZIP_DEFLATED) as dst:
 names=set(src.namelist())
 for info in src.infolist():
  data=src.read(info.filename)
  if info.filename in changes:
   record=changes[info.filename];assert hashlib.sha256(data).hexdigest()==record['previous_sha256']
   data=(r/'overlays'/a.service/info.filename).read_bytes();assert hashlib.sha256(data).hexdigest()==record['sha256']
  dst.writestr(info,data)
 for name in sorted(changes.keys()-names):
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  fields=meta[name];info=zipfile.ZipInfo(name,tuple(fields['date_time']))
  for key,value in fields.items():
   if key not in ['date_time','extra_b64','comment_b64']:setattr(info,key,value)
  info.extra=base64.b64decode(fields['extra_b64']);info.comment=base64.b64decode(fields['comment_b64'])
  data=(r/'overlays'/a.service/name).read_bytes();assert hashlib.sha256(data).hexdigest()==changes[name]['sha256'];dst.writestr(info,data)
assert hashlib.sha256(a.output_zip.read_bytes()).hexdigest()==m['zip_sha256'],'Archive bytes differ; do not deploy without requalification'
print(a.service+': exact release ZIP reproduced and verified')
