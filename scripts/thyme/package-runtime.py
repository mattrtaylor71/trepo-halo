#!/usr/bin/env python3
"""Overlay only reviewed runtime files on a pinned current Lambda ZIP; no deployment."""
import argparse, hashlib, json, pathlib, zipfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=pathlib.Path,required=True)
parser.add_argument('--base-sha256',required=True)
parser.add_argument('--target',choices=['legacy','native'],required=True)
parser.add_argument('--out',type=pathlib.Path,required=True)
a=parser.parse_args();root=pathlib.Path(__file__).resolve().parents[2]
actual=hashlib.sha256(a.base.read_bytes()).hexdigest()
if actual!=a.base_sha256:raise SystemExit('Base archive changed; inspect current serving code before packaging.')
files=list((root/'shared/thyme-runtime').glob('*.mjs'))
files += ([root/'playground12_voice_ack/trepo-quick-ack'/p for p in ['index-stream.js','lib/device-assistant.mjs']]
          if a.target=='legacy' else [root/'thyme-agent-v2/src'/p for p in ['direct-runner.mjs','entry.mjs','native.mjs','runner.mjs','service.mjs']])
if a.target=='native':files.append(root/'playground12_voice_ack/trepo-quick-ack/lib/action-claims.mjs')
updates={str(p.relative_to(root)):p.read_bytes() for p in files}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.base) as source,zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as target:
 names=source.namelist()
 if len(set(names))!=len(names):raise SystemExit('Duplicate entries in base archive.')
 if 'ai_consent_guard.cjs' not in names:raise SystemExit('Base lacks the deployed AI consent guard.')
 for info in source.infolist():
  if info.filename not in updates:target.writestr(info,source.read(info.filename))
 for name,data in updates.items():target.writestr(name,data)
manifest={'target':a.target,'base_sha256':actual,'archive_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),
          'files':{name:hashlib.sha256(data).hexdigest() for name,data in updates.items()},
          'runtime_enabled':False,'note':'Package only. Preserve deployment environment; actor allowlist defaults off. No credentials or test consent fixture added.'}
a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
