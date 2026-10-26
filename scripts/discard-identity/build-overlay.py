"""Build a candidate from a freshly downloaded serving ZIP. Does not deploy.

Never ship an old checkout wholesale. Require the touched serving files to match
the reviewed baseline, preserve every unrelated serving entry byte-for-byte,
then overlay only this change. Drift aborts the build and requires a review.
"""
import argparse,hashlib,json,pathlib,zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('kind',choices=['photo','voice'])
parser.add_argument('serving_zip',type=pathlib.Path)
parser.add_argument('output_zip',type=pathlib.Path)
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
manifest=json.loads(pathlib.Path(__file__).with_name('overlay-manifest.json').read_text())[args.kind]
sha=lambda data:hashlib.sha256(data).hexdigest()
if args.output_zip.exists():raise SystemExit('Output already exists; choose a new path.')
with zipfile.ZipFile(args.serving_zip) as old:
 entries={name:old.read(name) for name in old.namelist()}
 if len(entries)!=len(old.namelist()):raise SystemExit('Duplicate ZIP paths; aborting.')
 if 'ai_consent_guard.cjs' not in entries:raise SystemExit('Missing serving consent adapter.')
 for name,spec in manifest.items():
  if spec.get('base_sha256') and (name not in entries or sha(entries[name])!=spec['base_sha256']):
   raise SystemExit('Serving source changed: '+name+'. Rebase and review before release.')
  if not spec.get('base_sha256') and name in entries:raise SystemExit('New file already exists in serving ZIP: '+name)
 replacements={name:(root/spec['source']).read_bytes() for name,spec in manifest.items()}
 with zipfile.ZipFile(args.output_zip,'w',compression=zipfile.ZIP_DEFLATED) as out:
  for info in old.infolist():out.writestr(info,replacements.get(info.filename,entries[info.filename]))
  for name,data in replacements.items():
   if name not in entries:out.writestr(name,data)
with zipfile.ZipFile(args.output_zip) as result:
 for name,data in entries.items():
  if name not in replacements:assert result.read(name)==data,name
 for name,data in replacements.items():assert result.read(name)==data,name
receipt={'kind':args.kind,'base_sha256':sha(args.serving_zip.read_bytes()),
 'output_sha256':sha(args.output_zip.read_bytes()),'replacements':{name:sha(data) for name,data in replacements.items()},
 'unchanged_entries_verified':len(entries)-sum(name in entries for name in replacements),'deployed':False}
args.output_zip.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
