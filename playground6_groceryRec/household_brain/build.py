"""Build a deterministic standalone zip or minimal serving overlay; never deploy."""
import argparse
import base64
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MODULE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def python_files():
    return {str(p.relative_to(ROOT/'playground6_groceryRec')):p.read_bytes()
        for folder in ['family_brain','household_brain']
        for p in (ROOT/'playground6_groceryRec'/folder).glob('*.py')
        if not p.name.startswith('test_') and p.name not in {'build.py','build_overlay.py','cli.py','migrate.py','operator.py'}}


def build(output,base=None,expected_hash=None,kind='standalone'):
    files={};changed={}
    if base:
        raw=Path(base).read_bytes()
        if sha(raw)!=expected_hash:
            raise ValueError('base_package_hash_changed')
        with zipfile.ZipFile(base) as z:
            files={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    def replace_suffix(suffix,data,expected=None):
        names=[n for n in files if n.endswith(suffix)]
        if len(names)!=1:
            raise ValueError('ambiguous_or_missing_baseline:'+suffix)
        name=names[0]
        if expected is not None and sha(files[name])!=expected:
            raise ValueError('target_source_drift:'+suffix)
        changed[name]=data
        return name[:-len(suffix)]
    if kind=='standalone':
        if base:raise ValueError('standalone_has_no_base')
        import pymysql
        changed=python_files()
        folder=Path(pymysql.__file__).parent
        changed.update({'pymysql/'+str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*.py')})
    elif kind=='python':
        manifest=json.loads((ROOT/'docs/household-brain/imported-v1-manifest.json').read_text())
        prefix=None
        for name in ['store.py','api.py']:
            suffix='family_brain/'+name
            found=replace_suffix(suffix,(ROOT/'playground6_groceryRec'/suffix).read_bytes(),manifest['playground6_groceryRec/'+suffix])
            if prefix is not None and prefix!=found:raise ValueError('inconsistent_python_layout')
            prefix=found
        changed.update({prefix+k:v for k,v in python_files().items() if k.startswith('household_brain/')})
    elif kind=='thyme-v2':
        for name in ['gateway.mjs','runner.mjs','tools.mjs']:
            path='thyme-agent-v2/src/'+name
            original=subprocess.check_output(['git','show','e856563:'+path],cwd=ROOT)
            replace_suffix(path,(ROOT/path).read_bytes(),sha(original))
        suffix='playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs'
        prefix=replace_suffix(suffix,next(files[n] for n in files if n.endswith(suffix)))
        # Remove unchanged lookup sentinel from the overlay manifest.
        changed.pop(prefix+suffix)
        client='playground12_voice_ack/trepo-quick-ack/lib/household-brain-client.mjs'
        changed[prefix+client]=(ROOT/client).read_bytes()
    elif kind=='quick-ack':
        manifest=json.loads((ROOT/'docs/household-brain/imported-v1-manifest.json').read_text())
        path='playground12_voice_ack/trepo-quick-ack/lib/family-brain.mjs'
        replace_suffix(path,(ROOT/path).read_bytes(),manifest[path])
    else:
        raise ValueError('unsupported_overlay_kind')
    files.update(changed)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo(name,date_time=(2026,10,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,files[name])
    raw=output.read_bytes()
    manifest={'kind':kind,'deployed':False,'base_sha256':expected_hash,'sha256':sha(raw),
              'aws_code_sha256':base64.b64encode(hashlib.sha256(raw).digest()).decode(),
              'changed_files':{k:sha(v) for k,v in changed.items()},'bytes':len(raw)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--base-zip');p.add_argument('--expected-sha256')
    p.add_argument('--kind',choices=['standalone','python','thyme-v2','quick-ack'],default='standalone')
    a=p.parse_args();print(json.dumps(build(a.output,a.base_zip,a.expected_sha256,a.kind)))
