"""Import assembled handlers in the matching Linux Lambda runtime, network off.
Fake environment values only; no invocation of an application handler.
"""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path

def extract_with_modes(archive, root):
    if any(not (root/n).resolve().is_relative_to(root.resolve()) for n in archive.namelist()):
        raise ValueError('unsafe_zip_path')
    archive.extractall(root)
    # ZipFile.extractall drops POSIX executable bits. Match Lambda extraction so
    # bundled media tools are actually exercised instead of failing EACCES.
    for member in archive.infolist():
        mode = (member.external_attr >> 16) & 0o777
        if mode and not member.is_dir():
            (root/member.filename).chmod(mode)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('--only',nargs='*');args=ap.parse_args()
    manifest=Path(args.manifest);data=json.loads(manifest.read_text());output=manifest.parent/'startup-results.json'
    records=json.loads(output.read_text()) if args.only and output.exists() else []
    active={job['function'] for job in data['built']}
    records=[r for r in records if r['function'] in active]
    for job in data['built']:
        if args.only and job['function'] not in args.only:continue
        records=[r for r in records if r['function']!=job['function']]
        package=Path(job['package'])
        if hashlib.sha256(package.read_bytes()).hexdigest()!=job['package_sha256']:raise ValueError('package_drift')
        root=manifest.parent/'smoke'/job['function'];root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(package) as z:
            extract_with_modes(z,root)
        module,member=job['handler'].rsplit('.',1)
        runtime=job['runtime'];python=runtime.startswith('python')
        image='public.ecr.aws/lambda/'+('python:'+runtime[6:] if python else 'nodejs:'+runtime.removeprefix('nodejs').removesuffix('.x'))
        if python:
            script=f"import importlib; m=importlib.import_module({module!r}); assert callable(getattr(m,{member!r})); print('handler_import_ok')"
            imports=[c['target'][:-3].replace('/','.') for c in job['changes'] if c['target'].endswith('.py')]
            script+=f'; [importlib.import_module(n) for n in {imports!r}]; print("changed_modules_import_ok")'
            if (root/'bin'/'ffmpeg').exists():
                script+='; import subprocess; subprocess.run(["/var/task/bin/ffmpeg","-version"],check=True,stdout=subprocess.DEVNULL,timeout=10); print("media_decoder_executable_ok")'
            command=['python','-c',script]
        else:
            matches=[str(p.relative_to(root)) for p in root.glob(module+'.*') if p.suffix in {'.js','.mjs','.cjs'}]
            if len(matches)!=1:raise ValueError('ambiguous_handler:'+job['function'])
            script=f"globalThis.awslambda={{streamifyResponse: fn=>fn}}; const m=await import('/var/task/{matches[0]}'); if(typeof m[{member!r}]!=='function') throw Error('missing_handler'); console.log('handler_import_ok')"
            command=['node','--input-type=module','-e',script]
        result=subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none','--read-only',
            '--tmpfs','/tmp','-v',str(root)+':/var/task:ro','-w','/var/task','--entrypoint',command[0],
            '-e','AWS_REGION=us-east-1','-e','AWS_DEFAULT_REGION=us-east-1',
            '-e','AWS_ACCESS_KEY_ID=test','-e','AWS_SECRET_ACCESS_KEY=test','-e','AWS_EC2_METADATA_DISABLED=true',
            '-e','PYTHONPATH=/var/task:/opt/python:/var/runtime',
            '-e','NODE_PATH=/var/runtime/node_modules:/var/task/playground12_voice_ack/trepo-quick-ack/node_modules:/opt/nodejs/node_modules',
            '-e','OPENAI_API_KEY=sk-test-no-network','-e','DB_HOST=127.0.0.1','-e','DB_USER=test',
            '-e','DB_PASS=test','-e','DB_NAME=test',image,*command[1:]],capture_output=True,text=True,timeout=90)
        log=result.stdout+result.stderr;(root/'import.log').write_text(log)
        records.append({'function':job['function'],'image':image,'package_sha256':job['package_sha256'],'stream_runtime_shim':not python,'exit_code':result.returncode,'log':str(root/'import.log')})
        print(job['function'], 'PASS' if result.returncode==0 else 'FAIL',log[-1600:] if result.returncode else '',flush=True)
        (manifest.parent/'startup-results.json').write_text(json.dumps(records,indent=2)+'\n')
    return any(r['exit_code'] for r in records)
if __name__=='__main__':raise SystemExit(main())
