"""Read serving ZIPs, verify reviewed bases, overlay targeted files, build locally.
No deployment, infrastructure mutation, database access or user actions.
"""
import argparse,base64,hashlib,io,json,os,subprocess,tempfile,urllib.request,zipfile
from pathlib import Path
import boto3
from botocore.config import Config
ROOT=Path(__file__).resolve().parents[3]
AUDIT=Path('/Users/MattTaylor/Documents/Codex/2026-10-23/halo-cloud-audit/deep/source')

def gitbytes(rev,path):
    return subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT)

REVIEWED_VARIANTS={
 'trepo-quick-ack-sam-dev':{
  'playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs':
   '4b9895bbe460cbc55cccd611ed195329cb209246192cee8ddc435d40e5c155a4'
 }
}

def reconcile(name,target,previous,baseline,replacement):
    if previous==baseline:return replacement
    allowed=REVIEWED_VARIANTS.get(name,{}).get(target)
    if not previous or hashlib.sha256(previous).hexdigest()!=allowed:
        raise ValueError('unreviewed_source_drift:'+target)
    # Reviewed SAM difference: consent wiring is owned by that entrypoint's
    # deployment. Preserve it; apply only this repair's changes, fail on conflict.
    with tempfile.TemporaryDirectory() as directory:
        paths=[Path(directory)/part for part in ('serving','base','repair')]
        for path,data in zip(paths,(previous,baseline,replacement)):path.write_bytes(data)
        result=subprocess.run(['git','merge-file','-p',*[str(p) for p in paths]],capture_output=True)
        if result.returncode:raise ValueError('variant_merge_conflict:'+target)
        return result.stdout

def build(name,changes,out,client,expected_function=None):
    response=client.get_function(FunctionName=name);conf=response['Configuration']
    if expected_function and conf['CodeSha256']!=expected_function:
        raise ValueError('serving_function_changed_since_review')
    original=urllib.request.urlopen(response['Code']['Location'],timeout=30).read()
    if base64.b64encode(hashlib.sha256(original).digest()).decode()!=conf['CodeSha256']:
        raise ValueError('serving_zip_digest_mismatch')
    data={};infos={}
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        for entry in z.infolist():
            if entry.is_dir():continue
            data[entry.filename]=z.read(entry);infos[entry.filename]=entry
    records=[]
    for target,source,revision in changes:
        previous=data.get(target)
        replacement=(ROOT/source).read_bytes()
        if revision:
            baseline=gitbytes(revision,source)
            replacement=reconcile(name,target,previous,baseline,replacement)
        elif previous is not None and previous!=(ROOT/source).read_bytes():
            raise ValueError('new_module_already_exists_with_different_content:'+target)
        if source.endswith('.py'):compile(replacement,target,'exec')
        data[target]=replacement
        records.append({'target':target,'source':source,'before_sha256':hashlib.sha256(previous).hexdigest() if previous else None,
                        'after_sha256':hashlib.sha256(replacement).hexdigest()})
    target=out/(name+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for key,value in data.items():z.writestr(infos.get(key,key),value)
    # Unmodified files, native libraries, consent wrappers and handler paths stay byte-identical.
    with zipfile.ZipFile(target) as z:
        assert all(z.read(key)==value for key,value in data.items())
    return {'function':name,'base_code_sha256':conf['CodeSha256'],'base_revision':conf['RevisionId'],
            'handler':conf['Handler'],'runtime':conf['Runtime'],'architectures':conf.get('Architectures'),
            'package':str(target),'package_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'changes':records}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--only',nargs='*');args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    client=boto3.Session(profile_name='trepo',region_name='us-east-1').client('lambda',config=Config(connect_timeout=5,read_timeout=30,retries={'max_attempts':2}))
    rel='playground6_groceryRec/'
    targets={
     'trepo-grocery-backend-dev-MealPlanGeneratorFunctio-4rNstnlDTS1Z':[
        ('app.py',rel+'meal_plan_generator/app.py','8f6720f'),
        ('recipe_generation_guard.py',rel+'meal_plan_generator/recipe_generation_guard.py','8f6720f'),
        ('family_brain/recipe_safety.py',rel+'family_brain/recipe_safety.py','8f6720f'),
        ('generation_jobs.py',rel+'meal_plan_generator/generation_jobs.py',None),
        ('retry_dispatcher.py',rel+'meal_plan_generator/retry_dispatcher.py',None)],
     'trepo-grocery-backend-dev-SavedRecipeStatus':[('app.py',rel+'saved_recipe_status/app.py','8f6720f')],
     'trepo-grocery-backend-dev-SavedRecipesApiFunction-a7aWuWlJvEF2':[(file,rel+'saved_recipes_api/'+file,'8f6720f') for file in ['app.py','link_recipe_recovery.py','recipe_link_jobs.py']],
     'trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k':[
        ('app.py',rel+'kitchen_api/app.py','3f8c5ab'),('sql_retry.py',rel+'kitchen_api/sql_retry.py',None)],
     'trepo-ux-failure-monitor':[('app.py','obs/ux_failure_monitor/app.py','8f6720f')],
    }
    targets['trepo-grocery-backend-dev-MealPlanApiFunction-BPOVuOSkR8X2']=[
      ('app.py',rel+'meal_plan_api/app.py','8f6720f'),
      ('generation_claim.py',rel+'meal_plan_api/generation_claim.py','8f6720f'),
      ('plan_visibility.py',rel+'meal_plan_api/plan_visibility.py',None)]
    for file in ['family_brain/__init__.py','family_brain/core.py','family_brain/store.py',
                 'family_brain/recipe_safety.py','household_brain/__init__.py','household_brain/core.py']:
        targets['trepo-grocery-backend-dev-MealPlanApiFunction-BPOVuOSkR8X2'].append((file,rel+file,None))
    targets['trepo-imessage-context-projection']=[('index.mjs','ops/cloud-reliability/context-projection/index.mjs','e909d8d')]
    for name in ['trepo-item-icons-v2-Resolver-9pwJBED5nh32']:
        targets[name]=[('icon_service/delivery.py','ops/cloud-reliability/icons/icon_service/delivery.py','2374af5')]
    # Build each active voice entry point only when its serving data-access base
    # matches the reviewed worker. Source drift must be reconciled explicitly.
    prefix='playground12_voice_ack/trepo-quick-ack/lib/'
    for name in ['trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev','trepo-quick-ack-stream-dev']:
        targets[name]=[(prefix+'data-access.mjs',prefix+'data-access.mjs','8f6720f'),
                       (prefix+'household-kitchen-mutation.mjs',prefix+'household-kitchen-mutation.mjs',None)]
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if args.only and manifest_path.exists() else {'built':[],'blocked':[]}
    for key in ('built','blocked'):manifest[key]=[r for r in manifest[key] if r['function'] in targets]
    for name,changes in targets.items():
        if args.only and name not in args.only:continue
        for key in ('built','blocked'):manifest[key]=[r for r in manifest[key] if r['function']!=name]
        provenance=AUDIT/name/'_provenance.json'
        expected=json.loads(provenance.read_text()) if provenance.exists() else {}
        try:
            result=build(name,changes,out,client,expected.get('sha') or expected.get('CodeSha256'))
            manifest['built'].append(result);print(name,'built',flush=True)
        except Exception as exc:
            reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
            manifest['blocked'].append({'function':name,'reason':reason});print(name,'BLOCKED',reason,flush=True)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return bool(manifest['blocked'])

if __name__=='__main__':raise SystemExit(main())
