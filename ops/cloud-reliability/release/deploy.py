"""Deploy one reviewed overlay with revision CAS and immutable rollback receipts."""
import argparse,base64,datetime,hashlib,json,os,time
from pathlib import Path
import boto3
from botocore.config import Config

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('manifest');ap.add_argument('function');ap.add_argument('--receipts',required=True)
    ap.add_argument('--code-bucket')
    args=ap.parse_args();os.umask(0o077)
    manifest=json.loads(Path(args.manifest).read_text())
    item=next(x for x in manifest['built'] if x['function']==args.function)
    raw=Path(item['package']).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=item['package_sha256']:raise ValueError('package_changed')
    smoke=json.loads((Path(args.manifest).parent/'startup-results.json').read_text())
    if not any(x['function']==args.function and x['package_sha256']==sha and x['exit_code']==0 for x in smoke):
        raise ValueError('startup_not_qualified')
    session=boto3.Session(profile_name='trepo',region_name='us-east-1')
    client=session.client('lambda',config=Config(connect_timeout=5,read_timeout=90,retries={'total_max_attempts':1}))
    current=client.get_function_configuration(FunctionName=args.function)
    if current['CodeSha256']!=item['base_code_sha256'] or current['RevisionId']!=item['base_revision']:
        raise ValueError('serving_revision_changed')
    receipts=Path(args.receipts);receipts.mkdir(parents=True,exist_ok=True)
    backup=receipts/'before'/args.function
    previous=json.loads((backup/'configuration.private.json').read_text())
    original=(backup/'code.zip').read_bytes()
    if previous['RevisionId']!=current['RevisionId'] or base64.b64encode(hashlib.sha256(original).digest()).decode()!=current['CodeSha256']:
        raise ValueError('rollback_copy_not_current')
    kwargs={'ZipFile':raw}
    if len(raw)>45*1024*1024:
        if not args.code_bucket:raise ValueError('large_package_requires_private_code_bucket')
        key='cloud-reliability-20261023/'+args.function+'/'+sha+'.zip'
        session.client('s3').put_object(Bucket=args.code_bucket,Key=key,Body=raw,ServerSideEncryption='AES256')
        kwargs={'S3Bucket':args.code_bucket,'S3Key':key}
    receipt={'function':args.function,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'before_sha':current['CodeSha256'],'before_revision':current['RevisionId'],'package_sha256':sha,'state':'submitted'}
    path=receipts/(args.function+'.deploy.json')
    if path.exists():raise ValueError('deployment_receipt_already_exists')
    path.write_text(json.dumps(receipt,indent=2))
    result=client.update_function_code(FunctionName=args.function,RevisionId=current['RevisionId'],Publish=False,**kwargs)
    receipt['accepted_revision']=result['RevisionId'];path.write_text(json.dumps(receipt,indent=2))
    for _ in range(90):
        result=client.get_function_configuration(FunctionName=args.function)
        if result.get('LastUpdateStatus')!='InProgress':break
        time.sleep(2)
    expected=base64.b64encode(hashlib.sha256(raw).digest()).decode()
    if result.get('LastUpdateStatus')!='Successful' or result['CodeSha256']!=expected:
        receipt['state']='verification_failed';path.write_text(json.dumps(receipt,indent=2));raise ValueError('deployment_not_verified')
    for key in ('Handler','Runtime','Role','Timeout','MemorySize','Environment','Layers','VpcConfig','Architectures'):
        if result.get(key)!=current.get(key):raise ValueError('configuration_changed:'+key)
    receipt.update(state='verified',after_sha=result['CodeSha256'],after_revision=result['RevisionId'],
                   completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
