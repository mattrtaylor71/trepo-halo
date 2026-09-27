"""Code-only release after explicit founder approval; never use an entire checkout.
Usage: python3 deploy.py PRIVATE_EVIDENCE_DIRECTORY --approved
"""
from pathlib import Path
import argparse,base64,boto3,datetime,hashlib,json,time
parser=argparse.ArgumentParser();parser.add_argument('evidence');parser.add_argument('--approved',action='store_true');args=parser.parse_args()
if not args.approved:raise SystemExit('Explicit release approval required')
p=Path(args.evidence);m=json.loads((p/'candidate-manifest.json').read_text());tests=json.loads((p/'qualification.json').read_text());assert tests['status']=='passed_pending_release_approval'
for file,expected in tests['evidenceHashes'].items():assert hashlib.sha256((p/file).read_bytes()).hexdigest()==expected, 'Qualified evidence changed'
m=dict(sorted(m.items(),key=lambda pair:0 if 'stream' in pair[0] else 1))
l=boto3.Session(profile_name='trepo',region_name='us-east-1').client('lambda');before={}
keys=['Role','Handler','Runtime','Timeout','MemorySize','Environment','Layers','VpcConfig','Architectures','EphemeralStorage','TracingConfig','FileSystemConfigs','DeadLetterConfig','LoggingConfig']
def digest(blob):return base64.b64encode(hashlib.sha256(blob).digest()).decode()
def config_hash(cfg):return hashlib.sha256(json.dumps({k:cfg.get(k) for k in keys},sort_keys=True).encode()).hexdigest()
for name,item in m.items():
 cfg=l.get_function_configuration(FunctionName=name);assert cfg['CodeSha256']==item['base']['CodeSha256'] and cfg['RevisionId']==item['base']['RevisionId'], 'Serving identity changed; stop and requalify'
 assert digest((p/name/'candidate.zip').read_bytes())==item['candidateHash'];before[name]=cfg
report={'status':'deploying','startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'functions':{}}
receipt=p/'deployment-receipt.json'
def save():receipt.write_text(json.dumps(report,indent=2)+'\n')
save()
for name,item in m.items():
 l.update_function_code(FunctionName=name,ZipFile=(p/name/'candidate.zip').read_bytes(),RevisionId=before[name]['RevisionId'],Publish=False)
 for _ in range(60):
  cfg=l.get_function_configuration(FunctionName=name)
  if cfg['LastUpdateStatus']!='InProgress':break
  time.sleep(1)
 assert cfg['LastUpdateStatus']=='Successful' and cfg['CodeSha256']==item['candidateHash']
 assert config_hash(cfg)==config_hash(before[name]), 'Unexpected configuration change'
 report['functions'][name]={'hash':cfg['CodeSha256'],'revision':cfg['RevisionId'],'configurationUnchanged':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()};save();print(name,'code updated; configuration unchanged',flush=True)
report['status']='deployed_awaiting_public_verification';save()
