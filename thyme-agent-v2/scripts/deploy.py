"""Deploy only the new private pilot infrastructure. Never updates existing Trepo functions."""
import argparse,boto3,json,pathlib,secrets,time,hashlib
from botocore.exceptions import ClientError
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--base-config',required=True);args=p.parse_args()
root=pathlib.Path(args.private_dir);root.mkdir(exist_ok=True);root.chmod(0o700)
s=boto3.Session(profile_name='trepo-dev',region_name='us-east-1');ddb=s.client('dynamodb');iam=s.client('iam');lam=s.client('lambda');logs=s.client('logs');account=s.client('sts').get_caller_identity()['Account'];region='us-east-1';name='trepo-thyme-agent-v2-pilot'
secretfile=root/'bridge-secret';
if not secretfile.exists():secretfile.write_text(secrets.token_hex(32));secretfile.chmod(0o600)
try:table=ddb.describe_table(TableName=name)['Table']
except ddb.exceptions.ResourceNotFoundException:
 ddb.create_table(TableName=name,KeySchema=[{'AttributeName':'pk','KeyType':'HASH'},{'AttributeName':'sk','KeyType':'RANGE'}],AttributeDefinitions=[{'AttributeName':'pk','AttributeType':'S'},{'AttributeName':'sk','AttributeType':'S'}],BillingMode='PAY_PER_REQUEST',StreamSpecification={'StreamEnabled':True,'StreamViewType':'NEW_AND_OLD_IMAGES'},SSESpecification={'Enabled':True},Tags=[{'Key':'purpose','Value':'private-founder-pilot'}]);ddb.get_waiter('table_exists').wait(TableName=name);table=ddb.describe_table(TableName=name)['Table'];ddb.update_time_to_live(TableName=name,TimeToLiveSpecification={'Enabled':True,'AttributeName':'expires'});ddb.update_continuous_backups(TableName=name,PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled':True})
try:role=iam.get_role(RoleName=name)['Role']
except iam.exceptions.NoSuchEntityException:role=iam.create_role(RoleName=name,AssumeRolePolicyDocument=json.dumps({'Version':'2012-10-17','Statement':[{'Effect':'Allow','Principal':{'Service':'lambda.amazonaws.com'},'Action':'sts:AssumeRole'}]}))['Role']
base=json.loads(pathlib.Path(args.base_config).read_text())['Environment']['Variables'];generators=[v for k,v in base.items() if k.endswith('_ARN') and ':lambda:' in v]
policy={'Version':'2012-10-17','Statement':[{'Effect':'Allow','Action':['logs:CreateLogStream','logs:PutLogEvents'],'Resource':f'arn:aws:logs:{region}:{account}:log-group:/aws/lambda/{name}*:*'},{'Effect':'Allow','Action':['dynamodb:GetItem','dynamodb:Query','dynamodb:PutItem','dynamodb:UpdateItem','dynamodb:TransactWriteItems'],'Resource':table['TableArn']},{'Effect':'Allow','Action':['dynamodb:DescribeStream','dynamodb:GetRecords','dynamodb:GetShardIterator'],'Resource':table['LatestStreamArn']},{'Effect':'Allow','Action':['dynamodb:ListStreams'],'Resource':'*'},{'Effect':'Allow','Action':['lambda:InvokeFunction'],'Resource':generators}]}
iam.put_role_policy(RoleName=name,PolicyName='pilot-runtime',PolicyDocument=json.dumps(policy))
private=root/'deployment-private.json';old=json.loads(private.read_text()) if private.exists() else {}
principal_map=old.get('principal_map',{'qualification-read-only':'7d7df434-d942-4037-b054-2d3005ea6abc'})
env={**base,'STATE_TABLE':name,'BRIDGE_SECRET':secretfile.read_text(),'PRINCIPAL_MAP':json.dumps(principal_map),'PILOT_ACTORS':base['FAMILY_BRAIN_ACTORS'],'TREPO_TOOLS_ROOT':'/var/task'}
# Legacy session variables are irrelevant to this runtime; reclaim the Lambda env budget.
for k in ['SESSION_TABLE_NAME','SESSION_MEMORY_TTL_DAYS','SESSION_MEMORY_MAX_MESSAGES']:env.pop(k,None)
code=pathlib.Path(args.archive).read_bytes();functions=[]
for suffix,timeout in [('-api',30),('-worker',240)]:
 fn=name+suffix;group='/aws/lambda/'+fn
 try:logs.create_log_group(logGroupName=group)
 except logs.exceptions.ResourceAlreadyExistsException:pass
 logs.put_retention_policy(logGroupName=group,retentionInDays=14)
 try:
  existing=lam.get_function_configuration(FunctionName=fn)
  lam.update_function_configuration(FunctionName=fn,Role=role['Arn'],Runtime='nodejs22.x',Handler='thyme-agent-v2/src/entry.handler',Timeout=timeout,MemorySize=1024,Environment={'Variables':env},RevisionId=existing['RevisionId'])
  lam.get_waiter('function_updated_v2').wait(FunctionName=fn)
  current=lam.get_function_configuration(FunctionName=fn);lam.update_function_code(FunctionName=fn,ZipFile=code,RevisionId=current['RevisionId']);lam.get_waiter('function_updated_v2').wait(FunctionName=fn)
 except lam.exceptions.ResourceNotFoundException:
  for attempt in range(8):
   try:lam.create_function(FunctionName=fn,Role=role['Arn'],Runtime='nodejs22.x',Handler='thyme-agent-v2/src/entry.handler',Timeout=timeout,MemorySize=1024,Environment={'Variables':env},Code={'ZipFile':code},Architectures=['x86_64'],Tags={'purpose':'private-founder-pilot'});break
   except lam.exceptions.InvalidParameterValueException as e:
    if 'role' not in str(e).lower() or attempt==7:raise
    time.sleep(3)
  lam.get_waiter('function_active_v2').wait(FunctionName=fn)
 lam.put_function_concurrency(FunctionName=fn,ReservedConcurrentExecutions=3 if suffix=='-api' else 2);functions.append(fn)
api,worker=functions
try:url=lam.get_function_url_config(FunctionName=api)['FunctionUrl']
except lam.exceptions.ResourceNotFoundException:url=lam.create_function_url_config(FunctionName=api,AuthType='NONE',InvokeMode='BUFFERED')['FunctionUrl']
for sid,action,extra in [('url-entry','lambda:InvokeFunctionUrl',{'FunctionUrlAuthType':'NONE'}),('url-invoke','lambda:InvokeFunction',{'InvokedViaFunctionUrl':True})]:
 try:lam.add_permission(FunctionName=api,StatementId=sid,Action=action,Principal='*',**extra)
 except lam.exceptions.ResourceConflictException:pass
for fn,arn,kwargs in [(worker,table['LatestStreamArn'],{'StartingPosition':'TRIM_HORIZON','BatchSize':1,'FilterCriteria':{'Filters':[{'Pattern':json.dumps({'eventName':['INSERT','MODIFY'],'dynamodb':{'NewImage':{'type':{'S':['request']},'status':{'S':['queued']}}}})}]},'MaximumRetryAttempts':-1,'ParallelizationFactor':1} )]:
 existing=lam.list_event_source_mappings(FunctionName=fn,EventSourceArn=arn)['EventSourceMappings']
 if existing:lam.update_event_source_mapping(UUID=existing[0]['UUID'],BatchSize=1,Enabled=True)
 if not existing:lam.create_event_source_mapping(FunctionName=fn,EventSourceArn=arn,Enabled=True,FunctionResponseTypes=['ReportBatchItemFailures'],**kwargs)
result={'api_function':api,'worker_function':worker,'url':url,'table':name,'principal_map':principal_map,'archive_sha256':hashlib.sha256(code).hexdigest(),'deployed_at':time.time()};private.write_text(json.dumps(result,indent=2));private.chmod(0o600)
print(json.dumps({k:v for k,v in result.items() if k!='principal_map'},indent=2))
