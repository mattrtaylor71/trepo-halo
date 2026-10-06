"""Private IAM-only service. No function URL or API Gateway route is permitted."""
import json
import os
import ssl
from . import events,sources,store


def connect():
    import pymysql
    import boto3
    # Dedicated runtime secret. Never print secret values or provider exceptions.
    secret=json.loads(boto3.client('secretsmanager').get_secret_value(SecretId=os.environ['HOUSEHOLD_BRAIN_DB_SECRET'])['SecretString'])
    return pymysql.connect(host=secret['host'],port=int(secret.get('port',3306)),user=secret['username'],password=secret['password'],
        database=secret['dbname'],autocommit=True,cursorclass=pymysql.cursors.DictCursor,
        charset='utf8mb4',connect_timeout=3,read_timeout=10,write_timeout=10,
        ssl=ssl.create_default_context(cadata=secret.get('ca_bundle')))


def service(conn,request):
    if not isinstance(request,dict) or set(request)-{'operation','actor','household','audience','purpose','fence'}:
        return {'statusCode':400,'error':'invalid_request'}
    actor=request.get('actor')
    if not isinstance(actor,str) or not 1<=len(actor)<=64 or request.get('audience','personal') not in {'personal','household'}:
        return {'statusCode':400,'error':'invalid_identity'}
    with conn.cursor() as cur:
        if sources.membership(cur,actor)!=request.get('household'):
            return {'statusCode':403,'error':'membership_changed'}
    if request.get('operation','context')=='context':
        result=store.read_context(conn,actor,request.get('purpose','thyme'),actor=actor if request.get('audience','personal')=='personal' else None)
        return {'statusCode':200,**result} if result else {'statusCode':404,'error':'brain_not_enabled'}
    if request.get('operation')=='validate':
        fence=request.get('fence') or {}
        current=fence.get('actor')==actor and fence.get('household')==request.get('household') and store.is_current(conn,fence)
        return {'statusCode':200,'current':bool(current)}
    return {'statusCode':403,'error':'internally_managed'}


def reject_http(event):
    return not isinstance(event,dict) or any(k in event for k in ['requestContext','httpMethod','rawPath','body','headers'])


def reader_handler(event,context):
    if reject_http(event):
        return {'statusCode':403,'error':'private_service_only'}
    conn=None
    try:
        conn=connect()
        return service(conn,event)
    except (ValueError,PermissionError):
        return {'statusCode':409,'error':'context_changed_or_unavailable'}
    except Exception:
        print(json.dumps({'event':'household_brain','operation':'read','outcome':'unavailable'}))
        return {'statusCode':503,'error':'context_unavailable'}
    finally:
        if conn:conn.close()


def worker_handler(event,context):
    conn=connect()
    try:
        result=store.run_worker(conn)
        print(json.dumps({'event':'household_brain','operation':'refresh',**result}))
        if result['failed']:
            raise RuntimeError('household_brain_refresh_failed')
        return result
    finally:
        conn.close()


def event_handler(event,context):
    if reject_http(event):
        return {'statusCode':403,'error':'private_service_only'}
    conn=connect()
    try:
        # Fixed per Lambda, never chosen by a client or model.
        return events.ingest(conn,os.environ['HOUSEHOLD_BRAIN_PRODUCER'],event)
    finally:
        conn.close()
