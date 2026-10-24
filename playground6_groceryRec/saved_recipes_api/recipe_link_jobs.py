"""Bounded durable URL recovery. Paid work never runs in the sweeper.

A reserved existing GSI partition discovers due jobs. Each accepted recipe revision
has one job; leases fence deliveries, and SQL revisions fence user edits/deletions.
"""
import hashlib
import json
import os
import re
import time
import uuid
from types import SimpleNamespace
from boto3.dynamodb.conditions import Key

PARTITION = '__trepo_recipe_link_recovery_v1__'
TASK = 'process_saved_recipe_link_v1'
MAX_ATTEMPTS = 3
LEASE_SECONDS = 150
MAX_URL = 4000
_ready_tables = set()


def due(epoch):
    return '%012d' % int(epoch)


def conflict(exc):
    return getattr(exc, 'response', {}).get('Error', {}).get('Code') == 'ConditionalCheckFailedException'


def enqueue(api, owner, recipe_id, url, revision, request_id=None):
    if not revision or not api._valid_recipe_repair_url(url) or len(url) > MAX_URL:
        return False
    now = int(time.time())
    job_id = 'recipe-link-v1:' + hashlib.sha256((owner+'\n'+recipe_id+'\n'+revision).encode()).hexdigest()
    job = {'job_id': job_id, 'type': 'recipe_link_recovery_v1', 'owner': owner, 'ai_consent_actor': api._ai_recipe_owner.get(),
        'recipe_id': recipe_id, 'url': url, 'expected_revision': revision,
        'request_id': request_id or '', 'status': 'PENDING', 'attempts': 0,
        'user_id': PARTITION, 'created_at': due(now), 'registered_at': now,
        'retry_at': now, 'deadline': now + 1800, 'ttl': now + 30*86400}
    table = api._jobs_table()
    try:
        table.put_item(Item=job, ConditionExpression='attribute_not_exists(job_id)')
    except Exception as exc:
        if not conflict(exc):
            raise
        job = table.get_item(Key={'job_id': job_id}, ConsistentRead=True).get('Item') or job
    if job.get('status') in ('PENDING', 'RETRY'):
        try:
            dispatch(api, job_id)
        except Exception as exc:
            api._log_event(request_id, 'recipe_link_dispatch_pending', failure_type=type(exc).__name__)
    return True  # A durable due item survives dispatch failure.


def dispatch(api, job_id):
    response = api.boto3.client('lambda').invoke(
        FunctionName=os.environ['AWS_LAMBDA_FUNCTION_NAME'], InvocationType='Event',
        Payload=json.dumps({'async_task': TASK, 'job_id': job_id}).encode())
    if response.get('StatusCode') != 202:
        raise RuntimeError('link_dispatch_unavailable')


def finish_recipe(api, job):
    """End only the revision this job owns. No resurrection or edit overwrite."""
    conn = api._mysql_conn()
    try:
        table = api._saved_recipes_table(job['owner'])
        fields = api._drift_safe_select(conn, table)
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(f'SELECT {fields} FROM `{table}` WHERE _id=%s AND _owner=%s FOR UPDATE',
                        [job['recipe_id'], job['owner']])
            row = cur.fetchone()
            if not row or api._saved_recipe_content_revision(row) != job['expected_revision']:
                conn.rollback()
                return 'changed_or_deleted'
            if row.get('status') not in ('processing', 'repairing'):
                conn.rollback()
                return 'already_terminal'
            current = api._serialize_row(row)
            status = 'ready' if api.recipe_content_complete(current) else 'failed'
            notes = [n for n in current.get('notes', []) if n != "We couldn't read the full recipe from this post yet — we're still working on it."]
            cur.execute(f'UPDATE `{table}` SET status=%s,notes=%s WHERE _id=%s AND _owner=%s',
                        [status, json.dumps(notes), job['recipe_id'], job['owner']])
        api._dual_write_saved_recipe_to_shared(conn, job['owner'], job['recipe_id'],
            request_id=job.get('request_id'), transactional=True, required=True)
        conn.commit()
        return status
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def run(api, job_id):
    token = api._ai_recipe_owner.set(None)
    try:
        return _run_scoped(api, job_id)
    finally:
        api._ai_recipe_owner.reset(token)


def _run_scoped(api, job_id):
    table = api._jobs_table()
    job = table.get_item(Key={'job_id': job_id}, ConsistentRead=True).get('Item')
    if not job or job.get('type') != 'recipe_link_recovery_v1' or job.get('status') in ('COMPLETED', 'FAILED', 'SKIPPED'):
        return {'outcome': 'terminal_or_missing'}
    now = int(time.time()); token = uuid.uuid4().hex
    try:
        claimed = table.update_item(Key={'job_id': job_id},
            UpdateExpression='SET #s=:running, lease_until=:lease, lease_token=:token ADD attempts :one',
            ConditionExpression='(#s=:pending OR #s=:retry OR #s=:running) AND '
                '(attribute_not_exists(lease_until) OR lease_until <= :now) AND retry_at <= :now',
            ExpressionAttributeNames={'#s':'status'},
            ExpressionAttributeValues={':running':'RUNNING', ':pending':'PENDING', ':retry':'RETRY',
                ':lease':now+LEASE_SECONDS, ':token':token, ':one':1, ':now':now},
            ReturnValues='ALL_NEW')['Attributes']
    except Exception as exc:
        if conflict(exc):
            return {'outcome':'leased_or_not_due'}
        raise
    api._ai_recipe_owner.set(claimed.get('ai_consent_actor'))
    job = claimed; attempt = int(job['attempts']); retry = False
    mode = bool(job.get('force_media')); correction = job.get('correction') or None
    reason = 'exhausted'; outcome = 'FAILED'
    if attempt > MAX_ATTEMPTS or int(job['deadline']) <= now:
        reason = finish_recipe(api, job)
    else:
        try:
            result = api._repair_saved_recipe(job['owner'], job['recipe_id'], job['url'],
                request_id=job.get('request_id'), expected_revision=job['expected_revision'],
                recovery_attempt=attempt, force_media=mode, recovery_correction=correction)
            reason = result.get('reason') or result.get('outcome') or 'unknown'
            outcome = 'COMPLETED' if result.get('outcome') == 'ready' else 'SKIPPED' if result.get('outcome') == 'skipped' else 'FAILED'
            if outcome != 'COMPLETED':
                finish_recipe(api, job)
        except Exception as exc:
            mode = mode or bool(getattr(exc, 'force_media', False))
            correction = getattr(exc, 'correction', None) or correction
            safe_reasons={'pending','running','failed','media_continuation','budget_exhausted','model_transient','evidence_insufficient','source_unavailable'}
            reason = str(exc) if str(exc) in safe_reasons else type(exc).__name__
            retry = not isinstance(exc, api._ai_consent.AIConsentRequired) and attempt < MAX_ATTEMPTS
            if not retry:
                reason = finish_recipe(api, job)
    if retry:
        retry_at = int(time.time()) + min(240, 60 * 2**(attempt-1))
        table.update_item(Key={'job_id':job_id},
            UpdateExpression='SET #s=:retry,retry_at=:at,created_at=:due,force_media=:mode,correction=:correction,last_reason=:reason REMOVE lease_until,lease_token',
            ConditionExpression='lease_token=:token', ExpressionAttributeNames={'#s':'status'},
            ExpressionAttributeValues={':retry':'RETRY', ':at':retry_at, ':due':due(retry_at),
                ':mode':mode, ':correction':correction or '', ':reason':reason, ':token':token})
        outcome = 'RETRY'
    else:
        table.update_item(Key={'job_id':job_id},
            UpdateExpression='SET #s=:status,last_reason=:reason,completed_at=:now REMOVE user_id,created_at,lease_until,lease_token',
            ConditionExpression='lease_token=:token', ExpressionAttributeNames={'#s':'status'},
            ExpressionAttributeValues={':status':outcome, ':reason':reason, ':now':int(time.time()), ':token':token})
    api._log_event(job.get('request_id'), 'recipe_link_job_outcome', recipe_id=job['recipe_id'],
        owner=job['owner'], outcome=outcome, attempt=attempt, reason=reason)
    return {'outcome':outcome,'attempt':attempt}


def recover(api):
    now = int(time.time()); table = api._jobs_table()
    response = table.query(IndexName='owner-created-index',
        KeyConditionExpression=Key('user_id').eq(PARTITION) & Key('created_at').lte(due(now)), Limit=25)
    dispatched = 0
    for job in response.get('Items', []):
        if int(job.get('lease_until',0)) > now or int(job.get('retry_at',0)) > now:
            continue
        try:
            dispatch(api, job['job_id']); dispatched += 1
        except Exception as exc:
            api._log_event(None, 'recipe_link_sweep_dispatch_failed', failure_type=type(exc).__name__)
    api._log_event(None, 'recipe_link_recovery_sweep', due_count=len(response.get('Items',[])), dispatched=dispatched)
    return {'dispatched':dispatched}


def ensure_url_capacity(conn, tables, urls):
    """Expand only when a long URL arrives; never silently truncate accepted input."""
    if any(len(url or '') > MAX_URL for url in urls):
        raise ValueError('source_url_too_long')
    if max([len(url or '') for url in urls] or [0]) <= 1000:
        return
    for table in tables:
        if table in _ready_tables:
            continue
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', table):
            raise ValueError('invalid_table')
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{table}` WHERE Field IN ('source_url','resolved_url')")
            columns = cur.fetchall(); changes = []
            for column in columns:
                kind = column['Type'].lower()
                width = re.fullmatch(r'varchar\((\d+)\)', kind)
                if width and int(width[1]) < MAX_URL:
                    changes.append('MODIFY COLUMN `'+column['Field']+'` VARCHAR(4000) NOT NULL')
            if changes:
                cur.execute('SELECT @@SESSION.lock_wait_timeout previous_timeout')
                previous = cur.fetchone()['previous_timeout']
                cur.execute('SET SESSION lock_wait_timeout=2')
                try:
                    cur.execute(f'ALTER TABLE `{table}` '+','.join(changes)+', ALGORITHM=INPLACE, LOCK=NONE')
                finally:
                    cur.execute('SET SESSION lock_wait_timeout=%s',[previous])
        _ready_tables.add(table)
