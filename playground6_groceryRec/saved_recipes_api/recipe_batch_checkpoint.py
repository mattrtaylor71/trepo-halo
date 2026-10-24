import ai_consent_guard as _ai_consent
"""Recoverable image processing, one bounded stage per Lambda invocation.

The existing job remains the public status record. Recovery markers share its
owner-created-index in a reserved partition; checkpoints are immutable S3 objects.
No image bytes or OCR text are placed in DynamoDB or logs.
"""
import base64
import contextvars
import copy
import gzip
import hashlib
import json
import math
import os
import time
import uuid
from types import SimpleNamespace

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

VERSION = 2
PARTITION = '__trepo_recipe_recovery_v2__'
LEASE_SECONDS = 150
MAX_ATTEMPTS = 3
RETENTION_SECONDS = 30 * 86400
_deadline = contextvars.ContextVar('recipe_stage_deadline', default=None)


def _emit(api, name, job, **fields):
    api._log_event(job.get('request_id'), name, job_id=job['job_id'], owner=job['owner'],
        traffic_class='review' if job['owner'] == '8fcb37c1-0927-4c31-856a-5b761efc226c' else 'unclassified', **fields)


def bounded_client(client):
    """Do not multiply SDK retries by the durable stage retry budget."""
    if _deadline.get() is None:
        return client
    def create(**kwargs):
        remaining = _deadline.get() - time.monotonic() - 10
        if remaining < 2:
            raise TimeoutError('Recipe stage time budget exhausted')
        return _ai_consent.provider_call(client.with_options(timeout=min(20, remaining), max_retries=0).chat.completions.create)(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def enabled(owner):
    setting = os.getenv('RECIPE_CHECKPOINT_OWNERS', '')
    return setting == '*' or owner in [v.strip() for v in setting.split(',') if v.strip()]


def authorize(api, event, owner):
    import trepo_auth
    headers = {str(k).lower(): v for k, v in ((event or {}).get('headers') or {}).items()}
    header = headers.get('authorization') or ''
    claims = trepo_auth._verify_jwt(header[7:].strip()) if header.lower().startswith('bearer ') else None
    try:
        expiry = float(claims['exp'])
        valid = math.isfinite(expiry) and expiry > time.time() and owner in (
            str(claims.get('owner_id', '')), str(claims.get('user_id', '')))
    except (TypeError, KeyError, ValueError):
        valid = False
    if not valid:
        raise api.ServiceError('Please sign in again before importing recipes.', status_code=403)


def enqueue(api, owner, submission, event=None, request_id=None):
    authorize(api, event, owner)
    digest = hashlib.sha256(json.dumps(submission, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    job_id = uuid.uuid5(uuid.NAMESPACE_URL, 'trepo:recipe-upload:v2:' + owner + ':' + digest).hex
    table = api._jobs_table()
    existing = table.get_item(Key={'job_id': job_id}, ConsistentRead=True).get('Item')
    if existing:
        if existing.get('owner') != owner or existing.get('input_digest') != digest:
            raise api.ServiceError('Upload identity conflict.', status_code=409)
        _emit(api, 'recipe_checkpoint_accept_replayed', existing, status=existing['status'])
        return existing
    now = api._utc_now_iso()
    payload_key = 'recipe-images/checkpoints-v2/%s/%s/submission.json' % (owner, job_id)
    api.boto3.client('s3').put_object(Bucket=os.environ['BUCKET_NAME'], Key=payload_key,
        Body=json.dumps(submission).encode('utf-8'), ContentType='application/json', ServerSideEncryption='AES256')
    job = {'job_id': job_id, 'owner': owner, 'ai_consent_actor': api._ai_recipe_owner.get(), 'input_digest': digest,
        'type': api._SAVED_RECIPE_BATCH_JOB_TYPE, 'status': 'PENDING',
        'image_count': len(submission.get('images') or []), 'result_count': 0,
        'recipe_ids': [], 'results': [], 'partial_errors': [],
        'request_id': request_id, 'created_at': now, 'updated_at': now,
        'submission_kind': 'images', 'recovery_deadline': int(time.time()) + 86400,
        'payload_s3_key': payload_key}
    register(api, job)
    try:
        table.put_item(Item=job, ConditionExpression='attribute_not_exists(job_id)')
    except ClientError as exc:
        if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
            raise
        return table.get_item(Key={'job_id': job_id}, ConsistentRead=True)['Item']
    try:
        dispatch(api, job)
    except Exception as exc:
        api._log_event(request_id, 'recipe_checkpoint_dispatch_pending', error_type=type(exc).__name__)
    _emit(api, 'recipe_checkpoint_accepted', job, image_count=job['image_count'])
    return job


def _marker_id(job_id):
    return 'recipe-recovery-v2:' + job_id


def _due_string(epoch):
    return '%012d' % int(epoch)


def register(api, job):
    """Write discovery before accepting the parent; a failed put leaves a harmless marker."""
    job['checkpoint_version'] = VERSION
    now = int(time.time())
    api._jobs_table().put_item(Item={
        'job_id': _marker_id(job['job_id']), 'type': 'recipe_recovery_v2',
        'target_job_id': job['job_id'], 'user_id': PARTITION,
        'created_at': _due_string(now), 'registered_at': now,
        'ttl': now + RETENTION_SECONDS,
    })


def dispatch(api, job):
    result = api.boto3.client('lambda').invoke(
        FunctionName=os.environ['AWS_LAMBDA_FUNCTION_NAME'], InvocationType='Event',
        Payload=json.dumps({'async_task': api._ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH,
                            'owner': job['owner'], 'ai_consent_actor': job.get('ai_consent_actor'), 'job_id': job['job_id']}).encode())
    if result.get('StatusCode') != 202:
        raise RuntimeError('Recipe worker dispatch was not accepted')


def recover(api):
    """Bound each sweep to 25 due markers; no scan of customers' historical jobs."""
    table = api._jobs_table()
    now = int(time.time())
    response = table.query(IndexName='owner-created-index',
        KeyConditionExpression=Key('user_id').eq(PARTITION) & Key('created_at').lte(_due_string(now)), Limit=25)
    dispatched = 0
    oldest = {'review': 0, 'unclassified': 0}
    for marker in response.get('Items', []):
        job = table.get_item(Key={'job_id': marker['target_job_id']}, ConsistentRead=True).get('Item')
        if not job:
            # A registration may precede the parent by milliseconds; never delete that race.
            if now - int(marker.get('registered_at', now)) > 600:
                table.delete_item(Key={'job_id': marker['job_id']})
            continue
        if job.get('checkpoint_version') != VERSION or job.get('status') in ('COMPLETED', 'FAILED'):
            table.delete_item(Key={'job_id': marker['job_id']})
            continue
        traffic = 'review' if job['owner'] == '8fcb37c1-0927-4c31-856a-5b761efc226c' else 'unclassified'
        oldest[traffic] = max(oldest[traffic], now - int(marker.get('registered_at', now)))
        if int(job.get('lease_until', 0)) > now or int(job.get('retry_at', 0)) > now:
            continue
        if int(job.get('recovery_deadline', now + 1)) <= now:
            saved_indexes = {int(index) for result in job.get('results', []) for index in result.get('image_indexes', [])}
            errors = list(job.get('partial_errors', [])) + [{
                'image_indexes': [index for index in range(int(job.get('image_count', 0))) if index not in saved_indexes],
                'error': 'Processing could not finish. Please retry the remaining images.',
                'status_code': 503, 'retryable': True}]
            try:
                table.update_item(Key={'job_id': job['job_id']},
                    UpdateExpression='SET #status = :terminal, #error = :error, partial_errors = :errors, completed_at = :completed, #ttl = :ttl',
                    ConditionExpression='(#status = :pending OR #status = :running) AND '
                        '(attribute_not_exists(lease_until) OR lease_until <= :now)',
                    ExpressionAttributeNames={'#status': 'status', '#error': 'error', '#ttl': 'ttl'},
                    ExpressionAttributeValues={':terminal': 'COMPLETED' if job.get('result_count') else 'FAILED',
                        ':pending': 'PENDING', ':running': 'RUNNING', ':errors': errors,
                        ':now': now, ':completed': api._utc_now_iso(), ':ttl': None if job.get('admission_operation_id') else now + RETENTION_SECONDS,
                        ':error': 'Processing could not finish. Please retry the remaining images.'})
                table.delete_item(Key={'job_id': marker['job_id']})
                _emit(api, 'recipe_checkpoint_recovery_expired', job,
                      outcome='partial' if job.get('result_count') else 'failed')
            except ClientError as exc:
                if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
                    raise
            continue
        try:
            dispatch(api, job)
            dispatched += 1
        except Exception as exc:
            api._log_event(None, 'recipe_recovery_dispatch_failed', error_type=type(exc).__name__)
    api._log_event(None, 'recipe_recovery_sweep', due_count=len(response.get('Items', [])), dispatched=dispatched)
    for traffic, age in oldest.items():
        api._log_event(None, 'recipe_checkpoint_queue_age', TrafficClass=traffic, OldestPendingSeconds=age,
            _aws={'Timestamp': now * 1000, 'CloudWatchMetrics': [{'Namespace': 'Trepo/RecipeImports',
                'Dimensions': [['TrafficClass']], 'Metrics': [{'Name': 'OldestPendingSeconds', 'Unit': 'Seconds'}]}]})
    return {'ok': True, 'dispatched': dispatched}


def _encode(value):
    if isinstance(value, bytes):
        return {'__recipe_bytes__': base64.b64encode(value).decode('ascii')}
    raise TypeError('Unsupported checkpoint value: ' + type(value).__name__)


def _decode(value):
    if set(value) == {'__recipe_bytes__'}:
        return base64.b64decode(value['__recipe_bytes__'], validate=True)
    return value


def _load(api, job):
    key = job.get('checkpoint_key')
    if not key:
        return {'image_cursor': 0, 'fragments': [], 'clusters': None, 'cluster_cursor': 0,
                'results': [], 'errors': [], 'attempts': {}}
    body = api.boto3.client('s3').get_object(Bucket=os.environ['BUCKET_NAME'], Key=key)['Body']
    try:
        return json.loads(gzip.decompress(body.read()).decode('utf-8'), object_hook=_decode)
    finally:
        body.close()


def _save(api, job, state):
    key = 'recipe-images/checkpoints-v2/%s/%s/%s.json.gz' % (job['owner'], job['job_id'], uuid.uuid4().hex)
    data = gzip.compress(json.dumps(state, default=_encode).encode('utf-8'))
    api.boto3.client('s3').put_object(Bucket=os.environ['BUCKET_NAME'], Key=key,
        Body=data, ContentType='application/json', ContentEncoding='gzip', ServerSideEncryption='AES256')
    return key


def _freeze_fragment(api, job, fragment):
    """Keep bytes out of repeated JSON checkpoints, including eight-page recipes."""
    prepared = fragment['prepared_image']
    key = 'recipe-images/checkpoints-v2/%s/%s/image-%s-%s' % (
        job['owner'], job['job_id'], fragment['image_index'], uuid.uuid4().hex)
    api.boto3.client('s3').put_object(Bucket=os.environ['BUCKET_NAME'], Key=key,
        Body=prepared['image_bytes'], ContentType=prepared.get('content_type', 'application/octet-stream'),
        ServerSideEncryption='AES256')
    prepared['checkpoint_image_key'] = key
    prepared.pop('image_bytes')
    return fragment


def _hydrate_cluster(api, cluster):
    cluster = copy.deepcopy(cluster)
    for prepared in cluster.get('prepared_images', []):
        if 'image_bytes' in prepared:
            continue
        body = api.boto3.client('s3').get_object(Bucket=os.environ['BUCKET_NAME'],
                                               Key=prepared['checkpoint_image_key'])['Body']
        try:
            prepared['image_bytes'] = body.read()
        finally:
            body.close()
    return cluster


def _update(table, job_id, token, **fields):
    # A late worker can write an orphan S3 checkpoint, but cannot publish it or overwrite a successor.
    names = {'#lease': 'lease_token'}
    values = {':lease': token}
    sets = []
    for index, (key, value) in enumerate(fields.items()):
        names['#f%d' % index] = key
        values[':v%d' % index] = value
        sets.append('#f%d = :v%d' % (index, index))
    return table.update_item(Key={'job_id': job_id}, UpdateExpression='SET ' + ', '.join(sets),
        ConditionExpression='#lease = :lease', ExpressionAttributeNames=names,
        ExpressionAttributeValues=values, ReturnValues='ALL_NEW')['Attributes']


def _failure(api, exc, indexes):
    status = exc.status_code if isinstance(exc, api.ServiceError) else 503
    return {'image_indexes': indexes, 'status_code': status,
            'retryable': status in (408, 429) or status >= 500,
            'error': str(exc) if isinstance(exc, api.ServiceError) else 'This recipe could not be processed. Please try again.'}


def run(api, event, context=None):
    table = api._jobs_table()
    job_id, owner = event['job_id'], event['owner']
    token, now = uuid.uuid4().hex, int(time.time())
    try:
        job = table.update_item(Key={'job_id': job_id},
            UpdateExpression='SET lease_token = :token, lease_until = :until, #status = :running',
            ConditionExpression='#owner = :owner AND checkpoint_version = :version AND '
                '(#status = :pending OR #status = :running) AND '
                '(attribute_not_exists(lease_until) OR lease_until <= :now) AND '
                '(attribute_not_exists(retry_at) OR retry_at <= :now)',
            ExpressionAttributeNames={'#status': 'status', '#owner': 'owner'},
            ExpressionAttributeValues={':token': token, ':until': now + LEASE_SECONDS, ':owner': owner,
                ':version': VERSION, ':pending': 'PENDING', ':running': 'RUNNING', ':now': now},
            ReturnValues='ALL_NEW')['Attributes']
    except ClientError as exc:
        if exc.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return {'ok': True, 'job_id': job_id, 'not_claimed': True}
        raise
    remaining = getattr(context, 'get_remaining_time_in_millis', lambda: 120000)() / 1000
    deadline_token = _deadline.set(time.monotonic() + min(95, max(1, remaining - 20)))
    try:
        state = _load(api, job)
        images = api._load_saved_recipe_batch_payload(job).get('images') or []
        if not images or len(images) > 8:
            raise api.ServiceError('Choose between one and eight recipe images.', status_code=422)
        cursor = state['image_cursor']
        if cursor < len(images):
            step, indexes = 'image:%d' % cursor, [cursor]
        elif state['clusters'] is None:
            step, indexes = 'group', [f['image_index'] for f in state['fragments']]
        elif state['cluster_cursor'] < len(state['clusters']):
            step = 'save:%d' % state['cluster_cursor']
            indexes = state['clusters'][state['cluster_cursor']]['image_indexes']
        else:
            step, indexes = 'finish', []
        if step != 'finish':
            attempts = state['attempts'].get(step, 0) + 1
            state['attempts'][step] = attempts
            # Persist the attempt before invoking a provider so a killed worker consumes a bounded attempt.
            job = _update(table, job_id, token, checkpoint_key=_save(api, job, state))
            try:
                if attempts > MAX_ATTEMPTS:
                    raise api.ServiceError('Processing was interrupted repeatedly. Please retry these images.', status_code=503)
                if step.startswith('image:'):
                    prepared = api._prepare_image_for_vision(images[cursor])
                    fragment = api._extract_recipe_fragment_from_image(
                        prepared, image_index=cursor, request_id=event.get('request_id'))
                    state['fragments'].append(_freeze_fragment(api, job, fragment))
                    state['image_cursor'] += 1
                elif step == 'group':
                    state['clusters'] = [api._merge_image_cluster(c) for c in
                        api._group_image_fragments(state['fragments'], request_id=event.get('request_id'))]
                else:
                    cluster = state['clusters'][state['cluster_cursor']]
                    hydrated = _hydrate_cluster(api, cluster)
                    conn = api._mysql_conn()
                    api._ensure_saved_recipes_table(conn, owner)
                    result, _ = api._save_saved_recipe_record(conn, owner, cluster['extraction'],
                        request_id=event.get('request_id'), prepared_images=hydrated.get('prepared_images') or [])
                    recipe_id = ((result or {}).get('recipe') or {}).get('id')
                    if not recipe_id:
                        raise api.ServiceError('Recipe persistence was not confirmed.', status_code=503)
                    # The legacy insert/hash uniqueness repairs a commit-before-checkpoint crash by returning the same ID.
                    # Reconcile the shared mirror on that dedupe path too, then verify the active read surface.
                    api._dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=event.get('request_id'))
                    if os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true':
                        with conn.cursor() as cur:
                            cur.execute('SELECT _id FROM shared_saved_recipes WHERE owner_id=%s AND _id=%s', [owner, recipe_id])
                            if not cur.fetchone():
                                raise api.ServiceError('Recipe synchronization is pending.', status_code=503)
                    state['results'].append({'recipe_id': recipe_id, 'deduped': bool(result.get('deduped')), 'image_indexes': indexes})
                    state['cluster_cursor'] += 1
            except Exception as exc:
                failure = _failure(api, exc, indexes)
                if failure['retryable'] and attempts < MAX_ATTEMPTS:
                    delay = min(60, 10 * attempts)
                    job = _update(table, job_id, token, checkpoint_key=_save(api, job, state),
                        lease_until=0, retry_at=int(time.time()) + delay, updated_at=api._utc_now_iso())
                    _emit(api, 'recipe_checkpoint_retry', job, stage=step, attempt=attempts, status_code=failure['status_code'])
                    return {'ok': True, 'job_id': job_id, 'retry_scheduled': True}
                # Only a content rejection can split an inferred multipage recipe.
                if step.startswith('save:') and failure['status_code'] == 422 and len(cluster.get('fragments', [])) > 1:
                    state['clusters'][state['cluster_cursor']:state['cluster_cursor'] + 1] = [
                        api._merge_image_cluster({'fragments': [fragment]}) for fragment in cluster['fragments']]
                    state['attempts'].pop(step, None)
                else:
                    state['errors'].append(failure)
                    if step.startswith('image:'):
                        state['image_cursor'] += 1
                    elif step == 'group':
                        # Do not split continuation pages merely because the grouping provider is unavailable.
                        state['clusters'] = []
                    else:
                        state['cluster_cursor'] += 1
        terminal = state['image_cursor'] >= len(images) and state['clusters'] is not None and state['cluster_cursor'] >= len(state['clusters'])
        fields = {'checkpoint_key': _save(api, job, state), 'lease_until': 0, 'retry_at': 0,
                  'updated_at': api._utc_now_iso(), 'results': state['results'],
                  'recipe_ids': [r['recipe_id'] for r in state['results']],
                  'result_count': len(state['results']), 'partial_errors': state['errors']}
        if terminal:
            fields.update(status='COMPLETED' if state['results'] else 'FAILED', completed_at=api._utc_now_iso(),
                error=None if state['results'] else (state['errors'][0]['error'] if state['errors'] else 'No recipe could be read.'),
                ttl=None if job.get('admission_operation_id') else int(time.time()) + RETENTION_SECONDS)
        job = _update(table, job_id, token, **fields)
        if terminal:
            table.delete_item(Key={'job_id': _marker_id(job_id)})
            _emit(api, 'recipe_checkpoint_terminal', job, outcome=(
                'partial' if state['results'] and state['errors'] else 'complete' if state['results'] else 'failed'),
                result_count=len(state['results']), error_count=len(state['errors']))
        else:
            _emit(api, 'recipe_checkpoint_progress', job, stage=step, result_count=len(state['results']))
            try:
                dispatch(api, job)
            except Exception as exc:
                # The independent due-marker sweep owns recovery if dispatch fails.
                api._log_event(None, 'recipe_checkpoint_dispatch_pending', error_type=type(exc).__name__)
        return {'ok': True, 'job_id': job_id, 'status': job['status'], 'count': len(state['results'])}
    finally:
        _deadline.reset(deadline_token)
