"""Real DynamoDB Local tests; opt in with TREPO_RECIPE_DYNAMO_ENDPOINT."""
import copy
import base64
import hashlib
import hmac
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'saved_recipes_api'))
import recipe_batch_checkpoint as worker


class ServiceError(Exception):
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def fixture(monkeypatch):
    endpoint = os.getenv('TREPO_RECIPE_DYNAMO_ENDPOINT')
    if not endpoint:
        pytest.skip('Requires an explicit local DynamoDB fixture endpoint')
    assert endpoint.startswith('http://127.0.0.1:')
    db = boto3.resource('dynamodb', endpoint_url=endpoint, region_name='us-east-1',
                        aws_access_key_id='fixture', aws_secret_access_key='fixture')
    table = db.create_table(TableName='recipe-fixture-' + uuid.uuid4().hex,
        KeySchema=[{'AttributeName': 'job_id', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': k, 'AttributeType': 'S'} for k in ['job_id', 'user_id', 'created_at']],
        GlobalSecondaryIndexes=[{'IndexName': 'owner-created-index',
            'KeySchema': [{'AttributeName': 'user_id', 'KeyType': 'HASH'}, {'AttributeName': 'created_at', 'KeyType': 'RANGE'}],
            'Projection': {'ProjectionType': 'ALL'}}], BillingMode='PAY_PER_REQUEST')
    table.wait_until_exists()
    monkeypatch.setenv('BUCKET_NAME', 'fixture')
    monkeypatch.setenv('AWS_LAMBDA_FUNCTION_NAME', 'fixture')
    monkeypatch.setenv('READ_SHARED_SAVED_RECIPES', 'false')
    objects, saved = {}, {}
    s3 = Mock()
    s3.put_object.side_effect = lambda **kw: objects.__setitem__(kw['Key'], kw['Body'])
    s3.get_object.side_effect = lambda **kw: {'Body': io.BytesIO(objects[kw['Key']])}
    invoke = Mock(return_value={'StatusCode': 202})
    api = SimpleNamespace(_jobs_table=lambda: table, ServiceError=ServiceError,
        boto3=SimpleNamespace(client=lambda service: s3 if service == 's3' else SimpleNamespace(invoke=invoke)),
        _ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH='image_batch', _utc_now_iso=lambda: 'fixture-time',
        _SAVED_RECIPE_BATCH_JOB_TYPE='saved_recipe_image_batch',
        _put_saved_recipe_batch_payload=Mock(return_value='fixture-payload'),
        _log_event=Mock(), _load_saved_recipe_batch_payload=Mock(return_value={'images': ['a', 'b']}),
        _prepare_image_for_vision=Mock(side_effect=lambda image: {'image_bytes': image.encode(), 'content_type': 'image/png'}),
        _extract_recipe_fragment_from_image=Mock(side_effect=lambda prepared, image_index, **kw:
            {'image_index': image_index, 'content': prepared['image_bytes'].decode(), 'prepared_image': prepared}),
        _group_image_fragments=Mock(side_effect=lambda fragments, **kw: [{'fragments': [f]} for f in fragments]),
        _merge_image_cluster=lambda c: {'image_indexes': [f['image_index'] for f in c['fragments']],
            'fragments': c['fragments'], 'extraction': {'content': ''.join(f['content'] for f in c['fragments'])},
            'prepared_images': [f['prepared_image'] for f in c['fragments']]},
        _mysql_conn=Mock(), _ensure_saved_recipes_table=Mock(), _dual_write_saved_recipe_to_shared=Mock())
    def save(conn, owner, extraction, **kw):
        key = (owner, extraction['content'])
        already = key in saved
        saved.setdefault(key, str(uuid.uuid4()))
        return {'recipe': {'id': saved[key]}, 'deduped': already}, 201
    api._save_saved_recipe_record = Mock(side_effect=save)
    job = {'job_id': uuid.uuid4().hex, 'owner': 'fixture-owner', 'type': 'saved_recipe_image_batch', 'status': 'PENDING'}
    worker.register(api, job)
    table.put_item(Item=job)
    state = SimpleNamespace(api=api, table=table, job=job, event={'owner': job['owner'], 'job_id': job['job_id']},
        objects=objects, saved=saved, invoke=invoke, s3=s3)
    state.read = lambda: table.get_item(Key={'job_id': job['job_id']}, ConsistentRead=True)['Item']
    yield state
    table.delete()


def drain(f, maximum=20):
    for _ in range(maximum):
        result = worker.run(f.api, f.event)
        if f.read()['status'] in ('COMPLETED', 'FAILED'):
            return result
        f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET retry_at = :zero',
                            ExpressionAttributeValues={':zero': 0})
    raise AssertionError('Job never reached a terminal state')


def test_checkpoint_roundtrip_and_redelivery_preserve_exact_results(fixture):
    f = fixture
    drain(f)
    first = f.read()
    assert first['status'] == 'COMPLETED' and first['result_count'] == 2
    assert f.api._extract_recipe_fragment_from_image.call_count == 2
    assert worker.run(f.api, f.event)['not_claimed']
    assert f.api._save_saved_recipe_record.call_count == 2 and len(f.saved) == 2
    state = worker._load(f.api, first)
    assert 'image_bytes' not in state['fragments'][0]['prepared_image']
    assert worker._hydrate_cluster(f.api, state['clusters'][0])['prepared_images'][0]['image_bytes'] == b'a'
    assert worker.recover(f.api)['dispatched'] == 0


def test_maximum_eight_images_use_separate_invocations(fixture):
    f = fixture
    f.api._load_saved_recipe_batch_payload.return_value = {'images': list('abcdefgh')}
    drain(f)
    assert f.read()['result_count'] == 8 and len(f.saved) == 8
    assert f.api._prepare_image_for_vision.call_count == 8


def test_provider_retry_does_not_repeat_completed_image(fixture):
    f = fixture
    original = f.api._prepare_image_for_vision.side_effect
    fail = [True]
    def prepare(image):
        if image == 'b' and fail:
            fail.pop()
            raise ServiceError('provider busy', status_code=429)
        return original(image)
    f.api._prepare_image_for_vision.side_effect = prepare
    drain(f)
    assert f.read()['result_count'] == 2
    assert [c.args[0] for c in f.api._prepare_image_for_vision.call_args_list] == ['a', 'b', 'b']


def test_partial_result_keeps_success_and_precise_failed_indexes(fixture):
    f = fixture
    f.api._prepare_image_for_vision.side_effect = lambda image: (
        {'image_bytes': b'a'} if image == 'a' else (_ for _ in ()).throw(ServiceError('not a recipe', 422)))
    drain(f)
    job = f.read()
    assert job['status'] == 'COMPLETED' and job['result_count'] == 1
    assert job['partial_errors'][0]['image_indexes'] == [1]
    assert not job['partial_errors'][0]['retryable']


def test_storage_failure_retries_one_cluster_without_splitting_pages(fixture):
    f = fixture
    f.api._group_image_fragments.side_effect = lambda fragments, **kw: [{'fragments': fragments}]
    f.api._save_saved_recipe_record.side_effect = ServiceError('storage unavailable', 503)
    drain(f)
    assert f.read()['status'] == 'FAILED'
    assert f.api._save_saved_recipe_record.call_count == 3
    assert f.read()['partial_errors'][0]['image_indexes'] == [0, 1]
    assert f.api._extract_recipe_fragment_from_image.call_count == 2


def test_commit_before_checkpoint_crash_recovers_same_recipe_id(fixture):
    f = fixture
    for _ in range(3):
        worker.run(f.api, f.event)
    original = f.api._save_saved_recipe_record.side_effect
    crash = [True]
    class WorkerKilled(BaseException): pass
    def save(*args, **kwargs):
        result = original(*args, **kwargs)
        if crash:
            crash.pop()
            raise WorkerKilled()
        return result
    f.api._save_saved_recipe_record.side_effect = save
    with pytest.raises(WorkerKilled):
        worker.run(f.api, f.event)
    assert len(f.saved) == 1 and f.read()['lease_until'] > time.time()
    assert worker.run(f.api, f.event)['not_claimed']
    f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET lease_until = :zero',
                        ExpressionAttributeValues={':zero': 0})
    drain(f)
    assert len(f.saved) == 2 and f.read()['result_count'] == 2
    assert len(set(f.read()['recipe_ids'])) == 2
    assert f.read()['results'][0]['deduped']


def test_live_lease_and_wrong_owner_cannot_process_or_overwrite(fixture):
    f = fixture
    f.table.update_item(Key={'job_id': f.job['job_id']},
        UpdateExpression='SET lease_token = :token, lease_until = :until',
        ExpressionAttributeValues={':token': 'new-worker', ':until': int(time.time()) + 90})
    assert worker.run(f.api, f.event)['not_claimed']
    assert worker.run(f.api, {**f.event, 'owner': 'someone-else'})['not_claimed']
    with pytest.raises(ClientError) as exc:
        worker._update(f.table, f.job['job_id'], 'old-worker', status='COMPLETED')
    assert exc.value.response['Error']['Code'] == 'ConditionalCheckFailedException'
    f.api._prepare_image_for_vision.assert_not_called()


def test_lost_dispatch_is_found_by_indexed_sweep_without_scanning(fixture):
    f = fixture
    f.invoke.side_effect = RuntimeError('fixture dispatch failure')
    worker.run(f.api, f.event)
    assert f.read()['checkpoint_key']
    f.invoke.side_effect = None
    assert worker.recover(f.api)['dispatched'] == 1
    drain(f)
    assert worker.recover(f.api)['dispatched'] == 0


def test_repeated_kills_have_finite_attempts(fixture):
    f = fixture
    class WorkerKilled(BaseException): pass
    f.api._prepare_image_for_vision.side_effect = WorkerKilled()
    for _ in range(3):
        with pytest.raises(WorkerKilled):
            worker.run(f.api, f.event)
        f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET lease_until = :zero',
                            ExpressionAttributeValues={':zero': 0})
    worker.run(f.api, f.event)
    assert worker._load(f.api, f.read())['image_cursor'] == 1
    assert worker._load(f.api, f.read())['errors'][0]['image_indexes'] == [0]


def test_deadline_bounds_sdk_retries_and_is_reset_after_run(fixture):
    client = Mock()
    assert worker.bounded_client(client) is client
    token = worker._deadline.set(time.monotonic() + 50)
    try:
        worker.bounded_client(client).chat.completions.create(model='fixture', messages=[])
        assert client.with_options.call_args.kwargs == {'timeout': 20, 'max_retries': 0}
    finally:
        worker._deadline.reset(token)
    worker.run(fixture.api, fixture.event)
    assert worker._deadline.get() is None


def test_shared_read_must_exist_before_success(fixture, monkeypatch):
    f = fixture
    monkeypatch.setenv('READ_SHARED_SAVED_RECIPES', 'true')
    cursor = Mock()
    cursor.fetchone.return_value = None
    f.api._mysql_conn.return_value = SimpleNamespace(cursor=lambda: SimpleContext(cursor))
    drain(f)
    assert f.read()['status'] == 'FAILED' and f.read()['result_count'] == 0
    assert len(f.saved) == 2  # Saved legacy rows are retained for reconciliation, never duplicated.


def test_checkpoint_write_failure_cannot_publish_success(fixture):
    f = fixture
    f.s3.put_object.side_effect = RuntimeError('fixture storage unavailable')
    with pytest.raises(RuntimeError):
        worker.run(f.api, f.event)
    assert f.read()['status'] == 'RUNNING'
    f.api._prepare_image_for_vision.assert_not_called()
    assert f.read()['lease_until'] > time.time()


def test_temporary_grouping_failure_keeps_pages_together_for_retry(fixture):
    f = fixture
    f.api._group_image_fragments.side_effect = [ServiceError('provider unavailable', 503),
        {'unused': 'never read'}]
    worker.run(f.api, f.event)
    worker.run(f.api, f.event)
    result = worker.run(f.api, f.event)
    assert result['retry_scheduled'] and f.api._save_saved_recipe_record.call_count == 0
    f.api._group_image_fragments.side_effect = lambda fragments, **kw: [{'fragments': fragments}]
    f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET retry_at = :zero',
                        ExpressionAttributeValues={':zero': 0})
    drain(f)
    assert f.read()['result_count'] == 1
    assert f.read()['results'][0]['image_indexes'] == [0, 1]


def test_source_expired_is_permanent_without_duplicate_fetches(fixture):
    f = fixture
    f.api._prepare_image_for_vision.side_effect = ServiceError('Source is no longer available.', 422)
    drain(f)
    assert f.read()['status'] == 'FAILED' and f.api._prepare_image_for_vision.call_count == 2
    assert len(f.read()['partial_errors']) == 2


def signed_event(owner='fixture-owner', expiry=None):
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b'=').decode()
    value = encode({'alg': 'HS256'}) + '.' + encode({'iss': 'trepo-auth', 'owner_id': owner,
        'exp': time.time() + 60 if expiry is None else expiry})
    sig = base64.urlsafe_b64encode(hmac.new(b'fixture-secret', value.encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
    return {'headers': {'Authorization': 'Bearer ' + value + '.' + sig}}


def test_lost_accept_response_reuses_same_job_without_resetting_progress(fixture):
    import trepo_auth
    f = fixture
    with patch.object(trepo_auth, '_SECRET', 'fixture-secret'):
        first = worker.enqueue(f.api, 'fixture-owner', {'images': ['a']}, signed_event())
        f.table.update_item(Key={'job_id': first['job_id']}, UpdateExpression='SET checkpoint_key = :key',
                            ExpressionAttributeValues={':key': 'already-progressed'})
        second = worker.enqueue(f.api, 'fixture-owner', {'images': ['a']}, signed_event())
    assert first['job_id'] == second['job_id'] and second['checkpoint_key'] == 'already-progressed'
    assert f.invoke.call_count == 1 and f.s3.put_object.call_count == 1
    assert first['payload_s3_key'].startswith('recipe-images/checkpoints-v2/')


@pytest.mark.parametrize('event', [{}, signed_event('different-owner'), signed_event(expiry=float('nan')),
    signed_event(expiry=float('inf')), signed_event(expiry=time.time() - 1),
    {'headers': {'authorization': 'Bearer unsigned.invalid.token'}}])
def test_unauthorized_acceptance_has_no_storage_or_queue_mutation(fixture, event):
    import trepo_auth
    f = fixture
    before = f.table.scan()['Count']
    with patch.object(trepo_auth, '_SECRET', 'fixture-secret'):
        with pytest.raises(ServiceError) as exc:
            worker.enqueue(f.api, 'fixture-owner', {'images': ['a']}, event)
    assert exc.value.status_code == 403 and f.table.scan()['Count'] == before
    f.api._put_saved_recipe_batch_payload.assert_not_called()
    f.s3.put_object.assert_not_called()


def test_enqueue_dispatch_failure_still_has_durable_recovery_marker(fixture):
    import trepo_auth
    f = fixture
    f.invoke.side_effect = RuntimeError('fixture delivery unavailable')
    with patch.object(trepo_auth, '_SECRET', 'fixture-secret'):
        accepted = worker.enqueue(f.api, 'fixture-owner', {'images': ['a']}, signed_event())
    assert accepted['status'] == 'PENDING'
    marker = f.table.get_item(Key={'job_id': worker._marker_id(accepted['job_id'])}, ConsistentRead=True)['Item']
    assert marker['target_job_id'] == accepted['job_id']


def test_abandoned_job_stops_after_recovery_window(fixture):
    f = fixture
    f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET recovery_deadline = :deadline',
                        ExpressionAttributeValues={':deadline': int(time.time()) - 1})
    assert worker.recover(f.api)['dispatched'] == 0
    assert f.read()['status'] == 'FAILED' and f.read()['ttl'] > time.time()


def test_recovery_expiry_preserves_committed_partial_results(fixture):
    f = fixture
    for _ in range(4): worker.run(f.api, f.event)
    first = f.read()['recipe_ids']
    assert len(first) == 1
    f.table.update_item(Key={'job_id': f.job['job_id']}, UpdateExpression='SET recovery_deadline = :deadline, image_count = :count',
                        ExpressionAttributeValues={':deadline': int(time.time()) - 1, ':count': 2})
    worker.recover(f.api)
    assert f.read()['status'] == 'COMPLETED' and f.read()['recipe_ids'] == first
    assert f.read()['partial_errors'][-1]['image_indexes'] == [1]


class SimpleContext:
    def __init__(self, value): self.value = value
    def __enter__(self): return self.value
    def __exit__(self, *args): pass
