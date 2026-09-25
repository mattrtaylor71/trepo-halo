"""Synthetic regressions, loading the exact supplied candidate's application source."""
import json
import signal
import threading
import time
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import app
import recipe_work_budget as budget
from openai import OpenAI


@contextmanager
def invocation(seconds=120, sync=False):
    with budget.invocation(SimpleNamespace(get_remaining_time_in_millis=lambda: seconds * 1000), sync):
        yield


@contextmanager
def server(handler):
    instance = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    instance.daemon_threads = True
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%s' % instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=1)


@pytest.fixture(autouse=True)
def clean_state():
    with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'synthetic-fixture'}), patch.object(app, '_log_event'):
        app._mark_sync_request_start(False)
        yield
        app._mark_sync_request_start(False)


def test_async_rung_honors_native_deadline_and_warm_state_resets():
    with invocation(20):
        assert app._rung_fits('audio')[0] is False
        assert budget.remaining() > 0 and budget.terminal_reason() == 'deadline'
    assert budget.remaining() is None
    with invocation(120):
        assert app._rung_fits('audio')[0] is True


def test_actual_handler_establishes_and_resets_native_budget():
    seen = []
    def impl(*args):
        seen.append(budget.remaining())
        return {'ok': True}
    with patch.object(app, '_handler_impl', side_effect=impl):
        app.handler({'async_task': 'process_saved_recipe_url'}, SimpleNamespace(get_remaining_time_in_millis=lambda: 22000))
    assert seen[0] is not None and 6 < seen[0] <= 7
    assert budget.remaining() is None


def test_real_unknown_length_trickle_is_interrupted_and_closed(tmp_path):
    closed = threading.Event()
    class Slow(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            try:
                for _ in range(30):
                    self.wfile.write(b'x' * 1024); self.wfile.flush(); time.sleep(.03)
            except (BrokenPipeError, ConnectionResetError):
                closed.set()
        def log_message(self, *args): pass
    with server(Slow) as url, patch.object(budget, 'DOWNLOAD_SECONDS', .08), invocation():
        started = time.monotonic()
        with pytest.raises(budget.WorkLimit): app._download_media_to(tmp_path, url)
        assert time.monotonic() - started < .4
        assert closed.wait(.5)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize('known_length', [True, False])
def test_direct_oversize_is_rejected_before_upload(tmp_path, known_length):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {'Content-Length': '100'} if known_length else {}
    response.iter_content.return_value = iter([b'x' * 60, b'x' * 60])
    with patch.object(app.requests, 'get', return_value=response), patch.dict(app.os.environ, {'TRANSCRIBE_MEDIA_MAX_BYTES': '80'}), invocation():
        with pytest.raises(budget.WorkLimit): app._download_media_to(tmp_path, 'https://fixture.test/audio')
        assert budget.terminal_reason() == 'media_too_large'
    if known_length: response.iter_content.assert_not_called()


def fake_ytdlp_file(tmp_path, size=4):
    media = tmp_path / 'voice.m4a'
    with media.open('wb') as handle: handle.truncate(size)
    return {'requested_downloads': [{'filepath': str(media)}]}


def test_ytdlp_final_size_guard_prevents_whisper_even_if_provider_skips_hooks(tmp_path):
    info = fake_ytdlp_file(tmp_path, budget.MAX_MEDIA_BYTES + 1)
    client = Mock()
    with patch.object(app, '_extract_ytdlp_info', return_value=info), patch.object(app, 'OpenAI', return_value=client), invocation():
        with pytest.raises(RuntimeError): app._transcribe_audio_from_url('https://www.tiktok.com/video/fixture')
        assert budget.terminal_reason() == 'media_too_large'
    client.audio.transcriptions.create.assert_not_called()


def test_actual_ytdlp_options_enforce_unknown_size_progress_without_changing_format():
    options = []
    class Downloader:
        def __init__(self, opts): options.append(opts)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def extract_info(self, *args, **kwargs):
            for hook in options[-1].get('progress_hooks', []): hook({'downloaded_bytes': budget.MAX_MEDIA_BYTES + 1})
            return {}
    with patch.object(app.yt_dlp, 'YoutubeDL', Downloader), patch.object(app, '_resolve_ytdlp_cookiefile', return_value=None), invocation():
        with pytest.raises(budget.WorkLimit): app._extract_ytdlp_info('https://www.tiktok.com/video/fixture', download=True)
    assert len(options) == 1
    assert options[0]['format'] == 'bestaudio/best[url!*=media-video-hvc1]'


def test_slow_ytdlp_stops_without_retry_or_late_upload():
    calls = []
    def slow(*args, **kwargs): calls.append('download'); time.sleep(.5); return {}
    client = Mock()
    with patch.object(app, '_extract_ytdlp_info', side_effect=slow), patch.object(app, 'OpenAI', return_value=client), patch.object(budget, 'DOWNLOAD_SECONDS', .06), invocation():
        started = time.monotonic()
        with pytest.raises(RuntimeError): app._transcribe_audio_from_url('https://www.tiktok.com/video/fixture')
        assert time.monotonic() - started < .3
    assert calls == ['download']
    client.audio.transcriptions.create.assert_not_called()


def test_slow_transcription_is_wall_bounded_and_client_closed(tmp_path):
    client = Mock()
    client.audio.transcriptions.create.side_effect = lambda **kwargs: time.sleep(.5)
    with patch.object(app, '_extract_ytdlp_info', return_value=fake_ytdlp_file(tmp_path)), patch.object(app, 'OpenAI', return_value=client) as factory, patch.object(budget, 'TRANSCRIBE_SECONDS', .06), invocation():
        started = time.monotonic()
        with pytest.raises(RuntimeError): app._transcribe_audio_from_url('https://www.tiktok.com/video/fixture')
        assert time.monotonic() - started < .3
        assert factory.call_args.kwargs['max_retries'] == 0
        assert factory.call_args.kwargs['timeout'] <= .06
    client.close.assert_called_once()


def test_actual_audio_sdk_429_makes_one_attempt(tmp_path):
    calls = []
    class RateLimit(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length'])); calls.append(self.path)
            body = b'{"error":{"message":"synthetic limit","type":"rate_limit_error"}}'
            self.send_response(429); self.send_header('Content-Length', str(len(body))); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    with server(RateLimit) as url:
        def client(**kwargs): return OpenAI(base_url=url+'/v1', **kwargs)
        with patch.object(app, 'OpenAI', side_effect=client), patch.object(app, '_extract_ytdlp_info', return_value=fake_ytdlp_file(tmp_path)), invocation():
            with pytest.raises(RuntimeError): app._transcribe_audio_from_url('https://www.tiktok.com/video/fixture')
    assert len(calls) == 1


@pytest.mark.parametrize('text,language', [('Añade harina y mezcla hasta que esté suave.', 'spanish'), ('Stir gently and simmer until soft.', 'english')])
def test_valid_spoken_tiktok_without_digits_or_english_remains_accepted(tmp_path, text, language):
    client = Mock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text=text, language=language, segments=[])
    with patch.object(app, '_extract_ytdlp_info', return_value=fake_ytdlp_file(tmp_path)), patch.object(app, 'OpenAI', return_value=client), invocation():
        result = app._transcribe_audio_from_url('https://www.tiktok.com/video/fixture')
    assert result['audio_transcript'] == text


def test_apify_valid_speech_accepted_and_silence_filter_retained(tmp_path):
    client = Mock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text='Add 1 cup oats and simmer.', language='english', segments=[{'no_speech_prob': .01}])
    with patch.object(app, '_download_media_to', return_value=Path(fake_ytdlp_file(tmp_path)['requested_downloads'][0]['filepath'])), patch.object(app, 'OpenAI', return_value=client), invocation():
        assert app._transcribe_audio_from_url('https://www.instagram.com/reel/fixture', source_context={'media_url':'https://fixture.test/audio'})['audio_transcript']
        client.audio.transcriptions.create.return_value.segments = [{'no_speech_prob': 1.0}]
        with pytest.raises(RuntimeError): app._transcribe_audio_from_url('https://www.instagram.com/reel/fixture', source_context={'media_url':'https://fixture.test/audio'})


def test_source_timeout_retains_job_result_once_and_duplicate_does_not_extract():
    job = {'owner':'fixture-owner','job_id':'fixture-job','status':'PENDING'}
    conn = Mock()
    extracted = []
    def update(jobid, **fields): job.update(fields); return dict(job)
    def save(conn, owner, extraction, **kwargs):
        extracted.append(extraction)
        return {'recipe': {'id': 'same-retained-id', 'status': 'failed'}, 'deduped':False}, 201
    event={'owner':job['owner'],'job_id':job['job_id']}
    with ExitStack() as stack:
        for name, value in {'_get_saved_recipe_batch_job': lambda _: job, '_update_saved_recipe_batch_job': update, '_mysql_conn':lambda:conn, '_ensure_saved_recipes_table':Mock(), '_load_saved_recipe_batch_payload':lambda _: {'url':'https://fixture.test/recipe'}, '_explore_shortcircuit_extraction':lambda *a,**kw:None, '_extract_content':Mock(side_effect=budget.WorkLimit()), '_save_saved_recipe_record':save, '_report_backend_error':Mock()}.items(): stack.enter_context(patch.object(app,name,value))
        with invocation(15):
            result=app._handle_async_saved_recipe_url_task(event,'fixture')
        assert result['ok'] and job['status']=='COMPLETED' and job['recipe_ids']==['same-retained-id']
        app._handle_async_saved_recipe_url_task(event,'fixture')
    assert len(extracted)==1 and extracted[0]['terminal_work_reason']=='deadline'


@pytest.mark.parametrize('sync,expected_status,queued,limit_kind', [(False,'failed',False,'expired'),(True,'repairing',True,'expired'),(False,'failed',False,'rung_skip'),(True,'repairing',True,'rung_skip'),(False,'failed',False,'image_expired'),(True,'repairing',True,'image_expired')])
def test_persisted_timeout_link_has_terminal_readiness_and_sync_can_repair(sync, expected_status, queued, limit_kind):
    conn=MagicMock(); cursor=conn.cursor.return_value.__enter__.return_value
    values=[]
    def execute(sql, params):
        if sql.startswith('INSERT INTO'): values.append(params)
    cursor.execute.side_effect=execute
    extraction={'platform':'tiktok','url':'https://www.tiktok.com/video/fixture','resolved_url':'https://www.tiktok.com/video/fixture','title':'Source title','source':'fixture','caption':'','content':''}
    def row(conn,owner,rid):
        v=values[-1]
        return {'_id':rid,'_owner':owner,'title':v[6],'status':v[-1], 'ingredients':v[12], 'instructions':v[13], 'notes':v[14], 'source_url':v[3], 'resolved_url':v[4]}
    queue=Mock()
    with ExitStack() as stack:
        stubs={'_analyze_extraction':lambda *a,**kw:({'recipe':'Not enough recipe information.','model':None},{}),'_fetch_saved_recipe_by_hash':lambda *a:None,'_fetch_saved_recipe_by_id':row,'_ensure_recipe_personalization_tables':Mock(),'_build_kitchen_match_context':lambda *a:{},'_persist_owner_recipe_availability_rows':Mock(),'_apply_saved_recipe_meal_category':Mock(),'_dual_write_saved_recipe_to_shared':Mock(),'_fan_out_saved_recipe_to_household':Mock(),'_enqueue_saved_recipe_repair_or_finish':queue,'_serialize_saved_recipe_with_availability':lambda conn,owner,row,**kw:(app._serialize_row(row),{})}
        for name,value in stubs.items():stack.enter_context(patch.object(app,name,value))
        if limit_kind == 'image_expired':
            stack.enter_context(patch.object(app,'_prepare_saved_recipe_image_fields',side_effect=lambda *a,**kw:budget.stop('deadline')))
        with invocation(15 if limit_kind == 'expired' else 20,sync):
            if limit_kind == 'rung_skip': assert not app._rung_fits('audio')[0]
            result,status=app._save_saved_recipe_record(conn,'fixture-owner',extraction)
    assert values[-1][-1]==expected_status
    assert queue.called is queued
    recipe=result['recipe']
    if not sync:
        assert recipe['notes']==[]
        response=app._build_saved_recipe_batch_job_response('fixture-owner',{'status':'COMPLETED','job_id':'fixture-job','owner':'fixture-owner','result_count':1},recipes=[recipe])
        body=json.loads(response['body'])
        assert body['job']['status']=='failed' and body['content_outcome']=='link_retained' and len(body['retained_recipes'])==1


def test_limit_does_not_enter_transient_retry_and_timer_restores():
    operation=Mock(side_effect=budget.WorkLimit())
    prior=signal.getsignal(signal.SIGALRM)
    with pytest.raises(budget.WorkLimit): app._run_with_transient_retry(operation)
    assert operation.call_count==1
    with pytest.raises(budget.WorkLimit):
        with budget.phase(.02):
            try: time.sleep(.2)
            except Exception: pytest.fail('provider caught hard deadline')
    assert signal.getsignal(signal.SIGALRM)==prior and signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_native_deadline_bounds_initial_refinement_provider():
    with patch.object(app,'_refine_recipe_structured',side_effect=lambda *a,**kw:time.sleep(.4)), invocation(15.06):
        started=time.monotonic()
        with pytest.raises(budget.WorkLimit):app._recipe_response_from_content('1 cup oats. Simmer until soft.')
        assert time.monotonic()-started < .3


def test_finalization_uses_existing_deterministic_fallback_outside_sql_timer():
    conn=MagicMock(); cursor=conn.cursor.return_value.__enter__.return_value
    def sql(*args,**kwargs): assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    cursor.execute.side_effect=sql
    with patch.object(app.recipe_inventory_llm,'match_recipes_fast') as match, patch.object(app,'_classify_meal_category') as classify, invocation(15):
        available=app._compute_recipe_availability(['1 cup oats'], {'kitchen_candidates':[]})
        assert available['missing_count']==1 and available['can_make_exact'] is False
        app._apply_saved_recipe_meal_category(conn,'fixture-owner','fixture-id','Oat porridge',['1 cup oats'])
    match.assert_not_called(); classify.assert_not_called(); conn.commit.assert_called_once()


def test_nested_timer_keeps_earlier_deadline_and_restores_handler():
    original=signal.getsignal(signal.SIGALRM)
    started=time.monotonic()
    with pytest.raises(budget.WorkLimit):
        with budget.phase(.035):
            with budget.phase(.3):time.sleep(.5)
    assert time.monotonic()-started < .2
    assert signal.getsignal(signal.SIGALRM)==original and signal.getitimer(signal.ITIMER_REAL)[0]==0


from test_saved_recipe_edit_atomic import recipe, mirrored_recipe


def test_timed_out_url_worker_roundtrips_sql_and_dynamodb_as_one_terminal_retained_link(mirrored_recipe, monkeypatch):
    import os, uuid, boto3
    tested, conn = mirrored_recipe
    endpoint=os.environ.get('TREPO_RECIPE_DYNAMO_ENDPOINT')
    if not endpoint: pytest.skip('Explicit isolated DynamoDB endpoint required')
    assert endpoint=='http://127.0.0.1:38001'
    db=boto3.resource('dynamodb',endpoint_url=endpoint,region_name='us-east-1',aws_access_key_id='fixture',aws_secret_access_key='fixture')
    table=db.create_table(TableName='recipe-timeout-'+uuid.uuid4().hex,KeySchema=[{'AttributeName':'job_id','KeyType':'HASH'}],AttributeDefinitions=[{'AttributeName':'job_id','AttributeType':'S'}],BillingMode='PAY_PER_REQUEST')
    table.wait_until_exists()
    class Borrowed:
        def __getattr__(self,name):return getattr(conn,name)
        def close(self):pass
    job={'job_id':uuid.uuid4().hex,'owner':'acting','type':tested._SAVED_RECIPE_BATCH_JOB_TYPE,'status':'PENDING','submission_kind':'url','created_at':tested._utc_now_iso()}
    table.put_item(Item=job)
    extraction=Mock(side_effect=budget.WorkLimit())
    queue=Mock()
    try:
        monkeypatch.setattr(tested,'_jobs_table',lambda:table)
        monkeypatch.setattr(tested,'_mysql_conn',lambda:Borrowed())
        monkeypatch.setattr(tested,'_load_saved_recipe_batch_payload',lambda _: {'url':'https://fixture.invalid/new-recipe'})
        monkeypatch.setattr(tested,'_explore_shortcircuit_extraction',lambda *a,**kw:None)
        monkeypatch.setattr(tested,'_extract_content',extraction)
        monkeypatch.setattr(tested,'_enqueue_saved_recipe_repair_or_finish',queue)
        monkeypatch.setattr(tested,'_serialize_saved_recipe_with_availability',lambda conn,owner,row,**kw:(tested._serialize_row(row),{}))
        monkeypatch.setattr(tested,'_report_backend_error',Mock())
        with invocation(15): first=tested._handle_async_saved_recipe_url_task(job,'synthetic-timeout')
        assert first['ok']
        actual=table.get_item(Key={'job_id':job['job_id']},ConsistentRead=True)['Item']
        assert actual['status']=='COMPLETED' and actual['result_count']==1 and len(actual['recipe_ids'])==1
        rid=actual['recipe_ids'][0]
        for _ in range(2):
            body=json.loads(tested._get_saved_recipe_batch_job_status('acting',job['job_id'])['body'])
            assert body['job']['status']=='failed' and body['count']==0 and body['content_outcome']=='link_retained'
            assert body['retained_recipes'][0]['id']==rid
        replay=tested._handle_async_saved_recipe_url_task(job,'synthetic-replay')
        assert replay['already_completed'] and extraction.call_count==1
        queue.assert_not_called()
        with conn.cursor() as cursor:
            cursor.execute('SELECT _id,status,notes FROM acting_saved_recipes WHERE source_url=%s',['https://fixture.invalid/new-recipe'])
            rows=cursor.fetchall(); assert len(rows)==1 and rows[0]['_id']==rid and rows[0]['status']=='failed' and json.loads(rows[0]['notes'])==[]
            cursor.execute('SELECT _id,status FROM shared_saved_recipes WHERE owner_id=%s AND _id=%s',['acting',rid])
            assert cursor.fetchone()['status']=='failed'
            cursor.execute('SELECT title FROM member_saved_recipes WHERE _id=%s',['recipe']); assert cursor.fetchone()['title']=='Before'
    finally:table.delete()


@pytest.mark.parametrize('url', ['ftp://fixture.invalid/recipe','This is pasted recipe text','https://www.instagram.com/profile-only/'])
def test_expired_worker_does_not_retain_invalid_source(url):
    job={'owner':'fixture-owner','job_id':'fixture-job','status':'PENDING'}
    def update(jobid,**fields):job.update(fields);return dict(job)
    save=Mock()
    with ExitStack() as stack:
        for name,value in {'_get_saved_recipe_batch_job':lambda _:job,'_update_saved_recipe_batch_job':update,'_mysql_conn':Mock(),'_ensure_saved_recipes_table':Mock(),'_load_saved_recipe_batch_payload':lambda _: {'url':url},'_explore_shortcircuit_extraction':lambda *a,**kw:None,'_extract_content':Mock(side_effect=budget.WorkLimit()),'_save_saved_recipe_record':save,'_report_backend_error':Mock()}.items():stack.enter_context(patch.object(app,name,value))
        with invocation(15):result=app._handle_async_saved_recipe_url_task(job,'synthetic-invalid')
    assert result['ok'] is False and job['status']=='FAILED' and 400<=job['partial_errors'][0]['status_code']<500
    save.assert_not_called()
