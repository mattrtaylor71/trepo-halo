"""Real MySQL claims, retry leases and atomic publication; disposable local schema."""
import importlib.util,json,os,sys,uuid
from pathlib import Path
import pytest,pymysql
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generation_jobs',ROOT/'generation_jobs.py')
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)

@pytest.fixture
def db():
    port=os.getenv('RELIABILITY_MYSQL_PORT')
    if not port:pytest.skip('Set RELIABILITY_MYSQL_PORT to a disposable localhost MySQL')
    conn=pymysql.connect(host='127.0.0.1',port=int(port),user='root',autocommit=True,cursorclass=pymysql.cursors.DictCursor)
    name='reliability_plan_'+uuid.uuid4().hex[:12]
    with conn.cursor() as c:
        c.execute('CREATE DATABASE '+name);c.execute('USE '+name)
        for stmt in (ROOT/'generation_jobs.sql').read_text().split(';'):
            if stmt.strip():c.execute(stmt)
        c.execute('CREATE TABLE results (id INT PRIMARY KEY, value VARCHAR(100))')
    try:yield conn
    finally:
        conn.rollback()
        with conn.cursor() as c:c.execute('DROP DATABASE '+name)
        conn.close()

def row(db):
    with db.cursor() as c:c.execute('SELECT * FROM meal_plan_generation_jobs');return c.fetchone()

def pending(db,claim):
    db.begin();result=jobs.retry(db,claim,'provider_timeout',delay_seconds=0);db.commit();return result

def test_retries_are_bounded_and_new_revision_resets_budget(db):
    claim=jobs.begin(db,'home',{'owner':'a'});jobs.bind_inputs(db,claim,'a'*64)
    for attempt in (1,2):
        assert pending(db,claim)
        claim=jobs.begin(db,'home',{'owner':'a'},claim['token']);jobs.bind_inputs(db,claim,'a'*64)
    assert not pending(db,claim)
    assert row(db)['state']=='failed' and not jobs.due(db)
    claim=jobs.begin(db,'home',{'owner':'a'});jobs.bind_inputs(db,claim,'b'*64)
    assert row(db)['attempts']==1

def test_redelivery_and_late_worker_cannot_publish_or_overwrite_status(db):
    old=jobs.begin(db,'home',{'owner':'a'});pending(db,old)
    new=jobs.begin(db,'home',{'owner':'a'},old['token'])
    assert jobs.begin(db,'home',{'owner':'a'},old['token']) is None
    db.begin()
    with pytest.raises(jobs.Superseded):jobs.finish(db,old)
    db.rollback();assert row(db)['run_token']==new['token']

def test_result_and_queue_completion_are_atomic(db):
    claim=jobs.begin(db,'home',{'owner':'a'})
    db.begin()
    with db.cursor() as c:c.execute("INSERT INTO results VALUES (1,'plan')")
    jobs.finish(db,claim);db.rollback()
    assert row(db)['state']=='running'
    with db.cursor() as c:c.execute('SELECT * FROM results');assert not c.fetchall()
    db.begin()
    with db.cursor() as c:c.execute("INSERT INTO results VALUES (1,'plan')")
    jobs.finish(db,claim);db.commit();assert row(db)['state']=='ready'

def test_worker_death_lease_is_rescued_and_budget_exhaustion_is_explicit(db):
    claim=jobs.begin(db,'home',{'owner':'a'},lease_seconds=-1)
    jobs.bind_inputs(db,claim,'a'*64)
    assert len(jobs.due(db))==1
    for _ in range(2):
        claim=jobs.begin(db,'home',{'owner':'a'},claim['token'],lease_seconds=-1)
        jobs.bind_inputs(db,claim,'a'*64)
    claim=jobs.begin(db,'home',{'owner':'a'},claim['token'])
    with pytest.raises(jobs.RetryExhausted):jobs.bind_inputs(db,claim,'a'*64)
    assert not pending(db,claim) and row(db)['state']=='failed'

def test_new_input_during_retry_has_its_own_budget(db):
    claim=jobs.begin(db,'home',{'owner':'a'});jobs.bind_inputs(db,claim,'a'*64);pending(db,claim)
    claim=jobs.begin(db,'home',{'owner':'a'},claim['token']);jobs.bind_inputs(db,claim,'b'*64)
    assert row(db)['attempts']==1

def test_busy_household_keeps_latest_request_and_fences_original(db):
    old=jobs.begin(db,'home',{'owner':'a','focus':'old'})
    jobs.supersede_pending(db,'home',{'owner':'a','focus':'new'})
    assert json.loads(row(db)['payload'])['focus']=='new'
    db.begin()
    with pytest.raises(jobs.Superseded):jobs.finish(db,old)
    db.rollback()
    fresh=row(db);assert jobs.begin(db,'home',json.loads(fresh['payload']),fresh['run_token'])

@pytest.mark.parametrize('permanent',[True,False])
def test_consent_failure_cannot_retry_forever(db,permanent):
    claim=jobs.begin(db,'home',{'owner':'a'})
    for _ in range(3):jobs.admission_failed(db,'home',claim['token'],permanent=permanent)
    assert row(db)['state']=='blocked' and not jobs.due(db)

def test_admission_failure_for_old_token_cannot_block_new_work(db):
    old=jobs.begin(db,'home',{'owner':'a'});new=jobs.begin(db,'home',{'owner':'a'})
    jobs.admission_failed(db,'home',old['token'],True)
    assert row(db)['state']=='running' and row(db)['run_token']==new['token']


def test_platform_redelivery_does_not_reset_unchanged_revision_budget(db):
    for _ in range(3):
        claim=jobs.begin(db,'home',{'owner':'a'});jobs.bind_inputs(db,claim,'a'*64);pending(db,claim)
    claim=jobs.begin(db,'home',{'owner':'a'})
    with pytest.raises(jobs.RetryExhausted):jobs.bind_inputs(db,claim,'a'*64)
    assert not pending(db,claim)

def test_busy_requests_do_not_reset_retry_budget(db):
    claim=jobs.begin(db,'home',{'owner':'a'});jobs.bind_inputs(db,claim,'a'*64)
    for _ in range(2):
        jobs.supersede_pending(db,'home',{'owner':'a'})
        claim=jobs.begin(db,'home',{'owner':'a'},row(db)['run_token']);jobs.bind_inputs(db,claim,'a'*64)
    assert not pending(db,claim)


@pytest.mark.parametrize('first_operation', ['begin', 'supersede'])
def test_simultaneous_admissions_do_not_upgrade_shared_locks(db, first_operation):
    """Two duplicate INSERT IGNOREs hold S locks and deadlock on FOR UPDATE.

    Synchronize after that statement to reproduce the production race. An
    exclusive upsert serializes at admission and must not wait at this barrier.
    Both connections are real InnoDB transactions in the disposable schema.
    """
    import concurrent.futures
    import threading
    old = jobs.begin(db, 'home', {'owner': 'a', 'focus': 'old'})
    jobs.bind_inputs(db, old, 'a' * 64)
    with db.cursor() as cur:
        cur.execute('SELECT DATABASE() AS name')
        database = cur.fetchone()['name']
    start = threading.Barrier(2)
    after_shared_insert = threading.Barrier(2)

    class AdmissionCursor(pymysql.cursors.DictCursor):
        def execute(self, query, args=None):
            result = super().execute(query, args)
            if query.startswith('INSERT IGNORE INTO ' + jobs.TABLE):
                after_shared_insert.wait(timeout=5)
            return result

    def admit(index):
        conn = pymysql.connect(host='127.0.0.1', port=int(os.environ['RELIABILITY_MYSQL_PORT']),
            user='root', database=database, autocommit=True, cursorclass=AdmissionCursor,
            read_timeout=10, write_timeout=10)
        try:
            start.wait(timeout=5)
            if index == 0 and first_operation == 'begin':
                return jobs.begin(conn, 'home', {'owner': 'a', 'focus': 'worker'})
            jobs.supersede_pending(conn, 'home', {'owner': 'a', 'focus': 'new-' + str(index)})
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(admit, index) for index in range(2)]
        results = [future.result(timeout=15) for future in futures]
    current = row(db)
    assert current['input_hash'] == 'a' * 64
    assert current['attempts'] == (2 if first_operation == 'begin' else 1)
    assert current['run_token'] != old['token']
    assert current['state'] in ('pending', 'running')
    db.begin()
    with pytest.raises(jobs.Superseded):
        jobs.finish(db, old)
    db.rollback()
    if current['state'] == 'running':
        assert current['run_token'] == results[0]['token']
    else:
        assert json.loads(current['payload'])['focus'].startswith('new-')
