"""Actual create/insert functions against isolated MySQL; external effects are stubbed."""
import ast
import json
import concurrent.futures
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import threading
import uuid
import pymysql
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def fixture():
    port=int(os.environ['TREPO_SHELF_TEST_PORT'])
    schema='trepo_create_race_'+uuid.uuid4().hex
    root=pymysql.connect(host='127.0.0.1',port=port,user='root',password='',autocommit=True)
    with root.cursor() as cur:
        cur.execute('CREATE DATABASE `'+schema+'`')
        cur.execute('CREATE TABLE `'+schema+'`.shared_kitchen (_id CHAR(36) PRIMARY KEY,owner_id CHAR(36),job_id VARCHAR(100),product_name VARCHAR(255),_updatedDate DATETIME, INDEX(owner_id,job_id))')
    connections=[]
    def connect():
        c=pymysql.connect(host='127.0.0.1',port=port,user='root',password='',database=schema,autocommit=True,cursorclass=pymysql.cursors.DictCursor)
        connections.append(c);return c
    try:yield connect
    finally:
        for conn in connections:conn.close()
        with root.cursor() as cur:cur.execute('DROP DATABASE `'+schema+'`')
        root.close()

def load_create(connect,barrier=None):
    source=(ROOT/'kitchen_api/app.py').read_text();tree=ast.parse(source)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_create_kitchen_item','_shared_kitchen_insert']]
    feed=[]
    class Cursor:
        def __init__(self,cursor):self.cursor=cursor;self.duplicate_read=False
        def __enter__(self):return self
        def __exit__(self,*args):self.cursor.close()
        def execute(self,sql,params=None):
            self.duplicate_read='`job_id` = %s' in sql
            return self.cursor.execute(sql,params)
        @property
        def rowcount(self):return self.cursor.rowcount
        def fetchone(self):
            row=self.cursor.fetchone()
            if row is None and self.duplicate_read and barrier:
                # Inject latency after the empty read. A serialized implementation
                # may wait for this timeout; it must still produce exactly one row.
                try:barrier.wait(timeout=.6)
                except threading.BrokenBarrierError:pass
            return row
    class Connection:
        def __init__(self):self.raw=connect()
        def cursor(self):return Cursor(self.raw.cursor())
        def begin(self):return self.raw.begin()
        def rollback(self):return self.raw.rollback()
        def commit(self):return self.raw.commit()
        def close(self):return self.raw.close()
    ns={'json':json,'datetime':datetime,'_mysql_conn':Connection,'_build_create_payload':lambda owner,body:({'_id':str(uuid.uuid4()),'job_id':body['job_id'],'product_name':'Fixture beef','_updatedDate':datetime.now(timezone.utc).replace(tzinfo=None)},None),
        '_get_household_member_ids':lambda conn,owner:[owner], '_ensure_prod_kitchen_table':lambda *args:None,
        '_resolve_kitchen_table':lambda owner,*args:('shared_kitchen','owner_id = %s',[owner]),
        '_serialize_rows':lambda rows:rows,'_mark_recipe_refresh_needed_for_owners':lambda *args:None,
        '_record_master_feed_event':lambda *args:feed.append(args[-1]),'_report_backend_error':lambda *args,**kwargs:None,
        '_success_response':lambda body,status_code=200:{'statusCode':status_code,'body':body},
        '_error_response':lambda status,message:{'statusCode':status,'body':{'error':message}}}
    helper=ROOT/'kitchen_api/create_idempotency.py'
    if helper.exists():
        spec=importlib.util.spec_from_file_location('create_lock_test',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ns['_kitchen_create_lock']=module.kitchen_create_lock;ns['KitchenCreateBusy']=module.KitchenCreateBusy
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual-kitchen-create','exec'),ns)
    return ns['_create_kitchen_item'],feed

def rows(connect):
    with connect().cursor() as cur:cur.execute('SELECT * FROM shared_kitchen');return cur.fetchall()

def test_simultaneous_legacy_create_retries_insert_once(fixture):
    create,feed=load_create(fixture,threading.Barrier(2));owner=str(uuid.uuid4());body={'job_id':str(uuid.uuid4())+':bulk:1','defer_side_effects':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(lambda _:create(owner,body),range(2)))
    assert len(rows(fixture))==1,'Concurrent retry must not create a duplicate Kitchen item'
    assert sorted(r['statusCode'] for r in responses)==[200,201]
    assert responses[0]['body']['item']['_id']==responses[1]['body']['item']['_id']
    assert len(feed)==1

def test_lock_contention_is_retryable_and_does_not_block_other_jobs_or_owners(fixture):
    create,feed=load_create(fixture)
    helper=create.__globals__['_kitchen_create_lock'];module_globals=helper.__wrapped__.__globals__
    name=module_globals['create_lock_name'];owner=str(uuid.uuid4());job=str(uuid.uuid4());holder=fixture()
    create.__globals__['_kitchen_create_lock']=lambda conn,o,j:helper(conn,o,j,wait_seconds=0)
    with holder.cursor() as cur:cur.execute('SELECT GET_LOCK(%s,0)',(name(owner,job),))
    body={'job_id':job,'defer_side_effects':True}
    try:
        assert create(owner,body)['statusCode']==503
        assert len(rows(fixture))==0
        assert create(str(uuid.uuid4()),body)['statusCode']==201
        assert create(owner,{'job_id':str(uuid.uuid4()),'defer_side_effects':True})['statusCode']==201
    finally:
        with holder.cursor() as cur:cur.execute('SELECT RELEASE_LOCK(%s)',(name(owner,job),))
    assert create(owner,body)['statusCode']==201
    assert create(owner,body)['statusCode']==200
    assert len(rows(fixture))==3


def test_failed_insert_releases_lock_for_next_legacy_retry(fixture):
    create,feed=load_create(fixture);original=create.__globals__['_shared_kitchen_insert'];helper=create.__globals__['_kitchen_create_lock']
    create.__globals__['_kitchen_create_lock']=lambda conn,o,j:helper(conn,o,j,wait_seconds=0)
    def fail(*args):raise RuntimeError('Synthetic insert failure')
    create.__globals__['_shared_kitchen_insert']=fail
    owner=str(uuid.uuid4());body={'job_id':str(uuid.uuid4()),'defer_side_effects':True}
    assert create(owner,body)['statusCode']==500
    create.__globals__['_shared_kitchen_insert']=original
    assert create(owner,body)['statusCode']==201
    assert len(rows(fixture))==1


def test_thirty_concurrent_retries_preserve_one_item_and_one_creation(fixture):
    create,feed=load_create(fixture);owner=str(uuid.uuid4());job=str(uuid.uuid4())+':bulk:1'
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        answers=list(pool.map(lambda _:create(owner,{'job_id':job,'defer_side_effects':True}),range(30)))
    assert len(rows(fixture))==1
    assert sum(x['statusCode']==201 for x in answers)==1
    assert sum(x['statusCode']==200 for x in answers)==29
    assert len({x['body']['item']['_id'] for x in answers})==1
    assert len(feed)==1
