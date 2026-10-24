import ast,importlib.util,json,os,sys,threading,uuid
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import pytest,pymysql
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sql_retry import transaction

@pytest.mark.parametrize('code',[1213,1205,2013])
def test_only_deadlock_retries_complete_transaction(code):
    conn=Mock();op=Mock(side_effect=[pymysql.err.OperationalError(code,'fixture'),'saved'])
    if code==1213:
        assert transaction(conn,op,pause=lambda _:None)=='saved'
        assert conn.begin.call_count==2 and conn.rollback.call_count==1 and conn.commit.call_count==1
    else:
        with pytest.raises(pymysql.err.OperationalError):transaction(conn,op,pause=lambda _:None)
        assert op.call_count==1

def test_retry_budget_and_original_error_preserved():
    conn=Mock();op=Mock(side_effect=pymysql.err.OperationalError(1213,'deadlock fixture'))
    with pytest.raises(pymysql.err.OperationalError):transaction(conn,op,pause=lambda _:None)
    assert op.call_count==3 and conn.rollback.call_count==3

@pytest.fixture
def db():
    if os.getenv('RELIABILITY_MYSQL_PORT')!='33387':pytest.skip('Local fixture server only')
    opts=dict(host='127.0.0.1',port=33387,user='root',autocommit=True,cursorclass=pymysql.cursors.DictCursor)
    conn=pymysql.connect(**opts);name='reliability_sql_'+uuid.uuid4().hex[:12]
    with conn.cursor() as c:
        c.execute('CREATE DATABASE '+name);c.execute('USE '+name)
        c.execute('CREATE TABLE counters (id INT PRIMARY KEY,value INT) ENGINE=InnoDB')
        c.execute('INSERT INTO counters VALUES (1,0),(2,0)')
        c.execute('CREATE TABLE shared_kitchen (_id VARCHAR(36) PRIMARY KEY,owner_id VARCHAR(36),product_name TEXT,product_expiration VARCHAR(20),_updatedDate DATETIME)')
        c.execute('CREATE TABLE kitchen_item_edits (item_id VARCHAR(36) PRIMARY KEY,overrides JSON)')
    try:yield conn,{**opts,'database':name}
    finally:
        conn.rollback()
        with conn.cursor() as c:c.execute('DROP DATABASE '+name)
        conn.close()

def test_real_innodb_deadlock_retries_without_duplicate_effects(db):
    conn,opts=db;barrier=threading.Barrier(2);counts=[0,0]
    def worker(index):
        cx=pymysql.connect(**opts)
        def operation():
            counts[index]+=1
            with cx.cursor() as c:
                c.execute('UPDATE counters SET value=value+1 WHERE id=%s',[index+1])
                if counts[index]==1:barrier.wait(timeout=5)
                c.execute('UPDATE counters SET value=value+1 WHERE id=%s',[2-index])
        try:transaction(cx,operation)
        finally:cx.close()
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,[0,1]))
    assert sorted(counts)==[1,2]
    with conn.cursor() as c:c.execute('SELECT value FROM counters');assert [r['value'] for r in c.fetchall()]==[2,2]

def test_insert_and_expiration_provenance_rollback_together_and_replay_preserves_customer_edit(db):
    conn,_=db;scope={'datetime':datetime,'json':json,'_report_backend_error':Mock()}
    fn=next(n for n in ast.parse((ROOT/'app.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_kitchen_insert')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'app.py'),'exec'),scope)
    insert=scope['_shared_kitchen_insert'];payload={'_id':'one','product_name':'Milk','product_expiration':'2026-11-01'}
    with conn.cursor() as c:c.execute("CREATE TRIGGER fail_provenance BEFORE INSERT ON kitchen_item_edits FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='fixture failure'")
    with pytest.raises(pymysql.err.OperationalError):insert(conn,'owner',payload)
    with conn.cursor() as c:
        c.execute('SELECT * FROM shared_kitchen');assert c.fetchall()==()
        c.execute('DROP TRIGGER fail_provenance')
    insert(conn,'owner',payload)
    with conn.cursor() as c:c.execute("UPDATE shared_kitchen SET product_name='User correction' WHERE _id='one'")
    insert(conn,'owner',payload)
    with conn.cursor() as c:
        c.execute('SELECT * FROM shared_kitchen');assert c.fetchone()['product_name']=='User correction'
        c.execute('SELECT overrides FROM kitchen_item_edits');assert json.loads(c.fetchone()['overrides'])['__expiration_provenance']['value']=='2026-11-01'
