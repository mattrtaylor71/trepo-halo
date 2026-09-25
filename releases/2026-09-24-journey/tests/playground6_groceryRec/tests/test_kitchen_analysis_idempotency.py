import importlib.util
import os
from pathlib import Path
import uuid

import pytest
import pymysql

ROOT = Path(__file__).resolve().parents[1]
PORT = os.getenv('TREPO_TEXT_JOB_TEST_PORT', '33317')


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'kitchen_analysis_generator'))
    spec = importlib.util.spec_from_file_location('analysis_idempotency_test', ROOT / 'kitchen_analysis_generator/app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_busy_durable_operation_is_retryable_but_legacy_contract_is_preserved(app, monkeypatch):
    monkeypatch.setattr(app, '_acquire_named_lock', lambda *args: None)
    assert app.handler({'owner': 'fixture'}, None) == {'ok': True, 'skipped': True}
    result = app.handler({'owner': 'fixture', 'effect_id': str(uuid.uuid4())}, None)
    assert result == {'ok': False, 'retryable': True, 'code': 'owner_busy'}


@pytest.mark.parametrize('value', ['', 'not-a-uuid', {}, 1])
def test_invalid_effect_never_touches_database(app, monkeypatch, value):
    def forbidden(*args):
        pytest.fail('Invalid effect attempted a database lock')
    monkeypatch.setattr(app, '_acquire_named_lock', forbidden)
    assert app.handler({'owner': 'fixture', 'effect_id': value}, None)['code'] == 'invalid_effect_id'


@pytest.fixture
def database(app, monkeypatch):
    if not PORT:
        pytest.skip('Requires isolated MySQL fixture')
    conn = pymysql.connect(host='127.0.0.1', port=int(PORT), user='root', password='',
                           autocommit=True,
                           cursorclass=pymysql.cursors.DictCursor)
    monkeypatch.setattr(app, '_calculate_metrics', lambda *args: {'UPF': 0, 'harmful_ingredients': 0})
    monkeypatch.setattr(app, 'DUAL_WRITE_METRICS', False)
    owners = [str(uuid.uuid4()), str(uuid.uuid4())]
    schema='analysis_test_'+uuid.uuid4().hex
    with conn.cursor() as cur:
        cur.execute('CREATE DATABASE `'+schema+'`')
        cur.execute('USE `'+schema+'`')
        cur.execute('CREATE TABLE new_users(user_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64))')
        cur.executemany('INSERT INTO new_users VALUES(%s,%s)', [(owner, owners[0]) for owner in owners])
    try:
        yield conn, owners
    finally:
        with conn.cursor() as cur:cur.execute('DROP DATABASE `'+schema+'`')
        conn.close()


def snapshots(app, conn, owner):
    with conn.cursor() as cur:
        cur.execute('SELECT `_id`,Points FROM `' + app._metrics_table_name(owner) + '`')
        return cur.fetchall()


def test_points_and_snapshot_not_duplicated_on_redelivery(app, database):
    conn, owners = database
    effect = str(uuid.uuid4())
    first = app._insert_processing_snapshot(conn, owners[0], owners, 7, effect)
    second = app._insert_processing_snapshot(conn, owners[0], owners, 7, effect)
    assert first == second
    for owner in owners:
        assert snapshots(app, conn, owner) == [{'_id': first[0], 'Points': 7}]


def test_partial_household_commit_replays_only_missing_member(app, database, monkeypatch):
    conn, owners = database
    effect = str(uuid.uuid4())
    ensure = app._ensure_metrics_table
    def interrupted(c, table):
        if table == app._metrics_table_name(owners[1]):
            raise RuntimeError('Synthetic worker termination after first member commit')
        return ensure(c, table)
    monkeypatch.setattr(app, '_ensure_metrics_table', interrupted)
    with pytest.raises(RuntimeError):
        app._insert_processing_snapshot(conn, owners[0], owners, 5, effect)
    assert snapshots(app, conn, owners[0])[0]['Points'] == 5
    monkeypatch.setattr(app, '_ensure_metrics_table', ensure)
    app._insert_processing_snapshot(conn, owners[0], owners, 5, effect)
    for owner in owners:
        rows = snapshots(app, conn, owner)
        assert len(rows) == 1 and rows[0]['Points'] == 5


def test_later_points_are_preserved_when_old_effect_replays(app, database):
    conn, owners = database
    owner = owners[0]
    first = str(uuid.uuid4())
    app._insert_processing_snapshot(conn, owner, [owner], 4, first)
    second_id, _ = app._insert_processing_snapshot(conn, owner, [owner], 3, str(uuid.uuid4()))
    app._insert_processing_snapshot(conn, owner, [owner], 4, first)
    rows = snapshots(app, conn, owner)
    assert len(rows) == 2
    assert next(r['Points'] for r in rows if r['_id'] == second_id) == 7


def test_same_effect_is_scoped_to_owner(app, database):
    conn, owners = database
    effect = str(uuid.uuid4())
    ids = [app._insert_processing_snapshot(conn, owner, [owner], 2, effect)[0] for owner in owners]
    assert ids[0] != ids[1]


def test_effect_replay_rejects_foreign_snapshot_without_crediting_other_members(app, database, monkeypatch):
    conn, owners = database
    monkeypatch.setenv('METRICS_STATE_V1', 'false')
    effect = str(uuid.uuid4())
    ident, rows = app._insert_processing_snapshot(conn, owners[0], [owners[0]], 7, effect)
    with conn.cursor() as cur:
        cur.execute('UPDATE `' + rows[0][1] + '` SET _owner=%s WHERE _id=%s', [owners[1], ident])
        cur.execute('SELECT * FROM `' + rows[0][1] + '` WHERE _id=%s', [ident])
        original = cur.fetchone()
    with pytest.raises(RuntimeError, match='snapshot owner mismatch'):
        app._insert_processing_snapshot(conn, owners[0], owners, 7, effect)
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM `' + rows[0][1] + '` WHERE _id=%s', [ident])
        assert cur.fetchone() == original
        cur.execute('SHOW TABLES LIKE %s', [app._metrics_table_name(owners[1])])
        assert cur.fetchone() is None


def test_legacy_operations_still_create_distinct_snapshots(app, database):
    conn, owners = database
    one = app._insert_processing_snapshot(conn, owners[0], owners, 2)
    two = app._insert_processing_snapshot(conn, owners[0], owners, 2)
    assert one[0] != two[0]
    for owner in owners:
        assert len(snapshots(app, conn, owner)) == 2


def test_legacy_mode_completed_effect_skips_model_and_keeps_ready(app,database,monkeypatch):
    conn,owners=database;calls=[]
    monkeypatch.setenv('METRICS_STATE_V1','false')
    monkeypatch.setattr(app,'_mysql_conn',lambda:conn)
    monkeypatch.setattr(app,'_get_household_member_ids',lambda *_:owners)
    monkeypatch.setattr(app,'_acquire_named_lock',lambda *_:object())
    monkeypatch.setattr(app,'_release_named_lock',lambda *_:None)
    monkeypatch.setattr(app,'_get_kitchen_rows',lambda *_:[])
    def model(*_):
        calls.append(True)
        if len(calls)>1:raise RuntimeError('Replay must not invoke model')
        return 'Completed fixture analysis'
    monkeypatch.setattr(app,'_generate_analysis_with_gpt',model)
    event={'owner':owners[0],'effect_id':str(uuid.uuid4()),'points_delta':7}
    first=app.handler(event,None);assert first['ok']
    second=app.handler(event,None);assert second['ok'] and second['entry_id']==first['entry_id']
    assert len(calls)==1
    with conn.cursor() as cur:
        for owner in owners:
            cur.execute('SELECT Points,kitchen_analysis_status,kitchen_analysis_content FROM `'+app._metrics_table_name(owner)+'`')
            assert cur.fetchall()==[{'Points':7,'kitchen_analysis_status':'ready','kitchen_analysis_content':'Completed fixture analysis'}]


def test_legacy_mode_failed_replay_preserves_completed_member(app,database,monkeypatch):
    conn,owners=database
    monkeypatch.setenv('METRICS_STATE_V1','false')
    ident,rows=app._insert_processing_snapshot(conn,owners[0],owners,7,str(uuid.uuid4()))
    app._update_snapshot_status(conn,[rows[0]],ident,'ready',content='Already complete')
    app._update_snapshot_status(conn,rows,ident,'failed',error_message='Other member retry failed')
    with conn.cursor() as cur:
        cur.execute('SELECT kitchen_analysis_status,kitchen_analysis_content,kitchen_analysis_error FROM `'+rows[0][1]+'` WHERE _id=%s',[ident])
        assert cur.fetchone()=={'kitchen_analysis_status':'ready','kitchen_analysis_content':'Already complete','kitchen_analysis_error':None}
        cur.execute('SELECT kitchen_analysis_status FROM `'+rows[1][1]+'` WHERE _id=%s',[ident])
        assert cur.fetchone()['kitchen_analysis_status']=='failed'


def test_legacy_completed_result_is_immutable_on_successful_replay(app,database,monkeypatch):
    conn,owners=database;monkeypatch.setenv('METRICS_STATE_V1','false')
    ident,rows=app._insert_processing_snapshot(conn,owners[0],[owners[0]],7,str(uuid.uuid4()))
    app._update_snapshot_status(conn,rows,ident,'ready',content='Original completed analysis')
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM `'+rows[0][1]+'` WHERE _id=%s',[ident]);before=cur.fetchone()
    app._update_snapshot_status(conn,rows,ident,'ready',content='Different retry result')
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM `'+rows[0][1]+'` WHERE _id=%s',[ident]);assert cur.fetchone()==before


def test_legacy_analysis_status_update_does_not_touch_foreign_owner(app,database,monkeypatch):
    conn,owners=database;monkeypatch.setenv('METRICS_STATE_V1','false')
    ident,rows=app._insert_processing_snapshot(conn,owners[0],[owners[0]],7,str(uuid.uuid4()))
    with conn.cursor() as cur:
        cur.execute('UPDATE `'+rows[0][1]+'` SET _owner=%s WHERE _id=%s',[owners[1],ident])
        cur.execute('SELECT * FROM `'+rows[0][1]+'` WHERE _id=%s',[ident]);before=cur.fetchone()
    app._update_snapshot_status(conn,rows,ident,'ready',content='Must not overwrite another owner')
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM `'+rows[0][1]+'` WHERE _id=%s',[ident]);assert cur.fetchone()==before


def enable_shared(app,conn,table,monkeypatch):
    with conn.cursor() as cur:
        cur.execute('CREATE TABLE shared_metrics LIKE `'+table+'`')
        cur.execute('ALTER TABLE shared_metrics DROP PRIMARY KEY, ADD owner_id VARCHAR(64) NOT NULL, ADD PRIMARY KEY(owner_id,_id)')
    monkeypatch.setattr(app,'DUAL_WRITE_METRICS',True)


def test_legacy_mirror_does_not_copy_foreign_owner(app,database,monkeypatch):
    conn,owners=database;monkeypatch.setenv('METRICS_STATE_V1','false')
    ident,rows=app._insert_processing_snapshot(conn,owners[0],[owners[0]],7,str(uuid.uuid4()))
    enable_shared(app,conn,rows[0][1],monkeypatch)
    with conn.cursor() as cur:cur.execute('UPDATE `'+rows[0][1]+'` SET _owner=%s WHERE _id=%s',[owners[1],ident])
    app._update_snapshot_status(conn,rows,ident,'ready',content='Must not publish foreign row')
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) n FROM shared_metrics');assert cur.fetchone()['n']==0


def test_completed_replay_repairs_shared_copy_without_changing_result(app,database,monkeypatch):
    conn,owners=database;owner=owners[0];effect=str(uuid.uuid4());monkeypatch.setenv('METRICS_STATE_V1','false')
    ident,rows=app._insert_processing_snapshot(conn,owner,[owner],7,effect)
    enable_shared(app,conn,rows[0][1],monkeypatch)
    app._update_snapshot_status(conn,rows,ident,'ready',content='Completed result')
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM shared_metrics WHERE owner_id=%s',[owner]);original=cur.fetchone();assert original
        cur.execute('DELETE FROM shared_metrics WHERE owner_id=%s',[owner])
    assert app._insert_processing_snapshot(conn,owner,[owner],7,effect)==(ident,rows)
    assert app._analysis_already_ready(conn,rows,ident)
    app._update_snapshot_status(conn,rows,ident,'failed',error_message='Retry failed')
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM shared_metrics WHERE owner_id=%s',[owner]);assert cur.fetchone()==original


def test_legacy_analysis_uses_only_owned_balance_and_content(app,database,monkeypatch):
    conn,owners=database;owner=owners[0];table=app._metrics_table_name(owner);monkeypatch.setenv('METRICS_STATE_V1','false')
    app._ensure_metrics_table(conn,table);app._ensure_metrics_columns(conn,table)
    with conn.cursor() as cur:
        cur.execute('INSERT INTO `'+table+'` (_id,_owner,Points,_createdDate,kitchen_analysis_content) VALUES (%s,%s,50,%s,%s),(%s,%s,500,%s,%s)',['owned',owner,'2000-01-01','Owned analysis','foreign',owners[1],'2001-01-01','Foreign analysis'])
    ident,_=app._insert_processing_snapshot(conn,owner,[owner],5,str(uuid.uuid4()))
    with conn.cursor() as cur:
        cur.execute('SELECT Points,kitchen_analysis_content FROM `'+table+'` WHERE _id=%s',[ident])
        assert cur.fetchone()=={'Points':55,'kitchen_analysis_content':'Owned analysis'}


def test_async_busy_effect_raises_for_lambda_retry_without_changing_sync_contract(app, monkeypatch):
    monkeypatch.setattr(app, '_acquire_named_lock', lambda *args: None)
    event={'owner':'fixture','effect_id':str(uuid.uuid4()),'async_delivery':True}
    with pytest.raises(RuntimeError, match='retry asynchronous delivery'):
        app.handler(event,None)
    event.pop('async_delivery')
    assert app.handler(event,None)=={'ok':False,'retryable':True,'code':'owner_busy'}
