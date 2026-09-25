"""Real SQL failures must never acknowledge an unapplied saved-recipe edit."""
import json,os,uuid
from unittest.mock import Mock
import pymysql,pytest
from test_saved_recipes_dualwrite import load_app_module

@pytest.fixture
def recipe(monkeypatch):
    app=load_app_module()
    port=int(os.environ.get('TREPO_AMOUNT_TEST_PORT','33317'))
    assert port==33317,'Only the isolated audit database is allowed'
    schema='recipe_edit_'+uuid.uuid4().hex
    root=pymysql.connect(host='127.0.0.1',port=port,user='root',autocommit=True)
    with root.cursor() as cur:cur.execute('CREATE DATABASE '+schema)
    conn=pymysql.connect(host='127.0.0.1',port=port,user='root',database=schema,autocommit=True,cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cur:
            for owner in ['acting','member','unrelated']:
                cur.execute('CREATE TABLE `'+owner+'_saved_recipes` (_id VARCHAR(36) PRIMARY KEY,_owner VARCHAR(36),title VARCHAR(255),ingredients JSON,instructions JSON,notes JSON,_updatedDate DATETIME) ENGINE=InnoDB')
                cur.execute('INSERT INTO `'+owner+'_saved_recipes` VALUES (%s,%s,%s,%s,%s,%s,NOW())',['recipe',owner,'Before','["beef"]','["cook"]','[]'])
        monkeypatch.setattr(app,'_mysql_conn',lambda:conn)
        monkeypatch.setattr(app,'_ensure_saved_recipes_table',lambda *args:None)
        monkeypatch.setattr(app,'_drift_safe_select',lambda *args:'*')
        monkeypatch.setattr(app,'_get_household_member_ids',lambda *args:['member','acting'])
        monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',False)
        monkeypatch.setattr(app,'_ensure_recipe_personalization_tables',lambda *args:None)
        monkeypatch.setattr(app,'_build_kitchen_match_context',lambda *args:{})
        monkeypatch.setattr(app,'_serialize_saved_recipe_with_availability',lambda conn,owner,row,**kwargs:(dict(row),{}))
        monkeypatch.setattr(app,'_persist_response_availability',Mock(),raising=False)
        monkeypatch.setattr(app,'_persist_owner_recipe_availability_rows',Mock())
        app._audit_real_dual_write = app._dual_write_saved_recipe_to_shared
        monkeypatch.setattr(app,'_dual_write_saved_recipe_to_shared',Mock())
        yield app,conn
    finally:
        conn.close()
        with root.cursor() as cur:cur.execute('DROP DATABASE '+schema)
        root.close()

def title(conn,owner):
    with conn.cursor() as cur:
        cur.execute('SELECT title FROM `'+owner+'_saved_recipes` WHERE _id=%s',['recipe']);row=cur.fetchone();return row['title'] if row else None

def reject(conn,owner):
    with conn.cursor() as cur:cur.execute("CREATE TRIGGER reject_edit BEFORE UPDATE ON `"+owner+"_saved_recipes` FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='fixture rejected saved-recipe edit'")

def test_primary_failure_is_not_reported_as_success(recipe):
    app,conn=recipe;reject(conn,'acting')
    with pytest.raises(pymysql.err.OperationalError,match='fixture rejected'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')=='Before'
    app._dual_write_saved_recipe_to_shared.assert_not_called()
    app._persist_response_availability.assert_not_called()

def test_member_failure_rolls_back_every_required_edit(recipe,monkeypatch):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True);reject(conn,'member')
    with pytest.raises(pymysql.err.OperationalError,match='fixture rejected'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert [title(conn,owner) for owner in ['acting','member','unrelated']]==['Before']*3
    app._dual_write_saved_recipe_to_shared.assert_not_called()

@pytest.mark.parametrize('sharing',[False,True])
def test_legacy_success_contract_and_scope(recipe,monkeypatch,sharing):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',sharing)
    result=app._update_saved_recipe('acting','recipe',{'title':'After','steps':['New step']})
    assert result['statusCode']==200 and json.loads(result['body'])['recipe']['title']=='After'
    assert [title(conn,owner) for owner in ['acting','member','unrelated']]==['After','After' if sharing else 'Before','Before']
    with conn.cursor() as cur:
        cur.execute('SELECT instructions FROM acting_saved_recipes WHERE _id=%s',['recipe']);assert json.loads(cur.fetchone()['instructions'])==['New step']

def test_failure_before_commit_rolls_back(recipe,monkeypatch):
    app,conn=recipe
    class FailCommit:
        def __getattr__(self,name):return getattr(conn,name)
        def commit(self):raise RuntimeError('fixture failed before commit')
    monkeypatch.setattr(app,'_mysql_conn',FailCommit)
    with pytest.raises(RuntimeError,match='before commit'):app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')=='Before'

def test_household_lookup_failure_does_not_silently_skip_required_copies(recipe,monkeypatch):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    def fail(*args):raise RuntimeError('fixture membership lookup failure')
    monkeypatch.setattr(app,'_get_household_member_ids',fail)
    with pytest.raises(RuntimeError,match='membership lookup'):app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert [title(conn,owner) for owner in ['acting','member','unrelated']]==['Before']*3

def test_schema_preparation_failure_leaves_all_recipes_unchanged(recipe,monkeypatch):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    def prepare(client,owner):
        assert not client.server_status & 1,'DDL preparation entered the edit transaction'
        if owner=='member':raise RuntimeError('fixture preparation failure')
    monkeypatch.setattr(app,'_ensure_saved_recipes_table',prepare)
    with pytest.raises(RuntimeError,match='preparation failure'):app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert [title(conn,owner) for owner in ['acting','member','unrelated']]==['Before']*3

def test_acting_copy_deleted_between_preflight_and_transaction_cannot_edit_household(recipe,monkeypatch):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    calls=0
    def prepare(client,owner):
        nonlocal calls
        calls+=1
        if calls==2:
            with client.cursor() as cur:cur.execute('DELETE FROM acting_saved_recipes WHERE _id=%s',['recipe'])
    monkeypatch.setattr(app,'_ensure_saved_recipes_table',prepare)
    result=app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert result['statusCode']==404 and title(conn,'acting') is None
    assert title(conn,'member')==title(conn,'unrelated')=='Before'
    app._dual_write_saved_recipe_to_shared.assert_not_called()

def test_lost_commit_ack_preserves_both_copies_and_allows_legacy_retry(recipe,monkeypatch):
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    class LostAck:
        def __getattr__(self,name):return getattr(conn,name)
        def commit(self):
            conn.commit()
            raise pymysql.err.OperationalError(2013,'fixture committed acknowledgement lost')
    monkeypatch.setattr(app,'_mysql_conn',LostAck)
    with pytest.raises(pymysql.err.OperationalError,match='acknowledgement lost'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')==title(conn,'member')=='After'
    monkeypatch.setattr(app,'_mysql_conn',lambda:conn)
    assert app._update_saved_recipe('acting','recipe',{'title':'After'})['statusCode']==200
    assert title(conn,'unrelated')=='Before'

def test_concurrent_household_editors_keep_required_copies_equal(recipe,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    app,conn=recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    local=threading.local();start=threading.Barrier(8)
    monkeypatch.setattr(app,'_mysql_conn',lambda:local.conn)
    schema=conn.db.decode() if isinstance(conn.db,bytes) else conn.db
    def edit(i):
        local.conn=pymysql.connect(host='127.0.0.1',port=33317,user='root',database=schema,autocommit=True,cursorclass=pymysql.cursors.DictCursor,connect_timeout=5,read_timeout=15,write_timeout=15)
        try:
            start.wait(timeout=5)
            return app._update_saved_recipe('acting' if i%2 else 'member','recipe',{'title':'Edit '+str(i)})['statusCode']
        finally:local.conn.close()
    with ThreadPoolExecutor(max_workers=8) as pool:assert list(pool.map(edit,range(8)))==[200]*8
    assert title(conn,'acting')==title(conn,'member') and title(conn,'acting').startswith('Edit ')
    assert title(conn,'unrelated')=='Before'

@pytest.fixture
def mirrored_recipe(recipe,monkeypatch):
    import hashlib
    app,conn=recipe
    fields={'source_type':"VARCHAR(32) DEFAULT 'manual'",'source_url':'VARCHAR(1000)','resolved_url':'VARCHAR(1000)','resolved_url_hash':'CHAR(64)','image_url':'VARCHAR(1000)','image_urls':'JSON','source_image_url':'VARCHAR(1000)','source_image_urls':'JSON','image_storage_key':'VARCHAR(1000)','raw_caption':'TEXT','raw_content':'TEXT','extraction_source':'VARCHAR(32)','recipe_source_used':'VARCHAR(48)','author_name':'VARCHAR(255)','caption_field':'VARCHAR(64)','status':"VARCHAR(32) DEFAULT 'ready'",'meal_category':'VARCHAR(16)','_createdDate':'DATETIME DEFAULT CURRENT_TIMESTAMP'}
    with conn.cursor() as cur:
        for owner in ['acting','member','unrelated']:
            for field,ddl in fields.items():cur.execute('ALTER TABLE `'+owner+'_saved_recipes` ADD `'+field+'` '+ddl)
            url='https://example.invalid/fixture/'+owner
            cur.execute('UPDATE `'+owner+'_saved_recipes` SET source_url=%s,resolved_url=%s,resolved_url_hash=%s',[url,url,hashlib.sha256(url.encode()).hexdigest()])
        cur.execute('CREATE TABLE shared_saved_recipes LIKE acting_saved_recipes')
        cur.execute('ALTER TABLE shared_saved_recipes ADD owner_id VARCHAR(64) NOT NULL,DROP PRIMARY KEY,ADD PRIMARY KEY(owner_id,_id),ADD UNIQUE KEY owner_hash(owner_id,resolved_url_hash)')
    monkeypatch.setattr(app,'DUAL_WRITE_SAVED_RECIPES',True)
    monkeypatch.setattr(app,'_dual_write_saved_recipe_to_shared',app._audit_real_dual_write)
    for owner in ['acting','member','unrelated']:app._dual_write_saved_recipe_to_shared(conn,owner,'recipe')
    yield app,conn

def mirror_titles(conn):
    with conn.cursor() as cur:cur.execute('SELECT owner_id,title FROM shared_saved_recipes ORDER BY owner_id');return {row['owner_id']:row['title'] for row in cur.fetchall()}

@pytest.mark.parametrize('failed_owner',['acting','member'])
def test_required_shared_failure_rolls_back_personal_and_all_mirrors(mirrored_recipe,monkeypatch,failed_owner):
    app,conn=mirrored_recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    with conn.cursor() as cur:cur.execute("CREATE TRIGGER reject_mirror BEFORE INSERT ON shared_saved_recipes FOR EACH ROW BEGIN IF NEW.owner_id='"+failed_owner+"' THEN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='fixture mirror failure'; END IF; END")
    with pytest.raises(pymysql.err.OperationalError,match='mirror failure'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert [title(conn,owner) for owner in ['acting','member','unrelated']]==['Before']*3
    assert mirror_titles(conn)=={'acting':'Before','member':'Before','unrelated':'Before'}

@pytest.mark.parametrize('sharing',[False,True])
def test_successful_edit_keeps_shared_and_personal_copies_equal(mirrored_recipe,monkeypatch,sharing):
    app,conn=mirrored_recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',sharing)
    assert app._update_saved_recipe('acting','recipe',{'title':'After'})['statusCode']==200
    assert mirror_titles(conn)=={'acting':'After','member':'After' if sharing else 'Before','unrelated':'Before'}


def test_lost_ack_commits_shared_and_personal_edit_together(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;monkeypatch.setattr(app,'SAVED_RECIPE_HOUSEHOLD_FANOUT',True)
    class LostAck:
        def __getattr__(self,name):return getattr(conn,name)
        def commit(self):conn.commit();raise pymysql.err.OperationalError(2013,'fixture acknowledgement lost')
    monkeypatch.setattr(app,'_mysql_conn',LostAck)
    with pytest.raises(pymysql.err.OperationalError,match='acknowledgement lost'):app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')==title(conn,'member')=='After'
    assert mirror_titles(conn)=={'acting':'After','member':'After','unrelated':'Before'}

@pytest.mark.parametrize('dual,read_shared',[(False,False),(False,True),(True,False),(True,True)])
def test_read_cutover_never_returns_stale_edit_when_dual_write_flag_is_off(mirrored_recipe,monkeypatch,dual,read_shared):
    app,conn=mirrored_recipe;monkeypatch.setattr(app,'DUAL_WRITE_SAVED_RECIPES',dual);monkeypatch.setenv('READ_SHARED_SAVED_RECIPES','true' if read_shared else 'false')
    assert app._update_saved_recipe('acting','recipe',{'title':'After'})['statusCode']==200
    assert title(conn,'acting')=='After'
    assert mirror_titles(conn)=={'acting':'After' if dual or read_shared else 'Before','member':'Before','unrelated':'Before'}

def test_legacy_global_id_collision_never_edits_another_owner(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        cur.execute("DELETE FROM shared_saved_recipes WHERE owner_id IN ('acting','member')")
        cur.execute('ALTER TABLE shared_saved_recipes DROP PRIMARY KEY,ADD PRIMARY KEY(_id)')
    with pytest.raises(ValueError,match='identity collision'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')=='Before'
    assert mirror_titles(conn)=={'unrelated':'Before'}

def test_existing_url_identity_collision_never_overwrites_another_recipe(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:cur.execute("UPDATE shared_saved_recipes SET _id='different-recipe' WHERE owner_id='acting'")
    with pytest.raises(ValueError,match='identity collision'):
        app._update_saved_recipe('acting','recipe',{'title':'After'})
    assert title(conn,'acting')=='Before'
    assert mirror_titles(conn)=={'acting':'Before','member':'Before','unrelated':'Before'}

def test_legacy_global_identity_schema_accepts_normal_owned_edit(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        cur.execute("UPDATE shared_saved_recipes SET _id=CONCAT('other-',owner_id) WHERE owner_id<>'acting'")
        cur.execute('ALTER TABLE shared_saved_recipes DROP PRIMARY KEY,ADD PRIMARY KEY(_id)')
    assert app._update_saved_recipe('acting','recipe',{'title':'After'})['statusCode']==200
    assert title(conn,'acting')=='After'
    assert mirror_titles(conn)=={'acting':'After','member':'Before','unrelated':'Before'}

def test_legacy_best_effort_collision_keeps_foreign_recipe_unchanged(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        cur.execute("DELETE FROM shared_saved_recipes WHERE owner_id IN ('acting','member')")
        cur.execute('ALTER TABLE shared_saved_recipes DROP PRIMARY KEY,ADD PRIMARY KEY(_id)')
    app._dual_write_saved_recipe_to_shared(conn,'acting','recipe')
    assert mirror_titles(conn)=={'unrelated':'Before'}

@pytest.mark.parametrize('collision',['foreign_id','same_url'])
def test_transactional_mirror_rejects_identity_collision_directly(mirrored_recipe,collision):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        if collision=='foreign_id':
            cur.execute("DELETE FROM shared_saved_recipes WHERE owner_id IN ('acting','member')")
            cur.execute('ALTER TABLE shared_saved_recipes DROP PRIMARY KEY,ADD PRIMARY KEY(_id)')
        else:cur.execute("UPDATE shared_saved_recipes SET _id='different-recipe' WHERE owner_id='acting'")
    before=mirror_titles(conn)
    conn.begin()
    try:
        with pytest.raises(ValueError,match='identity collision'):
            app._dual_write_saved_recipe_to_shared(conn,'acting','recipe',transactional=True)
    finally:conn.rollback()
    assert mirror_titles(conn)==before
