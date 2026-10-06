import json
import threading
import uuid
from datetime import datetime,timedelta,timezone
import pymysql
import pytest
from household_brain import core,events,migrate,operator,runtime,sources,store
from .conftest import sql,edit,item,recipe

def brain(db,actor='a'):
    return store.read_context(db,actor,'thyme',actor=actor)

def test_commits_rollback_and_read_repair(db):
    initial=brain(db)
    db.begin();recipe(db);db.rollback()
    assert store.is_current(db,initial['fence'])
    recipe(db)
    assert not store.is_current(db,initial['fence'])
    assert brain(db)['context']['observations']=={'recipe_saved':['Bean tacos']}
    sql(db,"DELETE FROM shared_saved_recipes WHERE _id='r'")
    assert brain(db)['context']['observations']=={}

def test_full_household_private_and_foreign_boundaries(db):
    recipe(db);recipe(db,'other-recipe','b','Private b food');recipe(db,'foreign-recipe','foreign','Secret')
    item(db);item(db,'Eggs','eggs',owner_id='b');item(db,'Secret','secret',owner_id='foreign')
    a=brain(db)['context'];b=brain(db,'b')['context']
    shared=store.read_context(db,'a','recipe')['context']
    assert a['observations']=={'recipe_saved':['Bean tacos']}
    assert b['observations']=={'recipe_saved':['Private b food']}
    assert not shared['observations']
    assert {p['name'] for p in shared['products']}=={'Milk','Eggs'}
    assert 'Secret' not in core.encode(a)
    with pytest.raises(PermissionError):store.read_context(db,'a',actor='b')

def test_membership_change_revokes_and_removes_previous_shared_preferences(db):
    edit(db,'set',actor_id='b',key='cuisines',value=['Thai'],visibility='household')
    before=brain(db)
    sql(db,"UPDATE new_users SET owner_id='other' WHERE user_id='b'")
    assert not store.is_current(db,before['fence'])
    assert not brain(db)['context']['explicit_preferences']
    assert not brain(db,'b')['context']['explicit_preferences']

def test_controls_scopes_and_idempotency_audit(db):
    req={'household_id':'h','action':'set','actor_id':'a','key':'cuisines','value':['Thai'],'reason':'Founder approved preference',
         'expected_revision':sql(db,"SELECT input_revision FROM household_brain_state WHERE household_id='h'")[0]['input_revision'],'operation_id':'operation-123'}
    result=operator.change(db,'operator',req)
    assert operator.change(db,'operator',{**req,'expected_revision':999})==result
    with pytest.raises(ValueError,match='operation_conflict'):operator.change(db,'operator',{**req,'value':['French']})
    with pytest.raises(ValueError,match='revision_conflict'):operator.change(db,'operator',{**req,'operation_id':'different-id'})
    assert len(sql(db,'SELECT * FROM household_brain_audit'))==1
    assert brain(db)['context']['explicit_preferences']['cuisines']['value']==['Thai']
    assert not brain(db,'b')['context']['explicit_preferences']
    edit(db,'set',actor_id='a',key='cuisines',value=['Thai'],visibility='household')
    assert brain(db,'b')['context']['explicit_preferences']
    edit(db,'forget',actor_id='a',key='cuisines')
    assert not brain(db)['context']['explicit_preferences']
    with pytest.raises(ValueError):edit(db,'set',actor_id='a',key='allergies',value=['milk'])

def test_pause_resume_reset_keep_explicit_never_relearn_old_history(db):
    recipe(db);item(db,quantity_value=0)
    edit(db,'set',actor_id='a',key='favorite_foods',value=['Beans'])
    assert brain(db)['context']['observations']
    edit(db,'pause')
    paused=brain(db)['context']
    assert not paused['observations'] and not paused['products'] and paused['groceries']
    assert paused['explicit_preferences']
    edit(db,'resume')
    assert not brain(db)['context']['observations']
    edit(db,'reset')
    assert brain(db)['context']['explicit_preferences']
    with pytest.raises(ValueError,match='cannot_backfill_reset_history'):edit(db,'enable',backfill_days=90)

def test_suppress_product_survives_rebuild(db):
    item(db,quantity_value=0)
    key=brain(db)['context']['products'][0]['key']
    edit(db,'suppress_product',product_key=key)
    assert not brain(db)['context']['products'] and not brain(db)['context']['groceries']
    store.rebuild(db,'h')
    assert not brain(db)['context']['products']
    edit(db,'restore_product',product_key=key)
    assert brain(db)['context']['products']

def test_recipe_ingredient_use_is_distinct_from_save_and_undo_removes_it(db):
    recipe(db);item(db)
    changes=[{'mode':'consume','snapshot':{'product_name':'Milk','owner_id':'a'},'recipe_title':'Pancakes'}]
    sql(db,"INSERT INTO kitchen_amount_operations VALUES('a','consume',UTC_TIMESTAMP(6),NULL,%s)",[json.dumps(changes)])
    result=brain(db)['context']
    assert result['observations']['recipe_used']==['Pancakes']
    assert result['products'][0]['evidence']['consumed']==1
    sql(db,"UPDATE kitchen_amount_operations SET undone_by='undo' WHERE operation_id='consume'")
    result=brain(db)['context']
    assert 'recipe_used' not in result['observations']
    assert 'consumed' not in result['products'][0]['evidence']

def test_list_check_is_not_purchase_or_missing_stock(db):
    item(db,quantity_value=0)
    sql(db,"INSERT INTO shared_shopping_list VALUES('l','a','Milk','ADDED')")
    assert not brain(db)['context']['groceries']
    sql(db,"UPDATE shared_shopping_list SET action='CHECKED'")
    result=brain(db)['context']
    assert not result['groceries'] and 'purchase' not in result['products'][0]['evidence']

def test_receipt_date_required_and_pending_analysis_excluded(db):
    item(db,provisional_payload=json.dumps({'scan_source':'receipt_scan','source_job_id':'job'}))
    item(db,'Pending','pending',analysis_status='processing')
    item(db,'Review','review',needs_review=1)
    result=brain(db)['context']
    assert len(result['products'])==1
    assert result['products'][0]['evidence']=={'inventory_observed':1,'receipt_imported':1}
    sql(db,"UPDATE shared_kitchen SET provisional_payload=%s WHERE _id='milk'",[json.dumps({'scan_source':'receipt_scan','source_job_id':'job','receipt_purchase_date':(datetime.now()-timedelta(days=2)).date().isoformat()})])
    assert brain(db)['context']['products'][0]['evidence']['purchase']==1

def event(db,**changes):
    generation=sql(db,"SELECT generation FROM household_brain_state WHERE household_id='h'")[0]['generation']
    return {'event_id':uuid.uuid4().hex,'entity_id':'confirmed-selection','actor_id':'a','household_id':'h',
        'occurred_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),'generation':generation,
        'kind':'product_selected','payload':{'name':'Milk','visibility':'personal'},**changes}

def test_event_retry_payload_conflict_and_private_scope(db):
    e=event(db)
    assert not events.ingest(db,'shop',e)['duplicate']
    assert events.ingest(db,'shop',e)['duplicate']
    with pytest.raises(ValueError,match='event_id_conflict'):events.ingest(db,'shop',{**e,'payload':{'name':'Eggs','visibility':'personal'}})
    assert brain(db)['context']['products'][0]['evidence']=={'product_selected':1}
    assert not brain(db,'b')['context']['products']
    with pytest.raises(ValueError):events.ingest(db,'shop',event(db,kind='purchase'))
    with pytest.raises(PermissionError):events.ingest(db,'shop',event(db,actor_id='foreign'))

def test_reversal_before_original_and_reset_replay(db):
    e=event(db)
    rev=event(db,kind='reverse',reverses=e['event_id'])
    events.ingest(db,'shop',rev);events.ingest(db,'shop',e)
    assert not brain(db)['context']['products']
    edit(db,'reset')
    assert events.ingest(db,'shop',e)['duplicate']
    with pytest.raises(ValueError,match='generation'):events.ingest(db,'shop',{**e,'event_id':'new-old-generation'})

def test_hard_restrictions_fence_is_separate_and_current(db):
    old=brain(db)
    sql(db,"INSERT INTO user_dietary_preferences(owner_id,allergies) VALUES('h','[\"peanuts\"]')")
    assert not store.is_current(db,old['fence'])
    result=brain(db)
    assert result['dietary_preferences']['allergies']==['peanuts']
    assert 'peanuts' not in core.encode(result['context'])

def test_projection_concurrent_commit_cannot_publish_old_revision(db,monkeypatch):
    recipe(db)
    original=store.project
    called=False
    def race(*args,**kwargs):
        nonlocal called
        result=original(*args,**kwargs)
        if not called:
            called=True
            c=pymysql.connect(host='127.0.0.1',port=33826,user='root',database=sql(db,'SELECT DATABASE() name')[0]['name'],autocommit=True,cursorclass=pymysql.cursors.DictCursor)
            sql(c,"UPDATE shared_saved_recipes SET title='Edited tacos'");c.close()
        return result
    monkeypatch.setattr(store,'project',race)
    result=brain(db)
    assert result['context']['observations']['recipe_saved']==['Edited tacos']

def test_cache_expiry_worker_and_membership_cleanup(db):
    recipe(db);store.rebuild(db,'h')
    assert len(sql(db,"SELECT * FROM household_brain_contexts WHERE household_id='h'"))==3
    sql(db,"UPDATE new_users SET owner_id='other' WHERE user_id='b'")
    assert store.run_worker(db)['failed']==0
    assert len(sql(db,"SELECT * FROM household_brain_contexts WHERE household_id='h'"))==2
    sql(db,"UPDATE household_brain_contexts SET expires_at=UTC_TIMESTAMP()-INTERVAL 1 SECOND")
    assert brain(db)['context']['observations']

def test_missing_oversized_source_never_serves_old_projection(db,monkeypatch):
    recipe(db);brain(db)
    sql(db,"UPDATE shared_saved_recipes SET title='Changed'")
    original=sources.load
    monkeypatch.setattr(sources,'load',lambda *a: (_ for _ in ()).throw(ValueError('source_limit_exceeded')))
    with pytest.raises(ValueError,match='source_limit'):brain(db)
    assert store.run_worker(db)['failed']>=1
    monkeypatch.setattr(sources,'load',original)
    assert brain(db)['context']['observations']['recipe_saved']==['Changed']

def test_disable_and_capture_kill_switch(db):
    before=brain(db)
    edit(db,'disable')
    assert brain(db) is None and not store.is_current(db,before['fence'])
    edit(db,'enable')
    sql(db,'UPDATE household_brain_config SET capture_enabled=0')
    assert brain(db) is None

def test_install_repeat_and_default_off_provision(db):
    migrate.install(db)
    sql(db,"INSERT INTO new_users VALUES('new','newhome')")
    assert sql(db,"SELECT enabled FROM household_brain_state WHERE household_id='newhome'")==[{'enabled':0}]
    assert brain(db,'new') is None

def test_private_service_rejects_cross_household_http_and_controls(db):
    assert runtime.service(db,{'actor':'a','household':'other'})['statusCode']==403
    assert runtime.service(db,{'actor':'a','household':'h','operation':'control'})['statusCode']==403
    assert runtime.reader_handler({'requestContext':{},'actor':'a','household':'h'},None)['statusCode']==403
    assert runtime.service(db,{'actor':'a','household':'h'})['statusCode']==200

def test_existing_transaction_is_never_committed_by_brain(db):
    db.begin();recipe(db)
    with pytest.raises(RuntimeError,match='dedicated_connection_required'):brain(db)
    db.rollback()
    assert not sql(db,'SELECT * FROM shared_saved_recipes')

def test_personal_pause_reset_are_scoped_to_that_member(db):
    recipe(db);recipe(db,'r-b','b','Pasta')
    edit(db,'pause',actor_id='a')
    assert not brain(db)['context']['observations']
    assert brain(db,'b')['context']['observations']=={'recipe_saved':['Pasta']}
    edit(db,'resume',actor_id='a')
    assert not brain(db)['context']['observations']
    edit(db,'reset',actor_id='a')
    assert brain(db,'b')['context']['observations']=={'recipe_saved':['Pasta']}

def test_membership_revokes_shared_service_evidence(db):
    events.ingest(db,'shop',event(db,actor_id='b',payload={'name':'Former member choice','visibility':'household'}))
    assert brain(db)['context']['products']
    sql(db,"UPDATE new_users SET owner_id='other' WHERE user_id='b'")
    assert not brain(db)['context']['products']

def test_v2_legacy_signed_entrypoint_uses_shared_context_and_denies_controls(db,monkeypatch):
    from family_brain import api,store as legacy
    monkeypatch.setenv('HOUSEHOLD_BRAIN_V2','1')
    recipe(db)
    verify=lambda token:{'user_id':'a'} if token=='valid' else None
    request={'owner':'a','authorization':'Bearer valid','operation':'context'}
    result=api.handle(db,request,verify)
    assert result['statusCode']==200 and result['context']['schema']=='household-brain-v2'
    assert api.handle(db,{**request,'operation':'control'},verify)['statusCode']==403
    assert api.handle(db,{**request,'authorization':'Bearer forged'},verify)['statusCode']==403
    with pytest.raises(PermissionError,match='internally_managed'):legacy.control(db,'a',{})
    assert legacy.is_current(db,result['fence'])

def test_dissolved_household_does_not_starve_worker(db):
    store.rebuild(db,'h')
    sql(db,"DELETE FROM new_users WHERE owner_id='h'")
    store.run_worker(db)
    assert sql(db,"SELECT enabled FROM household_brain_state WHERE household_id='h'")==[{'enabled':0}]
    assert not sql(db,"SELECT * FROM household_brain_contexts WHERE household_id='h'")

def test_source_cap_raises_instead_of_silently_truncating(db):
    with db.cursor() as cur:
        with pytest.raises(ValueError,match='source_limit_exceeded'):
            sources.bounded(cur,'SELECT user_id FROM new_users',[],cap=1)
