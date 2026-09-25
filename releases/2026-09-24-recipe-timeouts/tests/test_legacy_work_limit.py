"""Legacy URL receipts and actionable source recovery under real bounded providers."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import recovery_web
import trepo_auth
from test_saved_recipe_edit_atomic import recipe, mirrored_recipe
from test_recipe_repair_preservation import stored, empty


def event(method, body, item=False):
    return {'rawPath':'/saved-recipes/acting'+('/recipe' if item else ''),
            'pathParameters':{'owner':'acting',**({'item_id':'recipe'} if item else {})},
            'body':json.dumps(body),'requestContext':{'requestId':'synthetic-legacy-timeout','http':{'method':method}}}


def setup(app, monkeypatch):
    monkeypatch.setattr(trepo_auth,'require_owner',lambda event:None)
    monkeypatch.setattr(app,'_log_event',Mock())
    monkeypatch.setattr(app,'_report_backend_error',Mock())
    monkeypatch.setattr(app,'_explore_shortcircuit_extraction',lambda *a,**kw:None)
    monkeypatch.setattr(app,'_serialize_saved_recipe_with_availability',lambda conn,owner,row,**kw:(app._serialize_row(row),{}))
    monkeypatch.setattr(app,'_resolve_url',lambda *a,**kw:time.sleep(.4))


@pytest.mark.parametrize('dispatch_ok',[True,False])
def test_legacy_timeout_receipt_retries_retain_one_identity_and_schedule_at_most_once(mirrored_recipe,monkeypatch,dispatch_ok):
    app,conn=mirrored_recipe;setup(app,monkeypatch)
    dispatch=Mock(return_value=dispatch_ok)
    monkeypatch.setattr(app,'_invoke_saved_recipe_repair_async',dispatch)
    request=event('POST',{'url':'https://example.invalid/new-slow-recipe'})
    context=SimpleNamespace(get_remaining_time_in_millis=lambda:15040)
    first=app.handler(request,context)
    assert first['statusCode']==201,first
    body=json.loads(first['body']);rid=body['recipe']['id']
    assert body['content_outcome']==('processing' if dispatch_ok else 'link_retained')
    assert body['recipe']['status']==('repairing' if dispatch_ok else 'failed')
    assert not app.recipe_content_complete(body['recipe']) and body['code']=='recipe_content_incomplete'
    assert body['recovery']['recipe_id']==rid
    second=app.handler(request,context)
    assert second['statusCode']==200 and json.loads(second['body'])['recipe']['id']==rid
    assert json.loads(second['body'])['deduped'] is True and dispatch.call_count==1
    with conn.cursor() as cursor:
        cursor.execute('SELECT _id,status FROM acting_saved_recipes WHERE source_url=%s',['https://example.invalid/new-slow-recipe'])
        rows=cursor.fetchall();assert len(rows)==1 and rows[0]['_id']==rid
        cursor.execute('SELECT status FROM shared_saved_recipes WHERE owner_id=%s AND _id=%s',['acting',rid])
        assert cursor.fetchone()['status']==rows[0]['status']
        cursor.execute("SELECT title FROM member_saved_recipes WHERE _id='recipe'");assert cursor.fetchone()['title']=='Before'
    app._report_backend_error.assert_not_called()


def test_legacy_timeout_dedupe_preserves_existing_complete_recipe(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;setup(app,monkeypatch);before=stored(conn)
    dispatch=Mock();monkeypatch.setattr(app,'_invoke_saved_recipe_repair_async',dispatch)
    result=app.handler(event('POST',{'url':before['source_url']}),SimpleNamespace(get_remaining_time_in_millis=lambda:15040))
    assert result['statusCode']==200 and json.loads(result['body'])['recipe']['id']=='recipe'
    assert stored(conn)==before and not dispatch.called


@pytest.mark.parametrize('url',['ftp://fixture.invalid/recipe','This is pasted recipe text','https://www.instagram.com/profile-only/'])
def test_legacy_timeout_revalidates_input_before_retaining(mirrored_recipe,monkeypatch,url):
    app,conn=mirrored_recipe;setup(app,monkeypatch)
    save=Mock();monkeypatch.setattr(app,'_save_saved_recipe_record',save)
    result=app.handler(event('POST',{'url':url}),SimpleNamespace(get_remaining_time_in_millis=lambda:15040))
    assert 400<=result['statusCode']<500
    save.assert_not_called()


@pytest.mark.parametrize('action',['replace_source','paste_recipe'])
@pytest.mark.parametrize('cutoff',['native','recovery'])
def test_explicit_recovery_provider_timeout_is_actionable_and_preserves_revision(mirrored_recipe,monkeypatch,action,cutoff):
    app,conn=mirrored_recipe;empty(conn);setup(app,monkeypatch)
    before=stored(conn)
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM shared_saved_recipes WHERE owner_id='acting' AND _id='recipe'");shared_before=cursor.fetchone()
    original_budget=recovery_web.recovery_budget
    if cutoff=='recovery':monkeypatch.setattr(recovery_web,'recovery_budget',lambda *a,**kw:original_budget(.04))
    monkeypatch.setattr(app,'_refine_recipe_structured',lambda *a,**kw:time.sleep(.4))
    # Keep the actual source parser and bounded refinement; avoid external network.
    monkeypatch.setattr(recovery_web,'fetch_page',lambda url:('<html><article>Ingredients: 1 cup oats. Simmer gently until tender.</article></html>',url))
    revision=app._saved_recipe_content_revision(app._fetch_saved_recipe_by_id(conn,'acting','recipe'))
    body={'content_revision':revision,'recovery':{'action':action,'url':'https://example.invalid/new-source','text':'Ingredients: 1 cup oats. Simmer gently until tender.'}}
    context=SimpleNamespace(get_remaining_time_in_millis=lambda:15040 if cutoff=='native' else 120000)
    result=app.handler(event('PUT',body,item=True),context)
    assert result['statusCode']==422 and 'has not changed' in json.loads(result['body'])['error'],result
    assert stored(conn)==before
    assert app._saved_recipe_content_revision(app._fetch_saved_recipe_by_id(conn,'acting','recipe'))==revision
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM shared_saved_recipes WHERE owner_id='acting' AND _id='recipe'");assert cursor.fetchone()==shared_before
    app._report_backend_error.assert_not_called()


def test_legacy_timeout_does_not_acknowledge_a_failed_insert(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;setup(app,monkeypatch)
    with conn.cursor() as cursor:
        cursor.execute("CREATE TRIGGER reject_timeout_insert BEFORE INSERT ON acting_saved_recipes FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='synthetic insert failed'")
    dispatch=Mock();monkeypatch.setattr(app,'_invoke_saved_recipe_repair_async',dispatch)
    result=app.handler(event('POST',{'url':'https://example.invalid/rejected-recipe'}),SimpleNamespace(get_remaining_time_in_millis=lambda:15040))
    assert result['statusCode']==500 and not dispatch.called
    with conn.cursor() as cursor:
        cursor.execute('SELECT _id FROM acting_saved_recipes WHERE source_url=%s',['https://example.invalid/rejected-recipe']);assert not cursor.fetchall()
