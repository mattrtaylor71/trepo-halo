import json
import pytest
from test_saved_recipe_edit_atomic import recipe, mirrored_recipe
from test_recipe_repair_preservation import stored, empty


def request(app,conn,action='replace_source',**extra):
    row=app._fetch_saved_recipe_by_id(conn,'acting','recipe')
    return {'content_revision':app._saved_recipe_content_revision(row),
            'recovery':{'action':action,'url':'https://example.invalid/new-recipe','text':'Beef\nIngredients: beef\nSteps: cook',**extra}}


def extraction(monkeypatch,app,hook=None):
    def extract(*args,**kwargs):
        if hook:hook()
        return {'title':'New title','ingredients':['new beef'],'instructions':['new step'],'notes':['source note']},'https://example.invalid/new-recipe'
    monkeypatch.setattr(app,'_extract_recovery_source',extract)
    monkeypatch.setattr(app,'_recipe_response_from_content',lambda *a,**k: ({},extract()[0]))


@pytest.mark.parametrize('action',['replace_source','paste_recipe'])
def test_recovery_keeps_identity_and_existing_manual_fields(mirrored_recipe,monkeypatch,action):
    app,conn=mirrored_recipe;empty(conn);extraction(monkeypatch,app)
    before=stored(conn)
    result=app._update_saved_recipe('acting','recipe',request(app,conn,action))
    assert result['statusCode']==200,result
    after=stored(conn);assert after['_id']==before['_id']
    assert after['title']==before['title'];assert after['notes']==before['notes']
    assert after['ingredients']==['new beef'];assert after['instructions']==['new step'];assert after['status']=='ready'
    if action=='replace_source':
        assert after['source_url'].endswith('/new-recipe')
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM shared_saved_recipes WHERE owner_id='acting' AND _id='recipe'")
            shared=cur.fetchone()
        for field in ['source_url','resolved_url','resolved_url_hash','status']:
            assert shared[field]==after[field]
        assert app._saved_recipe_content_revision(shared)==app._saved_recipe_content_revision(app._fetch_saved_recipe_by_id(conn,'acting','recipe'))


def test_replacement_requires_explicit_field_selection(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;extraction(monkeypatch,app)
    body=request(app,conn,replace_fields=['ingredients','instructions'])
    result=app._update_saved_recipe('acting','recipe',body)
    assert result['statusCode']==200,result
    assert stored(conn)['ingredients']==['new beef'];assert stored(conn)['title']=='Before'


@pytest.mark.parametrize('kind',['edit','delete'])
def test_recovery_cannot_overwrite_concurrent_user_decision(mirrored_recipe,monkeypatch,kind):
    app,conn=mirrored_recipe;empty(conn)
    def during():
        with conn.cursor() as cur:cur.execute("DELETE FROM acting_saved_recipes" if kind=='delete' else "UPDATE acting_saved_recipes SET title='My new title'")
    extraction(monkeypatch,app,during)
    result=app._update_saved_recipe('acting','recipe',request(app,conn))
    assert result['statusCode']==(404 if kind=='delete' else 409)
    if kind=='delete':assert stored(conn) is None
    else:assert stored(conn)['title']=='My new title' and stored(conn)['ingredients']==[]


def test_incomplete_and_missing_revision_preserve_original(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;empty(conn);before=stored(conn)
    monkeypatch.setattr(app,'_extract_recovery_source',lambda *a,**k: ({'ingredients':['url']},'https://example.invalid/new'))
    assert app._update_saved_recipe('acting','recipe',request(app,conn))['statusCode']==422
    assert app._update_saved_recipe('acting','recipe',{'recovery':{'action':'paste_recipe','text':'stuff'}})['statusCode']==400
    assert stored(conn)==before


@pytest.mark.parametrize('action', ['replace_source', 'paste_recipe'])
def test_recovery_timeout_keeps_all_stored_fields(mirrored_recipe, monkeypatch, action):
    import recovery_web
    import time
    app, conn = mirrored_recipe
    empty(conn)
    before = stored(conn)
    original_budget = recovery_web.recovery_budget
    monkeypatch.setattr(recovery_web, 'recovery_budget', lambda: original_budget(0.04))
    def slow(*a, **k):
        time.sleep(1)
        raise AssertionError('Timed out extraction must never complete')
    monkeypatch.setattr(app, '_extract_recovery_source', slow)
    monkeypatch.setattr(app, '_recipe_response_from_content', slow)
    response = app._update_saved_recipe('acting','recipe',request(app, conn, action))
    assert response['statusCode'] == 422
    assert 'has not changed' in json.loads(response['body'])['error']
    assert stored(conn) == before


def test_recovery_preserves_all_human_notes(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;empty(conn);extraction(monkeypatch,app)
    notes=['https://example.invalid/my-note','Ingredients','Keep this note']
    with conn.cursor() as cur:
        cur.execute('UPDATE acting_saved_recipes SET notes=%s',[json.dumps(notes)])
    assert app._update_saved_recipe('acting','recipe',request(app,conn,'paste_recipe'))['statusCode']==200
    assert stored(conn)['notes']==notes


@pytest.mark.parametrize('action', ['paste_recipe', 'replace_source'])
def test_recovery_removes_only_machine_pending_note_without_requiring_a_new_source(mirrored_recipe, monkeypatch, action):
    app, conn = mirrored_recipe
    empty(conn)
    extraction(monkeypatch, app)
    user_notes = ['Use my blue pan', 'https://example.invalid/my-personal-note']
    pending = "We couldn't read the full recipe from this post yet — we're still working on it."
    with conn.cursor() as cur:
        cur.execute('UPDATE acting_saved_recipes SET notes=%s', [json.dumps([pending, *user_notes])])
    before = stored(conn)
    result = app._update_saved_recipe('acting', 'recipe', request(app, conn, action))
    assert result['statusCode'] == 200, result
    after = stored(conn)
    assert after['_id'] == before['_id'] and after['status'] == 'ready'
    assert after['notes'] == user_notes
    assert after['ingredients'] == ['new beef'] and after['instructions'] == ['new step']
    if action == 'paste_recipe':
        for field in ['source_url', 'resolved_url', 'resolved_url_hash']:
            assert after[field] == before[field]
    with conn.cursor() as cur:
        cur.execute("SELECT notes, status FROM shared_saved_recipes WHERE owner_id='acting' AND _id='recipe'")
        shared = cur.fetchone()
    assert json.loads(shared['notes']) == user_notes and shared['status'] == 'ready'


@pytest.mark.parametrize('replace',[[],['ingredients'],['instructions']])
def test_recovery_never_merges_unrelated_cooking_content(mirrored_recipe,monkeypatch,replace):
    app,conn=mirrored_recipe;extraction(monkeypatch,app)
    with conn.cursor() as cur:
        cur.execute("UPDATE acting_saved_recipes SET ingredients='[\"2 cups flour\"]',instructions='[]'")
    before=stored(conn)
    response=app._update_saved_recipe('acting','recipe',request(app,conn,'paste_recipe',replace_fields=replace))
    assert response['statusCode']==409,response
    assert json.loads(response['body'])['code']=='recipe_replacement_confirmation_required'
    assert stored(conn)==before


@pytest.mark.parametrize('ingredient',['https://example.invalid/recipe',{'url':'https://example.invalid/recipe'}])
def test_malformed_json_ld_cannot_become_character_ingredients(recipe,monkeypatch,ingredient):
    import sys
    from unittest.mock import Mock
    app,_=recipe
    monkeypatch.setitem(sys.modules,'recovery_web',Mock(fetch_page=lambda url:('<html></html>',url)))
    monkeypatch.setattr(app,'_select_richest_recipe_node',lambda soup:{'name':'Recipe','recipeIngredient':ingredient,'recipeInstructions':['Cook it.']})
    result,_=app._extract_recovery_source('https://example.invalid/recipe')
    assert not app.recipe_content_complete(result)
