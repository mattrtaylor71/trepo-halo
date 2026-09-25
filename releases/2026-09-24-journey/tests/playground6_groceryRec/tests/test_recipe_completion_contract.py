import json
import pytest
from test_saved_recipes_dualwrite import load_app_module
from test_saved_recipe_edit_atomic import recipe, mirrored_recipe


def record(**updates):
    return {'id':'retained-id','title':'Chicken','source_url':'https://example.invalid/recipe',
            'status':'ready','ingredients':['Chicken'], 'instructions':['Cook the chicken.'], **updates}


@pytest.mark.parametrize('recipe', [record(ingredients=[]),record(instructions=[]),
    record(ingredients=['https://example.invalid/article'],instructions=[]),
    record(status='repairing',ingredients=[],instructions=[])])
def test_legacy_sync_cannot_receive_ready_success_for_incomplete_content(recipe):
    app = load_app_module()
    response = app._build_saved_recipe_post_response('owner', [{'recipe':recipe,'deduped':False}])
    assert response['statusCode'] == 422
    body = json.loads(response['body'])
    assert body['recipe']['id'] == 'retained-id'
    assert body['code'] == 'recipe_content_incomplete'
    assert body['recovery']['recipe_id'] == 'retained-id'


def test_real_simple_recipe_still_uses_the_shipped_success_contract():
    app = load_app_module()
    response = app._build_saved_recipe_post_response('owner', [{'recipe':record(),'deduped':False}])
    assert response['statusCode'] == 201
    assert json.loads(response['body'])['recipe']['title'] == 'Chicken'


def test_legacy_job_completion_requires_usable_content():
    app = load_app_module()
    job = {'status':'COMPLETED','job_id':'job','owner':'owner','recipe_ids':['retained-id'], 'result_count':1}
    response = app._build_saved_recipe_batch_job_response('owner', job,
        recipes=[record(status='failed',instructions=[])],
        results=[{'recipe':record(status='failed',instructions=[]),'deduped':False}])
    body = json.loads(response['body'])
    assert body['job']['status'] == 'failed'
    assert body['job']['error']
    assert body['retained_recipes'][0]['id'] == 'retained-id'
    assert body['count'] == 0


def test_repairing_job_keeps_polling_without_claiming_recipe_is_ready():
    app = load_app_module()
    job = {'status':'COMPLETED','job_id':'job','owner':'owner','result_count':1}
    recipe = record(status='repairing',ingredients=[],instructions=[])
    body = json.loads(app._build_saved_recipe_batch_job_response('owner',job,recipes=[recipe],results=[{'recipe':recipe}])['body'])
    assert body['job']['status'] == 'running'


def test_mixed_batch_reports_ready_and_retained_separately():
    app = load_app_module()
    good=record(id='ready-id'); bad=record(status='failed',instructions=[])
    job = {'status':'COMPLETED','job_id':'job','owner':'owner','result_count':2}
    body=json.loads(app._build_saved_recipe_batch_job_response('owner',job,recipes=[good,bad],results=[{'recipe':good},{'recipe':bad}])['body'])
    assert body['job']['status'] == 'completed'
    assert body['count'] == 1
    assert [r['id'] for r in body['recipes']] == ['ready-id']
    assert body['retained_recipes'][0]['id'] == 'retained-id'
    assert body['partial_errors'][0]['code'] == 'recipe_content_incomplete'


def test_complete_manual_recovery_sets_both_saved_copies_ready(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        cur.execute("UPDATE acting_saved_recipes SET ingredients='[]',instructions='[]',status='failed'")
    response=app._update_saved_recipe('acting','recipe',{'ingredients':['Chicken'],'steps':['Cook the chicken.']})
    assert response['statusCode']==200
    with conn.cursor() as cur:
        for table,scope in [('acting_saved_recipes','1=1'),('shared_saved_recipes',"owner_id='acting'")]:
            cur.execute(f"SELECT status FROM {table} WHERE {scope}")
            assert cur.fetchone()['status']=='ready'


def test_title_edit_does_not_make_an_incomplete_recipe_ready(mirrored_recipe):
    app,conn=mirrored_recipe
    with conn.cursor() as cur:
        cur.execute("UPDATE acting_saved_recipes SET instructions='[]',status='failed'")
    assert app._update_saved_recipe('acting','recipe',{'title':'Better name'})['statusCode']==200
    with conn.cursor() as cur:
        cur.execute('SELECT status FROM acting_saved_recipes')
        assert cur.fetchone()['status']=='failed'


def test_revision_aware_edit_rejects_stale_content_but_old_clients_keep_working(mirrored_recipe):
    app,conn=mirrored_recipe
    response=app._update_saved_recipe('acting','recipe',{'title':'Wrong', 'content_revision':'stale'})
    assert response['statusCode']==409
    assert app._update_saved_recipe('acting','recipe',{'title':'Legacy edit'})['statusCode']==200
