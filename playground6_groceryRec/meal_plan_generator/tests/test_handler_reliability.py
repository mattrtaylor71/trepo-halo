"""Exercise the real generator control flow and SQL queue without paid models."""
import ast,json,time,sys,types,importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_generation_jobs import db,jobs
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def runtime(db):
    with db.cursor() as c:
        c.execute('CREATE TABLE new_users (user_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64))')
        c.execute("INSERT INTO new_users VALUES ('a','home')")
        c.execute('CREATE TABLE shared_meal_plan (owner_id VARCHAR(64) PRIMARY KEY,_id VARCHAR(64),_owner VARCHAR(64),status VARCHAR(32),focus TEXT,explanation_title TEXT,explanation_paragraph TEXT,plan JSON,error_message TEXT,_createdDate DATETIME,_updatedDate DATETIME)')
        c.execute("INSERT INTO shared_meal_plan (owner_id,status,focus) VALUES ('a','regenerating','fresh')")
    state={'household':'home','members':['a'],'focuses':[]}
    digest=lambda x:__import__('hashlib').sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    @contextmanager
    def publication(conn,snapshot):
        conn.begin()
        try:
            if state['household']!=snapshot['household']:raise Changed('household_changed')
            yield
            conn.commit()
        except BaseException:
            conn.rollback();raise
    class Changed(Exception):
        def __init__(self,reason):self.reason=reason
    def capture(*a,**kw):return dict(household=state['household'],members=['a'],dietary=digest({}),versions={'a':1},inventory='same',controls=None,owner='a',deadline=time.monotonic()+280)
    def model(ingredients,focus,*args):
        state['focuses'].append(focus)
        return {'recipes':[{'title':'Meal','ingredients':['oats'],'steps':['Stir oats.']} for _ in range(12)]}
    scope=dict(generation_jobs=jobs,recipe_generation_guard=SimpleNamespace(digest=digest,publication=publication,capture=capture,GenerationInputsChanged=Changed),
        brain_store=SimpleNamespace(membership=lambda *a,**kw:state['household'],generation_context=lambda *a:None,dietary_preferences=lambda *a:{}),
        contextmanager=contextmanager,json=json,re=__import__('re'),os=SimpleNamespace(getenv=lambda key:'fixture'),time=time,
        _mysql_conn=lambda:db,_get_household_member_ids=lambda *a:['a'],_acquire_household_locks=lambda *a:[],_release_household_locks=lambda *a:None,
        DEFAULT_FOCUS='default',MIN_KITCHEN_ITEMS=5,ENABLE_MEAL_PLAN_IMAGES=False,SLOT_ORDER=[(i//4,'Today','meal') for i in range(12)],
        _resolve_table=lambda *a:('shared_kitchen',None,None),_get_kitchen_ingredients=lambda *a:[{'name':'oats'}]*5,
        _generate_plan_with_gpt=model,_recipe_violates=lambda *a:False,_format_preferences_block=lambda *a:'',_report_backend_error=lambda *a,**kw:None)
    tree=ast.parse((ROOT/'app.py').read_text())
    names={'_locked_members','_household_write','_generation_publication','_validate_generated_recipes','_record_publication','_repair_dietary_slots','_set_status','_set_empty'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    functions.append(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='handler'))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(ROOT/'app.py'),'exec'),scope)
    return scope,state

def get(db,table):
    with db.cursor() as c:c.execute('SELECT * FROM '+table);return c.fetchone()

def invoke(runtime,event=None):
    scope,state=runtime
    scope['handler']({'_ai_consent_actor':'requesting-member',**(event or {'owner':'a','focus':'stale'})},SimpleNamespace(get_remaining_time_in_millis=lambda:300000))

def test_success_publishes_plan_and_provenance_atomically(db,runtime):
    invoke(runtime)
    assert runtime[1]['focuses']==['fresh']
    assert get(db,'shared_meal_plan')['status']=='ready'
    assert len(json.loads(get(db,'shared_meal_plan')['plan']))==12
    assert get(db,'meal_plan_generation_jobs')['state']=='ready'
    assert get(db,'meal_plan_publications')['household_id']=='home'

def test_provider_timeout_leaves_durable_retry_without_empty_success(db,runtime):
    runtime[0]['_generate_plan_with_gpt']=lambda *a:(_ for _ in ()).throw(TimeoutError())
    invoke(runtime)
    assert get(db,'shared_meal_plan')['status']=='regenerating'
    assert get(db,'meal_plan_generation_jobs')['state']=='pending'

def test_focus_edit_mid_model_is_not_overwritten_and_retry_uses_new_focus(db,runtime):
    scope,state=runtime;model=scope['_generate_plan_with_gpt']
    def edit(*args):
        with db.cursor() as c:c.execute("UPDATE shared_meal_plan SET focus='changed'")
        return model(*args)
    scope['_generate_plan_with_gpt']=edit;invoke(runtime)
    queue=get(db,'meal_plan_generation_jobs')
    assert queue['state']=='pending' and get(db,'shared_meal_plan')['focus']=='changed'
    scope['_generate_plan_with_gpt']=model
    invoke(runtime,{'owner':'a','generation_retry_token':queue['run_token'],'generation_retry_scope':'home'})
    assert get(db,'shared_meal_plan')['status']=='ready' and state['focuses']==['fresh','changed']

def test_newer_request_prevents_old_success_and_failure(db,runtime):
    model=runtime[0]['_generate_plan_with_gpt']
    def supersede(*args):jobs.supersede_pending(db,'home',{'owner':'a','focus':'new'});return model(*args)
    runtime[0]['_generate_plan_with_gpt']=supersede;invoke(runtime)
    assert get(db,'meal_plan_generation_jobs')['state']=='pending'
    assert get(db,'shared_meal_plan')['plan'] is None

@pytest.mark.parametrize('field,value',[('steps',[]),('steps','text'),('ingredients',[None]),('title','')])
def test_malformed_recipe_never_becomes_ready(db,runtime,field,value):
    model=runtime[0]['_generate_plan_with_gpt']
    def malformed(*args):
        result=model(*args);result['recipes'][3][field]=value;return result
    runtime[0]['_generate_plan_with_gpt']=malformed;invoke(runtime)
    assert get(db,'shared_meal_plan')['plan'] is None and get(db,'meal_plan_generation_jobs')['state']=='pending'

def test_changed_household_cancels_without_writing_old_recipients(db,runtime):
    model=runtime[0]['_generate_plan_with_gpt']
    def moved(*args):runtime[1]['household']='other';return model(*args)
    runtime[0]['_generate_plan_with_gpt']=moved;invoke(runtime)
    assert get(db,'meal_plan_generation_jobs')['state']=='cancelled'
    assert get(db,'shared_meal_plan')['plan'] is None


def test_dietary_repair_only_replaces_rejected_slot_and_revalidates(db,runtime):
    scope,state=runtime;model=scope['_generate_plan_with_gpt']
    def unsafe(*args):
        result=model(*args);result['recipes'][4]['steps']=['Add peanut butter.'];return result
    scope['_generate_plan_with_gpt']=unsafe
    scope['_recipe_violates']=lambda r,p:'allergy:peanut(step)' if 'peanut' in str(r.get('steps')) else None
    calls=[]
    def repair(ingredients,focus,prefs,recipes,rejected,deadline):
        calls.append(rejected);return [{'index':4,'recipe':{'title':'Fixed slot','ingredients':['oats'],'steps':['Stir oats.']}}]
    scope['_generate_slot_repairs']=repair;invoke(runtime)
    saved=json.loads(get(db,'shared_meal_plan')['plan'])
    assert calls==[[4]] and saved[4]['recipe']['title']=='Fixed slot' and saved[3]['recipe']['title']=='Meal'

def test_unsafe_repair_is_never_published(db,runtime):
    scope,state=runtime;scope['_recipe_violates']=lambda *a:'allergy'
    scope['_generate_slot_repairs']=lambda ingredients,focus,prefs,recipes,rejected,deadline:[{'index':i,'recipe':recipes[i]} for i in rejected]
    invoke(runtime)
    assert get(db,'shared_meal_plan')['plan'] is None
    assert get(db,'meal_plan_generation_jobs')['state']=='pending'

def test_active_admission_preserves_latest_focus_without_second_dispatch(db,runtime):
    spec=importlib.util.spec_from_file_location('admission',ROOT.parent/'meal_plan_api/generation_claim.py')
    admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)
    with db.cursor() as c:c.execute("UPDATE shared_meal_plan SET _updatedDate=NOW()")
    calls=[]
    assert admission.claim_generation(db,['a'],'new focus',300,lambda:calls.append(1)) is False
    assert calls==[] and get(db,'shared_meal_plan')['focus']=='new focus'

def test_rejected_dispatch_restores_previous_ready_plan_and_focus(db,runtime):
    spec=importlib.util.spec_from_file_location('admission',ROOT.parent/'meal_plan_api/generation_claim.py')
    admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)
    with db.cursor() as c:c.execute("UPDATE shared_meal_plan SET status='ready',_updatedDate=NOW()")
    def reject():raise admission.DispatchRejected('fixture rejection')
    with pytest.raises(admission.DispatchRejected):admission.claim_generation(db,['a'],'different',300,reject)
    assert get(db,'shared_meal_plan')['status']=='ready' and get(db,'shared_meal_plan')['focus']=='fresh'


def test_retry_retains_original_verified_consent_actor(db,runtime):
    runtime[0]['_generate_plan_with_gpt']=lambda *a:(_ for _ in ()).throw(TimeoutError())
    invoke(runtime)
    payload=json.loads(get(db,'meal_plan_generation_jobs')['payload'])
    assert payload['user_id']=='requesting-member' and payload['owner']=='a'
