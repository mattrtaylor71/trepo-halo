import copy
import os
import json
from pathlib import Path
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
import pymysql
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recipe_inventory_layer/python'))
import quantity_confirmation as q


@pytest.fixture
def db():
    name = 'confirm_fixture_' + uuid.uuid4().hex
    root = pymysql.connect(host='127.0.0.1', port=int(os.environ.get('TREPO_AMOUNT_TEST_PORT','33316')), connect_timeout=5, read_timeout=15, write_timeout=15, user='root', password='', autocommit=True)
    with root.cursor() as cur:
        cur.execute('CREATE DATABASE ' + name)
    def connect():
        return pymysql.connect(host='127.0.0.1', port=int(os.environ.get('TREPO_AMOUNT_TEST_PORT','33316')), connect_timeout=5, read_timeout=15, write_timeout=15, user='root', password='', database=name,
                               autocommit=True, cursorclass=pymysql.cursors.DictCursor)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute('CREATE TABLE owner_kitchen_state (owner VARCHAR(64) PRIMARY KEY, kitchen_version BIGINT) ENGINE=InnoDB')
        columns = []
        for field in q.CACHE_COLUMNS:
            kind = 'BIGINT' if field in ('kitchen_version','can_make_exact','can_make_with_subs','matched_count','missing_count') else (
                'VARCHAR(255)' if field in ('owner','recipe_source','recipe_id') else 'TEXT')
            columns.append('`' + field + '` ' + kind)
        cur.execute('CREATE TABLE owner_recipe_availability (' + ','.join(columns) + ', PRIMARY KEY(owner,recipe_source,recipe_id)) ENGINE=InnoDB')
        cur.execute("INSERT INTO owner_kitchen_state VALUES ('owner',7),('other',7)")
    try:
        yield conn, connect
    finally:
        # The JSON-lines bridge closes this generator directly. Cleanup must run
        # on GeneratorExit as well as normal pytest teardown.
        try:
            conn.close()
            with root.cursor() as cur:
                cur.execute('DROP DATABASE ' + name)
        finally:
            root.close()


def record():
    matches = [
        {'recipe_ingredient':'salt to taste','match_status':'have','quantity_status':'unknown',
         'matched_kitchen_items':[{'item_id':'salt','display_name':'Salt'}]},
        {'recipe_ingredient':'1 lb beef','match_status':'have','quantity_status':'sufficient',
         'matched_kitchen_items':[{'item_id':'beef','display_name':'Beef'}]},
    ]
    return {'owner':'owner','recipe_source':'saved:q1','recipe_id':'recipe','kitchen_version':7,
            'can_make_exact':0,'can_make_with_subs':0,'matched_count':2,'missing_count':0,
            'ingredient_matches':json.dumps(matches),'missing_ingredients':'[]','substitution_candidates':'[]',
            'substitution_summary':None,'substitution_status':'none','analysis_status':'ready'}


def seed(db, row=None):
    row = row or record()
    q.persist_cache_rows(db[0], [row])
    return {'operation_id':str(uuid.uuid4()),'kitchen_version':7,'ingredient_indices':[0],
            'availability_token':q.availability_token({**row,'ingredient_matches':json.loads(row['ingredient_matches'])})}


def confirm(db, request, owner='owner'):
    return q.confirm_cached_availability(db[0], owner, 'saved', 'recipe', request)


def test_confirmation_and_same_operation_replay_are_durable(db):
    request = seed(db)
    first = confirm(db, request)
    assert first['can_make_exact'] is True
    assert first['ingredient_matches'][0]['quantity_status'] == 'confirmed_enough'
    assert confirm(db, request) == first
    other = db[1]()
    try:
        assert q.confirm_cached_availability(other, 'owner', 'saved', 'recipe', request) == first
    finally:
        other.close()


def test_old_background_refresh_preserves_confirmation(db):
    request = seed(db)
    confirm(db, request)
    q.persist_cache_rows(db[0], [record()])
    assert confirm(db, request)['can_make_exact'] is True


def test_partial_confirmations_survive_a_duplicate_refresh_together(db):
    incoming = record(); matches = json.loads(incoming['ingredient_matches'])
    matches[1]['quantity_status'] = 'unknown'; incoming['ingredient_matches'] = json.dumps(matches)
    first_request = seed(db,incoming); first = confirm(db,first_request)
    assert not first['can_make_exact']
    second_request = {**first_request,'operation_id':str(uuid.uuid4()),'ingredient_indices':[1],
                      'availability_token':q.availability_token(first)}
    assert confirm(db,second_request)['can_make_exact']
    q.persist_cache_rows(db[0],[incoming])
    assert confirm(db,first_request)['can_make_exact']
    assert confirm(db,second_request)['can_make_exact']


def test_kitchen_change_expires_confirmation(db):
    request = seed(db)
    confirm(db, request)
    with db[0].cursor() as cur:
        cur.execute("UPDATE owner_kitchen_state SET kitchen_version=8 WHERE owner='owner'")
    with pytest.raises(q.ConfirmationRejected, match='KITCHEN_CHANGED'):
        confirm(db, request)
    incoming = record(); incoming['kitchen_version'] = 8
    q.persist_cache_rows(db[0], [incoming])
    with db[0].cursor() as cur:
        cur.execute("SELECT * FROM owner_recipe_availability WHERE owner='owner'")
        row = cur.fetchone()
    assert not row['can_make_exact']
    assert json.loads(row['ingredient_matches'])[0]['quantity_status'] == 'unknown'
    q.persist_cache_rows(db[0], [record()])
    with db[0].cursor() as cur:
        cur.execute("SELECT kitchen_version FROM owner_recipe_availability WHERE owner='owner'")
        assert cur.fetchone()['kitchen_version'] == 8


def test_changed_recipe_or_match_invalidates_confirmation(db):
    request = seed(db); confirm(db, request)
    incoming = record(); matches = json.loads(incoming['ingredient_matches'])
    matches[0]['recipe_ingredient'] = 'salt for the entire batch'
    incoming['ingredient_matches'] = json.dumps(matches)
    q.persist_cache_rows(db[0], [incoming])
    with pytest.raises(q.ConfirmationRejected, match='AVAILABILITY_CHANGED'):
        confirm(db, request)


def test_known_shortage_cannot_be_overridden(db):
    row = record(); matches = json.loads(row['ingredient_matches'])
    matches[0].update(match_status='missing',quantity_status='insufficient')
    row['ingredient_matches'] = json.dumps(matches)
    request = seed(db,row)
    with pytest.raises(q.ConfirmationRejected, match='INGREDIENT_NOT_CONFIRMABLE'):
        confirm(db, request)


def test_mixed_valid_and_invalid_approval_rolls_back_entire_request(db):
    incoming = record(); matches = json.loads(incoming['ingredient_matches'])
    matches[1].update(match_status='missing',quantity_status='insufficient')
    incoming['ingredient_matches'] = json.dumps(matches)
    request = seed(db,incoming); request['ingredient_indices'] = [0,1]
    with pytest.raises(q.ConfirmationRejected,match='INGREDIENT_NOT_CONFIRMABLE'):
        confirm(db,request)
    with db[0].cursor() as cur:
        cur.execute('SELECT ingredient_matches FROM owner_recipe_availability')
        assert json.loads(cur.fetchone()['ingredient_matches'])[0]['quantity_status'] == 'unknown'


def test_owner_and_cache_source_boundaries(db):
    request = seed(db)
    with pytest.raises(q.ConfirmationRejected, match='REFRESH_AVAILABILITY'):
        confirm(db, request, owner='other')
    with pytest.raises(q.ConfirmationRejected, match='REFRESH_AVAILABILITY'):
        q.confirm_cached_availability(db[0],'owner','explore','recipe',request)


def test_reused_operation_cannot_broaden_approval(db):
    request = seed(db); confirm(db, request)
    with pytest.raises(q.ConfirmationRejected, match='OPERATION_REUSED'):
        confirm(db, {**request,'ingredient_indices':[0,1]})


def test_corrupt_cache_is_rejected_without_acknowledging_or_committing(db):
    request = seed(db)
    with db[0].cursor() as cur:
        cur.execute("UPDATE owner_recipe_availability SET missing_ingredients='invalid json'")
    with pytest.raises(q.ConfirmationRejected, match='REFRESH_AVAILABILITY'):
        confirm(db, request)
    with db[0].cursor() as cur:
        cur.execute('SELECT can_make_exact,ingredient_matches FROM owner_recipe_availability')
        row = cur.fetchone()
    assert not row['can_make_exact']
    assert json.loads(row['ingredient_matches'])[0]['quantity_status'] == 'unknown'


@pytest.mark.parametrize('change',[
    {'kitchen_version':True}, {'kitchen_version':-1}, {'ingredient_indices':[]},
    {'ingredient_indices':[True]}, {'ingredient_indices':[0,0]},
    {'availability_token':'wrong'}, {'operation_id':'not-a-uuid'},
])
def test_malformed_requests_rejected_before_writes(db, change):
    request = seed(db)
    with pytest.raises(q.ConfirmationRejected) as error:
        confirm(db, {**request,**change})
    assert error.value.status == 400


def test_concurrent_identical_confirmations_apply_once(db):
    request = seed(db)
    def run(_):
        conn = db[1]()
        try:
            return q.confirm_cached_availability(conn,'owner','saved','recipe',request)
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(run, range(4)))
    assert all(result == responses[0] for result in responses)


def test_confirmation_auth_is_strict_while_older_routes_stay_in_observe_mode(monkeypatch):
    import base64
    import hashlib
    import hmac
    import importlib.util
    import time
    path = Path(__file__).resolve().parents[1] / 'saved_recipes_api/trepo_auth.py'
    spec = importlib.util.spec_from_file_location('confirmation_auth_fixture', path)
    auth = importlib.util.module_from_spec(spec); spec.loader.exec_module(auth)
    auth._SECRET = 'fixture-only'; auth._ENFORCE = False
    monkeypatch.setitem(sys.modules, 'trepo_auth', auth)
    def encoded(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip('=')
    def signed(user):
        message = encoded({'alg':'HS256'}) + '.' + encoded({'iss':'trepo-auth','exp':time.time()+300,'user_id':user,'owner_id':'household'})
        return message + '.' + base64.urlsafe_b64encode(hmac.new(b'fixture-only',message.encode(),hashlib.sha256).digest()).decode().rstrip('=')
    assert auth.require_owner({'headers':{},'pathParameters':{'owner':'owner'}}) is None
    q.require_authenticated_owner({'headers':{'Authorization':'Bearer '+signed('owner')}}, 'owner')
    for event in [None, {'headers':[]}, {'headers':{}}, {'headers':{'Authorization':'Bearer '+signed('other')}},
                  {'headers':{'Authorization':'Bearer '+encoded({'user_id':'owner'})}}]:
        with pytest.raises(q.ConfirmationRejected) as error:
            q.require_authenticated_owner(event, 'owner')
        assert error.value.status == 403


def test_actual_saved_route_rejects_unsigned_confirmation_before_database_access(monkeypatch):
    import ast
    import types
    source_path = Path(__file__).resolve().parents[1] / 'saved_recipes_api/app.py'
    source = source_path.read_text()
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_confirm_recipe_amounts']
    assert len(selected) == 1
    def forbidden_database_access():
        raise AssertionError('Unauthenticated request reached database')
    namespace = {'_mysql_conn':forbidden_database_access,'_error':lambda status,code:{'status':status,'code':code}}
    monkeypatch.setitem(sys.modules,'trepo_auth',types.SimpleNamespace(_get_header=lambda *args:None,_verify_jwt=lambda *args:None))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source_path),'exec'),namespace)
    result = namespace['_confirm_recipe_amounts']('owner',{'recipe_source':'saved','recipe_id':'recipe'},{'headers':{}})
    assert result == {'status':403,'code':'SIGN_IN_REQUIRED'}


def test_saved_and_explore_writers_use_confirmation_preserving_storage(db):
    import ast
    import types
    root = Path(__file__).resolve().parents[2]
    for path, function_name, transform in [
        (root/'playground6_groceryRec/saved_recipes_api/app.py','_persist_owner_recipe_availability_rows',lambda row:row),
        (root/'playground_explore/explore_recipes_api/app.py','_persist_overlay_rows',lambda row:tuple(row[k] for k in q.CACHE_COLUMNS)),
    ]:
        request = seed(db); confirm(db,request)
        source = path.read_text()
        selected = [node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name==function_name]
        assert len(selected)==1
        namespace = {'_ensure_owner_recipe_availability_table':lambda conn:None,
                     '_reservation_state_for_owner':lambda *args,**kwargs:None,  # Existing disabled reservation contract.
                     'recipe_inventory_llm':types.SimpleNamespace(persist_quantity_availability_rows=q.persist_cache_rows)}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
        namespace[function_name](db[0],[transform(record())])
        assert confirm(db,request)['can_make_exact'] is True
        # Independent setup for the next function.
        with db[0].cursor() as cur:
            cur.execute('DELETE FROM owner_recipe_availability')


@pytest.mark.parametrize('source,function_name,source_path', [
    ('saved','_serialize_saved_recipe_with_availability','playground6_groceryRec/saved_recipes_api/app.py'),
    ('explore','_serialize_recipe_with_owner_context','playground_explore/explore_recipes_api/app.py'),
])
def test_recipe_reopen_response_preserves_confirmed_amount(db, source, function_name, source_path):
    import ast
    import recipe_inventory_llm as matching
    row = record(); row['recipe_source'] = source + ':q1'
    request = seed(db, row)
    q.confirm_cached_availability(db[0], 'owner', source, 'recipe', request)
    availability = {**row, 'ingredient_matches':json.loads(row['ingredient_matches']),
                    'missing_ingredients':[], 'substitution_candidates':[]}
    path = Path(__file__).resolve().parents[2] / source_path
    names = {function_name, '_availability_summary'}
    selected = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace = {'recipe_inventory_llm':matching, '_serialize_row':lambda row:dict(row),
                 '_build_owner_recipe_availability_record':lambda *args:{}, '_build_overlay_row':lambda *args:()}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    if source == 'saved':
        serialized, _ = namespace[function_name](db[0], 'owner', {'id':'recipe'}, kitchen_context={'fixture':True}, availability=availability)
    else:
        serialized, _ = namespace[function_name]({'id':'recipe'}, 'owner', db[0], kitchen_context={'fixture':True}, availability=availability)
    assert serialized['availability']['can_make_exact'] is True
    assert serialized['ingredient_matches'][0]['quantity_status'] == 'confirmed_enough'


@pytest.mark.parametrize('change', ['version','ingredients','owner','source'])
def test_response_confirmation_never_crosses_changed_evidence(db, change):
    request = seed(db); confirm(db, request)
    incoming = record(); incoming['ingredient_matches'] = json.loads(incoming['ingredient_matches'])
    owner, source = 'owner', 'saved'
    if change == 'version': incoming['kitchen_version'] = 8
    if change == 'ingredients': incoming['ingredient_matches'][0]['recipe_ingredient'] = '2 tsp salt'
    if change == 'owner': owner = 'other'
    if change == 'source': source = 'explore'
    result = q.confirmed_availability_for_response(db[0],owner,source,'recipe',incoming)
    assert not result['can_make_exact']
    assert result['ingredient_matches'][0]['quantity_status'] == 'unknown'


def test_explore_batch_response_keeps_legacy_fields_and_quantity_guidance():
    import ast
    import recipe_inventory_llm as matching
    path = Path(__file__).resolve().parents[2]/'playground_explore/explore_recipes_api/app.py'
    selected = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('personalize_recipes','_availability_summary','_persist_response_availability')]
    class Connection:
        def cursor(self): return self
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def execute(self,sql,params): pass
        def fetchall(self): return [{'id':'recipe'}]
        def fetchone(self): return None
        def close(self): pass
    row = record(); row['ingredient_matches'] = json.loads(row['ingredient_matches'])
    namespace = {'recipe_inventory_llm':matching, '_safe_owner':str, '_mysql_conn':Connection,
                 '_ensure_personalization_tables':lambda conn:None, '_build_kitchen_context':lambda *args:{},
                 '_serialize_row':lambda row:row, '_compute_explore_recipe_availability_batch':lambda *args:{'recipe':row},
                 '_build_overlay_row':lambda *args:(), '_persist_overlay_rows':lambda *args:None,
                 '_ok':lambda value:value, '_err':lambda *args:None}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    summary = namespace['personalize_recipes']({'owner':'owner','recipe_ids':['recipe']})['results']['recipe']
    assert not summary['can_make_exact']
    assert summary['quantity_unknown_ingredients'] == ['salt to taste']
    assert summary['quantity_confirmation_items'] == [{'index':0,'ingredient':'salt to taste'}]
    assert {'can_make_exact','can_make_with_subs','matched_count','missing_count'} <= set(summary)
