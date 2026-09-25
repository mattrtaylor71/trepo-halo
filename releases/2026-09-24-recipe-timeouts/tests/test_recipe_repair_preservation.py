"""Background extraction cannot erase or supersede the user's saved recipe."""
import json
import pytest
import pymysql
from test_saved_recipe_edit_atomic import recipe, mirrored_recipe

URL = 'https://example.invalid/fixture/acting'

def stored(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM acting_saved_recipes WHERE _id='recipe'")
        value = cur.fetchone()
    if value:
        for field in ('ingredients', 'instructions', 'notes'):
            value[field] = json.loads(value[field])
    return value

def configure(monkeypatch, app, conn, result, during=None):
    # The function owns its connection; preserve this fixture's inspection handle.
    class Connection:
        def __getattr__(self, key): return getattr(conn, key)
        def close(self): pass
    monkeypatch.setattr(app, '_mysql_conn', Connection)
    def extract(*args, **kwargs):
        if during: during()
        return {'content': ''}
    monkeypatch.setattr(app, '_extract_content', extract)
    monkeypatch.setattr(app, '_analyze_extraction', lambda *a, **k: ({'recipe_source_used': 'caption'}, result))

def empty(conn):
    with conn.cursor() as cur:
        cur.execute("UPDATE acting_saved_recipes SET ingredients='[]',instructions='[]',status='repairing'")

def test_empty_repair_never_clears_stored_content(mirrored_recipe, monkeypatch):
    app, conn = mirrored_recipe
    configure(monkeypatch, app, conn, {})
    app._repair_saved_recipe('acting', 'recipe', URL)
    assert stored(conn)['ingredients'] == ['beef']
    assert stored(conn)['instructions'] == ['cook']

def test_more_lines_do_not_authorize_replacing_existing_content(mirrored_recipe, monkeypatch):
    app, conn = mirrored_recipe
    configure(monkeypatch, app, conn, {'ingredients': ['chicken', 'oil'], 'instructions': ['fry', 'serve']})
    app._repair_saved_recipe('acting', 'recipe', URL)
    assert stored(conn)['ingredients'] == ['beef']
    assert stored(conn)['instructions'] == ['cook']

@pytest.mark.parametrize('change', ['manual', 'source', 'delete'])
def test_late_repair_cannot_overwrite_a_newer_decision(mirrored_recipe, monkeypatch, change):
    app, conn = mirrored_recipe
    empty(conn)
    def mutate():
        with conn.cursor() as cur:
            sql = {'manual': "UPDATE acting_saved_recipes SET ingredients='[\"user ingredient\"]',instructions='[\"user step\"]'",
                   'source': "UPDATE acting_saved_recipes SET source_url='https://example.invalid/replaced',resolved_url='https://example.invalid/replaced'",
                   'delete': "DELETE FROM acting_saved_recipes"}[change]
            cur.execute(sql)
    configure(monkeypatch, app, conn, {'ingredients': ['old source'], 'instructions': ['old instructions']}, mutate)
    result = app._repair_saved_recipe('acting', 'recipe', URL)
    assert result.get('repaired') is False
    row = stored(conn)
    if change == 'manual': assert row['ingredients'] == ['user ingredient'] and row['instructions'] == ['user step']
    elif change == 'source': assert row['ingredients'] == [] and row['source_url'].endswith('/replaced')
    else: assert row is None

@pytest.mark.parametrize('result', [{}, {'ingredients': ['beef']}, {'instructions': ['cook']}])
def test_partial_extraction_is_never_ready(mirrored_recipe, monkeypatch, result):
    app, conn = mirrored_recipe
    empty(conn)
    configure(monkeypatch, app, conn, result)
    app._repair_saved_recipe('acting', 'recipe', URL)
    assert stored(conn)['status'] != 'ready'

@pytest.mark.parametrize('shared', [False, True])
def test_full_empty_recipe_recovery_is_atomic_and_repeat_safe(mirrored_recipe, monkeypatch, shared):
    app, conn = mirrored_recipe
    monkeypatch.setenv('READ_SHARED_SAVED_RECIPES', 'true' if shared else 'false')
    empty(conn)
    configure(monkeypatch, app, conn, {'title': 'Beef', 'ingredients': ['beef'], 'instructions': ['cook']})
    assert app._repair_saved_recipe('acting', 'recipe', URL)['repaired'] is True
    first = stored(conn)
    assert first['status'] == 'ready'
    assert app._repair_saved_recipe('acting', 'recipe', URL)['repaired'] is False
    assert stored(conn) == first
    with conn.cursor() as cur:
        cur.execute("SELECT ingredients,instructions,status FROM shared_saved_recipes WHERE owner_id='acting'")
        shared = cur.fetchone()
        assert json.loads(shared['ingredients']) == first['ingredients']
        assert shared['status'] == 'ready'

def test_required_projection_failure_rolls_back_repair(mirrored_recipe, monkeypatch):
    app, conn = mirrored_recipe
    empty(conn)
    before = stored(conn)
    with conn.cursor() as cur:
        cur.execute("CREATE TRIGGER reject_repair BEFORE INSERT ON shared_saved_recipes FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='fixture projection rejection'")
    configure(monkeypatch, app, conn, {'ingredients': ['beef'], 'instructions': ['cook']})
    with pytest.raises(pymysql.err.OperationalError, match='projection rejection'):
        app._repair_saved_recipe('acting', 'recipe', URL)
    assert stored(conn) == before

def test_image_and_category_updates_do_not_cancel_content_recovery(mirrored_recipe, monkeypatch):
    app, conn = mirrored_recipe
    empty(conn)
    def enrich():
        with conn.cursor() as cur:
            cur.execute("UPDATE acting_saved_recipes SET meal_category='dinner',image_url='https://example.invalid/photo',_updatedDate=DATE_ADD(NOW(),INTERVAL 1 SECOND)")
    configure(monkeypatch, app, conn, {'ingredients':['beef'], 'instructions':['cook']}, enrich)
    assert app._repair_saved_recipe('acting', 'recipe', URL)['repaired'] is True
    assert stored(conn)['meal_category'] == 'dinner'
    assert stored(conn)['image_url'] == 'https://example.invalid/photo'

def test_success_removes_only_the_generated_processing_note(mirrored_recipe, monkeypatch):
    app, conn = mirrored_recipe
    empty(conn)
    with conn.cursor() as cur:
        cur.execute("UPDATE acting_saved_recipes SET notes=%s", [json.dumps([
            "We couldn't read the full recipe from this post yet — we're still working on it.",
            'Use my cast iron pan.'])])
    configure(monkeypatch, app, conn, {'ingredients':['beef'], 'instructions':['cook']})
    app._repair_saved_recipe('acting', 'recipe', URL)
    assert stored(conn)['notes'] == ['Use my cast iron pan.']

def test_provider_failure_becomes_retained_link_without_changing_content(mirrored_recipe,monkeypatch):
    app,conn=mirrored_recipe;empty(conn)
    configure(monkeypatch,app,conn,{})
    def fail(*a,**k):raise TimeoutError('fixture provider timeout')
    monkeypatch.setattr(app,'_extract_content',fail)
    result=app._repair_saved_recipe('acting','recipe',URL)
    assert result['outcome']=='link_retained'
    assert stored(conn)['status']=='failed'
    assert stored(conn)['ingredients']==[]


@pytest.mark.parametrize('existing',[False,True])
def test_placeholder_or_url_lists_are_not_a_complete_recipe(mirrored_recipe,monkeypatch,existing):
    app,conn=mirrored_recipe;empty(conn)
    invalid={'ingredients':['https://example.invalid/recipe'],'instructions':['Steps']}
    if existing:
        with conn.cursor() as cur:
            cur.execute('UPDATE acting_saved_recipes SET ingredients=%s,instructions=%s,status=\'ready\'',[json.dumps(invalid['ingredients']),json.dumps(invalid['instructions'])])
    configure(monkeypatch,app,conn,invalid)
    result=app._repair_saved_recipe('acting','recipe',URL)
    assert result['outcome']=='link_retained'
    assert stored(conn)['status']=='failed'


@pytest.mark.parametrize('accepted',[True,False])
def test_failed_dispatch_never_leaves_unscheduled_processing(mirrored_recipe,monkeypatch,accepted):
    app,conn=mirrored_recipe;empty(conn)
    monkeypatch.setattr(app,'_invoke_saved_recipe_repair_async',lambda *a,**k: accepted)
    assert app._enqueue_saved_recipe_repair_or_finish(conn,'acting','recipe',URL)==accepted
    assert stored(conn)['status']==('repairing' if accepted else 'failed')
    if not accepted:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM shared_saved_recipes WHERE owner_id='acting'")
            assert cur.fetchone()['status']=='failed'


@pytest.mark.parametrize('change',['edit','delete','source'])
def test_failed_dispatch_cannot_replace_a_concurrent_decision(mirrored_recipe,monkeypatch,change):
    app,conn=mirrored_recipe;empty(conn)
    def dispatch(*a,**k):
        with conn.cursor() as cur:
            cur.execute({'edit':"UPDATE acting_saved_recipes SET title='My edit',status='ready'",
                         'delete':'DELETE FROM acting_saved_recipes',
                         'source':"UPDATE acting_saved_recipes SET source_url='https://example.invalid/new'"}[change])
        return False
    monkeypatch.setattr(app,'_invoke_saved_recipe_repair_async',dispatch)
    app._enqueue_saved_recipe_repair_or_finish(conn,'acting','recipe',URL)
    row=stored(conn)
    if change=='delete':assert row is None
    elif change=='edit':assert row['status']=='ready' and row['title']=='My edit'
    else:assert row['status']=='repairing' and row['source_url'].endswith('/new')
