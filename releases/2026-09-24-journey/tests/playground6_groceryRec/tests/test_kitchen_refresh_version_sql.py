"""Execute the actual refresh helper against isolated MySQL, without AWS imports."""
import ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from test_quantity_confirmation import db


@pytest.fixture
def refresh(db):
    source = Path(__file__).resolve().parents[1] / 'kitchen_api/app.py'
    names = {'_sanitize_user_id', '_ensure_owner_kitchen_state_table',
             '_mark_recipe_refresh_needed_for_owners'}
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {'_OWNER_KITCHEN_STATE_TABLE': 'owner_kitchen_state'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    with db[0].cursor() as cur:
        cur.execute('DROP TABLE owner_kitchen_state')
    return namespace['_mark_recipe_refresh_needed_for_owners']


def rows(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT owner,kitchen_version,last_recipe_refresh_requested_version,recipe_refresh_needed '
                    'FROM owner_kitchen_state ORDER BY owner')
        return cur.fetchall()


def expected(owner, version):
    return dict(owner=owner, kitchen_version=version,
                last_recipe_refresh_requested_version=version, recipe_refresh_needed=1)


def test_existing_owner_requests_exactly_the_new_version(db, refresh):
    refresh(db[0], ['owner'])
    refresh(db[0], ['owner'])
    assert rows(db[0]) == [expected('owner', 2)]


@pytest.mark.parametrize('owners', [['owner', 'owner'], ['owner', 'own!er'], ['owner', '', None, 'owner']])
def test_same_normalized_owner_advances_once_per_call(db, refresh, owners):
    refresh(db[0], owners)
    assert rows(db[0]) == [expected('owner', 1)]


def test_household_refresh_leaves_unrelated_owner_untouched(db, refresh):
    refresh(db[0], ['unrelated'])
    refresh(db[0], ['owner', 'member'])
    refresh(db[0], ['owner', 'member'])
    assert rows(db[0]) == [expected('member', 2), expected('owner', 2), expected('unrelated', 1)]


def test_concurrent_requests_are_not_lost(db, refresh):
    refresh(db[0], ['owner'])
    def invoke(_):
        conn = db[1]()
        try:
            refresh(conn, ['owner'])
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(invoke, range(12)))
    assert rows(db[0]) == [expected('owner', 13)]


def test_empty_input_performs_no_schema_or_write(db, refresh):
    refresh(db[0], [None, '', '!'])
    with db[0].cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'owner_kitchen_state'")
        assert cur.fetchone() is None
