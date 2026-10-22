import time
import pytest
from household_brain import store, sources
from .conftest import sql,edit

def capture(db):
    # Capture the production fence fields from real MySQL. Projection inputs
    # outside this regression (external receipts/shelf cache) are not required.
    with db.cursor() as cur:
        state=store.state_for(cur,'h')
        members=sources.members(cur,'h')
        cur.execute("SELECT actor_id,generation,revision FROM family_brain_subjects WHERE household_id='h' AND enabled=1 ORDER BY actor_id")
        states=[[r['actor_id'],r['generation'],r['revision']] for r in cur.fetchall()]
        hard=store.dietary(cur,'h')
    return {'fence':{'schema':'household-brain-v2','actor':'a','household':'h','audience':'a',
        'revision':state['input_revision'],'generation':state['generation'],'states':states,
        'members_hash':store.digest(members),'dietary_hash':store.digest(hard),
        'expires':time.time()+300,'turn_expires':time.time()+600}}

def test_real_dish_trigger_invalidates_context_but_preserves_receipt(db):
    original=capture(db)
    sql(db,"INSERT INTO shared_dishes VALUES('synthetic-meal','a','a','Oatmeal','IN','complete',UTC_TIMESTAMP(6))")
    assert not store.is_current(db,original['fence'])
    assert store.is_action_scope_current(db,original['fence'])
    fresh=capture(db)
    assert store.is_current(db,fresh['fence'])
    # A historical receipt does not recreate a customer-deleted row. The fresh
    # plan must see the actual deletion rather than the old memory observation.
    sql(db,"DELETE FROM shared_dishes WHERE _id='synthetic-meal'")
    assert not store.is_current(db,fresh['fence'])
    assert store.is_action_scope_current(db,original['fence'])
    assert not sql(db,"SELECT * FROM shared_dishes")

@pytest.mark.parametrize('change',['allergy','actor_move','member_move','pause','reset','preference','forget'])
def test_real_scope_changes_revoke_receipts(db,change):
    edit(db,'set',actor_id='a',key='favorite_foods',value=['Beans'])
    original=capture(db)
    if change=='allergy':sql(db,"INSERT INTO user_dietary_preferences VALUES('h',JSON_ARRAY('peanut'),NULL,NULL,NULL,NULL)")
    elif change=='actor_move':sql(db,"UPDATE new_users SET owner_id='other' WHERE user_id='a'")
    elif change=='member_move':sql(db,"UPDATE new_users SET owner_id='other' WHERE user_id='b'")
    elif change=='preference':edit(db,'set',actor_id='a',key='favorite_foods',value=['Rice'])
    elif change=='forget':edit(db,'forget',actor_id='a',key='favorite_foods')
    else:edit(db,change)
    assert not store.is_action_scope_current(db,original['fence'])
