import ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
PATH=Path(__file__).resolve().parents[1]/'plan_visibility.py'

def runtime(publication=True):
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    prefs={key:[] for key in ('allergies','diets','religious','health','custom')}
    record={'household_id':'home','members_hash':digest(['a','b']),'dietary_hash':digest(prefs)} if publication else None
    cur=Mock();cur.__enter__=Mock(return_value=cur);cur.__exit__=Mock(return_value=False)
    cur.fetchall.return_value=[{'user_id':'a'},{'user_id':'b'}];cur.fetchone.side_effect=lambda: {'plan':[{'recipe':{'title':'Meal','steps':['Stir.']}}]} if 'SELECT plan ' in cur.execute.call_args.args[0] else record
    conn=Mock();conn.cursor.return_value=cur
    brain=SimpleNamespace(membership=lambda *a,**kw:'home',dietary_preferences=lambda *a,**kw:prefs)
    scope=dict(json=json,brain_store=brain,digest=digest,_recipe_violates=lambda r,p:False)
    node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),scope)
    return conn,cur,brain,scope,record,prefs

@pytest.mark.parametrize('changed',['none','household','members','dietary','ingredient','missing'])
def test_saved_plan_visibility_requires_current_membership_and_restrictions(changed):
    conn,cur,brain,scope,record,prefs=runtime(changed!='missing')
    if changed=='household':brain.membership=lambda *a,**kw:'other'
    if changed=='members':cur.fetchall.return_value=[{'user_id':'a'}]
    if changed=='dietary':prefs['allergies']=['peanuts']
    if changed=='ingredient':scope['_recipe_violates']=lambda *a:True
    plan=[{'recipe':{'title':'Meal','steps':['Stir.']}}]
    result=scope['visible_plan'](conn,'a',json.dumps(plan))
    assert result==(plan if changed=='none' else [])
    assert all(str(c.args[0]).startswith('SELECT') for c in cur.execute.call_args_list)


def test_concurrent_publication_does_not_validate_an_older_payload():
    conn,cur,brain,scope,record,prefs=runtime()
    old=[{'recipe':{'title':'Previous household meal','steps':['Stir.']}}]
    assert scope['visible_plan'](conn,'a',old)==[]
    assert not any('meal_plan_publications' in c.args[0] for c in cur.execute.call_args_list)
