"""Run exact serving helpers with IO traps; no credentials/model/DB writes."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
import pytest,pymysql

ROOT=Path(__file__).resolve().parents[1]
def load(path,name,scope):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

@pytest.fixture(params=['saved_recipe_status','saved_recipes_api'])
def runtime(request):
    calls=[]
    scope={'pymysql':pymysql,'_safe_text':lambda x:str(x or ''),'_JOB_STATUS_COMPLETED':'COMPLETED'}
    forbidden=['_ensure_recipe_personalization_tables','_build_kitchen_match_context','_ensure_owned_saved_recipe_image',
       '_compute_saved_recipe_availability_batch','_serialize_saved_recipe_with_availability',
       '_persist_owner_recipe_availability_rows','_ensure_saved_recipes_table']
    for name in forbidden:scope[name]=Mock(side_effect=AssertionError('Forbidden poll side effect: '+name))
    rows={'a':{'id':'a','title':'Current title','status':'ready'},'failed':{'id':'failed','status':'failed'}}
    scope['_fetch_saved_recipe_by_id']=lambda c,owner,rid:rows.get(rid) if owner=='alice' else None
    scope['_serialize_row']=lambda r:r.copy()
    path=ROOT/request.param/'app.py'
    fetch=load(path,'_fetch_saved_recipe_batch_results',scope)
    scope['_get_saved_recipe_batch_job']=lambda _: {'owner':'alice','status':'COMPLETED','results':[{'recipe_id':'a'}]}
    scope['_mysql_conn']=Mock(return_value=object())
    scope['_error']=lambda status,message:{'statusCode':status}
    scope['_build_saved_recipe_batch_job_response']=lambda *args,**kwargs:kwargs
    status=load(path,'_get_saved_recipe_batch_job_status',scope)
    return fetch,status,rows,scope

def test_completed_job_only_reads_current_rows_and_keeps_result_contract(runtime):
    fetch,_,rows,_=runtime
    recipes,results,errors=fetch(object(),'alice',{'results':[{'recipe_id':'a','deduped':True},{'recipe_id':'deleted'},{'recipe_id':'failed'}],'partial_errors':['existing']})
    assert [r['id'] for r in recipes]==['a','failed']
    assert results[0]['deduped'] is True and results[0]['recipe']['title']=='Current title'
    assert errors==['existing']
    del rows['a']
    assert fetch(object(),'alice',{'results':[{'recipe_id':'a'}]})[0]==[]

def test_status_route_does_not_ensure_schema(runtime):
    _,status,_,scope=runtime
    assert status('alice','job')['recipes'][0]['id']=='a'
    scope['_mysql_conn'].reset_mock()
    assert status('foreign','job')['statusCode']==404
    scope['_mysql_conn'].assert_not_called()

def test_100_concurrent_completed_polls_never_call_enrichment_or_writes(runtime):
    fetch,_,_,scope=runtime
    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes=list(pool.map(lambda _:fetch(object(),'alice',{'results':[{'recipe_id':'a'}]}),range(100)))
    assert all(len(item[0])==1 for item in outcomes)
    for name,value in scope.items():
        if name.startswith(('_ensure_','_compute_','_persist_')):value.assert_not_called()


def test_missing_table_is_absent_but_database_outages_are_not_empty_success(runtime):
    fetch,_,_,scope=runtime
    scope['_fetch_saved_recipe_by_id']=Mock(side_effect=pymysql.err.ProgrammingError(1146,'missing fixture'))
    assert fetch(object(),'alice',{'results':[{'recipe_id':'a'}]})[0]==[]
    scope['_fetch_saved_recipe_by_id']=Mock(side_effect=pymysql.err.OperationalError(2013,'connection lost'))
    with pytest.raises(pymysql.err.OperationalError):fetch(object(),'alice',{'results':[{'recipe_id':'a'}]})
