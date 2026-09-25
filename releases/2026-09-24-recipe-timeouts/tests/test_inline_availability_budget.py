"""Exercise the actual matching library, including its executor and provider boundary."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import app
import recipe_work_budget as budget

matching=app.recipe_inventory_llm


def response_for(kwargs):
    payload=json.loads(kwargs['messages'][-1]['content'])
    row=payload['recipes'][0]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'r':{row['recipe_id']:{'m':[['m'] for _ in row['ingredients']]}}})))])


def client(create):return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize('slow_stage,enabled',[('matching',True),('substitution',True),('substitution',False)])
def test_actual_library_deadline_has_no_worker_or_provider_continuation(slow_stage,enabled):
    events=[]
    def slow():
        events.append(('started',threading.current_thread() is threading.main_thread(),budget.remaining()))
        time.sleep(.3)
        events.append(('finished',))
    def create(**kwargs):
        if slow_stage=='matching':slow()
        return response_for(kwargs)
    def substitute(*args):
        if slow_stage=='substitution':slow()
        return {}
    with patch.object(matching,'llm_inventory_matching_enabled',return_value=enabled),patch.object(matching,'_openai_client',return_value=client(create)),patch.object(app,'_suggest_saved_recipe_substitutions',side_effect=substitute),patch.object(app,'_log_event'):
        started=time.monotonic()
        with budget.invocation(SimpleNamespace(get_remaining_time_in_millis=lambda:15040),False):
            result=app._compute_recipe_availability(['1 cup oats'],{'kitchen_candidates':[]})
        elapsed=time.monotonic()-started
    assert elapsed<.16, 'Executor cleanup must not wait for the blocked provider'
    assert result['missing_count']==1 and not result['can_make_exact'] and not result['can_make_with_subs']
    assert len(events)==1 and events[0][1] is True and events[0][2] is not None
    assert not any(t.name.startswith('ThreadPoolExecutor') and t.is_alive() for t in threading.enumerate())
    time.sleep(.32)
    assert len(events)==1,'No work may complete later after the deterministic response'


def test_inline_and_existing_single_matching_have_identical_results_and_prompts():
    create=Mock(side_effect=lambda **kwargs:response_for(kwargs))
    recipe={'id':'r1','title':'Porridge','ingredients':['1 cup oats','1 tablespoon peanut butter']}
    context={'kitchen_version':3,'kitchen_candidates':[]}
    callback=Mock(return_value={'can_make_with_subs':False,'substitution_candidates':[],'substitution_summary':None,'substitution_status':'none'})
    with patch.object(matching,'llm_inventory_matching_enabled',return_value=True),patch.object(matching,'_openai_client',return_value=client(create)):
        default,default_meta=matching.match_recipes_fast([recipe],context,substitution_callback=callback)
        inline,inline_meta=matching.match_recipes_fast([recipe],context,substitution_callback=callback,inline_single=True)
    assert inline==default and callback.call_count==2
    assert create.call_args_list[0]==create.call_args_list[1]
    for meta in [default_meta,inline_meta]:assert meta['used_llm'] and not meta['used_fallback'] and meta['recipe_count']==1


@pytest.mark.parametrize('count',[1,2])
def test_default_batch_callers_retain_executor_execution(count):
    barrier=threading.Barrier(count)
    threads=[]
    def create(**kwargs):
        threads.append(threading.current_thread())
        barrier.wait(timeout=1)
        return response_for(kwargs)
    recipes=[{'id':'r'+str(i),'ingredients':['1 cup oats']} for i in range(count)]
    with patch.object(matching,'llm_inventory_matching_enabled',return_value=True),patch.object(matching,'_openai_client',return_value=client(create)):
        result,meta=matching.match_recipes_fast(recipes,{'kitchen_candidates':[]})
    assert len(result)==count and meta['used_llm'] and not meta['used_fallback']
    assert len(threads)==count and all(t is not threading.main_thread() for t in threads)
    if count==2:assert len({t.ident for t in threads})==2


def test_inline_rejects_multiple_recipes_before_any_provider():
    provider=Mock()
    with patch.object(matching,'_openai_client',provider):
        with pytest.raises(ValueError,match='at most one'):
            matching.match_recipes_fast([{'id':'r1'},{'id':'r2'}],{},inline_single=True)
    provider.assert_not_called()


def test_inline_provider_failure_retains_existing_deterministic_contract():
    create=Mock(side_effect=RuntimeError('synthetic provider failure'))
    recipe={'id':'r1','ingredients':['1 cup oats']};context={'kitchen_candidates':[]}
    with patch.object(matching,'llm_inventory_matching_enabled',return_value=True),patch.object(matching,'_openai_client',return_value=client(create)):
        result,meta=matching.match_recipes_fast([recipe],context,inline_single=True)
    assert result['r1']==matching.deterministic_availability(recipe,context)
    assert meta['used_fallback'] and not meta['used_llm'] and meta['error']=='all_compact_calls_failed'
    assert create.call_count==1
