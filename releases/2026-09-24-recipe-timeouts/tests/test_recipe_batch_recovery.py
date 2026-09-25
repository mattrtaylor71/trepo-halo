from unittest.mock import Mock, patch

import pytest
from test_saved_recipes_image_mirroring import load_app_module


def cluster(index, pages=1):
    return {'extraction': {'fixture': index}, 'image_indexes': list(range(index, index + pages)),
            'fragments': [{'fixture': i} for i in range(pages)], 'prepared_images': []}


def test_later_internal_failure_preserves_earlier_saved_recipe():
    app = load_app_module()
    with patch.object(app, '_extract_recipe_clusters_from_images', return_value=([cluster(0), cluster(1)], [])), \
         patch.object(app, '_save_saved_recipe_record', side_effect=[({'recipe': {'id': 'saved'}}, 201), RuntimeError('fixture failure')]) as save:
        results, errors = app._process_saved_recipe_image_batch(Mock(), 'owner', ['a', 'b'])
    assert len(results) == 1 and results[0]['recipe_id'] == 'saved'
    assert errors[0]['image_indexes'] == [1] and errors[0]['status_code'] == 503
    assert save.call_count == 2


def test_transient_multpage_save_failure_retries_cluster_without_splitting():
    app = load_app_module()
    with patch.object(app, '_extract_recipe_clusters_from_images', return_value=([cluster(0, 2)], [])), \
         patch.object(app, '_save_saved_recipe_record', side_effect=app.ServiceError('fixture unavailable', status_code=503)) as save, \
         patch.object(app, '_merge_image_cluster') as split, patch.object(app.time, 'sleep') as sleep:
        with pytest.raises(app.ServiceError) as error:
            app._process_saved_recipe_image_batch(Mock(), 'owner', ['a', 'b'])
    assert save.call_count == 3 and sleep.call_count == 2
    split.assert_not_called()
    assert error.value.extra['partial_errors'][0]['image_indexes'] == [0, 1]


def test_rate_limit_recovery_keeps_exact_cluster_and_returns_one_result():
    app = load_app_module()
    with patch.object(app, '_extract_recipe_clusters_from_images', return_value=([cluster(0)], [])), \
         patch.object(app, '_save_saved_recipe_record', side_effect=[app.ServiceError('rate limited', status_code=429), ({'recipe': {'id': 'saved'}}, 201)]) as save, \
         patch.object(app.time, 'sleep'):
        results, errors = app._process_saved_recipe_image_batch(Mock(), 'owner', ['a'])
    assert len(results) == 1 and not errors
    assert save.call_args_list[0] == save.call_args_list[1]


def test_image_timeout_retains_other_fragments_without_reextracting_successes():
    app = load_app_module()
    calls = []
    def prepare(value):
        calls.append(value)
        if value == 'bad':
            raise TimeoutError('fixture timeout')
        return {'image': value}
    with patch.object(app, '_prepare_image_for_vision', side_effect=prepare), \
         patch.object(app, '_extract_recipe_fragment_from_image', side_effect=lambda image, **kw: image), \
         patch.object(app, '_group_image_fragments', side_effect=lambda fragments, **kw: fragments), \
         patch.object(app, '_merge_image_cluster', side_effect=lambda fragment: fragment), \
         patch.object(app.time, 'sleep'):
        results, errors = app._extract_recipe_clusters_from_images(['first', 'bad', 'last'])
    assert results == [{'image': 'first'}, {'image': 'last'}]
    assert calls == ['first', 'bad', 'bad', 'bad', 'last']
    assert errors[0]['image_index'] == 1 and errors[0]['retryable']


@pytest.mark.parametrize('handler', ['_handle_async_saved_recipe_batch_task', '_handle_async_saved_recipe_url_task'])
def test_completed_redelivery_does_not_repeat_provider_or_persistence(handler):
    app = load_app_module()
    with patch.object(app, '_get_saved_recipe_batch_job', return_value={'owner': 'owner', 'status': 'COMPLETED', 'result_count': 1}), \
         patch.object(app, '_mysql_conn', side_effect=AssertionError('Repeated completed work')):
        result = getattr(app, handler)({'owner': 'owner', 'job_id': 'job'}, 'fixture')
    assert result['already_completed'] and result['count'] == 1


def test_partial_outcome_is_explicit_without_breaking_legacy_terminal_status():
    app = load_app_module()
    result = app._serialize_saved_recipe_batch_job({'status': 'COMPLETED', 'partial_errors': [{'image_index': 1}]})
    assert result['status'] == 'completed' and result['outcome'] == 'partial'


def test_missing_persisted_recipe_id_cannot_be_reported_as_complete():
    app = load_app_module()
    with patch.object(app, '_extract_recipe_clusters_from_images', return_value=([cluster(0)], [])), \
         patch.object(app, '_save_saved_recipe_record', return_value=({'recipe': {}}, 201)):
        with pytest.raises(app.ServiceError):
            app._process_saved_recipe_image_batch(Mock(), 'owner', ['a'])


@pytest.mark.parametrize('status,retryable', [(403, False), (404, False), (410, False), (429, True), (503, True)])
def test_remote_image_status_is_classified_without_leaking_url(status, retryable):
    app = load_app_module()
    response = Mock(status_code=status)
    response.raise_for_status.side_effect = app.requests.HTTPError(
        'https://fixture.invalid/image?token=synthetic-secret', response=response)
    with patch.object(app.requests, 'get', return_value=response):
        with pytest.raises(app.ServiceError) as result:
            app._load_remote_image('https://fixture.invalid/image?token=synthetic-secret')
    assert app._is_transient_failure(result.value) == retryable
    assert 'synthetic-secret' not in str(result.value)
    response.close.assert_called_once()
