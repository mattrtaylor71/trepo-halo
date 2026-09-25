"""Exercise the real row serializer without cloud or database access."""
import ast
import datetime
import json
from decimal import Decimal
from pathlib import Path
import sys
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'kitchen_api'))
from storage_guidance_output import legacy_safe_guidance, legacy_safe_payload
source = ast.parse((root / 'kitchen_api/app.py').read_text())
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {
    'json_serial', '_decode_json_field_if_needed', '_compute_days_old', '_serialize_rows', '_success_response'}]
namespace = {'datetime': datetime.datetime, 'json': json, 'Decimal': Decimal, 'legacy_safe_guidance': legacy_safe_guidance, 'legacy_safe_payload': legacy_safe_payload}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<actual kitchen serializer>', 'exec'), namespace)
serialize = namespace['_serialize_rows']

def test_unexpected_guidance_failure_cannot_restore_unsafe_raw_output(monkeypatch):
    def failure(value):
        raise ValueError('controlled guard failure')
    monkeypatch.setitem(namespace, 'legacy_safe_guidance', failure)
    result = serialize([{'storage_guidance': {'max_days': 1e300}, 'product_name': 'fixture'}])[0]
    assert result['storage_guidance'] is None
    assert result['product_name'] == 'fixture'

@pytest.mark.parametrize('bad', [True, False, 2.5, -1, 0, 1e300, float('inf'), float('nan'), 'bad', '9223372036854775808'])
@pytest.mark.parametrize('encoded', [False, True])
def test_invalid_maximum_does_not_reach_old_clients(bad, encoded):
    guidance = {'summary': 'fixture', 'min_days': 1, 'max_days': bad}
    value = json.dumps(guidance) if encoded else guidance
    row = {'_id': 'fixture', 'storage_guidance': value, 'product_expiration': '2026-09-20'}
    result = serialize([row])[0]
    assert result['storage_guidance'] is None
    assert result['product_expiration'] == '2026-09-20'
    assert row['storage_guidance'] == value  # serialization never repairs stored data

@pytest.mark.parametrize('guidance', [
    {'summary': 'fixture', 'min_days': -1, 'max_days': 7},
    {'summary': 'fixture', 'min_days': 8, 'max_days': 7},
    {'summary': 'fixture', 'min_days': True, 'max_days': 7},
    {'summary': 'fixture', 'max_days': 30, 'scenarios': {'opened_fridge': {'max_days': 1e300}}},
    {'summary': 'fixture', 'max_days': 30, 'scenarios': {'opened_fridge': {'min_days': 4, 'max_days': 2}}},
])
def test_invalid_range_or_scenario_cannot_fall_back_to_general(guidance):
    assert serialize([{'storage_guidance': json.dumps(guidance)}])[0]['storage_guidance'] is None

@pytest.mark.parametrize('guidance', [
    {'summary': 'fixture', 'min_days': 0, 'max_days': 7},
    {'summary': 'fixture', 'min_days': 3, 'max_days': 3.0},
    {'summary': 'fixture', 'minDays': '2', 'maxDays': '7'},
    {'summary': 'fixture', 'max_days': None},
    {'summary': 'fixture', 'max_days': 30, 'scenarios': {'opened_fridge': {'max_days': 2}}},
])
@pytest.mark.parametrize('encoded', [False, True])
def test_valid_guidance_keeps_legacy_representation(guidance, encoded):
    value = json.dumps(guidance) if encoded else guidance
    result = serialize([{'storage_guidance': value, 'product_name': 'fixture'}])[0]
    assert result['storage_guidance'] == value
    assert result['product_name'] == 'fixture'

@pytest.mark.parametrize('wrapper', ['item', 'items', 'created'])
def test_mutation_response_boundary_blocks_raw_guidance(wrapper):
    row = {'_id': 'fixture', 'storage_guidance': {'summary': 'fixture', 'max_days': 1e300}}
    payload = {wrapper: row if wrapper == 'item' else [row]}
    result = namespace['_success_response'](payload)
    body = json.loads(result['body'])
    output = body[wrapper] if wrapper == 'item' else body[wrapper][0]
    assert output['storage_guidance'] is None
    assert result['statusCode'] == 200
    assert row['storage_guidance']['max_days'] == 1e300
