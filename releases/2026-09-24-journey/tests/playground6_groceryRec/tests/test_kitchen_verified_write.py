"""A successful SQL UPDATE is not proof that the requested edit persisted."""
import json
import uuid
import pytest
from test_amount_operations import db, row, run, a


def edit(fields):
    return {'operation_id': str(uuid.uuid4()), 'kind': 'edit',
            'items': [{'item_id': 'beef', 'revision': 0, 'fields': fields}]}


def test_persisted_mismatch_rolls_back_item_metadata_and_receipt(db):
    # A later trigger can silently override a valid write. Never acknowledge it.
    with db[0].cursor() as cur:
        cur.execute("""CREATE TRIGGER fixture_override BEFORE UPDATE ON shared_kitchen
            FOR EACH ROW FOLLOWS trepo_preserve_item_edits SET NEW.product_name=OLD.product_name""")
    body = edit({'product_name': 'Corrected name', 'brand': 'Requested brand'})
    with pytest.raises(a.Rejected, match='confirm') as failure:
        run(db, body)
    assert failure.value.status == 409
    assert row(db)['product_name'] == 'beef'
    assert row(db)['brand'] is None
    with db[0].cursor() as cur:
        for table in ('kitchen_item_edits', 'kitchen_amount_operations'):
            cur.execute('SELECT COUNT(*) AS n FROM '+table)
            assert cur.fetchone()['n'] == 0


def test_verified_rename_preserves_unrequested_fields_and_is_retryable(db):
    original = row(db)
    body = edit({'product_name': 'Garlic Olive Oil'})
    result = run(db, body)
    assert result['items'][0]['item']['product_name'] == 'Garlic Olive Oil'
    assert run(db, body) == result
    current = row(db)
    for field in ('quantity_value', 'quantity_unit', 'brand', 'storage_location', 'is_opened', 'product_expiration'):
        assert current[field] == original[field]


def test_readback_compares_database_types_without_false_conflicts(db):
    result = run(db, edit({'product_name': 'Correct', 'ingredients': ['Beef'],
                           'is_opened': True, 'product_expiration': '2026-12-01',
                           'quantity_value': '12.50', 'quantity_unit': 'oz', 'brand': None}))
    item = result['items'][0]['item']
    assert item['is_opened'] == 1
    assert json.loads(row(db)['ingredients']) == ['Beef']
