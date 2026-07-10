"""Characterization tests for the kitchen_api storage_location / COALESCE-sentinel
contract (the AnalyzeOnUpload create + enrichment-update path). Pins current behavior
so the Refactor Wave 1 changes stay behavior-preserving. No live DB — a fake cursor
records the SQL. Run: pytest kitchen_api/test_kitchen_contract.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402  kitchen_api/app.py


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, values=None):
        self.calls.append((" ".join(sql.split()), values))


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def test_coalesce_sentinel_emits_coalesce_and_only_fills_when_null():
    conn = _FakeConn()
    app._shared_kitchen_update(conn, "id-1", {"storage_location": app._CoalesceExisting("fridge")})
    sql, values = conn.cur.calls[-1]
    # Enrichment backfill must be conditional so it never overwrites a user's value.
    assert "COALESCE(`storage_location`, %s)" in sql
    assert "fridge" in values
    assert values[-1] == "id-1"  # WHERE `_id` = %s bound last
    assert conn.committed is True


def test_regular_value_is_a_plain_assignment_not_coalesce():
    conn = _FakeConn()
    app._shared_kitchen_update(conn, "id-2", {"product_name": "Milk"})
    sql, values = conn.cur.calls[-1]
    assert "`product_name` = %s" in sql
    assert "COALESCE" not in sql
    assert "Milk" in values


def test_mixed_update_keeps_direct_field_direct_and_sentinel_coalesced():
    conn = _FakeConn()
    app._shared_kitchen_update(conn, "id-3", {
        "product_name": "Yogurt",                       # direct write wins
        "storage_location": app._CoalesceExisting("pantry"),  # only if empty
    })
    sql, values = conn.cur.calls[-1]
    assert "`product_name` = %s" in sql
    assert "COALESCE(`storage_location`, %s)" in sql
    # storage_location value precedes the WHERE-clause id
    assert "pantry" in values and "Yogurt" in values
    assert values[-1] == "id-3"


def test_empty_update_is_a_noop():
    conn = _FakeConn()
    app._shared_kitchen_update(conn, "id-4", {})
    assert conn.cur.calls == []
    assert conn.committed is False
