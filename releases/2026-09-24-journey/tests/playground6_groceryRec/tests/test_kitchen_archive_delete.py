import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "kitchen_api" / "app.py"
APP_DIR = str(APP_PATH.parent)


def load_app_module():
    module_name = f"kitchen_api_app_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, APP_DIR)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class KitchenArchiveDeleteTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app_module()

    def test_live_shelf_life_does_not_dispatch_obsolete_cache_work(self):
        with patch.object(self.app.boto3, "client", side_effect=AssertionError("Redundant cloud work")):
            self.assertIsNone(self.app._refresh_shelf_life_cache("fixture-owner"))

    def test_archive_insert_defaults_null_is_opened_to_false(self):
        conn = Mock()
        cur = Mock()
        cur.fetchone.return_value = {
            "_id": "item-123",
            "product_name": "Milk",
            "is_opened": None,
        }

        with patch.object(self.app, "_ensure_archive_table"), patch.object(
            self.app,
            "_get_table_columns",
            return_value={"_id", "product_name", "is_opened", "archived_at", "archived_reason", "archived_from_table"},
        ):
            row = self.app._archive_kitchen_row(
                conn,
                cur,
                "owner_prod_kitchen",
                "owner_archive_kitchen",
                "item-123",
                "manual_delete",
            )

        self.assertEqual(row["_id"], "item-123")
        self.assertEqual(cur.execute.call_count, 2)
        _, insert_values = cur.execute.call_args_list[1].args
        self.assertIn(False, insert_values)
        self.assertNotIn(None, insert_values[:3])

    def test_backfill_null_is_opened_updates_existing_rows(self):
        conn = Mock()
        cur = Mock()
        cur.rowcount = 4

        with patch.object(self.app, "_get_table_columns", return_value={"is_opened"}):
            updated_count = self.app._backfill_null_is_opened(conn, cur, "owner_prod_kitchen")

        self.assertEqual(updated_count, 4)
        conn.commit.assert_called_once()
        query = cur.execute.call_args.args[0]
        self.assertIn("UPDATE `owner_prod_kitchen` SET `is_opened` = 0", query)

    def test_migrate_kitchen_is_opened_summarizes_prod_and_archive_work(self):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        mysql_conn = MagicMock()
        mysql_conn.__enter__.return_value = conn
        mysql_conn.__exit__.return_value = False

        with patch.object(self.app, "_mysql_conn", return_value=mysql_conn), patch.object(
            self.app,
            "_list_kitchen_table_names",
            return_value=["owner_archive_kitchen", "owner_prod_kitchen"],
        ), patch.object(self.app, "_ensure_archive_metadata_columns"), patch.object(
            self.app, "_ensure_prod_kitchen_columns"
        ), patch.object(
            self.app, "_backfill_null_is_opened", side_effect=[2, 5]
        ):
            summary = self.app.migrate_kitchen_is_opened(owner="owner")

        self.assertEqual(summary["owner"], "owner")
        self.assertEqual(summary["archive_tables_checked"], 1)
        self.assertEqual(summary["prod_tables_checked"], 1)
        self.assertEqual(summary["archive_rows_backfilled"], 2)
        self.assertEqual(summary["prod_rows_backfilled"], 5)
        self.assertEqual(
            summary["tables"],
            [
                {
                    "table_name": "owner_archive_kitchen",
                    "table_type": "archive",
                    "rows_backfilled": 2,
                },
                {
                    "table_name": "owner_prod_kitchen",
                    "table_type": "prod",
                    "rows_backfilled": 5,
                },
            ],
        )

    def test_should_bump_kitchen_version_for_recipe_relevant_fields(self):
        self.assertTrue(self.app._should_bump_kitchen_version_for_fields(["product_name"]))
        self.assertTrue(self.app._should_bump_kitchen_version_for_fields(["analysis_status"]))
        self.assertFalse(self.app._should_bump_kitchen_version_for_fields(["product_expiration"]))

    def test_mark_recipe_refresh_needed_for_owners_increments_versions(self):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False

        with patch.object(self.app, "_ensure_owner_kitchen_state_table"):
            self.app._mark_recipe_refresh_needed_for_owners(conn, ["owner-1", "owner-2"])

        cur.executemany.assert_called_once()
        query, params = cur.executemany.call_args.args
        self.assertIn("ON DUPLICATE KEY UPDATE", query)
        self.assertEqual(params, [("owner-1", 1, 1, 1), ("owner-2", 1, 1, 1)])
        conn.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
