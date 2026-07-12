"""M4 characterization: saved_recipes dual-write to shared_saved_recipes.
Verifies (a) OFF by default = no shared write (reads/writes untouched),
(b) ON = one owner_id-keyed upsert with the same values (behavior-preserving)."""
import importlib.util, os, sys, unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "saved_recipes_api" / "app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location(f"sr_dw_{uuid4().hex}", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"BUCKET_NAME": "b", "OPENAI_API_KEY": "k"}, clear=False):
        sys.path.insert(0, str(APP_PATH.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    return module


class _Cur:
    def __init__(self): self.calls = []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.calls.append((sql, params))


class _Conn:
    def __init__(self): self.cur = _Cur(); self.committed = False
    def cursor(self): return self.cur
    def commit(self): self.committed = True


ROW = {
    "_id": "rec-1", "_owner": "owner-1", "source_type": "tiktok", "source_url": "u",
    "resolved_url": "https://x/reel/1", "title": "T", "image_url": None, "image_urls": "[]",
    "source_image_url": None, "source_image_urls": "[]", "image_storage_key": None,
    "ingredients": '["a"]', "instructions": '["b"]', "notes": "[]", "raw_caption": "c",
    "raw_content": "d", "extraction_source": "apify", "author_name": None,
    "caption_field": None, "status": "ready",
}


class SavedRecipeDualWriteTests(unittest.TestCase):
    def setUp(self): self.app = load_app_module()

    def test_off_by_default_no_shared_write(self):
        self.app.DUAL_WRITE_SAVED_RECIPES = False
        conn = _Conn()
        with patch.object(self.app, "_fetch_saved_recipe_by_id", return_value=ROW):
            self.app._dual_write_saved_recipe_to_shared(conn, "owner-1", "rec-1")
        self.assertEqual(conn.cur.calls, [], "flag OFF must not write to shared")

    def test_on_writes_shared_with_owner_id_and_upsert(self):
        self.app.DUAL_WRITE_SAVED_RECIPES = True
        conn = _Conn()
        with patch.object(self.app, "_fetch_saved_recipe_by_id", return_value=ROW):
            self.app._dual_write_saved_recipe_to_shared(conn, "owner-1", "rec-1")
        self.assertEqual(len(conn.cur.calls), 1)
        sql, params = conn.cur.calls[0]
        self.assertIn("shared_saved_recipes", sql)
        self.assertIn("ON DUPLICATE KEY UPDATE", sql)     # per-owner upsert dedupe
        self.assertEqual(params[0], "owner-1")            # owner_id first
        self.assertEqual(params[1], "rec-1")              # _id
        self.assertEqual(params[6], self.app._sha256("https://x/reel/1"))  # recomputed hash
        self.assertTrue(conn.committed)

    def test_missing_row_is_noop(self):
        self.app.DUAL_WRITE_SAVED_RECIPES = True
        conn = _Conn()
        with patch.object(self.app, "_fetch_saved_recipe_by_id", return_value=None):
            self.app._dual_write_saved_recipe_to_shared(conn, "owner-1", "rec-1")
        self.assertEqual(conn.cur.calls, [])


if __name__ == "__main__":
    unittest.main()
