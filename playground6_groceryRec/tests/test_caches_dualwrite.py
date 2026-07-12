"""M4 characterization: recipes + meal_plan CACHE dual-writes to shared_recipes /
shared_meal_plan (one-row-per-owner caches, shared PK = owner_id).

Writers:
  recipes    -> recipes_generator (_dual_write_recipes_to_shared, flag DUAL_WRITE_RECIPES)
  meal_plan  -> meal_plan_generator + meal_plan_api (_dual_write_meal_plan_to_shared,
                flag DUAL_WRITE_MEAL_PLAN — duplicated helper, byte-equivalence guarded)

For each helper: (a) OFF by default => no shared write (per-user path untouched);
(b) ON => exactly one owner_id-keyed INSERT...SELECT-current upsert copying every
cache column.
"""
import importlib.util, os, re, sys, unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
LOAD_ENV = {
    "BUCKET_NAME": "b", "OPENAI_API_KEY": "k", "DB_HOST": "h", "DB_USER": "u",
    "DB_PASS": "p", "DB_NAME": "n", "DB_PORT": "3306", "AWS_DEFAULT_REGION": "us-east-1",
    "IOT_ENDPOINT": "x", "JOBS_TABLE": "j", "MEAL_PLAN_GENERATOR_ARN": "arn",
    "RECIPES_GENERATOR_ARN": "arn",
}
RECIPES_COLS = ['_id', '_owner', 'status', 'kitchen_only', 'need_grocery',
                'error_message', '_createdDate', '_updatedDate']
MEAL_PLAN_COLS = ['_id', '_owner', 'status', 'focus', 'explanation_title',
                  'explanation_paragraph', 'plan', 'error_message', '_createdDate', '_updatedDate']


def load(pkg):
    p = ROOT / pkg / "app.py"
    spec = importlib.util.spec_from_file_location(f"m_{uuid4().hex}", p)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    try:
        with patch.dict(os.environ, LOAD_ENV, clear=False):
            spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    return m


class _Cur:
    def __init__(self): self.calls = []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.calls.append((sql, params))


class _Conn:
    def __init__(self): self.cur = _Cur(); self.committed = False
    def cursor(self): return self.cur
    def commit(self): self.committed = True


class CacheDualWriteTests(unittest.TestCase):
    def _check(self, app, helper, flag, shared_tbl, cols):
        # OFF
        setattr(app, flag, False)
        conn = _Conn()
        getattr(app, helper)(conn, "owner-1")
        self.assertEqual(conn.cur.calls, [], f"{helper}: flag OFF must not write shared")
        # ON
        setattr(app, flag, True)
        conn = _Conn()
        getattr(app, helper)(conn, "owner-1")
        self.assertEqual(len(conn.cur.calls), 1, f"{helper}: one shared write expected")
        sql, params = conn.cur.calls[0]
        self.assertIn(shared_tbl, sql)
        self.assertIn("INSERT INTO", sql)
        self.assertIn("SELECT", sql)
        self.assertIn("_id`='current'", sql.replace(" ", ""))   # copy the current cache row
        self.assertIn("ON DUPLICATE KEY UPDATE", sql)
        self.assertIn("`owner_id`", sql)
        self.assertEqual(params, ("owner-1",))
        for col in cols:
            self.assertIn(f"`{col}`", sql, f"{helper}: missing col {col}")
        self.assertTrue(conn.committed)

    def test_recipes_cache_dualwrite(self):
        app = load("recipes_generator")
        self._check(app, "_dual_write_recipes_to_shared", "DUAL_WRITE_RECIPES",
                    "shared_recipes", RECIPES_COLS)

    def test_meal_plan_cache_dualwrite_generator(self):
        app = load("meal_plan_generator")
        self._check(app, "_dual_write_meal_plan_to_shared", "DUAL_WRITE_MEAL_PLAN",
                    "shared_meal_plan", MEAL_PLAN_COLS)

    def test_meal_plan_cache_dualwrite_api(self):
        app = load("meal_plan_api")
        self._check(app, "_dual_write_meal_plan_to_shared", "DUAL_WRITE_MEAL_PLAN",
                    "shared_meal_plan", MEAL_PLAN_COLS)

    def test_meal_plan_helper_byte_equivalent(self):
        blocks = []
        for pkg in ["meal_plan_generator", "meal_plan_api"]:
            src = (ROOT / pkg / "app.py").read_text()
            m = re.search(r"DUAL_WRITE_MEAL_PLAN = os\.getenv.*?def _dual_write_meal_plan_to_shared.*?\n\n\n",
                          src, re.DOTALL)
            self.assertIsNotNone(m, f"{pkg}: meal_plan dual-write block not found")
            blocks.append(m.group(0))
        self.assertEqual(blocks[0], blocks[1], "meal_plan dual-write block drifted between generator and api")


if __name__ == "__main__":
    unittest.main()
