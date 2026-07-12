"""M4 characterization: metrics dual-write to shared_metrics across all 4 writers
(redemptions_api, kitchen_analysis_generator, metrics_api, analyze_on_upload).

Verifies for each writer's `_dual_write_metrics_to_shared`:
  (a) OFF by default (DUAL_WRITE_METRICS unset) => no shared write — reads/writes
      of the per-user path are completely untouched.
  (b) ON => exactly one owner_id-keyed INSERT...SELECT upsert into shared_metrics
      that copies all 17 snapshot columns (behavior-preserving append/merge).
Plus a byte-equivalence guard: the helper is duplicated across the 4 packages and
must stay identical (single logical source until it is hoisted to a shared layer).
"""
import importlib.util, os, re, sys, unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
WRITERS = ["redemptions_api", "kitchen_analysis_generator", "metrics_api", "analyze_on_upload"]
LOAD_ENV = {
    "BUCKET_NAME": "b", "OPENAI_API_KEY": "k", "DB_HOST": "h", "DB_USER": "u",
    "DB_PASS": "p", "DB_NAME": "n", "DB_PORT": "3306", "AWS_DEFAULT_REGION": "us-east-1",
    "IOT_ENDPOINT": "x.iot.us-east-1.amazonaws.com", "JOBS_TABLE": "jobs",
}
METRICS_COLS = [
    "_id", "_owner", "_createdDate", "IQ", "Points", "UPF", "harmful_ingredients",
    "IQ_what", "IQ_suggestions", "UPF_what", "UPF_suggestions",
    "harmful_ingredients_what", "harmful_ingredients_suggestions",
    "kitchen_analysis_status", "kitchen_analysis_content", "kitchen_analysis_generated_at",
    "kitchen_analysis_error",
]


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


class MetricsDualWriteTests(unittest.TestCase):
    def test_off_by_default_no_shared_write(self):
        for pkg in WRITERS:
            app = load(pkg)
            app.DUAL_WRITE_METRICS = False
            conn = _Conn()
            app._dual_write_metrics_to_shared(conn, "owner-1", "owner-1-metrics", "snap-1")
            self.assertEqual(conn.cur.calls, [], f"{pkg}: flag OFF must not write to shared")

    def test_on_writes_owner_keyed_upsert_all_cols(self):
        for pkg in WRITERS:
            app = load(pkg)
            app.DUAL_WRITE_METRICS = True
            conn = _Conn()
            app._dual_write_metrics_to_shared(conn, "owner-1", "owner-1-metrics", "snap-1")
            self.assertEqual(len(conn.cur.calls), 1, f"{pkg}: expected exactly one shared write")
            sql, params = conn.cur.calls[0]
            self.assertIn("shared_metrics", sql, pkg)
            self.assertIn("INSERT INTO", sql, pkg)
            self.assertIn("SELECT", sql, pkg)                 # copy-by-_id
            self.assertIn("ON DUPLICATE KEY UPDATE", sql, pkg)  # idempotent re-sync
            self.assertIn("`owner_id`", sql, pkg)
            self.assertIn("owner-1-metrics", sql, pkg)        # source table interpolated
            self.assertEqual(params, ("owner-1", "snap-1"), f"{pkg}: (owner_id, _id) params")
            for col in METRICS_COLS:                          # all 17 mirrored
                self.assertIn(f"`{col}`", sql, f"{pkg}: missing col {col}")
            self.assertTrue(conn.committed, pkg)

    def test_helper_byte_equivalent_across_writers(self):
        blocks = []
        for pkg in WRITERS:
            src = (ROOT / pkg / "app.py").read_text()
            m = re.search(r"DUAL_WRITE_METRICS = os\.getenv.*?def _dual_write_metrics_to_shared.*?\n\n\ndef ",
                          src, re.DOTALL)
            self.assertIsNotNone(m, f"{pkg}: dual-write block not found")
            blocks.append(m.group(0))
        for other in blocks[1:]:
            self.assertEqual(blocks[0], other, "metrics dual-write block drifted between writers")


if __name__ == "__main__":
    unittest.main()
