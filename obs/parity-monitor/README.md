# trepo-parity-monitor

Scheduled guardrail for the shared-table migration (all 8 types read from shared as of
2026-07-13). Verifies every owner's per-owner (source-of-truth) tables are fully mirrored
in the shared_* tables — catches both "owner entirely missing" and "short by N rows"
(the Lexi saved-recipes class) before a user does.

- Lambda: `trepo-parity-monitor` (us-east-1, handler app.handler, pymysql bundled in zip)
- Types covered: saved_recipes, recipes, meal_plan, metrics, dishes, discards
- Emits `{"evt":"parity_miss",...}` per confirmed short owner (CloudWatch metric filter
  -> alarm -> SNS) and a `parity_scan_ok`/`parity_scan_found` heartbeat per run.
- Deploy: download live zip, overlay app.py, update-function-code (pymysql lives in the zip).
