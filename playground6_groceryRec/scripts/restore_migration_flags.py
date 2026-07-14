#!/usr/bin/env python3
"""Restore CLI-set shared-table migration env flags after a `sam deploy`.

WHY THIS EXISTS: the shared-table migration flags (DUAL_WRITE_*, READ_SHARED_*,
USE_SHARED_TABLES, WRITE_SHARED_ONLY, ...) are set via
`aws lambda update-function-configuration`, NOT in template.yaml. CloudFormation
resets a function's environment to the template's declaration whenever that
function's code changes in a deploy — silently wiping the flags. On 2026-07-14
this dropped DUAL_WRITE_RECIPES on the recipes generator: reads had been flipped
to shared_recipes but new generations wrote per-user only, so every new user saw
an empty "Use What You Have" (178 owners affected, backfilled same day).

USAGE (run after EVERY sam deploy of this stack until flags are template-ized):
    AWS_PROFILE=trepo-dev python3 scripts/restore_migration_flags.py            # dry-run diff
    AWS_PROFILE=trepo-dev python3 scripts/restore_migration_flags.py --apply    # restore

The desired state lives in migration_flags_snapshot.json (same dir). If you
intentionally change a flag, re-snapshot:
    python3 - <<'EOF' > scripts/migration_flags_snapshot.json
    ... (see snapshot_flags.py logic in session notes) ...
    EOF
"""
import json
import os
import sys

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "migration_flags_snapshot.json")

def main():
    apply = "--apply" in sys.argv
    lam = boto3.client("lambda")
    desired = json.load(open(SNAPSHOT))
    drift = 0
    for fn, flags in sorted(desired.items()):
        try:
            cfg = lam.get_function_configuration(FunctionName=fn)
        except lam.exceptions.ResourceNotFoundException:
            print(f"SKIP (gone) {fn}")
            continue
        env = cfg.get("Environment", {}).get("Variables", {})
        missing = {k: v for k, v in flags.items() if env.get(k) != v}
        if not missing:
            continue
        drift += 1
        print(f"DRIFT {fn}: {missing}")
        if apply:
            env.update(missing)
            lam.update_function_configuration(FunctionName=fn, Environment={"Variables": env})
            lam.get_waiter("function_updated").wait(FunctionName=fn)
            print(f"  restored")
    if drift == 0:
        print("no drift — all migration flags present")
    elif not apply:
        print(f"\n{drift} function(s) drifted. Re-run with --apply to restore.")

if __name__ == "__main__":
    main()
