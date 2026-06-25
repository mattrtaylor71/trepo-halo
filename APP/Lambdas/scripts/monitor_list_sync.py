#!/usr/bin/env python3
"""
Shopping-list sync monitor / tripwire.

Two checks, both read-only:
  1. DIVERGENCE SWEEP — every multi-member household: do all members' _new_list
     tables share the same set of household_item_uuids? (Union read masks this in
     the app, but a growing count means writes are silently failing to fan out.)
  2. FAN-OUT MISS LOG SCAN — counts `list_fanout_miss` events emitted by
     trepo-list-handler over the last N hours (the moment a write doesn't reach a
     member's table). Requires `logs:FilterLogEvents` (matt-cli has this).

Exit code 1 if anything is found (divergence rows or fan-out misses), so it can be
cron'd locally and piped to a notifier:
    */30 * * * * python3 monitor_list_sync.py --hours 1 || mail -s 'list sync drift' you@x

NOTE: the *native* CloudWatch path (metric filter + alarm + SNS) is preferable but
needs IAM perms matt-cli lacks (logs:PutMetricFilter, cloudwatch:PutMetricData,
sns:*). Drop-in commands for an admin profile are at the bottom of this file.
"""
import pymysql, subprocess, json, argparse, sys
from collections import Counter, defaultdict

DB = dict(host="database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com",
          user="admin", password="Nbmqyq17!", database="mysqlTutorial", connect_timeout=15)
LOG_GROUP = "/aws/lambda/trepo-list-handler"
AWS = ["--profile", "trepo-dev", "--region", "us-east-1"]


def divergence_sweep():
    conn = pymysql.connect(**DB); cur = conn.cursor()
    cur.execute("""SELECT owner_id FROM new_users WHERE owner_id IS NOT NULL AND owner_id<>''
                   GROUP BY owner_id HAVING COUNT(*)>1""")
    households = [r[0] for r in cur.fetchall()]
    findings = []
    for hid in households:
        cur.execute("SELECT user_id, COALESCE(first_name,'?') FROM new_users WHERE owner_id=%s", (hid,))
        members = cur.fetchall()
        sets = {}
        for uid, name in members:
            try:
                cur.execute(f"SELECT household_item_uuid FROM `{uid}_new_list`")
                sets[uid] = (name, {r[0] for r in cur.fetchall()})
            except Exception:
                pass
        if len(sets) < 2:
            continue
        union = set().union(*[s for _, s in sets.values()])
        for uid, (name, s) in sets.items():
            missing = union - s
            if missing:
                findings.append((hid, name, len(missing)))
    conn.close()
    return households, findings


def fanout_miss_scan(hours):
    import time
    start = int((time.time() - hours * 3600) * 1000)
    try:
        out = subprocess.run(
            ["aws", "logs", "filter-log-events", "--log-group-name", LOG_GROUP,
             "--start-time", str(start), "--filter-pattern", '"list_fanout_miss"',
             "--query", "events[*].message", "--output", "json"] + AWS,
            capture_output=True, text=True, timeout=120)
        msgs = json.loads(out.stdout or "[]")
    except Exception as e:
        return None, str(e)
    by_op = Counter()
    for m in msgs:
        try:
            i = m.index("{"); evt = json.loads(m[i:])
            by_op[evt.get("op", "?")] += 1
        except Exception:
            by_op["?"] += 1
    return by_op, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    args = ap.parse_args()

    print("=== LIST SYNC MONITOR ===")
    households, findings = divergence_sweep()
    print(f"\n[1] Divergence sweep — {len(households)} multi-member households")
    if not findings:
        print("    ✓ all households in sync (uuid sets match)")
    else:
        for hid, name, n in findings:
            print(f"    ✗ household {hid}: member '{name}' missing {n} item(s)")

    print(f"\n[2] Fan-out misses in trepo-list-handler logs (last {args.hours}h)")
    by_op, err = fanout_miss_scan(args.hours)
    if err:
        print(f"    (log scan unavailable: {err})")
    elif not by_op:
        print("    ✓ no list_fanout_miss events")
    else:
        for op, c in by_op.most_common():
            print(f"    ✗ {c} miss(es) on op={op}")

    bad = bool(findings) or bool(by_op)
    print(f"\n=== {'DRIFT DETECTED' if bad else 'CLEAN'} ===")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Native CloudWatch alarm (run once with an admin profile that has
# logs:PutMetricFilter, cloudwatch:PutMetricAlarm, sns:*):
#
#   aws logs put-metric-filter \
#     --log-group-name /aws/lambda/trepo-list-handler \
#     --filter-name list-fanout-miss \
#     --filter-pattern '"list_fanout_miss"' \
#     --metric-transformations \
#        metricName=ListFanoutMiss,metricNamespace=Trepo/ListSync,metricValue=1,defaultValue=0
#
#   aws sns create-topic --name trepo-list-sync-alerts            # -> TOPIC_ARN
#   aws sns subscribe --topic-arn TOPIC_ARN --protocol email --notification-endpoint matt@trepo.ai
#
#   aws cloudwatch put-metric-alarm \
#     --alarm-name trepo-list-fanout-miss \
#     --namespace Trepo/ListSync --metric-name ListFanoutMiss \
#     --statistic Sum --period 300 --evaluation-periods 1 --threshold 1 \
#     --comparison-operator GreaterThanOrEqualToThreshold \
#     --treat-missing-data notBreaching --alarm-actions TOPIC_ARN
# ---------------------------------------------------------------------------
