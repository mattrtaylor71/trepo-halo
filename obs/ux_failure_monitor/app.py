"""trepo-ux-failure-monitor — outcome-based UX-failure monitor.

Replaces ~20 noisy CloudWatch error-signal alarms (fire on ANY error string /
5xx / Lambda-error, all self-healing) with 4 OUTCOME checks: only surface a
failure when a user actually did NOT get a result AND it persists.

Runs on an EventBridge rate(10 minutes) schedule. Every run computes 4 counts
and emits them as CloudWatch metrics in namespace `Trepo/UX` (via put_metric_data)
so a CloudWatch alarm (EvaluationPeriods=2, Period=600 -> must persist ~20 min)
pages only on a real, sustained user-facing failure.

The 4 metrics (mirror the team's 10-min health-check logic):
  1. RecipesNotGenerated      -- owners with >=5 kitchen items but recipes failed/empty/stuck
  2. ItemsStuckInEnrichment   -- kitchen rows that never got an analysis_stage (recent window)
  3. CaptureNotAnalyzed       -- fridge/pantry/receipt/dish capture jobs stuck pending/processing
  4. RecipesNotAnalyzed       -- saved-recipe extraction jobs stuck PENDING/RUNNING

Design notes (why the windows exist):
  - shared_kitchen has ~84 ANCIENT rows (Mar-May 2026, real product names) with a
    NULL analysis_stage from a pre-`analysis_stage`-column code path. They are NOT
    stuck enrichment. So check #2 is bounded to a RECENT window (STUCK_ENRICH_MAX_AGE_HOURS,
    default 6h): a genuinely-stuck item is recent; a 100-day-old null-stage row is a fossil.
  - grocery-identifier-dev-jobs has exactly 5 April-2026 'pending' fossils. Check #3
    excludes created_at LIKE '2026-04%' (per spec) AND applies the same recent-window filter.
  - TrepoJobs-dev has a handful of zombie saved_recipe_batch jobs left in RUNNING
    (their async worker died without writing a terminal state), ages 14h..3mo. Check #4
    is bounded to SAVED_RECIPE_MAX_AGE_HOURS (default 6h) so it pages on actionable recent
    stalls, not long-dead zombies.

Never crashes the whole run: each check is isolated. On success it emits the count
(0 when clean, so "no data" != "no failure"). On error it emits NOTHING for that
metric (alarm TreatMissingData=notBreaching) and bumps `MonitorCheckErrors` + logs.
"""
import os
import json
import datetime

import boto3
import pymysql

# ---- config (env-overridable) ----
NAMESPACE = os.getenv('METRIC_NAMESPACE', 'Trepo/UX')
MIN_KITCHEN_ITEMS = int(os.getenv('MIN_KITCHEN_ITEMS', '5'))
RECIPE_STUCK_MIN = int(os.getenv('RECIPE_STUCK_MIN', '30'))        # regenerating/generating grace
ENRICH_STALE_MIN = int(os.getenv('ENRICH_STALE_MIN', '30'))       # item older than this = stale
STUCK_ENRICH_MAX_AGE_HOURS = int(os.getenv('STUCK_ENRICH_MAX_AGE_HOURS', '6'))
CAPTURE_STALE_MIN = int(os.getenv('CAPTURE_STALE_MIN', '15'))
CAPTURE_MAX_AGE_HOURS = int(os.getenv('CAPTURE_MAX_AGE_HOURS', '6'))
SAVED_RECIPE_STALE_MIN = int(os.getenv('SAVED_RECIPE_STALE_MIN', '15'))
SAVED_RECIPE_MAX_AGE_HOURS = int(os.getenv('SAVED_RECIPE_MAX_AGE_HOURS', '6'))
IDENTIFIER_JOBS_TABLE = os.getenv('IDENTIFIER_JOBS_TABLE', 'grocery-identifier-dev-jobs')
SAVED_RECIPE_JOBS_TABLE = os.getenv('SAVED_RECIPE_JOBS_TABLE', 'TrepoJobs-dev')

# test/synthetic owner prefixes to exclude (%% because pymysql %-escapes the query string)
TEST_PREFIXES = ('idemtest%%', 'synthverify%%', 'demo-%%', 'test-%%')


def _owner_excl(col='owner_id'):
    return " AND ".join(f"{col} NOT LIKE '{p}'" for p in TEST_PREFIXES)


def _conn():
    return pymysql.connect(
        host=os.environ['DB_HOST'], user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'], database=os.environ['DB_NAME'],
        port=int(os.getenv('DB_PORT', '3306')), cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=8, read_timeout=45, autocommit=True)


def _q(conn, sql, params=None):
    with conn.cursor() as c:
        c.execute(sql, params or [])
        return c.fetchall()


def _iso_ago(minutes=0, hours=0):
    t = datetime.datetime.utcnow() - datetime.timedelta(minutes=minutes, hours=hours)
    return t.replace(microsecond=0).isoformat() + 'Z'


# ---------------------------------------------------------------------------
# Check 1: RecipesNotGenerated
# ---------------------------------------------------------------------------
def check_recipes_not_generated(conn):
    """Owners with >=MIN_KITCHEN_ITEMS eligible kitchen items whose shared_recipes
    'current' row is a user-facing failure: status in ('failed','empty'), or
    'regenerating'/'generating' whose _updatedDate is older than RECIPE_STUCK_MIN.

    Kitchen items are counted with the SAME gate the RecipesGenerator uses
    (action='IN', analysis in-progress-or-ready, DISTINCT lower(name)) so that
    "empty with >=5 items" is a *real* failure and not a legit empty on an owner
    whose only rows are discards / half-analyzed dupes.
    """
    eligible = {
        r['owner_id'] for r in _q(conn, f"""
            SELECT owner_id, COUNT(DISTINCT LOWER(product_name)) AS n
            FROM shared_kitchen
            WHERE action = 'IN'
              AND (analysis_stage IN ('preliminary','fast','final') OR analysis_stage IS NULL)
              AND (analysis_status = 'ready' OR analysis_status IS NULL)
              AND {_owner_excl()}
            GROUP BY owner_id
            HAVING n >= %s
        """, [MIN_KITCHEN_ITEMS])
    }
    if not eligible:
        return 0, []
    status = {r['owner_id']: (r['status'] or '', r['_updatedDate'])
              for r in _q(conn, "SELECT owner_id, status, _updatedDate FROM shared_recipes WHERE _id='current'")}
    now = datetime.datetime.utcnow()
    failed = []
    for owner in eligible:
        st, upd = status.get(owner, ('', None))
        st = (st or '').lower()
        if st in ('failed', 'empty'):
            failed.append({'owner_id': owner, 'status': st})
        elif st in ('regenerating', 'generating'):
            age_min = (now - upd).total_seconds() / 60 if upd else 1e9
            if age_min > RECIPE_STUCK_MIN:
                failed.append({'owner_id': owner, 'status': st, 'age_min': int(age_min)})
    return len(failed), failed


# ---------------------------------------------------------------------------
# Check 2: ItemsStuckInEnrichment
# ---------------------------------------------------------------------------
def check_items_stuck_in_enrichment(conn):
    """Kitchen rows with no analysis_stage that are older than ENRICH_STALE_MIN but
    still within STUCK_ENRICH_MAX_AGE_HOURS (recent window kills the ~84 ancient
    null-stage fossils). Excludes test owners and Probe/Idem synthetic products."""
    n = _q(conn, f"""
        SELECT COUNT(*) AS n FROM shared_kitchen
        WHERE (analysis_stage IS NULL OR analysis_stage = '')
          AND _createdDate < (UTC_TIMESTAMP() - INTERVAL %s MINUTE)
          AND _createdDate > (UTC_TIMESTAMP() - INTERVAL %s HOUR)
          AND {_owner_excl()}
          AND product_name NOT LIKE '%%Probe%%'
          AND product_name NOT LIKE '%%Idem%%'
    """, [ENRICH_STALE_MIN, STUCK_ENRICH_MAX_AGE_HOURS])[0]['n']
    return int(n)


# ---------------------------------------------------------------------------
# Check 3: CaptureNotAnalyzed  (DynamoDB grocery-identifier-dev-jobs)
# ---------------------------------------------------------------------------
def _scan_all(table, **kwargs):
    ddb = boto3.resource('dynamodb')
    t = ddb.Table(table)
    items, start = [], None
    while True:
        if start:
            kwargs['ExclusiveStartKey'] = start
        resp = t.scan(**kwargs)
        items.extend(resp.get('Items', []))
        start = resp.get('LastEvaluatedKey')
        if not start:
            return items


def check_capture_not_analyzed():
    """Fridge/pantry/receipt/dish capture analysis jobs stuck non-terminal.

    status in (pending, processing), most-recent-activity older than CAPTURE_STALE_MIN,
    within CAPTURE_MAX_AGE_HOURS. Also explicitly excludes the 5 known April-2026
    'pending' fossils (created_at LIKE '2026-04%')."""
    from boto3.dynamodb.conditions import Attr
    items = _scan_all(
        IDENTIFIER_JOBS_TABLE,
        FilterExpression=Attr('status').is_in(['pending', 'processing']))
    stale_cut = _iso_ago(minutes=CAPTURE_STALE_MIN)
    fresh_cut = _iso_ago(hours=CAPTURE_MAX_AGE_HOURS)
    stuck = []
    for it in items:
        created = str(it.get('created_at') or '')
        updated = str(it.get('updated_at') or '') or created
        last = max(created, updated)                 # ISO strings sort lexically
        if created.startswith('2026-04'):
            continue                                  # known fossils
        if last < stale_cut and created > fresh_cut:
            stuck.append({'job_id': it.get('job_id'), 'status': it.get('status'),
                          'created_at': created, 'updated_at': updated})
    return len(stuck), stuck


# ---------------------------------------------------------------------------
# Check 4: RecipesNotAnalyzed  (DynamoDB TrepoJobs-dev, type=saved_recipe_batch)
# ---------------------------------------------------------------------------
def check_recipes_not_analyzed():
    """Saved-recipe extraction jobs (type=saved_recipe_batch) stuck non-terminal:
    status in (PENDING/RUNNING), most-recent-activity older than SAVED_RECIPE_STALE_MIN,
    within SAVED_RECIPE_MAX_AGE_HOURS (recent window skips long-dead RUNNING zombies)."""
    from boto3.dynamodb.conditions import Attr
    items = _scan_all(
        SAVED_RECIPE_JOBS_TABLE,
        FilterExpression=(Attr('type').eq('saved_recipe_batch')
                          & Attr('status').is_in(['PENDING', 'RUNNING', 'pending', 'running'])))
    stale_cut = _iso_ago(minutes=SAVED_RECIPE_STALE_MIN)
    fresh_cut = _iso_ago(hours=SAVED_RECIPE_MAX_AGE_HOURS)
    stuck = []
    for it in items:
        created = str(it.get('created_at') or '')
        updated = str(it.get('updated_at') or '') or created
        last = max(created, updated)
        if last < stale_cut and created > fresh_cut:
            stuck.append({'job_id': it.get('job_id'), 'status': it.get('status'),
                          'created_at': created, 'updated_at': updated})
    return len(stuck), stuck


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------
def handler(event, context):
    metrics = {}          # metric_name -> value (only for checks that SUCCEEDED)
    details = {}
    check_errors = 0

    conn = None
    # DB-backed checks (1 & 2) share one connection.
    try:
        conn = _conn()
    except Exception as e:
        conn = None
        check_errors += 2
        print(json.dumps({'evt': 'ux_check_error', 'check': 'db_connect', 'error': str(e)}))

    if conn is not None:
        try:
            n, failed = check_recipes_not_generated(conn)
            metrics['RecipesNotGenerated'] = n
            details['RecipesNotGenerated'] = failed[:25]
        except Exception as e:
            check_errors += 1
            print(json.dumps({'evt': 'ux_check_error', 'check': 'RecipesNotGenerated', 'error': str(e)}))
        try:
            metrics['ItemsStuckInEnrichment'] = check_items_stuck_in_enrichment(conn)
        except Exception as e:
            check_errors += 1
            print(json.dumps({'evt': 'ux_check_error', 'check': 'ItemsStuckInEnrichment', 'error': str(e)}))
        try:
            conn.close()
        except Exception:
            pass

    try:
        n, stuck = check_capture_not_analyzed()
        metrics['CaptureNotAnalyzed'] = n
        details['CaptureNotAnalyzed'] = stuck[:25]
    except Exception as e:
        check_errors += 1
        print(json.dumps({'evt': 'ux_check_error', 'check': 'CaptureNotAnalyzed', 'error': str(e)}))

    try:
        n, stuck = check_recipes_not_analyzed()
        metrics['RecipesNotAnalyzed'] = n
        details['RecipesNotAnalyzed'] = stuck[:25]
    except Exception as e:
        check_errors += 1
        print(json.dumps({'evt': 'ux_check_error', 'check': 'RecipesNotAnalyzed', 'error': str(e)}))

    # Emit to CloudWatch. Only emit metrics whose check succeeded (a missing metric
    # + TreatMissingData=notBreaching means "did not run" != "clean"). Always emit
    # MonitorCheckErrors so a broken monitor is itself observable.
    data = [{'MetricName': name, 'Value': float(val), 'Unit': 'Count'}
            for name, val in metrics.items()]
    data.append({'MetricName': 'MonitorCheckErrors', 'Value': float(check_errors), 'Unit': 'Count'})
    try:
        boto3.client('cloudwatch').put_metric_data(Namespace=NAMESPACE, MetricData=data)
    except Exception as e:
        print(json.dumps({'evt': 'ux_putmetric_error', 'error': str(e)}))

    print(json.dumps({'evt': 'ux_failure_scan', 'metrics': metrics,
                      'check_errors': check_errors, 'details': details}))
    return {'metrics': metrics, 'check_errors': check_errors}
