"""Outcome monitoring with explicit coverage reporting.

Runs every ten minutes. Existing capture/enrichment checks are joined by failed
meal-plan households, terminal recipe failures, durable generation retries,
partial voice actions, missing original uploads and pre-execution throttles.
Each check is isolated: failed reads omit that outcome count and emit
CoverageComplete=0 plus MonitorCheckErrors. Zero means a completed clean check,
never an unavailable source. A metric delivery failure fails the invocation.
See ops/cloud-reliability/outcome-alarms.json for sustained-impact thresholds.
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
def check_meal_plan_failures(conn):
    rows = _q(conn, """SELECT p.owner_id,p.status,p.plan,p._updatedDate,
        COALESCE(NULLIF(u.owner_id,''),u.user_id) AS household
        FROM shared_meal_plan p JOIN new_users u ON u.user_id=p.owner_id
        WHERE p._id='current' AND (p.status='failed' OR
          (p.status='regenerating' AND p._updatedDate<UTC_TIMESTAMP()-INTERVAL 30 MINUTE))""")
    failures={}
    for row in rows:
        if str(row['owner_id']).startswith(('idemtest','synthverify','demo-','test-')):
            continue
        key=row['household']
        value=row.get('plan')
        if isinstance(value,str):
            value=json.loads(value)
        entry=failures.setdefault(key,{'household':key,'without_plan':True})
        entry['without_plan'] &= not bool(value)
    return len(failures),list(failures.values())


def check_terminal_imports(conn):
    # Count actual retained failed records, not every failed provider attempt.
    rows=_q(conn,"""SELECT owner_id,_id AS recipe_id,status,_updatedDate
        FROM shared_saved_recipes WHERE status='failed'
        AND _updatedDate > UTC_TIMESTAMP()-INTERVAL 72 HOUR""")
    rows=[r for r in rows if not str(r['owner_id']).startswith(('idemtest','synthverify','demo-','test-'))]
    return len({(r['owner_id'],r['recipe_id']) for r in rows}),[
        {'owner_id':r['owner_id'],'recipe_id':r['recipe_id']} for r in rows]


def check_meal_retry_failures(conn):
    rows=_q(conn,"""SELECT scope_id,state,last_error FROM meal_plan_generation_jobs
        WHERE state IN ('failed','blocked') OR (state='running' AND lease_until<UTC_TIMESTAMP()-INTERVAL 5 MINUTE)
        OR (state='pending' AND due_at<UTC_TIMESTAMP()-INTERVAL 5 MINUTE)""")
    return len(rows),rows


def check_partial_actions():
    # Durable app operation receipts use verified tool outcomes, not response prose.
    # Older Halo sessions without this protocol remain a separate coverage gap.
    table=os.environ['VOICE_OPERATION_TABLE']
    from boto3.dynamodb.conditions import Attr
    now=datetime.datetime.now(datetime.timezone.utc).timestamp()
    rows=_scan_all(table,FilterExpression=Attr('request_status').eq('completed') &
                   Attr('operation_expires_at').gte(int(now)-2*86400))
    failed=[];unsupported=0
    for row in rows:
        body=row.get('response_body') or {}
        outcome=body.get('outcome') or {}
        if not outcome:
            unsupported+=1
        elif outcome.get('actions') in ('partial','failed','unconfirmed'):
            failed.append({'operation_id':row.get('operation_id'),'actions':outcome['actions']})
    if unsupported:
        raise ValueError('operation_receipts_missing_structured_outcome:'+str(unsupported))
    return len(failed),failed



def check_halo_voice_outcomes():
    from boto3.dynamodb.conditions import Attr
    rows=_scan_all(os.getenv('HALO_VOICE_JOBS_TABLE','trepo-quick-ack-voice-jobs-dev'),
                   FilterExpression=Attr('accepted_at').gte(_iso_ago(hours=72)))
    failures=[]
    for row in rows:
        if row.get('status')=='completed':
            outcome=row.get('outcome') or {}
            if not outcome:
                # Legacy index.js returns this fixed result BEFORE running the
                # assistant when transcription is empty. It is an explicit
                # no-action terminal branch, not model prose proving a write.
                if (row.get('transcript') is None and row.get('assistant_actions')==[]
                    and row.get('response_type')=='info_answer'
                    and row.get('response_text')=='No speech detected.'
                    and row.get('last_error') is None):
                    continue
                raise ValueError('halo_voice_outcome_coverage_missing')
            if outcome.get('actions') in ('partial','failed','unconfirmed'):
                failures.append({'job_id':row['job_id'],'actions':outcome['actions']})
        elif row.get('status') not in ('cancelled',) and str(row.get('updated_at') or row.get('accepted_at') or '')<_iso_ago(minutes=15):
            failures.append({'job_id':row['job_id'],'status':row.get('status')})
    return len(failures),failures


def check_missing_halo_media():
    from boto3.dynamodb.conditions import Attr
    import re
    buckets=json.loads(os.environ['HALO_MEDIA_BUCKETS'])
    if not buckets or not isinstance(buckets,list):raise ValueError('halo_media_buckets_missing')
    rows=_scan_all(SAVED_RECIPE_JOBS_TABLE,FilterExpression=Attr('status').eq('PENDING') &
        Attr('created_at').gte(_iso_ago(hours=72)))
    client=boto3.client('s3');missing=[]
    for row in rows:
        if not str(row.get('device_id','')).startswith('halo-') or row.get('type') not in ('grocery','dish','discard'):
            continue
        if str(row.get('created_at') or '')>=_iso_ago(minutes=15):continue
        expected='images/'+str(row.get('owner'))+'/'+str(row.get('device_id'))+'/'
        key=str(row.get('s3_key') or '')
        if not key.startswith(expected):raise ValueError('halo_media_path_coverage_missing')
        exists=False
        for bucket in buckets:
            try:
                result=client.head_object(Bucket=bucket,Key=key)
                if result['ContentLength']>0:exists=True;break
            except Exception as exc:
                code=getattr(exc,'response',{}).get('Error',{}).get('Code')
                if code in ('403','AccessDenied'):
                    # S3 hides absence as 403 without unrestricted ListBucket.
                    # An exact-prefix list uses the separately scoped listing
                    # grant; a denied list remains unknown coverage, not absence.
                    listing=client.list_objects_v2(Bucket=bucket,Prefix=key,MaxKeys=1)
                    if any(item.get('Key')==key and item.get('Size',0)>0
                           for item in listing.get('Contents',[])):
                        exists=True
                        break
                elif code not in ('404','NoSuchKey','NotFound'):raise
        if not exists:missing.append({'job_id':row['job_id'],'reason':'upload_not_received'})
    return len(missing),missing

def check_capacity_rejections():
    functions=json.loads(os.environ['MONITORED_LAMBDAS'])
    if not functions or not isinstance(functions,list):
        raise ValueError('monitored_functions_not_configured')
    cloud=boto3.client('cloudwatch')
    end=datetime.datetime.now(datetime.timezone.utc)
    total=0
    for function in functions:
        for metric in ('Throttles','AsyncEventsDropped'):
            response=cloud.get_metric_statistics(Namespace='AWS/Lambda',MetricName=metric,
                Dimensions=[{'Name':'FunctionName','Value':function}],StartTime=end-datetime.timedelta(minutes=10),
                EndTime=end,Period=600,Statistics=['Sum'])
            total+=sum(float(p['Sum']) for p in response.get('Datapoints',[]))
    return total


def handler(event, context):
    metrics, details, failures = {}, {}, []
    db_checks={'RecipesNotGenerated':check_recipes_not_generated,
        'ItemsStuckInEnrichment':check_items_stuck_in_enrichment,
        'MealPlanFailures':check_meal_plan_failures,'TerminalRecipeImports':check_terminal_imports,
        'MealPlanRetryFailures':check_meal_retry_failures}
    other_checks={'CaptureNotAnalyzed':check_capture_not_analyzed,
        'RecipesNotAnalyzed':check_recipes_not_analyzed,'PartialActions':check_partial_actions,
        'CapacityRejections':check_capacity_rejections,'HaloVoiceFailures':check_halo_voice_outcomes,
        'MissingHaloMedia':check_missing_halo_media}
    def run(name,check,*args):
        try:
            result=check(*args)
            if isinstance(result,tuple):
                metrics[name],detail=result
                details[name]=detail[:25]
                if name=='MealPlanFailures':
                    metrics['MealPlansWithoutFallback']=sum(bool(r['without_plan']) for r in detail)
            else:
                metrics[name]=result
        except Exception as exc:
            failures.append(name)
            print(json.dumps({'evt':'ux_check_error','check':name,'error_type':type(exc).__name__}))
    conn=None
    try:
        conn=_conn()
    except Exception:
        failures.extend(db_checks)
    if conn:
        try:
            for name,check in db_checks.items():run(name,check,conn)
        finally:
            conn.close()
    for name,check in other_checks.items():run(name,check)
    metrics['MonitorCheckErrors']=len(failures)
    metrics['CoverageComplete']=int(not failures)
    data=[{'MetricName':name,'Value':float(value),'Unit':'Count'} for name,value in metrics.items()]
    # A failed metric publication is a failed invocation, never a healthy report.
    boto3.client('cloudwatch').put_metric_data(Namespace=NAMESPACE,MetricData=data)
    print(json.dumps({'evt':'ux_failure_scan','metrics':metrics,'failed_checks':failures,'details':details},default=str))
    return {'metrics':metrics,'failed_checks':failures,'check_errors':len(failures)}
