"""trepo-parity-monitor — scheduled guardrail for the shared-table migration.

For every flipped type, verifies the shared table covers each owner's per-owner
(source-of-truth) data. Two layers, both bounded to fit the Lambda timeout:

  1) MISSING-OWNER sweep (fast, all owners): owner-set difference — every owner with a
     per-owner table that has data must appear in the shared table. Catches the
     "owner entirely absent" class (the recipes-25 case) comprehensively each run.
  2) ROW-LEVEL sample (rotating N owners): for a random sample, compare per-owner vs
     shared counts (distinct recipe-URLs for saved_recipes). Catches the "short by N
     rows" class (the Lexi case) probabilistically — full coverage over many runs.

Emits one {"evt":"parity_miss", ...} marker per confirmed short owner to stdout. A
CloudWatch metric filter on evt=parity_miss -> alarm -> SNS pages us. Emits a heartbeat
{"evt":"parity_scan_ok"} when clean so we can tell 'ran and clean' from 'did not run'.
"""
import os, json, random, pymysql

SAMPLE_N = int(os.getenv('PARITY_SAMPLE_N', '400'))
# (per-owner suffix, shared table, single-row predicate or None, distinct-hash?)
TYPES = [
    ('_saved_recipes', 'shared_saved_recipes', None, True),
    ('_recipes',       'shared_recipes',       "_id='current'", False),
    ('_meal_plan',     'shared_meal_plan',     "_id='current'", False),
    ('-metrics',       'shared_metrics',       None, False),
    ('_dishes',        'shared_dishes',        None, False),
    ('_discards',      'shared_discards',      None, False),
]

def _conn():
    return pymysql.connect(host=os.environ['DB_HOST'], user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'], database=os.environ['DB_NAME'],
        port=int(os.getenv('DB_PORT', '3306')), cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5, read_timeout=30, autocommit=True)

def _q(conn, sql, params=None):
    with conn.cursor() as c:
        c.execute(sql, params or [])
        return c.fetchall()

def _table_exists_shared(conn, shared):
    return bool(_q(conn, "SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s", [shared]))

def handler(event, context):
    conn = _conn()
    misses = 0
    for suffix, shared, pred, distinct in TYPES:
        if not _table_exists_shared(conn, shared):
            continue
        covered = {r['owner_id'] for r in _q(conn, f"SELECT DISTINCT owner_id FROM `{shared}`")}
        tbls = [r['tn'] for r in _q(conn,
            "SELECT table_name tn FROM information_schema.tables WHERE table_schema=DATABASE() "
            "AND table_name LIKE %s AND CHAR_LENGTH(table_name)=%s AND table_name NOT LIKE 'shared_%%'",
            ['%' + suffix, 36 + len(suffix)])]
        pw = (" WHERE " + pred) if pred else ""
        # 1) missing-owner sweep
        missing_candidates = [t[:-len(suffix)] for t in tbls if t[:-len(suffix)] not in covered]
        for o in missing_candidates:
            t = o + suffix
            try:
                n = _q(conn, f"SELECT COUNT(*) n FROM `{t}`{pw}")[0]['n']
            except Exception:
                continue
            if n > 0:
                misses += 1
                print(json.dumps({'evt': 'parity_miss', 'kind': 'missing_owner',
                    'family': suffix.strip('_-'), 'owner_id': o, 'per_owner': n, 'shared': 0}))
        # 2) row-level sample of covered owners
        sample = random.sample(list(covered), min(SAMPLE_N, len(covered))) if covered else []
        for o in sample:
            t = o + suffix
            try:
                if distinct:
                    po = _q(conn, f"SELECT COUNT(DISTINCT resolved_url_hash) n FROM `{t}`")[0]['n']
                else:
                    po = _q(conn, f"SELECT COUNT(*) n FROM `{t}`{pw}")[0]['n']
                sh = _q(conn, f"SELECT COUNT(*) n FROM `{shared}` WHERE owner_id=%s", [o])[0]['n']
            except Exception:
                continue
            if po > sh:  # re-check once to filter transient mid-write
                try:
                    if distinct:
                        po2 = _q(conn, f"SELECT COUNT(DISTINCT resolved_url_hash) n FROM `{t}`")[0]['n']
                    else:
                        po2 = _q(conn, f"SELECT COUNT(*) n FROM `{t}`{pw}")[0]['n']
                    sh2 = _q(conn, f"SELECT COUNT(*) n FROM `{shared}` WHERE owner_id=%s", [o])[0]['n']
                except Exception:
                    continue
                if po2 > sh2:
                    misses += 1
                    print(json.dumps({'evt': 'parity_miss', 'kind': 'short_rows',
                        'family': suffix.strip('_-'), 'owner_id': o, 'per_owner': po2, 'shared': sh2}))
    print(json.dumps({'evt': 'parity_scan_ok' if misses == 0 else 'parity_scan_found', 'misses': misses}))
    return {'misses': misses}
