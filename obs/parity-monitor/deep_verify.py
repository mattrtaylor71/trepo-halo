#!/usr/bin/env python3
"""Deep parity verifier: row-by-row, column-by-column checksum compare between every
per-owner table and its shared_* rows. Includes _createdDate/_updatedDate in the
checksum. Reports: missing rows (LOSS), extra rows (GHOSTS), content mismatches
(with per-column diff drill-down incl. timestamps), and per-owner columns absent
from shared (never-migrated data).

Usage: deep_verify.py <suffix> <shared_table> <mode>
  mode: full      — compare every row by _id (metrics/dishes/discards)
        current   — compare only the _id='current' row (recipes/meal_plan);
                    also counts non-current per-owner rows (not user-visible)
        by_hash   — saved_recipes: shared dedupes on (owner_id,resolved_url_hash);
                    shared row must match ONE of the per-owner rows w/ that hash;
                    null-hash rows compared by _id
"""
import boto3, pymysql, sys, time, json

SUFFIX, SHARED, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
s = boto3.Session(profile_name='trepo-dev', region_name='us-east-1')
E = s.client('lambda').get_function_configuration(
    FunctionName='trepo-grocery-backend-dev-RecipesApiFunction-zjwZ1RP9bUAQ')['Environment']['Variables']

def cn():
    return pymysql.connect(host=E['DB_HOST'], user=E['DB_USER'], password=E['DB_PASS'],
        database=E['DB_NAME'], port=int(E['DB_PORT']), cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10, read_timeout=60, autocommit=True)

conn = cn()

def q(sql, params=None):
    global conn
    for a in range(3):
        try:
            with conn.cursor() as c:
                c.execute(sql, params or []); return c.fetchall()
        except (pymysql.err.OperationalError, pymysql.err.InterfaceError):
            try: conn.close()
            except: pass
            time.sleep(1); conn = cn()
    raise RuntimeError('query failed 3x: ' + sql[:120])

shared_cols = {r['cn'] for r in q(
    "SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s", [SHARED])}
tbls = [r['tn'] for r in q(
    "SELECT table_name tn FROM information_schema.tables WHERE table_schema=DATABASE() "
    "AND table_name LIKE %s AND CHAR_LENGTH(table_name)=%s AND table_name NOT LIKE 'shared\\_%%'",
    ['%' + SUFFIX, 36 + len(SUFFIX)])]
print(f"[{SUFFIX}] {len(tbls)} per-owner tables | shared cols={len(shared_cols)}", flush=True)

# ---- schema audit: per-owner columns that DON'T exist in shared (never migrated) ----
col_rows = q(
    "SELECT table_name tn, COLUMN_NAME cnm FROM information_schema.columns WHERE table_schema=DATABASE() "
    "AND table_name LIKE %s AND CHAR_LENGTH(table_name)=%s AND table_name NOT LIKE 'shared\\_%%'",
    ['%' + SUFFIX, 36 + len(SUFFIX)])
from collections import defaultdict
tbl_cols = defaultdict(set)
for r in col_rows: tbl_cols[r['tn']].add(r['cnm'])
unmigrated = defaultdict(int)   # col -> number of tables having it
for t, cs in tbl_cols.items():
    for c in cs - shared_cols:
        unmigrated[c] += 1
if unmigrated:
    print(f"[{SUFFIX}] SCHEMA: per-owner columns ABSENT from shared: " +
          ", ".join(f"{c}(x{n})" for c, n in sorted(unmigrated.items(), key=lambda x: -x[1])), flush=True)
else:
    print(f"[{SUFFIX}] SCHEMA: every per-owner column exists in shared", flush=True)

NULLTOK = '~~NULL~~'
def checksum_select(cols, table, where='', params=None):
    expr = ", ".join(f"COALESCE(CAST(`{c}` AS CHAR), '{NULLTOK}')" for c in cols)
    rows = q(f"SELECT `_id` id, MD5(CONCAT_WS('|', {expr})) h FROM `{table}` {where}", params)
    return {r['id']: r['h'] for r in rows}

def col_diff(cols, po_table, owner, _id):
    """Drill into which columns differ for one row."""
    po = q(f"SELECT * FROM `{po_table}` WHERE `_id`=%s", [_id])
    sh = q(f"SELECT * FROM `{SHARED}` WHERE owner_id=%s AND `_id`=%s", [owner, _id])
    if not po or not sh: return ['<row fetch failed>']
    po, sh = po[0], sh[0]
    diffs = []
    for c in cols:
        a, b = po.get(c), sh.get(c)
        if str(a) != str(b):
            diffs.append(f"{c}: per-owner={str(a)[:60]!r} shared={str(b)[:60]!r}")
    return diffs

summary = {'tables_scanned': 0, 'tables_with_data': 0, 'rows_compared': 0,
           'missing_in_shared': 0, 'ghosts_in_shared': 0, 'content_mismatch': 0,
           'ts_mismatch_rows': 0, 'noncurrent_rows_skipped': 0, 'errors': 0}
detail = []
t0 = time.time()

for i, t in enumerate(tbls):
    owner = t[:-len(SUFFIX)]
    summary['tables_scanned'] += 1
    try:
        cols = sorted((tbl_cols[t] & shared_cols) - {'owner_id'})
        if '_id' not in cols:
            continue
        if MODE == 'current':
            po = checksum_select(cols, t, "WHERE `_id`='current'")
            nc = q(f"SELECT COUNT(*) n FROM `{t}` WHERE `_id`<>'current'")[0]['n']
            summary['noncurrent_rows_skipped'] += nc
        else:
            po = checksum_select(cols, t)
        if not po:
            continue
        summary['tables_with_data'] += 1
        summary['rows_compared'] += len(po)
        if MODE == 'current':
            sh = checksum_select(cols, SHARED, "WHERE owner_id=%s AND `_id`='current'", [owner])
        else:
            sh = checksum_select(cols, SHARED, "WHERE owner_id=%s", [owner])

        if MODE == 'by_hash':
            # group per-owner rows by resolved_url_hash; shared keeps 1 per hash.
            hpo = q(f"SELECT `_id` id, resolved_url_hash rh FROM `{t}`")
            hsh = q(f"SELECT `_id` id, resolved_url_hash rh FROM `{SHARED}` WHERE owner_id=%s", [owner])
            po_by_hash = defaultdict(list); po_null = {}
            for r in hpo:
                if r['rh'] is None: po_null[r['id']] = po.get(r['id'])
                else: po_by_hash[r['rh']].append(po.get(r['id']))
            sh_by_hash = {}; sh_null = {}
            for r in hsh:
                if r['rh'] is None: sh_null[r['id']] = sh.get(r['id'])
                else: sh_by_hash[r['rh']] = sh.get(r['id'])
            for rh, sums in po_by_hash.items():
                if rh not in sh_by_hash:
                    summary['missing_in_shared'] += 1
                    detail.append(f"MISSING {owner[:8]} hash={str(rh)[:12]}")
                elif sh_by_hash[rh] not in sums:
                    summary['content_mismatch'] += 1
                    detail.append(f"CONTENT {owner[:8]} hash={str(rh)[:12]} (shared row matches NO per-owner twin)")
            for rh in set(sh_by_hash) - set(po_by_hash):
                summary['ghosts_in_shared'] += 1
                detail.append(f"GHOST {owner[:8]} hash={str(rh)[:12]}")
            for _id, h in po_null.items():
                if _id not in sh_null:
                    summary['missing_in_shared'] += 1; detail.append(f"MISSING {owner[:8]} id={_id}(nullhash)")
                elif sh_null[_id] != h:
                    summary['content_mismatch'] += 1
                    d = col_diff(cols, t, owner, _id)
                    if any(x.startswith('_createdDate') or x.startswith('_updatedDate') for x in d):
                        summary['ts_mismatch_rows'] += 1
                    detail.append(f"CONTENT {owner[:8]} id={_id}: " + "; ".join(d[:4]))
            for _id in set(sh_null) - set(po_null):
                summary['ghosts_in_shared'] += 1; detail.append(f"GHOST {owner[:8]} id={_id}(nullhash)")
        else:
            for _id, h in po.items():
                if _id not in sh:
                    summary['missing_in_shared'] += 1
                    detail.append(f"MISSING {owner[:8]} id={_id}")
                elif sh[_id] != h:
                    summary['content_mismatch'] += 1
                    d = col_diff(cols, t, owner, _id)
                    if any(x.startswith('_createdDate') or x.startswith('_updatedDate') for x in d):
                        summary['ts_mismatch_rows'] += 1
                    detail.append(f"CONTENT {owner[:8]} id={_id}: " + "; ".join(d[:4]))
            for _id in set(sh) - set(po):
                summary['ghosts_in_shared'] += 1
                detail.append(f"GHOST {owner[:8]} id={_id}")
    except Exception as e:
        summary['errors'] += 1
        detail.append(f"ERR {owner[:8]}: {str(e)[:120]}")
    if (i + 1) % 1000 == 0:
        try: conn.close()
        except: pass
        conn = cn()
        print(f"[{SUFFIX}] {i+1}/{len(tbls)} data={summary['tables_with_data']} rows={summary['rows_compared']} "
              f"miss={summary['missing_in_shared']} ghost={summary['ghosts_in_shared']} "
              f"content={summary['content_mismatch']} {time.time()-t0:.0f}s", flush=True)

print(f"[{SUFFIX}] SUMMARY " + json.dumps(summary), flush=True)
for d in detail[:80]:
    print(f"[{SUFFIX}]   " + d, flush=True)
if len(detail) > 80:
    print(f"[{SUFFIX}]   ... +{len(detail)-80} more", flush=True)
verdict = 'PASS' if (summary['missing_in_shared'] == 0 and summary['content_mismatch'] == 0
                     and summary['ghosts_in_shared'] == 0 and summary['errors'] == 0) else 'FAIL'
print(f"[{SUFFIX}] VERDICT {verdict} ({time.time()-t0:.0f}s)", flush=True)
