# app.py — trepo-recall-deep-dive Lambda.
#
# A USER-TRIGGERED, ASYNC, web-searching "deep dive" recall investigation over the WHOLE
# live kitchen. This is the SLOW/DEEP sibling of the instant matcher (trepo-recall-check,
# recall_check/app.py): instead of a form-based gpt-4.1-mini pass over a pre-merged feed,
# it runs a GPT-5 + web_search agent PER kitchen item (see engine.py) that actually reads
# recall/outbreak notices and reasons about brand/product/lot/outbreak scope.
#
# Because it's expensive + slow (up to ~10 min) it ONLY runs on an explicit user tap, and
# it runs ASYNCHRONOUSLY: the POST kicks the work via a Lambda self-invoke (Event) and
# returns 202 immediately; the client polls GET until status == done.
#
# API CONTRACT (pinned; iOS is built against this):
#   POST {base}/deep-dive/{owner}[?force=1]
#        -> 202 {status, started_at, eta_seconds, progress?:{done,total}}
#           (or 200 with a cached {status:"done", result:...} if a <12h result exists and
#            no ?force=1; or the in-flight running record if one is already going)
#   GET  {base}/deep-dive/{owner}
#        -> {status:"none"|"running"|"done"|"failed", started_at?, finished_at?,
#            progress?:{done,total},
#            result?:{summary, total_checked, generated_at,
#                     matches:[{kitchen_item_id, kitchen_item_name, brand,
#                               verdict:"recalled"|"possible", confidence, reasoning,
#                               source_url, link_type, recall_title?}]},
#            error?}
#   internal (self-invoke): {"action":"run","owner":...,"force":bool} -> does the work.
#   keep-warm: {"action":"ping"} or ?warm=1 -> {ok, pong} (no DB/DDB/model).
#
# AUTH: mirrors the instant recall-check EXACTLY — there is no bearer token; the caller is
# identified solely by the {owner} in the path (recall_check/app.py reads `owner` straight
# off pathParameters). iOS calls this identically.

import concurrent.futures
import json
import os
import sys
import time
import urllib.parse

import boto3
from botocore.config import Config as _BotoConfig

try:
    import pymysql
    import pymysql.cursors
    _PYMYSQL_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = _e

import engine

# --- DB (mirrored off RecipesGeneratorFunction, same as recall_check + ux-failure-monitor)
DB_HOST = os.getenv('DB_HOST', 'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASS = os.getenv('DB_PASS', '')
DB_NAME = os.getenv('DB_NAME', 'mysqlTutorial')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
KITCHEN_TABLE = os.getenv('SHARED_KITCHEN_TABLE', 'shared_kitchen')

# --- Job store (DynamoDB) + async self-invoke
JOB_TABLE = os.getenv('DEEP_DIVE_TABLE', 'trepo-recall-deep-dive')
SELF_FUNCTION_NAME = os.getenv('SELF_FUNCTION_NAME', '')  # falls back to context.function_name
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# --- Run shaping (time/cost bounds)
MAX_ITEMS = int(os.getenv('DEEP_MAX_ITEMS', '80'))          # investigate most-recent-first
# Per-item calls are I/O-bound (minutes each), so we oversubscribe threads heavily to fit
# 80 items inside the wall-clock budget. Raise if OpenAI concurrency limits allow.
MAX_WORKERS = int(os.getenv('DEEP_MAX_WORKERS', '24'))
RUN_DEADLINE_S = int(os.getenv('DEEP_RUN_DEADLINE_S', '660'))  # ~11 min hard stop (< fn timeout)
CACHE_TTL_S = int(os.getenv('DEEP_CACHE_TTL_S', str(12 * 3600)))  # reuse a done result <12h
RUNNING_STALE_S = int(os.getenv('DEEP_RUNNING_STALE_S', '900'))   # a 'running' older than this is dead
# Cost guard: don't START another (even forced) run within this window of the last run's
# start — bounds rapid forced re-runs / burst abuse of this paid endpoint. Short enough to
# never block a realistic re-run (a run itself takes minutes) but kills back-to-back taps.
MIN_RUN_INTERVAL_S = int(os.getenv('DEEP_MIN_RUN_INTERVAL_S', '120'))
ETA_SECONDS = int(os.getenv('DEEP_ETA_SECONDS', '600'))
DDB_TTL_DAYS = int(os.getenv('DEEP_DDB_TTL_DAYS', '30'))

_ddb = boto3.resource('dynamodb', region_name=AWS_REGION)
_lambda = boto3.client('lambda', region_name=AWS_REGION,
                       config=_BotoConfig(read_timeout=15, connect_timeout=5, retries={'max_attempts': 2}))


# ---------------------------------------------------------------------------
# Kitchen read (mirrors recall_check.read_live_kitchen: action='IN', newest first)
# ---------------------------------------------------------------------------
def _connect():
    if pymysql is None:
        raise RuntimeError('pymysql unavailable: %r' % (_PYMYSQL_IMPORT_ERROR,))
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT,
        connect_timeout=DB_CONNECT_TIMEOUT, read_timeout=DB_READ_TIMEOUT,
        cursorclass=pymysql.cursors.DictCursor, autocommit=True)


def read_live_kitchen(owner):
    sql = ("SELECT _id, product_name, brand, variant, category, product_image_url "
           "FROM `%s` WHERE owner_id=%%s AND `action`='IN' "
           "ORDER BY `_createdDate` DESC" % KITCHEN_TABLE)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (owner,))
            return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------
def _table():
    return _ddb.Table(JOB_TABLE)


def _now():
    return int(time.time())


def _iso(ts):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))


def _get_job(owner):
    try:
        r = _table().get_item(Key={'owner_id': owner})
        return r.get('Item')
    except Exception as e:
        print(json.dumps({'evt': 'deep_dive_ddb_get_failed', 'err': str(e)[:300]}), file=sys.stderr)
        return None


def _job_view(item):
    """Shape a stored DDB item into the pinned GET response (result is stored as JSON str)."""
    if not item:
        return {'status': 'none'}
    out = {'status': item.get('status', 'none')}
    for k in ('started_at', 'finished_at', 'error'):
        if item.get(k) not in (None, ''):
            out[k] = item[k]
    prog = item.get('progress')
    if isinstance(prog, dict):
        out['progress'] = {'done': int(prog.get('done', 0)), 'total': int(prog.get('total', 0))}
    rj = item.get('result_json')
    if rj:
        try:
            out['result'] = json.loads(rj)
        except Exception:
            pass
    return out


def _put_running(owner, total_hint=0):
    now = _now()
    _table().put_item(Item={
        'owner_id': owner,
        'status': 'running',
        'started_at': _iso(now),
        'started_ts': now,
        'progress': {'done': 0, 'total': int(total_hint)},
        'ttl': now + DDB_TTL_DAYS * 86400,
    })


def _set_total(owner, total):
    try:
        _table().update_item(
            Key={'owner_id': owner},
            UpdateExpression='SET progress.#t = :t',
            ExpressionAttributeNames={'#t': 'total'},
            ExpressionAttributeValues={':t': int(total)})
    except Exception as e:
        print(json.dumps({'evt': 'deep_dive_set_total_failed', 'err': str(e)[:200]}), file=sys.stderr)


def _bump_done(owner, done):
    try:
        _table().update_item(
            Key={'owner_id': owner},
            UpdateExpression='SET progress.#d = :d',
            ExpressionAttributeNames={'#d': 'done'},
            ExpressionAttributeValues={':d': int(done)})
    except Exception:
        pass  # progress is best-effort; never fail the run over a heartbeat write


def _finish(owner, result):
    now = _now()
    _table().update_item(
        Key={'owner_id': owner},
        UpdateExpression=('SET #s = :s, finished_at = :f, result_json = :r, '
                          'progress.#d = :d REMOVE #e'),
        ExpressionAttributeNames={'#s': 'status', '#d': 'done', '#e': 'error'},
        ExpressionAttributeValues={
            ':s': 'done', ':f': _iso(now), ':r': json.dumps(result),
            ':d': int(result.get('total_checked', 0))})


def _fail(owner, err):
    now = _now()
    try:
        _table().update_item(
            Key={'owner_id': owner},
            UpdateExpression='SET #s = :s, finished_at = :f, #e = :e',
            ExpressionAttributeNames={'#s': 'status', '#e': 'error'},
            ExpressionAttributeValues={':s': 'failed', ':f': _iso(now), ':e': str(err)[:500]})
    except Exception as e:
        print(json.dumps({'evt': 'deep_dive_fail_write_failed', 'err': str(e)[:200]}), file=sys.stderr)


# ---------------------------------------------------------------------------
# The investigation (internal action=='run')
# ---------------------------------------------------------------------------
def run_investigation(owner):
    t_start = time.time()
    items = read_live_kitchen(owner)
    total_kitchen = len(items)
    capped = False
    if len(items) > MAX_ITEMS:
        capped = True
        items = items[:MAX_ITEMS]  # newest-first already, so this keeps the freshest adds
        print(json.dumps({'evt': 'deep_dive_capped', 'owner': owner,
                          'kitchen': total_kitchen, 'cap': MAX_ITEMS}), file=sys.stderr)
    total = len(items)
    _set_total(owner, total)

    matches = []
    done = 0
    errors = 0
    deadline = t_start + RUN_DEADLINE_S

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, total)))
    fut_to_item = {ex.submit(engine.investigate_item, it): it for it in items}
    try:
        for fut in concurrent.futures.as_completed(fut_to_item, timeout=max(1, deadline - time.time())):
            it = fut_to_item[fut]
            try:
                v = fut.result()
            except Exception as e:
                errors += 1
                print(json.dumps({'evt': 'deep_dive_item_failed', 'owner': owner,
                                  'item': (it.get('product_name') or '')[:80],
                                  'err': str(e)[:200]}), file=sys.stderr)
                v = None
            done += 1
            if v and v.get('verdict') in ('recalled', 'possible'):
                matches.append({
                    'kitchen_item_id': it.get('_id'),
                    'kitchen_item_name': it.get('product_name'),
                    'brand': it.get('brand'),
                    'verdict': v['verdict'],
                    'confidence': round(float(v.get('confidence', 0.0)), 3),
                    'reasoning': v.get('reasoning', ''),
                    'source_url': v.get('source_url', ''),
                    'link_type': v.get('link_type', 'search'),
                    'recall_title': v.get('recall_title', ''),
                })
            if done % 3 == 0 or done == total:
                _bump_done(owner, done)
    except concurrent.futures.TimeoutError:
        print(json.dumps({'evt': 'deep_dive_run_deadline', 'owner': owner,
                          'done': done, 'total': total}), file=sys.stderr)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    # recalled first, then possible, each by descending confidence.
    matches.sort(key=lambda m: (0 if m['verdict'] == 'recalled' else 1, -m['confidence']))

    n_recalled = sum(1 for m in matches if m['verdict'] == 'recalled')
    n_possible = sum(1 for m in matches if m['verdict'] == 'possible')
    if n_recalled:
        summary = ("%d item%s in your kitchen may be under an active recall or outbreak."
                   % (n_recalled, '' if n_recalled == 1 else 's'))
    elif n_possible:
        summary = ("%d item%s worth checking against current recalls/outbreaks."
                   % (n_possible, '' if n_possible == 1 else 's'))
    else:
        summary = "Good news — nothing in your kitchen matched a current recall or outbreak."

    result = {
        'summary': summary,
        'total_checked': done,
        'generated_at': _iso(int(time.time())),
        'matches': matches,
    }
    if capped:
        result['capped'] = True
        result['kitchen_size'] = total_kitchen
    if done < total:
        # The wall-clock deadline fired before every item finished — surface it so the UI
        # can show "checked N of M" rather than implying a clean bill of health.
        result['incomplete'] = True
        result['total_items'] = total
    print(json.dumps({'evt': 'deep_dive_done', 'owner': owner, 'checked': done,
                      'matches': len(matches), 'errors': errors,
                      'secs': round(time.time() - t_start, 1)}), file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# HTTP helpers + routing
# ---------------------------------------------------------------------------
def _resp(code, body):
    return {'statusCode': code,
            'headers': {'content-type': 'application/json',
                        'access-control-allow-origin': '*'},
            'body': json.dumps(body, ensure_ascii=False, default=str)}


def _method(event):
    m = event.get('httpMethod')
    if m:
        return m.upper()
    rc = event.get('requestContext') or {}
    return ((rc.get('http') or {}).get('method') or 'GET').upper()


def _owner(event):
    params = event.get('pathParameters') or {}
    if params.get('owner'):
        return urllib.parse.unquote(params['owner'])
    # Function URL / raw path fallback: .../deep-dive/{owner}
    raw = event.get('rawPath') or event.get('path') or ''
    parts = [p for p in raw.split('/') if p]
    if parts:
        return urllib.parse.unquote(parts[-1])
    qs = event.get('queryStringParameters') or {}
    return qs.get('owner')


def _truthy(v):
    return str(v).lower() in ('1', 'true', 'yes')


def _self_invoke_run(owner, force, context):
    fn = SELF_FUNCTION_NAME or (getattr(context, 'function_name', None) or '')
    if not fn:
        raise RuntimeError('no function name for self-invoke')
    _lambda.invoke(
        FunctionName=fn,
        InvocationType='Event',  # async fire-and-forget; POST returns 202 immediately
        Payload=json.dumps({'action': 'run', 'owner': owner, 'force': bool(force)}).encode('utf-8'))


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    event = event or {}
    action = event.get('action') or (event.get('detail') or {}).get('action')

    # keep-warm ping (kills cold-start on first user tap) — no DB/DDB/model.
    qs = event.get('queryStringParameters') or {}
    if action == 'ping' or _truthy(qs.get('warm', '')):
        return {'ok': True, 'pong': True}

    # internal async worker
    if action == 'run':
        owner = event.get('owner')
        if not owner:
            return {'ok': False, 'error': 'no owner'}
        try:
            result = run_investigation(owner)
            _finish(owner, result)
            return {'ok': True, 'owner': owner, 'matches': len(result.get('matches', []))}
        except Exception as e:
            print(json.dumps({'evt': 'deep_dive_run_failed', 'owner': owner,
                              'err': str(e)[:400]}), file=sys.stderr)
            _fail(owner, str(e)[:400])
            return {'ok': False, 'error': str(e)[:400]}

    # HTTP
    owner = _owner(event)
    if not owner:
        return _resp(400, {'error': 'missing_owner'})
    method = _method(event)

    if method == 'GET':
        return _resp(200, _job_view(_get_job(owner)))

    if method in ('POST', 'PUT'):
        force = _truthy(qs.get('force', ''))
        job = _get_job(owner)
        now = _now()

        # Already running (and not stale) -> return it, never double-run.
        if job and job.get('status') == 'running':
            started_ts = int(job.get('started_ts') or 0)
            if now - started_ts < RUNNING_STALE_S:
                view = _job_view(job)
                view['eta_seconds'] = max(0, ETA_SECONDS - (now - started_ts))
                return _resp(202, view)
            # else: stale 'running' (a prior invoke died) -> fall through and restart.

        # Fresh cached result -> serve it unless forced.
        if not force and job and job.get('status') == 'done':
            fin = job.get('finished_at')
            fin_ts = 0
            if fin:
                try:
                    fin_ts = int(time.mktime(time.strptime(fin, '%Y-%m-%dT%H:%M:%SZ')))
                except Exception:
                    fin_ts = 0
            if fin_ts and (now - fin_ts) < CACHE_TTL_S:
                return _resp(200, _job_view(job))

        # Cost guard: reject a fresh start (even forced) if a run started very recently, so
        # rapid re-taps / burst abuse can't stack up paid runs. Returns the current state.
        if job:
            last_start = int(job.get('started_ts') or 0)
            if last_start and (now - last_start) < MIN_RUN_INTERVAL_S:
                view = _job_view(job)
                code = 202 if view.get('status') == 'running' else 200
                return _resp(code, view)

        # Start a new run.
        try:
            _put_running(owner)
            _self_invoke_run(owner, force, context)
        except Exception as e:
            print(json.dumps({'evt': 'deep_dive_start_failed', 'owner': owner,
                              'err': str(e)[:300]}), file=sys.stderr)
            _fail(owner, 'failed to start: ' + str(e)[:200])
            return _resp(500, {'status': 'failed', 'error': 'could not start investigation'})

        return _resp(202, {'status': 'running', 'started_at': _iso(now),
                           'eta_seconds': ETA_SECONDS, 'progress': {'done': 0, 'total': 0}})

    return _resp(405, {'error': 'method_not_allowed'})
