"""Durable, fenced meal-plan retries. Schema is installed by migration, not GETs.

The existing household named locks still serialize workers. This row preserves a
retry after a worker dies and prevents an old delivery from publishing over a new
attempt. Dispatch is at-least-once; claims and publication are token-guarded.
"""
import json
import uuid

TABLE = 'meal_plan_generation_jobs'
MAX_ATTEMPTS = 3

class Superseded(Exception):
    pass

class RetryExhausted(Exception):
    pass

def _get(conn, scope):
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM '+TABLE+' WHERE scope_id=%s FOR UPDATE', [scope])
        return cur.fetchone()

def _ensure_exclusive_row(conn, scope):
    # INSERT IGNORE takes a shared lock on an existing unique key. Concurrent
    # admissions then deadlock when both upgrade it with SELECT FOR UPDATE.
    # A no-op duplicate UPDATE obtains the exclusive lock from the outset and
    # preserves the existing token, input hash and bounded attempt count.
    with conn.cursor() as cur:
        cur.execute('INSERT INTO '+TABLE+" (scope_id,state,attempts) VALUES (%s,'idle',0) "
                    'ON DUPLICATE KEY UPDATE scope_id=scope_id', [scope])

def begin(conn, scope, payload, retry_token=None, lease_seconds=360):
    conn.begin()
    try:
        _ensure_exclusive_row(conn, scope)
        old = _get(conn, scope)
        if retry_token and (old['run_token'] != retry_token or old['state'] not in ('pending','running')):
            conn.rollback()
            return None
        token = str(uuid.uuid4())
        continuing = bool(retry_token or old['state'] in ('pending','running','failed'))
        attempts = int(old['attempts']) + 1 if continuing else 1
        with conn.cursor() as cur:
            cur.execute('UPDATE '+TABLE+' SET run_token=%s,state=\'running\',payload=%s,attempts=%s,'
                        'input_hash=%s,due_at=NULL,lease_until=DATE_ADD(UTC_TIMESTAMP(),INTERVAL %s SECOND),last_error=NULL WHERE scope_id=%s',
                        [token,json.dumps(payload),attempts,old.get('input_hash') if continuing else None,lease_seconds,scope])
        conn.commit()
        return {'scope':scope,'token':token,'attempts':attempts}
    except BaseException:
        conn.rollback()
        raise

def require_current(conn, claim):
    row = _get(conn, claim['scope'])
    if not row or row['run_token'] != claim['token'] or row['state'] != 'running':
        raise Superseded('meal_plan_attempt_superseded')
    return row

def bind_inputs(conn, claim, input_hash):
    conn.begin()
    try:
        row = require_current(conn, claim)
        # A genuinely newer revision gets its own bounded retry budget.
        attempts = 1 if row.get('input_hash') and row['input_hash'] != input_hash else row['attempts']
        with conn.cursor() as cur:
            cur.execute('UPDATE '+TABLE+' SET input_hash=%s,attempts=%s WHERE scope_id=%s', [input_hash,attempts,claim['scope']])
        conn.commit()
        if attempts > MAX_ATTEMPTS:
            raise RetryExhausted("retry_budget_exhausted")
    except BaseException:
        conn.rollback()
        raise

def finish(conn, claim, state='ready', error=None):
    """Called inside the SAME transaction as the plan/status write."""
    require_current(conn, claim)
    with conn.cursor() as cur:
        cur.execute('UPDATE '+TABLE+' SET state=%s,due_at=NULL,lease_until=NULL,last_error=%s WHERE scope_id=%s',
                    [state,error,claim['scope']])

def retry(conn, claim, reason, payload=None, delay_seconds=15):
    """Called with the plan status transaction open; returns whether retry remains."""
    row = require_current(conn, claim)
    pending = int(row['attempts']) < MAX_ATTEMPTS
    with conn.cursor() as cur:
        cur.execute('UPDATE '+TABLE+' SET state=%s,lease_until=NULL,'
                    'due_at=IF(%s,DATE_ADD(UTC_TIMESTAMP(),INTERVAL %s SECOND),NULL),last_error=%s,payload=%s WHERE scope_id=%s',
                    ['pending' if pending else 'failed',pending,delay_seconds,str(reason)[:300],
                     json.dumps(payload) if payload is not None else row['payload'],claim['scope']])
    return pending

def due(conn, limit=50):
    with conn.cursor() as cur:
        cur.execute('SELECT scope_id,run_token,payload FROM '+TABLE+
                    ' WHERE (state=\'pending\' AND due_at<=UTC_TIMESTAMP()) OR '
                    '(state=\'running\' AND lease_until<=UTC_TIMESTAMP()) ORDER BY updated_at LIMIT %s', [limit])
        return cur.fetchall()


def supersede_pending(conn, scope, payload):
    """A request arriving under an active named lock must survive that worker."""
    conn.begin()
    try:
        _ensure_exclusive_row(conn, scope)
        with conn.cursor() as cur:
            cur.execute('UPDATE '+TABLE+" SET run_token=%s,state='pending',payload=%s,"
                        'due_at=DATE_ADD(UTC_TIMESTAMP(),INTERVAL 15 SECOND),lease_until=NULL WHERE scope_id=%s',
                        [str(uuid.uuid4()),json.dumps(payload),scope])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def admission_failed(conn, scope, token, permanent=False):
    """Consent denial/unavailability cannot leave an endless scheduled retry."""
    conn.begin()
    try:
        row = _get(conn, scope)
        if not row or row['run_token'] != token or row['state'] not in ('pending','running'):
            conn.rollback()
            return
        attempts = int(row['attempts']) + 1
        stopped = permanent or attempts >= MAX_ATTEMPTS
        with conn.cursor() as cur:
            cur.execute('UPDATE '+TABLE+" SET state=%s,attempts=%s,last_error=%s,lease_until=NULL,"
                        'due_at=IF(%s,NULL,DATE_ADD(UTC_TIMESTAMP(),INTERVAL 60 SECOND)) WHERE scope_id=%s',
                        ['blocked' if stopped else 'pending',attempts,
                         'consent_required' if permanent else 'consent_unavailable',stopped,scope])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
