# meal_plan_api/app.py - GET meal plan for owner, POST to regenerate with optional focus
# Uses per-owner table: {owner}_meal_plan (single row per table, _id = 'current')
import os
# Force republish after dependency-layer restore.
import json
import re
import boto3
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
REGEN_DEBOUNCE_SECONDS = 300


def _get_db_config():
    missing = [name for name in _DB_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {', '.join(missing)}")
    port_raw = os.getenv('DB_PORT', '3306')
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid DB_PORT value: {port_raw}")
    return {
        'host': os.getenv('DB_HOST'),
        'port': port,
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASS'),
        'database': os.getenv('DB_NAME'),
    }


_conn = None


def _mysql_conn():
    global _conn
    if _conn is not None:
        try:
            _conn.ping(reconnect=True)
            return _conn
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    if pymysql is None:
        raise RuntimeError(f"pymysql import failed: {_PYMYSQL_IMPORT_ERROR}")
    config = _get_db_config()
    _conn = pymysql.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5')),
        read_timeout=int(os.getenv('DB_READ_TIMEOUT_SECONDS', '30')),
        write_timeout=int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '30')),
    )
    return _conn


def json_serial(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, dict):
        return {k: json_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serial(item) for item in obj]
    return str(obj)


def _cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }


def _success(body, status=200):
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps(body, default=json_serial)}


def _error(status, message):
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps({'error': message})}


def _meal_plan_table(owner):
    """Safe table name: {owner}_meal_plan. Owner must be alphanumeric, hyphen, underscore only."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_meal_plan"


# ---- Shared-table migration M4: dual-write the meal_plan 'current' cache row to shared_meal_plan ----
# One-row-per-owner cache (shared PK = owner_id). Off by default (DUAL_WRITE_MEAL_PLAN).
# Copies the per-user 'current' row into shared_meal_plan via INSERT...SELECT so the
# mirror is a faithful copy of the latest write; ON DUPLICATE KEY UPDATE (owner_id PK)
# re-syncs on every state transition. Non-blocking: errors swallowed + surfaced as a
# metric-filterable {evt:'dual_write_miss', family:'meal_plan'} marker. (Duplicated in
# meal_plan_generator — keep byte-equivalent; follow-up: hoist to a shared layer.)
DUAL_WRITE_MEAL_PLAN = os.getenv('DUAL_WRITE_MEAL_PLAN', 'false').lower() == 'true'
_SHARED_MEAL_PLAN_TABLE = 'shared_meal_plan'
_SHARED_MEAL_PLAN_COLS = (
    '_id', '_owner', 'status', 'focus', 'explanation_title', 'explanation_paragraph',
    'plan', 'error_message', '_createdDate', '_updatedDate',
)


def _dual_write_meal_plan_to_shared(conn, owner):
    if not DUAL_WRITE_MEAL_PLAN:
        return
    try:
        table = _meal_plan_table(owner)
        cols = ', '.join(f'`{c}`' for c in _SHARED_MEAL_PLAN_COLS)
        src = ', '.join(f's.`{c}`' for c in _SHARED_MEAL_PLAN_COLS)
        upd = ', '.join(f'`{c}`=VALUES(`{c}`)' for c in _SHARED_MEAL_PLAN_COLS)
        sql = (
            f"INSERT INTO `{_SHARED_MEAL_PLAN_TABLE}` (`owner_id`, {cols}) "
            f"SELECT %s, {src} FROM `{table}` s WHERE s.`_id`='current' "
            f"ON DUPLICATE KEY UPDATE `owner_id`=VALUES(`owner_id`), {upd}"
        )
        with conn.cursor() as cur:
            cur.execute(sql, (owner,))
        conn.commit()
    except Exception as exc:
        try:
            import sys
            print(json.dumps({
                'evt': 'dual_write_miss', 'family': 'meal_plan',
                'owner_id': str(owner) if owner is not None else None,
                'error': (str(exc)[:500] if exc is not None else ''),
            }), file=sys.stderr)
        except Exception:
            pass


def _sanitize_user_id(user_id):
    return re.sub(r'[^a-zA-Z0-9_-]', '', (user_id or ''))


def _get_household_member_ids(conn, acting_user_id):
    safe_user_id = _sanitize_user_id(acting_user_id)
    if not safe_user_id:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT owner_id FROM new_users WHERE user_id = %s LIMIT 1", [safe_user_id])
        row = cur.fetchone() or {}
        household_id = row.get('owner_id')
        if not household_id:
            return [safe_user_id]
        cur.execute(
            "SELECT user_id FROM new_users WHERE owner_id = %s ORDER BY created_at ASC, user_id ASC",
            [household_id]
        )
        members = [_sanitize_user_id(item.get('user_id')) for item in (cur.fetchall() or [])]
    members = [member for member in members if member]
    return list(dict.fromkeys(members)) or [safe_user_id]


def _ensure_meal_plan_table(conn, owner):
    table = _meal_plan_table(owner)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                _id VARCHAR(36) PRIMARY KEY,
                _owner VARCHAR(36) NOT NULL,
                status ENUM('ready', 'regenerating', 'failed', 'empty') NOT NULL DEFAULT 'empty',
                focus TEXT,
                explanation_title VARCHAR(255),
                explanation_paragraph TEXT,
                plan JSON,
                error_message TEXT,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = (path_params.get('owner') or '').strip()
        if not owner:
            return _error(400, 'Missing owner parameter')

        if http_method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

        if http_method == 'GET':
            return _get_meal_plan(owner)
        if http_method == 'POST':
            body = json.loads(event.get('body', '{}') or '{}')
            return _post_meal_plan(owner, body)
        return _error(405, f'Method {http_method} not allowed')
    except json.JSONDecodeError as e:
        return _error(400, f'Invalid JSON: {str(e)}')
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return _error(500, str(e))


def _get_meal_plan(owner):
    conn = _mysql_conn()
    try:
        table = _meal_plan_table(owner)
        # Shared-table read cutover (reversible via env flag, default off = per-owner).
        # MANDATORY owner_id filter — shared_meal_plan holds every owner's row.
        use_shared = os.getenv('READ_SHARED_MEAL_PLAN', '').strip().lower() == 'true'
        empty_response = {
            'owner': owner,
            'status': 'empty',
            'focus': 'great overall health',
            'explanation_title': None,
            'explanation_paragraph': None,
            'plan': [],
            'error_message': None,
            '_createdDate': None,
            '_updatedDate': None,
        }
        with conn.cursor() as cur:
            if use_shared:
                cur.execute(
                    "SELECT _id, _owner, status, focus, explanation_title, explanation_paragraph, plan, error_message, _createdDate, _updatedDate "
                    "FROM shared_meal_plan WHERE owner_id = %s",
                    [owner],
                )
                row = cur.fetchone()
            else:
                cur.execute("""
                    SELECT COUNT(*) AS n FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table])
                if cur.fetchone()['n'] == 0:
                    return _success(empty_response)
                cur.execute(
                    f"SELECT _id, _owner, status, focus, explanation_title, explanation_paragraph, plan, error_message, _createdDate, _updatedDate FROM `{table}` WHERE _id = 'current'",
                )
                row = cur.fetchone()
        if not row:
            return _success({
                'owner': owner,
                'status': 'empty',
                'focus': 'great overall health',
                'explanation_title': None,
                'explanation_paragraph': None,
                'plan': [],
                'error_message': None,
                '_createdDate': None,
                '_updatedDate': None,
            })
        out = {k: json_serial(v) for k, v in row.items()}
        out['owner'] = owner  # response uses 'owner' for API consistency
        if out.get('plan') is None:
            out['plan'] = []
        return _success(out)
    finally:
        pass


def _post_meal_plan(owner, body):
    focus = (body.get('focus') or body.get('focus_text') or 'great overall health').strip() or 'great overall health'
    generator_arn = os.getenv('MEAL_PLAN_GENERATOR_ARN')
    if not generator_arn:
        return _error(500, 'Meal plan generator not configured')

    conn = _mysql_conn()
    should_invoke = False
    try:
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            for member_id in member_ids:
                _ensure_meal_plan_table(conn, member_id)
                table = _meal_plan_table(member_id)
                cur.execute(f"SELECT _id, status, _updatedDate FROM `{table}` WHERE _id = 'current'")
                row = cur.fetchone()
                is_recent_regen = False
                if row and row.get('status') == 'regenerating':
                    updated_at = row.get('_updatedDate')
                    if updated_at and (datetime.utcnow() - updated_at).total_seconds() < REGEN_DEBOUNCE_SECONDS:
                        is_recent_regen = True
                if row:
                    if not is_recent_regen:
                        cur.execute(
                            f"UPDATE `{table}` SET status = 'regenerating', focus = %s, error_message = NULL, _updatedDate = NOW() WHERE _id = 'current'",
                            (focus,)
                        )
                        should_invoke = True
                else:
                    cur.execute(
                        f"INSERT INTO `{table}` (_id, _owner, status, focus) VALUES ('current', %s, 'regenerating', %s)",
                        (member_id, focus)
                    )
                    should_invoke = True
        conn.commit()
        for member_id in member_ids:
            _dual_write_meal_plan_to_shared(conn, member_id)
    finally:
        pass

    if not should_invoke:
        return _success({
            'message': 'Meal plan regeneration already in progress. Poll GET /meal-plan/{owner} for status.',
            'owner': owner,
            'status': 'regenerating',
            'focus': focus,
        }, status=202)

    lambda_client = boto3.client('lambda')
    lambda_client.invoke(
        FunctionName=generator_arn,
        InvocationType='Event',
        Payload=json.dumps({'owner': owner, 'focus': focus})
    )

    return _success({
        'message': 'Meal plan regeneration started. Poll GET /meal-plan/{owner} for status.',
        'owner': owner,
        'status': 'regenerating',
        'focus': focus,
    }, status=202)
