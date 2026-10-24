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
        # SHARED ONLY as of the 2026-08-18 cutover. The per-user read branch is GONE on purpose,
        # not flag-gated: `os.getenv('READ_SHARED_MEAL_PLAN', '')` defaulted to the per-user table,
        # so once writes moved to shared, a lost env var (a sam deploy has wiped CLI-set vars on
        # this stack before) would have silently served every user an empty meal plan with no
        # error. A flag whose failure mode is silent data disappearance is the hazard, not the
        # safety net. Rollback is redeploying the previous version.
        # MANDATORY owner_id filter — shared_meal_plan holds every owner's row.
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
            cur.execute(
                "SELECT _id, _owner, status, focus, explanation_title, explanation_paragraph, plan, error_message, _createdDate, _updatedDate, "
                "TIMESTAMPDIFF(SECOND, _updatedDate, NOW()) AS _generation_age_seconds "
                "FROM shared_meal_plan WHERE owner_id = %s",
                [owner],
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
        age_seconds = row.pop('_generation_age_seconds', None)
        # Twice the maximum Lambda invocation duration. This is a read projection:
        # keep the persisted plan intact, and let the existing POST retry path work.
        if row.get('status') == 'regenerating' and age_seconds is not None and age_seconds > 1800:
            row['status'] = 'failed'
            row['error_message'] = row.get('error_message') or 'This meal plan did not finish. Please try again.'
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

    from botocore.config import Config
    from botocore.exceptions import ClientError, ParamValidationError
    from generation_claim import claim_generation, DispatchRejected
    # Keep the database reservation brief. SDK retries could turn a lost first
    # acknowledgement into an apparent rejection on a later attempt.
    lambda_client = boto3.client('lambda', config=Config(
        connect_timeout=2, read_timeout=5, retries={'total_max_attempts': 1},
    ))

    def dispatch():
        try:
            response = lambda_client.invoke(
                FunctionName=generator_arn, InvocationType='Event',
                Payload=json.dumps({'owner': owner, 'focus': focus}),
            )
        except (ClientError, ParamValidationError) as exc:
            raise DispatchRejected(str(exc)) from exc
        if response.get('StatusCode') != 202:
            raise RuntimeError('Meal plan dispatch acknowledgement was not confirmed')

    conn = _mysql_conn()
    member_ids = _get_household_member_ids(conn, owner)
    should_invoke = claim_generation(conn, member_ids, focus, REGEN_DEBOUNCE_SECONDS, dispatch)

    if not should_invoke:
        return _success({
            'message': 'Meal plan regeneration already in progress. Poll GET /meal-plan/{owner} for status.',
            'owner': owner,
            'status': 'regenerating',
            'focus': focus,
        }, status=202)

    return _success({
        'message': 'Meal plan regeneration started. Poll GET /meal-plan/{owner} for status.',
        'owner': owner,
        'status': 'regenerating',
        'focus': focus,
    }, status=202)
