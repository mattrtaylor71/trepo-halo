# recipes_api/app.py - GET recipes for owner (10 kitchen-only + 10 need-grocery). No POST; regeneration is triggered by kitchen add/remove.
# Uses per-owner table: {owner}_recipes (single row _id = 'current')
import os
# Force republish after dependency-layer restore.
import json
import re
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
RECIPE_MEAL_CATEGORIES = ('breakfast', 'lunch', 'dinner', 'snacks')


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
        cursorclass=pymysql.cursors.DictCursor,
        # autocommit=True: fresh read snapshot per statement (cached global _conn
        # otherwise freezes a REPEATABLE READ view on GET-only warm containers).
        autocommit=True,
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
        'Access-Control-Allow-Methods': 'GET,OPTIONS'
    }


def _success(body, status=200):
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps(body, default=json_serial)}


def _error(status, message):
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps({'error': message})}


def _recipes_table(owner):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_recipes"


def _ensure_recipes_table(conn, owner):
    table = _recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                _id VARCHAR(36) PRIMARY KEY,
                _owner VARCHAR(36) NOT NULL,
                status ENUM('ready', 'regenerating', 'failed', 'empty') NOT NULL DEFAULT 'empty',
                kitchen_only JSON,
                need_grocery JSON,
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
            return _get_recipes(owner)
        return _error(405, f'Method {http_method} not allowed')
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return _error(500, str(e))


def _parse_json_field(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def _normalize_recipe_meal_category(value):
    normalized = str(value or '').strip().lower()
    if normalized in RECIPE_MEAL_CATEGORIES:
        return normalized
    return None


def _infer_recipe_meal_category(title, ingredients, fallback_index=0):
    text_parts = [str(title or '').lower()]
    text_parts.extend(str(item or '').lower() for item in (ingredients or []))
    haystack = ' '.join(text_parts)

    keyword_groups = [
        ('breakfast', ('breakfast', 'omelet', 'omelette', 'oatmeal', 'pancake', 'waffle', 'bagel', 'toast', 'cereal', 'granola', 'yogurt', 'egg')),
        ('lunch', ('lunch', 'sandwich', 'wrap', 'salad', 'soup', 'quesadilla', 'taco', 'bowl')),
        ('snacks', ('snack', 'snacks', 'dip', 'bites', 'bar', 'trail mix', 'parfait', 'popcorn', 'chips', 'cookies', 'cracker')),
        ('dinner', ('dinner', 'pasta', 'curry', 'stir-fry', 'stir fry', 'skillet', 'roast', 'casserole', 'rice bowl', 'burger')),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in haystack for keyword in keywords):
            return category
    return RECIPE_MEAL_CATEGORIES[fallback_index % len(RECIPE_MEAL_CATEGORIES)]


def _normalize_recipe_item(recipe, fallback_index=0):
    if not isinstance(recipe, dict):
        recipe = {}
    normalized = dict(recipe)
    normalized['meal_category'] = _normalize_recipe_meal_category(
        normalized.get('meal_category') or normalized.get('category')
    ) or _infer_recipe_meal_category(
        normalized.get('title'),
        normalized.get('ingredients') or [],
        fallback_index=fallback_index,
    )
    return normalized


def _normalize_recipe_list(recipes):
    return [_normalize_recipe_item(recipe, fallback_index=i) for i, recipe in enumerate(recipes or [])]


def _get_recipes(owner):
    conn = _mysql_conn()
    try:
        table = _recipes_table(owner)

        # Fetch kitchen version state to detect staleness
        kitchen_version = 0
        recipe_generation_version = 0
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT kitchen_version, last_recipe_refresh_completed_version "
                    "FROM owner_kitchen_state WHERE owner = %s LIMIT 1",
                    [owner]
                )
                state_row = cur.fetchone() or {}
                kitchen_version = int(state_row.get('kitchen_version') or 0)
                recipe_generation_version = int(state_row.get('last_recipe_refresh_completed_version') or 0)
        except Exception:
            pass  # Graceful degradation if table doesn't exist yet

        is_stale = (kitchen_version > recipe_generation_version) if kitchen_version > 0 else False

        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if cur.fetchone()['n'] == 0:
                return _success({
                    'owner': owner,
                    'status': 'empty',
                    'kitchen_only': [],
                    'need_grocery': [],
                    'error_message': None,
                    '_createdDate': None,
                    '_updatedDate': None,
                    'kitchen_version': kitchen_version,
                    'recipe_generation_version': recipe_generation_version,
                    'is_stale': is_stale,
                })
            cur.execute(
                f"SELECT _id, _owner, status, kitchen_only, need_grocery, error_message, _createdDate, _updatedDate FROM `{table}` WHERE _id = 'current'",
            )
            row = cur.fetchone()
        if not row:
            return _success({
                'owner': owner,
                'status': 'empty',
                'kitchen_only': [],
                'need_grocery': [],
                'error_message': None,
                '_createdDate': None,
                '_updatedDate': None,
                'kitchen_version': kitchen_version,
                'recipe_generation_version': recipe_generation_version,
                'is_stale': is_stale,
            })
        kitchen_only = _normalize_recipe_list(_parse_json_field(row.get('kitchen_only')))
        need_grocery = _normalize_recipe_list(_parse_json_field(row.get('need_grocery')))
        out = {k: json_serial(v) for k, v in row.items()}
        out['owner'] = owner
        out['kitchen_only'] = kitchen_only
        out['need_grocery'] = need_grocery
        out['kitchen_version'] = kitchen_version
        out['recipe_generation_version'] = recipe_generation_version
        out['is_stale'] = is_stale
        return _success(out)
    finally:
        pass
