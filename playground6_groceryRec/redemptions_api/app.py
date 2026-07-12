import os
# Force republish after dependency-layer restore.
import json
import re
import uuid
import base64
from datetime import datetime, timedelta
from decimal import Decimal
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
try:
    import stripe
except ImportError as exc:
    stripe = None
    _STRIPE_IMPORT_ERROR = exc

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DEFAULT_IQ = 75
_DEFAULT_TTL_HOURS = 72
_DEFAULT_CODE_PREFIX = 'TREPO'
_STRIPE_SECRET_ENV = 'STRIPE_SECRET_KEY'
_DEFAULT_STRIPE_API_VERSION = '2020-08-27'
_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
_STRIPE_TIMEOUT_SECONDS = int(os.getenv('STRIPE_TIMEOUT_SECONDS', '10'))

_DEFAULT_REWARDS = [
    {
        'id': 'free_hot_sauce',
        'name': 'Free Hot Sauce',
        'points_cost': 100,
        'fulfillment': 'stripe',
        'stripe_url': 'https://buy.stripe.com/dRmaEZ63McU339b2Gr97G05',
        'code_prefix': 'SAUCE',
    },
    {
        'id': 'grocery_gift_card',
        'name': 'Grocery Gift Card',
        'points_cost': 500,
        'fulfillment': 'stripe',
        'stripe_url': 'https://buy.stripe.com/7sY6oJ2RAg6f2574Oz97G07',
        'code_prefix': 'GROC',
    },
    {
        'id': 'free_trepo_friend',
        'name': 'Free Trepo for Friend',
        'points_cost': 1000,
        'fulfillment': 'stripe',
        'stripe_url': 'https://buy.stripe.com/cNicN79fYbPZh01dl597G06',
        'code_prefix': 'FRIEND',
    },
    {
        'id': 'custom_trepo_merch_set',
        'name': 'Custom Trepo Merch Set',
        'points_cost': 2000,
        'fulfillment': 'stripe',
        'stripe_url': 'https://buy.stripe.com/eVq5kF1Nw5rBbFHdl597G08',
        'code_prefix': 'MERCH',
    },
]


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


def _sanitize_owner(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')


def _get_household_member_ids(conn, acting_user_id):
    safe_user_id = _sanitize_owner(acting_user_id)
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
        members = [_sanitize_owner(item.get('user_id')) for item in (cur.fetchall() or [])]
    members = [member for member in members if member]
    return list(dict.fromkeys(members)) or [safe_user_id]


# ---- Shared-table migration M4: dual-write metrics snapshots to shared_metrics ----
# Off by default (DUAL_WRITE_METRICS). Mirrors one snapshot row (by _id) into the
# owner_id-keyed shared_metrics append log via INSERT...SELECT, so the mirror is a
# faithful copy of whatever this writer wrote; ON DUPLICATE KEY re-syncs on re-call
# so an in-place snapshot UPDATE (async kitchen-analysis fill-in) is reflected.
# Non-blocking: errors are swallowed + surfaced as a metric-filterable
# {evt:'dual_write_miss', family:'metrics'} marker. (Duplicated across the 4 metrics
# writers — keep byte-equivalent; follow-up: hoist to a shared layer.)
DUAL_WRITE_METRICS = os.getenv('DUAL_WRITE_METRICS', 'false').lower() == 'true'
_SHARED_METRICS_TABLE = 'shared_metrics'
_SHARED_METRICS_COLS = (
    '_id', '_owner', '_createdDate', 'IQ', 'Points', 'UPF', 'harmful_ingredients',
    'IQ_what', 'IQ_suggestions', 'UPF_what', 'UPF_suggestions',
    'harmful_ingredients_what', 'harmful_ingredients_suggestions',
    'kitchen_analysis_status', 'kitchen_analysis_content', 'kitchen_analysis_generated_at',
    'kitchen_analysis_error',
)


def _dual_write_metrics_to_shared(conn, owner, metrics_table, metrics_id, request_id=None):
    if not DUAL_WRITE_METRICS:
        return
    try:
        cols = ', '.join(f'`{c}`' for c in _SHARED_METRICS_COLS)
        src = ', '.join(f's.`{c}`' for c in _SHARED_METRICS_COLS)
        upd = ', '.join(f'`{c}`=VALUES(`{c}`)' for c in _SHARED_METRICS_COLS if c != '_id')
        sql = (
            f"INSERT INTO `{_SHARED_METRICS_TABLE}` (`owner_id`, {cols}) "
            f"SELECT %s, {src} FROM `{metrics_table}` s WHERE s.`_id` = %s "
            f"ON DUPLICATE KEY UPDATE `owner_id`=VALUES(`owner_id`), {upd}"
        )
        with conn.cursor() as cur:
            cur.execute(sql, (owner, metrics_id))
        conn.commit()
    except Exception as exc:
        try:
            import sys
            print(json.dumps({
                'evt': 'dual_write_miss', 'family': 'metrics',
                'owner_id': str(owner) if owner is not None else None,
                'metrics_id': str(metrics_id) if metrics_id is not None else None,
                'error': (str(exc)[:500] if exc is not None else ''),
            }), file=sys.stderr)
        except Exception:
            pass


def _metrics_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}-metrics"


def _redemptions_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_redemptions"


def _table_exists(cur, table_name):
    cur.execute("""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    return cur.fetchone()['count'] > 0


def _get_table_columns(cur, table_name):
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    rows = cur.fetchall() or []
    columns = set()
    for row in rows:
        if isinstance(row, dict):
            name = row.get('column_name') or row.get('COLUMN_NAME')
            if name:
                columns.add(name)
        elif isinstance(row, (list, tuple)) and row:
            columns.add(row[0])
    return columns


def _ensure_metrics_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
              `_id` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
              `_owner` VARCHAR(36) NOT NULL COMMENT 'Owner UUID',
              `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the metrics snapshot was created',
              `IQ` INT NOT NULL DEFAULT {_DEFAULT_IQ} COMMENT 'IQ score out of 100',
              `Points` BIGINT NOT NULL DEFAULT 0 COMMENT 'Points total',
              `UPF` DECIMAL(5,2) NOT NULL DEFAULT 0 COMMENT 'UPF percentage (last 2 weeks)',
              `harmful_ingredients` INT NOT NULL DEFAULT 0 COMMENT 'Harmful ingredient count (last 2 weeks)',
              `IQ_what` TEXT COMMENT 'What this IQ score means',
              `IQ_suggestions` JSON COMMENT 'Suggestions to improve IQ',
              `UPF_what` TEXT COMMENT 'What this UPF score means',
              `UPF_suggestions` JSON COMMENT 'Suggestions to improve UPF score',
              `harmful_ingredients_what` TEXT COMMENT 'What this harmful ingredient count means',
              `harmful_ingredients_suggestions` JSON COMMENT 'Suggestions to reduce harmful ingredients',
              `kitchen_analysis_status` VARCHAR(32) DEFAULT NULL COMMENT 'Latest kitchen analysis job status',
              `kitchen_analysis_content` MEDIUMTEXT COMMENT 'Formatted AI summary of the kitchen',
              `kitchen_analysis_generated_at` DATETIME NULL COMMENT 'When the kitchen analysis was generated',
              `kitchen_analysis_error` TEXT COMMENT 'Latest kitchen analysis error, if any',
              INDEX `idx_owner` (`_owner`),
              INDEX `idx_created` (`_createdDate`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User metrics snapshots'
        """)
        conn.commit()


def _ensure_metrics_columns(conn, table_name):
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return
        columns = _get_table_columns(cur, table_name)
        alter_parts = []
        if 'IQ_what' not in columns:
            alter_parts.append("ADD COLUMN `IQ_what` TEXT COMMENT 'What this IQ score means' AFTER `harmful_ingredients`")
        if 'IQ_suggestions' not in columns:
            alter_parts.append("ADD COLUMN `IQ_suggestions` JSON COMMENT 'Suggestions to improve IQ' AFTER `IQ_what`")
        if 'UPF_what' not in columns:
            alter_parts.append("ADD COLUMN `UPF_what` TEXT COMMENT 'What this UPF score means' AFTER `IQ_suggestions`")
        if 'UPF_suggestions' not in columns:
            alter_parts.append("ADD COLUMN `UPF_suggestions` JSON COMMENT 'Suggestions to improve UPF score' AFTER `UPF_what`")
        if 'harmful_ingredients_what' not in columns:
            alter_parts.append("ADD COLUMN `harmful_ingredients_what` TEXT COMMENT 'What this harmful ingredient count means' AFTER `UPF_suggestions`")
        if 'harmful_ingredients_suggestions' not in columns:
            alter_parts.append("ADD COLUMN `harmful_ingredients_suggestions` JSON COMMENT 'Suggestions to reduce harmful ingredients' AFTER `harmful_ingredients_what`")
        if 'kitchen_analysis_status' not in columns:
            alter_parts.append("ADD COLUMN `kitchen_analysis_status` VARCHAR(32) DEFAULT NULL COMMENT 'Latest kitchen analysis job status' AFTER `harmful_ingredients_suggestions`")
        if 'kitchen_analysis_content' not in columns:
            alter_parts.append("ADD COLUMN `kitchen_analysis_content` MEDIUMTEXT COMMENT 'Formatted AI summary of the kitchen' AFTER `kitchen_analysis_status`")
        if 'kitchen_analysis_generated_at' not in columns:
            alter_parts.append("ADD COLUMN `kitchen_analysis_generated_at` DATETIME NULL COMMENT 'When the kitchen analysis was generated' AFTER `kitchen_analysis_content`")
        if 'kitchen_analysis_error' not in columns:
            alter_parts.append("ADD COLUMN `kitchen_analysis_error` TEXT COMMENT 'Latest kitchen analysis error, if any' AFTER `kitchen_analysis_generated_at`")
        if alter_parts:
            cur.execute(f"ALTER TABLE `{table_name}` {', '.join(alter_parts)}")
            conn.commit()


def _ensure_redemptions_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
              `_id` VARCHAR(36) PRIMARY KEY,
              `_owner` VARCHAR(36) NOT NULL,
              `reward_id` VARCHAR(64) NOT NULL,
              `reward_name` VARCHAR(255) NOT NULL,
              `reward_type` VARCHAR(32) NULL,
              `points_cost` BIGINT NOT NULL,
              `status` VARCHAR(20) NOT NULL,
              `code` VARCHAR(128) NULL,
              `stripe_url` TEXT NULL,
              `idempotency_key` VARCHAR(128) NULL,
              `metadata` JSON NULL,
              `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              `_updatedDate` DATETIME NULL,
              `expires_at` DATETIME NULL,
              `fulfilled_at` DATETIME NULL,
              INDEX `idx_owner` (`_owner`),
              INDEX `idx_status` (`status`),
              UNIQUE KEY `uniq_owner_idempotency` (`_owner`, `idempotency_key`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User reward redemptions'
        """)
        conn.commit()


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _pick_body_value(body, keys):
    for key in keys:
        if key in body and body[key] is not None:
            return body[key]
    return None


def _parse_limit(query):
    try:
        limit = int((query or {}).get('limit') or _DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT
    return max(1, min(limit, _MAX_LIST_LIMIT))


def _parse_before(query):
    raw = _normalize_text((query or {}).get('before'))
    if not raw:
        return None
    normalized = raw.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _coerce_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return [value]
    return []


def _load_rewards():
    raw = os.getenv('REWARDS_JSON')
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rewards = [_normalize_reward(entry) for entry in parsed if isinstance(entry, dict)]
                return [reward for reward in rewards if reward]
        except (TypeError, ValueError):
            print('[WARN] Invalid REWARDS_JSON, using defaults')
    return [_normalize_reward(entry) for entry in _DEFAULT_REWARDS]


def _normalize_reward(entry):
    reward_id = _normalize_text(entry.get('id') or entry.get('reward_id'))
    name = _normalize_text(entry.get('name'))
    points_cost = _safe_int(entry.get('points_cost') or entry.get('points') or entry.get('cost'), 0)
    if not reward_id or not name or points_cost <= 0:
        return None
    fulfillment = _normalize_text(entry.get('fulfillment') or entry.get('type') or 'manual')
    stripe_url = _normalize_text(entry.get('stripe_url'))
    code_prefix = _normalize_text(entry.get('code_prefix'))
    code_value = _normalize_text(entry.get('code'))
    description = _normalize_text(entry.get('description'))
    return {
        'id': reward_id,
        'name': name,
        'points_cost': points_cost,
        'fulfillment': fulfillment,
        'stripe_url': stripe_url,
        'code_prefix': code_prefix,
        'code': code_value,
        'description': description,
    }


def _get_reward(rewards, reward_id):
    for reward in rewards:
        if reward['id'] == reward_id:
            return reward
    return None


def _generate_code(reward):
    if reward.get('code'):
        return reward['code']
    prefix = reward.get('code_prefix') or _DEFAULT_CODE_PREFIX
    suffix = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{suffix}"


def _issue_stripe_promo_code(reward, owner, redemption_id, ttl_hours):
    if stripe is None:
        raise RuntimeError(f"stripe import failed: {_STRIPE_IMPORT_ERROR}")
    secret_key = os.getenv(_STRIPE_SECRET_ENV)
    if not secret_key:
        raise RuntimeError('Stripe secret key not configured')

    stripe.api_key = secret_key
    stripe.api_version = os.getenv('STRIPE_API_VERSION', _DEFAULT_STRIPE_API_VERSION)
    if hasattr(stripe, 'http_client') and hasattr(stripe.http_client, 'RequestsClient'):
        stripe.default_http_client = stripe.http_client.RequestsClient(timeout=_STRIPE_TIMEOUT_SECONDS)
    code = _generate_code(reward)
    expires_at = int((datetime.utcnow() + timedelta(hours=max(1, ttl_hours))).timestamp())
    coupon = stripe.Coupon.create(
        percent_off=100,
        duration='once',
        name=f"Trepo {reward['name']} redemption",
        metadata={
            'reward_id': reward['id'],
            'owner': owner,
            'redemption_id': redemption_id,
        }
    )
    promo = stripe.PromotionCode.create(
        coupon=coupon.id,
        code=code,
        max_redemptions=1,
        expires_at=expires_at,
        metadata={
            'reward_id': reward['id'],
            'owner': owner,
            'redemption_id': redemption_id,
        }
    )
    return {
        'code': code,
        'coupon_id': coupon.id,
        'promotion_code_id': promo.id,
        'expires_at': expires_at,
    }


def _get_latest_metrics(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM `{table_name}` ORDER BY `_createdDate` DESC LIMIT 1")
        return cur.fetchone()


def _insert_metrics_snapshot(conn, owner, table_name, previous, new_points):
    previous = previous or {}
    entry_id = str(uuid.uuid4())
    iq = _safe_int(previous.get('IQ'), _DEFAULT_IQ)
    upf = float(previous.get('UPF') or 0)
    harmful = _safe_int(previous.get('harmful_ingredients'), 0)
    iq_what = previous.get('IQ_what')
    upf_what = previous.get('UPF_what')
    harmful_what = previous.get('harmful_ingredients_what')
    iq_suggestions = _coerce_json_list(previous.get('IQ_suggestions'))
    upf_suggestions = _coerce_json_list(previous.get('UPF_suggestions'))
    harmful_suggestions = _coerce_json_list(previous.get('harmful_ingredients_suggestions'))
    kitchen_analysis_status = previous.get('kitchen_analysis_status')
    kitchen_analysis_content = previous.get('kitchen_analysis_content')
    kitchen_analysis_generated_at = previous.get('kitchen_analysis_generated_at')
    kitchen_analysis_error = previous.get('kitchen_analysis_error')
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO `{table_name}`
            (`_id`, `_owner`, `_createdDate`, `IQ`, `Points`, `UPF`, `harmful_ingredients`,
             `IQ_what`, `IQ_suggestions`, `UPF_what`, `UPF_suggestions`,
             `harmful_ingredients_what`, `harmful_ingredients_suggestions`,
             `kitchen_analysis_status`, `kitchen_analysis_content`, `kitchen_analysis_generated_at`,
             `kitchen_analysis_error`)
            VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                entry_id,
                owner,
                iq,
                new_points,
                upf,
                harmful,
                iq_what,
                json.dumps(iq_suggestions or []),
                upf_what,
                json.dumps(upf_suggestions or []),
                harmful_what,
                json.dumps(harmful_suggestions or []),
                kitchen_analysis_status,
                kitchen_analysis_content,
                kitchen_analysis_generated_at,
                kitchen_analysis_error,
            ]
        )
    _dual_write_metrics_to_shared(conn, owner, table_name, entry_id)


def _public_reward(reward, include_urls=False):
    payload = {
        'id': reward['id'],
        'name': reward['name'],
        'points_cost': reward['points_cost'],
        'fulfillment': reward.get('fulfillment'),
        'description': reward.get('description'),
    }
    if include_urls and reward.get('stripe_url'):
        payload['stripe_url'] = reward['stripe_url']
    return payload


def _scrub_redemption(row):
    if not isinstance(row, dict):
        return row
    status = row.get('status')
    if status != 'reserved':
        row = dict(row)
        row['code'] = None
        row['stripe_url'] = None
    return row


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path = event.get('rawPath') or event.get('requestContext', {}).get('http', {}).get('path', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        redemption_id = path_params.get('redemption_id')

        if path.rstrip('/').endswith('/stripe/webhook'):
            if http_method == 'POST':
                return _handle_stripe_webhook(event)
            if http_method == 'OPTIONS':
                return _options_response('POST,OPTIONS')
            return _error_response(405, f'Method {http_method} not allowed')

        if path.rstrip('/').endswith('/rewards'):
            if http_method == 'GET':
                include_urls = (event.get('queryStringParameters') or {}).get('include_urls') == 'true'
                return _get_rewards(include_urls=include_urls)
            if http_method == 'OPTIONS':
                return _options_response('GET,OPTIONS')
            return _error_response(405, f'Method {http_method} not allowed')

        if http_method == 'GET':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            if redemption_id:
                return _get_redemption(owner, redemption_id)
            return _list_redemptions(owner, event.get('queryStringParameters') or {})

        if http_method == 'POST':
            if not owner or redemption_id:
                return _error_response(400, 'Missing owner parameter')
            body = json.loads(event.get('body', '{}'))
            return _create_redemption(owner, body)

        if http_method == 'PATCH':
            if not owner or not redemption_id:
                return _error_response(400, 'Missing owner or redemption_id parameter')
            body = json.loads(event.get('body', '{}'))
            return _update_redemption(owner, redemption_id, body)

        if http_method == 'OPTIONS':
            if redemption_id:
                return _options_response('GET,PATCH,OPTIONS')
            return _options_response('GET,POST,OPTIONS')

        return _error_response(405, f'Method {http_method} not allowed')

    except json.JSONDecodeError as e:
        return _error_response(400, f'Invalid JSON in request body: {str(e)}')
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Internal server error: {str(e)}')


def _get_rewards(include_urls=False):
    rewards = _load_rewards()
    payload = [_public_reward(reward, include_urls=include_urls) for reward in rewards]
    return _success_response({'rewards': payload, 'count': len(payload)})


def _list_redemptions(owner, query):
    try:
        table_name = _redemptions_table_name(owner)
        status_filter = _normalize_text(query.get('status')) if isinstance(query, dict) else None
        active_only = isinstance(query, dict) and query.get('active') == 'true'
        limit = _parse_limit(query if isinstance(query, dict) else {})
        before = _parse_before(query if isinstance(query, dict) else {})
        conn = _mysql_conn()
        _ensure_redemptions_table(conn, table_name)
        with conn.cursor() as cur:
            where_clauses = ["`_owner` = %s"]
            values = [owner]
            if status_filter:
                where_clauses.append("`status` = %s")
                values.append(status_filter)
            elif active_only:
                where_clauses.append("`status` IN ('reserved')")
            if before:
                where_clauses.append("COALESCE(`_updatedDate`, `_createdDate`) < %s")
                values.append(before)
            query_sql = f"""
                SELECT * FROM `{table_name}`
                WHERE {' AND '.join(where_clauses)}
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC
                LIMIT {limit + 1}
            """
            cur.execute(query_sql, values)
            rows = cur.fetchall() or []
            has_more = len(rows) > limit
            rows = rows[:limit]
            items = [_scrub_redemption({k: json_serial(v) for k, v in row.items()}) for row in rows]
            return _success_response({
                'owner': owner,
                'redemptions': items,
                'count': len(items),
                'limit': limit,
                'has_more': has_more,
            })
    except Exception as e:
        print(f"[ERROR] Failed to list redemptions: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to list redemptions: {str(e)}')


def _get_redemption(owner, redemption_id):
    try:
        table_name = _redemptions_table_name(owner)
        conn = _mysql_conn()
        _ensure_redemptions_table(conn, table_name)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM `{table_name}` WHERE `_owner` = %s AND `_id` = %s",
                [owner, redemption_id]
            )
            row = cur.fetchone()
            if not row:
                return _error_response(404, f'Redemption {redemption_id} not found')
            item = _scrub_redemption({k: json_serial(v) for k, v in row.items()})
            return _success_response({'redemption': item})
    except Exception as e:
        print(f"[ERROR] Failed to get redemption: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to retrieve redemption: {str(e)}')


def _create_redemption(owner, body):
    reward_id = _normalize_text(_pick_body_value(body, ['reward_id', 'rewardId', 'reward']))
    if not reward_id:
        return _error_response(400, 'Missing reward_id')
    idempotency_key = _normalize_text(_pick_body_value(body, ['idempotency_key', 'idempotencyKey', 'request_id', 'requestId']))
    metadata = body.get('metadata') if isinstance(body.get('metadata'), dict) else None
    rewards = _load_rewards()
    reward = _get_reward(rewards, reward_id)
    if not reward:
        return _error_response(400, f'Unknown reward_id: {reward_id}')

    points_cost = _safe_int(reward['points_cost'], 0)
    if points_cost <= 0:
        return _error_response(400, 'Invalid reward points cost')

    fulfillment = reward.get('fulfillment') or 'manual'
    stripe_url = reward.get('stripe_url')
    code = None
    if fulfillment == 'stripe' and not stripe_url:
        return _error_response(400, 'Reward is missing a stripe_url')

    ttl_hours = _safe_int(os.getenv('REDEMPTION_TTL_HOURS'), _DEFAULT_TTL_HOURS)
    expires_at = datetime.utcnow() + timedelta(hours=max(1, ttl_hours))

    table_name = _redemptions_table_name(owner)
    metrics_table = _metrics_table_name(owner)

    try:
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        _ensure_redemptions_table(conn, table_name)
        _ensure_metrics_table(conn, metrics_table)
        _ensure_metrics_columns(conn, metrics_table)
        with conn.cursor() as cur:
            if idempotency_key:
                cur.execute(
                    f"""
                    SELECT * FROM `{table_name}`
                    WHERE `_owner` = %s AND `idempotency_key` = %s
                    ORDER BY `_createdDate` DESC
                    LIMIT 1
                    """,
                    [owner, idempotency_key]
                )
                existing = cur.fetchone()
                if existing:
                    item = _scrub_redemption({k: json_serial(v) for k, v in existing.items()})
                    return _success_response({'redemption': item, 'idempotent': True})

            prev_metrics = _get_latest_metrics(conn, metrics_table)
            prev_points = _safe_int((prev_metrics or {}).get('Points'), 0)
            if prev_points < points_cost:
                return _error_response(400, 'Not enough points to redeem this reward')

            redemption_id = str(uuid.uuid4())
            stripe_meta = None
            if fulfillment == 'stripe':
                try:
                    stripe_meta = _issue_stripe_promo_code(reward, owner, redemption_id, ttl_hours)
                    code = stripe_meta['code']
                except Exception as exc:
                    return _error_response(500, f'Failed to create Stripe promo code: {exc}')
            if fulfillment == 'code':
                code = _generate_code(reward)

            if metadata is None:
                metadata = {}
            if stripe_meta:
                metadata = dict(metadata)
                metadata['stripe_coupon_id'] = stripe_meta['coupon_id']
                metadata['stripe_promotion_code_id'] = stripe_meta['promotion_code_id']
            for member_id in member_ids:
                member_table_name = _redemptions_table_name(member_id)
                member_metrics_table = _metrics_table_name(member_id)
                _ensure_redemptions_table(conn, member_table_name)
                _ensure_metrics_table(conn, member_metrics_table)
                _ensure_metrics_columns(conn, member_metrics_table)
                cur.execute(
                    f"""
                    INSERT INTO `{member_table_name}`
                    (`_id`, `_owner`, `reward_id`, `reward_name`, `reward_type`,
                     `points_cost`, `status`, `code`, `stripe_url`, `idempotency_key`,
                     `metadata`, `expires_at`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        redemption_id,
                        member_id,
                        reward['id'],
                        reward['name'],
                        fulfillment,
                        points_cost,
                        'reserved',
                        code,
                        stripe_url,
                        idempotency_key,
                        json.dumps(metadata) if metadata else None,
                        expires_at.strftime('%Y-%m-%d %H:%M:%S'),
                    ]
                )
                member_prev_metrics = _get_latest_metrics(conn, member_metrics_table)
                member_prev_points = _safe_int((member_prev_metrics or {}).get('Points'), 0)
                _insert_metrics_snapshot(conn, member_id, member_metrics_table, member_prev_metrics, member_prev_points - points_cost)
            conn.commit()

            cur.execute(
                f"SELECT * FROM `{table_name}` WHERE `_id` = %s",
                [redemption_id]
            )
            row = cur.fetchone()
            item = _scrub_redemption({k: json_serial(v) for k, v in (row or {}).items()})
            return _success_response({'redemption': item})
    except Exception as e:
        print(f"[ERROR] Failed to create redemption: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to create redemption: {str(e)}')


def _handle_stripe_webhook(event):
    if stripe is None:
        return _error_response(500, f"stripe import failed: {_STRIPE_IMPORT_ERROR}")
    secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    if not secret:
        return _error_response(500, 'Stripe webhook secret not configured')

    payload = event.get('body') or ''
    if event.get('isBase64Encoded'):
        try:
            payload = base64.b64decode(payload).decode('utf-8')
        except (TypeError, ValueError) as exc:
            return _error_response(400, f'Invalid base64 payload: {exc}')

    signature = _get_header(event, 'stripe-signature')
    if not signature:
        return _error_response(400, 'Missing Stripe signature')

    try:
        stripe_event = stripe.Webhook.construct_event(payload, signature, secret)
    except stripe.error.SignatureVerificationError:
        return _error_response(400, 'Invalid Stripe signature')
    except Exception as exc:
        return _error_response(400, f'Webhook error: {exc}')

    event_type = stripe_event.get('type')
    if event_type == 'checkout.session.completed':
        session = stripe_event.get('data', {}).get('object', {}) or {}
        metadata = session.get('metadata') or {}
        owner = metadata.get('owner') or metadata.get('owner_id') or metadata.get('ownerId')
        redemption_id = (
            session.get('client_reference_id')
            or metadata.get('redemption_id')
            or metadata.get('redemptionId')
        )
        if owner and redemption_id:
            update_response = _update_redemption(owner, redemption_id, {'status': 'fulfilled'})
            if update_response.get('statusCode', 500) >= 400:
                print(f"[WARN] Redemption update failed: {update_response}")
            return _success_response({'received': True})
        print('[WARN] Stripe webhook missing owner/redemption_id metadata')
        return _success_response({'received': True, 'message': 'Missing owner/redemption_id metadata'})

    return _success_response({'received': True})


def _update_redemption(owner, redemption_id, body):
    new_status = _normalize_text(_pick_body_value(body, ['status', 'state']))
    allowed = {'reserved', 'fulfilled', 'expired', 'cancelled'}
    if not new_status or new_status not in allowed:
        return _error_response(400, 'Invalid status')

    table_name = _redemptions_table_name(owner)
    metrics_table = _metrics_table_name(owner)

    try:
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        _ensure_redemptions_table(conn, table_name)
        _ensure_metrics_table(conn, metrics_table)
        _ensure_metrics_columns(conn, metrics_table)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM `{table_name}` WHERE `_owner` = %s AND `_id` = %s",
                [owner, redemption_id]
            )
            row = cur.fetchone()
            if not row:
                return _error_response(404, f'Redemption {redemption_id} not found')

            current_status = row.get('status')
            if current_status == new_status:
                item = _scrub_redemption({k: json_serial(v) for k, v in row.items()})
                return _success_response({'redemption': item})

            if current_status not in ['reserved']:
                return _error_response(400, f'Redemption is already {current_status}')

            update_parts = ["`status` = %s", "`_updatedDate` = NOW()"]
            update_values = [new_status]
            if new_status == 'fulfilled':
                update_parts.append("`fulfilled_at` = NOW()")
                update_parts.append("`code` = NULL")
                update_parts.append("`stripe_url` = NULL")
            if new_status in ['expired', 'cancelled']:
                update_parts.append("`code` = NULL")
                update_parts.append("`stripe_url` = NULL")

            for member_id in member_ids:
                member_table_name = _redemptions_table_name(member_id)
                member_metrics_table = _metrics_table_name(member_id)
                member_update_values = list(update_values) + [member_id, redemption_id]
                cur.execute(
                    f"""
                    UPDATE `{member_table_name}`
                    SET {', '.join(update_parts)}
                    WHERE `_owner` = %s AND `_id` = %s
                    """,
                    member_update_values
                )

                if new_status in ['expired', 'cancelled']:
                    points_cost = _safe_int(row.get('points_cost'), 0)
                    if points_cost > 0:
                        prev_metrics = _get_latest_metrics(conn, member_metrics_table)
                        prev_points = _safe_int((prev_metrics or {}).get('Points'), 0)
                        _insert_metrics_snapshot(conn, member_id, member_metrics_table, prev_metrics, prev_points + points_cost)

            conn.commit()

            cur.execute(
                f"SELECT * FROM `{table_name}` WHERE `_owner` = %s AND `_id` = %s",
                [owner, redemption_id]
            )
            updated = cur.fetchone()
            item = _scrub_redemption({k: json_serial(v) for k, v in (updated or {}).items()})
            return _success_response({'redemption': item})
    except Exception as e:
        print(f"[ERROR] Failed to update redemption: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to update redemption: {str(e)}')


def _success_response(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PATCH,OPTIONS'
        },
        'body': json.dumps(data, default=json_serial)
    }


def _options_response(allow_methods):
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': allow_methods
        },
        'body': ''
    }


def _error_response(status_code, message):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PATCH,OPTIONS'
        },
        'body': json.dumps({'error': message})
    }


def _get_header(event, header_name):
    headers = event.get('headers') or {}
    target = header_name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None
