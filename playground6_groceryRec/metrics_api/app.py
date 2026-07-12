# metrics_api/app.py
# Force republish after dependency-layer restore.
import os
import json
import re
import uuid
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal

USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_METRICS_WINDOW_DAYS = 14
_DEFAULT_IQ = 75
_METRIC_TEXT_FIELDS = (
    'IQ_what',
    'UPF_what',
    'harmful_ingredients_what',
)
_METRIC_SUGGESTION_FIELDS = (
    'IQ_suggestions',
    'UPF_suggestions',
    'harmful_ingredients_suggestions',
)
_KITCHEN_ANALYSIS_STATUS_FIELD = 'kitchen_analysis_status'
_KITCHEN_ANALYSIS_CONTENT_FIELD = 'kitchen_analysis_content'
_KITCHEN_ANALYSIS_GENERATED_AT_FIELD = 'kitchen_analysis_generated_at'
_KITCHEN_ANALYSIS_ERROR_FIELD = 'kitchen_analysis_error'
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))


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


def _kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_prod_kitchen"


def _new_kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_new_kitchen"


def _discards_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_discards"


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')

        if http_method == 'GET':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            return _get_metrics(owner)

        elif http_method in ['POST', 'PUT']:
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            body = json.loads(event.get('body', '{}'))
            return _update_metrics(owner, body)

        elif http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,OPTIONS'
                },
                'body': ''
            }

        return _error_response(405, f'Method {http_method} not allowed')

    except json.JSONDecodeError as e:
        return _error_response(400, f'Invalid JSON in request body: {str(e)}')
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Internal server error: {str(e)}')


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


def _ensure_grocery_columns(conn, table_name):
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return
        existing_columns = _get_table_columns(cur, table_name)
        upf_position = " AFTER `nutrition_summary`" if 'nutrition_summary' in existing_columns else ""
        cur.execute("""
            SELECT column_name, column_type
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s
              AND LOWER(column_name) IN ('upf', 'harmful_ingredients')
        """, [table_name])
        rows = cur.fetchall() or []
        column_info = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get('column_name') or row.get('COLUMN_NAME') or ''
            col_type = row.get('column_type') or row.get('COLUMN_TYPE')
            if name:
                column_info[name.lower()] = col_type

        if 'upf' not in column_info:
            cur.execute(
                f"ALTER TABLE `{table_name}` "
                f"ADD COLUMN `upf` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'{upf_position}"
            )
            conn.commit()
        else:
            col_type = column_info.get('upf') or ''
            normalized_type = re.sub(r'\s+', '', col_type)
            if normalized_type != "enum('yes','no')":
                cur.execute(
                    f"ALTER TABLE `{table_name}` "
                    "MODIFY COLUMN `upf` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'"
                )
                cur.execute(f"UPDATE `{table_name}` SET `upf` = LOWER(`upf`) WHERE `upf` IS NOT NULL")
                conn.commit()

        if 'harmful_ingredients' not in column_info:
            cur.execute(
                f"ALTER TABLE `{table_name}` "
                "ADD COLUMN `harmful_ingredients` JSON COMMENT 'Array of harmful ingredient strings' AFTER `upf`"
            )
            conn.commit()


def _recent_where_clause(columns):
    clauses = [f"`_createdDate` >= DATE_SUB(NOW(), INTERVAL {_METRICS_WINDOW_DAYS} DAY)"]
    if 'action' in columns:
        clauses.append("`action` = 'IN'")
    return " AND ".join(clauses)


def _get_upf_counts(cur, table_name, extra_where='', extra_params=None):
    if not _table_exists(cur, table_name):
        return 0, 0
    columns = _get_table_columns(cur, table_name)
    where_clause = _recent_where_clause(columns)
    if extra_where:
        where_clause = f"{extra_where} AND {where_clause}"
    params = list(extra_params or [])
    if 'upf' in columns:
        cur.execute(
            f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN LOWER(`upf`) = 'yes' THEN 1 ELSE 0 END) as upf_count
            FROM `{table_name}`
            WHERE {where_clause}
            """,
            params
        )
        row = cur.fetchone() or {}
        total = int(row.get('total') or 0)
        upf_count = int(row.get('upf_count') or 0)
        return total, upf_count

    cur.execute(
        f"""
        SELECT COUNT(*) as total
        FROM `{table_name}`
        WHERE {where_clause}
        """,
        params
    )
    row = cur.fetchone() or {}
    total = int(row.get('total') or 0)
    return total, 0


def _get_harmful_count(cur, table_name, extra_where='', extra_params=None):
    if not _table_exists(cur, table_name):
        return 0
    columns = _get_table_columns(cur, table_name)
    if 'harmful_ingredients' not in columns:
        return 0
    where_clause = _recent_where_clause(columns)
    if extra_where:
        where_clause = f"{extra_where} AND {where_clause}"
    params = list(extra_params or [])
    cur.execute(
        f"""
        SELECT SUM(COALESCE(JSON_LENGTH(`harmful_ingredients`), 0)) as harmful_count
        FROM `{table_name}`
        WHERE {where_clause}
        """,
        params
    )
    row = cur.fetchone() or {}
    return int(row.get('harmful_count') or 0)


def _calculate_metrics(conn, owner):
    if USE_SHARED_TABLES:
        kitchen_table = 'shared_kitchen'
        member_ids = _get_household_member_ids(conn, owner)
        placeholders = ','.join(['%s'] * len(member_ids))
        owner_where = f"`owner_id` IN ({placeholders})"
        owner_params = member_ids
    else:
        kitchen_table = _kitchen_table_name(owner)
        owner_where = ''
        owner_params = []
    new_kitchen_table = _new_kitchen_table_name(owner)
    discards_table = _discards_table_name(owner)
    if not USE_SHARED_TABLES:
        _ensure_grocery_columns(conn, kitchen_table)
    _ensure_grocery_columns(conn, new_kitchen_table)
    _ensure_grocery_columns(conn, discards_table)

    with conn.cursor() as cur:
        kitchen_total, kitchen_upf = _get_upf_counts(cur, kitchen_table, owner_where, owner_params)
        new_kitchen_total, new_kitchen_upf = _get_upf_counts(cur, new_kitchen_table)
        discards_total, discards_upf = _get_upf_counts(cur, discards_table)
        harmful_total = (
            _get_harmful_count(cur, kitchen_table, owner_where, owner_params)
            + _get_harmful_count(cur, new_kitchen_table)
            + _get_harmful_count(cur, discards_table)
        )

    total = kitchen_total + new_kitchen_total + discards_total
    upf_percent = round((kitchen_upf + new_kitchen_upf + discards_upf) / total * 100, 2) if total > 0 else 0
    harmful_total = max(0, min(1000, harmful_total))

    return {
        'UPF': upf_percent,
        'harmful_ingredients': harmful_total,
    }


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _get_latest_metrics(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM `{table_name}` ORDER BY `_createdDate` DESC LIMIT 1"
        )
        return cur.fetchone()


def _pick_body_value(body, keys):
    for key in keys:
        if key in body and body[key] is not None:
            return body[key]
    return None


def _normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_suggestions(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            cleaned.append(line[2:].strip() if line.startswith('- ') else line)
        return cleaned or []
    return [str(value).strip()]


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


def _normalize_metrics_response(metrics):
    if not isinstance(metrics, dict):
        return metrics
    for field in _METRIC_SUGGESTION_FIELDS:
        if field in metrics:
            metrics[field] = _coerce_json_list(metrics[field])
    return metrics


def _get_metrics(owner):
    try:
        table_name = _metrics_table_name(owner)
        conn = _mysql_conn()
        with conn.cursor() as cur:
            if not _table_exists(cur, table_name):
                _ensure_metrics_table(conn, table_name)
                _ensure_metrics_columns(conn, table_name)
                computed = _calculate_metrics(conn, owner)
                metrics = {
                    '_id': None,
                    '_owner': owner,
                    'IQ': _DEFAULT_IQ,
                    'Points': 0,
                    'UPF': computed['UPF'],
                    'harmful_ingredients': computed['harmful_ingredients'],
                    'IQ_what': None,
                    'IQ_suggestions': [],
                    'UPF_what': None,
                    'UPF_suggestions': [],
                    'harmful_ingredients_what': None,
                    'harmful_ingredients_suggestions': [],
                    'kitchen_analysis_status': None,
                    'kitchen_analysis_content': None,
                    'kitchen_analysis_generated_at': None,
                    'kitchen_analysis_error': None,
                }
                return _success_response({'owner': owner, 'metrics': metrics, 'count': 0})

        latest = _get_latest_metrics(conn, table_name)
        if not latest:
            _ensure_metrics_columns(conn, table_name)
            computed = _calculate_metrics(conn, owner)
            metrics = {
                '_id': None,
                '_owner': owner,
                'IQ': _DEFAULT_IQ,
                'Points': 0,
                'UPF': computed['UPF'],
                'harmful_ingredients': computed['harmful_ingredients'],
                'IQ_what': None,
                'IQ_suggestions': [],
                'UPF_what': None,
                'UPF_suggestions': [],
                'harmful_ingredients_what': None,
                'harmful_ingredients_suggestions': [],
                'kitchen_analysis_status': None,
                'kitchen_analysis_content': None,
                'kitchen_analysis_generated_at': None,
                'kitchen_analysis_error': None,
            }
            return _success_response({'owner': owner, 'metrics': metrics, 'count': 0})

        metrics = _normalize_metrics_response({k: json_serial(v) for k, v in latest.items()})
        return _success_response({'owner': owner, 'metrics': metrics, 'count': 1})

    except Exception as e:
        print(f"[ERROR] Failed to get metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to retrieve metrics: {str(e)}')


def _update_metrics(owner, body):
    try:
        table_name = _metrics_table_name(owner)
        points_delta = body.get('points_delta')
        if points_delta is None:
            points_delta = body.get('points')
        if points_delta is None:
            points_delta = body.get('pointsDelta')
        points_delta = _safe_int(points_delta, 0)

        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        computed = _calculate_metrics(conn, owner)
        entry_id = str(uuid.uuid4())
        row = None

        with conn.cursor() as cur:
            for member_id in member_ids:
                member_table_name = _metrics_table_name(member_id)
                _ensure_metrics_table(conn, member_table_name)
                _ensure_metrics_columns(conn, member_table_name)
                latest = _get_latest_metrics(conn, member_table_name) or {}
                prev_points = _safe_int(latest.get('Points'), 0)
                new_points = max(0, prev_points + points_delta)
                iq_what = _normalize_text(_pick_body_value(body, ['iq_what', 'IQ_what', 'iqWhat']))
                iq_suggestions = _normalize_suggestions(_pick_body_value(body, ['iq_suggestions', 'IQ_suggestions', 'iqSuggestions']))
                upf_what = _normalize_text(_pick_body_value(body, ['upf_what', 'UPF_what', 'upfWhat']))
                upf_suggestions = _normalize_suggestions(_pick_body_value(body, ['upf_suggestions', 'UPF_suggestions', 'upfSuggestions']))
                harmful_what = _normalize_text(_pick_body_value(body, ['harmful_ingredients_what', 'harmfulIngredientsWhat', 'harmful_what']))
                harmful_suggestions = _normalize_suggestions(_pick_body_value(body, ['harmful_ingredients_suggestions', 'harmfulIngredientsSuggestions', 'harmful_suggestions']))
                kitchen_analysis_status = _normalize_text(_pick_body_value(body, ['kitchen_analysis_status', 'kitchenAnalysisStatus']))
                kitchen_analysis_content = _normalize_text(_pick_body_value(body, ['kitchen_analysis_content', 'kitchenAnalysisContent']))
                kitchen_analysis_error = _normalize_text(_pick_body_value(body, ['kitchen_analysis_error', 'kitchenAnalysisError']))
                kitchen_analysis_generated_at = _pick_body_value(body, ['kitchen_analysis_generated_at', 'kitchenAnalysisGeneratedAt'])

                if iq_what is None:
                    iq_what = latest.get('IQ_what')
                if iq_suggestions is None:
                    iq_suggestions = _coerce_json_list(latest.get('IQ_suggestions'))
                if upf_what is None:
                    upf_what = latest.get('UPF_what')
                if upf_suggestions is None:
                    upf_suggestions = _coerce_json_list(latest.get('UPF_suggestions'))
                if harmful_what is None:
                    harmful_what = latest.get('harmful_ingredients_what')
                if harmful_suggestions is None:
                    harmful_suggestions = _coerce_json_list(latest.get('harmful_ingredients_suggestions'))
                if kitchen_analysis_status is None:
                    kitchen_analysis_status = latest.get('kitchen_analysis_status')
                if kitchen_analysis_content is None:
                    kitchen_analysis_content = latest.get('kitchen_analysis_content')
                if kitchen_analysis_error is None:
                    kitchen_analysis_error = latest.get('kitchen_analysis_error')
                if kitchen_analysis_generated_at is None:
                    kitchen_analysis_generated_at = latest.get('kitchen_analysis_generated_at')

                cur.execute(
                    f"""
                    INSERT INTO `{member_table_name}`
                    (`_id`, `_owner`, `_createdDate`, `IQ`, `Points`, `UPF`, `harmful_ingredients`,
                     `IQ_what`, `IQ_suggestions`, `UPF_what`, `UPF_suggestions`,
                     `harmful_ingredients_what`, `harmful_ingredients_suggestions`,
                     `kitchen_analysis_status`, `kitchen_analysis_content`, `kitchen_analysis_generated_at`,
                     `kitchen_analysis_error`)
                    VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        entry_id,
                        member_id,
                        _DEFAULT_IQ,
                        new_points,
                        computed['UPF'],
                        computed['harmful_ingredients'],
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
                _dual_write_metrics_to_shared(conn, member_id, member_table_name, entry_id)
                if member_id == owner:
                    cur.execute(f"SELECT * FROM `{member_table_name}` WHERE `_id` = %s", [entry_id])
                    row = cur.fetchone()
            conn.commit()

        metrics = _normalize_metrics_response({k: json_serial(v) for k, v in (row or {}).items()})
        return _success_response({
            'message': 'Metrics updated successfully',
            'owner': owner,
            'points_delta': points_delta,
            'metrics': metrics,
        })

    except Exception as e:
        print(f"[ERROR] Failed to update metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to update metrics: {str(e)}')


def _success_response(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,OPTIONS'
        },
        'body': json.dumps(data, default=json_serial)
    }


def _error_response(status_code, message):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,OPTIONS'
        },
        'body': json.dumps({'error': message})
    }
