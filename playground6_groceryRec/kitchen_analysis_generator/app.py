import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal

try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc


def _report_backend_error(op, owner_id=None, code=None, error=None, job_id=None, service='kitchen'):
    """Emit a metric-filterable marker (evt=backend_error) for a swallowed soft failure. Never throws."""
    try:
        import sys as _sys
        msg = ''
        if error is not None:
            msg = error if isinstance(error, str) else str(error)
            if len(msg) > 500:
                msg = msg[:500]
        print(json.dumps({
            'evt': 'backend_error',
            'service': service,
            'op': op,
            'owner_id': str(owner_id) if owner_id is not None else None,
            'code': code or 'error',
            'error': msg,
            'job_id': str(job_id) if job_id is not None else None,
        }), file=_sys.stderr)
    except Exception:
        pass

USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DEFAULT_IQ = 75
_METRICS_WINDOW_DAYS = 14
_RECENT_EXPIRED_DAYS = int(os.getenv('KITCHEN_ANALYSIS_RECENT_EXPIRED_DAYS', '7'))
_EXPIRING_SOON_DAYS = int(os.getenv('KITCHEN_ANALYSIS_EXPIRING_SOON_DAYS', '7'))
_OPENAI_TIMEOUT_SECONDS = int(os.getenv('OPENAI_TIMEOUT_SECONDS', '45'))
_OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '2'))
_KITCHEN_ANALYSIS_MODEL = os.getenv('KITCHEN_ANALYSIS_MODEL') or os.getenv('OPENAI_MODEL') or 'gpt-4o'
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))


def _get_db_config():
    missing = [name for name in _DB_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {', '.join(missing)}")
    return {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', '3306')),
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


def _sanitize_owner(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')


def _metrics_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f'{safe}-metrics'


def _kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f'{safe}_prod_kitchen'


def _new_kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f'{safe}_new_kitchen'


def _discards_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f'{safe}_discards'


def _owner_lock_name(owner):
    return f'kitchen-analysis-owner:{_sanitize_owner(owner)}'


def _lock_mysql_conn(timeout_seconds):
    if pymysql is None:
        raise RuntimeError(f"pymysql import failed: {_PYMYSQL_IMPORT_ERROR}")
    config = _get_db_config()
    wait_timeout = max(
        _DB_READ_TIMEOUT,
        _DB_WRITE_TIMEOUT,
        int(timeout_seconds or 0) + 5,
    )
    return pymysql.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=max(_DB_CONNECT_TIMEOUT, 5),
        read_timeout=wait_timeout,
        write_timeout=wait_timeout,
    )


def _acquire_named_lock(lock_name, timeout_seconds=0):
    conn = _lock_mysql_conn(timeout_seconds)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT GET_LOCK(%s, %s) AS acquired', [lock_name, timeout_seconds])
            row = cur.fetchone() or {}
        if row.get('acquired') == 1:
            return conn
    except Exception:
        conn.close()
        raise
    conn.close()
    return None


def _release_named_lock(conn, lock_name):
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT RELEASE_LOCK(%s)', [lock_name])
    except Exception:
        pass
    finally:
        conn.close()


def json_serial(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
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


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _table_exists(cur, table_name):
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    row = cur.fetchone() or {}
    return int(row.get('count') or 0) > 0


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
    return columns


def _get_household_member_ids(conn, acting_user_id):
    safe_user_id = _sanitize_owner(acting_user_id)
    if not safe_user_id:
        return []
    with conn.cursor() as cur:
        cur.execute('SELECT owner_id FROM new_users WHERE user_id = %s LIMIT 1', [safe_user_id])
        row = cur.fetchone() or {}
        household_id = row.get('owner_id')
        if not household_id:
            return [safe_user_id]
        cur.execute(
            'SELECT user_id FROM new_users WHERE owner_id = %s ORDER BY created_at ASC, user_id ASC',
            [household_id]
        )
        members = [_sanitize_owner(item.get('user_id')) for item in (cur.fetchall() or [])]
    members = [member for member in members if member]
    return list(dict.fromkeys(members)) or [safe_user_id]


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
            name = (row.get('column_name') or row.get('COLUMN_NAME') or '').lower()
            if name:
                column_info[name] = row.get('column_type') or row.get('COLUMN_TYPE')

        if 'upf' not in column_info:
            cur.execute(
                f"ALTER TABLE `{table_name}` "
                f"ADD COLUMN `upf` ENUM('yes', 'no') COMMENT 'Ultra-processed food flag'{upf_position}"
            )
            conn.commit()
        else:
            normalized_type = re.sub(r'\s+', '', column_info.get('upf') or '')
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
    return ' AND '.join(clauses)


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
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN LOWER(`upf`) = 'yes' THEN 1 ELSE 0 END) AS upf_count
            FROM `{table_name}`
            WHERE {where_clause}
            """,
            params
        )
        row = cur.fetchone() or {}
        return int(row.get('total') or 0), int(row.get('upf_count') or 0)

    cur.execute(
        f"""
        SELECT COUNT(*) AS total
        FROM `{table_name}`
        WHERE {where_clause}
        """,
        params
    )
    row = cur.fetchone() or {}
    return int(row.get('total') or 0), 0


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
        SELECT SUM(COALESCE(JSON_LENGTH(`harmful_ingredients`), 0)) AS harmful_count
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
    return {'UPF': upf_percent, 'harmful_ingredients': harmful_total}


def _get_latest_metrics(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM `{table_name}` ORDER BY `_createdDate` DESC LIMIT 1")
        return cur.fetchone() or {}


def _parse_expiration(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).date()
    except ValueError:
        return None


def _get_kitchen_rows(conn, owner):
    if USE_SHARED_TABLES:
        member_ids = _get_household_member_ids(conn, owner)
        placeholders = ','.join(['%s'] * len(member_ids))
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM `shared_kitchen`
                WHERE `owner_id` IN ({placeholders})
                  AND `action` = 'IN'
                  AND (`analysis_stage` = 'final' OR `analysis_stage` IS NULL)
                  AND (`analysis_status` = 'ready' OR `analysis_status` IS NULL)
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC, `_createdDate` DESC
            """, member_ids)
            return cur.fetchall() or []
    table_name = _kitchen_table_name(owner)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        columns = _get_table_columns(cur, table_name)
        where_parts = []
        if 'action' in columns:
            where_parts.append("`action` = 'IN'")
        if 'analysis_stage' in columns:
            where_parts.append("(`analysis_stage` = 'final' OR `analysis_stage` IS NULL)")
        if 'analysis_status' in columns:
            where_parts.append("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)")
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ''
        order_clause = "ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC" if '_updatedDate' in columns else "ORDER BY `_createdDate` DESC"
        cur.execute(f"SELECT * FROM `{table_name}` {where_clause} {order_clause}")
        return cur.fetchall() or []


def _category_breakdown(rows):
    counts = {}
    for row in rows:
        category = str(row.get('category') or 'uncategorized').strip() or 'uncategorized'
        counts[category] = counts.get(category, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return [{'category': category, 'count': count} for category, count in ordered[:8]]


def _row_brief(row):
    parts = []
    for key in ('product_name', 'brand', 'variant', 'category'):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return ' | '.join(parts) if parts else str(row.get('_id') or 'Unknown item')


def _compute_kitchen_facts(rows):
    today = datetime.utcnow().date()
    recent_cutoff = today - timedelta(days=max(0, _RECENT_EXPIRED_DAYS))
    soon_cutoff = today + timedelta(days=max(0, _EXPIRING_SOON_DAYS))
    expired_recently = []
    expiring_soon = []
    no_expiration_count = 0

    for row in rows:
        expiration = _parse_expiration(row.get('product_expiration'))
        if expiration is None:
            no_expiration_count += 1
            continue
        if expiration < today and expiration >= recent_cutoff:
            expired_recently.append({
                'item': _row_brief(row),
                'expiration_date': expiration.isoformat(),
                'days_ago': (today - expiration).days,
            })
        elif today <= expiration <= soon_cutoff:
            expiring_soon.append({
                'item': _row_brief(row),
                'expiration_date': expiration.isoformat(),
                'days_until': (expiration - today).days,
            })

    expired_recently.sort(key=lambda item: (item['days_ago'], item['item'].lower()))
    expiring_soon.sort(key=lambda item: (item['days_until'], item['item'].lower()))

    return {
        'total_items': len(rows),
        'categories': _category_breakdown(rows),
        'expired_recently': expired_recently[:10],
        'expiring_soon': expiring_soon[:10],
        'missing_expiration_count': no_expiration_count,
        'recent_expired_window_days': _RECENT_EXPIRED_DAYS,
        'expiring_soon_window_days': _EXPIRING_SOON_DAYS,
    }


def _empty_kitchen_summary():
    return (
        "Overview\n"
        "- The kitchen is currently empty or has no active items checked in.\n\n"
        "Expired Recently\n"
        "- No recently expired items were found.\n\n"
        "Expiring Soon\n"
        "- No items are expiring soon."
    )


def _generate_analysis_with_gpt(rows, facts):
    if not rows:
        return _empty_kitchen_summary()
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=_OPENAI_TIMEOUT_SECONDS,
        max_retries=_OPENAI_MAX_RETRIES,
    )
    system = """You analyze a user's kitchen inventory. Return plain text only, no markdown code fences.
Use exactly these section headings in this order:
Overview
Expired Recently
Expiring Soon
Notes

Rules:
- Ground every claim in the provided facts or kitchen rows.
- Do not invent items, dates, categories, or counts.
- Keep it concise and easy for a mobile app to render.
- If a section has nothing notable, say so briefly.
- Mention broad inventory patterns, recently expired items, and soon-to-expire items.
- You may mention missing expiration coverage or category imbalance in Notes when clearly supported."""
    user = (
        "Computed kitchen facts:\n"
        f"{json.dumps(facts, default=json_serial)}\n\n"
        "Full active prod_kitchen rows with all fields:\n"
        f"{json.dumps(rows, default=json_serial)}"
    )
    resp = client.chat.completions.create(
        model=_KITCHEN_ANALYSIS_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return text


def _insert_processing_snapshot(conn, owner, member_ids, points_delta):
    computed = _calculate_metrics(conn, owner)
    entry_id = str(uuid.uuid4())
    generated_rows = []
    with conn.cursor() as cur:
        for member_id in member_ids:
            table_name = _metrics_table_name(member_id)
            _ensure_metrics_table(conn, table_name)
            _ensure_metrics_columns(conn, table_name)
            latest = _get_latest_metrics(conn, table_name)
            new_points = max(0, _safe_int(latest.get('Points'), 0) + _safe_int(points_delta, 0))
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
                    member_id,
                    _DEFAULT_IQ,
                    new_points,
                    computed['UPF'],
                    computed['harmful_ingredients'],
                    latest.get('IQ_what'),
                    json.dumps(_coerce_json_list(latest.get('IQ_suggestions'))),
                    latest.get('UPF_what'),
                    json.dumps(_coerce_json_list(latest.get('UPF_suggestions'))),
                    latest.get('harmful_ingredients_what'),
                    json.dumps(_coerce_json_list(latest.get('harmful_ingredients_suggestions'))),
                    'processing',
                    latest.get('kitchen_analysis_content'),
                    latest.get('kitchen_analysis_generated_at'),
                    None,
                ],
            )
            generated_rows.append((member_id, table_name))
        conn.commit()
    return entry_id, generated_rows


def _update_snapshot_status(conn, generated_rows, entry_id, status, content=None, error_message=None):
    generated_at = datetime.utcnow() if status == 'ready' else None
    with conn.cursor() as cur:
        for _, table_name in generated_rows:
            cur.execute(
                f"""
                UPDATE `{table_name}`
                SET `kitchen_analysis_status` = %s,
                    `kitchen_analysis_content` = COALESCE(%s, `kitchen_analysis_content`),
                    `kitchen_analysis_generated_at` = %s,
                    `kitchen_analysis_error` = %s
                WHERE `_id` = %s
                """,
                [status, content, generated_at, error_message, entry_id],
            )
        conn.commit()


def handler(event, context):
    owner = str((event or {}).get('owner') or '').strip()
    if not owner:
        print('[kitchen_analysis_generator] Missing owner')
        return {'ok': False, 'error': 'missing owner'}
    points_delta = _safe_int((event or {}).get('points_delta'), 0)
    owner_lock_name = _owner_lock_name(owner)
    owner_lock_conn = _acquire_named_lock(owner_lock_name, 0)
    if owner_lock_conn is None:
        print(f'[kitchen_analysis_generator] Skipping duplicate run owner={owner}')
        return {'ok': True, 'skipped': True}

    conn = None
    entry_id = None
    generated_rows = []
    try:
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        entry_id, generated_rows = _insert_processing_snapshot(conn, owner, member_ids, points_delta)
        kitchen_rows = _get_kitchen_rows(conn, owner)
        facts = _compute_kitchen_facts(kitchen_rows)
        content = _generate_analysis_with_gpt(kitchen_rows, facts)
        _update_snapshot_status(conn, generated_rows, entry_id, 'ready', content=content, error_message=None)
        print(
            '[kitchen_analysis_generator] Analysis generated',
            json.dumps({'owner': owner, 'rows': len(kitchen_rows), 'entry_id': entry_id})
        )
        return {'ok': True, 'owner': owner, 'entry_id': entry_id}
    except Exception as exc:
        print(f'[kitchen_analysis_generator] Failed: {exc}')
        if conn is not None and entry_id and generated_rows:
            try:
                _update_snapshot_status(conn, generated_rows, entry_id, 'failed', content=None, error_message=str(exc))
            except Exception as update_exc:
                print(f'[kitchen_analysis_generator] Failed to persist error state: {update_exc}')
                _report_backend_error('generate', owner_id=owner, code='persist_failed', error=update_exc, job_id=entry_id)
        raise
    finally:
        _release_named_lock(owner_lock_conn, owner_lock_name)
