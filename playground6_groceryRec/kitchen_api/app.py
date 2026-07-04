# kitchen_api/app.py
# Force republish after dependency-layer restore.
import os
import json
import time
import boto3
import uuid
import base64
import urllib.request
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal
from openai import OpenAI
from master_feed import write_master_feed_event

# Database configuration keys (read lazily to avoid init failures)
_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
_OWNER_KITCHEN_STATE_TABLE = 'owner_kitchen_state'
DUAL_WRITE_ENABLED = os.getenv('DUAL_WRITE_ENABLED', 'false').lower() == 'true'  # dishes only
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'
WRITE_SHARED_ONLY = os.getenv('WRITE_SHARED_ONLY', 'false').lower() == 'true'
_RECIPE_RELEVANT_KITCHEN_FIELDS = {
    'product_name',
    'product_description',
    'category',
    'ingredients',
    'action',
    'analysis_stage',
    'analysis_status',
}


def _get_db_config():
    """Resolve DB config from environment with validation."""
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


def _invoke_meal_plan_generator(owner):
    """Fire-and-forget invoke of meal plan generator (e.g. after kitchen change)."""
    arn = os.getenv('MEAL_PLAN_GENERATOR_ARN')
    if not arn:
        return
    try:
        boto3.client('lambda').invoke(
            FunctionName=arn,
            InvocationType='Event',
            Payload=json.dumps({'owner': owner}),
        )
    except Exception as e:
        print(f"[WARN] Meal plan generator invoke failed: {e}")
        _report_backend_error('invoke_meal_plan_generator', owner_id=owner, code='invoke_failed', error=e)


def _invoke_recipes_generator(owner):
    """Fire-and-forget invoke of recipes generator (e.g. after kitchen change)."""
    arn = os.getenv('RECIPES_GENERATOR_ARN')
    if not arn:
        return
    try:
        boto3.client('lambda').invoke(
            FunctionName=arn,
            InvocationType='Event',
            Payload=json.dumps({'owner': owner}),
        )
    except Exception as e:
        print(f"[WARN] Recipes generator invoke failed: {e}")
        _report_backend_error('invoke_recipes_generator', owner_id=owner, code='invoke_failed', error=e)


def _invoke_kitchen_analysis_generator(owner, points_delta=0):
    """Fire-and-forget invoke of kitchen analysis generator after kitchen mutations."""
    arn = os.getenv('KITCHEN_ANALYSIS_GENERATOR_ARN')
    if not arn:
        return
    try:
        boto3.client('lambda').invoke(
            FunctionName=arn,
            InvocationType='Event',
            Payload=json.dumps({'owner': owner, 'points_delta': int(points_delta or 0)}),
        )
    except Exception as e:
        print(f"[WARN] Kitchen analysis generator invoke failed: {e}")
        _report_backend_error('invoke_analysis_generator', owner_id=owner, code='invoke_failed', error=e)


def _refresh_shelf_life_cache(owner):
    """Fire-and-forget: recalculate shelf life cache after kitchen mutations."""
    arn = os.getenv('HOME_SUGGESTIONS_ARN')
    if not arn:
        return
    try:
        boto3.client('lambda').invoke(
            FunctionName=arn,
            InvocationType='Event',
            Payload=json.dumps({
                'rawPath': f'/home/items-to-watch/{owner}',
                'requestContext': {'http': {'method': 'POST'}},
                'pathParameters': {'owner': owner},
            }),
        )
    except Exception as e:
        print(f"[WARN] Shelf life cache refresh failed: {e}")
        _report_backend_error('refresh_shelf_life_cache', owner_id=owner, code='invoke_failed', error=e)


def _shared_kitchen_insert(conn, owner, payload):
    """Insert a row into shared_kitchen."""
    with conn.cursor() as cur:
        # Add owner_id to payload
        row = dict(payload)
        row['owner_id'] = owner

        # Filter to columns that exist in shared_kitchen
        # Use a broad set — extra columns are harmlessly ignored
        cols = [k for k in row.keys() if k not in ('__table',)]
        placeholders = ', '.join(['%s'] * len(cols))
        columns_sql = ', '.join(f'`{c}`' for c in cols)
        values = [row[c] for c in cols]

        cur.execute(
            f"INSERT INTO `shared_kitchen` ({columns_sql}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE `_updatedDate` = NOW()",
            values
        )
        conn.commit()


class _CoalesceExisting:
    """Sentinel value for _shared_kitchen_update: emit `col = COALESCE(col, %s)`
    so the column is only filled when it's currently NULL. Used for server-side
    storage_location backfill during enrichment so we never overwrite a value a
    user (or an earlier write) already set."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _shared_kitchen_update(conn, item_id, updates_dict):
    """Update a row in shared_kitchen."""
    with conn.cursor() as cur:
        if not updates_dict:
            return
        set_parts = []
        values = []
        for k, v in updates_dict.items():
            if isinstance(v, _CoalesceExisting):
                set_parts.append(f'`{k}` = COALESCE(`{k}`, %s)')
                values.append(v.value)
            else:
                set_parts.append(f'`{k}` = %s')
                values.append(v)
        set_clauses = ', '.join(set_parts)
        values.append(item_id)
        cur.execute(
            f"UPDATE `shared_kitchen` SET {set_clauses}, `_updatedDate` = NOW() WHERE `_id` = %s",
            values
        )
        conn.commit()


def _shared_kitchen_delete(conn, owner, item_id):
    """Archive and delete from shared tables."""
    with conn.cursor() as cur:
        # Copy to shared_archive_kitchen
        cur.execute(
            "INSERT IGNORE INTO `shared_archive_kitchen` "
            "SELECT *, NOW() as archived_at, 'deleted' as archived_reason, 'shared_kitchen' as archived_from_table "
            "FROM `shared_kitchen` WHERE `_id` = %s",
            [item_id]
        )
        # Delete from shared_kitchen
        cur.execute("DELETE FROM `shared_kitchen` WHERE `_id` = %s", [item_id])
        conn.commit()


def _shared_dishes_insert(conn, owner, payload):
    """Insert/update a row in shared_dishes (non-blocking)."""
    if not DUAL_WRITE_ENABLED:
        return
    try:
        with conn.cursor() as cur:
            row = dict(payload)
            row['owner_id'] = owner
            cols = [k for k in row.keys()]
            placeholders = ', '.join(['%s'] * len(cols))
            columns_sql = ', '.join(f'`{c}`' for c in cols)
            values = [row[c] for c in cols]

            cur.execute(
                f"INSERT INTO `shared_dishes` ({columns_sql}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE `_updatedDate` = NOW()",
                values
            )
            conn.commit()
    except Exception as e:
        print(f"[DUAL-WRITE] shared_dishes INSERT failed (non-fatal): {e}")
        _report_backend_error('shared_dishes_dualwrite', owner_id=owner, code='dualwrite_failed',
                              error=e, job_id=(payload or {}).get('job_id'))


def _openai_client():
    """Lazy OpenAI client initializer."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set.')
    return OpenAI(api_key=api_key)


_conn = None


def _mysql_conn():
    """Create or reuse MySQL connection."""
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
    """JSON serializer for objects not serializable by default json code"""
    # Handle basic types that are already JSON-serializable
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Handle Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    # Handle bytes
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    # Handle dict - recursively serialize values
    if isinstance(obj, dict):
        return {k: json_serial(v) for k, v in obj.items()}
    # Handle list/tuple - recursively serialize items
    if isinstance(obj, (list, tuple)):
        return [json_serial(item) for item in obj]
    # For any other type, convert to string
    return str(obj)


def _decode_json_field_if_needed(key, value):
    if key != 'swaps' or value is None:
        return value
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _sanitize_user_id(user_id):
    return ''.join(ch for ch in str(user_id or '') if ch.isalnum() or ch in '-_')


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


def _resolve_kitchen_table(owner, suffix, conn=None):
    """Route reads to shared table (with owner_id filter) or per-user table.
    Returns (table_name, extra_where_sql, extra_where_params).
    """
    safe = _sanitize_user_id(owner)
    if not safe:
        raise ValueError('Invalid owner')
    if not USE_SHARED_TABLES:
        return f"{safe}{suffix}", "", []
    shared_map = {
        '_prod_kitchen': 'shared_kitchen',
        '_archive_kitchen': 'shared_archive_kitchen',
    }
    shared_name = shared_map.get(suffix)
    if not shared_name:
        return f"{safe}{suffix}", "", []
    member_ids = _get_household_member_ids(conn, owner) if conn else [safe]
    placeholders = ','.join(['%s'] * len(member_ids))
    return shared_name, f"owner_id IN ({placeholders})", member_ids


def _query_param_truthy(event, name):
    value = (event.get('queryStringParameters') or {}).get(name)
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _table_exists(cur, table_name):
    cur.execute("""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
        AND table_name = %s
    """, [table_name])
    row = cur.fetchone() or {}
    return row.get('count', 0) > 0


def _ensure_owner_kitchen_state_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_OWNER_KITCHEN_STATE_TABLE}` (
                owner VARCHAR(36) PRIMARY KEY,
                kitchen_version BIGINT NOT NULL DEFAULT 0,
                last_recipe_refresh_requested_version BIGINT DEFAULT NULL,
                last_recipe_refresh_completed_version BIGINT DEFAULT NULL,
                recipe_refresh_needed TINYINT(1) NOT NULL DEFAULT 0,
                last_recipe_refresh_started_at DATETIME NULL,
                last_recipe_refresh_completed_at DATETIME NULL,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def _mark_recipe_refresh_needed_for_owners(conn, owners):
    normalized_owners = [
        _sanitize_user_id(owner)
        for owner in (owners or [])
        if _sanitize_user_id(owner)
    ]
    if not normalized_owners:
        return
    _ensure_owner_kitchen_state_table(conn)
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO `{_OWNER_KITCHEN_STATE_TABLE}` (
                owner,
                kitchen_version,
                last_recipe_refresh_requested_version,
                recipe_refresh_needed
            ) VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                kitchen_version = kitchen_version + 1,
                last_recipe_refresh_requested_version = kitchen_version + 1,
                recipe_refresh_needed = 1,
                _updatedDate = NOW()
            """,
            [(owner, 1, 1, 1) for owner in normalized_owners],
        )
    conn.commit()


def _should_bump_kitchen_version_for_fields(changed_fields):
    return bool(set(changed_fields or []) & _RECIPE_RELEVANT_KITCHEN_FIELDS)


def _get_table_columns(cur, table_name):
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
        AND table_name = %s
    """, [table_name])
    columns = set()
    for row in (cur.fetchall() or []):
        name = row.get('column_name') or row.get('COLUMN_NAME')
        if name:
            columns.add(name)
    return columns


def _get_column_metadata(cur, table_name):
    cur.execute("""
        SELECT column_name, column_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
        AND table_name = %s
    """, [table_name])
    metadata = {}
    for row in (cur.fetchall() or []):
        name = row.get('column_name') or row.get('COLUMN_NAME')
        if not name:
            continue
        metadata[name] = {
            'column_type': (row.get('column_type') or row.get('COLUMN_TYPE') or '').lower(),
            'is_nullable': (row.get('is_nullable') or row.get('IS_NULLABLE') or '').upper(),
            'column_default': row.get('column_default') if 'column_default' in row else row.get('COLUMN_DEFAULT'),
        }
    return metadata


def _serialize_rows(rows):
    serialized = []
    for item in rows or []:
        item_dict = {}
        for key, value in item.items():
            try:
                item_dict[key] = json_serial(_decode_json_field_if_needed(key, value))
            except Exception as e:
                print(f"[WARN] Failed to serialize {key}: {type(value)} - {str(e)}")
                item_dict[key] = str(value) if value is not None else None
        serialized.append(item_dict)
    return serialized


def _record_master_feed_event(conn, owner, member_ids, record):
    try:
        write_master_feed_event(conn, owner, record, member_ids=member_ids)
    except Exception as exc:
        print(f"[WARN] Master feed write failed: {exc}")
        _report_backend_error('master_feed_write', owner_id=owner, code='feed_write_failed',
                              error=exc, job_id=(record or {}).get('job_id'))


def _clean_nullable_string(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_optional_int(value, field_name, minimum=None, maximum=None):
    if value in [None, '']:
        return None
    if isinstance(value, bool):
        raise ValueError(f'Invalid {field_name} value')
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid {field_name} value')
    if minimum is not None and parsed < minimum:
        raise ValueError(f'Invalid {field_name} value')
    if maximum is not None and parsed > maximum:
        raise ValueError(f'Invalid {field_name} value')
    return parsed


def _parse_optional_float(value, field_name, minimum=None, maximum=None):
    if value in [None, '']:
        return None
    if isinstance(value, bool):
        raise ValueError(f'Invalid {field_name} value')
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid {field_name} value')
    if minimum is not None and parsed < minimum:
        raise ValueError(f'Invalid {field_name} value')
    if maximum is not None and parsed > maximum:
        raise ValueError(f'Invalid {field_name} value')
    return parsed


def _parse_optional_bool(value, field_name):
    if value in [None, '']:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid {field_name} value')


def _ensure_archive_metadata_columns(conn, cur, archive_table_name):
    columns = _get_table_columns(cur, archive_table_name)
    metadata = _get_column_metadata(cur, archive_table_name)
    alter_parts = []
    if 'archived_at' not in columns:
        alter_parts.append(
            "ADD COLUMN `archived_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the row was archived from live kitchen'"
        )
    if 'archived_reason' not in columns:
        alter_parts.append(
            "ADD COLUMN `archived_reason` VARCHAR(64) DEFAULT NULL COMMENT 'Why the row was archived'"
        )
    if 'archived_from_table' not in columns:
        alter_parts.append(
            "ADD COLUMN `archived_from_table` VARCHAR(128) DEFAULT NULL COMMENT 'Source live kitchen table name'"
        )
    if 'is_opened' in columns:
        column = metadata.get('is_opened') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `is_opened` TINYINT(1) DEFAULT NULL COMMENT 'Whether the item appears opened or partially used'"
            )
    if alter_parts:
        cur.execute(f"ALTER TABLE `{archive_table_name}` {', '.join(alter_parts)}")
        conn.commit()


def _ensure_archive_table(conn, cur, source_table_name, archive_table_name):
    if not _table_exists(cur, archive_table_name):
        if not _table_exists(cur, source_table_name):
            raise RuntimeError(f'Source kitchen table not found for archive bootstrap: {source_table_name}')
        cur.execute(f"CREATE TABLE IF NOT EXISTS `{archive_table_name}` LIKE `{source_table_name}`")
        conn.commit()
    _ensure_archive_metadata_columns(conn, cur, archive_table_name)


def _archive_kitchen_row(conn, cur, source_table_name, archive_table_name, item_id, archived_reason):
    if USE_SHARED_TABLES:
        src_table, src_where, src_params = source_table_name, "", []
        # When reading from shared table, use the shared name directly
        if source_table_name.endswith('_prod_kitchen') and source_table_name != 'shared_kitchen':
            owner = source_table_name.replace('_prod_kitchen', '')
            src_table, src_where, src_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)
        where_clause = "`_id` = %s"
        if src_where:
            where_clause = f"{src_where} AND {where_clause}"
        cur.execute(f"SELECT * FROM `{src_table}` WHERE {where_clause} LIMIT 1", src_params + [item_id])
    else:
        cur.execute(f"SELECT * FROM `{source_table_name}` WHERE `_id` = %s LIMIT 1", [item_id])
    row = cur.fetchone()
    if not row:
        return None

    _ensure_archive_table(conn, cur, source_table_name, archive_table_name)
    archive_columns = _get_table_columns(cur, archive_table_name)

    insert_columns = [column for column in row.keys() if column in archive_columns]
    values = [row[column] for column in insert_columns]
    # Fix NULLs for columns that may have NOT NULL constraints in older archive tables
    for col, default in [('is_opened', False), ('_device', ''), ('_owner', '')]:
        if col in insert_columns:
            idx = insert_columns.index(col)
            if values[idx] is None:
                values[idx] = default

    if 'archived_at' in archive_columns:
        insert_columns.append('archived_at')
        values.append(datetime.utcnow())
    if 'archived_reason' in archive_columns:
        insert_columns.append('archived_reason')
        values.append(archived_reason)
    if 'archived_from_table' in archive_columns:
        insert_columns.append('archived_from_table')
        values.append(source_table_name)

    columns_sql = ', '.join(f"`{column}`" for column in insert_columns)
    placeholders_sql = ', '.join(['%s'] * len(insert_columns))
    updates = []
    for column in ('archived_at', 'archived_reason', 'archived_from_table'):
        if column in insert_columns:
            updates.append(f"`{column}` = VALUES(`{column}`)")

    query = f"INSERT INTO `{archive_table_name}` ({columns_sql}) VALUES ({placeholders_sql})"
    if updates:
        query += f" ON DUPLICATE KEY UPDATE {', '.join(updates)}"

    cur.execute(query, values)
    return row


def _backfill_null_is_opened(conn, cur, table_name):
    columns = _get_table_columns(cur, table_name)
    if 'is_opened' not in columns:
        return 0
    cur.execute(f"UPDATE `{table_name}` SET `is_opened` = 0 WHERE `is_opened` IS NULL")
    updated_count = int(cur.rowcount or 0)
    if updated_count:
        conn.commit()
    return updated_count


def _list_kitchen_table_names(cur, owner=None):
    patterns = []
    if owner:
        safe_owner = _sanitize_user_id(owner)
        if not safe_owner:
            return []
        patterns = [
            f'{safe_owner}\\_prod\\_kitchen',
            f'{safe_owner}\\_archive\\_kitchen',
        ]
    else:
        patterns = [
            '%\\_prod\\_kitchen',
            '%\\_archive\\_kitchen',
        ]

    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND (
    """
    clauses = []
    params = []
    for pattern in patterns:
        clauses.append("table_name LIKE %s ESCAPE '\\\\'")
        params.append(pattern)
    query += " OR ".join(clauses)
    query += """
          )
        ORDER BY table_name ASC
    """
    cur.execute(query, params)
    return [
        row.get('table_name') or row.get('TABLE_NAME')
        for row in (cur.fetchall() or [])
        if (row.get('table_name') or row.get('TABLE_NAME'))
    ]


def migrate_kitchen_is_opened(owner=None):
    summary = {
        'owner': _sanitize_user_id(owner) if owner else None,
        'prod_tables_checked': 0,
        'archive_tables_checked': 0,
        'prod_rows_backfilled': 0,
        'archive_rows_backfilled': 0,
        'tables': [],
    }
    conn = _mysql_conn()
    with conn.cursor() as cur:
        for table_name in _list_kitchen_table_names(cur, owner=owner):
            entry = {
                'table_name': table_name,
                'table_type': 'archive' if table_name.endswith('_archive_kitchen') else 'prod',
                'rows_backfilled': 0,
            }
            if entry['table_type'] == 'archive':
                summary['archive_tables_checked'] += 1
                _ensure_archive_metadata_columns(conn, cur, table_name)
                entry['rows_backfilled'] = _backfill_null_is_opened(conn, cur, table_name)
                summary['archive_rows_backfilled'] += entry['rows_backfilled']
            else:
                summary['prod_tables_checked'] += 1
                _ensure_prod_kitchen_columns(conn, cur, table_name)
                entry['rows_backfilled'] = _backfill_null_is_opened(conn, cur, table_name)
                summary['prod_rows_backfilled'] += entry['rows_backfilled']
            summary['tables'].append(entry)
    return summary


def _ensure_prod_kitchen_columns(conn, cur, table_name):
    columns = _get_table_columns(cur, table_name)
    metadata = _get_column_metadata(cur, table_name)
    alter_parts = []
    if 'remaining_quantity' not in columns:
        alter_parts.append(
            "ADD COLUMN `remaining_quantity` VARCHAR(128) DEFAULT NULL COMMENT 'Human-readable remaining amount from bulk scan'"
        )
    else:
        column = metadata.get('remaining_quantity') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `remaining_quantity` VARCHAR(128) DEFAULT NULL COMMENT 'Human-readable remaining amount from bulk scan'"
            )
    if 'quantity_value' not in columns:
        alter_parts.append(
            "ADD COLUMN `quantity_value` DECIMAL(10,2) DEFAULT NULL COMMENT 'Structured visible quantity value from scan'"
        )
    else:
        column = metadata.get('quantity_value') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `quantity_value` DECIMAL(10,2) DEFAULT NULL COMMENT 'Structured visible quantity value from scan'"
            )
    if 'quantity_unit' not in columns:
        alter_parts.append(
            "ADD COLUMN `quantity_unit` VARCHAR(32) DEFAULT NULL COMMENT 'Structured visible quantity unit such as count or jar'"
        )
    else:
        column = metadata.get('quantity_unit') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `quantity_unit` VARCHAR(32) DEFAULT NULL COMMENT 'Structured visible quantity unit such as count or jar'"
            )
    if 'is_opened' not in columns:
        alter_parts.append(
            "ADD COLUMN `is_opened` TINYINT(1) DEFAULT NULL COMMENT 'Whether the item appears opened or partially used'"
        )
    else:
        column = metadata.get('is_opened') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `is_opened` TINYINT(1) DEFAULT NULL COMMENT 'Whether the item appears opened or partially used'"
            )
    if 'storage_location' not in columns:
        alter_parts.append(
            "ADD COLUMN `storage_location` VARCHAR(32) DEFAULT NULL COMMENT 'Where the item is stored: fridge, freezer, or pantry'"
        )
    if 'fill_percent' not in columns:
        alter_parts.append(
            "ADD COLUMN `fill_percent` INT DEFAULT NULL COMMENT 'Approximate remaining fill percentage from 0 to 100'"
        )
    else:
        column = metadata.get('fill_percent') or {}
        if column.get('is_nullable') != 'YES':
            alter_parts.append(
                "MODIFY COLUMN `fill_percent` INT DEFAULT NULL COMMENT 'Approximate remaining fill percentage from 0 to 100'"
            )
    # Enrichment-era columns that older tables may lack
    _enrichment_columns = {
        'storage_guidance': "ADD COLUMN `storage_guidance` JSON DEFAULT NULL",
        'analysis_stage': "ADD COLUMN `analysis_stage` VARCHAR(32) DEFAULT NULL",
        'analysis_status': "ADD COLUMN `analysis_status` VARCHAR(32) DEFAULT NULL",
        'analysis_source': "ADD COLUMN `analysis_source` VARCHAR(128) DEFAULT NULL",
        'analysis_updated_at': "ADD COLUMN `analysis_updated_at` DATETIME NULL",
        'provisional_payload': "ADD COLUMN `provisional_payload` JSON DEFAULT NULL",
        'product_image_url': "ADD COLUMN `product_image_url` VARCHAR(1000) DEFAULT NULL",
        'product_image_key': "ADD COLUMN `product_image_key` VARCHAR(500) DEFAULT NULL",
        'resized_image_url': "ADD COLUMN `resized_image_url` VARCHAR(1000) DEFAULT NULL",
        'resized_image_key': "ADD COLUMN `resized_image_key` VARCHAR(500) DEFAULT NULL",
        'nutrition_summary': "ADD COLUMN `nutrition_summary` TEXT DEFAULT NULL",
        'ingredients': "ADD COLUMN `ingredients` JSON DEFAULT NULL",
        'harmful_ingredients': "ADD COLUMN `harmful_ingredients` JSON DEFAULT NULL",
        'healthier_alternatives': "ADD COLUMN `healthier_alternatives` JSON DEFAULT NULL",
        'store_availability': "ADD COLUMN `store_availability` JSON DEFAULT NULL",
        'needs_review': "ADD COLUMN `needs_review` TINYINT(1) DEFAULT 0",
        'swaps': "ADD COLUMN `swaps` JSON DEFAULT NULL",
        'upf': "ADD COLUMN `upf` ENUM('yes','no') DEFAULT NULL",
    }
    for col_name, ddl in _enrichment_columns.items():
        if col_name not in columns:
            alter_parts.append(ddl)

    for ddl in alter_parts:
        try:
            cur.execute(f"ALTER TABLE `{table_name}` {ddl}")
            conn.commit()
        except Exception as e:
            # 1060 = Duplicate column name (race condition with concurrent requests)
            if getattr(e, 'args', (None,))[0] == 1060:
                conn.rollback()
            else:
                raise


def _ensure_prod_kitchen_table(conn, cur, table_name):
    if _table_exists(cur, table_name):
        _ensure_prod_kitchen_columns(conn, cur, table_name)
        # One-time: ensure indexes on shared tables (idempotent)
        try:
            cur.execute("CREATE INDEX `idx_new_users_user_id` ON `new_users`(`user_id`)")
            conn.commit()
        except Exception:
            pass  # Index already exists
        try:
            cur.execute("CREATE INDEX `idx_new_users_owner_id` ON `new_users`(`owner_id`)")
            conn.commit()
        except Exception:
            pass  # Index already exists
        return

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `_id` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
            `_owner` VARCHAR(36) NOT NULL COMMENT 'Owner UUID',
            `_device` VARCHAR(255) NOT NULL COMMENT 'Device ID',
            `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
            `_updatedDate` DATETIME NULL COMMENT 'Last update timestamp',
            `product_name` VARCHAR(500) DEFAULT NULL,
            `brand` VARCHAR(255) DEFAULT NULL,
            `variant` VARCHAR(255) DEFAULT NULL,
            `category` VARCHAR(255) DEFAULT NULL,
            `remaining_quantity` VARCHAR(128) DEFAULT NULL,
            `quantity_value` DECIMAL(10,2) DEFAULT NULL,
            `quantity_unit` VARCHAR(32) DEFAULT NULL,
            `is_opened` TINYINT(1) DEFAULT NULL,
            `fill_percent` INT DEFAULT NULL,
            `confidence` DECIMAL(3,2) DEFAULT NULL,
            `explanation` TEXT DEFAULT NULL,
            `product_description` TEXT DEFAULT NULL,
            `barcode` VARCHAR(100) DEFAULT NULL,
            `country_guess` VARCHAR(100) DEFAULT NULL,
            `estimated_price` VARCHAR(50) DEFAULT NULL,
            `ingredients` JSON DEFAULT NULL,
            `nutrition_summary` TEXT DEFAULT NULL,
            `upf` ENUM('yes', 'no') DEFAULT NULL,
            `harmful_ingredients` JSON DEFAULT NULL,
            `similar_items` JSON DEFAULT NULL,
            `alternatives` JSON DEFAULT NULL,
            `healthier_alternatives` JSON DEFAULT NULL,
            `store_availability` JSON DEFAULT NULL,
            `images` VARCHAR(1000) DEFAULT NULL,
            `s3_key` VARCHAR(500) DEFAULT NULL,
            `action` ENUM('IN', 'OUT') NOT NULL DEFAULT 'IN',
            `product_expiration` DATE DEFAULT NULL,
            `storage_guidance` JSON DEFAULT NULL,
            `resized_image_url` VARCHAR(1000) DEFAULT NULL,
            `resized_image_key` VARCHAR(500) DEFAULT NULL,
            `product_image_url` VARCHAR(1000) DEFAULT NULL,
            `product_image_key` VARCHAR(500) DEFAULT NULL,
            `job_id` VARCHAR(100) DEFAULT NULL,
            `user_id` VARCHAR(255) DEFAULT NULL,
            `analysis_stage` VARCHAR(32) DEFAULT NULL,
            `analysis_status` VARCHAR(32) DEFAULT NULL,
            `analysis_source` VARCHAR(255) DEFAULT NULL,
            `analysis_updated_at` DATETIME DEFAULT NULL,
            `needs_review` TINYINT(1) DEFAULT 0,
            `swaps` JSON DEFAULT NULL,
            `provisional_payload` JSON DEFAULT NULL,
            INDEX `idx_owner` (`_owner`),
            INDEX `idx_device` (`_device`),
            INDEX `idx_product_name` (`product_name`),
            INDEX `idx_category` (`category`),
            INDEX `idx_action` (`action`),
            INDEX `idx_created` (`_createdDate`),
            INDEX `idx_expiration` (`product_expiration`),
            INDEX `idx_job_id` (`job_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    conn.commit()

    # One-time: ensure indexes on shared tables (idempotent)
    try:
        cur.execute("CREATE INDEX `idx_new_users_user_id` ON `new_users`(`user_id`)")
        conn.commit()
    except Exception:
        pass  # Index already exists
    try:
        cur.execute("CREATE INDEX `idx_new_users_owner_id` ON `new_users`(`owner_id`)")
        conn.commit()
    except Exception:
        pass  # Index already exists


def _json_column_value(value, default_empty_list=False):
    if value is None:
        return json.dumps([]) if default_empty_list else None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _build_create_payload(owner, body):
    product_name = str(body.get('product_name') or '').strip()
    if not product_name:
        return None, _error_response(400, 'Missing required field: product_name')

    action = str(body.get('action') or 'IN').strip().upper()
    if action not in ['IN', 'OUT']:
        return None, _error_response(400, 'Invalid action value. Must be "IN" or "OUT"')

    expiration = body.get('product_expiration')
    if expiration:
        try:
            datetime.strptime(expiration, '%Y-%m-%d')
        except ValueError:
            return None, _error_response(400, 'Invalid date format for product_expiration. Use YYYY-MM-DD')

    confidence = body.get('confidence')
    if confidence not in [None, '']:
        try:
            confidence = max(0, min(1, float(confidence)))
        except (TypeError, ValueError):
            return None, _error_response(400, 'Invalid confidence value. Must be a number between 0 and 1')
    else:
        confidence = None

    try:
        quantity_value = _parse_optional_float(body.get('quantity_value'), 'quantity_value', minimum=0)
    except ValueError:
        return None, _error_response(400, 'Invalid quantity_value. Must be a non-negative number or null')

    try:
        fill_percent = _parse_optional_int(body.get('fill_percent'), 'fill_percent', minimum=0, maximum=100)
    except ValueError:
        return None, _error_response(400, 'Invalid fill_percent. Must be an integer between 0 and 100 or null')

    try:
        is_opened = _parse_optional_bool(body.get('is_opened'), 'is_opened')
    except ValueError:
        return None, _error_response(400, 'Invalid is_opened. Must be true, false, or null')

    upf = body.get('upf')
    if upf in ['', None]:
        upf = None
    elif upf not in ['yes', 'no']:
        return None, _error_response(400, 'Invalid upf value. Must be "yes", "no", or null')

    # Storage location: real guess from the writer (fridge/freezer/pantry). Normalize
    # to the app's lowercase enum; anything else (incl. blank) stays null/unset.
    storage_location = body.get('storage_location')
    if storage_location is not None:
        storage_location = str(storage_location).strip().lower() or None
        if storage_location not in ('fridge', 'freezer', 'pantry'):
            storage_location = None

    payload = {
        '_id': str(body.get('_id') or uuid.uuid4()),
        '_owner': owner,
        '_device': str(body.get('device_id') or 'gemini_live_backend'),
        'product_name': product_name,
        'brand': body.get('brand'),
        'variant': body.get('variant'),
        'category': body.get('category'),
        'remaining_quantity': _clean_nullable_string(body.get('remaining_quantity')),
        'quantity_value': quantity_value,
        'quantity_unit': _clean_nullable_string(body.get('quantity_unit')),
        'is_opened': is_opened,
        'fill_percent': fill_percent,
        'confidence': confidence,
        'explanation': body.get('explanation'),
        'product_description': body.get('product_description'),
        'barcode': body.get('barcode'),
        'country_guess': body.get('country_guess'),
        'estimated_price': body.get('estimated_price'),
        'ingredients': _json_column_value(body.get('ingredients'), default_empty_list=True),
        'nutrition_summary': body.get('nutrition_summary'),
        'upf': upf,
        'harmful_ingredients': _json_column_value(body.get('harmful_ingredients'), default_empty_list=True),
        'similar_items': _json_column_value(body.get('similar_items'), default_empty_list=True),
        'alternatives': _json_column_value(body.get('alternatives'), default_empty_list=True),
        'healthier_alternatives': _json_column_value(body.get('healthier_alternatives'), default_empty_list=True),
        'store_availability': _json_column_value(body.get('store_availability'), default_empty_list=True),
        'images': body.get('images'),
        's3_key': body.get('s3_key'),
        'action': action,
        'product_expiration': expiration or None,
        'storage_location': storage_location,
        'storage_guidance': _json_column_value(body.get('storage_guidance')),
        'resized_image_url': body.get('resized_image_url'),
        'resized_image_key': body.get('resized_image_key'),
        'product_image_url': body.get('product_image_url'),
        'product_image_key': body.get('product_image_key'),
        'job_id': str(body.get('job_id') or uuid.uuid4()),
        'user_id': str(body.get('user_id') or owner),
        'analysis_stage': body.get('analysis_stage'),
        'analysis_status': body.get('analysis_status'),
        'analysis_source': body.get('analysis_source'),
        'analysis_updated_at': datetime.utcnow(),
        'needs_review': 1 if body.get('needs_review') else 0,
        'swaps': _json_column_value(body.get('swaps')),
        'provisional_payload': _json_column_value(body.get('provisional_payload')),
    }

    return payload, None


def _create_kitchen_item(owner, body):
    payload, error = _build_create_payload(owner, body)
    if error:
        return error

    table_name = f"{owner}_prod_kitchen"

    try:
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            _ensure_prod_kitchen_table(conn, cur, table_name)

            read_table, read_where, read_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)
            dup_where = "`job_id` = %s"
            if read_where:
                dup_where = f"{read_where} AND {dup_where}"
            cur.execute(
                f"SELECT * FROM `{read_table}` WHERE {dup_where} LIMIT 1",
                read_params + [payload['job_id']]
            )
            existing_item = cur.fetchone()
            if existing_item:
                serialized = _serialize_rows([existing_item])[0]
                return _success_response({
                    'message': 'Item already exists',
                    'created': False,
                    'item': serialized,
                })

            # Write to shared table (primary path)
            _shared_kitchen_insert(conn, owner, payload)

            _mark_recipe_refresh_needed_for_owners(conn, member_ids)

            rb_table, rb_where, rb_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)
            rb_clause = "`_id` = %s"
            if rb_where:
                rb_clause = f"{rb_where} AND {rb_clause}"
            cur.execute(f"SELECT * FROM `{rb_table}` WHERE {rb_clause} LIMIT 1", rb_params + [payload['_id']])
            created_item = cur.fetchone()
            if not created_item:
                return _error_response(500, 'Item was inserted but could not be read back')

            _invoke_kitchen_analysis_generator(owner, points_delta=1)
            _refresh_shelf_life_cache(owner)
            _invoke_meal_plan_generator(owner)
            if not body.get('defer_recipes'):
                _invoke_recipes_generator(owner)

            serialized = _serialize_rows([created_item])[0]
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f"kitchen-add:{payload['_id']}",
                    'device_id': serialized.get('_device'),
                    'event_type': 'kitchen_add',
                    'entity_type': 'kitchen_item',
                    'action': serialized.get('action'),
                    'title': serialized.get('product_name'),
                    'brand': serialized.get('brand'),
                    'item_id': payload['_id'],
                    'job_id': serialized.get('job_id'),
                    'user_id': serialized.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/kitchen/{owner}',
                    'source_system': 'kitchen_api',
                    'primary_image_url': serialized.get('product_image_url') or serialized.get('images'),
                    'secondary_image_url': serialized.get('images') or serialized.get('resized_image_url'),
                    'metadata': {
                        'category': serialized.get('category'),
                        'product_expiration': serialized.get('product_expiration'),
                        'upf': serialized.get('upf'),
                        'harmful_ingredients': serialized.get('harmful_ingredients'),
                        'analysis_status': serialized.get('analysis_status'),
                    },
                },
            )
            return _success_response({
                'message': 'Item created successfully',
                'created': True,
                'item': serialized,
            }, status_code=201)

    except Exception as e:
        print(f"[ERROR] Failed to create kitchen item: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to create item: {str(e)}')


# ──────────────────────────── Text-add (free-text → structured items) ────────────────────────────
# POST /kitchen/{owner}/text-add — parse ANY free text the user types (quantities,
# brands, units, variants, sentences) into structured kitchen items via an LLM,
# create each one, fire enrichment, and log every input as a searchable corpus.

TEXT_ADD_PARSE_MODEL = os.getenv('KITCHEN_TEXT_PARSE_MODEL', 'gpt-4o-mini')
TEXT_ADD_MAX_ITEMS = 25
# Public enrich endpoint (grocery-identifier stack). KitchenApiFunction's IAM role
# can only invoke the 3 generator lambdas, so we dispatch enrichment over HTTP.
ENRICH_KITCHEN_ITEM_URL = os.getenv(
    'ENRICH_KITCHEN_ITEM_URL',
    'https://6m9t6wosh9.execute-api.us-east-1.amazonaws.com/enrich-kitchen-item',
)

_TEXT_ADD_SYSTEM_PROMPT = """You parse a user's free-text grocery/kitchen note into structured items for a kitchen inventory app. The user may type a single item, a comma / newline / "and"-separated list, or full sentences. Extract each DISTINCT product the user wants to add to their kitchen.

Return JSON: {"items":[{fragment, product_name, brand, variant, quantity_value, quantity_unit}], "skipped":[{fragment, reason}]}.

FIELD RULES
- product_name: the canonical product name in Title Case, with NO brand, NO quantity, and NO unit embedded. Examples: "Eggs", "Whole Milk", "Chicken Breast", "Greek Yogurt", "Ground Beef".
- brand: the brand/manufacturer ONLY when clearly present. These store brands ARE brands: 365, Kirkland, Trader Joe's, Good & Gather, Great Value, Siete. Otherwise null.
- variant: descriptive qualifiers such as "organic", "2%", "low fat", "grain free", "unsalted", "extra virgin", "whole", "skim", "reduced fat". If several apply, join them with a space in natural reading order (e.g. "organic low fat"). null if none. Do NOT repeat in variant any word that is already part of product_name: color/descriptor words that belong to the item's common name (e.g. "Red Onion", "Green Bell Pepper", "White Bread") stay in product_name and are NOT also a variant - variant stays null unless there is an ADDITIONAL qualifier like "organic".
- quantity_value: the numeric amount the user wants, or null if unspecified.
- quantity_unit: exactly one of "count", "oz", "lb", "g", "kg", "gallon", "liter", or null. If quantity_value is null, quantity_unit MUST also be null. Normalize synonyms: lbs/pound/pounds -> lb; ounce/ounces -> oz; gram/grams -> g; kilogram/kilograms -> kg; gallons -> gallon; liter/litre/liters/litres -> liter. A bare number with no stated unit (e.g. "2 eggs") uses unit "count".

CRITICAL - numbers that are PART OF A PRODUCT OR BRAND NAME are NOT quantities:
- "2% milk" -> product_name "Milk", variant "2%", quantity_value null, quantity_unit null.
- "5 hour energy" -> product_name "5 Hour Energy", quantity null.
- "7up" -> "7UP", quantity null. "1893 cola" -> "1893 Cola", quantity null.
- "365 whole milk" / "365 organic whole milk" -> brand "365", product_name "Whole Milk" (365 is a store brand), NOT quantity 365.
Only treat a leading number as a quantity when it is clearly a count/measure of the item, not part of its name or brand.

QUANTITY WORDS
- A bare product with NO number and NO quantity word (e.g. "eggs", "milk", "bananas", including each side of "eggs and milk") has quantity_value null and quantity_unit null. Do NOT default a bare noun to 1.
- "a" / "an" / "one" before an item -> quantity_value 1, unit "count" (e.g. "a red onion" -> "Red Onion", 1 count).
- "a dozen" / "dozen" -> 12 count; "2 dozen" -> 24 count; "half a dozen" -> 6 count. NEVER output the unit "dozen" - always convert to count.
- "a couple" -> 2 count; "a few" -> 3 count; "several" -> 3 count.
- "some", "a bunch of", "a bag of", "a pack of", "a box of" with NO number -> quantity_value null (unit null).

SPLITTING INTO SEPARATE ITEMS
- Split on commas, on newlines, and on the word "and" when they separate DISTINCT products: "eggs and milk" -> 2 items; "2 apples and 3 bananas" -> Apples (2 count) + Bananas (3 count).
- Do NOT split a product name that CONTAINS "and" or "&": "half and half", "mac and cheese", "salt and pepper", "sweet and sour sauce", "chips and salsa", "cookies and cream" are each a SINGLE product.
- In full sentences, ignore filler/narration and extract only the products: "I bought 2 lbs of ground beef and a dozen eggs" -> Ground Beef (2 lb) + Eggs (12 count).

SKIPPING
- If a fragment is not plausibly a grocery/kitchen product (gibberish like "asdf", punctuation-only, or empty), put it in "skipped" with a short reason. Never invent a product for gibberish.

For every returned item, set "fragment" to the portion of the user's text that produced it. Title-case product_name. Be consistent and precise. Return ONLY the JSON object."""

_TEXT_ADD_SCHEMA = {
    "name": "kitchen_text_parse",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items", "skipped"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["fragment", "product_name", "brand", "variant", "quantity_value", "quantity_unit"],
                    "properties": {
                        "fragment": {"type": "string"},
                        "product_name": {"type": "string"},
                        "brand": {"type": ["string", "null"]},
                        "variant": {"type": ["string", "null"]},
                        "quantity_value": {"type": ["number", "null"]},
                        "quantity_unit": {"type": ["string", "null"], "enum": ["count", "oz", "lb", "g", "kg", "gallon", "liter", None]},
                    },
                },
            },
            "skipped": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["fragment", "reason"],
                    "properties": {"fragment": {"type": "string"}, "reason": {"type": "string"}},
                },
            },
        },
    },
}


def _parse_text_items_with_llm(text):
    """Parse free text into (items, skipped) via OpenAI structured outputs. Raises on failure."""
    client = _openai_client()
    response = client.chat.completions.create(
        model=TEXT_ADD_PARSE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _TEXT_ADD_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_schema", "json_schema": _TEXT_ADD_SCHEMA},
    )
    data = json.loads(response.choices[0].message.content)
    items = data.get('items') or []
    skipped = data.get('skipped') or []
    return items, skipped


def _fire_kitchen_enrichment(owner, item_id, product_name, brand, variant):
    """Fire-and-forget enrichment dispatch (ingredients/nutrition/storage_location) over HTTP.
    Never raises — enrichment is best-effort and must not fail the text-add request."""
    try:
        seed = {"product_name": product_name, "brand": brand, "variant": variant}
        payload = {
            "owner": owner,
            "item_id": item_id,
            "preliminary_item": {"product_name": product_name, "needs_review": False},
            "initial_payload": seed,
        }
        req = urllib.request.Request(
            ENRICH_KITCHEN_ITEM_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            r.read()
        return True
    except Exception as e:
        print(f"[WARN] text-add enrichment dispatch failed for item {item_id}: {e}")
        return False


def _handle_text_add(owner, body):
    """POST /kitchen/{owner}/text-add — parse free text into items, create each, fire enrichment, log."""
    t0 = time.time()
    text = body.get('text')
    if not isinstance(text, str) or not text.strip():
        return _error_response(400, 'Missing required field: text')
    text = text.strip()
    device_id = str(body.get('device_id') or 'ios-app-text')
    user_id = str(body.get('user_id') or owner)

    # 1) LLM parse
    try:
        items, skipped = _parse_text_items_with_llm(text)
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        print(json.dumps({
            'event': 'kitchen_text_add_parse_error', 'owner': owner, 'raw_text': text,
            'error': str(e), 'latency_ms': latency_ms,
        }))
        return _error_response(502, f'Failed to parse text: {str(e)}')

    skipped = [s for s in skipped if isinstance(s, dict)]

    # 2) Enforce the 25-item cap (log + skip the overflow)
    if len(items) > TEXT_ADD_MAX_ITEMS:
        for extra in items[TEXT_ADD_MAX_ITEMS:]:
            skipped.append({'fragment': (extra or {}).get('fragment') or (extra or {}).get('product_name'), 'reason': 'over_limit'})
        items = items[:TEXT_ADD_MAX_ITEMS]

    # 3) Nothing parseable → 422 (still return skipped for visibility)
    if not items:
        latency_ms = int((time.time() - t0) * 1000)
        print(json.dumps({
            'event': 'kitchen_text_add', 'owner': owner, 'raw_text': text,
            'parse': [], 'created_ids': [], 'skipped': skipped, 'latency_ms': latency_ms,
        }))
        return _success_response({'created': [], 'count': 0, 'parse': [], 'skipped': skipped,
                                  'error': 'No parseable items found in text'}, status_code=422)

    # 4) Create each parsed item, then dispatch enrichment
    parse_out = []
    created = []
    created_ids = []
    for idx, it in enumerate(items):
        pn = str((it.get('product_name') or '')).strip()
        brand = it.get('brand')
        variant = it.get('variant')
        qv = it.get('quantity_value')
        qu = it.get('quantity_unit')
        fragment = it.get('fragment') or pn
        parse_out.append({'fragment': fragment, 'product_name': pn, 'brand': brand,
                          'variant': variant, 'quantity_value': qv, 'quantity_unit': qu})
        if not pn:
            skipped.append({'fragment': fragment, 'reason': 'empty_product_name'})
            continue
        is_last = (idx == len(items) - 1)
        item_body = {
            'product_name': pn, 'brand': brand, 'variant': variant,
            'quantity_value': qv, 'quantity_unit': qu,
            'action': 'IN', 'device_id': device_id, 'user_id': user_id,
            'analysis_stage': 'preliminary', 'analysis_status': 'ready',
            'analysis_source': 'kitchen_text_add', 'defer_recipes': not is_last,
            'provisional_payload': {'source': 'text_add', 'raw_fragment': fragment, 'enrichment_status': 'pending'},
        }
        try:
            resp = _create_kitchen_item(owner, item_body)
            if resp.get('statusCode') not in (200, 201):
                err = json.loads(resp.get('body') or '{}').get('error', 'create failed')
                skipped.append({'fragment': fragment, 'reason': f'create_failed: {err}'})
                continue
            item = json.loads(resp['body']).get('item') or {}
            created.append(item)
            item_id = item.get('_id')
            if item_id:
                created_ids.append(item_id)
                _fire_kitchen_enrichment(owner, item_id, pn, brand, variant)
        except Exception as e:
            skipped.append({'fragment': fragment, 'reason': f'create_error: {str(e)}'})

    latency_ms = int((time.time() - t0) * 1000)
    # 5) Structured corpus log: one line per request (CloudWatch-searchable)
    print(json.dumps({
        'event': 'kitchen_text_add', 'owner': owner, 'raw_text': text,
        'parse': parse_out, 'created_ids': created_ids, 'skipped': skipped,
        'latency_ms': latency_ms,
    }))

    return _success_response({'created': created, 'count': len(created),
                              'parse': parse_out, 'skipped': skipped}, status_code=201)


def _correct_kitchen_item_with_llm(item_dict, correction_text):
    """Call OpenAI to re-derive item fields based on a user correction."""
    system_prompt = (
        "You are a kitchen inventory AI. A user has corrected a misidentified kitchen item. "
        "Given the current item data and the correction, return a JSON object with updated fields. "
        "Return ONLY valid JSON."
    )
    user_prompt = f"""Current item (may be wrong):
product_name: {item_dict.get('product_name', '')}
brand: {item_dict.get('brand', '')}
category: {item_dict.get('category', '')}
description: {item_dict.get('product_description', '')}
ingredients: {item_dict.get('ingredients', [])}
nutrition_summary: {item_dict.get('nutrition_summary', '')}
upf: {item_dict.get('upf', 'no')}

User correction: "{correction_text}"

Return JSON with these fields updated to match the corrected item:
product_name, brand, category (one of: produce/dairy_eggs/meat_seafood/pantry/beverages/snacks_sweets/prepared_other/condiments_sauces/frozen/leftovers), description, ingredients (array), nutrition_summary, upf ("yes"/"no"), harmful_ingredients (array), healthier_alternatives (array of objects with name/brand/why_healthier). IMPORTANT: If the current category is 'leftovers', keep it as 'leftovers' unless the user's correction clearly indicates otherwise."""

    client = _openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1000,
        timeout=30
    )
    return json.loads(response.choices[0].message.content)


def _handle_correct_kitchen_item(owner, item_id, body, conn):
    """Apply a user correction to a kitchen item via LLM re-analysis."""
    correction = (body.get('correction') or '').strip()
    if not correction:
        return _error_response(400, 'Missing required field: correction')

    table_name = f"{owner}_prod_kitchen"

    with conn.cursor() as cur:
        read_table, read_where, read_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)
        if not USE_SHARED_TABLES and not _table_exists(cur, table_name):
            return _error_response(404, f'Kitchen table not found for owner: {owner}')

        verify_where = "`_id` = %s AND `action` = 'IN'"
        if read_where:
            verify_where = f"{read_where} AND {verify_where}"
        cur.execute(
            f"SELECT * FROM `{read_table}` WHERE {verify_where}",
            read_params + [item_id]
        )
        current_row = cur.fetchone()
        if not current_row:
            return _error_response(404, f'Item with id {item_id} not found')

        current_dict = _serialize_rows([current_row])[0]

        try:
            corrected = _correct_kitchen_item_with_llm(current_dict, correction)
        except Exception as e:
            print(f"[ERROR] LLM correction failed: {e}")
            return _error_response(500, f'AI correction failed: {str(e)}')

        # Build UPDATE clause from corrected fields
        field_map = {
            'product_name': corrected.get('product_name'),
            'brand': corrected.get('brand'),
            'category': corrected.get('category'),
            'product_description': corrected.get('description'),
            'ingredients': _json_column_value(corrected.get('ingredients'), default_empty_list=True),
            'nutrition_summary': corrected.get('nutrition_summary'),
            'upf': corrected.get('upf'),
            'harmful_ingredients': _json_column_value(corrected.get('harmful_ingredients'), default_empty_list=True),
            'healthier_alternatives': _json_column_value(corrected.get('healthier_alternatives'), default_empty_list=True),
            'analysis_stage': 'corrected',
        }

        updates = []
        values = []
        for col, val in field_map.items():
            if val is not None:
                updates.append(f"`{col}` = %s")
                values.append(val)

        if not updates:
            return _error_response(500, 'AI returned no corrected fields')

        updates.append("`_updatedDate` = NOW()")
        values.append(item_id)

        update_sql = f"UPDATE `{{tbl}}` SET {', '.join(updates)} WHERE `_id` = %s"

        member_ids = _get_household_member_ids(conn, owner)
        update_dict = {col: val for col, val in field_map.items() if val is not None}
        _shared_kitchen_update(conn, item_id, update_dict)

        # Re-fetch updated row
        refetch_where = "`_id` = %s"
        if read_where:
            refetch_where = f"{read_where} AND {refetch_where}"
        cur.execute(f"SELECT * FROM `{read_table}` WHERE {refetch_where}", read_params + [item_id])
        updated_row = cur.fetchone()
        if not updated_row:
            return _error_response(500, 'Item not found after correction')

        item_dict = _serialize_rows([updated_row])[0]

        _mark_recipe_refresh_needed_for_owners(conn, member_ids)
        _record_master_feed_event(
            conn,
            owner,
            member_ids,
            {
                'event_key': f"kitchen-correct:{item_id}",
                'device_id': item_dict.get('_device'),
                'event_type': 'kitchen_correct',
                'entity_type': 'kitchen_item',
                'action': item_dict.get('action'),
                'title': item_dict.get('product_name'),
                'brand': item_dict.get('brand'),
                'item_id': item_id,
                'job_id': item_dict.get('job_id'),
                'user_id': item_dict.get('user_id'),
                'source_table': table_name,
                'source_path': '/kitchen/{owner}/items/{item_id}/correct',
                'source_system': 'kitchen_api',
                'primary_image_url': item_dict.get('product_image_url') or item_dict.get('images'),
                'secondary_image_url': item_dict.get('images') or item_dict.get('resized_image_url'),
                'metadata': {
                    'correction_text': correction,
                    'analysis_stage': 'corrected',
                },
            },
        )
        _invoke_kitchen_analysis_generator(owner)
        _refresh_shelf_life_cache(owner)

        return _success_response({'item': item_dict})


def _run_sprint4_phase1(event):
    """One-time migration: backup per-user tables + create shared tables."""
    from datetime import datetime as dt
    results = {'backups': [], 'tables_created': [], 'errors': []}
    timestamp = 'bk'  # Short suffix to stay under 64-char MySQL table name limit
    dry_run = event.get('dry_run', False)

    conn = _mysql_conn()
    with conn.cursor() as cur:
        # Step 1: Find and backup all per-user tables
        suffixes = ['_prod_kitchen', '_archive_kitchen', '_dishes', '_new_list', '_master_feed']
        all_tables = []
        for suffix in suffixes:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name LIKE %s
                AND table_name NOT LIKE '%%backup%%'
                AND table_name NOT LIKE 'shared_%%'
            """, [f'%{suffix}'])
            all_tables.extend([row.get('table_name') or row.get('TABLE_NAME') for row in cur.fetchall()])

        print(f"[MIGRATE] Found {len(all_tables)} per-user tables")

        for table_name in sorted(all_tables):
            backup_name = f"{table_name}_backup_{timestamp}"
            cur.execute("SELECT COUNT(*) as c FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s", [backup_name])
            if cur.fetchone()['c'] > 0:
                results['backups'].append({'table': table_name, 'status': 'already_exists'})
                continue
            if dry_run:
                results['backups'].append({'table': table_name, 'status': 'dry_run'})
                continue
            try:
                cur.execute(f"CREATE TABLE `{backup_name}` LIKE `{table_name}`")
                cur.execute(f"INSERT INTO `{backup_name}` SELECT * FROM `{table_name}`")
                conn.commit()
                cur.execute(f"SELECT COUNT(*) as c FROM `{table_name}`")
                orig = cur.fetchone()['c']
                cur.execute(f"SELECT COUNT(*) as c FROM `{backup_name}`")
                bkup = cur.fetchone()['c']
                results['backups'].append({'table': table_name, 'backup': backup_name, 'rows': orig, 'verified': orig == bkup})
            except Exception as e:
                results['errors'].append({'table': table_name, 'error': str(e)})

        # Step 2: Create shared tables
        shared_ddl = {
            'shared_kitchen': """
                CREATE TABLE IF NOT EXISTS `shared_kitchen` (
                    `_id` VARCHAR(36) NOT NULL, `owner_id` VARCHAR(36) NOT NULL, `_owner` VARCHAR(36) NOT NULL,
                    `_device` VARCHAR(255) DEFAULT NULL, `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                    `product_name` VARCHAR(500), `brand` VARCHAR(500), `variant` VARCHAR(500), `category` VARCHAR(255),
                    `barcode` VARCHAR(100), `country_guess` VARCHAR(100),
                    `remaining_quantity` VARCHAR(255), `quantity_value` DECIMAL(10,2), `quantity_unit` VARCHAR(50),
                    `is_opened` TINYINT(1), `fill_percent` INT,
                    `confidence` DECIMAL(3,2), `explanation` TEXT, `product_description` TEXT, `estimated_price` VARCHAR(100),
                    `ingredients` JSON, `nutrition_summary` TEXT, `upf` VARCHAR(10), `harmful_ingredients` JSON,
                    `similar_items` JSON, `alternatives` JSON, `healthier_alternatives` JSON, `store_availability` JSON,
                    `images` VARCHAR(1000), `s3_key` VARCHAR(500),
                    `resized_image_url` VARCHAR(1000), `resized_image_key` VARCHAR(500),
                    `product_image_url` VARCHAR(1000), `product_image_key` VARCHAR(500),
                    `action` VARCHAR(10) NOT NULL DEFAULT 'IN', `product_expiration` VARCHAR(50),
                    `storage_guidance` JSON, `usage_guide` JSON,
                    `analysis_stage` VARCHAR(50) DEFAULT 'preliminary', `analysis_status` VARCHAR(50) DEFAULT 'ready',
                    `analysis_source` VARCHAR(128), `analysis_updated_at` DATETIME,
                    `needs_review` TINYINT(1) DEFAULT 0, `swaps` JSON, `provisional_payload` JSON,
                    `job_id` VARCHAR(255), `user_id` VARCHAR(255),
                    PRIMARY KEY (`_id`), INDEX `idx_owner_id` (`owner_id`),
                    INDEX `idx_owner_action` (`owner_id`, `action`), INDEX `idx_owner_created` (`owner_id`, `_createdDate`),
                    INDEX `idx_job_id` (`job_id`), INDEX `idx_analysis` (`analysis_stage`, `analysis_status`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'shared_archive_kitchen': """
                CREATE TABLE IF NOT EXISTS `shared_archive_kitchen` (
                    `_id` VARCHAR(36) NOT NULL, `owner_id` VARCHAR(36) NOT NULL, `_owner` VARCHAR(36) NOT NULL,
                    `_device` VARCHAR(255), `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                    `product_name` VARCHAR(500), `brand` VARCHAR(500), `variant` VARCHAR(500), `category` VARCHAR(255),
                    `barcode` VARCHAR(100), `country_guess` VARCHAR(100),
                    `remaining_quantity` VARCHAR(255), `quantity_value` DECIMAL(10,2), `quantity_unit` VARCHAR(50),
                    `is_opened` TINYINT(1), `fill_percent` INT,
                    `confidence` DECIMAL(3,2), `explanation` TEXT, `product_description` TEXT, `estimated_price` VARCHAR(100),
                    `ingredients` JSON, `nutrition_summary` TEXT, `upf` VARCHAR(10), `harmful_ingredients` JSON,
                    `similar_items` JSON, `alternatives` JSON, `healthier_alternatives` JSON, `store_availability` JSON,
                    `images` VARCHAR(1000), `s3_key` VARCHAR(500),
                    `resized_image_url` VARCHAR(1000), `resized_image_key` VARCHAR(500),
                    `product_image_url` VARCHAR(1000), `product_image_key` VARCHAR(500),
                    `action` VARCHAR(10) NOT NULL DEFAULT 'IN', `product_expiration` VARCHAR(50),
                    `storage_guidance` JSON, `usage_guide` JSON,
                    `analysis_stage` VARCHAR(50), `analysis_status` VARCHAR(50), `analysis_source` VARCHAR(128),
                    `analysis_updated_at` DATETIME, `needs_review` TINYINT(1) DEFAULT 0,
                    `swaps` JSON, `provisional_payload` JSON,
                    `job_id` VARCHAR(255), `user_id` VARCHAR(255),
                    `archived_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `archived_reason` VARCHAR(100), `archived_from_table` VARCHAR(255),
                    PRIMARY KEY (`_id`), INDEX `idx_owner_id` (`owner_id`),
                    INDEX `idx_owner_archived` (`owner_id`, `archived_at`), INDEX `idx_job_id` (`job_id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'shared_dishes': """
                CREATE TABLE IF NOT EXISTS `shared_dishes` (
                    `_id` VARCHAR(36) NOT NULL, `owner_id` VARCHAR(36) NOT NULL, `_owner` VARCHAR(36) NOT NULL,
                    `_device` VARCHAR(255) NOT NULL DEFAULT 'assistant',
                    `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                    `dish_name` VARCHAR(500), `confidence` DECIMAL(3,2), `explanation` TEXT, `serving_size` VARCHAR(100),
                    `calories` DECIMAL(10,2), `total_fat` DECIMAL(10,2), `saturated_fat` DECIMAL(10,2),
                    `trans_fat` DECIMAL(10,2), `cholesterol` DECIMAL(10,2), `sodium` DECIMAL(10,2),
                    `total_carbohydrates` DECIMAL(10,2), `dietary_fiber` DECIMAL(10,2), `sugars` DECIMAL(10,2),
                    `protein` DECIMAL(10,2), `vitamin_a` DECIMAL(10,2), `vitamin_c` DECIMAL(10,2),
                    `calcium` DECIMAL(10,2), `iron` DECIMAL(10,2),
                    `ingredients` JSON, `components` JSON, `allergens` JSON,
                    `images` VARCHAR(1000), `s3_key` VARCHAR(500),
                    `resized_image_url` VARCHAR(1000), `resized_image_key` VARCHAR(500),
                    `dish_image_url` VARCHAR(1000), `dish_image_key` VARCHAR(500),
                    `action` VARCHAR(10) NOT NULL DEFAULT 'IN',
                    `job_id` VARCHAR(255), `user_id` VARCHAR(255),
                    `analysis_status` VARCHAR(50), `analysis_error` TEXT,
                    PRIMARY KEY (`_id`), INDEX `idx_owner_id` (`owner_id`),
                    INDEX `idx_owner_created` (`owner_id`, `_createdDate`), INDEX `idx_job_id` (`job_id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'shared_list': """
                CREATE TABLE IF NOT EXISTS `shared_list` (
                    `_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    `owner_id` VARCHAR(36) NOT NULL, `_owner` VARCHAR(36) NOT NULL,
                    `_device` VARCHAR(64) NOT NULL, `product_name` VARCHAR(255) NOT NULL,
                    `product_brand` VARCHAR(255), `images` TEXT, `product_barcode` VARCHAR(64),
                    `store` VARCHAR(100), `action` VARCHAR(32) NOT NULL DEFAULT 'ADDED',
                    `sort_order` INT DEFAULT NULL,
                    `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    `household_item_uuid` CHAR(36),
                    PRIMARY KEY (`_id`), INDEX `idx_owner_id` (`owner_id`),
                    INDEX `idx_owner_device` (`owner_id`, `_device`),
                    INDEX `idx_household_uuid` (`household_item_uuid`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        }

        for table_name, ddl in shared_ddl.items():
            if not dry_run:
                try:
                    cur.execute(ddl)
                    conn.commit()
                    results['tables_created'].append(table_name)
                    print(f"[MIGRATE] Created {table_name}")
                except Exception as e:
                    results['errors'].append({'table': table_name, 'error': str(e)})
            else:
                results['tables_created'].append(f"{table_name} (dry_run)")

    return _success_response(results)


# ── Owner block list & token validation ──────────────────────────────
# Blocked owner_ids — manually curated. These users are denied all API access.
_BLOCKED_OWNERS = frozenset([
    'deadbeef-cafe-babe-feed-1234567890ab',
])

def _validate_owner_token(event, owner):
    """Decode the Bearer token and verify the user_id matches the path owner.

    The token is currently a base64-encoded JSON blob (no signature).
    Returns (ok: bool, message: str).
    """
    auth_header = (event.get('headers') or {}).get('authorization', '')
    if not auth_header.startswith('Bearer '):
        # No token — allow for backwards compat (many paths don't send one yet)
        return True, None
    try:
        payload = json.loads(base64.b64decode(auth_header[7:].strip()))
        token_user_id = payload.get('user_id', '')
        if token_user_id and token_user_id != owner:
            print(f"[SECURITY] Owner mismatch: path={owner} token_user_id={token_user_id}")
            return False, 'Owner mismatch: token user_id does not match path owner'
    except Exception:
        # Malformed token — log but don't block (token may be absent/old format)
        pass
    return True, None


def handler(event, context):
    """Handle kitchen API requests"""
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        # Check for internal migration event
        if event.get('sprint4_phase1'):
            return _run_sprint4_phase1(event)

        # Extract HTTP method and path
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        item_id = path_params.get('item_id')

        # ── Security checks ──
        if owner and owner in _BLOCKED_OWNERS:
            print(f"[SECURITY] Blocked owner denied: {owner}")
            return _error_response(403, 'Access denied')

        if owner and http_method != 'OPTIONS':
            ok, msg = _validate_owner_token(event, owner)
            if not ok:
                return _error_response(403, msg)

        # Route based on HTTP method
        if http_method == 'GET':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            return _get_kitchen_items(owner, event)

        elif http_method == 'POST':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            body = json.loads(event.get('body', '{}'))
            if body.get('trigger_recipes_only'):
                _invoke_recipes_generator(owner)
                _invoke_meal_plan_generator(owner)
                return _success_response({'message': 'Recipe and meal plan generation triggered', 'owner': owner})
            raw_path = event.get('rawPath') or event.get('requestContext', {}).get('http', {}).get('path', '')
            if raw_path.rstrip('/').endswith('/text-add'):
                return _handle_text_add(owner, body)
            if raw_path.rstrip('/').endswith('/correct') and item_id:
                conn = _mysql_conn()
                return _handle_correct_kitchen_item(owner, item_id, body, conn)
            return _create_kitchen_item(owner, body)
        
        elif http_method in ['PUT', 'PATCH']:
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            body = json.loads(event.get('body', '{}'))
            return _update_kitchen_item(owner, item_id, body)
        
        elif http_method == 'DELETE':
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            return _delete_kitchen_item(owner, item_id, event)
        
        elif http_method == 'OPTIONS':
            # Handle CORS preflight
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS'
                },
                'body': ''
            }
        
        else:
            return _error_response(405, f'Method {http_method} not allowed')
    
    except json.JSONDecodeError as e:
        return _error_response(400, f'Invalid JSON in request body: {str(e)}')
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Internal server error: {str(e)}')


def _get_kitchen_items(owner, event=None):
    """Get all items from owner's kitchen table"""
    try:
        include_archived = _query_param_truthy(event or {}, 'include_archived')

        conn = _mysql_conn()
        with conn.cursor() as cur:
            table_name, owner_where, owner_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)

            items_list = []
            if not USE_SHARED_TABLES and not _table_exists(cur, table_name):
                response_body = {
                    'owner': owner,
                    'items': [],
                    'count': 0
                }
            else:
                where_clause = "`action` = 'IN'"
                if owner_where:
                    where_clause = f"{owner_where} AND {where_clause}"
                cur.execute(f"""
                    SELECT * FROM `{table_name}`
                    WHERE {where_clause}
                    ORDER BY `_createdDate` DESC
                """, owner_params)
                items_list = _serialize_rows(cur.fetchall())
                response_body = {
                    'owner': owner,
                    'items': items_list,
                    'count': len(items_list)
                }

            if include_archived:
                archived_items = []
                arch_table, arch_where, arch_params = _resolve_kitchen_table(owner, '_archive_kitchen', conn)
                if USE_SHARED_TABLES or _table_exists(cur, arch_table):
                    if not USE_SHARED_TABLES:
                        _ensure_archive_metadata_columns(conn, cur, arch_table)
                    arch_where_clause = "1=1"
                    if arch_where:
                        arch_where_clause = arch_where
                    cur.execute(f"""
                        SELECT * FROM `{arch_table}`
                        WHERE {arch_where_clause}
                        ORDER BY `archived_at` DESC, `_createdDate` DESC
                    """, arch_params)
                    archived_items = _serialize_rows(cur.fetchall())
                response_body['archived_items'] = archived_items
                response_body['archived_count'] = len(archived_items)

            return _success_response(response_body)
    
    except Exception as e:
        print(f"[ERROR] Failed to get kitchen items: {str(e)}")
        return _error_response(500, f'Failed to retrieve kitchen items: {str(e)}')


def _update_kitchen_item(owner, item_id, body):
    """Update a kitchen item (e.g., expiration date)"""
    try:
        table_name = f"{owner}_prod_kitchen"
        
        # Validate allowed fields for update
        field_types = {
            'product_name': 'TEXT',
            'brand': 'TEXT',
            'variant': 'TEXT',
            'category': 'TEXT',
            'remaining_quantity': 'TEXT',
            'quantity_value': 'FLOAT',
            'quantity_unit': 'TEXT',
            'is_opened': 'BOOL',
            'storage_location': 'TEXT',
            'fill_percent': 'INT',
            'confidence': 'CONFIDENCE',
            'explanation': 'TEXT',
            'product_description': 'TEXT',
            'barcode': 'TEXT',
            'country_guess': 'TEXT',
            'estimated_price': 'TEXT',
            'ingredients': 'JSON_LIST',
            'nutrition_summary': 'TEXT',
            'upf': 'UPF',
            'harmful_ingredients': 'JSON_LIST',
            'similar_items': 'JSON',
            'alternatives': 'JSON',
            'healthier_alternatives': 'JSON',
            'store_availability': 'JSON',
            'action': 'ENUM',
            'product_expiration': 'DATE',
            'storage_guidance': 'JSON',
            'resized_image_url': 'TEXT',
            'resized_image_key': 'TEXT',
            'product_image_url': 'TEXT',
            'product_image_key': 'TEXT',
            'analysis_stage': 'TEXT',
            'analysis_status': 'TEXT',
            'analysis_source': 'TEXT',
            'needs_review': 'BOOL',
            'swaps': 'JSON',
            'provisional_payload': 'JSON',
        }
        
        # Build UPDATE query dynamically
        updates = []
        values = []
        update_dict = {}

        for field, field_type in field_types.items():
            if field in body:
                value = body[field]
                
                # Validate expiration date format
                if field == 'product_expiration' and value:
                    try:
                        # Validate date format (YYYY-MM-DD)
                        datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        return _error_response(400, f'Invalid date format for product_expiration. Use YYYY-MM-DD')
                
                # Validate action enum
                if field == 'action' and value:
                    if value not in ['IN', 'OUT']:
                        return _error_response(400, f'Invalid action value. Must be "IN" or "OUT"')

                if field_type == 'TEXT':
                    value = _clean_nullable_string(value)

                if field_type == 'FLOAT':
                    try:
                        value = _parse_optional_float(value, field, minimum=0)
                    except ValueError:
                        return _error_response(400, f'Invalid {field}. Must be a non-negative number or null')

                if field_type == 'INT':
                    try:
                        value = _parse_optional_int(value, field, minimum=0, maximum=100 if field == 'fill_percent' else None)
                    except ValueError:
                        if field == 'fill_percent':
                            return _error_response(400, 'Invalid fill_percent. Must be an integer between 0 and 100 or null')
                        return _error_response(400, f'Invalid {field} value')

                if field_type == 'BOOL':
                    try:
                        parsed_bool = _parse_optional_bool(value, field)
                        value = None if parsed_bool is None else (1 if parsed_bool else 0)
                    except ValueError:
                        return _error_response(400, f'Invalid {field}. Must be true, false, or null')

                if field_type == 'CONFIDENCE':
                    if value in [None, '']:
                        value = None
                    else:
                        try:
                            value = max(0, min(1, float(value)))
                        except (TypeError, ValueError):
                            return _error_response(400, 'Invalid confidence value. Must be a number between 0 and 1')

                if field_type == 'UPF':
                    if value in ['', None]:
                        value = None
                    elif value not in ['yes', 'no']:
                        return _error_response(400, 'Invalid upf value. Must be "yes", "no", or null')

                if field_type == 'JSON':
                    value = _json_column_value(value)

                if field_type == 'JSON_LIST':
                    value = _json_column_value(value, default_empty_list=True)

                updates.append(f"`{field}` = %s")
                values.append(value)
                # Accumulate the VALIDATED/serialized value for the shared-table write.
                update_dict[field] = value

        # Conditional storage_location fill (server-side enrichment): only set it
        # when the column is currently NULL so a user's explicit choice is never
        # clobbered. Skipped entirely if the caller also sent an explicit
        # `storage_location` (that direct field wins).
        sl_if_empty = body.get('storage_location_if_empty')
        if 'storage_location' not in body and sl_if_empty is not None:
            sl_val = str(sl_if_empty).strip().lower() or None
            if sl_val in ('fridge', 'freezer', 'pantry'):
                update_dict['storage_location'] = _CoalesceExisting(sl_val)

        if not update_dict:
            return _error_response(400, 'No valid fields to update.')

        # Stamp analysis_updated_at for parity with CREATE (which sets it in _build_create_payload).
        update_dict['analysis_updated_at'] = datetime.utcnow()
        
        # Add updated timestamp
        updates.append("`_updatedDate` = NOW()")
        
        # Add item_id to values for WHERE clause
        values.append(item_id)
        
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            # Ensure table schema is up to date (adds missing columns like storage_guidance)
            _ensure_prod_kitchen_table(conn, cur, table_name)

            # Check if table exists
            read_table, read_where, read_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)
            if not USE_SHARED_TABLES:
                cur.execute("""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    AND table_name = %s
                """, [table_name])
                table_exists = cur.fetchone()['count'] > 0
                if not table_exists:
                    return _error_response(404, f'Kitchen table not found for owner: {owner}')

            # Check if item exists
            verify_where = "`_id` = %s"
            if read_where:
                verify_where = f"{read_where} AND {verify_where}"
            cur.execute(f"SELECT `_id` FROM `{read_table}` WHERE {verify_where}", read_params + [item_id])
            if not cur.fetchone():
                return _error_response(404, f'Item with id {item_id} not found')
                
            # Ensure _updatedDate column exists (for tracking updates)
            cur.execute(f"""
                SELECT COUNT(*) as count 
                FROM information_schema.columns 
                WHERE table_schema = DATABASE() 
                AND table_name = %s 
                AND column_name = '_updatedDate'
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                cur.execute(f"""
                    ALTER TABLE `{table_name}` 
                    ADD COLUMN `_updatedDate` DATETIME NULL 
                    COMMENT 'Last update timestamp'
                """)
                conn.commit()
                
            # Perform update using the VALIDATED/serialized values accumulated above
            # (raw-body rebuild removed: it passed Python dicts/lists straight to pymysql).
            _shared_kitchen_update(conn, item_id, update_dict)

            # Fetch updated item
            rb_where = "`_id` = %s"
            if read_where:
                rb_where = f"{read_where} AND {rb_where}"
            cur.execute(f"SELECT * FROM `{read_table}` WHERE {rb_where}", read_params + [item_id])
            updated_item = cur.fetchone()

            if not updated_item:
                return _error_response(404, 'Item not found after update')

            # Auto-delete if quantity dropped to 0 or below
            qty_val = updated_item.get('quantity_value')
            if qty_val is not None and float(qty_val) <= 0:
                print(f"[UPDATE] quantity_value={qty_val} for item {item_id}, auto-deleting")
                return _delete_kitchen_item(owner, item_id)
                
            # Convert to JSON-serializable format
            item_dict = {}
            for key, value in updated_item.items():
                try:
                    item_dict[key] = json_serial(value)
                except Exception as e:
                    print(f"[WARN] Failed to serialize {key}: {type(value)} - {str(e)}")
                    item_dict[key] = str(value) if value is not None else None
            changed_fields = sorted(field for field in field_types.keys() if field in body)
            if _should_bump_kitchen_version_for_fields(changed_fields):
                _mark_recipe_refresh_needed_for_owners(conn, member_ids)
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f"kitchen-update:{item_id}:{item_dict.get('_updatedDate') or ''}",
                    'device_id': item_dict.get('_device'),
                    'event_type': 'kitchen_update',
                    'entity_type': 'kitchen_item',
                    'action': item_dict.get('action'),
                    'title': item_dict.get('product_name'),
                    'brand': item_dict.get('brand'),
                    'item_id': item_id,
                    'job_id': item_dict.get('job_id'),
                    'user_id': item_dict.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/kitchen/{owner}/{item_id}',
                    'source_system': 'kitchen_api',
                    'primary_image_url': item_dict.get('product_image_url') or item_dict.get('images'),
                    'secondary_image_url': item_dict.get('images') or item_dict.get('resized_image_url'),
                    'metadata': {
                        'changed_fields': changed_fields,
                        'product_expiration': item_dict.get('product_expiration'),
                        'analysis_status': item_dict.get('analysis_status'),
                    },
                },
            )
            _invoke_kitchen_analysis_generator(owner)
            _refresh_shelf_life_cache(owner)
            return _success_response({
                'message': 'Item updated successfully',
                'item': item_dict
            })
    
    except Exception as e:
        print(f"[ERROR] Failed to update kitchen item: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to update item: {str(e)}')


def _delete_archived_kitchen_item(owner, item_id):
    """Delete an already-archived kitchen item from archive history/feed."""
    try:
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            # Delete from shared archive table
            cur.execute("SELECT `_id` FROM `shared_archive_kitchen` WHERE `_id` = %s", [item_id])
            if not cur.fetchone():
                return _error_response(404, f'Archived item with id {item_id} not found')
            cur.execute("DELETE FROM `shared_archive_kitchen` WHERE `_id` = %s", [item_id])
            deleted_count = int(cur.rowcount or 0)
            conn.commit()

            return _success_response({
                'message': 'Archived item deleted successfully',
                'item_id': item_id,
                'archived': True,
                'deleted_from_archive': True,
                'deleted_count': deleted_count,
            })

    except Exception as e:
        print(f"[ERROR] Failed to delete archived kitchen item: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to delete archived item: {str(e)}')


def _delete_kitchen_item(owner, item_id, event=None):
    """Delete a kitchen item (mark as used/consumed)"""
    try:
        if _query_param_truthy(event or {}, 'archived'):
            return _delete_archived_kitchen_item(owner, item_id)

        table_name = f"{owner}_prod_kitchen"

        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            read_table, read_where, read_params = _resolve_kitchen_table(owner, '_prod_kitchen', conn)

            if not USE_SHARED_TABLES and not _table_exists(cur, table_name):
                return _error_response(404, f'Kitchen table not found for owner: {owner}')

            # Check if item exists
            verify_where = "`_id` = %s"
            if read_where:
                verify_where = f"{read_where} AND {verify_where}"
            cur.execute(f"SELECT * FROM `{read_table}` WHERE {verify_where} LIMIT 1", read_params + [item_id])
            existing_item = cur.fetchone()
            if not existing_item:
                return _error_response(404, f'Item with id {item_id} not found')
            existing_serialized = _serialize_rows([existing_item])[0]
                
            # Delete from shared table (primary path)
            _shared_kitchen_delete(conn, owner, item_id)

            _mark_recipe_refresh_needed_for_owners(conn, member_ids)

        # Trigger meal plan and recipes regeneration (async)
        _invoke_kitchen_analysis_generator(owner)
        _refresh_shelf_life_cache(owner)
        _invoke_meal_plan_generator(owner)
        _invoke_recipes_generator(owner)
        _record_master_feed_event(
            conn,
            owner,
            member_ids,
            {
                'event_key': f'kitchen-remove:{item_id}',
                'device_id': existing_serialized.get('_device'),
                'event_type': 'kitchen_remove',
                'entity_type': 'kitchen_item',
                'action': 'OUT',
                'title': existing_serialized.get('product_name'),
                'brand': existing_serialized.get('brand'),
                'item_id': item_id,
                'job_id': existing_serialized.get('job_id'),
                'user_id': existing_serialized.get('user_id'),
                'source_table': f'{owner}_archive_kitchen',
                'source_path': '/kitchen/{owner}/{item_id}',
                'source_system': 'kitchen_api',
                'primary_image_url': existing_serialized.get('product_image_url') or existing_serialized.get('images'),
                'secondary_image_url': existing_serialized.get('images') or existing_serialized.get('resized_image_url'),
                'metadata': {
                    'archived': True,
                    'archived_reason': 'manual_delete',
                    'removed_from_table': table_name,
                },
            },
        )

        return _success_response({
            'message': 'Item deleted successfully',
            'item_id': item_id,
            'archived': True,
        })
    
    except Exception as e:
        print(f"[ERROR] Failed to delete kitchen item: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to delete item: {str(e)}')


def _success_response(data, status_code=200):
    """Return successful response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS'
        },
        'body': json.dumps(data, default=json_serial)
    }


def _error_response(status_code, message):
    """Return error response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS'
        },
        'body': json.dumps({
            'error': message
        })
    }
