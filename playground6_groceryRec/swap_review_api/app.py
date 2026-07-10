import json
import os
import re
from datetime import datetime
from decimal import Decimal

try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
_ALLOWED_STATUSES = {'pending', 'dismissed', 'applied'}


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
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5')),
        read_timeout=int(os.getenv('DB_READ_TIMEOUT_SECONDS', '30')),
        write_timeout=int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '30')),
        autocommit=True,
    )
    return _conn


def _sanitize_owner(value):
    return re.sub(r'[^a-zA-Z0-9_-]', '', str(value or ''))


def _swap_reviews_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_swap_reviews"


def _kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_prod_kitchen"


def _success_response(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        },
        'body': json.dumps(data, default=_json_serial),
    }


def _error_response(status_code, message):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        },
        'body': json.dumps({'error': message}),
    }


def _json_serial(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, dict):
        return {k: _json_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serial(item) for item in obj]
    return str(obj)


def _decode_json_field(value):
    if value is None or isinstance(value, (dict, list)):
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


def _table_exists(cur, table_name):
    cur.execute(
        """
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        [table_name],
    )
    row = cur.fetchone() or {}
    return row.get('count', 0) > 0


def _get_table_columns(cur, table_name):
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        [table_name],
    )
    return {
        row.get('column_name') or row.get('COLUMN_NAME')
        for row in (cur.fetchall() or [])
        if (row.get('column_name') or row.get('COLUMN_NAME'))
    }


def _ensure_swap_reviews_table(conn, table_name):
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                  `_id` VARCHAR(36) PRIMARY KEY COMMENT 'Stable swap review prompt id',
                  `_owner` VARCHAR(36) NOT NULL COMMENT 'Owner/user table owner',
                  `user_id` VARCHAR(255) NOT NULL COMMENT 'App user who sees this prompt',
                  `kitchen_item_id` VARCHAR(36) NOT NULL COMMENT 'Checked-in kitchen row id',
                  `job_id` VARCHAR(100) NOT NULL COMMENT 'Check-in job id',
                  `status` ENUM('pending','dismissed','applied') NOT NULL DEFAULT 'pending' COMMENT 'Per-user review status',
                  `reviewed_at` DATETIME NULL COMMENT 'When the user first reviewed this prompt',
                  `dismissed_at` DATETIME NULL COMMENT 'When the user dismissed this prompt',
                  `applied_at` DATETIME NULL COMMENT 'When the user applied this prompt',
                  `selected_candidate_ids` JSON NULL COMMENT 'Candidate kitchen ids chosen by the user',
                  `candidate_snapshot` JSON NULL COMMENT 'Snapshot of swap suggestions shown to the user',
                  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Prompt creation time',
                  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last prompt update time',
                  UNIQUE KEY `uniq_user_item` (`user_id`, `kitchen_item_id`),
                  INDEX `idx_owner_status_created` (`_owner`, `status`, `created_at`),
                  INDEX `idx_job_id` (`job_id`),
                  INDEX `idx_kitchen_item_id` (`kitchen_item_id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Per-user swap review prompts'
                """
            )
            conn.commit()
            return

        columns = _get_table_columns(cur, table_name)
        alter_parts = []
        if 'selected_candidate_ids' not in columns:
            alter_parts.append(
                "ADD COLUMN `selected_candidate_ids` JSON NULL COMMENT 'Candidate kitchen ids chosen by the user' AFTER `applied_at`"
            )
        if 'candidate_snapshot' not in columns:
            alter_parts.append(
                "ADD COLUMN `candidate_snapshot` JSON NULL COMMENT 'Snapshot of swap suggestions shown to the user' AFTER `selected_candidate_ids`"
            )
        if 'updated_at' not in columns:
            alter_parts.append(
                "ADD COLUMN `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last prompt update time' AFTER `created_at`"
            )
        # Each column as its OWN ALTER, tolerating errno 1060 (Duplicate column) so a
        # concurrent request that already added it doesn't fail this one. Per-column so
        # a dup on one never blocks the others.
        for ddl in alter_parts:
            try:
                cur.execute(f"ALTER TABLE `{table_name}` {ddl}")
                conn.commit()
            except Exception as e:
                if getattr(e, 'args', (None,))[0] == 1060:
                    conn.rollback()
                else:
                    raise


def _parse_limit(query):
    try:
        limit = int((query or {}).get('limit') or _DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT
    return max(1, min(limit, _MAX_LIST_LIMIT))


def _parse_statuses(query):
    raw = str((query or {}).get('status') or 'pending').strip().lower()
    if not raw:
        return ['pending']
    statuses = []
    for part in raw.split(','):
        status = part.strip().lower()
        if not status:
            continue
        if status not in _ALLOWED_STATUSES:
            raise ValueError('status must be pending, dismissed, or applied')
        if status not in statuses:
            statuses.append(status)
    return statuses or ['pending']


def _require_user_id(value):
    user_id = _sanitize_owner(value)
    if not user_id:
        raise ValueError('user_id is required')
    return user_id


def _parse_body(event):
    raw = event.get('body')
    if not raw:
        return {}
    return json.loads(raw)


def _candidate_ids_from_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return []
    raw_ids = snapshot.get('candidate_ids')
    if not isinstance(raw_ids, list):
        return []
    result = []
    for value in raw_ids:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in result:
            result.append(text)
    return result


def _load_kitchen_items_by_ids(cur, owner, item_ids):
    use_shared = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'
    table_name = 'shared_kitchen' if use_shared else _kitchen_table_name(owner)
    if not item_ids or (not use_shared and not _table_exists(cur, table_name)):
        return {}
    placeholders = ', '.join(['%s'] * len(item_ids))
    cur.execute(
        f"""
        SELECT * FROM `{table_name}`
        WHERE `_id` IN ({placeholders}) AND `action` = 'IN'
        """,
        item_ids,
    )
    items = {}
    for row in (cur.fetchall() or []):
        item = {}
        for key, value in row.items():
            if key == 'swaps':
                item[key] = _decode_json_field(value)
            else:
                item[key] = _json_serial(value)
        items[str(row.get('_id'))] = item
    return items


def _list_swap_reviews(owner, event):
    query = event.get('queryStringParameters') or {}
    user_id = _require_user_id(query.get('user_id'))
    limit = _parse_limit(query)
    statuses = _parse_statuses(query)
    table_name = _swap_reviews_table_name(owner)

    conn = _mysql_conn()
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return _success_response({
                'owner': owner,
                'user_id': user_id,
                'items': [],
                'count': 0,
                'limit': limit,
            })

        placeholders = ', '.join(['%s'] * len(statuses))
        cur.execute(
            f"""
            SELECT *
            FROM `{table_name}`
            WHERE `user_id` = %s AND `status` IN ({placeholders})
            ORDER BY `updated_at` DESC, `created_at` DESC
            LIMIT %s
            """,
            [user_id, *statuses, limit],
        )
        rows = cur.fetchall() or []

        referenced_ids = []
        prompt_rows = []
        for row in rows:
            snapshot = _decode_json_field(row.get('candidate_snapshot'))
            selected_ids = _decode_json_field(row.get('selected_candidate_ids'))
            kitchen_item_id = str(row.get('kitchen_item_id') or '')
            if kitchen_item_id:
                referenced_ids.append(kitchen_item_id)
            candidate_ids = _candidate_ids_from_snapshot(snapshot)
            referenced_ids.extend(candidate_ids)
            prompt_rows.append({
                'prompt_id': row.get('_id'),
                'owner': row.get('_owner'),
                'user_id': row.get('user_id'),
                'kitchen_item_id': kitchen_item_id,
                'job_id': row.get('job_id'),
                'status': row.get('status'),
                'reviewed_at': _json_serial(row.get('reviewed_at')),
                'dismissed_at': _json_serial(row.get('dismissed_at')),
                'applied_at': _json_serial(row.get('applied_at')),
                'selected_candidate_ids': selected_ids if isinstance(selected_ids, list) else [],
                'candidate_snapshot': snapshot,
                'created_at': _json_serial(row.get('created_at')),
                'updated_at': _json_serial(row.get('updated_at')),
                'candidate_ids': candidate_ids,
            })

        referenced_ids = list(dict.fromkeys([item_id for item_id in referenced_ids if item_id]))
        kitchen_items = _load_kitchen_items_by_ids(cur, owner, referenced_ids)

        response_items = []
        for prompt in prompt_rows:
            checked_in_item = kitchen_items.get(prompt['kitchen_item_id'])
            if prompt['status'] == 'pending' and not checked_in_item:
                continue
            candidate_items = [
                kitchen_items[item_id]
                for item_id in prompt['candidate_ids']
                if item_id in kitchen_items
            ]
            response_items.append({
                **prompt,
                'checked_in_item': checked_in_item,
                'candidate_items': candidate_items,
                'missing_candidate_ids': [
                    item_id for item_id in prompt['candidate_ids'] if item_id not in kitchen_items
                ],
            })

        return _success_response({
            'owner': owner,
            'user_id': user_id,
            'items': response_items,
            'count': len(response_items),
            'limit': limit,
        })


def _load_prompt_for_update(cur, owner, kitchen_item_id, user_id):
    table_name = _swap_reviews_table_name(owner)
    if not _table_exists(cur, table_name):
        return table_name, None
    cur.execute(
        f"""
        SELECT *
        FROM `{table_name}`
        WHERE `user_id` = %s AND `kitchen_item_id` = %s
        LIMIT 1
        """,
        [user_id, kitchen_item_id],
    )
    return table_name, cur.fetchone()


def _dismiss_swap_review(owner, kitchen_item_id, body):
    user_id = _require_user_id(body.get('user_id'))
    conn = _mysql_conn()
    with conn.cursor() as cur:
        table_name, row = _load_prompt_for_update(cur, owner, kitchen_item_id, user_id)
        if not row:
            return _error_response(404, 'Swap review prompt not found')

        cur.execute(
            f"""
            UPDATE `{table_name}`
            SET `status` = 'dismissed',
                `reviewed_at` = COALESCE(`reviewed_at`, NOW()),
                `dismissed_at` = NOW()
            WHERE `user_id` = %s AND `kitchen_item_id` = %s
            """,
            [user_id, kitchen_item_id],
        )
        conn.commit()

    return _success_response({
        'message': 'Swap review dismissed',
        'owner': owner,
        'user_id': user_id,
        'kitchen_item_id': kitchen_item_id,
        'status': 'dismissed',
    })


def _apply_swap_review(owner, kitchen_item_id, body):
    user_id = _require_user_id(body.get('user_id'))
    selected_candidate_ids = body.get('selected_candidate_ids')
    if not isinstance(selected_candidate_ids, list):
        return _error_response(400, 'selected_candidate_ids must be an array of kitchen row ids')

    cleaned_ids = []
    for value in selected_candidate_ids:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in cleaned_ids:
            cleaned_ids.append(text)
    if not cleaned_ids:
        return _error_response(400, 'selected_candidate_ids must contain at least one kitchen row id')

    conn = _mysql_conn()
    with conn.cursor() as cur:
        table_name, row = _load_prompt_for_update(cur, owner, kitchen_item_id, user_id)
        if not row:
            return _error_response(404, 'Swap review prompt not found')
        snapshot = _decode_json_field(row.get('candidate_snapshot'))
        allowed_ids = set(_candidate_ids_from_snapshot(snapshot))
        if not allowed_ids:
            return _error_response(409, 'Swap review prompt no longer has selectable candidates')
        invalid_ids = [item_id for item_id in cleaned_ids if item_id not in allowed_ids]
        if invalid_ids:
            return _error_response(400, f'Invalid selected_candidate_ids: {invalid_ids}')

        cur.execute(
            f"""
            UPDATE `{table_name}`
            SET `status` = 'applied',
                `reviewed_at` = COALESCE(`reviewed_at`, NOW()),
                `applied_at` = NOW(),
                `selected_candidate_ids` = %s
            WHERE `user_id` = %s AND `kitchen_item_id` = %s
            """,
            [json.dumps(cleaned_ids), user_id, kitchen_item_id],
        )
        conn.commit()

    return _success_response({
        'message': 'Swap review marked applied',
        'owner': owner,
        'user_id': user_id,
        'kitchen_item_id': kitchen_item_id,
        'status': 'applied',
        'selected_candidate_ids': cleaned_ids,
    })


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        kitchen_item_id = path_params.get('item_id')
        raw_path = (event.get('rawPath') or '').rstrip('/')

        if http_method == 'GET':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            return _list_swap_reviews(owner, event)

        if http_method == 'POST':
            if not owner or not kitchen_item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            body = _parse_body(event)
            if raw_path.endswith('/dismiss'):
                return _dismiss_swap_review(owner, kitchen_item_id, body)
            if raw_path.endswith('/apply'):
                return _apply_swap_review(owner, kitchen_item_id, body)
            return _error_response(404, 'Unsupported swap review action')

        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                },
                'body': '',
            }

        return _error_response(405, f'Method {http_method} not allowed')
    except ValueError as exc:
        return _error_response(400, str(exc))
    except json.JSONDecodeError as exc:
        return _error_response(400, f'Invalid JSON in request body: {str(exc)}')
    except Exception as exc:
        print(f"[ERROR] Unexpected swap review error: {str(exc)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Internal server error: {str(exc)}')
