# dishes_api/app.py
# Force republish after dependency-layer restore.
# API to fetch and manage dish records from {owner}_dishes MySQL table.
import os
import json
import boto3
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal
from master_feed import write_master_feed_event

s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'trepo-grocery-uploads-dev')
PRESIGN_TTL_S = 3600

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'


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


def _sanitize_user_id(user_id):
    return ''.join(ch for ch in str(user_id or '') if ch.isalnum() or ch in '-_')


def _parse_limit(query):
    try:
        limit = int((query or {}).get('limit') or _DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT
    return max(1, min(limit, _MAX_LIST_LIMIT))


def _parse_before(query):
    raw = str((query or {}).get('before') or '').strip()
    if not raw:
        return None
    normalized = raw.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _presign_s3_object(bucket, key):
    if not bucket or not key:
        return None
    return s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=PRESIGN_TTL_S
    )


def _report_backend_error(op, owner_id=None, code=None, error=None, job_id=None, service='dishes'):
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


def _presign_dish_images(row):
    if not row:
        return row
    try:
        if row.get('resized_image_key'):
            row['resized_image_url'] = _presign_s3_object(BUCKET_NAME, row['resized_image_key'])
        if row.get('s3_key'):
            row['images'] = _presign_s3_object(BUCKET_NAME, row['s3_key'])
        if row.get('dish_image_key'):
            row['dish_image_url'] = _presign_s3_object(BUCKET_NAME, row['dish_image_key'])
    except Exception as e:
        print(f"[presign] Failed to presign dish images: {str(e)}")
        _report_backend_error('presign_images', owner_id=row.get('user_id') or row.get('_owner'),
                              code='presign_failed', error=e, job_id=row.get('job_id'))
    return row


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


def _resolve_table(owner, suffix, conn=None):
    """Route reads to shared table (with owner_id filter) or per-user table."""
    safe = _sanitize_user_id(owner)
    if not safe:
        raise ValueError('Invalid owner')
    if not USE_SHARED_TABLES:
        return f"{safe}{suffix}", "", []
    shared_map = {
        '_prod_kitchen': 'shared_kitchen',
        '_archive_kitchen': 'shared_archive_kitchen',
        '_dishes': 'shared_dishes',
    }
    shared_name = shared_map.get(suffix)
    if not shared_name:
        return f"{safe}{suffix}", "", []
    # Dishes are user-specific — read from per-user table, not shared
    if suffix == '_dishes':
        return f"{safe}{suffix}", "", []
    member_ids = _get_household_member_ids(conn, owner) if conn else [safe]
    placeholders = ','.join(['%s'] * len(member_ids))
    return shared_name, f"owner_id IN ({placeholders})", member_ids


def _record_master_feed_event(conn, owner, member_ids, record):
    try:
        write_master_feed_event(conn, owner, record, member_ids=member_ids)
    except Exception as exc:
        print(f"[WARN] Master feed write failed: {exc}")
        _report_backend_error('master_feed_write', owner_id=owner, code='feed_write_failed',
                              error=exc, job_id=(record or {}).get('job_id'))


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        item_id = path_params.get('item_id')

        if http_method == 'GET':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            if item_id:
                return _get_dish_by_id(owner, item_id)
            return _get_dishes(owner, event.get('queryStringParameters') or {})

        elif http_method in ['PUT', 'PATCH']:
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            body = json.loads(event.get('body', '{}'))
            return _update_dish(owner, item_id, body)

        elif http_method == 'POST':
            if not owner:
                return _error_response(400, 'Missing owner parameter')
            body = json.loads(event.get('body', '{}') or '{}')
            return _create_dish(owner, body)

        elif http_method == 'DELETE':
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            return _delete_dish(owner, item_id)

        elif http_method == 'OPTIONS':
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


def _ensure_dishes_table(conn, table_name, owner):
    """Create the dishes table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `_id` varchar(36) NOT NULL,
                `_owner` varchar(36) NOT NULL,
                `_device` varchar(255) NOT NULL DEFAULT 'assistant',
                `_createdDate` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `_updatedDate` datetime DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                `dish_name` varchar(500) DEFAULT NULL,
                `confidence` decimal(3,2) DEFAULT NULL,
                `explanation` text,
                `serving_size` varchar(100) DEFAULT NULL,
                `calories` decimal(10,2) DEFAULT NULL,
                `total_fat` decimal(10,2) DEFAULT NULL,
                `saturated_fat` decimal(10,2) DEFAULT NULL,
                `trans_fat` decimal(10,2) DEFAULT NULL,
                `cholesterol` decimal(10,2) DEFAULT NULL,
                `sodium` decimal(10,2) DEFAULT NULL,
                `total_carbohydrates` decimal(10,2) DEFAULT NULL,
                `dietary_fiber` decimal(10,2) DEFAULT NULL,
                `sugars` decimal(10,2) DEFAULT NULL,
                `protein` decimal(10,2) DEFAULT NULL,
                `vitamin_a` decimal(10,2) DEFAULT NULL,
                `vitamin_c` decimal(10,2) DEFAULT NULL,
                `calcium` decimal(10,2) DEFAULT NULL,
                `iron` decimal(10,2) DEFAULT NULL,
                `ingredients` json DEFAULT NULL,
                `components` json DEFAULT NULL,
                `allergens` json DEFAULT NULL,
                `images` varchar(1000) DEFAULT NULL,
                `s3_key` varchar(500) DEFAULT NULL,
                `action` enum('IN','OUT') NOT NULL DEFAULT 'IN',
                `resized_image_key` varchar(500) DEFAULT NULL,
                `resized_image_url` varchar(1000) DEFAULT NULL,
                `dish_image_url` varchar(1000) DEFAULT NULL,
                `dish_image_key` varchar(500) DEFAULT NULL,
                `job_id` varchar(100) DEFAULT NULL,
                `user_id` varchar(255) DEFAULT NULL,
                `analysis_status` varchar(32) DEFAULT NULL,
                `analysis_error` text,
                PRIMARY KEY (`_id`),
                KEY `idx_owner` (`_owner`),
                KEY `idx_created` (`_createdDate`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        conn.commit()


def _create_dish(owner, body):
    """POST /dishes/{owner} - create a preliminary dish record (e.g., from assistant voice/text log)."""
    import uuid
    try:
        dish_name = (body.get('dish_name') or '').strip()
        if not dish_name:
            return _error_response(400, 'dish_name is required')

        dish_id = str(uuid.uuid4())
        table_name = _table_name(owner)

        conn = _mysql_conn()
        _ensure_dishes_table(conn, table_name, owner)
        # Dishes are user-specific — no household sync
        member_ids = [_sanitize_user_id(owner)]

        calories = body.get('calories')
        protein = body.get('protein')
        total_fat = body.get('total_fat')
        total_carbs = body.get('total_carbohydrates')
        serving_size = body.get('serving_size')
        explanation = body.get('explanation')
        ingredients = body.get('ingredients')
        device_id = body.get('device_id', 'assistant')
        user_id = body.get('user_id')
        analysis_status = body.get('analysis_status', 'preliminary')

        ingredients_json = json.dumps(ingredients) if ingredients else None

        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO `{table_name}`
                (_id, _owner, _device, dish_name, calories, protein, total_fat,
                 total_carbohydrates, serving_size, explanation, ingredients,
                 user_id, analysis_status, action)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'IN')
            """, (dish_id, owner, device_id, dish_name,
                  calories, protein, total_fat, total_carbs,
                  serving_size, explanation, ingredients_json,
                  user_id, analysis_status))
            conn.commit()

            read_tbl, read_where, read_params = _resolve_table(owner, '_dishes', conn)
            read_q = f"SELECT * FROM `{read_tbl}` WHERE `_id` = %s"
            read_p = [dish_id]
            if read_where:
                read_q = f"SELECT * FROM `{read_tbl}` WHERE {read_where} AND `_id` = %s"
                read_p = list(read_params) + [dish_id]
            cur.execute(read_q, read_p)
            row = cur.fetchone()

        dish_dict = {k: json_serial(v) for k, v in (row or {}).items()}
        _record_master_feed_event(conn, owner, member_ids, dish_dict)

        return _success_response({'dish': dish_dict}, status_code=201)

    except ValueError as e:
        return _error_response(400, str(e))
    except Exception as e:
        print(f"[ERROR] Failed to create dish: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to create dish: {str(e)}')


def _table_name(owner):
    safe = _sanitize_user_id(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_dishes"


def _get_dishes(owner, query=None):
    """GET /dishes/{owner} - list all dishes for owner (from {owner}_dishes)."""
    try:
        limit = _parse_limit(query)
        before = _parse_before(query)
        conn = _mysql_conn()
        table_name, owner_where, owner_params = _resolve_table(owner, '_dishes', conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                return _success_response({
                    'owner': owner,
                    'dishes': [],
                    'count': 0
                })

            where_parts = []
            params = list(owner_params)
            if owner_where:
                where_parts.append(owner_where)
            if before:
                where_parts.append("COALESCE(`_updatedDate`, `_createdDate`) < %s")
                params.append(before)
            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
            cur.execute(f"""
                SELECT * FROM `{table_name}`
                {where_clause}
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC
                LIMIT {limit + 1}
            """, params)
            rows = cur.fetchall() or []
            has_more = len(rows) > limit
            rows = rows[:limit]

            dishes_list = []
            for row in rows:
                dish_dict = {}
                for key, value in row.items():
                    try:
                        dish_dict[key] = json_serial(value)
                    except Exception as e:
                        print(f"[WARN] Serialize {key}: {e}")
                        dish_dict[key] = str(value) if value is not None else None
                _presign_dish_images(dish_dict)
                dishes_list.append(dish_dict)

            return _success_response({
                'owner': owner,
                'dishes': dishes_list,
                'count': len(dishes_list),
                'limit': limit,
                'has_more': has_more
            })
    except ValueError as e:
        return _error_response(400, str(e))
    except Exception as e:
        print(f"[ERROR] Failed to get dishes: {str(e)}")
        return _error_response(500, f'Failed to retrieve dishes: {str(e)}')


def _get_dish_by_id(owner, item_id):
    """GET /dishes/{owner}/{item_id} - get a single dish by _id."""
    try:
        conn = _mysql_conn()
        table_name, owner_where, owner_params = _resolve_table(owner, '_dishes', conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                return _error_response(404, f'Dishes table not found for owner: {owner}')

            where = "`_id` = %s"
            params = [item_id]
            if owner_where:
                where = f"{owner_where} AND {where}"
                params = list(owner_params) + params
            cur.execute(f"SELECT * FROM `{table_name}` WHERE {where}", params)
            row = cur.fetchone()
            if not row:
                return _error_response(404, f'Dish with id {item_id} not found')

            dish_dict = {k: json_serial(v) for k, v in row.items()}
            _presign_dish_images(dish_dict)
            return _success_response({'dish': dish_dict})
    except ValueError as e:
        return _error_response(400, str(e))
    except Exception as e:
        print(f"[ERROR] Failed to get dish: {str(e)}")
        return _error_response(500, f'Failed to retrieve dish: {str(e)}')


def _update_dish(owner, item_id, body):
    """PATCH/PUT /dishes/{owner}/{item_id} - update dish (e.g. action, serving_size)."""
    try:
        table_name = _table_name(owner)
        allowed_fields = {
            'action': ('IN', 'OUT'),
            'serving_size': None,
        }
        updates = []
        values = []

        for field, constraint in allowed_fields.items():
            if field not in body:
                continue
            value = body[field]
            if constraint and value not in constraint:
                return _error_response(400, f'Invalid {field}. Allowed: {constraint}')
            updates.append(f"`{field}` = %s")
            values.append(value)

        if not updates:
            return _error_response(400, 'No valid fields to update. Allowed: action, serving_size')

        updates.append("`_updatedDate` = NOW()")
        values.append(item_id)

        conn = _mysql_conn()
        # Dishes are user-specific — no household sync
        member_ids = [_sanitize_user_id(owner)]
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                return _error_response(404, f'Dishes table not found for owner: {owner}')

            cur.execute(f"SELECT `_id` FROM `{table_name}` WHERE `_id` = %s", [item_id])
            if not cur.fetchone():
                return _error_response(404, f'Dish with id {item_id} not found')

            query = f"UPDATE `{table_name}` SET {', '.join(updates)} WHERE `_id` = %s"
            cur.execute(query, values)
            conn.commit()

            cur.execute(f"SELECT * FROM `{table_name}` WHERE `_id` = %s", [item_id])
            row = cur.fetchone()
            dish_dict = {k: json_serial(v) for k, v in row.items()}
            changed_fields = sorted(field for field in allowed_fields.keys() if field in body)
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f"dish-update:{item_id}:{dish_dict.get('_updatedDate') or ''}",
                    'device_id': dish_dict.get('_device'),
                    'event_type': 'dish_update',
                    'entity_type': 'dish',
                    'action': dish_dict.get('action'),
                    'title': dish_dict.get('dish_name'),
                    'item_id': item_id,
                    'job_id': dish_dict.get('job_id'),
                    'user_id': dish_dict.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/dishes/{owner}/{item_id}',
                    'source_system': 'dishes_api',
                    'primary_image_url': dish_dict.get('dish_image_url') or dish_dict.get('images'),
                    'secondary_image_url': dish_dict.get('images') or dish_dict.get('resized_image_url'),
                    'metadata': {
                        'changed_fields': changed_fields,
                        'serving_size': dish_dict.get('serving_size'),
                    },
                },
            )
            return _success_response({'message': 'Dish updated successfully', 'dish': dish_dict})
    except ValueError as e:
        return _error_response(400, str(e))
    except Exception as e:
        print(f"[ERROR] Failed to update dish: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to update dish: {str(e)}')


def _delete_dish(owner, item_id):
    """DELETE /dishes/{owner}/{item_id} - delete a dish record."""
    try:
        table_name = _table_name(owner)
        conn = _mysql_conn()
        # Dishes are user-specific — no household sync
        member_ids = [_sanitize_user_id(owner)]
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                return _error_response(404, f'Dishes table not found for owner: {owner}')

            cur.execute(f"SELECT * FROM `{table_name}` WHERE `_id` = %s LIMIT 1", [item_id])
            existing_row = cur.fetchone()
            if not existing_row:
                return _error_response(404, f'Dish with id {item_id} not found')
            dish_dict = {k: json_serial(v) for k, v in existing_row.items()}

            cur.execute(f"DELETE FROM `{table_name}` WHERE `_id` = %s", [item_id])
            conn.commit()
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f'dish-remove:{item_id}',
                    'device_id': dish_dict.get('_device'),
                    'event_type': 'dish_remove',
                    'entity_type': 'dish',
                    'action': dish_dict.get('action'),
                    'title': dish_dict.get('dish_name'),
                    'item_id': item_id,
                    'job_id': dish_dict.get('job_id'),
                    'user_id': dish_dict.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/dishes/{owner}/{item_id}',
                    'source_system': 'dishes_api',
                    'primary_image_url': dish_dict.get('dish_image_url') or dish_dict.get('images'),
                    'secondary_image_url': dish_dict.get('images') or dish_dict.get('resized_image_url'),
                    'metadata': {
                        'removed': True,
                    },
                },
            )
            return _success_response({'message': 'Dish deleted successfully', 'item_id': item_id})
    except ValueError as e:
        return _error_response(400, str(e))
    except Exception as e:
        print(f"[ERROR] Failed to delete dish: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to delete dish: {str(e)}')


def _success_response(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,PUT,PATCH,DELETE,OPTIONS'
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
            'Access-Control-Allow-Methods': 'GET,PUT,PATCH,DELETE,OPTIONS'
        },
        'body': json.dumps({'error': message})
    }
