# discards_api/app.py - API for {owner}_discards (discarded groceries)
import os
# Force republish after dependency-layer restore.
import json
import boto3
from urllib.parse import urlparse
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal
from master_feed import write_master_feed_event

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
_IMAGE_URL_TTL_SECONDS = int(os.getenv('IMAGE_URL_TTL_SECONDS', '604800'))

_s3_client = None


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


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def _parse_s3_url(value):
    text = str(value or '').strip()
    if not text or 'amazonaws.com/' not in text:
        return None, None
    try:
        parsed = urlparse(text)
        host = parsed.netloc or ''
        path = (parsed.path or '').lstrip('/')
        if not path:
            return None, None
        if '.s3.' in host or host.endswith('.s3.amazonaws.com'):
            bucket = host.split('.s3', 1)[0]
            return bucket, path
        if host == 's3.amazonaws.com' and '/' in path:
            bucket, key = path.split('/', 1)
            return bucket, key
    except Exception:
        return None, None
    return None, None


def _report_backend_error(op, owner_id=None, code=None, error=None, job_id=None, service='discards'):
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


def _generate_read_url(bucket, key):
    if not bucket or not key:
        return None
    try:
        return _get_s3_client().generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=_IMAGE_URL_TTL_SECONDS,
        )
    except Exception as exc:
        print(f"[WARN] Failed to presign s3://{bucket}/{key}: {exc}")
        _report_backend_error('presign_image', code='presign_failed', error=exc)
        return None


def _field_bucket_and_key(item_dict, key_field, url_field, default_bucket_env=None):
    key = item_dict.get(key_field)
    if key:
        bucket = os.getenv(default_bucket_env, '').strip() if default_bucket_env else ''
        return bucket or None, str(key).strip()
    return _parse_s3_url(item_dict.get(url_field))


def _normalize_discard_image_fields(item_dict):
    normalized = dict(item_dict or {})

    images_bucket, images_key = _field_bucket_and_key(
        normalized,
        's3_key',
        'images',
        'DISCARDS_BUCKET_NAME',
    )
    resized_bucket, resized_key = _field_bucket_and_key(
        normalized,
        'resized_image_key',
        'resized_image_url',
        'DISCARDS_BUCKET_NAME',
    )
    product_bucket, product_key = _field_bucket_and_key(
        normalized,
        'product_image_key',
        'product_image_url',
        'UPLOADS_BUCKET_NAME',
    )

    signed_product = _generate_read_url(product_bucket, product_key)
    signed_resized = _generate_read_url(resized_bucket, resized_key)
    signed_images = _generate_read_url(images_bucket, images_key)

    if signed_product:
        normalized['product_image_url'] = signed_product
    if signed_resized:
        normalized['resized_image_url'] = signed_resized
    elif signed_product:
        normalized['resized_image_url'] = signed_product
    if signed_images:
        normalized['images'] = signed_images
    elif signed_resized:
        normalized['images'] = signed_resized
    elif signed_product:
        normalized['images'] = signed_product

    return normalized


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
                return _get_discard_by_id(owner, item_id)
            return _get_discards(owner, event.get('queryStringParameters') or {})
        elif http_method in ['PUT', 'PATCH']:
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            body = json.loads(event.get('body', '{}'))
            return _update_discard(owner, item_id, body)
        elif http_method == 'DELETE':
            if not owner or not item_id:
                return _error_response(400, 'Missing owner or item_id parameter')
            return _delete_discard(owner, item_id)
        elif http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET,PUT,PATCH,DELETE,OPTIONS'
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


# Shared-table migration: mirror discard writes into shared_discards (owner_id-keyed,
# PK (owner_id,_id)). Discards fan out to household members, each written with owner_id =
# that member. Off by default; non-blocking. Per-call column intersection handles the
# per-owner schema drift. Read stays per-owner until READ_SHARED_DISCARDS flips.
DUAL_WRITE_DISCARDS = os.getenv('DUAL_WRITE_DISCARDS', 'false').lower() == 'true'
_SHARED_DISCARDS_TABLE = 'shared_discards'


def _shared_discards_cols(conn, member_table):
    with conn.cursor() as cur:
        cur.execute("SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s", [member_table])
        src = {r['cn'] for r in cur.fetchall()}
        cur.execute("SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s", [_SHARED_DISCARDS_TABLE])
        shd = {r['cn'] for r in cur.fetchall()}
    return [c for c in shd if c in src and c != 'owner_id']


def _dual_write_discard_to_shared(conn, member_id, item_id):
    """Upsert one discard row (by _id) from a member's table into shared_discards."""
    if not DUAL_WRITE_DISCARDS:
        return
    try:
        member_table = f"{member_id}_discards"
        cols = _shared_discards_cols(conn, member_table)
        if not cols:
            return
        collist = ', '.join(f'`{c}`' for c in cols)
        src = ', '.join(f's.`{c}`' for c in cols)
        upd = ', '.join(f'`{c}`=VALUES(`{c}`)' for c in cols if c != '_id')
        sql = (f"INSERT INTO `{_SHARED_DISCARDS_TABLE}` (`owner_id`, {collist}) "
               f"SELECT %s, {src} FROM `{member_table}` s WHERE s.`_id` = %s "
               f"ON DUPLICATE KEY UPDATE `owner_id`=VALUES(`owner_id`), {upd}")
        with conn.cursor() as cur:
            cur.execute(sql, (member_id, item_id))
        conn.commit()
    except Exception as exc:
        try:
            import sys
            print(json.dumps({'evt': 'dual_write_miss', 'family': 'discards',
                'owner_id': str(member_id), 'item_id': str(item_id), 'error': str(exc)[:500]}), file=sys.stderr)
        except Exception:
            pass


def _dual_delete_discard_from_shared(conn, member_id, item_id):
    if not DUAL_WRITE_DISCARDS:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM `{_SHARED_DISCARDS_TABLE}` WHERE `owner_id`=%s AND `_id`=%s LIMIT 1", (member_id, item_id))
        conn.commit()
    except Exception:
        pass


def _get_discards(owner, query=None):
    """Get all items from owner's discards table ({owner}_discards)"""
    try:
        table_name = f"{_sanitize_user_id(owner)}_discards"
        limit = _parse_limit(query)
        before = _parse_before(query)
        conn = _mysql_conn()
        # Shared-table read cutover (reversible via env flag, default off = per-owner).
        # MANDATORY owner_id filter — shared_discards holds every owner's rows.
        use_shared = os.getenv('READ_SHARED_DISCARDS', '').strip().lower() == 'true'
        with conn.cursor() as cur:
            if not use_shared:
                cur.execute("""
                    SELECT COUNT(*) as count FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table_name])
                if cur.fetchone()['count'] == 0:
                    return _success_response({'owner': owner, 'items': [], 'count': 0})
                cur.execute("""
                    SELECT COUNT(*) as count FROM information_schema.columns
                    WHERE table_schema = DATABASE() AND table_name = %s AND column_name = '_updatedDate'
                """, [table_name])
                has_updated_date = cur.fetchone()['count'] > 0
            else:
                has_updated_date = True
            sort_expression = "COALESCE(`_updatedDate`, `_createdDate`)" if has_updated_date else "`_createdDate`"
            params = []
            if use_shared:
                from_ref = "`shared_discards`"
                where_clause = "`owner_id` = %s AND `action` = 'IN'"
                params.append(_sanitize_user_id(owner))
            else:
                from_ref = f"`{table_name}`"
                where_clause = "`action` = 'IN'"
            before_clause = ""
            if before:
                before_clause = f"AND {sort_expression} < %s"
                params.append(before)
            cur.execute(f"""
                SELECT * FROM {from_ref}
                WHERE {where_clause}
                {before_clause}
                ORDER BY {sort_expression} DESC
                LIMIT {limit + 1}
            """, params)
            items = cur.fetchall() or []
            has_more = len(items) > limit
            items = items[:limit]
            items_list = []
            for item in items:
                item_dict = {}
                for key, value in item.items():
                    try:
                        item_dict[key] = json_serial(value)
                    except Exception as e:
                        print(f"[WARN] Failed to serialize {key}: {e}")
                        item_dict[key] = str(value) if value is not None else None
                items_list.append(_normalize_discard_image_fields(item_dict))
            return _success_response({
                'owner': owner,
                'items': items_list,
                'count': len(items_list),
                'limit': limit,
                'has_more': has_more,
            })
    except Exception as e:
        print(f"[ERROR] Failed to get discards: {str(e)}")
        return _error_response(500, f'Failed to retrieve discards: {str(e)}')


def _get_discard_by_id(owner, item_id):
    """GET /discards/{owner}/{item_id} - get a single discard by _id."""
    try:
        table_name = f"{_sanitize_user_id(owner)}_discards"
        conn = _mysql_conn()
        use_shared = os.getenv('READ_SHARED_DISCARDS', '').strip().lower() == 'true'
        with conn.cursor() as cur:
            if use_shared:
                cur.execute("SELECT * FROM `shared_discards` WHERE `owner_id` = %s AND `_id` = %s", [_sanitize_user_id(owner), item_id])
            else:
                cur.execute("""
                    SELECT COUNT(*) as count FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table_name])
                if cur.fetchone()['count'] == 0:
                    return _error_response(404, f'Discards table not found for owner: {owner}')
                cur.execute(f"SELECT * FROM `{table_name}` WHERE `_id` = %s", [item_id])
            row = cur.fetchone()
            if not row:
                return _error_response(404, f'Discard with id {item_id} not found')
            item_dict = _normalize_discard_image_fields({k: json_serial(v) for k, v in row.items()})
            return _success_response({'item': item_dict})
    except Exception as e:
        print(f"[ERROR] Failed to get discard: {str(e)}")
        return _error_response(500, f'Failed to retrieve discard: {str(e)}')


def _update_discard(owner, item_id, body):
    """Update a discard item (e.g., expiration date, action)"""
    try:
        table_name = f"{_sanitize_user_id(owner)}_discards"
        allowed_fields = {'product_expiration': 'DATE', 'action': 'ENUM'}
        updates = []
        values = []
        for field, field_type in allowed_fields.items():
            if field not in body:
                continue
            value = body[field]
            if field == 'product_expiration' and value:
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    return _error_response(400, 'Invalid date format for product_expiration. Use YYYY-MM-DD')
            if field == 'action' and value and value not in ['IN', 'OUT']:
                return _error_response(400, 'Invalid action value. Must be "IN" or "OUT"')
            updates.append(f"`{field}` = %s")
            values.append(value)
        if not updates:
            return _error_response(400, 'No valid fields to update. Allowed: product_expiration, action')
        updates.append("`_updatedDate` = NOW()")
        values.append(item_id)
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        # Same flag the READ path uses. Reads cut over to shared_discards a while ago but this
        # write did not, so it still required a per-owner table that is being retired - a user
        # could SEE a discard (served from shared) and get a 404 trying to edit it. Both halves
        # now follow the flag together.
        use_shared = os.getenv('READ_SHARED_DISCARDS', '').strip().lower() == 'true'
        with conn.cursor() as cur:
            if use_shared:
                # MANDATORY owner scoping — shared_discards holds every owner's rows, so an
                # unscoped WHERE `_id` would edit a stranger's discard. Scoped to the household
                # rather than the acting user alone, because the per-owner path below fans the
                # same edit out to every member's table and that behaviour must be preserved.
                owner_ids = [_sanitize_user_id(m) for m in member_ids] or [_sanitize_user_id(owner)]
                placeholders = ','.join(['%s'] * len(owner_ids))
                cur.execute(
                    f"SELECT `_id` FROM `shared_discards` WHERE `_id` = %s AND `owner_id` IN ({placeholders})",
                    [item_id] + owner_ids)
                if not cur.fetchone():
                    return _error_response(404, f'Item with id {item_id} not found')
                cur.execute(
                    f"UPDATE `shared_discards` SET {', '.join(updates)} "
                    f"WHERE `_id` = %s AND `owner_id` IN ({placeholders})",
                    values + owner_ids)
                conn.commit()
                # No _dual_write_discard_to_shared here: shared IS the table we just wrote.
                cur.execute(
                    "SELECT * FROM `shared_discards` WHERE `_id` = %s AND `owner_id` = %s",
                    [item_id, _sanitize_user_id(owner)])
            else:
                cur.execute("""
                    SELECT COUNT(*) as count FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table_name])
                if cur.fetchone()['count'] == 0:
                    return _error_response(404, f'Discards table not found for owner: {owner}')
                cur.execute(f"SELECT `_id` FROM `{table_name}` WHERE `_id` = %s", [item_id])
                if not cur.fetchone():
                    return _error_response(404, f'Item with id {item_id} not found')
                cur.execute(f"""
                    SELECT COUNT(*) as count FROM information_schema.columns
                    WHERE table_schema = DATABASE() AND table_name = %s AND column_name = '_updatedDate'
                """, [table_name])
                if cur.fetchone()['count'] == 0:
                    # Tolerate errno 1060: two concurrent requests can both see the column
                    # missing and both ALTER — a duplicate just means the other won the race.
                    try:
                        cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `_updatedDate` DATETIME NULL")
                        conn.commit()
                    except Exception as e:
                        if getattr(e, 'args', (None,))[0] == 1060:
                            conn.rollback()
                        else:
                            raise
                query = f"UPDATE `{{table_name}}` SET {', '.join(updates)} WHERE `_id` = %s"
                for member_id in member_ids:
                    member_table_name = f"{member_id}_discards"
                    cur.execute(query.format(table_name=member_table_name), values)
                conn.commit()
                for member_id in member_ids:
                    _dual_write_discard_to_shared(conn, member_id, item_id)
                cur.execute(f"SELECT * FROM `{table_name}` WHERE `_id` = %s", [item_id])
            updated_item = cur.fetchone()
            if not updated_item:
                return _error_response(404, 'Item not found after update')
            item_dict = _normalize_discard_image_fields({k: json_serial(v) for k, v in updated_item.items()})
            changed_fields = sorted(field for field in allowed_fields.keys() if field in body)
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f"discard-update:{item_id}:{item_dict.get('_updatedDate') or ''}",
                    'device_id': item_dict.get('_device'),
                    'event_type': 'discard_update',
                    'entity_type': 'discard_item',
                    'action': item_dict.get('action'),
                    'title': item_dict.get('product_name'),
                    'brand': item_dict.get('brand'),
                    'item_id': item_id,
                    'job_id': item_dict.get('job_id'),
                    'user_id': item_dict.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/discards/{owner}/{item_id}',
                    'source_system': 'discards_api',
                    'primary_image_url': item_dict.get('product_image_url') or item_dict.get('images'),
                    'secondary_image_url': item_dict.get('images') or item_dict.get('resized_image_url'),
                    'metadata': {
                        'changed_fields': changed_fields,
                        'product_expiration': item_dict.get('product_expiration'),
                    },
                },
            )
            return _success_response({'message': 'Item updated successfully', 'item': item_dict})
    except Exception as e:
        print(f"[ERROR] Failed to update discard: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to update item: {str(e)}')


def _delete_discard(owner, item_id):
    """Delete a discard item"""
    try:
        table_name = f"{_sanitize_user_id(owner)}_discards"
        conn = _mysql_conn()
        member_ids = _get_household_member_ids(conn, owner)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if cur.fetchone()['count'] == 0:
                return _error_response(404, f'Discards table not found for owner: {owner}')
            cur.execute(f"SELECT * FROM `{table_name}` WHERE `_id` = %s LIMIT 1", [item_id])
            existing_item = cur.fetchone()
            if not existing_item:
                return _error_response(404, f'Item with id {item_id} not found')
            item_dict = _normalize_discard_image_fields({k: json_serial(v) for k, v in existing_item.items()})
            for member_id in member_ids:
                member_table_name = f"{member_id}_discards"
                cur.execute(f"DELETE FROM `{member_table_name}` WHERE `_id` = %s", [item_id])
            conn.commit()
            for member_id in member_ids:
                _dual_delete_discard_from_shared(conn, member_id, item_id)
            _record_master_feed_event(
                conn,
                owner,
                member_ids,
                {
                    'event_key': f'discard-remove:{item_id}',
                    'device_id': item_dict.get('_device'),
                    'event_type': 'discard_remove',
                    'entity_type': 'discard_item',
                    'action': item_dict.get('action'),
                    'title': item_dict.get('product_name'),
                    'brand': item_dict.get('brand'),
                    'item_id': item_id,
                    'job_id': item_dict.get('job_id'),
                    'user_id': item_dict.get('user_id'),
                    'source_table': table_name,
                    'source_path': '/discards/{owner}/{item_id}',
                    'source_system': 'discards_api',
                    'primary_image_url': item_dict.get('product_image_url') or item_dict.get('images'),
                    'secondary_image_url': item_dict.get('images') or item_dict.get('resized_image_url'),
                    'metadata': {
                        'removed': True,
                    },
                },
            )
            return _success_response({'message': 'Item deleted successfully', 'item_id': item_id})
    except Exception as e:
        print(f"[ERROR] Failed to delete discard: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, f'Failed to delete item: {str(e)}')


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
