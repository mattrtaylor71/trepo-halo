"""meal_calendar_api — user-curated weekly meal calendar.

A slot is (plan_date, meal_slot) and holds a denormalized snapshot of a recipe
pulled from one of three sources (saved / kitchen / explore). We store the
snapshot (title, image, ingredients, steps) so the calendar renders and expands
without re-fetching the source — robust to the source recipe changing or being
deleted, and to the three sources having different / unstable ids.

Dual-table infrastructure (matches the shared-table migration mid-flight state):
  - PRIMARY writes go to the per-owner table `{owner}_meal_calendar`.
  - Every mutation also mirrors into `shared_meal_calendar` (owner_id-keyed),
    gated by DUAL_WRITE_MEAL_CALENDAR (default true), non-blocking.
  - READS come from the per-owner table until READ_SHARED_MEAL_CALENDAR flips.
This keeps the feature consistent with every other type and ready for the
eventual read cutover with no data migration.

Routes (owner_id is the sole identity — matches the grocery API convention):
  GET    /meal-calendar/{owner}?start=YYYY-MM-DD&end=YYYY-MM-DD
  POST   /meal-calendar/{owner}          body: a slot entry
  PUT    /meal-calendar/{owner}/{item_id} body: partial update (move day/slot, edit)
  DELETE /meal-calendar/{owner}/{item_id}
"""
import os
import re
import json
import uuid
import datetime

import pymysql

_MEAL_SLOTS = {'breakfast', 'lunch', 'dinner', 'snack'}
_SOURCE_TYPES = {'saved', 'kitchen', 'explore', 'manual'}
_SHARED_TABLE = 'shared_meal_calendar'
DUAL_WRITE = os.getenv('DUAL_WRITE_MEAL_CALENDAR', 'true').strip().lower() == 'true'
READ_SHARED = os.getenv('READ_SHARED_MEAL_CALENDAR', 'false').strip().lower() == 'true'
_MAX_RANGE_DAYS = 60

# Columns carried on both the per-owner and shared tables (shared adds owner_id).
_COLS = [
    '_id', '_owner', 'plan_date', 'meal_slot', 'source_type', 'source_id',
    'title', 'image_url', 'ingredients', 'instructions', 'notes',
    'meal_category', 'need_grocery', 'missing_ingredients',
]
_JSON_FIELDS = {'ingredients', 'instructions', 'notes', 'missing_ingredients'}


def _sanitize_owner(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))


def _table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_meal_calendar"


def _mysql_conn():
    return pymysql.connect(
        host=os.environ['DB_HOST'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'],
        database=os.environ['DB_NAME'],
        port=int(os.getenv('DB_PORT', '3306')),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '10')),
        read_timeout=int(os.getenv('DB_READ_TIMEOUT_SECONDS', '30')),
        write_timeout=int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '30')),
        autocommit=True,
        charset='utf8mb4',
    )


def _ensure_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `_id` VARCHAR(36) NOT NULL,
                `_owner` VARCHAR(36) NULL,
                `plan_date` DATE NOT NULL,
                `meal_slot` VARCHAR(16) NOT NULL,
                `source_type` VARCHAR(16) NOT NULL DEFAULT 'saved',
                `source_id` VARCHAR(128) NULL,
                `title` VARCHAR(500) NULL,
                `image_url` TEXT NULL,
                `ingredients` JSON NULL,
                `instructions` JSON NULL,
                `notes` JSON NULL,
                `meal_category` VARCHAR(32) NULL,
                `need_grocery` TINYINT(1) NOT NULL DEFAULT 0,
                `missing_ingredients` JSON NULL,
                `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `_updatedDate` DATETIME NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`_id`),
                INDEX `idx_plan_date` (`plan_date`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)


def _ensure_shared_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_SHARED_TABLE}` (
                `_id` VARCHAR(36) NOT NULL,
                `owner_id` VARCHAR(36) NOT NULL,
                `_owner` VARCHAR(36) NULL,
                `plan_date` DATE NOT NULL,
                `meal_slot` VARCHAR(16) NOT NULL,
                `source_type` VARCHAR(16) NOT NULL DEFAULT 'saved',
                `source_id` VARCHAR(128) NULL,
                `title` VARCHAR(500) NULL,
                `image_url` TEXT NULL,
                `ingredients` JSON NULL,
                `instructions` JSON NULL,
                `notes` JSON NULL,
                `meal_category` VARCHAR(32) NULL,
                `need_grocery` TINYINT(1) NOT NULL DEFAULT 0,
                `missing_ingredients` JSON NULL,
                `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `_updatedDate` DATETIME NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`owner_id`, `_id`),
                INDEX `idx_owner_date` (`owner_id`, `plan_date`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)


# ---------------- dual-write helpers ----------------

def _dual_write_to_shared(conn, owner, item_id):
    """Mirror one row (by _id) from the per-owner table into shared_meal_calendar.
    Non-blocking: errors are swallowed + logged under a metric-filterable marker."""
    if not DUAL_WRITE:
        return
    try:
        table = _table_name(owner)
        _ensure_shared_table(conn)
        cols = ', '.join(f'`{c}`' for c in _COLS)
        src = ', '.join(f's.`{c}`' for c in _COLS)
        upd = ', '.join(f'`{c}`=VALUES(`{c}`)' for c in _COLS if c != '_id')
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO `{_SHARED_TABLE}` (`owner_id`, {cols}) "
                f"SELECT %s, {src} FROM `{table}` s WHERE s.`_id` = %s "
                f"ON DUPLICATE KEY UPDATE `owner_id`=VALUES(`owner_id`), {upd}",
                (owner, item_id),
            )
    except Exception as exc:
        _log_miss('dual_write_miss', owner, item_id, exc)


def _dual_delete_from_shared(conn, owner, item_id):
    if not DUAL_WRITE:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM `{_SHARED_TABLE}` WHERE `owner_id` = %s AND `_id` = %s",
                (owner, item_id),
            )
    except Exception as exc:
        _log_miss('dual_delete_miss', owner, item_id, exc)


def _log_miss(evt, owner, item_id, exc):
    try:
        print(json.dumps({
            'evt': evt, 'family': 'meal_calendar',
            'owner_id': str(owner) if owner else None,
            'item_id': str(item_id) if item_id else None,
            'error': str(exc)[:400],
        }))
    except Exception:
        pass


# ---------------- serialization ----------------

def _json_default(v):
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return str(v)


def _serialize_row(row):
    out = {}
    for k, v in row.items():
        if k in _JSON_FIELDS:
            out[k] = _decode_json(v)
        elif isinstance(v, (datetime.date, datetime.datetime)):
            out[k] = v.isoformat()
        elif k == 'need_grocery':
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _decode_json(v):
    if v is None:
        return []
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return []


def _encode_json(v):
    if v is None:
        return None
    if isinstance(v, str):
        # allow a pre-encoded string, but normalize simple lists
        return v
    return json.dumps(v)


# ---------------- responses ----------------

_CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
    'Content-Type': 'application/json',
}


def _resp(status, body):
    return {'statusCode': status, 'headers': _CORS,
            'body': json.dumps(body, default=_json_default)}


def _err(status, message):
    return _resp(status, {'error': message})


# ---------------- validation ----------------

def _parse_date(s):
    try:
        return datetime.date.fromisoformat(str(s).strip()[:10])
    except Exception:
        return None


def _validate_entry(body, partial=False):
    """Return (cleaned_dict, error_message)."""
    cleaned = {}
    if not partial or 'plan_date' in body:
        d = _parse_date(body.get('plan_date'))
        if d is None:
            return None, 'plan_date must be YYYY-MM-DD'
        cleaned['plan_date'] = d.isoformat()
    if not partial or 'meal_slot' in body:
        slot = str(body.get('meal_slot') or '').strip().lower()
        if slot not in _MEAL_SLOTS:
            return None, f"meal_slot must be one of {sorted(_MEAL_SLOTS)}"
        cleaned['meal_slot'] = slot
    if not partial or 'source_type' in body:
        st = str(body.get('source_type') or 'saved').strip().lower()
        if st not in _SOURCE_TYPES:
            return None, f"source_type must be one of {sorted(_SOURCE_TYPES)}"
        cleaned['source_type'] = st
    # optional passthrough fields
    for f in ('source_id', 'title', 'image_url', 'meal_category'):
        if f in body:
            cleaned[f] = (str(body[f])[:2000] if body[f] is not None else None)
    for f in _JSON_FIELDS:
        if f in body:
            cleaned[f] = _encode_json(body[f])
    if 'need_grocery' in body:
        cleaned['need_grocery'] = 1 if body.get('need_grocery') else 0
    if not partial and not cleaned.get('title'):
        return None, 'title is required'
    return cleaned, None


# ---------------- handlers ----------------

def _get_calendar(owner, query):
    conn = _mysql_conn()
    try:
        query = query or {}
        start = _parse_date(query.get('start')) or datetime.date.today()
        end = _parse_date(query.get('end')) or (start + datetime.timedelta(days=20))
        if end < start:
            start, end = end, start
        if (end - start).days > _MAX_RANGE_DAYS:
            end = start + datetime.timedelta(days=_MAX_RANGE_DAYS)

        if READ_SHARED:
            table = _SHARED_TABLE
            where = "`owner_id` = %s AND `plan_date` BETWEEN %s AND %s"
            params = [owner, start.isoformat(), end.isoformat()]
            _ensure_shared_table(conn)
        else:
            table = _table_name(owner)
            where = "`plan_date` BETWEEN %s AND %s"
            params = [start.isoformat(), end.isoformat()]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) c FROM information_schema.tables "
                    "WHERE table_schema=DATABASE() AND table_name=%s", [table])
                if cur.fetchone()['c'] == 0:
                    return _resp(200, {'owner': owner, 'start': start.isoformat(),
                                       'end': end.isoformat(), 'entries': []})
        slot_order = "FIELD(`meal_slot`,'breakfast','lunch','dinner','snack')"
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM `{table}` WHERE {where} "
                f"ORDER BY `plan_date` ASC, {slot_order}, `_createdDate` ASC",
                params)
            entries = [_serialize_row(r) for r in cur.fetchall()]
        return _resp(200, {'owner': owner, 'start': start.isoformat(),
                           'end': end.isoformat(), 'entries': entries})
    finally:
        conn.close()


def _create_entry(owner, body):
    cleaned, err = _validate_entry(body, partial=False)
    if err:
        return _err(400, err)
    conn = _mysql_conn()
    try:
        table = _table_name(owner)
        _ensure_table(conn, table)
        item_id = str(uuid.uuid4())
        fields = ['_id', '_owner'] + list(cleaned.keys())
        placeholders = ', '.join(['%s'] * len(fields))
        values = [item_id, owner] + [cleaned[k] for k in cleaned]
        col_sql = ', '.join(f'`{c}`' for c in fields)
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO `{table}` ({col_sql}) VALUES ({placeholders})", values)
        _dual_write_to_shared(conn, owner, item_id)
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` WHERE `_id` = %s", [item_id])
            row = cur.fetchone()
        return _resp(201, {'message': 'Added to meal calendar', 'entry': _serialize_row(row)})
    finally:
        conn.close()


def _update_entry(owner, item_id, body):
    cleaned, err = _validate_entry(body or {}, partial=True)
    if err:
        return _err(400, err)
    if not cleaned:
        return _err(400, 'No updatable fields provided')
    conn = _mysql_conn()
    try:
        table = _table_name(owner)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) c FROM information_schema.tables "
                "WHERE table_schema=DATABASE() AND table_name=%s", [table])
            if cur.fetchone()['c'] == 0:
                return _err(404, 'Meal calendar not found for owner')
            cur.execute(f"SELECT `_id` FROM `{table}` WHERE `_id` = %s", [item_id])
            if not cur.fetchone():
                return _err(404, f'Entry {item_id} not found')
            set_sql = ', '.join(f'`{k}` = %s' for k in cleaned)
            cur.execute(f"UPDATE `{table}` SET {set_sql} WHERE `_id` = %s",
                        list(cleaned.values()) + [item_id])
        _dual_write_to_shared(conn, owner, item_id)
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` WHERE `_id` = %s", [item_id])
            row = cur.fetchone()
        return _resp(200, {'message': 'Entry updated', 'entry': _serialize_row(row)})
    finally:
        conn.close()


def _delete_entry(owner, item_id):
    conn = _mysql_conn()
    try:
        table = _table_name(owner)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) c FROM information_schema.tables "
                "WHERE table_schema=DATABASE() AND table_name=%s", [table])
            if cur.fetchone()['c'] == 0:
                # nothing to delete anywhere; keep DELETE idempotent
                _dual_delete_from_shared(conn, owner, item_id)
                return _resp(200, {'message': 'Entry deleted', 'item_id': item_id})
            cur.execute(f"DELETE FROM `{table}` WHERE `_id` = %s", [item_id])
        _dual_delete_from_shared(conn, owner, item_id)
        return _resp(200, {'message': 'Entry deleted', 'item_id': item_id})
    finally:
        conn.close()


def handler(event, context):
    try:
        method = (event.get('requestContext', {}).get('http', {}).get('method')
                  or event.get('httpMethod') or '').upper()
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        item_id = path_params.get('item_id')

        if method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _CORS, 'body': ''}
        if not owner:
            return _err(400, 'Missing owner parameter')

        if method == 'GET':
            return _get_calendar(owner, event.get('queryStringParameters') or {})
        if method == 'POST':
            body = json.loads(event.get('body') or '{}')
            return _create_entry(owner, body)
        if method in ('PUT', 'PATCH'):
            if not item_id:
                return _err(400, 'Missing item_id parameter')
            body = json.loads(event.get('body') or '{}')
            return _update_entry(owner, item_id, body)
        if method == 'DELETE':
            if not item_id:
                return _err(400, 'Missing item_id parameter')
            return _delete_entry(owner, item_id)
        return _err(405, f'Method {method} not allowed')
    except json.JSONDecodeError as e:
        return _err(400, f'Invalid JSON body: {e}')
    except ValueError as e:
        return _err(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({'evt': 'backend_error', 'service': 'meal_calendar',
                          'code': 'unhandled_500', 'error': str(e)[:400]}))
        return _err(500, f'Internal server error: {e}')
