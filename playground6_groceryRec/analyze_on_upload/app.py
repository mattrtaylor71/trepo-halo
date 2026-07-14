# analyze_on_upload/app.py
import os
import re
import json
import time
import base64
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
import pymysql

# ---------- Optional OpenAI ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
except Exception:
    openai_client = None

# ---------- AWS clients (reuse across invocations) ----------
dynamodb = boto3.resource("dynamodb")
s3       = boto3.client("s3")
iot      = boto3.client("iot-data", endpoint_url=f"https://{os.environ['IOT_ENDPOINT']}")

# ---------- Environment ----------
JOBS_TABLE     = os.environ["JOBS_TABLE"]
RESULTS_TABLE  = os.environ.get("RESULTS_TABLE")  # optional
BUCKET_NAME    = os.environ["BUCKET_NAME"]
KEY_PREFIX     = os.environ.get("KEY_PREFIX", "images/")
TOPIC_TEMPLATE = os.environ.get("RESULT_TOPIC_TEMPLATE", "trepo/{user_id}/{device_id}/jobs/{job_id}/result")

# MySQL
DB_HOST  = os.environ["DB_HOST"]
DB_PORT  = int(os.environ.get("DB_PORT", "3306"))
DB_USER  = os.environ["DB_USER"]
DB_PASS  = os.environ["DB_PASS"]
DB_NAME  = os.environ["DB_NAME"]
DB_TABLE = os.environ.get("DB_TABLE", "grocery_events")  # Legacy default, will be overridden by owner

# Behavior toggles
FORCE_LABEL     = os.environ.get("FORCE_LABEL", "").strip()  # e.g. "banana"
ALLOW_NEGATIVE  = int(os.environ.get("ALLOW_NEGATIVE", "1")) # 1=allow, 0=block OUT when qty<=0

# ---------- Utilities ----------
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _log(msg: str, data=None):
    if data is None:
        print(msg)
    else:
        try:
            print(f"{msg} {json.dumps(data)[:1500]}")
        except Exception:
            print(msg)

def _ai_op(service, op, model, latency_ms, status,
           input_summary=None, output_summary=None, error=None,
           owner_id=None, job_id=None):
    """Emit one single-line ai_op telemetry marker. Additive only — never raises."""
    try:
        rec = {
            "evt": "ai_op",
            "service": service,
            "op": op,
            "model": model,
            "latency_ms": int(latency_ms),
            "status": status,
        }
        if owner_id:
            rec["owner_id"] = str(owner_id)
        if input_summary is not None:
            rec["input"] = str(input_summary)[:400]
        if output_summary is not None:
            rec["output"] = str(output_summary)[:1200]
        if error is not None:
            rec["error"] = str(error)[:500]
        if job_id:
            rec["job_id"] = str(job_id)
        print(json.dumps(rec, default=str))
    except Exception:
        pass

def _extract_s3_from_event(event):
    """
    Supports:
      1) Native S3 notifications (Records[].s3.bucket.name / .object.key)
      2) EventBridge S3 events (detail.bucket.name / detail.object.key)
    Returns: (bucket, key) or (None, None)
    """
    # 1) Native S3
    if isinstance(event, dict) and "Records" in event:
        try:
            rec = event["Records"][0]
            if rec.get("eventSource", "").startswith("aws:s3"):
                b = rec["s3"]["bucket"]["name"]
                k = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
                return b, k
        except Exception as e:
            _log("[extract] Records parse failed:", {"error": str(e)})

    # 2) EventBridge S3
    try:
        if event.get("detail-type") == "Object Created" and "detail" in event:
            d = event["detail"]
            b = d["bucket"]["name"]
            k = urllib.parse.unquote_plus(d["object"]["key"])
            return b, k
    except Exception as e:
        _log("[extract] EventBridge parse failed:", {"error": str(e)})

    _log("[extract] Unrecognized event shape:", event)
    return None, None

# images/<user>/<device>/YYYY/MM/DD/<job_id>.jpg or .png
def _extract_ids_from_key(key: str):
    pat = rf"^{re.escape(KEY_PREFIX)}([^/]+)/([^/]+)/\d{{4}}/\d{{2}}/\d{{2}}/([^.]+)\.(jpg|png)$"
    m = re.match(pat, key)
    if not m:
        raise ValueError(f"Key does not match expected pattern: {key}")
    return {"user_id": m.group(1), "device_id": m.group(2), "job_id": m.group(3)}

def _to_dynamo(value):
    """Recursively convert floats → Decimal for DynamoDB."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamo(v) for v in value]
    return value

def _update_job(job_id, **attrs):
    table = dynamodb.Table(JOBS_TABLE)
    attrs = _to_dynamo(attrs)
    names = {f"#{k}": k for k in attrs.keys()}
    vals  = {f":{k}": v for k, v in attrs.items()}
    expr  = "SET " + ", ".join(f"#{k} = :{k}" for k in attrs.keys())
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=expr,
        ExpressionAttributeValues=vals,
        ExpressionAttributeNames=names
    )

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

    _conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        autocommit=True,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.environ.get('DB_CONNECT_TIMEOUT_SECONDS', '5')),
        read_timeout=int(os.environ.get('DB_READ_TIMEOUT_SECONDS', '30')),
        write_timeout=int(os.environ.get('DB_WRITE_TIMEOUT_SECONDS', '30')),
    )
    return _conn

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


def _metrics_table_name(owner: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')
    if not safe:
        raise ValueError("Invalid owner")
    return f"{safe}-metrics"

def _prod_kitchen_table_name(owner: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')
    if not safe:
        raise ValueError("Invalid owner")
    return f"{safe}_prod_kitchen"

def _new_kitchen_table_name(owner: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')
    if not safe:
        raise ValueError("Invalid owner")
    return f"{safe}_new_kitchen"

def _discards_table_name(owner: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')
    if not safe:
        raise ValueError("Invalid owner")
    return f"{safe}_discards"

def _table_exists(cur, table_name: str) -> bool:
    cur.execute("""
        SELECT COUNT(*) AS count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    row = cur.fetchone() or {"count": 0}
    return int(row.get("count") or 0) > 0

def _get_table_columns(cur, table_name: str):
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    rows = cur.fetchall() or []
    return {
        (row.get("column_name") or row.get("COLUMN_NAME"))
        for row in rows
        if isinstance(row, dict) and (row.get("column_name") or row.get("COLUMN_NAME"))
    }

def _ensure_metrics_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
              `_id` VARCHAR(36) PRIMARY KEY COMMENT 'UUID for this record',
              `_owner` VARCHAR(36) NOT NULL COMMENT 'Owner UUID',
              `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the metrics snapshot was created',
              `IQ` INT NOT NULL DEFAULT 75 COMMENT 'IQ score out of 100',
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

def _ensure_metrics_columns(conn, table_name: str):
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

def _ensure_grocery_columns(conn, table_name: str):
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
            col_type = row.get('column_type') or row.get('COLUMN_TYPE')
            if name:
                column_info[name] = col_type

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

def _recent_where_clause(columns) -> str:
    clauses = ["`_createdDate` >= DATE_SUB(NOW(), INTERVAL 14 DAY)"]
    if 'action' in columns:
        clauses.append("`action` = 'IN'")
    return " AND ".join(clauses)

def _get_upf_counts(cur, table_name: str):
    if not _table_exists(cur, table_name):
        return 0, 0
    columns = _get_table_columns(cur, table_name)
    where_clause = _recent_where_clause(columns)
    if 'upf' in columns:
        cur.execute(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN LOWER(`upf`) = 'yes' THEN 1 ELSE 0 END) AS upf_count
            FROM `{table_name}`
            WHERE {where_clause}
            """
        )
        row = cur.fetchone() or {}
        return int(row.get('total') or 0), int(row.get('upf_count') or 0)
    cur.execute(
        f"""
        SELECT COUNT(*) AS total
        FROM `{table_name}`
        WHERE {where_clause}
        """
    )
    row = cur.fetchone() or {}
    return int(row.get('total') or 0), 0

def _get_harmful_count(cur, table_name: str) -> int:
    if not _table_exists(cur, table_name):
        return 0
    columns = _get_table_columns(cur, table_name)
    if 'harmful_ingredients' not in columns:
        return 0
    where_clause = _recent_where_clause(columns)
    cur.execute(
        f"""
        SELECT SUM(COALESCE(JSON_LENGTH(`harmful_ingredients`), 0)) AS harmful_count
        FROM `{table_name}`
        WHERE {where_clause}
        """
    )
    row = cur.fetchone() or {}
    return int(row.get('harmful_count') or 0)

def _calculate_metrics(conn, owner: str):
    prod_kitchen_table = _prod_kitchen_table_name(owner)
    new_kitchen_table = _new_kitchen_table_name(owner)
    discards_table = _discards_table_name(owner)
    _ensure_grocery_columns(conn, prod_kitchen_table)
    _ensure_grocery_columns(conn, new_kitchen_table)
    _ensure_grocery_columns(conn, discards_table)

    with conn.cursor() as cur:
        prod_total, prod_upf = _get_upf_counts(cur, prod_kitchen_table)
        new_total, new_upf = _get_upf_counts(cur, new_kitchen_table)
        discard_total, discard_upf = _get_upf_counts(cur, discards_table)
        harmful_total = (
            _get_harmful_count(cur, prod_kitchen_table)
            + _get_harmful_count(cur, new_kitchen_table)
            + _get_harmful_count(cur, discards_table)
        )

    total = prod_total + new_total + discard_total
    upf_percent = round((prod_upf + new_upf + discard_upf) / total * 100, 2) if total > 0 else 0
    harmful_total = max(0, min(1000, harmful_total))
    return {"UPF": upf_percent, "harmful_ingredients": harmful_total}

def _safe_int(value, default=0) -> int:
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

def _append_metrics_snapshot(owner: str):
    table_name = _metrics_table_name(owner)
    conn = _mysql_conn()
    _ensure_metrics_table(conn, table_name)
    _ensure_metrics_columns(conn, table_name)
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM `{table_name}` ORDER BY `_createdDate` DESC LIMIT 1")
        latest = cur.fetchone() or {}

    computed = _calculate_metrics(conn, owner)
    entry_id = str(uuid4())
    points = _safe_int(latest.get("Points"), 0)
    iq_what = latest.get("IQ_what")
    iq_suggestions = _coerce_json_list(latest.get("IQ_suggestions"))
    upf_what = latest.get("UPF_what")
    upf_suggestions = _coerce_json_list(latest.get("UPF_suggestions"))
    harmful_what = latest.get("harmful_ingredients_what")
    harmful_suggestions = _coerce_json_list(latest.get("harmful_ingredients_suggestions"))
    kitchen_analysis_status = latest.get("kitchen_analysis_status")
    kitchen_analysis_content = latest.get("kitchen_analysis_content")
    kitchen_analysis_generated_at = latest.get("kitchen_analysis_generated_at")
    kitchen_analysis_error = latest.get("kitchen_analysis_error")

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
                75,
                points,
                computed["UPF"],
                computed["harmful_ingredients"],
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
            ],
        )
        conn.commit()
    _dual_write_metrics_to_shared(conn, owner, table_name, entry_id)

# Table creation removed - tables already exist

def _current_qty(owner: str, product_name: str, table_name: str) -> int:
    """
    Compute net qty for an owner/product (IN=+1, OUT=-1).
    Used only if ALLOW_NEGATIVE=0 to guard OUT when qty<=0.
    """
    sql = f"""
      SELECT COALESCE(SUM(CASE WHEN action='IN' THEN 1 WHEN action='OUT' THEN -1 ELSE 0 END),0) AS qty
      FROM `{table_name}`
      WHERE _owner=%s AND product_name=%s
    """
    conn = _mysql_conn()
    with conn.cursor() as cur:
        cur.execute(sql, (owner, product_name))
        row = cur.fetchone() or {"qty": 0}
        return int(row["qty"])

def _insert_event(owner: str, device_id: str, action: str, product_name: str, product_brand: str, image_key: str, table_name: str, product_expiration: str = None):
    """
    Insert event into owner-specific table with new schema.
    _id: UUID
    _owner: from frontend
    _device: from frontend
    product_name: from OpenAI classification
    product_brand: from OpenAI classification
    product_expiration: leave blank for now
    images: S3 link to image
    product_barcode: leave blank for now (will come from ChatGPT later)
    action: IN or OUT
    _createdDate: current timestamp
    """
    # Generate UUID for _id
    entry_id = str(uuid4())
    
    # Construct S3 URL from image key
    s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{image_key}"
    
    # Column order must match the actual table structure:
    # _id, _owner, _device, product_name, product_brand, product_expiration, images, product_barcode, action, _createdDate
    sql = f"""
    INSERT INTO `{table_name}`
      (_id, _owner, _device, product_name, product_brand, product_expiration, images, product_barcode, action, _createdDate)
    VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s, NOW())
    """
    conn = _mysql_conn()
    with conn.cursor() as cur:
        cur.execute(
            sql,
            [
                entry_id,           # _id
                owner,              # _owner
                device_id,          # _device
                product_name,       # product_name
                product_brand or None,  # product_brand (empty string becomes NULL)
                product_expiration or None,  # product_expiration (YYYY-MM-DD format or NULL)
                s3_url,            # images (S3 URL)
                action,            # action
            ],
        )

def _publish_result(user_id, device_id, job_id, product_name, confidence, action):
    topic = TOPIC_TEMPLATE.format(user_id=user_id, device_id=device_id, job_id=job_id)
    payload = {
        "job_id": job_id,
        "mode": "grocery",
        "action": action,
        "product_name": product_name,
        "confidence": float(confidence),
        "summary": f"{action} {product_name} ({confidence:.0f}%)",
        "created_at": _iso_now(),
    }
    iot.publish(topic=topic, qos=1, payload=json.dumps(payload).encode("utf-8"))

# ---------- Classification ----------
def _classify(bucket: str, key: str):
    """
    Returns (product_name: str, confidence: float, classification: dict).
    Honors FORCE_LABEL; uses OpenAI Vision when available; falls back to 'unknown'.
    """
    # 1) FORCE_LABEL for fast pipeline tests
    if FORCE_LABEL:
        return FORCE_LABEL.strip().lower(), 100.0, {
            "title": FORCE_LABEL.strip().lower(),
            "category": "",
            "source": "forced"
        }

    # 2) If no OpenAI key, return unknown
    if not openai_client:
        return "unknown", 0.0, {"error": "openai_disabled"}

    # 3) Download image and call OpenAI Vision chat.completions (JSON output)
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        image_bytes = obj["Body"].read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        sys_prompt = (
            "You are a product classification expert for groceries. "
            "Return ONLY a JSON object with keys: title, brand, flavor, size, packaging_type, "
            "category, subcategory, claims, other_text. "
            "IMPORTANT: Pay special attention to identifying the brand name if visible on the product packaging. "
            "If unclear, set fields to empty strings. "
            "If no grocery is visible, set title='unknown' and category='unknown'."
        )
        user_text = (
            "Identify the grocery item, paying special attention to the brand name if visible. "
            "Respond with JSON only. "
            "Keys: title, brand, flavor, size, packaging_type, category, subcategory, claims, other_text."
        )

        _ai_t0 = time.time()
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}" }},
                        ],
                    },
                ],
                max_tokens=800,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as _ai_err:
            _ai_op("capture", "classify_item", "gpt-4o",
                   (time.time() - _ai_t0) * 1000, "error",
                   input_summary=key.rsplit("/", 1)[-1], error=str(_ai_err))
            raise
        _ai_latency_ms = int((time.time() - _ai_t0) * 1000)

        content = resp.choices[0].message.content.strip()
        # Strip code fences if any
        if content.startswith("```"):
            content = content.split("```", 2)[-2] if "```" in content else content
        try:
            classification = json.loads(content)
        except json.JSONDecodeError:
            # Last resort fallback
            classification = {"title": "unknown", "category": "unknown", "other_text": content[:200]}

        product_name = (classification.get("title") or "").strip().lower()
        if not product_name:
            product_name = (classification.get("category") or "unknown").strip().lower()

        confidence = 0.0 if product_name == "unknown" else 100.0
        _log("[openai] classification:", classification)
        try:
            _ai_op("capture", "classify_item", "gpt-4o", _ai_latency_ms, "success",
                   input_summary=key.rsplit("/", 1)[-1],
                   output_summary=json.dumps(classification, default=str))
        except Exception:
            pass
        return product_name, confidence, classification

    except Exception as e:
        _log("[openai] error:", {"error": str(e)})
        return "unknown", 0.0, {"error": str(e)}

# ---------- Lambda handler ----------
def handler(event, context):
    _log("[handler] raw event:", event)

    bucket, key = _extract_s3_from_event(event)
    if not bucket or not key:
        _log("[handler] No bucket/key; ignoring.")
        return {"statusCode": 200, "body": json.dumps({"ok": True, "ignored": True})}

    _log("[handler] S3 object:", {"bucket": bucket, "key": key})

    if KEY_PREFIX and not key.startswith(KEY_PREFIX):
        _log("[handler] Key not under prefix; skipping.", {"prefix": KEY_PREFIX})
        return {"statusCode": 200, "body": json.dumps({"ok": True, "ignored": True})}

    job_id = None
    try:
        ids = _extract_ids_from_key(key)
        user_id   = ids["user_id"]
        device_id = ids["device_id"]
        job_id    = ids["job_id"]
        _log("[handler] Parsed IDs:", ids)

        # Mark job -> PROCESSING
        _update_job(job_id, status="PROCESSING", t_start_proc=_iso_now())

        # Load job (to get action and owner)
        jobs_tbl = dynamodb.Table(JOBS_TABLE)
        job = jobs_tbl.get_item(Key={"job_id": job_id}).get("Item", {**ids, "job_id": job_id})
        action = job.get("action", "IN")
        owner = job.get("owner")  # Get owner from job
        
        # Determine table name: <owner>_new_kitchen
        if not owner:
            _log("[error] owner parameter required in job")
            _update_job(job_id, status="FAILED", error_msg="owner parameter required")
            return {"statusCode": 400, "body": json.dumps({"ok": False, "error": "owner parameter required"})}
        
        table_name = f"{owner}_new_kitchen"
        
        _log("[dynamo] Loaded job:", {
            "has_action": bool(job.get("action")), 
            "action": action,
            "owner": owner,
            "table_name": table_name
        })

        # Classify
        product_name, conf, classification = _classify(bucket, key)
        product_brand = (classification.get("brand") or "").strip()  # Extract brand from classification
        _log("[classify] result:", {
            "product": product_name, 
            "brand": product_brand,
            "confidence": conf
        })

        # Guard OUT if not allowed to go negative (only if owner provided)
        if owner and action == "OUT" and not ALLOW_NEGATIVE and product_name != "unknown":
            qty = _current_qty(owner, product_name, table_name)
            if qty <= 0:
                # Mark failed and publish a message
                msg = f"OUT not allowed: no inventory for '{product_name}' (qty={qty})"
                _log("[inventory] block OUT:", {"owner": owner, "product": product_name, "qty": qty})
                _update_job(job_id, status="FAILED", error_msg=msg, last_product=product_name, confidence=float(conf))
                _publish_result(user_id, device_id, job_id, product_name, conf, action="OUT_BLOCKED")
                return {"statusCode": 200, "body": json.dumps({"ok": False, "blocked": True, "reason": msg})}

        # Get product_expiration from job if provided
        product_expiration = job.get("product_expiration")
        
        # Insert event into existing table
        _insert_event(owner, device_id, action, product_name, product_brand, key, table_name, product_expiration)
        _log("[mysql] insert OK")
        _append_metrics_snapshot(owner)
        _log("[metrics] snapshot append OK")

        # Update job -> DONE
        _update_job(
            job_id,
            status="DONE",
            t_done=_iso_now(),
            last_product=product_name,
            confidence=float(conf),
        )
        _log("[dynamo] job DONE")

        # Optional: write a result row
        if RESULTS_TABLE:
            dyn = dynamodb.Table(RESULTS_TABLE)
            item = _to_dynamo({
                "job_id": job_id,
                "result": {
                    "product_name": product_name,
                    "confidence": float(conf),
                    "classification": classification,
                    "image_s3_key": key,
                    "action": action,
                },
                "created_at": _iso_now(),
            })
            dyn.put_item(Item=item)
            _log("[dynamo] results row put")

        # MQTT publish back to device/UI
        _publish_result(user_id, device_id, job_id, product_name, conf, action)
        _log("[mqtt] publish OK")

        return {"statusCode": 200, "body": json.dumps({"ok": True, "job_id": job_id})}

    except Exception as e:
        _log("[error] exception:", {"error": str(e)})
        if job_id:
            try:
                _update_job(job_id, status="FAILED", error_msg=str(e))
            except Exception as e2:
                _log("[error] failed to mark FAILED:", {"error": str(e2)})
        raise
