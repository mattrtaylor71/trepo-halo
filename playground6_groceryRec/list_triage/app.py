import json
# Force republish after dependency-layer restore.
import math
import os
from datetime import date, datetime, timezone
from decimal import Decimal

import boto3
import pymysql
from botocore.exceptions import ClientError

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
JOBS_TABLE = os.environ.get("JOBS_TABLE")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
DISCARDS_BUCKET_NAME = os.environ.get("DISCARDS_BUCKET_NAME")
ACCEPTED_STALE_TIMEOUT_SECONDS = int(os.environ.get("ACCEPTED_STALE_TIMEOUT_SECONDS", "600"))
TRIAGE_RECONCILE_BATCH_LIMIT = max(1, int(os.environ.get("TRIAGE_RECONCILE_BATCH_LIMIT", "100")))
TRIAGE_RECONCILE_TABLE_LIMIT = max(1, int(os.environ.get("TRIAGE_RECONCILE_TABLE_LIMIT", "250")))
DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT_SECONDS", "5"))
DB_READ_TIMEOUT = int(os.environ.get("DB_READ_TIMEOUT_SECONDS", "10"))
DB_WRITE_TIMEOUT = int(os.environ.get("DB_WRITE_TIMEOUT_SECONDS", "10"))

dynamodb = boto3.resource("dynamodb") if JOBS_TABLE else None
s3 = boto3.client("s3") if (BUCKET_NAME or DISCARDS_BUCKET_NAME) else None


def _response(status_code, payload=None):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Cache-Control": "no-store",
    }
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": "" if payload is None else json.dumps(payload),
    }


def _json_serial(obj):
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (dict, list, str, int, bool)):
        return obj
    return str(obj)


def _safe_serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, (Decimal, datetime, date)):
        return _json_serial(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)


def _sanitize_owner(owner):
    return "".join(ch for ch in str(owner or "") if ch.isalnum() or ch in {"_", "-"})


def _triage_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        return None
    return f"{safe}_triage"


def _parse_json_body(event):
    raw_body = event.get("body")
    if not raw_body:
        return {}
    return json.loads(raw_body)


def _clean_optional_string(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_string(body, field_name):
    value = _clean_optional_string((body or {}).get(field_name))
    if not value:
        raise ValueError(f"Missing {field_name}")
    return value


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(value)


def _normalize_triage_status(value):
    normalized = str(value or "").strip().lower()
    if normalized in {"accepted", "processing", "completed", "failed"}:
        return normalized
    raise ValueError("status must be accepted, processing, completed, or failed")


def _normalize_triage_filter_status(value):
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized == "pending":
        return "accepted"
    if normalized in {"accepted", "processing", "completed", "failed"}:
        return normalized
    raise ValueError("status must be accepted, pending, processing, completed, or failed")


def _parse_status_filters(value):
    if value is None:
        return []

    parts = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    statuses = []
    seen = set()
    for part in parts:
        normalized = _normalize_triage_filter_status(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        statuses.append(normalized)
    return statuses


def _default_raw_status_for_status(status):
    if status == "accepted":
        return "ADMIN_ACCEPTED"
    if status == "processing":
        return "ADMIN_PROCESSING"
    if status == "completed":
        return "ADMIN_COMPLETED"
    return "ADMIN_FAILED"


def _default_admin_message(status, raw_status):
    if status == "failed" and raw_status == "MISSED_INGEST":
        return "Upload was accepted but never picked up by the processor. Reconciled by admin."
    if status == "failed":
        return "Triage item failed after admin reconciliation."
    if status == "completed":
        return "Triage item marked completed by admin reconciliation."
    if status == "processing":
        return "Triage item marked processing by admin reconciliation."
    return "Triage item marked accepted by admin reconciliation."


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


def _parse_before(value):
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_row_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _table_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
            """,
            [table_name],
        )
        row = cur.fetchone() or {}
    return int(row.get("count") or 0) > 0


def _get_triage_row(cur, table_name, job_id):
    cur.execute(
        f"""
        SELECT *
        FROM `{table_name}`
        WHERE job_id = %s
        LIMIT 1
        """,
        [job_id],
    )
    return cur.fetchone()


def _map_row(row):
    serialized = _safe_serialize(row)
    return {
        "id": serialized.get("_id"),
        "owner_id": serialized.get("_owner"),
        "device_id": serialized.get("_device"),
        "user_id": serialized.get("user_id"),
        "job_id": serialized.get("job_id"),
        "source": serialized.get("source"),
        "job_type": serialized.get("job_type"),
        "status": serialized.get("status_normalized"),
        "raw_status": serialized.get("raw_status"),
        "action": serialized.get("action"),
        "title": serialized.get("title"),
        "detail": serialized.get("detail"),
        "error_message": serialized.get("error_message"),
        "response_text": serialized.get("response_text"),
        "session_id": serialized.get("session_id"),
        "s3_key": serialized.get("s3_key"),
        "table_source": serialized.get("table_source"),
        "created_at": serialized.get("created_at"),
        "updated_at": serialized.get("updated_at"),
        "accepted_at": serialized.get("accepted_at"),
        "processing_started_at": serialized.get("processing_started_at"),
        "completed_at": serialized.get("completed_at"),
        "failed_at": serialized.get("failed_at"),
        "time": serialized.get("updated_at") or serialized.get("created_at"),
        "metadata": serialized.get("metadata"),
    }


def _candidate_buckets(job_type):
    buckets = []
    normalized_job_type = str(job_type or "").strip().lower()
    if normalized_job_type == "discard" and DISCARDS_BUCKET_NAME:
        buckets.append(DISCARDS_BUCKET_NAME)
    if BUCKET_NAME:
        buckets.append(BUCKET_NAME)
    if DISCARDS_BUCKET_NAME and DISCARDS_BUCKET_NAME not in buckets:
        buckets.append(DISCARDS_BUCKET_NAME)
    return buckets


def _s3_object_exists(job_type, s3_key):
    key = str(s3_key or "").strip()
    if not key or s3 is None:
        return False
    for bucket in _candidate_buckets(job_type):
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            error_code = str((exc.response or {}).get("Error", {}).get("Code") or "")
            # HeadObject may return 403 for a missing object in a private bucket
            # when the caller lacks ListBucket. For stale-triage reconciliation,
            # treat that the same as "object not found" so GET /triage stays healthy.
            if error_code not in {"403", "404", "Forbidden", "NoSuchKey", "NotFound"}:
                raise
    return False


def _expire_job_record(job_id, now_iso, error_message):
    if not dynamodb or not JOBS_TABLE or not job_id:
        return
    try:
        dynamodb.Table(JOBS_TABLE).update_item(
            Key={"job_id": job_id},
            ConditionExpression=(
                "attribute_not_exists(#t_start_proc) AND "
                "(attribute_not_exists(#status) OR #status = :pending OR #status = :accepted)"
            ),
            UpdateExpression=(
                "SET #status = :expired, #grocery_status = :expired, "
                "#updated_at = :now, #failed_at = :now, #error_msg = :error"
            ),
            ExpressionAttributeNames={
                "#status": "status",
                "#grocery_status": "grocery_status",
                "#updated_at": "updated_at",
                "#failed_at": "failed_at",
                "#error_msg": "error_msg",
                "#t_start_proc": "t_start_proc",
            },
            ExpressionAttributeValues={
                ":pending": "PENDING",
                ":accepted": "ACCEPTED",
                ":expired": "EXPIRED",
                ":now": now_iso,
                ":error": error_message,
            },
        )
    except ClientError as exc:
        error_code = str((exc.response or {}).get("Error", {}).get("Code") or "")
        if error_code != "ConditionalCheckFailedException":
            raise


def _expire_triage_row(cur, table_name, row, now):
    now_mysql = now.strftime("%Y-%m-%d %H:%M:%S")
    now_iso = now.isoformat()
    error_message = "Upload URL expired before the file was uploaded."
    cur.execute(
        f"""
        UPDATE `{table_name}`
        SET status_normalized = %s,
            raw_status = %s,
            detail = %s,
            error_message = %s,
            updated_at = %s,
            failed_at = %s
        WHERE job_id = %s
          AND LOWER(status_normalized) = 'accepted'
          AND processing_started_at IS NULL
        """,
        ["failed", "EXPIRED", error_message, error_message, now_mysql, now_mysql, row.get("job_id")],
    )
    _expire_job_record(row.get("job_id"), now_iso, error_message)


def _expire_stale_accepted_rows(conn, table_name):
    cutoff = datetime.now(timezone.utc).timestamp() - ACCEPTED_STALE_TIMEOUT_SECONDS
    cutoff_mysql = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT job_id, job_type, s3_key, accepted_at, created_at, updated_at, processing_started_at
            FROM `{table_name}`
            WHERE LOWER(status_normalized) = 'accepted'
              AND processing_started_at IS NULL
              AND COALESCE(accepted_at, created_at, updated_at) < %s
            ORDER BY COALESCE(accepted_at, created_at, updated_at) ASC
            LIMIT {TRIAGE_RECONCILE_BATCH_LIMIT}
            """,
            [cutoff_mysql],
        )
        stale_rows = cur.fetchall() or []

        expired_count = 0
        for row in stale_rows:
            accepted_time = (
                _parse_row_datetime(row.get("accepted_at"))
                or _parse_row_datetime(row.get("created_at"))
                or _parse_row_datetime(row.get("updated_at"))
            )
            if not accepted_time:
                continue
            if accepted_time.timestamp() >= cutoff:
                continue
            if _s3_object_exists(row.get("job_type"), row.get("s3_key")):
                continue
            _expire_triage_row(cur, table_name, row, datetime.now(timezone.utc))
            expired_count += 1
    return {
        "table": table_name,
        "scanned": len(stale_rows),
        "expired": expired_count,
    }


def _list_triage_tables(conn):
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        rows = cur.fetchall() or []
    table_names = []
    for row in rows:
        if not isinstance(row, dict) or not row:
            continue
        table_name = next(iter(row.values()))
        if isinstance(table_name, str) and table_name.endswith("_triage"):
            table_names.append(table_name)
    table_names.sort()
    return table_names[:TRIAGE_RECONCILE_TABLE_LIMIT]


def _run_scheduled_reconciliation():
    try:
        conn = _mysql_conn()
        table_names = _list_triage_tables(conn)
        results = []
        total_scanned = 0
        total_expired = 0
        for table_name in table_names:
            summary = _expire_stale_accepted_rows(conn, table_name)
            results.append(summary)
            total_scanned += summary["scanned"]
            total_expired += summary["expired"]
        return _response(
            200,
            {
                "ok": True,
                "tables": len(table_names),
                "scanned": total_scanned,
                "expired": total_expired,
                "results": results,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        return _response(500, {"error": str(exc), "message": "Failed to reconcile stale triage rows"})


def _reconcile_triage_row(cur, table_name, job_id, status, raw_status, message, now):
    now_mysql = now.strftime("%Y-%m-%d %H:%M:%S")
    assignments = [
        "status_normalized = %s",
        "raw_status = %s",
        "updated_at = %s",
        "detail = %s",
    ]
    params = [status, raw_status, now_mysql, message]

    if status == "failed":
        assignments.extend([
            "error_message = %s",
            "failed_at = COALESCE(failed_at, %s)",
            "completed_at = NULL",
        ])
        params.extend([message, now_mysql])
    elif status == "completed":
        assignments.extend([
            "error_message = NULL",
            "completed_at = COALESCE(completed_at, %s)",
            "failed_at = NULL",
        ])
        params.append(now_mysql)
    elif status == "processing":
        assignments.append("processing_started_at = COALESCE(processing_started_at, %s)")
        params.append(now_mysql)
    elif status == "accepted":
        assignments.append("accepted_at = COALESCE(accepted_at, %s)")
        params.append(now_mysql)

    params.append(job_id)
    cur.execute(
        f"""
        UPDATE `{table_name}`
        SET {', '.join(assignments)}
        WHERE job_id = %s
        """,
        params,
    )


def _get_job_record(job_id):
    if not dynamodb or not JOBS_TABLE or not job_id:
        return None
    response = dynamodb.Table(JOBS_TABLE).get_item(Key={"job_id": job_id})
    return response.get("Item")


def _ddb_status_for_triage_status(status):
    if status == "accepted":
        return "PENDING"
    if status == "processing":
        return "PROCESSING"
    if status == "completed":
        return "DONE"
    return "FAILED"


def _reconcile_job_record(job_id, status, message, now, force=False, current_job=None):
    if not dynamodb or not JOBS_TABLE or not job_id:
        return {
            "updated": False,
            "reason": "jobs_table_unavailable",
            "job": None,
        }

    table = dynamodb.Table(JOBS_TABLE)
    job = current_job or _get_job_record(job_id)
    if not job:
        return {
            "updated": False,
            "reason": "job_not_found",
            "job": None,
        }

    current_status = str(job.get("status") or "").strip().upper()
    if not force and current_status in {"DONE", "FAILED", "EXPIRED"}:
        return {
            "updated": False,
            "reason": f"job_already_{current_status.lower()}",
            "job": _safe_serialize(job),
        }

    now_iso = now.isoformat()
    next_job = dict(job)
    next_job["status"] = _ddb_status_for_triage_status(status)
    next_job["updated_at"] = now_iso

    if "grocery_status" in next_job or next_job.get("type") != "voice":
        next_job["grocery_status"] = _ddb_status_for_triage_status(status)

    if status == "processing":
        next_job["t_start_proc"] = next_job.get("t_start_proc") or now_iso
    elif status == "completed":
        next_job["t_done"] = next_job.get("t_done") or now_iso
        next_job.pop("error_msg", None)
        next_job.pop("failed_at", None)
    elif status == "failed":
        next_job["failed_at"] = next_job.get("failed_at") or now_iso
        if message:
            next_job["error_msg"] = message
    else:
        next_job.setdefault("t_presign", now_iso)

    table.put_item(Item=next_job)
    return {
        "updated": True,
        "reason": "updated",
        "job": _safe_serialize(next_job),
    }


def _list_triage(owner, query):
    table_name = _triage_table_name(owner)
    if not table_name:
        return _response(400, {"error": "owner is required"})

    try:
        limit = int(query.get("limit") or 100)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 500))
    statuses = _parse_status_filters(query.get("status"))
    source = str(query.get("source") or "").strip().lower()
    job_type = str(query.get("job_type") or "").strip().lower()
    job_id = str(query.get("job_id") or "").strip()
    device_id = str(query.get("device_id") or "").strip()
    session_id = str(query.get("session_id") or "").strip()
    before = _parse_before(query.get("before"))

    clauses = []
    params = []
    if statuses:
        if len(statuses) == 1:
            clauses.append("LOWER(status_normalized) = %s")
            params.append(statuses[0])
        else:
            placeholders = ", ".join(["%s"] * len(statuses))
            clauses.append(f"LOWER(status_normalized) IN ({placeholders})")
            params.extend(statuses)
    if source:
        clauses.append("LOWER(source) = %s")
        params.append(source)
    if job_type:
        clauses.append("LOWER(job_type) = %s")
        params.append(job_type)
    if job_id:
        clauses.append("job_id = %s")
        params.append(job_id)
    if device_id:
        clauses.append("_device = %s")
        params.append(device_id)
    if session_id:
        clauses.append("session_id = %s")
        params.append(session_id)
    if before:
        clauses.append("COALESCE(updated_at, created_at) < %s")
        params.append(before)
    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    try:
        conn = _mysql_conn()
        if not _table_exists(conn, table_name):
            return _response(
                200,
                {
                    "owner": owner,
                    "count": 0,
                    "items": [],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT *
                FROM `{table_name}`
                {where_clause}
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT {limit}
                """,
                params,
            )
            rows = cur.fetchall() or []
    except Exception as exc:
        return _response(500, {"error": str(exc), "message": "Failed to load triage feed"})

    items = [_map_row(row) for row in rows]
    return _response(
        200,
        {
            "owner": owner,
            "count": len(items),
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _handle_admin_reconcile(event):
    body = _parse_json_body(event)
    job_id = _require_string(body, "job_id")
    target_status = _normalize_triage_status(body.get("status") or "failed")
    raw_status = _clean_optional_string(body.get("raw_status")) or _default_raw_status_for_status(target_status)
    force = _coerce_bool(body.get("force"), default=False)

    job = _get_job_record(job_id)
    owner = _clean_optional_string(body.get("owner")) or _clean_optional_string((job or {}).get("owner"))
    table_name = _triage_table_name(owner)
    if not table_name:
        raise ValueError("Missing owner")

    message = _clean_optional_string(body.get("message"))
    now = datetime.now(timezone.utc)

    conn = _mysql_conn()
    if not _table_exists(conn, table_name):
        return _response(404, {"error": "triage table not found", "owner": owner, "job_id": job_id})

    with conn.cursor() as cur:
        row = _get_triage_row(cur, table_name, job_id)
        if not row:
            return _response(404, {"error": "triage row not found", "owner": owner, "job_id": job_id})
        if not message:
            message = _default_admin_message(target_status, raw_status)
        _reconcile_triage_row(cur, table_name, job_id, target_status, raw_status, message, now)
        reconciled_row = _get_triage_row(cur, table_name, job_id)

    job_result = _reconcile_job_record(
        job_id,
        target_status,
        message,
        now,
        force=force,
        current_job=job,
    )

    return _response(
        200,
        {
            "ok": True,
            "owner": owner,
            "job_id": job_id,
            "status": target_status,
            "raw_status": raw_status,
            "message": message,
            "triage": _map_row(reconciled_row),
            "job_reconciliation": job_result,
        },
    )


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        if event.get("source") == "aws.events":
            return _run_scheduled_reconciliation()

        http_method = event.get("requestContext", {}).get("http", {}).get("method", "")
        raw_path = (
            event.get("rawPath")
            or event.get("requestContext", {}).get("http", {}).get("path", "")
            or ""
        )
        normalized_path = raw_path.rstrip("/")

        if http_method == "OPTIONS":
            return _response(200, {})

        if http_method == "POST" and normalized_path.endswith("/triage/admin/reconcile"):
            return _handle_admin_reconcile(event)

        if http_method == "DELETE":
            # Extract job_id from path: /triage/{owner}/{job_id}
            path_parts = normalized_path.strip("/").split("/")
            if len(path_parts) >= 3 and path_parts[0] == "triage":
                job_id = path_parts[2]
                owner = _sanitize_owner(path_parts[1])
                if not owner or not job_id:
                    return _response(400, {"error": "Missing owner or job_id"})
                conn = _mysql_conn()
                table = _triage_table_name(owner)
                if not _table_exists(conn, table):
                    return _response(404, {"error": "No triage table found"})
                with conn.cursor() as cur:
                    row = _get_triage_row(cur, table, job_id)
                    if not row:
                        return _response(404, {"error": "Job not found"})
                    # Only cancel if still in-flight
                    current_status = (row.get("status") or "").lower()
                    if current_status in ("pending", "accepted", "processing"):
                        cur.execute(
                            f"UPDATE `{table}` SET status='cancelled', raw_status='CANCELLED', "
                            f"updated_at=NOW() WHERE job_id=%s",
                            [job_id]
                        )
                        conn.commit()
                    # Also delete the preliminary kitchen item written by fast analysis.
                    # Archive-then-delete: snapshot the full row into shared_archive_kitchen
                    # first (reason 'triage_cancel') so a cancelled provisional item is still
                    # kept on file, mirroring kitchen_api's _shared_kitchen_delete.
                    try:
                        cur.execute(
                            "INSERT IGNORE INTO `shared_archive_kitchen` "
                            "SELECT *, NOW() as archived_at, 'triage_cancel' as archived_reason, "
                            "'shared_kitchen' as archived_from_table "
                            "FROM `shared_kitchen` WHERE `job_id` = %s AND `_owner` = %s",
                            [job_id, owner]
                        )
                        cur.execute(
                            "DELETE FROM `shared_kitchen` WHERE `job_id` = %s AND `_owner` = %s",
                            [job_id, owner]
                        )
                        deleted_count = cur.rowcount
                        conn.commit()
                        if deleted_count > 0:
                            print(f"[cancel] Deleted {deleted_count} preliminary kitchen row(s) for job {job_id}")
                    except Exception as e:
                        print(f"[cancel] Failed to delete preliminary kitchen row: {e}")
                return _response(200, {"cancelled": True, "job_id": job_id})

        owner = (event.get("pathParameters") or {}).get("owner")
        return _list_triage(owner, event.get("queryStringParameters") or {})
    except json.JSONDecodeError as exc:
        return _response(400, {"error": f"Invalid JSON body: {exc}"})
    except ValueError as exc:
        return _response(400, {"error": str(exc)})
    except Exception as exc:
        return _response(500, {"error": str(exc), "message": "Failed to process triage request"})
