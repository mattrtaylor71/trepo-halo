import base64
import hashlib
import json
import os
import time
import zlib
from decimal import Decimal

import boto3
import analytics
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


OTA_REPORT_EVENTS_TABLE = os.environ["OTA_REPORT_EVENTS_TABLE"]
OTA_DEVICE_LATEST_TABLE = os.environ["OTA_DEVICE_LATEST_TABLE"]

KNOWN_REPORT_TYPES = {
    "boot",
    "maintenance_enter",
    "ota_result",
    "pre_sleep",
    "manifest_ok",
    "manifest_err",
    "maintenance_done",
    "lcd_ota_result",
    "heartbeat",
    "health",
    "action_result",
}

RECENT_REPORT_TYPES = [
    "pre_sleep",
    "ota_result",
    "maintenance_enter",
    "boot",
    "manifest_ok",
    "manifest_err",
    "maintenance_done",
    "lcd_ota_result",
    "heartbeat",
]

INT_FIELDS = {
    "ts_epoch",
    "boot_count",
    "wake_cause",
    "uptime_ms",
    "maintenance_mode",
    "maintenance_in_window",
    "maint_scheduled",
    "maint_start_epoch",
    "maint_end_epoch",
    "maint_wake_epoch",
    "sched_fetch_http",
    "maint_sync_pending",
    "maint_sync_attempts",
    "lcd_maint_ack",
    "rssi",
}

dynamodb = boto3.resource("dynamodb")


def _json_default(value):
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    return str(value)


def _response(status_code, payload=None):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Device-Id,X-Owner-Id,X-Request-Id,Authorization",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Cache-Control": "no-store",
    }
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": "" if payload is None else json.dumps(payload, default=_json_default),
    }


def _log(message, **fields):
    payload = {"message": message, **fields}
    print(json.dumps(payload, default=_json_default))


def _lower_headers(event):
    headers = event.get("headers") or {}
    normalized = {}
    for key, value in headers.items():
        if key is None:
            continue
        normalized[str(key).lower()] = value
    return normalized


def _parse_json_body(event):
    raw_body = event.get("body")
    if not raw_body:
        return {}
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")
    body = json.loads(raw_body)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return body


def _clean_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    value = str(value).strip()
    return value or None


def _safe_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_payload(body, headers):
    normalized = {}
    for key, value in body.items():
        normalized[key] = value.strip() if isinstance(value, str) else value

    header_fallbacks = {
        "device_id": headers.get("x-device-id"),
        "owner_id": headers.get("x-owner-id"),
        "request_id": headers.get("x-request-id"),
    }
    for key, value in header_fallbacks.items():
        if _clean_string(normalized.get(key)) is None and _clean_string(value) is not None:
            normalized[key] = _clean_string(value)

    for field in INT_FIELDS:
        if field in normalized:
            converted = _safe_int(normalized.get(field))
            if converted is not None:
                normalized[field] = converted

    for field in ["device_id", "owner_id", "device_type", "board", "fw", "build", "report_type", "channel", "request_id"]:
        if field in normalized:
            cleaned = _clean_string(normalized.get(field))
            if cleaned is None:
                normalized.pop(field, None)
            else:
                normalized[field] = cleaned

    return normalized


def _require_string(payload, field_name):
    value = _clean_string(payload.get(field_name))
    if value is None:
        raise ValueError(f"missing_required_field:{field_name}")
    return value


def _validate_payload(payload):
    device_id = _require_string(payload, "device_id")
    device_type = _require_string(payload, "device_type")
    fw = _require_string(payload, "fw")
    build = _require_string(payload, "build")
    report_type = _require_string(payload, "report_type")
    ts_epoch = _safe_int(payload.get("ts_epoch"))

    if ts_epoch is None:
        raise ValueError("invalid_field:ts_epoch")
    if report_type not in KNOWN_REPORT_TYPES:
        raise ValueError("invalid_field:report_type")

    payload["device_id"] = device_id
    payload["device_type"] = device_type
    payload["fw"] = fw
    payload["build"] = build
    payload["report_type"] = report_type
    payload["ts_epoch"] = ts_epoch

    return payload


def _diagnostic_blob(payload, name, size):
    value = payload.get(name)
    if not isinstance(value, str) or len(value) != ((size + 2) // 3) * 4:
        raise ValueError("invalid_field:diagnostic_export")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("invalid_field:diagnostic_export") from None
    if len(raw) != size or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("invalid_field:diagnostic_export")
    return raw


def _diagnostic_text(raw, offset, size):
    field = raw[offset:offset + size]
    end = field.find(b"\0")
    if end <= 0 or any(c < 32 or c > 126 for c in field[:end]):
        raise ValueError("invalid_field:diagnostic_export")
    return field[:end].decode("ascii")


def _validated_diagnostic_export(payload):
    # Ordinary diagnostic discovery fields are still ordinary device reports.
    # Only these reserved export-routing fields request exclusion from latest.
    routing = {"diag_export", "diag_handoff", "diag_context_b64", "diag_record_b64",
               "diag_handoff_b64", "diag_handoff_schema"}
    if not routing.intersection(payload):
        return False

    def require(ok):
        if not ok:
            raise ValueError("invalid_field:diagnostic_export")

    def integer(name, low, high):
        value = payload.get(name)
        require(type(value) is int and low <= value <= high)
        return value

    require(integer("diag_export", 1, 1) == 1 and integer("diag_schema", 3, 3) == 3)
    require(payload["report_type"] == "heartbeat")
    slot = integer("diag_slot", 1, 3)
    sequence = integer("diag_sequence", 2, (1 << 64) - 1)
    record_crc = integer("diag_crc", 0, (1 << 32) - 1)
    context = _diagnostic_blob(payload, "diag_context_b64", 256)
    record = _diagnostic_blob(payload, "diag_record_b64", 256)
    u16 = lambda b, p: int.from_bytes(b[p:p + 2], "little")
    u32 = lambda b, p: int.from_bytes(b[p:p + 4], "little")
    u64 = lambda b, p: int.from_bytes(b[p:p + 8], "little")

    def header(raw, kind, index, length):
        return (raw[:4] == b"HDG1" and raw[4:8] == bytes([3, kind, index, 0]) and
                u16(raw, 20) == length and u16(raw, 22) == 0 and
                u32(raw, 252) == (zlib.crc32(raw[:252]) & 0xffffffff))

    require(header(context, 1, 0, 212) and u64(context, 8) == 1 and u32(context, 16) == 0)
    c = context[24:]
    require(any(c[:16]) and any(c[16:32]) and any(c[32:64]) and any(c[160:166]))
    require(c[166] in (1, 2) and c[167] == 1 and u32(c, 168) > 0)
    _diagnostic_text(c, 64, 32)
    _diagnostic_text(c, 96, 64)
    require(header(record, 3 if slot == 3 else 2, slot, 228))
    require(u32(record, 16) == u32(context, 252) and u64(record, 8) == sequence and
            u32(record, 252) == record_crc)
    p = record[24:]
    if slot == 3:
        require(u64(p, 0) != 0 and 1 <= u16(p, 136) <= 22 and p[138] <= 1 and
                p[207] <= 2 and p[208] <= 3)
        build, fw, epoch = _diagnostic_text(p, 20, 96), _diagnostic_text(p, 140, 12), u64(p, 8)
    else:
        require(any(p[:16]) and u64(p, 16) != 0 and 1 <= u16(p, 36) <= 22 and
                u32(p, 44) <= u32(p, 48) == u32(c, 168) and p[212] == slot - 1 and p[213] <= 3)
        build, fw, epoch = _diagnostic_text(p, 72, 96), _diagnostic_text(p, 216, 12), u64(p, 24)
    device = "halo-" + c[162:164].hex() + "-" + c[164:166].hex()
    require(payload["device_id"] == device and payload["device_type"] == ("sense" if c[166] == 1 else "lcd"))
    require(payload["build"] == build and payload["fw"] == fw)

    handoff_fields = {"diag_handoff", "diag_handoff_schema", "diag_handoff_b64"}
    if handoff_fields.intersection(payload):
        require(integer("diag_handoff", 1, 1) == 1 and integer("diag_handoff_schema", 4, 4) == 4)
        handoff = _diagnostic_blob(payload, "diag_handoff_b64", 72)
        require(handoff[:16] == c[:16] and handoff[69:72] == b"\0\0\0" and handoff[68] == slot)
        require(u32(handoff, 52) == u32(context, 252))
        epoch = u32(handoff, 56)
        require(epoch >= 1577836800)
        tuples = [(u64(handoff, 16 + i * 12), u32(handoff, 24 + i * 12)) for i in range(3)]
        require(all((n == 0 and crc == 0) or n >= 2 for n, crc in tuples))
        require(tuples[slot - 1] == (sequence, record_crc) and sequence == max(n for n, _ in tuples))
        require(len({n for n, _ in tuples if n}) == sum(bool(n) for n, _ in tuples))
        request = f"h4-{c[:16].hex()}-{zlib.crc32(handoff) & 0xffffffff:08x}"
    else:
        request = f"d3-{c[:16].hex()}-{sequence:016x}-{record_crc:08x}"
    require(payload["ts_epoch"] == epoch and payload.get("request_id") == request)
    return True


def _utc_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_event_keys(payload):
    base = "|".join(
        [
            payload["device_id"],
            payload["report_type"],
            str(payload["ts_epoch"]),
            payload["build"],
            payload.get("request_id") or "",
        ]
    )
    ingest_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    event_ts_key = f'{payload["ts_epoch"]:010d}#{payload["report_type"]}#{ingest_id}'
    return ingest_id, event_ts_key


def _put_event(table, payload):
    now_epoch = int(time.time())
    now_iso = _utc_now_iso()
    ingest_id, event_ts_key = _make_event_keys(payload)

    item = {
        "device_id": payload["device_id"],
        "event_ts_key": event_ts_key,
        "ingest_id": ingest_id,
        "device_type": payload["device_type"],
        "fw": payload["fw"],
        "build": payload["build"],
        "report_type": payload["report_type"],
        "ts_epoch": payload["ts_epoch"],
        "payload": payload,
        "ingested_at": now_iso,
        "ingested_at_epoch": now_epoch,
    }

    for field in ["owner_id", "board", "channel", "request_id", "ota_result", "sched_fetch", "lcd_ota_result"]:
        value = payload.get(field)
        if value is not None and value != "":
            item[field] = value

    table.put_item(Item=item)
    return item, now_epoch


def _build_latest_item(payload, event_item, now_epoch):
    latest = {
        "device_id": payload["device_id"],
        "device_type": payload["device_type"],
        "last_seen_epoch": payload["ts_epoch"],
        "last_report_type": payload["report_type"],
        "last_fw": payload["fw"],
        "last_build": payload["build"],
        "last_payload": payload,
        "last_event_ts_key": event_item["event_ts_key"],
        "updated_at": _utc_now_iso(),
        "updated_at_epoch": now_epoch,
    }

    optional_mappings = {
        "owner_id": "owner_id",
        "board": "board",
        "channel": "channel",
        "request_id": "last_request_id",
        "ota_result": "last_ota_result",
        "manifest_version": "last_manifest_version",
        "manifest_build_id": "last_manifest_build_id",
        "sched_fetch": "last_sched_fetch",
        "sched_fetch_http": "last_sched_fetch_http",
        "sched_fetch_request_id": "last_sched_fetch_request_id",
        "maintenance_mode": "last_maintenance_mode",
        "maintenance_in_window": "last_maintenance_in_window",
        "lcd_fw": "last_lcd_fw",
        "lcd_ota_result": "last_lcd_ota_result",
        "lcd_maint_ack": "last_lcd_maint_ack",
        "maint_sync_resolution": "last_maint_sync_resolution",
        "health": "last_health",
        "health_flags": "last_health_flags",
    }

    for source_key, target_key in optional_mappings.items():
        value = payload.get(source_key)
        if value is not None and value != "":
            latest[target_key] = value

    return latest


def _update_latest_if_newer(table, latest_item):
    expression_names = {}
    expression_values = {":last_seen_epoch": latest_item["last_seen_epoch"]}
    set_parts = []
    remove_parts = []

    for key, value in latest_item.items():
        if key == "device_id":
            continue
        name_key = f"#n{len(expression_names)}"
        expression_names[name_key] = key
        if value is None:
            remove_parts.append(name_key)
            continue
        value_key = f":v{len(expression_values)}"
        expression_values[value_key] = value
        set_parts.append(f"{name_key} = {value_key}")

    update_parts = []
    if set_parts:
        update_parts.append("SET " + ", ".join(set_parts))
    if remove_parts:
        update_parts.append("REMOVE " + ", ".join(remove_parts))

    try:
        table.update_item(
            Key={"device_id": latest_item["device_id"]},
            UpdateExpression=" ".join(update_parts),
            ConditionExpression="attribute_not_exists(last_seen_epoch) OR last_seen_epoch <= :last_seen_epoch",
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values,
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def _query_device_events(table, device_id, limit):
    response = table.query(
        KeyConditionExpression=Key("device_id").eq(device_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])


def _query_request_events(table, request_id, limit):
    response = table.query(
        IndexName="RequestIdTsIndex",
        KeyConditionExpression=Key("request_id").eq(request_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])


def _query_report_type_events(table, report_type, limit):
    response = table.query(
        IndexName="ReportTypeTsIndex",
        KeyConditionExpression=Key("report_type").eq(report_type),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])


def _list_recent_events(table, limit):
    merged = []
    for report_type in RECENT_REPORT_TYPES:
        merged.extend(_query_report_type_events(table, report_type, limit))
    merged.sort(
        key=lambda item: (
            _safe_int(item.get("ts_epoch"), 0),
            str(item.get("event_ts_key") or ""),
        ),
        reverse=True,
    )
    deduped = []
    seen = set()
    for item in merged:
        event_key = (item.get("device_id"), item.get("event_ts_key"))
        if event_key in seen:
            continue
        seen.add(event_key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _scan_latest_states(table, limit):
    items = []
    scan_kwargs = {"Limit": min(limit, 200)}
    while len(items) < limit:
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response or len(items) >= limit:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return items


def _query_latest_by_owner(table, owner_id, limit):
    response = table.query(
        IndexName="OwnerLatestIndex",
        KeyConditionExpression=Key("owner_id").eq(owner_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])


def _get_latest_state(latest_table, query):
    device_id = _clean_string((query or {}).get("device_id"))
    owner_id = _clean_string((query or {}).get("owner_id"))
    stale_hours = _safe_int((query or {}).get("stale_hours"))
    limit = min(max(_safe_int((query or {}).get("limit"), 50), 1), 200)

    if device_id:
        item = latest_table.get_item(Key={"device_id": device_id}).get("Item")
        items = [item] if item else []
    elif owner_id:
        items = _query_latest_by_owner(latest_table, owner_id, limit)
    else:
        items = _scan_latest_states(latest_table, limit)

    now_epoch = int(time.time())
    stale_before_epoch = None
    if stale_hours is not None and stale_hours >= 0:
        stale_before_epoch = now_epoch - (stale_hours * 3600)
        items = [item for item in items if _safe_int(item.get("last_seen_epoch"), now_epoch) <= stale_before_epoch]

    items = sorted(
        items,
        key=lambda item: (
            _safe_int(item.get("last_seen_epoch"), 0),
            str(item.get("device_id") or ""),
        ),
        reverse=True,
    )[:limit]

    return _response(
        200,
        {
            "ok": True,
            "count": len(items),
            "items": items,
            "server_time_epoch": now_epoch,
            "stale_before_epoch": stale_before_epoch,
        },
    )


def _get_events(events_table, query):
    query = query or {}
    device_id = _clean_string(query.get("device_id"))
    request_id = _clean_string(query.get("request_id"))
    report_type = _clean_string(query.get("report_type"))
    limit = min(max(_safe_int(query.get("limit"), 50), 1), 200)

    if device_id:
        items = _query_device_events(events_table, device_id, limit)
    elif request_id:
        items = _query_request_events(events_table, request_id, limit)
    elif report_type:
        if report_type not in KNOWN_REPORT_TYPES:
            return _response(400, {"ok": False, "error": "invalid_field", "field": "report_type"})
        items = _query_report_type_events(events_table, report_type, limit)
    else:
        items = _list_recent_events(events_table, limit)

    return _response(
        200,
        {
            "ok": True,
            "count": len(items),
            "items": items,
        },
    )


def _ingest_report(events_table, latest_table, event):
    headers = _lower_headers(event)
    raw_payload = _parse_json_body(event)
    payload = _validate_payload(_normalize_payload(raw_payload, headers))
    diagnostic_export = _validated_diagnostic_export(payload)

    _log(
        "ota_report_received",
        device_id=payload["device_id"],
        owner_id=payload.get("owner_id"),
        report_type=payload["report_type"],
        request_id=payload.get("request_id"),
        fw=payload["fw"],
        build=payload["build"],
    )

    event_item, now_epoch = _put_event(events_table, payload)
    latest_updated = False
    analytics_latest_updated = False
    if not diagnostic_export:
        latest_item = _build_latest_item(payload, event_item, now_epoch)
        latest_updated = _update_latest_if_newer(latest_table, latest_item)
        try:
            analytics_latest_updated = analytics.update_latest(latest_table, payload, event_item, now_epoch)
        except Exception:
            # The accepted legacy report remains acknowledged if its optional
            # projection fails. No payload or exception contents in this log.
            _log("analytics_latest_projection_failed", status="error")

    _log(
        "ota_report_ingested",
        device_id=payload["device_id"],
        report_type=payload["report_type"],
        request_id=payload.get("request_id"),
        latest_updated=latest_updated,
        event_ts_key=event_item["event_ts_key"],
        status="ok",
    )

    return _response(
        200,
        {
            "ok": True,
            "ingested": True,
            "latest_updated": latest_updated,
            "analytics_latest_updated": analytics_latest_updated,
            "server_time_epoch": now_epoch,
            "event_ts_key": event_item["event_ts_key"],
        },
    )


def handler(event, context):
    try:
        http = event.get("requestContext", {}).get("http", {})
        http_method = http.get("method", "")
        raw_path = http.get("path", "")
        normalized_path = raw_path.rstrip("/")

        if http_method == "OPTIONS":
            return _response(200, {})

        events_table = dynamodb.Table(OTA_REPORT_EVENTS_TABLE)
        latest_table = dynamodb.Table(OTA_DEVICE_LATEST_TABLE)

        if http_method == "POST" and normalized_path.endswith("/ota/report"):
            return _ingest_report(events_table, latest_table, event)

        if http_method == "GET" and (event.get("queryStringParameters") or {}).get("view") == "analytics":
            return analytics.serve(event, events_table, latest_table, _response)

        if http_method == "GET" and normalized_path.endswith("/ota/report/latest"):
            return _get_latest_state(latest_table, event.get("queryStringParameters"))

        if http_method == "GET" and normalized_path.endswith("/ota/report/events"):
            return _get_events(events_table, event.get("queryStringParameters"))

        return _response(405, {"ok": False, "error": f"Method {http_method} not allowed for {normalized_path}"})
    except json.JSONDecodeError as exc:
        _log("ota_report_invalid_json", status="error", error=str(exc))
        return _response(400, {"ok": False, "error": "invalid_json", "detail": str(exc)})
    except ValueError as exc:
        error_text = str(exc)
        if error_text.startswith("missing_required_field:"):
            field_name = error_text.split(":", 1)[1]
            _log("ota_report_validation_failed", status="error", error="missing_required_field", field=field_name)
            return _response(400, {"ok": False, "error": "missing_required_field", "field": field_name})
        if error_text.startswith("invalid_field:"):
            field_name = error_text.split(":", 1)[1]
            _log("ota_report_validation_failed", status="error", error="invalid_field", field=field_name)
            return _response(400, {"ok": False, "error": "invalid_field", "field": field_name})
        _log("ota_report_validation_failed", status="error", error=error_text)
        return _response(400, {"ok": False, "error": error_text})
    except Exception as exc:
        _log("ota_report_handler_exception", status="error", error=str(exc))
        return _response(500, {"ok": False, "error": "internal_server_error", "detail": str(exc)})
