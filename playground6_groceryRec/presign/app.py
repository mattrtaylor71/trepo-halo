# presign/app.py
# Force source hash changes so Lambda repackages bundled dependencies.
import os, json, re
from urllib.parse import urlencode
from typing import Optional
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from triage import upsert_triage_row

ALLOWED_ACTIONS = {"IN", "OUT"}
_SAVED_RECIPE_UPLOAD_PREFIX = "recipe-images"
_SAVED_RECIPE_UPLOAD_MAX_FILES = 8
_SAVED_RECIPE_ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
    "image/avif",
}

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

BUCKET_NAME   = os.environ["BUCKET_NAME"]
DISCARDS_BUCKET_NAME = os.environ.get("DISCARDS_BUCKET_NAME")

# HALO production buckets. Both default to empty, and an empty allow-list means no request is
# ever routed to them, so this whole feature is inert until a device id is explicitly added.
PROD_BUCKET_NAME = (os.environ.get("HALO_PROD_BUCKET_NAME") or "").strip()
PROD_DISCARDS_BUCKET_NAME = (os.environ.get("HALO_PROD_DISCARDS_BUCKET_NAME") or "").strip()
HALO_PROD_DEVICE_IDS = {
    d.strip() for d in (os.environ.get("HALO_PROD_DEVICE_IDS") or "").split(",") if d.strip()
}


def _is_halo_prod_device(device_id):
    """True only for device ids explicitly allow-listed for the production buckets.

    Exact match on purpose. See the routing block below for why a prefix rule is unsafe here.
    """
    return bool(device_id) and str(device_id).strip() in HALO_PROD_DEVICE_IDS
JOBS_TABLE    = os.environ["JOBS_TABLE"]
KEY_PREFIX    = os.environ.get("KEY_PREFIX", "images/")
PRESIGN_TTL_S = int(os.environ.get("PRESIGN_TTL_SECONDS", "300"))
_MYSQL_IDENTIFIER_LIMIT = 64
_OWNER_TABLE_SUFFIXES = (
    "_prod_kitchen",
    "_discards",
    "_dishes",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_url(event) -> Optional[str]:
    request_context = event.get("requestContext") or {}
    domain_name = request_context.get("domainName")
    if not domain_name:
        return None
    stage = request_context.get("stage")
    if stage and stage != "$default":
        return f"https://{domain_name}/{stage}"
    return f"https://{domain_name}"


def _dish_result_url(event, user_id: str, device_id: str, job_id: str) -> Optional[str]:
    base_url = _base_url(event)
    if not base_url:
        return None
    query = urlencode({
        "user_id": user_id,
        "device_id": device_id,
        "job_id": job_id,
    })
    return f"{base_url}/dish/result?{query}"


def _s3_key(
    user_id: str,
    device_id: str,
    job_id: str,
    content_type: str = "image/jpeg",
    suffix: Optional[str] = None,
) -> str:
    dt = datetime.now(timezone.utc)
    # Map content-type -> extension (default to jpg)
    if content_type.lower() == "image/png":
        ext = "png"
    else:
        # treat image/jpeg or anything else as jpg
        ext = "jpg"
    key_id = f"{job_id}_{suffix}" if suffix else job_id
    return f"{KEY_PREFIX}{user_id}/{device_id}/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{key_id}.{ext}"


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    if value is None:
        return False
    return bool(value)


def _clean_optional_string(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    text = str(value).strip()
    return text or None


def _validate_owner_for_tables(owner: str) -> Optional[str]:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", owner or "")
    if not sanitized:
        return "owner must include at least one alphanumeric character"
    max_owner_length = min(_MYSQL_IDENTIFIER_LIMIT - len(suffix) for suffix in _OWNER_TABLE_SUFFIXES)
    if len(sanitized) > max_owner_length:
        return f"owner is too long for backend table naming (max {max_owner_length} characters)"
    return None


def _extract_camera_meta(value) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    print(json.dumps({
        "event": "presign_ignored_invalid_camera_meta",
        "reason": "camera_meta must be an object when provided",
        "provided_type": type(value).__name__,
    }))
    return None


def _normalize_saved_recipe_image_content_type(value: Optional[str]) -> str:
    normalized = _clean_optional_string(value or "image/jpeg")
    normalized = (normalized or "image/jpeg").lower()
    if normalized == "image/jpg":
        return "image/jpeg"
    if normalized not in _SAVED_RECIPE_ALLOWED_IMAGE_CONTENT_TYPES:
        raise ValueError(
            "content_type must be one of image/jpeg, image/png, image/webp, image/gif, image/heic, image/heif, or image/avif"
        )
    return normalized


def _saved_recipe_extension(content_type: str, filename: Optional[str] = None) -> str:
    normalized = (content_type or "").lower()
    extension_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/heic": "heic",
        "image/heif": "heif",
        "image/avif": "avif",
    }
    if normalized in extension_map:
        return extension_map[normalized]
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp", "gif", "heic", "heif", "avif"}:
        return "jpg" if suffix == "jpeg" else suffix
    return "jpg"


def _saved_recipe_public_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _saved_recipe_upload_key(owner: str, session_id: str, file_index: int, content_type: str, filename: Optional[str] = None) -> str:
    ext = _saved_recipe_extension(content_type, filename=filename)
    return (
        f"{_SAVED_RECIPE_UPLOAD_PREFIX}/{owner}/saved-recipes/uploads/"
        f"{session_id}/{int(file_index) + 1:02d}-{uuid4().hex}.{ext}"
    )


def _build_saved_recipe_presign_response(owner: str, files: list) -> dict:
    if not owner:
        raise ValueError("owner is required")
    owner_validation_error = _validate_owner_for_tables(owner)
    if owner_validation_error:
        raise ValueError(owner_validation_error)
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty array")
    if len(files) > _SAVED_RECIPE_UPLOAD_MAX_FILES:
        raise ValueError(f"files must contain at most {_SAVED_RECIPE_UPLOAD_MAX_FILES} items")

    session_id = uuid4().hex
    uploads = []
    for index, item in enumerate(files):
        payload = item or {}
        filename = _clean_optional_string(payload.get("filename")) or f"recipe-image-{index + 1}"
        content_type = _normalize_saved_recipe_image_content_type(payload.get("content_type"))
        key = _saved_recipe_upload_key(owner, session_id, index, content_type, filename=filename)
        put_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": BUCKET_NAME, "Key": key, "ContentType": content_type},
            ExpiresIn=PRESIGN_TTL_S,
        )
        uploads.append({
            "file_id": f"file-{index + 1}",
            "filename": filename,
            "content_type": content_type,
            "put_url": put_url,
            "image_url": _saved_recipe_public_url(BUCKET_NAME, key),
            "s3_key": key,
            "expires_in": PRESIGN_TTL_S,
        })

    print(json.dumps({
        "event": "saved_recipe_presign_created",
        "owner": owner,
        "file_count": len(uploads),
        "upload_session_id": session_id,
    }))
    return {
        "owner": owner,
        "upload_session_id": session_id,
        "uploads": uploads,
        "count": len(uploads),
        "expires_in": PRESIGN_TTL_S,
    }


def handler(event, context):
    try:
        path = event.get("rawPath") or event.get("requestContext", {}).get("http", {}).get("path", "")
        if path.rstrip("/").endswith("/saved-recipes/presign"):
            body = json.loads(event.get("body") or "{}")
            response = _build_saved_recipe_presign_response(
                owner=_clean_optional_string(body.get("owner")),
                files=body.get("files") or [],
            )
            return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(response)}
        is_discard_route = path.rstrip("/").endswith("/presign/discard")
        body = json.loads(event.get("body") or "{}")

        # Required fields
        user_id   = body["user_id"]
        device_id = body["device_id"]
        owner = _clean_optional_string(body.get("owner"))

        # Optional fields
        action = (body.get("action") or "IN").upper()
        if action not in ALLOWED_ACTIONS:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "action must be IN or OUT"}),
            }

        content_type = (body.get("content_type") or "image/jpeg").lower()
        expiration_content_type = (body.get("expiration_content_type") or content_type).lower()
        default_type = "discard" if is_discard_route else "grocery"
        type_ = body.get("type", default_type)
        route_kind = "discard" if is_discard_route else "presign"
        product_expiration = body.get("product_expiration")  # Optional: YYYY-MM-DD format
        include_expiration_image = bool(body.get("include_expiration_image"))
        add_to_shopping_list = _coerce_bool(body.get("add_to_shopping_list"))
        camera_meta = _extract_camera_meta(body.get("camera_meta"))
        if not owner:
            print(json.dumps({
                "event": "presign_rejected_missing_owner",
                "route": route_kind,
                "type": type_,
                "user_id": user_id,
                "device_id": device_id,
                "owner_present": False,
            }))
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "owner is required"}),
            }
        owner_validation_error = _validate_owner_for_tables(owner)
        if owner_validation_error:
            print(json.dumps({
                "event": "presign_rejected_invalid_owner",
                "route": route_kind,
                "type": type_,
                "user_id": user_id,
                "device_id": device_id,
                "owner": owner,
                "reason": owner_validation_error,
            }))
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": owner_validation_error}),
            }
        quantity_raw = body.get("quantity", 1)
        try:
            quantity = int(quantity_raw)
        except (TypeError, ValueError):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "quantity must be a positive integer"}),
            }
        if quantity < 1:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "quantity must be a positive integer"}),
            }

        # New job
        job_id = uuid4().hex
        key = _s3_key(user_id, device_id, job_id, content_type)
        expiration_key = None
        if include_expiration_image:
            expiration_key = _s3_key(user_id, device_id, job_id, expiration_content_type, suffix="exp")

        # Put job row (status starts PENDING; analyzer will flip to RUNNING/DONE)
        jobs = dynamodb.Table(JOBS_TABLE)
        now = _iso_now()
        job_item = {
            "job_id": job_id,
            "user_id": user_id,
            "device_id": device_id,
            "type": type_,
            "action": action,
            "status": "PENDING",
            "grocery_status": "PENDING",
            "created_at": now,
            "updated_at": now,
            "t_presign": now,
            "s3_key": key,
            "owner": owner,
        }
        if product_expiration:
            job_item["product_expiration"] = product_expiration
        if camera_meta is not None:
            job_item["camera_meta"] = camera_meta
        if expiration_key:
            job_item["expiration_s3_key"] = expiration_key
            job_item["expiration_status"] = "PENDING"
            job_item["expiration_expected"] = True
            job_item["expiration_content_type"] = expiration_content_type
        job_item["quantity"] = quantity
        job_item["add_to_shopping_list"] = add_to_shopping_list
        if add_to_shopping_list:
            job_item["shopping_list_status"] = "PENDING"
        jobs.put_item(Item=job_item)
        triage_metadata = {
            "route_kind": route_kind,
            "expiration_expected": bool(expiration_key),
            "quantity": quantity,
            "add_to_shopping_list": add_to_shopping_list,
        }
        if camera_meta is not None:
            triage_metadata["camera_meta"] = camera_meta
        try:
            upsert_triage_row({
                "owner": owner,
                "device_id": device_id,
                "user_id": user_id,
                "job_id": job_id,
                "source": "image",
                "job_type": type_,
                "status_normalized": "accepted",
                "raw_status": "PENDING",
                "action": action,
                "title": "Dish image upload" if type_ == "dish" else ("Discard image upload" if type_ == "discard" else "Kitchen image upload"),
                "detail": "Upload accepted and waiting for processing.",
                "s3_key": key,
                "created_at": now,
                "updated_at": now,
                "accepted_at": now,
                "metadata": triage_metadata,
            })
        except Exception as triage_error:
            print(f"[triage] Failed to seed triage row: {triage_error}")

        # Presign URL (PUT; requires matching Content-Type on upload)
        target_bucket = BUCKET_NAME
        if (is_discard_route or type_ == "discard") and DISCARDS_BUCKET_NAME:
            target_bucket = DISCARDS_BUCKET_NAME

        # HALO production cutover. This route is NOT device-only: the iOS app posts the same
        # body shape here from every capture flow (UploadAPIService.swift), and 27% of the
        # objects in the upload bucket come from the app. So routing is opt-in per device and
        # defaults to today's behaviour for everything else.
        #
        # Deliberately an explicit allow-list, NOT a `device_id.startswith("halo-")` prefix:
        # the app sends `userService.deviceName`, a user-editable iPhone name, so a prefix rule
        # is one oddly-named phone away from silently filing a real customer's photo into the
        # device bucket. An allow-list cannot do that.
        #
        # Empty (the default) => no device routes anywhere new and this block is a no-op.
        if _is_halo_prod_device(device_id):
            prod_bucket = (PROD_DISCARDS_BUCKET_NAME
                           if (is_discard_route or type_ == "discard")
                           else PROD_BUCKET_NAME)
            if prod_bucket:
                target_bucket = prod_bucket
        put_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": target_bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=PRESIGN_TTL_S,
        )
        expiration_put_url = None
        if expiration_key:
            expiration_put_url = s3.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": target_bucket, "Key": expiration_key, "ContentType": expiration_content_type},
                ExpiresIn=PRESIGN_TTL_S,
            )

        resp = {
            "job_id": job_id,
            "put_url": put_url,
            "s3_key": key,
            "expires_in": PRESIGN_TTL_S,
            "ttl_s": PRESIGN_TTL_S,
            "content_type": content_type,
        }
        if type_ == "dish":
            result_url = _dish_result_url(event, user_id, device_id, job_id)
            if result_url:
                resp["result_url"] = result_url
        if expiration_put_url and expiration_key:
            resp["expiration_put_url"] = expiration_put_url
            resp["expiration_s3_key"] = expiration_key
        print(json.dumps({
            "event": "presign_created",
            "route": route_kind,
            "type": type_,
            "job_id": job_id,
            "user_id": user_id,
            "device_id": device_id,
            "owner": owner,
            "owner_present": True,
            "add_to_shopping_list": add_to_shopping_list,
            "include_expiration_image": include_expiration_image,
            "camera_meta_present": camera_meta is not None,
            "camera_meta": camera_meta,
        }))
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(resp)}

    except KeyError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"missing field: {e.args[0]}"}),
        }
    except ValueError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
    except ClientError as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"aws error: {e.response.get('Error', {}).get('Message', str(e))}"}),
        }
    except Exception as e:
        return {"statusCode": 500, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": str(e)})}
