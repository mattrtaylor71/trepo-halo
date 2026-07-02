import base64
import hashlib
# Force republish after dependency-layer restore.
import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime
from decimal import Decimal
from html import unescape
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import boto3
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
import yt_dlp

USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

# Saved recipes are PER-USER (private) by default. When this is off (the default), a save is
# NOT copied to other household members, a delete does not remove it from other members, and an
# edit only touches the acting user's own copy. Set to 'true' only to restore household sharing.
SAVED_RECIPE_HOUSEHOLD_FANOUT = os.getenv('SAVED_RECIPE_HOUSEHOLD_FANOUT', 'false').lower() == 'true'

_LAYER_PYTHON = Path(__file__).resolve().parents[1] / 'recipe_inventory_layer' / 'python'
if _LAYER_PYTHON.exists() and str(_LAYER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_LAYER_PYTHON))

import recipe_inventory_llm


_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_REQUEST_TIMEOUT_SECONDS = 15
_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/136.0.0.0 Safari/537.36'
)
_VALID_TIKTOK_HOSTS = {
    'tiktok.com',
    'www.tiktok.com',
    'm.tiktok.com',
    'vm.tiktok.com',
    'vt.tiktok.com',
}
_VALID_INSTAGRAM_HOSTS = {
    'instagram.com',
    'www.instagram.com',
    'm.instagram.com',
}
_EXTRACT_CONTENT_PATHS = {'/extract-caption', '/extract-content'}
_RECIPE_FROM_CONTENT_PATHS = {'/recipe-from-caption', '/recipe-from-content'}
_ANALYZE_URL_PATHS = {'/analyze-tiktok', '/analyze-url'}
_ASYNC_TASK_GENERATE_SAVED_RECIPE_IMAGE = 'generate_saved_recipe_image'
_ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH = 'process_saved_recipe_batch'
_ASYNC_TASK_REFINE_SAVED_RECIPE_TEXT = 'refine_saved_recipe_text'
_SAVED_RECIPE_BATCH_JOB_TYPE = 'saved_recipe_batch'
_JOB_STATUS_PENDING = 'PENDING'
_JOB_STATUS_RUNNING = 'RUNNING'
_JOB_STATUS_COMPLETED = 'COMPLETED'
_JOB_STATUS_FAILED = 'FAILED'
_DEFAULT_TRANSCRIPTION_MODEL = os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'whisper-1')
_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200
_OPENAI_TIMEOUT_SECONDS = int(os.getenv('OPENAI_TIMEOUT_SECONDS', '45'))
_OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '2'))
_YTDLP_MAX_ATTEMPTS = max(1, int(os.getenv('YTDLP_MAX_ATTEMPTS', '3')))
_YTDLP_RETRY_SLEEP_SECONDS = max(0.0, float(os.getenv('YTDLP_RETRY_SLEEP_SECONDS', '0.75')))
_APIFY_TOKEN_ENV = 'APIFY_TOKEN'
# ScrapeCreators — the only provider that handles TikTok PHOTO/slideshow posts
# (yt-dlp/oembed/html all 502 on /photo/ URLs). Returns caption (desc) + the full
# carousel of slide image URLs.
_SCRAPECREATORS_API_KEY_ENV = 'SCRAPECREATORS_API_KEY'
_SCRAPECREATORS_VIDEO_ENDPOINT = 'https://api.scrapecreators.com/v2/tiktok/video'
# Cap slide OCR to bound latency/cost within the 120s Lambda timeout.
_SLIDESHOW_OCR_MAX_SLIDES = max(1, int(os.getenv('SLIDESHOW_OCR_MAX_SLIDES', '10')))
_APIFY_INSTAGRAM_ACTOR_ENV = 'APIFY_INSTAGRAM_ACTOR'
_APIFY_INSTAGRAM_ACTOR = os.getenv(_APIFY_INSTAGRAM_ACTOR_ENV, 'apify~instagram-scraper')
_APIFY_RUN_TIMEOUT_SECONDS = max(1, int(os.getenv('APIFY_RUN_TIMEOUT_SECONDS', '45')))
_YTDLP_COOKIEFILE_ENV = 'YTDLP_COOKIEFILE'
_YTDLP_COOKIEFILE_B64_ENV = 'YTDLP_COOKIEFILE_B64'
_YTDLP_SSM_PARAM_NAME = '/trepo/ytdlp-cookiefile-b64'
_ytdlp_cookiefile_b64_cache = None
_ytdlp_cookiefile_cache_loaded = False
_OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
_OPENAI_RECIPE_IMAGE_MODEL = os.getenv('OPENAI_RECIPE_IMAGE_MODEL', 'gpt-image-1')
_OPENAI_SUBSTITUTION_MODEL = os.getenv('OPENAI_SUBSTITUTION_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))
_MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS = max(0, int(os.getenv('SAVED_RECIPES_MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS', '2')))
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
_MAX_BATCH_IMAGE_COUNT = max(1, int(os.getenv('SAVED_RECIPE_MAX_BATCH_IMAGES', '8')))
_VISION_SUPPORTED_CONTENT_TYPES = {
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/webp',
    'image/gif',
}
_CONVERTIBLE_PHONE_CONTENT_TYPES = {
    'image/heic',
    'image/heif',
    'image/avif',
}
_CONVERTIBLE_PHONE_EXTENSIONS = {'heic', 'heif', 'avif'}
_SAVED_RECIPE_SELECT_FIELDS = """_id, _owner, source_type, source_url, resolved_url, title, image_url, image_urls,
                       source_image_url, source_image_urls, image_storage_key,
                       ingredients, instructions, notes, raw_caption, raw_content, extraction_source,
                       author_name, caption_field, status, _createdDate, _updatedDate"""
_OWNER_KITCHEN_STATE_TABLE = 'owner_kitchen_state'
_OWNER_RECIPE_AVAILABILITY_TABLE = 'owner_recipe_availability'
_INGREDIENT_NOISE_TOKENS = {
    'a', 'an', 'and', 'fresh', 'organic', 'large', 'small', 'medium', 'lean', 'extra', 'virgin',
    'boneless', 'skinless', 'shredded', 'chopped', 'diced', 'minced', 'sliced', 'ground',
    'raw', 'cooked', 'dry', 'plain', 'whole', 'halves', 'pieces', 'piece', 'pack',
    'packs', 'package', 'packages', 'bag', 'bags', 'box', 'boxes', 'bottle', 'bottles', 'jar',
    'jars', 'can', 'cans', 'count', 'ct', 'lb', 'lbs', 'oz', 'g', 'kg', 'ml', 'l',
    'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp',
    'pound', 'pounds', 'ounce', 'ounces', 'pinch', 'dash', 'to', 'taste', 'for', 'of',
    'cut', 'into', 'in', 'about', 'roughly', 'thinly', 'finely', 'thick', 'thin',
    'divided', 'optional', 'needed', 'serving', 'garnish', 'topping',
    'halved', 'quartered', 'cubed', 'chunk', 'chunks', 'strip', 'strips', 'clove', 'cloves',
    'plus', 'plu', 'more', 'adjust', 'as', 'sized', 'each', 'per', 'with', 'or', 'cooking',
    'deseeded', 'peeled', 'trimmed', 'rinsed', 'drained', 'crushed', 'pressed', 'grated',
    'juiced', 'zested', 'squeezed', 'melted', 'softened', 'thawed', 'warmed', 'chilled',
    'room', 'temperature', 'beaten', 'whisked', 'sifted', 'toasted', 'roasted',
    # Quantity/descriptor words from natural-language recipe amounts
    'little', 'less', 'than', 'just', 'under', 'over', 'around', 'approximately',
    'generous', 'heaping', 'scant', 'level', 'splash', 'drizzle', 'handful', 'bit',
    'some', 'few', 'several', 'couple', 'good', 'big', 'tiny', 'slight', 'light',
    'heavy', 'full', 'half', 'quarter', 'third', 'basically', 'overflowing', 'not',
    'really', 'very', 'super', 'like', 'the', 'your', 'my', 'any', 'all', 'no',
    'amount', 'generou', 'generous',
}
_PANTRY_CANONICAL_INGREDIENTS = {
    'salt', 'kosher salt', 'sea salt', 'table salt', 'flaky salt',
    'pepper', 'black pepper', 'white pepper',
    'oil', 'olive oil', 'vegetable oil', 'canola oil', 'neutral oil', 'cooking oil', 'sesame oil',
    'water', 'cold water', 'warm water', 'hot water', 'ice water', 'ice',
    'butter', 'salted butter', 'unsalted butter',
    'cooking spray', 'nonstick cooking spray',
    'sugar', 'white sugar', 'granulated sugar', 'brown sugar', 'powdered sugar',
    'flour', 'all purpose flour', 'wheat flour',
    'baking soda', 'baking powder',
    'vinegar', 'white vinegar', 'apple cider vinegar',
    'basic spice', 'spice', 'basic spices', 'seasoning', 'seasonings',
    'garlic powder', 'onion powder', 'paprika', 'cumin', 'oregano', 'cinnamon',
    'cornstarch', 'corn starch',
}
_PANTRY_TOKENS = {'salt', 'pepper', 'oil', 'water', 'ice', 'butter', 'spray',
                  'sugar', 'flour', 'spice', 'spices', 'seasoning', 'seasonings'}
_INGREDIENT_CONFLICT_TOKENS = {
    'apple', 'avocado', 'banana', 'bean', 'beef', 'berry', 'bread', 'broccoli', 'broth',
    'butter', 'cabbage', 'carrot', 'cauliflower', 'celery', 'cheese', 'cherry', 'chicken',
    'chili', 'corn', 'cream', 'egg', 'fish', 'flour', 'garlic', 'grape', 'juice', 'kale',
    'lemon', 'lettuce', 'lime', 'mango', 'milk', 'mushroom', 'onion', 'orange', 'pasta',
    'paste', 'pea', 'peach', 'pear', 'pepper', 'pork', 'potato', 'powder', 'rice', 'salmon',
    'sauce', 'seasoning', 'seasonings', 'shrimp', 'spinach', 'stock', 'strawberry', 'sugar',
    'tomatillo', 'tomato', 'tuna', 'turkey', 'vinegar', 'yogurt',
}

# A generic category token (e.g. "cheese") is redundant — NOT a real conflict — when a
# specific variety of that category (e.g. "parmesan") is already present in the match
# context. Without this, the conflict-token guard wrongly blocks valid matches like
# kitchen "Parmesan" vs recipe "parmesan cheese". It must NOT loosen genuine conflicts:
# kitchen "cream" vs recipe "cream cheese" still fails (no cheese variety present).
_CATEGORY_VARIETY_TOKENS = {
    'cheese': {
        'parmesan', 'parmigiano', 'reggiano', 'cheddar', 'mozzarella', 'feta', 'gouda',
        'brie', 'provolone', 'gruyere', 'asiago', 'romano', 'pecorino', 'ricotta',
        'mascarpone', 'manchego', 'colby', 'swiss', 'havarti', 'gorgonzola', 'fontina',
        'halloumi', 'paneer', 'cotija', 'burrata', 'camembert', 'edam', 'emmental',
        'jarlsberg', 'muenster', 'queso',
    },
}


def _strip_redundant_category_conflicts(conflict_tokens, context_tokens):
    """Drop category conflict tokens (e.g. 'cheese') that are redundant because a specific
    variety of that category (e.g. 'parmesan') is present in context_tokens."""
    if not conflict_tokens:
        return conflict_tokens
    return {
        t for t in conflict_tokens
        if not (_CATEGORY_VARIETY_TOKENS.get(t) and (context_tokens & _CATEGORY_VARIETY_TOKENS[t]))
    }


class ServiceError(RuntimeError):
    def __init__(self, message, status_code=500, extra=None):
        super().__init__(message)
        self.status_code = status_code
        self.extra = extra or {}


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
        autocommit=True,
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


def _safe_text(value):
    return str(value or '').strip()


def _utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def _parse_limit(query):
    try:
        limit = int((query or {}).get('limit') or _DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT
    return max(1, min(limit, _MAX_LIST_LIMIT))


def _parse_before(query):
    raw = _safe_text((query or {}).get('before'))
    if not raw:
        return None
    normalized = raw.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _unique_texts(values):
    seen = set()
    items = []
    for value in values or []:
        text = _safe_text(value)
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return items


def _clean_string_list(values):
    return _unique_texts([str(value or '').strip() for value in (values or [])])


def _log_event(request_id, event_name, **fields):
    payload = {
        'request_id': request_id,
        'event': event_name,
        **fields,
    }
    print(json.dumps(payload, default=json_serial))


def _report_backend_error(op, owner_id=None, code=None, error=None, job_id=None, service='recipes'):
    """Emit a metric-filterable marker (evt=backend_error) for a swallowed soft failure. Never throws."""
    try:
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
        }), file=sys.stderr)
    except Exception:
        pass


def _cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS'
    }


def _success(body, status=200):
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps(body, default=json_serial)}


def _error(status, message, extra=None):
    payload = {'error': message}
    if extra:
        payload.update(extra)
    return {'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps(payload, default=json_serial)}


def _safe_owner_token(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))


def _get_household_member_ids(conn, acting_user_id):
    """Return all user_ids sharing the same household (owner_id) as the acting user."""
    safe = _safe_owner_token(acting_user_id)
    if not safe:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT owner_id FROM new_users WHERE user_id = %s LIMIT 1", [safe])
        row = cur.fetchone() or {}
        household_id = row.get('owner_id')
        if not household_id:
            return [safe]
        cur.execute(
            "SELECT user_id FROM new_users WHERE owner_id = %s ORDER BY created_at ASC, user_id ASC",
            [household_id],
        )
        members = [_safe_owner_token(r.get('user_id')) for r in (cur.fetchall() or [])]
    members = [m for m in members if m]
    return list(dict.fromkeys(members)) or [safe]


def _fan_out_saved_recipe_to_household(conn, primary_owner, recipe_id, request_id=None):
    """Copy a saved recipe from the primary owner's table to all other household members."""
    if not SAVED_RECIPE_HOUSEHOLD_FANOUT:
        return  # saved recipes are per-user; do not copy to other household members
    members = _get_household_member_ids(conn, primary_owner)
    other_members = [m for m in members if m != _safe_owner_token(primary_owner)]
    if not other_members:
        return
    # Fetch the full row from the primary owner's table
    row = _fetch_saved_recipe_by_id(conn, primary_owner, recipe_id)
    if not row:
        return
    # _SAVED_RECIPE_SELECT_FIELDS doesn't carry resolved_url_hash, so recompute it
    # the same way the owner's row was hashed (_sha256(resolved_url)) — reproduces
    # the identical hash so the dedup-by-hash check still matches. resolved_url_hash
    # is NOT NULL in the member table; without this the fan-out INSERT threw
    # (1048, "Column 'resolved_url_hash' cannot be null") and household members
    # silently never received the shared recipe.
    member_hash = row.get('resolved_url_hash') or _sha256(
        row.get('resolved_url') or row.get('source_url') or recipe_id
    )
    for member_id in other_members:
        try:
            _ensure_saved_recipes_table(conn, member_id)
            member_table = _saved_recipes_table(member_id)
            # Check if this recipe already exists in the member's table (by hash or id)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT _id FROM `{member_table}` WHERE _id = %s OR resolved_url_hash = %s LIMIT 1",
                    [recipe_id, member_hash],
                )
                if cur.fetchone():
                    continue  # Already exists, skip
                cur.execute(
                    f"""INSERT INTO `{member_table}` (
                            _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                            title, image_url, image_urls, source_image_url, source_image_urls, image_storage_key,
                            ingredients, instructions, notes, raw_caption, raw_content,
                            extraction_source, author_name, caption_field, status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ready')""",
                    (
                        recipe_id,
                        member_id,
                        row.get('source_type'),
                        row.get('source_url'),
                        row.get('resolved_url'),
                        member_hash,
                        row.get('title'),
                        row.get('image_url'),
                        json.dumps(row.get('image_urls') or []) if isinstance(row.get('image_urls'), list) else row.get('image_urls'),
                        row.get('source_image_url'),
                        json.dumps(row.get('source_image_urls') or []) if isinstance(row.get('source_image_urls'), list) else row.get('source_image_urls'),
                        row.get('image_storage_key'),
                        json.dumps(row.get('ingredients') or []) if isinstance(row.get('ingredients'), list) else row.get('ingredients'),
                        json.dumps(row.get('instructions') or []) if isinstance(row.get('instructions'), list) else row.get('instructions'),
                        json.dumps(row.get('notes') or []) if isinstance(row.get('notes'), list) else row.get('notes'),
                        row.get('raw_caption'),
                        row.get('raw_content'),
                        row.get('extraction_source'),
                        row.get('author_name'),
                        row.get('caption_field'),
                    ),
                )
            conn.commit()
            # Also compute and persist availability for this member
            try:
                _ensure_recipe_personalization_tables(conn)
                kitchen_context = _build_kitchen_match_context(conn, member_id)
                member_row = _fetch_saved_recipe_by_id(conn, member_id, recipe_id)
                if member_row:
                    _, overlay_row = _serialize_saved_recipe_with_availability(
                        conn, member_id, member_row,
                        kitchen_context=kitchen_context,
                        request_id=request_id,
                    )
                    _persist_owner_recipe_availability_rows(conn, [overlay_row])
            except Exception:
                pass  # Non-critical — availability will be computed on next refresh
        except Exception as exc:
            _log_event(request_id, 'household_recipe_fanout_error', member=member_id, error=str(exc))
            continue


def _fan_out_delete_to_household(conn, primary_owner, recipe_id, request_id=None):
    """Delete a saved recipe from all other household members' tables."""
    if not SAVED_RECIPE_HOUSEHOLD_FANOUT:
        return  # saved recipes are per-user; a delete only affects the acting user
    members = _get_household_member_ids(conn, primary_owner)
    other_members = [m for m in members if m != _safe_owner_token(primary_owner)]
    if not other_members:
        return
    for member_id in other_members:
        try:
            member_table = _saved_recipes_table(member_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT COUNT(*) AS n FROM information_schema.tables
                        WHERE table_schema = DATABASE() AND table_name = %s""",
                    [member_table],
                )
                if cur.fetchone()['n'] == 0:
                    continue
                cur.execute(f"DELETE FROM `{member_table}` WHERE _id = %s LIMIT 1", [recipe_id])
            conn.commit()
            _delete_owner_recipe_availability(conn, member_id, 'saved', recipe_id)
        except Exception as exc:
            _log_event(request_id, 'household_recipe_delete_fanout_error', member=member_id, error=str(exc))
            continue


def _saved_recipes_table(owner):
    safe = _safe_owner_token(owner)
    if not safe:
        raise ServiceError('Invalid owner', status_code=400)
    return f"{safe}_saved_recipes"


def _table_has_column(conn, table, column_name):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS n
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s
        """, [table, column_name])
        return (cur.fetchone() or {}).get('n', 0) > 0


def _ensure_column(conn, table, column_name, ddl):
    if _table_has_column(conn, table, column_name):
        return
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE `{table}` ADD COLUMN {ddl}")
    conn.commit()


def _ensure_saved_recipes_table(conn, owner):
    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                _id VARCHAR(36) PRIMARY KEY,
                _owner VARCHAR(36) NOT NULL,
                source_type VARCHAR(32) NOT NULL DEFAULT 'tiktok',
                source_url VARCHAR(1000) NOT NULL,
                resolved_url VARCHAR(1000) NOT NULL,
                resolved_url_hash CHAR(64) NOT NULL,
                title VARCHAR(255) NOT NULL,
                image_url VARCHAR(1000) NULL,
                image_urls JSON NULL,
                source_image_url VARCHAR(1000) NULL,
                source_image_urls JSON NULL,
                image_storage_key VARCHAR(1000) NULL,
                ingredients JSON,
                instructions JSON,
                notes JSON,
                raw_caption TEXT,
                raw_content TEXT NULL,
                extraction_source VARCHAR(32),
                author_name VARCHAR(255),
                caption_field VARCHAR(64) NULL,
                status ENUM('ready', 'failed') NOT NULL DEFAULT 'ready',
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_resolved_url_hash (resolved_url_hash),
                KEY idx_created (_createdDate),
                KEY idx_title (title)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()
    _ensure_column(conn, table, 'raw_content', "`raw_content` TEXT NULL AFTER `raw_caption`")
    _ensure_column(conn, table, 'caption_field', "`caption_field` VARCHAR(64) NULL AFTER `author_name`")
    _ensure_column(conn, table, 'image_url', "`image_url` VARCHAR(1000) NULL AFTER `title`")
    _ensure_column(conn, table, 'image_urls', "`image_urls` JSON NULL AFTER `image_url`")
    _ensure_column(conn, table, 'source_image_url', "`source_image_url` VARCHAR(1000) NULL AFTER `image_urls`")
    _ensure_column(conn, table, 'source_image_urls', "`source_image_urls` JSON NULL AFTER `source_image_url`")
    _ensure_column(conn, table, 'image_storage_key', "`image_storage_key` VARCHAR(1000) NULL AFTER `source_image_urls`")


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


def _ensure_owner_recipe_availability_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_OWNER_RECIPE_AVAILABILITY_TABLE}` (
                owner VARCHAR(36) NOT NULL,
                recipe_source VARCHAR(32) NOT NULL,
                recipe_id VARCHAR(64) NOT NULL,
                kitchen_version BIGINT NOT NULL,
                can_make_exact TINYINT(1) NOT NULL DEFAULT 0,
                can_make_with_subs TINYINT(1) NOT NULL DEFAULT 0,
                matched_count INT NOT NULL DEFAULT 0,
                missing_count INT NOT NULL DEFAULT 0,
                ingredient_matches JSON NULL,
                missing_ingredients JSON NULL,
                substitution_candidates JSON NULL,
                substitution_summary TEXT NULL,
                substitution_status VARCHAR(32) NULL,
                analysis_status VARCHAR(32) NOT NULL DEFAULT 'ready',
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (owner, recipe_source, recipe_id),
                KEY idx_owner_source (owner, recipe_source),
                KEY idx_owner_exact (owner, can_make_exact),
                KEY idx_owner_subs (owner, can_make_with_subs)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def _ensure_recipe_personalization_tables(conn):
    _ensure_owner_kitchen_state_table(conn)
    _ensure_owner_recipe_availability_table(conn)


def _parse_json_field(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def _serialize_row(row):
    out = {k: json_serial(v) for k, v in row.items()}
    out['id'] = out.pop('_id', None)
    out['ingredients'] = _parse_json_field(row.get('ingredients'))
    out['instructions'] = _parse_json_field(row.get('instructions'))
    out['notes'] = _parse_json_field(row.get('notes'))
    out['image_urls'] = _parse_json_field(row.get('image_urls'))
    out['source_image_urls'] = _parse_json_field(row.get('source_image_urls'))
    out['platform'] = out.get('source_type')
    out['content'] = out.get('raw_content') or out.get('raw_caption') or None
    out['caption'] = out.get('raw_caption') or out.get('raw_content') or None
    return out


def _singularize_token(token):
    token = _safe_text(token).lower()
    if len(token) <= 3:
        return token
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    if token.endswith('oes') and len(token) > 4:
        return token[:-2]
    if token.endswith('s') and not token.endswith('ss'):
        return token[:-1]
    return token


_COMPOUND_WORD_SPLITS = {
    'flaxseed': ['flax', 'seed'], 'chickpea': ['chick', 'pea'], 'cornstarch': ['corn', 'starch'],
    'cornmeal': ['corn', 'meal'], 'oatmeal': ['oat', 'meal'], 'applesauce': ['apple', 'sauce'],
    'buttermilk': ['butter', 'milk'], 'sourdough': ['sour', 'dough'], 'breadcrumb': ['bread', 'crumb'],
    'popcorn': ['pop', 'corn'], 'arrowroot': ['arrow', 'root'], 'beeswax': ['bee', 'wax'],
    'cheesecloth': ['cheese', 'cloth'], 'eggplant': ['egg', 'plant'], 'grapefruit': ['grape', 'fruit'],
    'horseradish': ['horse', 'radish'], 'peppercorn': ['pepper', 'corn'], 'sugarcane': ['sugar', 'cane'],
    'sunflower': ['sun', 'flower'], 'watercress': ['water', 'cress'], 'watermelon': ['water', 'melon'],
    'wheatgerm': ['wheat', 'germ'], 'sweetpotato': ['sweet', 'potato'], 'sourcream': ['sour', 'cream'],
    'creamcheese': ['cream', 'cheese'], 'peanutbutter': ['peanut', 'butter'],
    'almondbutter': ['almond', 'butter'], 'cashewbutter': ['cashew', 'butter'],
    'coconutmilk': ['coconut', 'milk'], 'almondmilk': ['almond', 'milk'],
    'oatmilk': ['oat', 'milk'], 'soymilk': ['soy', 'milk'], 'ricemilk': ['rice', 'milk'],
}


def _ingredient_tokens(text):
    cleaned = unescape(_safe_text(text).lower())
    cleaned = re.sub(r'\([^)]*\)', ' ', cleaned)
    cleaned = cleaned.replace('&', ' and ')
    cleaned = re.sub(r'[^a-z0-9]+', ' ', cleaned)
    tokens = []
    for raw in cleaned.split():
        if not raw or raw.isdigit():
            continue
        if re.fullmatch(r'\d+(?:oz|lb|lbs|g|kg|ml|l|ct|pack)?', raw):
            continue
        token = _singularize_token(raw)
        if not token or token in _INGREDIENT_NOISE_TOKENS:
            continue
        if token in _COMPOUND_WORD_SPLITS:
            tokens.extend(_COMPOUND_WORD_SPLITS[token])
        else:
            tokens.append(token)
    return tokens


def _canonical_ingredient_name(text):
    return ' '.join(_ingredient_tokens(text))


def _is_pantry_ingredient(text):
    canonical = _canonical_ingredient_name(text)
    if not canonical:
        return False
    if canonical in _PANTRY_CANONICAL_INGREDIENTS:
        return True
    tokens = canonical.split()
    return bool(tokens) and all(token in _PANTRY_TOKENS for token in tokens)


def _get_owner_kitchen_version(conn, owner):
    _ensure_owner_kitchen_state_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT kitchen_version FROM `{_OWNER_KITCHEN_STATE_TABLE}` WHERE owner = %s LIMIT 1",
            [_safe_text(owner)],
        )
        row = cur.fetchone() or {}
    return int(row.get('kitchen_version') or 0)


def _get_cached_recipe_availability(conn, owner, recipe_ids, min_kitchen_version=None):
    """Return a dict keyed by recipe_id with cached availability data from owner_recipe_availability table.

    If min_kitchen_version is provided, discard cache rows with a lower version (stale).
    Returns an empty dict if no data is found or the table does not exist.
    """
    if not recipe_ids:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [_OWNER_RECIPE_AVAILABILITY_TABLE])
            if (cur.fetchone() or {}).get('n', 0) == 0:
                return {}
        safe_owner = _safe_text(owner)
        safe_ids = [_safe_text(rid) for rid in recipe_ids if _safe_text(rid)]
        if not safe_ids:
            return {}
        placeholders = ', '.join(['%s'] * len(safe_ids))
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT recipe_id, kitchen_version, can_make_exact, can_make_with_subs,
                           matched_count, missing_count, ingredient_matches, missing_ingredients,
                           substitution_candidates, substitution_summary, substitution_status, analysis_status
                    FROM `{_OWNER_RECIPE_AVAILABILITY_TABLE}`
                    WHERE owner = %s AND recipe_source = 'saved' AND recipe_id IN ({placeholders})""",
                [safe_owner] + safe_ids,
            )
            rows = cur.fetchall() or []
        result = {}
        for row in rows:
            rid = _safe_text(row.get('recipe_id'))
            if not rid:
                continue
            cached_version = int(row.get('kitchen_version') or 0)
            if min_kitchen_version is not None and cached_version < min_kitchen_version:
                continue  # Stale cache — skip
            result[rid] = {
                'kitchen_version': cached_version,
                'can_make_exact': bool(row.get('can_make_exact')),
                'can_make_with_subs': bool(row.get('can_make_with_subs')),
                'matched_count': int(row.get('matched_count') or 0),
                'missing_count': int(row.get('missing_count') or 0),
                'ingredient_matches': _parse_json_field(row.get('ingredient_matches')),
                'missing_ingredients': _parse_json_field(row.get('missing_ingredients')),
                'substitution_candidates': _parse_json_field(row.get('substitution_candidates')),
                'substitution_summary': row.get('substitution_summary'),
                'substitution_status': row.get('substitution_status'),
                'analysis_status': _safe_text(row.get('analysis_status') or 'ready'),
            }
        return result
    except Exception:
        return {}


def _get_kitchen_items_for_matching(conn, owner):
    if USE_SHARED_TABLES:
        member_ids = _get_household_member_ids(conn, owner)
        placeholders = ','.join(['%s'] * len(member_ids))
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT `_id`, `product_name`, `product_description`
                FROM `shared_kitchen`
                WHERE `owner_id` IN ({placeholders})
                  AND `action` = 'IN'
                  AND (`analysis_status` = 'ready' OR `analysis_status` IS NULL)
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC, `_createdDate` DESC
            """, member_ids)
            rows = cur.fetchall() or []
    else:
        table_name = f"{owner}_prod_kitchen"
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table_name])
            if (cur.fetchone() or {}).get('n', 0) == 0:
                return []
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s
                  AND column_name IN ('product_description', 'analysis_stage', 'analysis_status')
            """, [table_name])
            column_names = {
                (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
                for row in (cur.fetchall() or [])
                if (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
            }
            where_parts = ["action = 'IN'"]
            if 'analysis_stage' in column_names:
                where_parts.append("(`analysis_stage` = 'final' OR `analysis_stage` IS NULL)")
            if 'analysis_status' in column_names:
                where_parts.append("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)")
            select_fields = "`_id`, `product_name`"
            if 'product_description' in column_names:
                select_fields += ", `product_description`"
            cur.execute(f"""
                SELECT {select_fields}
                FROM `{table_name}`
                WHERE {' AND '.join(where_parts)}
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC, `_createdDate` DESC
            """)
            rows = cur.fetchall() or []
    items = []
    seen = set()
    for row in rows:
        display_name = _safe_text(row.get('product_name'))
        if not display_name:
            continue
        dedupe_key = display_name.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        description = _safe_text(row.get('product_description')) or None
        name_parts = [display_name]
        if description:
            name_parts.append(description)
        tokens = _ingredient_tokens(' '.join(name_parts))
        items.append({
            'item_id': _safe_text(row.get('_id')) or None,
            'display_name': display_name,
            'description': description,
            'canonical_ingredient': ' '.join(tokens),
            'tokens': tokens,
        })
    return items


def _build_kitchen_match_context(conn, owner):
    return {
        'kitchen_version': _get_owner_kitchen_version(conn, owner),
        'kitchen_candidates': _get_kitchen_items_for_matching(conn, owner),
    }


def _score_kitchen_candidate(recipe_tokens, candidate_tokens):
    recipe_set = set(recipe_tokens or [])
    candidate_set = set(candidate_tokens or [])
    if not recipe_set or not candidate_set:
        return -1
    if recipe_set == candidate_set:
        return 300 + len(recipe_set)
    context_tokens = recipe_set | candidate_set
    # Forward: recipe tokens fully contained in kitchen item tokens
    extra_tokens = candidate_set - recipe_set
    forward_conflicts = _strip_redundant_category_conflicts(
        {t for t in extra_tokens if t in _INGREDIENT_CONFLICT_TOKENS}, context_tokens)
    if recipe_set.issubset(candidate_set) and not forward_conflicts:
        return 200 + (len(recipe_set) * 10) - max(0, len(candidate_set) - len(recipe_set))
    # Reverse: kitchen item tokens fully contained in recipe tokens
    # e.g. recipe "2 lemons juiced" (tokens: lemon) vs kitchen "Organic Lemons" (tokens: lemon)
    if candidate_set.issubset(recipe_set):
        recipe_extra = recipe_set - candidate_set
        conflict_in_extra = _strip_redundant_category_conflicts(
            {t for t in recipe_extra if t in _INGREDIENT_CONFLICT_TOKENS}, context_tokens)
        if not conflict_in_extra:
            return 150 + (len(candidate_set) * 10) - max(0, len(recipe_set) - len(candidate_set))
    return -1


def _match_recipe_ingredient(recipe_ingredient, kitchen_candidates):
    recipe_text = _safe_text(recipe_ingredient)
    tokens = _ingredient_tokens(recipe_text)
    canonical = ' '.join(tokens)
    if not recipe_text:
        return None
    if _is_pantry_ingredient(recipe_text):
        return {
            'recipe_ingredient': recipe_text,
            'canonical_ingredient': canonical,
            'match_status': 'pantry',
            'matched_kitchen_items': [],
            'substitute_kitchen_items': [],
            'missing_reason': None,
        }
    best_candidate = None
    best_score = -1
    for candidate in kitchen_candidates or []:
        score = _score_kitchen_candidate(tokens, candidate.get('tokens') or [])
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate:
        match_type = 'canonical_exact'
        if set(tokens or []) != set(best_candidate.get('tokens') or []):
            match_type = 'canonical_subset'
        return {
            'recipe_ingredient': recipe_text,
            'canonical_ingredient': canonical,
            'match_status': 'have',
            'matched_kitchen_items': [{
                'item_id': best_candidate.get('item_id'),
                'display_name': best_candidate.get('display_name'),
                'match_type': match_type,
            }],
            'substitute_kitchen_items': [],
            'missing_reason': None,
        }
    return {
        'recipe_ingredient': recipe_text,
        'canonical_ingredient': canonical,
        'match_status': 'missing',
        'matched_kitchen_items': [],
        'substitute_kitchen_items': [],
        'missing_reason': 'No direct kitchen match found.',
    }


def _log_inventory_match_event(request_id, event_name, **kwargs):
    _log_event(request_id, event_name, **kwargs)


def _compute_recipe_availability(recipe_ingredients, kitchen_context, request_id=None, recipe=None):
    recipe_payload = {
        'id': _safe_text(((recipe or {}).get('id') or 'inline-recipe')),
        'title': _safe_text((recipe or {}).get('title')),
        'ingredients': _clean_string_list((recipe or {}).get('ingredients') or recipe_ingredients or []),
    }
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        [recipe_payload],
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label='saved',
        substitution_callback=_suggest_saved_recipe_substitutions,
    )
    result = availability_map.get(recipe_payload['id']) or recipe_inventory_llm.deterministic_availability(
        recipe_payload,
        kitchen_context,
        substitution_callback=_suggest_saved_recipe_substitutions,
    )
    return _local_deterministic_merge(result, kitchen_context)


def _local_deterministic_merge(availability, kitchen_context):
    """Safety-net merge: if deterministic matching finds a 'have' that the LLM missed, upgrade it."""
    ingredient_matches = availability.get('ingredient_matches') or []
    kitchen_candidates = (kitchen_context or {}).get('kitchen_candidates') or []
    changed = False
    for match in ingredient_matches:
        if match.get('match_status') != 'missing':
            continue
        recipe_text = _safe_text(match.get('recipe_ingredient'))
        if not recipe_text:
            continue
        det = _match_recipe_ingredient(recipe_text, kitchen_candidates)
        if det and det.get('match_status') == 'have':
            match['match_status'] = 'have'
            match['matched_kitchen_items'] = det.get('matched_kitchen_items') or []
            match['missing_reason'] = None
            changed = True
    if changed:
        matched = sum(1 for m in ingredient_matches if m.get('match_status') in {'have', 'pantry'})
        missing = len(ingredient_matches) - matched
        availability['matched_count'] = matched
        availability['missing_count'] = missing
        availability['can_make_exact'] = missing == 0
        availability['missing_ingredients'] = [
            _safe_text(m.get('recipe_ingredient'))
            for m in ingredient_matches if m.get('match_status') == 'missing'
        ]
    return availability


def _compute_saved_recipe_availability_batch(recipes, kitchen_context, request_id=None):
    recipe_payloads = []
    for recipe in recipes or []:
        recipe_id = _safe_text((recipe or {}).get('id'))
        if not recipe_id:
            continue
        recipe_payloads.append({
            'id': recipe_id,
            'title': _safe_text((recipe or {}).get('title')),
            'ingredients': _clean_string_list((recipe or {}).get('ingredients') or []),
        })
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        recipe_payloads,
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label='saved_batch',
        substitution_callback=_suggest_saved_recipe_substitutions,
    )
    # Safety-net: upgrade any LLM "missing" that deterministic can match
    for recipe_id, avail in availability_map.items():
        availability_map[recipe_id] = _local_deterministic_merge(avail, kitchen_context)
    return availability_map


def _suggest_saved_recipe_substitutions(recipe, kitchen_context, missing_ingredients):
    if not missing_ingredients or len(missing_ingredients) > _MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS:
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': None,
            'can_make_with_subs': False,
        }
    kitchen_items = [
        item.get('display_name')
        for item in ((kitchen_context or {}).get('kitchen_candidates') or [])
        if item.get('display_name')
    ]
    if not kitchen_items:
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': None,
            'can_make_with_subs': False,
        }
    system = """You help users substitute recipe ingredients with items already in their kitchen.
Return valid JSON only in this schema:
{
  "substitutions": [
    {
      "missing_ingredient": "ingredient name",
      "use_instead": "kitchen item name",
      "confidence": "high|medium|low",
      "notes": "brief explanation"
    }
  ],
  "summary": "one short paragraph for the app"
}
Only include substitutions that are genuinely plausible. If no good substitutions exist, return an empty substitutions array and summary as null."""
    user = json.dumps({
        'recipe_title': recipe.get('title'),
        'recipe_ingredients': recipe.get('ingredients') or [],
        'missing_ingredients': missing_ingredients,
        'kitchen_items': kitchen_items,
    })
    try:
        response = _openai_client().chat.completions.create(
            model=_OPENAI_SUBSTITUTION_MODEL,
            temperature=0.2,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        )
        payload = _best_effort_json_parse(_strip_code_fences(response.choices[0].message.content), {'substitutions': [], 'summary': None})
    except Exception:
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': 'failed',
            'can_make_with_subs': False,
        }
    substitutions = payload.get('substitutions') or []
    normalized = []
    covered_missing = set()
    for item in substitutions:
        missing_ingredient = _safe_text((item or {}).get('missing_ingredient'))
        use_instead = _safe_text((item or {}).get('use_instead'))
        if not missing_ingredient or not use_instead:
            continue
        covered_missing.add(missing_ingredient.lower())
        normalized.append({
            'missing_ingredient': missing_ingredient,
            'use_instead': use_instead,
            'confidence': _safe_text((item or {}).get('confidence') or 'medium').lower() or 'medium',
            'notes': _safe_text((item or {}).get('notes')) or None,
        })
    return {
        'substitution_candidates': normalized,
        'substitution_summary': payload.get('summary'),
        'substitution_status': 'ready' if normalized else 'none',
        'can_make_with_subs': bool(normalized) and len(covered_missing) >= len({item.lower() for item in missing_ingredients}),
    }


def _build_owner_recipe_availability_record(owner, recipe_source, recipe_id, availability):
    return {
        'owner': _safe_text(owner),
        'recipe_source': _safe_text(recipe_source),
        'recipe_id': _safe_text(recipe_id),
        'kitchen_version': int((availability or {}).get('kitchen_version') or 0),
        'can_make_exact': 1 if (availability or {}).get('can_make_exact') else 0,
        'can_make_with_subs': 1 if (availability or {}).get('can_make_with_subs') else 0,
        'matched_count': int((availability or {}).get('matched_count') or 0),
        'missing_count': int((availability or {}).get('missing_count') or 0),
        'ingredient_matches': json.dumps((availability or {}).get('ingredient_matches') or []),
        'missing_ingredients': json.dumps((availability or {}).get('missing_ingredients') or []),
        'substitution_candidates': json.dumps((availability or {}).get('substitution_candidates') or []),
        'substitution_summary': (availability or {}).get('substitution_summary'),
        'substitution_status': (availability or {}).get('substitution_status'),
        'analysis_status': _safe_text((availability or {}).get('analysis_status') or 'ready'),
    }


def _persist_owner_recipe_availability_rows(conn, rows):
    rows = [row for row in (rows or []) if row.get('owner') and row.get('recipe_source') and row.get('recipe_id')]
    if not rows:
        return
    _ensure_owner_recipe_availability_table(conn)
    params = [
        (
            row['owner'],
            row['recipe_source'],
            row['recipe_id'],
            row['kitchen_version'],
            row['can_make_exact'],
            row['can_make_with_subs'],
            row['matched_count'],
            row['missing_count'],
            row['ingredient_matches'],
            row['missing_ingredients'],
            row['substitution_candidates'],
            row['substitution_summary'],
            row['substitution_status'],
            row['analysis_status'],
        )
        for row in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO `{_OWNER_RECIPE_AVAILABILITY_TABLE}` (
                owner, recipe_source, recipe_id, kitchen_version, can_make_exact, can_make_with_subs,
                matched_count, missing_count, ingredient_matches, missing_ingredients,
                substitution_candidates, substitution_summary, substitution_status, analysis_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                kitchen_version = VALUES(kitchen_version),
                can_make_exact = VALUES(can_make_exact),
                can_make_with_subs = VALUES(can_make_with_subs),
                matched_count = VALUES(matched_count),
                missing_count = VALUES(missing_count),
                ingredient_matches = VALUES(ingredient_matches),
                missing_ingredients = VALUES(missing_ingredients),
                substitution_candidates = VALUES(substitution_candidates),
                substitution_summary = VALUES(substitution_summary),
                substitution_status = VALUES(substitution_status),
                analysis_status = VALUES(analysis_status),
                _updatedDate = NOW()
            """,
            params,
        )
    conn.commit()


def _delete_owner_recipe_availability(conn, owner, recipe_source, recipe_id):
    if not _safe_text(owner) or not _safe_text(recipe_source) or not _safe_text(recipe_id):
        return
    _ensure_owner_recipe_availability_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM `{_OWNER_RECIPE_AVAILABILITY_TABLE}` WHERE owner = %s AND recipe_source = %s AND recipe_id = %s",
            [_safe_text(owner), _safe_text(recipe_source), _safe_text(recipe_id)],
        )
    conn.commit()


def _serialize_saved_recipe_with_availability(conn, owner, row, kitchen_context=None, availability=None, request_id=None):
    kitchen_context = kitchen_context or _build_kitchen_match_context(conn, owner)
    recipe = _serialize_row(row)
    availability = availability or _compute_recipe_availability(
        recipe.get('ingredients') or [],
        kitchen_context,
        request_id=request_id,
        recipe=recipe,
    )
    recipe['availability'] = {
        'kitchen_version': availability['kitchen_version'],
        'can_make_exact': availability['can_make_exact'],
        'can_make_with_subs': availability['can_make_with_subs'],
        'matched_count': availability['matched_count'],
        'missing_count': availability['missing_count'],
    }
    recipe['ingredient_matches'] = availability['ingredient_matches']
    recipe['missing_ingredients'] = availability['missing_ingredients']
    recipe['substitution_candidates'] = availability['substitution_candidates']
    recipe['substitution_summary'] = availability['substitution_summary']
    recipe['substitution_status'] = availability['substitution_status']
    overlay_row = _build_owner_recipe_availability_record(owner, 'saved', recipe.get('id'), availability)
    return recipe, overlay_row


def _sha256(value):
    return hashlib.sha256(str(value or '').encode('utf-8')).hexdigest()


def _sha256_bytes(value):
    return hashlib.sha256(value or b'').hexdigest()


def _owned_images_bucket():
    return _safe_text(os.getenv('BUCKET_NAME'))


def _owned_recipe_image_prefix():
    bucket = _owned_images_bucket()
    if not bucket:
        return ''
    return f"https://{bucket}.s3.amazonaws.com/recipe-images/"


def _is_owned_recipe_image_url(url):
    prefix = _owned_recipe_image_prefix()
    text = _safe_text(url)
    return bool(prefix and text.startswith(prefix))


def _jobs_table():
    table_name = _safe_text(os.getenv('JOBS_TABLE'))
    if not table_name:
        raise RuntimeError('JOBS_TABLE is not set.')
    return boto3.resource('dynamodb').Table(table_name)


def _saved_recipe_batch_payload_key(owner, job_id):
    return f"recipe-images/{owner}/saved-recipes/jobs/{job_id}-submission.json"


def _saved_recipe_job_path(owner, job_id):
    return f"/saved-recipes/{owner}/jobs/{job_id}"


def _base_url_from_event(event):
    request_context = (event or {}).get('requestContext') or {}
    domain_name = _safe_text(request_context.get('domainName'))
    if not domain_name:
        return ''
    stage = _safe_text(request_context.get('stage'))
    if stage and stage != '$default':
        return f"https://{domain_name}/{stage}"
    return f"https://{domain_name}"


def _saved_recipe_job_url(owner, job_id, event=None):
    path = _saved_recipe_job_path(owner, job_id)
    base_url = _base_url_from_event(event)
    return f"{base_url}{path}" if base_url else path


def _serialize_saved_recipe_batch_job(job, event=None):
    job = dict(job or {})
    owner = _safe_text(job.get('owner'))
    job_id = _safe_text(job.get('job_id'))
    return {
        'job_id': job_id,
        'type': _safe_text(job.get('type')) or _SAVED_RECIPE_BATCH_JOB_TYPE,
        'status': _safe_text(job.get('status')).lower() or 'unknown',
        'owner': owner or None,
        'image_count': int(job.get('image_count') or 0),
        'count': int(job.get('result_count') or 0),
        'created_at': job.get('created_at'),
        'updated_at': job.get('updated_at'),
        'started_at': job.get('started_at'),
        'completed_at': job.get('completed_at'),
        'error': _safe_text(job.get('error')) or None,
        'poll_path': _saved_recipe_job_path(owner, job_id) if owner and job_id else None,
        'poll_url': _saved_recipe_job_url(owner, job_id, event=event) if owner and job_id else None,
    }


def _put_saved_recipe_batch_payload(owner, job_id, submission):
    bucket = _owned_images_bucket()
    if not bucket:
        raise RuntimeError('BUCKET_NAME is not set.')
    key = _saved_recipe_batch_payload_key(owner, job_id)
    boto3.client('s3').put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(submission, default=json_serial).encode('utf-8'),
        ContentType='application/json',
    )
    return key


def _load_saved_recipe_batch_payload(job):
    bucket = _owned_images_bucket()
    if not bucket:
        raise RuntimeError('BUCKET_NAME is not set.')
    payload_key = _safe_text((job or {}).get('payload_s3_key'))
    if not payload_key:
        raise ServiceError('Batch job payload is missing.', status_code=500)
    response = boto3.client('s3').get_object(Bucket=bucket, Key=payload_key)
    return json.loads(response['Body'].read().decode('utf-8'))


def _get_saved_recipe_batch_job(job_id):
    item = _jobs_table().get_item(Key={'job_id': job_id}).get('Item')
    if not item or _safe_text(item.get('type')) != _SAVED_RECIPE_BATCH_JOB_TYPE:
        return None
    return item


def _put_saved_recipe_batch_job(job):
    _jobs_table().put_item(Item=job)
    return job


def _update_saved_recipe_batch_job(job_id, **fields):
    job = _get_saved_recipe_batch_job(job_id)
    if not job:
        raise ServiceError('Saved recipe batch job not found.', status_code=404)
    job.update(fields)
    job['updated_at'] = _utc_now_iso()
    _put_saved_recipe_batch_job(job)
    return job


def _guess_image_extension(content_type, source_url):
    normalized = _safe_text(content_type).split(';', 1)[0].lower()
    content_type_map = {
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'image/png': 'png',
        'image/webp': 'webp',
        'image/gif': 'gif',
        'image/heic': 'jpg',
        'image/heif': 'jpg',
        'image/avif': 'avif',
    }
    if normalized in content_type_map:
        return content_type_map[normalized]
    suffix = Path(urlparse(_safe_text(source_url)).path or '').suffix.lower().lstrip('.')
    if suffix in {'jpg', 'jpeg', 'png', 'webp', 'gif', 'avif', 'heic', 'heif'}:
        return 'jpg' if suffix == 'jpeg' else suffix
    return 'jpg'


def _mirror_recipe_image(owner, recipe_id, source_image_url, request_id=None):
    bucket = _owned_images_bucket()
    source_url = _safe_text(source_image_url)
    if not bucket:
        raise RuntimeError('BUCKET_NAME is not set.')
    if not source_url:
        raise RuntimeError('No source image URL provided.')

    started = time.time()
    safe_owner = _safe_owner_token(owner)
    digest = _sha256(source_url)[:16]
    response = None
    try:
        response = requests.get(
            source_url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
        content_type = _safe_text(response.headers.get('Content-Type')).split(';', 1)[0].lower()
        if not content_type.startswith('image/'):
            raise RuntimeError(f'Unexpected image content type: {content_type or "unknown"}')
        image_bytes = response.content
        if not image_bytes:
            raise RuntimeError('Downloaded image was empty.')
        ext = _guess_image_extension(content_type, response.url or source_url)
        key = f"recipe-images/{safe_owner}/saved-recipes/{recipe_id}-{digest}.{ext}"
        boto3.client('s3').put_object(
            Bucket=bucket,
            Key=key,
            Body=image_bytes,
            ContentType=content_type or 'image/jpeg',
            CacheControl='public, max-age=31536000, immutable',
        )
        owned_url = f"https://{bucket}.s3.amazonaws.com/{key}"
        _log_event(
            request_id,
            'saved_recipe_image_mirrored',
            owner=owner,
            recipe_id=recipe_id,
            source_image_url=source_url,
            image_storage_key=key,
            latency_ms=int((time.time() - started) * 1000),
        )
        return {
            'image_url': owned_url,
            'image_urls': [owned_url],
            'source_image_url': source_url,
            'source_image_urls': [source_url],
            'image_storage_key': key,
        }
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_image_mirror_failed',
            owner=owner,
            recipe_id=recipe_id,
            source_image_url=source_url,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        raise
    finally:
        if response is not None:
            response.close()


def _put_owned_recipe_image(key, image_bytes, content_type):
    bucket = _owned_images_bucket()
    if not bucket:
        raise RuntimeError('BUCKET_NAME is not set.')
    boto3.client('s3').put_object(
        Bucket=bucket,
        Key=key,
        Body=image_bytes,
        ContentType=content_type or 'image/jpeg',
        CacheControl='public, max-age=31536000, immutable',
    )
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _prepare_saved_recipe_image_fields(owner, recipe_id, extraction, request_id=None):
    source_image_url = _safe_text((extraction or {}).get('image_url'))
    fallback_image_urls = _unique_texts((extraction or {}).get('image_urls') or [source_image_url])
    if not source_image_url:
        return {
            'image_url': None,
            'image_urls': [],
            'source_image_url': None,
            'source_image_urls': [],
            'image_storage_key': None,
        }
    try:
        return _mirror_recipe_image(owner, recipe_id, source_image_url, request_id=request_id)
    except Exception:
        return {
            'image_url': source_image_url,
            'image_urls': fallback_image_urls,
            'source_image_url': source_image_url,
            'source_image_urls': fallback_image_urls,
            'image_storage_key': None,
        }


def _openai_client():
    if not os.getenv('OPENAI_API_KEY'):
        raise ServiceError('OPENAI_API_KEY is not set.', status_code=500)
    return OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=_OPENAI_TIMEOUT_SECONDS,
        max_retries=_OPENAI_MAX_RETRIES,
    )


def _strip_code_fences(text):
    cleaned = _safe_text(text)
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return cleaned


def _best_effort_json_parse(text, fallback):
    cleaned = _safe_text(text)
    if not cleaned:
        return fallback
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def _normalize_image_payload_item(item):
    payload = item or {}
    image_url = _safe_text(payload.get('image_url') or payload.get('imageUrl'))
    image_base64 = _safe_text(payload.get('image_base64') or payload.get('imageBase64') or payload.get('image_data') or payload.get('imageData'))
    if bool(image_url) == bool(image_base64):
        raise ServiceError(
            'Each image must provide exactly one of image_url or image_base64.',
            status_code=400,
        )
    return {
        'image_url': image_url,
        'image_base64': image_base64,
        'image_content_type': _safe_text(payload.get('image_content_type') or payload.get('imageContentType')),
    }


def _build_saved_recipe_input(body):
    payload = body or {}
    url = _safe_text(payload.get('url'))
    content = _safe_text(payload.get('content') or payload.get('text') or payload.get('caption'))
    image_url = _safe_text(payload.get('image_url') or payload.get('imageUrl'))
    image_base64 = _safe_text(payload.get('image_base64') or payload.get('imageBase64') or payload.get('image_data') or payload.get('imageData'))
    raw_images = payload.get('images')
    images = []
    if raw_images is not None:
        if not isinstance(raw_images, list):
            raise ServiceError('images must be an array.', status_code=400)
        if not raw_images:
            raise ServiceError('Provide at least one image.', status_code=400)
        if len(raw_images) > _MAX_BATCH_IMAGE_COUNT:
            raise ServiceError(
                f'Provide no more than {_MAX_BATCH_IMAGE_COUNT} images per request.',
                status_code=400,
            )
        images = [_normalize_image_payload_item(item) for item in raw_images]
    provided_modes = [bool(url), bool(content), bool(image_url or image_base64), bool(images)]
    if sum(provided_modes) != 1:
        raise ServiceError(
            'Provide exactly one of url, content/text, image_url/image_base64, or images[].',
            status_code=400,
        )
    if url:
        return {
            'kind': 'url',
            'url': url,
        }
    if content:
        return {
            'kind': 'content',
            'content': content,
        }
    if images:
        return {
            'kind': 'images',
            'images': images,
        }
    return {
        'kind': 'image',
        'image_url': image_url,
        'image_base64': image_base64,
        'image_content_type': _safe_text(payload.get('image_content_type') or payload.get('imageContentType')),
    }


def _manual_recipe_url(kind, fingerprint):
    return f"app://saved-recipes/{kind}/{fingerprint}"


def _build_text_recipe_extraction(content):
    source_text = _safe_text(content)
    if not source_text:
        raise ServiceError('Provide recipe text.', status_code=400)
    fingerprint = _sha256(source_text)
    synthetic_url = _manual_recipe_url('text', fingerprint)
    return {
        'url': synthetic_url,
        'resolved_url': synthetic_url,
        'platform': 'text',
        'content': source_text,
        'caption': source_text,
        'title': '',
        'image_url': '',
        'image_urls': [],
        'source': 'user_text',
        'author_name': None,
        'caption_field': 'content',
        'warnings': [],
    }


def _decode_image_payload(image_base64, content_type_hint=None):
    raw = _safe_text(image_base64)
    if not raw:
        raise ServiceError('Provide image data.', status_code=400)
    content_type = _safe_text(content_type_hint) or 'image/jpeg'
    payload = raw
    if raw.startswith('data:'):
        header, _, data = raw.partition(',')
        if ';base64' not in header.lower() or not data:
            raise ServiceError('image_base64 must be valid base64 image data.', status_code=400)
        content_type = _safe_text(header[5:].split(';', 1)[0]) or content_type
        payload = data
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ServiceError(f'Invalid image_base64: {exc}', status_code=400)
    if not image_bytes:
        raise ServiceError('Decoded image was empty.', status_code=400)
    return image_bytes, content_type


def _normalize_image_content_type(content_type, source_url=''):
    normalized = _safe_text(content_type).split(';', 1)[0].lower()
    if normalized:
        if normalized == 'image/jpg':
            return 'image/jpeg'
        return normalized
    suffix = Path(urlparse(_safe_text(source_url)).path or '').suffix.lower().lstrip('.')
    suffix_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
        'gif': 'image/gif',
        'heic': 'image/heic',
        'heif': 'image/heif',
        'avif': 'image/avif',
    }
    return suffix_map.get(suffix, 'image/jpeg')


def _normalize_image_submission(submission):
    image_url = _safe_text((submission or {}).get('image_url'))
    image_base64 = _safe_text((submission or {}).get('image_base64'))
    content_type_hint = _safe_text((submission or {}).get('image_content_type'))
    if image_base64 or image_url.startswith('data:'):
        image_bytes, content_type = _decode_image_payload(image_base64 or image_url, content_type_hint=content_type_hint)
        digest = _sha256_bytes(image_bytes)
        return {
            'kind': 'bytes',
            'image_bytes': image_bytes,
            'content_type': _normalize_image_content_type(content_type),
            'fingerprint': digest,
        }
    if image_url:
        return {
            'kind': 'url',
            'image_url': image_url,
            'content_type': _normalize_image_content_type('', source_url=image_url),
            'fingerprint': _sha256(image_url),
        }
    raise ServiceError('Missing image_url or image_base64.', status_code=400)


def _load_remote_image(image_url):
    response = None
    try:
        response = requests.get(
            image_url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
        content_type = _normalize_image_content_type(response.headers.get('Content-Type'), source_url=response.url or image_url)
        image_bytes = response.content
        if not content_type.startswith('image/'):
            raise ServiceError(f'Unexpected image content type: {content_type or "unknown"}', status_code=400)
        if not image_bytes:
            raise ServiceError('Downloaded image was empty.', status_code=400)
        return image_bytes, content_type, response.url or image_url
    except requests.RequestException as exc:
        raise ServiceError(f'Could not fetch image URL: {exc}', status_code=502)
    finally:
        if response is not None:
            response.close()


def _convert_image_bytes_to_jpeg(image_bytes, content_type=None):
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
    except Exception as exc:
        raise ServiceError(f'Image conversion dependency is unavailable: {exc}', status_code=500)

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image)
            if normalized.mode != 'RGB':
                normalized = normalized.convert('RGB')
            output = BytesIO()
            normalized.save(output, format='JPEG', quality=92)
            return output.getvalue(), 'image/jpeg'
    except Exception as exc:
        normalized_content_type = _normalize_image_content_type(content_type)
        if normalized_content_type in _CONVERTIBLE_PHONE_CONTENT_TYPES:
            try:
                heif_image = pillow_heif.open_heif(image_bytes)
                normalized = heif_image.to_pillow()
                if normalized.mode != 'RGB':
                    normalized = normalized.convert('RGB')
                output = BytesIO()
                normalized.save(output, format='JPEG', quality=92)
                return output.getvalue(), 'image/jpeg'
            except Exception as heif_exc:
                raise ServiceError(f'Could not convert uploaded image: {heif_exc}', status_code=400)
        raise ServiceError(f'Could not convert uploaded image: {exc}', status_code=400)


def _prepare_image_for_vision(submission):
    image_input = _normalize_image_submission(submission)
    source_image_url = None
    if image_input['kind'] == 'url':
        image_bytes, content_type, source_image_url = _load_remote_image(image_input['image_url'])
    else:
        image_bytes = image_input['image_bytes']
        content_type = image_input['content_type']

    normalized_content_type = _normalize_image_content_type(content_type, source_url=source_image_url or '')
    if normalized_content_type not in _VISION_SUPPORTED_CONTENT_TYPES:
        image_bytes, normalized_content_type = _convert_image_bytes_to_jpeg(
            image_bytes,
            content_type=normalized_content_type,
        )

    prepared = {
        'kind': 'bytes',
        'image_bytes': image_bytes,
        'content_type': normalized_content_type,
        'fingerprint': _sha256_bytes(image_bytes),
        'source_image_url': source_image_url,
        'original_content_type': image_input.get('content_type'),
    }
    return prepared


def _recipe_image_transcription_prompt():
    return """Read the provided recipe image and return valid JSON only with this exact schema:
{
  "transcribed_text": "visible recipe text rewritten clearly as plain text",
  "title_hint": "best guess title if visible",
  "author_name": "author name if clearly visible, else empty string",
  "not_recipe": false
}

Rules:
- This may be a cookbook page, screenshot, social post, or photo of a written recipe.
- Extract only text that helps reconstruct the recipe.
- Keep ingredient lists and instructions readable, but do not invent missing recipe details.
- If the image does not contain enough recipe information, return:
  {"transcribed_text":"","title_hint":"","author_name":"","not_recipe":true}
- Do not return markdown fences or any extra prose."""


def _recipe_image_grouping_prompt():
    return """Group OCR fragments into recipes and return valid JSON only with this exact schema:
{
  "clusters": [
    {
      "image_indexes": [0, 1],
      "title_hint": "best guess recipe title"
    }
  ],
  "discard_indexes": [2]
}

Rules:
- Group screenshots/pages that belong to the same recipe into one cluster.
- Different recipes must be in different clusters.
- Preserve input order within each cluster.
- Use each valid image index at most once.
- Put unusable or non-recipe fragments in discard_indexes.
- If unsure whether adjacent pages are the same recipe, prefer grouping them together only when the text clearly continues the same recipe.
- Do not return markdown fences or extra prose."""


def _extract_recipe_fragment_from_image(prepared_image, request_id=None, image_index=None):
    client = _openai_client()
    image_reference = f"data:{prepared_image['content_type']};base64,{base64.b64encode(prepared_image['image_bytes']).decode('ascii')}"
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=_OPENAI_VISION_MODEL,
            temperature=0.1,
            messages=[
                {'role': 'system', 'content': _recipe_image_transcription_prompt()},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Extract the recipe from this image.'},
                        {'type': 'image_url', 'image_url': {'url': image_reference}},
                    ],
                },
            ],
        )
        text = _strip_code_fences(response.choices[0].message.content)
        payload = json.loads(text)
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_image_extract_failed',
            model=_OPENAI_VISION_MODEL,
            image_index=image_index,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        raise ServiceError(f'Image recipe extraction failed: {exc}', status_code=502)

    content = _safe_text(payload.get('transcribed_text'))
    if bool(payload.get('not_recipe')) or not content:
        raise ServiceError('Not enough recipe information.', status_code=422)

    synthetic_url = _manual_recipe_url('image', _sha256(content))
    author_name = _safe_text(payload.get('author_name')) or None
    _log_event(
        request_id,
        'saved_recipe_image_extract_success',
        model=_OPENAI_VISION_MODEL,
        image_index=image_index,
        latency_ms=int((time.time() - started) * 1000),
    )
    return {
        'image_index': image_index,
        'title_hint': _safe_text(payload.get('title_hint')),
        'author_name': author_name,
        'content': content,
        'fingerprint': prepared_image['fingerprint'],
        'source_image_url': prepared_image.get('source_image_url'),
        'prepared_image': prepared_image,
    }


def _build_image_recipe_extraction(content, title_hint='', author_name=None, source='openai_vision'):
    cleaned_content = _safe_text(content)
    synthetic_url = _manual_recipe_url('image', _sha256(cleaned_content))
    return {
        'url': synthetic_url,
        'resolved_url': synthetic_url,
        'platform': 'image',
        'content': cleaned_content,
        'caption': cleaned_content,
        'title': _safe_text(title_hint),
        'image_url': '',
        'image_urls': [],
        'source': source,
        'author_name': author_name or None,
        'caption_field': 'image_text',
        'warnings': [],
    }


def _extract_recipe_from_image(image_submission, request_id=None):
    prepared_image = _prepare_image_for_vision(image_submission)
    fragment = _extract_recipe_fragment_from_image(prepared_image, request_id=request_id, image_index=0)
    extraction = _build_image_recipe_extraction(
        fragment['content'],
        title_hint=fragment.get('title_hint'),
        author_name=fragment.get('author_name'),
    )
    return extraction, prepared_image


def _extract_recipe_from_social_preview_images(extraction, request_id=None):
    candidate_urls = _unique_texts([
        extraction.get('image_url'),
        *(extraction.get('image_urls') or []),
    ])
    last_exc = None
    for image_index, image_url in enumerate(candidate_urls[:3]):
        try:
            prepared_image = _prepare_image_for_vision({'image_url': image_url})
            fragment = _extract_recipe_fragment_from_image(
                prepared_image,
                request_id=request_id,
                image_index=f'preview-{image_index}',
            )
            _log_event(
                request_id,
                'social_preview_recipe_extract_success',
                platform=extraction.get('platform'),
                resolved_url=extraction.get('resolved_url'),
                preview_image_url=image_url,
                image_index=image_index,
            )
            return {
                'content': fragment.get('content'),
                'title_hint': fragment.get('title_hint'),
                'author_name': fragment.get('author_name'),
                'source_image_url': image_url,
            }
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise ServiceError('No usable social preview image was available.', status_code=422)


def _group_image_fragments(fragments, request_id=None):
    if not fragments:
        return []
    if len(fragments) == 1:
        return [{'title_hint': fragments[0].get('title_hint') or '', 'fragments': fragments}]

    summary = []
    for index, fragment in enumerate(fragments):
        summary.append({
            'image_index': index,
            'title_hint': fragment.get('title_hint') or '',
            'author_name': fragment.get('author_name') or '',
            'content_preview': _safe_text(fragment.get('content'))[:1800],
        })

    started = time.time()
    try:
        response = _openai_client().chat.completions.create(
            model=_OPENAI_VISION_MODEL,
            temperature=0.1,
            messages=[
                {'role': 'system', 'content': _recipe_image_grouping_prompt()},
                {'role': 'user', 'content': json.dumps(summary, ensure_ascii=True)},
            ],
        )
        payload = json.loads(_strip_code_fences(response.choices[0].message.content))
        raw_clusters = payload.get('clusters') or []
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_image_group_failed',
            model=_OPENAI_VISION_MODEL,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        return [{'title_hint': fragment.get('title_hint') or '', 'fragments': [fragment]} for fragment in fragments]

    used = set()
    clusters = []
    for raw_cluster in raw_clusters:
        indexes = []
        for value in (raw_cluster or {}).get('image_indexes') or []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(fragments) or idx in used:
                continue
            indexes.append(idx)
            used.add(idx)
        if not indexes:
            continue
        indexes = sorted(indexes)
        clusters.append({
            'title_hint': _safe_text((raw_cluster or {}).get('title_hint')),
            'fragments': [fragments[idx] for idx in indexes],
        })

    for idx, fragment in enumerate(fragments):
        if idx not in used:
            clusters.append({'title_hint': fragment.get('title_hint') or '', 'fragments': [fragment]})

    return clusters


def _merge_image_cluster(cluster):
    fragments = list((cluster or {}).get('fragments') or [])
    merged_content = '\n\n'.join(
        [_safe_text(fragment.get('content')) for fragment in fragments if _safe_text(fragment.get('content'))]
    ).strip()
    title_hint = _safe_text((cluster or {}).get('title_hint'))
    if not title_hint:
        for fragment in fragments:
            title_hint = _safe_text(fragment.get('title_hint'))
            if title_hint:
                break
    author_name = None
    for fragment in fragments:
        author_name = _safe_text(fragment.get('author_name')) or None
        if author_name:
            break
    prepared_images = [fragment.get('prepared_image') for fragment in fragments if fragment.get('prepared_image')]
    extraction = _build_image_recipe_extraction(
        merged_content,
        title_hint=title_hint,
        author_name=author_name,
        source='openai_vision_batch' if len(fragments) > 1 else 'openai_vision',
    )
    extraction['source_image_urls'] = _unique_texts([fragment.get('source_image_url') for fragment in fragments])
    return {
        'extraction': extraction,
        'prepared_images': prepared_images,
        'image_indexes': [fragment.get('image_index') for fragment in fragments],
    }


def _extract_recipe_clusters_from_images(image_submissions, request_id=None):
    partial_errors = []
    fragments = []
    for image_index, image_submission in enumerate(image_submissions or []):
        try:
            prepared_image = _prepare_image_for_vision(image_submission)
            fragment = _extract_recipe_fragment_from_image(
                prepared_image,
                request_id=request_id,
                image_index=image_index,
            )
            fragments.append(fragment)
        except ServiceError as exc:
            partial_errors.append({
                'image_index': image_index,
                'error': str(exc),
                'status_code': exc.status_code,
            })

    if not fragments:
        status_code = partial_errors[0]['status_code'] if partial_errors else 422
        message = partial_errors[0]['error'] if partial_errors else 'Not enough recipe information.'
        raise ServiceError(message, status_code=status_code, extra={'partial_errors': partial_errors})

    clusters = [_merge_image_cluster(cluster) for cluster in _group_image_fragments(fragments, request_id=request_id)]
    return clusters, partial_errors


def _build_recipe_image_prompt(recipe):
    title = _safe_text((recipe or {}).get('title')) or 'Recipe'
    ingredients = [str(item).strip() for item in ((recipe or {}).get('ingredients') or []) if _safe_text(item)]
    instructions = [str(item).strip() for item in ((recipe or {}).get('instructions') or []) if _safe_text(item)]
    ingredient_text = ', '.join(ingredients[:8])
    instruction_text = ' '.join(instructions[:2])
    title_lower = title.lower()

    if re.search(r'\b(coffee|tea|latte|espresso|cappuccino|smoothie|juice|soda|drink|beverage)\b', title_lower):
        vessel = 'Serve it in a simple white mug or clear glass on a clean neutral tabletop.'
    elif re.search(r'\b(soup|stew|curry|ramen|noodles|oatmeal|porridge|risotto)\b', title_lower):
        vessel = 'Serve it in a simple white ceramic bowl on a clean neutral tabletop.'
    else:
        vessel = 'Serve it on a simple white ceramic plate or bowl, whichever fits the dish naturally, on a clean neutral tabletop.'

    ingredient_clause = (
        f"Make the dish clearly match these key ingredients where natural: {ingredient_text}. "
        if ingredient_text else
        ''
    )
    instruction_clause = (
        f"The finished dish should visually fit this recipe description: {instruction_text}. "
        if instruction_text else
        ''
    )
    return (
        f"Create an ultra-realistic 4k-style food photograph of {title}. "
        "Photorealistic food photography only with realistic textures, natural colors, and believable plating. "
        "Do not make it look like an illustration, cartoon, painting, CGI render, or stylized concept art. "
        "Use a single plated serving, softly lit like a premium food magazine photo, with a slightly angled close composition. "
        f"{vessel} "
        f"{ingredient_clause}"
        f"{instruction_clause}"
        "Keep the dish centered and clearly visible as the main focus. "
        "Accurately match the named dish so it looks like the real food being described. "
        "Do not add text, labels, watermarks, extra hands, or unrelated side dishes."
    )


def _recipe_image_digest(recipe):
    payload = json.dumps({
        'title': (recipe or {}).get('title') or '',
        'ingredients': (recipe or {}).get('ingredients') or [],
        'instructions': (recipe or {}).get('instructions') or [],
        'notes': (recipe or {}).get('notes') or [],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _generate_recipe_image(recipe, request_id=None):
    started = time.time()
    try:
        response = _openai_client().images.generate(
            model=_OPENAI_RECIPE_IMAGE_MODEL,
            prompt=_build_recipe_image_prompt(recipe),
            size='1024x1024',
            quality='high',
            output_format='jpeg',
            output_compression=70,
            n=1,
        )
        if not response.data:
            return None
        b64 = getattr(response.data[0], 'b64_json', None)
        if not b64:
            return None
        return base64.b64decode(b64)
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_generated_image_failed',
            model=_OPENAI_RECIPE_IMAGE_MODEL,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        return None


def _upload_saved_recipe_source_image(owner, recipe_id, image_bytes, content_type, request_id=None, image_index=None):
    safe_owner = _safe_owner_token(owner)
    digest = _sha256_bytes(image_bytes)[:16]
    ext = _guess_image_extension(content_type, '')
    index_suffix = f"-{int(image_index) + 1}" if image_index is not None else ''
    key = f"recipe-images/{safe_owner}/saved-recipes/{recipe_id}-source{index_suffix}-{digest}.{ext}"
    url = _put_owned_recipe_image(key, image_bytes, content_type or 'image/jpeg')
    _log_event(
        request_id,
        'saved_recipe_source_image_uploaded',
        owner=owner,
        recipe_id=recipe_id,
        image_index=image_index,
        image_storage_key=key,
    )
    return url


def _upload_saved_recipe_source_images(owner, recipe_id, prepared_images, request_id=None):
    urls = []
    for image_index, prepared_image in enumerate(prepared_images or []):
        if not prepared_image:
            continue
        try:
            urls.append(
                _upload_saved_recipe_source_image(
                    owner,
                    recipe_id,
                    prepared_image['image_bytes'],
                    prepared_image['content_type'],
                    request_id=request_id,
                    image_index=image_index,
                )
            )
        except Exception as exc:
            _log_event(
                request_id,
                'saved_recipe_source_image_upload_failed',
                owner=owner,
                recipe_id=recipe_id,
                image_index=image_index,
                failure_reason=str(exc),
            )
    return _unique_texts(urls)


def _prepare_generated_saved_recipe_image_fields(owner, recipe_id, recipe, source_image_urls=None, request_id=None):
    image_bytes = _generate_recipe_image(recipe, request_id=request_id)
    source_urls = _unique_texts(source_image_urls or [])
    primary_source_url = source_urls[0] if source_urls else None
    if image_bytes:
        safe_owner = _safe_owner_token(owner)
        digest = _recipe_image_digest(recipe)
        key = f"recipe-images/{safe_owner}/saved-recipes/{recipe_id}-generated-{digest}.jpg"
        image_url = _put_owned_recipe_image(key, image_bytes, 'image/jpeg')
        _log_event(
            request_id,
            'saved_recipe_generated_image_uploaded',
            owner=owner,
            recipe_id=recipe_id,
            image_storage_key=key,
        )
        return {
            'image_url': image_url,
            'image_urls': [image_url],
            'source_image_url': primary_source_url,
            'source_image_urls': source_urls,
            'image_storage_key': key,
        }

    fallback_urls = [primary_source_url] if primary_source_url else []
    return {
        'image_url': primary_source_url,
        'image_urls': fallback_urls,
        'source_image_url': primary_source_url,
        'source_image_urls': source_urls,
        'image_storage_key': None,
    }


def _build_source_backed_saved_recipe_image_fields(source_image_urls=None):
    source_urls = _unique_texts(source_image_urls or [])
    primary_source_url = source_urls[0] if source_urls else None
    fallback_urls = [primary_source_url] if primary_source_url else []
    return {
        'image_url': primary_source_url,
        'image_urls': fallback_urls,
        'source_image_url': primary_source_url,
        'source_image_urls': source_urls,
        'image_storage_key': None,
    }


def _normalize_url(raw_url):
    text = _safe_text(raw_url)
    if not text:
        raise ServiceError('Provide a URL.', status_code=400)
    if '://' not in text:
        text = f'https://{text}'
    parsed = urlparse(text)
    if parsed.scheme not in ('http', 'https'):
        raise ServiceError('URL must use http or https.', status_code=400)
    if not parsed.netloc:
        raise ServiceError('URL is missing a hostname.', status_code=400)
    cleaned = parsed._replace(fragment='')
    normalized = urlunparse(cleaned)
    host = (cleaned.netloc or '').lower()
    path = (cleaned.path or '').lower()
    if host in _VALID_INSTAGRAM_HOSTS and '/reel/' not in path and '/reels/' not in path:
        raise ServiceError('Only public Instagram Reel URLs are supported.', status_code=400)
    return normalized


def _resolve_url(url):
    try:
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
        resolved = response.url or url
    except requests.RequestException as exc:
        raise ServiceError(f'Could not reach URL: {exc}', status_code=502)
    parsed = urlparse(resolved)
    return urlunparse(parsed._replace(fragment=''))


def _detect_platform(url):
    parsed = urlparse(url)
    host = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    if host in _VALID_TIKTOK_HOSTS or host.endswith('.tiktok.com'):
        return 'tiktok'
    if host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com'):
        if '/reel/' not in path and '/reels/' not in path:
            raise ServiceError('Only public Instagram Reel URLs are supported.', status_code=400)
        return 'instagram'
    return 'web_recipe'


def _absolute_url(base_url, value):
    text = _safe_text(value)
    if not text:
        return ''
    return urljoin(base_url, text)


def _normalize_image_values(value, base_url=None):
    items = []
    if isinstance(value, str):
        items.append(_absolute_url(base_url or '', value))
    elif isinstance(value, dict):
        candidate = value.get('url') or value.get('@id')
        if candidate:
            items.append(_absolute_url(base_url or '', candidate))
    elif isinstance(value, list):
        for item in value:
            items.extend(_normalize_image_values(item, base_url=base_url))
    return _unique_texts(items)


def _fetch_html_soup(url):
    try:
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ServiceError(f'HTML fetch failed: {exc}', status_code=502)
    return BeautifulSoup(response.text, 'html.parser')


def _meta_content(soup, *names):
    for name in names:
        tag = soup.find('meta', attrs={'property': name}) or soup.find('meta', attrs={'name': name})
        if tag:
            content = _safe_text(tag.get('content'))
            if content:
                return unescape(content)
    return ''


def _strip_social_prefixes(text):
    value = _safe_text(text)
    prefixes = ('TikTok', 'Watch', 'Replying to', 'Instagram')
    for prefix in prefixes:
        if value.startswith(prefix) and ':' in value:
            _, _, remainder = value.partition(':')
            if _safe_text(remainder):
                return _safe_text(remainder)
    return value


def _extract_visible_text_snippet(soup, max_lines=24):
    blocked = {
        'instagram', 'log in', 'log in to continue', 'sign up', 'meta', 'about', 'blog',
        'jobs', 'help', 'api', 'privacy', 'terms', 'locations', 'threads', 'contact uploading',
        'meta verified', 'english', 'cookie settings',
    }
    lines = []
    seen = set()
    for raw_line in soup.get_text('\n', strip=True).splitlines():
        line = _safe_text(raw_line)
        normalized = line.lower()
        if len(line) < 12 or normalized in blocked or normalized.startswith('©'):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return '\n'.join(lines).strip()


def _is_ytdlp_retryable_error(message):
    text = _safe_text(message).lower()
    return any(fragment in text for fragment in (
        'rate-limit reached',
        'login required',
        'requested content is not available',
        'http error 429',
        'timed out',
        'temporarily unavailable',
    ))


def _merge_recipe_source_chunks(chunks):
    merged = []
    seen = set()
    for label, text in chunks:
        cleaned = _safe_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        if label:
            merged.append(f"{label}:\n{cleaned}")
        else:
            merged.append(cleaned)
    return '\n\n'.join(merged).strip()


def _resolve_ytdlp_cookiefile():
    configured_path = _safe_text(os.getenv(_YTDLP_COOKIEFILE_ENV))
    if configured_path:
        if Path(configured_path).exists():
            return configured_path
        raise RuntimeError(f'{_YTDLP_COOKIEFILE_ENV} does not exist: {configured_path}')

    cookiefile_b64 = _safe_text(os.getenv(_YTDLP_COOKIEFILE_B64_ENV))
    if not cookiefile_b64:
        global _ytdlp_cookiefile_b64_cache, _ytdlp_cookiefile_cache_loaded
        if not _ytdlp_cookiefile_cache_loaded:
            _ytdlp_cookiefile_cache_loaded = True
            try:
                ssm = boto3.client('ssm', region_name='us-east-1')
                resp = ssm.get_parameter(Name=_YTDLP_SSM_PARAM_NAME, WithDecryption=True)
                _ytdlp_cookiefile_b64_cache = _safe_text(resp['Parameter']['Value'])
            except Exception:
                _ytdlp_cookiefile_b64_cache = None
        cookiefile_b64 = _ytdlp_cookiefile_b64_cache
    if not cookiefile_b64:
        return None
    try:
        cookie_bytes = base64.b64decode(cookiefile_b64.encode('ascii'), validate=True)
    except Exception as exc:
        raise RuntimeError(f'{_YTDLP_COOKIEFILE_B64_ENV} is not valid base64: {exc}')
    if not cookie_bytes.strip():
        raise RuntimeError(f'{_YTDLP_COOKIEFILE_B64_ENV} decoded to empty content')
    digest = _sha256_bytes(cookie_bytes)[:16]
    cookie_path = Path('/tmp') / f'yt-dlp-cookies-{digest}.txt'
    if not cookie_path.exists():
        cookie_path.write_bytes(cookie_bytes)
        os.chmod(cookie_path, 0o600)
    return str(cookie_path)


def _extract_tiktok_oembed(resolved_url):
    try:
        response = requests.get(
            'https://www.tiktok.com/oembed',
            params={'url': resolved_url},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise ServiceError(f'TikTok oEmbed failed: {exc}', status_code=502)
    caption = _safe_text(payload.get('title'))
    if not caption:
        raise ServiceError('TikTok oEmbed did not include caption text.', status_code=502)
    image_urls = _normalize_image_values(payload.get('thumbnail_url'), base_url=resolved_url)
    return {
        'content': caption,
        'title': caption,
        'image_url': image_urls[0] if image_urls else '',
        'image_urls': image_urls,
        'source': 'oembed',
        'author_name': _safe_text(payload.get('author_name')) or None,
        'caption_field': 'title',
        'warnings': [],
    }


def _extract_ytdlp_info(resolved_url, download=False, outtmpl=None):
    options = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': not download,
        'extract_flat': False,
        'socket_timeout': _REQUEST_TIMEOUT_SECONDS,
        'http_headers': {'User-Agent': _USER_AGENT},
        'noplaylist': True,
    }
    if outtmpl:
        options['outtmpl'] = outtmpl
    if download:
        options['format'] = 'bestaudio/best'
    cookiefile = _resolve_ytdlp_cookiefile()
    if cookiefile:
        options['cookiefile'] = cookiefile
    last_exc = None
    for attempt in range(1, _YTDLP_MAX_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(resolved_url, download=download)
        except Exception as exc:
            last_exc = exc
            if attempt >= _YTDLP_MAX_ATTEMPTS or not _is_ytdlp_retryable_error(exc):
                break
            time.sleep(_YTDLP_RETRY_SLEEP_SECONDS * attempt)
    raise ServiceError(f'yt-dlp extraction failed: {last_exc}', status_code=502)


def _extract_ytdlp(resolved_url):
    info = _extract_ytdlp_info(resolved_url)
    description = _safe_text(info.get('description'))
    title = _safe_text(info.get('title'))
    content = description or title
    if not content:
        raise ServiceError('yt-dlp did not return source text.', status_code=502)
    warnings = []
    caption_field = 'description'
    if not description:
        caption_field = 'title'
        warnings.append('yt-dlp did not provide a description, so the title was used instead.')
    image_urls = _unique_texts(
        _normalize_image_values(info.get('thumbnail'), base_url=resolved_url)
        + [item.get('url') for item in (info.get('thumbnails') or []) if isinstance(item, dict) and item.get('url')]
    )
    return {
        'content': content,
        'title': title or content,
        'image_url': image_urls[0] if image_urls else '',
        'image_urls': image_urls,
        'source': 'yt-dlp',
        'author_name': _safe_text(info.get('uploader') or info.get('creator') or info.get('channel')) or None,
        'caption_field': caption_field,
        'warnings': warnings,
    }


def _extract_tiktok_scrapecreators(resolved_url):
    """TikTok provider that handles PHOTO/slideshow posts (aweme_type 150) as well
    as videos. Returns the caption (desc) plus EVERY carousel slide image URL, so
    the downstream OCR fallback can read recipes that live in the slide images."""
    api_key = _safe_text(os.getenv(_SCRAPECREATORS_API_KEY_ENV))
    if not api_key:
        raise ServiceError(f'{_SCRAPECREATORS_API_KEY_ENV} is not set.', status_code=502)
    try:
        response = requests.get(
            _SCRAPECREATORS_VIDEO_ENDPOINT,
            params={'url': resolved_url},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={'x-api-key': api_key, 'Accept': 'application/json'},
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise ServiceError(f'ScrapeCreators TikTok fetch failed: {exc}', status_code=502)

    detail = payload.get('aweme_detail') if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        detail = payload if isinstance(payload, dict) else {}

    caption = _safe_text(detail.get('desc'))

    # Carousel slide image URLs (present on photo/slideshow posts).
    image_urls = []
    image_post_info = detail.get('image_post_info') or {}
    for img in (image_post_info.get('images') or []):
        if not isinstance(img, dict):
            continue
        url_list = (img.get('display_image') or {}).get('url_list') or []
        if url_list:
            image_urls.append(url_list[0])
    image_urls = _unique_texts(image_urls)
    # For plain videos there's no carousel — fall back to the cover thumbnail.
    if not image_urls:
        cover = (detail.get('video') or {}).get('cover') or {}
        image_urls = _unique_texts((cover.get('url_list') or [])[:1])

    if not caption and not image_urls:
        raise ServiceError('ScrapeCreators returned no caption or images.', status_code=502)

    author = detail.get('author') or {}
    author_name = _safe_text(author.get('nickname') or author.get('unique_id')) or None
    is_slideshow = bool(image_post_info.get('images'))
    # The pipeline rejects empty content; for a caption-less slideshow use a neutral
    # placeholder so the request proceeds to the slide-OCR fallback.
    content = caption or ('TikTok photo slideshow (recipe is in the slide images).' if is_slideshow else '')
    if not content:
        raise ServiceError('ScrapeCreators returned no usable text.', status_code=502)
    return {
        'content': content,
        'title': (caption[:80] if caption else ''),
        'image_url': image_urls[0] if image_urls else '',
        'image_urls': image_urls,
        'source': 'scrapecreators',
        'author_name': author_name,
        'caption_field': 'desc',
        'warnings': [],
    }


def _extract_recipe_text_via_rekognition(image_urls, request_id=None):
    """Fast/cheap OCR over carousel slide images via AWS Rekognition DetectText.
    Used for TikTok photo slideshows whose recipe lives in the slide text (not the
    caption). ~10x faster than the VLM preview path. Returns concatenated per-slide
    text in carousel order; HEIC slides are converted to JPEG first."""
    if not image_urls:
        return ''
    client = boto3.client('rekognition')
    slides = []
    for idx, url in enumerate(image_urls[:_SLIDESHOW_OCR_MAX_SLIDES]):
        try:
            image_bytes, content_type, _ = _load_remote_image(url)
            if _normalize_image_content_type(content_type) not in {'image/jpeg', 'image/png'}:
                image_bytes, _ = _convert_image_bytes_to_jpeg(image_bytes, content_type=content_type)
            resp = client.detect_text(Image={'Bytes': image_bytes})
            lines = [
                _safe_text(d.get('DetectedText'))
                for d in (resp.get('TextDetections') or [])
                if d.get('Type') == 'LINE' and _safe_text(d.get('DetectedText'))
            ]
            text = '\n'.join(lines).strip()
            if text:
                slides.append(f'[Slide {idx + 1}]\n{text}')
        except Exception as exc:
            _log_event(request_id, 'slide_ocr_skip', slide_index=idx, error=str(exc))
            continue
    return '\n\n'.join(slides)


def _extract_first_apify_post(items, fallback_url=''):
    if not isinstance(items, list):
        raise RuntimeError(f'Unexpected Apify response type: {type(items).__name__}')

    posts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        latest_posts = item.get('latestPosts')
        if isinstance(latest_posts, list):
            posts.extend([post for post in latest_posts if isinstance(post, dict)])
            continue
        if item.get('shortCode') or item.get('shortcode') or item.get('url') or item.get('postUrl'):
            posts.append(item)
    if not posts:
        posts = [item for item in items if isinstance(item, dict)]
    if not posts:
        raise RuntimeError('Apify returned no Instagram posts.')

    post = posts[0]
    shortcode = _safe_text(post.get('shortCode') or post.get('shortcode'))
    url = _safe_text(post.get('url') or post.get('postUrl') or post.get('post_url'))
    if not url and shortcode:
        url = f'https://www.instagram.com/p/{shortcode}/'
    caption = _safe_text(post.get('caption') or post.get('text') or post.get('postText'))
    image_url = _safe_text(
        post.get('displayUrl')
        or post.get('imageUrl')
        or post.get('image_url')
        or post.get('display_url')
        or post.get('thumbnailUrl')
    )
    author_name = _safe_text(
        post.get('ownerUsername')
        or post.get('owner_username')
        or post.get('username')
    )
    return {
        'url': url or fallback_url,
        'caption': caption,
        'image_url': image_url,
        'image_urls': _unique_texts([image_url]),
        'author_name': author_name,
    }


def _extract_instagram_apify(resolved_url, request_id=None):
    token = _safe_text(os.getenv(_APIFY_TOKEN_ENV))
    if not token:
        raise RuntimeError(f'{_APIFY_TOKEN_ENV} is not set.')

    started = time.time()
    endpoint = f'https://api.apify.com/v2/acts/{_APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items'
    payload = {
        'directUrls': [resolved_url],
        'resultsType': 'posts',
        'resultsLimit': 1,
    }
    try:
        response = requests.post(
            endpoint,
            params={'token': token},
            json=payload,
            timeout=_APIFY_RUN_TIMEOUT_SECONDS,
            headers={'User-Agent': _USER_AGENT, 'Accept': 'application/json'},
        )
        response.raise_for_status()
        post = _extract_first_apify_post(response.json(), fallback_url=resolved_url)
        content = _safe_text(post.get('caption'))
        if not content:
            raise RuntimeError('Apify response did not include caption text.')
        _log_event(
            request_id,
            'instagram_apify_extract_success',
            resolved_url=resolved_url,
            actor=_APIFY_INSTAGRAM_ACTOR,
            latency_ms=int((time.time() - started) * 1000),
        )
        return {
            'content': content,
            'title': content[:80],
            'image_url': _safe_text(post.get('image_url')),
            'image_urls': _unique_texts(post.get('image_urls') or []),
            'source': 'apify',
            'author_name': post.get('author_name'),
            'caption_field': 'caption',
            'warnings': [],
        }
    except Exception as exc:
        _log_event(
            request_id,
            'instagram_apify_extract_failed',
            resolved_url=resolved_url,
            actor=_APIFY_INSTAGRAM_ACTOR,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        raise RuntimeError(f'Apify Instagram extraction failed: {exc}')


def _extract_html_metadata(resolved_url):
    soup = _fetch_html_soup(resolved_url)
    title = _safe_text(_meta_content(soup, 'og:title', 'twitter:title'))
    if not title and soup.title:
        title = _safe_text(soup.title.get_text())
    description = _safe_text(_meta_content(soup, 'og:description', 'description', 'twitter:description'))
    content = _strip_social_prefixes(description or title)
    host = (urlparse(resolved_url).netloc or '').lower()
    if host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com'):
        visible_text = _extract_visible_text_snippet(soup)
        content = _merge_recipe_source_chunks([
            ('Meta description', _strip_social_prefixes(description)),
            ('Page text', visible_text),
            ('Title', title),
        ]) or content
    if not content:
        raise ServiceError('HTML metadata did not include source text.', status_code=502)
    image_urls = _unique_texts([
        _absolute_url(resolved_url, _meta_content(soup, 'og:image')),
        _absolute_url(resolved_url, _meta_content(soup, 'twitter:image')),
    ])
    return {
        'content': content,
        'title': title or content,
        'image_url': image_urls[0] if image_urls else '',
        'image_urls': image_urls,
        'source': 'html',
        'author_name': _safe_text(_meta_content(soup, 'article:author', 'author')) or None,
        'caption_field': 'description' if description else 'title',
        'warnings': [],
    }


def _parse_json_ld_blocks(soup):
    blocks = []
    for tag in soup.find_all('script', attrs={'type': re.compile(r'application/ld\+json', re.I)}):
        raw = tag.string or tag.get_text() or ''
        raw = raw.strip()
        if not raw:
            continue
        try:
            blocks.append(json.loads(unescape(raw)))
        except json.JSONDecodeError:
            continue
    return blocks


def _iter_json_nodes(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            for item in _iter_json_nodes(nested):
                yield item
    elif isinstance(value, list):
        for item in value:
            for nested in _iter_json_nodes(item):
                yield nested


def _node_types(node):
    raw = node.get('@type')
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [_safe_text(item) for item in raw if _safe_text(item)]
    return []


def _parse_author_name(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _safe_text(value.get('name'))
    if isinstance(value, list):
        return ', '.join([_parse_author_name(item) for item in value if _parse_author_name(item)])
    return ''


def _parse_instructions(value):
    steps = []
    if isinstance(value, str):
        return [_safe_text(value)] if _safe_text(value) else []
    if isinstance(value, list):
        for item in value:
            steps.extend(_parse_instructions(item))
        return steps
    if isinstance(value, dict):
        if _safe_text(value.get('@type')) == 'HowToSection':
            name = _safe_text(value.get('name'))
            nested_steps = _parse_instructions(value.get('itemListElement'))
            if name and nested_steps:
                return [name] + nested_steps
            return nested_steps
        text = _safe_text(value.get('text'))
        if text:
            return [text]
        name = _safe_text(value.get('name'))
        if name and _safe_text(value.get('@type')).startswith('HowTo'):
            return [name]
        return _parse_instructions(value.get('itemListElement'))
    return []


def _extract_json_ld_recipe(resolved_url):
    soup = _fetch_html_soup(resolved_url)
    recipe_node = None
    for block in _parse_json_ld_blocks(soup):
        for node in _iter_json_nodes(block):
            if isinstance(node, dict) and 'Recipe' in _node_types(node):
                recipe_node = node
                break
        if recipe_node:
            break
    if not recipe_node:
        raise ServiceError('No Recipe JSON-LD was found on the page.', status_code=502)

    title = _safe_text(recipe_node.get('name'))
    description = _safe_text(recipe_node.get('description'))
    author_name = _parse_author_name(recipe_node.get('author')) or None
    ingredients = recipe_node.get('recipeIngredient') if isinstance(recipe_node.get('recipeIngredient'), list) else []
    instructions = _parse_instructions(recipe_node.get('recipeInstructions'))
    image_urls = _normalize_image_values(recipe_node.get('image'), base_url=resolved_url)

    lines = []
    if title:
        lines.extend(['Title:', title, ''])
    if description:
        lines.extend(['Description:', description, ''])
    if ingredients:
        lines.append('Ingredients:')
        lines.extend([f"- {_safe_text(item)}" for item in ingredients if _safe_text(item)])
        lines.append('')
    if instructions:
        lines.append('Instructions:')
        lines.extend([f"{index}. {step}" for index, step in enumerate(instructions, start=1) if step])
        lines.append('')
    notes = []
    if author_name:
        notes.append(f'Author: {author_name}')
    for label, key in [('Yield', 'recipeYield'), ('Prep time', 'prepTime'), ('Cook time', 'cookTime'), ('Total time', 'totalTime')]:
        value = _safe_text(recipe_node.get(key))
        if value:
            notes.append(f'{label}: {value}')
    notes.append(f'Source URL: {resolved_url}')
    if notes:
        lines.append('Notes:')
        lines.extend([f'- {note}' for note in notes])
    content = '\n'.join(lines).strip()
    if not content:
        raise ServiceError('Recipe page did not yield usable structured text.', status_code=502)

    return {
        'content': content,
        'title': title or description or 'Recipe',
        'image_url': image_urls[0] if image_urls else '',
        'image_urls': image_urls,
        'source': 'json-ld',
        'author_name': author_name,
        'caption_field': '',
        'warnings': [],
    }


def _run_provider_pipeline(normalized_url, resolved_url, platform, providers, request_id=None):
    attempts = []
    warnings = []
    started = time.time()
    for provider_name, provider in providers:
        provider_started = time.time()
        try:
            result = provider(resolved_url)
            merged_warnings = warnings + list(result.get('warnings') or [])
            response = {
                'url': normalized_url,
                'resolved_url': resolved_url,
                'platform': platform,
                'content': _safe_text(result.get('content')),
                'caption': _safe_text(result.get('content')),
                'title': _safe_text(result.get('title')),
                'image_url': _safe_text(result.get('image_url')),
                'image_urls': _unique_texts(result.get('image_urls') or []),
                'source': result.get('source') or provider_name,
                'author_name': result.get('author_name'),
                'caption_field': result.get('caption_field') or '',
                'warnings': merged_warnings,
            }
            if not response['content']:
                raise ServiceError(f'{provider_name} returned empty content.', status_code=502)
            if not response['image_url'] and response['image_urls']:
                response['image_url'] = response['image_urls'][0]
            _log_event(
                request_id,
                'extract_success',
                platform=platform,
                resolved_url=resolved_url,
                extraction_source=response['source'],
                latency_ms=int((time.time() - started) * 1000),
                attempts=attempts + [{
                    'source': provider_name,
                    'ok': True,
                    'latency_ms': int((time.time() - provider_started) * 1000),
                }],
            )
            return response
        except Exception as exc:
            attempts.append({
                'source': provider_name,
                'ok': False,
                'latency_ms': int((time.time() - provider_started) * 1000),
                'error': str(exc),
            })
            warnings.append(f'{provider_name} failed')

    _log_event(
        request_id,
        'extract_failed',
        platform=platform,
        resolved_url=resolved_url,
        latency_ms=int((time.time() - started) * 1000),
        failure_reason=' | '.join([attempt.get('error') or attempt.get('source') for attempt in attempts]),
        attempts=attempts,
    )
    _report_backend_error('extract_url', code='all_providers_failed',
                          error=f"{platform}: " + ' | '.join([attempt.get('error') or attempt.get('source') for attempt in attempts]))
    raise ServiceError(f'Could not extract content from this {platform} URL.', status_code=502, extra={'warnings': warnings})


def _extract_content(url, request_id=None):
    normalized_url = _normalize_url(url)
    resolved_url = _resolve_url(normalized_url)
    platform = _detect_platform(resolved_url)
    providers = {
        'tiktok': [
            ('yt-dlp', _extract_ytdlp),
            ('oembed', _extract_tiktok_oembed),
            # ScrapeCreators handles photo/slideshow posts (and videos) — the only
            # provider that doesn't 502 on /photo/ URLs. After the cheaper providers
            # so existing video behavior is unchanged.
            ('scrapecreators', _extract_tiktok_scrapecreators),
            ('html', _extract_html_metadata),
        ],
        'instagram': [
            ('yt-dlp', _extract_ytdlp),
            ('apify', _extract_instagram_apify),
            ('html', _extract_html_metadata),
        ],
        'web_recipe': [
            ('json-ld', _extract_json_ld_recipe),
            ('html', _extract_html_metadata),
        ],
    }
    return _run_provider_pipeline(normalized_url, resolved_url, platform, providers[platform], request_id=request_id)


def _recipe_is_incomplete(recipe_text):
    return _safe_text(recipe_text) == 'Not enough recipe information.'


def _audio_fallback_supported(platform):
    return platform in {'tiktok', 'instagram'}


def _choose_downloaded_media_file(temp_dir, info):
    requested_downloads = info.get('requested_downloads')
    if isinstance(requested_downloads, list):
        for item in requested_downloads:
            if isinstance(item, dict):
                filepath = _safe_text(item.get('filepath'))
                if filepath and Path(filepath).exists():
                    return Path(filepath)

    ext = _safe_text(info.get('ext'))
    media_id = _safe_text(info.get('id'))
    if media_id and ext:
        candidate = temp_dir / f'{media_id}.{ext}'
        if candidate.exists():
            return candidate

    files = sorted([path for path in temp_dir.iterdir() if path.is_file()])
    if files:
        return files[0]

    raise RuntimeError('Audio download finished, but no media file was found.')


def _transcribe_audio_from_url(resolved_url, request_id=None, source_context=None):
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set.')

    started = time.time()
    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=_OPENAI_TIMEOUT_SECONDS,
            max_retries=_OPENAI_MAX_RETRIES,
        )
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            info = _extract_ytdlp_info(
                resolved_url,
                download=True,
                outtmpl=str(temp_dir / '%(id)s.%(ext)s'),
            )
            media_path = _choose_downloaded_media_file(temp_dir, info)
            with media_path.open('rb') as media_file:
                response = client.audio.transcriptions.create(
                    model=_DEFAULT_TRANSCRIPTION_MODEL,
                    file=media_file,
                )
        transcript = _safe_text(getattr(response, 'text', response))
        if not transcript:
            raise RuntimeError('Audio transcription returned empty text.')
        _log_event(
            request_id,
            'audio_transcription_success',
            platform=(source_context or {}).get('platform'),
            resolved_url=resolved_url,
            transcription_model=_DEFAULT_TRANSCRIPTION_MODEL,
            latency_ms=int((time.time() - started) * 1000),
        )
        return {
            'audio_transcript': transcript,
            'transcript_source': 'openai_audio',
            'transcription_model': _DEFAULT_TRANSCRIPTION_MODEL,
        }
    except Exception as exc:
        _log_event(
            request_id,
            'audio_transcription_failed',
            platform=(source_context or {}).get('platform'),
            resolved_url=resolved_url,
            transcription_model=_DEFAULT_TRANSCRIPTION_MODEL,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        raise RuntimeError(f'Audio transcription failed: {exc}')


def _merge_text_and_transcript(content, transcript):
    parts = []
    if _safe_text(content):
        parts.append(f'Caption or page text:\n{_safe_text(content)}')
    if _safe_text(transcript):
        parts.append(f'Audio transcript:\n{_safe_text(transcript)}')
    return '\n\n'.join(parts).strip()


def _recipe_json_prompt():
    return """Convert extracted source text into structured recipe JSON.
Return valid JSON only with this exact schema:
{
  "title": "Recipe title",
  "ingredients": ["ingredient 1", "ingredient 2"],
  "instructions": ["step 1", "step 2"],
  "notes": ["optional note 1"],
  "not_enough": false
}

Rules:
- Extract only recipe-relevant information.
- If there is not enough information to produce a useful recipe, return:
  {"title":"","ingredients":[],"instructions":[],"notes":[],"not_enough":true}
- Keep ingredients and instructions concise but useful.
- Do not return markdown fences or prose outside the JSON object."""


def _format_recipe_text(recipe):
    if recipe.get('not_enough'):
        return 'Not enough recipe information.'
    lines = []
    title = _safe_text(recipe.get('title'))
    if title:
        lines.extend(['Title:', title, ''])
    lines.append('Ingredients:')
    ingredients = _clean_string_list(recipe.get('ingredients') or [])
    if ingredients:
        lines.extend([f'- {item}' for item in ingredients])
    lines.append('')
    lines.append('Instructions:')
    instructions = _clean_string_list(recipe.get('instructions') or [])
    if instructions:
        lines.extend([f'{index}. {step}' for index, step in enumerate(instructions, start=1)])
    lines.append('')
    notes = _clean_string_list(recipe.get('notes') or [])
    lines.append('Notes:')
    if notes:
        lines.extend([f'- {note}' for note in notes])
    return '\n'.join(lines).strip()


def _refine_recipe_structured(content, request_id=None, source_context=None):
    source_text = _safe_text(content)
    if not source_text:
        raise ServiceError('Provide source text to refine.', status_code=400)

    started = time.time()
    model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {'role': 'system', 'content': _recipe_json_prompt()},
                {'role': 'user', 'content': source_text},
            ],
        )
        text = _strip_code_fences(response.choices[0].message.content)
        payload = json.loads(text)
    except Exception as exc:
        _log_event(
            request_id,
            'recipe_refine_failed',
            model=model,
            platform=(source_context or {}).get('platform'),
            resolved_url=(source_context or {}).get('resolved_url'),
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        raise ServiceError(f'OpenAI refinement failed: {exc}', status_code=502)

    recipe = {
        'title': _safe_text(payload.get('title'))[:255],
        'ingredients': _clean_string_list(payload.get('ingredients') or []),
        'instructions': _clean_string_list(payload.get('instructions') or []),
        'notes': _clean_string_list(payload.get('notes') or []),
        'not_enough': bool(payload.get('not_enough')),
    }
    if not recipe['not_enough'] and not recipe['title'] and not recipe['ingredients'] and not recipe['instructions']:
        recipe['not_enough'] = True

    _log_event(
        request_id,
        'recipe_refine_success',
        model=model,
        platform=(source_context or {}).get('platform'),
        resolved_url=(source_context or {}).get('resolved_url'),
        latency_ms=int((time.time() - started) * 1000),
        not_enough=recipe['not_enough'],
    )
    return recipe, model


def _recipe_response_from_content(content, request_id=None, source_context=None):
    recipe, model = _refine_recipe_structured(content, request_id=request_id, source_context=source_context)
    return {
        'recipe': _format_recipe_text(recipe),
        'model': model,
    }, recipe


def _append_extraction_source(existing_source, new_source):
    parts = [part.strip() for part in _safe_text(existing_source).split('+') if part.strip()]
    new_value = _safe_text(new_source)
    if new_value and new_value not in parts:
        parts.append(new_value)
    return '+'.join(parts)


def _overlay_extraction_fallback(extraction, fallback_result, merged_content=None):
    image_urls = _unique_texts(list(extraction.get('image_urls') or []) + list(fallback_result.get('image_urls') or []))
    extraction['content'] = _safe_text(merged_content or fallback_result.get('content') or extraction.get('content'))
    extraction['caption'] = extraction['content']
    extraction['source'] = _append_extraction_source(extraction.get('source'), fallback_result.get('source'))
    extraction['image_urls'] = image_urls
    extraction['image_url'] = _safe_text(extraction.get('image_url')) or (image_urls[0] if image_urls else '')
    if _safe_text(fallback_result.get('author_name')) and not _safe_text(extraction.get('author_name')):
        extraction['author_name'] = fallback_result.get('author_name')
    if _safe_text(fallback_result.get('caption_field')):
        extraction['caption_field'] = fallback_result.get('caption_field')


def _analyze_extraction(extraction, request_id=None):
    response, structured_recipe = _recipe_response_from_content(
        extraction['content'],
        request_id=request_id,
        source_context=extraction,
    )
    result = {
        **response,
        'audio_transcript': '',
        'transcript_source': '',
        'transcription_model': '',
        'recipe_source_used': 'content',
    }
    if not _recipe_is_incomplete(response.get('recipe')):
        return result, structured_recipe

    warnings = list(extraction.get('warnings') or [])
    preview_fallback = None
    transcript_result = None
    if extraction.get('platform') == 'instagram':
        try:
            apify_fallback = _extract_instagram_apify(extraction['resolved_url'], request_id=request_id)
            merged_apify_content = _merge_recipe_source_chunks([
                ('Caption or description', extraction.get('content')),
                ('Apify caption', apify_fallback.get('content')),
            ])
            _overlay_extraction_fallback(extraction, apify_fallback, merged_content=merged_apify_content)
            refined_with_apify, structured_with_apify = _recipe_response_from_content(
                extraction['content'],
                request_id=request_id,
                source_context=extraction,
            )
            warnings.append(
                'Caption text was insufficient, so Apify Instagram extraction was used as a fallback.'
            )
            result.update(refined_with_apify)
            result['recipe_source_used'] = 'content+apify'
            result['warnings'] = list(warnings)
            structured_recipe = structured_with_apify
            if not _recipe_is_incomplete(refined_with_apify.get('recipe')):
                return result, structured_with_apify
        except Exception as exc:
            warnings.append(str(exc))

    if not _audio_fallback_supported(extraction.get('platform')):
        result['warnings'] = warnings
        return result, structured_recipe

    # Fast slide-OCR fallback (Rekognition) — for photo slideshows whose recipe
    # lives in the slide images, not the caption. Runs BEFORE the slower VLM
    # preview path below; only when we actually have carousel image URLs.
    if extraction.get('image_urls'):
        try:
            slide_ocr_text = _extract_recipe_text_via_rekognition(
                extraction.get('image_urls'), request_id=request_id
            )
            if slide_ocr_text:
                merged_ocr_content = _merge_recipe_source_chunks([
                    ('Caption or description', extraction.get('content')),
                    ('Slide OCR', slide_ocr_text),
                ])
                refined_with_ocr, structured_with_ocr = _recipe_response_from_content(
                    merged_ocr_content,
                    request_id=request_id,
                    source_context={**extraction, 'content': merged_ocr_content},
                )
                if not _recipe_is_incomplete(refined_with_ocr.get('recipe')):
                    result.update(refined_with_ocr)
                    result['recipe_source_used'] = 'content+slide_ocr'
                    result['warnings'] = warnings + [
                        'Caption text was insufficient, so slide-image OCR was used as a fallback.'
                    ]
                    return result, structured_with_ocr
        except Exception as exc:
            warnings.append(str(exc))

    try:
        preview_fallback = _extract_recipe_from_social_preview_images(extraction, request_id=request_id)
        merged_preview_content = _merge_recipe_source_chunks([
            ('Caption or description', extraction.get('content')),
            ('Preview image OCR', preview_fallback.get('content')),
        ])
        refined_with_preview, structured_with_preview = _recipe_response_from_content(
            merged_preview_content,
            request_id=request_id,
            source_context={**extraction, 'content': merged_preview_content},
        )
        if not _recipe_is_incomplete(refined_with_preview.get('recipe')):
            result.update(refined_with_preview)
            result['recipe_source_used'] = 'content+image_ocr'
            result['warnings'] = warnings + [
                'Caption text was insufficient, so preview-image OCR was used as a fallback.'
            ]
            return result, structured_with_preview
    except Exception as exc:
        warnings.append(str(exc))

    try:
        transcript_result = _transcribe_audio_from_url(
            extraction['resolved_url'],
            request_id=request_id,
            source_context=extraction,
        )
        merged_content = _merge_text_and_transcript(extraction['content'], transcript_result['audio_transcript'])
        result.update({
            **transcript_result,
            'recipe_source_used': 'content+audio',
        })
        result['warnings'] = warnings + [
            'Caption text was insufficient, so audio transcription was used as a fallback.'
        ]
    except Exception as exc:
        result['warnings'] = warnings + [str(exc)]
        return result, structured_recipe

    try:
        refined_with_audio, structured_with_audio = _recipe_response_from_content(
            merged_content,
            request_id=request_id,
            source_context={**extraction, 'content': merged_content},
        )
        result.update(refined_with_audio)
        if not _recipe_is_incomplete(refined_with_audio.get('recipe')):
            return result, structured_with_audio
    except Exception as exc:
        result['warnings'] = list(result.get('warnings') or []) + [str(exc)]
        return result, structured_recipe

    if preview_fallback and transcript_result:
        try:
            merged_all_content = _merge_recipe_source_chunks([
                ('Caption or description', extraction.get('content')),
                ('Preview image OCR', preview_fallback.get('content')),
                ('Audio transcript', transcript_result.get('audio_transcript')),
            ])
            refined_with_all, structured_with_all = _recipe_response_from_content(
                merged_all_content,
                request_id=request_id,
                source_context={**extraction, 'content': merged_all_content},
            )
            result.update(refined_with_all)
            result['recipe_source_used'] = 'content+image_ocr+audio'
            result['warnings'] = warnings + [
                'Caption text was insufficient, so preview-image OCR and audio transcription were used as fallbacks.'
            ]
            return result, structured_with_all
        except Exception as exc:
            result['warnings'] = list(result.get('warnings') or []) + [str(exc)]
            return result, structured_recipe
    return result, structured_with_audio


def _update_saved_recipe_image_fields(conn, owner, recipe_id, image_fields):
    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""UPDATE `{table}`
                SET image_url = %s,
                    image_urls = %s,
                    source_image_url = %s,
                    source_image_urls = %s,
                    image_storage_key = %s
                WHERE _id = %s
                LIMIT 1""",
            [
                image_fields.get('image_url') or None,
                json.dumps(image_fields.get('image_urls') or []),
                image_fields.get('source_image_url') or None,
                json.dumps(image_fields.get('source_image_urls') or []),
                image_fields.get('image_storage_key') or None,
                recipe_id,
            ],
        )
    conn.commit()


def _invoke_saved_recipe_image_generation_async(owner, recipe_id, request_id=None):
    function_name = _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not function_name:
        _log_event(
            request_id,
            'saved_recipe_async_image_invoke_skipped',
            owner=owner,
            recipe_id=recipe_id,
            failure_reason='AWS_LAMBDA_FUNCTION_NAME is not available',
        )
        return False
    try:
        boto3.client('lambda').invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps({
                'async_task': _ASYNC_TASK_GENERATE_SAVED_RECIPE_IMAGE,
                'owner': owner,
                'recipe_id': recipe_id,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        _log_event(
            request_id,
            'saved_recipe_async_image_invoked',
            owner=owner,
            recipe_id=recipe_id,
        )
        return True
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_async_image_invoke_failed',
            owner=owner,
            recipe_id=recipe_id,
            failure_reason=str(exc),
        )
        return False


def _invoke_saved_recipe_text_refinement_async(owner, recipe_id, content, request_id=None):
    function_name = _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not function_name:
        _log_event(request_id, 'saved_recipe_async_text_invoke_skipped', owner=owner, recipe_id=recipe_id,
                   failure_reason='AWS_LAMBDA_FUNCTION_NAME is not available')
        return False
    try:
        boto3.client('lambda').invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps({
                'async_task': _ASYNC_TASK_REFINE_SAVED_RECIPE_TEXT,
                'owner': owner,
                'recipe_id': recipe_id,
                'content': content,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        _log_event(request_id, 'saved_recipe_async_text_invoked', owner=owner, recipe_id=recipe_id)
        return True
    except Exception as exc:
        _log_event(request_id, 'saved_recipe_async_text_invoke_failed', owner=owner, recipe_id=recipe_id,
                   failure_reason=str(exc))
        return False


def _handle_async_saved_recipe_text_task(event, request_id=None):
    owner = _safe_text(event.get('owner'))
    recipe_id = _safe_text(event.get('recipe_id'))
    content = _safe_text(event.get('content'))
    if not owner or not recipe_id or not content:
        print(f"[async-text-refine] Missing required fields: owner={owner}, recipe_id={recipe_id}, content={'yes' if content else 'no'}")
        return {'statusCode': 400}
    started = time.time()
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        extraction = _build_text_recipe_extraction(content)
        recipe_response, structured_recipe = _analyze_extraction(extraction, request_id=request_id)
        if recipe_response['recipe'] == 'Not enough recipe information.':
            table = _saved_recipes_table(owner)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE `{table}` SET status = 'failed' WHERE _id = %s AND _owner = %s", (recipe_id, owner))
            conn.commit()
            _log_event(request_id, 'saved_recipe_async_text_failed', owner=owner, recipe_id=recipe_id, reason='not_enough_info')
            _report_backend_error('save_recipe_text', owner_id=owner, code='not_enough_info',
                                  error='extraction returned Not enough recipe information.', job_id=recipe_id)
            return {'statusCode': 200}
        table = _saved_recipes_table(owner)
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE `{table}` SET
                    title = %s, ingredients = %s, instructions = %s, notes = %s,
                    extraction_source = %s, status = 'ready'
                WHERE _id = %s AND _owner = %s""",
                (
                    structured_recipe['title'] or extraction['title'] or 'Recipe',
                    json.dumps(structured_recipe['ingredients']),
                    json.dumps(structured_recipe['instructions']),
                    json.dumps(structured_recipe['notes']),
                    extraction['source'],
                    recipe_id,
                    owner,
                ),
            )
        conn.commit()
        # Generate AI image
        _invoke_saved_recipe_image_generation_async(owner, recipe_id, request_id=request_id)
        # Compute kitchen availability
        row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        if row:
            _ensure_recipe_personalization_tables(conn)
            kitchen_context = _build_kitchen_match_context(conn, owner)
            _, overlay_row = _serialize_saved_recipe_with_availability(
                conn, owner, row, kitchen_context=kitchen_context, request_id=request_id)
            _persist_owner_recipe_availability_rows(conn, [overlay_row])
            try:
                _fan_out_saved_recipe_to_household(conn, owner, recipe_id, request_id=request_id)
            except Exception as fan_exc:
                _report_backend_error('household_fanout', owner_id=owner, code='fanout_failed',
                                      error=fan_exc, job_id=recipe_id)
        _log_event(request_id, 'saved_recipe_async_text_complete', owner=owner, recipe_id=recipe_id,
                   latency_ms=int((time.time() - started) * 1000))
        return {'statusCode': 200}
    except Exception as exc:
        print(f"[async-text-refine] Error: {exc}")
        import traceback
        traceback.print_exc()
        _report_backend_error('save_recipe_text', owner_id=owner, code='refine_error', error=exc, job_id=recipe_id)
        try:
            table = _saved_recipes_table(owner)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE `{table}` SET status = 'failed' WHERE _id = %s AND _owner = %s", (recipe_id, owner))
            conn.commit()
        except Exception:
            pass
        return {'statusCode': 500}


def _refresh_saved_recipe_generated_image(conn, owner, recipe_id, request_id=None):
    row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
    if not row:
        _log_event(
            request_id,
            'saved_recipe_async_image_missing',
            owner=owner,
            recipe_id=recipe_id,
        )
        return None
    recipe = _serialize_row(row)
    if _safe_text(recipe.get('image_storage_key')) and _is_owned_recipe_image_url(recipe.get('image_url')):
        _log_event(
            request_id,
            'saved_recipe_async_image_skipped',
            owner=owner,
            recipe_id=recipe_id,
            reason='image_already_generated',
        )
        return recipe
    image_fields = _prepare_generated_saved_recipe_image_fields(
        owner,
        recipe_id,
        recipe,
        source_image_urls=recipe.get('source_image_urls') or [],
        request_id=request_id,
    )
    _update_saved_recipe_image_fields(conn, owner, recipe_id, image_fields)
    refreshed_row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
    refreshed_recipe = _serialize_row(refreshed_row) if refreshed_row else recipe
    _log_event(
        request_id,
        'saved_recipe_async_image_complete',
        owner=owner,
        recipe_id=recipe_id,
        image_storage_key=refreshed_recipe.get('image_storage_key'),
    )
    return refreshed_recipe


def _ensure_owned_saved_recipe_image(conn, owner, row, request_id=None):
    if not row:
        return row
    recipe_id = _safe_text(row.get('_id'))
    current_image_url = _safe_text(row.get('image_url'))
    if not recipe_id or not current_image_url:
        return row
    if _is_owned_recipe_image_url(current_image_url):
        return row

    candidate_urls = _unique_texts([
        row.get('source_image_url'),
        *(_parse_json_field(row.get('source_image_urls')) or []),
        current_image_url,
    ])
    for candidate_url in candidate_urls:
        try:
            image_fields = _mirror_recipe_image(owner, recipe_id, candidate_url, request_id=request_id)
            _update_saved_recipe_image_fields(conn, owner, recipe_id, image_fields)
            row.update(image_fields)
            return row
        except Exception:
            continue

    refresh_url = _safe_text(row.get('source_url') or row.get('resolved_url'))
    if not refresh_url:
        return row

    try:
        refreshed_extraction = _extract_content(refresh_url, request_id=request_id)
        image_fields = _prepare_saved_recipe_image_fields(
            owner,
            recipe_id,
            refreshed_extraction,
            request_id=request_id,
        )
        if image_fields.get('image_url') and _is_owned_recipe_image_url(image_fields.get('image_url')):
            _update_saved_recipe_image_fields(conn, owner, recipe_id, image_fields)
            row.update(image_fields)
            return row
    except Exception as exc:
        _log_event(
            request_id,
            'saved_recipe_image_refresh_failed',
            owner=owner,
            recipe_id=recipe_id,
            source_url=refresh_url,
            failure_reason=str(exc),
        )
    return row


def _fetch_saved_recipe_by_id(conn, owner, item_id):
    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT {_SAVED_RECIPE_SELECT_FIELDS}
                FROM `{table}`
                WHERE _id = %s
                LIMIT 1""",
            [item_id]
        )
        return cur.fetchone()


def _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash):
    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT {_SAVED_RECIPE_SELECT_FIELDS}
                FROM `{table}`
                WHERE resolved_url_hash = %s
                LIMIT 1""",
            [resolved_url_hash]
        )
        return cur.fetchone()


def _get_saved_recipes(owner, query=None, request_id=None):
    conn = _mysql_conn()
    try:
        table = _saved_recipes_table(owner)
        limit = _parse_limit(query)
        before = _parse_before(query)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if cur.fetchone()['n'] == 0:
                return _success({'owner': owner, 'recipes': [], 'count': 0})
        _ensure_saved_recipes_table(conn, owner)
        with conn.cursor() as cur:
            params = []
            where_clause = ""
            if before:
                where_clause = "WHERE COALESCE(_updatedDate, _createdDate) < %s"
                params.append(before)
            cur.execute(
                f"""SELECT {_SAVED_RECIPE_SELECT_FIELDS}
                    FROM `{table}`
                    {where_clause}
                    ORDER BY COALESCE(_updatedDate, _createdDate) DESC
                    LIMIT {limit + 1}""",
                params,
            )
            rows = cur.fetchall() or []
        has_more = len(rows) > limit
        rows = rows[:limit]
        rows = [_ensure_owned_saved_recipe_image(conn, owner, row, request_id=request_id) for row in rows]
        serialized_rows = [_serialize_row(row) for row in rows]
        recipe_ids = [s.get('id') for s in serialized_rows if s.get('id')]
        current_kv = _get_owner_kitchen_version(conn, owner)
        availability_map = _get_cached_recipe_availability(conn, owner, recipe_ids, min_kitchen_version=current_kv)
        recipes = []
        for serialized in serialized_rows:
            avail = availability_map.get(serialized.get('id'))
            if avail:
                serialized['availability'] = {
                    'kitchen_version': avail['kitchen_version'],
                    'can_make_exact': avail['can_make_exact'],
                    'can_make_with_subs': avail['can_make_with_subs'],
                    'matched_count': avail['matched_count'],
                    'missing_count': avail['missing_count'],
                }
                serialized['ingredient_matches'] = avail['ingredient_matches']
                serialized['missing_ingredients'] = avail['missing_ingredients']
                serialized['substitution_candidates'] = avail['substitution_candidates']
                serialized['substitution_summary'] = avail['substitution_summary']
                serialized['substitution_status'] = avail['substitution_status']
            else:
                serialized['availability'] = None
                serialized['ingredient_matches'] = None
                serialized['missing_ingredients'] = None
                serialized['substitution_candidates'] = None
                serialized['substitution_summary'] = None
                serialized['substitution_status'] = None
            recipes.append(serialized)
        return _success({
            'owner': owner,
            'recipes': recipes,
            'count': len(recipes),
            'limit': limit,
            'has_more': has_more,
        })
    finally:
        pass


def _personalize_saved_recipes(owner, request_id=None, recipe_ids=None):
    """Phase 2: run the LLM batch, persist results, and return per-recipe personalization data."""
    conn = _mysql_conn()
    try:
        table = _saved_recipes_table(owner)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if cur.fetchone()['n'] == 0:
                return _success({'owner': owner, 'recipes': []})
        _ensure_saved_recipes_table(conn, owner)
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT {_SAVED_RECIPE_SELECT_FIELDS}
                    FROM `{table}`
                    ORDER BY COALESCE(_updatedDate, _createdDate) DESC""",
            )
            rows = cur.fetchall() or []
        _ensure_recipe_personalization_tables(conn)
        kitchen_context = _build_kitchen_match_context(conn, owner)
        serialized_rows = [_serialize_row(row) for row in rows]
        if recipe_ids:
            serialized_rows = [r for r in serialized_rows if r.get('id') in recipe_ids]
        availability_map = _compute_saved_recipe_availability_batch(serialized_rows, kitchen_context, request_id=request_id)
        personalization_list = []
        overlay_rows = []
        for serialized in serialized_rows:
            recipe_id = serialized.get('id')
            avail = availability_map.get(recipe_id) or {}
            personalization_list.append({
                'id': recipe_id,
                'availability': {
                    'kitchen_version': avail.get('kitchen_version'),
                    'can_make_exact': avail.get('can_make_exact'),
                    'can_make_with_subs': avail.get('can_make_with_subs'),
                    'matched_count': avail.get('matched_count'),
                    'missing_count': avail.get('missing_count'),
                } if avail else None,
                'ingredient_matches': avail.get('ingredient_matches'),
                'substitution_candidates': avail.get('substitution_candidates'),
                'substitution_summary': avail.get('substitution_summary'),
                'substitution_status': avail.get('substitution_status'),
            })
            if avail and recipe_id:
                overlay_rows.append(_build_owner_recipe_availability_record(owner, 'saved', recipe_id, avail))
        _persist_owner_recipe_availability_rows(conn, overlay_rows)
        return _success({'owner': owner, 'recipes': personalization_list})
    finally:
        pass


def _save_saved_recipe_record(conn, owner, extraction, request_id=None, prepared_images=None):
    resolved_url_hash = _sha256(extraction['resolved_url'])
    existing = _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash)
    if existing:
        existing = _ensure_owned_saved_recipe_image(conn, owner, existing, request_id=request_id)
        _ensure_recipe_personalization_tables(conn)
        kitchen_context = _build_kitchen_match_context(conn, owner)
        serialized_existing, overlay_row = _serialize_saved_recipe_with_availability(
            conn,
            owner,
            existing,
            kitchen_context=kitchen_context,
            request_id=request_id,
        )
        _persist_owner_recipe_availability_rows(conn, [overlay_row])
        _log_event(
            request_id,
            'saved_recipe_deduped',
            owner=owner,
            platform=extraction['platform'],
            resolved_url=extraction['resolved_url'],
            extraction_source=extraction['source'],
        )
        return {
            'recipe': serialized_existing,
            'deduped': True,
            'resolved_url_hash': resolved_url_hash,
        }, 200

    recipe_response, structured_recipe = _analyze_extraction(
        extraction,
        request_id=request_id,
    )
    if recipe_response['recipe'] == 'Not enough recipe information.':
        raise ServiceError('Not enough recipe information.', status_code=422)

    recipe_id = str(uuid.uuid4())
    manual_platform = extraction['platform'] in {'text', 'image'}
    if manual_platform:
        source_image_urls = _upload_saved_recipe_source_images(
            owner,
            recipe_id,
            prepared_images or [],
            request_id=request_id,
        )
        image_fields = _build_source_backed_saved_recipe_image_fields(source_image_urls=source_image_urls)
    else:
        image_fields = _prepare_saved_recipe_image_fields(
            owner,
            recipe_id,
            extraction,
            request_id=request_id,
        )

    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO `{table}` (
                    _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                    title, image_url, image_urls, source_image_url, source_image_urls, image_storage_key,
                    ingredients, instructions, notes, raw_caption, raw_content,
                    extraction_source, author_name, caption_field, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ready')""",
            (
                recipe_id,
                owner,
                extraction['platform'],
                extraction['url'],
                extraction['resolved_url'],
                resolved_url_hash,
                structured_recipe['title'] or extraction['title'] or 'Recipe',
                image_fields.get('image_url') or None,
                json.dumps(image_fields.get('image_urls') or []),
                image_fields.get('source_image_url') or None,
                json.dumps(image_fields.get('source_image_urls') or []),
                image_fields.get('image_storage_key') or None,
                json.dumps(structured_recipe['ingredients']),
                json.dumps(structured_recipe['instructions']),
                json.dumps(structured_recipe['notes']),
                extraction['caption'],
                extraction['content'],
                extraction['source'],
                extraction.get('author_name'),
                extraction.get('caption_field') or None,
            )
        )
    conn.commit()
    if manual_platform and extraction['platform'] != 'image':
        _invoke_saved_recipe_image_generation_async(owner, recipe_id, request_id=request_id)
    row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
    _ensure_recipe_personalization_tables(conn)
    kitchen_context = _build_kitchen_match_context(conn, owner)
    serialized_row, overlay_row = _serialize_saved_recipe_with_availability(
        conn,
        owner,
        row,
        kitchen_context=kitchen_context,
        request_id=request_id,
    )
    _persist_owner_recipe_availability_rows(conn, [overlay_row])
    _log_event(
        request_id,
        'saved_recipe_created',
        owner=owner,
        recipe_id=recipe_id,
        platform=extraction['platform'],
        resolved_url=extraction['resolved_url'],
        extraction_source=extraction['source'],
    )
    # Fan out to household members
    try:
        _fan_out_saved_recipe_to_household(conn, owner, recipe_id, request_id=request_id)
    except Exception as exc:
        _log_event(request_id, 'household_fanout_failed', owner=owner, recipe_id=recipe_id, error=str(exc))
    return {
        'recipe': serialized_row,
        'deduped': False,
        'resolved_url_hash': resolved_url_hash,
    }, 201


def _build_saved_recipe_post_response(owner, results, partial_errors=None):
    results = list(results or [])
    partial_errors = list(partial_errors or [])
    if not results:
        raise ServiceError('Not enough recipe information.', status_code=422, extra={'partial_errors': partial_errors})

    has_created = any(not result.get('deduped') for result in results)
    status = 201 if has_created else 200
    if len(results) == 1:
        result = results[0]
        body = {
            'owner': owner,
            'recipe': result['recipe'],
            'deduped': bool(result.get('deduped')),
        }
        if partial_errors:
            body['partial_errors'] = partial_errors
        return _success(body, status=status)

    body = {
        'owner': owner,
        'recipes': [result['recipe'] for result in results],
        'results': [
            {
                'recipe': result['recipe'],
                'deduped': bool(result.get('deduped')),
            }
            for result in results
        ],
        'count': len(results),
    }
    if partial_errors:
        body['partial_errors'] = partial_errors
    return _success(body, status=status)


def _build_saved_recipe_batch_accept_response(owner, job, event=None):
    return _success({
        'owner': owner,
        'accepted': True,
        'job': _serialize_saved_recipe_batch_job(job, event=event),
    }, status=202)


def _build_saved_recipe_batch_job_response(owner, job, recipes=None, results=None, partial_errors=None, event=None):
    body = {
        'owner': owner,
        'job': _serialize_saved_recipe_batch_job(job, event=event),
        'recipes': recipes or [],
        'results': results or [],
        'count': len(results or []),
    }
    if partial_errors:
        body['partial_errors'] = partial_errors
    return _success(body)


def _process_saved_recipe_image_batch(conn, owner, image_submissions, request_id=None):
    clusters, partial_errors = _extract_recipe_clusters_from_images(image_submissions, request_id=request_id)
    results = []
    for cluster in clusters:
        try:
            result, _ = _save_saved_recipe_record(
                conn,
                owner,
                cluster['extraction'],
                request_id=request_id,
                prepared_images=cluster.get('prepared_images') or [],
            )
            results.append({
                'recipe_id': ((result or {}).get('recipe') or {}).get('id'),
                'deduped': bool((result or {}).get('deduped')),
                'image_indexes': list(cluster.get('image_indexes') or []),
            })
        except ServiceError as exc:
            partial_errors.append({
                'image_indexes': cluster.get('image_indexes') or [],
                'error': str(exc),
                'status_code': exc.status_code,
            })
    if not results:
        first_error = partial_errors[0] if partial_errors else {'error': 'Not enough recipe information.', 'status_code': 422}
        raise ServiceError(
            first_error.get('error') or 'Not enough recipe information.',
            status_code=first_error.get('status_code') or 422,
            extra={'partial_errors': partial_errors},
        )
    return results, partial_errors


def _enqueue_saved_recipe_batch_job(owner, submission, request_id=None, event=None):
    job_id = uuid.uuid4().hex
    now = _utc_now_iso()
    payload_key = _put_saved_recipe_batch_payload(owner, job_id, submission)
    job = {
        'job_id': job_id,
        'type': _SAVED_RECIPE_BATCH_JOB_TYPE,
        'owner': owner,
        'status': _JOB_STATUS_PENDING,
        'image_count': len(submission.get('images') or []),
        'result_count': 0,
        'recipe_ids': [],
        'results': [],
        'partial_errors': [],
        'request_id': request_id,
        'created_at': now,
        'updated_at': now,
        'payload_s3_key': payload_key,
    }
    _put_saved_recipe_batch_job(job)
    function_name = _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not function_name:
        raise RuntimeError('AWS_LAMBDA_FUNCTION_NAME is not available')
    try:
        boto3.client('lambda').invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps({
                'async_task': _ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH,
                'owner': owner,
                'job_id': job_id,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        _log_event(
            request_id,
            'saved_recipe_batch_job_enqueued',
            owner=owner,
            job_id=job_id,
            image_count=job['image_count'],
        )
        return job
    except Exception as exc:
        failed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_FAILED,
            error=str(exc),
            completed_at=_utc_now_iso(),
        )
        _log_event(
            request_id,
            'saved_recipe_batch_job_enqueue_failed',
            owner=owner,
            job_id=job_id,
            failure_reason=str(exc),
        )
        raise RuntimeError(str(exc))


def _fetch_saved_recipe_batch_results(conn, owner, job, request_id=None):
    result_entries = list((job or {}).get('results') or [])
    recipes = []
    results = []
    overlay_rows = []
    partial_errors = list((job or {}).get('partial_errors') or [])
    _ensure_recipe_personalization_tables(conn)
    kitchen_context = _build_kitchen_match_context(conn, owner)
    serialized_rows = []
    for entry in result_entries:
        recipe_id = _safe_text((entry or {}).get('recipe_id'))
        if not recipe_id:
            continue
        row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        if not row:
            continue
        row = _ensure_owned_saved_recipe_image(conn, owner, row, request_id=request_id)
        serialized_rows.append((entry, row, _serialize_row(row)))
    availability_map = _compute_saved_recipe_availability_batch(
        [serialized for _, _, serialized in serialized_rows],
        kitchen_context,
        request_id=request_id,
    )
    for entry, row, serialized in serialized_rows:
        recipe, overlay_row = _serialize_saved_recipe_with_availability(
            conn,
            owner,
            row,
            kitchen_context=kitchen_context,
            availability=availability_map.get((serialized or {}).get('id')),
            request_id=request_id,
        )
        recipes.append(recipe)
        overlay_rows.append(overlay_row)
        results.append({
            'recipe': recipe,
            'deduped': bool((entry or {}).get('deduped')),
        })
    _persist_owner_recipe_availability_rows(conn, overlay_rows)
    return recipes, results, partial_errors


def _post_saved_recipe(owner, body, request_id=None):
    try:
        submission = _build_saved_recipe_input(body)
    except ServiceError as exc:
        return _error(exc.status_code, str(exc), exc.extra)
    if submission['kind'] == 'images':
        try:
            job = _enqueue_saved_recipe_batch_job(owner, submission, request_id=request_id, event=body.get('_event'))
        except Exception as exc:
            return _error(500, str(exc))
        return _build_saved_recipe_batch_accept_response(owner, job, event=body.get('_event'))
    started = time.time()
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        response_result_count = 0
        if submission['kind'] == 'url':
            extraction = _extract_content(submission['url'], request_id=request_id)
            result, _ = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id)
            response = _build_saved_recipe_post_response(owner, [result])
            response_result_count = 1
        elif submission['kind'] == 'content':
            extraction = _build_text_recipe_extraction(submission['content'])
            resolved_url_hash = _sha256(extraction['resolved_url'])
            existing = _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash)
            if existing:
                # Duplicate — return synchronously
                existing = _ensure_owned_saved_recipe_image(conn, owner, existing, request_id=request_id)
                serialized_existing = _serialize_row(existing)
                response = _success({'owner': owner, 'recipe': serialized_existing, 'deduped': True}, status=200)
                response_result_count = 1
            else:
                # Insert placeholder, refine asynchronously
                recipe_id = str(uuid.uuid4())
                # Quick-parse title and ingredients from the text
                content_text = submission['content']
                lines = content_text.strip().split('\n')
                placeholder_title = lines[0].strip() if lines else 'Recipe'
                raw_ingredients = []
                in_ingredients = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.lower().startswith('ingredients'):
                        in_ingredients = True
                        continue
                    if stripped.lower().startswith('steps') or stripped.lower().startswith('directions'):
                        in_ingredients = False
                        continue
                    if in_ingredients and stripped.startswith('- '):
                        raw_ingredients.append(stripped[2:])
                table = _saved_recipes_table(owner)
                with conn.cursor() as cur:
                    cur.execute(
                        f"""INSERT INTO `{table}` (
                                _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                                title, ingredients, instructions, notes, raw_caption, raw_content,
                                extraction_source, status
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'processing')""",
                        (
                            recipe_id, owner, extraction['platform'],
                            extraction['url'], extraction['resolved_url'], resolved_url_hash,
                            placeholder_title,
                            json.dumps(raw_ingredients), json.dumps([]), json.dumps([]),
                            extraction['caption'], extraction['content'], extraction['source'],
                        ),
                    )
                conn.commit()
                _invoke_saved_recipe_text_refinement_async(owner, recipe_id, content_text, request_id=request_id)
                placeholder_row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
                serialized = _serialize_row(placeholder_row) if placeholder_row else {
                    'id': recipe_id, 'title': placeholder_title, 'status': 'processing',
                    'ingredients': raw_ingredients, 'instructions': [], 'notes': [],
                }
                response = _success({'owner': owner, 'recipe': serialized, 'deduped': False, 'async': True}, status=202)
                response_result_count = 1
        elif submission['kind'] == 'image':
            extraction, prepared_image = _extract_recipe_from_image(submission, request_id=request_id)
            result, _ = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id, prepared_images=[prepared_image])
            response = _build_saved_recipe_post_response(owner, [result])
            response_result_count = 1

        _log_event(
            request_id,
            'saved_recipe_post_complete',
            owner=owner,
            submission_kind=submission['kind'],
            result_count=response_result_count,
            latency_ms=int((time.time() - started) * 1000),
        )
        return response
    finally:
        pass


def _get_saved_recipe_batch_job_status(owner, job_id, event=None, request_id=None):
    job = _get_saved_recipe_batch_job(job_id)
    if not job or _safe_text(job.get('owner')) != owner:
        return _error(404, 'Saved recipe batch job not found')
    job_status = _safe_text(job.get('status'))
    if job_status != _JOB_STATUS_COMPLETED:
        return _build_saved_recipe_batch_job_response(
            owner,
            job,
            recipes=[],
            results=[],
            partial_errors=list(job.get('partial_errors') or []),
            event=event,
        )
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        recipes, results, partial_errors = _fetch_saved_recipe_batch_results(conn, owner, job, request_id=request_id)
        return _build_saved_recipe_batch_job_response(
            owner,
            job,
            recipes=recipes,
            results=results,
            partial_errors=partial_errors,
            event=event,
        )
    finally:
        pass


def _delete_saved_recipe(owner, item_id):
    conn = _mysql_conn()
    try:
        table = _saved_recipes_table(owner)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if cur.fetchone()['n'] == 0:
                return _error(404, 'Saved recipe not found')
        _ensure_saved_recipes_table(conn, owner)
        row = _fetch_saved_recipe_by_id(conn, owner, item_id)
        if not row:
            return _error(404, 'Saved recipe not found')
        _ensure_recipe_personalization_tables(conn)
        kitchen_context = _build_kitchen_match_context(conn, owner)
        serialized_row, _ = _serialize_saved_recipe_with_availability(
            conn,
            owner,
            row,
            kitchen_context=kitchen_context,
        )
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM `{table}` WHERE _id = %s LIMIT 1", [item_id])
        conn.commit()
        _delete_owner_recipe_availability(conn, owner, 'saved', item_id)
        # Fan out delete to household members
        try:
            _fan_out_delete_to_household(conn, owner, item_id)
        except Exception:
            pass  # Non-critical — other members can delete independently
        return _success({'owner': owner, 'recipe': serialized_row})
    finally:
        pass


def _update_saved_recipe(owner, item_id, body, request_id=None):
    """Edit a saved recipe's user-editable fields (title, ingredients, steps/instructions,
    notes). Partial updates allowed. Applies to the owner + any household members that have
    the recipe, then recomputes availability since ingredient changes affect kitchen matching."""
    updates = {}
    if 'title' in body:
        title = _safe_text(body.get('title'))
        if not title:
            return _error(400, 'Title cannot be empty')
        updates['title'] = title[:255]
    if 'ingredients' in body:
        updates['ingredients'] = json.dumps(_clean_string_list(body.get('ingredients') or []))
    # iOS sends "steps"; the column is "instructions". Accept either key.
    if 'steps' in body or 'instructions' in body:
        steps = body.get('steps') if 'steps' in body else body.get('instructions')
        updates['instructions'] = json.dumps(_clean_string_list(steps or []))
    if 'notes' in body:
        updates['notes'] = json.dumps(_clean_string_list(body.get('notes') or []))
    if not updates:
        return _error(400, 'No editable fields provided')

    conn = _mysql_conn()
    table = _saved_recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
        """, [table])
        if cur.fetchone()['n'] == 0:
            return _error(404, 'Saved recipe not found')
    _ensure_saved_recipes_table(conn, owner)
    if not _fetch_saved_recipe_by_id(conn, owner, item_id):
        return _error(404, 'Saved recipe not found')

    # Per-user by default: an edit only touches the acting user's own copy. Household
    # propagation happens only when saved-recipe sharing is explicitly enabled.
    targets = set()
    if SAVED_RECIPE_HOUSEHOLD_FANOUT:
        try:
            targets = set(_get_household_member_ids(conn, owner) or [])
        except Exception:
            targets = set()
    targets.add(_safe_owner_token(owner))
    set_clause = ', '.join(f"`{col}` = %s" for col in updates) + ", `_updatedDate` = NOW()"
    vals = list(updates.values())
    for tid in targets:
        try:
            _ensure_saved_recipes_table(conn, tid)
            mtable = _saved_recipes_table(tid)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE `{mtable}` SET {set_clause} WHERE _id = %s LIMIT 1", vals + [item_id])
        except Exception as exc:
            _log_event(request_id, 'saved_recipe_update_member_error', member=tid, error=str(exc))
    conn.commit()

    # Ingredients may have changed → recompute availability against the owner's kitchen.
    _ensure_recipe_personalization_tables(conn)
    updated_row = _fetch_saved_recipe_by_id(conn, owner, item_id)
    kitchen_context = _build_kitchen_match_context(conn, owner)
    serialized_row, overlay_row = _serialize_saved_recipe_with_availability(
        conn, owner, updated_row, kitchen_context=kitchen_context, request_id=request_id,
    )
    try:
        _persist_owner_recipe_availability_rows(conn, [overlay_row])
    except Exception:
        pass
    _log_event(request_id, 'saved_recipe_updated', item_id=item_id, fields=list(updates.keys()))
    return _success({'owner': owner, 'recipe': serialized_row})


def _parse_json_body(event):
    try:
        body = json.loads(event.get('body', '{}') or '{}')
    except json.JSONDecodeError as exc:
        raise ServiceError(f'Invalid JSON: {str(exc)}', status_code=400)
    if not isinstance(body, dict):
        raise ServiceError('Request body must be a JSON object.', status_code=400)
    return body


def _handle_extract_content(body, request_id):
    extraction = _extract_content(body.get('url'), request_id=request_id)
    return _success(extraction)


def _handle_recipe_from_content(body, request_id):
    content = body.get('content') or body.get('caption')
    response, _ = _recipe_response_from_content(content, request_id=request_id)
    return _success(response)


def _handle_analyze_url(body, request_id):
    extraction = _extract_content(body.get('url'), request_id=request_id)
    response, _ = _analyze_extraction(extraction, request_id=request_id)
    return _success({**extraction, **response})


def _handle_async_saved_recipe_image_task(event, request_id):
    owner = _safe_text((event or {}).get('owner'))
    recipe_id = _safe_text((event or {}).get('recipe_id'))
    if not owner or not recipe_id:
        raise ServiceError('Async image task requires owner and recipe_id.', status_code=400)
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        recipe = _refresh_saved_recipe_generated_image(conn, owner, recipe_id, request_id=request_id)
        return {
            'ok': True,
            'owner': owner,
            'recipe_id': recipe_id,
            'image_url': (recipe or {}).get('image_url'),
            'image_storage_key': (recipe or {}).get('image_storage_key'),
        }
    finally:
        pass


def _handle_async_saved_recipe_batch_task(event, request_id):
    owner = _safe_text((event or {}).get('owner'))
    job_id = _safe_text((event or {}).get('job_id'))
    if not owner or not job_id:
        raise ServiceError('Async batch task requires owner and job_id.', status_code=400)
    job = _get_saved_recipe_batch_job(job_id)
    if not job or _safe_text(job.get('owner')) != owner:
        raise ServiceError('Saved recipe batch job not found.', status_code=404)
    _update_saved_recipe_batch_job(job_id, status=_JOB_STATUS_RUNNING, started_at=_utc_now_iso(), error=None)
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        submission = _load_saved_recipe_batch_payload(job)
        results, partial_errors = _process_saved_recipe_image_batch(
            conn,
            owner,
            submission.get('images') or [],
            request_id=request_id,
        )
        completed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_COMPLETED,
            result_count=len(results),
            recipe_ids=[entry.get('recipe_id') for entry in results if entry.get('recipe_id')],
            results=results,
            partial_errors=partial_errors,
            completed_at=_utc_now_iso(),
            error=None,
        )
        _log_event(
            request_id,
            'saved_recipe_batch_job_complete',
            owner=owner,
            job_id=job_id,
            result_count=len(results),
            partial_error_count=len(partial_errors),
        )
        return {
            'ok': True,
            'owner': owner,
            'job_id': job_id,
            'status': completed_job.get('status'),
            'count': len(results),
        }
    except ServiceError as exc:
        failed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_FAILED,
            result_count=0,
            recipe_ids=[],
            results=[],
            partial_errors=exc.extra.get('partial_errors') or [],
            completed_at=_utc_now_iso(),
            error=str(exc),
        )
        _log_event(
            request_id,
            'saved_recipe_batch_job_failed',
            owner=owner,
            job_id=job_id,
            failure_reason=str(exc),
            partial_error_count=len(failed_job.get('partial_errors') or []),
        )
        _report_backend_error('save_recipe_image_batch', owner_id=owner, code='batch_failed',
                              error=exc, job_id=job_id)
        return {
            'ok': False,
            'owner': owner,
            'job_id': job_id,
            'status': failed_job.get('status'),
            'error': str(exc),
        }
    except Exception as exc:
        failed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_FAILED,
            result_count=0,
            recipe_ids=[],
            results=[],
            partial_errors=[],
            completed_at=_utc_now_iso(),
            error=str(exc),
        )
        _log_event(
            request_id,
            'saved_recipe_batch_job_failed',
            owner=owner,
            job_id=job_id,
            failure_reason=str(exc),
            partial_error_count=0,
        )
        _report_backend_error('save_recipe_image_batch', owner_id=owner, code='batch_error',
                              error=exc, job_id=job_id)
        return {
            'ok': False,
            'owner': owner,
            'job_id': job_id,
            'status': failed_job.get('status'),
            'error': str(exc),
        }
    finally:
        pass


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    request_id = (
        _safe_text((event or {}).get('request_id'))
        or _safe_text((event or {}).get('requestContext', {}).get('requestId'))
        or _safe_text(getattr(context, 'aws_request_id', ''))
    )
    if (event or {}).get('async_task') == _ASYNC_TASK_GENERATE_SAVED_RECIPE_IMAGE:
        return _handle_async_saved_recipe_image_task(event, request_id=request_id)
    if (event or {}).get('async_task') == _ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH:
        return _handle_async_saved_recipe_batch_task(event, request_id=request_id)
    if (event or {}).get('async_task') == _ASYNC_TASK_REFINE_SAVED_RECIPE_TEXT:
        return _handle_async_saved_recipe_text_task(event, request_id=request_id)

    http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
    path_params = event.get('pathParameters') or {}
    owner = _safe_text(path_params.get('owner'))
    item_id = _safe_text(path_params.get('item_id'))
    job_id = _safe_text(path_params.get('job_id'))
    raw_path = event.get('rawPath') or event.get('requestContext', {}).get('http', {}).get('path') or ''

    try:
        if http_method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

        if raw_path in _EXTRACT_CONTENT_PATHS:
            if http_method != 'POST':
                return _error(405, f'Method {http_method} not allowed')
            return _handle_extract_content(_parse_json_body(event), request_id)

        if raw_path in _RECIPE_FROM_CONTENT_PATHS:
            if http_method != 'POST':
                return _error(405, f'Method {http_method} not allowed')
            return _handle_recipe_from_content(_parse_json_body(event), request_id)

        if raw_path in _ANALYZE_URL_PATHS:
            if http_method != 'POST':
                return _error(405, f'Method {http_method} not allowed')
            return _handle_analyze_url(_parse_json_body(event), request_id)

        if not owner:
            return _error(400, 'Missing owner parameter')
        if http_method == 'GET' and raw_path.endswith('/personalize'):
            qsp = event.get('queryStringParameters') or {}
            ids_param = qsp.get('ids', '')
            recipe_ids = [i.strip() for i in ids_param.split(',') if i.strip()] if ids_param else None
            return _personalize_saved_recipes(owner, request_id=request_id, recipe_ids=recipe_ids)
        if http_method == 'GET':
            if job_id:
                return _get_saved_recipe_batch_job_status(owner, job_id, event=event, request_id=request_id)
            return _get_saved_recipes(owner, event.get('queryStringParameters') or {}, request_id=request_id)
        if http_method == 'POST':
            body = _parse_json_body(event)
            body['_event'] = event
            return _post_saved_recipe(owner, body, request_id=request_id)
        if http_method == 'DELETE':
            if not item_id:
                return _error(400, 'Missing item_id parameter')
            return _delete_saved_recipe(owner, item_id)
        if http_method == 'PUT':
            if not item_id:
                return _error(400, 'Missing item_id parameter')
            return _update_saved_recipe(owner, item_id, _parse_json_body(event), request_id=request_id)
        return _error(405, f'Method {http_method} not allowed')
    except ServiceError as exc:
        _log_event(request_id, 'request_failed', path=raw_path, status_code=exc.status_code, failure_reason=str(exc))
        return _error(exc.status_code, str(exc), exc.extra)
    except Exception as exc:
        _log_event(request_id, 'request_failed', path=raw_path, status_code=500, failure_reason=str(exc))
        import traceback
        traceback.print_exc()
        return _error(500, str(exc))
