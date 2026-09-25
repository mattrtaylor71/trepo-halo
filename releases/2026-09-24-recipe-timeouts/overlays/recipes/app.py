from recipe_readiness import usable_lines as recipe_usable_lines, complete as recipe_content_complete, outcome as recipe_content_outcome, recovery as recipe_content_recovery, incomplete_error as recipe_incomplete_error, INCOMPLETE_MESSAGE
from safe_logging import safe_print as print, redact_text
import base64
import recipe_batch_checkpoint
import recipe_work_budget
import admission
from types import SimpleNamespace
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
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, unquote, quote

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

# Shared-table migration Phase 1: dual-write each saved recipe to shared_saved_recipes
# (owner_id-keyed) IN ADDITION to the per-user table. Reads untouched. Default OFF —
# flip ON only after parity verification. A missed dual-write = redundant data, never lost.
DUAL_WRITE_SAVED_RECIPES = os.getenv('DUAL_WRITE_SAVED_RECIPES', 'false').lower() == 'true'
_SHARED_SAVED_RECIPES_TABLE = 'shared_saved_recipes'

_LAYER_PYTHON = Path(__file__).resolve().parents[1] / 'recipe_inventory_layer' / 'python'
if _LAYER_PYTHON.exists() and str(_LAYER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_LAYER_PYTHON))

import recipe_inventory_llm


_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_REQUEST_TIMEOUT_SECONDS = 15
# Image mirroring runs INSIDE the saved-recipes read path, so its timeout is a direct tax
# on every list request. At 15s one unreachable host (kroger.com, which does not answer
# bots) blew past API Gateway's 30s ceiling: 2 candidate URLs x 15s = 31.2s, the Lambda
# returned 200 to nobody, and the user got a 503 plus an endless "Finding recipes..."
# spinner on EVERY load, forever. A thumbnail is not worth 15 seconds.
_IMAGE_MIRROR_TIMEOUT_SECONDS = float(os.getenv('IMAGE_MIRROR_TIMEOUT_SECONDS', '3'))
# Hard ceiling on mirroring across a WHOLE request, however many rows or candidate URLs.
# Past this, rows keep their original image URL and mirror on a later request. Bounded
# degradation beats a response that never arrives.
_IMAGE_MIRROR_BUDGET_SECONDS = float(os.getenv('IMAGE_MIRROR_BUDGET_SECONDS', '6'))
_image_mirror_spent = {'seconds': 0.0}

# How long a failed image URL is left alone before we try it again. A host that refused
# us today is overwhelmingly likely to refuse us tomorrow; a week means a genuinely
# transient outage still self-heals without us re-paying the timeout every request.
_IMAGE_MIRROR_FAILURE_TTL_SECONDS = float(
    os.getenv('IMAGE_MIRROR_FAILURE_TTL_SECONDS', str(7 * 24 * 3600)))
# url -> epoch seconds after which it may be retried. Per-container, so a cold start
# retries once; the DB column below is what makes the memory durable.
_image_mirror_failures = {}


def _mirror_url_recently_failed(url):
    expiry = _image_mirror_failures.get(url)
    return bool(expiry and expiry > time.time())


def _note_mirror_failure(url, permanent=False):
    """Remember a dead URL, and keep the map from growing without bound in a long-lived
    container by dropping entries that have already expired."""
    if not url:
        return
    _image_mirror_failures[url] = float('inf') if permanent else time.time() + 300
    if len(_image_mirror_failures) > 5000:
        now = time.time()
        for key, expiry in list(_image_mirror_failures.items()):
            if expiry <= now:
                _image_mirror_failures.pop(key, None)
        while len(_image_mirror_failures) > 5000:
            _image_mirror_failures.pop(next(iter(_image_mirror_failures)))


def _ensure_mirror_failed_column(conn, table):
    """Lazy, idempotent ALTER — same pattern used elsewhere in this codebase."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN image_mirror_failed_at DATETIME NULL")
        conn.commit()
    except Exception:
        pass          # already present, or the table is gone; both are fine


def _mark_mirror_failed_in_db(conn, owner, recipe_id, failure_kind='legacy', source_url=None):
    """Persist each read-path marker independently; a missing legacy table must
    not prevent the shared row from retaining its failure classification."""
    metadata = json.dumps({'kind': failure_kind, 'url_hash': _sha256(source_url) if source_url else None})
    for table, predicate, values, family in [
        (_saved_recipes_table(owner), '_id = %s', [recipe_id], 'legacy'),
        ('shared_saved_recipes', 'owner_id = %s AND _id = %s', [owner, recipe_id], 'shared'),
    ]:
        try:
            _ensure_mirror_failed_column(conn, table)
            with conn.cursor() as cur:
                try:
                    cur.execute(f'ALTER TABLE `{table}` ADD COLUMN image_mirror_failure JSON NULL')
                except pymysql.err.OperationalError as error:
                    if not error.args or error.args[0] != 1060:
                        raise
                cur.execute(f'UPDATE `{table}` SET image_mirror_failed_at=NOW(), image_mirror_failure=%s '
                            f'WHERE {predicate} LIMIT 1', [metadata] + values)
            conn.commit()
        except Exception as error:
            code = error.args[0] if getattr(error, 'args', ()) and type(error.args[0]) is int else None
            if code == 1146 and family == 'legacy':
                continue
            _log_event(None, 'image_failure_marker_write_failed', table_family=family,
                       error_type=type(error).__name__, mysql_code=code)


def _row_mirror_failed_recently(row):
    failed_at = row.get('image_mirror_failed_at') if isinstance(row, dict) else None
    if not failed_at:
        return False
    metadata = row.get('image_mirror_failure')
    try:
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (ValueError, TypeError):
        metadata = None
    if isinstance(metadata, dict) and metadata.get('url_hash'):
        if metadata['url_hash'] != _sha256(_safe_text(row.get('image_url'))):
            return False
        if metadata.get('kind') == 'source_unavailable':
            return True
    try:
        age = (datetime.now() - failed_at).total_seconds()
    except Exception:
        return False
    ttl = 300 if isinstance(metadata, dict) and metadata.get('kind') == 'transient' else _IMAGE_MIRROR_FAILURE_TTL_SECONDS
    return age < ttl



def _image_mirror_budget_left():
    return _IMAGE_MIRROR_BUDGET_SECONDS - _image_mirror_spent['seconds']


def _reset_image_mirror_budget():
    _image_mirror_spent['seconds'] = 0.0
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
_ASYNC_TASK_PROCESS_SAVED_RECIPE_URL = 'process_saved_recipe_url'
_ASYNC_TASK_REFINE_SAVED_RECIPE_TEXT = 'refine_saved_recipe_text'
_ASYNC_TASK_REPAIR_SAVED_RECIPE = 'repair_saved_recipe'
_ASYNC_TASK_PERSONALIZE_WARM = 'personalize_warm'
_SAVED_RECIPE_BATCH_JOB_TYPE = 'saved_recipe_batch'
_JOB_STATUS_PENDING = 'PENDING'
_JOB_STATUS_RUNNING = 'RUNNING'
_JOB_STATUS_COMPLETED = 'COMPLETED'
_JOB_STATUS_FAILED = 'FAILED'
_DEFAULT_TRANSCRIPTION_MODEL = os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'whisper-1')
_DEFAULT_LIST_LIMIT = 200
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
# Apify's instagram-scraper actor WAS non-functional (0 successful runs), so the read
# timeout was cut to 10s to fail fast into the html fallback. The actor has since
# recovered — but its run-sync latency is 6-25s (median ~8.5s), so a 10s budget timed
# out on ~45% of real reels and dropped them onto the html source, whose og:description
# caption is usually too thin to yield a recipe ("Not enough recipe information."). 30s
# covers the observed distribution with headroom and still leaves room inside the 120s
# Lambda timeout. Env-overridable.
_APIFY_RUN_TIMEOUT_SECONDS = max(1, int(os.getenv('APIFY_RUN_TIMEOUT_SECONDS', '30')))
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
                       recipe_source_used,
                       author_name, caption_field, status, meal_category, _createdDate, _updatedDate"""
# Canonical field order the readers expect (parsed from the list above). Per-owner
# saved_recipes tables drifted over time — ~5185 of 7643 predate columns like
# source_image_url/source_image_urls/image_storage_key/image_urls/raw_content/
# caption_field/meal_category. Reads MUST NOT hard-list those or a stale table 500s with
# 1054 "Unknown column". _drift_safe_select() NULL-fills any expected-but-missing column.
_SAVED_RECIPE_FIELDS = [f.strip() for f in _SAVED_RECIPE_SELECT_FIELDS.replace('\n', ' ').split(',') if f.strip()]


def _table_column_set(conn, table):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            [table],
        )
        return {
            (row.get('column_name') or row.get('COLUMN_NAME'))
            for row in (cur.fetchall() or [])
            if (row.get('column_name') or row.get('COLUMN_NAME'))
        }


def _drift_safe_select(conn, table, extra_fields=None):
    """Build a schema-drift-tolerant SELECT list for a saved_recipes table: columns that
    exist are selected as-is; expected columns the table LACKS are NULL-filled (aliased)
    so a legacy per-owner table missing e.g. source_image_url can never raise 1054. Field
    set/order matches _SAVED_RECIPE_SELECT_FIELDS so downstream mapping is unchanged."""
    fields = list(_SAVED_RECIPE_FIELDS)
    for extra in (extra_fields or []):
        if extra not in fields:
            fields.append(extra)
    present = _table_column_set(conn, table)
    return ', '.join(
        (f'`{f}`' if f in present else f'NULL AS `{f}`') for f in fields
    )


_OWNER_KITCHEN_STATE_TABLE = 'owner_kitchen_state'
_OWNER_RECIPE_AVAILABILITY_TABLE = 'owner_recipe_availability'
# Max saved recipes to LLM-match synchronously in one /personalize call. Cache-read
# serves the rest instantly; a cold recompute beyond this cap defers to the next call
# so a single request can never exceed the API-gateway timeout.
_MAX_PERSONALIZE_SYNC_COMPUTE = int(os.getenv('MAX_PERSONALIZE_SYNC_COMPUTE', '8'))
# Background warm: after the sync cap serves the first slice, self-invoke to compute the
# rest (persisting to cache) so subsequent polls/opens hit the cache instead of the LLM.
# Fully reversible: PERSONALIZE_ASYNC_WARM=false restores the pre-warm behavior.
_PERSONALIZE_ASYNC_WARM = os.getenv('PERSONALIZE_ASYNC_WARM', 'true').strip().lower() == 'true'
_PERSONALIZE_WARM_MAX = int(os.getenv('PERSONALIZE_WARM_MAX', '60'))
_PERSONALIZE_WARM_CHUNK = int(os.getenv('PERSONALIZE_WARM_CHUNK', '12'))
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
            msg = redact_text(error if isinstance(error, str) else str(error))
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
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
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


def _household_owner_filter(conn, owner):
    """(sql_fragment, params) restricting a shared_saved_recipes read to the acting user's
    HOUSEHOLD rather than just the user.

    Saved recipes are written per-user but read from the shared table, so a household
    member who has saved nothing sees an empty list even though their household has
    hundreds — household 43375 is the live case (one member 207 recipes, the other 0
    visible). This mirrors exactly how _get_kitchen_items_for_matching already scopes
    kitchen reads, via _get_household_member_ids.

    READ-ONLY. Mutations continue to resolve by the acting user's own per-owner table
    (_fetch_saved_recipe_by_id), so cross-member edit/delete still 404s.
    Falls back to the acting user alone if household resolution fails, so a lookup
    problem degrades to today's behaviour rather than erroring.
    """
    try:
        members = _get_household_member_ids(conn, owner) or []
    except Exception:
        members = []
    members = [m for m in members if m]
    if not members:
        members = [owner]
    placeholders = ', '.join(['%s'] * len(members))
    return f"owner_id IN ({placeholders})", list(members)


def _dedupe_household_rows(rows, acting_owner):
    """Collapse the same recipe saved by multiple members, preferring the ACTING user's
    own copy so tapping it opens an editable row. Mirrors the dedupe already used by
    _read_household_saved_recipe_rows. Rows without a hash are always kept."""
    acting = _safe_owner_token(acting_owner)
    seen = {}
    out = []
    for row in rows:
        key = row.get('resolved_url_hash')
        if not key:
            out.append(row)
            continue
        if key not in seen:
            seen[key] = len(out)
            out.append(row)
            continue
        idx = seen[key]
        if _safe_owner_token(row.get('owner_id')) == acting and \
                _safe_owner_token(out[idx].get('owner_id')) != acting:
            out[idx] = row
    return out


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


def _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=None, transactional=False, required=False):
    """Migration dual-write: mirror a just-saved recipe into shared_saved_recipes
    (owner_id-keyed; per-owner dedupe on resolved_url_hash). saved_recipes is
    user-private, so owner_id identifies the copy being mirrored. Default calls log
    and swallow errors. Transactional callers propagate errors and never commit;
    required=True retains the projection when shared reads require it even if the
    optional dual-write flag is off."""
    if not DUAL_WRITE_SAVED_RECIPES and not required:
        return
    try:
        row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        if not row:
            return
        # _SAVED_RECIPE_SELECT_FIELDS omits resolved_url_hash — recompute identically.
        # Uses the same canonical key as the writer so the recomputed value matches what
        # _save_saved_recipe_record stored (identical to the old value for non-Instagram).
        resolved_url_hash = row.get('resolved_url_hash') or _resolved_url_hash(
            row.get('resolved_url') or row.get('source_url') or recipe_id)

        def _j(v):
            if v is None:
                return None
            return v if isinstance(v, str) else json.dumps(v)

        with conn.cursor() as cur:
            # Copy the per-owner timestamps verbatim — shared must show the same
            # "saved on" time as the source table, never the mirror time.
            cur.execute(
                f"SELECT `_createdDate`, `_updatedDate` FROM `{_safe_owner_token(owner)}_saved_recipes` WHERE `_id` = %s",
                (recipe_id,),
            )
            ts = cur.fetchone() or {}
            cur.execute(
                f"""INSERT INTO `{_SHARED_SAVED_RECIPES_TABLE}` (
                        owner_id, _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                        title, image_url, image_urls, source_image_url, source_image_urls, image_storage_key,
                        ingredients, instructions, notes, raw_caption, raw_content,
                        extraction_source, recipe_source_used, author_name, caption_field, status, meal_category,
                        _createdDate, _updatedDate
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s,NOW()),%s)
                    ON DUPLICATE KEY UPDATE
                        source_url=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(source_url), source_url),
                        resolved_url=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(resolved_url), resolved_url),
                        resolved_url_hash=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(resolved_url_hash), resolved_url_hash),
                        title=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(title), title), image_url=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(image_url), image_url), image_urls=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(image_urls), image_urls),
                        source_image_url=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(source_image_url), source_image_url), source_image_urls=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(source_image_urls), source_image_urls),
                        image_storage_key=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(image_storage_key), image_storage_key), ingredients=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(ingredients), ingredients),
                        instructions=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(instructions), instructions), notes=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(notes), notes), raw_caption=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(raw_caption), raw_caption),
                        raw_content=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(raw_content), raw_content), extraction_source=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(extraction_source), extraction_source),
                        recipe_source_used=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), COALESCE(VALUES(recipe_source_used), recipe_source_used), recipe_source_used),
                        author_name=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(author_name), author_name), caption_field=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(caption_field), caption_field),
                        status=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(status), status), meal_category=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(meal_category), meal_category),
                        _createdDate=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(_createdDate), _createdDate), _updatedDate=IF(owner_id=VALUES(owner_id) AND _id=VALUES(_id), VALUES(_updatedDate), _updatedDate)""",
                (
                    owner, row.get('_id'), row.get('_owner'), row.get('source_type'), row.get('source_url'),
                    row.get('resolved_url'), resolved_url_hash, row.get('title'), row.get('image_url'),
                    _j(row.get('image_urls')), row.get('source_image_url'), _j(row.get('source_image_urls')),
                    row.get('image_storage_key'), _j(row.get('ingredients')), _j(row.get('instructions')),
                    _j(row.get('notes')), row.get('raw_caption'), row.get('raw_content'),
                    row.get('extraction_source'), row.get('recipe_source_used'),
                    row.get('author_name'), row.get('caption_field'),
                    row.get('status') or 'ready', row.get('meal_category'),
                    ts.get('_createdDate'), ts.get('_updatedDate'),
                ),
            )
        # Legacy shared tables may have a globally unique _id. Guard every
        # duplicate-key assignment above, then verify this owner's exact target.
        # A foreign ID or same-URL/different-ID collision must not edit another row.
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT owner_id, _id, title, ingredients, instructions, notes, source_url, resolved_url, resolved_url_hash, status FROM `{_SHARED_SAVED_RECIPES_TABLE}` WHERE owner_id=%s AND _id=%s",
                [owner, recipe_id],
            )
            confirmed = cur.fetchone()
        if not confirmed:
            raise ValueError('Saved recipe projection identity collision')
        for field in ('source_url','resolved_url','resolved_url_hash','status'):
            expected = resolved_url_hash if field=='resolved_url_hash' else (row.get('status') or 'ready') if field=='status' else row.get(field)
            if confirmed.get(field) != expected:
                raise ValueError('Saved recipe projection did not retain source/status')
        if confirmed.get('title') != row.get('title'):
            raise ValueError('Saved recipe projection did not retain the edited title')
        for field in ('ingredients', 'instructions', 'notes'):
            expected = row.get(field)
            actual = confirmed.get(field)
            expected = json.loads(expected) if isinstance(expected, str) else expected
            actual = json.loads(actual) if isinstance(actual, str) else actual
            if actual != expected:
                raise ValueError('Saved recipe projection did not retain the edited content')
        if not transactional:
            conn.commit()
    except Exception as exc:
        # LOUD, metric-filterable miss marker. The dual-write is non-blocking/
        # swallowed, so the parity count is our only signal a mirror is missing —
        # surface every miss to stderr under a stable {evt:'dual_write_miss'}
        # shape so a CloudWatch metric filter alarms without waiting for the
        # daily parity check.
        try:
            print(json.dumps({
                'evt': 'dual_write_miss',
                'family': 'saved_recipes',
                'owner_id': str(owner) if owner is not None else None,
                'recipe_id': str(recipe_id) if recipe_id is not None else None,
                'error': (str(exc)[:500] if exc is not None else ''),
            }), file=sys.stderr)
        except Exception:
            pass
        _log_event(request_id, 'saved_recipe_shared_dualwrite_failed', owner=owner, recipe_id=recipe_id, error=str(exc))

        if transactional:
            raise


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
    member_hash = row.get('resolved_url_hash') or _resolved_url_hash(
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
            # Mirror the MEMBER's delete too. _delete_saved_recipe mirrors the ACTING owner's
            # row, but this fan-out only ever touched the members' per-owner tables — so every
            # household member's shared_saved_recipes row survived the delete and, now that
            # READ_SHARED_SAVED_RECIPES is true, reads come FROM that mirror: the recipe
            # reappears for the household members who didn't press delete. Same pattern, same
            # flag gate, same non-blocking contract as the acting-owner delete above.
            if DUAL_WRITE_SAVED_RECIPES:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM `{_SHARED_SAVED_RECIPES_TABLE}` WHERE owner_id = %s AND _id = %s LIMIT 1",
                            [member_id, recipe_id],
                        )
                    conn.commit()
                except Exception as exc:
                    print(json.dumps({'evt': 'dual_write_miss',
                                      'family': 'saved_recipes_delete_fanout',
                                      'owner_id': str(member_id), 'recipe_id': str(recipe_id),
                                      'error': str(exc)[:500]}), file=sys.stderr)
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
    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN {ddl}")
        conn.commit()
    except pymysql.err.OperationalError as exc:
        # 1060 = ER_DUP_FIELDNAME. The has-column check above then the ALTER is check-then-act:
        # two concurrent requests for the same owner (the app fires several /saved-recipes calls
        # on load; autocommit=True so this is genuine concurrency, not a stale snapshot) can both
        # pass the check on a fresh table and both fire the ALTER — the loser gets 1060. The
        # column now exists (the desired end state), so treat it as success. Re-raise anything
        # else (a bad DDL is ProgrammingError; a missing table is 1146 — neither matches here).
        if exc.args and exc.args[0] == 1060:
            return
        raise


_PROCESSING_STATUS_READY = set()


def _ensure_saved_recipe_processing_status(conn, table):
    """Append the missing enum value without rewriting existing status values.

    This runs before save transactions. The old production SQL mode silently
    coerced 'processing' to '', so the worker never recognized accepted text.
    """
    if table in _PROCESSING_STATUS_READY:
        return True
    with conn.cursor() as cur:
        cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE 'status'")
        column = cur.fetchone() or {}
        kind = column.get('Type') or ''
        if kind.lower().startswith('enum(') and "'processing'" not in kind.lower():
            expanded = kind[:-1] + ",'processing')"
            cur.execute("SELECT @@SESSION.lock_wait_timeout AS timeout")
            previous_timeout = cur.fetchone()['timeout']
            try:
                cur.execute("SET SESSION lock_wait_timeout=1")
                cur.execute(f"ALTER TABLE `{table}` MODIFY COLUMN `status` {expanded} NOT NULL DEFAULT 'ready', ALGORITHM=INPLACE, LOCK=NONE")
            except pymysql.err.OperationalError as exc:
                if exc.args[0] in (1205, 1213):
                    return False  # Use the synchronous save path; never wait out the HTTP deadline.
                raise
            finally:
                cur.execute("SET SESSION lock_wait_timeout=%s", (previous_timeout,))
    _PROCESSING_STATUS_READY.add(table)
    return True


def _text_processing_schema_ready(conn, owner):
    if not _ensure_saved_recipe_processing_status(conn, _saved_recipes_table(owner)):
        return False
    return not DUAL_WRITE_SAVED_RECIPES or _ensure_saved_recipe_processing_status(conn, _SHARED_SAVED_RECIPES_TABLE)


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
                status ENUM('ready', 'failed', 'processing') NOT NULL DEFAULT 'ready',
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
    # Meal-category grouping in "My Recipes" (breakfast/lunch/dinner/snacks/other).
    _ensure_column(conn, table, 'meal_category', "`meal_category` VARCHAR(16) NULL AFTER `status`")
    # WHICH source text the recipe was actually built from: content, content+audio,
    # content+slide_ocr, content+image_ocr(+audio), or explore.
    #
    # The pipeline has always computed this and then discarded it, which is what made the
    # repair agent dangerous. Repair judged "was this fabricated?" by re-testing the CAPTION,
    # while `_analyze_extraction` legitimately builds from caption PLUS audio transcript — so
    # a recipe correctly recovered from speech looked ungrounded and got wiped. That cost 8
    # rows once and 3 again the same day, 3 of them unrecoverable. Recording the fact removes
    # the guess.
    _ensure_column(conn, table, 'recipe_source_used',
                   "`recipe_source_used` VARCHAR(48) NULL AFTER `extraction_source`")



# Same 4-value taxonomy the recipe generator uses for "Use What I Have", so saved
# and generated recipes group consistently. 'other' is a last-resort bucket only.
RECIPE_MEAL_CATEGORIES = ('breakfast', 'lunch', 'dinner', 'snacks')


def _meal_category_heuristic(title, ingredients):
    """Keyword fallback used only when the LLM classify call fails. Returns one of
    the 4 categories or None (never a round-robin guess — a wrong label is worse
    than no label, which renders as 'other')."""
    hay = ' '.join([str(title or '').lower()] + [str(i or '').lower() for i in (ingredients or [])])
    keyword_groups = [
        ('breakfast', ('breakfast', 'omelet', 'omelette', 'oatmeal', 'pancake', 'waffle', 'bagel',
                       'french toast', 'cereal', 'granola', 'frittata', 'hash brown', 'scramble', 'egg')),
        ('snacks', ('snack', 'dessert', 'cookie', 'brownie', 'cake', 'muffin', 'dip', 'bites',
                    'energy ball', 'protein ball', 'trail mix', 'parfait', 'popcorn', 'chips',
                    'cracker', 'smoothie', 'granola bar')),
        ('lunch', ('lunch', 'sandwich', 'wrap', 'salad', 'soup', 'quesadilla', 'panini')),
        ('dinner', ('dinner', 'pasta', 'curry', 'stir-fry', 'stir fry', 'skillet', 'roast',
                    'casserole', 'lasagna', 'risotto', 'chili', 'enchilada', 'steak')),
    ]
    for category, keywords in keyword_groups:
        if any(k in hay for k in keywords):
            return category
    return None


def _classify_meal_category(title, ingredients, request_id=None):
    """Classify a recipe into one meal category. LLM-primary (accurate), heuristic
    fallback on LLM failure, 'other' as a last resort. Used both on save and by the
    one-time backfill."""
    title = _safe_text(title)
    ings = [_safe_text(i) for i in (ingredients or []) if _safe_text(i)]
    if not title and not ings:
        return 'other'
    try:
        client = _openai_client()
        model = os.getenv('OPENAI_MEAL_CATEGORY_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        system = (
            "You classify a recipe into exactly one meal category. "
            "Respond ONLY with JSON: {\"meal_category\": \"<value>\"} where value is one of: "
            "breakfast, lunch, dinner, snacks. Use 'snacks' for desserts, sweets, baked goods, "
            "dips, drinks, and small bites. Pick the single best fit."
        )
        user = f"Title: {title}\nIngredients: {', '.join(ings[:20])}"
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        )
        payload = json.loads(_strip_code_fences(response.choices[0].message.content) or '{}')
        category = _safe_text(payload.get('meal_category')).lower()
        if category in RECIPE_MEAL_CATEGORIES:
            return category
    except Exception as exc:
        _log_event(request_id, 'meal_category_llm_failed', error=str(exc))
    return _meal_category_heuristic(title, ings) or 'other'


def _apply_saved_recipe_meal_category(conn, owner, recipe_id, title, ingredients, request_id=None, provided_category=None):
    """Classify + persist meal_category onto the owner's saved-recipe row. Called
    right BEFORE the shared dual-write so the mirror picks it up. Non-blocking: a
    failure leaves meal_category NULL (renders as 'other'; backfill/next-write fixes).

    If provided_category is already a valid category (e.g. a CLAIM reusing the share
    snapshot's stored meal_category), use it directly and SKIP the classify LLM — no
    reason to re-derive what we already have. Only the prestructured/claim path passes
    it; normal URL/text/image saves pass None and still classify via LLM."""
    try:
        provided = _safe_text(provided_category).lower()
        if provided in RECIPE_MEAL_CATEGORIES:
            category = provided
        else:
            try:
                with recipe_work_budget.phase(max_seconds=6):
                    category = _classify_meal_category(title, ingredients, request_id=request_id)
            except recipe_work_budget.WorkLimit:
                category = _meal_category_heuristic(title, ingredients) or 'other'
        table = _saved_recipes_table(owner)
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE `{table}` SET meal_category = %s WHERE _id = %s AND _owner = %s",
                (category, recipe_id, owner),
            )
        conn.commit()
        return category
    except Exception as exc:
        _log_event(request_id, 'meal_category_persist_failed', owner=owner, recipe_id=recipe_id, error=str(exc))
        return None


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


def _canonical_explore_source_url(value):
    """Put the content ID in the path for clients that discard URL queries."""
    from urllib.parse import urlparse, parse_qs
    import uuid
    if not isinstance(value, str):
        return value
    try:
        parsed = urlparse(value)
        if (parsed.hostname or '').lower() not in ('instagram.com', 'www.instagram.com') or parsed.path.rstrip('/').lower() != '/trepohq':
            return value
        candidate = (parse_qs(parsed.query).get('trepo_recipe') or [''])[0]
        return 'https://trepo.ai/explore/' + str(uuid.UUID(candidate))
    except (ValueError, TypeError):
        return value


def _serialize_row(row):
    out = {k: json_serial(v) for k, v in row.items()}
    out['id'] = out.pop('_id', None)
    out['ingredients'] = _parse_json_field(row.get('ingredients'))
    out['instructions'] = _parse_json_field(row.get('instructions'))
    out['notes'] = _parse_json_field(row.get('notes'))
    out['image_urls'] = _parse_json_field(row.get('image_urls'))
    out['source_image_urls'] = _parse_json_field(row.get('source_image_urls'))
    for field in ('source_url', 'resolved_url'):
        out[field] = _canonical_explore_source_url(out.get(field))
    out['platform'] = out.get('source_type')
    out['content'] = out.get('raw_content') or out.get('raw_caption') or None
    out['caption'] = out.get('raw_caption') or out.get('raw_content') or None
    out['content_outcome'] = recipe_content_outcome(out)
    out['content_revision'] = _saved_recipe_content_revision(row)
    if out['content_outcome'] != 'ready':
        out['recovery'] = recipe_content_recovery(out)
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
                    WHERE owner = %s AND recipe_source = %s AND recipe_id IN ({placeholders})""",
                [safe_owner, _availability_cache_source('saved')] + safe_ids,
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
                SELECT `_id`, `product_name`, `product_description`, `quantity_value`, `quantity_unit`
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
                  AND column_name IN ('product_description', 'analysis_stage', 'analysis_status', 'quantity', 'quantity_value', 'quantity_unit')
            """, [table_name])
            column_names = {
                (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
                for row in (cur.fetchall() or [])
                if (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
            }
            where_parts = ["action = 'IN'"]
            # Gate on analysis_status only, NOT analysis_stage: voice check-in items rest
            # at analysis_stage='preliminary' forever (no capture image to promote them),
            # so the stage clause hid them from the saved-recipe inventory for voice users.
            if 'analysis_status' in column_names:
                where_parts.append("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)")
            select_fields = "`_id`, `product_name`"
            if 'product_description' in column_names:
                select_fields += ", `product_description`"
            for field in ('quantity', 'quantity_value', 'quantity_unit'):
                if field in column_names:
                    select_fields += f", `{field}`"
            cur.execute(f"""
                SELECT {select_fields}
                FROM `{table_name}`
                WHERE {' AND '.join(where_parts)}
                ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC, `_createdDate` DESC
            """)
            rows = cur.fetchall() or []
    items = []
    for row in rows:
        display_name = _safe_text(row.get('product_name'))
        if not display_name:
            continue
        description = _safe_text(row.get('product_description')) or None
        name_parts = [display_name]
        if description:
            name_parts.append(description)
        tokens = _ingredient_tokens(' '.join(name_parts))
        items.append({
            'item_id': _safe_text(row.get('_id')) or None,
            'display_name': display_name,
            'quantity': row.get('quantity'),
            'quantity_value': row.get('quantity_value'),
            'quantity_unit': row.get('quantity_unit'),
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
    try:
        with recipe_work_budget.phase():
            availability_map, _ = recipe_inventory_llm.match_recipes_fast(
                [recipe_payload],
                kitchen_context,
                request_id=request_id,
                log_fn=_log_inventory_match_event,
                source_label='saved',
                substitution_callback=_suggest_saved_recipe_substitutions,
                inline_single=True,
            )
            result = availability_map.get(recipe_payload['id']) or recipe_inventory_llm.deterministic_availability(
                recipe_payload,
                kitchen_context,
                substitution_callback=_suggest_saved_recipe_substitutions,
            )
    except recipe_work_budget.WorkLimit:
        result = recipe_inventory_llm.deterministic_availability(recipe_payload, kitchen_context)
    return _local_deterministic_merge(result, kitchen_context)


def _local_deterministic_merge(availability, kitchen_context):
    """Safety-net merge: if deterministic matching finds a 'have' that the LLM missed, upgrade it."""
    ingredient_matches = availability.get('ingredient_matches') or []
    kitchen_candidates = (kitchen_context or {}).get('kitchen_candidates') or []
    changed = False
    for match in ingredient_matches:
        if match.get('match_status') != 'missing' or match.get('quantity_status') == 'insufficient':
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
    # Older layer versions remain deployable during the additive rollout.
    enforce = getattr(recipe_inventory_llm, 'enforce_quantity_availability', None)
    if enforce and ingredient_matches:
        return enforce({'ingredients': [m['recipe_ingredient'] for m in ingredient_matches]}, kitchen_context, availability)
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
    # An EMPTY kitchen has exactly one correct answer: nothing is on hand. Running the LLM
    # matcher against no inventory cannot improve on that, and it made the result unstable -
    # a user with an empty kitchen reported the same recipe reading "0 ingredients on hand"
    # one moment and "some other random amount" the next. Answer it deterministically and
    # skip both the matcher and the substitution calls (there is nothing to substitute WITH).
    # This is not a rare edge case: ~58% of households currently have an empty kitchen.
    if not ((kitchen_context or {}).get('kitchen_candidates') or []):
        return {
            payload['id']: recipe_inventory_llm.deterministic_availability(
                payload, kitchen_context, None)
            for payload in recipe_payloads
        }

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
        'recipe_source': _availability_cache_source(_safe_text(recipe_source)),
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
    persist_quantities = getattr(recipe_inventory_llm, 'persist_quantity_availability_rows', None)
    if persist_quantities:
        persist_quantities(conn, rows)
        return
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
            f"DELETE FROM `{_OWNER_RECIPE_AVAILABILITY_TABLE}` WHERE owner = %s AND recipe_source IN (%s, %s) AND recipe_id = %s",
            [_safe_text(owner), _safe_text(recipe_source), _availability_cache_source(_safe_text(recipe_source)), _safe_text(recipe_id)],
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
    preserve = getattr(recipe_inventory_llm, 'confirmed_availability_for_response', None)
    if preserve:
        availability = preserve(conn, owner, 'saved', recipe.get('id'), availability)
    recipe['availability'] = _availability_summary(availability)
    recipe['ingredient_matches'] = availability['ingredient_matches']
    recipe['missing_ingredients'] = availability['missing_ingredients']
    recipe['substitution_candidates'] = availability['substitution_candidates']
    recipe['substitution_summary'] = availability['substitution_summary']
    recipe['substitution_status'] = availability['substitution_status']
    overlay_row = _build_owner_recipe_availability_record(owner, 'saved', recipe.get('id'), availability)
    return recipe, overlay_row


def _sha256(value):
    return hashlib.sha256(str(value or '').encode('utf-8')).hexdigest()


# Instagram share links carry a per-share ?igsh= token, so the SAME reel shared twice
# yields two different URLs and exact-string dedupe can NEVER fire. Reduce any Instagram
# content URL to a stable identity key built from its shortcode, ignoring /reel/ vs /p/
# (Instagram serves the same content under both) and dropping query + fragment.
_INSTAGRAM_SHORTCODE_RE = re.compile(r'^/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)')


def _dedupe_key_for_url(url):
    """Identity key used for duplicate detection. Non-Instagram URLs are returned
    unchanged so their existing hashes stay valid."""
    text = _safe_text(url)
    if not text:
        return text
    try:
        parsed = urlparse(text)
    except ValueError:
        return text
    host = (parsed.netloc or '').lower()
    if not (host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com')):
        return text
    match = _INSTAGRAM_SHORTCODE_RE.match(parsed.path or '')
    if not match:
        return text
    return f'instagram:{match.group(1)}'


def _resolved_url_hash(resolved_url):
    """Hash written on NEW rows: the canonical identity key."""
    return _sha256(_dedupe_key_for_url(resolved_url))


def _resolved_url_hash_candidates(resolved_url):
    """Hashes to CHECK when deciding whether we already hold this recipe. The canonical
    key first, then the legacy raw-URL hash, so rows written before canonicalization
    still dedupe instead of silently duplicating. Order matters only for readability —
    both are looked up."""
    keys = []
    canonical = _dedupe_key_for_url(resolved_url)
    raw = _safe_text(resolved_url)
    for key in (canonical, raw):
        if key and key not in keys:
            keys.append(key)
    return [_sha256(key) for key in keys]


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
        'outcome': ('partial' if job.get('partial_errors') else 'complete')
                   if job.get('status') == _JOB_STATUS_COMPLETED else _safe_text(job.get('status')).lower(),
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
    item = _jobs_table().get_item(Key={'job_id': job_id}, ConsistentRead=True).get('Item')
    if not item or _safe_text(item.get('type')) != _SAVED_RECIPE_BATCH_JOB_TYPE:
        return None
    return item


def _put_saved_recipe_batch_job(job):
    _jobs_table().put_item(Item=job)
    return job


# Retention for FAILED jobs. The submission payload (and, for image saves, the uploaded
# photos it references) is what makes a failed save re-drivable — it is deliberately kept
# so a fix-then-sweep can replay it. But users submitted these intending a saved recipe,
# not indefinite storage of failures, so failed job records carry a TTL and age out.
# Successful jobs are untouched by this.
_FAILED_JOB_TTL_DAYS = max(1, int(os.getenv('FAILED_JOB_TTL_DAYS', '30')))


def _failed_job_ttl_epoch():
    return int(time.time()) + (_FAILED_JOB_TTL_DAYS * 86400)


# BUILD_STAMP is written into the deployment zip by scripts/deploy_overlay.sh at deploy
# time (deploy timestamp + sha256 of app.py). We deploy to $LATEST without publishing
# versions, so AWS_LAMBDA_FUNCTION_VERSION is the literal string "$LATEST" on every build
# and answers nothing — on 2026-07-25 five deploys shared it. The stamp is what actually
# identifies which build handled a job.
#
# Read once at cold start and FAIL SOFT: a missing or unreadable stamp records
# 'unstamped' and must never break the function. An unstamped build is a small
# observability loss; a build that won't import is an outage.
def _read_build_stamp():
    try:
        stamp_path = Path(__file__).resolve().parent / 'BUILD_STAMP'
        if not stamp_path.exists():
            return 'unstamped'
        return _safe_text(stamp_path.read_text(encoding='utf-8')).splitlines()[0][:120] or 'unstamped'
    except Exception:
        return 'unstamped'


_BUILD_STAMP = _read_build_stamp()


def _build_identity():
    """Which build handled this job. Today proved we need to know WHICH deploy failed a
    save: three deploys landed within 25 minutes and the only way to attribute a failure
    was to correlate timestamps by hand."""
    return {
        'build_stamp': _BUILD_STAMP,
        'function_version': _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_VERSION')),
        'log_stream': _safe_text(os.getenv('AWS_LAMBDA_LOG_STREAM_NAME')),
    }


def _update_saved_recipe_batch_job(job_id, **fields):
    job = _get_saved_recipe_batch_job(job_id)
    if not job:
        raise ServiceError('Saved recipe batch job not found.', status_code=404)
    job.update(fields)
    job['updated_at'] = _utc_now_iso()
    # On a terminal failure, stamp the job so it is re-drivable AND attributable:
    # where the original input lives, what failed, and which build failed it.
    if _safe_text(job.get('status')) == _JOB_STATUS_FAILED:
        identity = _build_identity()
        job.setdefault('failed_at', job['updated_at'])
        # Prefer the deploy stamp; function_version is "$LATEST" for every build.
        job['failed_build'] = identity.get('build_stamp') or 'unstamped'
        job['failed_function_version'] = identity.get('function_version')
        job['failed_log_stream'] = identity.get('log_stream')
        # input_refs makes the re-drive contract explicit instead of implicit: the
        # submission JSON is never deleted and image objects it points at survive, so
        # these two fields are all a sweep needs.
        job['input_refs'] = {
            'payload_s3_key': _safe_text(job.get('payload_s3_key')),
            'kind': _safe_text(job.get('submission_kind')),
        }
        job['ttl'] = None if job.get('admission_operation_id') else (job.get('ttl') or _failed_job_ttl_epoch())
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


_HERO_IMAGE_MAX_ASPECT = 1.2


def _normalize_hero_image_bytes(image_bytes, content_type=None):
    """Web-scraped recipe hero images are usually landscape Open-Graph cards (e.g.
    1200x630, aspect ~1.90). A wide hero inflates the recipe-detail layout past screen
    width on older iOS builds (renders zoomed/clipped). If aspect (w/h) exceeds 1.2 we
    center-crop to a 1:1 square before storing — square is the safest bound: it can never
    exceed the screen width regardless of layout. Aspect <= 1.2 (already near-square or
    portrait) is stored byte-for-byte unchanged. Fully error-safe: any decode/encode
    failure returns the ORIGINAL bytes so a save never crashes on an odd image."""
    if not image_bytes:
        return image_bytes, content_type
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with Image.open(BytesIO(image_bytes)) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if not width or not height:
                return image_bytes, content_type
            if (width / height) <= _HERO_IMAGE_MAX_ASPECT:
                return image_bytes, content_type
            # Landscape hero: center-crop to a square on the shorter side.
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            square = oriented.crop((left, top, left + side, top + side))
            if square.mode != 'RGB':
                square = square.convert('RGB')
            output = BytesIO()
            square.save(output, format='JPEG', quality=92)
            return output.getvalue(), 'image/jpeg'
    except Exception:
        # Never let hero normalization break a save — fall back to the original bytes.
        return image_bytes, content_type


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
        image_bytes, content_type, resolved_url = _load_remote_image(
            source_url, timeout_seconds=_IMAGE_MIRROR_TIMEOUT_SECONDS,
            deadline_seconds=_IMAGE_MIRROR_TIMEOUT_SECONDS)
        # The card thumbnail may be cropped; keep the complete source separately
        # so image zoom can show the original recipe without losing its edges.
        original_url = _upload_saved_recipe_source_image(
            owner, recipe_id, image_bytes, content_type, request_id=request_id)
        # Normalize wide landscape heroes to a square before storing so they can't blow out
        # the recipe-detail layout on older iOS builds. No-op for near-square/portrait; on
        # any decode error the original bytes pass through unchanged.
        image_bytes, content_type = _normalize_hero_image_bytes(image_bytes, content_type)
        ext = _guess_image_extension(content_type, resolved_url)
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
            'source_image_urls': [original_url],
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
    return recipe_batch_checkpoint.bounded_client(OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=_OPENAI_TIMEOUT_SECONDS,
        max_retries=_OPENAI_MAX_RETRIES,
    ))


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
    # Attribution hints. Deliberately NOT part of provided_modes below: they describe where
    # already-extracted content came from, they are not another way of supplying a recipe.
    source_url_hint = _safe_text(payload.get('source_url') or payload.get('sourceUrl'))
    source_image_hint = _safe_text(payload.get('source_image_url') or payload.get('sourceImageUrl'))
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
            'source_url': source_url_hint,
            'image_url': source_image_hint,
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


def _text_hero_image(extraction):
    """The real photo a text save was handed, if any. Only http(s) — never a synthetic URL."""
    url = _safe_text((extraction or {}).get('image_url'))
    return url if url.lower().startswith(('http://', 'https://')) else None


def _manual_recipe_url(kind, fingerprint):
    return f"app://saved-recipes/{kind}/{fingerprint}"


def _build_text_recipe_extraction(content, source_url=None, image_url=None):
    """Recipe from text the caller already holds.

    `source_url` is an attribution hint, not a fetch instruction: the caller is saving
    content it has already extracted (and, for the personalised Explore feed, already run
    through the dietary filter), so nothing here re-reads the page. Recording the real URL
    instead of the synthetic app:// one keeps the saved recipe pointing at where it came
    from, lets it dedupe against the same recipe saved by URL, and lets the client recognise
    it as saved — matching on a synthetic hash it cannot compute is why bookmarks on
    agent-found recipes lit up and then went dark again.
    """
    source_text = _safe_text(content)
    if not source_text:
        raise ServiceError('Provide recipe text.', status_code=400)
    fingerprint = _sha256(source_text)
    synthetic_url = _manual_recipe_url('text', fingerprint)
    attributed = _safe_text(source_url)
    # Only trust a real http(s) URL; anything else falls back to the synthetic one.
    if attributed and not attributed.lower().startswith(('http://', 'https://')):
        attributed = ''
    hero = _safe_text(image_url)
    return {
        'url': attributed or synthetic_url,
        'resolved_url': attributed or synthetic_url,
        'platform': 'text',
        'content': source_text,
        'caption': source_text,
        'title': '',
        'image_url': hero,
        'image_urls': [hero] if hero else [],
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


_GENERIC_IMAGE_TYPES = {'', 'application/octet-stream', 'binary/octet-stream', 'application/binary'}

def _verified_generic_image_type(image_bytes):
    """Trust a decoded raster format, never the URL extension or binary MIME label."""
    from PIL import Image
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > 25_000_000:
                raise ServiceError('This image has too many pixels.', status_code=413)
            content_type = Image.MIME.get(image.format, '')
            if image.format not in {'JPEG', 'PNG', 'GIF', 'WEBP', 'AVIF', 'HEIF'}:
                raise ValueError('Unsupported raster format')
            image.load()  # Reject corrupt/truncated bodies before uploading to owned storage.
            return content_type
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError('The source did not return a valid image.', status_code=422,
                           extra={'error_code': 'invalid_image'}) from exc


def _load_remote_image(image_url, *, timeout_seconds=None, deadline_seconds=20):
    response = None
    try:
        response = requests.get(
            image_url,
            timeout=_REQUEST_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
            stream=True,
            allow_redirects=True,
            headers={'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        )
        response.raise_for_status()
        content_type = _safe_text(response.headers.get('Content-Type')).split(';', 1)[0].lower()
        if not content_type.startswith('image/') and content_type not in _GENERIC_IMAGE_TYPES:
            raise ServiceError(f'Unexpected image content type: {content_type or "unknown"}', status_code=400)
        limit = 25 * 1024 * 1024
        try:
            advertised_size = int(response.headers.get('Content-Length') or 0)
        except (TypeError, ValueError):
            advertised_size = 0
        if advertised_size > limit:
            raise ServiceError('This image is too large. Please choose an image under 25 MB.', status_code=413)
        chunks, size = [], 0
        deadline = time.monotonic() + deadline_seconds
        for chunk in response.iter_content(chunk_size=65536):
            if time.monotonic() > deadline:
                raise ServiceError('The image download took too long. Please try again.', status_code=503)
            size += len(chunk)
            if size > limit:
                raise ServiceError('This image is too large. Please choose an image under 25 MB.', status_code=413)
            chunks.append(chunk)
        image_bytes = b''.join(chunks)
        if not image_bytes:
            raise ServiceError('Downloaded image was empty.', status_code=400)
        if content_type in _GENERIC_IMAGE_TYPES:
            content_type = _verified_generic_image_type(image_bytes)
        return image_bytes, _normalize_image_content_type(content_type), response.url or image_url
    except requests.HTTPError as exc:
        status = getattr(exc.response, 'status_code', None)
        if status in (403, 404, 410):
            raise ServiceError('This image is unavailable at its source. Please upload it again.',
                               status_code=422, extra={'error_code': 'source_unavailable'}) from exc
        if status in (408, 429):
            raise ServiceError('The image provider is temporarily busy. Please try again.',
                               status_code=status, extra={'error_code': 'provider_busy'}) from exc
        raise ServiceError('The image provider is temporarily unavailable. Please try again.',
                           status_code=502, extra={'error_code': 'provider_unavailable'}) from exc
    except requests.RequestException as exc:
        raise ServiceError('The image could not be downloaded. Please try again.',
                           status_code=502, extra={'error_code': 'provider_unavailable'}) from exc
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

    if submission.get('_expected_sha256') is not None:
        if len(image_bytes) != submission.get('_expected_size') or _sha256_bytes(image_bytes) != submission['_expected_sha256']:
            raise ServiceError('The uploaded photo changed. Please import the original image again.', status_code=409)

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


def _is_uniform_recipe_image(image_bytes):
    """Reject only provably empty static rasters, not faint text or sparse pages."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if getattr(image, 'n_frames', 1) != 1:
                return False
            if image.width * image.height > 25_000_000:
                raise ServiceError('This image has too many pixels.', status_code=413)
            with image.convert('RGBA') as pixels:
                extrema = pixels.getextrema()
                # Fully transparent pixels cannot supply visible recipe information.
                return extrema[3] == (0, 0) or all(low == high for low, high in extrema)
    except (UnidentifiedImageError, OSError):
        # This guard is not the format validator; retain the existing decoder path.
        return False



def _extract_recipe_fragment_from_image(prepared_image, request_id=None, image_index=None,
                                        escalate_on_not_recipe=True):
    if _is_uniform_recipe_image(prepared_image['image_bytes']):
        raise ServiceError('Not enough recipe information. Try a clearer photo with ingredients and steps.', status_code=422)
    client = _openai_client()
    image_reference = f"data:{prepared_image['content_type']};base64,{base64.b64encode(prepared_image['image_bytes']).decode('ascii')}"
    # A transient OpenAI blip was flagging VALID recipe images as not_recipe → the whole
    # save failed permanently (real case: a Kimchi Beef Stew recipe rejected ~11x, then
    # read fine on retry). Retry the vision model, then FAIL OVER to the flagship (a
    # stronger read) before giving up — only a genuinely non-recipe image reaches the 422.
    fallback_model = os.getenv('OPENAI_RECIPE_VISION_FALLBACK_MODEL', 'gpt-5.4-2026-03-05')
    attempts = [(_OPENAI_VISION_MODEL, 0.0), (_OPENAI_VISION_MODEL, 0.5), (fallback_model, 0.4)]
    last_reason = 'not_recipe'
    provider_failures = 0
    for attempt_idx, (model, delay) in enumerate(attempts):
        if delay:
            time.sleep(delay)
        started = time.time()
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': _recipe_image_transcription_prompt()},
                    {'role': 'user', 'content': [
                        {'type': 'text', 'text': 'Extract the recipe from this image.'},
                        {'type': 'image_url', 'image_url': {'url': image_reference}},
                    ]},
                ],
            }
            # gpt-5.x / o-series reject a custom temperature; only set it for the 4.x model.
            if not str(model).startswith('gpt-5') and not str(model).startswith('o'):
                kwargs['temperature'] = 0.1
            response = client.chat.completions.create(**kwargs)
            payload = json.loads(_strip_code_fences(response.choices[0].message.content))
        except Exception as exc:
            provider_failures += 1
            last_reason = type(exc).__name__
            _log_event(request_id, 'saved_recipe_image_extract_retry', model=model,
                       image_index=image_index, attempt=attempt_idx + 1, failure_reason=last_reason)
            continue
        content = _safe_text(payload.get('transcribed_text'))
        if not bool(payload.get('not_recipe')) and content:
            author_name = _safe_text(payload.get('author_name')) or None
            _log_event(request_id, 'saved_recipe_image_extract_success', model=model,
                       image_index=image_index, attempt=attempt_idx + 1,
                       latency_ms=int((time.time() - started) * 1000))
            return {
                'image_index': image_index,
                'title_hint': _safe_text(payload.get('title_hint')),
                'author_name': author_name,
                'content': content,
                'fingerprint': prepared_image['fingerprint'],
                'source_image_url': prepared_image.get('source_image_url'),
                'prepared_image': prepared_image,
            }
        last_reason = 'not_recipe' if payload.get('not_recipe') else 'empty_transcription'
        _log_event(request_id, 'saved_recipe_image_extract_retry', model=model,
                   image_index=image_index, attempt=attempt_idx + 1, failure_reason=last_reason)
        # A confident "this isn't a recipe" on a VIDEO COVER FRAME is almost always right,
        # and re-asking twice more cannot turn a photo of a person in a kitchen into a recipe
        # card. Those six doomed model calls were eating the whole request budget, so audio
        # never ran — and audio is the only rung that recovers the steps for a talking video.
        # The escalation still applies in full to user-uploaded images, where the picture
        # really is a recipe and a transient blip is the likely explanation.
        if not escalate_on_not_recipe and last_reason == 'not_recipe':
            break
    # Provider outages and malformed output are not evidence that the photo is unsuitable.
    if provider_failures:
        raise ServiceError('The recipe reader is temporarily unavailable. Please try again.', status_code=503)
    # All completed attempts agree it's not a usable recipe.
    raise ServiceError('Not enough recipe information.', status_code=422)


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


@recipe_work_budget.bounded
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
                escalate_on_not_recipe=False,
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


# A single recipe photographed across pages is 1-3 images in practice (recipe page +
# continuation, or page + finished-dish shot). Nothing in the grouping call constrained
# cluster size, so the model could return ONE cluster spanning every submitted image; the
# merged text then refined into a single unusable recipe and the whole batch 422'd.
# Observed 2026-07-25: four separate 8-image batches failed this way on three attempts
# each, while a 5-image batch from the same user succeeded — deterministic at-cap failure,
# not nondeterminism.
_MAX_IMAGES_PER_RECIPE_CLUSTER = max(1, int(os.getenv('MAX_IMAGES_PER_RECIPE_CLUSTER', '3')))


def _split_oversized_cluster(items, limit):
    """Break an oversized cluster into chunks of at most `limit` items. Splits rather than
    drops, so no submitted image is ever lost."""
    return [items[i:i + limit] for i in range(0, len(items), limit)]


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
            # Deterministic grouping: identical retries should group identically, so a
            # re-drive is a genuine retry rather than a fresh roll of the dice.
            temperature=0,
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
        if recipe_batch_checkpoint._deadline.get() is not None:
            raise ServiceError('Recipe page grouping is temporarily unavailable.', status_code=503) from exc
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

    # NOTE: grouping deliberately does NOT second-guess a single all-images cluster.
    # A sample of 507 real multi-image batches showed 51% are ONE recipe photographed
    # across several pages, so "one big cluster" is most often CORRECT. Pre-emptively
    # splitting it would break the majority case. If the merged extraction turns out to
    # yield no recipe, the SAVE path falls back to per-image (see
    # _save_recipe_image_clusters) — a fallback on evidence, not a guess from image count.
    #
    # CAP EXEMPTION for that same shape: a SINGLE cluster spanning EVERY image must not be
    # capped either. 48 historical batches with >3 images resolved to exactly 1 recipe, and
    # chopping one recipe into 3+3+2 would produce three partial recipes — re-introducing,
    # one layer down, the very harm merged-first removes. The cap still applies to genuine
    # multi-cluster results, where an over-large cluster really does indicate over-merging.
    if len(clusters) == 1 and len(clusters[0].get('fragments') or []) == len(fragments):
        return clusters

    capped = []
    for cluster in clusters:
        frags = list(cluster.get('fragments') or [])
        if len(frags) <= _MAX_IMAGES_PER_RECIPE_CLUSTER:
            capped.append(cluster)
            continue
        _log_event(
            request_id,
            'saved_recipe_image_cluster_capped',
            cluster_size=len(frags),
            cap=_MAX_IMAGES_PER_RECIPE_CLUSTER,
        )
        for chunk in _split_oversized_cluster(frags, _MAX_IMAGES_PER_RECIPE_CLUSTER):
            capped.append({
                'title_hint': cluster.get('title_hint') or '',
                'fragments': chunk,
            })
    return capped


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
        # Retained so a failed merged save can be retried as individual images.
        'fragments': fragments,
    }


def _extract_recipe_clusters_from_images(image_submissions, request_id=None):
    partial_errors = []
    fragments = []
    for image_index, image_submission in enumerate(image_submissions or []):
        try:
            prepared_image = _run_with_transient_retry(
                lambda: _prepare_image_for_vision(image_submission), request_id=request_id, label='prepare_recipe_image')
            fragment = _run_with_transient_retry(
                lambda: _extract_recipe_fragment_from_image(prepared_image, request_id=request_id, image_index=image_index),
                request_id=request_id, label='extract_recipe_image')
            fragments.append(fragment)
        except Exception as exc:
            partial_errors.append({
                'image_index': image_index,
                'error': str(exc) if isinstance(exc, ServiceError) else 'This image could not be processed. Please try again.',
                'status_code': exc.status_code if isinstance(exc, ServiceError) else 503,
                'retryable': _is_transient_failure(exc),
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
            # A successful recipe save must not silently lose one of its source
            # pages. This runs before the recipe INSERT; the batch can report the
            # failed cluster and retry it using the retained submission.
            raise ServiceError('The original image could not be stored. Please try again.', status_code=503) from exc
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
    # Reject pasted recipe TEXT masquerading as a URL BEFORE we try to fetch it —
    # otherwise a schemeless blob like "Chicken Parmesan: 2 lbs chicken..." gets
    # https:// prepended, passes netloc, and 502s on fetch (bogus 5xx alarm trips).
    # A real host has no whitespace and a dot-TLD (or is localhost / an IP).
    try:
        host = (parsed.hostname or '').strip().lower()
    except ValueError:
        host = ''
    _friendly_not_url = ("That doesn't look like a recipe link. Paste a webpage URL "
                         "(or use the paste-text option to add a recipe by text).")
    if (not host) or (' ' in parsed.netloc) or (
        '.' not in host and host != 'localhost' and not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host)
    ):
        raise ServiceError(_friendly_not_url, status_code=422)
    cleaned = parsed._replace(fragment='')
    normalized = urlunparse(cleaned)
    # NOTE: Instagram content-type validation is deliberately NOT done here.
    # Instagram share-sheet links (instagram.com/share/...) are short-links that
    # only reveal their real path (/reel/, /p/, ...) AFTER redirect resolution.
    # We let every Instagram URL through normalization and validate the RESOLVED
    # URL in _detect_platform (resolve-then-validate), so /share/ links work.
    return normalized


# Full modern-browser header set. Many recipe sites bot-block a bare requests
# User-Agent but pass on a complete Chrome header profile. (No brotli in
# Accept-Encoding — requests only decodes gzip/deflate without the brotli lib;
# tier-2 curl_cffi handles br natively.)
_BROWSER_HEADERS = {
    'User-Agent': _USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Referer': 'https://www.google.com/',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Sec-Ch-Ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
}

# Statuses that indicate a bot-wall / WAF block worth escalating to tier 2
# (TLS-fingerprint impersonation). A plain 404/410 is NOT a block — fail fast.
_FETCH_BLOCK_STATUSES = {401, 402, 403, 406, 429, 500, 502, 503, 520, 521, 522, 523, 524, 526}
_CHALLENGE_MARKERS = (
    'just a moment', 'attention required', 'cf-browser-verification', 'cf-challenge',
    '/cdn-cgi/challenge-platform', 'access denied', 'enable javascript and cookies to continue',
)


class _FetchResult:
    __slots__ = ('final_url', 'status_code', 'text', 'tier')

    def __init__(self, final_url, status_code, text, tier):
        self.final_url = final_url
        self.status_code = status_code
        self.text = text
        self.tier = tier


def _looks_like_challenge(text):
    # Interstitial/challenge pages are small and carry tell-tale markers; real
    # recipe pages are large. Only inspect a short prefix of smallish bodies.
    if not text or len(text) > 200000:
        return False
    low = text[:5000].lower()
    return any(marker in low for marker in _CHALLENGE_MARKERS)


def _curl_cffi_fetch(url):
    # Lazy import so the module still loads (degrading to tier 1) if the wheel is
    # ever missing. curl_cffi impersonates a real Chrome TLS/JA3 fingerprint,
    # which defeats most datacenter-IP bot walls that header spoofing cannot.
    from curl_cffi import requests as _cffi_requests
    return _cffi_requests.get(
        url,
        impersonate='chrome',
        timeout=_REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )


# --- Tier 3: residential-egress proxy --------------------------------------
# Tiers 1 and 2 both lose on the Dotdash Meredith / People Inc. network (Allrecipes,
# Food & Wine, Serious Eats, Simply Recipes): those hosts block on IP REPUTATION —
# 402 for tier-1 requests, then 403 for tier-2 curl_cffi — so nothing we spoof from a
# Lambda datacenter IP can win. The only remaining lever is egressing from a
# residential IP. Tier 3 therefore re-runs the SAME Chrome-impersonating fetch
# through a residential proxy: the IP is what changes, but the TLS fingerprint still
# has to be right, which is why this is curl_cffi and not plain requests (verified —
# a residential IP with curl's own fingerprint still gets a 403 bot wall).
# The provider is pluggable via env so the vendor can be swapped without a deploy.
_FETCH_PROXY_PROVIDER_ENV = 'RECIPE_FETCH_PROXY_PROVIDER'
_FETCH_PROXY_URL_ENV = 'RECIPE_FETCH_PROXY_URL'
_FETCH_PROXY_TIMEOUT_ENV = 'RECIPE_FETCH_PROXY_TIMEOUT_SECONDS'
_APIFY_PROXY_PASSWORD_ENV = 'APIFY_PROXY_PASSWORD'
_APIFY_PROXY_GROUPS_ENV = 'APIFY_PROXY_GROUPS'
_APIFY_PROXY_COUNTRY_ENV = 'APIFY_PROXY_COUNTRY'
_APIFY_PROXY_ENDPOINT = 'proxy.apify.com:8000'
# Residential egress adds a hop plus real-ISP latency, so tier 3 gets a bigger budget
# than the 15s direct tiers — but a bounded one. A single web_recipe import can run the
# whole ladder up to 3x (resolve, then json-ld, then the html fallback), so the ceiling
# that matters is 3 x (15 + 15 + tier3). _PROXY_FETCH_CACHE collapses those repeats, and
# 25s keeps even the uncached worst case under the function's 120s timeout.
_PROXY_FETCH_TIMEOUT_SECONDS = max(1, min(30, int(os.getenv(_FETCH_PROXY_TIMEOUT_ENV, '25'))))
# Tier 3 is metered (residential proxies bill per GB) and a recipe page is ~500KB, so the
# 3 ladder runs of one import would otherwise buy the same HTML three times. Short-TTL
# memo, tier-3 ONLY — tiers 1 and 2 keep their exact current fetch-every-time behavior.
# The TTL is per-container and short enough that a later import re-fetches fresh.
_PROXY_FETCH_CACHE = {}
_PROXY_FETCH_CACHE_TTL_SECONDS = 180
_PROXY_FETCH_CACHE_MAX = 8
_APIFY_PROXY_PASSWORD_CACHE = {}


def _apify_proxy_password():
    """Apify proxy password. Prefers the explicit env var; otherwise derives it once
    from APIFY_TOKEN (the account API exposes it) and memoizes it for the container.
    Returns '' when it cannot be resolved, which leaves tier 3 dark."""
    explicit = _safe_text(os.getenv(_APIFY_PROXY_PASSWORD_ENV))
    if explicit:
        return explicit
    if 'password' in _APIFY_PROXY_PASSWORD_CACHE:
        return _APIFY_PROXY_PASSWORD_CACHE['password']
    password = ''
    token = _safe_text(os.getenv(_APIFY_TOKEN_ENV))
    if token:
        try:
            resp = requests.get('https://api.apify.com/v2/users/me',
                                headers={'Authorization': f'Bearer {token}'}, timeout=10)
            if resp.status_code < 400:
                data = (resp.json() or {}).get('data') or {}
                password = _safe_text((data.get('proxy') or {}).get('password'))
        except Exception:
            password = ''
    _APIFY_PROXY_PASSWORD_CACHE['password'] = password
    return password


def _fetch_proxy_endpoint():
    """Proxy URL for the configured tier-3 provider, or '' when tier 3 is dark (no
    provider configured — the ladder then ends at tier 2 exactly as it does today)."""
    override = _safe_text(os.getenv(_FETCH_PROXY_URL_ENV))
    if override:
        # Generic escape hatch: any vendor that speaks HTTP proxy (Bright Data,
        # Oxylabs, Zyte, ScrapingBee proxy mode, ...) needs only this one var set.
        return override
    provider = _safe_text(os.getenv(_FETCH_PROXY_PROVIDER_ENV)).lower()
    if provider == 'apify':
        password = _apify_proxy_password()
        if not password:
            return ''
        groups = _safe_text(os.getenv(_APIFY_PROXY_GROUPS_ENV)) or 'RESIDENTIAL'
        country = _safe_text(os.getenv(_APIFY_PROXY_COUNTRY_ENV)) or 'US'
        username = f'groups-{groups}'
        if country:
            username += f',country-{country}'
        return f'http://{quote(username, safe="-,")}:{quote(password, safe="")}@{_APIFY_PROXY_ENDPOINT}'
    return ''


def _proxy_fetch(url, proxy_url):
    from curl_cffi import requests as _cffi_requests
    return _cffi_requests.get(
        url,
        impersonate='chrome',
        timeout=_PROXY_FETCH_TIMEOUT_SECONDS,
        allow_redirects=True,
        proxy=proxy_url,
    )


def _proxy_cache_get(url):
    entry = _PROXY_FETCH_CACHE.get(url)
    if not entry:
        return None
    cached_at, result = entry
    if (time.time() - cached_at) > _PROXY_FETCH_CACHE_TTL_SECONDS:
        _PROXY_FETCH_CACHE.pop(url, None)
        return None
    return result


def _proxy_cache_put(url, result):
    if len(_PROXY_FETCH_CACHE) >= _PROXY_FETCH_CACHE_MAX:
        _PROXY_FETCH_CACHE.clear()
    _PROXY_FETCH_CACHE[url] = (time.time(), result)


def _is_upstream_block(exc):
    """True when the failure is purely the remote site refusing us (every fetch tier
    hit a bot wall / IP block), as opposed to a fault on our side. Callers use this to
    keep pure blocks out of the paging error feed while still failing the job."""
    return bool(isinstance(exc, ServiceError) and (exc.extra or {}).get('upstream_block'))


def _fetch_page(url, request_id=None):
    """Layered page fetch: cheap browser-header requests first, then Chrome
    TLS-impersonation (curl_cffi) only if the first tier looks bot-blocked, then the
    same impersonating fetch through a residential proxy if tier 2 is blocked too.
    Returns a _FetchResult or raises a friendly ServiceError."""
    # Social hosts (Instagram/TikTok) get the ORIGINAL minimal headers: the
    # aggressive Referer/Sec-Fetch-Site=cross-site profile makes Instagram
    # redirect public reels to a login wall from datacenter IPs. Web-recipe
    # hosts get the full modern-browser header set (+ curl_cffi escalation).
    host = (urlparse(url).netloc or '').lower()
    is_social = (
        host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com')
        or host in _VALID_TIKTOK_HOSTS or host.endswith('.tiktok.com')
    )
    tier1_headers = (
        {'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'}
        if is_social else _BROWSER_HEADERS
    )
    attempts = []
    blocked = False
    # Did any tier get an actual REFUSAL from the site (a block-signature HTTP status or
    # a challenge page) as opposed to a transport error? Only a refusal proves the remote
    # end is deliberately turning us away. If every tier merely errored out (DNS, TLS,
    # timeout) that could equally be OUR egress broken — an incident we still need paged,
    # so it must not be laundered into a low-severity blocked_host marker.
    refused = False
    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS, allow_redirects=True, headers=tier1_headers)
        attempts.append({'tier': 'requests', 'status': r.status_code})
        if r.status_code < 400 and not _looks_like_challenge(r.text):
            return _FetchResult(r.url or url, r.status_code, r.text, 'requests')
        blocked = r.status_code in _FETCH_BLOCK_STATUSES or _looks_like_challenge(r.text)
        refused = refused or blocked
        if not blocked:
            # Non-block 4xx (e.g. 404/410) — genuinely unreachable, don't escalate.
            raise ServiceError(f'Could not reach URL (HTTP {r.status_code}).', status_code=502)
    except requests.RequestException as exc:
        attempts.append({'tier': 'requests', 'status': f'error: {exc}'})
        blocked = True

    # Tier 2: TLS-fingerprint impersonation, only when tier 1 looked blocked.
    if blocked:
        try:
            cr = _curl_cffi_fetch(url)
            attempts.append({'tier': 'curl_cffi', 'status': cr.status_code})
            if cr.status_code < 400 and not _looks_like_challenge(cr.text):
                # Corpus: record which sites required TLS-impersonation to get past
                # a bot wall (tier-1 requests was blocked, tier-2 curl_cffi won).
                _log_event(request_id, 'web_fetch_recovered', url=url,
                           tier1_status=attempts[0]['status'], status=cr.status_code,
                           tier='curl_cffi')
                return _FetchResult(str(cr.url) or url, cr.status_code, cr.text, 'curl_cffi')
            refused = refused or (cr.status_code in _FETCH_BLOCK_STATUSES
                                  or _looks_like_challenge(cr.text))
        except Exception as exc:
            attempts.append({'tier': 'curl_cffi', 'status': f'error: {exc}'})

    # Tier 3: residential-egress proxy — ONLY when tier 2 came back blocked too, and
    # never for social hosts. Instagram/TikTok have their own extraction path and their
    # own auth walls; a residential IP does not help them and would just burn metered
    # bandwidth. `blocked` is already the tier-1/tier-2 block gate, so a tier-1 success
    # or a non-block 4xx (404/410, which raises above) never reaches here.
    if blocked and not is_social:
        cached = _proxy_cache_get(url)
        if cached is not None:
            return cached
        proxy_url = _fetch_proxy_endpoint()
        if proxy_url:
            try:
                pr = _proxy_fetch(url, proxy_url)
                attempts.append({'tier': 'proxy', 'status': pr.status_code})
                if pr.status_code < 400 and not _looks_like_challenge(pr.text):
                    # Corpus: the sites that need a residential IP rather than just a
                    # browser fingerprint — i.e. the tier-2-lost / tier-3-won population.
                    _log_event(request_id, 'web_fetch_recovered', url=url,
                               tier1_status=attempts[0]['status'], status=pr.status_code,
                               tier='proxy')
                    result = _FetchResult(str(pr.url) or url, pr.status_code, pr.text, 'proxy')
                    _proxy_cache_put(url, result)
                    return result
                refused = refused or (pr.status_code in _FETCH_BLOCK_STATUSES
                                      or _looks_like_challenge(pr.text))
            except Exception as exc:
                attempts.append({'tier': 'proxy', 'status': f'error: {exc}'})

    _log_event(request_id, 'web_fetch_blocked', url=url,
               tiers_tried=[a['tier'] for a in attempts], statuses=attempts, refused=refused)
    # `upstream_block` only when the site actually refused us (see `refused` above): that
    # is the site's decision, not a fault in our backend, and the user already gets the
    # graceful copy-the-text fallback — so _is_upstream_block lets the job runner skip
    # paging. An all-transport-error failure leaves the flag off and pages as it does today.
    raise ServiceError('This site is blocking us — try copying the recipe text instead.',
                       status_code=502,
                       extra={'upstream_block': refused, 'blocked_host': host})


def _resolve_url(url, request_id=None):
    result = _fetch_page(url, request_id=request_id)
    parsed = urlparse(result.final_url or url)
    return urlunparse(parsed._replace(fragment=''))


# Instagram paths that point at a specific piece of content we can attempt to
# extract: reels, feed posts (/p/ — videos AND photo carousels), IGTV, and
# unresolved share short-links (let the pipeline try). Everything else on an
# Instagram host (profiles, /accounts/, /stories/, /explore/) is not a single
# saveable post and is rejected with a clear, actionable message.
_INSTAGRAM_CONTENT_MARKERS = ('/reel/', '/reels/', '/p/', '/tv/', '/share/')


def _instagram_content_url_from(candidate):
    """If `candidate` (a full URL or a bare path) points at Instagram content,
    return a clean canonical https://www.instagram.com/<path> URL; else None."""
    text = _safe_text(candidate)
    if not text:
        return None
    if '://' not in text:
        text = 'https://www.instagram.com' + (text if text.startswith('/') else '/' + text)
    parsed = urlparse(text)
    path = (parsed.path or '').lower()
    if any(marker in path for marker in _INSTAGRAM_CONTENT_MARKERS):
        return urlunparse(('https', 'www.instagram.com', parsed.path, '', '', ''))
    return None


def _recover_instagram_content_url(resolved_url, raw_url=None, normalized_url=None, request_id=None):
    """Instagram now login-wall-redirects EVERY resolve fetch from datacenter IPs
    (resolved_url becomes /accounts/login/?next=%2Freel%2F...). Recover the real
    content URL from the login redirect's `next` param (or the raw/normalized
    submission) so the gate passes and extraction (Apify) gets the canonical URL."""
    parsed = urlparse(resolved_url)
    host = (parsed.netloc or '').lower()
    if not (host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com')):
        return resolved_url
    if any(marker in (parsed.path or '').lower() for marker in _INSTAGRAM_CONTENT_MARKERS):
        return resolved_url  # already a content URL — nothing to recover

    next_param = None
    qs = parse_qs(parsed.query or '')
    if qs.get('next'):
        next_param = unquote(qs['next'][0])
    # Prefer the login redirect's next= (the canonical target IG was about to show),
    # then the user's own submission.
    for candidate in (next_param, normalized_url, raw_url):
        canonical = _instagram_content_url_from(candidate)
        if canonical:
            _log_event(request_id, 'instagram_login_wall_fallback',
                       raw_url=raw_url, resolved_url=resolved_url,
                       next_param=next_param, canonical_url=canonical)
            return canonical
    return resolved_url  # genuine profile/login/etc. — _detect_platform will 400 it


def _detect_platform(url, raw_url=None, request_id=None):
    parsed = urlparse(url)
    host = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    if host in _VALID_TIKTOK_HOSTS or host.endswith('.tiktok.com'):
        return 'tiktok'
    if host in _VALID_INSTAGRAM_HOSTS or host.endswith('.instagram.com'):
        if any(marker in path for marker in _INSTAGRAM_CONTENT_MARKERS):
            return 'instagram'
        # Non-content Instagram URL (profile / accounts / stories / explore).
        # Log the offending URL so we build a corpus of what users actually share.
        _log_event(
            request_id,
            'instagram_url_rejected',
            raw_url=raw_url,
            resolved_url=url,
            reason='non_content_path',
        )
        raise ServiceError('Share a link to a specific Instagram post or reel.', status_code=400)
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


def _fetch_html_soup(url, request_id=None):
    # Uses the same layered fetch as _resolve_url so the extraction re-fetch also
    # gets past bot walls (a site that blocks resolve would otherwise block here).
    result = _fetch_page(url, request_id=request_id)
    return BeautifulSoup(result.text, 'html.parser')


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
    # A login/content wall does not change when we immediately repeat the same
    # request and credentials. Let the existing provider chain try its next source.
    # Instagram's combined message mentions rate limits too; that is not an
    # explicit HTTP 429 and must not stall every caption/audio attempt three times.
    if any(fragment in text for fragment in ('login required', 'use --cookies', 'requested content is not available')):
        return False
    return any(fragment in text for fragment in (
        'rate-limit reached',
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
        options.update(max_filesize=recipe_work_budget.MAX_MEDIA_BYTES,
                       progress_hooks=[recipe_work_budget.download_progress], noprogress=True)
        options['format'] = 'bestaudio/best'
        host = (urlparse(resolved_url).hostname or '').lower()
        if host == 'tiktok.com' or host.endswith('.tiktok.com'):
            # Older yt-dlp labels TikTok's /media-video-hvc1/ variants as AAC,
            # although they contain video only. Prefer real audio/muxed media;
            # sending the silent variant to transcription always fails.
            options['format'] = 'bestaudio/best[url!*=media-video-hvc1]'
    cookiefile = _resolve_ytdlp_cookiefile()
    if cookiefile:
        options['cookiefile'] = cookiefile
    last_exc = None
    for attempt in range(1, _YTDLP_MAX_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(resolved_url, download=download)
        except recipe_work_budget.WorkLimit:
            raise
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


@recipe_work_budget.bounded
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
    # The actor already returns direct CDN media links in the same response we pay for.
    # yt-dlp cannot fetch Instagram media at all any more (IG answers unauthenticated
    # requests with an empty media response), so this is our only working route to the
    # audio — and it costs no extra actor run. Prefer audioUrl when present: it is far
    # smaller than the mp4 and Whisper accepts either.
    audio_url = _safe_text(post.get('audioUrl') or post.get('audio_url'))
    video_url = _safe_text(post.get('videoUrl') or post.get('video_url'))
    return {
        'url': url or fallback_url,
        'caption': caption,
        'image_url': image_url,
        'image_urls': _unique_texts([image_url]),
        'author_name': author_name,
        'audio_url': audio_url,
        'video_url': video_url,
        'media_url': audio_url or video_url,
    }


# A single saved-recipe job can call the Apify extractor twice for the same URL: once
# in the provider chain, and again as the `_analyze_extraction` fallback when the caption
# turned out to be too thin. With a 30s read timeout that would mean paying the timeout
# twice inside one 120s Lambda. Memoize the attempt (success OR failure) per REQUEST so
# the second call reuses the first outcome. Keying on request_id — not just the URL —
# keeps the memo inside one job: a user retrying a minute later still gets a fresh
# attempt rather than a cached failure from a warm container.
_APIFY_IG_ATTEMPT_CACHE = {}
_APIFY_IG_ATTEMPT_CACHE_MAX = 16


def _apify_ig_attempt_remember(cache_key, outcome, payload):
    """Record this request's Apify outcome. No-op when we have no request_id to scope
    the memo to (a bare URL key could leak across jobs in a warm container)."""
    if not cache_key[0]:
        return
    if len(_APIFY_IG_ATTEMPT_CACHE) >= _APIFY_IG_ATTEMPT_CACHE_MAX:
        _APIFY_IG_ATTEMPT_CACHE.clear()
    _APIFY_IG_ATTEMPT_CACHE[cache_key] = (outcome, payload)


@recipe_work_budget.bounded
def _extract_instagram_apify(resolved_url, request_id=None):
    token = _safe_text(os.getenv(_APIFY_TOKEN_ENV))
    if not token:
        raise RuntimeError(f'{_APIFY_TOKEN_ENV} is not set.')

    cache_key = (_safe_text(request_id), _safe_text(resolved_url))
    cached = _APIFY_IG_ATTEMPT_CACHE.get(cache_key) if cache_key[0] else None
    if cached is not None:
        outcome, payload = cached
        if outcome == 'ok':
            return payload
        raise payload

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
            json=payload,
            timeout=_APIFY_RUN_TIMEOUT_SECONDS,
            headers={'User-Agent': _USER_AGENT, 'Accept': 'application/json',
                     'Authorization': f'Bearer {token}'},
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
        result = {
            'content': content,
            'title': content[:80],
            'image_url': _safe_text(post.get('image_url')),
            'image_urls': _unique_texts(post.get('image_urls') or []),
            'source': 'apify',
            'author_name': post.get('author_name'),
            'caption_field': 'caption',
            'warnings': [],
            'media_url': _safe_text(post.get('media_url')),
        }
        _apify_ig_attempt_remember(cache_key, 'ok', result)
        return result
    except Exception as exc:
        _log_event(
            request_id,
            'instagram_apify_extract_failed',
            resolved_url=resolved_url,
            actor=_APIFY_INSTAGRAM_ACTOR,
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=str(exc),
        )
        failure = RuntimeError(f'Apify Instagram extraction failed: {exc}')
        _apify_ig_attempt_remember(cache_key, 'err', failure)
        raise failure


# --- Web article body extraction --------------------------------------------------
# The html fallback previously built its content from og:description ALONE, which is a
# ~130-char marketing blurb. Two consequences, both measured on live data:
#   1. Pages with no Recipe json-ld (sipnfeel, thepastatable) 422'd, or worse
#   2. the refiner INVENTED a plausible recipe from the dish name — 50.7% of successful
#      web html saves had no digit in any ingredient.
# The recipe text is almost always sitting in the page body we never read. This reads it.
#
# Scoped deliberately: only for non-social web hosts, and only in the html fallback,
# which already runs solely when json-ld found no Recipe node. Social hosts are EXCLUDED
# because their og:description IS the full post caption and already works (503 such saves
# in 30 days) — adding a server-side-fetched social page body would inject login-wall
# junk into content that is currently correct.
_WEB_SOCIAL_HOSTS = (
    'facebook.com', 'fb.watch', 'fb.com', 'instagram.com', 'tiktok.com', 'threads.net',
    'twitter.com', 'x.com', 'pinterest.com', 'pin.it', 'youtube.com', 'youtu.be',
    'reddit.com', 'snapchat.com', 'linkedin.com', 'tumblr.com',
)
# Narrowest-first: a page that marks its article body explicitly is far more reliable
# than falling back to <body>, which drags in nav, related-posts and comment threads.
_WEB_ARTICLE_SELECTORS = (
    '[itemprop="articleBody"]', 'article', 'main', '.entry-content', '.post-content', '#content',
)
_WEB_ARTICLE_DROP_TAGS = (
    'script', 'style', 'noscript', 'nav', 'header', 'footer', 'aside', 'form', 'svg',
    'button', 'iframe',
)
# Cap: measured 127-6081 chars across a 7-page sample, so 6000 binds on the long ones.
# ~1.5k tokens of extra prompt input on a path that previously sent ~35 — bounded, and
# only on saves that were already failing or fabricating.
_WEB_BODY_MAX_CHARS = int(os.getenv('WEB_BODY_MAX_CHARS', '6000'))
# Below this the body is not worth trusting over the description (example.com yields 127).
_WEB_BODY_MIN_CHARS = int(os.getenv('WEB_BODY_MIN_CHARS', '200'))


# A body with no quantities anywhere cannot support a real recipe — and handing one to
# the refiner is precisely how fabrication happens. browneyedbaker.com proved this in
# production: its article body is 6,025 chars of prose and navigation with ZERO quantity
# tokens (the actual recipe lives in a WP Recipe Maker card that reaches us only as
# json-ld). Fed that, the refiner invented an ingredient list whose own text admitted
# "Danish pastry dough (ingredients not explicitly listed)" — strictly worse for the user
# than the honest 422 it used to return.
# Measured separation on the sample: sipnfeel 27, thepastatable 12 (both real recipes)
# vs browneyedbaker 0 and example.com 0. A floor of 3 sits well clear of both clusters.
_WEB_BODY_MIN_QUANTITIES = int(os.getenv('WEB_BODY_MIN_QUANTITIES', '3'))
_WEB_QUANTITY_PATTERN = re.compile(
    r'\b\d+\s*(?:\d*/\d+\s*)?(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons'
    r'|g|gram|grams|kg|oz|ounce|ounces|lb|lbs|pound|pounds|ml|clove|cloves|can|cans'
    r'|slice|slices|stick|sticks)\b'
    r'|[\u00bc\u00bd\u00be\u2153\u2154\u215b]\s*(?:cup|tsp|tbsp|lb|oz)',
    re.I,
)


def _web_body_has_recipe_substance(text):
    """True when the extracted body carries enough measured quantities to plausibly BE a
    recipe. Fails closed: on doubt we keep today's og:description behaviour, which may be
    an honest 422 — a miss the user can act on, unlike an invented recipe."""
    return len(_WEB_QUANTITY_PATTERN.findall(_safe_text(text))) >= _WEB_BODY_MIN_QUANTITIES


def _is_web_social_host(host):
    host = (host or '').lower()
    return any(host == h or host.endswith('.' + h) for h in _WEB_SOCIAL_HOSTS)


def _extract_web_article_text(soup, max_chars=None):
    """Readable article text for a recipe web page. Returns '' when nothing usable."""
    limit = max_chars or _WEB_BODY_MAX_CHARS
    root = None
    for selector in _WEB_ARTICLE_SELECTORS:
        try:
            found = soup.select(selector)
        except Exception:
            found = []
        if found:
            # Several <article> tags can match (related-post cards); take the biggest.
            root = max(found, key=lambda node: len(node.get_text(' ', strip=True)))
            break
    if root is None:
        root = soup.body or soup
    for tag in root.find_all(_WEB_ARTICLE_DROP_TAGS):
        tag.decompose()
    lines = []
    seen = set()
    total = 0
    for raw_line in root.get_text('\n', strip=True).splitlines():
        line = _safe_text(raw_line)
        normalized = line.lower()
        # Keep short lines here (unlike the Instagram snippet's 12-char floor): ingredient
        # rows are frequently 3-10 chars ("1/2 lb.", "2 eggs") and dropping them would
        # discard exactly the quantities this change exists to recover.
        if len(line) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
        total += len(line) + 1
        if total >= limit:
            break
    return '\n'.join(lines).strip()


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
    elif not _is_web_social_host(host):
        # Non-social web page: read the article body instead of trusting a marketing
        # blurb. The description is kept FIRST and the body added as context, so a page
        # whose body extraction comes back thin degrades to exactly today's behaviour.
        body_text = _extract_web_article_text(soup)
        if len(body_text) >= _WEB_BODY_MIN_CHARS and _web_body_has_recipe_substance(body_text):
            merged = _merge_recipe_source_chunks([
                ('Meta description', _strip_social_prefixes(description)),
                ('Page text', body_text),
                ('Title', title),
            ])
            if merged:
                _log_event(None, 'web_body_text_extracted', host=host,
                           description_chars=len(_safe_text(description)),
                           body_chars=len(body_text), merged_chars=len(merged),
                           capped=len(body_text) >= _WEB_BODY_MAX_CHARS)
                content = merged
        elif body_text:
            # Body found but too thin on quantities to trust. This is the step-2
            # population — a page whose recipe is NOT in its prose — so log it by host
            # to size that cohort without escalating anything yet.
            _log_event(None, 'web_body_text_rejected', host=host,
                       body_chars=len(body_text),
                       quantities=len(_WEB_QUANTITY_PATTERN.findall(body_text)))
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


def _json_ld_recipe_richness(node):
    """How much actual recipe content a Recipe-typed node carries."""
    ingredients = node.get('recipeIngredient')
    ingredient_count = len(ingredients) if isinstance(ingredients, list) else 0
    return ingredient_count + len(_parse_instructions(node.get('recipeInstructions')))


def _select_richest_recipe_node(soup):
    """Pick the Recipe node with the MOST content, not the first one encountered.

    Recipe blogs commonly emit several ld+json blocks, and the first Recipe-typed node is
    often a STUB — name and description only, no ingredients or steps — with the real
    recipe in a later block. poulef.com/banana-bread-brownies-2 is the confirmed case:
    three Recipe nodes, the first with 0 ingredients / 0 instructions and the next two
    with 13 / 11. First-match-wins read the stub, produced title-plus-description content,
    and the refiner correctly said there was no recipe — so a perfectly good recipe blog
    422'd on the user.

    Ties keep the earliest node, so single-Recipe pages are completely unaffected. If no
    node has any content we still return the first bare node, preserving today's behaviour
    (and today's 502) for genuinely empty pages.
    """
    best = None
    best_score = -1
    first = None
    for block in _parse_json_ld_blocks(soup):
        for node in _iter_json_nodes(block):
            if not (isinstance(node, dict) and 'Recipe' in _node_types(node)):
                continue
            if first is None:
                first = node
            score = _json_ld_recipe_richness(node)
            if score > best_score:
                best, best_score = node, score
    if best is not None and best_score > 0:
        return best
    return first


def _extract_json_ld_recipe(resolved_url):
    soup = _fetch_html_soup(resolved_url)
    recipe_node = _select_richest_recipe_node(soup)
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
    block_exc = None
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
                # Direct CDN media link when the provider supplied one (Apify does).
                # The audio fallback prefers it over yt-dlp, which is dead for Instagram.
                'media_url': _safe_text(result.get('media_url')),
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
            if block_exc is None and _is_upstream_block(exc):
                block_exc = exc
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
    if block_exc is not None:
        # A provider lost because the site refused every fetch tier (the block gate in
        # _fetch_page), not because extraction is broken. That is an upstream decision
        # with nothing for us to fix, so it must not page — re-raise with the flag
        # intact and let the job runner emit the low-severity blocked_host marker.
        raise ServiceError(str(block_exc), status_code=502,
                           extra={'upstream_block': True,
                                  'blocked_host': (block_exc.extra or {}).get('blocked_host'),
                                  'warnings': warnings})
    _report_backend_error('extract_url', code='all_providers_failed',
                          error=f"{platform}: " + ' | '.join([attempt.get('error') or attempt.get('source') for attempt in attempts]))
    raise ServiceError(f'Could not extract content from this {platform} URL.', status_code=502, extra={'warnings': warnings})


@recipe_work_budget.bounded
def _extract_content(url, request_id=None):
    normalized_url = _normalize_url(url)
    resolved_url = _resolve_url(normalized_url, request_id=request_id)
    # Recover from Instagram's login-wall redirect before gating/extraction so a
    # real reel/post isn't mis-rejected and Apify gets the canonical content URL.
    resolved_url = _recover_instagram_content_url(
        resolved_url, raw_url=url, normalized_url=normalized_url, request_id=request_id)
    platform = _detect_platform(resolved_url, raw_url=url, request_id=request_id)
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
            # Pass request_id so this attempt shares the per-request Apify memo with the
            # `_analyze_extraction` fallback (one timeout per job, not two) and so its
            # success/failure events are attributable to the job in the logs.
            ('apify', lambda u: _extract_instagram_apify(u, request_id=request_id)),
            ('html', _extract_html_metadata),
        ],
        'web_recipe': [
            ('json-ld', _extract_json_ld_recipe),
            ('html', _extract_html_metadata),
        ],
    }
    return _run_provider_pipeline(normalized_url, resolved_url, platform, providers[platform], request_id=request_id)


def _recipe_completeness_score(structured):
    """Sortable richness of a refine result, most significant first.

    The ladder used to keep whatever a later rung produced, so a thinner roll could
    overwrite a richer earlier one. Comparing scores lets a rung REPLACE the incumbent
    only when it is genuinely better."""
    if not isinstance(structured, dict):
        return (0, 0, 0)
    ingredients = [item for item in (structured.get('ingredients') or []) if _safe_text(item)]
    instructions = [item for item in (structured.get('instructions') or []) if _safe_text(item)]
    return (1 if (ingredients and instructions) else 0, len(instructions), len(ingredients))


def _recipe_is_incomplete(recipe_text, structured=None):
    """Should the recovery ladder keep going?

    Completeness used to be an exact-string match on the model's refusal sentinel. That
    made any output that was not literally 'Not enough recipe information.' count as a
    finished recipe - including a full ingredient list with an empty Instructions section
    (82.5% of all defective saves) and a title-only result manufactured by slide-OCR off a
    video cover frame (a further 15.8%). Both stopped the ladder before Apify, OCR or audio
    transcription ever ran, which is why audio was reached only ~159 times against ~1,050
    saves a day. Measured cost of stopping early: those requests finish in ~7s, i.e. with
    ~22s of gateway headroom unused.

    A result is usable only if the user could cook from it: at least one ingredient AND at
    least one instruction. Anything less keeps the ladder climbing.

    `structured=None` reproduces the old sentinel-only behaviour byte-for-byte, so any
    caller that has no structured recipe to hand is unaffected."""
    if _safe_text(recipe_text) == 'Not enough recipe information.':
        return True
    if structured is None:
        return False
    return _recipe_completeness_score(structured)[0] == 0


# --- Sync-request time budget for the audio fallback -------------------------------
# Phase-1 added an Apify media download + Whisper round-trip to the Instagram fallback.
# Measured: the transcription leg alone runs 3.7-5.0s, and end-to-end a reel that took
# 14.2s before phase-1 now takes 20.8-25.7s. API Gateway's integration timeout is 30,000ms
# on every saved-recipes route (verified across all 29 integrations), so a slow sync save
# can cross the ceiling and 504/503 — losing the very recoveries phase-1 added.
#
# Async work has a separate native Lambda deadline (recipe_work_budget). Both paths
# reserve time to persist an honest terminal outcome before their response deadline.
#
# A skipped attempt is NOT a lost recovery: the job is self-heal captured (payload_s3_key +
# input_refs + ttl), and scripts/redrive_failed_saves.py speaks the async path — so an
# operator sweep recovers budget victims out of band. The distinct event below is what
# makes that sweep targetable, and what lets us price the budget.
_SYNC_AUDIO_BUDGET_SECONDS = float(os.getenv('SYNC_AUDIO_BUDGET_SECONDS', '18'))
# Off by default: a saved recipe shows a REAL photo or a title emoji, never an invented one.
# Kept as a flag rather than deleting the generator so the behaviour is a one-line decision
# and the meal-plan path (which still generates) is unaffected.
GENERATE_RECIPE_IMAGES = os.getenv('GENERATE_RECIPE_IMAGES', 'false').strip().lower() == 'true'
# Set by the handler wrapper. None => not a synchronous HTTP request (async task, warmup,
# or a direct invoke), in which case no budget is enforced.
_SYNC_REQUEST_STARTED_AT = None


def _mark_sync_request_start(is_sync):
    """Record when a synchronous HTTP request began, so deep-stack code can tell how much
    of the gateway's 30s has already been spent."""
    global _SYNC_REQUEST_STARTED_AT
    _SYNC_REQUEST_STARTED_AT = time.time() if is_sync else None


def _sync_elapsed_seconds():
    if _SYNC_REQUEST_STARTED_AT is None:
        return None
    return time.time() - _SYNC_REQUEST_STARTED_AT


def _audio_budget_exhausted():
    """DEPRECATED — superseded by the deadline-aware _rung_fits('audio').

    This measured elapsed time against a fixed constant (15s in prod) rather than against
    what was actually left of the gateway window. That was fine while the ladder usually
    quit at rung 0, but once the completeness fix let it climb, the earlier rungs routinely
    spent 15s on their own, so this fired on nearly every video save and audio never ran.
    The visible result was TikTok and Instagram recipes with ingredients (from the caption)
    and ZERO steps, because for a talking video the steps only exist in the transcript.

    Kept as a no-op rather than deleted so the constant and this note stay together as the
    record of why a fixed-constant budget cannot work here.
    """
    return False


# --- Deadline-aware rung budgeting -------------------------------------------------
# The audio budget above guards ONE rung against a constant measured from request start.
# That was adequate while the ladder usually quit at rung 0; once the completeness fix let
# it climb, the earlier rungs (Apify, slide OCR, preview OCR — each a fetch plus a refine)
# began consuming the budget themselves, and the request blew the 29s gateway ceiling
# BEFORE audio was ever considered. Measured after that fix, same clock window as the day
# before: p90 11.4s -> 24.1s, requests over 29s 1.4% -> 6.2%. Skipping audio no longer
# protects the ceiling, because audio is not where the time goes.
#
# So the budget has to be a DEADLINE, checked before EVERY rung, expressed as time
# REMAINING rather than time spent: is there enough left to run this rung and still
# return? Async invocations have no gateway in front of them and are unaffected.
_GATEWAY_CEILING_SECONDS = float(os.getenv('GATEWAY_CEILING_SECONDS', '29'))
# Headroom kept back for the work that must still happen after a rung returns: persisting
# the row, image mirroring, availability overlay and serialization. Measured at 1.5-4s.
_RESERVE_AFTER_LADDER_SECONDS = float(os.getenv('RESERVE_AFTER_LADDER_SECONDS', '5'))
# What each rung typically costs, so we can ask "will this fit?" rather than "have we
# already overrun?". Conservative p75-ish figures from production timings.
_RUNG_COST_SECONDS = {
    'apify': 6.0,        # Apify fetch + refine
    'slide_ocr': 4.0,    # Rekognition over carousel slides + refine
    'preview_ocr': 7.0,  # VLM preview read + refine
    'audio': 10.0,       # media download + Whisper + refine
    'merged': 4.0,       # one more refine over already-fetched content
}


class _DeadlineSkip(Exception):
    """Raised to skip a rung that will not fit before the gateway ceiling. Caught by the
    rung's existing `except Exception` so the ladder continues rather than aborting."""


def _sync_time_remaining():
    """Seconds left before the gateway gives up on this sync request. None when async."""
    elapsed = _sync_elapsed_seconds()
    if elapsed is None:
        return None
    return _GATEWAY_CEILING_SECONDS - elapsed


def _rung_fits(rung):
    """Should we attempt `rung`? True whenever there is no gateway in front of us, or the
    rung's expected cost plus the post-ladder reserve still fits in the time remaining.

    Returns (fits, remaining, needed) so the caller can log why it skipped."""
    remaining = _sync_time_remaining()
    native_work = recipe_work_budget.remaining()
    if native_work is not None:
        remaining = min(remaining, native_work) if remaining is not None else native_work
    if remaining is None:
        return True, None, None
    needed = _RUNG_COST_SECONDS.get(rung, 5.0) + _RESERVE_AFTER_LADDER_SECONDS
    if remaining < needed:
        recipe_work_budget.note_rung_skip()
    return remaining >= needed, remaining, needed


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


# Whisper invents fluent text from silence or from music-only audio — the classic
# failure is a confident transcript in an unrelated language. verbose_json gives us
# no_speech_prob and a detected language, which together catch it cheaply.
_NO_SPEECH_PROB_MAX = float(os.getenv('TRANSCRIPT_NO_SPEECH_PROB_MAX', '0.6'))
_TRANSCRIPT_EXPECTED_LANGUAGES = {'english'}
# Quantity/measure signal and imperative cooking verbs. A real spoken recipe has at
# least one of each; marketing narration ("if summer had a signature salmon dinner")
# has neither, and must NOT be handed to the refiner as if it were a recipe.
_QUANTITY_PATTERN = re.compile(
    # NOTE the trailing `s?` — spoken recipes say "200 grams" / "10 ounces", and a bare
    # \b after the singular unit rejects every plural. That false-negative would block
    # genuine recoveries, so plurals are matched explicitly.
    r'\b\d+\s*(?:cup|tbsp|tablespoon|tsp|teaspoon|oz|ounce|lb|pound|gram|g|kg|ml|liter|litre'
    r'|clove|can|slice|minute|min|hour|degree)s?\b'
    r'|\b\d{2,3}\s*(?:f|c|°)\b'
    r'|\b(?:half|quarter|third)\s+(?:a\s+)?cup\b'
    r'|\b(?:one|two|three|four|five|six|eight|ten|twelve)\s+'
    r'(?:cup|tbsp|tablespoon|tsp|teaspoon|oz|ounce|lb|pound|gram|clove|can|slice)s?\b',
    re.I,
)
_IMPERATIVE_PATTERN = re.compile(
    r'\b(?:add|mix|stir|bake|boil|blend|chop|combine|cook|fold|fry|grill|heat|knead|marinate'
    r'|mash|melt|pour|preheat|roast|saute|sauté|season|simmer|slice|whisk|toss|drain|layer|top)\b',
    re.I,
)


def _transcript_looks_hallucinated(segments, language):
    """True when Whisper most likely transcribed silence/music rather than speech."""
    lang = _safe_text(language).lower()
    if lang and _TRANSCRIPT_EXPECTED_LANGUAGES and lang not in _TRANSCRIPT_EXPECTED_LANGUAGES:
        return True, f'unexpected transcript language: {lang}'
    probs = [
        seg.get('no_speech_prob')
        for seg in (segments or [])
        if isinstance(seg, dict) and isinstance(seg.get('no_speech_prob'), (int, float))
    ]
    if probs and (sum(probs) / len(probs)) > _NO_SPEECH_PROB_MAX:
        return True, f'no_speech_prob {sum(probs) / len(probs):.2f} over {_NO_SPEECH_PROB_MAX}'
    return False, ''


def _transcript_has_recipe_signal(transcript):
    """True when the transcript actually carries a recipe: at least one quantity AND at
    least one imperative cooking step. Narration that merely describes a dish fails this
    and is treated as 'no recipe found' — the refiner would otherwise invent a method
    from the dish name, which is worse for the user than an honest miss."""
    text = _safe_text(transcript)
    if not text:
        return False
    return bool(_QUANTITY_PATTERN.search(text)) and bool(_IMPERATIVE_PATTERN.search(text))


def _download_media_to(temp_dir, media_url):
    """Stream a direct CDN media link to disk. Bounded so a surprise large asset can't
    blow the Lambda's /tmp or Whisper's 25MB upload ceiling."""
    limit = min(recipe_work_budget.MAX_MEDIA_BYTES, int(os.getenv('TRANSCRIBE_MEDIA_MAX_BYTES', str(recipe_work_budget.MAX_MEDIA_BYTES))))
    path = temp_dir / 'apify_media.mp4'
    total = 0
    with recipe_work_budget.phase(recipe_work_budget.DOWNLOAD_SECONDS), requests.get(media_url, stream=True, timeout=min(_REQUEST_TIMEOUT_SECONDS, recipe_work_budget.phase_seconds(recipe_work_budget.DOWNLOAD_SECONDS))) as resp:
        resp.raise_for_status()
        try:
            recipe_work_budget.check_bytes(int(resp.headers.get('Content-Length', 0)), limit)
        except (TypeError, ValueError):
            pass
        with path.open('wb') as handle:
            for chunk in resp.iter_content(chunk_size=262144):
                if not chunk:
                    continue
                total += len(chunk)
                recipe_work_budget.check_bytes(total, limit)
                handle.write(chunk)
    if not total:
        raise RuntimeError('Media download was empty.')
    return path


def _transcribe_audio_from_url(resolved_url, request_id=None, source_context=None):
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set.')

    started = time.time()
    # Prefer the direct CDN link the Apify actor already handed us. yt-dlp cannot fetch
    # Instagram media any more (empty media response since ~07-15), so for Instagram this
    # is the only route that works; other platforms keep the yt-dlp path unchanged.
    media_url = _safe_text((source_context or {}).get('media_url'))
    media_source = 'apify_media' if media_url else 'yt-dlp'
    try:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if media_url:
                # Whisper accepts mp4/m4a directly, so no ffmpeg (there is no ffmpeg
                # binary in the deployment package and no layer providing one).
                media_path = _download_media_to(temp_dir, media_url)
            else:
                with recipe_work_budget.phase(recipe_work_budget.DOWNLOAD_SECONDS):
                    info = _extract_ytdlp_info(
                        resolved_url,
                        download=True,
                        outtmpl=str(temp_dir / '%(id)s.%(ext)s'),
                    )
                recipe_work_budget.check_bytes(info.get('filesize'))
                media_path = _choose_downloaded_media_file(temp_dir, info)
            # Both download routes must pass the same final guard before upload.
            recipe_work_budget.check_bytes(media_path.stat().st_size)
            with recipe_work_budget.phase(recipe_work_budget.TRANSCRIBE_SECONDS, reserve=5), media_path.open('rb') as media_file:
                client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=recipe_work_budget.phase_seconds(recipe_work_budget.TRANSCRIBE_SECONDS, reserve=5),
                    max_retries=0,
                )
                try:
                    response = client.audio.transcriptions.create(
                        model=_DEFAULT_TRANSCRIPTION_MODEL, file=media_file, response_format='verbose_json',
                    )
                finally:
                    client.close()
        transcript = _safe_text(getattr(response, 'text', ''))
        segments = getattr(response, 'segments', None) or []
        if isinstance(segments, list):
            segments = [seg if isinstance(seg, dict) else getattr(seg, '__dict__', {}) for seg in segments]
        language = _safe_text(getattr(response, 'language', ''))
        if not transcript:
            raise RuntimeError('Audio transcription returned empty text.')
        # The two gates below are SCOPED TO THE APIFY CDN MEDIA PATH on purpose.
        # They were designed against Instagram audio-only reels, where the audio route
        # has ZERO successes today — so there a gate can only ever turn a 422 into a
        # better-reasoned 422. The yt-dlp path is a different story: it carries ~108
        # SUCCESSFUL TikTok transcriptions a day, and both gates have a real
        # false-negative surface there (non-English recipes fail GATE 1; quantity-free
        # method narration fails GATE 2 — 26% of social saves have no digit in their
        # ingredient list at all). Applying a rule inferred from one broken path across
        # a healthy one is exactly what the 07-25 empty-instructions guard did.
        if media_source == 'apify_media':
            # GATE 1 — did we transcribe actual speech, or hallucinate over silence/music?
            hallucinated, why = _transcript_looks_hallucinated(segments, language)
            if hallucinated:
                raise RuntimeError(f'Audio transcript rejected ({why}).')
            # GATE 2 — is there a recipe in the speech, or just narration about a dish?
            if not _transcript_has_recipe_signal(transcript):
                raise RuntimeError('Audio transcript has no recipe signal (no quantities and/or no steps).')
        _log_event(
            request_id,
            'audio_transcription_success',
            platform=(source_context or {}).get('platform'),
            resolved_url=resolved_url,
            transcription_model=_DEFAULT_TRANSCRIPTION_MODEL,
            latency_ms=int((time.time() - started) * 1000),
            media_source=media_source,
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
            media_source=media_source,
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
- An ingredient list is a valid, useful recipe on its own. If the source has a
  usable list of ingredients but no explicit cooking steps, that is STILL enough:
  return the title + ingredients and leave "instructions" as []. Do NOT set
  not_enough just because steps are missing.
- Only set not_enough=true when there is genuinely nothing to save — no usable
  ingredients AND no usable instructions:
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


def _repair_truncated_json(text):
    """Best-effort salvage of a truncated/unterminated JSON object.

    Closes an open string and any unbalanced braces/brackets so a model
    response that was cut off mid-generation still yields the recipe fields it
    did emit, instead of exploding the whole save with a 502.
    """
    s = _safe_text(text).strip()
    start = s.find('{')
    if start < 0:
        return None
    s = s[start:]
    in_str = False
    esc = False
    stack = []
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
    repaired = s
    if in_str:
        repaired += '"'
    repaired = repaired.rstrip().rstrip(',')
    for opener in reversed(stack):
        repaired += '}' if opener == '{' else ']'
    try:
        return json.loads(repaired)
    except Exception:
        return None


def _loads_recipe_json(text):
    """Parse recipe JSON, tolerating a truncated/unterminated model response."""
    cleaned = _safe_text(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return _repair_truncated_json(cleaned)


def _refine_recipe_structured(content, request_id=None, source_context=None):
    source_text = _safe_text(content)
    if not source_text:
        raise ServiceError('Provide source text to refine.', status_code=400)

    started = time.time()
    model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    client = _openai_client()
    messages = [
        {'role': 'system', 'content': _recipe_json_prompt()},
        {'role': 'user', 'content': source_text},
    ]

    # Attempt 1 forces JSON mode so the model cannot emit malformed/unterminated
    # JSON. On a parse miss we retry once WITHOUT json_object (fallback for models
    # that reject it) and best-effort-repair a truncated object. A rare
    # unrecoverable response degrades to a 422 retry prompt instead of the 502
    # that was tripping the main-grocery 5xx alarm.
    payload = None
    last_error = None
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                'model': model,
                'temperature': 0.1,
                'messages': messages,
            }
            if attempt == 1:
                kwargs['response_format'] = {'type': 'json_object'}
            response = client.chat.completions.create(**kwargs)
            payload = _loads_recipe_json(_strip_code_fences(response.choices[0].message.content))
            if payload is not None:
                break
            last_error = 'model returned unparseable recipe JSON'
        except Exception as exc:
            last_error = str(exc)
        _log_event(
            request_id,
            'recipe_refine_retry' if attempt < max_attempts else 'recipe_refine_failed',
            model=model,
            attempt=attempt,
            platform=(source_context or {}).get('platform'),
            resolved_url=(source_context or {}).get('resolved_url'),
            latency_ms=int((time.time() - started) * 1000),
            failure_reason=last_error,
        )

    if payload is None:
        # A user's recipe save that the LLM couldn't parse — surface it so it pages
        # (RecipesBackendError) instead of being invisible behind a generic 422.
        _report_backend_error('save_recipe_refine', code='unreadable',
                              error=last_error or 'LLM returned unparseable recipe',
                              job_id=(source_context or {}).get('resolved_url'))
        raise ServiceError(
            'Could not read this recipe right now — please try again.',
            status_code=503,
        )

    recipe = {
        'title': _safe_text(payload.get('title'))[:255],
        'ingredients': _clean_string_list(payload.get('ingredients') or []),
        'instructions': _clean_string_list(payload.get('instructions') or []),
        'notes': _clean_string_list(payload.get('notes') or []),
        'not_enough': bool(payload.get('not_enough')),
    }
    if not recipe['not_enough'] and not recipe['title'] and not recipe['ingredients'] and not recipe['instructions']:
        recipe['not_enough'] = True
    # Ingredients-only recipes are valid saves (product decision 2026-07-13). The
    # refine model is inconsistent about honoring this in the prompt, so override:
    # if it flagged not_enough but there ARE usable ingredients, keep the recipe.
    if recipe['not_enough'] and recipe['ingredients']:
        recipe['not_enough'] = False

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


# A caption this short cannot contain a recipe. Below this, asking a model to "extract the
# recipe" is asking it to invent one, and it obliges: a real save built 7 ingredients and 8
# steps out of the caption "#quickrecipes".
_MIN_GROUNDING_CHARS = int(os.getenv('MIN_RECIPE_GROUNDING_CHARS', '200'))
# Words that indicate the text actually describes cooking rather than just naming a dish.
_RECIPE_SIGNAL_RE = re.compile(
    r'\b(ingredient|cup|cups|tbsp|tsp|tablespoon|teaspoon|gram|grams|\d+\s*(g|kg|ml|oz|lb)\b'
    r'|preheat|bake|boil|simmer|saut|fry|mix|stir|whisk|blend|chop|dice|marinate|season'
    r'|step\s*\d|instructions|directions|recipe:)', re.I)


def _content_can_support_a_recipe(content):
    """Is there enough real source material here to EXTRACT a recipe, rather than invent one?

    The pipeline had no such test. `_recipe_is_incomplete` asks whether the OUTPUT has
    ingredients and steps — never whether they came from the input — so a hallucination
    passed every check and logged recipe_refine_success. Roughly 14% of video saves over the
    last 30 days were full recipes generated from a caption too short to contain one, which
    is what a user reported: a jerk chicken reel captioned only "#JerkChicken" came back with
    invented ingredients (including a tomato that was never in the video) and invented steps.

    A wrong recipe is worse than no recipe — silently so, because it looks right.
    """
    text = _safe_text(content)
    if len(text) >= _MIN_GROUNDING_CHARS:
        return True
    # Short but explicitly recipe-shaped (a terse ingredient list) is still real content.
    return bool(_RECIPE_SIGNAL_RE.search(text))


@recipe_work_budget.bounded
def _recipe_response_from_content(content, request_id=None, source_context=None):
    if not _content_can_support_a_recipe(content):
        _log_event(request_id, 'recipe_refine_refused_ungrounded',
                   content_chars=len(_safe_text(content)),
                   preview=_safe_text(content)[:80])
        return {'recipe': 'Not enough recipe information.', 'model': None}, None
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
    # Keep any media link the fallback discovered — the html provider has none, so this
    # is how an Apify-sourced media URL reaches the audio step when html won the chain.
    if _safe_text(fallback_result.get('media_url')) and not _safe_text(extraction.get('media_url')):
        extraction['media_url'] = fallback_result.get('media_url')


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
    if not _recipe_is_incomplete(response.get('recipe'), structured_recipe):
        return result, structured_recipe

    # Keep the best result any rung has produced. Rungs below may return something
    # THINNER than what we already have; without this the last roll wins and a 9-ingredient
    # result can be replaced by a 5-ingredient one.
    best_result = dict(result)
    best_structured = structured_recipe
    best_score = _recipe_completeness_score(structured_recipe)

    def _keep_if_better(candidate_result, candidate_structured, source_used, candidate_warnings):
        """Promote a rung's output to 'best so far' only if it beats the incumbent."""
        nonlocal best_result, best_structured, best_score
        score = _recipe_completeness_score(candidate_structured)
        if score <= best_score:
            return False
        best_result = {**result, **candidate_result}
        best_result['recipe_source_used'] = source_used
        best_result['warnings'] = list(candidate_warnings)
        best_structured = candidate_structured
        best_score = score
        return True

    warnings = list(extraction.get('warnings') or [])
    preview_fallback = None
    transcript_result = None
    # Only worth re-fetching from Apify if the chain did NOT already extract via Apify.
    # When it did, this fallback re-fetches the identical caption, merges it into itself
    # and re-refines — adding zero new information while handing the model a second,
    # unconstrained roll. On reels that deliberately withhold the method ("comment FULL
    # RECIPE below") that second roll fabricates a plausible-looking method from the dish
    # title, which is worse than an honest "couldn't find a recipe". Skipping keeps the
    # fallback for the case it was built for: the chain fell back to the thin html
    # caption, so Apify genuinely supplies text we don't have yet.
    already_extracted_via_apify = 'apify' in _safe_text(extraction.get('source')).lower()
    apify_fits, _rem, _need = _rung_fits('apify')
    if not apify_fits:
        _log_event(request_id, 'rung_skipped_deadline', rung='apify',
                   remaining_seconds=round(_rem, 2), needed_seconds=round(_need, 2))
    if extraction.get('platform') == 'instagram' and not already_extracted_via_apify and apify_fits:
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
            # This rung used to overwrite `result` unconditionally, BEFORE testing the new
            # roll - so a worse Apify roll replaced a better caption one. Promote only if better.
            if _keep_if_better(refined_with_apify, structured_with_apify, 'content+apify', warnings):
                structured_recipe = structured_with_apify
            if not _recipe_is_incomplete(refined_with_apify.get('recipe'), structured_with_apify):
                result.update(refined_with_apify)
                result['recipe_source_used'] = 'content+apify'
                result['warnings'] = list(warnings)
                return result, structured_with_apify
        except Exception as exc:
            warnings.append(str(exc))

    if not _audio_fallback_supported(extraction.get('platform')):
        best_result['warnings'] = warnings
        return best_result, best_structured

    # Fast slide-OCR fallback (Rekognition) — for photo slideshows whose recipe
    # lives in the slide images, not the caption. Runs BEFORE the slower VLM
    # preview path below; only when we actually have carousel image URLs.
    slide_fits, _rem, _need = _rung_fits('slide_ocr')
    if extraction.get('image_urls') and not slide_fits:
        _log_event(request_id, 'rung_skipped_deadline', rung='slide_ocr',
                   remaining_seconds=round(_rem, 2), needed_seconds=round(_need, 2))
    if extraction.get('image_urls') and slide_fits:
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
                ocr_warnings = warnings + [
                    'Caption text was insufficient, so slide-image OCR was used as a fallback.'
                ]
                # This rung is meant for photo slideshows, but it fires on any post that has
                # image_urls - for a VIDEO that is the single cover frame, and OCR of a cover
                # frame yields a title and nothing else. Under the old sentinel-only test that
                # title-only result counted as complete and returned here, so audio never ran.
                _keep_if_better(refined_with_ocr, structured_with_ocr, 'content+slide_ocr', ocr_warnings)
                if not _recipe_is_incomplete(refined_with_ocr.get('recipe'), structured_with_ocr):
                    result.update(refined_with_ocr)
                    result['recipe_source_used'] = 'content+slide_ocr'
                    result['warnings'] = ocr_warnings
                    return result, structured_with_ocr
        except Exception as exc:
            warnings.append(str(exc))

    preview_fits, _rem, _need = _rung_fits('preview_ocr')
    if not preview_fits:
        _log_event(request_id, 'rung_skipped_deadline', rung='preview_ocr',
                   remaining_seconds=round(_rem, 2), needed_seconds=round(_need, 2))
    try:
        if not preview_fits:
            raise _DeadlineSkip('preview_ocr')
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
        preview_warnings = warnings + [
            'Caption text was insufficient, so preview-image OCR was used as a fallback.'
        ]
        _keep_if_better(refined_with_preview, structured_with_preview, 'content+image_ocr', preview_warnings)
        if not _recipe_is_incomplete(refined_with_preview.get('recipe'), structured_with_preview):
            result.update(refined_with_preview)
            result['recipe_source_used'] = 'content+image_ocr'
            result['warnings'] = preview_warnings
            return result, structured_with_preview
    except Exception as exc:
        warnings.append(str(exc))

    try:
        _audio_fits, _rem, _need = _rung_fits('audio')
        if not _audio_fits or _audio_budget_exhausted():
            # Out of gateway headroom. Skip the audio leg so the request returns an honest
            # result instead of being cut off mid-flight by the 30s integration timeout.
            elapsed = _sync_elapsed_seconds()
            _log_event(
                request_id,
                'audio_skipped_time_budget',
                platform=extraction.get('platform'),
                resolved_url=extraction.get('resolved_url'),
                elapsed_seconds=round(elapsed, 2) if elapsed is not None else None,
                budget_seconds=_SYNC_AUDIO_BUDGET_SECONDS,
            )
            best_result['warnings'] = warnings + [
                'Audio transcription was skipped because the request ran out of time.'
            ]
            return best_result, best_structured
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
        best_result['warnings'] = warnings + [str(exc)]
        return best_result, best_structured

    try:
        refined_with_audio, structured_with_audio = _recipe_response_from_content(
            merged_content,
            request_id=request_id,
            source_context={**extraction, 'content': merged_content},
        )
        # Also an UNCONDITIONAL overwrite before the check, like the Apify rung above.
        _keep_if_better(refined_with_audio, structured_with_audio, 'content+audio', warnings)
        if not _recipe_is_incomplete(refined_with_audio.get('recipe'), structured_with_audio):
            result.update(refined_with_audio)
            return result, structured_with_audio
    except Exception as exc:
        best_result['warnings'] = list(best_result.get('warnings') or []) + [str(exc)]
        return best_result, best_structured

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
            all_warnings = warnings + [
                'Caption text was insufficient, so preview-image OCR and audio transcription were used as fallbacks.'
            ]
            _keep_if_better(refined_with_all, structured_with_all, 'content+image_ocr+audio', all_warnings)
            if not _recipe_is_incomplete(refined_with_all.get('recipe'), structured_with_all):
                result.update(refined_with_all)
                result['recipe_source_used'] = 'content+image_ocr+audio'
                result['warnings'] = all_warnings
                return result, structured_with_all
        except Exception as exc:
            best_result['warnings'] = list(best_result.get('warnings') or []) + [str(exc)]
            return best_result, best_structured
    # Ladder exhausted with nothing complete: return the RICHEST result any rung produced,
    # not simply the last one. Previously the final roll won even when it was thinner.
    return best_result, best_structured


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
    # Migration dual-write: mirror the enriched image fields into shared_saved_recipes
    # (re-upsert the row) so generated images survive the read cutover. Non-blocking.
    _dual_write_saved_recipe_to_shared(conn, owner, recipe_id)


def _valid_recipe_repair_url(url):
    """Reject permanent non-web inputs before dispatch and on queued redelivery."""
    from urllib.parse import urlsplit
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlsplit(url.strip())
        return (parsed.scheme.lower() in ('http', 'https') and bool(parsed.hostname)
                and parsed.username is None and parsed.password is None
                and not any(char.isspace() for char in parsed.netloc))
    except ValueError:
        return False



def _invoke_saved_recipe_repair_async(owner, recipe_id, url, request_id=None):
    """Hand an unreadable save to the background repair agent.

    The sync save is bounded by the gateway's 29s, which is why audio transcription gets
    skipped and why a recipe can end up with nothing real in it. This runs with the Lambda's
    full timeout and no user waiting, so it can afford the leg that actually recovers the
    steps for a talking video.
    """
    function_name = _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not function_name or not _valid_recipe_repair_url(url):
        return False
    try:
        response = boto3.client('lambda').invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps({
                'async_task': _ASYNC_TASK_REPAIR_SAVED_RECIPE,
                'owner': owner, 'recipe_id': recipe_id, 'url': url,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        if response.get('StatusCode') != 202:
            return False
        _log_event(request_id, 'saved_recipe_repair_enqueued', owner=owner, recipe_id=recipe_id)
        return True
    except Exception as exc:
        _log_event(request_id, 'saved_recipe_repair_enqueue_failed',
                   owner=owner, recipe_id=recipe_id, failure_reason=str(exc)[:160])
        return False


def _saved_recipe_content_revision(row):
    """Opaque optimistic revision; no schema migration for installed clients.

    Include source, status, content and admission identity. Network/model work must not
    hold row locks, and any concurrent edit or source replacement wins.
    """
    fields = ('_id', '_owner', 'source_url', 'resolved_url', 'title', 'ingredients',
              'instructions', 'notes', 'extraction_source', 'recipe_source_used',
              'status', '_createdDate')
    values = {field: row.get(field) for field in fields}
    for field in ('ingredients', 'instructions', 'notes'):
        if isinstance(values[field], str):
            try:
                values[field] = json.loads(values[field])
            except ValueError:
                pass
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def _enqueue_saved_recipe_repair_or_finish(conn, owner, recipe_id, url, request_id=None):
    """A failed admission must not leave an unscheduled bookmark processing forever."""
    before = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
    if not before or before.get('status') != 'repairing':
        return False
    revision = _saved_recipe_content_revision(before)
    if _invoke_saved_recipe_repair_async(owner, recipe_id, url, request_id=request_id):
        return True
    table = _saved_recipes_table(owner)
    fields = _drift_safe_select(conn, table)
    conn.begin()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {fields} FROM `{table}` WHERE _id=%s AND _owner=%s FOR UPDATE", [recipe_id, owner])
            row = cur.fetchone()
            if not row or _saved_recipe_content_revision(row) != revision:
                conn.rollback()
                return False
            cur.execute(f"UPDATE `{table}` SET status='failed' WHERE _id=%s AND _owner=%s", [recipe_id, owner])
        _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=request_id,
            transactional=True, required=os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true')
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    _log_event(request_id, 'saved_recipe_repair_outcome', owner=owner, recipe_id=recipe_id,
               outcome='link_retained', reason='dispatch_unavailable')
    return False


def _repair_saved_recipe(owner, recipe_id, url, request_id=None):
    """Recover an empty bookmark, without replacing content of unknown provenance.

    Legacy rows do not identify human versus generated fields. Nonempty content
    therefore belongs to the user; list length is never permission to replace it.
    Compare the pre-extraction revision under lock and commit required mirrors in
    the same transaction. A late job cannot resurrect a deleted recipe.
    """
    if not _valid_recipe_repair_url(url):
        return {'error': 'invalid source URL', 'code': 'invalid_source_url', 'retryable': False}
    conn = _mysql_conn()
    try:
        row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        if not row:
            return {'repaired': False, 'outcome': 'skipped', 'reason': 'deleted'}
        sources = [row.get('source_url'), row.get('resolved_url')]
        if not any(source and _resolved_url_hash(source) == _resolved_url_hash(url) for source in sources):
            return {'repaired': False, 'outcome': 'skipped', 'reason': 'source_changed'}
        revision = _saved_recipe_content_revision(row)
        current = _serialize_row(row)
        before = (len(current.get('ingredients') or []), len(current.get('instructions') or []))
        current_complete = recipe_content_complete(current)
        if current_complete and row.get('status') == 'ready':
            return {'repaired': False, 'outcome': 'ready', 'reason': 'existing_content', 'before': before}

        _mark_sync_request_start(False)
        try:
            extraction = _extract_content(url, request_id=request_id)
            response, structured = _analyze_extraction(extraction, request_id=request_id)
        except Exception as exc:
            # End processing honestly; the lock below still protects newer edits.
            _log_event(request_id, 'saved_recipe_repair_extraction_failed', owner=owner, recipe_id=recipe_id, failure_type=type(exc).__name__)
            response, structured = {}, {}
        structured = structured or {}
        ing = _clean_string_list(structured.get('ingredients') or [])
        ins = _clean_string_list(structured.get('instructions') or [])
        after = (len(ing), len(ins))
        # A coherent source-backed recipe needs ingredients AND directions.
        # Preserve partial old content for explicit user recovery, rather than
        # silently combining lists extracted on different occasions.
        replace_empty = before == (0, 0) and recipe_content_complete({'ingredients': ing, 'instructions': ins})
        pending_note = "We couldn't read the full recipe from this post yet — we're still working on it."
        # Remove only our own obsolete processing copy. Other notes may be human
        # edits and must survive; a failed re-extraction is never proof otherwise.
        recovered_notes = [note for note in (current.get('notes') or []) if note != pending_note]
        ready = current_complete or replace_empty
        table = _saved_recipes_table(owner)
        select_fields = _drift_safe_select(conn, table)
        conn.begin()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {select_fields} FROM `{table}` WHERE _id=%s AND _owner=%s FOR UPDATE",
                            [recipe_id, owner])
                locked = cur.fetchone()
                if not locked or _saved_recipe_content_revision(locked) != revision:
                    conn.rollback()
                    return {'repaired': False, 'outcome': 'skipped', 'reason': 'changed_or_deleted'}
                if replace_empty:
                    cur.execute(f"""UPDATE `{table}` SET title=%s, ingredients=%s, instructions=%s, notes=%s,
                                extraction_source=%s, recipe_source_used=%s, status='ready', _updatedDate=NOW()
                                WHERE _id=%s AND _owner=%s""",
                                [current.get('title') or structured.get('title') or 'Recipe',
                                 json.dumps(ing), json.dumps(ins), json.dumps(recovered_notes),
                                 _append_extraction_source(current.get('extraction_source'), 'repair'),
                                 _safe_text(response.get('recipe_source_used')) or None, recipe_id, owner])
                else:
                    cur.execute(f"UPDATE `{table}` SET status=%s WHERE _id=%s AND _owner=%s",
                                ['ready' if ready else 'failed', recipe_id, owner])
            require_shared = os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true'
            _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=request_id,
                                              transactional=True, required=require_shared)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        outcome = 'ready' if ready else 'link_retained'
        _log_event(request_id, 'saved_recipe_repair_outcome', owner=owner, recipe_id=recipe_id,
                   outcome=outcome, replaced_empty=replace_empty, before=str(before), after=str(after))
        return {'repaired': replace_empty, 'outcome': outcome, 'before': before, 'after': after}
    finally:
        conn.close()


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


def _invoke_personalize_warm(owner, recipe_ids, request_id=None):
    """Self-invoke to compute + cache the recipes deferred past the sync cap.
    Mirrors the other async self-invoke helpers; never raises (background best-effort)."""
    function_name = _safe_text(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not function_name:
        _log_event(request_id, 'saved_personalize_warm_invoke_skipped', owner=owner,
                   failure_reason='AWS_LAMBDA_FUNCTION_NAME is not available')
        return False
    try:
        boto3.client('lambda').invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps({
                'async_task': _ASYNC_TASK_PERSONALIZE_WARM,
                'owner': owner,
                'recipe_ids': recipe_ids,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        _log_event(request_id, 'saved_personalize_warm_invoked', owner=owner,
                   count=len(recipe_ids or []))
        return True
    except Exception as exc:
        _log_event(request_id, 'saved_personalize_warm_invoke_failed', owner=owner,
                   failure_reason=str(exc))
        return False


def _handle_personalize_warm_task(event, request_id=None):
    """Background task: compute availability for the deferred recipe_ids and persist to the
    cache table so subsequent personalize polls/opens hit the cache. Never re-fires warm."""
    owner = _safe_text((event or {}).get('owner'))
    recipe_ids = [
        _safe_text(rid) for rid in ((event or {}).get('recipe_ids') or []) if _safe_text(rid)
    ]
    if not owner or not recipe_ids:
        return {'statusCode': 200, 'body': 'warm-noop'}
    conn = _mysql_conn()
    try:
        _ensure_recipe_personalization_tables(conn)
        kitchen_context = _build_kitchen_match_context(conn, owner)
        current_version = int((kitchen_context or {}).get('kitchen_version') or 0)
        cached = _get_cached_recipe_availability(conn, owner, recipe_ids,
                                                 min_kitchen_version=current_version)
        todo_ids = [rid for rid in recipe_ids if rid not in cached]
        if not todo_ids:
            _log_event(request_id, 'saved_personalize_warm', owner=owner, warmed=0)
            return {'statusCode': 200, 'body': 'warm'}
        # Load the recipe rows using the SAME read path _personalize_saved_recipes uses
        # (respect READ_SHARED_SAVED_RECIPES + table existence identically).
        table = _saved_recipes_table(owner)
        use_shared = os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true'
        if not use_shared:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) AS n FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table])
                if cur.fetchone()['n'] == 0:
                    _log_event(request_id, 'saved_personalize_warm', owner=owner, warmed=0)
                    return {'statusCode': 200, 'body': 'warm'}
            _ensure_saved_recipes_table(conn, owner)
        with conn.cursor() as cur:
            if use_shared:
                cur.execute(
                    f"""SELECT {_drift_safe_select(conn, 'shared_saved_recipes', extra_fields=['image_mirror_failed_at', 'image_mirror_failure'])}
                        FROM `shared_saved_recipes`
                        WHERE {_household_owner_filter(conn, owner)[0]}
                        ORDER BY COALESCE(_updatedDate, _createdDate) DESC""",
                    _household_owner_filter(conn, owner)[1],
                )
            else:
                cur.execute(
                    f"""SELECT {_drift_safe_select(conn, table)}
                        FROM `{table}`
                        ORDER BY COALESCE(_updatedDate, _createdDate) DESC""",
                )
            rows = cur.fetchall() or []
        todo_set = set(todo_ids)
        recipes = [r for r in (_serialize_row(row) for row in rows)
                   if r.get('id') in todo_set]
        warmed = 0
        for start in range(0, len(recipes), _PERSONALIZE_WARM_CHUNK):
            batch = recipes[start:start + _PERSONALIZE_WARM_CHUNK]
            if not batch:
                continue
            computed = _compute_saved_recipe_availability_batch(
                batch, kitchen_context, request_id=request_id) or {}
            # _compute_* does NOT persist; write the cache ourselves so warm actually helps.
            overlay_rows = [
                _build_owner_recipe_availability_record(owner, 'saved', rid, avail)
                for rid, avail in computed.items() if rid and avail
            ]
            _persist_owner_recipe_availability_rows(conn, overlay_rows)
            warmed += len(overlay_rows)
        _log_event(request_id, 'saved_personalize_warm', owner=owner, warmed=warmed)
        return {'statusCode': 200, 'body': 'warm'}
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
        existing = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        conn.commit()
        if not existing or existing.get('status') != 'processing':
            return {'statusCode': 200}
        extraction = _build_text_recipe_extraction(content)
        recipe_response, structured_recipe = _analyze_extraction(extraction, request_id=request_id)
        if recipe_response['recipe'] == 'Not enough recipe information.':
            _mark_saved_recipe_text_failed(conn, owner, recipe_id, request_id=request_id)
            _log_event(request_id, 'saved_recipe_async_text_failed', owner=owner, recipe_id=recipe_id, reason='not_enough_info')
            # The pasted text simply had no recipe in it. That is an upstream CONTENT
            # outcome the user already sees, not a fault in our backend — so record a
            # low-severity marker instead of paging, mirroring the blocked_host pattern.
            # Genuine extraction crashes still reach _report_backend_error elsewhere.
            _log_event(request_id, 'not_enough_info', owner=owner, recipe_id=recipe_id,
                       severity='low', source='save_recipe_text')
            return {'statusCode': 200}
        table = _saved_recipes_table(owner)
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE `{table}` SET
                    title = %s, ingredients = %s, instructions = %s, notes = %s,
                    extraction_source = %s, status = 'ready'
                WHERE _id = %s AND _owner = %s AND status = 'processing'""",
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
            changed = cur.rowcount
        if not changed:
            conn.commit()
            return {'statusCode': 200}
        _dual_write_saved_recipe_to_shared(conn, owner, recipe_id,
                                          request_id=request_id, transactional=True)
        conn.commit()
        # Saved recipes show a real photo or a title emoji, never an invented one. Skip the
        # invoke entirely rather than paying for a Lambda that returns immediately.
        if GENERATE_RECIPE_IMAGES:
            _invoke_saved_recipe_image_generation_async(owner, recipe_id, request_id=request_id)
        # Compute kitchen availability
        row = _fetch_saved_recipe_by_id(conn, owner, recipe_id)
        if row:
            _ensure_recipe_personalization_tables(conn)
            kitchen_context = _build_kitchen_match_context(conn, owner)
            _, overlay_row = _serialize_saved_recipe_with_availability(
                conn, owner, row, kitchen_context=kitchen_context, request_id=request_id)
            _persist_owner_recipe_availability_rows(conn, [overlay_row])
            _apply_saved_recipe_meal_category(
                conn, owner, recipe_id,
                structured_recipe['title'] or extraction.get('title'),
                structured_recipe['ingredients'],
                request_id=request_id,
            )
            _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=request_id)
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
        conn.rollback()
        try:
            _mark_saved_recipe_text_failed(conn, owner, recipe_id, request_id=request_id)
        except Exception:
            # An asynchronous Lambda return code does not trigger a retry.
            raise RuntimeError('saved_recipe_terminal_persistence_failed') from None
        return {'statusCode': 500}
    finally:
        conn.close()


def _refresh_saved_recipe_generated_image(conn, owner, recipe_id, request_id=None):
    """Attach a hero image to a saved recipe — from its SOURCE, never invented.

    This used to synthesise a photorealistic food photo whenever a save had no image of its
    own. That is every text-based save: Use What I Have recipes, and Explore agent recipes.
    The result was 1,602 recipes across 483 users carrying a picture of a dish nobody cooked,
    of food that was never photographed — and it reads as fake, because it is.

    A recipe with no real photo now gets no photo, and the app draws a title-derived emoji
    instead, exactly as Use What I Have already does on its own screen.
    """
    if not GENERATE_RECIPE_IMAGES:
        _log_event(request_id, 'saved_recipe_image_generation_disabled',
                   owner=owner, recipe_id=recipe_id)
        return None
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


def _hide_unavailable_recipe_image(row):
    # Response-only fallback. Keep original URLs and stored recipe data so a
    # replacement upload/recovery can still restore the image later.
    row['image_url'] = None
    row['image_urls'] = []
    return row


def _ensure_owned_saved_recipe_image(conn, owner, row, request_id=None):
    if not row:
        return row
    if _row_mirror_failed_recently(row):
        metadata = row.get('image_mirror_failure')
        try:
            metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        except (ValueError, TypeError):
            metadata = None
        if isinstance(metadata, dict) and metadata.get('kind') == 'source_unavailable':
            return _hide_unavailable_recipe_image(row)
        return row
    # Out of budget for this request: hand the row back untouched. It keeps its original
    # image URL and gets mirrored on a later request. Serving the list is what matters.
    if _image_mirror_budget_left() <= 0:
        return row
    # Known-dead image: skip entirely rather than re-pay the timeout on every single
    # read. This is what stops one unreachable host taxing a user's list forever.
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
    attempted_any = False
    permanent_failures = []
    for candidate_url in candidate_urls:
        if _image_mirror_budget_left() <= 0:
            permanent_failures.append(False)  # Remaining candidates were not classified.
            break
        if _mirror_url_recently_failed(candidate_url):
            permanent_failures.append(_image_mirror_failures.get(candidate_url) == float('inf'))
            continue                      # already known dead, in this container
        attempted_any = True
        attempt_started = time.time()
        try:
            image_fields = _mirror_recipe_image(owner, recipe_id, candidate_url, request_id=request_id)
            _update_saved_recipe_image_fields(conn, owner, recipe_id, image_fields)
            row.update(image_fields)
            return row
        except Exception as error:
            permanent = (
                isinstance(error, requests.HTTPError)
                and getattr(error.response, 'status_code', None) in (403, 404, 410)
            ) or (
                isinstance(error, ServiceError)
                and (error.extra or {}).get('error_code') == 'source_unavailable'
            )
            permanent_failures.append(permanent)
            _note_mirror_failure(candidate_url, permanent=permanent)
            continue
        finally:
            # Charge every attempt, successful or not. Failures are the expensive ones.
            _image_mirror_spent['seconds'] += time.time() - attempt_started
    if attempted_any or (permanent_failures and all(permanent_failures)):
        # Persist the failure so a cold container inherits the knowledge instead of
        # rediscovering it at the user's expense.
        _mark_mirror_failed_in_db(conn, owner, recipe_id,
                                 failure_kind='source_unavailable' if permanent_failures and all(permanent_failures) else 'transient',
                                 source_url=current_image_url)
        if permanent_failures and all(permanent_failures):
            return _hide_unavailable_recipe_image(row)

    refresh_url = _safe_text(row.get('source_url') or row.get('resolved_url'))
    if not refresh_url:
        return row
    # Re-extracting the page is far more expensive than the mirror fetch that just failed.
    # Never start one on a read once the budget is gone.
    if _image_mirror_budget_left() <= 0:
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
            f"""SELECT {_drift_safe_select(conn, table)}
                FROM `{table}`
                WHERE _id = %s
                LIMIT 1""",
            [item_id]
        )
        return cur.fetchone()


def _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash):
    """Accepts a single hash or a list of candidate hashes (canonical + legacy raw)."""
    hashes = resolved_url_hash if isinstance(resolved_url_hash, (list, tuple)) else [resolved_url_hash]
    hashes = [h for h in hashes if h]
    if not hashes:
        return None
    table = _saved_recipes_table(owner)
    placeholders = ', '.join(['%s'] * len(hashes))
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT {_drift_safe_select(conn, table)}
                FROM `{table}`
                WHERE resolved_url_hash IN ({placeholders})
                LIMIT 1""",
            list(hashes)
        )
        return cur.fetchone()


def _read_own_saved_recipe_rows(conn, owner, limit, before):
    """Existing single-owner read (source of truth). Returns (rows, has_more)."""
    table = _saved_recipes_table(owner)
    # Shared-table read cutover (reversible via env flag, default off = per-owner).
    # MANDATORY owner_id filter. Internal fetch-by-id/hash + mutations stay per-owner
    # (source of truth); only this user-facing list read flips.
    use_shared = os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true'
    if not use_shared:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if cur.fetchone()['n'] == 0:
                return [], False
        _ensure_saved_recipes_table(conn, owner)
    with conn.cursor() as cur:
        params = []
        where_parts = []
        if use_shared:
            from_table = 'shared_saved_recipes'
            from_ref = '`shared_saved_recipes`'
            where_parts.append("owner_id = %s")
            params.append(owner)
        else:
            from_table = table
            from_ref = f'`{table}`'
        if before:
            where_parts.append("COALESCE(_updatedDate, _createdDate) < %s")
            params.append(before)
        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        cur.execute(
            f"""SELECT {_drift_safe_select(conn, from_table, extra_fields=['image_mirror_failed_at', 'image_mirror_failure'])}
                FROM {from_ref}
                {where_clause}
                ORDER BY COALESCE(_updatedDate, _createdDate) DESC
                LIMIT {limit + 1}""",
            params,
        )
        rows = cur.fetchall() or []
    has_more = len(rows) > limit
    return rows[:limit], has_more


def _read_household_saved_recipe_rows(conn, acting_owner, member_ids, limit, before):
    """Household read-union: merge every member's `{member}_saved_recipes` (same fields
    /order/limit as own reads), sort by recency DESC, then collapse duplicates by
    resolved_url_hash — preferring the ACTING user's own copy so tapping opens an
    editable row — while keeping null-hash rows as-is. Limit is applied AFTER merge.
    Returns (rows, has_more). Each row carries `_owner` (for saved_by) and
    `resolved_url_hash` (popped from the serialized payload later)."""
    acting = _safe_owner_token(acting_owner)
    collected = []
    for member_id in member_ids:
        table = _saved_recipes_table(member_id)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [table])
            if (cur.fetchone() or {}).get('n', 0) == 0:
                continue  # member never saved a recipe yet
            # Drift-tolerant per-member SELECT: legacy tables lacking source_image_url (etc)
            # NULL-fill instead of 500ing the whole household read (the bug this fixes).
            fields = _drift_safe_select(conn, table, extra_fields=['resolved_url_hash'])
            params = []
            where = ""
            if before:
                where = "WHERE COALESCE(_updatedDate, _createdDate) < %s"
                params.append(before)
            cur.execute(
                f"""SELECT {fields}
                    FROM `{table}`
                    {where}
                    ORDER BY COALESCE(_updatedDate, _createdDate) DESC
                    LIMIT {limit + 1}""",
                params,
            )
            collected.extend(cur.fetchall() or [])

    def _recency(r):
        return r.get('_updatedDate') or r.get('_createdDate') or datetime.min

    collected.sort(key=_recency, reverse=True)

    seen = {}          # resolved_url_hash -> index into deduped
    deduped = []
    for r in collected:
        h = _safe_text(r.get('resolved_url_hash'))
        if not h:
            deduped.append(r)  # null/blank hash never collapses (legacy rows)
            continue
        if h not in seen:
            seen[h] = len(deduped)
            deduped.append(r)
        else:
            idx = seen[h]
            existing = deduped[idx]
            # Prefer the acting user's own copy for the surviving row's content.
            if _safe_owner_token(r.get('_owner')) == acting and _safe_owner_token(existing.get('_owner')) != acting:
                deduped[idx] = r

    has_more = len(deduped) > limit
    return deduped[:limit], has_more


def _stamp_saved_by(conn, recipes):
    """Stamp each serialized recipe with `saved_by` (the row's owner user_id) and
    `saved_by_name` (first_name from new_users; one query for all owners; fallback
    'Housemate'). Applied in BOTH scopes (cheap). Never raises."""
    owner_ids = list(dict.fromkeys(
        _safe_owner_token(r.get('_owner')) for r in recipes if r.get('_owner')
    ))
    owner_ids = [o for o in owner_ids if o]
    names = {}
    if owner_ids:
        try:
            placeholders = ','.join(['%s'] * len(owner_ids))
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT user_id, first_name FROM new_users WHERE user_id IN ({placeholders})",
                    owner_ids,
                )
                for r in (cur.fetchall() or []):
                    uid = _safe_owner_token(r.get('user_id'))
                    if uid:
                        names[uid] = _safe_text(r.get('first_name')) or 'Housemate'
        except Exception:
            names = {}
    for r in recipes:
        oid = _safe_owner_token(r.get('_owner'))
        r['saved_by'] = oid or None
        r['saved_by_name'] = names.get(oid) or 'Housemate'
    return recipes


def _get_saved_recipes(owner, query=None, request_id=None):
    conn = _mysql_conn()
    limit = _parse_limit(query)
    before = _parse_before(query)
    scope = _safe_text((query or {}).get('scope')).lower()

    # Household read-union (additive; default = own). Solo users, or any non-household
    # scope, fall through to the identical own-table read (cheap early-out).
    member_ids = _get_household_member_ids(conn, owner) if scope == 'household' else []
    others = [m for m in member_ids if m != _safe_owner_token(owner)]
    if scope == 'household' and others:
        rows, has_more = _read_household_saved_recipe_rows(conn, owner, member_ids, limit, before)
        effective_scope = 'household'
    else:
        rows, has_more = _read_own_saved_recipe_rows(conn, owner, limit, before)
        effective_scope = 'own'

    # Self-heal images against each row's REAL owner table so a household read never
    # cross-writes one member's recipe into the acting user's table. Budget is per
    # request: Lambda containers are reused, so a stale spend would starve later calls.
    _reset_image_mirror_budget()
    rows = [
        _ensure_owned_saved_recipe_image(conn, _safe_owner_token(r.get('_owner')) or owner, r, request_id=request_id)
        for r in rows
    ]
    serialized_rows = [_serialize_row(row) for row in rows]
    for s in serialized_rows:
        s.pop('resolved_url_hash', None)  # keep the payload field set identical to own reads
    recipe_ids = [s.get('id') for s in serialized_rows if s.get('id')]
    # Availability is keyed to the ACTING owner's kitchen (housemate copies simply
    # show availability=None until personalized — same as an uncomputed own recipe).
    current_kv = _get_owner_kitchen_version(conn, owner)
    availability_map = _get_cached_recipe_availability(conn, owner, recipe_ids, min_kitchen_version=current_kv)
    recipes = []
    for serialized in serialized_rows:
        avail = availability_map.get(serialized.get('id'))
        if avail:
            serialized['availability'] = _availability_summary(avail)
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
    _stamp_saved_by(conn, recipes)
    return _success({
        'owner': owner,
        'recipes': recipes,
        'count': len(recipes),
        'limit': limit,
        'has_more': has_more,
        'scope': effective_scope,
    })


def _personalize_saved_recipes(owner, request_id=None, recipe_ids=None):
    """Phase 2: run the LLM batch, persist results, and return per-recipe personalization data."""
    conn = _mysql_conn()
    try:
        table = _saved_recipes_table(owner)
        # Shared-table read cutover (reversible via env flag, default off = per-owner).
        use_shared = os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true'
        if not use_shared:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) AS n FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = %s
                """, [table])
                if cur.fetchone()['n'] == 0:
                    return _success({'owner': owner, 'recipes': []})
            _ensure_saved_recipes_table(conn, owner)
        with conn.cursor() as cur:
            if use_shared:
                cur.execute(
                    f"""SELECT {_drift_safe_select(conn, 'shared_saved_recipes', extra_fields=['image_mirror_failed_at', 'image_mirror_failure'])}
                        FROM `shared_saved_recipes`
                        WHERE {_household_owner_filter(conn, owner)[0]}
                        ORDER BY COALESCE(_updatedDate, _createdDate) DESC""",
                    _household_owner_filter(conn, owner)[1],
                )
            else:
                cur.execute(
                    f"""SELECT {_drift_safe_select(conn, table)}
                        FROM `{table}`
                        ORDER BY COALESCE(_updatedDate, _createdDate) DESC""",
                )
            rows = cur.fetchall() or []
        _ensure_recipe_personalization_tables(conn)
        kitchen_context = _build_kitchen_match_context(conn, owner)
        serialized_rows = [_serialize_row(row) for row in rows]
        if recipe_ids:
            serialized_rows = [r for r in serialized_rows if r.get('id') in recipe_ids]

        # Cache-read: availability is deterministic for a given (recipe, kitchen_version),
        # so reuse fresh cached rows and only run the LLM matcher on cache misses / stale
        # entries. Warm calls (kitchen unchanged) do zero LLM work. The per-call compute is
        # capped so a cold recompute can never exceed the API-gateway timeout — any overflow
        # is left uncached this call and fills in on the next call.
        current_version = int((kitchen_context or {}).get('kitchen_version') or 0)
        all_ids = [r.get('id') for r in serialized_rows if r.get('id')]
        cached_map = _get_cached_recipe_availability(conn, owner, all_ids, min_kitchen_version=current_version)
        pending = [r for r in serialized_rows if r.get('id') and r.get('id') not in cached_map]
        to_compute = pending[:_MAX_PERSONALIZE_SYNC_COMPUTE]
        computed_map = (
            _compute_saved_recipe_availability_batch(to_compute, kitchen_context, request_id=request_id)
            if to_compute else {}
        )
        availability_map = dict(cached_map)
        availability_map.update(computed_map)
        _log_event(request_id, 'saved_personalize_availability', owner=owner,
                   total=len(all_ids), from_cache=len(cached_map),
                   computed=len(computed_map), deferred=max(0, len(pending) - len(to_compute)))
        # Warm the recipes deferred past the sync cap in a background self-invoke so the
        # next poll/open hits the cache. Non-fatal: never breaks the sync response.
        remaining_ids = [r.get('id') for r in pending[len(to_compute):] if r.get('id')]
        if _PERSONALIZE_ASYNC_WARM and remaining_ids:
            _invoke_personalize_warm(owner, remaining_ids[:_PERSONALIZE_WARM_MAX], request_id=request_id)
        personalization_list = []
        overlay_rows = []
        for serialized in serialized_rows:
            recipe_id = serialized.get('id')
            avail = availability_map.get(recipe_id) or {}
            preserve = getattr(recipe_inventory_llm, 'confirmed_availability_for_response', None)
            if preserve and recipe_id in computed_map:
                avail = preserve(conn, owner, 'saved', recipe_id, avail)
            personalization_list.append({
                'id': recipe_id,
                'availability': _availability_summary(avail) if avail else None,
                'ingredient_matches': avail.get('ingredient_matches'),
                'substitution_candidates': avail.get('substitution_candidates'),
                'substitution_summary': avail.get('substitution_summary'),
                'substitution_status': avail.get('substitution_status'),
            })
            if avail and recipe_id and recipe_id in computed_map:
                overlay_rows.append(_build_owner_recipe_availability_record(owner, 'saved', recipe_id, avail))
        _persist_owner_recipe_availability_rows(conn, overlay_rows)
        return _success({'owner': owner, 'recipes': personalization_list})
    finally:
        pass


# --- Explore short-circuit ----------------------------------------------------
# When a user saves a URL that already exists as a curated explore_recipes row,
# skip scrape/Apify/LLM-refine entirely and build the saved recipe straight from
# that row. explore_recipes lives in the SAME database/schema as saved recipes
# (both Lambdas use DB_HOST/DB_NAME -> database-1 / mysqlTutorial), so this is a
# plain read on the shared table. De-attribution is preserved: creator fields are
# never carried into the saved copy.
_EXPLORE_SYNTHETIC_URL_RE = re.compile(
    r'^https?://(?:www\.)?trepo\.ai/explore/([0-9a-fA-F-]{36})/?$', re.IGNORECASE)
# The app submits curated-recipe saves as an Instagram PROFILE URL that opens the
# Trepo IG page for humans while carrying the recipe id in a query param IG
# ignores: https://www.instagram.com/trepohq?trepo_recipe={id}
_EXPLORE_IG_PROFILE_HOSTS = {'instagram.com', 'www.instagram.com'}
_EXPLORE_IG_PROFILE_PATH = '/trepohq'


def _explore_synthetic_id(raw_url):
    """Parse a curated-recipe id from either synthetic scheme, reading the RAW
    submitted URL BEFORE any canonicalization can strip the query string:
      - https://trepo.ai/explore/{id}
      - https://www.instagram.com/trepohq?trepo_recipe={id}
    Returns (explore_id_or_None, is_trepo_profile). is_trepo_profile is True for ANY
    instagram.com/trepohq URL (even with a missing/unknown param) so the caller can
    return 422 instead of EVER scraping the profile page."""
    text = _safe_text(raw_url).strip()
    if not text:
        return None, False
    m = _EXPLORE_SYNTHETIC_URL_RE.match(text)
    if m:
        return m.group(1), False
    try:
        parsed = urlparse(text if '://' in text else f'https://{text}')
    except Exception:
        return None, False
    host = (parsed.hostname or '').lower()
    path = (parsed.path or '').rstrip('/').lower()
    if host in _EXPLORE_IG_PROFILE_HOSTS and path == _EXPLORE_IG_PROFILE_PATH:
        rid = (parse_qs(parsed.query or '').get('trepo_recipe') or [None])[0]
        return (_safe_text(rid) or None), True
    return None, False


_EXPLORE_SELECT_COLS = (
    "id, title, ingredients, instructions, notes, image_url, original_image_url, "
    "image_urls, source_url, resolved_url, resolved_url_hash"
)


def _lookup_explore_recipe(conn, explore_id=None, match_url=None):
    """Read one ready explore_recipes row by id (synthetic-URL path) or by
    source_url/resolved_url/hash equality (real-URL path). Returns the row dict or
    None. Read-only; returns None (never raises) on any miss/error."""
    try:
        with conn.cursor() as cur:
            if explore_id:
                cur.execute(
                    f"SELECT {_EXPLORE_SELECT_COLS} FROM explore_recipes "
                    "WHERE id=%s AND status='ready' LIMIT 1",
                    (explore_id,),
                )
                return cur.fetchone()
            candidate = _safe_text(match_url)
            if not candidate:
                return None
            # Match the raw candidate AND its canonical Instagram identity, so the same
            # reel arriving with a different ?igsh= share token still hits the curated
            # row. Existing conditions are kept verbatim so curated rows hashed the old
            # way keep matching.
            cur.execute(
                f"SELECT {_EXPLORE_SELECT_COLS} FROM explore_recipes "
                "WHERE status='ready' AND (source_url=%s OR resolved_url=%s "
                "OR resolved_url_hash=%s OR resolved_url_hash=%s) "
                "LIMIT 1",
                (candidate, candidate, _sha256(candidate), _resolved_url_hash(candidate)),
            )
            return cur.fetchone()
    except Exception:
        return None


def _explore_extraction_from_row(row, submitted_url):
    """Build a pre-structured extraction dict from an explore_recipes row. The
    'prestructured' key signals _save_saved_recipe_record to skip refine."""
    structured = {
        'title': _safe_text(row.get('title')),
        'ingredients': _clean_string_list(_parse_json_field(row.get('ingredients')) or []),
        'instructions': _clean_string_list(_parse_json_field(row.get('instructions')) or []),
        'notes': _clean_string_list(_parse_json_field(row.get('notes')) or []),
    }
    image_url = _safe_text(row.get('image_url')) or _safe_text(row.get('original_image_url'))
    explore_id = _safe_text(row.get('id'))
    # Stable per-recipe dedup identity, independent of how the save was matched:
    # prefer the row's own resolved/source URL, else the synthetic explore URL.
    dedup_url = (
        _safe_text(row.get('resolved_url'))
        or _safe_text(row.get('source_url'))
        or f'https://trepo.ai/explore/{explore_id}'
    )
    content = _format_recipe_text(structured)
    return {
        'url': _safe_text(submitted_url) or dedup_url,
        'resolved_url': dedup_url,
        'platform': 'explore',
        'content': content,
        'caption': content,
        'title': structured['title'],
        'image_url': image_url,
        'image_urls': [image_url] if image_url else [],
        'source': 'explore',
        'author_name': None,        # de-attribution stays — never carry creator fields
        'caption_field': 'explore',
        'warnings': [],
        'prestructured': structured,
        'explore_id': explore_id,
    }


def _explore_shortcircuit_extraction(conn, url, request_id=None):
    """If the submitted URL maps to a curated explore_recipes row, return a
    pre-structured extraction (skips scrape/Apify/refine). Synthetic schemes
    (trepo.ai/explore/{id} and instagram.com/trepohq?trepo_recipe={id}) are matched
    from the RAW url before canonicalization. A Trepo IG-profile URL that cannot be
    resolved to a recipe raises 422 rather than EVER falling through to a profile
    scrape (which returns junk). Real explore URLs match by source_url/resolved_url
    equality. Returns an extraction dict, or None to fall through to normal extract."""
    explore_id, is_trepo_profile = _explore_synthetic_id(url)
    if explore_id:
        row = _lookup_explore_recipe(conn, explore_id=explore_id)
        if row:
            extraction = _explore_extraction_from_row(row, url)
            _log_event(
                request_id,
                'saved_recipe_explore_shortcircuit',
                explore_id=extraction['explore_id'],
                matched='trepo_profile' if is_trepo_profile else 'synthetic_id',
                resolved_url=extraction['resolved_url'],
            )
            return extraction
    if is_trepo_profile:
        # A Trepo IG-profile URL with a missing/unknown recipe id. NEVER scrape a
        # profile page — surface a clean not-a-recipe 422 instead of falling through.
        _log_event(
            request_id,
            'saved_recipe_explore_profile_unmatched',
            explore_id=explore_id or '',
            url=_safe_text(url)[:200],
        )
        raise ServiceError("That link doesn't point to a saved recipe.", status_code=422)
    # Non-synthetic URL: match real explore rows by source_url/resolved_url equality.
    try:
        normalized = _normalize_url(url)
    except Exception:
        return None
    row = _lookup_explore_recipe(conn, match_url=normalized)
    if not row:
        return None
    extraction = _explore_extraction_from_row(row, url)
    _log_event(
        request_id,
        'saved_recipe_explore_shortcircuit',
        explore_id=extraction['explore_id'],
        matched='source_url',
        resolved_url=extraction['resolved_url'],
    )
    return extraction


def _save_saved_recipe_record(conn, owner, extraction, request_id=None, prepared_images=None):
    # Write the canonical-identity hash, but LOOK UP both it and the legacy raw-URL hash
    # so reels saved before canonicalization still dedupe instead of duplicating.
    # For non-Instagram URLs the canonical key is the URL itself, so both are identical
    # and nothing about existing behaviour changes.
    resolved_url_hash = _resolved_url_hash(extraction['resolved_url'])
    existing = _fetch_saved_recipe_by_hash(
        conn, owner, _resolved_url_hash_candidates(extraction['resolved_url']))
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

    prestructured = extraction.get('prestructured')
    if prestructured:
        # Explore short-circuit: recipe is already structured, skip LLM refine.
        structured_recipe = {
            'title': _safe_text(prestructured.get('title')),
            'ingredients': _clean_string_list(prestructured.get('ingredients') or []),
            'instructions': _clean_string_list(prestructured.get('instructions') or []),
            'notes': _clean_string_list(prestructured.get('notes') or []),
        }
        # Explore recipes arrive already structured from a real web page, so they are grounded
        # by construction and never went through the refine step that computes a source.
        recipe_response = {'recipe': _format_recipe_text(structured_recipe), 'model': 'explore',
                           'recipe_source_used': 'explore'}
    else:
        try:
            recipe_response, structured_recipe = _analyze_extraction(extraction, request_id=request_id)
        except recipe_work_budget.WorkLimit:
            recipe_response = {'recipe': 'Not enough recipe information.', 'model': None}
            structured_recipe = {}
    # Nothing usable in the source. Do NOT invent a recipe, and do NOT throw the save away
    # either: keep what is genuinely known (title, link, image), mark it honestly, and hand it
    # to the repair agent, which can spend minutes on audio transcription that a 29s request
    # cannot. The user keeps their save and gets the real recipe shortly after, instead of
    # either losing it or being handed a confident fabrication.
    ungrounded = recipe_response['recipe'] == 'Not enough recipe information.'
    terminal_reason = recipe_work_budget.terminal_reason() or extraction.get('terminal_work_reason')
    _repairable_platform = extraction.get('platform') not in {'text', 'image', 'explore'} and not terminal_reason

    # A save can also come back GROUNDED but incomplete: a video whose caption lists the
    # ingredients while the method is only ever spoken aloud. Those pass the grounding check,
    # save as 'ready' with zero instructions, and nothing ever revisits them — 49 of them
    # accumulated in a single day while only 4 saves reached the repair agent. A recipe with
    # ingredients and no steps is exactly what the repair agent is for: it can afford the
    # audio transcription the 29s sync path skips.
    _ins = (structured_recipe or {}).get('instructions') or []
    _ing = (structured_recipe or {}).get('ingredients') or []
    incomplete_video = (
        not ungrounded
        and extraction.get('platform') in {'tiktok', 'instagram'}
        and len(_ins) == 0
        and len(_ing) > 0
    )

    if ungrounded:
        _log_event(request_id, 'saved_recipe_queued_for_repair',
                   platform=extraction.get('platform'),
                   resolved_url=extraction.get('resolved_url'))
        structured_recipe = {
            'title': _safe_text(extraction.get('title')) or _safe_text(structured_recipe and structured_recipe.get('title')) or 'Recipe',
            'ingredients': [], 'instructions': [],
            'notes': [] if terminal_reason else ["We couldn't read the full recipe from this post yet — we're still working on it."],
        }
        recipe_response = {'recipe': _format_recipe_text(structured_recipe), 'model': None}

    recipe_id = str(uuid.uuid4())
    manual_platform = extraction['platform'] in {'text', 'image'}
    if manual_platform:
        source_image_urls = _upload_saved_recipe_source_images(
            owner,
            recipe_id,
            prepared_images or [],
            request_id=request_id,
        )
        # A text save can carry an attribution image: the photo already being shown for this
        # recipe in Explore. There are no uploaded bytes for it (nothing was scraped — the
        # caller handed us content it already had), so without this the save loses a perfectly
        # good real photo and falls back to an emoji. The feed already renders this same
        # remote URL, so referencing it here is consistent rather than novel.
        hinted = _safe_text(extraction.get('image_url'))
        if hinted.lower().startswith(('http://', 'https://')) and hinted not in (source_image_urls or []):
            source_image_urls = [hinted] + list(source_image_urls or [])
        image_fields = _build_source_backed_saved_recipe_image_fields(source_image_urls=source_image_urls)
    else:
        try:
            with recipe_work_budget.phase(max_seconds=6):
                image_fields = _prepare_saved_recipe_image_fields(owner, recipe_id, extraction, request_id=request_id)
        except recipe_work_budget.WorkLimit:
            image_fields = _build_source_backed_saved_recipe_image_fields(
                source_image_urls=_unique_texts(extraction.get('image_urls') or [extraction.get('image_url')]))

    # Image enrichment can consume the remaining provider budget after analysis.
    # Recheck before choosing status or scheduling repair; sync may still defer.
    terminal_reason = terminal_reason or recipe_work_budget.terminal_reason()
    if terminal_reason:
        _repairable_platform = False
        if ungrounded:
            structured_recipe['notes'] = []

    table = _saved_recipes_table(owner)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO `{table}` (
                        _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                        title, image_url, image_urls, source_image_url, source_image_urls, image_storage_key,
                        ingredients, instructions, notes, raw_caption, raw_content,
                        extraction_source, recipe_source_used, author_name, caption_field, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
                    # Ungrounded saves record NOTHING rather than a source, so "we refused to
                    # invent this" is distinguishable from "we never looked" (a pre-migration
                    # row, which is also NULL but for a different reason — those are resolved
                    # by extraction_source instead).
                    (None if ungrounded else
                     _safe_text(recipe_response.get('recipe_source_used')) or None),
                    extraction.get('author_name'),
                    extraction.get('caption_field') or None,
                    'ready' if recipe_content_complete(structured_recipe) else 'repairing' if _repairable_platform else 'failed',
                )
            )
        conn.commit()
        if _repairable_platform and not recipe_content_complete(structured_recipe):
            # Hand it to the background repair agent, which is not bound by the gateway's 29s
            # and can afford the audio transcription this save could not.
            if incomplete_video:
                _log_event(request_id, 'saved_recipe_queued_for_repair_incomplete',
                           platform=extraction.get('platform'),
                           ingredients=len(_ing),
                           resolved_url=extraction.get('resolved_url'))
            _enqueue_saved_recipe_repair_or_finish(conn, owner, recipe_id,
                                              extraction.get('resolved_url') or extraction.get('url'),
                                              request_id=request_id)
    except pymysql.err.IntegrityError as exc:
        # 1062 = another save of the same URL committed between our hash pre-check
        # and this INSERT (double-tap / client retry — the pre-check-to-insert gap
        # spans seconds of LLM + image work). The recipe exists, so this save
        # SUCCEEDED from the user's perspective: re-enter to take the dedupe path.
        if exc.args and exc.args[0] == 1062:
            conn.rollback()
            _log_event(
                request_id,
                'saved_recipe_insert_race_deduped',
                owner=owner,
                resolved_url=extraction['resolved_url'],
            )
            return _save_saved_recipe_record(
                conn, owner, extraction,
                request_id=request_id, prepared_images=prepared_images,
            )
        raise
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
    # Classify meal category (breakfast/lunch/dinner/snacks) for "My Recipes" grouping,
    # BEFORE the dual-write so the shared mirror carries it too.
    _apply_saved_recipe_meal_category(
        conn, owner, recipe_id,
        structured_recipe['title'] or extraction.get('title'),
        structured_recipe['ingredients'],
        request_id=request_id,
        # Prestructured/claim path carries the snapshot's meal_category → reuse it (skip LLM).
        # None for normal saves → classify as before.
        provided_category=extraction.get('meal_category'),
    )
    # Migration dual-write to shared_saved_recipes (flag-gated, non-blocking).
    _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=request_id)
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

    retained = [result for result in results if not recipe_content_complete(result.get('recipe'))]
    results = [result for result in results if recipe_content_complete(result.get('recipe'))]
    partial_errors.extend(recipe_incomplete_error(result['recipe']) for result in retained)
    if not results:
        retained_recipe = retained[0]['recipe']
        return _success({'owner':owner, 'recipe':retained_recipe,
                         'deduped':bool(retained[0].get('deduped')),
                         'retained_recipes':[result['recipe'] for result in retained],
                         'code':'recipe_content_incomplete', 'error':INCOMPLETE_MESSAGE,
                         'recovery':recipe_content_recovery(retained_recipe)}, status=422)
    has_created = any(not result.get('deduped') for result in results)
    status = 201 if has_created else 200
    if len(results) == 1:
        result = results[0]
        body = {
            'owner': owner,
            'recipe': result['recipe'],
            'deduped': bool(result.get('deduped')),
        }
        if retained:
            body['retained_recipes'] = [entry['recipe'] for entry in retained]
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
    if retained:
        body['retained_recipes'] = [entry['recipe'] for entry in retained]
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
    serialized_job = _serialize_saved_recipe_batch_job(job, event=event)
    recipes = list(recipes or [])
    results = list(results or [])
    partial_errors = list(partial_errors or [])
    retained = [recipe for recipe in recipes if not recipe_content_complete(recipe)]
    ready = [recipe for recipe in recipes if recipe_content_complete(recipe)]
    if serialized_job.get('status') == 'completed' and retained:
        pending = any(recipe_content_outcome(recipe) == 'processing' for recipe in retained)
        serialized_job['status'] = 'running' if pending else 'completed' if ready else 'failed'
        if not pending:
            partial_errors.extend(recipe_incomplete_error(recipe) for recipe in retained)
            if not ready:
                serialized_job['error'] = INCOMPLETE_MESSAGE
                serialized_job['code'] = 'recipe_content_incomplete'
        serialized_job['result_count'] = len(ready)
    if serialized_job.get('status') == 'completed' and not recipes:
        serialized_job.update(status='failed',error='No usable recipe result is available.',code='recipe_result_unavailable',result_count=0)
    ready_ids = {recipe.get('id') for recipe in ready}
    verified_results = [entry for entry in results if (entry.get('recipe') or {}).get('id') in ready_ids]
    body = {'owner':owner, 'job':serialized_job, 'recipes':ready,
            'results':verified_results, 'count':len(verified_results)}
    if retained:
        body['retained_recipes'] = retained
        body['content_outcome'] = 'processing' if any(recipe_content_outcome(r) == 'processing' for r in retained) else 'partial' if ready else 'link_retained'
    if partial_errors:
        body['partial_errors'] = partial_errors
    return _success(body)


def _process_saved_recipe_image_batch(conn, owner, image_submissions, request_id=None):
    clusters, partial_errors = _extract_recipe_clusters_from_images(image_submissions, request_id=request_id)
    results = []

    def _save_cluster(cluster):
        result, _ = _run_with_transient_retry(
            lambda: _save_saved_recipe_record(conn, owner, cluster['extraction'],
                request_id=request_id, prepared_images=cluster.get('prepared_images') or []),
            request_id=request_id, label='persist_recipe_image_cluster')
        if not _safe_text(((result or {}).get('recipe') or {}).get('id')):
            raise ServiceError('Recipe persistence was not confirmed. Please try again.', status_code=503)
        return {
            'recipe_id': ((result or {}).get('recipe') or {}).get('id'),
            'deduped': bool((result or {}).get('deduped')),
            'image_indexes': list(cluster.get('image_indexes') or []),
        }

    for cluster in clusters:
        try:
            results.append(_save_cluster(cluster))
            continue
        except Exception as exc:
            fragments = list(cluster.get('fragments') or [])
            # MERGED-FIRST, PER-IMAGE ON FAILURE. The merged attempt above is what a
            # one-recipe-across-pages batch needs (51% of real multi-image batches), so it
            # always runs first. Only once it has actually failed do we treat the grouping
            # as wrong and retry the images individually — a fallback on evidence rather
            # than a guess from image count. Losing the whole batch to one combined 422 is
            # the failure this exists to prevent.
            if len(fragments) < 2 or not isinstance(exc, ServiceError) or exc.status_code != 422:
                partial_errors.append({
                    'image_indexes': cluster.get('image_indexes') or [],
                    'error': str(exc) if isinstance(exc, ServiceError) else 'This recipe could not be saved. Please try again.',
                    'status_code': exc.status_code if isinstance(exc, ServiceError) else 503,
                    'retryable': _is_transient_failure(exc),
                })
                continue
            _log_event(
                request_id,
                'saved_recipe_image_merged_failed_per_image_fallback',
                image_count=len(fragments),
                image_indexes=cluster.get('image_indexes') or [],
                failure_reason=str(exc)[:200],
            )
            recovered_any = False
            for fragment in fragments:
                single = _merge_image_cluster({
                    'title_hint': fragment.get('title_hint') or '',
                    'fragments': [fragment],
                })
                try:
                    results.append(_save_cluster(single))
                    recovered_any = True
                except Exception as inner:
                    partial_errors.append({
                        'image_indexes': single.get('image_indexes') or [],
                        'error': str(inner) if isinstance(inner, ServiceError) else 'This image could not be saved. Please try again.',
                        'status_code': inner.status_code if isinstance(inner, ServiceError) else 503,
                        'retryable': _is_transient_failure(inner),
                    })
            _log_event(
                request_id,
                'saved_recipe_image_per_image_fallback_complete',
                image_count=len(fragments),
                recovered=recovered_any,
            )
    if not results:
        first_error = partial_errors[0] if partial_errors else {'error': 'Not enough recipe information.', 'status_code': 422}
        raise ServiceError(
            first_error.get('error') or 'Not enough recipe information.',
            status_code=first_error.get('status_code') or 422,
            extra={'partial_errors': partial_errors},
        )
    return results, partial_errors


def _enqueue_saved_recipe_batch_job(owner, submission, request_id=None, event=None):
    if recipe_batch_checkpoint.enabled(owner):
        return recipe_batch_checkpoint.enqueue(SimpleNamespace(**globals()), owner, submission, event, request_id)
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
        # Recorded at enqueue so a failure sweep knows how to replay the input without
        # having to fetch and sniff the payload first.
        'submission_kind': _safe_text(submission.get('kind')),
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


def _enqueue_saved_recipe_url_job(owner, url, request_id=None, event=None):
    """Slow web/social URL saves (yt-dlp/Apify extraction chain) can exceed API
    Gateway's hard 30s integration cap, killing the connection with a 503 even though
    the Lambda completes. Run the extraction+save in the background worker and hand the
    client a job_id to poll — mirrors the image-batch async path and reuses the SAME
    job document, GET /jobs/{job_id} endpoint, and DynamoDB jobs table."""
    job_id = uuid.uuid4().hex
    now = _utc_now_iso()
    payload_key = _put_saved_recipe_batch_payload(owner, job_id, {'kind': 'url', 'url': url})
    job = {
        'job_id': job_id,
        'type': _SAVED_RECIPE_BATCH_JOB_TYPE,
        'owner': owner,
        'status': _JOB_STATUS_PENDING,
        'image_count': 0,
        'result_count': 0,
        'recipe_ids': [],
        'results': [],
        'partial_errors': [],
        'request_id': request_id,
        'created_at': now,
        'updated_at': now,
        'payload_s3_key': payload_key,
        'source_url': url,
        'submission_kind': 'url',
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
                'async_task': _ASYNC_TASK_PROCESS_SAVED_RECIPE_URL,
                'owner': owner,
                'job_id': job_id,
                'request_id': request_id,
            }).encode('utf-8'),
        )
        _log_event(
            request_id,
            'saved_recipe_url_job_enqueued',
            owner=owner,
            job_id=job_id,
        )
        return job
    except Exception as exc:
        _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_FAILED,
            error=str(exc),
            completed_at=_utc_now_iso(),
        )
        _log_event(
            request_id,
            'saved_recipe_url_job_enqueue_failed',
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


def _client_supports_async_url_save(body):
    """Whether the calling client understands a 202 + {job:{job_id}} for a URL save and
    will poll GET /jobs/{job_id}. Shipped iOS routes image/text saves through a polling
    helper but its URL-save path (saveRecipe(url:)) accepts ONLY 200/201 — so async MUST
    be opt-in, or every un-updated client would break on 202. An updated client signals
    support via body flag or the X-Trepo-Async-Save header; absent the signal we keep the
    synchronous path (current behavior). Flip this to always-True once the async-capable
    iOS build is the floor."""
    payload = body or {}
    for key in ('async_save', 'async', 'supports_async'):
        val = payload.get(key)
        if val is True:
            return True
        if isinstance(val, str) and val.strip().lower() in ('1', 'true', 'yes', 'url'):
            return True
    event = payload.get('_event') or {}
    headers = event.get('headers') or {}
    header_val = ''
    for k, v in headers.items():
        if _safe_text(k).lower() == 'x-trepo-async-save':
            header_val = _safe_text(v)
            break
    return header_val.lower() in ('1', 'true', 'yes', 'url')


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
            # Curated explore recipes short-circuit scrape/Apify/refine entirely — always
            # fast, so they stay synchronous (201/200) regardless of client capability.
            extraction = _explore_shortcircuit_extraction(conn, submission['url'], request_id=request_id)
            if extraction is None and _client_supports_async_url_save(body):
                # Slow path: real web/social extraction (yt-dlp/Apify) can blow past API
                # Gateway's 30s cap. Hand an async-capable client a job to poll and return
                # 202 immediately. Idempotency is preserved downstream: the worker's
                # _save_saved_recipe_record dedupes on resolved_url_hash (unique index +
                # 1062 handling), so a client retry can't create a duplicate save.
                job = _enqueue_saved_recipe_url_job(
                    owner, submission['url'], request_id=request_id, event=body.get('_event'))
                _log_event(
                    request_id,
                    'saved_recipe_post_complete',
                    owner=owner,
                    submission_kind='url',
                    result_count=0,
                    latency_ms=int((time.time() - started) * 1000),
                    async_job=job.get('job_id'),
                )
                return _build_saved_recipe_batch_accept_response(owner, job, event=body.get('_event'))
            source_deferred = False
            if extraction is None:
                try:
                    extraction = _extract_content(submission['url'], request_id=request_id)
                except recipe_work_budget.WorkLimit:
                    # Legacy URL clients accept a recipe receipt, not an async job.
                    # Retain only an input that passes the existing pure validators;
                    # the sync budget may then defer one longer repair attempt.
                    normalized_url = _normalize_url(submission['url'])
                    platform = _detect_platform(normalized_url, raw_url=submission['url'], request_id=request_id)
                    extraction = {'url': normalized_url, 'resolved_url': normalized_url, 'platform': platform,
                                  'title': 'Recipe', 'caption': '', 'content': '', 'source': 'deadline_retained'}
                    source_deferred = True
            result, _ = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id)
            if source_deferred and not recipe_content_complete(result.get('recipe')):
                # 200/201 acknowledges the retained link, never ready recipe content.
                retained = result['recipe']
                response = _success({'owner': owner, 'recipe': retained,
                    'deduped': bool(result.get('deduped')), 'code': 'recipe_content_incomplete',
                    'content_outcome': recipe_content_outcome(retained),
                    'recovery': recipe_content_recovery(retained)},
                    status=200 if result.get('deduped') else 201)
            else:
                response = _build_saved_recipe_post_response(owner, [result])
            response_result_count = 1
        elif submission['kind'] == 'content':
            extraction = _build_text_recipe_extraction(
                submission['content'],
                source_url=submission.get('source_url'),
                image_url=submission.get('image_url'),
            )
            resolved_url_hash = _sha256(extraction['resolved_url'])
            existing = _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash)
            if existing:
                # Duplicate — return synchronously
                existing = _ensure_owned_saved_recipe_image(conn, owner, existing, request_id=request_id)
                # A prior accepted text save may predate projection publication.
                _dual_write_saved_recipe_to_shared(conn, owner, existing['_id'], request_id=request_id, transactional=True)
                conn.commit()
                serialized_existing = _serialize_row(existing)
                response = _build_saved_recipe_post_response(owner, [{'recipe':serialized_existing,'deduped':True}])
                response_result_count = 1
            elif not _text_processing_schema_ready(conn, owner):
                # An older table may have an active metadata lock. Preserve the shipped
                # synchronous contract instead of blocking or storing an invalid status.
                result, _ = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id)
                response = _build_saved_recipe_post_response(owner, [result])
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
                dup_row = None
                try:
                    conn.begin()
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""INSERT INTO `{table}` (
                                    _id, _owner, source_type, source_url, resolved_url, resolved_url_hash,
                                    title, ingredients, instructions, notes, raw_caption, raw_content,
                                    extraction_source, image_url, image_urls, source_image_url,
                                    source_image_urls, status
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'processing')""",
                            (
                                recipe_id, owner, extraction['platform'],
                                extraction['url'], extraction['resolved_url'], resolved_url_hash,
                                placeholder_title,
                                json.dumps(raw_ingredients), json.dumps([]), json.dumps([]),
                                extraction['caption'], extraction['content'], extraction['source'],
                                # Keep the caller's photo. This row is written directly rather
                                # than through _save_saved_recipe_record, so the image columns
                                # have to be carried here too — omitting them is what dropped
                                # the real Explore photo and left the recipe showing an emoji.
                                _text_hero_image(extraction),
                                json.dumps([_text_hero_image(extraction)] if _text_hero_image(extraction) else []),
                                _text_hero_image(extraction),
                                json.dumps([_text_hero_image(extraction)] if _text_hero_image(extraction) else []),
                            ),
                        )
                    # The list projection must exist before returning a saved recipe.
                    # Refinement is optional background work, never a visibility gate.
                    _dual_write_saved_recipe_to_shared(conn, owner, recipe_id, request_id=request_id, transactional=True)
                    conn.commit()
                except pymysql.err.IntegrityError as exc:
                    # 1062 = concurrent save of identical text won the race — the
                    # recipe exists, so return it as a dedupe instead of a 500.
                    if not (exc.args and exc.args[0] == 1062):
                        raise
                    conn.rollback()
                    dup_row = _fetch_saved_recipe_by_hash(conn, owner, resolved_url_hash)
                    if not dup_row:
                        raise
                except Exception:
                    conn.rollback()
                    raise
                if dup_row:
                    _dual_write_saved_recipe_to_shared(conn, owner, dup_row['_id'], request_id=request_id, transactional=True)
                    conn.commit()
                    dup_row = _ensure_owned_saved_recipe_image(conn, owner, dup_row, request_id=request_id)
                    response = _build_saved_recipe_post_response(owner, [{'recipe':_serialize_row(dup_row),'deduped':True}])
                    response_result_count = 1
                else:
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


def _delete_saved_recipe_atomic(owner, item_id):
    from saved_recipe_deletions import (
        PreparationRequired, RecipeIdentityReused, delete_with_receipt,
    )

    conn = _mysql_conn()
    members = [owner]
    if SAVED_RECIPE_HOUSEHOLD_FANOUT:
        members = sorted(set(members + list(_get_household_member_ids(conn, owner) or [])))
    options = {
        'serialize': json_serial,
        # List reads may use shared storage; mutation ownership remains personal.
        'read_shared': False,
        'delete_shared': DUAL_WRITE_SAVED_RECIPES or os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true',
        'cache_sources': ['saved', _availability_cache_source('saved')],
    }
    for attempt in range(4):
        try:
            result = delete_with_receipt(conn, owner, item_id, members, **options)
            if result is None:
                return _error(404, 'Saved recipe not found')
            return _success({'owner': owner, 'recipe': result})
        except PreparationRequired as pending:
            if attempt == 3:
                return _error(503, 'Recipe changed during deletion. Please retry.')
            # Availability calculation can perform external work or schema setup.
            # The kernel has released all transaction locks before asking for it.
            _ensure_recipe_personalization_tables(conn)
            context = _build_kitchen_match_context(conn, owner)
            prepared, _ = _serialize_saved_recipe_with_availability(
                conn, owner, pending.row, kitchen_context=context,
            )
            options.update(prepared=prepared, fingerprint=pending.fingerprint)
        except RecipeIdentityReused:
            return _error(409, 'Saved recipe identifier was reused after deletion')



def _delete_saved_recipe(owner, item_id):
    if os.getenv('SAVED_RECIPE_ATOMIC_DELETE_V1', '').strip().lower() == 'true':
        return _delete_saved_recipe_atomic(owner, item_id)
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
        # Migration dual-write: mirror the delete into shared_saved_recipes so a deleted
        # recipe can't reappear once reads flip to shared (per-owner stays source of truth).
        # Non-blocking — a missed shared delete is caught by the reconcile/parity sweep.
        if DUAL_WRITE_SAVED_RECIPES:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM `{_SHARED_SAVED_RECIPES_TABLE}` WHERE owner_id = %s AND _id = %s LIMIT 1",
                        [owner, item_id],
                    )
                conn.commit()
            except Exception as exc:
                print(json.dumps({'evt': 'dual_write_miss', 'family': 'saved_recipes_delete',
                    'owner_id': str(owner), 'recipe_id': str(item_id), 'error': str(exc)[:500]}), file=sys.stderr)
        # Fan out delete to household members
        try:
            _fan_out_delete_to_household(conn, owner, item_id)
        except Exception:
            pass  # Non-critical — other members can delete independently
        return _success({'owner': owner, 'recipe': serialized_row})
    finally:
        pass


def _extract_recovery_source(url, request_id=None):
    from recovery_web import fetch_page
    html, resolved = fetch_page(url)
    soup = BeautifulSoup(html, 'html.parser')
    node = _select_richest_recipe_node(soup)
    if node:
        ingredients = node.get('recipeIngredient')
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        elif not isinstance(ingredients, list):
            ingredients = []
        structured = {'title':_safe_text(node.get('name')),
                      'ingredients':_clean_string_list([value for value in ingredients if isinstance(value, str)]),
                      'instructions':_parse_instructions(node.get('recipeInstructions'))}
    else:
        # No inferred link crawling, audio scrape, images or embedded subrequests.
        _, structured = _recipe_response_from_content(_extract_web_article_text(soup), request_id=request_id)
    return structured or {}, resolved


def _recover_saved_recipe(owner, item_id, body, request_id=None):
    recovery = body.get('recovery')
    if not isinstance(recovery, dict) or recovery.get('action') not in ('replace_source','paste_recipe'):
        return _error(400, 'Choose a recipe recovery action.')
    revision = body.get('content_revision')
    if not isinstance(revision, str) or not revision:
        return _error(400, 'Refresh the recipe before changing its source.')
    conn = _mysql_conn()
    row = _fetch_saved_recipe_by_id(conn, owner, item_id)
    if not row: return _error(404, 'Saved recipe not found')
    if _saved_recipe_content_revision(row) != revision:
        return _error(409, 'This recipe changed. Refresh it before saving.')
    replace = recovery.get('replace_fields') or []
    if not isinstance(replace,list) or set(replace)-{'title','ingredients','instructions','notes'}:
        return _error(400, 'Choose which recipe fields to replace.')
    # A source is one coherent recipe. Never attach new directions to ingredients
    # retained from another version, even when only one old field is populated.
    current_content = _serialize_row(row)
    if (current_content.get('ingredients') or current_content.get('instructions')) and not {'ingredients','instructions'}.issubset(replace):
        return _error(409, 'Replace the ingredients and steps together to recover this recipe.',
                      {'code':'recipe_replacement_confirmation_required','replace_fields':['ingredients','instructions']})
    source_updates = {}
    if recovery['action'] == 'replace_source':
        url = _safe_text(recovery.get('url'))
        if not _valid_recipe_repair_url(url): return _error(400,'Use a public recipe page URL.')
        try:
            from recovery_web import recovery_budget
            with recovery_budget():
                structured, resolved = _extract_recovery_source(url, request_id=request_id)
        except recipe_work_budget.WorkLimit:
            return _error(422, 'This is taking too long. Your saved recipe has not changed. Try again or paste its recipe text.')
        except ValueError as exc:
            return _error(422, str(exc))
        source_updates = {'source_url':url, 'resolved_url':resolved, 'resolved_url_hash':_resolved_url_hash(resolved)}
        other = _fetch_saved_recipe_by_hash(conn, owner, source_updates['resolved_url_hash'])
        if other and other.get('_id') != item_id:
            return _error(409, 'This source is already saved.', {'existing_recipe_id':other['_id']})
    else:
        text = _safe_text(recovery.get('text'))
        if not text or len(text)>30000: return _error(400, 'Paste up to 30,000 characters of recipe text.')
        try:
            from recovery_web import recovery_budget
            with recovery_budget():
                _, structured = _recipe_response_from_content(text, request_id=request_id)
        except recipe_work_budget.WorkLimit:
            return _error(422, 'This is taking too long. Your saved recipe has not changed. Try again or paste its recipe text.')
        except ValueError as exc:
            return _error(422, str(exc))
        structured = structured or {}
    if not recipe_content_complete(structured):
        return _error(422, INCOMPLETE_MESSAGE, {'code':'recipe_content_incomplete','recipe':_serialize_row(row),'recovery':recipe_content_recovery(_serialize_row(row))})
    updates = {'content_revision':revision}
    for field in ('title','ingredients','instructions','notes'):
        if field == 'notes' and field not in replace:
            notes = current_content.get('notes') or []
            pending_note = "We couldn't read the full recipe from this post yet — we're still working on it."
            if pending_note in notes:
                # Bypass input normalization: these are already-stored user notes.
                source_updates['notes'] = json.dumps([note for note in notes if note != pending_note])
            continue
        current = _safe_text(row.get(field)) if field=='title' else current_content.get(field)
        if field in replace or not current:
            value = structured.get(field)
            if value is not None: updates[field]=value
    # Revision and uniqueness are revalidated inside the same edit transaction.
    return _update_saved_recipe(owner,item_id,updates,request_id=request_id,_source_updates=source_updates)


def _update_saved_recipe(owner, item_id, body, request_id=None, _source_updates=None):
    """Edit a saved recipe's user-editable fields (title, ingredients, steps/instructions,
    notes). Partial updates allowed. Applies to the owner + any household members that have
    the recipe, then recomputes availability since ingredient changes affect kitchen matching."""
    if 'recovery' in body:
        return _recover_saved_recipe(owner,item_id,body,request_id=request_id)
    updates = dict(_source_updates or {})
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
        targets = set(_get_household_member_ids(conn, owner) or [])
    targets.add(_safe_owner_token(owner))
    targets = sorted(targets)
    # Schema preparation may commit implicitly. Finish it before the edit transaction.
    for tid in targets:
        _ensure_saved_recipes_table(conn, tid)
    conn.begin()
    try:
        with conn.cursor() as cur:
            # Lock all existing copies in the same order for concurrent household
            # edits. Recheck the acting copy before changing any required target.
            acting_exists = False
            locked_rows = {}
            for tid in targets:
                target_table = _saved_recipes_table(tid)
                cur.execute(f"SELECT * FROM `{target_table}` WHERE _id = %s LIMIT 1 FOR UPDATE", [item_id])
                locked = cur.fetchone()
                if locked is not None: locked_rows[tid] = locked
                if target_table == table:
                    acting_exists = locked is not None
            if not acting_exists:
                conn.rollback()
                return _error(404, 'Saved recipe not found')
            expected_revision = body.get('content_revision')
            if expected_revision is not None and expected_revision != _saved_recipe_content_revision(locked_rows[_safe_owner_token(owner)]):
                conn.rollback()
                return _error(409, 'This recipe changed. Refresh it before saving.')
            if 'resolved_url_hash' in (_source_updates or {}):
                for tid in locked_rows:
                    cur.execute(f"SELECT _id FROM `{_saved_recipes_table(tid)}` WHERE resolved_url_hash=%s AND _id<>%s FOR UPDATE",[_source_updates['resolved_url_hash'],item_id])
                    if cur.fetchone():
                        conn.rollback()
                        return _error(409, 'This source is already saved.')
            for tid, locked in locked_rows.items():
                target_updates = dict(updates)
                # Old clients omit content_revision; retain their partial-edit
                # contract, but compute readiness from the entire locked row.
                if 'status' in locked:
                    target_updates['status'] = 'ready' if recipe_content_complete({**locked, **updates}) else 'failed'
                set_clause = ', '.join(f"`{col}` = %s" for col in target_updates) + ", `_updatedDate` = NOW()"
                vals = list(target_updates.values())
                mtable = _saved_recipes_table(tid)
                cur.execute(f"UPDATE `{mtable}` SET {set_clause} WHERE _id = %s LIMIT 1", vals + [item_id])
        # Required projections join the same transaction. A shared-read cutover
        # must never acknowledge an edit while leaving its displayed copy stale.
        require_shared = os.getenv('READ_SHARED_SAVED_RECIPES', '').strip().lower() == 'true'
        for tid in targets:
            _dual_write_saved_recipe_to_shared(conn, tid, item_id, request_id=request_id,
                                              transactional=True, required=require_shared)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

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


# Bounded auto-retry for TRANSIENT failures only. A provider 5xx, a timeout or a
# throttle is worth retrying — the same input may well succeed seconds later. A content
# 4xx ("not a recipe", "not a recipe link") is NOT: the input has not changed, so a retry
# just re-fails and burns another LLM/provider call. Those wait for a fix plus a sweep,
# which is what the re-drive runbook is for.
_TRANSIENT_RETRY_ATTEMPTS = min(2, max(0, int(os.getenv('TRANSIENT_RETRY_ATTEMPTS', '2'))))
_TRANSIENT_RETRY_BASE_SECONDS = float(os.getenv('TRANSIENT_RETRY_BASE_SECONDS', '2'))
_TRANSIENT_ERROR_MARKERS = (
    'timed out', 'timeout', 'throttl', 'rate limit', 'too many requests',
    'connection reset', 'connection aborted', 'temporarily unavailable',
    'service unavailable', 'bad gateway', 'internal server error',
)


def _is_transient_failure(exc):
    """True when the failure looks like infrastructure, not content."""
    if isinstance(exc, recipe_work_budget.WorkLimit):
        return False
    if isinstance(exc, ServiceError):
        # Upstream refusing us (bot wall) is not transient — a retry hits the same wall.
        if _is_upstream_block(exc):
            return False
        if exc.status_code in (408, 429):
            return True
        if exc.status_code and exc.status_code < 500:
            return False
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def _run_with_transient_retry(operation, request_id=None, label=''):
    """Run `operation`, retrying only transient failures with linear backoff."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= _TRANSIENT_RETRY_ATTEMPTS or not _is_transient_failure(exc):
                raise
            attempt += 1
            delay = _TRANSIENT_RETRY_BASE_SECONDS * attempt
            _log_event(request_id, 'transient_retry', label=label, attempt=attempt,
                       delay_seconds=delay, failure_reason=str(exc)[:200])
            time.sleep(delay)


def _handle_async_saved_recipe_url_task(event, request_id):
    """Background worker for slow URL saves. Runs the same extraction+save the sync path
    ran (so image gen, availability, household fan-out all still happen via
    _save_saved_recipe_record) and records the result on the job so GET /jobs/{job_id}
    returns the recipe. Reuses the batch job document + status endpoint."""
    owner = _safe_text((event or {}).get('owner'))
    job_id = _safe_text((event or {}).get('job_id'))
    if not owner or not job_id:
        raise ServiceError('Async url task requires owner and job_id.', status_code=400)
    job = _get_saved_recipe_batch_job(job_id)
    if not job or _safe_text(job.get('owner')) != owner:
        raise ServiceError('Saved recipe url job not found.', status_code=404)
    if job.get('status') == _JOB_STATUS_COMPLETED:
        return {'ok': True, 'owner': owner, 'job_id': job_id, 'status': _JOB_STATUS_COMPLETED,
                'count': int(job.get('result_count') or 0), 'already_completed': True}
    _update_saved_recipe_batch_job(job_id, status=_JOB_STATUS_RUNNING, started_at=_utc_now_iso(), error=None)
    started = time.time()
    conn = _mysql_conn()
    try:
        _ensure_saved_recipes_table(conn, owner)
        submission = _load_saved_recipe_batch_payload(job)
        url = _safe_text((submission or {}).get('url'))
        if not url:
            raise ServiceError('URL job payload is missing url.', status_code=400)
        extraction = _explore_shortcircuit_extraction(conn, url, request_id=request_id)
        if extraction is None:
            # Transient-only retry: a provider 5xx/timeout/throttle gets 2 more goes with
            # backoff; a content 422 raises straight through and waits for a fix + sweep.
            try:
                extraction = _run_with_transient_retry(
                    lambda: _extract_content(url, request_id=request_id),
                    request_id=request_id, label='extract_content')
            except recipe_work_budget.WorkLimit as exc:
                # Keep the accepted link even when its provider never responds.
                # No network here: budget expiry must not bypass input validation.
                normalized_url = _normalize_url(url)
                platform = _detect_platform(normalized_url, raw_url=url, request_id=request_id)
                extraction = {'url': normalized_url, 'resolved_url': normalized_url, 'platform': platform,
                              'title': 'Recipe', 'caption': '', 'content': '',
                              'source': 'deadline_retained', 'terminal_work_reason': exc.reason}
        result, _ = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id)
        recipe_id = ((result or {}).get('recipe') or {}).get('id')
        if not _safe_text(recipe_id):
            raise ServiceError('Recipe persistence was not confirmed. Please try again.', status_code=503)
        entry = {'recipe_id': recipe_id, 'deduped': bool((result or {}).get('deduped'))}
        completed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_COMPLETED,
            result_count=1 if recipe_id else 0,
            recipe_ids=[recipe_id] if recipe_id else [],
            results=[entry] if recipe_id else [],
            partial_errors=[],
            completed_at=_utc_now_iso(),
            error=None,
        )
        _log_event(
            request_id,
            'saved_recipe_url_job_complete',
            owner=owner,
            job_id=job_id,
            deduped=entry['deduped'],
            latency_ms=int((time.time() - started) * 1000),
        )
        return {
            'ok': True,
            'owner': owner,
            'job_id': job_id,
            'status': completed_job.get('status'),
            'count': 1 if recipe_id else 0,
        }
    except ServiceError as exc:
        failed_job = _update_saved_recipe_batch_job(
            job_id,
            status=_JOB_STATUS_FAILED,
            result_count=0,
            recipe_ids=[],
            results=[],
            partial_errors=[{'error': str(exc), 'status_code': exc.status_code}],
            completed_at=_utc_now_iso(),
            error=str(exc),
        )
        _log_event(
            request_id,
            'saved_recipe_url_job_failed',
            owner=owner,
            job_id=job_id,
            failure_reason=str(exc),
            status_code=exc.status_code,
        )
        # 4xx (e.g. 422 "not a recipe") is an expected user outcome the client surfaces,
        # not a backend fault — don't page. Only 5xx pages, matching the sync handler.
        if exc.status_code >= 500:
            if _is_upstream_block(exc):
                # Every fetch tier (including the residential proxy) was refused by the
                # site. The job still fails and the user still gets the graceful
                # copy-the-text fallback, but there is no backend defect to action — so
                # record a low-severity marker instead of paging the errors feed. Genuine
                # failures (extraction crashes, our own 5xx) fall through and page as before.
                _log_event(request_id, 'blocked_host', owner=owner, job_id=job_id,
                           host=(exc.extra or {}).get('blocked_host'),
                           severity='low', failure_reason=str(exc))
            else:
                _report_backend_error('save_recipe_url', owner_id=owner, code='url_job_failed',
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
            'saved_recipe_url_job_failed',
            owner=owner,
            job_id=job_id,
            failure_reason=str(exc),
        )
        _report_backend_error('save_recipe_url', owner_id=owner, code='url_job_error',
                              error=exc, job_id=job_id)
        import traceback
        traceback.print_exc()
        return {
            'ok': False,
            'owner': owner,
            'job_id': job_id,
            'status': failed_job.get('status'),
            'error': str(exc),
        }


def _handle_async_saved_recipe_batch_task(event, request_id, context=None):
    owner = _safe_text((event or {}).get('owner'))
    job_id = _safe_text((event or {}).get('job_id'))
    if not owner or not job_id:
        raise ServiceError('Async batch task requires owner and job_id.', status_code=400)
    job = _get_saved_recipe_batch_job(job_id)
    if not job or _safe_text(job.get('owner')) != owner:
        raise ServiceError('Saved recipe batch job not found.', status_code=404)
    if job.get('status') == _JOB_STATUS_COMPLETED:
        return {'ok': True, 'owner': owner, 'job_id': job_id, 'status': _JOB_STATUS_COMPLETED,
                'count': int(job.get('result_count') or 0), 'already_completed': True}
    if job.get('checkpoint_version') == recipe_batch_checkpoint.VERSION:
        return recipe_batch_checkpoint.run(SimpleNamespace(**globals()), event, context)
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
        if partial_errors:
            # Some photos in a multi-image recipe save failed while others succeeded — the
            # user silently gets fewer recipes than they added. Distinct marker (pattern alarm),
            # so a spike surfaces without paging on every mixed batch.
            try:
                print(json.dumps({'evt': 'recipe_batch_partial_failure', 'service': 'recipes',
                                  'owner_id': owner, 'job_id': job_id,
                                  'failed': len(partial_errors), 'saved': len(results)}))
            except Exception:
                pass
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


# =====================================================================
# Custom recipe categories — user-defined tags for saved recipes.
# Shared-first: two owner_id-keyed shared tables, no per-owner sprawl.
# A saved recipe's _id is identical in the per-owner and shared tables,
# so these mappings work regardless of READ_SHARED_SAVED_RECIPES. The
# owner_id here matches shared_saved_recipes.owner_id (the acting user).
# Many-to-many + orthogonal to meal_category: a recipe can carry any
# number of custom tags on top of its single breakfast/lunch/... slot.
# =====================================================================
_SHARED_RECIPE_CATEGORIES_TABLE = 'shared_recipe_categories'
_SHARED_RECIPE_CATEGORY_MAP_TABLE = 'shared_recipe_category_map'
_MAX_CATEGORIES_PER_OWNER = 100
_MAX_CATEGORY_NAME_LEN = 64


def _ensure_recipe_category_tables(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_SHARED_RECIPE_CATEGORIES_TABLE}` (
                owner_id VARCHAR(36) NOT NULL,
                category_id VARCHAR(36) NOT NULL,
                name VARCHAR(64) NOT NULL,
                color VARCHAR(16) NULL,
                sort_order INT NOT NULL DEFAULT 0,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (owner_id, category_id),
                UNIQUE KEY uniq_owner_name (owner_id, name),
                KEY idx_owner_sort (owner_id, sort_order)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_SHARED_RECIPE_CATEGORY_MAP_TABLE}` (
                owner_id VARCHAR(36) NOT NULL,
                recipe_id VARCHAR(36) NOT NULL,
                category_id VARCHAR(36) NOT NULL,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (owner_id, recipe_id, category_id),
                KEY idx_owner_category (owner_id, category_id),
                KEY idx_owner_recipe (owner_id, recipe_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def _category_row(r):
    return {
        'id': r.get('category_id'),
        'name': r.get('name'),
        'color': r.get('color'),
        'sort_order': int(r.get('sort_order') or 0),
    }


def _list_recipe_categories(owner, request_id=None):
    """Return the owner's custom categories + a recipe_id -> [category_id] map."""
    safe = _safe_owner_token(owner)
    conn = _mysql_conn()
    _ensure_recipe_category_tables(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT category_id, name, color, sort_order FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` "
            f"WHERE owner_id = %s ORDER BY sort_order ASC, name ASC",
            [safe],
        )
        categories = [_category_row(r) for r in (cur.fetchall() or [])]
        cur.execute(
            f"SELECT recipe_id, category_id FROM `{_SHARED_RECIPE_CATEGORY_MAP_TABLE}` WHERE owner_id = %s",
            [safe],
        )
        assignments = {}
        for r in (cur.fetchall() or []):
            assignments.setdefault(r.get('recipe_id'), []).append(r.get('category_id'))
    return _success({'categories': categories, 'assignments': assignments})


def _create_recipe_category(owner, body, request_id=None):
    name = _safe_text((body or {}).get('name'))
    if not name:
        return _error(400, 'Category name is required')
    name = name[:_MAX_CATEGORY_NAME_LEN]
    color = _safe_text((body or {}).get('color')) or None
    safe = _safe_owner_token(owner)
    conn = _mysql_conn()
    _ensure_recipe_category_tables(conn)
    with conn.cursor() as cur:
        # Dedupe by case-insensitive name — return the existing one instead of erroring,
        # so the client's "create" is idempotent.
        cur.execute(
            f"SELECT category_id, name, color, sort_order FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` "
            f"WHERE owner_id = %s AND LOWER(name) = LOWER(%s) LIMIT 1",
            [safe, name],
        )
        existing = cur.fetchone()
        if existing:
            return _success({'category': _category_row(existing)}, status=200)
        cur.execute(
            f"SELECT COUNT(*) AS n FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` WHERE owner_id = %s",
            [safe],
        )
        if int((cur.fetchone() or {}).get('n', 0)) >= _MAX_CATEGORIES_PER_OWNER:
            return _error(400, 'Category limit reached')
        cur.execute(
            f"SELECT COALESCE(MAX(sort_order), -1) AS m FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` WHERE owner_id = %s",
            [safe],
        )
        next_sort = int((cur.fetchone() or {}).get('m', -1)) + 1
        category_id = str(uuid.uuid4())
        cur.execute(
            f"INSERT INTO `{_SHARED_RECIPE_CATEGORIES_TABLE}` (owner_id, category_id, name, color, sort_order) "
            f"VALUES (%s,%s,%s,%s,%s)",
            [safe, category_id, name, color, next_sort],
        )
    conn.commit()
    return _success({'category': {'id': category_id, 'name': name, 'color': color, 'sort_order': next_sort}}, status=201)


def _update_recipe_category(owner, category_id, body, request_id=None):
    safe = _safe_owner_token(owner)
    cid = _safe_text(category_id)
    if not cid:
        return _error(400, 'Missing category_id')
    body = body or {}
    sets, params = [], []
    if 'name' in body:
        name = _safe_text(body.get('name'))
        if not name:
            return _error(400, 'Category name cannot be empty')
        sets.append('name = %s')
        params.append(name[:_MAX_CATEGORY_NAME_LEN])
    if 'color' in body:
        sets.append('color = %s')
        params.append(_safe_text(body.get('color')) or None)
    if 'sort_order' in body:
        try:
            so = int(body.get('sort_order'))
        except (TypeError, ValueError):
            so = 0
        sets.append('sort_order = %s')
        params.append(so)
    if not sets:
        return _error(400, 'Nothing to update')
    conn = _mysql_conn()
    _ensure_recipe_category_tables(conn)
    with conn.cursor() as cur:
        try:
            cur.execute(
                f"UPDATE `{_SHARED_RECIPE_CATEGORIES_TABLE}` SET {', '.join(sets)} "
                f"WHERE owner_id = %s AND category_id = %s",
                params + [safe, cid],
            )
        except pymysql.err.IntegrityError:
            return _error(409, 'A category with that name already exists')
        if cur.rowcount == 0:
            cur.execute(
                f"SELECT 1 FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` WHERE owner_id = %s AND category_id = %s",
                [safe, cid],
            )
            if not cur.fetchone():
                return _error(404, 'Category not found')
    conn.commit()
    return _success({'ok': True, 'id': cid})


def _delete_recipe_category(owner, category_id, request_id=None):
    safe = _safe_owner_token(owner)
    cid = _safe_text(category_id)
    if not cid:
        return _error(400, 'Missing category_id')
    conn = _mysql_conn()
    _ensure_recipe_category_tables(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM `{_SHARED_RECIPE_CATEGORY_MAP_TABLE}` WHERE owner_id = %s AND category_id = %s",
            [safe, cid],
        )
        cur.execute(
            f"DELETE FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` WHERE owner_id = %s AND category_id = %s",
            [safe, cid],
        )
    conn.commit()
    return _success({'ok': True, 'id': cid})


def _set_recipe_categories(owner, recipe_id, body, request_id=None):
    """Replace the full set of custom categories on a saved recipe (multi-select)."""
    safe = _safe_owner_token(owner)
    rid = _safe_text(recipe_id)
    if not rid:
        return _error(400, 'Missing item_id')
    raw_ids = (body or {}).get('category_ids')
    if raw_ids is None:
        raw_ids = (body or {}).get('categories')
    if not isinstance(raw_ids, list):
        return _error(400, 'category_ids must be an array')
    desired = [c for c in (_safe_text(c) for c in raw_ids) if c]
    conn = _mysql_conn()
    _ensure_recipe_category_tables(conn)
    with conn.cursor() as cur:
        valid = set()
        if desired:
            placeholders = ','.join(['%s'] * len(desired))
            cur.execute(
                f"SELECT category_id FROM `{_SHARED_RECIPE_CATEGORIES_TABLE}` "
                f"WHERE owner_id = %s AND category_id IN ({placeholders})",
                [safe] + desired,
            )
            valid = {r.get('category_id') for r in (cur.fetchall() or [])}
        # Preserve request order, drop dupes + ids that aren't this owner's categories.
        final_ids = [c for c in dict.fromkeys(desired) if c in valid]
        cur.execute(
            f"DELETE FROM `{_SHARED_RECIPE_CATEGORY_MAP_TABLE}` WHERE owner_id = %s AND recipe_id = %s",
            [safe, rid],
        )
        for cid in final_ids:
            cur.execute(
                f"INSERT INTO `{_SHARED_RECIPE_CATEGORY_MAP_TABLE}` (owner_id, recipe_id, category_id) "
                f"VALUES (%s,%s,%s)",
                [safe, rid, cid],
            )
    conn.commit()
    return _success({'ok': True, 'recipe_id': rid, 'category_ids': final_ids})


# ============================================================================
# RECIPE SHARING (additive) — shareable links for any recipe.
# Create a denormalized snapshot -> share_id; resolve it as JSON (in-app) or a
# beautiful on-brand HTML landing page (browser/social); claim it into an
# owner's saved recipes by reusing the exact save path (_save_saved_recipe_record).
# Isolated: no existing route/DTO touched; reuses _mysql_conn/_success/_error.
# ============================================================================
import secrets as _secrets
import string as _string

_SHARE_LINKS_TABLE = 'shared_recipe_links'
_SHARE_ID_ALPHABET = _string.ascii_letters + _string.digits  # base62
_SHARE_ID_LENGTH = 8
# App Store link for the "Get Trepo" button — REPLACE the id before GA.
_TREPO_APP_STORE_URL = 'https://apps.apple.com/app/id6764135986'
# Fallback web host if the request context has no domainName (e.g. direct invoke).
_SHARE_WEB_HOST = 'https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com'


def _ensure_shared_recipe_links_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_SHARE_LINKS_TABLE}` (
                share_id VARCHAR(16) PRIMARY KEY,
                sharer_owner_id VARCHAR(36) NOT NULL,
                sharer_name VARCHAR(255) NULL,
                title VARCHAR(255) NOT NULL,
                image_url VARCHAR(1000) NULL,
                ingredients JSON NULL,
                instructions JSON NULL,
                notes JSON NULL,
                meal_category VARCHAR(16) NULL,
                source_type VARCHAR(32) NULL,
                source_id VARCHAR(128) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                claim_count INT NOT NULL DEFAULT 0,
                view_count INT NOT NULL DEFAULT 0,
                KEY idx_sharer (sharer_owner_id),
                KEY idx_created (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()
    # Nullable Branch deep link for this share, minted on create when Branch is on.
    # The web card uses it as the "Save in Trepo" button target; NULL -> the card
    # falls back to the trepo:// scheme + App Store buttons. Added post-hoc (guarded)
    # so pre-existing tables gain the column too.
    _ensure_column(conn, _SHARE_LINKS_TABLE, 'branch_url', 'branch_url VARCHAR(1000) NULL')
    # Recipe emoji (iOS sends it; derived from the title if absent). The web card renders it
    # as the hero when there's no image, matching the app's emoji hero. Guarded add.
    _ensure_column(conn, _SHARE_LINKS_TABLE, 'emoji', 'emoji VARCHAR(16) NULL')


def _ensure_shared_recipe_claims_table(conn):
    """One row per (share_id, owner) claim so repeat claims are idempotent and
    claim_count only increments on the first claim by a given owner."""
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `shared_recipe_claims` (
                share_id VARCHAR(16) NOT NULL,
                owner_id VARCHAR(36) NOT NULL,
                saved_recipe_id VARCHAR(36) NULL,
                claimed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (share_id, owner_id),
                KEY idx_owner (owner_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def _generate_share_id(conn):
    """8-char URL-safe base62 id with collision-retry against the links table."""
    for _ in range(8):
        candidate = ''.join(_secrets.choice(_SHARE_ID_ALPHABET) for _ in range(_SHARE_ID_LENGTH))
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM `{_SHARE_LINKS_TABLE}` WHERE share_id = %s LIMIT 1", [candidate])
            if not cur.fetchone():
                return candidate
    raise ServiceError('Could not allocate a share id.', status_code=500)


def _resolve_sharer_name(conn, owner):
    """First name from new_users; None if unknown (never raises)."""
    safe = _safe_owner_token(owner)
    if not safe:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT first_name FROM new_users WHERE user_id = %s LIMIT 1", [safe])
            row = cur.fetchone() or {}
        name = _safe_text(row.get('first_name'))
        return name or None
    except Exception:
        return None


def _share_web_url(share_id, event=None):
    base = _base_url_from_event(event) or _SHARE_WEB_HOST
    return f"{base}/r/{share_id}"


_BRANCH_URL_ENDPOINT = 'https://api2.branch.io/v1/url'


def _mint_branch_link(share_id, title, image_url, landing_url):
    """Mint a Branch.io deep link for a shared recipe.

    Returns the Branch URL string, or None on ANY failure (missing key, non-200,
    timeout, network/parse error) — this must NEVER break share-create; the caller
    falls back to the landing URL. Uses stdlib urllib (no new deps) and only the
    PUBLIC branch_key (which also ships in the app), never the Branch secret.
    """
    import urllib.request

    branch_key = os.getenv('BRANCH_KEY', '').strip()
    if not branch_key:
        return None

    data = {
        'share_id': str(share_id),
        '$deeplink_path': f'r/{share_id}',
        '$canonical_identifier': f'recipe/{share_id}',
        '$og_title': title,
        '$og_description': 'A recipe shared with you on Trepo',
        '$desktop_url': landing_url,
        '$fallback_url': landing_url,
        '$ios_url': 'https://apps.apple.com/app/id6764135986',
        '$ios_deeplink_path': f'r/{share_id}',
        '$uri_redirect_mode': 1,
    }
    if image_url:
        data['$og_image_url'] = image_url

    body = {
        'branch_key': branch_key,
        'channel': 'trepo-share',
        'feature': 'recipe-share',
        'campaign': 'recipe-sharing',
        'data': data,
    }

    try:
        raw = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            _BRANCH_URL_ENDPOINT,
            data=raw,
            method='POST',
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.getcode() != 200:
                return None
            parsed = json.loads(resp.read().decode('utf-8'))
        url = parsed.get('url')
        return url if isinstance(url, str) and url else None
    except Exception as exc:  # noqa: BLE001 — mint must never break share-create
        try:
            status = getattr(exc, 'code', None)
            status = status if type(status) is int else None
            code = 'provider_unavailable'
            if status == 403:
                code = 'provider_forbidden'
                try:
                    detail = json.loads(exc.read(4096)).get('error', {}).get('message', '')
                    if 'do not have access to this link creation via API' in detail:
                        code = 'branch_api_not_entitled'
                except Exception:
                    pass
            _log_event(None, 'branch_mint_failed', share_id=str(share_id),
                       error_code=code, http_status=status)
        except Exception:
            pass
        return None


# Food-keyword → emoji, first match wins (specific before general). Used only when the
# share payload has no emoji (iOS now sends the recipe's emoji); 🍽️ is the safe fallback.
_RECIPE_EMOJI_KEYWORDS = [
    ('pizza', '🍕'), ('taco', '🌮'), ('burrito', '🌯'), ('quesadilla', '🌮'),
    ('sushi', '🍣'), ('ramen', '🍜'), ('noodle', '🍜'), ('spaghetti', '🍝'), ('pasta', '🍝'),
    ('curry', '🍛'), ('fried rice', '🍚'), ('rice', '🍚'), ('soup', '🍲'), ('stew', '🍲'),
    ('salad', '🥗'), ('sandwich', '🥪'), ('wrap', '🌯'), ('burger', '🍔'), ('fries', '🍟'),
    ('omelet', '🍳'), ('egg', '🍳'), ('pancake', '🥞'), ('waffle', '🧇'), ('bacon', '🥓'),
    ('steak', '🥩'), ('beef', '🥩'), ('chicken', '🍗'), ('turkey', '🍗'), ('pork', '🥓'),
    ('lamb', '🍖'), ('bbq', '🍖'), ('grill', '🍖'), ('meat', '🍖'),
    ('shrimp', '🍤'), ('prawn', '🍤'), ('salmon', '🐟'), ('fish', '🐟'), ('dumpling', '🥟'),
    ('bagel', '🥯'), ('croissant', '🥐'), ('pretzel', '🥨'), ('toast', '🍞'), ('bread', '🍞'),
    ('cheese', '🧀'), ('cookie', '🍪'), ('cupcake', '🧁'), ('cake', '🍰'), ('pie', '🥧'),
    ('donut', '🍩'), ('doughnut', '🍩'), ('chocolate', '🍫'), ('candy', '🍬'),
    ('ice cream', '🍨'), ('popcorn', '🍿'), ('potato', '🥔'), ('corn', '🌽'),
    ('broccoli', '🥦'), ('avocado', '🥑'), ('tomato', '🍅'), ('mushroom', '🍄'),
    ('apple', '🍎'), ('banana', '🍌'), ('strawberr', '🍓'), ('grape', '🍇'),
    ('orange', '🍊'), ('lemon', '🍋'), ('peach', '🍑'), ('cherr', '🍒'),
    ('watermelon', '🍉'), ('pineapple', '🍍'), ('mango', '🥭'),
    ('smoothie', '🥤'), ('juice', '🧃'), ('coffee', '☕'), ('tea', '🍵'),
    ('oat', '🥣'), ('cereal', '🥣'), ('yogurt', '🥣'), ('parfait', '🥣'), ('bean', '🫘'),
]


def _derive_recipe_emoji(title):
    """Best-effort single food emoji from the recipe title; 🍽️ fallback. Only used when the
    share payload carries no emoji."""
    t = (title or '').lower()
    for keyword, emoji in _RECIPE_EMOJI_KEYWORDS:
        if keyword in t:
            return emoji
    return '🍽️'


def _create_shared_recipe_link(owner, body, request_id=None):
    """POST /share/recipe/{owner} — insert a denormalized snapshot (survives the
    sharer editing/deleting the source). Returns 201 {share_id, app_url, web_url}."""
    payload = body or {}
    title = _safe_text(payload.get('title'))
    if not title:
        return _error(400, 'title is required')
    ingredients = _clean_string_list(payload.get('ingredients') or [])
    instructions = _clean_string_list(payload.get('instructions') or [])
    notes = _clean_string_list(payload.get('notes') or [])
    image_url = _safe_text(payload.get('image_url')) or None
    meal_category = _safe_text(payload.get('meal_category')) or None
    source_type = _safe_text(payload.get('source_type')) or None
    source_id = _safe_text(payload.get('source_id')) or None
    # Emoji: prefer what iOS sends; else derive from the title (never empty -> 🍽️ fallback).
    emoji = (_safe_text(payload.get('emoji')) or _derive_recipe_emoji(title))[:16]
    event = payload.get('_event')

    conn = _mysql_conn()
    _ensure_shared_recipe_links_table(conn)
    sharer_name = _resolve_sharer_name(conn, owner)
    share_id = _generate_share_id(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO `{_SHARE_LINKS_TABLE}` (
                    share_id, sharer_owner_id, sharer_name, title, image_url, emoji,
                    ingredients, instructions, notes, meal_category, source_type, source_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                share_id, owner, sharer_name, title[:255], image_url, emoji,
                json.dumps(ingredients), json.dumps(instructions), json.dumps(notes),
                meal_category, source_type, source_id,
            )
        )
    conn.commit()
    _log_event(request_id, 'recipe_share_created', owner=owner, share_id=share_id,
               source_type=source_type or '', title=title[:80])

    # web_url is ALWAYS our own recipe card — so the shared/copied link lands on the
    # card first (shows the recipe), never the raw Branch link (which auto-bounces a
    # browser to the App Store before the recipe is ever seen). When Branch is on and
    # mints a link, we STORE it on the row instead; the card renders it as the tap
    # target of the "Save in Trepo" button, where Branch does the proper has-app-open
    # vs App-Store+deferred logic. branch_url stays NULL when the flag is off or the
    # mint fails -> the card falls back to the trepo:// scheme + App Store buttons.
    web_url = _share_web_url(share_id, event)
    branch_url = None
    branch_stored = False
    if str(os.getenv('SHARE_LINKS_USE_BRANCH', '')).strip().lower() in ('1', 'true', 'yes', 'on'):
        branch_url = _mint_branch_link(share_id, title, image_url, web_url)
    if branch_url:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE `{_SHARE_LINKS_TABLE}` SET branch_url = %s WHERE share_id = %s",
                    [branch_url, share_id],
                )
            conn.commit()
            branch_stored = True
        except Exception as exc:  # noqa: BLE001 — never fail share-create on this
            _log_event(request_id, 'branch_url_store_failed', share_id=share_id, error=str(exc)[:200])

    link_mode = 'enhanced' if branch_stored else 'web_fallback'
    _log_event(request_id, 'recipe_share_link_outcome', share_id=share_id,
               link_mode=link_mode,
               traffic_class='review' if owner == '8fcb37c1-0927-4c31-856a-5b761efc226c' else 'unclassified')
    return _success({
        'share_id': share_id,
        'app_url': f'trepo://r/{share_id}',
        'web_url': web_url,
        'link_mode': link_mode,
    }, status=201)


def _fetch_shared_recipe_link(conn, share_id):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM `{_SHARE_LINKS_TABLE}` WHERE share_id = %s LIMIT 1", [share_id])
        return cur.fetchone()


def _shared_link_snapshot_json(row):
    return {
        'share_id': _safe_text(row.get('share_id')),
        'sharer_name': _safe_text(row.get('sharer_name')) or None,
        'recipe': {
            'title': _safe_text(row.get('title')),
            'image_url': _safe_text(row.get('image_url')) or None,
            'emoji': _safe_text(row.get('emoji')) or None,
            'ingredients': _clean_string_list(_parse_json_field(row.get('ingredients')) or []),
            'instructions': _clean_string_list(_parse_json_field(row.get('instructions')) or []),
            'notes': _clean_string_list(_parse_json_field(row.get('notes')) or []),
            'meal_category': _safe_text(row.get('meal_category')) or None,
            'source_type': _safe_text(row.get('source_type')) or None,
        },
    }


def _wants_json_share(event):
    qsp = (event or {}).get('queryStringParameters') or {}
    if _safe_text(qsp.get('format')).lower() == 'json':
        return True
    accept = ''
    for k, v in ((event or {}).get('headers') or {}).items():
        if k.lower() == 'accept':
            accept = _safe_text(v).lower()
            break
    if 'application/json' in accept:
        return True
    return False


def _resolve_shared_recipe(share_id, event, request_id=None):
    """GET /r/{share_id} — content-negotiated. JSON (Accept: application/json or
    ?format=json) returns the snapshot; anything else returns the HTML landing
    page. Increments view_count. 404 if the share_id is unknown."""
    conn = _mysql_conn()
    _ensure_shared_recipe_links_table(conn)
    row = _fetch_shared_recipe_link(conn, share_id)
    want_json = _wants_json_share(event)
    if not row:
        if want_json:
            return _error(404, 'Shared recipe not found')
        return _shared_recipe_not_found_html()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE `{_SHARE_LINKS_TABLE}` SET view_count = view_count + 1 WHERE share_id = %s", [share_id])
        conn.commit()
    except Exception:
        pass
    _log_event(request_id, 'recipe_share_viewed', share_id=share_id, mode='json' if want_json else 'html')
    if want_json:
        return _success(_shared_link_snapshot_json(row), status=200)
    return _render_shared_recipe_html(row)


def _claim_shared_recipe(share_id, owner, request_id=None):
    """POST /share/recipe/{share_id}/claim/{owner} — copy the snapshot into the
    owner's saved recipes using the SAME save path as a normal save
    (_save_saved_recipe_record via the prestructured/explore short-circuit).
    Idempotent per (share_id, owner): a repeat claim returns the existing saved
    recipe and does NOT re-increment claim_count. Returns 200 {saved_recipe_id,
    already_saved}."""
    conn = _mysql_conn()
    _ensure_shared_recipe_links_table(conn)
    _ensure_shared_recipe_claims_table(conn)
    row = _fetch_shared_recipe_link(conn, share_id)
    if not row:
        return _error(404, 'Shared recipe not found')

    # Idempotency: has this owner already claimed this share_id?
    with conn.cursor() as cur:
        cur.execute(
            "SELECT saved_recipe_id FROM `shared_recipe_claims` WHERE share_id = %s AND owner_id = %s LIMIT 1",
            [share_id, owner],
        )
        prior = cur.fetchone()

    _ensure_saved_recipes_table(conn, owner)
    extraction = _shared_link_claim_extraction(share_id, row)
    result, _status = _save_saved_recipe_record(conn, owner, extraction, request_id=request_id)
    saved_recipe_id = _safe_text((result.get('recipe') or {}).get('id'))
    already_saved = bool(result.get('deduped')) or prior is not None

    if prior is None:
        # First claim by this owner → record it + bump claim_count once.
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `shared_recipe_claims` (share_id, owner_id, saved_recipe_id) VALUES (%s, %s, %s)",
                    [share_id, owner, saved_recipe_id or None],
                )
                cur.execute(
                    f"UPDATE `{_SHARE_LINKS_TABLE}` SET claim_count = claim_count + 1 WHERE share_id = %s",
                    [share_id],
                )
            conn.commit()
        except pymysql.err.IntegrityError:
            # Concurrent double-claim by the same owner — the other insert won the
            # PK race; treat as already-saved (don't double-count).
            conn.rollback()
            already_saved = True

    _log_event(request_id, 'recipe_share_claimed', share_id=share_id, owner=owner,
               already_saved=already_saved, saved_recipe_id=saved_recipe_id)
    return _success({'saved_recipe_id': saved_recipe_id, 'already_saved': already_saved}, status=200)


def _shared_link_claim_extraction(share_id, row):
    """Build a pre-structured extraction (same shape as the explore short-circuit)
    from a shared_recipe_links snapshot so claim reuses _save_saved_recipe_record.
    The synthetic per-share resolved_url gives free idempotency via the existing
    resolved_url_hash unique index (a re-claim dedupes to the same saved recipe)."""
    structured = {
        'title': _safe_text(row.get('title')),
        'ingredients': _clean_string_list(_parse_json_field(row.get('ingredients')) or []),
        'instructions': _clean_string_list(_parse_json_field(row.get('instructions')) or []),
        'notes': _clean_string_list(_parse_json_field(row.get('notes')) or []),
    }
    image_url = _safe_text(row.get('image_url'))
    dedup_url = f'https://trepo.ai/shared/{share_id}'
    content = _format_recipe_text(structured)
    return {
        'url': dedup_url,
        'resolved_url': dedup_url,
        'platform': 'explore',        # non-manual → mirrors image; skips generation
        'content': content,
        'caption': content,
        'title': structured['title'],
        'image_url': image_url,
        'image_urls': [image_url] if image_url else [],
        'source': 'shared_link',
        'author_name': None,          # de-attribution — never carry creator fields
        'caption_field': 'shared',
        'warnings': [],
        'prestructured': structured,
        # Reuse the snapshot's stored meal_category so the claim's save skips the classify LLM.
        'meal_category': _safe_text(row.get('meal_category')) or None,
    }


def _share_html_escape(value):
    return (
        _safe_text(value)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )


_SHARED_RECIPE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__ — shared on Trepo</title>
<meta property="og:type" content="website">
<meta property="og:title" content="__OG_TITLE__">
<meta property="og:description" content="A recipe shared with you on Trepo">
<meta property="og:image" content="__OG_IMAGE__">
<meta property="og:url" content="__WEB_URL__">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="__OG_TITLE__">
<meta name="twitter:description" content="A recipe shared with you on Trepo">
<meta name="twitter:image" content="__OG_IMAGE__">
<style>
:root{--cream:#F5E9D8;--ink:#1A1A1A;--thyme:#296065;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{background:var(--cream);color:var(--ink);
  font-family:ui-rounded,"SF Pro Rounded",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;}
body{padding:24px 18px 48px;display:flex;justify-content:center;}
.wrap{width:100%;max-width:520px;}
.brand{font-weight:900;font-size:20px;letter-spacing:-0.5px;text-transform:lowercase;margin-bottom:18px;}
.brand .dot{color:var(--thyme);}
.card{background:#fff;border:2px solid var(--ink);border-radius:20px;
  box-shadow:4px 4px 0 var(--ink);overflow:hidden;}
.hero{width:100%;aspect-ratio:16/10;object-fit:cover;display:block;border-bottom:2px solid var(--ink);background:#e7d8c2;}
.hero-emoji{width:100%;aspect-ratio:16/10;display:flex;align-items:center;justify-content:center;font-size:80px;line-height:1;border-bottom:2px solid var(--ink);background:#e7d8c2;}
.pad{padding:22px 20px;}
.eyebrow{display:inline-block;background:var(--thyme);color:var(--cream);font-weight:800;
  font-size:11px;letter-spacing:1.5px;text-transform:uppercase;padding:5px 10px;border-radius:8px;
  border:2px solid var(--ink);box-shadow:2px 2px 0 var(--ink);margin-bottom:14px;}
h1{font-weight:900;font-size:28px;line-height:1.1;letter-spacing:-0.5px;margin-bottom:8px;}
.sharedby{font-weight:700;font-size:15px;color:var(--ink);opacity:.75;margin-bottom:4px;}
.sectlabel{font-weight:800;font-size:12px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--thyme);margin:0 0 10px;}
ul.ings{list-style:none;margin:0 0 6px;}
ul.ings li{font-weight:600;font-size:15px;line-height:1.35;padding:8px 0;border-bottom:1.5px dashed rgba(26,26,26,.18);}
ul.ings li:last-child{border-bottom:0;}
.more{font-weight:700;font-size:14px;color:var(--ink);opacity:.6;margin-top:8px;}
.sect{margin-top:22px;}
ol.steps{list-style:none;margin:0;counter-reset:step;}
ol.steps li{position:relative;font-weight:600;font-size:15px;line-height:1.4;padding:9px 0 9px 40px;
  border-bottom:1.5px dashed rgba(26,26,26,.18);counter-increment:step;}
ol.steps li:last-child{border-bottom:0;}
ol.steps li::before{content:counter(step);position:absolute;left:0;top:8px;width:26px;height:26px;
  background:var(--thyme);color:var(--cream);font-weight:900;font-size:13px;border-radius:50%;
  border:2px solid var(--ink);display:flex;align-items:center;justify-content:center;}
ul.notes{list-style:none;margin:0;}
ul.notes li{font-weight:600;font-size:14px;line-height:1.4;padding:10px 12px;margin-bottom:8px;
  background:#FFF4CC;border:2px solid var(--ink);border-radius:12px;box-shadow:2px 2px 0 var(--ink);}
ul.notes li:last-child{margin-bottom:0;}
.btns{margin-top:26px;display:flex;flex-direction:column;gap:12px;}
.btn{display:block;text-align:center;text-decoration:none;font-weight:900;font-size:17px;
  padding:16px 18px;border-radius:14px;border:2px solid var(--ink);box-shadow:4px 4px 0 var(--ink);
  transition:transform .05s ease,box-shadow .05s ease;}
.btn:active{transform:translate(2px,2px);box-shadow:2px 2px 0 var(--ink);}
.btn-primary{background:var(--thyme);color:var(--cream);}
.btn-secondary{background:#fff;color:var(--ink);}
.foot{text-align:center;font-weight:600;font-size:12px;opacity:.5;margin-top:22px;}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">trepo<span class="dot">.</span></div>
  <div class="card">
    __HERO__
    <div class="pad">
      <span class="eyebrow">Shared with you</span>
      <h1>__TITLE__</h1>
      <div class="sharedby">Shared by __SHARER__</div>
      __INGREDIENTS_BLOCK__
      __INSTRUCTIONS_BLOCK__
      __NOTES_BLOCK__
      <div class="btns">__BTNS_BLOCK__</div>
      <div class="foot">Save recipes from anywhere. Cook what you have.</div>
    </div>
  </div>
</div>
<script>
(function(){
  // Dependency-free deferred deep link: stash the share token on the clipboard
  // BEFORE bouncing to the App Store; the app reads it on first launch.
  var el = document.getElementById('get-trepo');
  if(!el) return;
  el.addEventListener('click', function(){
    try{
      if(navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText('trepo-share:__SHARE_ID__');
      }
    }catch(e){}
    // Let the default navigation to the App Store proceed.
  });
})();
(function(){
  // "Save in Trepo": try the DIRECT app scheme first (reliable has-app open -> resolves +
  // saves the recipe). If the app doesn't take over within ~1.2s (no app installed), the page
  // is still foregrounded, so fall back to the Branch link (App Store + Branch deferred). The
  // has-app case backgrounds the page, delaying this timer, so the fallback does not fire.
  var s = document.getElementById('save-in-trepo');
  if(!s) return;
  s.addEventListener('click', function(ev){
    ev.preventDefault();
    var scheme = s.getAttribute('href');
    var fallback = s.getAttribute('data-fallback');
    var ts = Date.now();
    window.location = scheme;
    setTimeout(function(){
      if(Date.now() - ts < 1500 && fallback){ window.location = fallback; }
    }, 1200);
  });
})();
</script>
</body>
</html>"""


def _render_shared_recipe_html(row):
    share_id = _safe_text(row.get('share_id'))
    title = _safe_text(row.get('title')) or 'A recipe'
    sharer = _safe_text(row.get('sharer_name')) or 'a friend'
    image_url = _safe_text(row.get('image_url'))
    ingredients = _clean_string_list(_parse_json_field(row.get('ingredients')) or [])
    instructions = _clean_string_list(_parse_json_field(row.get('instructions')) or [])
    notes = _clean_string_list(_parse_json_field(row.get('notes')) or [])

    emoji = _safe_text(row.get('emoji'))
    if image_url:
        hero = f'<img class="hero" src="{_share_html_escape(image_url)}" alt="{_share_html_escape(title)}">'
    elif emoji:
        # No photo but we have the recipe emoji — render a big centered emoji hero (matches the
        # app's emoji hero), reusing the .hero box dims + border/bg.
        hero = f'<div class="hero-emoji" role="img" aria-label="{_share_html_escape(title)}">{_share_html_escape(emoji)}</div>'
    else:
        hero = ''

    # Full recipe on the shared web card — ALL ingredients (no truncation), the steps, and
    # any notes — so a recipient sees the whole thing before they open/get the app.
    if ingredients:
        items = ''.join(f'<li>{_share_html_escape(i)}</li>' for i in ingredients)
        ingredients_block = f'<div class="sect"><div class="sectlabel">Ingredients</div><ul class="ings">{items}</ul></div>'
    else:
        ingredients_block = ''

    if instructions:
        steps = ''.join(f'<li>{_share_html_escape(s)}</li>' for s in instructions)
        instructions_block = f'<div class="sect"><div class="sectlabel">Steps</div><ol class="steps">{steps}</ol></div>'
    else:
        instructions_block = ''

    if notes:
        note_items = ''.join(f'<li>{_share_html_escape(n)}</li>' for n in notes)
        notes_block = f'<div class="sect"><div class="sectlabel">Notes</div><ul class="notes">{note_items}</ul></div>'
    else:
        notes_block = ''

    # Buttons. When a Branch link was minted for this share, show ONE primary CTA whose
    # href is the Branch link — a TAP on it triggers Branch's proper has-app-open (opens
    # the app to the recipe) or App-Store+deferred (recipe waiting after install). This
    # is reliable precisely because it's a tap from this page, not a pasted address-bar
    # URL. Fallback (no branch_url): the trepo:// scheme opener + an App Store link that
    # stashes the deferred share token on the clipboard (existing #get-trepo JS below).
    branch_url = _safe_text(row.get('branch_url'))
    if branch_url:
        # "Save in Trepo" opens the app via the DIRECT trepo:// scheme first — a reliable
        # has-app open that resolves + saves the recipe (Branch's has-app open was the flaky
        # part: it opened the app but didn't reliably pass the share). Branch is kept ONLY as
        # the no-app fallback (App-Store + deferred), fired by the timeout JS below when the app
        # doesn't take over. share_id-based scheme; branch_url carried in data-fallback.
        btns_block = (
            f'<a class="btn btn-primary" id="save-in-trepo" '
            f'href="trepo://r/{_share_html_escape(share_id)}" '
            f'data-fallback="{_share_html_escape(branch_url)}">Save in Trepo</a>'
        )
    else:
        btns_block = (
            f'<a class="btn btn-primary" href="trepo://r/{_share_html_escape(share_id)}">Open in Trepo</a>'
            f'<a class="btn btn-secondary" id="get-trepo" href="{_share_html_escape(_TREPO_APP_STORE_URL)}">'
            "Get Trepo — it's free</a>"
        )

    html = _SHARED_RECIPE_HTML_TEMPLATE
    replacements = {
        '__TITLE__': _share_html_escape(title),
        '__OG_TITLE__': _share_html_escape(title),
        '__OG_IMAGE__': _share_html_escape(image_url),
        '__WEB_URL__': _share_html_escape(_share_web_url(share_id)),
        '__SHARER__': _share_html_escape(sharer),
        '__HERO__': hero,
        '__INGREDIENTS_BLOCK__': ingredients_block,
        '__INSTRUCTIONS_BLOCK__': instructions_block,
        '__NOTES_BLOCK__': notes_block,
        '__BTNS_BLOCK__': btns_block,
        '__SHARE_ID__': _share_html_escape(share_id),
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html; charset=utf-8', 'Access-Control-Allow-Origin': '*'},
        'body': html,
    }


def _shared_recipe_not_found_html():
    body = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Recipe not found — Trepo</title>"
            "<style>body{background:#F5E9D8;color:#1A1A1A;font-family:ui-rounded,-apple-system,system-ui,sans-serif;"
            "display:flex;min-height:100vh;align-items:center;justify-content:center;text-align:center;padding:24px;margin:0}"
            "h1{font-weight:900;font-size:26px;margin:0 0 10px}p{font-weight:600;opacity:.7;margin:0 0 22px}"
            "a{display:inline-block;font-weight:900;background:#296065;color:#F5E9D8;text-decoration:none;"
            "padding:14px 20px;border-radius:14px;border:2px solid #1A1A1A;box-shadow:4px 4px 0 #1A1A1A}</style></head>"
            "<body><div><h1>This recipe link expired</h1>"
            "<p>The shared recipe couldn't be found.</p>"
            f"<a href=\"{_TREPO_APP_STORE_URL}\">Get Trepo — it's free</a></div></body></html>")
    return {
        'statusCode': 404,
        'headers': {'Content-Type': 'text/html; charset=utf-8', 'Access-Control-Allow-Origin': '*'},
        'body': body,
    }


def _match_share_route(raw_path):
    """Right-anchored match so a stage prefix in rawPath doesn't break routing.
    Returns (kind, params) or (None, None). Kinds:
      'claim'  -> {'share_id','owner'}   for /share/recipe/{share_id}/claim/{owner}
      'create' -> {'owner'}              for /share/recipe/{owner}
      'resolve'-> {'share_id'}           for /r/{share_id}
    """
    segs = [s for s in _safe_text(raw_path).split('/') if s]
    if len(segs) >= 5 and segs[-5] == 'share' and segs[-4] == 'recipe' and segs[-2] == 'claim':
        return 'claim', {'share_id': segs[-3], 'owner': segs[-1]}
    if len(segs) >= 3 and segs[-3] == 'share' and segs[-2] == 'recipe':
        return 'create', {'owner': segs[-1]}
    if len(segs) >= 2 and segs[-2] == 'r':
        return 'resolve', {'share_id': segs[-1]}
    return None, None


def _handler_impl(event, context):
    # Keep-warm ping — MUST be the first branch: return instantly with NO auth, NO DB, NO work.
    # An EventBridge rule fires this every few minutes to hold a container hot so a user's Save
    # / claim tap doesn't eat the ~3s cold start (this function has no provisioned concurrency
    # and is low-traffic). Zero side effects.
    if isinstance(event, dict) and (event.get('warmup') is True or event.get('source') == 'keepwarm'):
        return {'statusCode': 200, 'body': '{"ok":true,"warm":true}'}
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
    if (event or {}).get('async_task') == 'recover_recipe_checkpoints_v2' and not (event or {}).get('requestContext'):
        return recipe_batch_checkpoint.recover(SimpleNamespace(**globals()))
    if (event or {}).get('async_task') == _ASYNC_TASK_PROCESS_SAVED_RECIPE_BATCH:
        return _handle_async_saved_recipe_batch_task(event, request_id=request_id, context=context)
    if (event or {}).get('async_task') == _ASYNC_TASK_PROCESS_SAVED_RECIPE_URL:
        return _handle_async_saved_recipe_url_task(event, request_id=request_id)
    if (event or {}).get('async_task') == _ASYNC_TASK_REPAIR_SAVED_RECIPE:
        return _repair_saved_recipe(_safe_text(event.get('owner')),
                                    _safe_text(event.get('recipe_id')),
                                    _safe_text(event.get('url')),
                                    request_id=request_id)
    if (event or {}).get('async_task') == _ASYNC_TASK_REFINE_SAVED_RECIPE_TEXT:
        return _handle_async_saved_recipe_text_task(event, request_id=request_id)
    if (event or {}).get('async_task') == _ASYNC_TASK_PERSONALIZE_WARM:
        return _handle_personalize_warm_task(event, request_id=request_id)

    http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
    path_params = event.get('pathParameters') or {}
    owner = _safe_text(path_params.get('owner'))
    item_id = _safe_text(path_params.get('item_id'))
    job_id = _safe_text(path_params.get('job_id'))
    raw_path = event.get('rawPath') or event.get('requestContext', {}).get('http', {}).get('path') or ''

    try:
        if http_method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

        # --- Recipe sharing (additive; right-anchored so a stage prefix is OK) ---
        share_kind, share_params = _match_share_route(raw_path)
        if share_kind == 'resolve':
            if http_method != 'GET':
                return _error(405, f'Method {http_method} not allowed')
            return _resolve_shared_recipe(share_params['share_id'], event, request_id=request_id)
        if share_kind == 'claim':
            if http_method != 'POST':
                return _error(405, f'Method {http_method} not allowed')
            if not share_params['owner']:
                return _error(400, 'Missing owner parameter')
            return _claim_shared_recipe(share_params['share_id'], share_params['owner'], request_id=request_id)
        if share_kind == 'create':
            if http_method != 'POST':
                return _error(405, f'Method {http_method} not allowed')
            if not share_params['owner']:
                return _error(400, 'Missing owner parameter')
            body = _parse_json_body(event)
            body['_event'] = event
            return _create_shared_recipe_link(share_params['owner'], body, request_id=request_id)

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

        admission_query = event.get('queryStringParameters') or {}
        admission_body = _parse_json_body(event) if http_method == 'POST' else {}
        if admission_query.get('protocol') == admission.PROTOCOL or admission_body.get('protocol') == admission.PROTOCOL:
            return admission.handle(SimpleNamespace(**globals()), event, owner, http_method,
                admission_body, admission_query, request_id)


        # --- Custom recipe categories (self-contained; owner-scoped tags) ---
        if raw_path.endswith('/categories') and '/saved-recipes/' in raw_path:
            if http_method != 'PUT':
                return _error(405, f'Method {http_method} not allowed')
            if not item_id:
                return _error(400, 'Missing item_id parameter')
            return _set_recipe_categories(owner, item_id, _parse_json_body(event), request_id=request_id)
        if '/recipe-categories' in raw_path:
            category_id = _safe_text(path_params.get('category_id'))
            if http_method == 'GET':
                return _list_recipe_categories(owner, request_id=request_id)
            if http_method == 'POST':
                return _create_recipe_category(owner, _parse_json_body(event), request_id=request_id)
            if http_method == 'PUT':
                if not category_id:
                    return _error(400, 'Missing category_id parameter')
                return _update_recipe_category(owner, category_id, _parse_json_body(event), request_id=request_id)
            if http_method == 'DELETE':
                if not category_id:
                    return _error(400, 'Missing category_id parameter')
                return _delete_recipe_category(owner, category_id, request_id=request_id)
            return _error(405, f'Method {http_method} not allowed')

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
            if body.get('operation') == 'confirm_recipe_amounts':
                return _confirm_recipe_amounts(owner, body, event)
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
        # Any unhandled 500 in saved-recipes (DB insert failure, availability crash, etc.)
        # → surface it so it pages (RecipesBackendError) rather than being a silent 5xx.
        _report_backend_error('saved_recipes_request', owner_id=owner, code='unhandled_500',
                              error=exc, job_id=raw_path)
        import traceback
        traceback.print_exc()
        return _error(500, str(exc))


# --- Tail-latency instrumentation (log-only, zero behaviour change) -----------------
# Sporadic 32-40s sync requests hit the API Gateway 29s ceiling and 504 with NOTHING in
# the logs to explain them: the gateway gives up while the Lambda is still running, so no
# existing code path ever reports the outcome. `request_start` proves the request arrived
# and pins the API GW request id; `request_end` sits in a finally block so the duration is
# recorded even when the gateway has already hung up on the caller, and even when the
# handler raises.
#
# Fail-soft BY CONSTRUCTION: every logging statement is individually wrapped, so a bug in
# instrumentation can never turn a working request into a 500. The handler's return value
# and any exception pass through completely untouched.


def _request_log_fields(event):
    ctx = (event or {}).get('requestContext') or {}
    http = ctx.get('http') or {}
    async_task = _safe_text((event or {}).get('async_task'))
    return {
        'path': _safe_text((event or {}).get('rawPath') or http.get('path')),
        'route': _safe_text((event or {}).get('routeKey') or ctx.get('resourcePath')),
        'method': _safe_text(http.get('method')),
        'apigw_request_id': _safe_text(ctx.get('requestId')),
        'invocation': f'async_task:{async_task}' if async_task else 'http',
    }


def handler(event, context):
    # Keep-warm pings are excluded deliberately: they fire every few minutes, carry no
    # user-visible latency, and must stay instant with zero side effects.
    if isinstance(event, dict) and (event.get('warmup') is True or event.get('source') == 'keepwarm'):
        return _handler_impl(event, context)

    started = time.time()
    fields = {}
    try:
        fields = _request_log_fields(event)
        _log_event(fields.get('apigw_request_id'), 'request_start', **fields)
    except Exception:
        pass

    # Keep gateway timing; recipe_work_budget also bounds native async invocations.
    try:
        _mark_sync_request_start(
            fields.get('invocation') == 'http' and not _safe_text((event or {}).get('async_task'))
        )
    except Exception:
        pass

    outcome = 'exception'
    status_code = None
    try:
        with recipe_work_budget.invocation(context, fields.get('invocation') == 'http'):
            result = _handler_impl(event, context)
        outcome = 'returned'
        if isinstance(result, dict):
            status_code = result.get('statusCode')
        return result
    finally:
        try:
            remaining_ms = None
            get_remaining = getattr(context, 'get_remaining_time_in_millis', None)
            if callable(get_remaining):
                remaining_ms = get_remaining()
            _log_event(
                fields.get('apigw_request_id'),
                'request_end',
                duration_ms=int((time.time() - started) * 1000),
                outcome=outcome,
                status_code=status_code,
                # How close we came to the Lambda timeout. A small value here on a 504 is
                # the difference between "we were slow" and "we were still working".
                remaining_ms=remaining_ms,
                **fields,
            )
        except Exception:
            pass
        # Clear it: containers are reused, and a stale start time would make the NEXT
        # request think it had already burned its budget.
        try:
            _mark_sync_request_start(False)
        except Exception:
            pass

BUILD_STAMP = "2026-08-02T05:59:36Z sha256:a6ad62e45734106c"


def _mark_saved_recipe_text_failed(conn, owner, recipe_id, request_id=None):
    """Persist one terminal outcome in both stores; never undo ready or deleted data."""
    table = _saved_recipes_table(owner)
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(f"SELECT status FROM `{table}` WHERE _id = %s AND _owner = %s FOR UPDATE",
                        (recipe_id, owner))
            row = cur.fetchone()
            if not row or row.get('status') != 'processing':
                conn.commit()
                return False
            cur.execute(f"UPDATE `{table}` SET status = 'failed' WHERE _id = %s AND _owner = %s AND status = 'processing'",
                        (recipe_id, owner))
        _dual_write_saved_recipe_to_shared(conn, owner, recipe_id,
                                          request_id=request_id, transactional=True)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise



def _availability_summary(availability):
    summarize = getattr(recipe_inventory_llm, 'availability_summary', None)
    if summarize:
        return summarize(availability)
    return {key: availability.get(key) for key in (
        'kitchen_version', 'can_make_exact', 'can_make_with_subs',
        'matched_count', 'missing_count',
    )}



def _availability_cache_source(source):
    versioned = getattr(recipe_inventory_llm, 'inventory_cache_source', None)
    return versioned(source) if versioned else source



def _confirm_recipe_amounts(owner, body, event):
    from quantity_confirmation import confirm_cached_availability, require_authenticated_owner, ConfirmationRejected
    try:
        require_authenticated_owner(event, owner)
        conn = _mysql_conn()
        availability = confirm_cached_availability(conn, owner, body.get('recipe_source'), body.get('recipe_id'), body)
        return _success({'recipe_id': body.get('recipe_id'), 'recipe_source': body.get('recipe_source'),
                         'availability': _availability_summary(availability),
                         'ingredient_matches': availability['ingredient_matches']})
    except ConfirmationRejected as error:
        return _error(error.status, error.code)
