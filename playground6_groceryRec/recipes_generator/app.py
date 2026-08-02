# recipes_generator/app.py - Async Lambda: read kitchen, GPT 10 kitchen-only + 10 need-grocery recipes, write {owner}_recipes
import os
import re
import json
import sys
import time
import hashlib
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc

_LAYER_PYTHON = Path(__file__).resolve().parents[1] / 'recipe_inventory_layer' / 'python'
if _LAYER_PYTHON.exists() and str(_LAYER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_LAYER_PYTHON))

import recipe_inventory_llm

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
MIN_KITCHEN_ITEMS = 5
RECIPE_MEAL_CATEGORIES = ('breakfast', 'lunch', 'dinner', 'snacks')
OWNER_LOCK_TIMEOUT_SECONDS = 5
_OWNER_KITCHEN_STATE_TABLE = 'owner_kitchen_state'
OPENAI_TIMEOUT_SECONDS = int(os.getenv('OPENAI_TIMEOUT_SECONDS', '45'))
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '2'))
# Large kitchens (e.g. 195 items) made the single 20-recipe generation call exceed
# the 45s OpenAI client timeout ("Request timed out" -> handler_error, gutting the
# refresh). Cap the ingredient CONTEXT for generation to the N most-recently-added
# items (_get_kitchen_ingredients returns _createdDate DESC) — plenty for 20 varied
# recipes and deterministically fast. Matching still uses the full kitchen. Also give
# the generation call more timeout headroom (Lambda timeout is 420s).
RECIPE_GEN_MAX_INGREDIENTS = int(os.getenv('RECIPE_GEN_MAX_INGREDIENTS', '120'))
# Beyond the detailed head, list the remaining kitchen items compactly (name
# only, no description) so a 200+ item kitchen can still reach OLDER items in
# recipes instead of them being invisible past the cap. Name-only keeps the
# token cost small (~2-4 tokens/item), so a ~180-item tail is only ~a few
# hundred tokens. Total context (head + tail) is bounded by RECIPE_GEN_MAX_TOTAL.
RECIPE_GEN_MAX_TOTAL = int(os.getenv('RECIPE_GEN_MAX_TOTAL', '320'))
RECIPE_GEN_TIMEOUT_SECONDS = int(os.getenv('RECIPE_GEN_TIMEOUT_SECONDS', '90'))
# gpt-5.5's long-generation tail intermittently exceeds the 90s per-attempt client
# timeout on EVERY retry (~3% of runs -> APITimeoutError at timeout*(retries+1)).
# Cap the primary model at 2 attempts, then regenerate once on the fast fallback
# model instead of failing the refresh. Rollback = env flip (empty fallback model
# disables failover; bump RECIPE_GEN_MAX_RETRIES to restore the old 3-attempt budget).
RECIPE_GEN_MAX_RETRIES = int(os.getenv('RECIPE_GEN_MAX_RETRIES', '1'))
RECIPE_GEN_FALLBACK_MODEL = (os.getenv('RECIPE_GEN_FALLBACK_MODEL', 'gpt-5.4') or '').strip()
RECIPE_GEN_FALLBACK_TIMEOUT_SECONDS = int(os.getenv('RECIPE_GEN_FALLBACK_TIMEOUT_SECONDS', '75'))
SUBSTITUTION_MODEL = os.getenv('OPENAI_SUBSTITUTION_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o'))


def _create_chat(client, **kwargs):
    """chat.completions.create with self-healing param fallback. gpt-5.x reject
    some legacy/explicit params (max_tokens, custom temperature) with a 400 that
    names the offending param; drop exactly that param and retry so future model
    families self-heal instead of silently failing to the fallback path. Driven by
    the API's actual response, not brittle model-name matching."""
    for _ in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Extract the offending param from the structured body, the exception's
            # .param attr, or (most robust across SDK versions) the quoted token in
            # the 400 message, e.g. "'temperature' does not support 0.4".
            param = None
            body = getattr(exc, 'body', None)
            if isinstance(body, dict):
                err = body.get('error') if isinstance(body.get('error'), dict) else body
                param = err.get('param')
            if not param:
                param = getattr(exc, 'param', None)
            if not param:
                m = re.search(r"'([A-Za-z_]+)'", str(exc))
                if m:
                    param = m.group(1)
            if not param or param not in kwargs:
                raise
            print(json.dumps({'evt': 'openai_param_dropped', 'service': 'recipes_generator', 'param': param}))
            kwargs.pop(param, None)
    return client.chat.completions.create(**kwargs)
MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS = max(0, int(os.getenv('RECIPES_MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS', '2')))
DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

# Self-healing sweep for owners stuck with an unfulfilled recipe refresh (crashed/timed-out
# generator run left kitchen_version > last_recipe_refresh_completed_version with
# recipe_refresh_needed=1 and nothing to re-fire it). Triggered by an EventBridge rule with
# input {"sweep": true}. RECIPE_SWEEP_ENABLED=false fully disables the branch.
_RECIPE_SWEEP_ENABLED = os.getenv('RECIPE_SWEEP_ENABLED', 'true').strip().lower() == 'true'
_RECIPE_SWEEP_BATCH = int(os.getenv('RECIPE_SWEEP_BATCH', '25'))  # owners re-driven per sweep run
_RECIPE_SWEEP_CLAIM_TTL_MIN = int(os.getenv('RECIPE_SWEEP_CLAIM_TTL_MIN', '8'))  # don't re-fire an owner claimed < TTL ago (in-flight / backoff)


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
def _ai_op(op, model, latency_ms, status, input_summary='', output_summary='',
           error=None, owner_id=None, job_id=None, service='recipes'):
    """Emit one per-LLM-call telemetry marker (evt=ai_op) to stdout. Best-effort:
    a logging failure must never break the op."""
    try:
        rec = {
            'evt': 'ai_op',
            'service': service,
            'op': op,
            'owner_id': str(owner_id) if owner_id is not None else None,
            'model': str(model) if model is not None else None,
            'latency_ms': int(latency_ms) if latency_ms is not None else None,
            'status': status,
            'input': (str(input_summary) if input_summary is not None else '')[:400],
            'output': (str(output_summary) if output_summary is not None else '')[:1200],
        }
        if error is not None:
            rec['error'] = (error if isinstance(error, str) else str(error))[:500]
        if job_id is not None:
            rec['job_id'] = str(job_id)
        print(json.dumps(rec))
    except Exception:
        pass


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


def _get_db_config():
    missing = [n for n in _DB_ENV_VARS if not os.getenv(n)]
    if missing:
        raise RuntimeError(f"Missing DB env: {', '.join(missing)}")
    port = int(os.getenv('DB_PORT', '3306'))
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


def _lock_mysql_conn(timeout_seconds):
    if pymysql is None:
        raise RuntimeError(f"pymysql import failed: {_PYMYSQL_IMPORT_ERROR}")
    wait_timeout = max(
        DB_READ_TIMEOUT,
        DB_WRITE_TIMEOUT,
        int(timeout_seconds or 0) + 5,
    )
    return pymysql.connect(
        **_get_db_config(),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=max(DB_CONNECT_TIMEOUT, 5),
        read_timeout=wait_timeout,
        write_timeout=wait_timeout,
    )


def _acquire_named_lock(lock_name, timeout_seconds):
    conn = _lock_mysql_conn(timeout_seconds)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT GET_LOCK(%s, %s) AS acquired", [lock_name, timeout_seconds])
            row = cur.fetchone() or {}
        if row.get('acquired') == 1:
            return conn
    except Exception:
        conn.close()
        raise
    conn.close()
    return None


def _release_named_lock(conn, lock_name):
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT RELEASE_LOCK(%s)", [lock_name])
    except Exception:
        pass
    finally:
        conn.close()


def _recipes_table(owner):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_recipes"


# ---- Shared-table migration M4: dual-write the recipes 'current' cache row to shared_recipes ----
# One-row-per-owner cache (shared PK = owner_id). Off by default (DUAL_WRITE_RECIPES).
# Copies the per-user 'current' row into shared_recipes via INSERT...SELECT so the
# mirror is a faithful copy of the latest write; ON DUPLICATE KEY UPDATE (owner_id PK)
# re-syncs on every state transition. Non-blocking: errors swallowed + surfaced as a
# metric-filterable {evt:'dual_write_miss', family:'recipes'} marker.
DUAL_WRITE_RECIPES = os.getenv('DUAL_WRITE_RECIPES', 'false').lower() == 'true'
_SHARED_RECIPES_TABLE = 'shared_recipes'
_SHARED_RECIPES_COLS = (
    '_id', '_owner', 'status', 'kitchen_only', 'need_grocery', 'error_message',
    '_createdDate', '_updatedDate',
)


def _dual_write_recipes_to_shared(conn, owner):
    if not DUAL_WRITE_RECIPES:
        return
    try:
        table = _recipes_table(owner)
        cols = ', '.join(f'`{c}`' for c in _SHARED_RECIPES_COLS)
        src = ', '.join(f's.`{c}`' for c in _SHARED_RECIPES_COLS)
        upd = ', '.join(f'`{c}`=VALUES(`{c}`)' for c in _SHARED_RECIPES_COLS)
        sql = (
            f"INSERT INTO `{_SHARED_RECIPES_TABLE}` (`owner_id`, {cols}) "
            f"SELECT %s, {src} FROM `{table}` s WHERE s.`_id`='current' "
            f"ON DUPLICATE KEY UPDATE `owner_id`=VALUES(`owner_id`), {upd}"
        )
        with conn.cursor() as cur:
            cur.execute(sql, (owner,))
        conn.commit()
    except Exception as exc:
        try:
            import sys
            print(json.dumps({
                'evt': 'dual_write_miss', 'family': 'recipes',
                'owner_id': str(owner) if owner is not None else None,
                'error': (str(exc)[:500] if exc is not None else ''),
            }), file=sys.stderr)
        except Exception:
            pass


def _sanitize_user_id(user_id):
    return re.sub(r'[^a-zA-Z0-9_-]', '', (user_id or ''))


def _owner_lock_name(owner):
    return f"recipes-owner:{_sanitize_user_id(owner)}"


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
    member_ids = _get_household_member_ids(conn, owner) if conn else [safe]
    placeholders = ','.join(['%s'] * len(member_ids))
    return shared_name, f"owner_id IN ({placeholders})", member_ids


def _ensure_recipes_table(conn, owner):
    table = _recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                _id VARCHAR(36) PRIMARY KEY,
                _owner VARCHAR(36) NOT NULL,
                status ENUM('ready', 'regenerating', 'failed', 'empty') NOT NULL DEFAULT 'empty',
                kitchen_only JSON,
                need_grocery JSON,
                error_message TEXT,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()


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


def _get_owner_kitchen_version(conn, owner):
    _ensure_owner_kitchen_state_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT kitchen_version FROM `{_OWNER_KITCHEN_STATE_TABLE}` WHERE owner = %s LIMIT 1",
            [_sanitize_user_id(owner)],
        )
        row = cur.fetchone() or {}
    return int(row.get('kitchen_version') or 0)


def _mark_recipe_refresh_complete(conn, owners):
    normalized_owners = [_sanitize_user_id(owner) for owner in (owners or []) if _sanitize_user_id(owner)]
    if not normalized_owners:
        return
    _ensure_owner_kitchen_state_table(conn)
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            UPDATE `{_OWNER_KITCHEN_STATE_TABLE}`
            SET last_recipe_refresh_completed_version = kitchen_version,
                recipe_refresh_needed = 0,
                last_recipe_refresh_completed_at = NOW(),
                _updatedDate = NOW()
            WHERE owner = %s
            """,
            [(owner,) for owner in normalized_owners],
        )
    conn.commit()


def _get_kitchen_ingredients(conn, owner):
    """Return list of dicts: [{"name": str, "description": str|None}, ...]. Deduplicated by name (latest row wins)."""
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
        """, [table_name])
        if cur.fetchone()['n'] == 0:
            return []
        cur.execute("""
            SELECT COUNT(*) AS n FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s AND column_name = 'product_description'
        """, [table_name])
        has_desc = cur.fetchone()['n'] > 0
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s
              AND column_name IN ('analysis_stage', 'analysis_status')
        """, [table_name])
        column_names = {
            (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
            for row in cur.fetchall()
            if (row.get('column_name') or row.get('COLUMN_NAME') or '').strip()
        }
        where_parts = []
        params = []
        if owner_where:
            where_parts.append(owner_where)
            params = list(owner_params)
        where_parts.append("action = 'IN'")
        if 'analysis_stage' in column_names:
            where_parts.append("(`analysis_stage` IN ('preliminary', 'fast', 'final') OR `analysis_stage` IS NULL)")
        if 'analysis_status' in column_names:
            where_parts.append("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)")
        where_clause = " AND ".join(where_parts)
        if has_desc:
            cur.execute(f"""
                SELECT product_name, product_description FROM `{table_name}`
                WHERE {where_clause}
                ORDER BY _createdDate DESC
            """, params)
        else:
            cur.execute(f"""
                SELECT product_name FROM `{table_name}`
                WHERE {where_clause}
                ORDER BY _createdDate DESC
            """, params)
        rows = cur.fetchall()
    result = []
    seen = set()
    for row in rows:
        name = (row.get('product_name') or '').strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        desc = ((row.get('product_description') or '').strip() or None) if has_desc else None
        result.append({'name': name, 'description': desc})
    return result


def _set_status(conn, owner, status, error_message=None):
    table = _recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE `{table}` SET status = %s, error_message = %s, _updatedDate = NOW() WHERE _id = 'current'",
            (status, error_message)
        )
        conn.commit()
    _dual_write_recipes_to_shared(conn, owner)


def _set_empty(conn, owner):
    table = _recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE `{table}`
            SET status = 'empty',
                kitchen_only = %s,
                need_grocery = %s,
                error_message = NULL,
                _updatedDate = NOW()
            WHERE _id = 'current'
            """,
            (json.dumps([]), json.dumps([]))
        )
        conn.commit()
    _dual_write_recipes_to_shared(conn, owner)


def _format_kitchen_for_prompt(items):
    """Format list of {name, description} for GPT: 'Name (what it is)' or just 'Name' if no description."""
    parts = []
    for it in items:
        name = it.get('name') or ''
        desc = it.get('description')
        if desc:
            parts.append(f"{name} ({desc})")
        else:
            parts.append(name)
    return ', '.join(parts) if parts else 'No specific ingredients (suggest pantry staples)'


def _singularize_token(token):
    token = str(token or '').strip().lower()
    if len(token) <= 3:
        return token
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    if token.endswith('oes') and len(token) > 4:
        return token[:-2]
    if token.endswith('s') and not token.endswith('ss'):
        return token[:-1]
    return token


def _ingredient_tokens(text):
    cleaned = str(text or '').strip().lower()
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


def _build_kitchen_match_context(ingredients, kitchen_version=0):
    kitchen_candidates = []
    for item in ingredients or []:
        display_name = str((item or {}).get('name') or '').strip()
        description = str((item or {}).get('description') or '').strip()
        if not display_name:
            continue
        tokens = _ingredient_tokens(' '.join([display_name, description]).strip())
        kitchen_candidates.append({
            'item_id': None,
            'display_name': display_name,
            'description': description or None,
            'canonical_ingredient': ' '.join(tokens),
            'tokens': tokens,
        })
    return {
        'kitchen_version': int(kitchen_version or 0),
        'kitchen_candidates': kitchen_candidates,
    }


def _score_kitchen_candidate(recipe_tokens, candidate_tokens):
    recipe_set = set(recipe_tokens or [])
    candidate_set = set(candidate_tokens or [])
    if not recipe_set or not candidate_set:
        return -1
    if recipe_set == candidate_set:
        return 300 + len(recipe_set)
    context_tokens = recipe_set | candidate_set
    extra_tokens = candidate_set - recipe_set
    forward_conflicts = _strip_redundant_category_conflicts(
        {t for t in extra_tokens if t in _INGREDIENT_CONFLICT_TOKENS}, context_tokens)
    if recipe_set.issubset(candidate_set) and not forward_conflicts:
        return 200 + (len(recipe_set) * 10) - max(0, len(candidate_set) - len(recipe_set))
    return -1


def _build_recipe_id(section_name, recipe):
    payload = json.dumps({
        'section_name': section_name,
        'title': recipe.get('title') or '',
        'ingredients': recipe.get('ingredients') or [],
    }, sort_keys=True)
    return f"rec_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _best_effort_json_parse(text, fallback):
    cleaned = str(text or '').strip()
    if not cleaned:
        return fallback
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def _parse_json_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return _best_effort_json_parse(value, [])
    return []


def _suggest_substitutions_with_gpt(recipe, kitchen_context, missing_ingredients):
    if not os.getenv('OPENAI_API_KEY'):
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': 'unavailable',
            'can_make_with_subs': False,
        }
    if not missing_ingredients or len(missing_ingredients) > MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS:
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': None,
            'can_make_with_subs': False,
        }
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    kitchen_items = [item.get('display_name') for item in (kitchen_context or {}).get('kitchen_candidates') or [] if item.get('display_name')]
    if not kitchen_items:
        return {
            'substitution_candidates': [],
            'substitution_summary': None,
            'substitution_status': None,
            'can_make_with_subs': False,
        }
    system = """You help users substitute ingredients in recipes using items they already have.
Return valid JSON only with this schema:
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
Only include substitutions that are genuinely plausible in a home kitchen. If no good substitutions exist, return an empty substitutions array and summary as null."""
    user = json.dumps({
        'recipe_title': recipe.get('title'),
        'recipe_ingredients': recipe.get('ingredients') or [],
        'missing_ingredients': missing_ingredients,
        'kitchen_items': kitchen_items,
    })
    _sub_t0 = time.time()
    _sub_input = f"recipe={recipe.get('title')} missing={missing_ingredients}"
    try:
        response = _create_chat(
            client,
            model=SUBSTITUTION_MODEL,
            temperature=0.2,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        )
        payload = _best_effort_json_parse((response.choices[0].message.content or '').strip(), {'substitutions': [], 'summary': None})
        _ai_op('suggest_substitutions', SUBSTITUTION_MODEL, int((time.time() - _sub_t0) * 1000), 'success',
               input_summary=_sub_input,
               output_summary=f"substitutions={len(payload.get('substitutions') or [])}")
    except Exception as exc:
        _ai_op('suggest_substitutions', SUBSTITUTION_MODEL, int((time.time() - _sub_t0) * 1000), 'error',
               input_summary=_sub_input, error=exc)
        print(f"[recipes_generator] Substitution suggestion failed for '{recipe.get('title')}': {exc}")
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
        missing_ingredient = str((item or {}).get('missing_ingredient') or '').strip()
        use_instead = str((item or {}).get('use_instead') or '').strip()
        if not missing_ingredient or not use_instead:
            continue
        covered_missing.add(missing_ingredient.lower())
        normalized.append({
            'missing_ingredient': missing_ingredient,
            'use_instead': use_instead,
            'confidence': str((item or {}).get('confidence') or 'medium').strip().lower() or 'medium',
            'notes': str((item or {}).get('notes') or '').strip() or None,
        })
    return {
        'substitution_candidates': normalized,
        'substitution_summary': payload.get('summary'),
        'substitution_status': 'ready' if normalized else 'none',
        'can_make_with_subs': bool(normalized) and len(covered_missing) >= len({item.lower() for item in missing_ingredients}),
    }


def _log_inventory_match_event(request_id, event_name, **kwargs):
    payload = {
        'request_id': request_id,
        'event': event_name,
    }
    payload.update(kwargs)
    print(json.dumps(payload, default=str))


def _compute_recipe_availability(recipe, kitchen_context, request_id=None):
    recipe_id = str((recipe or {}).get('id') or (recipe or {}).get('_id') or 'inline-recipe').strip()
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        [{
            'id': recipe_id,
            'title': str((recipe or {}).get('title') or '').strip(),
            'ingredients': [str(item).strip() for item in ((recipe or {}).get('ingredients') or []) if str(item).strip()],
        }],
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label='recipes_generator',
        substitution_callback=_suggest_substitutions_with_gpt,
    )
    return availability_map.get(recipe_id) or recipe_inventory_llm.deterministic_availability(
        recipe,
        kitchen_context,
        substitution_callback=_suggest_substitutions_with_gpt,
    )


def _compute_recipe_availability_batch(recipes, kitchen_context, request_id=None, source_label='recipes_generator_batch'):
    recipe_payloads = []
    for recipe in recipes or []:
        recipe_id = str((recipe or {}).get('id') or (recipe or {}).get('_id') or '').strip()
        if not recipe_id:
            continue
        recipe_payloads.append({
            'id': recipe_id,
            'title': str((recipe or {}).get('title') or '').strip(),
            'ingredients': [str(item).strip() for item in ((recipe or {}).get('ingredients') or []) if str(item).strip()],
        })
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        recipe_payloads,
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label=source_label,
        substitution_callback=_suggest_substitutions_with_gpt,
    )
    return availability_map


def _recipe_can_survive(recipe, section_name):
    availability = recipe.get('availability') or {}
    if section_name == 'kitchen_only':
        return bool(availability.get('can_make_exact') or availability.get('can_make_with_subs'))
    return bool(availability.get('matched_count') or availability.get('can_make_with_subs'))


def _normalize_existing_recipe(recipe, section_name, kitchen_context, fallback_index=0, availability=None, request_id=None):
    base = _build_recipe_record(recipe, fallback_index=fallback_index)
    base['id'] = (recipe or {}).get('id') or _build_recipe_id(section_name, base)
    availability = availability or _compute_recipe_availability(base, kitchen_context, request_id=request_id)
    base['availability'] = {
        'kitchen_version': availability.get('kitchen_version'),
        'can_make_exact': bool(availability.get('can_make_exact')),
        'can_make_with_subs': bool(availability.get('can_make_with_subs')),
        'matched_count': int(availability.get('matched_count') or 0),
        'missing_count': int(availability.get('missing_count') or 0),
    }
    base['ingredient_matches'] = availability.get('ingredient_matches') or []
    base['substitution_candidates'] = availability.get('substitution_candidates') or []
    base['substitution_summary'] = availability.get('substitution_summary')
    base['substitution_status'] = availability.get('substitution_status')
    base['generation_version'] = int((recipe or {}).get('generation_version') or availability.get('kitchen_version') or 0)
    return base


def _normalize_recipe_batch(recipes, section_name, kitchen_context, request_id=None, start_index=0, source_label='recipes_generator_batch'):
    bases = []
    for index, recipe in enumerate(recipes or []):
        base = _build_recipe_record(recipe, fallback_index=start_index + index)
        base['id'] = (recipe or {}).get('id') or _build_recipe_id(section_name, base)
        bases.append(base)
    availability_map = _compute_recipe_availability_batch(
        bases,
        kitchen_context,
        request_id=request_id,
        source_label=source_label,
    )
    return [
        _normalize_existing_recipe(
            dict(recipe or {}, id=base.get('id')),
            section_name,
            kitchen_context,
            fallback_index=start_index + index,
            availability=availability_map.get(base.get('id')),
            request_id=request_id,
        )
        for index, (recipe, base) in enumerate(zip(recipes or [], bases))
    ]


def _dedupe_recipes(recipes):
    seen = set()
    output = []
    for recipe in recipes or []:
        dedupe_key = (
            str(recipe.get('title') or '').strip().lower(),
            tuple(str(item).strip().lower() for item in (recipe.get('ingredients') or [])),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        output.append(recipe)
    return output


def _normalize_recipe_meal_category(value):
    normalized = str(value or '').strip().lower()
    if normalized in RECIPE_MEAL_CATEGORIES:
        return normalized
    return None


def _infer_recipe_meal_category(title, ingredients, fallback_index=0):
    text_parts = [str(title or '').lower()]
    text_parts.extend(str(item or '').lower() for item in (ingredients or []))
    haystack = ' '.join(text_parts)

    keyword_groups = [
        ('breakfast', ('breakfast', 'omelet', 'omelette', 'oatmeal', 'pancake', 'waffle', 'bagel', 'toast', 'cereal', 'granola', 'yogurt', 'egg')),
        ('lunch', ('lunch', 'sandwich', 'wrap', 'salad', 'soup', 'quesadilla', 'taco', 'bowl')),
        ('snacks', ('snack', 'snacks', 'dip', 'bites', 'bar', 'trail mix', 'parfait', 'popcorn', 'chips', 'cookies', 'cracker')),
        ('dinner', ('dinner', 'pasta', 'curry', 'stir-fry', 'stir fry', 'skillet', 'roast', 'casserole', 'rice bowl', 'burger')),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in haystack for keyword in keywords):
            return category
    return RECIPE_MEAL_CATEGORIES[fallback_index % len(RECIPE_MEAL_CATEGORIES)]


def _recipe_meal_category(recipe, fallback_index=0):
    normalized = _normalize_recipe_meal_category(
        recipe.get('meal_category') or recipe.get('category')
    )
    if normalized:
        return normalized
    return _infer_recipe_meal_category(
        recipe.get('title'),
        recipe.get('ingredients') or [],
        fallback_index=fallback_index,
    )


# --- Dietary preferences (household-scoped hard constraints on generation) ----------------
_DIETARY_PREFS_TABLE = 'user_dietary_preferences'
_DIETARY_PREF_KEYS = ('allergies', 'diets', 'religious', 'health', 'custom')


def _resolve_household_owner_id(conn, owner):
    """Map an acting identity (a member's user_id) to the shared HOUSEHOLD owner_id that keys
    the ONE dietary-prefs record, so every household member's generation reads the same prefs
    no matter who set them. Falls back to `owner` itself for a solo user (not in new_users) or
    an id already at the owner_id level — so the single-user case is identity-preserving."""
    safe = _sanitize_user_id(owner)
    if not safe:
        return safe
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id FROM new_users WHERE user_id = %s LIMIT 1", [safe])
            row = cur.fetchone() or {}
        hh = row.get('owner_id') or row.get('OWNER_ID')
        return _sanitize_user_id(hh) if hh else safe
    except Exception:
        return safe


def _get_user_preferences(conn, owner):
    """Return {allergies, diets, religious, health, custom} (string lists) for the household
    owner; all-empty if there is no row / the table is absent. Best-effort — never raises, so a
    prefs read can't break recipe generation (no prefs = current behavior)."""
    prefs = {k: [] for k in _DIETARY_PREF_KEYS}
    if not owner:
        return prefs
    try:
        household_owner = _resolve_household_owner_id(conn, owner)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT allergies, diets, religious, health, custom "
                f"FROM `{_DIETARY_PREFS_TABLE}` WHERE owner_id = %s LIMIT 1", [household_owner])
            row = cur.fetchone()
        if not row:
            return prefs
        for k in _DIETARY_PREF_KEYS:
            v = row.get(k)
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = []
            prefs[k] = [str(x).strip() for x in (v or []) if str(x).strip()]
    except Exception as e:
        print(json.dumps({'evt': 'dietary_prefs_read_failed', 'owner': str(owner),
                          'error': str(e)[:200]}))
    return prefs


# ---------------------------------------------------------------------------
# Deterministic post-generation dietary SCRUBBER
# ---------------------------------------------------------------------------
# Battle-tested ingredient detectors ported from the taxonomy stress-test
# (confidence_sweep). Each detector takes a lowercased text blob (title +
# ingredient strings) and returns the offending token, or None. Plant/vegan
# qualifiers are stripped first so e.g. "peanut butter", "almond milk",
# "vegan cheese", "rice noodle", "corn tortilla", "tamari" do NOT false-fire.
# This is a DROP-only v1: recipes that violate a HARD exclusion are removed
# (not regenerated). Hard-exclusion violation rates are low, so drops are rare.

DIET_POST_FILTER_ENABLED = os.getenv('DIET_POST_FILTER_ENABLED', 'true').strip().lower() != 'false'
# Spare recipes to request per section for users WITH dietary preferences, because the scrub
# runs after generation and every drop would otherwise leave the set short of ten. Measured
# drop rate on real runs was 9-12 recipes per full regeneration across both sections.
RECIPE_DIET_HEADROOM = int(os.getenv('RECIPE_DIET_HEADROOM', '5'))
# Recipes in the fast first batch, published before the full set is generated. Small enough
# to come back quickly, big enough to fill the screen.
RECIPE_FIRST_BATCH = int(os.getenv('RECIPE_FIRST_BATCH', '4'))

_PLANT_BUTTER = r"(peanut|almond|cashew|sunflower|seed|nut|soy|plant|vegan|coconut)\s*butter"
_PLANT_MILK = r"(coconut|almond|oat|soy|cashew|rice|hemp|plant|non-?dairy|nut)\s*milk"


def _scrub_dairy(b):
    b = re.sub(_PLANT_BUTTER, "", b)
    b = re.sub(_PLANT_MILK, "", b)
    for kill in ("vegan cheese", "dairy-free cheese", "vegan yogurt", "vegan butter",
                 "dairy-free", "plant-based cheese"):
        b = b.replace(kill, "")
    for w in ["cheese", "cheddar", "mozzarella", "parmesan", "parmigiano", "pecorino",
              "feta", "ricotta", "mascarpone", "halloumi", "paneer", "burrata", "brie",
              "gruyere", "gouda", "yogurt", "yoghurt", "ghee", "whey", "casein", "custard",
              "creme fraiche", "crème fraîche", "clotted cream", "condensed milk",
              "evaporated milk", "heavy cream", "double cream", "sour cream",
              "cream cheese", "ice cream", "kefir", "quark", "curd", " milk", " butter",
              "buttermilk"]:
        if w in b:
            return w.strip()
    return None


def _scrub_egg(b):
    return "egg" if (re.search(r"\begg", b) and "eggplant" not in b) else None


def _scrub_meat(b):
    for w in ["chicken", "beef", "bacon", " pork", "sausage", "turkey", "lamb", "steak",
              "ham ", "salami", "pepperoni", "chorizo", "meatball", "prosciutto", "veal"]:
        if w in b:
            return w.strip()
    return None


def _scrub_seafood(b):
    for w in ["shrimp", "salmon", "fish", "tuna", "cod", "crab", "lobster", "clam", "oyster",
              "anchovy", "tilapia", "scallop", "mussel", "sardine", "prawn", "halibut"]:
        if w in b:
            return w
    return None


def _scrub_fish(b):
    for w in ["salmon", "tuna", "cod", "tilapia", "halibut", "anchovy", "sardine", "trout",
              "bass", "mackerel", "fish"]:
        if w in b:
            return w
    return None


def _scrub_shellfish(b):
    for w in ["shrimp", "crab", "lobster", "clam", "oyster", "mussel", "scallop", "prawn", "crawfish"]:
        if w in b:
            return w
    return None


_ALLERGEN_NEGATIONS = (
    "peanut-free", "peanut free", "nut-free", "nut free", "dairy-free", "dairy free",
    "gluten-free", "gluten free", "egg-free", "egg free", "soy-free", "soy free",
    "sesame-free", "sesame free", "wheat-free", "wheat free", "no peanuts", "no nuts",
)


def _strip_allergen_negations(b):
    """Remove 'peanut-free'/'nut free'/etc BEFORE matching, so a label advertising the
    ABSENCE of an allergen is not read as its presence. Without this, 'peanut-free
    chocolate' is flagged as containing peanuts and a safe recipe is hidden."""
    for phrase in _ALLERGEN_NEGATIONS:
        b = b.replace(phrase, " ")
    return b


def _scrub_peanut(b):
    # "peanut" alone missed the names peanuts actually travel under: groundnut (UK/India/
    # Africa), arachis (the botanical name used on labels and in oils), satay and its
    # spellings, and monkey nut. Every one of these is a real ingredient line on real
    # recipe sites, and each is a hospital visit for an allergic user.
    b = _strip_allergen_negations(b)
    for w in ("peanut", "groundnut", "ground nut", "arachis", "satay", "sate sauce",
              "monkey nut", "beer nut", "goober"):
        if w in b:
            return w
    return None


def _scrub_treenut(b):
    # Beyond the bare nut names, tree nuts hide inside prepared ingredients that never
    # say "nut": marzipan and frangipane are almond, praline/nougat/gianduja/nutella are
    # hazelnut, amaretto is apricot-kernel/almond, and pesto is pine nut unless it
    # explicitly says otherwise. These are the forms that reach a recipe page.
    b = _strip_allergen_negations(b)
    # A peanut is a legume, not a tree nut. Remove peanut forms before the generic
    # "nut butter"/"nut flour" sweep, or "Spicy Peanut Butter Noodles" is withheld from
    # someone whose only allergy is tree nuts. Peanuts remain fully covered by
    # _scrub_peanut for anyone who declares that allergy.
    b = b.replace("peanut butter", " ").replace("peanut flour", " ").replace("peanut", " ")
    for w in ("almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut", "macadamia",
              "brazil nut", "pine nut", "pignoli", "marzipan", "frangipane", "praline",
              "nougat", "gianduja", "nutella", "amaretto", "amaretti", "filbert",
              "chestnut", "nut butter", "nut flour", "nut meal"):
        if w in b:
            return w
    # Pesto is pine nut by default. A pesto that names a different nut is caught above;
    # one that says nut-free has already been stripped by the negation pass.
    if "pesto" in b:
        return "pesto(pine nut)"
    return None


def _scrub_sesame(b):
    for w in ["sesame", "tahini"]:
        if w in b:
            return w
    return None


def _scrub_soy(b):
    b = b.replace("soy-free", "")
    for w in ["soy sauce", "soybean", " soy ", "tofu", "edamame", "tempeh", "miso"]:
        if w in b:
            return w.strip()
    return None


def _scrub_wheat(b):
    hits = []
    for w in ["wheat", "bread", "flour", "spaghetti", "couscous", "breadcrumb", "cracker", " bun "]:
        if w in b and "gluten-free" not in b:
            hits.append(w.strip())
    if "pasta" in b and "gluten-free" not in b and "rice pasta" not in b:
        hits.append("pasta")
    if "noodle" in b and "rice noodle" not in b and "gluten-free" not in b:
        hits.append("noodle")
    if "tortilla" in b and "corn tortilla" not in b:
        hits.append("tortilla")
    return hits[0] if hits else None


def _scrub_gluten(b):
    # "noodle" alone is not gluten: zucchini noodles, rice noodles, glass/mung-bean
    # noodles and shirataki are all gluten-free staples of exactly the recipes a
    # gluten-free user wants. Measured: it silently dropped "Avocado Pesto Zucchini
    # Noodles" from a gluten-free feed. Remove the gluten-free noodle forms before the
    # sweep, the same way plant milks are removed before the dairy sweep.
    for gf in ("zucchini noodle", "zoodle", "rice noodle", "glass noodle", "shirataki",
               "sweet potato noodle", "kelp noodle", "veggie noodle", "vegetable noodle",
               "chickpea noodle", "lentil noodle", "buckwheat noodle", "gluten-free noodle",
               "gluten free noodle", "gluten-free pasta", "gluten free pasta",
               "chickpea pasta", "lentil pasta", "rice pasta", "corn pasta"):
        b = b.replace(gf, " ")
    w = _scrub_wheat(b)
    if w:
        return w
    for x in ["barley", "rye", "malt", "seitan", "farro", "bulgur"]:
        if x in b:
            return x
    if "soy sauce" in b and "tamari" not in b and "gluten-free soy" not in b:
        return "soy sauce(wheat)"
    return None


def _scrub_honey(b):
    return "honey" if "honey" in b else None


def _scrub_pork(b):
    for w in ["pork", "bacon", "ham ", "prosciutto", "pancetta", "lard", "gelatin"]:
        if w in b:
            return w.strip()
    return None


def _scrub_alcohol(b):
    for w in ["wine", "beer", " rum", "vodka", "bourbon", "sake", "mirin", "sherry",
              "brandy", "liqueur", "whiskey"]:
        if w in b:
            return w.strip()
    return None


# Allergen display-name -> detector. Keys are matched case-insensitively and by
# substring so "Milk/Dairy", "Tree nuts", etc. resolve.
_ALLERGEN_DETECTORS = [
    ('dairy', _scrub_dairy), ('milk', _scrub_dairy),
    ('egg', _scrub_egg),
    ('peanut', _scrub_peanut),
    ('tree nut', _scrub_treenut), ('treenut', _scrub_treenut),
    ('shellfish', _scrub_shellfish),
    ('fish', _scrub_fish),
    ('wheat', _scrub_wheat),
    ('soy', _scrub_soy),
    ('sesame', _scrub_sesame),
    ('gluten', _scrub_gluten),
]

# Exclusion diets -> list of detectors that must all pass. Quantitative diets
# (keto/low-carb/paleo/etc) have NO deterministic check and are skipped here
# (they stay prompt-only best-effort).
_DIET_DETECTORS = {
    'vegetarian': [_scrub_meat, _scrub_seafood],
    'vegan': [_scrub_meat, _scrub_seafood, _scrub_dairy, _scrub_egg, _scrub_honey],
    'pescatarian': [_scrub_meat],
    'dairy-free': [_scrub_dairy],
    'dairy free': [_scrub_dairy],
    'gluten-free': [_scrub_gluten],
    'gluten free': [_scrub_gluten],
}


def _recipe_blob(recipe):
    """Lowercased title + ingredients + missing_ingredients (+ UWIH subtitle and
    detail.matched/missing) for a recipe dict. Robust to missing keys."""
    parts = [str(recipe.get('title', '') or ''), str(recipe.get('name', '') or ''),
             str(recipe.get('subtitle', '') or '')]
    for key in ('ingredients', 'missing_ingredients', 'matched_ingredients'):
        v = recipe.get(key)
        if isinstance(v, list):
            parts += [str(x) for x in v]
    detail = recipe.get('detail')
    if isinstance(detail, dict):
        for key in ('matched_ingredients', 'missing_ingredients'):
            v = detail.get(key)
            if isinstance(v, list):
                parts += [str(x) for x in v]
    return ' '.join(parts).lower()


def _recipe_violates(recipe, prefs):
    """Return a reason string if the recipe violates any HARD dietary exclusion in
    prefs, else None. prefs = {allergies, diets, religious, health, custom}."""
    b = _recipe_blob(recipe)
    prefs = prefs or {}

    # Allergies (safety-critical hard exclusions)
    for name in prefs.get('allergies', []) or []:
        low = str(name).strip().lower()
        for token, fn in _ALLERGEN_DETECTORS:
            if token in low:
                hit = fn(b)
                if hit:
                    return f'allergy:{name}={hit}'

    # Exclusion diets
    for name in prefs.get('diets', []) or []:
        low = str(name).strip().lower()
        for diet_key, fns in _DIET_DETECTORS.items():
            if diet_key == low or diet_key in low.replace(' ', '-'):
                for fn in fns:
                    hit = fn(b)
                    if hit:
                        return f'diet:{name}={hit}'
                break

    # Religious rules
    for name in prefs.get('religious', []) or []:
        low = str(name).strip().lower()
        if 'kosher' in low:
            p = _scrub_pork(b)
            if p:
                return f'kosher:pork={p}'
            s = _scrub_shellfish(b)
            if s:
                return f'kosher:shellfish={s}'
            m = _scrub_meat(b)
            d = _scrub_dairy(b)
            if m and d:
                return f'kosher:meat+dairy={m}+{d}'
        if 'halal' in low:
            p = _scrub_pork(b)
            if p:
                return f'halal:pork={p}'
            a = _scrub_alcohol(b)
            if a:
                return f'halal:alcohol={a}'

    # Custom "no X" / "avoid X" / "no more X" -> substring match on X
    for entry in prefs.get('custom', []) or []:
        term = str(entry).strip().lower()
        for prefix in ('no more ', 'avoid ', 'no '):
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                break
        if term and term in b:
            return f'custom:{entry}={term}'

    # Health entries are soft ("tailor toward"), not hard exclusions -> skip.
    return None


def _scrub_recipes(recipes, prefs, owner=None):
    """Return recipes with any HARD-exclusion violator dropped. No-op when prefs
    are empty or the feature flag is off (byte-unchanged behavior)."""
    if not recipes:
        return recipes
    if not DIET_POST_FILTER_ENABLED:
        return recipes
    prefs = prefs or {}
    if not any(prefs.get(k) for k in _DIETARY_PREF_KEYS):
        return recipes
    kept = []
    for recipe in recipes:
        if not isinstance(recipe, dict):
            kept.append(recipe)
            continue
        reason = _recipe_violates(recipe, prefs)
        if reason:
            print(json.dumps({'evt': 'diet_scrub_dropped', 'owner': str(owner),
                              'title': str(recipe.get('title') or recipe.get('name') or ''),
                              'reason': reason}))
            continue
        kept.append(recipe)
    return kept


def _format_preferences_block(prefs):
    """Render the DIETARY CONSTRAINTS prompt block. Empty categories are omitted; returns ''
    when there are NO preferences at all (so the prompt is byte-for-byte unchanged)."""
    prefs = prefs or {}
    if not any(prefs.get(k) for k in _DIETARY_PREF_KEYS):
        return ''
    lines = ['DIETARY CONSTRAINTS (every recipe MUST comply):']
    if prefs.get('allergies'):
        lines.append('- ALLERGIES — NEVER include these or any derivative/trace '
                     '(safety-critical): ' + ', '.join(prefs['allergies']))
    if prefs.get('diets'):
        lines.append('- DIET — recipes MUST be: ' + ', '.join(prefs['diets']))
    if prefs.get('religious'):
        lines.append('- RELIGIOUS: ' + ', '.join(prefs['religious']))
        _religious_low = ' '.join(str(r).lower() for r in prefs['religious'])
        if 'kosher' in _religious_low:
            lines.append('  For KOSHER: no pork or shellfish, and never combine meat and '
                         'dairy in the same recipe.')
        if 'halal' in _religious_low:
            lines.append('  For HALAL: no pork or pork derivatives (bacon, ham, lard, '
                         'gelatin) and no alcohol including wine, beer, or cooking wine.')
    if prefs.get('health'):
        lines.append('- HEALTH — tailor toward (informational, not medical advice): '
                     + ', '.join(prefs['health']))
    if prefs.get('custom'):
        lines.append("- ALSO AVOID/HONOR (user's own words): " + ', '.join(prefs['custom']))
    lines.append(
        'INGREDIENT-LEVEL CHECK: verify EVERY ingredient in each recipe against the '
        'constraints above, including staples pulled from the kitchen. Do not assume a '
        'staple is compliant just because it is common. Butter, milk, cream, cheese, '
        'yogurt, and eggs are NOT vegan or dairy-free (use plant milk, vegan butter, or '
        'omit them); honey is not vegan; regular soy sauce, teriyaki, most bread, pasta, '
        'flour, and breadcrumbs contain gluten (use tamari or certified gluten-free '
        'versions). Substitute any non-compliant ingredient with a compliant alternative '
        'or leave it out.')
    lines.append('Any recipe that violates an ALLERGY or DIET is unacceptable — omit it '
                 'entirely and generate a compliant one instead.')
    return '\n'.join(lines)


def _generate_recipes_with_gpt(ingredients_list, kitchen_only_count=10, need_grocery_count=10, excluded_titles=None, owner=None, user_preferences_block="", prefer_fast_model=False):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=RECIPE_GEN_TIMEOUT_SECONDS,
        max_retries=RECIPE_GEN_MAX_RETRIES,
    )
    kitchen_only_count = max(0, int(kitchen_only_count or 0))
    need_grocery_count = max(0, int(need_grocery_count or 0))
    if kitchen_only_count == 0 and need_grocery_count == 0:
        return {'kitchen_only': [], 'need_grocery': []}
    # Cap the DETAILED generation context to the most-recent items so a huge
    # kitchen can't blow the OpenAI timeout. ingredients_list is already
    # _createdDate DESC. Items beyond the detailed head are still surfaced as a
    # compact name-only tail so OLDER items remain reachable in recipes (a 200+
    # item kitchen previously lost everything past the cap). Matching still uses
    # the full kitchen regardless.
    full_ingredient_count = len(ingredients_list or [])
    all_items = list(ingredients_list or [])
    detailed_items = all_items[:RECIPE_GEN_MAX_INGREDIENTS]
    tail_items = all_items[RECIPE_GEN_MAX_INGREDIENTS:RECIPE_GEN_MAX_TOTAL]
    if full_ingredient_count > RECIPE_GEN_MAX_INGREDIENTS:
        print(f"[recipes_generator] Kitchen context: {len(detailed_items)} detailed + {len(tail_items)} compact (of {full_ingredient_count} total)")
    ingredients_list = detailed_items
    ingredients_str = _format_kitchen_for_prompt(detailed_items) if detailed_items else 'No specific ingredients (suggest pantry staples)'
    # Append a compact name-only list of the remaining (older) items so they are
    # still usable, without paying the descriptive-format token cost for all of
    # them.
    tail_names = [str((it or {}).get('name') or '').strip() for it in tail_items]
    tail_names = [name for name in tail_names if name]
    if tail_names:
        ingredients_str = (
            f"{ingredients_str}\n\n"
            f"Additional kitchen items (also available; listed by name only): "
            + ', '.join(tail_names)
        )
    excluded_titles = [str(item).strip() for item in (excluded_titles or []) if str(item).strip()]
    exclusion_text = ''
    if excluded_titles:
        exclusion_text = '\nAvoid recreating these existing recipe titles or close variants: ' + ', '.join(excluded_titles[:30])
    system = """You are a recipe assistant. Return valid JSON only, no markdown.
Output schema:
{
  "kitchen_only": [ exactly the requested number of recipes. Each: { "title": "Recipe name", "emoji": "🍝", "meal_category": "breakfast|lunch|dinner|snacks", "ingredients": ["1 lb Chicken Breasts", "2 cups White Rice"], "steps": ["Step 1.", "Step 2."] }. Every ingredient string MUST begin with a realistic amount/measurement, then the item name. Use ONLY the provided kitchen list of grocery products (keep each item's EXACT product name after the amount, e.g. "1 lb Chicken Breasts", so it can be matched) plus pantry staples (salt, pepper, oil, water, basic spices), each with an amount (e.g. "1 tsp salt", "2 tbsp olive oil"). ],
  "need_grocery": [ exactly the requested number of recipes. Each: { "title": "Recipe name", "emoji": "🌮", "meal_category": "breakfast|lunch|dinner|snacks", "ingredients": ["1 lb Chicken Breasts", "1 cup Shredded Cheese", ...], "steps": ["Step 1.", "Step 2."], "missing_ingredients": ["item to buy 1", "item to buy 2"] }. Every ingredient string begins with a realistic amount/measurement followed by the item name (keep kitchen items' exact product name after the amount). Each recipe should use some grocery products from the kitchen list (reference by exact product name) but require at least one additional ingredient the user must buy. List those in missing_ingredients. ]
}
Each kitchen item may include a short description in parentheses (e.g. "Ice Cubes Gum (chewing gum, not edible ice)"). Use that to avoid misuse: do NOT suggest recipes that treat a product as something it is not (e.g. do not use gum as ice in drinks). Reference items by their exact product name in the ingredients array. Keep titles short. Steps concise.

Coherence rules:
- Recipes must be normal, common-sense dishes. Avoid odd pairings or contrived combos.
- Do NOT use beverage items (sparkling water, soda, seltzer, etc.) as cooking liquids or cereal bases.
- Recipes should be FOOD dishes, not beverages. If a beverage item exists, it can be omitted.
- Every recipe must include exactly one meal_category value chosen from: breakfast, lunch, dinner, snacks.
- Every recipe must include an "emoji" field: exactly ONE emoji that best represents the finished dish (e.g. 🍝 pasta, 🌮 tacos, 🥞 pancakes, 🍜 stir-fry/noodles, 🥗 salad, 🍲 soup/stew, 🍳 eggs, 🥪 sandwich). It MUST be a single food or dish emoji — never a flag, letter, number, punctuation, symbol, or more than one emoji.
- If a recipe cannot be made coherently with the available items, choose a different combination of available items (still only from the list + pantry).

Meal substantiality rules (IMPORTANT):
- "dinner" and "lunch" recipes MUST be substantial, complete meals — the kind you would sit down and eat as a full plate. Think: pasta dishes, stir-fries, casseroles, grain bowls, tacos, soups, curries, sandwiches, salads with protein, etc.
- Do NOT categorize simple condiments, toppings, sauces, dips, relishes, or 1-2 ingredient combinations as "dinner" or "lunch". A recipe that is just mixing two condiments together (e.g. BBQ sauce + jalapenos) is NOT a meal.
- Simple items like dips, salsas, toppings, and spreads should be categorized as "snacks" if included at all.
- Each dinner/lunch recipe should have at least 3-4 meaningful ingredients (beyond pantry staples) and involve actual cooking or assembly of a complete dish.
- Prioritize recipes people actually make at home — classics, weeknight staples, popular cuisines. Avoid contrived combinations just to use up inventory.

Measurement rules (IMPORTANT):
- EVERY entry in "ingredients" MUST include a realistic quantity/measurement (e.g. "2 cups", "1 lb", "3 cloves", "1/2 tsp", "1 (14 oz) can") placed BEFORE the item name. Never output a bare ingredient with no amount.
- Keep the kitchen product's EXACT name immediately after the amount so it still matches the user's inventory (kitchen item "Chicken Breasts" -> "1 lb Chicken Breasts", not "1 lb chicken").
- Scale amounts sensibly for about 2 servings unless the dish implies otherwise, and make the steps reference those amounts naturally (e.g. "Add the 2 cups White Rice ...").
- "missing_ingredients" stays a plain list of item names to buy (no amount required there)."""
    # Inject household dietary constraints (allergies as HARD exclusions, diets/health as
    # requirements) right before the closing instruction. Empty when the owner has no prefs,
    # so the prompt is unchanged for them.
    if user_preferences_block:
        system += "\n\n" + user_preferences_block
    system += "\nReturn the JSON object only."
    user = (
        f"Kitchen grocery products (use exact product name in recipe ingredients; descriptions in parentheses clarify what each item is): {ingredients_str}\n\n"
        f"Generate exactly {kitchen_only_count} kitchen_only recipes using ONLY these items (plus pantry), and exactly {need_grocery_count} need_grocery recipes that use some of these items but need extra ingredients to buy (include missing_ingredients for each)."
        f"{exclusion_text}\nReturn the JSON object only."
    )
    _gen_input = f"kitchen_items={full_ingredient_count} kitchen_only={kitchen_only_count} need_grocery={need_grocery_count}"

    def _generate_once(gen_client, gen_model):
        # Fast in-handler retry: a single malformed/blank generation (JSONDecodeError)
        # otherwise fails the invocation and detours through the 4-min refresh-flag class.
        # Retry the generate+parse 1x immediately before giving up.
        t0 = time.time()
        last_err = None
        try:
            for _attempt in range(2):
                resp = _create_chat(
                    gen_client,
                    model=gen_model,
                    messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                    temperature=0.4,
                )
                text = (resp.choices[0].message.content or '').strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                try:
                    parsed = json.loads(text)
                    if _attempt > 0:
                        print(json.dumps({'evt': 'openai_fast_retry_saved', 'service': 'recipes_generator', 'op': 'generate'}))
                    _titles = []
                    for _section in ('kitchen_only', 'need_grocery'):
                        for _r in (parsed.get(_section) or []):
                            _title = (_r or {}).get('title')
                            if _title:
                                _titles.append(_title)
                    _ai_op('generate_recipes', gen_model, int((time.time() - t0) * 1000), 'success',
                           input_summary=_gen_input,
                           output_summary=(f"kitchen_only={len(parsed.get('kitchen_only') or [])} "
                                           f"need_grocery={len(parsed.get('need_grocery') or [])} "
                                           f"titles={_titles[:20]}"),
                           owner_id=owner)
                    return parsed
                except json.JSONDecodeError as e:
                    last_err = e
                    if _attempt == 0:
                        time.sleep(2)
            raise last_err
        except Exception as gen_err:
            _ai_op('generate_recipes', gen_model, int((time.time() - t0) * 1000), 'error',
                   input_summary=_gen_input, error=gen_err, owner_id=owner)
            raise

    # The first batch goes straight to the fast model. The primary times out often enough
    # (~182s before failing over) that paying that cost TWICE — once for the teaser and again
    # for the full set — turned a 121s wait into 287s. Speed is the entire point of the first
    # batch; the full set still gets the primary model's quality.
    _gen_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    if prefer_fast_model and RECIPE_GEN_FALLBACK_MODEL:
        _gen_model = RECIPE_GEN_FALLBACK_MODEL
    try:
        return _generate_once(client, _gen_model)
    except Exception as _primary_err:
        if not RECIPE_GEN_FALLBACK_MODEL or RECIPE_GEN_FALLBACK_MODEL == _gen_model:
            raise
        print(json.dumps({'evt': 'recipes_model_failover', 'service': 'recipes_generator',
                          'from_model': _gen_model, 'to_model': RECIPE_GEN_FALLBACK_MODEL,
                          'primary_error': str(_primary_err)[:200], 'owner_id': owner}))
        fallback_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=RECIPE_GEN_FALLBACK_TIMEOUT_SECONDS,
            max_retries=1,
        )
        return _generate_once(fallback_client, RECIPE_GEN_FALLBACK_MODEL)


# Leading emoji grapheme: one pictographic base (excludes the regional-indicator
# block used for flags, letters, and ASCII symbols) plus any skin-tone modifier /
# variation selector. ZWJ-joined extras are intentionally dropped to the base.
_RECIPE_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
    "[\U0001F3FB-\U0001F3FF️]*"
)


def _clean_recipe_emoji(value):
    """Return a single food/dish emoji grapheme, or None. Keeps only the first
    grapheme when the model returns more than one char, and drops anything that
    isn't in the emoji unicode ranges (flags, letters, numbers, symbols)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _RECIPE_EMOJI_RE.match(text)
    return match.group(0) if match else None


def _build_recipe_record(recipe, fallback_index=0):
    recipe = recipe or {}
    record = dict(recipe)
    record.update({
        'title': recipe.get('title') or 'Recipe',
        'meal_category': _recipe_meal_category(recipe, fallback_index=fallback_index),
        'ingredients': recipe.get('ingredients') or [],
        'steps': recipe.get('steps') or [],
        # One food emoji for the dish; None when absent/invalid (old parses stay valid).
        'emoji': _clean_recipe_emoji(recipe.get('emoji')),
    })
    if 'missing_ingredients' in (recipe or {}):
        record['missing_ingredients'] = recipe.get('missing_ingredients') or []
    return record


def _publish_partial_recipes(conn, target_owners, kitchen_only, need_grocery):
    """Write what is ready so far, keeping status 'regenerating'.

    A full regeneration takes tens of seconds and used to write NOTHING until every recipe was
    finished, so the user watched a loading screen the whole time even though half the set had
    been ready for a while. The client already renders a partial set during 'regenerating';
    it just never received one.

    Best-effort: a failure here must not fail the generation that is still in flight.
    """
    try:
        kitchen_list, need_list = _build_recipes(kitchen_only or [], need_grocery or [])
        if not kitchen_list and not need_list:
            return
        with conn.cursor() as cur:
            for target_owner in target_owners:
                table = _recipes_table(target_owner)
                cur.execute(f"""
                    UPDATE `{table}`
                    SET kitchen_only = %s, need_grocery = %s, _updatedDate = NOW()
                    WHERE _id = 'current'
                """, (json.dumps(kitchen_list), json.dumps(need_list)))
        conn.commit()
        for target_owner in target_owners:
            _dual_write_recipes_to_shared(conn, target_owner)
        print(json.dumps({'evt': 'recipes_partial_published',
                          'kitchen_only': len(kitchen_list), 'need_grocery': len(need_list)}))
    except Exception as exc:
        print(json.dumps({'evt': 'recipes_partial_publish_failed', 'error': str(exc)[:160]}))


def _build_recipes(kitchen_only, need_grocery):
    """Build recipe dicts with normalized meal categories."""
    out_kitchen = []
    for i, r in enumerate((kitchen_only or [])[:10]):
        rec = _build_recipe_record(r, fallback_index=i)
        out_kitchen.append(rec)
    out_need = []
    for i, r in enumerate((need_grocery or [])[:10]):
        rec = _build_recipe_record(r, fallback_index=i)
        out_need.append(rec)
    return out_kitchen, out_need


def _load_current_recipes(conn, owner):
    table = _recipes_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT kitchen_only, need_grocery FROM `{table}` WHERE _id = 'current' LIMIT 1"
        )
        row = cur.fetchone() or {}
    return (
        row.get('kitchen_only'),
        row.get('need_grocery'),
    )


def _sweep_stuck_recipe_owners(context):
    """Scheduled self-heal: re-drive owners stranded with an unfulfilled recipe refresh.

    A crashed/timed-out generator run leaves kitchen_version >
    last_recipe_refresh_completed_version with recipe_refresh_needed=1 and nothing to re-fire
    it (the normal path only re-invokes on the success tail or a fresh kitchen change), so the
    app shows "Updating recipes…" forever. This sweep CLAIMS each stuck owner (stamping
    last_recipe_refresh_started_at so the next tick won't double-fire and a crashed child
    naturally retries after the TTL) and self-invokes the generator with {"owner": id}. It
    NEVER takes the per-owner lock and NEVER generates recipes inline — the real work runs in
    those child invocations, which take the lock, generate, and clear the gap on success via
    _mark_recipe_refresh_complete. Never raises out of handler.
    """
    if not _RECIPE_SWEEP_ENABLED:
        return {'swept': 0, 'disabled': True}
    try:
        conn = _mysql_conn()
        try:
            _ensure_owner_kitchen_state_table(conn)

            # Total stuck population (same condition, minus the claim-TTL/in-flight guard and
            # the batch LIMIT) for observability of the backlog.
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM `{_OWNER_KITCHEN_STATE_TABLE}`
                    WHERE kitchen_version > COALESCE(last_recipe_refresh_completed_version, 0)
                      AND recipe_refresh_needed = 1
                    """
                )
                stuck_total = int((cur.fetchone() or {}).get('n') or 0)

            # Stuck owners not claimed within the TTL (in-flight guard + crash-retry backoff),
            # oldest-completed first so the most-stale owners drain first.
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT owner, kitchen_version FROM `{_OWNER_KITCHEN_STATE_TABLE}`
                    WHERE kitchen_version > COALESCE(last_recipe_refresh_completed_version, 0)
                      AND recipe_refresh_needed = 1
                      AND (last_recipe_refresh_started_at IS NULL
                           OR last_recipe_refresh_started_at < (UTC_TIMESTAMP() - INTERVAL %s MINUTE))
                    ORDER BY last_recipe_refresh_completed_at ASC
                    LIMIT %s
                    """,
                    [_RECIPE_SWEEP_CLAIM_TTL_MIN, _RECIPE_SWEEP_BATCH],
                )
                rows = cur.fetchall() or []

            import boto3
            lambda_client = boto3.client('lambda')
            arn = context.invoked_function_arn
            swept = 0
            for row in rows:
                owner = row.get('owner')
                if not owner:
                    continue
                try:
                    # CLAIM first, then invoke — so a failure between claim and invoke still
                    # backs the owner off for the TTL (retried on a later tick), and a
                    # concurrent sweep tick won't double-fire this owner.
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            UPDATE `{_OWNER_KITCHEN_STATE_TABLE}`
                            SET last_recipe_refresh_started_at = UTC_TIMESTAMP(),
                                last_recipe_refresh_requested_version = kitchen_version
                            WHERE owner = %s
                            """,
                            [_sanitize_user_id(owner)],
                        )
                    conn.commit()
                    lambda_client.invoke(
                        FunctionName=arn,
                        InvocationType='Event',
                        Payload=json.dumps({'owner': owner}).encode('utf-8'),
                    )
                    swept += 1
                except Exception as owner_err:
                    _report_backend_error('recipe_sweep', owner_id=owner, code='claim_invoke_failed', error=owner_err)

            # Structured marker for backlog + throughput observability.
            print(json.dumps({
                'evt': 'recipe_refresh_sweep',
                'service': 'recipes',
                'swept': swept,
                'stuck_total': stuck_total,
                'batch': _RECIPE_SWEEP_BATCH,
            }))
            return {'swept': swept, 'stuck_total': stuck_total}
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        _report_backend_error('recipe_sweep', code='sweep_failed', error=e)
        return {'swept': 0, 'error': 'sweep_failed'}


# ============================================================================
# Personalised Explore — agent-sourced, dietary-safe
# ============================================================================
# Explore is a single global table of 337 creator recipes, identical for everyone and
# blind to dietary preferences. Filtering that pool is not enough: measured against the
# production filter, a vegan+gluten-free user has only 90 of 337 left, and those 90
# merely SURVIVED rather than being chosen for them.
#
# So this builds a per-user Explore instead. The agent proposes REAL recipe URLs from
# well-known sites; each is then fetched and parsed by the existing, proven extractor
# behind POST /analyze-url. That split matters: **the model suggests, the extractor
# verifies.** A hallucinated URL simply fails to extract and is dropped, so no invented
# recipe can reach a user — the recipe text always comes from the real page.
#
# Every surviving recipe then passes the SAME deterministic dietary filter used for
# UseWhatIHave (_recipe_violates), applied AFTER generation regardless of what the model
# was told. With agent sourcing this filter is the only line of defence, which is why it
# was hardened first. Fail-closed: if filtering leaves few, we show few, never unsafe.

_EXPLORE_PERSONAL_TABLE = 'shared_explore_personal'
EXPLORE_AGENT_TARGET = int(os.getenv('EXPLORE_AGENT_TARGET', '15'))
EXPLORE_AGENT_CANDIDATES = int(os.getenv('EXPLORE_AGENT_CANDIDATES', '20'))
# 15 is a hard requirement, and a single pass cannot guarantee it: the funnel loses ~55%
# at extraction (sites with no structured recipe data) and more again at the safety
# filter. So the agent runs ROUNDS until it has 15, telling the model which dishes it has
# already used so each round brings new ones, and stopping on the job deadline.
EXPLORE_AGENT_MAX_ROUNDS = int(os.getenv('EXPLORE_AGENT_MAX_ROUNDS', '6'))
EXPLORE_AGENT_FIRST_ROUND = int(os.getenv('EXPLORE_AGENT_FIRST_ROUND', '6'))
EXPLORE_AGENT_WORKERS = int(os.getenv('EXPLORE_AGENT_WORKERS', '10'))
# Suggesting known recipe URLs is recall, not reasoning. The default OPENAI_MODEL here is
# gpt-5.5, which measured 138-216s for this prompt and left nothing of the Lambda's 420s
# for the extraction pass. The cheap structured model this app already uses elsewhere does
# the same job in seconds.
EXPLORE_AGENT_MODEL = os.getenv('EXPLORE_AGENT_MODEL',
                                os.getenv('OPENAI_INVENTORY_MODEL', 'gpt-4.1-mini'))
# Whole-job deadline. The extraction fan-out must never run the function into its own
# ceiling: past this we stop collecting and store what we already have, rather than dying
# with nothing. Same principle as the recipe ladder's deadline budgeting.
EXPLORE_AGENT_DEADLINE_SECONDS = float(os.getenv('EXPLORE_AGENT_DEADLINE_SECONDS', '330'))
# A build holding the claim longer than this is treated as dead so the next trigger can take
# over. Comfortably above the agent deadline plus the Lambda's own ceiling, so a slow-but-alive
# build is never stolen from underneath itself.
_EXPLORE_BUILD_STALE_SECONDS = float(os.getenv('EXPLORE_BUILD_STALE_SECONDS', '480'))
EXPLORE_ANALYZE_URL = os.getenv(
    'EXPLORE_ANALYZE_URL',
    'https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com/analyze-url')


def _prefs_hash(prefs):
    """Stable fingerprint of a preference set, so we only rebuild when it actually
    changes — not on every save of the same values."""
    norm = {k: sorted([str(x).strip().lower() for x in (prefs.get(k) or []) if str(x).strip()])
            for k in _DIETARY_PREF_KEYS}
    return hashlib.md5(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:32]


def _ensure_explore_personal_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_EXPLORE_PERSONAL_TABLE}` (
              `_id` VARCHAR(64) NOT NULL, `owner_id` VARCHAR(64) NOT NULL,
              `prefs_hash` VARCHAR(32) NOT NULL, `rank` INT NOT NULL DEFAULT 0,
              `title` VARCHAR(255) NOT NULL, `image_url` VARCHAR(1000) DEFAULT NULL,
              `source_url` VARCHAR(1000) DEFAULT NULL, `source_domain` VARCHAR(255) DEFAULT NULL,
              `ingredients` JSON DEFAULT NULL, `instructions` JSON DEFAULT NULL,
              `notes` JSON DEFAULT NULL, `meal_category` VARCHAR(32) DEFAULT NULL,
              `why` VARCHAR(255) DEFAULT NULL,
              `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`_id`), KEY `idx_owner_rank` (`owner_id`,`rank`),
              KEY `idx_owner_prefs` (`owner_id`,`prefs_hash`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    conn.commit()


# Search results for a broad query ("vegan gluten free dinner recipe") are dominated by
# roundups and category pages, which carry no single ingredient list. Individual recipe
# pages come from searching a SPECIFIC DISH. So the work is split by what each side is
# actually good at: the model names dishes, the search engine finds the real page.
_EXPLORE_ROUNDUP_RE = re.compile(
    r'^\d+\s|\b\d+\s+(of\s+)?(the\s+)?(best|favou?rite|incredible|easy|amazing|top)\b'
    r'|/recipes/?$|/category/|/categories/|/tag/|/collections?/|reddit\.com|pinterest\.'
    r'|youtube\.com|facebook\.com|/search', re.I)
# Brave's free tier allows roughly one query per second; anything faster returns 429.
BRAVE_QPS_DELAY = float(os.getenv('BRAVE_QPS_DELAY_SECONDS', '1.15'))


def _diet_query_terms(prefs):
    """The restriction words to put INTO the search query. Searching a dish name alone
    returns the ordinary version of that dish, so a vegan feed fills with recipes that
    then fail the safety filter — one run dropped 10 of 12 that way. Putting the diet in
    the query makes the search engine do the filtering first, and the safety filter then
    catches what slips through rather than doing all the work."""
    terms = []
    for d in (prefs.get('diets') or []):
        terms.append(str(d).strip().lower())
    for a in (prefs.get('allergies') or []):
        terms.append(f"{str(a).strip().lower()}-free")
    for r in (prefs.get('religious') or []):
        terms.append(str(r).strip().lower())
    seen, out = set(), []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ' '.join(out[:4])


def _brave_find_recipe_url(dish, diet_terms=''):
    """Search one dish name and return the first result that looks like a single recipe
    page. Roundups, category pages and social links are skipped — they have no one
    ingredient list to extract."""
    key = os.getenv('BRAVE_API_KEY', '').strip()
    if not key:
        return None
    query = f'{diet_terms} {dish} recipe'.strip() if diet_terms else f'{dish} recipe'
    url = 'https://api.search.brave.com/res/v1/web/search?' + urllib.parse.urlencode(
        {'q': query, 'count': 8})
    try:
        req = urllib.request.Request(
            url, headers={'Accept': 'application/json', 'X-Subscription-Token': key})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception as exc:
        print(json.dumps({'evt': 'brave_search_failed', 'dish': dish, 'error': str(exc)[:120]}))
        return None
    for item in ((data.get('web') or {}).get('results') or []):
        title, link = item.get('title', ''), item.get('url', '')
        if not link or _EXPLORE_ROUNDUP_RE.search(title) or _EXPLORE_ROUNDUP_RE.search(link):
            continue
        return {'url': link, 'title': title}
    return None


_EXPLORE_BUILD_TABLE = 'shared_explore_build'


def _ensure_explore_build_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{_EXPLORE_BUILD_TABLE}` (
              `owner_id` VARCHAR(64) NOT NULL, `prefs_hash` VARCHAR(32) NOT NULL,
              `status` VARCHAR(16) NOT NULL DEFAULT 'building',
              `found` INT NOT NULL DEFAULT 0, `target` INT NOT NULL DEFAULT 15,
              `claim_token` VARCHAR(36) NULL,
              `started_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (`owner_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        # Tables created before the claim existed need the column adding. MySQL has no
        # ADD COLUMN IF NOT EXISTS, so the duplicate-column error is the "already there" path.
        try:
            cur.execute(f"ALTER TABLE `{_EXPLORE_BUILD_TABLE}` "
                        f"ADD COLUMN `claim_token` VARCHAR(36) NULL")
        except Exception:
            pass
    conn.commit()


def _explore_try_claim(conn, owner, fingerprint, target, token):
    """Atomically claim the right to build this owner's feed. True if we own it.

    Two triggers land for the same person seconds apart routinely: a second save of the
    same preferences, a household fan-out, a client retry. Without a claim each one runs
    the whole agent - double the API spend, and two writers racing on the same rows. This
    is a compare-and-swap on the build row: whoever gets their token in owns the build,
    everyone else backs off.

    A claim older than _EXPLORE_BUILD_STALE_SECONDS counts as abandoned, so a run that died
    mid-build can never lock someone out of ever getting a feed again.

    MySQL evaluates ON DUPLICATE KEY UPDATE assignments left to right and later expressions
    see the NEW value of earlier columns - so every column after `claim_token` applies only
    when the claim was actually won.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO `{_EXPLORE_BUILD_TABLE}`
                  (owner_id, prefs_hash, status, found, target, claim_token,
                   started_at, updated_at)
                VALUES (%s,%s,'building',0,%s,%s,UTC_TIMESTAMP(),UTC_TIMESTAMP())
                ON DUPLICATE KEY UPDATE
                  claim_token = IF(status <> 'building'
                                   OR updated_at < UTC_TIMESTAMP() - INTERVAL %s SECOND,
                                   VALUES(claim_token), claim_token),
                  prefs_hash = IF(claim_token = VALUES(claim_token),
                                  VALUES(prefs_hash), prefs_hash),
                  status     = IF(claim_token = VALUES(claim_token), 'building', status),
                  found      = IF(claim_token = VALUES(claim_token), 0, found),
                  target     = IF(claim_token = VALUES(claim_token), VALUES(target), target),
                  started_at = IF(claim_token = VALUES(claim_token),
                                  UTC_TIMESTAMP(), started_at),
                  updated_at = UTC_TIMESTAMP()""",
            [owner, fingerprint, target, token, int(_EXPLORE_BUILD_STALE_SECONDS)])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f"SELECT claim_token FROM `{_EXPLORE_BUILD_TABLE}` WHERE owner_id = %s",
                    [owner])
        row = cur.fetchone() or {}
    held = row.get('claim_token') if isinstance(row, dict) else (row[0] if row else None)
    return held == token


def _explore_build_mark(conn, owner, fingerprint, status, found, target):
    """Publish progress so the app can show recipes arriving instead of a blank wait.
    A full build takes ~163s for the hardest preference sets; nobody should watch a
    spinner for that long when the first recipe is ready in about twenty seconds."""
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO `{_EXPLORE_BUILD_TABLE}`
                  (owner_id, prefs_hash, status, found, target)
                VALUES (%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE prefs_hash=VALUES(prefs_hash),
                  status=VALUES(status), found=VALUES(found), target=VALUES(target)""",
            [owner, fingerprint, status, found, target])
    conn.commit()


def _explore_store_one(conn, owner, fingerprint, rank, r):
    """Insert ONE recipe the moment it survives the safety filter, so it is visible to the
    next poll rather than waiting for the whole batch."""
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO `{_EXPLORE_PERSONAL_TABLE}`
                (_id, owner_id, prefs_hash, `rank`, title, image_url, source_url,
                 source_domain, ingredients, instructions, notes, why)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [str(uuid.uuid4()), owner, fingerprint, rank, r['title'], r.get('image_url'),
             r.get('source_url'), r.get('source_domain'), json.dumps(r['ingredients']),
             json.dumps(r['instructions']), json.dumps(r.get('notes') or []),
             (r.get('why') or '')[:255]])
    conn.commit()


def _explore_agent_propose_dishes(prefs, count, exclude=None):
    """Ask the model for DISH NAMES, not URLs.

    Measured: asked for URLs it produced 0 of 16 that existed, because it generates
    plausible slugs rather than recalling real pages. Naming good dishes that fit a set of
    restrictions is the thing it is genuinely reliable at, so that is all it is asked for."""
    from openai import OpenAI
    exclude = exclude or set()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=float(os.getenv('EXPLORE_AGENT_TIMEOUT_SECONDS', '90')))
    lines = []
    for key, label in (('allergies', 'MUST NOT CONTAIN (allergy)'), ('diets', 'diet'),
                       ('religious', 'religious rule'), ('health', 'health goal'),
                       ('custom', 'personal preference')):
        vals = [str(x).strip() for x in (prefs.get(key) or []) if str(x).strip()]
        if vals:
            lines.append(f"- {label}: {', '.join(vals)}")
    block = "\n".join(lines) or "- no restrictions"
    system = (
        "You name excellent, well-known dishes. Return dish names only — no URLs, no "
        "recipes, no commentary.\n"
        "Each must fully satisfy every requirement given.\n"
        "Vary cuisine, meal type and effort. Favour proper meals over two-ingredient "
        "snacks. Prefer dishes a good food blog would have a dedicated recipe page for.\n"
        'Respond as JSON: {"dishes":[{"name":"","why":"max 12 words"}]}')
    user = f"Name {count} dishes for someone with these requirements:\n{block}"
    if exclude:
        user += ("\n\nAlready suggested — do NOT repeat these or close variants:\n"
                 + ", ".join(sorted(exclude)[:60]))
    resp = _create_chat(client, model=EXPLORE_AGENT_MODEL,
                        messages=[{'role': 'system', 'content': system},
                                  {'role': 'user', 'content': user}],
                        response_format={'type': 'json_object'})
    text = resp.choices[0].message.content or '{}'
    try:
        dishes = (json.loads(text) or {}).get('dishes') or []
    except Exception:
        dishes = [json.loads(c) for c in re.findall(r'\{[^{}]*"name"[^{}]*\}', text)]
    out = [{'name': str(d.get('name') or '').strip(), 'why': str(d.get('why') or '').strip()[:255]}
           for d in dishes if str((d or {}).get('name') or '').strip()]
    print(json.dumps({'evt': 'explore_agent_dishes', 'count': len(out)}))
    return out[:count]


def _explore_agent_iter_urls(prefs, count, exclude=None, seen_urls=None):
    """Dish names from the model, real URLs from search, YIELDED as each one is found.

    Search is rate-limited to about one query a second, so a round of fifteen spends ~17s
    here. Returning a list meant nothing could be extracted until the last search came back
    and the whole round then landed at once. Yielding lets each candidate start extracting
    while the next is still being searched for, so recipes reach the screen steadily instead
    of in clumps separated by silence.
    """
    dishes = _explore_agent_propose_dishes(prefs, count, exclude=exclude)
    diet_terms = _diet_query_terms(prefs)
    seen_urls = seen_urls if seen_urls is not None else set()
    seen_domains, found = {}, 0
    for dish in dishes:
        hit = _brave_find_recipe_url(dish['name'], diet_terms)
        time.sleep(BRAVE_QPS_DELAY)
        if not hit:
            continue
        if hit['url'] in seen_urls:
            continue
        domain = urllib.parse.urlparse(hit['url']).netloc.lower()
        if seen_domains.get(domain, 0) >= 3:      # no single blog fills the feed
            continue
        seen_urls.add(hit['url'])
        seen_domains[domain] = seen_domains.get(domain, 0) + 1
        found += 1
        yield {'url': hit['url'], 'title': hit.get('title') or dish['name'],
               'why': dish.get('why') or ''}
    print(json.dumps({'evt': 'explore_agent_proposed', 'dishes': len(dishes),
                      'urls': found, 'domains': len(seen_domains)}))


def _explore_screen_and_fetch(candidate):
    """Liveness check and extraction for ONE candidate, as a single unit of work.

    Kept together so a candidate can go search -> screen -> extract -> stored without waiting
    on its neighbours. Returns (was_live, recipe_or_None) so the caller can keep its funnel
    counts accurate without a second pass.
    """
    try:
        if not _explore_url_is_live(candidate['url']):
            return (False, None)
    except Exception:
        return (False, None)
    try:
        return (True, _explore_fetch_one(candidate))
    except Exception:
        return (True, None)


def _explore_url_is_live(url):
    """Cheap liveness check before spending a full extraction on a URL.

    web_search cuts fabrication but does not eliminate it — a measured run still returned
    URLs that 404. A HEAD costs ~0.3s against ~10s for an extraction, so screening first
    lets us over-propose candidates without burning the job's deadline on dead links.
    Sites that reject HEAD but serve GET are retried; anything still failing is dropped."""
    for method in ('HEAD', 'GET'):
        try:
            req = urllib.request.Request(
                url, method=method,
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                       'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36'})
            with urllib.request.urlopen(req, timeout=12) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            continue
    return False


def _explore_fetch_one(candidate):
    """Fetch + parse ONE candidate through the shared extractor. Returns None on any
    failure — a hallucinated or dead URL is simply dropped, never guessed at."""
    try:
        req = urllib.request.Request(
            EXPLORE_ANALYZE_URL,
            data=json.dumps({'url': candidate['url']}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=35) as r:
            payload = json.load(r)
    except Exception as exc:
        print(json.dumps({'evt': 'explore_agent_fetch_failed', 'url': candidate['url'],
                          'error': str(exc)[:160]}))
        return None
    text = payload.get('recipe') or ''
    if not text or text.strip() == 'Not enough recipe information.':
        return None

    def section(name):
        m = re.search(rf'{name}:\s*\n(.*?)(\n\n[A-Z][a-z]+:|\Z)', text, re.S)
        if not m:
            return []
        return [ln.lstrip('- ').strip() for ln in m.group(1).split('\n') if ln.strip()]

    ingredients, instructions = section('Ingredients'), section('Instructions')
    if not ingredients or not instructions:
        return None                      # same usability bar as a saved recipe
    title = (re.search(r'Title:\s*\n(.+)', text) or [None, candidate.get('title') or ''])
    title = title.group(1).strip() if hasattr(title, 'group') else (candidate.get('title') or '')
    return {
        'title': (title or candidate.get('title') or 'Recipe')[:255],
        'ingredients': ingredients, 'instructions': instructions,
        'notes': section('Notes'),
        'image_url': payload.get('image_url') or (payload.get('image_urls') or [None])[0],
        'source_url': candidate['url'],
        'source_domain': urllib.parse.urlparse(candidate['url']).netloc,
        'why': candidate.get('why') or '',
    }


def _personalize_explore(conn, owner, request_id=None):
    """Build this owner's personalised Explore. Safe to call repeatedly: a no-op when the
    preference fingerprint already matches what is stored."""
    prefs = _get_user_preferences(conn, owner)
    _ensure_explore_personal_table(conn)
    _ensure_explore_build_table(conn)
    if not any(prefs.get(k) for k in _DIETARY_PREF_KEYS):
        # Preferences were cleared. Drop the personalised feed rather than leaving it in
        # place: the app decides which feed to show from what is stored here, so a stale
        # feed means someone who removed every filter still sees a filtered Explore.
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM `{_EXPLORE_PERSONAL_TABLE}` WHERE owner_id = %s", [owner])
            cur.execute(f"DELETE FROM `{_EXPLORE_BUILD_TABLE}` WHERE owner_id = %s", [owner])
        conn.commit()
        return {'skipped': 'no_preferences', 'cleared': True}
    fingerprint = _prefs_hash(prefs)

    # "Already built" means a COMPLETED build for these exact preferences. Keying it on a
    # row count instead strands anyone whose build died early - a run that stopped at five
    # recipes looks good enough forever and never retries, which is what left a tester
    # staring at five recipes and a progress bar that never moved.
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT b.status, b.prefs_hash, b.target,
                       (SELECT COUNT(*) FROM `{_EXPLORE_PERSONAL_TABLE}` p
                         WHERE p.owner_id = b.owner_id AND p.prefs_hash = b.prefs_hash) AS n
                  FROM `{_EXPLORE_BUILD_TABLE}` b WHERE b.owner_id = %s""", [owner])
        row = cur.fetchone()
    if row:
        _get = (lambda k, i: row.get(k)) if isinstance(row, dict) else (lambda k, i: row[i])
        if (_get('status', 0) == 'complete' and _get('prefs_hash', 1) == fingerprint
                and int(_get('n', 3) or 0) >= int(_get('target', 2) or EXPLORE_AGENT_TARGET)):
            return {'skipped': 'unchanged', 'count': int(_get('n', 3) or 0)}

    token = uuid.uuid4().hex
    if not _explore_try_claim(conn, owner, fingerprint, EXPLORE_AGENT_TARGET, token):
        # Someone else is already building this feed. Backing off is the whole point:
        # running anyway would double the spend and race them writing the same rows.
        print(json.dumps({'evt': 'explore_build_skipped_concurrent', 'owner': owner}))
        return {'skipped': 'already_building'}

    started = time.monotonic()
    deadline = started + EXPLORE_AGENT_DEADLINE_SECONDS
    # Clear the previous feed now that the build is ours, so the app shows an honest
    # "finding your recipes" state and then watches them arrive one at a time.
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM `{_EXPLORE_PERSONAL_TABLE}` WHERE owner_id = %s", [owner])
    conn.commit()
    safe, seen_dishes, seen_urls = [], set(), set()
    proposed_total = live_total = extracted_total = dropped_total = 0

    # Rounds until the target is met. Each round asks for fresh dish names (the model is
    # told what it has already given), searches them, screens for liveness, extracts, and
    # applies the safety filter. A round that adds nothing still costs a round, so the
    # round cap and the deadline both bound the work.
    for round_no in range(1, EXPLORE_AGENT_MAX_ROUNDS + 1):
        if len(safe) >= EXPLORE_AGENT_TARGET or time.monotonic() > deadline:
            break
        want = EXPLORE_AGENT_TARGET - len(safe)
        # Round 1 is deliberately small. Search is rate-limited to ~1 query/sec, so asking
        # for 20 dishes up front means ~23s before extraction even starts and the screen
        # stays empty. A short first round puts a real recipe on screen in ~20s, and the
        # later rounds do the bulk while the user already has something to look at.
        round_size = (EXPLORE_AGENT_FIRST_ROUND if round_no == 1
                      else max(EXPLORE_AGENT_CANDIDATES, want * 2))
        round_live = round_extracted = 0
        handled = set()

        def _publish(future):
            """Screen result -> safety filter -> stored, for one candidate.

            Filtering and storing per recipe rather than per round is what lets the app show
            them arriving; the safety gate still runs on every single recipe before it is
            kept, regardless of what the model was told."""
            nonlocal round_live, round_extracted, dropped_total
            try:
                was_live, result = future.result()
            except Exception:
                was_live, result = False, None
            if was_live:
                round_live += 1
            if not result:
                return
            round_extracted += 1
            reason = _recipe_violates(result, prefs) if DIET_POST_FILTER_ENABLED else None
            if reason:
                dropped_total += 1
                print(json.dumps({'evt': 'diet_scrub_dropped', 'owner': str(owner),
                                  'title': str(result.get('title') or ''),
                                  'reason': reason}))
            elif (result.get('source_url')
                  and all(result['source_url'] != x.get('source_url') for x in safe)
                  and len(safe) < EXPLORE_AGENT_TARGET):
                safe.append(result)
                try:
                    _explore_store_one(conn, owner, fingerprint, len(safe) - 1, result)
                    _explore_build_mark(conn, owner, fingerprint, 'building',
                                        len(safe), EXPLORE_AGENT_TARGET)
                except Exception as exc:
                    print(json.dumps({'evt': 'explore_store_failed',
                                      'owner': owner, 'error': str(exc)[:160]}))

        futures = []
        with ThreadPoolExecutor(max_workers=EXPLORE_AGENT_WORKERS) as pool:
            # Submit each candidate the moment search produces it, and publish anything that
            # has already finished. Extraction therefore overlaps the next search instead of
            # waiting for the whole round.
            for c in _explore_agent_iter_urls(prefs, round_size, exclude=seen_dishes,
                                              seen_urls=seen_urls):
                proposed_total += 1
                seen_dishes.add((c.get('title') or '').lower()[:60])
                futures.append(pool.submit(_explore_screen_and_fetch, c))
                for f in [f for f in futures if f.done() and f not in handled]:
                    handled.add(f)
                    _publish(f)
                if len(safe) >= EXPLORE_AGENT_TARGET or time.monotonic() > deadline:
                    break
            for f in as_completed(futures):
                if f in handled:
                    continue
                handled.add(f)
                _publish(f)
                if len(safe) >= EXPLORE_AGENT_TARGET or time.monotonic() > deadline:
                    for pf in futures:
                        pf.cancel()
                    break

        live_total += round_live
        extracted_total += round_extracted
        print(json.dumps({'evt': 'explore_agent_round', 'round': round_no,
                          'live': round_live, 'extracted': round_extracted,
                          'kept_total': len(safe)}))

    fetched = []          # kept for the shape of the summary below
    dropped = dropped_total
    # The safety gate ran inside every round, on every recipe, after generation and
    # regardless of what the model was told — it is the only thing standing between an
    # allergic user and a web-sourced recipe, so nothing reaches this point unfiltered.
    safe = safe[:EXPLORE_AGENT_TARGET]

    # Rows were written as they were found; nothing left to flush. Close the build so the
    # app stops polling and drops the progress hero.
    _explore_build_mark(conn, owner, fingerprint,
                        'complete' if safe else 'failed', len(safe), EXPLORE_AGENT_TARGET)
    out = {'owner': owner, 'proposed': proposed_total, 'live': live_total,
           'extracted': extracted_total, 'dropped_unsafe': dropped,
           'stored': len(safe), 'target': EXPLORE_AGENT_TARGET,
           'met_target': len(safe) >= EXPLORE_AGENT_TARGET,
           'prefs_hash': fingerprint,
           'elapsed_ms': int((time.monotonic() - started) * 1000)}
    print(json.dumps({'evt': 'explore_personalized', **out}))
    return out

def handler(event, context):
    if (event or {}).get('sweep'):
        return _sweep_stuck_recipe_owners(context)
    # Personalised-Explore build. Separate entry point so it can be invoked on its own
    # (from a preferences save) without running the whole UseWhatIHave regeneration.
    if (event or {}).get('explore_personalize'):
        _owner = (event.get('owner') or '').strip()
        if not _owner:
            print('[explore_personalize] Missing owner')
            return {'error': 'missing owner'}
        _conn = _mysql_conn()
        try:
            return _personalize_explore(_conn, _owner)
        except Exception as _exc:
            print(json.dumps({'evt': 'explore_personalize_failed', 'owner': _owner,
                              'error': str(_exc)[:300]}))
            _report_backend_error('explore_personalize', owner_id=_owner,
                                  code='explore_personalize_failed', error=_exc)
            # Close the build out. Left in 'building' it holds the claim until the stale
            # window expires and the app keeps polling a build that is never coming back,
            # so a crash has to publish a terminal state, not just log one.
            try:
                with _conn.cursor() as _cur:
                    _cur.execute(f"UPDATE `{_EXPLORE_BUILD_TABLE}` SET status = 'failed', "
                                 f"claim_token = NULL, updated_at = UTC_TIMESTAMP() "
                                 f"WHERE owner_id = %s", [_owner])
                _conn.commit()
            except Exception:
                pass
            return {'error': str(_exc)[:200]}
        finally:
            try:
                _conn.close()
            except Exception:
                pass
    owner = (event.get('owner') or '').strip()
    if not owner:
        print('[recipes_generator] Missing owner')
        return
    owner_lock_name = _owner_lock_name(owner)
    owner_lock_conn = _acquire_named_lock(owner_lock_name, OWNER_LOCK_TIMEOUT_SECONDS)
    if owner_lock_conn is None:
        print(f'[recipes_generator] Lock held, marking refresh needed owner={owner}')
        try:
            mark_conn = _mysql_conn()
            _ensure_owner_kitchen_state_table(mark_conn)
            with mark_conn.cursor() as cur:
                cur.execute(
                    f"UPDATE `{_OWNER_KITCHEN_STATE_TABLE}` SET recipe_refresh_needed = 1 WHERE owner = %s",
                    [_sanitize_user_id(owner)]
                )
            mark_conn.commit()
        except Exception as e:
            print(f'[recipes_generator] Failed to mark refresh needed: {e}')
            _report_backend_error('mark_refresh_needed', owner_id=owner, code='mark_failed', error=e)
        return

    conn = _mysql_conn()
    target_owners = [owner]
    try:
        target_owners = _get_household_member_ids(conn, owner)
        has_existing_recipes = False
        for target_owner in target_owners:
            _ensure_recipes_table(conn, target_owner)
            table = _recipes_table(target_owner)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT _id, status, (kitchen_only IS NOT NULL AND JSON_LENGTH(kitchen_only) > 0) AS has_recipes "
                    f"FROM `{table}` WHERE _id = 'current'"
                )
                row = cur.fetchone()
                if not row:
                    cur.execute(
                        f"INSERT INTO `{table}` (_id, _owner, status) VALUES ('current', %s, 'regenerating')",
                        (target_owner,)
                    )
                    conn.commit()
                    _dual_write_recipes_to_shared(conn, target_owner)
                else:
                    # A row that already holds recipes must NEVER be gutted or hidden by
                    # a crash/timeout mid-regeneration. Treat existing content as
                    # has_existing_recipes (even if a prior timeout left status stuck at
                    # 'regenerating'), keep it VISIBLE ('ready') during regen, and swap
                    # the new set in atomically on success (the UPDATE below). Only show a
                    # bare 'regenerating' state when there is nothing to display.
                    if row.get('status') == 'ready' or row.get('has_recipes'):
                        has_existing_recipes = True
                    if not has_existing_recipes:
                        _set_status(conn, target_owner, 'regenerating')
                    elif row.get('status') != 'ready':
                        # Recover a set stranded in 'regenerating'/'failed' by a prior run.
                        _set_status(conn, target_owner, 'ready')

        ingredients = _get_kitchen_ingredients(conn, owner)
        kitchen_version = _get_owner_kitchen_version(conn, owner)
        kitchen_context = _build_kitchen_match_context(ingredients, kitchen_version=kitchen_version)
        if len(ingredients) < MIN_KITCHEN_ITEMS:
            for target_owner in target_owners:
                _set_empty(conn, target_owner)
            _mark_recipe_refresh_complete(conn, target_owners)
            print(f'[recipes_generator] Not enough kitchen items owner={owner} count={len(ingredients)} min={MIN_KITCHEN_ITEMS}')
            return

        # The current (old) recipes are about to be replaced by the full regeneration below and
        # are not referenced anywhere after this point, so we SKIP the expensive availability +
        # substitution matching on them. That matching was ~half of the regen's per-recipe
        # substitution LLM calls, all computed and discarded — a pure waste that slowed regen.
        current_kitchen = []
        current_need = []

        # Always do a full regeneration — generate fresh 10+10 recipes every time
        # the kitchen changes so the user sees new suggestions.
        gpt_started_at = time.monotonic()
        _structured_prefs = _get_user_preferences(conn, owner)
        _prefs_block = _format_preferences_block(_structured_prefs)
        # The dietary scrub runs AFTER generation, so asking for exactly ten guarantees a
        # short set the moment anything is dropped - and a short set used to leave the client
        # polling forever. Ask for headroom up front for anyone with preferences: one larger
        # call costs less latency than a second round trip, and the extras are trimmed to ten
        # below. Users without preferences are unaffected (headroom is zero).
        _has_prefs = any(_structured_prefs.get(k) for k in _DIETARY_PREF_KEYS)
        _ask = 10 + (RECIPE_DIET_HEADROOM if _has_prefs else 0)

        # A FAST FIRST BATCH, then the rest.
        #
        # Asking for the whole set in one call meant nothing existed for ~120s — the entire
        # wait is the model generating, so publishing "as sections finish" saved about three
        # seconds of it. Generating a handful first puts real recipes on screen in a fraction
        # of the time, and the remainder arrives while the user is already reading. Same total
        # work, and the app already renders a partial set during 'regenerating'.
        _first = min(RECIPE_FIRST_BATCH, _ask)
        first_out = _generate_recipes_with_gpt(
            ingredients,
            kitchen_only_count=_first,
            need_grocery_count=_first,
            excluded_titles=[],
            owner=owner,
            user_preferences_block=_prefs_block,
            prefer_fast_model=True,
        )
        first_out = {
            'kitchen_only': _scrub_recipes(first_out.get('kitchen_only') or [], _structured_prefs, owner=owner),
            'need_grocery': _scrub_recipes(first_out.get('need_grocery') or [], _structured_prefs, owner=owner),
        }
        _first_titles = [str((r or {}).get('title') or '').strip() for r in
                         (first_out.get('kitchen_only') or []) + (first_out.get('need_grocery') or [])]
        try:
            with ThreadPoolExecutor(max_workers=2) as _fp:
                _fk = _fp.submit(_normalize_recipe_batch, (first_out.get('kitchen_only') or []),
                                 'kitchen_only', kitchen_context, owner, 0, 'recipes_generator_first_kitchen')
                _fn = _fp.submit(_normalize_recipe_batch, (first_out.get('need_grocery') or []),
                                 'need_grocery', kitchen_context, owner, 0, 'recipes_generator_first_need')
                _publish_partial_recipes(conn, target_owners,
                                         _dedupe_recipes(_fk.result()),
                                         _dedupe_recipes(_fn.result()))
        except Exception as _exc:
            print(json.dumps({'evt': 'recipes_first_batch_failed', 'error': str(_exc)[:160]}))

        # The remainder, excluding what we already showed so the set does not repeat itself.
        gpt_out = _generate_recipes_with_gpt(
            ingredients,
            kitchen_only_count=_ask,
            need_grocery_count=_ask,
            excluded_titles=[t for t in _first_titles if t],
            owner=owner,
            user_preferences_block=_prefs_block,
        )
        # Deterministic post-generation scrubber: drop any recipe that violates a
        # HARD dietary exclusion the LLM prompt missed. No-op when prefs are empty
        # or DIET_POST_FILTER_ENABLED=false.
        gpt_out = {
            'kitchen_only': _scrub_recipes(gpt_out.get('kitchen_only') or [], _structured_prefs, owner=owner),
            'need_grocery': _scrub_recipes(gpt_out.get('need_grocery') or [], _structured_prefs, owner=owner),
        }
        print(
            f"[recipes_generator] GPT recipe generation owner={owner} "
            f"elapsed_ms={int((time.monotonic() - gpt_started_at) * 1000)} "
            f"full_regen=True"
        )
        # Normalize the two new sections CONCURRENTLY. Each fires parallel per-recipe
        # substitution matching internally, and the two sections are independent, so overlap
        # them instead of running one after the other.
        with ThreadPoolExecutor(max_workers=2) as _norm_pool:
            # Normalize the headroom too, not just the first ten - deduping afterwards can
            # itself drop the set below ten, and there is no point generating spares if they
            # are sliced off before they can cover a gap.
            _fut_new_kitchen = _norm_pool.submit(
                _normalize_recipe_batch,
                (gpt_out.get('kitchen_only') or [])[:_ask],
                'kitchen_only', kitchen_context, owner, 0, 'recipes_generator_new_kitchen',
            )
            _fut_new_need = _norm_pool.submit(
                _normalize_recipe_batch,
                (gpt_out.get('need_grocery') or [])[:_ask],
                'need_grocery', kitchen_context, owner, 0, 'recipes_generator_new_need',
            )
            # Publish as each section lands rather than after both. The app already renders
            # whatever exists while status is 'regenerating' — it was simply never given
            # anything until the very end, so a dietary change looked like minutes of nothing.
            new_kitchen = _fut_new_kitchen.result()
            _publish_partial_recipes(conn, target_owners,
                                     _dedupe_recipes(new_kitchen)[:10], [])
            new_need = _fut_new_need.result()

        kitchen_only = _dedupe_recipes(new_kitchen)[:10]
        need_grocery = _dedupe_recipes(new_need)[:10]
        if len(kitchen_only) < 10 or len(need_grocery) < 10:
            _report_backend_error('generate', owner_id=owner, code='incomplete_set',
                                  error=f'kitchen_only={len(kitchen_only)} need_grocery={len(need_grocery)} has_existing={has_existing_recipes}')
            if not has_existing_recipes:
                for target_owner in target_owners:
                    _set_status(conn, target_owner, 'failed', 'Could not build a full 10+10 recipe set after survivor backfill')
            else:
                # Their previous recipes are still on screen and still fine, but the row was
                # moved to 'regenerating' when this run started. Returning without resolving
                # it left the client polling a regeneration that was never going to land -
                # that is the loader that sticks at 94%. Every exit from here has to publish
                # a terminal status, even the ones where nothing is wrong.
                for target_owner in target_owners:
                    _set_status(conn, target_owner, 'ready')
            return

        kitchen_only_list, need_grocery_list = _build_recipes(kitchen_only, need_grocery)

        with conn.cursor() as cur:
            for target_owner in target_owners:
                table = _recipes_table(target_owner)
                cur.execute(f"""
                    UPDATE `{table}`
                    SET status = 'ready', kitchen_only = %s, need_grocery = %s, error_message = NULL, _updatedDate = NOW()
                    WHERE _id = 'current'
                """, (json.dumps(kitchen_only_list), json.dumps(need_grocery_list)))
            conn.commit()
        for target_owner in target_owners:
            _dual_write_recipes_to_shared(conn, target_owner)
        _mark_recipe_refresh_complete(conn, target_owners)
        print(f'[recipes_generator] Done owner={owner}')

        # Check if kitchen changed while we were generating -- if so, re-invoke (up to 3 retries)
        retry_count = int(event.get('retry_count') or (1 if event.get('is_retry') else 0))
        if retry_count < 3:
            try:
                _ensure_owner_kitchen_state_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT recipe_refresh_needed, kitchen_version, last_recipe_refresh_completed_version FROM `{_OWNER_KITCHEN_STATE_TABLE}` WHERE owner = %s",
                        [_sanitize_user_id(owner)]
                    )
                    state_row = cur.fetchone() or {}
                if state_row.get('recipe_refresh_needed') or (state_row.get('kitchen_version', 0) > state_row.get('last_recipe_refresh_completed_version', 0)):
                    print(f'[recipes_generator] Kitchen changed during generation, re-invoking for owner={owner} (retry {retry_count + 1}/3)')
                    import boto3
                    arn = context.invoked_function_arn
                    boto3.client('lambda').invoke(
                        FunctionName=arn,
                        InvocationType='Event',
                        Payload=json.dumps({'owner': owner, 'retry_count': retry_count + 1}),
                    )
            except Exception as re_err:
                print(f'[recipes_generator] Re-invoke check failed: {re_err}')
    except Exception as e:
        print(f'[recipes_generator] Error: {e}')
        import traceback
        traceback.print_exc()
        _report_backend_error('generate', owner_id=owner, code='handler_error', error=e)
        if not has_existing_recipes:
            try:
                for target_owner in target_owners:
                    _set_status(conn, target_owner, 'failed', str(e))
            except Exception:
                pass
    finally:
        _release_named_lock(owner_lock_conn, owner_lock_name)
