# recipes_generator/app.py - Async Lambda: read kitchen, GPT 10 kitchen-only + 10 need-grocery recipes, write {owner}_recipes
import os
import re
import json
import sys
import time
import hashlib
from pathlib import Path
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
SUBSTITUTION_MODEL = os.getenv('OPENAI_SUBSTITUTION_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o'))
MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS = max(0, int(os.getenv('RECIPES_MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS', '2')))
DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'
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
    try:
        response = client.chat.completions.create(
            model=SUBSTITUTION_MODEL,
            temperature=0.2,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        )
        payload = _best_effort_json_parse((response.choices[0].message.content or '').strip(), {'substitutions': [], 'summary': None})
    except Exception as exc:
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


def _generate_recipes_with_gpt(ingredients_list, kitchen_only_count=10, need_grocery_count=10, excluded_titles=None):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    kitchen_only_count = max(0, int(kitchen_only_count or 0))
    need_grocery_count = max(0, int(need_grocery_count or 0))
    if kitchen_only_count == 0 and need_grocery_count == 0:
        return {'kitchen_only': [], 'need_grocery': []}
    ingredients_str = _format_kitchen_for_prompt(ingredients_list) if ingredients_list else 'No specific ingredients (suggest pantry staples)'
    excluded_titles = [str(item).strip() for item in (excluded_titles or []) if str(item).strip()]
    exclusion_text = ''
    if excluded_titles:
        exclusion_text = '\nAvoid recreating these existing recipe titles or close variants: ' + ', '.join(excluded_titles[:30])
    system = """You are a recipe assistant. Return valid JSON only, no markdown.
Output schema:
{
  "kitchen_only": [ exactly the requested number of recipes. Each: { "title": "Recipe name", "meal_category": "breakfast|lunch|dinner|snacks", "ingredients": ["item1", "item2"], "steps": ["Step 1.", "Step 2."] }. Use ONLY the provided kitchen list of grocery products (reference by exact product name) plus pantry staples (salt, pepper, oil, water, basic spices). ],
  "need_grocery": [ exactly the requested number of recipes. Each: { "title": "Recipe name", "meal_category": "breakfast|lunch|dinner|snacks", "ingredients": ["item1", "item2", ...], "steps": ["Step 1.", "Step 2."], "missing_ingredients": ["item to buy 1", "item to buy 2"] }. Each recipe should use some grocery products from the kitchen list (reference by exact product name) but require at least one additional ingredient the user must buy. List those in missing_ingredients. ]
}
Each kitchen item may include a short description in parentheses (e.g. "Ice Cubes Gum (chewing gum, not edible ice)"). Use that to avoid misuse: do NOT suggest recipes that treat a product as something it is not (e.g. do not use gum as ice in drinks). Reference items by their exact product name in the ingredients array. Keep titles short. Steps concise.

Coherence rules:
- Recipes must be normal, common-sense dishes. Avoid odd pairings or contrived combos.
- Do NOT use beverage items (sparkling water, soda, seltzer, etc.) as cooking liquids or cereal bases.
- Recipes should be FOOD dishes, not beverages. If a beverage item exists, it can be omitted.
- Every recipe must include exactly one meal_category value chosen from: breakfast, lunch, dinner, snacks.
- If a recipe cannot be made coherently with the available items, choose a different combination of available items (still only from the list + pantry).

Meal substantiality rules (IMPORTANT):
- "dinner" and "lunch" recipes MUST be substantial, complete meals — the kind you would sit down and eat as a full plate. Think: pasta dishes, stir-fries, casseroles, grain bowls, tacos, soups, curries, sandwiches, salads with protein, etc.
- Do NOT categorize simple condiments, toppings, sauces, dips, relishes, or 1-2 ingredient combinations as "dinner" or "lunch". A recipe that is just mixing two condiments together (e.g. BBQ sauce + jalapenos) is NOT a meal.
- Simple items like dips, salsas, toppings, and spreads should be categorized as "snacks" if included at all.
- Each dinner/lunch recipe should have at least 3-4 meaningful ingredients (beyond pantry staples) and involve actual cooking or assembly of a complete dish.
- Prioritize recipes people actually make at home — classics, weeknight staples, popular cuisines. Avoid contrived combinations just to use up inventory.
Return the JSON object only."""
    user = (
        f"Kitchen grocery products (use exact product name in recipe ingredients; descriptions in parentheses clarify what each item is): {ingredients_str}\n\n"
        f"Generate exactly {kitchen_only_count} kitchen_only recipes using ONLY these items (plus pantry), and exactly {need_grocery_count} need_grocery recipes that use some of these items but need extra ingredients to buy (include missing_ingredients for each)."
        f"{exclusion_text}\nReturn the JSON object only."
    )
    resp = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        temperature=0.4,
    )
    text = (resp.choices[0].message.content or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def _build_recipe_record(recipe, fallback_index=0):
    recipe = recipe or {}
    record = dict(recipe)
    record.update({
        'title': recipe.get('title') or 'Recipe',
        'meal_category': _recipe_meal_category(recipe, fallback_index=fallback_index),
        'ingredients': recipe.get('ingredients') or [],
        'steps': recipe.get('steps') or [],
    })
    if 'missing_ingredients' in (recipe or {}):
        record['missing_ingredients'] = recipe.get('missing_ingredients') or []
    return record


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


def handler(event, context):
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
                cur.execute(f"SELECT _id, status FROM `{table}` WHERE _id = 'current'")
                row = cur.fetchone()
                if not row:
                    cur.execute(
                        f"INSERT INTO `{table}` (_id, _owner, status) VALUES ('current', %s, 'regenerating')",
                        (target_owner,)
                    )
                    conn.commit()
                else:
                    if row.get('status') == 'ready':
                        has_existing_recipes = True
                    _set_status(conn, target_owner, 'regenerating')

        ingredients = _get_kitchen_ingredients(conn, owner)
        kitchen_version = _get_owner_kitchen_version(conn, owner)
        kitchen_context = _build_kitchen_match_context(ingredients, kitchen_version=kitchen_version)
        if len(ingredients) < MIN_KITCHEN_ITEMS:
            for target_owner in target_owners:
                _set_empty(conn, target_owner)
            _mark_recipe_refresh_complete(conn, target_owners)
            print(f'[recipes_generator] Not enough kitchen items owner={owner} count={len(ingredients)} min={MIN_KITCHEN_ITEMS}')
            return

        current_kitchen_raw, current_need_raw = _load_current_recipes(conn, owner)
        current_kitchen_source = _parse_json_field(current_kitchen_raw)
        current_need_source = _parse_json_field(current_need_raw)
        current_kitchen = _normalize_recipe_batch(
            current_kitchen_source,
            'kitchen_only',
            kitchen_context,
            request_id=owner,
            start_index=0,
            source_label='recipes_generator_current_kitchen',
        )
        current_need = _normalize_recipe_batch(
            current_need_source,
            'need_grocery',
            kitchen_context,
            request_id=owner,
            start_index=0,
            source_label='recipes_generator_current_need',
        )

        # Always do a full regeneration — generate fresh 10+10 recipes every time
        # the kitchen changes so the user sees new suggestions.
        gpt_started_at = time.monotonic()
        gpt_out = _generate_recipes_with_gpt(
            ingredients,
            kitchen_only_count=10,
            need_grocery_count=10,
            excluded_titles=[],
        )
        print(
            f"[recipes_generator] GPT recipe generation owner={owner} "
            f"elapsed_ms={int((time.monotonic() - gpt_started_at) * 1000)} "
            f"full_regen=True"
        )
        new_kitchen = _normalize_recipe_batch(
            (gpt_out.get('kitchen_only') or [])[:10],
            'kitchen_only',
            kitchen_context,
            request_id=owner,
            start_index=0,
            source_label='recipes_generator_new_kitchen',
        )
        new_need = _normalize_recipe_batch(
            (gpt_out.get('need_grocery') or [])[:10],
            'need_grocery',
            kitchen_context,
            request_id=owner,
            start_index=0,
            source_label='recipes_generator_new_need',
        )

        kitchen_only = _dedupe_recipes(new_kitchen)[:10]
        need_grocery = _dedupe_recipes(new_need)[:10]
        if len(kitchen_only) < 10 or len(need_grocery) < 10:
            if not has_existing_recipes:
                for target_owner in target_owners:
                    _set_status(conn, target_owner, 'failed', 'Could not build a full 10+10 recipe set after survivor backfill')
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
        if not has_existing_recipes:
            try:
                for target_owner in target_owners:
                    _set_status(conn, target_owner, 'failed', str(e))
            except Exception:
                pass
    finally:
        _release_named_lock(owner_lock_conn, owner_lock_name)
