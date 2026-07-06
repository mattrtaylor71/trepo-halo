# home_suggestions_api/app.py
import os
import json
import re
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
from datetime import date, datetime
from decimal import Decimal

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

_DEFAULT_SUGGESTIONS = [
    {"item_name": "Milk", "reason": "A staple that most households use regularly", "confidence": 0.7},
    {"item_name": "Eggs", "reason": "Versatile protein source used in many recipes", "confidence": 0.7},
    {"item_name": "Bread", "reason": "Common pantry staple for meals and snacks", "confidence": 0.7},
    {"item_name": "Bananas", "reason": "Popular fruit with a short shelf life", "confidence": 0.6},
    {"item_name": "Chicken breast", "reason": "Lean protein that pairs with many dishes", "confidence": 0.6},
    {"item_name": "Rice", "reason": "Versatile grain that complements many meals", "confidence": 0.6},
    {"item_name": "Butter", "reason": "Essential for cooking and baking", "confidence": 0.6},
    {"item_name": "Onions", "reason": "Base ingredient for most savory dishes", "confidence": 0.6},
    {"item_name": "Cheese", "reason": "Pairs well with many meals and snacks", "confidence": 0.5},
    {"item_name": "Tomatoes", "reason": "Fresh produce staple for salads and cooking", "confidence": 0.5},
]


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


def _sanitize_owner(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')


def _cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    }


def _success(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': _cors_headers(),
        'body': json.dumps(data, default=json_serial),
    }


def _error(status_code, message):
    return {
        'statusCode': status_code,
        'headers': _cors_headers(),
        'body': json.dumps({'error': message}),
    }


def _kitchen_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_prod_kitchen"


def _get_household_member_ids(conn, acting_user_id):
    safe_user_id = _sanitize_owner(acting_user_id)
    if not safe_user_id:
        return [acting_user_id] if acting_user_id else []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id FROM new_users WHERE user_id = %s LIMIT 1", [safe_user_id])
            row = cur.fetchone()
            if not row:
                return [safe_user_id]
            household_id = row.get('owner_id') or row.get('OWNER_ID') or ''
            if not household_id:
                return [safe_user_id]
            cur.execute("SELECT user_id FROM new_users WHERE owner_id = %s", [household_id])
            members = [r.get('user_id') or r.get('USER_ID') for r in (cur.fetchall() or [])]
            members = [m for m in members if m]
            return members or [safe_user_id]
    except Exception:
        return [safe_user_id]


def _resolve_table(owner, suffix, conn=None):
    """Route reads to shared table (with owner_id filter) or per-user table."""
    safe = _sanitize_owner(owner)
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


def _table_exists(cur, table_name):
    cur.execute("""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """, [table_name])
    return cur.fetchone()['count'] > 0


def _fetch_kitchen_items(conn, owner):
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        where = "action = 'IN'"
        params = []
        if owner_where:
            where = f"{owner_where} AND {where}"
            params = list(owner_params)
        cur.execute(
            f"""
            SELECT product_name AS item_name, category,
                   product_expiration AS estimated_expiration,
                   DATEDIFF(product_expiration, CURDATE()) AS days_until_expiration,
                   _createdDate AS added_at
            FROM `{table_name}`
            WHERE {where}
            ORDER BY _createdDate DESC
            LIMIT 200
            """, params
        )
        return cur.fetchall() or []


def _build_prompt(kitchen_items, exclude_items=None):
    items_text = ""
    for item in kitchen_items:
        name = item.get('item_name', 'Unknown')
        category = item.get('category', 'Unknown')
        days_left = item.get('days_until_expiration')
        expiration = item.get('estimated_expiration')
        line = f"- {name} (category: {category}"
        if days_left is not None:
            line += f", expires in {days_left} days"
        elif expiration is not None:
            line += f", expires: {expiration}"
        line += ")"
        items_text += line + "\n"

    exclude_section = ""
    if exclude_items:
        exclude_list = ", ".join(exclude_items)
        exclude_section = f"""
ALREADY ON SHOPPING LIST (do NOT suggest these or very similar items):
{exclude_list}

"""

    return f"""You are a smart grocery shopping assistant. Based on the user's current kitchen contents below, suggest exactly 30 grocery items they likely need to buy this week.

IMPORTANT: Each item_name must be a specific, concrete product — something you'd actually write on a shopping list. NOT a category or vague description.
- Good: "Spinach", "Chicken Breast", "Greek Yogurt", "Yellow Onions", "Coconut Milk"
- Bad: "Fresh Vegetables (e.g. spinach or kale)", "Protein Source", "Healthy Snacks", "Dairy Products"

Consider:
- Items that are expiring soon or may have run out
- Common complementary items that go with what they already have
- Staples that are missing from their kitchen
- Variety and nutritional balance
- Do NOT suggest items the user already has on their shopping list
{exclude_section}
Current kitchen contents:
{items_text}

Respond with a JSON object in this exact format:
{{
  "suggestions": [
    {{"item_name": "string", "reason": "short reason why they need it", "confidence": 0.0}}
  ]
}}

The confidence should be a float between 0.0 and 1.0 indicating how confident you are they need this item. Keep item_name to 1-3 words max. Provide exactly 30 suggestions. Return ONLY valid JSON, no other text."""


def _generate_suggestions(kitchen_items, exclude_items=None):
    if OpenAI is None:
        print(f"[WARN] OpenAI import failed: {_OPENAI_IMPORT_ERROR}")
        return []

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[WARN] OPENAI_API_KEY not set")
        return []

    try:
        client = OpenAI(api_key=api_key)
        prompt = _build_prompt(kitchen_items, exclude_items=exclude_items)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a grocery suggestion assistant. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        suggestions = parsed.get('suggestions', [])

        # Validate structure
        validated = []
        for s in suggestions[:30]:
            if isinstance(s, dict) and 'item_name' in s:
                validated.append({
                    'item_name': str(s['item_name']),
                    'reason': str(s.get('reason', '')),
                    'confidence': float(s.get('confidence', 0.5)),
                })
        return validated

    except Exception as e:
        print(f"[ERROR] OpenAI call failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def _get_suggestions(owner, exclude_items=None):
    try:
        conn = _mysql_conn()
        kitchen_items = _fetch_kitchen_items(conn, owner)

        if not kitchen_items:
            return _success({'suggestions': _DEFAULT_SUGGESTIONS})

        serialized_items = [
            {k: json_serial(v) for k, v in item.items()}
            for item in kitchen_items
        ]

        suggestions = _generate_suggestions(serialized_items, exclude_items=exclude_items)

        if not suggestions:
            return _success({'suggestions': _DEFAULT_SUGGESTIONS})

        return _success({'suggestions': suggestions})

    except Exception as e:
        print(f"[ERROR] Failed to get suggestions: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Failed to retrieve suggestions: {str(e)}')


# ── Items To Watch ────────────────────────────────────────────────────

def _fetch_kitchen_items_full(conn, owner):
    """Fetch kitchen items with storage guidance and usage guide for items-to-watch."""
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        # Check which columns exist (usage_guide may not be present in all tables)
        cur.execute("""
            SELECT COLUMN_NAME FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s
        """, [table_name])
        existing_cols = {row['COLUMN_NAME'] for row in cur.fetchall()}

        select_cols = ['product_name', 'brand', 'category', 'product_expiration',
                       'storage_guidance', '_createdDate', '_id']
        # Only include optional columns if they exist
        if 'usage_guide' in existing_cols:
            select_cols.append('usage_guide')
        if 'product_image_url' in existing_cols:
            select_cols.append('product_image_url')
        if 'quantity' in existing_cols:
            select_cols.append('quantity')
        if 'is_opened' in existing_cols:
            select_cols.append('is_opened')
        if 'storage_location' in existing_cols:
            select_cols.append('storage_location')

        cols_sql = ', '.join(f'`{c}`' for c in select_cols)
        where = "action = 'IN'"
        params = []
        if owner_where:
            where = f"{owner_where} AND {where}"
            params = list(owner_params)
        cur.execute(
            f"""
            SELECT {cols_sql},
                   DATEDIFF(CURDATE(), DATE(_createdDate)) AS days_old
            FROM `{table_name}`
            WHERE {where}
            ORDER BY _createdDate DESC
            LIMIT 200
            """, params
        )
        return cur.fetchall() or []


def _parse_storage_guidance_text(sg_raw):
    """Extract the summary text from storage_guidance (may be JSON string or dict)."""
    if not sg_raw:
        return None
    if isinstance(sg_raw, str):
        try:
            sg = json.loads(sg_raw)
        except (json.JSONDecodeError, TypeError):
            return sg_raw
    elif isinstance(sg_raw, dict):
        sg = sg_raw
    else:
        return None
    return sg.get('summary', '')


def _parse_usage_guide_text(ug_raw):
    """Extract usage guide text (may be JSON string, dict, or plain text)."""
    if not ug_raw:
        return None
    if isinstance(ug_raw, str):
        try:
            ug = json.loads(ug_raw)
            if isinstance(ug, dict):
                return ug.get('content', ug.get('text', ug.get('summary', json.dumps(ug))))
            return str(ug)
        except (json.JSONDecodeError, TypeError):
            return ug_raw
    return str(ug_raw)


# ── Category perishability (mirrors iOS cleanPriority) ────────────────

_CATEGORY_MULTIPLIER = {
    'produce': 1.5, 'fruits': 1.5, 'vegetables': 1.5, 'fresh produce': 1.5,
    'meat': 1.4, 'seafood': 1.4, 'meat & seafood': 1.4, 'poultry': 1.4,
    'dairy': 1.2, 'dairy & eggs': 1.2, 'dairy alternatives': 1.1, 'eggs': 1.2,
    'deli': 1.1, 'deli & prepared foods': 1.1, 'prepared': 1.1, 'bakery': 1.1,
    'snacks': 0.6, 'snacks & sweets': 0.6, 'candy': 0.5,
    'beverages': 0.5, 'drinks': 0.5,
    'pantry': 0.4, 'condiment': 0.5, 'condiments': 0.5,
    'spices': 0.2, 'seasonings': 0.2, 'baking': 0.3,
}


def _compute_clean_priority(item):
    """Deterministic shelf-life score — mirrors iOS cleanPriority.

    User-entered expiration date is AUTHORITATIVE: when an item has a valid
    `product_expiration`, urgency / days_over / days_left are driven SOLELY by
    days-until-that-date and storage guidance is ignored for that item. Items
    without a user date fall back to the storage-guidance logic (unchanged).
    Returns a dict.
    """
    days_old = int(item.get('days_old', 0) or 0)
    cat = (item.get('category', '') or '').lower().strip()
    score = 0.0

    max_days = None
    min_days = None
    days_over = 0
    days_left = None

    # User-entered expiration date (authoritative when present + valid)
    exp_str = item.get('product_expiration', '') or ''
    exp_date_norm = None
    user_days_until = None
    if exp_str:
        try:
            exp_dt = datetime.strptime(str(exp_str)[:10], '%Y-%m-%d')
            user_days_until = (exp_dt - datetime.now()).days
            exp_date_norm = str(exp_str)[:10]
        except (ValueError, TypeError):
            user_days_until = None

    if user_days_until is not None:
        # AUTHORITATIVE — ignore storage guidance for this item.
        expiry_source = 'user'
        d = user_days_until
        if d < 0:
            score += 100 + abs(d) * 2
            days_over = abs(d)
        elif d <= 3:
            score += 80 + (3 - d) * 5
            days_left = d
        elif d <= 7:
            score += 50 + (7 - d) * 4
            days_left = d
        else:
            score += max(0, 30 - d)
            days_left = d
    else:
        # No user date — storage-guidance logic (unchanged behavior).
        expiry_source = 'estimate'
        sg_raw = item.get('storage_guidance')
        sg = None
        if sg_raw:
            if isinstance(sg_raw, str):
                try:
                    sg = json.loads(sg_raw)
                except (json.JSONDecodeError, TypeError):
                    sg = None
            elif isinstance(sg_raw, dict):
                sg = sg_raw

        if sg:
            # Resolve scenario-specific shelf life based on item's current state
            is_opened = bool(item.get('is_opened'))
            storage_loc = (item.get('storage_location') or 'fridge').lower().strip()
            scenario_key = f"{'opened' if is_opened else 'sealed'}_{storage_loc}"
            scenarios = sg.get('scenarios') or {}
            scenario = scenarios.get(scenario_key) or {}

            max_days = scenario.get('max_days') or sg.get('max_days') or sg.get('maxDays')
            min_days = scenario.get('min_days') or sg.get('min_days') or sg.get('minDays')
            if max_days is not None:
                try:
                    max_days = int(max_days)
                except (TypeError, ValueError):
                    max_days = None
            if min_days is not None:
                try:
                    min_days = int(min_days)
                except (TypeError, ValueError):
                    min_days = None

        # Sanity filter: dry/pantry goods with < 7-day shelf life are AI errors
        storage_zone = (sg.get('storage_zone', '') or '').lower() if sg else ''
        is_dry = ('pantry' in cat or 'spice' in cat or 'season' in cat or 'baking' in cat
                  or 'dry' in storage_zone or 'shelf' in storage_zone
                  or 'pantry' in storage_zone or 'room temp' in storage_zone)

        if max_days is not None:
            if is_dry and max_days < 7:
                # Bad AI data — ignore storage guidance, use minimal age penalty
                score += days_old * 0.1
            elif days_old >= max_days:
                score += 100 + (days_old - max_days) * 2
            elif min_days is not None and days_old >= min_days:
                urgency_frac = (days_old - min_days) / max(max_days - min_days, 1)
                score += 50 + urgency_frac * 40
            else:
                score += (days_old / max(max_days, 1)) * 30
            if days_old > max_days:
                days_over = days_old - max_days
            else:
                days_left = max_days - days_old
        else:
            score += days_old * 0.5

    # Category perishability multiplier
    multiplier = _CATEGORY_MULTIPLIER.get(cat, 0.7)
    score *= multiplier

    urgency = 'expired' if days_over > 0 else 'expiring_soon'

    return {
        'score': score,
        'days_over': days_over,
        'days_left': days_left,
        'urgency': urgency,
        'max_days': max_days,
        'expiration_date': exp_date_norm,
        'expiry_source': expiry_source,
    }


def _generate_items_to_watch(kitchen_items):
    """Deterministic shelf-life scoring (mirrors iOS cleanPriority) — no LLM needed."""

    scored = []
    for i, item in enumerate(kitchen_items):
        result = _compute_clean_priority(item)
        score = result['score']
        days_over = result['days_over']
        days_left = result['days_left']
        urgency = result['urgency']
        max_days = result['max_days']
        expiration_date = result['expiration_date']
        expiry_source = result['expiry_source']

        # Include items nearing shelf life (lower threshold catches approaching items)
        if score < 15:
            continue

        days_old = int(item.get('days_old', 0) or 0)
        name = item.get('product_name', 'Unknown')
        category = (item.get('category', '') or '').strip()

        # Build reason text
        if expiry_source == 'user':
            # Honor the date the user typed.
            if days_over > 0:
                reason = f"Past the expiration date you set by {days_over} day{'s' if days_over != 1 else ''}."
            elif days_left == 0:
                reason = "Expires today (the date you set)."
            elif days_left is not None:
                reason = f"Expires in {days_left} day{'s' if days_left != 1 else ''} (the date you set)."
            else:
                reason = "Nearing the expiration date you set."
        elif days_over > 0 and max_days:
            reason = f"Typical shelf life is {max_days} days — this has been in your kitchen {days_old} days."
        elif days_over > 0:
            reason = f"Past its expiration date by {days_over} day{'s' if days_over != 1 else ''}."
        elif max_days:
            remaining = max_days - days_old
            reason = f"Nearing end of {max_days}-day shelf life — about {remaining} day{'s' if remaining != 1 else ''} left."
        else:
            reason = f"Perishable {category.lower()} item, {days_old} days in your kitchen."

        # Extract emoji
        img_url = item.get('product_image_url', '') or ''
        emoji = img_url[6:] if isinstance(img_url, str) and img_url.startswith('emoji:') else None

        # Store _createdDate so days_old can be recomputed from cache
        created_date_raw = item.get('_createdDate')
        created_date_str = str(created_date_raw)[:10] if created_date_raw else None

        scored.append({
            'index': i,
            'item_id': item.get('_id'),
            'product_name': name,
            'reason': reason,
            'urgency': urgency,
            'days_over': days_over,
            'days_left': days_left,
            'days_old': days_old,
            'emoji': emoji,
            'quantity': item.get('quantity') or None,
            'confidence': min(score / 100.0, 1.0),
            'expiration_date': expiration_date,
            'expiry_source': expiry_source,
            '_score': score,
            '_created_date': created_date_str,
            '_max_days': max_days,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x['_score'], reverse=True)

    # Split into nearing vs expired, cap each list
    nearing = [s for s in scored if s['urgency'] == 'expiring_soon'][:10]
    expired = [s for s in scored if s['urgency'] == 'expired'][:10]

    result = nearing + expired

    # Remove internal score field
    for item in result:
        del item['_score']

    return result, None  # No recipe — iOS handles the Thyme button


def _ensure_shelf_life_cache_table(conn):
    """Create the shelf_life_cache table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shelf_life_cache (
                owner_id VARCHAR(64) PRIMARY KEY,
                items_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def _read_shelf_life_cache(conn, owner):
    """Read cached shelf life data. Returns (items_list, updated_at) or (None, None)."""
    try:
        _ensure_shelf_life_cache_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT items_json, updated_at FROM shelf_life_cache WHERE owner_id = %s",
                [owner]
            )
            row = cur.fetchone()
            if row and row.get('items_json'):
                items = json.loads(row['items_json'])
                return items, row.get('updated_at')
    except Exception as e:
        print(f"[WARN] Failed to read shelf life cache: {e}")
    return None, None


def _write_shelf_life_cache(conn, owner, cache_data):
    """Write shelf life results (items + recipe) to cache."""
    try:
        _ensure_shelf_life_cache_table(conn)
        data_json = json.dumps(cache_data, default=json_serial)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO shelf_life_cache (owner_id, items_json)
                   VALUES (%s, %s)
                   ON DUPLICATE KEY UPDATE items_json = VALUES(items_json), updated_at = NOW()""",
                [owner, data_json]
            )
            conn.commit()
        items_count = len(cache_data.get('items_to_watch', []))
        print(f"[CACHE] Wrote {items_count} shelf life items for owner {owner[:8]}...")
    except Exception as e:
        print(f"[WARN] Failed to write shelf life cache: {e}")


def _recompute_days_old(items):
    """Recompute days_old from _created_date so cached values stay accurate."""
    today = date.today()
    for item in items:
        # User-entered expiration is authoritative — recompute from the date so
        # a cached item's urgency stays correct as days pass (mirrors scoring).
        if item.get('expiry_source') == 'user' and item.get('expiration_date'):
            try:
                exp = datetime.strptime(str(item['expiration_date'])[:10], '%Y-%m-%d').date()
                d = (exp - today).days
                if d < 0:
                    item['days_over'] = abs(d)
                    item['days_left'] = None
                    item['urgency'] = 'expired'
                    item['reason'] = f"Past the expiration date you set by {abs(d)} day{'s' if abs(d) != 1 else ''}."
                else:
                    item['days_over'] = 0
                    item['days_left'] = d
                    item['urgency'] = 'expiring_soon'
                    item['reason'] = "Expires today (the date you set)." if d == 0 else f"Expires in {d} day{'s' if d != 1 else ''} (the date you set)."
            except (ValueError, TypeError):
                pass
            continue
        cd = item.get('_created_date')
        if cd:
            try:
                created = datetime.strptime(str(cd)[:10], '%Y-%m-%d').date()
                fresh_days_old = (today - created).days
                old_days_old = item.get('days_old', 0)
                item['days_old'] = max(0, fresh_days_old)
                # Update reason text and days_over if they reference stale days_old
                max_days = item.get('_max_days')
                if max_days and isinstance(max_days, (int, float)):
                    max_days = int(max_days)
                    item['days_over'] = max(0, fresh_days_old - max_days)
                    if item['days_over'] > 0:
                        item['urgency'] = 'expired'
                        item['reason'] = f"Typical shelf life is {max_days} days — this has been in your kitchen {fresh_days_old} days."
                    else:
                        remaining = max_days - fresh_days_old
                        if remaining <= 0:
                            item['urgency'] = 'expired'
                            item['reason'] = f"Past its {max_days}-day shelf life."
                        else:
                            item['urgency'] = 'expiring_soon'
                            item['reason'] = f"Nearing end of {max_days}-day shelf life — about {remaining} day{'s' if remaining != 1 else ''} left."
            except (ValueError, TypeError):
                pass


def _get_items_to_watch(owner):
    """GET handler — returns cached shelf life data instantly."""
    try:
        conn = _mysql_conn()
        cached_data, updated_at = _read_shelf_life_cache(conn, owner)

        if cached_data is not None:
            # Handle both old format (list) and new format (dict with items + recipe)
            if isinstance(cached_data, list):
                cached_data = {'items_to_watch': cached_data, 'recipe': None}
            # Recompute days_old from stored _created_date so values are always current
            if 'items_to_watch' in cached_data and isinstance(cached_data['items_to_watch'], list):
                _recompute_days_old(cached_data['items_to_watch'])
            result = cached_data
            result['cached'] = True
            result['updated_at'] = json_serial(updated_at) if updated_at else None
            return _success(result)

        # No cache yet — return empty so iOS shows loading state
        return _success({'items_to_watch': [], 'recipe': None, 'cached': False})

    except Exception as e:
        print(f"[ERROR] Failed to get items to watch: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Failed to retrieve items to watch: {str(e)}')


def _refresh_items_to_watch(owner):
    """POST handler — recalculates shelf life via LLM and writes to cache."""
    try:
        conn = _mysql_conn()
        kitchen_items = _fetch_kitchen_items_full(conn, owner)

        if not kitchen_items:
            empty_data = {'items_to_watch': [], 'recipe': None}
            try:
                conn = _mysql_conn()
                _write_shelf_life_cache(conn, owner, empty_data)
            except Exception:
                pass
            return _success({**empty_data, 'recalculated': True})

        serialized_items = [
            {k: json_serial(v) for k, v in item.items()}
            for item in kitchen_items
        ]

        items, recipe = _generate_items_to_watch(serialized_items)

        cache_data = {'items_to_watch': items, 'recipe': recipe}

        # Write to cache
        try:
            conn = _mysql_conn()
            _write_shelf_life_cache(conn, owner, cache_data)
        except Exception as e:
            print(f"[WARN] Cache write failed after recalc: {e}")

        return _success({**cache_data, 'recalculated': True})

    except Exception as e:
        print(f"[ERROR] Failed to refresh items to watch: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Failed to refresh items to watch: {str(e)}')


# ── Kitchen Activity Timeline ─────────────────────────────────────────

def _archive_table_name(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_archive_kitchen"


def _fetch_kitchen_activity(conn, owner):
    """Fetch daily aggregated kitchen activity + recent individual events."""
    prod_table, prod_owner_where, prod_owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    archive_table, arch_owner_where, arch_owner_params = _resolve_table(owner, '_archive_kitchen', conn)

    daily_summary = []
    recent_events = []

    with conn.cursor() as cur:
        prod_exists = _table_exists(cur, prod_table)
        archive_exists = _table_exists(cur, archive_table)

        if not prod_exists and not archive_exists:
            return daily_summary, recent_events

        # ── Daily summary (last 14 days) ──
        # Count check-ins per day from prod table
        union_parts = []
        query_params = []
        if prod_exists:
            prod_w = "_createdDate >= DATE_SUB(CURDATE(), INTERVAL 14 DAY)"
            if prod_owner_where:
                prod_w = f"{prod_owner_where} AND {prod_w}"
                query_params.extend(prod_owner_params)
            union_parts.append(f"""
                SELECT DATE(_createdDate) AS day, 'in' AS action_type, COUNT(*) AS cnt
                FROM `{prod_table}`
                WHERE {prod_w}
                GROUP BY DATE(_createdDate)
            """)
        if archive_exists:
            # Check-ins that were later archived (still count as adds on the day they were added)
            arch_in_w = "_createdDate >= DATE_SUB(CURDATE(), INTERVAL 14 DAY)"
            if arch_owner_where:
                arch_in_w = f"{arch_owner_where} AND {arch_in_w}"
                query_params.extend(arch_owner_params)
            union_parts.append(f"""
                SELECT DATE(_createdDate) AS day, 'in' AS action_type, COUNT(*) AS cnt
                FROM `{archive_table}`
                WHERE {arch_in_w}
                GROUP BY DATE(_createdDate)
            """)
            # Removals on the day they were archived
            arch_out_w = "archived_at >= DATE_SUB(CURDATE(), INTERVAL 14 DAY) AND archived_at IS NOT NULL"
            if arch_owner_where:
                arch_out_w = f"{arch_owner_where} AND {arch_out_w}"
                query_params.extend(arch_owner_params)
            union_parts.append(f"""
                SELECT DATE(archived_at) AS day, 'out' AS action_type, COUNT(*) AS cnt
                FROM `{archive_table}`
                WHERE {arch_out_w}
                GROUP BY DATE(archived_at)
            """)

        if union_parts:
            query = " UNION ALL ".join(union_parts)
            cur.execute(f"""
                SELECT day, action_type, SUM(cnt) AS total
                FROM ({query}) AS combined
                GROUP BY day, action_type
                ORDER BY day ASC
            """, query_params)
            rows = cur.fetchall() or []

            # Pivot into {day: {in: N, out: N}}
            day_map = {}
            for row in rows:
                d = str(row['day'])
                if d not in day_map:
                    day_map[d] = {'date': d, 'items_in': 0, 'items_out': 0}
                if row['action_type'] == 'in':
                    day_map[d]['items_in'] += int(row['total'])
                else:
                    day_map[d]['items_out'] += int(row['total'])
            daily_summary = sorted(day_map.values(), key=lambda x: x['date'])

        # ── Recent individual events (last 20) ──
        event_parts = []
        event_params = []
        if prod_exists:
            # Check which columns exist for emoji
            cur.execute("""
                SELECT COLUMN_NAME FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [prod_table])
            prod_cols = {r['COLUMN_NAME'] for r in cur.fetchall()}
            emoji_col = "product_image_url" if "product_image_url" in prod_cols else "NULL"

            prod_ev_where = "1=1"
            if prod_owner_where:
                prod_ev_where = prod_owner_where
                event_params.extend(prod_owner_params)
            event_parts.append(f"""
                (SELECT product_name, category, {emoji_col} AS image_url,
                       _createdDate AS event_time, 'added' AS event_type
                FROM `{prod_table}`
                WHERE {prod_ev_where}
                ORDER BY _createdDate DESC
                LIMIT 20)
            """)

        if archive_exists:
            cur.execute("""
                SELECT COLUMN_NAME FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s
            """, [archive_table])
            arch_cols = {r['COLUMN_NAME'] for r in cur.fetchall()}
            emoji_col_arch = "product_image_url" if "product_image_url" in arch_cols else "NULL"
            archived_at_col = "archived_at" if "archived_at" in arch_cols else "_updatedDate"

            arch_rm_where = f"{archived_at_col} IS NOT NULL"
            if arch_owner_where:
                arch_rm_where = f"{arch_owner_where} AND {arch_rm_where}"
                event_params.extend(arch_owner_params)
            event_parts.append(f"""
                (SELECT product_name, category, {emoji_col_arch} AS image_url,
                       {archived_at_col} AS event_time, 'removed' AS event_type
                FROM `{archive_table}`
                WHERE {arch_rm_where}
                ORDER BY {archived_at_col} DESC
                LIMIT 20)
            """)
            # Also add archived items as "added" events (on their _createdDate)
            arch_add_where = "1=1"
            if arch_owner_where:
                arch_add_where = arch_owner_where
                event_params.extend(arch_owner_params)
            event_parts.append(f"""
                (SELECT product_name, category, {emoji_col_arch} AS image_url,
                       _createdDate AS event_time, 'added' AS event_type
                FROM `{archive_table}`
                WHERE {arch_add_where}
                ORDER BY _createdDate DESC
                LIMIT 20)
            """)

        if event_parts:
            combined = " UNION ALL ".join(event_parts)
            cur.execute(f"""
                SELECT * FROM ({combined}) AS all_events
                ORDER BY event_time DESC
                LIMIT 20
            """, event_params)
            rows = cur.fetchall() or []
            for row in rows:
                emoji = None
                img = row.get('image_url') or ''
                if isinstance(img, str) and img.startswith('emoji:'):
                    emoji = img[6:]
                recent_events.append({
                    'product_name': row.get('product_name', 'Unknown'),
                    'category': row.get('category', ''),
                    'emoji': emoji,
                    'event_type': row.get('event_type', 'added'),
                    'event_time': row.get('event_time'),
                })

    return daily_summary, recent_events


def _get_kitchen_activity(owner):
    try:
        conn = _mysql_conn()
        daily_summary, recent_events = _fetch_kitchen_activity(conn, owner)

        return _success({
            'daily_summary': [json_serial(d) for d in daily_summary],
            'recent_events': [json_serial(e) for e in recent_events],
        })

    except Exception as e:
        print(f"[ERROR] Failed to get kitchen activity: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Failed to retrieve kitchen activity: {str(e)}')


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
        path_params = event.get('pathParameters') or {}
        owner = path_params.get('owner')
        raw_path = event.get('rawPath', '')
        query_params = event.get('queryStringParameters') or {}

        if http_method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

        if not owner:
            return _error(400, 'Missing owner parameter')

        if http_method == 'GET':
            if '/items-to-watch/' in raw_path:
                return _get_items_to_watch(owner)
            elif '/kitchen-activity/' in raw_path:
                return _get_kitchen_activity(owner)
            else:
                # Parse exclude list from query param (comma-separated shopping list items)
                exclude_raw = query_params.get('exclude', '')
                exclude_items = [item.strip() for item in exclude_raw.split(',') if item.strip()] if exclude_raw else None
                return _get_suggestions(owner, exclude_items=exclude_items)

        if http_method == 'POST':
            if '/items-to-watch/' in raw_path:
                return _refresh_items_to_watch(owner)
            return _error(400, 'POST not supported for this path')

        return _error(405, f'Method {http_method} not allowed')

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback; traceback.print_exc()
        return _error(500, f'Internal server error: {str(e)}')
