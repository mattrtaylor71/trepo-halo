# kitchen_insights_api/app.py
"""
Generates kitchen insights for a user based on their kitchen data.
Insight types: shelf_life_alert, kitchen_value, recipe_match, restock_reminder, healthier_swaps
"""
import os
import json
import re
import threading
import time
from datetime import datetime, date, timedelta
from decimal import Decimal

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

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
_DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
_DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '15'))
_DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'

# ── In-memory cache (persists across warm Lambda invocations) ────────────
_CACHE_TTL_SECONDS = 300  # 5 minutes
_insights_cache = {}  # {owner: {'data': response_body, 'ts': float}}


def _cache_get(owner):
    entry = _insights_cache.get(owner)
    if entry and (time.time() - entry['ts']) < _CACHE_TTL_SECONDS:
        return entry['data']
    return None


def _cache_set(owner, data):
    _insights_cache[owner] = {'data': data, 'ts': time.time()}


# ── Helpers ──────────────────────────────────────────────────────────────

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


def _sanitize_owner(owner):
    return re.sub(r'[^a-zA-Z0-9_-]', '', owner or '')


def _json_serial(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, dict):
        return {k: _json_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serial(i) for i in obj]
    return str(obj)


def _cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
    }


def _success(data, status_code=200):
    return {
        'statusCode': status_code,
        'headers': _cors_headers(),
        'body': json.dumps(data, default=_json_serial),
    }


def _error(status_code, message):
    return {
        'statusCode': status_code,
        'headers': _cors_headers(),
        'body': json.dumps({'error': message}),
    }


def _table_exists(cur, table_name):
    cur.execute("""
        SELECT COUNT(*) as cnt
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """, (table_name,))
    return cur.fetchone()['cnt'] > 0


def _kitchen_table(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError("Invalid owner")
    return f"`{safe}_prod_kitchen`"


def _archive_table(owner):
    safe = _sanitize_owner(owner)
    if not safe:
        raise ValueError("Invalid owner")
    return f"`{safe}_archive_kitchen`"


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


def _parse_date(val):
    """Parse a date value from DB into a date object."""
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if val:
        try:
            return datetime.strptime(str(val)[:10], '%Y-%m-%d').date()
        except (ValueError, TypeError):
            pass
    return None


def _parse_json_field(raw):
    """Safely parse a JSON string field, returning the parsed object or None."""
    if not raw:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _parse_price(price_str):
    """Parse price string like '$5-$8' into midpoint float."""
    if not price_str or str(price_str) in ('N/A', 'null', 'None', ''):
        return None
    s = str(price_str).replace('$', '').replace(',', '').strip()
    if '-' in s:
        parts = s.split('-')
        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return round((low + high) / 2, 2)
        except (ValueError, IndexError):
            pass
    try:
        return round(float(s), 2)
    except ValueError:
        return None


# ── Data Fetching ────────────────────────────────────────────────────────

def _fetch_current_items(conn, owner):
    """Fetch all current kitchen items with full fields for insights."""
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        where = "action = 'IN'"
        params = []
        if owner_where:
            where = f"{owner_where} AND {where}"
            params = list(owner_params)
        cur.execute(f"""
            SELECT _id, product_name, brand, category,
                   _createdDate, product_image_url, images,
                   DATEDIFF(CURDATE(), DATE(_createdDate)) AS days_old,
                   remaining_quantity, quantity_value, quantity_unit,
                   storage_guidance, estimated_price, healthier_alternatives,
                   product_expiration
            FROM `{table_name}`
            WHERE {where}
            ORDER BY _createdDate DESC
            LIMIT 300
        """, params)
        return cur.fetchall()


def _fetch_weekly_spend(conn, owner, weeks=8):
    """Get weekly value of items added, for the kitchen value timeline."""
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        where = "action = 'IN' AND _createdDate >= DATE_SUB(CURDATE(), INTERVAL %s WEEK)"
        params = list(owner_params) + [weeks] if owner_where else [weeks]
        if owner_where:
            where = f"{owner_where} AND {where}"
        cur.execute(f"""
            SELECT
                DATE(DATE_SUB(_createdDate, INTERVAL WEEKDAY(_createdDate) DAY)) AS week_start,
                COUNT(*) AS item_count,
                GROUP_CONCAT(estimated_price SEPARATOR '||') AS prices
            FROM `{table_name}`
            WHERE {where}
            GROUP BY week_start
            ORDER BY week_start ASC
        """, params)
        return cur.fetchall()


def _fetch_weekly_category_counts(conn, owner, weeks=6):
    """Get check-in counts per normalized category per week."""
    table_name, owner_where, owner_params = _resolve_table(owner, '_prod_kitchen', conn)
    with conn.cursor() as cur:
        if not _table_exists(cur, table_name):
            return []
        where = "action = 'IN' AND _createdDate >= DATE_SUB(CURDATE(), INTERVAL %s WEEK)"
        params = list(owner_params) + [weeks] if owner_where else [weeks]
        if owner_where:
            where = f"{owner_where} AND {where}"
        cur.execute(f"""
            SELECT
                YEARWEEK(_createdDate, 1) AS yw,
                DATE(DATE_SUB(_createdDate, INTERVAL WEEKDAY(_createdDate) DAY)) AS week_start,
                LOWER(category) AS category,
                COUNT(*) AS cnt
            FROM `{table_name}`
            WHERE {where}
            GROUP BY yw, week_start, LOWER(category)
            ORDER BY yw ASC
        """, params)
        return cur.fetchall()


# ── Price Estimation ─────────────────────────────────────────────────────

_CATEGORY_AVG_PRICES = {
    'produce': 3.50, 'fruits': 3.50, 'vegetables': 3.00,
    'dairy': 4.50, 'dairy_eggs': 4.50, 'dairy & eggs': 4.50,
    'meat': 8.00, 'meat_seafood': 8.00, 'meat & seafood': 8.00,
    'seafood': 9.00,
    'pantry': 4.00, 'pantry_staples': 4.00,
    'beverages': 3.00, 'drinks': 3.00,
    'snacks': 4.00, 'frozen': 5.00,
    'bakery': 4.50, 'deli': 6.00,
    'condiments': 3.50, 'condiments_sauces': 3.50,
    'condiments & sauces': 3.50,
}


def _fallback_price_estimate(category):
    """Get a category-based fallback price."""
    cat = (category or 'other').lower().replace('_', ' ')
    for key, price in _CATEGORY_AVG_PRICES.items():
        if key in cat or cat in key:
            return price
    return 4.00


def _estimate_missing_prices(unpriced_items):
    """Use GPT to estimate prices for items missing price data."""
    if not unpriced_items:
        return []

    if OpenAI is None or not os.getenv('OPENAI_API_KEY'):
        return [
            {'name': i['name'], 'price': _fallback_price_estimate(i.get('category')),
             'category': i.get('category', ''), 'brand': i.get('brand', ''),
             'image': i.get('image', ''), 'source': 'estimated'}
            for i in unpriced_items
        ]

    batch = unpriced_items[:40]
    item_list = '\n'.join(
        f"- {i['name']}" + (f" ({i['brand']})" if i.get('brand') else '')
        for i in batch
    )

    prompt = f"""Estimate the typical US grocery store price for each item. Return JSON only.

Items:
{item_list}

Return: {{"prices": [{{"name": "item name", "price": 4.99}}]}}

Rules:
- Typical US grocery prices (2024-2025)
- Round to nearest $0.50
- If unsure use category averages ($3-5 produce, $4-7 dairy, $6-12 meat, $3-6 pantry)
- Return ONLY valid JSON"""

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You estimate grocery prices. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2048,
        )
        data = json.loads(response.choices[0].message.content)
        price_map = {}
        for p in data.get('prices', []):
            name = (p.get('name') or '').lower().strip()
            pr = p.get('price')
            if name and isinstance(pr, (int, float)) and 0 < pr < 200:
                price_map[name] = round(float(pr), 2)

        results = []
        for item in batch:
            price = price_map.get(item['name'].lower().strip())
            if price is None:
                price = _fallback_price_estimate(item.get('category'))
            results.append({
                'name': item['name'], 'price': price,
                'category': item.get('category', ''), 'brand': item.get('brand', ''),
                'image': item.get('image', ''), 'source': 'estimated',
            })

        for item in unpriced_items[40:]:
            results.append({
                'name': item['name'],
                'price': _fallback_price_estimate(item.get('category')),
                'category': item.get('category', ''), 'brand': item.get('brand', ''),
                'image': item.get('image', ''), 'source': 'estimated',
            })

        return results
    except Exception as e:
        print(f"[WARN] GPT price estimation failed: {e}")
        return [
            {'name': i['name'], 'price': _fallback_price_estimate(i.get('category')),
             'category': i.get('category', ''), 'brand': i.get('brand', ''),
             'image': i.get('image', ''), 'source': 'estimated'}
            for i in unpriced_items
        ]


# ── Insight Generators ───────────────────────────────────────────────────

def _generate_shelf_life_alerts(items):
    """Find items past or approaching shelf life using storage_guidance."""
    today = date.today()
    expired = []
    approaching = []
    fresh = []

    for item in items:
        name = item.get('product_name', 'Unknown')
        image = item.get('product_image_url') or ''
        category = (item.get('category') or '').lower()

        # User-entered expiration is AUTHORITATIVE — classify by the date and
        # ignore storage guidance for this item (matches Clean / items-to-watch).
        exp_raw = item.get('product_expiration') or ''
        user_days_until = None
        exp_norm = None
        if exp_raw:
            try:
                exp_dt = datetime.strptime(str(exp_raw)[:10], '%Y-%m-%d').date()
                user_days_until = (exp_dt - today).days
                exp_norm = str(exp_raw)[:10]
            except (ValueError, TypeError):
                user_days_until = None

        if user_days_until is not None:
            entry = {
                'name': name, 'image': image,
                'category': item.get('category', ''),
                'expiration_date': exp_norm,
                'expiry_source': 'user',
            }
            if user_days_until < 0:
                entry['status'] = 'expired'
                entry['days_over'] = abs(user_days_until)
                expired.append(entry)
            elif user_days_until <= 7:
                entry['status'] = 'use_soon'
                entry['days_left'] = user_days_until
                approaching.append(entry)
            else:
                fresh.append(entry)
            continue

        # No user date — storage-guidance logic (unchanged behavior).
        sg = _parse_json_field(item.get('storage_guidance'))
        if not sg:
            continue

        min_days = sg.get('min_days')
        max_days = sg.get('max_days')
        storage_zone = (sg.get('storage_zone') or '').lower()

        if min_days is None or max_days is None:
            continue
        try:
            min_days = int(min_days)
            max_days = int(max_days)
        except (TypeError, ValueError):
            continue

        # Sanity filter: dry/pantry goods with <7 day shelf life are AI errors
        is_dry = ('pantry' in category or 'dry' in storage_zone
                  or 'shelf' in storage_zone or 'room temp' in storage_zone)
        if is_dry and max_days < 7:
            continue

        created_date = _parse_date(item.get('_createdDate'))
        if not created_date:
            continue

        age_days = (today - created_date).days

        entry = {
            'name': name, 'image': image,
            'category': item.get('category', ''),
            'age_days': age_days,
            'min_days': min_days, 'max_days': max_days,
            'storage_zone': sg.get('storage_zone', ''),
            'expiry_source': 'estimate',
        }

        if age_days > max_days:
            entry['status'] = 'expired'
            entry['days_over'] = age_days - max_days
            expired.append(entry)
        elif age_days >= min_days:
            entry['status'] = 'use_soon'
            entry['days_left'] = max_days - age_days
            approaching.append(entry)
        else:
            fresh.append(entry)

    total_tracked = len(expired) + len(approaching) + len(fresh)
    if total_tracked == 0:
        return []

    freshness_pct = int((len(fresh) / total_tracked) * 100)

    expired.sort(key=lambda x: x.get('days_over', 0), reverse=True)
    approaching.sort(key=lambda x: x.get('days_left', 999))

    if expired:
        title = f"{len(expired)} item{'s' if len(expired) != 1 else ''} past shelf life"
        names = [e['name'] for e in expired[:3]]
        subtitle = ', '.join(names)
        if len(expired) > 3:
            subtitle += f" + {len(expired) - 3} more"
        priority = 0
    elif approaching:
        title = f"{len(approaching)} item{'s' if len(approaching) != 1 else ''} to use soon"
        names = [e['name'] for e in approaching[:3]]
        subtitle = ', '.join(names)
        if len(approaching) > 3:
            subtitle += f" + {len(approaching) - 3} more"
        priority = 1
    else:
        title = f"All {total_tracked} tracked items are fresh"
        subtitle = f"Your kitchen freshness score is {freshness_pct}%."
        priority = 5

    return [{
        'type': 'shelf_life_alert',
        'icon': 'clock.badge.exclamationmark',
        'title': title,
        'subtitle': subtitle,
        'metric': f"{freshness_pct}% FRESH",
        'priority': priority,
        'detail': {
            'freshness_pct': freshness_pct,
            'expired_count': len(expired),
            'approaching_count': len(approaching),
            'fresh_count': len(fresh),
            'total_tracked': total_tracked,
            'expired_items': expired[:10],
            'approaching_items': approaching[:10],
            'explanation': f"Based on storage guidance for {total_tracked} items. "
                           f"{len(expired)} past shelf life, {len(approaching)} approaching, "
                           f"{len(fresh)} fresh.",
        }
    }]


def _generate_kitchen_value(items, conn, owner):
    """Calculate total kitchen value with GPT price estimation + timeline."""
    priced_items = []
    unpriced_items = []

    for item in items:
        name = item.get('product_name', 'Unknown')
        category = (item.get('category') or 'other').lower()
        brand = item.get('brand') or ''
        image = item.get('product_image_url') or ''

        parsed = _parse_price(item.get('estimated_price'))
        if parsed is not None:
            priced_items.append({
                'name': name, 'price': parsed, 'category': category,
                'brand': brand, 'image': image, 'source': 'listed',
            })
        else:
            unpriced_items.append({
                'name': name, 'category': category,
                'brand': brand, 'image': image,
            })

    # Estimate missing prices
    estimated = _estimate_missing_prices(unpriced_items)
    all_priced = priced_items + estimated

    total_value = sum(p['price'] for p in all_priced)

    # Category breakdown
    cat_values = {}
    for p in all_priced:
        cat = (p.get('category') or 'other').replace('_', ' & ').title()
        cat_values[cat] = cat_values.get(cat, 0) + p['price']
    cat_breakdown = sorted(cat_values.items(), key=lambda x: x[1], reverse=True)

    # Value timeline from weekly spend
    timeline = _build_value_timeline(conn, owner, all_priced, total_value)

    listed_count = len(priced_items)
    estimated_count = len(estimated)

    return [{
        'type': 'kitchen_value',
        'icon': 'dollarsign.circle',
        'title': f"Your kitchen is worth ~${int(total_value)}",
        'subtitle': f"{listed_count} priced + {estimated_count} estimated across {len(items)} items.",
        'metric': f"${int(total_value)}",
        'priority': 2,
        'is_pinned': True,
        'detail': {
            'total_value': round(total_value, 2),
            'item_count': len(items),
            'listed_count': listed_count,
            'estimated_count': estimated_count,
            'category_breakdown': [
                {'category': c, 'value': round(v, 2)} for c, v in cat_breakdown
            ],
            'top_items': sorted(
                [{'name': p['name'], 'price': p['price'], 'source': p['source'],
                  'image': p.get('image', '')}
                 for p in all_priced],
                key=lambda x: x['price'], reverse=True
            )[:10],
            'timeline': timeline,
            'explanation': f"Total estimated value of {len(items)} kitchen items. "
                           f"{listed_count} have listed prices, {estimated_count} were AI-estimated.",
        }
    }]


def _build_value_timeline(conn, owner, all_priced, current_total):
    """Build weekly kitchen value timeline using check-in data."""
    weekly_data = _fetch_weekly_spend(conn, owner, weeks=8)
    timeline = []

    # Compute average item price from our priced items
    avg_price = current_total / max(len(all_priced), 1)

    for row in weekly_data:
        ws = row.get('week_start')
        count = row.get('item_count', 0)
        # Parse actual prices from the concatenated string
        prices_str = row.get('prices') or ''
        week_value = 0
        if prices_str:
            for p in prices_str.split('||'):
                parsed = _parse_price(p)
                if parsed is not None:
                    week_value += parsed
                else:
                    week_value += avg_price
        else:
            week_value = count * avg_price

        if ws:
            timeline.append({
                'date': str(ws),
                'value': round(week_value, 2),
                'item_count': count,
            })

    # Add current week snapshot
    timeline.append({
        'date': date.today().isoformat(),
        'value': round(current_total, 2),
        'item_count': len(all_priced),
    })

    return timeline


def _generate_recipe_match(items):
    """Use GPT to find recipes the user can make with current kitchen items."""
    if not items or len(items) < 3:
        return []

    if OpenAI is None:
        print(f"[WARN] OpenAI import failed: {_OPENAI_IMPORT_ERROR}")
        return []
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[WARN] OPENAI_API_KEY not set")
        return []

    item_names = [item.get('product_name', '') for item in items[:50] if item.get('product_name')]
    if len(item_names) < 3:
        return []

    prompt = f"""You are a recipe matching assistant. Given the user's current kitchen inventory, find 2 recipes they can make RIGHT NOW with what they have. Prefer recipes where they have ALL or nearly all ingredients.

Current kitchen items:
{chr(10).join(f'- {name}' for name in item_names)}

Return a JSON object:
{{
  "recipes": [
    {{
      "name": "Recipe name",
      "matched_ingredients": ["ingredient1", "ingredient2"],
      "total_ingredients": 7,
      "missing": ["ingredient they don't have"],
      "description": "One sentence about the dish"
    }}
  ]
}}

Rules:
- matched_ingredients must be items from the kitchen list above. Treat the same ingredient under a different name/brand/form as a match (tamari = soy sauce, scallion = green onion, cilantro = coriander, prawns = shrimp); do NOT count a genuinely different food as a match.
- Prefer recipes with 0-1 missing ingredients
- Keep recipe names short (2-4 words)
- Return ONLY valid JSON"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("KITCHEN_INSIGHTS_RECIPE_MODEL", "gpt-5.4-mini"),
            messages=[
                {"role": "system", "content": "You are a recipe matching assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
        )
        data = json.loads(response.choices[0].message.content)
        recipes = data.get('recipes', [])

        insights = []
        for recipe in recipes[:2]:
            matched = recipe.get('matched_ingredients', [])
            total = recipe.get('total_ingredients', len(matched))
            missing = recipe.get('missing', [])
            match_ratio = f"{len(matched)}/{total}"

            if missing:
                subtitle = f"You have {len(matched)} of {total} ingredients — just need {', '.join(missing[:2])}."
            else:
                subtitle = f"All {total} ingredients on hand — {', '.join(matched[:4])}."

            insights.append({
                'type': 'recipe_match',
                'icon': 'fork.knife',
                'title': recipe.get('name', 'Recipe'),
                'subtitle': subtitle,
                'metric': match_ratio,
                'priority': 0 if not missing else 2,
                'detail': {
                    'description': recipe.get('description', ''),
                    'matched_ingredients': matched,
                    'missing_ingredients': missing,
                    'total_ingredients': total,
                }
            })
        return insights

    except Exception as e:
        print(f"[ERROR] Recipe match GPT call failed: {str(e)}")
        return []


def _generate_restock_reminder(items):
    """Find perishable items that need restocking based on shelf life."""
    today = date.today()
    use_soon = []
    expired_list = []

    for item in items:
        sg = _parse_json_field(item.get('storage_guidance'))
        if not sg:
            continue

        min_days = sg.get('min_days')
        max_days = sg.get('max_days')
        if min_days is None or max_days is None:
            continue
        try:
            min_days = int(min_days)
            max_days = int(max_days)
        except (TypeError, ValueError):
            continue

        # Only perishables (max shelf life <= 14 days)
        if max_days > 14:
            continue

        # Sanity check for AI errors on dry goods
        category = (item.get('category') or '').lower()
        storage_zone = (sg.get('storage_zone') or '').lower()
        is_dry = ('pantry' in category or 'dry' in storage_zone
                  or 'shelf' in storage_zone or 'room temp' in storage_zone)
        if is_dry and max_days < 7:
            continue

        created_date = _parse_date(item.get('_createdDate'))
        if not created_date:
            continue

        age_days = (today - created_date).days
        name = item.get('product_name', 'Unknown')

        entry = {
            'name': name,
            'category': item.get('category', ''),
            'image': item.get('product_image_url') or '',
            'age_days': age_days,
            'max_days': max_days,
        }

        if age_days > max_days:
            entry['urgency'] = 'expired'
            expired_list.append(entry)
        elif age_days >= min_days:
            entry['urgency'] = 'use_soon'
            entry['days_left'] = max_days - age_days
            use_soon.append(entry)

    total = len(expired_list) + len(use_soon)
    if total == 0:
        return []

    expired_list.sort(key=lambda x: x['age_days'], reverse=True)
    use_soon.sort(key=lambda x: x.get('days_left', 999))

    if expired_list:
        title = f"{len(expired_list)} perishable{'s' if len(expired_list) != 1 else ''} need replacing"
        subtitle = (f"Plus {len(use_soon)} more to use soon."
                    if use_soon else "Check your fridge and freezer.")
        priority = 1
    else:
        title = f"{len(use_soon)} perishable{'s' if len(use_soon) != 1 else ''} to use this week"
        top_names = ', '.join(e['name'] for e in use_soon[:3])
        subtitle = f"Plan meals around {top_names}."
        priority = 3

    return [{
        'type': 'restock_reminder',
        'icon': 'cart.badge.plus',
        'title': title,
        'subtitle': subtitle,
        'metric': f"{total} ITEMS",
        'priority': priority,
        'detail': {
            'expired_items': expired_list[:10],
            'use_soon_items': use_soon[:10],
            'total_perishables': total,
            'explanation': f"{total} perishable items (shelf life ≤14 days) need attention. "
                           f"{len(expired_list)} expired, {len(use_soon)} approaching.",
        }
    }]


def _generate_healthier_swaps(items):
    """Surface items that have healthier alternatives."""
    swappable = []

    for item in items:
        alts = _parse_json_field(item.get('healthier_alternatives'))
        if not isinstance(alts, list) or not alts:
            continue

        name = item.get('product_name', 'Unknown')
        top_alt = alts[0] if alts else {}

        swappable.append({
            'name': name,
            'image': item.get('product_image_url') or '',
            'category': item.get('category', ''),
            'swap_name': top_alt.get('name', ''),
            'swap_brand': top_alt.get('brand', ''),
            'why_healthier': top_alt.get('why_healthier', ''),
            'trade_offs': top_alt.get('trade_offs', ''),
            'all_alternatives': [_json_serial(a) for a in alts[:3]],
        })

    if not swappable:
        return []

    title = f"{len(swappable)} healthier swap{'s' if len(swappable) != 1 else ''} available"
    top_names = [s['name'] for s in swappable[:3]]
    if len(swappable) > 3:
        subtitle = f"Starting with {', '.join(top_names)} + {len(swappable) - 3} more."
    else:
        subtitle = f"Try swapping {', '.join(top_names)}."

    return [{
        'type': 'healthier_swaps',
        'icon': 'leaf.arrow.triangle.circlepath',
        'title': title,
        'subtitle': subtitle,
        'metric': f"{len(swappable)} SWAPS",
        'priority': 4,
        'detail': {
            'swappable_items': swappable[:15],
            'total_swappable': len(swappable),
            'explanation': f"{len(swappable)} items in your kitchen have healthier "
                           f"alternatives based on nutritional analysis.",
        }
    }]


def _generate_category_trends(items, weekly_cat_data):
    """Identify notable category trends comparing recent weeks."""
    insights = []

    if not weekly_cat_data:
        return insights

    # Normalize categories into broader groups for meaningful trends
    _GROUP_MAP = {
        'produce': 'Produce', 'fruits': 'Produce', 'vegetables': 'Produce',
        'meat': 'Protein', 'meat_seafood': 'Protein', 'meat & seafood': 'Protein',
        'seafood': 'Protein', 'poultry': 'Protein', 'deli meat': 'Protein',
        'dairy': 'Dairy', 'dairy_eggs': 'Dairy', 'dairy & eggs': 'Dairy',
        'eggs': 'Dairy',
        'pantry': 'Pantry', 'pantry_staples': 'Pantry', 'pantry staples': 'Pantry',
        'condiments': 'Pantry', 'condiments_sauces': 'Pantry',
        'condiments & sauces': 'Pantry', 'baking': 'Pantry',
        'beverages': 'Beverages', 'drinks': 'Beverages',
        'snacks': 'Snacks', 'frozen': 'Frozen', 'bakery': 'Bakery',
        'prepared foods': 'Prepared', 'deli': 'Prepared',
    }

    _GROUP_ICONS = {
        'Produce': 'leaf.fill', 'Protein': 'flame.fill',
        'Dairy': 'drop.fill', 'Pantry': 'cabinet.fill',
        'Beverages': 'cup.and.saucer.fill', 'Snacks': 'popcorn.fill',
        'Frozen': 'snowflake', 'Bakery': 'birthday.cake.fill',
        'Prepared': 'takeoutbag.and.cup.and.straw.fill',
    }

    # Group weekly data by normalized category and yearweek
    grouped = {}  # {group: {yw: count}}
    for row in weekly_cat_data:
        raw_cat = (row.get('category') or 'other').lower().strip()
        group = _GROUP_MAP.get(raw_cat)
        if not group:
            # Try partial matching
            for key, grp in _GROUP_MAP.items():
                if key in raw_cat or raw_cat in key:
                    group = grp
                    break
        if not group:
            group = 'Other'
        yw = row['yw']
        if group not in grouped:
            grouped[group] = {}
        grouped[group][yw] = grouped[group].get(yw, 0) + row['cnt']

    all_yws = sorted(set(r['yw'] for r in weekly_cat_data))
    if len(all_yws) < 2:
        return insights

    this_yw = all_yws[-1]
    prev_yws = all_yws[:-1]

    for group, weeks_data in grouped.items():
        this_cnt = weeks_data.get(this_yw, 0)
        # Average of previous weeks for a more stable comparison
        prev_counts = [weeks_data.get(yw, 0) for yw in prev_yws]
        avg_prev = sum(prev_counts) / len(prev_counts) if prev_counts else 0

        if avg_prev == 0 and this_cnt >= 3:
            change_pct = 100
        elif avg_prev > 0:
            change_pct = ((this_cnt - avg_prev) / avg_prev) * 100
        else:
            continue

        # Only surface meaningful changes
        if abs(change_pct) < 20 or (this_cnt < 2 and avg_prev < 2):
            continue

        direction = "up" if change_pct > 0 else "down"
        icon = _GROUP_ICONS.get(group, 'chart.line.uptrend.xyaxis')

        # Build weekly data for chart
        weekly_chart = [
            {'week': str(yw), 'count': weeks_data.get(yw, 0)}
            for yw in all_yws[-6:]
        ]

        # Find example items driving this trend
        drivers = []
        for item in items:
            item_cat = (item.get('category') or '').lower().strip()
            item_group = _GROUP_MAP.get(item_cat)
            if not item_group:
                for key, grp in _GROUP_MAP.items():
                    if key in item_cat or item_cat in key:
                        item_group = grp
                        break
            if item_group == group and (item.get('days_old') or 999) <= 7:
                drivers.append({
                    'name': item.get('product_name', 'Unknown'),
                    'image': item.get('product_image_url') or '',
                })
                if len(drivers) >= 4:
                    break

        insights.append({
            'type': 'category_trend',
            'icon': icon,
            'title': f"{group} is {direction} {abs(int(change_pct))}%",
            'subtitle': f"{this_cnt} items this week vs ~{int(avg_prev)} avg. "
                        f"{', '.join(d['name'] for d in drivers[:3])}.",
            'metric': f"{'+'if change_pct > 0 else ''}{int(change_pct)}%",
            'priority': 3 if abs(change_pct) >= 40 else 5,
            'detail': {
                'group': group,
                'this_week': this_cnt,
                'avg_previous': round(avg_prev, 1),
                'change_pct': int(change_pct),
                'direction': direction,
                'weekly_data': weekly_chart,
                'drivers': drivers,
                'explanation': f"You checked in {this_cnt} {group.lower()} items this week "
                               f"compared to an average of {avg_prev:.0f} over the prior "
                               f"{len(prev_yws)} weeks — a {abs(int(change_pct))}% "
                               f"{'increase' if change_pct > 0 else 'decrease'}.",
            }
        })

    # Sort by magnitude of change, keep top 2
    insights.sort(key=lambda x: abs(x['detail']['change_pct']), reverse=True)
    return insights[:2]


# ── Backfill Storage Guidance ─────────────────────────────────────────────

def _estimate_storage_via_llm(item):
    """Call GPT to estimate storage guidance for a single item."""
    if OpenAI is None or not os.getenv('OPENAI_API_KEY'):
        return None

    payload = json.dumps({
        'product_name': item.get('product_name'),
        'brand': item.get('brand'),
        'variant': item.get('variant'),
        'category': item.get('category'),
        'product_description': item.get('product_description'),
        'explanation': item.get('explanation'),
    }, indent=2)

    system_prompt = (
        'You are a food storage expert. Estimate how long a grocery item typically stays good in a home kitchen.\n\n'
        'CRITICAL: The product_description field tells you what this item ACTUALLY IS. Read it carefully before deciding.\n'
        '- "Simply Organic Thyme" with description "Glass jar of dried thyme leaves" → pantry, 180-365 days\n'
        '- "Fresh Thyme" with description "Bundle of fresh thyme sprigs" → refrigerated, 5-10 days\n'
        '- "Chicken Broth" with description "32oz carton of chicken broth" → refrigerated after opening, 5-7 days\n'
        '- "Canned Tomatoes" with description "28oz can of whole peeled tomatoes" → pantry, 365-730 days unopened\n\n'
        'The product_description is the ground truth for what the item is. Do NOT rely solely on the product_name.\n\n'
        'Guidelines:\n'
        '- Fresh produce, meat, seafood, bread, deli: estimate from check-in (just brought home)\n'
        '- Packaged/sealed items: estimate how long quality lasts once opened\n'
        '- Dried, canned, jarred shelf-stable items: long pantry shelf life (months to years)\n'
        '- Use conservative, practical consumer guidance\n\n'
        'Return JSON with: summary (string), min_days (int), max_days (int), '
        'timing_start ("from_check_in" or "after_opening"), '
        'storage_zone ("counter", "refrigerated", "pantry", "frozen", or "mixed"), '
        'confidence (0.0-1.0). Return ONLY valid JSON.'
    )

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=200,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[WARN] Storage LLM failed for {item.get('product_name')}: {e}")
        return None


def _backfill_storage_guidance(owner):
    """Re-estimate storage_guidance for all items using LLM."""
    try:
        conn = _mysql_conn()
        table = _kitchen_table(owner)

        with conn.cursor() as cur:
            raw_name = table.strip('`')
            if not _table_exists(cur, raw_name):
                return _error(404, 'Kitchen table not found')

            # Fetch all IN items with their descriptions
            cur.execute(f"""
                SELECT _id, product_name, brand, variant, category,
                       product_description, explanation, storage_guidance
                FROM {table}
                WHERE action = 'IN'
                ORDER BY _createdDate DESC
                LIMIT 300
            """)
            items = cur.fetchall()

        if not items:
            return _success({'message': 'No items to backfill', 'updated': 0})

        # Check coverage first
        has_desc = sum(1 for i in items if i.get('product_description'))
        total = len(items)

        updated = 0
        skipped = 0
        errors = 0
        sample_updates = []

        for item in items:
            new_guidance = _estimate_storage_via_llm(item)
            if not new_guidance:
                errors += 1
                continue

            old_sg = _parse_json_field(item.get('storage_guidance'))
            old_summary = (old_sg or {}).get('summary', 'none')

            guidance_json = json.dumps(new_guidance)
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {table}
                    SET storage_guidance = %s
                    WHERE _id = %s
                    LIMIT 1
                """, (guidance_json, item['_id']))
            conn.commit()
            updated += 1

            if len(sample_updates) < 10:
                sample_updates.append({
                    'name': item.get('product_name'),
                    'description': item.get('product_description') or '(none)',
                    'old': old_summary,
                    'new': new_guidance.get('summary', ''),
                    'new_days': f"{new_guidance.get('min_days')}-{new_guidance.get('max_days')}",
                    'zone': new_guidance.get('storage_zone', ''),
                })

        return _success({
            'message': f'Backfill complete. {updated} updated, {errors} errors, {skipped} skipped.',
            'total_items': total,
            'has_product_description': has_desc,
            'missing_product_description': total - has_desc,
            'updated': updated,
            'errors': errors,
            'sample_updates': sample_updates,
        })

    except Exception as e:
        print(f"[ERROR] Backfill failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Backfill failed: {str(e)}')


# ── Main Entry Point ─────────────────────────────────────────────────────

def _get_insights(owner):
    """Generate all insights for a user, with caching and parallel GPT calls."""
    # Check cache first
    cached = _cache_get(owner)
    if cached:
        print(f"[CACHE HIT] Returning cached insights for {owner}")
        return _success(cached)

    try:
        conn = _mysql_conn()
        items = _fetch_current_items(conn, owner)

        if not items:
            empty = {
                'insights': [],
                'generated_at': datetime.utcnow().isoformat(),
                'item_count': 0,
            }
            _cache_set(owner, empty)
            return _success(empty)

        # Fetch category trend data
        weekly_cat_data = _fetch_weekly_category_counts(conn, owner)

        # Run GPT-based generators in parallel (these are the slow ones)
        kitchen_value_result = []
        recipe_match_result = []

        def _run_kitchen_value():
            nonlocal kitchen_value_result
            kitchen_value_result = _generate_kitchen_value(items, conn, owner)

        def _run_recipe_match():
            nonlocal recipe_match_result
            recipe_match_result = _generate_recipe_match(items)

        t_value = threading.Thread(target=_run_kitchen_value)
        t_recipe = threading.Thread(target=_run_recipe_match)
        t_value.start()
        t_recipe.start()

        # Meanwhile, run the fast generators on the main thread
        all_insights = []
        all_insights.extend(_generate_shelf_life_alerts(items))
        all_insights.extend(_generate_restock_reminder(items))
        all_insights.extend(_generate_healthier_swaps(items))
        all_insights.extend(_generate_category_trends(items, weekly_cat_data))

        # Wait for GPT threads
        t_value.join()
        t_recipe.join()
        all_insights.extend(kitchen_value_result)
        all_insights.extend(recipe_match_result)

        # Sort by priority (lower = more important)
        all_insights.sort(key=lambda x: x.get('priority', 99))

        result = {
            'insights': [_json_serial(i) for i in all_insights],
            'generated_at': datetime.utcnow().isoformat(),
            'item_count': len(items),
        }
        _cache_set(owner, result)
        return _success(result)

    except Exception as e:
        print(f"[ERROR] Failed to generate insights: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error(500, f'Failed to generate insights: {str(e)}')


def handler(event, context):
    from trepo_auth import require_owner
    _denied = require_owner(event)
    if _denied is not None:
        return _denied
    try:
        http_method = event.get('requestContext', {}).get('http', {}).get(
            'method', event.get('httpMethod', 'GET'))
        path_params = event.get('pathParameters', {}) or {}
        owner = path_params.get('owner', '')

        if http_method == 'OPTIONS':
            return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

        if http_method == 'GET':
            if not owner:
                return _error(400, 'Missing owner parameter')
            return _get_insights(owner)

        if http_method == 'POST':
            if not owner:
                return _error(400, 'Missing owner parameter')
            raw_path = event.get('rawPath', event.get('path', ''))
            if 'backfill-storage' in raw_path:
                return _backfill_storage_guidance(owner)
            return _error(404, 'Unknown POST endpoint')

        return _error(405, f'Method {http_method} not allowed')

    except Exception as e:
        print(f"[ERROR] Unhandled: {str(e)}")
        return _error(500, f'Internal server error: {str(e)}')
