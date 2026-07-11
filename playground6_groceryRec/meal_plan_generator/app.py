# meal_plan_generator/app.py - Async Lambda: read kitchen, GPT plan + 12 recipes, generate images, write {owner}_meal_plan
import os
import re
import json
import base64
import hashlib
import time
import boto3
try:
    import pymysql
except ImportError as exc:
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc
from datetime import datetime
from decimal import Decimal


def _report_backend_error(op, owner_id=None, code=None, error=None, job_id=None, service='mealplan'):
    """Emit a metric-filterable marker (evt=backend_error) for a swallowed soft failure. Never throws."""
    try:
        import sys as _sys
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
        }), file=_sys.stderr)
    except Exception:
        pass

_DB_ENV_VARS = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
DEFAULT_FOCUS = 'great overall health'
MIN_KITCHEN_ITEMS = 5
OWNER_LOCK_TIMEOUT_SECONDS = 0
IMAGE_LOCK_TIMEOUT_SECONDS = 180
IMAGE_MIN_INTERVAL_SECONDS = 13
IMAGE_RATE_LIMIT_RETRIES = 4
IMAGE_RATE_LIMIT_BUFFER_SECONDS = 2
IMAGE_SLOT_COUNT = max(1, int(os.getenv('MEAL_PLAN_IMAGE_SLOT_COUNT', '3')))
ENABLE_MEAL_PLAN_IMAGES = str(os.getenv('MEAL_PLAN_GENERATE_IMAGES', 'false')).strip().lower() not in {'0', 'false', 'no', 'off'}
OPENAI_TIMEOUT_SECONDS = int(os.getenv('OPENAI_TIMEOUT_SECONDS', '45'))
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '2'))
DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
DB_WRITE_TIMEOUT = int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '10'))
USE_SHARED_TABLES = os.getenv('USE_SHARED_TABLES', 'false').lower() == 'true'
SLOT_ORDER = [
    (0, 'Today', 'Breakfast'), (0, 'Today', 'Snack'), (0, 'Today', 'Lunch'), (0, 'Today', 'Dinner'),
    (1, 'Tomorrow', 'Breakfast'), (1, 'Tomorrow', 'Snack'), (1, 'Tomorrow', 'Lunch'), (1, 'Tomorrow', 'Dinner'),
    (2, 'Day after', 'Breakfast'), (2, 'Day after', 'Snack'), (2, 'Day after', 'Lunch'), (2, 'Day after', 'Dinner'),
]


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
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5')),
        read_timeout=int(os.getenv('DB_READ_TIMEOUT_SECONDS', '30')),
        write_timeout=int(os.getenv('DB_WRITE_TIMEOUT_SECONDS', '30')),
    )
    return _conn


def _lock_mysql_conn(timeout_seconds):
    if pymysql is None:
        raise RuntimeError(f"pymysql import failed: {_PYMYSQL_IMPORT_ERROR}")
    config = _get_db_config()
    wait_timeout = max(
        DB_READ_TIMEOUT,
        DB_WRITE_TIMEOUT,
        int(timeout_seconds or 0) + 5,
    )
    return pymysql.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
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


def _meal_plan_table(owner):
    """Safe table name: {owner}_meal_plan."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', (owner or ''))
    if not safe:
        raise ValueError('Invalid owner')
    return f"{safe}_meal_plan"


def _sanitize_user_id(user_id):
    return re.sub(r'[^a-zA-Z0-9_-]', '', (user_id or ''))


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


def _ensure_meal_plan_table(conn, owner):
    table = _meal_plan_table(owner)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                _id VARCHAR(36) PRIMARY KEY,
                _owner VARCHAR(36) NOT NULL,
                status ENUM('ready', 'regenerating', 'failed', 'empty') NOT NULL DEFAULT 'empty',
                focus TEXT,
                explanation_title VARCHAR(255),
                explanation_paragraph TEXT,
                plan JSON,
                error_message TEXT,
                _createdDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                _updatedDate DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
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
            SELECT column_name AS col_name FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s
              AND column_name IN ('analysis_stage', 'analysis_status')
        """, [table_name])
        column_names = {row['col_name'] for row in cur.fetchall()}
        where_parts = []
        params = []
        if owner_where:
            where_parts.append(owner_where)
            params = list(owner_params)
        where_parts.append("action = 'IN'")
        # Gate on analysis_STATUS only (usable when 'ready'/NULL), NOT analysis_stage.
        # Voice check-in items are written analysis_stage='preliminary' by bulkKitchenWriter
        # and NEVER promoted to 'final' — promotion is bulkEnrichmentEngine's image-path
        # job, and voice adds have no capture image. The old `analysis_stage='final' OR
        # NULL` clause therefore excluded every voice-checkin kitchen -> count=0 -> empty
        # meal plans for voice users (recipes_generator has no such clause, which is why
        # recipes worked for them). Matching recipes_generator's live semantics.
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


def _set_status(conn, owner, status, error_message=None):
    table = _meal_plan_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE `{table}` SET status = %s, error_message = %s, _updatedDate = NOW() WHERE _id = 'current'",
            (status, error_message)
        )
        conn.commit()


def _set_empty(conn, owner, focus):
    table = _meal_plan_table(owner)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE `{table}`
            SET status = 'empty',
                focus = %s,
                explanation_title = NULL,
                explanation_paragraph = NULL,
                plan = %s,
                error_message = NULL,
                _updatedDate = NOW()
            WHERE _id = 'current'
            """,
            (focus, json.dumps([]))
        )
        conn.commit()


def _generate_plan_with_gpt(ingredients_list, focus):
    from openai import OpenAI, RateLimitError
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    ingredients_str = _format_kitchen_for_prompt(ingredients_list) if ingredients_list else 'No specific ingredients (suggest pantry staples)'
    slot_desc = ', '.join(f"{d} {m}" for _, d, m in SLOT_ORDER)
    system = """You are a meal planning assistant. Return valid JSON only, no markdown.
Output schema:
{
  "explanation_title": "short title e.g. Muscle gain meal plan",
  "explanation_paragraph": "One short paragraph explaining how this plan supports the user's focus.",
  "recipes": [ exactly 12 items, in order: Today Breakfast, Today Snack, Today Lunch, Today Dinner, Tomorrow Breakfast, ... Day after Dinner.
    Each item: { "title": "Recipe name", "reason": "One sentence explaining how this specific recipe supports the user's goal.", "ingredients": ["item1", "item2"], "steps": ["Step 1.", "Step 2."] }
  ]
}
Each kitchen item may include a short description in parentheses (e.g. "Ice Cubes Gum (chewing gum, not edible ice)"). Use that to avoid misuse: do NOT suggest meals that treat a product as something it is not (e.g. do not use gum as ice in drinks). Use ONLY grocery products from the provided kitchen list (and normal pantry: salt, pepper, oil, water). Each recipe must reference those grocery items by their exact product names in the ingredients array. Keep titles short. Steps concise.

Coherence rules:
- Recipes must be normal, common-sense dishes. Avoid odd pairings or contrived combos.
- Do NOT use beverage items (sparkling water, soda, seltzer, etc.) as cooking liquids or cereal bases.
- Meal plan slots are FOOD dishes, not beverages. If a beverage item exists, it can be omitted.
- If a slot cannot be filled coherently with the available items, choose a different combination of available items (still only from the list + pantry)."""
    user = f"Kitchen grocery products (use only these, reference by exact product name in recipe ingredients; descriptions in parentheses clarify what each item is): {ingredients_str}\n\nUser focus: {focus}\n\nGenerate a 12-slot meal plan. Slots order: {slot_desc}. Return the JSON object only."
    # Retry generate AND parse (mirrors recipes_generator): a malformed-JSON LLM
    # response regenerates instead of throwing, so the alarm means "twice-failed",
    # not "one bad sample". Rate-limit handling preserved.
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
                messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                temperature=0.4,
            )
        except RateLimitError as e:
            last_exc = e
            if attempt == 3:
                raise
            match = re.search(r'try again in ([\d.]+)s', str(e), flags=re.IGNORECASE)
            wait = float(match.group(1)) + 0.5 if match else 3.0
            print(f"[meal_plan_generator] Rate limited, retrying in {wait:.1f}s (attempt {attempt}/3)")
            time.sleep(wait)
            continue
        text = (resp.choices[0].message.content or '').strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        try:
            parsed = json.loads(text)
            if attempt > 1:
                print(json.dumps({'evt': 'meal_plan_parse_retry_saved', 'service': 'mealplan', 'attempt': attempt}))
            return parsed
        except json.JSONDecodeError as e:
            last_exc = e
            print(f"[meal_plan_generator] Malformed JSON (attempt {attempt}/3), regenerating: {e}")
            if attempt == 3:
                raise
            continue
    # Loop exhausted without returning (all attempts failed).
    if last_exc:
        raise last_exc
    raise RuntimeError('meal plan generation failed with no captured error')


def _build_recipe_image_prompt(title):
    title_lower = str(title or '').lower()
    if re.search(r'\b(coffee|tea|latte|espresso|cappuccino|smoothie|juice|soda|drink|beverage)\b', title_lower):
        vessel = 'Serve it in a simple white mug or clear glass on the same neutral tabletop instead of a plate.'
    elif re.search(r'\b(soup|stew|curry|ramen|noodles|oatmeal|porridge)\b', title_lower):
        vessel = 'Serve it in a simple white ceramic bowl on the same neutral tabletop.'
    else:
        vessel = 'Serve it on a simple white ceramic plate or bowl, whichever fits the dish naturally, on the same neutral tabletop.'

    return (
        f"Create a realistic food photograph of the dish: {title}. "
        "Photorealistic food photography only. Real textures, natural colors, realistic plating. "
        "Do not make it look like an illustration, cartoon, comic, animation frame, painting, or 3D render. "
        "Use the same standardized setting for every image: a clean neutral tabletop, soft natural lighting, "
        "minimal styling, and a slightly angled close food-photo composition. "
        f"{vessel} "
        "Keep the dish centered and clearly visible as the main focus. "
        "Accurately match the named meal so it looks like the actual food being described. "
        "Do not add unrelated side dishes, props, labels, or text."
    )


def _generate_recipe_image(title):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    prompt = _build_recipe_image_prompt(title)
    resp = client.images.generate(
        model='gpt-image-1',
        prompt=prompt,
        size='1024x1024',
        quality='medium',
        output_format='jpeg',
        output_compression=75,
        n=1,
    )
    if not resp.data or len(resp.data) == 0:
        return None
    b64 = getattr(resp.data[0], 'b64_json', None)
    if not b64:
        return None
    return base64.b64decode(b64)


def _owner_lock_name(owner):
    return f"meal-plan-owner:{_sanitize_user_id(owner)}"


def _image_lock_name(owner, slot_index):
    return f"meal-plan-image-rate-limit:{_sanitize_user_id(owner)}:{slot_index}"


def _is_rate_limit_error(error):
    status = getattr(error, 'status_code', None) or getattr(error, 'status', None)
    text = str(error)
    return (
        status == 429
        or 'rate_limit_exceeded' in text
        or 'Rate limit reached' in text
    )


def _retry_after_seconds(error):
    match = re.search(r'try again in (\d+)s', str(error), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _image_slot_index(owner, title):
    digest = hashlib.sha256(f"{_sanitize_user_id(owner)}:{title}".encode("utf-8")).hexdigest()
    return int(digest, 16) % IMAGE_SLOT_COUNT


def _generate_recipe_image_with_rate_limit(owner, title):
    last_error = None
    for attempt in range(IMAGE_RATE_LIMIT_RETRIES):
        slot_index = _image_slot_index(owner, title)
        lock_name = _image_lock_name(owner, slot_index)
        lock_conn = _acquire_named_lock(lock_name, IMAGE_LOCK_TIMEOUT_SECONDS)
        if lock_conn is None:
            raise RuntimeError('Timed out waiting for image generation slot')

        hold_seconds = IMAGE_MIN_INTERVAL_SECONDS
        started_at = time.monotonic()
        try:
            return _generate_recipe_image(title)
        except Exception as error:
            last_error = error
            if not _is_rate_limit_error(error) or attempt == IMAGE_RATE_LIMIT_RETRIES - 1:
                raise
            retry_after = _retry_after_seconds(error) or IMAGE_MIN_INTERVAL_SECONDS
            hold_seconds = max(hold_seconds, retry_after + IMAGE_RATE_LIMIT_BUFFER_SECONDS)
            print(
                f"[meal_plan_generator] Image rate limited for '{title}' "
                f"(attempt {attempt + 1}/{IMAGE_RATE_LIMIT_RETRIES}); retrying after cooldown"
            )
        finally:
            elapsed = time.monotonic() - started_at
            remaining = hold_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
            _release_named_lock(lock_conn, lock_name)

    raise last_error or RuntimeError('Image generation failed')


def _upload_image(bucket, owner, slot_index, image_bytes):
    key = f"meal-plan-images/{owner}/{slot_index}.jpg"
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=image_bytes,
        ContentType='image/jpeg',
    )
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def handler(event, context):
    owner = (event.get('owner') or '').strip()
    if not owner:
        print('[meal_plan_generator] Missing owner')
        return
    focus = (event.get('focus') or '').strip() or None
    conn = _mysql_conn()
    target_owners = [owner]
    try:
        if focus is None:
            table = _meal_plan_table(owner)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = %s", [table])
                if cur.fetchone()['n'] > 0:
                    cur.execute(f"SELECT focus FROM `{table}` WHERE _id = 'current'")
                    row = cur.fetchone()
                    focus = (row.get('focus') if row else None) or DEFAULT_FOCUS
                else:
                    focus = DEFAULT_FOCUS
        focus = (focus or DEFAULT_FOCUS).strip() or DEFAULT_FOCUS
    except Exception as e:
        print(f'[meal_plan_generator] DB read focus: {e}')
        focus = DEFAULT_FOCUS
    finally:
        pass

    bucket = os.getenv('BUCKET_NAME')
    if not bucket:
        print('[meal_plan_generator] BUCKET_NAME not set')
        return

    owner_lock_name = _owner_lock_name(owner)
    owner_lock_conn = _acquire_named_lock(owner_lock_name, OWNER_LOCK_TIMEOUT_SECONDS)
    if owner_lock_conn is None:
        print(f'[meal_plan_generator] Skipping duplicate run owner={owner}')
        return

    conn = None
    try:
        conn = _mysql_conn()
        target_owners = _get_household_member_ids(conn, owner)
        for target_owner in target_owners:
            _ensure_meal_plan_table(conn, target_owner)
            table = _meal_plan_table(target_owner)
            with conn.cursor() as cur:
                cur.execute(f"SELECT _id FROM `{table}` WHERE _id = 'current'")
                row = cur.fetchone()
                if not row:
                    cur.execute(
                        f"INSERT INTO `{table}` (_id, _owner, status, focus) VALUES ('current', %s, 'regenerating', %s)",
                        (target_owner, focus)
                    )
                    conn.commit()
            _set_status(conn, target_owner, 'regenerating')

        ingredients = _get_kitchen_ingredients(conn, owner)
        if len(ingredients) < MIN_KITCHEN_ITEMS:
            for target_owner in target_owners:
                _set_empty(conn, target_owner, focus)
            print(f'[meal_plan_generator] Not enough kitchen items owner={owner} count={len(ingredients)} min={MIN_KITCHEN_ITEMS}')
            return
        gpt_out = _generate_plan_with_gpt(ingredients, focus)
        recipes = gpt_out.get('recipes') or []
        if len(recipes) < 12:
            _report_backend_error('generate', owner_id=owner, code='insufficient_recipes',
                                  error=f'GPT returned {len(recipes)} recipes (need 12)')
            for target_owner in target_owners:
                _set_status(conn, target_owner, 'failed', 'GPT returned fewer than 12 recipes')
            return
        recipes = recipes[:12]

        plan = []
        for i, (day_idx, day_label, meal_type) in enumerate(SLOT_ORDER):
            r = recipes[i]
            title = r.get('title') or f'Meal {i+1}'
            reason = (r.get('reason') or '').strip()
            ings = r.get('ingredients') or []
            steps = r.get('steps') or []
            image_url = None
            if ENABLE_MEAL_PLAN_IMAGES:
                try:
                    img_bytes = _generate_recipe_image_with_rate_limit(owner, title)
                    if img_bytes:
                        image_url = _upload_image(bucket, owner, i, img_bytes)
                except Exception as e:
                    print(f'[meal_plan_generator] Image gen/upload failed for slot {i}: {e}')
            plan.append({
                'day_index': day_idx,
                'day_label': day_label,
                'meal_type': meal_type,
                'recipe': {
                    'title': title,
                    'reason': reason,
                    'ingredients': ings,
                    'steps': steps,
                    'image_url': image_url,
                },
            })

        explanation_title = (gpt_out.get('explanation_title') or '').strip() or f'Meal plan: {focus[:50]}'
        explanation_paragraph = (gpt_out.get('explanation_paragraph') or '').strip()

        with conn.cursor() as cur:
            for target_owner in target_owners:
                table = _meal_plan_table(target_owner)
                cur.execute(f"""
                    UPDATE `{table}`
                    SET status = 'ready', focus = %s, explanation_title = %s, explanation_paragraph = %s,
                        plan = %s, error_message = NULL, _updatedDate = NOW()
                    WHERE _id = 'current'
                """, (focus, explanation_title, explanation_paragraph, json.dumps(plan)))
            conn.commit()
        print(f'[meal_plan_generator] Done owner={owner}')
    except Exception as e:
        print(f'[meal_plan_generator] Error: {e}')
        import traceback
        traceback.print_exc()
        _report_backend_error('generate', owner_id=owner, code='handler_error', error=e)
        try:
            for target_owner in target_owners:
                _set_status(conn, target_owner, 'failed', str(e))
        except Exception:
            pass
    finally:
        _release_named_lock(owner_lock_conn, owner_lock_name)
