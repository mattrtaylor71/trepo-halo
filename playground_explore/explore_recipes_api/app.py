import base64
import json
import os
import re
import sys
import uuid
from pathlib import Path

import boto3
import pymysql

_LAYER_PYTHON = Path(__file__).resolve().parents[1] / "recipe_inventory_layer" / "python"
if _LAYER_PYTHON.exists() and str(_LAYER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_LAYER_PYTHON))

import recipe_inventory_llm

INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")
THUMBNAIL_BUCKET = os.getenv("THUMBNAIL_BUCKET", "")
THUMBNAIL_CDN_BASE = os.getenv("THUMBNAIL_CDN_BASE", "").rstrip("/")
OWNER_KITCHEN_STATE_TABLE = "owner_kitchen_state"
OWNER_RECIPE_AVAILABILITY_TABLE = "owner_recipe_availability"
INGREDIENT_NOISE_TOKENS = {
    "a", "an", "and", "fresh", "organic", "large", "small", "medium", "lean", "extra", "virgin",
    "boneless", "skinless", "shredded", "chopped", "diced", "minced", "sliced", "raw", "ground",
    "cooked", "dry", "plain", "whole", "halves", "pieces", "piece", "pack", "packs", "package",
    "packages", "bag", "bags", "box", "boxes", "bottle", "bottles", "jar", "jars", "can", "cans",
    "count", "ct", "lb", "lbs", "oz", "g", "kg", "ml", "l",
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons", "tsp",
    "pound", "pounds", "ounce", "ounces", "pinch", "dash", "to", "taste", "for", "of",
    "cut", "into", "in", "about", "roughly", "thinly", "finely", "thick", "thin",
    "divided", "optional", "needed", "serving", "garnish", "topping",
    "halved", "quartered", "cubed", "chunk", "chunks", "strip", "strips", "clove", "cloves",
    "plus", "plu", "more", "adjust", "as", "sized", "each", "per", "with", "or", "cooking",
    "deseeded", "peeled", "trimmed", "rinsed", "drained", "crushed", "pressed", "grated",
    "juiced", "zested", "squeezed", "melted", "softened", "thawed", "warmed", "chilled",
    "room", "temperature", "beaten", "whisked", "sifted", "toasted", "roasted",
}
PANTRY_CANONICAL_INGREDIENTS = {
    "salt", "kosher salt", "sea salt", "table salt", "flaky salt",
    "pepper", "black pepper", "white pepper",
    "oil", "olive oil", "vegetable oil", "canola oil", "neutral oil", "cooking oil", "sesame oil",
    "water", "cold water", "warm water", "hot water", "ice water", "ice",
    "butter", "salted butter", "unsalted butter",
    "cooking spray", "nonstick cooking spray",
    "sugar", "white sugar", "granulated sugar", "brown sugar", "powdered sugar",
    "flour", "all purpose flour", "wheat flour",
    "baking soda", "baking powder",
    "vinegar", "white vinegar", "apple cider vinegar",
    "basic spice", "spice", "basic spices", "seasoning", "seasonings",
    "garlic powder", "onion powder", "paprika", "cumin", "oregano", "cinnamon",
    "cornstarch", "corn starch",
}
PANTRY_TOKENS = {"salt", "pepper", "oil", "water", "ice", "butter", "spray",
                 "sugar", "flour", "spice", "spices", "seasoning", "seasonings"}
INGREDIENT_CONFLICT_TOKENS = {
    "apple", "avocado", "banana", "bean", "beef", "berry", "bread", "broccoli", "broth",
    "butter", "cabbage", "carrot", "cauliflower", "celery", "cheese", "cherry", "chicken",
    "chili", "corn", "cream", "egg", "fish", "flour", "garlic", "grape", "juice", "kale",
    "lemon", "lettuce", "lime", "mango", "milk", "mushroom", "onion", "orange", "pasta",
    "paste", "pea", "peach", "pear", "pepper", "pork", "potato", "powder", "rice", "salmon",
    "sauce", "seasoning", "seasonings", "shrimp", "spinach", "stock", "strawberry", "sugar",
    "tomatillo", "tomato", "tuna", "turkey", "vinegar", "yogurt",
}

# A generic category token (e.g. "cheese") is redundant — NOT a real conflict — when a
# specific variety of that category (e.g. "parmesan") is already present in the match
# context. Without this, the conflict-token guard wrongly blocks valid matches like
# kitchen "Parmesan" vs recipe "parmesan cheese". It must NOT loosen genuine conflicts:
# kitchen "cream" vs recipe "cream cheese" still fails (no cheese variety present).
CATEGORY_VARIETY_TOKENS = {
    "cheese": {
        "parmesan", "parmigiano", "reggiano", "cheddar", "mozzarella", "feta", "gouda",
        "brie", "provolone", "gruyere", "asiago", "romano", "pecorino", "ricotta",
        "mascarpone", "manchego", "colby", "swiss", "havarti", "gorgonzola", "fontina",
        "halloumi", "paneer", "cotija", "burrata", "camembert", "edam", "emmental",
        "jarlsberg", "muenster", "queso",
    },
}


def strip_redundant_category_conflicts(conflict_tokens, context_tokens):
    """Drop category conflict tokens (e.g. 'cheese') that are redundant because a specific
    variety of that category (e.g. 'parmesan') is present in context_tokens."""
    if not conflict_tokens:
        return conflict_tokens
    return {
        t for t in conflict_tokens
        if not (CATEGORY_VARIETY_TOKENS.get(t) and (context_tokens & CATEGORY_VARIETY_TOKENS[t]))
    }


OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_SUBSTITUTION_MODEL = os.getenv("OPENAI_SUBSTITUTION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS = max(0, int(os.getenv("EXPLORE_MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS", "2")))

s3 = boto3.client("s3")


def _mysql_conn():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        database=os.environ["DB_NAME"],
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
    )


def _cors():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _ok(body, status=200):
    return {
        "statusCode": status,
        "headers": _cors(),
        "body": json.dumps(body, default=str),
    }


def _err(status, message):
    return _ok({"error": message}, status=status)


def _parse_int(value, default, maximum=None):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if maximum is not None:
        n = min(n, maximum)
    return max(0, n)


def _parse_json_field(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return []


def _safe_text(value):
    return str(value or "").strip()


def _best_effort_json_parse(text, fallback):
    cleaned = _safe_text(text)
    if not cleaned:
        return fallback
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def _openai_client():
    api_key = _safe_text(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def _serialize_row(row):
    d = dict(row)
    for key in ("ingredients", "instructions", "notes", "image_urls"):
        d[key] = _parse_json_field(d.get(key))
    return d


_CATEGORY_ENSURED = False


def _ensure_category_column(conn):
    """Add the `category` column (+ index) to explore_recipes if missing.
    Cached per warm Lambda container so it's a no-op after the first call."""
    global _CATEGORY_ENSURED
    if _CATEGORY_ENSURED:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM information_schema.COLUMNS
            WHERE table_schema = DATABASE()
              AND table_name = 'explore_recipes' AND column_name = 'category'
            """
        )
        if cur.fetchone()["n"] == 0:
            cur.execute("ALTER TABLE explore_recipes ADD COLUMN category VARCHAR(64) DEFAULT NULL")
            try:
                cur.execute("CREATE INDEX idx_explore_category ON explore_recipes (category)")
            except Exception:
                pass
    conn.commit()
    _CATEGORY_ENSURED = True


def _safe_owner(owner):
    return re.sub(r"[^a-zA-Z0-9_-]", "", str(owner or ""))


def _ensure_owner_kitchen_state_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{OWNER_KITCHEN_STATE_TABLE}` (
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
            CREATE TABLE IF NOT EXISTS `{OWNER_RECIPE_AVAILABILITY_TABLE}` (
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
                PRIMARY KEY (owner, recipe_source, recipe_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def _ensure_personalization_tables(conn):
    _ensure_owner_kitchen_state_table(conn)
    _ensure_owner_recipe_availability_table(conn)


def _get_owner_kitchen_version(conn, owner):
    _ensure_owner_kitchen_state_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT kitchen_version FROM `{OWNER_KITCHEN_STATE_TABLE}` WHERE owner = %s LIMIT 1",
            [_safe_owner(owner)],
        )
        row = cur.fetchone() or {}
    return int(row.get("kitchen_version") or 0)


def _singularize_token(token):
    token = str(token or "").strip().lower()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("oes") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _ingredient_tokens(text):
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    tokens = []
    for raw in cleaned.split():
        if not raw or raw.isdigit():
            continue
        if re.fullmatch(r"\d+(?:oz|lb|lbs|g|kg|ml|l|ct|pack)?", raw):
            continue
        token = _singularize_token(raw)
        if not token or token in INGREDIENT_NOISE_TOKENS:
            continue
        tokens.append(token)
    return tokens


def _canonical_ingredient_name(text):
    return " ".join(_ingredient_tokens(text))


def _is_pantry_ingredient(text):
    canonical = _canonical_ingredient_name(text)
    if not canonical:
        return False
    if canonical in PANTRY_CANONICAL_INGREDIENTS:
        return True
    tokens = canonical.split()
    return bool(tokens) and all(token in PANTRY_TOKENS for token in tokens)


def _get_kitchen_items_for_matching(conn, owner):
    table_name = f"{_safe_owner(owner)}_prod_kitchen"
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
        """, [table_name])
        if (cur.fetchone() or {}).get("n", 0) == 0:
            return []
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = %s
              AND column_name IN ('product_description', 'analysis_stage', 'analysis_status')
        """, [table_name])
        column_names = {
            (row.get("column_name") or row.get("COLUMN_NAME") or "").strip()
            for row in (cur.fetchall() or [])
            if (row.get("column_name") or row.get("COLUMN_NAME") or "").strip()
        }
        where_parts = ["action = 'IN'"]
        if "analysis_stage" in column_names:
            where_parts.append("(`analysis_stage` = 'final' OR `analysis_stage` IS NULL)")
        if "analysis_status" in column_names:
            where_parts.append("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)")
        select_fields = "`product_name`"
        if "product_description" in column_names:
            select_fields += ", `product_description`"
        cur.execute(
            f"SELECT {select_fields} FROM `{table_name}` WHERE {' AND '.join(where_parts)} ORDER BY COALESCE(`_updatedDate`, `_createdDate`) DESC, `_createdDate` DESC"
        )
        rows = cur.fetchall() or []
    items = []
    seen = set()
    for row in rows:
        display_name = str((row or {}).get("product_name") or "").strip()
        if not display_name:
            continue
        dedupe_key = display_name.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        description = str((row or {}).get("product_description") or "").strip()
        tokens = _ingredient_tokens(" ".join(part for part in [display_name, description] if part))
        items.append({
            "display_name": display_name,
            "tokens": tokens,
        })
    return items


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
    forward_conflicts = strip_redundant_category_conflicts(
        {t for t in extra_tokens if t in INGREDIENT_CONFLICT_TOKENS}, context_tokens)
    if recipe_set.issubset(candidate_set) and not forward_conflicts:
        return 200 + (len(recipe_set) * 10) - max(0, len(candidate_set) - len(recipe_set))
    # Reverse: kitchen item tokens fully contained in recipe tokens
    if candidate_set.issubset(recipe_set):
        recipe_extra = recipe_set - candidate_set
        conflict_in_extra = strip_redundant_category_conflicts(
            {t for t in recipe_extra if t in INGREDIENT_CONFLICT_TOKENS}, context_tokens)
        if not conflict_in_extra:
            return 150 + (len(candidate_set) * 10) - max(0, len(recipe_set) - len(candidate_set))
    return -1


def _build_kitchen_context(conn, owner):
    return {
        "kitchen_version": _get_owner_kitchen_version(conn, owner),
        "kitchen_candidates": _get_kitchen_items_for_matching(conn, owner),
    }


def _log_inventory_match_event(request_id, event_name, **kwargs):
    payload = {
        "request_id": request_id,
        "event": event_name,
    }
    payload.update(kwargs)
    print(json.dumps(payload, default=str))


def _compute_recipe_availability(recipe, kitchen_context, request_id=None):
    recipe_id = _safe_text((recipe or {}).get("id") or (recipe or {}).get("_id") or "inline-recipe")
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        [{
            "id": recipe_id,
            "title": _safe_text((recipe or {}).get("title")),
            "ingredients": [_safe_text(item) for item in ((recipe or {}).get("ingredients") or []) if _safe_text(item)],
        }],
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label="explore",
        substitution_callback=_suggest_explore_recipe_substitutions,
    )
    return availability_map.get(recipe_id) or recipe_inventory_llm.deterministic_availability(
        recipe,
        kitchen_context,
        substitution_callback=_suggest_explore_recipe_substitutions,
    )


def _compute_explore_recipe_availability_batch(recipes, kitchen_context, request_id=None):
    recipe_payloads = []
    for recipe in recipes or []:
        recipe_id = _safe_text((recipe or {}).get("id") or (recipe or {}).get("_id"))
        if not recipe_id:
            continue
        recipe_payloads.append({
            "id": recipe_id,
            "title": _safe_text((recipe or {}).get("title")),
            "ingredients": [_safe_text(item) for item in ((recipe or {}).get("ingredients") or []) if _safe_text(item)],
        })
    availability_map, _ = recipe_inventory_llm.match_recipes_fast(
        recipe_payloads,
        kitchen_context,
        request_id=request_id,
        log_fn=_log_inventory_match_event,
        source_label="explore_batch",
        substitution_callback=_suggest_explore_recipe_substitutions,
    )
    return availability_map


def _suggest_explore_recipe_substitutions(recipe, kitchen_context, missing_ingredients):
    if not missing_ingredients or len(missing_ingredients) > MAX_AI_SUBSTITUTION_MISSING_INGREDIENTS:
        return {
            "substitution_candidates": [],
            "substitution_summary": None,
            "substitution_status": None,
            "can_make_with_subs": False,
        }
    client = _openai_client()
    if client is None:
        return {
            "substitution_candidates": [],
            "substitution_summary": None,
            "substitution_status": "unavailable",
            "can_make_with_subs": False,
        }
    kitchen_items = [
        item.get("display_name")
        for item in ((kitchen_context or {}).get("kitchen_candidates") or [])
        if item.get("display_name")
    ]
    if not kitchen_items:
        return {
            "substitution_candidates": [],
            "substitution_summary": None,
            "substitution_status": None,
            "can_make_with_subs": False,
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
        "recipe_title": recipe.get("title"),
        "recipe_ingredients": recipe.get("ingredients") or [],
        "missing_ingredients": missing_ingredients,
        "kitchen_items": kitchen_items,
    })
    try:
        response = client.chat.completions.create(
            model=OPENAI_SUBSTITUTION_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        payload = _best_effort_json_parse((response.choices[0].message.content or "").strip(), {"substitutions": [], "summary": None})
    except Exception as exc:
        print(f"explore substitution suggestion failed: {exc}")
        return {
            "substitution_candidates": [],
            "substitution_summary": None,
            "substitution_status": "failed",
            "can_make_with_subs": False,
        }
    substitutions = payload.get("substitutions") or []
    normalized = []
    covered_missing = set()
    for item in substitutions:
        missing_ingredient = _safe_text((item or {}).get("missing_ingredient"))
        use_instead = _safe_text((item or {}).get("use_instead"))
        if not missing_ingredient or not use_instead:
            continue
        covered_missing.add(missing_ingredient.lower())
        normalized.append({
            "missing_ingredient": missing_ingredient,
            "use_instead": use_instead,
            "confidence": _safe_text((item or {}).get("confidence") or "medium").lower() or "medium",
            "notes": _safe_text((item or {}).get("notes")) or None,
        })
    return {
        "substitution_candidates": normalized,
        "substitution_summary": payload.get("summary"),
        "substitution_status": "ready" if normalized else "none",
        "can_make_with_subs": bool(normalized) and len(covered_missing) >= len({item.lower() for item in missing_ingredients}),
    }


def _build_overlay_row(owner, recipe_id, availability):
    return (
        _safe_owner(owner),
        "explore",
        str(recipe_id or ""),
        int((availability or {}).get("kitchen_version") or 0),
        1 if (availability or {}).get("can_make_exact") else 0,
        1 if (availability or {}).get("can_make_with_subs") else 0,
        int((availability or {}).get("matched_count") or 0),
        int((availability or {}).get("missing_count") or 0),
        json.dumps((availability or {}).get("ingredient_matches") or []),
        json.dumps((availability or {}).get("missing_ingredients") or []),
        json.dumps((availability or {}).get("substitution_candidates") or []),
        (availability or {}).get("substitution_summary"),
        (availability or {}).get("substitution_status"),
        str((availability or {}).get("analysis_status") or "ready"),
    )


def _persist_overlay_rows(conn, rows):
    rows = [row for row in (rows or []) if row[0] and row[2]]
    if not rows:
        return
    _ensure_owner_recipe_availability_table(conn)
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO `{OWNER_RECIPE_AVAILABILITY_TABLE}` (
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
            rows,
        )
    conn.commit()


def _serialize_recipe_with_owner_context(recipe, owner, conn, kitchen_context=None, availability=None, request_id=None):
    serialized = _serialize_row(recipe)
    kitchen_context = kitchen_context or _build_kitchen_context(conn, owner)
    availability = availability or _compute_recipe_availability(serialized, kitchen_context, request_id=request_id)
    serialized["availability"] = {
        "kitchen_version": availability["kitchen_version"],
        "can_make_exact": availability["can_make_exact"],
        "can_make_with_subs": availability["can_make_with_subs"],
        "matched_count": availability["matched_count"],
        "missing_count": availability["missing_count"],
    }
    serialized["ingredient_matches"] = availability["ingredient_matches"]
    serialized["missing_ingredients"] = availability["missing_ingredients"]
    serialized["substitution_summary"] = availability["substitution_summary"]
    serialized["substitution_status"] = availability["substitution_status"]
    overlay_row = _build_overlay_row(owner, serialized.get("id"), availability)
    return serialized, overlay_row


# -------- personalize endpoint --------


def personalize_recipes(body):
    """Lightweight endpoint: compute availability for a small batch of recipe IDs."""
    owner = _safe_owner((body or {}).get("owner"))
    recipe_ids = (body or {}).get("recipe_ids") or []
    if not owner:
        return _err(400, "owner is required")
    if not recipe_ids or not isinstance(recipe_ids, list):
        return _err(400, "recipe_ids is required (list)")
    # Cap batch size to prevent abuse
    recipe_ids = [str(rid) for rid in recipe_ids[:10]]

    conn = _mysql_conn()
    try:
        _ensure_personalization_tables(conn)
        kitchen_context = _build_kitchen_context(conn, owner)

        # Fetch the requested recipes from DB
        placeholders = ", ".join(["%s"] * len(recipe_ids))
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, title, ingredients
                    FROM explore_recipes
                    WHERE id IN ({placeholders}) AND status = 'ready'""",
                recipe_ids,
            )
            rows = cur.fetchall()

        if not rows:
            return _ok({"results": {}})

        base_recipes = [_serialize_row(row) for row in rows]
        availability_map = _compute_explore_recipe_availability_batch(base_recipes, kitchen_context)

        results = {}
        overlay_rows = []
        for recipe in base_recipes:
            rid = recipe.get("id")
            avail = availability_map.get(rid)
            if not avail:
                continue
            results[rid] = {
                "can_make_exact": avail.get("can_make_exact", False),
                "can_make_with_subs": avail.get("can_make_with_subs", False),
                "matched_count": avail.get("matched_count", 0),
                "missing_count": avail.get("missing_count", 0),
            }
            overlay_rows.append(_build_overlay_row(owner, rid, avail))

        _persist_overlay_rows(conn, overlay_rows)
        return _ok({"results": results})
    finally:
        conn.close()


# -------- read endpoints --------


def list_recipes(qs):
    conn = _mysql_conn()
    try:
        _ensure_category_column(conn)
        owner = _safe_owner((qs or {}).get("owner"))
        limit = _parse_int((qs or {}).get("limit"), 20, maximum=100) or 20
        offset = _parse_int((qs or {}).get("offset"), 0)
        creator = (qs or {}).get("creator")
        platform = (qs or {}).get("platform")
        category = (qs or {}).get("category")
        sort = (qs or {}).get("sort") or "recent"

        where = ["status = %s"]
        params = ["ready"]
        if creator:
            where.append("creator_username = %s")
            params.append(creator.lstrip("@"))
        if platform:
            where.append("source_platform = %s")
            params.append(platform)
        if category:
            where.append("category = %s")
            params.append(category)

        order = (
            "likes_count DESC, post_timestamp DESC"
            if sort == "popular"
            else "post_timestamp DESC, _createdDate DESC"
        )

        sql = f"""
            SELECT id, source_platform, source_url, shortcode, post_type,
                   post_timestamp, likes_count, comments_count,
                   creator_username, creator_name, creator_url,
                   title, image_url, image_urls,
                   ingredients, instructions, notes,
                   extraction_source, category, _createdDate
            FROM explore_recipes
            WHERE {' AND '.join(where)}
            ORDER BY {order}
            LIMIT %s OFFSET %s
        """
        count_sql = (
            f"SELECT COUNT(*) AS n FROM explore_recipes WHERE {' AND '.join(where)}"
        )
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()["n"]
            cur.execute(sql, params + [limit, offset])
            rows = cur.fetchall()
        if owner:
            _ensure_personalization_tables(conn)
            kitchen_context = _build_kitchen_context(conn, owner)
            base_recipes = [_serialize_row(row) for row in rows]
            availability_map = _compute_explore_recipe_availability_batch(base_recipes, kitchen_context)
            serialized_rows = [
                _serialize_recipe_with_owner_context(
                    row,
                    owner,
                    conn,
                    kitchen_context=kitchen_context,
                    availability=availability_map.get((base_recipe or {}).get("id")),
                )
                for row, base_recipe in zip(rows, base_recipes)
            ]
            _persist_overlay_rows(conn, [overlay_row for _, overlay_row in serialized_rows])
            recipes = [recipe for recipe, _ in serialized_rows]
        else:
            recipes = [_serialize_row(r) for r in rows]
        return _ok(
            {
                "total": total,
                "limit": limit,
                "offset": offset,
                "recipes": recipes,
            }
        )
    finally:
        conn.close()


def get_recipe(recipe_id, qs=None):
    if not recipe_id:
        return _err(400, "Missing recipe id")
    conn = _mysql_conn()
    try:
        owner = _safe_owner((qs or {}).get("owner"))
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM explore_recipes WHERE id = %s LIMIT 1",
                [recipe_id],
            )
            row = cur.fetchone()
        if not row:
            return _err(404, "Recipe not found")
        if owner:
            _ensure_personalization_tables(conn)
            recipe, overlay_row = _serialize_recipe_with_owner_context(row, owner, conn, request_id=recipe_id)
            _persist_overlay_rows(conn, [overlay_row])
            return _ok({"recipe": recipe})
        return _ok({"recipe": _serialize_row(row)})
    finally:
        conn.close()


def list_creators(qs):
    conn = _mysql_conn()
    try:
        limit = _parse_int((qs or {}).get("limit"), 50, maximum=200) or 50
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT creator_username,
                       MAX(creator_name) AS creator_name,
                       MAX(creator_url) AS creator_url,
                       COUNT(*) AS recipe_count,
                       SUM(likes_count) AS total_likes,
                       MAX(post_timestamp) AS latest_post_ts
                FROM explore_recipes
                WHERE status = 'ready' AND creator_username IS NOT NULL
                GROUP BY creator_username
                ORDER BY recipe_count DESC
                LIMIT %s
                """,
                [limit],
            )
            rows = cur.fetchall()
        return _ok({"creators": rows, "count": len(rows)})
    finally:
        conn.close()


def get_creator(username, qs):
    uname = (username or "").lstrip("@")
    if not uname:
        return _err(400, "Missing username")
    conn = _mysql_conn()
    try:
        owner = _safe_owner((qs or {}).get("owner"))
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT creator_username,
                       MAX(creator_name) AS creator_name,
                       MAX(creator_url) AS creator_url,
                       COUNT(*) AS recipe_count,
                       SUM(likes_count) AS total_likes,
                       MAX(post_timestamp) AS latest_post_ts
                FROM explore_recipes
                WHERE creator_username = %s AND status = 'ready'
                GROUP BY creator_username
                """,
                [uname],
            )
            profile = cur.fetchone()
            if not profile:
                return _err(404, "Creator not found")

            limit = _parse_int((qs or {}).get("limit"), 20, maximum=100) or 20
            offset = _parse_int((qs or {}).get("offset"), 0)
            cur.execute(
                """
                SELECT id, title, image_url, shortcode, source_url,
                       post_timestamp, likes_count, comments_count, ingredients
                FROM explore_recipes
                WHERE creator_username = %s AND status = 'ready'
                ORDER BY post_timestamp DESC
                LIMIT %s OFFSET %s
                """,
                [uname, limit, offset],
            )
            recipes = cur.fetchall()
        if owner:
            _ensure_personalization_tables(conn)
            kitchen_context = _build_kitchen_context(conn, owner)
            base_recipes = [_serialize_row(row) for row in recipes]
            availability_map = _compute_explore_recipe_availability_batch(base_recipes, kitchen_context)
            serialized_rows = [
                _serialize_recipe_with_owner_context(
                    row,
                    owner,
                    conn,
                    kitchen_context=kitchen_context,
                    availability=availability_map.get((base_recipe or {}).get("id")),
                )
                for row, base_recipe in zip(recipes, base_recipes)
            ]
            _persist_overlay_rows(conn, [overlay_row for _, overlay_row in serialized_rows])
            recipes = [recipe for recipe, _ in serialized_rows]
        return _ok({"creator": profile, "recipes": recipes})
    finally:
        conn.close()


def recipe_index(qs):
    """Lightweight index of all recipes — id, title, image_url, creator_username.
    Used by the iOS client for instant client-side search."""
    conn = _mysql_conn()
    try:
        sql = """
            SELECT id, title, image_url, creator_username
            FROM explore_recipes
            WHERE status = 'ready'
            ORDER BY likes_count DESC, post_timestamp DESC
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        items = [
            {
                "id": row["id"],
                "title": _safe_text(row.get("title")),
                "image_url": row.get("image_url"),
                "creator_username": _safe_text(row.get("creator_username")),
            }
            for row in rows
        ]
        return _ok({"recipes": items, "total": len(items)})
    finally:
        conn.close()


def search_recipes(qs):
    q = ((qs or {}).get("q") or "").strip()
    if not q:
        return _err(400, "Missing q")
    conn = _mysql_conn()
    try:
        owner = _safe_owner((qs or {}).get("owner"))
        limit = _parse_int((qs or {}).get("limit"), 20, maximum=100) or 20
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, image_url, creator_username, source_url,
                       post_timestamp, likes_count, ingredients,
                       MATCH(title, raw_caption) AGAINST (%s IN NATURAL LANGUAGE MODE) AS score
                FROM explore_recipes
                WHERE status = 'ready'
                  AND MATCH(title, raw_caption) AGAINST (%s IN NATURAL LANGUAGE MODE)
                ORDER BY score DESC
                LIMIT %s
                """,
                [q, q, limit],
            )
            rows = cur.fetchall()
        if owner:
            _ensure_personalization_tables(conn)
            kitchen_context = _build_kitchen_context(conn, owner)
            base_rows = [_serialize_row(row) for row in rows]
            availability_map = _compute_explore_recipe_availability_batch(base_rows, kitchen_context)
            serialized_rows = [
                _serialize_recipe_with_owner_context(
                    row,
                    owner,
                    conn,
                    kitchen_context=kitchen_context,
                    availability=availability_map.get((base_row or {}).get("id")),
                )
                for row, base_row in zip(rows, base_rows)
            ]
            _persist_overlay_rows(conn, [overlay_row for _, overlay_row in serialized_rows])
            rows = [recipe for recipe, _ in serialized_rows]
        return _ok({"query": q, "results": rows})
    finally:
        conn.close()


# -------- write endpoint (ingest) --------


def _get_header(event, name):
    headers = event.get("headers") or {}
    return headers.get(name) or headers.get(name.lower()) or headers.get(name.upper())


def ingest(event, body):
    provided = _get_header(event, "X-Api-Key")
    if not INGEST_API_KEY or provided != INGEST_API_KEY:
        return _err(401, "Unauthorized")

    recipes = (body or {}).get("recipes") or []
    if not isinstance(recipes, list):
        return _err(400, 'Expected {"recipes": [...]}')

    inserted = 0
    deduped = 0
    errors = []
    conn = _mysql_conn()
    try:
        _ensure_category_column(conn)
        with conn.cursor() as cur:
            for idx, r in enumerate(recipes):
                try:
                    resolved_url_hash = r["resolved_url_hash"]
                    cur.execute(
                        "SELECT id FROM explore_recipes WHERE resolved_url_hash = %s",
                        [resolved_url_hash],
                    )
                    if cur.fetchone():
                        # Already stored — refresh mutable curation fields (category).
                        if r.get("category") is not None:
                            cur.execute(
                                "UPDATE explore_recipes SET category = %s WHERE resolved_url_hash = %s",
                                [r.get("category"), resolved_url_hash],
                            )
                        deduped += 1
                        continue

                    row_id = r.get("id") or str(uuid.uuid4())
                    original = r.get("original_image_url") or r.get("image_url")
                    image_url = r.get("image_url")

                    if r.get("thumbnail_bytes_b64") and THUMBNAIL_BUCKET:
                        content_type = r.get("thumbnail_content_type", "image/jpeg")
                        ext = "jpg"
                        if "png" in content_type:
                            ext = "png"
                        elif "webp" in content_type:
                            ext = "webp"
                        key = f"recipes/{row_id}.{ext}"
                        s3.put_object(
                            Bucket=THUMBNAIL_BUCKET,
                            Key=key,
                            Body=base64.b64decode(r["thumbnail_bytes_b64"]),
                            ContentType=content_type,
                            CacheControl="public, max-age=31536000, immutable",
                        )
                        image_url = f"{THUMBNAIL_CDN_BASE}/{key}"

                    cur.execute(
                        """
                        INSERT INTO explore_recipes (
                            id, source_platform, source_domain, source_url, resolved_url,
                            resolved_url_hash, shortcode, post_type, post_timestamp,
                            likes_count, comments_count, video_duration,
                            creator_username, creator_name, creator_url,
                            title, image_url, original_image_url, image_urls,
                            ingredients, instructions, notes,
                            raw_caption, extraction_source, refine_model, status, category
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        [
                            row_id,
                            r["source_platform"],
                            r.get("source_domain"),
                            r["source_url"],
                            r["resolved_url"],
                            resolved_url_hash,
                            r.get("shortcode"),
                            r.get("post_type"),
                            r.get("post_timestamp"),
                            r.get("likes_count"),
                            r.get("comments_count"),
                            r.get("video_duration"),
                            (r.get("creator_username") or "").lstrip("@") or None,
                            r.get("creator_name"),
                            r.get("creator_url"),
                            (r.get("title") or "Untitled")[:255],
                            image_url,
                            original,
                            json.dumps(r.get("image_urls") or []),
                            json.dumps(r.get("ingredients") or []),
                            json.dumps(r.get("instructions") or []),
                            json.dumps(r.get("notes") or []),
                            r.get("raw_caption"),
                            r.get("extraction_source"),
                            r.get("refine_model"),
                            r.get("status") or "ready",
                            r.get("category"),
                        ],
                    )
                    inserted += 1
                except Exception as exc:
                    errors.append({"index": idx, "error": str(exc)})
        conn.commit()
    finally:
        conn.close()

    return _ok(
        {
            "inserted": inserted,
            "deduped": deduped,
            "errors": errors[:10],
            "error_count": len(errors),
            "received": len(recipes),
        }
    )


def delete_by_creator(event, body):
    provided = _get_header(event, "X-Api-Key")
    if not INGEST_API_KEY or provided != INGEST_API_KEY:
        return _err(401, "Unauthorized")

    creator = (body or {}).get("creator_username", "").strip().lstrip("@")
    if not creator:
        return _err(400, 'Expected {"creator_username": "..."}')

    conn = _mysql_conn()
    try:
        with conn.cursor() as cur:
            # Get S3 keys to clean up thumbnails
            cur.execute(
                "SELECT id, image_url FROM explore_recipes WHERE creator_username = %s",
                [creator],
            )
            rows = cur.fetchall()
            s3_keys = []
            for row in rows:
                img = row.get("image_url") or ""
                if THUMBNAIL_BUCKET and THUMBNAIL_CDN_BASE and THUMBNAIL_CDN_BASE in img:
                    key = img.split(THUMBNAIL_CDN_BASE + "/", 1)[-1] if THUMBNAIL_CDN_BASE in img else None
                    if key:
                        s3_keys.append(key)

            cur.execute(
                "DELETE FROM explore_recipes WHERE creator_username = %s",
                [creator],
            )
            deleted = cur.rowcount
        conn.commit()

        # Clean up S3 thumbnails
        for key in s3_keys:
            try:
                s3.delete_object(Bucket=THUMBNAIL_BUCKET, Key=key)
            except Exception:
                pass

    finally:
        conn.close()

    return _ok({"deleted": deleted, "creator_username": creator})


# -------- router --------


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    raw_path = event.get("rawPath") or ""
    path_params = event.get("pathParameters") or {}
    query = event.get("queryStringParameters") or {}

    if method == "OPTIONS":
        return _ok({}, status=204)

    try:
        if raw_path == "/explore/recipes" and method == "GET":
            return list_recipes(query)
        if raw_path == "/explore/recipes/index" and method == "GET":
            return recipe_index(query)
        if raw_path.startswith("/explore/recipes/") and method == "GET":
            return get_recipe(path_params.get("recipe_id") or "", query)
        if raw_path == "/explore/creators" and method == "GET":
            return list_creators(query)
        if raw_path.startswith("/explore/creators/") and method == "GET":
            return get_creator(path_params.get("username") or "", query)
        if raw_path == "/explore/search" and method == "GET":
            return search_recipes(query)
        if raw_path == "/explore/recipes/personalize" and method == "POST":
            body = json.loads(event.get("body") or "{}")
            return personalize_recipes(body)
        if raw_path == "/explore/ingest" and method == "POST":
            body = json.loads(event.get("body") or "{}")
            return ingest(event, body)
        if raw_path == "/explore/delete" and method == "POST":
            body = json.loads(event.get("body") or "{}")
            return delete_by_creator(event, body)
        return _err(404, f"Unknown route {method} {raw_path}")
    except Exception as exc:
        print(f"handler error: {exc}")
        return _err(500, "Internal error")
