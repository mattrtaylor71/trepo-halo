import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape

from openai import OpenAI


INGREDIENT_NOISE_TOKENS = {
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
PANTRY_CANONICAL_INGREDIENTS = {
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
PANTRY_TOKENS = {'salt', 'pepper', 'oil', 'water', 'ice', 'butter', 'spray',
                 'sugar', 'flour', 'spice', 'spices', 'seasoning', 'seasonings'}
INGREDIENT_CONFLICT_TOKENS = {
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
CATEGORY_VARIETY_TOKENS = {
    'cheese': {
        'parmesan', 'parmigiano', 'reggiano', 'cheddar', 'mozzarella', 'feta', 'gouda',
        'brie', 'provolone', 'gruyere', 'asiago', 'romano', 'pecorino', 'ricotta',
        'mascarpone', 'manchego', 'colby', 'swiss', 'havarti', 'gorgonzola', 'fontina',
        'halloumi', 'paneer', 'cotija', 'burrata', 'camembert', 'edam', 'emmental',
        'jarlsberg', 'muenster', 'queso',
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


DEFAULT_MATCH_TYPE = 'llm_match'


def _safe_text(value):
    return str(value or '').strip()


def _best_effort_json_parse(text, fallback):
    cleaned = _safe_text(text)
    if not cleaned:
        return fallback
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


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


def ingredient_tokens(text):
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
        if not token or token in INGREDIENT_NOISE_TOKENS:
            continue
        tokens.append(token)
    return tokens


def canonical_ingredient_name(text):
    return ' '.join(ingredient_tokens(text))


def is_pantry_ingredient(text):
    canonical = canonical_ingredient_name(text)
    if not canonical:
        return False
    if canonical in PANTRY_CANONICAL_INGREDIENTS:
        return True
    tokens = canonical.split()
    return bool(tokens) and all(token in PANTRY_TOKENS for token in tokens)


def llm_inventory_matching_enabled():
    return _safe_text(os.getenv('ENABLE_LLM_INVENTORY_MATCHING', 'false')).lower() in {'1', 'true', 'yes', 'on'}


def _inventory_model():
    return (
        _safe_text(os.getenv('OPENAI_INVENTORY_MODEL'))
        or _safe_text(os.getenv('OPENAI_SUBSTITUTION_MODEL'))
        or _safe_text(os.getenv('OPENAI_MODEL'))
        or 'gpt-4.1-mini'
    )


def _chat_temperature_kwargs(model):
    # gpt-5.x / o-series reject a non-default temperature; omit it for them.
    m = (model or '').lower()
    if m.startswith('gpt-5') or m.startswith('o1') or m.startswith('o3') or m.startswith('o4'):
        return {}
    return {'temperature': 0}


def _openai_client():
    api_key = _safe_text(os.getenv('OPENAI_API_KEY'))
    if not api_key:
        return None
    timeout_seconds = int(os.getenv('OPENAI_INVENTORY_TIMEOUT_SECONDS', os.getenv('OPENAI_TIMEOUT_SECONDS', '12')))
    max_retries = int(os.getenv('OPENAI_INVENTORY_MAX_RETRIES', '0'))
    return OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)


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
    # Catches cases where recipe has extra prep/quantity words (e.g. "1 pound chicken breast cut into chunks" vs "Chicken Breasts")
    if candidate_set.issubset(recipe_set):
        recipe_extra = recipe_set - candidate_set
        conflict_in_extra = strip_redundant_category_conflicts(
            {t for t in recipe_extra if t in INGREDIENT_CONFLICT_TOKENS}, context_tokens)
        if not conflict_in_extra:
            return 150 + (len(candidate_set) * 10) - max(0, len(recipe_set) - len(candidate_set))
    return -1


def _deterministic_ingredient_match(ingredient_text, kitchen_candidates):
    canonical = canonical_ingredient_name(ingredient_text)
    if is_pantry_ingredient(ingredient_text):
        return {
            'recipe_ingredient': ingredient_text,
            'canonical_ingredient': canonical,
            'match_status': 'pantry',
            'matched_kitchen_items': [],
            'substitute_kitchen_items': [],
            'missing_reason': None,
        }
    recipe_tokens = ingredient_tokens(ingredient_text)
    best_candidate = None
    best_score = -1
    for candidate in kitchen_candidates or []:
        score = _score_kitchen_candidate(recipe_tokens, candidate.get('tokens') or [])
        if score > best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate:
        return {
            'recipe_ingredient': ingredient_text,
            'canonical_ingredient': canonical,
            'match_status': 'have',
            'matched_kitchen_items': [{
                'item_id': best_candidate.get('item_id'),
                'display_name': best_candidate.get('display_name'),
                'match_type': 'canonical_exact' if set(recipe_tokens) == set(best_candidate.get('tokens') or []) else 'canonical_subset',
            }],
            'substitute_kitchen_items': [],
            'missing_reason': None,
        }
    return {
        'recipe_ingredient': ingredient_text,
        'canonical_ingredient': canonical,
        'match_status': 'missing',
        'matched_kitchen_items': [],
        'substitute_kitchen_items': [],
        'missing_reason': 'No direct kitchen match found.',
    }


def _apply_substitution_callback(recipe, kitchen_context, availability, substitution_callback):
    if not substitution_callback:
        availability['substitution_candidates'] = []
        availability['substitution_summary'] = None
        availability['substitution_status'] = None
        availability['can_make_with_subs'] = False
        return availability
    substitution = substitution_callback(recipe, kitchen_context, availability.get('missing_ingredients') or [])
    availability['can_make_with_subs'] = bool((substitution or {}).get('can_make_with_subs'))
    availability['substitution_candidates'] = (substitution or {}).get('substitution_candidates') or []
    availability['substitution_summary'] = (substitution or {}).get('substitution_summary')
    availability['substitution_status'] = (substitution or {}).get('substitution_status')
    return availability


def deterministic_availability(recipe, kitchen_context, substitution_callback=None):
    recipe = dict(recipe or {})
    ingredient_matches = []
    missing_ingredients = []
    matched_count = 0
    missing_count = 0
    kitchen_candidates = (kitchen_context or {}).get('kitchen_candidates') or []
    for ingredient in [_safe_text(item) for item in (recipe.get('ingredients') or []) if _safe_text(item)]:
        match = _deterministic_ingredient_match(ingredient, kitchen_candidates)
        ingredient_matches.append(match)
        if match.get('match_status') in {'have', 'pantry'}:
            matched_count += 1
        else:
            missing_count += 1
            missing_ingredients.append(ingredient)
    availability = {
        'kitchen_version': int((kitchen_context or {}).get('kitchen_version') or 0),
        'can_make_exact': missing_count == 0,
        'can_make_with_subs': False,
        'matched_count': matched_count,
        'missing_count': missing_count,
        'ingredient_matches': ingredient_matches,
        'missing_ingredients': missing_ingredients,
        'substitution_candidates': [],
        'substitution_summary': None,
        'substitution_status': None,
        'analysis_status': 'ready',
    }
    return _apply_substitution_callback(recipe, kitchen_context, availability, substitution_callback)


def _recipe_payload(recipes):
    payload = []
    for recipe in recipes or []:
        recipe_id = _safe_text((recipe or {}).get('id') or (recipe or {}).get('_id'))
        if not recipe_id:
            continue
        ingredients = [_safe_text(item) for item in ((recipe or {}).get('ingredients') or []) if _safe_text(item)]
        payload.append({
            'recipe_id': recipe_id,
            'title': _safe_text((recipe or {}).get('title')),
            'ingredients': ingredients,
        })
    return payload


def _kitchen_payload(kitchen_context):
    items = []
    for candidate in (kitchen_context or {}).get('kitchen_candidates') or []:
        display_name = _safe_text((candidate or {}).get('display_name'))
        if not display_name:
            continue
        items.append({
            'item_id': _safe_text((candidate or {}).get('item_id')) or None,
            'display_name': display_name,
            'description': _safe_text((candidate or {}).get('description')) or None,
        })
    return items


def _kitchen_index(kitchen_context):
    by_item_id = {}
    by_display_name = {}
    for candidate in (kitchen_context or {}).get('kitchen_candidates') or []:
        item_id = _safe_text((candidate or {}).get('item_id'))
        display_name = _safe_text((candidate or {}).get('display_name'))
        if item_id:
            by_item_id[item_id] = candidate
        if display_name:
            by_display_name[display_name.lower()] = candidate
    return by_item_id, by_display_name


def _normalize_kitchen_refs(raw_refs, kitchen_context, default_match_type=None):
    by_item_id, by_display_name = _kitchen_index(kitchen_context)
    normalized = []
    seen = set()
    for raw_ref in raw_refs or []:
        item_id = _safe_text((raw_ref or {}).get('item_id'))
        display_name = _safe_text((raw_ref or {}).get('display_name'))
        candidate = None
        if item_id:
            candidate = by_item_id.get(item_id)
        if candidate is None and display_name:
            candidate = by_display_name.get(display_name.lower())
        if candidate is None and not display_name:
            continue
        final_item_id = item_id or _safe_text((candidate or {}).get('item_id')) or None
        final_display_name = display_name or _safe_text((candidate or {}).get('display_name'))
        if not final_display_name:
            continue
        dedupe_key = (final_item_id or '', final_display_name.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_ref = {
            'display_name': final_display_name,
        }
        if final_item_id:
            normalized_ref['item_id'] = final_item_id
        match_type = _safe_text((raw_ref or {}).get('match_type')) or default_match_type
        if match_type:
            normalized_ref['match_type'] = match_type
        normalized.append(normalized_ref)
    return normalized


def _system_prompt():
    return """You analyze whether a user can make recipes from the items already in their kitchen.
Return valid JSON only with this schema:
{
  "recipes": [
    {
      "recipe_id": "string",
      "ingredient_matches": [
        {
          "recipe_ingredient": "string",
          "canonical_ingredient": "short normalized name",
          "match_status": "have|pantry|missing",
          "matched_kitchen_items": [
            {
              "item_id": "string or null",
              "display_name": "string",
              "match_type": "llm_exact|llm_variant|llm_form"
            }
          ],
          "substitute_kitchen_items": [
            {
              "item_id": "string or null",
              "display_name": "string"
            }
          ],
          "missing_reason": "string or null"
        }
      ],
      "substitution_candidates": [
        {
          "missing_ingredient": "string",
          "use_instead": "string",
          "confidence": "high|medium|low",
          "notes": "string or null"
        }
      ],
      "substitution_summary": "string or null",
      "substitution_status": "ready|none",
      "can_make_with_subs": true
    }
  ]
}
Rules:
- Pantry staples should be marked as pantry when reasonable, including water, salt, pepper, olive oil, neutral oil, canola oil, butter, cooking spray, and basic spices or seasonings.
- Do not make false-positive noun swaps. Example: frozen cherries is not the same as cherry tomatoes.
- Only mark have when the kitchen item clearly satisfies the ingredient.
- If an ingredient is missing but a kitchen item is a plausible substitution, keep match_status as missing and list the substitute in substitute_kitchen_items and substitution_candidates.
- Preserve each recipe_ingredient exactly as written in the recipe input.
- Return one result object for every recipe_id in the input."""


def _compact_system_prompt():
    return """Match recipe ingredients to kitchen items. Return compact JSON.
For each recipe, return an array "m" with one entry per ingredient (same order as input).
Each entry: [status, kitchen_item_name_or_null, sub_kitchen_item_or_null]
- status: "h" = have (kitchen has it), "p" = pantry staple, "m" = missing
- kitchen_item_name: exact display_name from the kitchen list (only for "h")
- sub_kitchen_item: display_name of a plausible substitute (only for "m", optional)

Format: {"r":{"recipe_id":{"m":[["h","Chicken Breasts"],["p"],["m",null,"Greek Yogurt"],["m"]]}}}

Rules:
- Pantry staples (salt, pepper, oil, water, butter, cooking spray, basic spices/seasonings) → "p"
- Use "h" when a kitchen item IS the ingredient, INCLUDING the same food under a different name, brand, or regional/equivalent form. Same-ingredient examples that MUST be "h": tamari = soy sauce; scallion = green onion; cilantro = coriander; garbanzo = chickpea; prawns = shrimp; passata = tomato sauce; confectioners sugar = powdered sugar. Put that kitchen item's exact display_name.
- "m" with a substitute is ONLY for a genuinely DIFFERENT ingredient that could stand in (e.g., Greek yogurt for sour cream). Never demote a true same-ingredient match to a substitute.
- Still no false-positive noun swaps between DIFFERENT foods (frozen cherries ≠ cherry tomatoes; green onion ≠ green bell pepper)
- One entry per ingredient, same order as input"""


def _normalize_llm_recipe(recipe, raw_recipe, kitchen_context):
    ingredients = [_safe_text(item) for item in ((recipe or {}).get('ingredients') or []) if _safe_text(item)]
    raw_matches = list((raw_recipe or {}).get('ingredient_matches') or [])
    raw_by_ingredient = {
        _safe_text((match or {}).get('recipe_ingredient')).lower(): (match or {})
        for match in raw_matches
        if _safe_text((match or {}).get('recipe_ingredient'))
    }
    normalized_matches = []
    missing_ingredients = []
    matched_count = 0
    missing_count = 0
    for ingredient in ingredients:
        raw_match = raw_by_ingredient.get(ingredient.lower())
        if not raw_match:
            return None
        match_status = _safe_text((raw_match or {}).get('match_status')).lower()
        if match_status not in {'have', 'pantry', 'missing'}:
            return None
        normalized_match = {
            'recipe_ingredient': ingredient,
            'canonical_ingredient': _safe_text((raw_match or {}).get('canonical_ingredient')) or canonical_ingredient_name(ingredient),
            'match_status': match_status,
            'matched_kitchen_items': _normalize_kitchen_refs(
                (raw_match or {}).get('matched_kitchen_items') or [],
                kitchen_context,
                default_match_type=DEFAULT_MATCH_TYPE,
            ),
            'substitute_kitchen_items': _normalize_kitchen_refs(
                (raw_match or {}).get('substitute_kitchen_items') or [],
                kitchen_context,
            ),
            'missing_reason': _safe_text((raw_match or {}).get('missing_reason')) or None,
        }
        if is_pantry_ingredient(ingredient):
            normalized_match['match_status'] = 'pantry'
            normalized_match['matched_kitchen_items'] = []
            normalized_match['missing_reason'] = None
            match_status = 'pantry'
        if match_status == 'have' and not normalized_match['matched_kitchen_items']:
            return None
        if match_status in {'have', 'pantry'}:
            matched_count += 1
        else:
            missing_count += 1
            missing_ingredients.append(ingredient)
            if not normalized_match['missing_reason']:
                normalized_match['missing_reason'] = 'No direct kitchen match found.'
        normalized_matches.append(normalized_match)

    raw_candidates = []
    covered_missing = set()
    for candidate in (raw_recipe or {}).get('substitution_candidates') or []:
        missing_ingredient = _safe_text((candidate or {}).get('missing_ingredient'))
        use_instead = _safe_text((candidate or {}).get('use_instead'))
        if not missing_ingredient or not use_instead:
            continue
        raw_candidates.append({
            'missing_ingredient': missing_ingredient,
            'use_instead': use_instead,
            'confidence': _safe_text((candidate or {}).get('confidence') or 'medium').lower() or 'medium',
            'notes': _safe_text((candidate or {}).get('notes')) or None,
        })
        covered_missing.add(missing_ingredient.lower())
    substitution_status = _safe_text((raw_recipe or {}).get('substitution_status')).lower()
    if substitution_status not in {'ready', 'none'}:
        substitution_status = 'ready' if raw_candidates else 'none'
    can_make_with_subs = bool((raw_recipe or {}).get('can_make_with_subs'))
    if raw_candidates and len(covered_missing) >= len({item.lower() for item in missing_ingredients}):
        can_make_with_subs = True
    return {
        'kitchen_version': int((kitchen_context or {}).get('kitchen_version') or 0),
        'can_make_exact': missing_count == 0,
        'can_make_with_subs': can_make_with_subs,
        'matched_count': matched_count,
        'missing_count': missing_count,
        'ingredient_matches': normalized_matches,
        'missing_ingredients': missing_ingredients,
        'substitution_candidates': raw_candidates,
        'substitution_summary': _safe_text((raw_recipe or {}).get('substitution_summary')) or None,
        'substitution_status': substitution_status,
        'analysis_status': 'ready',
    }


def match_recipes(recipes, kitchen_context, request_id=None, log_fn=None, source_label='inventory', substitution_callback=None):
    recipes = [dict(recipe or {}) for recipe in (recipes or []) if _safe_text((recipe or {}).get('id') or (recipe or {}).get('_id'))]
    meta = {
        'used_llm': False,
        'used_fallback': False,
        'recipe_count': len(recipes),
        'latency_ms': 0,
        'prompt_tokens': None,
        'completion_tokens': None,
        'error': None,
    }
    if not recipes:
        return {}, meta

    def fallback(error_message=None):
        if error_message:
            meta['error'] = error_message
        meta['used_fallback'] = True
        results = {}
        for recipe in recipes:
            recipe_id = _safe_text(recipe.get('id') or recipe.get('_id'))
            results[recipe_id] = deterministic_availability(recipe, kitchen_context, substitution_callback=substitution_callback)
        if log_fn:
            log_fn(
                request_id,
                'llm_inventory_match_fallback',
                source=source_label,
                recipe_count=len(recipes),
                error=error_message,
            )
        return results, meta

    if not llm_inventory_matching_enabled():
        return fallback('llm_inventory_matching_disabled')

    client = _openai_client()
    if client is None:
        return fallback('openai_client_unavailable')

    payload = {
        'kitchen_items': _kitchen_payload(kitchen_context),
        'recipes': _recipe_payload(recipes),
    }
    started_at = time.monotonic()
    try:
        _model = _inventory_model()
        response = client.chat.completions.create(
            model=_model,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': _system_prompt()},
                {'role': 'user', 'content': json.dumps(payload)},
            ],
            **_chat_temperature_kwargs(_model),
        )
        meta['latency_ms'] = int((time.monotonic() - started_at) * 1000)
        meta['used_llm'] = True
        usage = getattr(response, 'usage', None)
        if usage is not None:
            meta['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
            meta['completion_tokens'] = getattr(usage, 'completion_tokens', None)
        body = _best_effort_json_parse(_safe_text(response.choices[0].message.content), {'recipes': []})
        raw_recipes = {
            _safe_text((item or {}).get('recipe_id')): (item or {})
            for item in (body.get('recipes') or [])
            if _safe_text((item or {}).get('recipe_id'))
        }
        normalized = {}
        for recipe in recipes:
            recipe_id = _safe_text(recipe.get('id') or recipe.get('_id'))
            llm_recipe = raw_recipes.get(recipe_id)
            if not llm_recipe:
                return fallback(f'missing_recipe_result:{recipe_id}')
            normalized_recipe = _normalize_llm_recipe(recipe, llm_recipe, kitchen_context)
            if normalized_recipe is None:
                return fallback(f'invalid_recipe_result:{recipe_id}')
            normalized[recipe_id] = normalized_recipe
        if log_fn:
            log_fn(
                request_id,
                'llm_inventory_match_completed',
                source=source_label,
                recipe_count=len(recipes),
                latency_ms=meta['latency_ms'],
                prompt_tokens=meta['prompt_tokens'],
                completion_tokens=meta['completion_tokens'],
            )
        return normalized, meta
    except Exception as exc:
        meta['latency_ms'] = int((time.monotonic() - started_at) * 1000)
        return fallback(str(exc))


def _normalize_compact_llm_result(recipe, compact_matches, kitchen_context, substitution_callback=None):
    """Expand compact LLM output [["h","Item"],["p"],["m"]] into full availability format."""
    ingredients = [_safe_text(item) for item in ((recipe or {}).get('ingredients') or []) if _safe_text(item)]
    if len(compact_matches) != len(ingredients):
        return None
    by_item_id, by_display_name = _kitchen_index(kitchen_context)
    normalized_matches = []
    missing_ingredients = []
    matched_count = 0
    missing_count = 0
    for i, ingredient in enumerate(ingredients):
        entry = compact_matches[i] if i < len(compact_matches) else []
        if not isinstance(entry, list) or not entry:
            return None
        status_code = _safe_text(entry[0]).lower()
        kitchen_name = _safe_text(entry[1]) if len(entry) > 1 else ''
        sub_name = _safe_text(entry[2]) if len(entry) > 2 else ''
        if status_code == 'h':
            match_status = 'have'
        elif status_code == 'p':
            match_status = 'pantry'
        elif status_code == 'm':
            match_status = 'missing'
        else:
            return None
        # Override to pantry if code detects it
        if is_pantry_ingredient(ingredient):
            match_status = 'pantry'
            kitchen_name = ''
        # Resolve kitchen item reference
        matched_kitchen_items = []
        if match_status == 'have' and kitchen_name:
            candidate = by_display_name.get(kitchen_name.lower())
            if candidate:
                matched_kitchen_items.append({
                    'item_id': _safe_text((candidate or {}).get('item_id')) or None,
                    'display_name': _safe_text((candidate or {}).get('display_name')),
                    'match_type': DEFAULT_MATCH_TYPE,
                })
            else:
                # LLM returned a name we can't resolve — mark missing instead
                match_status = 'missing'
        if match_status == 'have' and not matched_kitchen_items:
            match_status = 'missing'
        # Resolve substitute reference
        substitute_kitchen_items = []
        if match_status == 'missing' and sub_name:
            sub_candidate = by_display_name.get(sub_name.lower())
            if sub_candidate:
                substitute_kitchen_items.append({
                    'item_id': _safe_text((sub_candidate or {}).get('item_id')) or None,
                    'display_name': _safe_text((sub_candidate or {}).get('display_name')),
                })
        normalized_matches.append({
            'recipe_ingredient': ingredient,
            'canonical_ingredient': canonical_ingredient_name(ingredient),
            'match_status': match_status,
            'matched_kitchen_items': matched_kitchen_items,
            'substitute_kitchen_items': substitute_kitchen_items,
            'missing_reason': 'No direct kitchen match found.' if match_status == 'missing' else None,
        })
        if match_status in {'have', 'pantry'}:
            matched_count += 1
        else:
            missing_count += 1
            missing_ingredients.append(ingredient)
    availability = {
        'kitchen_version': int((kitchen_context or {}).get('kitchen_version') or 0),
        'can_make_exact': missing_count == 0,
        'can_make_with_subs': False,
        'matched_count': matched_count,
        'missing_count': missing_count,
        'ingredient_matches': normalized_matches,
        'missing_ingredients': missing_ingredients,
        'substitution_candidates': [],
        'substitution_summary': None,
        'substitution_status': 'none',
        'analysis_status': 'ready',
    }
    return _apply_substitution_callback(recipe, kitchen_context, availability, substitution_callback)


def _compact_llm_call_single(client, model, kitchen_items, recipe, kitchen_context, substitution_callback):
    """Make a single compact LLM call for one recipe. Returns (recipe_id, availability_or_None)."""
    recipe_id = _safe_text(recipe.get('id') or recipe.get('_id'))
    ingredients = [_safe_text(item) for item in ((recipe or {}).get('ingredients') or []) if _safe_text(item)]
    payload = {
        'kitchen_items': kitchen_items,
        'recipes': [{
            'recipe_id': recipe_id,
            'title': _safe_text((recipe or {}).get('title')),
            'ingredients': ingredients,
        }],
    }
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': _compact_system_prompt()},
                {'role': 'user', 'content': json.dumps(payload)},
            ],
            **_chat_temperature_kwargs(model),
        )
        body = _best_effort_json_parse(_safe_text(response.choices[0].message.content), {})
        raw_results = body.get('r') or {}
        if not isinstance(raw_results, dict):
            return recipe_id, None
        raw_recipe = raw_results.get(recipe_id) or {}
        compact_matches = raw_recipe.get('m') if isinstance(raw_recipe, dict) else None
        if not compact_matches or not isinstance(compact_matches, list):
            return recipe_id, None
        result = _normalize_compact_llm_result(recipe, compact_matches, kitchen_context, substitution_callback=substitution_callback)
        return recipe_id, result
    except Exception:
        return recipe_id, None


def match_recipes_fast(recipes, kitchen_context, request_id=None, log_fn=None, source_label='inventory', substitution_callback=None):
    """Hybrid matching: deterministic first, then parallel compact LLM calls per recipe."""
    recipes = [dict(recipe or {}) for recipe in (recipes or []) if _safe_text((recipe or {}).get('id') or (recipe or {}).get('_id'))]
    meta = {
        'used_llm': False,
        'used_fallback': False,
        'recipe_count': len(recipes),
        'latency_ms': 0,
        'prompt_tokens': None,
        'completion_tokens': None,
        'error': None,
    }
    if not recipes:
        return {}, meta

    # Step 1: Run deterministic matching for all recipes (instant fallback)
    deterministic_results = {}
    for recipe in recipes:
        recipe_id = _safe_text(recipe.get('id') or recipe.get('_id'))
        deterministic_results[recipe_id] = deterministic_availability(recipe, kitchen_context, substitution_callback=substitution_callback)

    if not llm_inventory_matching_enabled():
        meta['used_fallback'] = True
        meta['error'] = 'llm_inventory_matching_disabled'
        return deterministic_results, meta

    client = _openai_client()
    if client is None:
        meta['used_fallback'] = True
        meta['error'] = 'openai_client_unavailable'
        return deterministic_results, meta

    # Step 2: Fire parallel compact LLM calls (one per recipe)
    kitchen_items = _kitchen_payload(kitchen_context)
    model = _inventory_model()
    started_at = time.monotonic()
    normalized = dict(deterministic_results)  # start with deterministic, upgrade with LLM
    llm_success_count = 0

    max_workers = min(len(recipes), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compact_llm_call_single, client, model, kitchen_items, recipe, kitchen_context, substitution_callback
            ): recipe
            for recipe in recipes
        }
        for future in as_completed(futures):
            recipe_id, llm_result = future.result()
            if llm_result is not None:
                normalized[recipe_id] = llm_result
                llm_success_count += 1

    meta['latency_ms'] = int((time.monotonic() - started_at) * 1000)
    meta['used_llm'] = llm_success_count > 0
    meta['used_fallback'] = llm_success_count < len(recipes)
    if llm_success_count == 0:
        meta['error'] = 'all_compact_calls_failed'

    if log_fn:
        event = 'llm_inventory_match_completed' if llm_success_count > 0 else 'llm_inventory_match_fallback'
        log_fn(
            request_id,
            event,
            source=source_label,
            recipe_count=len(recipes),
            llm_success_count=llm_success_count,
            latency_ms=meta['latency_ms'],
        )
    return normalized, meta
