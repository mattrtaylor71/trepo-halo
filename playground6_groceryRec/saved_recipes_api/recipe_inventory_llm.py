import ai_consent_guard as _ai_consent
"""Shared versioned identity, quantity checks and bounded semantic resolution."""
import recipe_generation_budget
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
import ingredient_identity as identity

from quantity_sufficiency import check_amount
from quantity_confirmation import (availability_token, persist_cache_rows as persist_quantity_availability_rows,
                                   confirmed_availability_for_response)


PANTRY_CANONICAL_INGREDIENTS = {
    'salt', 'kosher salt', 'sea salt', 'table salt', 'flaky salt',
    'pepper', 'black pepper', 'white pepper',
    'oil', 'olive oil', 'vegetable oil', 'canola oil', 'neutral oil', 'cooking oil', 'sesame oil',
    'water', 'cold water', 'warm water', 'hot water', 'ice water', 'ice',
    'butter', 'salted butter', 'unsalted butter',
    'cooking spray', 'nonstick cooking spray',
    'sugar', 'white sugar', 'granulated sugar', 'brown sugar', 'powdered sugar', 'confectioners sugar',
    'flour', 'all purpose flour', 'wheat flour',
    'baking soda', 'baking powder',
    'vinegar', 'white vinegar', 'apple cider vinegar',
    'basic spice', 'spice', 'basic spices', 'seasoning', 'seasonings',
    'garlic powder', 'onion powder', 'paprika', 'cumin', 'oregano', 'cinnamon',
    'cornstarch', 'corn starch',
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


def ingredient_tokens(text):
    parsed = identity.food(str(text or ''))
    return parsed.canonical.split() + sorted(parsed.modifiers | parsed.forms)


def canonical_ingredient_name(text):
    return identity.food(str(text or '')).canonical


def is_pantry_ingredient(text):
    parsed = identity.food(str(text or ''))
    return (not parsed.modifiers.intersection(identity.PROTECTED)
        and (parsed.canonical in PANTRY_CANONICAL_INGREDIENTS
             and parsed.modifiers.issubset(parsed.canonical.split())
             or not parsed.modifiers and bool(parsed.families)
             and parsed.families.issubset(PANTRY_CANONICAL_INGREDIENTS)))


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
    import ai_consent_guard
    try:
        if not ai_consent_guard.allowed(ai_consent_guard.actor_context.get()):
            return None
    except Exception:
        return None  # Retain deterministic availability when consent cannot be checked.
    from openai import OpenAI
    api_key = _safe_text(os.getenv('OPENAI_API_KEY'))
    if not api_key:
        return None
    timeout_seconds = int(os.getenv('OPENAI_INVENTORY_TIMEOUT_SECONDS', os.getenv('OPENAI_TIMEOUT_SECONDS', '12')))
    max_retries = int(os.getenv('OPENAI_INVENTORY_MAX_RETRIES', '0'))
    return _ai_consent.provider_client(OpenAI)(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)


def _score_kitchen_candidate(recipe_tokens, candidate_tokens):
    return 300 if identity.compatible(' '.join(recipe_tokens or []), ' '.join(candidate_tokens or [])) else -1


def _deterministic_ingredient_match(ingredient_text, kitchen_candidates, assume_pantry=True, *, confirmed_staple=False):
    intent = identity.parse(ingredient_text)
    matches = []
    selected = None
    for alt in intent.alternatives:
        matches = [c for c in kitchen_candidates or [] if identity.compatible(alt, identity.candidate_name(c))]
        if matches:
            selected = alt
            break
    status = 'have' if matches else ('optional' if intent.optional else
             'pantry' if confirmed_staple or (assume_pantry and any(is_pantry_ingredient(a) for a in intent.alternatives)) else 'missing')
    return {
        'recipe_ingredient': ingredient_text,
        'canonical_ingredient': identity.food(selected or intent.alternatives[0]).canonical,
        'match_status': status,
        'matched_kitchen_items': [{'item_id': c.get('item_id'), 'display_name': identity.candidate_name(c),
                                  'match_type': 'food_identity'} for c in matches],
        'substitute_kitchen_items': [],
        'missing_reason': 'No direct kitchen match found.' if status == 'missing' else None,
        'is_optional': intent.optional,
        'selected_alternative': selected,
        'matcher_version': identity.VERSION,
    }


def inventory_cache_source(source):
    """Invalidate old identity/quantity verdicts without changing inventory versions or DDL."""
    return str(source) + ':q4' + identity.VERSION


def availability_summary(availability):
    """Same quantity guidance for fresh results and existing JSON cache columns.

    No schema migration is needed: ingredient_matches already stores each
    quantity verdict. The original five fields remain readable by old clients.
    """
    result = {key: availability.get(key) for key in (
        'kitchen_version', 'can_make_exact', 'can_make_with_subs',
        'matched_count', 'missing_count',
    )}
    matches = availability.get('ingredient_matches') or []
    unknown = [m['recipe_ingredient'] for m in matches
               if isinstance(m, dict) and m.get('quantity_status') == 'unknown' and m.get('recipe_ingredient')]
    insufficient = [m['recipe_ingredient'] for m in matches
                    if isinstance(m, dict) and m.get('quantity_status') == 'insufficient' and m.get('recipe_ingredient')]
    result.update(
        quantity_unknown_ingredients=unknown,
        insufficient_quantity_ingredients=insufficient,
        quantity_confirmation_required=bool(unknown),
        quantity_confirmation_items=[{'index': index, 'ingredient': m['recipe_ingredient']}
                                     for index, m in enumerate(matches)
                                     if isinstance(m, dict) and m.get('quantity_status') == 'unknown'
                                     and m.get('match_status') == 'have' and m.get('matched_kitchen_items')
                                     and m.get('recipe_ingredient')],
        availability_token=availability_token(availability) if unknown else None,
    )
    if unknown or insufficient:
        result['can_make_exact'] = False
        result['can_make_with_subs'] = False
    if any(isinstance(m, dict) and m.get('reservation_unavailable') for m in matches):
        result.update(can_make_exact=False, can_make_with_subs=False,
                      quantity_confirmation_required=False, quantity_confirmation_items=[],
                      availability_token=None, availability_status='temporarily_unavailable')
    return result


def enforce_quantity_availability(recipe, kitchen_context, availability):
    """Resolve current identity and one OR branch before allocating measured stock.

    Recompute on every call: cached semantic evidence never caches stock, and
    rerunning the guard must not double-spend or retain a removed item's reference.
    """
    context = kitchen_context or {}
    matches = availability.get('ingredient_matches') or []
    if context.get('reservation_unavailable'):
        for match in matches:
            if not identity.parse(match['recipe_ingredient']).optional:
                match.update(reservation_unavailable=True, quantity_status='unknown',
                             missing_reason='Kitchen availability is temporarily unavailable.')
                match.pop('reservation_token', None)
                match.pop('quantity_confirmation', None)
        availability.update(can_make_exact=False, can_make_with_subs=False)
        return availability
    candidates = context.get('kitchen_candidates') or []
    confirmed = {canonical_ingredient_name(item) for item in context.get('confirmed_staples', [])}
    token = context.get('reservation_token')
    if token is not None and not re.fullmatch(r'[0-9a-f]{64}', str(token)):
        raise ValueError('Invalid meal reservation evidence')
    remaining, unknown, insufficient, missing = {}, [], [], []
    for match in matches:
        ingredient = match['recipe_ingredient']
        intent = identity.parse(ingredient)
        # Only validated semantic evidence can supplement local food identity.
        semantic_refs = match.get('semantic_evidence') or []
        options = []
        for alt in intent.alternatives:
            current = [(index, c) for index, c in enumerate(candidates)
                       if identity.compatible(alt, identity.candidate_name(c)) or (
                           identity.compatible(alt, identity.candidate_name(c), semantic=True)
                           and any(e.get('requirement') == alt and e.get('product') == identity.candidate_name(c)
                                   and e.get('version') == identity.VERSION for e in semantic_refs))]
            if current:
                trial = dict(remaining)
                quantity = check_amount(alt, current, trial)
                rank = {'sufficient': 3, 'unknown': 2, 'insufficient': 1}[quantity['status']]
                options.append((rank, alt, current, quantity, trial))
        # Remove stale amount flags before restoring independently verified evidence.
        for key in ('reservation_unavailable', 'quantity_status', 'quantity_confirmation'):
            match.pop(key, None)
        match.update(_deterministic_ingredient_match(ingredient, candidates,
                     assume_pantry=availability.get('pantry_assumptions_allowed', True),
                     confirmed_staple=any(canonical_ingredient_name(a) in confirmed for a in intent.alternatives)))
        if token is not None: match['reservation_token'] = token
        else: match.pop('reservation_token', None)
        if options:
            _, alt, current, quantity, trial = max(options, key=lambda x: x[0])
            match.update(match_status='have', selected_alternative=alt,
                         canonical_ingredient=identity.food(alt).canonical,
                         matched_kitchen_items=[{'item_id': c.get('item_id'),
                                                 'display_name': identity.candidate_name(c),
                                                 'match_type': 'food_identity' if identity.compatible(alt, identity.candidate_name(c)) else 'semantic_identity'}
                                                for _, c in current],
                         quantity_status=quantity['status'], missing_reason=quantity.get('reason'))
            if not intent.optional:
                remaining = trial
                if quantity['status'] == 'insufficient':
                    match['match_status'] = 'missing'; insufficient.append(ingredient)
                elif quantity['status'] == 'unknown': unknown.append(ingredient)
        elif match['match_status'] == 'pantry':
            if any(canonical_ingredient_name(a) in confirmed for a in intent.alternatives):
                match['quantity_status'] = 'confirmed_staple'
            else:
                match['quantity_status'] = 'unknown'; unknown.append(ingredient)
        if intent.optional:
            # Optional stock never consumes quantities needed by required lines.
            match['quantity_status'] = 'not_required'
        elif match['match_status'] == 'missing': missing.append(ingredient)
    required = [m for m in matches if not m.get('is_optional')]
    availability.update(missing_ingredients=missing, missing_count=len(missing),
                        matched_count=len(required)-len(missing),
                        quantity_unknown_ingredients=unknown, insufficient_quantity_ingredients=insufficient,
                        optional_count=len(matches)-len(required), matcher_version=identity.VERSION,
                        can_make_exact=bool(matches) and not missing and not unknown)
    if missing or unknown or insufficient: availability['can_make_with_subs'] = False
    _filter_substitutions(availability)
    availability['substitution_quantity_status'] = ('needs_confirmation' if missing and availability.get('substitution_candidates') else None)
    return availability


def _filter_substitutions(availability):
    """A shortage cannot be solved by relabelling the same ingredient.

    Also discard cached suggestions for ingredients that are now present. A
    model-written paragraph can mention rejected suggestions, so clear it when
    filtering rather than showing prose that contradicts the ingredient rows.
    """
    original = availability.get('substitution_candidates') or []
    missing = set(availability.get('missing_ingredients') or [])
    kept = []
    for suggestion in original:
        if not isinstance(suggestion, dict):
            continue
        ingredient = _safe_text(suggestion.get('missing_ingredient'))
        replacement = _safe_text(suggestion.get('use_instead'))
        if ingredient not in missing or identity.is_placeholder(replacement):
            continue
        if any(identity.compatible(alt, replacement) for alt in identity.parse(ingredient).alternatives):
            continue
        kept.append(suggestion)
    availability['substitution_candidates'] = kept
    if len(kept) != len(original):
        availability['substitution_summary'] = None
        availability['substitution_status'] = 'ready' if kept else 'none'
        availability['can_make_with_subs'] = False
    if not kept:
        availability['substitution_summary'] = None


def _apply_substitution_callback(recipe, kitchen_context, availability, substitution_callback):
    if (kitchen_context or {}).get('reservation_unavailable'):
        return enforce_quantity_availability(recipe, kitchen_context, availability)
    # Resolve shortages and unconfirmed staples before asking for alternatives.
    availability = enforce_quantity_availability(recipe, kitchen_context, availability)
    if not substitution_callback:
        availability['substitution_candidates'] = []
        availability['substitution_summary'] = None
        availability['substitution_status'] = None
        availability['can_make_with_subs'] = False
        return enforce_quantity_availability(recipe, kitchen_context, availability)
    # Unknown/processing rows must not become proposed substitutes either.
    substitution_context = dict(kitchen_context or {})
    substitution_context['kitchen_candidates'] = [
        c for c in substitution_context.get('kitchen_candidates') or [] if identity.candidate_name(c)]
    substitution = substitution_callback(recipe, substitution_context, availability.get('missing_ingredients') or [])
    availability['can_make_with_subs'] = bool((substitution or {}).get('can_make_with_subs'))
    availability['substitution_candidates'] = (substitution or {}).get('substitution_candidates') or []
    availability['substitution_summary'] = (substitution or {}).get('substitution_summary')
    availability['substitution_status'] = (substitution or {}).get('substitution_status')
    return enforce_quantity_availability(recipe, kitchen_context, availability)


def deterministic_availability(recipe, kitchen_context, substitution_callback=None, *, assume_pantry=True):
    recipe = dict(recipe or {})
    ingredient_matches = []
    missing_ingredients = []
    matched_count = 0
    missing_count = 0
    kitchen_candidates = (kitchen_context or {}).get('kitchen_candidates') or []
    for ingredient in [_safe_text(item) for item in (recipe.get('ingredients') or []) if _safe_text(item)]:
        match = _deterministic_ingredient_match(ingredient, kitchen_candidates, assume_pantry=assume_pantry)
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
    availability['pantry_assumptions_allowed'] = assume_pantry
    return _apply_substitution_callback(recipe, kitchen_context, availability, substitution_callback)


def _kitchen_payload(kitchen_context):
    items = []
    for candidate in (kitchen_context or {}).get('kitchen_candidates') or []:
        display_name = identity.candidate_name(candidate)
        if not display_name:
            continue
        items.append({
            'item_id': _safe_text((candidate or {}).get('item_id')) or None,
            'display_name': display_name,
            'description': _safe_text((candidate or {}).get('description')) or None,
            'quantity_value': str(candidate['quantity_value']) if candidate.get('quantity_value') is not None else None,
            'quantity_unit': candidate.get('quantity_unit'),
            'reserved': bool(candidate.get('reserved')),
        })
    return items


def _kitchen_index(kitchen_context):
    by_item_id = {}
    by_display_name = {}
    for candidate in (kitchen_context or {}).get('kitchen_candidates') or []:
        item_id = _safe_text((candidate or {}).get('item_id'))
        display_name = identity.candidate_name(candidate)
        if not display_name:
            continue
        if item_id:
            by_item_id[item_id] = candidate
        if display_name:
            by_display_name[display_name.lower()] = candidate
    return by_item_id, by_display_name


def _normalize_kitchen_refs(raw_refs, kitchen_context, default_match_type=None):
    by_id, by_name = _kitchen_index(kitchen_context)
    output, seen = [], set()
    for ref in raw_refs or []:
        if not isinstance(ref, dict): continue
        supplied_id = _safe_text(ref.get('item_id'))
        candidate = by_id.get(supplied_id) if supplied_id else by_name.get(_safe_text(ref.get('display_name')).lower())
        if not candidate: continue
        key = (candidate.get('item_id'), candidate.get('display_name'))
        if key in seen: continue
        seen.add(key)
        output.append({'item_id':candidate.get('item_id'), 'display_name':candidate.get('display_name'),
                       'match_type': default_match_type or DEFAULT_MATCH_TYPE})
    return output


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
    availability = {
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


    return enforce_quantity_availability(recipe, kitchen_context, availability)


def match_recipes(recipes, kitchen_context, request_id=None, log_fn=None, source_label='inventory', substitution_callback=None):
    return match_recipes_fast(recipes, kitchen_context, request_id, log_fn, source_label, substitution_callback)


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
        # Legacy schema adapter: current identity and stock are revalidated below.
        matched_kitchen_items = []
        if match_status == 'have' and kitchen_name:
            candidate = by_display_name.get(kitchen_name.lower())
            if not candidate:
                # Fuzzy fallback: try token-based matching for display purposes
                llm_tokens = ingredient_tokens(kitchen_name)
                if llm_tokens:
                    best_candidate = None
                    best_score = -1
                    for c in (kitchen_context or {}).get('kitchen_candidates') or []:
                        score = _score_kitchen_candidate(llm_tokens, c.get('tokens') or [])
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                    if best_candidate:
                        candidate = best_candidate
            if candidate:
                matched_kitchen_items.append({
                    'item_id': _safe_text((candidate or {}).get('item_id')) or None,
                    'display_name': _safe_text((candidate or {}).get('display_name')),
                    'match_type': DEFAULT_MATCH_TYPE,
                })
            else:
                # The quantity/identity guard below rejects unresolved references.
                matched_kitchen_items.append({
                    'item_id': None,
                    'display_name': kitchen_name,
                    'match_type': 'llm_match',
                })
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


def _merge_availability(det, llm):
    """Keep deterministic verified identity and amounts; model claims cannot overwrite them.

    New semantic enrichment is validated before quantity enforcement. This legacy
    helper remains conservative for callers still providing old model results.
    """
    return dict(det)


def match_recipes_fast(recipes, kitchen_context, request_id=None, log_fn=None, source_label='inventory', substitution_callback=None, *, inline_single=False):
    """Shared local identity with one bounded unresolved-name model batch.

    The call stays on the caller's thread, preserving deadline cancellation.
    No model call is made for known identity, optional garnish or pantry guesses.
    """
    recipes = [dict(r) for r in recipes or [] if _safe_text((r or {}).get('id') or (r or {}).get('_id'))]
    if inline_single and len(recipes) > 1: raise ValueError('inline_single requires at most one recipe')
    results = {_safe_text(r.get('id') or r.get('_id')): deterministic_availability(r, kitchen_context)
               for r in recipes}
    meta = dict(used_llm=False, used_fallback=False, recipe_count=len(recipes), latency_ms=0,
                prompt_tokens=None, completion_tokens=None, error=None)
    if not recipes or (kitchen_context or {}).get('reservation_unavailable'): return results, meta
    candidates = (kitchen_context or {}).get('kitchen_candidates') or []
    unresolved = [m['recipe_ingredient'] for result in results.values() for m in result['ingredient_matches']
                  if m['match_status'] == 'missing' and not m.get('matched_kitchen_items') and not m.get('is_optional')]
    model = _inventory_model()
    pairs = identity.unresolved_pairs(unresolved, candidates, model)
    if pairs and llm_inventory_matching_enabled():
        client = _openai_client()
        if client is not None:
            if hasattr(client, 'with_options'):
                client = client.with_options(timeout=6.0, max_retries=0)
            started = time.monotonic()
            try:
                response = recipe_generation_budget.chat(client,
                    model=model, timeout=min(6.0, recipe_generation_budget.remaining(6.0)), max_tokens=1600,
                    response_format={'type': 'json_object'},
                    messages=[{'role':'system', 'content':identity.SEMANTIC_PROMPT},
                              {'role':'user', 'content':json.dumps({'pairs':[
                                  {'id':i, 'ingredient':a, 'product':b} for i,(a,b) in enumerate(pairs)]})}],
                    **_chat_temperature_kwargs(model))
                meta['used_llm'] = True
                body = _best_effort_json_parse(response.choices[0].message.content, {})
                if identity.accept_semantic(body, pairs, model) != len(pairs):
                    meta.update(used_fallback=True, error='invalid_semantic_response')
                usage = getattr(response, 'usage', None)
                if usage:
                    meta['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
                    meta['completion_tokens'] = getattr(usage, 'completion_tokens', None)
            except Exception:
                meta.update(used_fallback=True, error='semantic_lookup_unavailable')
            meta['latency_ms'] = int((time.monotonic()-started)*1000)
        else: meta.update(used_fallback=True, error='openai_client_unavailable')
    elif pairs: meta.update(used_fallback=True, error='llm_inventory_matching_disabled')
    for recipe in recipes:
        result = results[_safe_text(recipe.get('id') or recipe.get('_id'))]
        for match in result['ingredient_matches']:
            if match['recipe_ingredient'] not in unresolved: continue
            evidence = []
            for alt in identity.parse(match['recipe_ingredient']).alternatives:
                for candidate in candidates:
                    name = identity.candidate_name(candidate)
                    if (llm_inventory_matching_enabled() and identity.compatible(alt, name, semantic=True)
                            and identity.cached_semantic(alt, name, model) is True):
                        evidence.append({'requirement':alt, 'product':name, 'version':identity.VERSION})
            if evidence: match['semantic_evidence'] = evidence
    # Preserve existing concurrent substitution generation. The semantic lookup
    # above is one batch; optional substitutions must not become one slow call
    # per recipe in a serial loop. Scope/deadline propagates to every worker.
    if substitution_callback and len(recipes) > 1:
        workers = max(1, min(len(recipes), 24, int(os.getenv('OPENAI_INVENTORY_MAX_WORKERS', '24'))))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [recipe_generation_budget.submit(executor, _apply_substitution_callback,
                recipe, kitchen_context, results[_safe_text(recipe.get('id') or recipe.get('_id'))], substitution_callback)
                for recipe in recipes]
            for future in as_completed(futures): future.result()
    else:
        for recipe in recipes:
            _apply_substitution_callback(recipe, kitchen_context,
                results[_safe_text(recipe.get('id') or recipe.get('_id'))], substitution_callback)
    if log_fn:
        log_fn(request_id, 'ingredient_identity_completed', source=source_label,
               recipe_count=len(recipes), unresolved_pairs=len(pairs), used_llm=meta['used_llm'],
               latency_ms=meta['latency_ms'], error=meta['error'])
    return results, meta


def preview_availability(recipe, kitchen_context):
    recipe = dict(recipe, id=recipe.get('id') or 'preview')
    results, _ = match_recipes_fast([recipe], kitchen_context, source_label='preview', inline_single=True)
    return results[recipe['id']]
