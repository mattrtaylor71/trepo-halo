"""Conservative amount checks using US customary volume units.

Package sizes, density conversions and unmeasured reservations are never inferred.
A shared remaining balance prevents repeated ingredient lines spending stock twice.
"""
import re
import ingredient_identity as identity
import hashlib
import json
from decimal import Decimal, InvalidOperation

UNITS = {
    'g': ('mass', Decimal('1')),
    'kg': ('mass', Decimal('1000')),
    'oz': ('mass', Decimal('28.349523125')),
    'lb': ('mass', Decimal('453.59237')),
    'ml': ('volume', Decimal('1')),
    'l': ('volume', Decimal('1000')),
    'fl oz': ('volume', Decimal('29.5735295625')),
    'cup': ('volume', Decimal('236.5882365')),
    'tbsp': ('volume', Decimal('14.78676478125')),
    'tsp': ('volume', Decimal('4.92892159375')),
    'pint': ('volume', Decimal('473.176473')),
    'quart': ('volume', Decimal('946.352946')),
    'gallon': ('volume', Decimal('3785.411784')),
    'count': ('count', Decimal('1')),
    # Never assume how many cloves a head of garlic contains.
    'clove': ('clove', Decimal('1')),
    'head': ('head', Decimal('1')),
}
ALIASES = {
    'grams': 'g', 'gram': 'g', 'kilogram': 'kg', 'kilograms': 'kg',
    'ounce': 'oz', 'ounces': 'oz', 'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb',
    'liter': 'l', 'litre': 'l', 'liters': 'l', 'litres': 'l', 'milliliters': 'ml',
    'cups': 'cup', 'c': 'cup', 'tablespoon': 'tbsp', 'tablespoons': 'tbsp',
    'teaspoon': 'tsp', 'teaspoons': 'tsp', 'fluid ounce': 'fl oz', 'fluid ounces': 'fl oz',
    'each': 'count', 'items': 'count', 'piece': 'count', 'pieces': 'count', 'ct': 'count',
    'pints': 'pint', 'pt': 'pint', 'quarts': 'quart', 'qt': 'quart',
    'gallons': 'gallon', 'gal': 'gallon', 'cloves': 'clove', 'heads': 'head',
    'milliliter': 'ml', 'millilitre': 'ml', 'millilitres': 'ml',
}
FRACTIONS = {
    '¼': '1/4', '½': '1/2', '¾': '3/4', '⅓': '1/3', '⅔': '2/3',
    '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
}
UNMEASURED_UNITS = {
    'can', 'cans', 'bag', 'bags', 'bottle', 'bottles', 'pack', 'packs',
    'package', 'packages', 'jar', 'jars', 'clove', 'cloves', 'bunch', 'bunches',
    'pinch', 'handful', 'scoop', 'scoops', 'pint', 'pints', 'quart', 'quarts',
    'gallon', 'gallons', 'pt', 'qt', 'gal', 'mg', 'milligram', 'milligrams',
    'dl', 'cl', 't', 'stick', 'sticks',
}


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def normalized_unit(value):
    value = str(value or '').strip().lower().rstrip('.')
    return ALIASES.get(value, value)


def requested_amount(text):
    text = str(text or '').lower().strip()
    for symbol, fraction in FRACTIONS.items():
        text = text.replace(symbol, ' ' + fraction)
    match = re.match(r'^(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|\.\d+)\s*(\D.+)$', text.strip())
    if not match:
        return None
    try:
        amount = Decimal(0)
        for part in match[1].split():
            if '/' in part:
                numerator, denominator = part.split('/')
                amount += Decimal(numerator) / Decimal(denominator)
            else:
                amount += Decimal(part)
    except (InvalidOperation, ZeroDivisionError):
        return None
    rest = match[2]
    if rest.startswith(('-', '–', '/', '.', 'to ', 'x ', '× ')):
        return None
    if rest.startswith('('):
        package = re.match(r'^\(\s*([^()]+)\s*\)\s*(?:cans?|jars?|packages?|packs?|bottles?|bags?)\b', rest)
        if not package:
            return None
        size = requested_amount(package[1].strip() + ' contents')
        if size and UNITS[size[1]][0] in ('mass', 'volume') and size[0] > 0:
            return amount * size[0], size[1]
        return None
    for alias in sorted(set(UNITS) | set(ALIASES), key=len, reverse=True):
        if re.match(re.escape(alias) + r'\.?(?:\s|$)', rest):
            return amount, normalized_unit(alias)
    # Compact notation is accepted only for recognized units (100g), not an
    # unknown suffix or exponent that could silently become an item count.
    if not text.strip()[match.end(1):match.end(1)+1].isspace():
        return None
    if re.match(r'^garlic\s+cloves?\b', rest):
        return amount, 'clove'
    # Includes descriptions such as "two large cans", not just "two cans".
    if any(word.rstrip('.') in UNMEASURED_UNITS for word in rest.split()[:3]):
        return None
    return amount, 'count'


def available_amount(candidate):
    quantity = number(candidate.get('quantity_value'))
    unit_name = normalized_unit(candidate.get('quantity_unit'))
    unit = UNITS.get(unit_name)
    # Receipt scans count purchases/packages, not necessarily their contents.
    # "1 count" with this unmeasured receipt marker is not proof of one egg.
    # Preserve explicit zero and measured mass/volume; never guess carton sizes.
    remaining = str(candidate.get('remaining_quantity') or '').strip().lower().rstrip('.')
    if unit_name == 'count' and quantity != 0 and remaining == 'purchased on receipt':
        return None, None
    if quantity is not None:
        return quantity, unit
    # Older writers can retain a measured display quantity without populating
    # quantity_value. Honor an explicit unit; never infer bag/can sizes or turn
    # an unlabelled display number into a measured amount.
    legacy = str(candidate.get('quantity') or '').strip().lower()
    if unit and number(legacy) is not None:
        return number(legacy), unit
    for alias in sorted(set(UNITS) | set(ALIASES), key=len, reverse=True):
        if re.search(r'(?:\d|\s)' + re.escape(alias) + r'\.?$', legacy):
            parsed = requested_amount(legacy + ' contents')
            if parsed:
                return parsed[0], UNITS.get(parsed[1])
    return None, None


def reservation_token(planned):
    """Bind cached evidence to the active ingredient multiset, including repeats."""
    if not isinstance(planned, list) or any(not isinstance(x, str) for x in planned):
        raise ValueError('Invalid planned ingredient snapshot')
    canonical = sorted(' '.join(text.split()) for text in planned)
    return hashlib.sha256(json.dumps({'version': 1, 'ingredients': canonical},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reserve_planned_amounts(planned, candidates, tokenize, score, threshold=150):
    """Reserve listed meal amounts, without changing physical inventory quantities.

    Unknown sizes retain the legacy whole-row uncertainty. Known reservations are
    expressed in the unit dimension's base measure and bounded by physical stock.
    These are availability estimates, not durable locks or serving-size guesses.
    """
    unique = {}
    aliases = {}
    for index, candidate in enumerate(candidates):
        key = candidate.get('item_id') or ('row', index)
        unique.setdefault(key, candidate)
        aliases.setdefault(key, []).append(candidate)
    for ingredient in planned or []:
        intent = identity.parse(ingredient)
        if intent.optional:
            continue
        choices = []
        for alt in intent.alternatives:
            hits = [(key, item, 300) for key, item in unique.items()
                    if identity.compatible(alt, identity.candidate_name(item))]
            if hits:
                verdict = check_amount(alt, [(i, item) for i, (_, item, _) in enumerate(hits)], {})
                rank = {'sufficient': 3, 'unknown': 2, 'insufficient': 1}[verdict['status']]
                choices.append((rank, alt, hits))
        if not choices:
            continue
        _, selected, hits = max(choices, key=lambda x: x[0])
        requested = requested_amount(selected)
        if requested and requested[0] == 0:
            continue
        if not requested:
            # Without an amount, no matching row has a proven free remainder.
            for _, item, _ in hits:
                item['reserved'] = True
                item.pop('reserved_amount_base', None)
                item.pop('reserved_dimension', None)
            continue
        amount, unit_name = requested
        dimension, factor = UNITS[unit_name]
        needed = amount * factor
        uncertain = []
        for _, item, _ in hits:
            quantity, unit = available_amount(item)
            previous = number(item.get('reserved_amount_base'))
            if (quantity is None or unit is None or unit[0] != dimension
                    or (item.get('reserved') and (previous is None
                        or item.get('reserved_dimension') != dimension))):
                uncertain.append(item)
                continue
            total = quantity * unit[1]
            previous = previous or Decimal(0)
            free = max(Decimal(0), total - previous)
            used = min(free, needed)
            if used:
                item.update(reserved=True, reserved_amount_base=str(previous + used),
                            reserved_dimension=dimension)
                needed -= used
            if needed <= 0:
                break
        if needed > 0:
            # A known shortage may use uncertain stock, but cannot promise a
            # numerical remainder from it. Keep that uncertainty visible.
            for item in uncertain:
                item['reserved'] = True
                item.pop('reserved_amount_base', None)
                item.pop('reserved_dimension', None)
    for key, copies in aliases.items():
        source = unique[key]
        for item in copies:
            for field in ('reserved', 'reserved_amount_base', 'reserved_dimension'):
                if field in source:
                    item[field] = source[field]
                else:
                    item.pop(field, None)


def check_amount(ingredient, candidates, remaining):
    requested = requested_amount(ingredient)
    if not requested or requested[0] <= 0:
        return {'status': 'unknown', 'reason': 'Confirm how much this recipe needs.'}
    amount, unit_name = requested
    dimension, factor = UNITS[unit_name]
    needed = amount * factor
    available = Decimal(0)
    buckets, seen = [], set()
    unknown = False
    for index, candidate in candidates:
        key = candidate.get('item_id') or ('row', index)
        if key in seen:
            continue
        seen.add(key)
        quantity, unit = available_amount(candidate)
        if quantity is None or unit is None or unit[0] != dimension:
            unknown = True
            continue
        total = quantity * unit[1]
        reserved = Decimal(0)
        if candidate.get('reserved'):
            reserved = number(candidate.get('reserved_amount_base'))
            if (reserved is None or candidate.get('reserved_dimension') != dimension
                    or reserved > total):
                unknown = True
                continue
        free = remaining.get(key, total - reserved)
        available += free
        buckets.append((key, free))
    if available < needed:
        return {
            'status': 'unknown' if unknown else 'insufficient',
            'reason': 'Confirm the available amount.' if unknown else 'There is not enough of this ingredient left.',
        }
    for key, free in buckets:
        used = min(free, needed)
        remaining[key] = free - used
        needed -= used
        if needed <= 0:
            break
    return {'status': 'sufficient', 'reason': None}
