"""Deterministic evidence projection. No inference calls and no domain writes."""
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

VERSION = 'household-brain-v2'
MAX_CONTEXT_BYTES = 12288
SOFT_FIELDS = {'cuisines', 'favorite_foods', 'avoid_foods', 'preferred_products',
               'cooking_style', 'max_prep_minutes'}


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=iso)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def text(value, limit=160):
    return re.sub(r'[\x00-\x1f\x7f]', ' ', str(value or '')).strip()[:limit]


def stamp(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)


def iso(value):
    parsed = stamp(value)
    if parsed:
        return parsed.isoformat()
    raise TypeError('unsupported_json_value')


def obj(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {} if default is None else default
    return value if value is not None else ({} if default is None else default)


def product_key(name, brand='', variant='', barcode=''):
    # Exact normalized product families only. Never merge variants/pack sizes by guess.
    return digest([re.sub(r'\s+', ' ', text(v, 240)).casefold() for v in (name, brand, variant, barcode)])


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (TypeError, ValueError):
        return None


def explicit_preferences(states, actor, now):
    result = {}
    for state in sorted(states, key=lambda s: (s['actor_id'] == actor, s['actor_id'])):
        for key, fact in obj(state.get('facts')).items():
            if key not in SOFT_FIELDS or not isinstance(fact, dict):
                continue
            if fact.get('visibility') != 'household' and state['actor_id'] != actor:
                continue
            if fact.get('expires_at') and (not stamp(fact['expires_at']) or stamp(fact['expires_at']) <= now):
                continue
            value = fact.get('value')
            if key == 'max_prep_minutes':
                if type(value) is not int or not 5 <= value <= 240:
                    continue
            elif not isinstance(value, list) or not 1 <= len(value) <= 8 or any(not isinstance(v, str) or not v.strip() or len(v) > 80 for v in value):
                continue
            if key not in result or state['actor_id']==actor:
                result[key] = {'value': value, 'scope': fact.get('visibility', 'personal'), 'basis': 'explicit'}
    return result


def project(source, state, actor=None, now=None):
    """source is current, complete, authorized canonical data, never arbitrary client input.

    source -> qualified evidence -> derived patterns -> bounded optional context
    Hard restrictions and operational writes remain outside this projection.
    """
    now = stamp(now) or datetime.now(timezone.utc)
    local = ZoneInfo(state.get('timezone', 'UTC'))
    today = now.astimezone(local).date()
    since = max(stamp(state.get('learning_since')) or now, now - timedelta(days=90))
    members = set(source['members'])
    if actor is not None and actor not in members:
        raise PermissionError('membership_changed')
    subjects = [s for s in source.get('subjects', []) if s['actor_id'] in members and s['household_id'] == state['household_id'] and s['enabled']]
    explicit = explicit_preferences(subjects, actor, now)
    own = next((s for s in source.get('subjects',[]) if s['actor_id']==actor and s['household_id']==state['household_id']), None)
    personal_since = max(since, stamp(own.get('learning_since')) or now) if own else since
    learn_personal = bool(actor and not state['paused'] and not (own or {}).get('paused') and (own or {}).get('enabled',True))
    suppressed = set(source.get('suppressed', []))
    products, evidence, observations = {}, [], defaultdict(list)
    seen = set()

    def record(kind, row, at, identity, audience='household'):
        at = stamp(at)
        if state['paused'] or not at or not since <= at <= now or (audience != 'household' and (not learn_personal or audience != actor or at < personal_since)):
            return
        name = text(row.get('product_name') or row.get('name') or row.get('title') or row.get('dish_name'))
        if not name:
            return
        brand, variant, barcode = text(row.get('brand')), text(row.get('variant')), text(row.get('barcode'))
        key = product_key(name, brand, variant, barcode)
        if key in suppressed:
            return
        unique = (kind, str(identity), audience)
        if unique in seen:
            return
        seen.add(unique)
        e = {'id': digest(unique)[:24], 'kind': kind, 'name': name, 'at': iso(at),
             'scope': 'household' if audience == 'household' else 'personal', 'product_key': key}
        evidence.append(e)
        if kind in {'recipe_saved', 'dish_logged', 'recipe_used', 'recipe_liked', 'recipe_disliked', 'meal_planned'}:
            observations[kind].append(e)
            return
        p = products.setdefault(key, {'key': key, 'name': name, 'brand': brand, 'variant': variant,
                                     'evidence': defaultdict(set), 'evidence_ids': [], 'latest_at': iso(at)})
        p['evidence'][kind].add(at.astimezone(local).date().isoformat())
        p['evidence_ids'].append(e['id'])
        p['latest_at'] = max(p['latest_at'], iso(at))

    kitchen = []
    for row in source.get('kitchen', []):
        if row.get('needs_review') or str(row.get('analysis_status') or '').lower() not in {'','ready','complete','completed','success','succeeded'} or str(row.get('action', 'IN')).upper() != 'IN':
            continue
        kitchen.append(row)
    all_inventory = kitchen + source.get('archive', [])
    for row in all_inventory:
        if row.get('needs_review') or str(row.get('analysis_status') or '').lower() not in {'','ready','complete','completed','success','succeeded'}:
            continue
        record('inventory_observed', row, row.get('_createdDate'), row['_id'])
        payload = obj(row.get('provisional_payload'))
        # Undated receipt imports establish recorded receipt evidence, not a purchase
        # date. Several old receipts imported today must not invent a buying rhythm.
        if isinstance(payload, dict) and payload.get('scan_source') == 'receipt_scan' and (payload.get('source_job_id') or row.get('job_id')):
            receipt_id=str(payload.get('source_job_id') or row['job_id']) + ':' + str(row['_id'])
            record('receipt_imported', row, row.get('_createdDate'), receipt_id)
            purchase_date=payload.get('receipt_purchase_date')
            if isinstance(purchase_date,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',purchase_date) and stamp(purchase_date):
                purchase_date=datetime.fromisoformat(purchase_date).replace(tzinfo=local)
            record('purchase', row, purchase_date, receipt_id)
    for row in source.get('recipes', []):
        if row.get('owner_id') == actor and row.get('status') == 'ready':
            record('recipe_saved', row, row.get('_createdDate'), row['_id'], actor)
    for row in source.get('dishes', []):
        if row.get('user_id') == actor and str(row.get('action', '')).upper() == 'IN' and str(row.get('analysis_status') or '').lower() not in {'failed','pending','processing'}:
            record('dish_logged', row, row.get('_createdDate'), row['_id'], actor)
    for operation in source.get('amounts', []):
        if operation.get('undone_by') or operation.get('actor') != actor:
            continue
        for index, change in enumerate(obj(operation.get('changes'), [])):
            if not isinstance(change, dict) or change.get('mode') not in {'consume', 'discard'}:
                continue
            row = change.get('snapshot') or {}
            if not isinstance(row, dict):
                continue
            key = str(operation['operation_id']) + ':' + str(index)
            record('consumed' if change['mode'] == 'consume' else 'discarded', row, operation['created_at'], key, actor)
            if change['mode'] == 'consume' and change.get('recipe_title'):
                record('recipe_used', {'name': change['recipe_title']}, operation['created_at'], str(operation['operation_id']), actor)
    for row in source.get('calendar', []):
        if row.get('owner_id') == actor:
            record('meal_planned', row, row.get('_createdDate'), row['_id'], actor)
    for event in source.get('events', []):
        if event.get('reversed_by') or event['generation'] != state['generation'] or event.get('actor_id') not in members:
            continue
        payload = obj(event['payload'])
        audience = 'household' if payload.get('visibility') == 'household' else event.get('actor_id')
        if not audience or (audience != 'household' and audience not in members):
            continue
        record(event['kind'], payload, event['occurred_at'], event['producer'] + ':' + event['event_id'], audience)

    output_products = []
    for p in products.values():
        dates = sorted(p['evidence'].get('purchase', []))
        cadence = None
        if len(dates) >= 3:
            values = [datetime.fromisoformat(d).date() for d in dates]
            gaps = [(b-a).days for a,b in zip(values, values[1:])]
            median = statistics.median(gaps)
            if median > 0 and max(gaps) <= 2 * median and (today-values[-1]).days <= 3 * median:
                cadence = {'median_days': median, 'next_window_start': (values[-1]+timedelta(days=min(gaps))).isoformat(),
                           'next_window_end': (values[-1]+timedelta(days=max(gaps))).isoformat(),
                           'interpretation': 'Recorded purchase rhythm; check stock, not proof of depletion.'}
        output_products.append({**p, 'evidence': {k: len(v) for k,v in p['evidence'].items()},
                                'evidence_ids': p['evidence_ids'][:8], 'cadence': cadence})
    output_products.sort(key=lambda p: (-p['evidence'].get('purchase',0), -p['evidence'].get('product_selected',0), p['name'], p['key']))

    # Current lists/dates always remain current even when learning is paused.
    listed = {text(r.get('product_name')).casefold() for r in source.get('shopping', []) if str(r.get('action','')).upper() in {'ADDED','CHECKED'}}
    use_soon, check_dates, groceries = [], [], []
    for row in kitchen:
        name = text(row.get('product_name'))
        key = product_key(name, row.get('brand'), row.get('variant'), row.get('barcode'))
        if key in suppressed:
            continue
        date = stamp(row.get('product_expiration'))
        if date:
            days = (date.date()-today).days
            target = check_dates if days < 0 else use_soon
            if days <= 5:
                target.append({'name': name, 'date': date.date().isoformat(), 'days_until': days, 'basis': 'recorded_date', 'key': key})
        quantity, fill = number(row.get('quantity_value')), number(row.get('fill_percent'))
        if name.casefold() not in listed and (quantity == 0 or (fill is not None and fill <= 25)):
            groceries.append({'key': key, 'name': name, 'reason': 'Recorded amount is low; check stock.', 'basis': 'recorded_amount'})
    for p in output_products:
        if p['name'].casefold() not in listed and p['cadence'] and p['cadence']['next_window_start'] <= today.isoformat() and p['evidence'].get('discarded',0) < 2:
            groceries.append({'key': p['key'], 'name': p['name'], 'reason': 'Approaching the recorded purchase interval; check stock.', 'basis': 'purchase_interval'})
    groceries = list({g['key']: g for g in groceries}.values())[:20]
    context = {'schema': VERSION, 'explicit_preferences': explicit,
               'observations': {k: [e['name'] for e in sorted(v,key=lambda x:x['at'],reverse=True)[:8]] for k,v in observations.items()},
               'products': output_products[:40], 'groceries': groceries,
               'use_soon': sorted(use_soon,key=lambda x:x['date'])[:12], 'check_dates': sorted(check_dates,key=lambda x:x['date'])[:12],
               'evidence': sorted(evidence,key=lambda x:x['at'],reverse=True)[:30],
               'coverage': {'as_of': iso(now), 'history_since': iso(since), 'sources': source.get('sources',[]),
                            'complete_history': False, 'learning_paused': bool(state['paused'] or (own or {}).get('paused')),
                            'caveats': ['Saves indicate interest, not liking.', 'Dish logs do not identify who cooked.',
                                        'List checks and cart selections are not purchases.', 'Undated receipt imports do not establish purchase cadence.',
                                        'Recorded dates do not establish food safety.']}}
    # Explicit preferences survive before optional derived evidence; hard restrictions are separate.
    for field in ('evidence','products','check_dates','use_soon','groceries'):
        while len(encode(context).encode()) > MAX_CONTEXT_BYTES and context[field]:
            context[field].pop()
    while len(encode(context).encode()) > MAX_CONTEXT_BYTES and context['observations']:
        key = next(reversed(context['observations']))
        context['observations'][key].pop()
        if not context['observations'][key]:
            del context['observations'][key]
    # Unicode explicit strings can consume four bytes per character. The full
    # operator record is retained; a prompt is only a bounded optional excerpt.
    while len(encode(context).encode()) > MAX_CONTEXT_BYTES and context['explicit_preferences']:
        context['explicit_preferences'].pop(next(reversed(context['explicit_preferences'])))
        context['coverage']['explicit_excerpt'] = True
    if len(encode(context).encode()) > MAX_CONTEXT_BYTES:
        raise ValueError('context_budget_exceeded')
    return context
