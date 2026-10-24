"""Owner-scoped confirmations in the existing versioned availability cache.

Call only after authenticating the owner. No inventory rows or quantities are
changed. A kitchen-version change invalidates the normal availability cache,
including these confirmations. No request-time schema changes.
"""
import ingredient_identity as identity
import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timezone


class ConfirmationRejected(Exception):
    def __init__(self, code, status=409):
        self.code = code
        self.status = status
        super().__init__(code)


def require_authenticated_owner(event, owner):
    """New confirmation writes require a signature, independent of legacy observe mode."""
    from trepo_auth import _get_header, _verify_jwt
    if not isinstance(event, dict) or not isinstance(event.get('headers') or {}, dict) or not isinstance(owner, str) or not owner:
        raise ConfirmationRejected('SIGN_IN_REQUIRED', 403)
    header = _get_header(event, 'authorization')
    if not isinstance(header, str) or not header.lower().startswith('bearer '):
        raise ConfirmationRejected('SIGN_IN_REQUIRED', 403)
    claims = _verify_jwt(header.split(' ', 1)[1].strip())
    if not claims:
        raise ConfirmationRejected('SIGN_IN_REQUIRED', 403)
    claimed_owner = claims.get('user_id') or claims.get('owner_id')
    if not isinstance(claimed_owner, str) or claimed_owner.lower() != str(owner or '').lower():
        raise ConfirmationRejected('OWNER_MISMATCH', 403)


def availability_token(availability):
    payload = {
        'version': 1,
        'kitchen_version': int(availability.get('kitchen_version') or 0),
        'ingredient_matches': availability.get('ingredient_matches') or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, default=str).encode()).hexdigest()


def evidence_token(availability):
    """Stable across duplicate computations of the same inventory evidence."""
    matches = []
    for match in availability.get('ingredient_matches') or []:
        if not isinstance(match, dict):
            raise ConfirmationRejected('REFRESH_AVAILABILITY')
        matches.append({
            'ingredient': match.get('recipe_ingredient'),
            'status': match.get('match_status'),
            'optional': bool(match.get('is_optional')),
            'alternative': match.get('selected_alternative'),
            'matcher_version': match.get('matcher_version'),
            'quantity': 'unknown' if match.get('quantity_status') == 'confirmed_enough' else match.get('quantity_status'),
            'items': sorted((match.get('matched_kitchen_items') or []), key=lambda item: json.dumps(item, sort_keys=True)),
        })
        if 'saved_recipe_evidence' in match:
            matches[-1]['saved_recipe_evidence'] = match['saved_recipe_evidence']
        if 'reservation_token' in match:
            matches[-1]['reservation_token'] = match['reservation_token']
        if match.get('reservation_unavailable'):
            matches[-1]['reservation_unavailable'] = True
    return availability_token({'kitchen_version': availability.get('kitchen_version'), 'ingredient_matches': matches})


def _ready(matches):
    return bool(matches) and all(
        m.get('is_optional') or m.get('match_status') == 'optional' or (
            m.get('match_status') in ('have', 'pantry') and
            m.get('quantity_status') in ('sufficient', 'confirmed_enough', 'confirmed_staple')
        ) for m in matches
    )


def _json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            raise ConfirmationRejected('REFRESH_AVAILABILITY')
    return value


def confirm_cached_availability(conn, owner, source, recipe_id, request, saved_recipe_guard=None, reservation_state_reader=None):
    if not owner or source not in ('saved', 'explore', 'explore_personal') or not isinstance(recipe_id, str) or not recipe_id or len(recipe_id) > 255:
        raise ConfirmationRejected('INVALID_RECIPE', 400)
    if not isinstance(request, dict):
        raise ConfirmationRejected('INVALID_CONFIRMATION', 400)
    version = request.get('kitchen_version')
    indices = request.get('ingredient_indices')
    token = request.get('availability_token')
    try:
        operation_id = str(uuid.UUID(request.get('operation_id', '')))
    except (ValueError, TypeError, AttributeError):
        raise ConfirmationRejected('INVALID_OPERATION_ID', 400)
    if type(version) is not int or version < 0 or not isinstance(token, str) or not re.fullmatch('[a-f0-9]{64}', token):
        raise ConfirmationRejected('INVALID_AVAILABILITY', 400)
    if not isinstance(indices, list) or not 1 <= len(indices) <= 200 or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ConfirmationRejected('INVALID_INGREDIENTS', 400)
    indices = sorted(indices)
    conn.begin()
    try:
        validate_recipe = (saved_recipe_guard(conn, owner, recipe_id)
                           if source == 'saved' and saved_recipe_guard else None)
        with conn.cursor() as cur:
            # Lock in the same owner/version order for every confirmation.
            cur.execute('SELECT kitchen_version FROM owner_kitchen_state WHERE owner=%s FOR UPDATE', [owner])
            state = cur.fetchone() or {}
            if int(state.get('kitchen_version') or 0) != version:
                raise ConfirmationRejected('KITCHEN_CHANGED')
            cur.execute('''SELECT * FROM owner_recipe_availability
                           WHERE owner=%s AND recipe_source=%s AND recipe_id=%s FOR UPDATE''',
                        [owner, source + ':q4' + identity.VERSION, recipe_id])
            row = cur.fetchone()
            if not row or int(row.get('kitchen_version') or 0) != version:
                raise ConfirmationRejected('REFRESH_AVAILABILITY')
            if validate_recipe and not validate_recipe(row):
                raise ConfirmationRejected('REFRESH_AVAILABILITY')
            matches = _json(row.get('ingredient_matches'), [])
            if not isinstance(matches, list) or not matches or len(matches) > 400 or any(not isinstance(m, dict) for m in matches) or indices[-1] >= len(matches):
                raise ConfirmationRejected('REFRESH_AVAILABILITY')
            if reservation_state_reader is not None:
                from meal_reservation_state import cached_matches_reservations, ReservationSnapshotUnavailable
                try:
                    reservation_state = reservation_state_reader(conn)
                except ReservationSnapshotUnavailable:
                    raise ConfirmationRejected('RESERVATIONS_UNAVAILABLE', 503)
                if not cached_matches_reservations(matches, reservation_state):
                    raise ConfirmationRejected('RESERVATIONS_CHANGED')
            row['ingredient_matches'] = matches
            for field in ('missing_ingredients', 'substitution_candidates'):
                row[field] = _json(row.get(field), [])
                if not isinstance(row[field], list):
                    raise ConfirmationRejected('REFRESH_AVAILABILITY')
            receipt = {'operation_id': operation_id, 'input_token': token,
                       'kitchen_version': version, 'ingredient_indices': indices}
            previous = [m.get('quantity_confirmation') for m in matches
                        if isinstance(m.get('quantity_confirmation'), dict)
                        and m['quantity_confirmation'].get('operation_id') == operation_id]
            if previous:
                # Retrying an acknowledged request never changes quantities or
                # broadens the list of ingredients the person confirmed.
                if any(any(p.get(k) != v for k, v in receipt.items()) for p in previous) or any(
                    matches[i].get('quantity_confirmation', {}).get('operation_id') != operation_id for i in indices
                ):
                    raise ConfirmationRejected('OPERATION_REUSED')
            else:
                if not hmac.compare_digest(availability_token(row), token):
                    raise ConfirmationRejected('AVAILABILITY_CHANGED')
                for i in indices:
                    match = matches[i]
                    if match.get('match_status') != 'have' or match.get('quantity_status') != 'unknown' or not match.get('matched_kitchen_items'):
                        raise ConfirmationRejected('INGREDIENT_NOT_CONFIRMABLE')
                receipt['confirmed_at'] = datetime.now(timezone.utc).isoformat()
                receipt['evidence_token'] = evidence_token(row)
                for i in indices:
                    matches[i].update(quantity_status='confirmed_enough', quantity_confirmation=receipt,
                                      missing_reason=None)
                row['can_make_exact'] = _ready(matches)
                # A confirmation never approves unmeasured substitute ratios.
                row['can_make_with_subs'] = False
                cur.execute('''UPDATE owner_recipe_availability SET ingredient_matches=%s,
                               can_make_exact=%s, can_make_with_subs=0
                               WHERE owner=%s AND recipe_source=%s AND recipe_id=%s AND kitchen_version=%s''',
                            [json.dumps(matches), int(row['can_make_exact']), owner, source + ':q4' + identity.VERSION, recipe_id, version])
        conn.commit()
        row['can_make_exact'] = bool(row.get('can_make_exact'))
        row['can_make_with_subs'] = bool(row.get('can_make_with_subs'))
        return row
    except Exception:
        conn.rollback()
        raise


CACHE_COLUMNS = (
    'owner', 'recipe_source', 'recipe_id', 'kitchen_version', 'can_make_exact', 'can_make_with_subs',
    'matched_count', 'missing_count', 'ingredient_matches', 'missing_ingredients',
    'substitution_candidates', 'substitution_summary', 'substitution_status', 'analysis_status',
)


def preserve_confirmations(previous, incoming):
    if not previous:
        return incoming
    previous_version = int(previous.get('kitchen_version') or 0)
    incoming_version = int(incoming.get('kitchen_version') or 0)
    if previous_version > incoming_version:
        return previous  # A late worker cannot replace a newer kitchen check.
    if previous_version != incoming_version:
        return incoming
    old_matches = _json(previous.get('ingredient_matches'), [])
    new_matches = _json(incoming.get('ingredient_matches'), [])
    evidence = evidence_token({**incoming, 'ingredient_matches': new_matches})
    if evidence != evidence_token({**previous, 'ingredient_matches': old_matches}):
        return incoming
    restored = False
    for index, old in enumerate(old_matches):
        receipt = old.get('quantity_confirmation') or {}
        if old.get('quantity_status') == 'confirmed_enough' and receipt.get('evidence_token') == evidence and receipt.get('kitchen_version') == incoming_version:
            new_matches[index] = old
            restored = True
    if not restored:
        return incoming
    return {**incoming, 'ingredient_matches': json.dumps(new_matches),
            'can_make_exact': int(_ready(new_matches)), 'can_make_with_subs': 0}


def confirmed_availability_for_response(conn, owner, source, recipe_id, availability):
    """Reopening a recipe must show the same confirmation that its cache retains.

    This is a read, not a confirmation write. Only identical ingredient/inventory
    evidence at the same version can carry a previous explicit choice forward.
    """
    if source not in ('saved', 'explore', 'explore_personal') or not recipe_id:
        return availability
    if not any(m.get('quantity_status') == 'unknown' for m in availability.get('ingredient_matches') or []):
        return availability
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM owner_recipe_availability WHERE owner=%s AND recipe_source=%s AND recipe_id=%s',
                    [owner, source + ':q4' + identity.VERSION, recipe_id])
        previous = cur.fetchone()
    if not previous or int(previous.get('kitchen_version') or 0) != int(availability.get('kitchen_version') or 0):
        return availability
    try:
        merged = preserve_confirmations(previous, availability)
        return {**availability, 'ingredient_matches': _json(merged.get('ingredient_matches'), []),
                'can_make_exact': bool(merged.get('can_make_exact')),
                'can_make_with_subs': bool(merged.get('can_make_with_subs'))}
    except ConfirmationRejected:
        # A broken old cache must not turn a successful fresh recipe read into an error.
        return availability


def persist_cache_rows(conn, rows, saved_recipe_guard=None, reservation_state_reader=None):
    """Serialize cache refreshes with confirmations; preserve independent sources."""
    records = [dict(row) if isinstance(row, dict) else dict(zip(CACHE_COLUMNS, row)) for row in rows]
    # Transient outage results remain response-only; never replace durable evidence.
    records = [record for record in records if not any(
        isinstance(match, dict) and match.get('reservation_unavailable')
        for match in _json(record.get('ingredient_matches'), []))]
    if not records:
        return []
    reservation_states = {}
    records.sort(key=lambda row: (row['owner'], row['recipe_source'], row['recipe_id']))
    conn.begin()
    try:
        guards = {}
        saved_source = 'saved:q4' + identity.VERSION
        if saved_recipe_guard:
            for owner in sorted({r['owner'] for r in records if r.get('recipe_source') == saved_source}):
                ids = [r['recipe_id'] for r in records if r['owner'] == owner and r.get('recipe_source') == saved_source]
                guards[owner] = saved_recipe_guard(conn, owner, ids)
        accepted = []
        with conn.cursor() as cur:
            for record in records:
                if record.get('recipe_source') == saved_source and record['owner'] in guards and not guards[record['owner']](record):
                    continue
                if record.get('recipe_source') not in tuple(source + ':q4' + identity.VERSION for source in ('saved', 'explore', 'explore_personal')):
                    raise ValueError('Unsupported quantity cache source')
                key = [record['owner'], record['recipe_source'], record['recipe_id']]
                cur.execute('SELECT * FROM owner_recipe_availability WHERE owner=%s AND recipe_source=%s AND recipe_id=%s FOR UPDATE', key)
                previous = cur.fetchone()
                if reservation_state_reader is not None:
                    from meal_reservation_state import cached_matches_reservations, ReservationSnapshotUnavailable
                    try:
                        if record['owner'] not in reservation_states:
                            reservation_states[record['owner']] = reservation_state_reader(conn, record['owner'])
                    except ReservationSnapshotUnavailable:
                        raise ConfirmationRejected('RESERVATIONS_UNAVAILABLE', 503)
                    if not cached_matches_reservations(_json(record.get('ingredient_matches'), []), reservation_states[record['owner']]):
                        raise ConfirmationRejected('RESERVATIONS_CHANGED')
                merged = preserve_confirmations(previous, record)
                names = ','.join('`' + name + '`' for name in CACHE_COLUMNS)
                updates = ','.join('`' + name + '`=VALUES(`' + name + '`)' for name in CACHE_COLUMNS[3:])
                cur.execute('INSERT INTO owner_recipe_availability (' + names + ') VALUES (' + ','.join(['%s'] * len(CACHE_COLUMNS)) + ') ON DUPLICATE KEY UPDATE ' + updates,
                            [merged.get(name) for name in CACHE_COLUMNS])
                accepted.append(record)
        conn.commit()
        return accepted
    except Exception:
        conn.rollback()
        raise
