"""Bounded shared-calendar snapshots for reservation-aware availability.

Callers resolve authorized calendar owners before calling. Read failures are not
an empty calendar. Manual notes carry no ingredient reservation. No writes or DDL.
"""
import datetime
import json
from quantity_sufficiency import reservation_token

class ReservationSnapshotUnavailable(Exception):
    pass

def read_reservation_state(conn, owners, today=None, max_rows=1000, for_update=False, max_ingredients=1000, clock_owner=None):
    if (not isinstance(owners, (list, tuple)) or not owners or len(owners)>100
            or any(not isinstance(x,str) or not x for x in owners)):
        raise ReservationSnapshotUnavailable('Invalid calendar scope')
    if type(max_rows) is not int or not 1 <= max_rows <= 1000:
        raise ReservationSnapshotUnavailable('Invalid snapshot bound')
    if type(max_ingredients) is not int or not 1 <= max_ingredients <= 1000:
        raise ReservationSnapshotUnavailable('Invalid ingredient bound')
    if type(for_update) is not bool:
        raise ReservationSnapshotUnavailable('Invalid snapshot lock mode')
    scope=sorted(set(owners))
    if clock_owner is not None and clock_owner not in scope:
        raise ReservationSnapshotUnavailable('Clock owner outside calendar scope')
    try:
        if clock_owner is not None:
            if today is not None:
                raise ReservationSnapshotUnavailable('Conflicting calendar clocks')
            from reservation_clock import current_owner_day
            today = current_owner_day(clock_owner)
        with conn.cursor() as cur:
            if today is None:
                cur.execute('SELECT CURRENT_DATE AS reservation_day')
                row=cur.fetchone()
                today=row['reservation_day'] if isinstance(row,dict) else None
            if type(today) is not datetime.date:
                raise ReservationSnapshotUnavailable('A calendar date is required')
            placeholders=', '.join(['%s']*len(scope))
            cur.execute('SELECT CASE WHEN OCTET_LENGTH(ingredients)>16384 THEN NULL ELSE ingredients END AS ingredients, '
                        '(OCTET_LENGTH(ingredients)>16384) AS ingredients_oversized, source_type, meal_category FROM shared_meal_calendar '
                        f'WHERE owner_id IN ({placeholders}) AND plan_date >= %s '
                        'ORDER BY owner_id, plan_date, _id LIMIT %s'+(' FOR UPDATE' if for_update else ''),scope+[today.isoformat(),max_rows+1])
            rows=cur.fetchall()
        if not isinstance(rows,(list,tuple)) or len(rows)>max_rows:
            raise ReservationSnapshotUnavailable('Calendar snapshot incomplete')
        ingredients=[]
        ingredient_count=0
        for row in rows:
            if not isinstance(row,dict):
                raise ReservationSnapshotUnavailable('Malformed calendar row')
            if row.get('meal_category')=='__note__':
                continue
            if row.get('ingredients_oversized'):
                raise ReservationSnapshotUnavailable('Planned ingredient row too large')
            raw=row.get('ingredients')
            if raw is None and row.get('source_type')=='manual':
                continue  # Legacy free-text calendar cards contain no recipe snapshot.
            parsed=json.loads(raw) if isinstance(raw,str) else raw
            if not isinstance(parsed,list) or any(not isinstance(x,str) for x in parsed):
                raise ReservationSnapshotUnavailable('Malformed planned ingredients')
            ingredient_count += len(parsed)
            if ingredient_count > max_ingredients or any(len(text)>512 for text in parsed):
                raise ReservationSnapshotUnavailable('Planned ingredient workload too large')
            ingredients.extend(text.strip() for text in parsed if text.strip())
        return {'ingredients':ingredients,'reservation_token':reservation_token(ingredients),
                'calendar_date':today.isoformat()}
    except ReservationSnapshotUnavailable:
        raise
    except Exception as exc:
        # Do not put database credentials, customer ingredients or driver errors
        # into the caller-visible error. Preserve the cause for controlled handling.
        raise ReservationSnapshotUnavailable('Calendar snapshot unavailable') from exc

def cached_matches_reservations(matches, state):
    """Legacy mode keeps old cache contracts; enabled snapshots require evidence."""
    if isinstance(matches,list) and any(isinstance(m,dict) and m.get('reservation_unavailable') for m in matches):
        return False
    if state is None:
        return True
    token=state.get('reservation_token') if isinstance(state,dict) else None
    if not isinstance(token,str) or len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
        return False
    return isinstance(matches,list) and bool(matches) and all(
        isinstance(match,dict) and match.get('reservation_token')==token for match in matches)

def calendar_owner_scope(conn, owner, household_enabled=False, for_update=False):
    if not isinstance(owner,str) or not owner:
        raise ReservationSnapshotUnavailable('Invalid calendar owner')
    if not household_enabled:
        return [owner]
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT owner_id FROM new_users WHERE user_id=%s LIMIT 1',[owner])
            row=cur.fetchone()
            if row is None:
                if for_update:
                    cur.execute('SELECT owner_id FROM new_users WHERE user_id=%s LIMIT 1 FOR UPDATE',[owner])
                    if cur.fetchone() is not None:
                        raise ReservationSnapshotUnavailable('Household membership changed')
                return [owner]
            if not isinstance(row,dict):
                raise ReservationSnapshotUnavailable('Invalid household evidence')
            household=row.get('owner_id')
            if not household:
                if for_update:
                    cur.execute('SELECT owner_id FROM new_users WHERE user_id=%s LIMIT 1 FOR UPDATE',[owner])
                    if cur.fetchone() != row:
                        raise ReservationSnapshotUnavailable('Household membership changed')
                return [owner]
            cur.execute('SELECT user_id FROM new_users WHERE owner_id=%s ORDER BY user_id LIMIT 101'+(' FOR UPDATE' if for_update else ''),[household])
            rows=cur.fetchall()
        if not isinstance(rows,(list,tuple)) or len(rows)>100 or any(not isinstance(x,dict) or not isinstance(x.get('user_id'),str) or not x['user_id'] for x in rows):
            raise ReservationSnapshotUnavailable('Invalid household evidence')
        members=sorted(set(x['user_id'] for x in rows))
        if owner not in members:
            raise ReservationSnapshotUnavailable('Household membership changed')
        return members
    except ReservationSnapshotUnavailable:
        raise
    except Exception as exc:
        raise ReservationSnapshotUnavailable('Household scope unavailable') from exc
