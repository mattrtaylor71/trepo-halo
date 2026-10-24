"""Bounded, provider-free updates of existing recipe quantities after plan changes."""
import copy
import re
import meal_reservation_state as reservations
import recipe_inventory_llm as matching
import quantity_sufficiency


def kitchen_context(conn, owner, state, expected_version, shared=False, max_rows=1000):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner or '') or type(max_rows) is not int or not 1 <= max_rows <= 1000:
        raise reservations.ReservationSnapshotUnavailable('Invalid kitchen scope')
    try:
        owners = reservations.calendar_owner_scope(conn, owner, household_enabled=True) if shared else [owner]
        table = 'shared_kitchen' if shared else owner + '_prod_kitchen'
        with conn.cursor() as cur:
            cur.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s', [table])
            columns = {r.get('column_name') or r.get('COLUMN_NAME') for r in cur.fetchall()}
            if not {'_id', 'product_name', 'action'} <= columns:
                raise reservations.ReservationSnapshotUnavailable('Kitchen schema unavailable')
            fields = ['_id', 'product_name'] + [c for c in ('product_description','quantity','quantity_value','quantity_unit') if c in columns]
            where = ["action='IN'"]
            params = []
            if shared:
                where.append('owner_id IN (' + ','.join(['%s']*len(owners)) + ')')
                params.extend(owners)
            if 'analysis_status' in columns:
                where.append("(analysis_status='ready' OR analysis_status IS NULL)")
            cur.execute('SELECT '+','.join('`'+f+'`' for f in fields)+' FROM `'+table+'` WHERE '+ ' AND '.join(where)+' ORDER BY `_id` LIMIT %s',params+[max_rows+1])
            rows = cur.fetchall()
            if len(rows) > max_rows:
                raise reservations.ReservationSnapshotUnavailable('Kitchen snapshot incomplete')
            cur.execute('SELECT kitchen_version FROM owner_kitchen_state WHERE owner=%s LIMIT 1',[owner])
            version = cur.fetchone()
            if not version or int(version['kitchen_version']) != int(expected_version):
                raise reservations.ReservationSnapshotUnavailable('Kitchen changed during read')
        candidates=[]
        for row in rows:
            name = str(row.get('product_name') or '').strip()
            if not name:
                continue
            desc = str(row.get('product_description') or '').strip()
            candidates.append({'item_id':row['_id'],'display_name':name,'description':desc,
                               'tokens':matching.ingredient_tokens(name+' '+desc),
                               'quantity':row.get('quantity'),'quantity_value':row.get('quantity_value'),'quantity_unit':row.get('quantity_unit')})
        quantity_sufficiency.reserve_planned_amounts(state['ingredients'], candidates, matching.ingredient_tokens, matching._score_kitchen_candidate)
        return {'kitchen_candidates':candidates,'kitchen_version':int(expected_version),'reservation_token':state['reservation_token']}
    except reservations.ReservationSnapshotUnavailable:
        raise
    except Exception as exc:
        raise reservations.ReservationSnapshotUnavailable('Kitchen snapshot unavailable') from exc


def update_recipe(recipe, context):
    """Retain identity matches; re-evaluate amounts from current physical stock."""
    raw_matches = recipe.get('ingredient_matches')
    matches = copy.deepcopy(raw_matches) if isinstance(raw_matches,list) and all(isinstance(m,dict) for m in raw_matches) else []
    if not matches:
        availability = matching.deterministic_availability(recipe, context)
    else:
        availability = copy.deepcopy(recipe.get('availability') or {})
        availability['ingredient_matches'] = matches
        for match in matches:
            match.pop('quantity_confirmation', None)
            match.pop('reservation_unavailable', None)
            if match.pop('quantity_status', None) == 'insufficient' and match.get('matched_kitchen_items'):
                match['match_status'] = 'have'
            match['missing_reason'] = None if match.get('match_status')=='have' else match.get('missing_reason')
        availability = matching.enforce_quantity_availability(recipe, context, availability)
    recipe['ingredient_matches'] = availability['ingredient_matches']
    recipe['availability'] = matching.availability_summary(availability)
    # Generated recipes have no confirmation-cache source. Saving a recipe uses
    # the existing Saved confirmation flow; do not advertise an unsupported action.
    recipe['availability'].update(quantity_confirmation_required=False,
                                  quantity_confirmation_items=[], availability_token=None)
    recipe['missing_ingredients'] = availability.get('missing_ingredients') or []
