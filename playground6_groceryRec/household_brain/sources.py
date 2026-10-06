"""Allowlisted, bounded canonical reads. A missing/oversized source fails closed."""
from datetime import datetime, timedelta, timezone

CAP = 2000
MEMBER_CAP = 64
INVENTORY = ('_id,owner_id,product_name,brand,variant,barcode,_createdDate,'
             'quantity_value,fill_percent,product_expiration,action,needs_review,'
             'analysis_status,provisional_payload,job_id')


def bounded(cur, sql, args, cap=CAP):
    cur.execute(sql + ' LIMIT %s', [*args, cap+1])
    rows = list(cur.fetchall())
    if len(rows) > cap:
        raise ValueError('source_limit_exceeded')
    return rows


def membership(cur, actor, lock=False):
    cur.execute("SELECT COALESCE(NULLIF(owner_id,''),user_id) household FROM new_users WHERE user_id=%s"
                + (' LOCK IN SHARE MODE' if lock else ''), [actor])
    row = cur.fetchone()
    return row['household'] if row else None


def members(cur, household):
    return [r['user_id'] for r in bounded(cur,
        "SELECT user_id FROM new_users WHERE COALESCE(NULLIF(owner_id,''),user_id)=%s ORDER BY user_id",
        [household], MEMBER_CAP)]


def load(cur, state, actor, now):
    household = state['household_id']
    member_ids = members(cur, household)
    if not member_ids or (actor and actor not in member_ids):
        raise PermissionError('membership_changed')
    placeholders = ','.join(['%s']*len(member_ids))
    # Actual shared inventory is owned by current users; never broaden to a supplied
    # household identifier which is not itself a current member.
    owner_filter = 'owner_id IN (' + placeholders + ')'
    since = max(state['learning_since'], (now-timedelta(days=90)).replace(tzinfo=None))
    source = {'members': member_ids, 'sources': []}
    def get(key, table, fields, where, args):
        source[key] = bounded(cur, f'SELECT {fields} FROM {table} WHERE {where}', args)
        source['sources'].append(table)
    get('subjects','family_brain_subjects','*','household_id=%s',[household])
    get('kitchen','shared_kitchen',INVENTORY,owner_filter,member_ids)
    get('archive','shared_archive_kitchen',INVENTORY,owner_filter+' AND _createdDate>=%s',[*member_ids,since])
    # Lists are mirrored per member. Use exactly one current member's view.
    get('shopping','shared_shopping_list','_id,product_name,action', 'owner_id=%s', [actor or member_ids[0]])
    if actor:
        get('recipes','shared_saved_recipes','_id,owner_id,title,status,_createdDate',
            'owner_id=%s AND _createdDate>=%s',[actor,since])
        get('dishes','shared_dishes','_id,owner_id,user_id,dish_name,action,analysis_status,_createdDate',
            f'{owner_filter} AND user_id=%s AND _createdDate>=%s',[*member_ids,actor,since])
        get('calendar','shared_meal_calendar','_id,owner_id,title,_createdDate',
            'owner_id=%s AND _createdDate>=%s',[actor,since])
        get('amounts','kitchen_amount_operations','actor,operation_id,created_at,undone_by,changes',
            'actor=%s AND created_at>=%s',[actor,since])
    get('events','household_brain_events','*',
        "household_id=%s AND generation=%s AND occurred_at>=%s AND (actor_id=%s OR JSON_UNQUOTE(JSON_EXTRACT(payload,'$.visibility'))='household')",
        [household,state['generation'],since,actor])
    cur.execute("SELECT product_key FROM household_brain_suppressions WHERE household_id=%s AND audience IN ('household',%s)",
                [household,actor or 'household'])
    source['suppressed'] = [r['product_key'] for r in cur.fetchall()]
    return source
