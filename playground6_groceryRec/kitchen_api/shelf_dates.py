"""Explicit event-date corrections; observation time never replaces unknown history."""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from shelf_lifecycle import object_value
from shelf_lifecycle_store import enabled_for, materialize, INPUT_FIELDS

KEYS = {'opened_on':'__opened_at', 'frozen_on':'__storage_started_at',
        'acquired_on':'__acquired_on', 'cooked_on':'__cooked_on'}


def apply(conn, cur, actor, operation, row, values):
    from amount_operations import Rejected, dumps
    if not isinstance(values,dict) or not values or set(values)-set(KEYS):raise Rejected('Invalid shelf dates.')
    # Bought-date corrections are available to every household. Other lifecycle
    # date controls retain their existing rollout gate.
    if set(values)!={'acquired_on'} and not enabled_for(conn, actor):raise Rejected('Shelf date corrections are not available.',404)
    state=materialize(conn,[row])[row['_id']]
    today=datetime.now(timezone.utc).astimezone(ZoneInfo(state['time_zone'])).date()
    clean={}
    for key,value in values.items():
        if value is not None:
            try:
                if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value or date.fromisoformat(value)>today:raise ValueError()
            except (ValueError,TypeError):raise Rejected('Choose a valid date that is not in the future.')
        clean[KEYS[key]]=value
    cur.execute('SELECT overrides FROM kitchen_item_edits WHERE item_id=%s FOR UPDATE',(row['_id'],))
    saved=object_value(cur.fetchone()['overrides']);saved.update(clean)
    saved['__shelf_clock_correction']=operation
    saved['__shelf_corrected_keys']=sorted(values)
    history=object_value(saved.get('__shelf_history'))
    if 'opened_on' in values:history['first_opened_at']=values['opened_on']
    saved['__shelf_history']=history
    cur.execute('UPDATE kitchen_item_edits SET overrides=%s,revision=revision+1 WHERE item_id=%s',(dumps(saved),row['_id']))
    cur.execute('SELECT * FROM shared_kitchen WHERE _id=%s',(row['_id'],))
    current=cur.fetchone()
    evidence={key:current.get(key) for key in INPUT_FIELDS}
    evidence.update(_shelf_edits=saved,_shelf_event='CORRECT_DATE')
    cur.execute('INSERT INTO kitchen_shelf_events(item_id,owner_id,evidence) VALUES(%s,%s,%s)',(row['_id'],row['owner_id'],dumps(evidence)))
    cur.execute('UPDATE kitchen_shelf_state SET needs_work=TRUE WHERE item_id=%s',(row['_id'],))
