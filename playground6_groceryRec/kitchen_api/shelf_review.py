"""Attention decisions stored in the existing protected edit sidecar.

Called inside the authenticated amount operation transaction, after item and
revision locks. The operation ledger supplies idempotency and lost-ack recovery.
"""
import json
from datetime import datetime, timedelta, timezone, time, date
from zoneinfo import ZoneInfo
from shelf_lifecycle_store import enabled_for, materialize
from shelf_lifecycle import object_value


def apply(conn, cur, actor, operation, row, meta, entry, now=None):
    from amount_operations import Rejected, dumps
    if not enabled_for(conn, actor):
        raise Rejected('Shelf review is not available for this household.', 404)
    action = entry.get('decision')
    if action not in ('keep', 'snooze', 'clear'):
        raise Rejected('Choose Keep, Remind tomorrow, or Undo.')
    now = now or datetime.now(timezone.utc)
    state = materialize(conn, [row], now=now)[row['_id']]
    if action == 'snooze' and not state.get('reminders_enabled'):
        raise Rejected('Reminders are not enabled for this household yet.', 409)
    if entry.get('generation') != state['generation']:
        raise Rejected('This item changed. Review its latest date first.', 409)
    edits = object_value(meta['overrides'])
    prior = object_value(edits.get('__shelf_review'))
    if action == 'clear':
        if not entry.get('undo_of') or prior.get('operation_id') != entry['undo_of']:
            raise Rejected('A newer review has replaced this action.', 409)
        edits.pop('__shelf_review', None)
    else:
        zone = ZoneInfo(state['time_zone'])
        local = now.astimezone(zone)
        days = 1 if state.get('days_left') is None or state['days_left'] <= 3 else 3
        until = local + timedelta(days=days)
        if state.get('deadline') and state.get('days_left', 0) > 0:
            until = min(until, datetime.combine(date.fromisoformat(state['deadline']), time(), zone))
        reminder = datetime.combine(local.date() + timedelta(days=1), time(9), zone)
        edits['__shelf_review'] = dict(operation_id=operation, actor=actor,
            generation=state['generation'], reviewed_at=now.isoformat(),
            clean_until=until.astimezone(timezone.utc).isoformat(),
            remind_after=reminder.astimezone(timezone.utc).isoformat() if action == 'snooze' else prior.get('remind_after') if prior.get('generation') == state['generation'] else None)
    cur.execute('UPDATE kitchen_item_edits SET overrides=%s,revision=revision+1 WHERE item_id=%s',
                (dumps(edits), row['_id']))
    return edits
