"""Durable receipts from trusted producers, not arbitrary client analytics."""
import re
from datetime import datetime,timedelta,timezone
from . import sources,store
from .core import digest,encode

PRODUCERS={'shop':{'product_selected'},'orders':{'purchase'},'recipe-feedback':{'recipe_liked','recipe_disliked'}}


def ingest(conn, producer, event):
    store.dedicated(conn)
    if producer not in PRODUCERS:
        raise PermissionError('producer_not_allowed')
    for key in ['event_id','entity_id','actor_id','household_id']:
        if not isinstance(event.get(key),str) or not 1<=len(event[key])<=(160 if key in {'event_id','entity_id'} else 64):
            raise ValueError('invalid_event_identity')
    if set(event)-{'event_id','entity_id','actor_id','household_id','occurred_at','kind','payload','reverses','generation'}:
        raise ValueError('unsupported_event_field')
    from .core import stamp
    at=stamp(event.get('occurred_at'))
    now=datetime.now(timezone.utc)
    if not at or not now-timedelta(days=90)<=at<=now:
        raise ValueError('invalid_event_time')
    payload=event.get('payload')
    if not isinstance(payload,dict) or set(payload)-{'name','brand','variant','barcode','visibility'}:
        raise ValueError('invalid_event_payload')
    if payload.get('visibility') not in {'personal','household'}:
        raise ValueError('explicit_visibility_required')
    if any(not isinstance(v,str) or not v.strip() or len(v)>160 for v in payload.values()):
        raise ValueError('invalid_event_value')
    reverse=event.get('kind')=='reverse'
    if reverse:
        if not isinstance(event.get('reverses'),str) or not 1<=len(event['reverses'])<=160:
            raise ValueError('reversal_target_required')
    elif event.get('kind') not in PRODUCERS[producer] or not payload.get('name') or event.get('reverses'):
        raise ValueError('unsupported_event_kind')
    intent=digest(event)
    conn.begin()
    try:
        with conn.cursor() as cur:
            if sources.membership(cur,event['actor_id'],lock=True)!=event['household_id']:
                raise PermissionError('membership_changed')
            state=store.state_for(cur,event['household_id'],True)
            if not store.active(cur) or not state or not state['enabled']:
                raise ValueError('brain_not_enabled')
            cur.execute('SELECT request_hash FROM household_brain_events WHERE producer=%s AND event_id=%s',[producer,event['event_id']])
            previous=cur.fetchone()
            if previous:
                if previous['request_hash']!=intent:
                    raise ValueError('event_id_conflict')
                conn.commit()
                return {'ok':True,'duplicate':True}
            if type(event.get('generation')) is not int or event['generation']!=state['generation']:
                raise ValueError('event_generation_changed')
            if state['paused']:
                raise ValueError('learning_paused')
            if at.replace(tzinfo=None)<state['learning_since']:
                raise ValueError('event_predates_learning_window')
            reversed_by=None
            if reverse:
                cur.execute('SELECT household_id,actor_id FROM household_brain_events WHERE producer=%s AND event_id=%s',[producer,event['reverses']])
                target=cur.fetchone()
                if target and (target['household_id']!=event['household_id'] or target['actor_id']!=event['actor_id']):
                    raise PermissionError('reversal_scope_mismatch')
                cur.execute('INSERT INTO household_brain_reversals(producer,event_id,household_id,actor_id,reversal_id) VALUES(%s,%s,%s,%s,%s)',
                    [producer,event['reverses'],event['household_id'],event['actor_id'],event['event_id']])
                cur.execute('UPDATE household_brain_events SET reversed_by=%s WHERE producer=%s AND event_id=%s',[event['event_id'],producer,event['reverses']])
                reversed_by=event['event_id']
            else:
                cur.execute('SELECT * FROM household_brain_reversals WHERE producer=%s AND event_id=%s',[producer,event['event_id']])
                reversal=cur.fetchone()
                if reversal:
                    if reversal['household_id']!=event['household_id'] or reversal['actor_id']!=event['actor_id']:
                        raise PermissionError('reversal_scope_mismatch')
                    reversed_by=reversal['reversal_id']
            cur.execute('''INSERT INTO household_brain_events(producer,event_id,household_id,actor_id,generation,kind,entity_id,occurred_at,payload,request_hash,reversed_by)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',[producer,event['event_id'],event['household_id'],event['actor_id'],state['generation'],event['kind'],event['entity_id'],at.replace(tzinfo=None),encode(payload),intent,reversed_by])
            cur.execute('UPDATE household_brain_state SET input_revision=input_revision+1,dirty=1,updated_at=UTC_TIMESTAMP(6) WHERE household_id=%s',[event['household_id']])
        conn.commit()
        return {'ok':True,'duplicate':False}
    except BaseException:
        conn.rollback()
        raise
