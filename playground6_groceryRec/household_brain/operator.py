"""Internal management only. Never mount on a customer endpoint or agent tool."""
import re
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from family_brain.core import corrected_state
from .core import digest,encode,obj,stamp
from . import sources,store


def change(conn, operator_id, request):
    """Operator identity is supplied by the trusted CLI's STS check, never a body."""
    store.dedicated(conn)
    operation_id=request.get('operation_id','')
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,64}',operation_id):
        raise ValueError('invalid_operation_id')
    household=request.get('household_id')
    if not isinstance(household,str) or not 1<=len(household)<=64 or not operator_id:
        raise ValueError('invalid_identity')
    reason=request.get('reason')
    if not isinstance(reason,str) or not 5<=len(reason.strip())<=500:
        raise ValueError('operator_reason_required')
    intent=digest({k:v for k,v in request.items() if k!='expected_revision'})
    now=datetime.now(timezone.utc).replace(tzinfo=None)
    conn.begin()
    try:
        with conn.cursor() as cur:
            state=store.state_for(cur,household,True)
            if not state:
                raise ValueError('household_not_provisioned')
            cur.execute('SELECT * FROM household_brain_audit WHERE operation_id=%s',[operation_id])
            old=cur.fetchone()
            if old:
                if old['operator_id']!=operator_id or old['request_hash']!=intent:
                    raise ValueError('operation_conflict')
                conn.commit()
                return obj(old['result'])
            if type(request.get('expected_revision')) is not int or request['expected_revision']!=state['input_revision']:
                raise ValueError('revision_conflict')
            member_ids=sources.members(cur,household)
            if not member_ids:
                raise ValueError('household_has_no_members')
            action=request.get('action')
            actor=request.get('actor_id')
            if actor and actor not in member_ids:
                raise PermissionError('actor_not_current_member')
            if action in {'set','forget'} or (actor and action in {'pause','resume','reset'}):
                if not actor:
                    raise ValueError('preference_actor_required')
                cur.execute('SELECT * FROM family_brain_subjects WHERE actor_id=%s FOR UPDATE',[actor])
                subject=cur.fetchone()
                if not subject or subject['household_id']!=household:
                    learning_since=now if subject else state['learning_since']
                    subject={'actor_id':actor,'household_id':household,'enabled':1,'paused':0,'generation':1,
                             'revision':1,'facts':{},'learning_since':learning_since}
                subject=corrected_state(subject,request)
                if action=='set':
                    subject['facts'][request['key']]['provenance']={'operator':operator_id,'operation_id':operation_id,'reason':reason,'at':now.isoformat()+'Z'}
                cur.execute('''INSERT INTO family_brain_subjects(actor_id,household_id,enabled,paused,generation,revision,learning_since,facts)
                    VALUES(%s,%s,1,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE household_id=VALUES(household_id),enabled=1,
                    paused=VALUES(paused),generation=VALUES(generation),revision=VALUES(revision),learning_since=VALUES(learning_since),
                    facts=VALUES(facts),updated_at=UTC_TIMESTAMP(6)''',
                    [actor,household,subject['paused'],subject['generation'],subject['revision'],stamp(subject['learning_since']).replace(tzinfo=None),encode(subject['facts'])])
            elif action in {'suppress_product','restore_product'}:
                key=request.get('product_key','')
                if not re.fullmatch('[a-f0-9]{64}',key):
                    raise ValueError('invalid_product_key')
                audience=actor or 'household'
                if action=='suppress_product':
                    cur.execute('INSERT IGNORE INTO household_brain_suppressions(household_id,audience,product_key) VALUES(%s,%s,%s)',[household,audience,key])
                else:
                    cur.execute('DELETE FROM household_brain_suppressions WHERE household_id=%s AND audience=%s AND product_key=%s',[household,audience,key])
            elif action in {'enable','disable','pause','resume','reset','timezone'}:
                if actor:
                    raise ValueError('household_action_has_no_actor')
                if action=='enable':
                    backfill=request.get('backfill_days',0)
                    if type(backfill) is not int or not 0<=backfill<=90:
                        raise ValueError('invalid_backfill_days')
                    # Enrollment is not a backdoor to undo a reset on an existing brain.
                    if state['generation']>1 and backfill:
                        raise ValueError('cannot_backfill_reset_history')
                    cur.execute('UPDATE household_brain_state SET enabled=1,learning_since=%s WHERE household_id=%s',
                                [min(state['learning_since'],now-timedelta(days=backfill)) if state['generation']==1 else state['learning_since'],household])
                elif action=='disable':
                    cur.execute('UPDATE household_brain_state SET enabled=0 WHERE household_id=%s',[household])
                elif action in {'pause','resume','reset'}:
                    cur.execute('UPDATE household_brain_state SET paused=%s,generation=generation+1,learning_since=%s WHERE household_id=%s',
                                [action=='pause',now if action!='pause' else state['learning_since'],household])
                else:
                    tz=request.get('timezone')
                    ZoneInfo(tz)
                    cur.execute('UPDATE household_brain_state SET timezone=%s WHERE household_id=%s',[tz,household])
            else:
                raise ValueError('unsupported_action')
            cur.execute('UPDATE household_brain_state SET input_revision=input_revision+1,dirty=1,updated_at=UTC_TIMESTAMP(6) WHERE household_id=%s',[household])
            cur.execute('DELETE FROM household_brain_contexts WHERE household_id=%s',[household])
            updated=store.state_for(cur,household)
            result={'ok':True,'household_id':household,'revision':updated['input_revision'],'generation':updated['generation']}
            cur.execute('INSERT INTO household_brain_audit(operation_id,operator_id,household_id,action,request_hash,request,result) VALUES(%s,%s,%s,%s,%s,%s,%s)',
                        [operation_id,operator_id,household,action,intent,encode(request),encode(result)])
        conn.commit()
        return result
    except BaseException:
        conn.rollback()
        raise


def inspect(conn,household):
    with conn.cursor() as cur:
        state=store.state_for(cur,household)
        if not state:
            return None
        result={'state':state,'members':sources.members(cur,household)}
        for name,table in [('preferences','family_brain_subjects'),('suppressions','household_brain_suppressions'),('contexts','household_brain_contexts')]:
            cur.execute(f'SELECT * FROM {table} WHERE household_id=%s',[household])
            result[name]=cur.fetchall()
        cur.execute('SELECT * FROM household_brain_audit WHERE household_id=%s ORDER BY created_at DESC LIMIT 50',[household])
        result['audit']=cur.fetchall()
        cur.execute('SELECT producer,event_id,kind,actor_id,occurred_at,reversed_by FROM household_brain_events WHERE household_id=%s ORDER BY occurred_at DESC LIMIT 50',[household])
        result['recent_events']=cur.fetchall()
        return result
