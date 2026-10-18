"""Coalesced source refresh driven by existing DynamoDB change streams.

Streams are invalidation hints, never preference evidence. The worker re-reads
canonical sources. Replayed hints cannot multiply observations. All sources reconcile daily as recovery for expiry or missed events.
"""
import os
import json
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from . import sources, store
from .core import stamp
from .engagement import feature

KINDS = {'features', 'chat_native', 'chat_legacy'}


def enqueue(conn, actor, kind, *, household_hint=None, partition=None, now=None, retract=True):
    if kind not in KINDS or not isinstance(actor,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',actor):
        return False
    store.dedicated(conn)
    at=(stamp(now) or datetime.now(timezone.utc)).replace(tzinfo=None)
    conn.begin()
    try:
        with conn.cursor() as cur:
            household=sources.membership(cur,actor,lock=True)
            state=store.state_for(cur,household,True) if household else None
            if not state or not store.active(cur) or not state['enabled'] or state['paused'] or household_hint and household_hint!=household:
                conn.rollback(); return False
            if partition is not None:
                if kind!='chat_legacy' or not isinstance(partition,str) or len(partition)>256 or not (partition==household or partition.startswith('family-brain:'+actor+':')):
                    conn.rollback(); return False
                cur.execute('''INSERT INTO household_brain_external_partitions(household_id,actor_id,partition_key,updated_at)
                  VALUES(%s,%s,%s,%s) ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)''',[household,actor,partition,at])
            cur.execute('''INSERT INTO household_brain_external_work(household_id,actor_id,source,dirty_since,next_refresh_at,updated_at)
              VALUES(%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE revision=revision+1,
              dirty_since=IF(dirty=0,VALUES(dirty_since),dirty_since),dirty=1,
              next_refresh_at=VALUES(next_refresh_at),updated_at=VALUES(updated_at)''',
              [household,actor,kind,at,at,at])
            # Retract the previous source immediately on an edit/delete. Context
            # reads may omit optional evidence until the replacement is ready.
            if retract:
                cur.execute('DELETE FROM household_brain_external WHERE household_id=%s AND actor_id=%s AND source=%s',[household,actor,kind])
                cur.execute('''UPDATE household_brain_state SET input_revision=input_revision+1,dirty=1,
                  dirty_since=COALESCE(dirty_since,%s),next_refresh_at=%s,updated_at=%s WHERE household_id=%s''',[at,at,at,household])
        conn.commit(); return True
    except BaseException:
        conn.rollback(); raise


def decode_image(record, name):
    from boto3.dynamodb.types import TypeDeserializer
    decoder=TypeDeserializer()
    return {k:decoder.deserialize(v) for k,v in record.get('dynamodb',{}).get(name,{}).items()}


def hints(record, kind):
    keys=decode_image(record,'Keys')
    images=[decode_image(record,k) for k in ('OldImage','NewImage')]
    if kind=='features':
        if record.get('eventName')=='INSERT' and not any(feature(r.get('event_name')) for r in images):
            return []
        # Deletes on NEW_IMAGE streams retain keys but not the former event name.
        return [(keys.get('owner_id'),None,None)]
    if kind=='chat_native':
        match=re.fullmatch(r'A#([^#]+)#H#([^#]+)',keys.get('pk',''))
        if not match or not keys.get('sk','').startswith('S#'):return []
        return [(match[1],match[2],None)]
    if kind=='chat_legacy':
        partition=keys.get('owner_id');entry=keys.get('session_entry_id','')
        if not entry.startswith('session#') or '#msg#' not in entry:return []
        return list({(r.get('request_owner_id') or r.get('user_id'),None,partition)
                     for r in images if r.get('role')=='user'})
    return []


def stream(conn, event, streams):
    """Configured exact stream ARNs determine source kind; HTTP is unsupported."""
    if not isinstance(event,dict) or set(event)!={'Records'} or not isinstance(event['Records'],list):
        raise ValueError('invalid_stream_event')
    tasks={}
    for record in event['Records']:
        kind=streams.get(record.get('eventSourceARN'))
        if kind not in KINDS or record.get('eventSource')!='aws:dynamodb':
            raise ValueError('untrusted_stream')
        for a,h,p in hints(record,kind):
            if a:
                key=(a,kind,h,p)
                tasks[key]=tasks.get(key,False) or record.get('eventName')!='INSERT'
    # Additions coalesce without interrupting the very chat turn that created
    # them. Corrections and removals retract stale evidence immediately.
    checked=sum(enqueue(conn,a,k,household_hint=h,partition=p,retract=tasks[(a,k,h,p)])
                for a,k,h,p in sorted(tasks,key=str))
    return {'checked':checked}


def sync(conn, client, tables, limit=50, now=None, budget_seconds=45):
    from . import external
    store.dedicated(conn)
    at=stamp(now) or datetime.now(timezone.utc)
    result={'checked':0,'failed':0,'changed_during_read':0}
    if not tables:return result
    if set(tables)-KINDS or type(limit) is not int or not 1<=limit<=100:
        raise ValueError('invalid_external_configuration')
    with conn.cursor() as cur:
        if not store.active(cur):return result
        marks=','.join(['%s']*len(tables))
        cur.execute(f'''SELECT w.* FROM household_brain_external_work w
          JOIN household_brain_state s ON s.household_id=w.household_id
          WHERE s.enabled=1 AND s.paused=0 AND w.source IN ({marks}) AND w.next_refresh_at<=%s
            AND (w.dirty=0 OR w.updated_at<=%s OR w.dirty_since<=%s)
          ORDER BY w.next_refresh_at,w.household_id,w.actor_id,w.source LIMIT %s''',
          [*tables,at.replace(tzinfo=None),(at-timedelta(seconds=30)).replace(tzinfo=None),
           (at-timedelta(minutes=5)).replace(tzinfo=None),limit])
        tasks=cur.fetchall()
    deadline=time.monotonic()+budget_seconds
    for task in tasks:
        if time.monotonic()>=deadline-8:break
        actor=task['actor_id'];home=task['household_id'];kind=task['source']
        with conn.cursor() as cur:
            state=store.state_for(cur,home)
            if sources.membership(cur,actor)!=home:
                cur.execute('DELETE FROM household_brain_external_work WHERE household_id=%s AND actor_id=%s AND source=%s AND revision=%s',[home,actor,kind,task['revision']])
                continue
            cur.execute('SELECT * FROM family_brain_subjects WHERE actor_id=%s AND household_id=%s',[actor,home])
            subject=cur.fetchone() or {}
            cur.execute('SELECT partition_key FROM household_brain_external_partitions WHERE household_id=%s AND actor_id=%s AND updated_at>=%s',[home,actor,(at-timedelta(days=30)).replace(tzinfo=None)])
            partitions=[r['partition_key'] for r in cur.fetchall()]
        if not state:continue
        status='ready';body={}
        if subject.get('paused') or not subject.get('enabled',True):status='paused'
        else:
            try:
                since=max(stamp(state['learning_since']),stamp(subject.get('learning_since')) or stamp(state['learning_since']))
                body=external.collect(client,tables[kind],kind,actor,home,since,at,ZoneInfo(state['timezone']),deadline,partitions=partitions)
            except Exception as exc:
                status='unavailable';result['failed']+=1
                # Operational diagnostics must not log source text, identities,
                # query arguments, provider exception bodies or credentials.
                reason=str(exc) if isinstance(exc,ValueError) and str(exc) in {
                    'external_time_budget','external_source_limit','external_partition_limit'
                } else type(exc).__name__
                print(json.dumps({'event':'household_brain_source_unavailable','source':kind,'reason':reason}))
        try:
            external.publish(conn,state,actor,kind,body,status,at,work=task)
            result['checked']+=1
        except ValueError:
            result['changed_during_read']+=1
    return result


def finish(cur, task, body, status, now):
    if status=='unavailable':delay=timedelta(minutes=5)
    else:delay=timedelta(hours=20)
    cur.execute('''UPDATE household_brain_external_work SET dirty=0,next_refresh_at=%s
      WHERE household_id=%s AND actor_id=%s AND source=%s AND revision=%s''',
      [(now+delay).replace(tzinfo=None),task['household_id'],task['actor_id'],task['source'],task['revision']])
    if cur.rowcount!=1:raise ValueError('external_inputs_changed')
