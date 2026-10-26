"""Durable shelf state and change journal for every Kitchen writer.

Install schema/triggers with the explicit migration, never in a request handler.
Feature flags permit shadow qualification without changing legacy responses.
Readers lock the same item before processing its event journal. Caller owns the
transaction; no hidden commit can break a quantity operation's atomicity.
"""
import json
import os
import pymysql
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo
from shelf_lifecycle import advance, object_value, public_snapshot, fingerprint, review_projection

INPUT_FIELDS = ('_id','owner_id','action','product_name','brand','variant','category',
                'product_expiration','storage_guidance','storage_location','is_opened',
                '_createdDate','analysis_stage','analysis_status')


def enabled():
    return os.getenv('SHELF_LIFECYCLE_V1', '').lower() == 'true'


def enabled_for(conn, owner):
    """Fail closed unless the current household is in the configured cohort.

    Membership is read on every boundary; joining/leaving cannot retain access
    through a process-global cache. '*' is the explicit full-rollout setting.
    """
    if not enabled() or not owner:
        return False
    try:
        cohort = json.loads(os.getenv('SHELF_LIFECYCLE_HOUSEHOLDS', '[]'))
    except (ValueError, TypeError):
        return False
    if not isinstance(cohort, list) or not all(isinstance(v, str) for v in cohort):
        return False
    if '*' in cohort:
        return True
    if not cohort:
        return False
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute('SELECT owner_id FROM new_users WHERE user_id=%s', (owner,))
        account = cur.fetchone()
    return bool(account and (account.get('owner_id') or owner) in cohort)


def schema_sql():
    yield '''CREATE TABLE IF NOT EXISTS kitchen_shelf_calendars (
      household_id VARCHAR(64) PRIMARY KEY, time_zone VARCHAR(100) NOT NULL,
      created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
    yield '''CREATE TABLE IF NOT EXISTS kitchen_shelf_events (
      event_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
      item_id VARCHAR(64) NOT NULL, owner_id VARCHAR(64) NOT NULL,
      observed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
      evidence JSON NOT NULL, INDEX item_events(item_id,event_id), INDEX event_age(observed_at,event_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
    yield '''CREATE TABLE IF NOT EXISTS kitchen_shelf_state (
      item_id VARCHAR(64) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL,
      event_id BIGINT NOT NULL DEFAULT 0, input_hash CHAR(64) NOT NULL,
      state JSON NOT NULL, deadline DATE NULL, generation BIGINT NOT NULL,
      needs_work BOOLEAN NOT NULL DEFAULT TRUE,
      updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
      INDEX due_items(deadline,owner_id), INDEX work_queue(needs_work,item_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
    yield '''CREATE TABLE IF NOT EXISTS kitchen_shelf_reminders (
      recipient VARCHAR(64) NOT NULL, item_id VARCHAR(64) NOT NULL,
      generation BIGINT NOT NULL, milestone VARCHAR(16) NOT NULL,
      local_day DATE NOT NULL, status VARCHAR(24) NOT NULL,
      claimed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
      PRIMARY KEY(recipient,item_id,generation,milestone), INDEX recipient_day(recipient,local_day)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
    for event, ref in [('INSERT','NEW'),('UPDATE','NEW'),('DELETE','OLD')]:
        fields = []
        for key in INPUT_FIELDS:
            value = "'OUT'" if event == 'DELETE' and key == 'action' else ref+'.`'+key+'`'
            fields.extend(["'"+key+"'", value])
        fields.extend(["'_shelf_edits'", '(SELECT overrides FROM kitchen_item_edits WHERE item_id=BINARY '+ref+'._id)'])
        fields.extend(["'_shelf_event'", "'"+event+"'"])
        payload = 'JSON_OBJECT('+','.join(fields)+')'
        condition = 'TRUE'
        if event == 'UPDATE':
            condition = 'NOT ('+' AND '.join('OLD.`'+key+'` <=> NEW.`'+key+'`' for key in INPUT_FIELDS if key not in ('_id',)) + ')'
        yield f'''CREATE TRIGGER trepo_shelf_{event.lower()} AFTER {event} ON shared_kitchen FOR EACH ROW
        BEGIN
          IF {condition} THEN
            INSERT INTO kitchen_shelf_events(item_id,owner_id,evidence) VALUES({ref}._id,{ref}.owner_id,{payload});
            INSERT INTO kitchen_shelf_state(item_id,owner_id,input_hash,state,generation,needs_work)
              VALUES({ref}._id,{ref}.owner_id,'',JSON_OBJECT(),0,TRUE)
              ON DUPLICATE KEY UPDATE needs_work=TRUE;
          END IF;
        END'''


def _clock(raw):
    if isinstance(raw,(int,float,Decimal)): return datetime.fromtimestamp(float(raw),timezone.utc)
    if isinstance(raw, datetime): value = raw
    else: value = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def materialize(conn, rows, *, time_zone='America/Los_Angeles', today=None, now=None, max_events=2000):
    """Project current evidence under an existing transaction, no network calls.

    Always re-read the item after locking, rather than accepting a caller's stale
    serialized row as evidence. Replays are chronologically ordered and bounded.
    """
    result = {}
    if not rows: return result
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        # Two indexed batch reads avoid a cross-collation binary join. That
        # join can scan the entire kitchen table once for each requested item.
        marks = ','.join(['%s'] * len(rows))
        ids = [r['_id'] for r in rows]
        cur.execute('SELECT * FROM kitchen_shelf_state WHERE item_id IN ('+marks+')', ids)
        cached = {r['item_id']:r for r in cur.fetchall()}
        cur.execute('SELECT _id,owner_id,action FROM shared_kitchen WHERE _id IN ('+marks+')',ids)
        current = {r['_id']:r for r in cur.fetchall()}
        for ident,known in cached.items():
            known['current_owner'] = current.get(ident,{}).get('owner_id')
            known['current_action'] = current.get(ident,{}).get('action')
        for supplied in sorted(rows, key=lambda r:r['_id']):
            ident = supplied['_id']
            known = cached.get(ident)
            if known and not known['needs_work'] and object_value(known['state']).get('version'):
                if supplied.get('owner_id') and known['current_owner'] and known['current_owner'] != supplied['owner_id']:
                    raise RuntimeError('shelf_owner_changed')
                result[ident] = public_snapshot(object_value(known['state']),today=today,now=now)
                continue
            cur.execute('SELECT * FROM shared_kitchen WHERE _id=%s FOR UPDATE', (ident,))
            row = cur.fetchone()
            if row and supplied.get('owner_id') and row['owner_id'] != supplied['owner_id']:
                raise RuntimeError('shelf_owner_changed')
            cur.execute('SELECT * FROM kitchen_shelf_state WHERE item_id=%s FOR UPDATE', (ident,))
            saved = cur.fetchone()
            previous = object_value(saved['state']) if saved else {}
            calendar_owner = (row or supplied).get('owner_id')
            cur.execute('SELECT owner_id FROM new_users WHERE user_id=%s FOR SHARE', (calendar_owner,))
            account = cur.fetchone()
            if account is None: raise RuntimeError('shelf_account_missing')
            household = (account or {}).get('owner_id') or calendar_owner
            # One calendar per household; timezone changes/travel cannot move dates.
            ZoneInfo(time_zone)
            cur.execute('INSERT IGNORE INTO kitchen_shelf_calendars(household_id,time_zone) VALUES(%s,%s)', (household,time_zone))
            cur.execute('SELECT time_zone FROM kitchen_shelf_calendars WHERE household_id=%s', (household,))
            calendar = cur.fetchone()['time_zone']
            last = int(saved['event_id']) if saved else 0
            cur.execute('SELECT event_id,UNIX_TIMESTAMP(observed_at) AS observed_epoch,evidence FROM kitchen_shelf_events WHERE item_id=%s AND event_id>%s ORDER BY event_id LIMIT %s', (ident,last,max_events+1))
            events = cur.fetchall()
            if len(events) > max_events: raise RuntimeError('shelf_event_replay_budget')
            for event in events:
                previous = advance(object_value(event['evidence']),previous,now=_clock(event['observed_epoch']),time_zone=calendar)
                last = int(event['event_id'])
            if row:
                cur.execute('SELECT overrides FROM kitchen_item_edits WHERE item_id=%s', (ident,))
                edits = cur.fetchone()
                row['_shelf_edits'] = object_value(edits['overrides']) if edits else {}
            else:
                # The deletion journal retains the same product identity and
                # accepted guidance so Undo cannot pick up a later model estimate.
                row = dict(object_value(events[-1]['evidence']) if events else supplied,action='OUT')
            inputs = {key:row.get(key) for key in INPUT_FIELDS}
            inputs['_shelf_edits'] = row.get('_shelf_edits',{})
            signature = fingerprint(inputs)
            # Empty/new queue records are initialized, including old rows without
            # an event journal. Never backdate events or fabricate opening dates.
            state = advance(row,previous,time_zone=calendar,now=now)
            generation = state['generation']
            if not saved or saved['input_hash'] != signature or saved['needs_work'] or last != saved['event_id']:
                cur.execute('''INSERT INTO kitchen_shelf_state(item_id,owner_id,event_id,input_hash,state,deadline,generation,needs_work)
                  VALUES(%s,%s,%s,%s,%s,%s,%s,FALSE) ON DUPLICATE KEY UPDATE
                  owner_id=VALUES(owner_id),event_id=VALUES(event_id),input_hash=VALUES(input_hash),state=VALUES(state),
                  deadline=VALUES(deadline),generation=VALUES(generation),needs_work=FALSE''',
                  (ident,row.get('owner_id') or (saved or {}).get('owner_id') or '',last,signature,json.dumps(state),state['deadline'],generation))
            result[ident] = public_snapshot(state,today=today,now=now)
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute('SELECT item_id,overrides FROM kitchen_item_edits WHERE item_id IN (' + ','.join(['%s']*len(result)) + ')', list(result))
        edits = {row['item_id']:row['overrides'] for row in cur.fetchall()}
    reminders_enabled = os.getenv('SHELF_LIFECYCLE_NOTIFICATIONS','').lower() == 'true'
    return {key:dict(review_projection(value, edits.get(key), now=now), reminders_enabled=reminders_enabled) for key,value in result.items()}


def annotate(conn, rows, **kwargs):
    if not enabled() or not rows: return rows
    owners = {r.get('owner_id') for r in rows}
    allowed = {owner for owner in owners if enabled_for(conn, owner)}
    selected = [r for r in rows if r.get('owner_id') in allowed]
    projections = materialize(conn,selected,**kwargs)
    for row in selected: row['shelf_life'] = projections[row['_id']]
    return rows


def reconcile(conn, *, limit=250, time_zone='America/Los_Angeles'):
    """Bounded durable worker; interrupted work is retried without duplicate state."""
    try:
        cohort=json.loads(os.getenv('SHELF_LIFECYCLE_HOUSEHOLDS','[]'))
    except (ValueError,TypeError):return 0
    if not enabled() or not isinstance(cohort,list) or not cohort or not all(isinstance(v,str) for v in cohort):return 0
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        args=[];clause=''
        if '*' not in cohort:
            marks=','.join(['%s']*len(cohort))
            cur.execute("SELECT user_id FROM new_users WHERE owner_id IN ("+marks+") OR ((owner_id IS NULL OR owner_id='') AND user_id IN ("+marks+'))',cohort+cohort)
            args=[r['user_id'] for r in cur.fetchall()]
            if not args:return 0
            clause=' AND owner_id IN ('+','.join(['%s']*len(args))+')'
        cur.execute('SELECT item_id AS _id,owner_id FROM kitchen_shelf_state WHERE needs_work=TRUE'+clause+' ORDER BY updated_at,item_id LIMIT %s',args+[limit])
        rows=cur.fetchall()
    for row in rows:
        try:
            conn.begin();materialize(conn,[row],time_zone=time_zone);conn.commit()
        except Exception as error:
            conn.rollback()
            # Keep failed work pending, but move it behind other dirty items.
            # One orphan or malformed row must not block every household's digest.
            conn.begin()
            with conn.cursor() as cur:
                cur.execute('UPDATE kitchen_shelf_state SET updated_at=CURRENT_TIMESTAMP(6) WHERE item_id=%s AND needs_work=TRUE',(row['_id'],))
            conn.commit()
            print(json.dumps({'evt':'shelf_projection_unavailable','error_type':type(error).__name__}))
    return len(rows)


def seed_calendars(conn, profiles):
    """Seed known household calendars before legacy backfill; never overwrite one."""
    valid = {}
    for owner, profile in profiles.items():
        if not enabled_for(conn,owner):continue
        try:
            zone = profile.get('time_zone')
            ZoneInfo(zone)
            valid[owner] = zone
        except (ValueError, KeyError, TypeError): continue
    owners = sorted(valid)
    for start in range(0,len(owners),500):
        batch = owners[start:start+500]
        conn.begin()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute('SELECT user_id,owner_id FROM new_users WHERE user_id IN ('+','.join(['%s']*len(batch))+') ORDER BY user_id FOR SHARE',batch)
                values = [(r['owner_id'] or r['user_id'],valid[r['user_id']]) for r in cur.fetchall()]
                if values:
                    cur.executemany('INSERT IGNORE INTO kitchen_shelf_calendars(household_id,time_zone) VALUES(%s,%s)',values)
            conn.commit()
        except Exception:
            conn.rollback();raise


def compact_journal(conn,limit=1000):
    """Keep 30 days for diagnosis; never delete unapplied evidence or the state."""
    conn.begin()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute('''SELECT e.event_id FROM kitchen_shelf_events e
              JOIN kitchen_shelf_state s ON s.item_id=e.item_id
              WHERE e.event_id<=s.event_id AND e.observed_at<CURRENT_TIMESTAMP - INTERVAL 30 DAY
              ORDER BY e.observed_at,e.event_id LIMIT %s''',(limit,))
            ids=[r['event_id'] for r in cur.fetchall()]
            if ids:cur.execute('DELETE FROM kitchen_shelf_events WHERE event_id IN ('+','.join(['%s']*len(ids))+')',ids)
        conn.commit();return len(ids)
    except Exception:conn.rollback();raise
