"""Transactional recipe dependencies, separate from the brain projection cache.

Activity evidence may age during one bounded generation. Membership, dietary
restrictions, stock and explicit memory controls may not change under a write.
The general brain fence remains strict for chat/history and other consumers.
"""
from contextlib import contextmanager
from decimal import Decimal
import json
import re
import time

from family_brain import store as brain_store
from household_brain.core import digest, obj, stamp


class GenerationInputsChanged(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def normalize(prefs):
    return {key: [s.strip() for s in prefs.get(key, []) if s.strip()]
            for key in ('allergies', 'diets', 'religious', 'health', 'custom')}


def _inventory(cur, table, members):
    if table is None:
        return None
    if not re.fullmatch(r'[A-Za-z0-9_-]+', table):
        raise ValueError('invalid_inventory_table')
    wanted = ['_id','owner_id','product_name','product_description','quantity','quantity_value',
              'quantity_unit','action','analysis_stage','analysis_status','_createdDate']
    cur.execute('SELECT COLUMN_NAME FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s', [table])
    present = {row['COLUMN_NAME'] for row in cur.fetchall()}
    if not {'_id','product_name','action'}.issubset(present):
        raise ValueError('inventory_schema_unavailable')
    columns = [name for name in wanted if name in present]
    where, args = ["action='IN'"], []
    if table == 'shared_kitchen':
        where.append('owner_id IN ('+','.join(['%s']*len(members))+')')
        args = members
    if 'analysis_stage' in present:
        where.append("(analysis_stage IN ('preliminary','fast','final') OR analysis_stage IS NULL)")
    if 'analysis_status' in present:
        where.append("(analysis_status='ready' OR analysis_status IS NULL)")
    cur.execute('SELECT '+','.join('`'+name+'`' for name in columns)+' FROM `'+table+'` WHERE '+
                ' AND '.join(where)+' ORDER BY _id'+(',owner_id' if 'owner_id' in present else '')+
                ' LIMIT 2001 LOCK IN SHARE MODE', args)
    rows = cur.fetchall()
    if len(rows)>2000:
        raise ValueError('inventory_limit_exceeded')
    # Some legacy writers do not advance owner_kitchen_state. Read the actual
    # inputs too, and hold their row/range locks until the recipe write commits.
    return digest([{key: str(value) if isinstance(value, Decimal) else value
                    for key, value in row.items() if key not in ('analysis_stage', 'analysis_status')} for row in rows])


def _dependencies(conn, owner, with_brain, inventory_table=None):
    # All reads are locking/current reads, including when the caller already
    # has a repeatable-read snapshot. Hold these locks through publication.
    with conn.cursor() as cur:
        household = brain_store.membership(cur, owner, current=True)
        if not household:
            raise GenerationInputsChanged('household_missing')
        cur.execute("SELECT user_id FROM new_users WHERE owner_id=%s OR "
                    "(user_id=%s AND (owner_id IS NULL OR owner_id='')) "
                    "ORDER BY user_id LOCK IN SHARE MODE", [household, household])
        members = [row['user_id'] for row in cur.fetchall()]
        prefs = normalize(brain_store.dietary_preferences(conn, owner, current=True))
        inventory = _inventory(cur, inventory_table, members)
        controls = None
        if with_brain:
            cur.execute('SELECT capture_enabled FROM household_brain_config WHERE singleton=1 LOCK IN SHARE MODE')
            if not (cur.fetchone() or {}).get('capture_enabled'):
                raise GenerationInputsChanged('brain_capture_disabled')
            cur.execute('SELECT enabled,paused,generation FROM household_brain_state WHERE household_id=%s FOR UPDATE', [household])
            state = cur.fetchone()
            if not state or not state['enabled']:
                raise GenerationInputsChanged('brain_disabled')
            cur.execute('SELECT actor_id,enabled,paused,generation,revision,facts FROM family_brain_subjects '
                        'WHERE household_id=%s ORDER BY actor_id LOCK IN SHARE MODE', [household])
            subjects = [row for row in cur.fetchall() if row['actor_id'] in members]
            # Detect an explicit preference expiring even without a SQL write.
            now = time.time()
            for subject in subjects:
                facts = obj(subject['facts'])
                subject['facts'] = facts
                subject['expired_facts'] = sorted(key for key, fact in facts.items()
                    if isinstance(fact, dict) and stamp(fact.get('expires_at'))
                    and stamp(fact['expires_at']).timestamp() <= now)
            cur.execute('SELECT audience,product_key FROM household_brain_suppressions WHERE household_id=%s '
                        'ORDER BY audience,product_key LOCK IN SHARE MODE', [household])
            suppressions = cur.fetchall()
            cur.execute('SELECT operation_id FROM household_brain_audit WHERE household_id=%s '
                        'ORDER BY operation_id LOCK IN SHARE MODE', [household])
            controls = digest([state, subjects, suppressions, cur.fetchall()])
        versions = {}
        for member in members:
            cur.execute('SELECT kitchen_version FROM owner_kitchen_state WHERE owner=%s FOR UPDATE', [member])
            versions[member] = int((cur.fetchone() or {}).get('kitchen_version') or 0)
    return dict(household=household, members=members, dietary=digest(prefs),
                versions=versions, inventory=inventory, controls=controls)


def capture(conn, *, owner, recipients, dietary, memory=None, inventory_table=None, max_seconds=480):
    conn.begin()
    try:
        # Validate freshness ONCE, before any expensive model work. A cached
        # context's five-minute read TTL is not a recipe-job expiration date.
        try:
            brain_store.require_publishable(conn, memory, owner, recipients, dietary)
        except ValueError as exc:
            if str(exc) in ('Food preferences changed during generation',
                            'Household changed during generation',
                            'Dietary preferences changed during generation'):
                raise GenerationInputsChanged('inputs_changed_before_generation') from exc
            raise
        if memory and memory['fence'].get('audience') is not None:
            raise GenerationInputsChanged('personal_context_in_shared_recipe')
        with_brain = bool(memory and memory['fence'].get('schema') == 'household-brain-v2')
        data = _dependencies(conn, owner, with_brain, inventory_table)
        if data['members'] != sorted(set(recipients)):
            raise GenerationInputsChanged('household_membership_changed')
        if data['dietary'] != digest(normalize(dietary)):
            raise GenerationInputsChanged('dietary_preferences_changed')
        conn.commit()
        return dict(data, owner=owner, with_brain=with_brain, inventory_table=inventory_table,
                    deadline=time.monotonic()+max(1, min(480, max_seconds)),
                    legacy_memory=memory if memory and not with_brain else None)
    except BaseException:
        conn.rollback()
        raise


def require_current(conn, snapshot):
    if time.monotonic() >= snapshot['deadline']:
        raise GenerationInputsChanged('generation_deadline_expired')
    legacy = snapshot.get('legacy_memory')
    if legacy and not brain_store.is_current(conn, legacy['fence']):
        raise GenerationInputsChanged('legacy_memory_changed')
    current = _dependencies(conn, snapshot['owner'], snapshot['with_brain'], snapshot['inventory_table'])
    for key, reason in [('household','household_membership_changed'), ('members','household_membership_changed'),
                        ('dietary','dietary_preferences_changed'), ('versions','kitchen_inventory_changed'),
                        ('inventory','kitchen_inventory_changed'),
                        ('controls','brain_controls_changed')]:
        if current[key] != snapshot[key]:
            raise GenerationInputsChanged(reason)


@contextmanager
def publication(conn, snapshot=None, **legacy):
    if snapshot is None:
        with brain_store.publication(conn, **legacy) as guarded:
            yield guarded
        return
    conn.begin()
    try:
        require_current(conn, snapshot)
        yield True
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
