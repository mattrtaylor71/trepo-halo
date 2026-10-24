"""Atomic saved-recipe deletion. Callers prepare response data outside transactions."""
import hashlib
import json
import re
import random
import time

from pymysql.err import OperationalError

RECEIPTS = 'saved_recipe_delete_results_v1'
CACHE = 'owner_recipe_availability'
SHARED = 'shared_saved_recipes'
# Preserve the original response for short-term legacy retries, then retain only
# the completion identity so delayed requests cannot delete a recreated recipe.
REPLAY_DAYS = 7


class PreparationRequired(Exception):
    def __init__(self, row, fingerprint):
        super().__init__('Prepare the deletion response outside the transaction')
        self.row = dict(row)
        self.fingerprint = fingerprint


class RecipeIdentityReused(ValueError):
    pass


def _fingerprint(row, serialize):
    raw = json.dumps(row, sort_keys=True, separators=(',', ':'), default=serialize, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _table(cur, name):
    cur.execute('SELECT TABLE_NAME, ENGINE FROM information_schema.tables '
                'WHERE TABLE_SCHEMA=DATABASE() AND BINARY TABLE_NAME=BINARY %s', [name])
    row = cur.fetchone()
    if row and row['ENGINE'] != 'InnoDB':
        raise ValueError('Saved-recipe deletion requires transactional tables')
    return row is not None


def expire_response_batch(conn, *, limit=100, owner=None):
    """Clear a bounded batch of expired response payloads, retaining tombstones.

    The optional owner filter is for scoped qualification. This never deletes
    recipes, completion identities, or pending/recent operation results.
    """
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('Invalid retention batch limit')
    if owner is not None and (not isinstance(owner, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', owner)):
        raise ValueError('Invalid retention owner')
    if conn.server_status & 1:
        raise ValueError('Retention must not inherit an active transaction')
    with conn.cursor() as cur:
        if not _table(cur, RECEIPTS):
            raise ValueError('Saved-recipe deletion schema is not ready')
    conn.begin()
    try:
        with conn.cursor() as cur:
            sql = ('UPDATE saved_recipe_delete_results_v1 SET result=NULL '
                   'WHERE has_response=1 AND completed_at <= UTC_TIMESTAMP(6) - INTERVAL %s DAY')
            args = [REPLAY_DAYS]
            if owner is not None:
                sql += ' AND actor=%s'
                args.append(owner)
            cur.execute(sql + ' ORDER BY completed_at,actor,item_id LIMIT %s', args + [limit])
            changed = cur.rowcount
        conn.commit()
        return changed
    except Exception:
        conn.rollback()
        raise


def delete_with_receipt(conn, owner, item_id, member_ids, **kwargs):
    """Retry only confirmed deadlock victims, never ambiguous connection errors."""
    for attempt in range(8):
        try:
            return _delete_once(conn, owner, item_id, member_ids, **kwargs)
        except OperationalError as error:
            if not error.args or error.args[0] != 1213 or attempt == 7:
                raise
            time.sleep(random.uniform(0.005, min(0.16, 0.01 * 2 ** attempt)))


def _delete_once(conn, owner, item_id, member_ids, *, serialize,
                        prepared=None, fingerprint=None, read_shared=False,
                        delete_shared=True, cache_sources=('saved',)):
    """Return the stored recipe response, None for missing, or request preparation.

    The same exact legacy actor/item request has one durable completion. Recreated
    identities are rejected instead of allowing a delayed retry to delete new data.
    No schema creation, external calls or response enrichment happens here.
    """
    owners = set([owner] + list(member_ids))
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value) for value in owners):
        raise ValueError('Invalid owner')
    if not isinstance(item_id, str) or not 1 <= len(item_id) <= 128:
        raise ValueError('Invalid recipe identifier')
    if not callable(serialize):
        raise ValueError('A response serializer is required')
    sources = sorted(set(cache_sources))
    if not sources or any(not isinstance(value, str) or not value or len(value) > 64 for value in sources):
        raise ValueError('Invalid availability cache sources')
    if conn.server_status & 1:
        raise ValueError('Deletion must not inherit an active transaction')
    use_shared = read_shared or delete_shared
    with conn.cursor() as cur:
        for required in [RECEIPTS, CACHE] + ([SHARED] if use_shared else []):
            if not _table(cur, required):
                raise ValueError('Saved-recipe deletion schema is not ready')
        if use_shared:
            # Match the database's owner comparison, while preserving exact physical
            # personal-table names for the stored spelling of that same owner.
            cur.execute('SELECT owner_id FROM shared_saved_recipes WHERE owner_id=%s AND _id=%s LIMIT 1', [owner, item_id])
            canonical = cur.fetchone()
            if canonical:
                value = canonical['owner_id']
                if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
                    raise ValueError('Invalid stored owner')
                owners.add(value)
        owners = sorted(owners)
        tables = [(actor, actor + '_saved_recipes') for actor in owners if _table(cur, actor + '_saved_recipes')]
    placeholders = ','.join(['%s'] * len(owners))
    conn.begin()
    try:
        with conn.cursor() as cur:
            cur.execute('INSERT INTO saved_recipe_delete_results_v1(actor,item_id) VALUES(%s,%s) '
                        'ON DUPLICATE KEY UPDATE item_id=item_id', [owner, item_id])
            cur.execute('SELECT result,completed_at, '
                        '(completed_at <= UTC_TIMESTAMP(6) - INTERVAL %s DAY) AS expired '
                        'FROM saved_recipe_delete_results_v1 WHERE actor=%s AND item_id=%s FOR UPDATE',
                        [REPLAY_DAYS, owner, item_id])
            receipt = cur.fetchone()
            # Same order as edits: personal rows first, then shared projections.
            personal = {}
            for actor, table in tables:
                cur.execute('SELECT * FROM `' + table + '` WHERE _id=%s FOR UPDATE', [item_id])
                personal[actor] = cur.fetchone()
            shared = []
            if use_shared:
                cur.execute('SELECT *, (owner_id=%s) AS __acting FROM shared_saved_recipes '
                            'WHERE _id=%s AND owner_id IN (' + placeholders + ') ORDER BY owner_id FOR UPDATE',
                            [owner, item_id] + owners)
                shared = cur.fetchall()
            acting_shared = None
            for row in shared:
                if row.pop('__acting'):
                    acting_shared = row
            primary = acting_shared if read_shared else personal.get(owner)
            if receipt['result'] is not None or receipt['completed_at'] is not None:
                # Required acting-owner copies must still be absent on a replay.
                if primary is not None or acting_shared is not None or personal.get(owner) is not None:
                    raise RecipeIdentityReused('Saved recipe identifier was reused after deletion')
                if receipt['expired'] or receipt['result'] is None:
                    if receipt['result'] is not None:
                        cur.execute('UPDATE saved_recipe_delete_results_v1 SET result=NULL '
                                    'WHERE actor=%s AND item_id=%s', [owner, item_id])
                    conn.commit()
                    return None
                stored = json.loads(receipt['result']) if isinstance(receipt['result'], str) else receipt['result']
                if not isinstance(stored, dict) or not isinstance(stored.get('id'), str):
                    raise ValueError('Invalid saved-recipe deletion receipt')
                conn.commit()
                return stored
            if primary is None:
                conn.rollback()
                return None
            current_fingerprint = _fingerprint(primary, serialize)
            if prepared is None or fingerprint != current_fingerprint:
                raise PreparationRequired(primary, current_fingerprint)
            if not isinstance(prepared, dict) or prepared.get('id') != primary['_id']:
                raise ValueError('Prepared response does not identify the deleted recipe')
            encoded = json.dumps(prepared, default=serialize, allow_nan=False)
            for actor, table in tables:
                cur.execute('DELETE FROM `' + table + '` WHERE _id=%s', [item_id])
            if use_shared:
                cur.execute('DELETE FROM shared_saved_recipes WHERE _id=%s AND owner_id IN (' + placeholders + ')', [item_id] + owners)
            cur.execute('DELETE FROM owner_recipe_availability WHERE owner IN (' + placeholders + ') '
                        'AND recipe_source IN (' + ','.join(['%s'] * len(sources)) + ') AND recipe_id=%s',
                        owners + sources + [item_id])
            cur.execute('UPDATE saved_recipe_delete_results_v1 SET result=%s,completed_at=UTC_TIMESTAMP(6) '
                        'WHERE actor=%s AND item_id=%s', [encoded, owner, item_id])
        conn.commit()
        return json.loads(encoded)
    except Exception:
        conn.rollback()
        raise
