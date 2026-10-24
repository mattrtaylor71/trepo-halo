"""Serialize household meal-plan admission using the existing shared cache rows."""
import re


class DispatchRejected(Exception):
    """The downstream service explicitly refused the request before execution."""


def claim_generation(conn, members, focus, debounce_seconds, dispatch=None):
    owners = sorted(set(members))
    if not owners or len(owners) > 100 or any(not isinstance(owner, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', owner) for owner in owners):
        raise ValueError('Invalid meal-plan household')
    conn.begin()
    try:
        with conn.cursor() as cur:
            # Create or lock every key in a stable order. A read followed by an
            # upsert lets simultaneous first requests each start generation.
            for owner in owners:
                cur.execute(
                    "INSERT INTO shared_meal_plan (owner_id, _id, _owner, status, _createdDate, _updatedDate) "
                    "VALUES (%s, 'current', %s, 'empty', NOW(), NOW()) "
                    "ON DUPLICATE KEY UPDATE owner_id=owner_id",
                    (owner, owner),
                )
            canonical_owners = set()
            active = False
            for owner in owners:
                cur.execute(
                    "SELECT owner_id, status, focus, TIMESTAMPDIFF(SECOND, _updatedDate, NOW()) AS age_seconds "
                    "FROM shared_meal_plan WHERE owner_id=%s FOR UPDATE", (owner,),
                )
                row = cur.fetchone()
                canonical_owners.add(row['owner_id'])
                if row['status'] == 'regenerating' and row['age_seconds'] is not None and row['age_seconds'] < debounce_seconds:
                    active = True
            if active:
                # Preserve a focus change; the running worker checks it before
                # publication and queues a fresh bounded attempt. No new dispatch.
                for owner in sorted(canonical_owners):
                    cur.execute('UPDATE shared_meal_plan SET focus=%s WHERE owner_id=%s', (focus,owner))
                conn.commit()
                return False
            for owner in sorted(canonical_owners):
                cur.execute(
                    "UPDATE shared_meal_plan SET status='regenerating', focus=%s, error_message=NULL, "
                    "_updatedDate=NOW() WHERE owner_id=%s", (focus, owner),
                )
                # Older generator packages still write personal rows then mirror
                # them. Keep existing copies aligned before dispatch, without DDL.
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', owner):
                    raise ValueError('Invalid stored meal-plan owner')
                table = owner + '_meal_plan'
                cur.execute(
                    "SELECT TABLE_NAME FROM information_schema.tables WHERE TABLE_SCHEMA=DATABASE() "
                    "AND BINARY TABLE_NAME=BINARY %s", (table,),
                )
                if cur.fetchone():
                    cur.execute(
                        "UPDATE `" + table + "` SET status='regenerating', focus=%s, error_message=NULL, "
                        "_updatedDate=NOW() WHERE _id='current'", (focus,),
                    )
        dispatch_error = None
        if dispatch is not None:
            try:
                dispatch()
            except DispatchRejected:
                # Restore every previous value, including an existing ready plan.
                raise
            except Exception as exc:
                # A lost acknowledgement can follow acceptance. Keep the claim
                # so an immediate legacy retry cannot start another generation.
                dispatch_error = exc
        conn.commit()
        if dispatch_error is not None:
            raise dispatch_error
        return True
    except Exception:
        conn.rollback()
        raise
