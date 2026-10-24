"""Validate new publication provenance; retain legacy owner reads during migration.

Legacy rows have no provable historical household stamp. Do not manufacture one
or hide every existing plan during rollout. They retain their previous owner-only
read contract, with current dietary validation. New publications always require
their exact content, household and restrictions to match. Full legacy enforcement
can be enabled only after legacy reconciliation is complete.
"""
import json,os
from family_brain import store as brain_store
from family_brain.recipe_safety import _recipe_violates
from household_brain.core import digest


def visible_plan(conn, owner, plan):
    if isinstance(plan,str):
        plan=json.loads(plan)
    if not plan:
        return []
    conn.begin()
    try:
        with conn.cursor() as cur:
            household=brain_store.membership(cur,owner,current=True)
            cur.execute("SELECT user_id FROM new_users WHERE COALESCE(NULLIF(owner_id,''),user_id)=%s ORDER BY user_id LOCK IN SHARE MODE",[household])
            members=[r['user_id'] for r in cur.fetchall()]
            cur.execute('SELECT plan FROM shared_meal_plan WHERE owner_id=%s LOCK IN SHARE MODE',[owner])
            current=cur.fetchone()
            current_plan=current.get('plan') if current else None
            if isinstance(current_plan,str):
                current_plan=json.loads(current_plan)
            # The outer GET read must belong to the publication we validate.
            # A concurrent publication must not lend its stamp to an older payload.
            if current_plan != plan:
                conn.commit()
                return []
            cur.execute('SELECT * FROM meal_plan_publications WHERE owner_id=%s LOCK IN SHARE MODE',[owner])
            publication=cur.fetchone()
        prefs=brain_store.dietary_preferences(conn,owner,current=True)
        normalized={key:[s.strip() for s in prefs.get(key,[]) if s.strip()] for key in ('allergies','diets','religious','health','custom')}
        if publication:
            valid=bool(publication['household_id']==household and publication['members_hash']==digest(members)
                       and publication['dietary_hash']==digest(normalized) and publication.get('plan_hash')==digest(plan))
        else:
            valid=os.getenv('MEAL_PLAN_REQUIRE_LEGACY_PROVENANCE','false').lower()!='true'
        valid=valid and isinstance(plan,list) and all(isinstance(slot,dict) and isinstance(slot.get('recipe'),dict)
                   and not _recipe_violates(slot['recipe'],prefs) for slot in plan)
        conn.commit()
        return plan if valid else []
    except BaseException:
        conn.rollback()
        raise
