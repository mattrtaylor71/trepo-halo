"""MySQL adapter: fresh membership, current rows, bounded reads, actor-only controls."""
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from .core import corrected_state, encode, json_value, project, prompt, timestamp


def allowlist():
    try:
        value = json.loads(os.getenv("FAMILY_BRAIN_ACTORS", "[]"))
        return set(value) if isinstance(value, list) and len(value) <= 2 else set()
    except (TypeError, ValueError):
        return set()


def enabled_owner(owner):
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        return isinstance(owner, str) and bool(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', owner))
    return str(owner) in allowlist()


def generation_subject(current_member_ids):
    """Use existing authoritative fan-out membership; no extra reads for other families.

    A non-pilot member's kitchen edit can regenerate the founders' shared feed. Such
    work may use deliberately shared preferences, never that founder's personal history.
    read_context still rechecks the enrollment's current household before returning data.
    """
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        return next((str(actor) for actor in sorted(current_member_ids) if enabled_owner(str(actor))), None)
    allowed = allowlist()
    return next((str(actor) for actor in sorted(current_member_ids) if str(actor) in allowed), None)


def generation_context(conn, owner, current_member_ids, purpose):
    subject = generation_subject(current_member_ids)
    if not subject:
        return None
    result = optional_context(conn, subject, purpose)
    if result:
        try:
            with conn.cursor() as cur:
                if membership(cur, owner, current=True) != result['fence']['household']:
                    return None
        except Exception:
            return None
    return result


def membership(cur, actor, current=False):
    cur.execute("SELECT user_id, owner_id FROM new_users WHERE user_id=%s LIMIT 1"
                + (" LOCK IN SHARE MODE" if current else ""), [actor])
    row = cur.fetchone()
    return str(row.get("owner_id") or row["user_id"]) if row else None


def states_for(cur, household, current=False):
    cur.execute("""SELECT b.* FROM family_brain_subjects b JOIN new_users u
        ON u.user_id=b.actor_id AND COALESCE(NULLIF(u.owner_id,''),u.user_id)=b.household_id
        WHERE b.household_id=%s AND b.enabled=1 ORDER BY b.actor_id LIMIT 2"""
        + (" LOCK IN SHARE MODE" if current else ""), [household])
    return [r for r in cur.fetchall() if r["actor_id"] in allowlist()]


def evidence_for(cur, actor, household, since):
    rows = []
    # Canonical current rows, so deletion/unsave/duplicate delivery cannot inflate learning.
    # Never read another member's saved library or a household-wide imported profile.
    cur.execute("""SELECT _id AS entity_id, LEFT(title,100) AS name, _createdDate AS at
        FROM shared_saved_recipes WHERE owner_id=%s AND status='ready'
        AND _createdDate >= %s ORDER BY _createdDate DESC, _id LIMIT 80""", [actor, since])
    for r in cur.fetchall():
        rows.append(dict(r, kind="recipe_saved", actor_id=actor, household_id=household, committed=True))
    # A logged dish is observational. Only explicitly attributed actor records qualify.
    cur.execute("""SELECT _id AS entity_id, LEFT(dish_name,100) AS name, _createdDate AS at
        FROM shared_dishes WHERE user_id=%s AND owner_id IN (%s,%s)
        AND _createdDate >= %s AND action='IN'
        ORDER BY _createdDate DESC, _id LIMIT 40""", [actor, actor, household, since])
    for r in cur.fetchall():
        rows.append(dict(r, kind="dish_logged", actor_id=actor, household_id=household, committed=True))
    # The amount-operation journal commits with inventory and records Undo. Manual
    # reductions, discards, failed writes and reversed operations are never cooking evidence.
    cur.execute("""SELECT operation_id AS entity_id, created_at AS at,
        LEFT(JSON_UNQUOTE(JSON_EXTRACT(changes,'$[0].recipe_title')),100) AS name
        FROM kitchen_amount_operations WHERE actor=%s AND created_at>=%s AND undone_by IS NULL
        AND JSON_UNQUOTE(JSON_EXTRACT(changes,'$[0].mode'))='consume'
        AND JSON_UNQUOTE(JSON_EXTRACT(changes,'$[0].recipe_title')) NOT IN ('','null')
        ORDER BY created_at DESC, operation_id LIMIT 40""", [actor, since])
    seen_uses = set()
    for r in cur.fetchall():
        key = (str(r.get("name", "")).casefold(), str(r.get("at", ""))[:10])
        if key not in seen_uses:
            seen_uses.add(key)
            rows.append(dict(r, kind="recipe_used", actor_id=actor, household_id=household, committed=True))
    return rows


def read_context(conn, owner, purpose, *, actor=None, now=None):
    """actor=None is deliberately shared-only for household-visible generation caches."""
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        from household_brain import store as shared_brain
        return shared_brain.read_context(conn, owner, purpose, actor=actor, now=now)
    if not enabled_owner(owner):
        return None
    if actor is not None and (actor != owner or actor not in allowlist()):
        return None
    now = now or datetime.now(timezone.utc)
    with conn.cursor() as cur:
        household = membership(cur, owner)
        if not household:
            return None
        states = states_for(cur, household)
        own = next((s for s in states if s["actor_id"] == owner), None)
        if not own:
            return None
        hard_preferences = dietary_preferences(conn, owner)
        evidence = evidence_for(cur, actor, household,
            (now - timedelta(days=90)).replace(tzinfo=None)) if actor and not own["paused"] else []
        context = project(actor=actor, household=household, purpose=purpose,
                          states=states, evidence=evidence, now=now)
        # A caller must revalidate before publishing a long-running model response.
        fence = {"actor": owner, "household": household,
                 "dietary_hash": hashlib.sha256(encode(hard_preferences).encode()).hexdigest(),
                 "states": [[s["actor_id"], int(s["generation"]), int(s["revision"])] for s in states],
                 "expires": min([timestamp(f.get("expires_at")).timestamp()
                    for s in states for f in json_value(s.get("facts"), {}).values()
                    if isinstance(f, dict) and timestamp(f.get("expires_at")) and timestamp(f.get("expires_at")) > now]
                    + [(now + timedelta(minutes=10)).timestamp()])}
        return {"context": context, "prompt": prompt(context), "fence": fence,
                "dietary_preferences": hard_preferences}


def is_current(conn, fence):
    if fence and fence.get('schema') == 'household-brain-v2':
        from household_brain import store as shared_brain
        return shared_brain.is_current(conn, fence)
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        return False
    if not fence or not enabled_owner(fence.get("actor")):
        return False
    if datetime.now(timezone.utc).timestamp() >= fence.get("expires", 0):
        return False
    with conn.cursor() as cur:
        if membership(cur, fence["actor"], current=True) != fence["household"]:
            return False
        states = states_for(cur, fence["household"], current=True)
        actual = [[s["actor_id"], int(s["generation"]), int(s["revision"])] for s in states]
        return (actual == fence["states"] and fence.get("dietary_hash") ==
                hashlib.sha256(encode(dietary_preferences(conn, fence["actor"], current=True)).encode()).hexdigest())


def dietary_preferences(conn, owner, current=False):
    """Separate safety input, never bounded/truncated with optional soft memory."""
    with conn.cursor() as cur:
        household = membership(cur, owner, current=current)
        if not household:
            raise PermissionError("membership_unknown")
        cur.execute("SELECT allergies,diets,religious,health,custom FROM user_dietary_preferences WHERE owner_id=%s LIMIT 1"
                    + (" LOCK IN SHARE MODE" if current else ""), [household])
        row = cur.fetchone() or {}
    prefs = {}
    for key in ("allergies", "diets", "religious", "health", "custom"):
        raw = row.get(key)
        value = json.loads(raw) if isinstance(raw, str) else raw
        if value is not None and (not isinstance(value, list) or any(not isinstance(x, str) for x in value)):
            raise ValueError("dietary_preferences_unavailable")
        prefs[key] = value or []
    return prefs


def require_publishable(conn, memory=None, owner=None, recipients=(), dietary=None):
    """Call inside a transaction; hold membership/consent locks through payload writes."""
    if memory is None and dietary is None:
        return
    if memory and not is_current(conn, memory['fence']):
        raise ValueError('Food preferences changed during generation')
    with conn.cursor() as cur:
        household = memory['fence']['household'] if memory else membership(cur, owner, current=True)
        if not household:
            raise ValueError('Household changed during generation')
        for recipient in sorted(set(recipients) | {owner}):
            if membership(cur, recipient, current=True) != household:
                raise ValueError('Household changed during generation')
    if dietary is not None:
        # Match the recipe generator's existing whitespace normalization.
        normalize = lambda prefs: {k: [s.strip() for s in prefs.get(k, []) if s.strip()]
            for k in ('allergies', 'diets', 'religious', 'health', 'custom')}
        if normalize(dietary) != normalize(dietary_preferences(conn, owner, current=True)):
            raise ValueError('Dietary preferences changed during generation')


@contextmanager
def publication(conn, memory=None, owner=None, recipients=(), dietary=None):
    """Optional pilot transaction. Existing non-pilot writers keep their own commits."""
    if memory is None and dietary is None:
        yield False
        return
    conn.begin()
    try:
        require_publishable(conn, memory, owner, recipients, dietary)
        yield True
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def signed_actor(event, owner, verify_jwt):
    if not enabled_owner(owner):
        return None
    headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}
    auth = str(headers.get("authorization", ""))
    claims = verify_jwt(auth[7:]) if auth.lower().startswith("bearer ") else None
    return owner if claims and claims.get("user_id") == owner else None


def control(conn, actor, operation):
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        raise PermissionError('internally_managed')
    if actor not in allowlist():
        raise PermissionError("not_enrolled")
    op_id = str(operation.get("operation_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", op_id):
        raise ValueError("invalid_operation_id")
    # A lost response followed by a fresh read can advance expected_revision. The same
    # operation ID + intent still returns its original outcome; different intent conflicts.
    request_hash = hashlib.sha256(encode({k: v for k, v in operation.items() if k != "expected_revision"}).encode()).hexdigest()
    conn.begin()
    try:
        with conn.cursor() as cur:
            # Lock only this actor. A second member can never reset their private history.
            cur.execute("SELECT * FROM family_brain_subjects WHERE actor_id=%s FOR UPDATE", [actor])
            state = cur.fetchone()
            household = membership(cur, actor, current=True)
            if not state or not state["enabled"] or state["household_id"] != household:
                raise PermissionError("membership_changed")
            cur.execute("SELECT request_hash,result_json FROM family_brain_controls WHERE actor_id=%s AND operation_id=%s", [actor, op_id])
            previous = cur.fetchone()
            if previous:
                if previous["request_hash"] != request_hash:
                    raise ValueError("operation_conflict")
                conn.commit()
                return json_value(previous["result_json"])
            if type(operation.get("expected_revision")) is not int or operation["expected_revision"] != state["revision"]:
                raise ValueError("revision_conflict")
            updated = corrected_state(state, operation)
            since = updated["learning_since"]
            if isinstance(since, str):
                since = datetime.fromisoformat(since).replace(tzinfo=None)
            cur.execute("""UPDATE family_brain_subjects SET facts=%s,paused=%s,generation=%s,
                revision=%s,learning_since=%s,updated_at=UTC_TIMESTAMP(6) WHERE actor_id=%s""",
                [encode(updated["facts"]), updated["paused"], updated["generation"],
                 updated["revision"], since, actor])
            result = {"ok": True, "revision": updated["revision"], "generation": updated["generation"]}
            cur.execute("INSERT INTO family_brain_controls(actor_id,operation_id,request_hash,result_json) VALUES (%s,%s,%s,%s)",
                        [actor, op_id, request_hash, encode(result)])
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def optional_context(conn, owner, purpose, *, actor=None):
    if not enabled_owner(owner):
        return None
    try:
        result = read_context(conn, owner, purpose, actor=actor)
        if result:
            print(json.dumps({"evt": "family_brain", "purpose": purpose, "outcome": "used",
                              "context_bytes": len(encode(result["context"]).encode())}))
        return result
    except Exception:
        # No content, account IDs, SQL or credentials in failure telemetry.
        print(json.dumps({"evt": "family_brain", "purpose": purpose, "outcome": "fallback"}))
        return None
