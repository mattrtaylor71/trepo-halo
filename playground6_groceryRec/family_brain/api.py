"""Private founder control/context API, mounted in Home; signed identity is mandatory."""
import base64
import hashlib
import hmac
import json
import os
import time
from . import store
from .core import encode, json_value


def _sign(payload):
    body = base64.urlsafe_b64encode(encode(payload).encode()).decode().rstrip("=")
    sig = hmac.new(os.environ["TOKEN_SIGNING_SECRET"].encode(), body.encode(), hashlib.sha256).hexdigest()
    return body + "." + sig


def ticket_actor(ticket):
    try:
        body, sig = ticket.split(".")
        expected = hmac.new(os.environ["TOKEN_SIGNING_SECRET"].encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        p = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if p.get("aud") != "trepo-family-brain" or not time.time() < p["exp"] <= time.time() + 601:
            return None
        return p
    except Exception:
        return None


def handle(conn, request, verify_jwt):
    """Never trust actor/household from a body, decoded token, or a legacy owner header."""
    owner = str(request.get("owner", ""))
    authorization = str(request.get("authorization", ""))
    ticket = ticket_actor(request.get("ticket", ""))
    if ticket:
        actor = ticket["actor"]
    else:
        token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        claims = verify_jwt(token) if token else None
        actor = claims.get("user_id") if claims else None
    if not actor or actor != owner or not store.enabled_owner(actor):
        return {"statusCode": 403, "error": "not_enrolled_or_authenticated"}
    purpose = request.get("purpose", "thyme")
    operation = request.get("operation", "context")
    if os.getenv('HOUSEHOLD_BRAIN_V2') == '1' and operation not in {'context','validate','authorize','history','refresh_turn','validate_receipt'}:
        return {'statusCode':403,'error':'internally_managed'}
    # Async jobs may read and validate only. A ticket must not mint its own
    # replacement and turn a ten-minute delegation into indefinite access.
    if ticket and operation not in {"context", "validate", "history", "refresh_turn", "validate_receipt"}:
        return {"statusCode": 403, "error": "read_only_ticket"}
    # Receipt checks do not rebuild optional context (or publish its old text).
    # Refresh is a new read, bound to the original actor/household/control scope.
    from household_brain.store import is_action_scope_current
    if operation == 'validate' and os.getenv('HOUSEHOLD_BRAIN_V2') == '1':
        fence = request.get('fence')
        current = (isinstance(fence, dict) and fence.get('actor') == actor
                   and (not ticket or is_action_scope_current(conn, ticket.get('fence')))
                   and store.is_current(conn, fence))
        return {'statusCode': 200, 'current': bool(current)}
    if operation in {'refresh_turn', 'validate_receipt'}:
        fence = request.get('fence')
        bound = isinstance(fence, dict) and fence.get('actor') == actor
        current = (bound and is_action_scope_current(conn, fence)
                   and (not ticket or is_action_scope_current(conn, ticket.get('fence'))))
        if operation == 'validate_receipt':
            return {'statusCode': 200, 'current': bool(current)}
        if not current:
            return {'statusCode': 200, 'current': False}
        fresh = store.read_context(conn, owner, purpose, actor=actor)
        if not fresh or not is_action_scope_current(conn, fence) or not store.is_current(conn, fresh['fence']):
            return {'statusCode': 200, 'current': False}
        fresh['fence']['turn_expires'] = fence.get('turn_expires', fence.get('expires', 0))
        return {'statusCode': 200, 'current': True, **fresh}
    data = store.read_context(conn, owner, purpose, actor=actor if operation in {"context", "inspect"} else None)
    if not data or (ticket and not store.is_current(conn, ticket.get("fence"))):
        return {"statusCode": 403, "error": "membership_or_controls_changed"}
    fence = request.get("fence")
    fence_bound = (isinstance(fence, dict) and fence.get("actor") == actor
                   and fence.get("household") == data["fence"]["household"])
    if operation == "history":
        if not fence_bound or not store.is_current(conn,fence):
            return {"statusCode":409,"error":"history_context_changed"}
        from household_brain.history import lookup
        from household_brain.receipt_lookup import configured_resolver
        history=lookup(conn,actor,data["fence"]["household"],request.get("query"),receipt_resolver=configured_resolver())
        if not store.is_current(conn,fence):
            return {"statusCode":409,"error":"history_context_changed"}
        return {"statusCode":200,"history":history}
    if operation == "context":
        return {"statusCode": 200, **data}
    if operation == "validate":
        return {"statusCode": 200, "current": fence_bound and store.is_current(conn, fence)}
    if operation == "authorize":
        return {"statusCode": 200, "ticket": _sign({"aud": "trepo-family-brain", "actor": actor,
            "fence": data["fence"], "exp": int(time.time()) + 600})}
    if operation == "inspect":
        with conn.cursor() as cur:
            cur.execute("SELECT revision,generation,paused,facts FROM family_brain_subjects WHERE actor_id=%s", [actor])
            state = cur.fetchone()
        state["facts"] = json_value(state["facts"], {})
        return {"statusCode": 200, "state": state, **data}
    if operation == "control":
        return {"statusCode": 200, **store.control(conn, actor, request.get("change", {}))}
    return {"statusCode": 400, "error": "invalid_operation"}
