"""Server gate for new AI work. Absence is legacy compatibility, never consent.

The new app atomically activates a pending account record before showing its
disclosure. Once activated, every worker requires this exact policy's explicit
allow. No positive cache: withdrawal applies to the next admitted request.
"""
import json
import base64
import os
import re
import boto3
import urllib.request
import urllib.error
from contextvars import ContextVar
actor_context = ContextVar("trepo_ai_actor", default=None)
UNVERIFIED_ACTOR = "unverified-ai-actor"

VERSION = "2026-10-22.v1"
TABLE = "trepo-current-user-consents"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_client = None


class AIConsentRequired(PermissionError):
    pass


def allowed(owner, client=None):
    global _client
    if not isinstance(owner, str) or not UUID.fullmatch(owner):
        return False
    if client is None:
        if _client is None:
            _client = boto3.client("dynamodb")
        client = _client
    item = client.get_item(TableName=TABLE, Key={"user_id": {"S": "ai:" + owner.lower()}},
                           ConsistentRead=True).get("Item")
    if item is None:
        # Remove this compatibility branch only with the approved minimum-version rollout.
        return os.environ.get("AI_CONSENT_REQUIRE_ALL") != "true"
    return item.get("policy_version", {}).get("S") == VERSION and item.get("allowed", {}).get("BOOL") is True


def require(owner, client=None):
    if not allowed(owner, client):
        raise AIConsentRequired("AI processing is off. Review your choice in Trepo > Profile > AI privacy.")


def response():
    return {"statusCode": 428, "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
            "body": json.dumps({"error": "ai_consent_required", "message": "Review AI privacy in Trepo before using this feature."})}


def event_owner(event):
    body = event.get("body") or {}
    if isinstance(body, str):
        try:
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode()
            body = json.loads(body)
        except (ValueError, UnicodeError):
            body = {}
    if not isinstance(body, dict):
        body = {}
    headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}
    for key in ["user_id", "userId", "owner", "owner_id", "ownerId"]:
        for container in [body, event.get("payload") or {}, event.get("pathParameters") or {}, event]:
            value = container.get(key)
            if isinstance(value, str) and UUID.fullmatch(value):
                return value.lower()
    for key in ["x-owner-id", "x-user-id"]:
        value = headers.get(key)
        if isinstance(value, str) and UUID.fullmatch(value):
            return value.lower()
    return None


def request_actor(event):
    headers = {str(k).lower():str(v) for k,v in (event.get("headers") or {}).items()}
    token = headers.get("authorization")
    if not token:
        return event_owner(event)
    request = urllib.request.Request("https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com/v1/consents/ai",
        data=json.dumps({"version":VERSION,"action":"check"}).encode(), method="POST",
        headers={"Authorization":token,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result=json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401,403): raise AIConsentRequired("Verified identity required") from None
        raise
    actor=result.get("user_id")
    if not isinstance(actor,str) or not UUID.fullmatch(actor): raise ValueError("Invalid consent identity")
    return actor


def guard_handler(handler):
    def guarded(event, context):
        method = (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "").upper()
        if method == "OPTIONS" or event.get("warmup") is True or event.get("source") == "keepwarm":
            return handler(event, context)
        try:
            actor = request_actor(event)
            require(actor)
            event = dict(event, _ai_consent_actor=actor)
        except AIConsentRequired:
            return response()
        except Exception:
            return {"statusCode": 503, "body": '{"error":"ai_consent_unavailable"}'}
        token = actor_context.set(actor)
        try:
            return handler(event, context)
        finally:
            actor_context.reset(token)
    return guarded

def context_handler(handler):
    """Keep identity separate from household ownership without gating manual work."""
    def scoped(event, context):
        try:
            actor = request_actor(event)
        except AIConsentRequired:
            return response()
        except Exception:
            actor = UNVERIFIED_ACTOR
        token = actor_context.set(actor)
        try:
            return handler(event, context)
        except AIConsentRequired:
            return response()
        finally:
            actor_context.reset(token)
    return scoped


def body_operation(event):
    try:
        body = event.get('body') or {}
        if isinstance(body,str):
            if event.get('isBase64Encoded'): body=base64.b64decode(body).decode()
            body=json.loads(body)
        return body.get('operation') if isinstance(body,dict) else None
    except (ValueError,TypeError,UnicodeError): return None

def provider_call(call):
    def checked(*args, **kwargs):
        require(actor_context.get())
        return call(*args, **kwargs)
    return checked


def submit(executor, function, *args, **kwargs):
    from contextvars import copy_context
    return executor.submit(copy_context().run, function, *args, **kwargs)


def provider_client(constructor):
    """Check the transport for every SDK attempt, including internal retries."""
    def create(*args, **kwargs):
        client = constructor(*args, **kwargs)
        hooks = client._client.event_hooks
        if constructor.__name__ == 'AsyncOpenAI':
            async def check(request):
                require(actor_context.get())
        else:
            def check(request):
                require(actor_context.get())
        hooks.setdefault('request', []).append(check)
        return client
    return create


def ensure_refresh_actor_column(conn, table):
    # Schema changes happen alongside the existing lazy table preparation, before
    # inventory transactions. Only the expected duplicate-column error is benign.
    with conn.cursor() as cur:
        cur.execute('SELECT 1 FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s', [table, 'ai_consent_actor'])
        if cur.fetchone():
            return
        try:
            cur.execute('ALTER TABLE `' + table + '` ADD COLUMN ai_consent_actor VARCHAR(36) NULL')
        except Exception as error:
            if not error.args or error.args[0] != 1060:
                raise
    conn.commit()
