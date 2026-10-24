"""Redact credentials at this application's logging boundary, never in API payloads."""
import builtins
import json
import re
from urllib.parse import unquote

_REDACTED = '[REDACTED]'
_SENSITIVE = re.compile(r'^(?:authorization|proxy.authorization|cookie|set.cookie|token|access.token|refresh.token|api.key|apikey|secret|password|client.secret|x.api.key|credential|signature|x.amz.signature|x.amz.credential|x.amz.security.token)$', re.I)
_QUERY = re.compile(r'([?&](?:[^=\s&"\'<>]+)=)([^&\s"\'<>]*)')
_TOKEN = re.compile(r'\b(?:apify_api_|sk_live_|sk_test_|sk-proj-|sk-ant-)[A-Za-z0-9_-]+')
_BEARER = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*')
_HEADER = re.compile(r'(?im)\b(authorization|proxy-authorization|x-api-key|cookie|set-cookie)\s*:\s*[^\r\n]+')
_URL_USERINFO = re.compile(r'(https?://)[^/\s:@]+:[^@\s/]+@', re.I)
_ASSIGNMENT = re.compile(r'''(?i)(["']?\b(?:token|access_token|refresh_token|api_key|apikey|client_secret|password|secret)["']?\s*[:=]\s*)(?:\[REDACTED\]|"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}\]&]+)''')


def redact_text(value):
    def query(match):
        key = unquote(match.group(1)[1:-1])
        return match.group(1) + (_REDACTED if _SENSITIVE.match(key) else match.group(2))
    text = _QUERY.sub(query, value)
    text = _URL_USERINFO.sub(lambda m: m.group(1) + _REDACTED + '@', text)
    text = _TOKEN.sub(_REDACTED, text)
    text = _ASSIGNMENT.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _BEARER.sub('Bearer ' + _REDACTED, text)
    return _HEADER.sub(lambda m: m.group(1) + ': ' + _REDACTED, text)


def redact(value):
    if isinstance(value, dict):
        return {key: _REDACTED if _SENSITIVE.match(str(key)) else redact(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        # Many existing emitters serialize before printing; handle their nested fields too.
        if value.lstrip().startswith(('{', '[')):
            try:
                return json.dumps(redact(json.loads(value)), ensure_ascii=False)
            except (ValueError, TypeError, RecursionError):
                pass
        return redact_text(value)
    return value


def safe_print(*values, **kwargs):
    # Preserve print's destination/flush contract and existing JSON event field names.
    cleaned = [redact(v if isinstance(v, (dict, list, tuple, str)) else str(v)) for v in values]
    builtins.print(*cleaned, **kwargs)
