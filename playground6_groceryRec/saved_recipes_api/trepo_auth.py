# trepo_auth.py — SEC-001 Phase B: shared token verification + owner authorization.
# Observe mode (ENFORCE_AUTH != 'true'): never blocks; logs would-deny decisions.
# Enforce mode: returns a 403 response dict when the token does not authorize `owner`.
import os, json, hmac, hashlib, base64, time

_SECRET = os.getenv('TOKEN_SIGNING_SECRET', '')
_ENFORCE = os.getenv('ENFORCE_AUTH', 'false').strip().lower() == 'true'
_ISSUER = 'trepo-auth'

def _b64url_decode(seg):
    pad = '=' * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)

def _verify_jwt(token):
    try:
        if not _SECRET:
            return None
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = (header_b64 + '.' + payload_b64).encode('ascii')
        expected = hmac.new(_SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            return None
        header = json.loads(_b64url_decode(header_b64))
        if header.get('alg') != 'HS256':
            return None
        claims = json.loads(_b64url_decode(payload_b64))
        if claims.get('iss') != _ISSUER:
            return None
        exp = claims.get('exp')
        if exp is not None and time.time() > float(exp) + 60:
            return None
        return claims
    except Exception:
        return None

def _parse_legacy(token):
    try:
        return json.loads(base64.b64decode(token).decode('utf-8'))
    except Exception:
        return None

def _get_header(event, name):
    for k, v in (event.get('headers') or {}).items():
        if k.lower() == name:
            return v
    return None

def _forbidden(owner, reason):
    return {
        'statusCode': 403,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'error': 'Forbidden', 'reason': reason})
    }

def require_owner(event, owner=None):
    """Return None to ALLOW; return a 403 dict to DENY (enforce mode only). NEVER raises.
    Usage at top of handler:
        denied = require_owner(event)
        if denied is not None:
            return denied
    """
    try:
        method = (((event.get('requestContext') or {}).get('http') or {}).get('method')
                  or event.get('httpMethod') or 'GET').upper()
        if method == 'OPTIONS':
            return None
        if owner is None:
            owner = (event.get('pathParameters') or {}).get('owner')
        if not owner:
            return None
        auth = _get_header(event, 'authorization') or ''
        token = auth[7:].strip() if auth.lower().startswith('bearer ') else ''
        auth_kind = 'none'
        authorized_ids = set()
        if token:
            claims = _verify_jwt(token)
            if claims is not None:
                auth_kind = 'jwt'
            else:
                claims = _parse_legacy(token)
                if claims is not None:
                    auth_kind = 'legacy'
            if claims:
                for k in ('owner_id', 'user_id'):
                    if claims.get(k):
                        authorized_ids.add(str(claims[k]))
        match = str(owner) in authorized_ids
        decision = 'allow' if (match and auth_kind == 'jwt') else 'deny'
        print(json.dumps({'evt': 'authz', 'owner': str(owner), 'auth_kind': auth_kind,
                          'match': match, 'enforce': _ENFORCE, 'would': decision}))
        if _ENFORCE and decision == 'deny':
            return _forbidden(owner, 'auth_kind=%s match=%s' % (auth_kind, match))
        return None
    except Exception as e:
        try:
            print(json.dumps({'evt': 'authz_error', 'error': str(e)}))
        except Exception:
            pass
        return None
