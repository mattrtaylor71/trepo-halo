# trepo_auth.py — SEC-001 Phase B: shared token verification + owner authorization.
# Observe mode (ENFORCE_AUTH != 'true'): never blocks; logs would-deny decisions.
# Enforce mode: returns a 403 response dict when the token does not authorize `owner`.
import os, json, hmac, hashlib, base64, time, math

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
        if not isinstance(header, dict) or header.get('alg') != 'HS256':
            return None
        claims = json.loads(_b64url_decode(payload_b64))
        if not isinstance(claims, dict) or claims.get('iss') != _ISSUER:
            return None
        exp = claims.get('exp')
        if isinstance(exp, bool) or not isinstance(exp, (int, float)) or not math.isfinite(exp):
            return None
        if time.time() > exp + 60:
            return None
        nbf = claims.get('nbf')
        if nbf is not None and (isinstance(nbf, bool) or not isinstance(nbf, (int, float)) or not math.isfinite(nbf) or nbf > time.time() + 60):
            return None
        if not any(isinstance(claims.get(k), str) and claims[k] for k in ('owner_id', 'user_id')):
            return None
        return claims
    except Exception:
        return None

def _parse_legacy(token):
    try:
        claims = json.loads(base64.b64decode(token, validate=True).decode('utf-8'))
        return claims if isinstance(claims, dict) else None
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
            return _forbidden(owner, 'missing_owner') if _ENFORCE else None
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
                    if isinstance(claims.get(k), str) and claims[k]:
                        authorized_ids.add(claims[k])
        match = str(owner) in authorized_ids
        decision = 'allow' if (match and auth_kind == 'jwt') else 'deny'
        print(json.dumps({'evt': 'authz', 'owner': str(owner), 'auth_kind': auth_kind,
                          'match': match, 'enforce': _ENFORCE, 'would': decision}))
        if _ENFORCE and decision == 'deny':
            return _forbidden(owner, 'auth_kind=%s match=%s' % (auth_kind, match))
        return None
    except Exception as e:
        try:
            print(json.dumps({'evt': 'authz_error', 'error': type(e).__name__}))
        except Exception:
            pass
        return _forbidden(owner, 'invalid_authorization') if _ENFORCE else None
