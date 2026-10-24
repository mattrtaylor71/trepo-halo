import base64,hashlib,hmac,json,os,re,time
UUID=re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",re.I)

def authenticate(event, owner):
    secret = os.getenv('TOKEN_SIGNING_SECRET')
    if not secret:
        return 503, 'AUTH_UNAVAILABLE'
    try:
        headers = [v for k, v in (event.get('headers') or {}).items() if k.lower() == 'authorization']
        assert len(headers) == 1 and isinstance(headers[0], str)
        match = re.fullmatch(r'Bearer ([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)', headers[0].strip(), re.I)
        assert match and len(match[1]) <= 8192
        head, body, signature = match[1].split('.')
        decode = lambda s: base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))
        assert hmac.compare_digest(hmac.new(secret.encode(), (head + '.' + body).encode(), hashlib.sha256).digest(), decode(signature))
        header, claims = json.loads(decode(head)), json.loads(decode(body))
        assert header.get('alg') == 'HS256' and not header.get('crit') and claims.get('iss') == 'trepo-auth'
        assert type(claims.get('exp')) in (int, float) and claims['exp'] > time.time()
        assert 'nbf' not in claims or type(claims['nbf']) in (int, float) and claims['nbf'] <= time.time()
        assert isinstance(claims.get('user_id'), str) and UUID.fullmatch(claims['user_id'])
    except Exception:
        return 401, 'AUTH_REQUIRED'
    if claims['user_id'] != owner:
        return 403, 'OWNER_MISMATCH'
    return None

