const { createHmac, timingSafeEqual } = require('crypto');
function authorizeCommit(event, owner, secret = process.env.TOKEN_SIGNING_SECRET) {
  if (!secret) return { status: 503, code: 'auth_unavailable' };
  try {
    const headers = Object.entries(event.headers || {}).filter(([key]) => key.toLowerCase() === 'authorization');
    if (headers.length !== 1 || typeof headers[0][1] !== 'string') throw new Error();
    const match = /^Bearer ([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)$/i.exec(headers[0][1].trim());
    if (!match || match[1].length > 8192) throw new Error();
    const [head, body, signature] = match[1].split('.');
    const expected = createHmac('sha256', secret).update(head + '.' + body).digest();
    const actual = Buffer.from(signature, 'base64url');
    if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) throw new Error();
    const header = JSON.parse(Buffer.from(head, 'base64url'));
    const claims = JSON.parse(Buffer.from(body, 'base64url'));
    if (header?.alg !== 'HS256' || header.crit || claims?.iss !== 'trepo-auth' ||
        typeof claims.exp !== 'number' || !Number.isFinite(claims.exp) || claims.exp <= Date.now()/1000 ||
        (claims.nbf !== undefined && (typeof claims.nbf !== 'number' || !Number.isFinite(claims.nbf) || claims.nbf > Date.now()/1000))) throw new Error();
    if (![claims.owner_id, claims.user_id].some(value => typeof value === 'string' && value === owner)) return { status: 403, code: 'owner_forbidden' };
    return { userId: typeof claims.user_id === "string" && claims.user_id ? claims.user_id : owner };
  } catch { return { status: 401, code: 'authentication_required' }; }
}
// Released iOS <=1.13 sends no Authorization header. Its unguessable scan UUID
// is the legacy capability: constrain it to a completed inventory scan and owner.
// Never downgrade a supplied invalid/expired signed credential to this path.
function isLegacyCommit(event, sourceId) {
  return !Object.keys(event.headers || {}).some(key => key.toLowerCase() === 'authorization') &&
    typeof sourceId === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(sourceId);
}
function authorizeLegacySource(source, owner) {
  validateSourceOwner(source, owner);
  if (source.status !== 'completed' || !['receipt_inventory_deep', 'bulk_inventory_deep'].includes(source.analysis_mode)) {
    const error = new Error('This scan is not ready to add. Please reopen your scan.');
    error.code = 'source_owner_conflict'; throw error;
  }
  // Do not trust a separately supplied body user_id for attribution.
  return { userId: typeof source.meta.user_id === 'string' && source.meta.user_id ? source.meta.user_id : owner };
}
function validateSourceOwner(source, owner) {
  if (!source?.meta || typeof source.meta.owner !== 'string' || source.meta.owner !== owner) {
    const error = new Error('This scan does not belong to the signed-in account. Please reopen your scan.');
    error.statusCode = 409; error.code = 'source_owner_conflict'; throw error;
  }
}
module.exports = { authorizeCommit, validateSourceOwner, isLegacyCommit, authorizeLegacySource };
