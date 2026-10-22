import crypto from 'node:crypto';
import {readKitchenEditRequest, retainKitchenEditRequest} from './session-store.mjs';

const DEFAULT_API = 'https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com';
const EDIT_FIELDS = new Set(['product_name','brand','category','storage_location','is_opened',
  'product_expiration','quantity_value','quantity_unit']);

// Keep this finite alias contract aligned with Kitchen amount_operations.unit.
const UNIT_ALIASES = {"lbs": "lb", "pound": "lb", "pounds": "lb", "ounces": "oz", "ounce": "oz", "grams": "g", "kilograms": "kg", "liter": "l", "litre": "l", "liters": "l", "pieces": "count", "each": "count", "packs": "pack", "gallons": "gallon", "quarts": "quart", "pints": "pint", "cups": "cup", "tablespoon": "tbsp", "tablespoons": "tbsp", "teaspoon": "tsp", "teaspoons": "tsp", "cans": "can", "jars": "jar", "bottles": "bottle", "bags": "bag", "slices": "slice", "servings": "serving", "containers": "container", "package": "pack", "packages": "pack", "cloves": "clove", "heads": "head"};
const UNITS = new Set(["bag", "bottle", "bunch", "can", "clove", "container", "count", "cup", "fl oz", "g", "gallon", "head", "jar", "kg", "l", "lb", "ml", "oz", "pack", "pint", "quart", "serving", "slice", "tbsp", "tsp"]);

// IDs seen in model grounding are not a user selection. For a new rename,
// resolve the actual label again and require an unambiguous target. A pending
// clarification carries a server-held revision and may use its selected ID.
export function renameReference(reference, updates, options, resolvedRow) {
  if (!updates?.new_name || !options?.sourceTranscript || options.expectedRevision != null) return reference;
  const transcript=String(options.sourceTranscript);
  const id=String(reference.item_id || '');
  if (id && transcript.split(/\s+/).some(word=>word.replace(/^["'(]+|["'),.!?]+$/g,'')===id)) return reference;
  const lower=transcript.toLowerCase(), replacement=String(updates.new_name).toLowerCase();
  const lastReplacement=lower.lastIndexOf(replacement);
  const withoutName=lastReplacement<0?lower:lower.slice(0,lastReplacement)+lower.slice(lastReplacement+replacement.length);
  const label=reference.item_name || resolvedRow?.product_name;
  // The old label must be grounded in the user's words, not an unrelated row
  // selected from the inventory by the model. Pronouns need a pending target.
  const normalized=value=>String(value||'').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
  if (!label || !(` ${normalized(withoutName)} `).includes(` ${normalized(label)} `))
    throw rejected('Please repeat the item and its full new name to confirm this correction.');
  const selector=normalized(withoutName).replace(normalized(label),'');
  const newest=/\b(?:most recent|newest|latest)\b/.test(selector);
  const oldest=/\b(?:oldest|earliest)\b/.test(selector);
  return {item_name:label, item_id:null,
    selection_hint:newest !== oldest ? (newest?'most_recent':'oldest') : null};
}

export function pickRenameByCreation(rows, hint) {
  if (!['most_recent','oldest'].includes(hint)) return null;
  const timestamp=row=>new Date(row._createdDate ?? row.created_at).getTime();
  if (!rows.length || rows.some(row=>!Number.isFinite(timestamp(row)))) return null;
  const sorted=[...rows].sort((a,b)=>timestamp(b)-timestamp(a));
  if (hint==='oldest') sorted.reverse();
  if (sorted[1] && timestamp(sorted[0])===timestamp(sorted[1])) return null;
  return sorted[0];
}
export function canonicalKitchenUnit(value) {
  const text = String(value || "").trim().toLowerCase();
  const unit = ({piece:'count',item:'count',items:'count'}[text]) || UNIT_ALIASES[text] || text;
  if (!UNITS.has(unit)) throw rejected("Set an amount and supported unit before changing this item.", 400);
  return unit;
}

function rejected(message, statusCode = 409) {
  return Object.assign(new Error(message), {statusCode});
}

// Request metadata is supplied by the entry point, never by model arguments.
// The object travels in action options only: it is not prompt/session content.
export function createKitchenMutationContext(request = {}) {
  return {requestId: request?.operationId || crypto.randomUUID(),
    authorization: request?.authorization || null, operations: new Map(), editIntents:new Map()};
}

function kitchenEditOperationId(mutation, ordinal) {
  const bytes = crypto.createHash('sha256').update(`${mutation.requestId}:kitchen:${ordinal}`).digest().subarray(0,16);
  bytes[6] = (bytes[6] & 15) | 80; bytes[8] = (bytes[8] & 63) | 128;
  const hex = bytes.toString('hex');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}

function stableJSON(value) {
  return JSON.stringify(value, (_, entry) => entry && typeof entry === 'object' && !Array.isArray(entry)
    ? Object.fromEntries(Object.entries(entry).sort(([a],[b])=>a.localeCompare(b))) : entry);
}

// Read before resolving the item: a committed rename may have removed the old
// lookup name, and a later human edit must not become this retry's new revision.
export async function prepareKitchenEdit(context, reference, updates, options, resolve) {
  const mutation = options.mutationContext;
  const intentHash = crypto.createHash('sha256').update(stableJSON({reference,updates,
    transcript:options.sourceTranscript || null,expectedRevision:options.expectedRevision ?? null})).digest('hex');
  if (!mutation.editIntents) mutation.editIntents = new Map();
  if (!mutation.editIntents.has(intentHash))
    mutation.editIntents.set(intentHash, kitchenEditOperationId(mutation, mutation.editIntents.size));
  const scope = {household:context?.ownerId, actor:context?.userId || context?.tableOwnerId,
    operationId:mutation.editIntents.get(intentHash), intentHash};
  const env = options.env || process.env;
  let body;
  try {
    body = await readKitchenEditRequest(scope, env);
    if (!body) {
      const prepared = await resolve();
      body = await retainKitchenEditRequest(scope,
        kitchenEditBody(mutation, prepared.row, prepared.fields, scope.operationId), env);
    }
  } catch (error) {
    if (error?.statusCode) throw error;
    throw rejected('Could not confirm the correction request. No new change was sent. Retry it.', 503);
  }
  const entry = body?.items?.[0];
  if (body?.operation_id !== scope.operationId || body?.kind !== 'edit' || body?.items?.length !== 1 ||
      typeof entry?.item_id !== 'string' || !entry.item_id || !Number.isSafeInteger(entry?.revision) ||
      !entry.fields || !Object.keys(entry.fields).length || Object.keys(entry.fields).some(key=>!EDIT_FIELDS.has(key)))
    throw rejected('Could not confirm the original correction request. Refresh your kitchen.', 503);
  const fields = Object.fromEntries(Object.entries(entry.fields).sort(([a],[b])=>a.localeCompare(b)));
  mutation.operations.set(JSON.stringify([entry.item_id, fields]), body);
  return {row:{_id:entry.item_id,amount_revision:entry.revision}, fields};
}

export function kitchenEditBody(mutation, row, fields, explicitOperationId) {
  if (!fields || !Object.keys(fields).length || Object.keys(fields).some(k => !EDIT_FIELDS.has(k)))
    throw rejected('Unsupported kitchen correction fields.', 400);
  if (!row?._id || !Number.isSafeInteger(Number(row.amount_revision)))
    throw rejected('Refresh this item before changing it.');
  const normalized = {...fields, ...(fields.quantity_unit != null ? {quantity_unit:canonicalKitchenUnit(fields.quantity_unit)} : {})};
  const patch = Object.fromEntries(Object.entries(normalized).sort(([a],[b]) => a.localeCompare(b)));
  const key = JSON.stringify([row._id, patch]);
  if (mutation.operations.has(key)) return mutation.operations.get(key);
  // Derive from the request and action position, not the model's tool-call ID or
  // patch contents. A changed retry cannot disguise itself as the same write.
  const operation_id = explicitOperationId || kitchenEditOperationId(mutation, mutation.operations.size);
  const body = {operation_id, kind:'edit', items:[{item_id:row._id, revision:Number(row.amount_revision), fields:patch}]};
  mutation.operations.set(key, body);
  return body;
}

export async function saveKitchenEdit(context, row, fields, options = {}) {
  const actor = context?.userId || context?.tableOwnerId;
  if (typeof actor !== 'string' || !/^[a-zA-Z0-9_-]{1,64}$/.test(actor))
    throw rejected('Could not confirm your kitchen account.', 403);
  const mutation = options.mutationContext || createKitchenMutationContext();
  const body = kitchenEditBody(mutation, row, fields);
  const base = new URL(options.env?.KITCHEN_API_BASE_URL || DEFAULT_API);
  if (base.protocol !== 'https:' || base.username || base.password || base.search || base.hash)
    throw rejected('Kitchen editing is temporarily unavailable.', 503);
  const response = await fetch(`${base.href.replace(/\/$/, '')}/kitchen/${encodeURIComponent(actor)}?amount_operation=v1`, {
    method:'POST', redirect:'error', signal:AbortSignal.timeout(20000),
    headers:{'Content-Type':'application/json', ...(mutation.authorization ? {Authorization:mutation.authorization} : {})},
    body:JSON.stringify(body)
  });
  let payload;
  try { payload = await response.json(); } catch { throw rejected('Could not confirm the saved change. Retry this request.', 503); }
  if (!response.ok) throw rejected(payload?.error || payload?.message || 'Could not save this change.', response.status);
  const entry = payload?.items?.[0];
  if (payload?.operation_id !== body.operation_id || payload?.items?.length !== 1 || entry?.removed
      || entry?.item?._id !== row._id || !Number.isSafeInteger(Number(entry.item.amount_revision)))
    throw rejected('Could not confirm the saved change. Refresh your kitchen.', 503);
  for (const [key, requested] of Object.entries(body.items[0].fields)) {
    const saved = entry.item[key];
    const matches = key === 'quantity_value' ? saved != null && Number(saved) === Number(requested)
      : key === 'is_opened' ? [true,false,0,1].includes(saved) && Boolean(saved) === requested
      : saved === requested;
    if (!matches) throw rejected('Could not confirm the saved change. Refresh your kitchen.', 409);
  }
  return {...entry.item, mutation_outcome:'applied', operation_id:body.operation_id};
}
