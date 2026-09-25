// Model-supplied correction fields must stay within the user's requested change.
// This complements the server's ownership/revision checks; it is not authorization.
export function correctionFieldDenial(toolName, args, transcript) {
  const original = String(transcript || '');
  // Names are data, even when they contain words such as Opened or Brand.
  let text = original.replace(/"[^"\n]*"|“[^”\n]*”|`[^`\n]*`/g, ' ');
  if (args?.new_name) text = text.replace(new RegExp(String(args.new_name).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), ' ');
  text = text.replace(/\b(?:rename|named?|call(?:ed)?)\b[^,.;!?]*?(?:\bto\b|\bis\b)[^,.;!?]*(?=[,.;!?]|$)/gi,
    clause => clause.replace(/\s+(?:and|also)\s+(?=(?:change|set|mark|move|store|put|open)\b)/i, '|').split('|').slice(1).join(' '));
  text = text.split(/[,;.!?]|\band\b|\bbut\b/i).filter(clause => !/\b(?:don['’]t|do not|not|never|without|keep|leave|unchanged)\b/i.test(clause)).join('. ');
  const renamed = Boolean(args?.new_name) || /\b(?:rename|renaming|name|called|spelled|spelling)\b/i.test(original);
  const opened = /\b(?:opened|unopened|sealed|unsealed|closed|resealed)\b/i.test(text)
    || /\b(?:mark|set)\b[^.?!]{0,30}\bopen\b/i.test(text)
    || /^\s*(?:please\s+)?open\s+(?:it|that|this|the|my)\b/i.test(text);
  if ((toolName === 'mark_item_opened' || (toolName === 'update_item_details' && typeof args?.is_opened === 'boolean')) && !opened)
    return 'opened_state_not_requested';
  if (!renamed) return null;
  const allowed = {
    brand:/\b(?:change|set|update|correct)\b[^.?!]{0,30}\bbrand\b|\bbrand\b[^.?!]{0,20}\b(?:is|should be)\b/i.test(text),
    category:/\b(?:categorize|categorise|classify)\b|\b(?:change|set|update|correct)\b[^.?!]{0,30}\bcategory\b/i.test(text),
    location:/\b(?:move|store|stored|storage|put)\b[^.?!]{0,40}\b(?:fridge|freezer|pantry)\b/i.test(text),
    expiration_date:/\b(?:expiration|expiry|expires|best before|use by)\b/i.test(text),
    is_opened:opened,
  };
  const amount = /\b(?:quantity|amount|remaining|left|half full|quarter full)\b/i.test(text);
  for (const key of ['quantity','remaining_quantity','quantity_value','quantity_unit','fill_percent']) allowed[key]=amount;
  if (toolName === 'update_item_details') {
    for (const [key, value] of Object.entries(args || {}))
      if (value != null && Object.hasOwn(allowed,key) && !allowed[key]) return `correction_field_not_requested:${key}`;
  } else if (toolName === 'update_item_quantity' && !amount) return 'quantity_not_requested_during_rename';
  else if (toolName === 'update_item_location' && !allowed.location) return 'location_not_requested_during_rename';
  else if (toolName === 'update_item_expiration' && !allowed.expiration_date) return 'expiry_not_requested_during_rename';
  else if (['discard_item','clear_kitchen_inventory','check_in_item','check_in_many_items'].includes(toolName)
    && !/\b(?:also|and then)\b[^.?!]{0,50}\b(?:add|check[ -]?in|discard|remove|delete)\b/i.test(text))
    return 'inventory_change_not_requested_during_rename';
  return null;
}
