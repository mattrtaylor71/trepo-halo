// Product labels are not ingredient lists. This narrow guard catches the
// demonstrated infused-oil ambiguity; other uncertainty stays with clarification.
export function kitchenLabelDenial(transcript, toolName, args) {
  if (!['check_in_item','check_in_many_items'].includes(toolName)) return null;
  const text=String(transcript || '').toLowerCase();
  const separate=/\b(?:separate|different products|two products|both products)\b/.test(text);
  if (!separate && /\b(?:garlic|cilantro|basil|chili|chilli)(?:\s+oil)?\s*,\s*(?:olive\s+)?oil\b/.test(text))
    return 'Is that one flavored oil, or two separate products?';
  const compound=text.match(/\b(?:garlic|cilantro|basil|chili|chilli)(?:[- ]infused)?\s+(?:extra virgin\s+)?olive oil\b/)?.[0];
  if (!compound || separate) return null;
  const names=toolName==='check_in_many_items' ? (args?.items || []).map(x=>x.item_name) : [args?.item_name];
  const flavor=compound.split(/[- ]/)[0];
  if (names.some(name=>{
    const normalized=String(name || '').toLowerCase();
    return /\boil\b/.test(normalized) && !normalized.includes(compound)
      && (normalized.includes(flavor) || /^(?:extra virgin )?olive oil$/.test(normalized.trim()));
  })) return 'Keep the flavored olive oil as one product. If you mean separate products, please confirm that.';
  return null;
}
