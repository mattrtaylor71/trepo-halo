// A narrow routing hint from the user's original command, never from assistant
// narration. It neither executes a write nor bypasses existing tool safeguards.
export function explicitLeftoverCheckIn(text) {
  const value=String(text || '').trim();
  if (value.length>300 || /["“”`;\n]/.test(value)
      || /\b(?:not|never|don['’]t|do not|without|instead|if|unless)\b/i.test(value)
      || /\b(?:shopping|grocery|list|calendar|recipe|recipes)\b/i.test(value)) return false;
  return /^(?:(?:please|can you|could you|would you|i want you to)\s+)*(?:add|put|save|check[ -]?in)\s+[^.?!]{1,180}\s+(?:as\s+(?:a\s+|some\s+)?leftovers?|(?:to|in|into)\s+(?:(?:my|the|our)\s+)?leftovers?)\s*[.!?]?$/i.test(value)
    || /^(?:(?:please|can you|could you|would you)\s+)*(?:add|save|check[ -]?in)\s+(?:(?:my|the|some)\s+)?leftover\s+[^.?!]{1,180}[.!?]?$/i.test(value);
}

export function leftoverFirstToolChoice(text) {
  if (!explicitLeftoverCheckIn(text)) return null;
  // The many-item tool also supports a single compound food such as mac and cheese.
  return {type:'function',function:{name:/\band\b|,/i.test(text) ? 'check_in_many_items' : 'check_in_item'}};
}
