// Narrow destination policy for a fresh Halo add command. This does not parse
// item names, execute tools, or infer a destination from conversation history.
const SHOPPING_TOOLS = new Set(['add_to_shopping_list', 'add_many_to_shopping_list']);
const KITCHEN_TOOLS = new Set(['check_in_item', 'check_in_many_items', 'clarify_kitchen_check_in']);
const READ_TOOLS = new Set([
  'get_shopping_list', 'get_shopping_aisles', 'list_store_tabs',
  'get_kitchen_overview', 'search_kitchen_item', 'get_recent_discards',
  'get_recent_dishes', 'get_dish_detail', 'get_health_metrics',
  'get_recipe_suggestions', 'get_recipe_detail', 'get_saved_recipes',
  'get_saved_recipe_detail', 'search_web_recipes', 'get_meal_plan',
  'recommend_items_to_buy', 'recommend_items_to_use_up',
  'list_recipe_categories', 'get_meal_calendar',
]);

// Optional politeness is still a command ("could you add milk?"). Questions
// about how/whether to add, quoted examples, and ordinary conversation do not
// match this anchored command prefix.
const COMMAND = /^(?:(?:hey\s+)?(?:halo|thyme)[, ]+)?(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i(?:'d| would)\s+like\s+you\s+to\s+|i\s+(?:want|need)\s+you\s+to\s+)?(add|check[ -]?in)\s+(.+)$/i;
const DESTINATION = /\b(?:to|in|into|on|onto)\s+(?:(?:my|our|the|a)\s+)?(shopping\s+list|grocery\s+list|list|kitchen|pantry|fridge|freezer|inventory)\s*$/i;
const OTHER_SCOPE = /\b(?:recipe|recipes|calendar|meal[ -]?plan|meal[ -]?calendar|dish\s+log|breakfast|lunch|dinner|reminder|reminders|timer|timers|playlist|contact|contacts|account|note|notes)\b/i;
const ANOTHER_ACTION = /\b(?:and(?:\s+then)?|then|but|also)\s+(?:please\s+)?(?:add|put|check[ -]?in|remove|delete|discard|throw|toss|clear|empty|log|save|move|mark|set|update|cook|mix|stir|make|tell|show|find)\b/i;

export function haloAddDestination(transcript, responseSurface = 'halo') {
  if (String(responseSurface).trim().toLowerCase() !== 'halo') return null;
  if (typeof transcript !== 'string') return null;
  const text = transcript.replace(/[’‘]/g, "'").trim().replace(/[.!?]+$/, '').trim();
  if (!text || /[;\n!?]|(?<!\d)\.|\.(?!\d)/.test(text)) return null;
  if (/\b(?:don't|do not|never|not|instead of|except)\b/i.test(text)) return null;
  const command = text.match(COMMAND);
  if (!command) return null;
  const checkIn = /^check/i.test(command[1]);
  const body = command[2].replace(/,?\s+please$/i, '').trim();
  if (!body || OTHER_SCOPE.test(body) || ANOTHER_ACTION.test(body)) return null;

  const scoped = body.match(DESTINATION);
  const items = scoped ? body.slice(0, scoped.index).trim() : body;
  if (!items || /\b(?:to|into|onto|on|in)\b/i.test(items)) return null;
  // A second destination, conditional, or clause is deliberately left to the
  // existing assistant. A comma/"and" between item names remains supported.
  if (/\b(?:if|unless|because|when|whether|so that)\b/i.test(items)) return null;
  if (scoped) {
    const destination = /^(?:shopping\s+list|grocery\s+list|list)$/i.test(scoped[1])
      ? 'shopping' : 'kitchen';
    return checkIn && destination === 'shopping' ? null : destination;
  }
  return checkIn ? 'kitchen' : 'shopping';
}

// This guard must run before mutation in both sync and streaming callers.
// It leaves app behavior and compound/out-of-scope requests unchanged.
export function haloAddToolDenial(transcript, toolName, responseSurface = 'halo') {
  const destination = haloAddDestination(transcript, responseSurface);
  if (!destination || READ_TOOLS.has(toolName)) return null;
  const allowed = destination === 'shopping' ? SHOPPING_TOOLS : KITCHEN_TOOLS;
  return allowed.has(toolName) ? null : `halo_add_requires_${destination}`;
}
