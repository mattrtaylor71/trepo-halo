// Public activity is server-owned copy, never model thoughts or tool arguments.
export const STAGES = Object.freeze({
  connecting: ['Connecting to Thyme', 'connection'],
  received: ['Request received', 'connection'],
  transcribing: ['Listening to your recording', 'voice'],
  household: ['Checking your kitchen and preferences', 'kitchen'],
  thinking: ['Putting your answer together', 'answer'],
  validating: ['Checking the result', 'answer'],
  recovering: ['Reconnecting to your saved request', 'connection'],
});

export function activity(stage, {id = stage, state = 'running', requestId = null} = {}) {
  const [label, category] = STAGES[stage] || STAGES.thinking;
  return {type: 'progress', version: 1, id, stage: STAGES[stage] ? stage : 'thinking',
    label, category, state, ...(requestId ? {request_id: requestId} : {})};
}

export function toolActivity(name, {id = name, state = 'running', requestId = null, args = {}} = {}) {
  const proposal = name.startsWith('request_');
  const n = name.replace(/^request_/, '');
  let label = 'Working through your request', category = 'answer';
  if (name === 'read_trepo') {
    [label, category] = ({kitchen: ['Checking your kitchen', 'kitchen'], shopping: ['Checking your shopping list', 'shopping'],
      preferences: ['Checking your food preferences', 'kitchen'], saved_recipes: ['Looking through your saved recipes', 'recipes'],
      calendar: ['Checking your meal calendar', 'calendar'], meal_plan: ['Checking your meal plan', 'calendar'],
      dishes: ['Checking your dish history', 'history'], discards: ['Checking your kitchen history', 'history']})[args?.resource]
      || ['Checking your Trepo data', 'kitchen'];
  } else if (/history|recent_dish|recent_discard/.test(n)) { label = 'Looking through your history'; category = 'history'; }
  else if (/clarif/.test(n)) label = 'Preparing a question for you';
  else if (/memory|preferences/.test(n)) { label = 'Checking your preferences'; category = 'kitchen'; }
  else if (/calendar|meal_plan/.test(n)) { label = /^(get_|read_)/.test(n) ? 'Checking your meal plan' : 'Working on your meal plan'; category = 'calendar'; }
  else if (/recipe/.test(n)) { label = /^(get_|search_|list_)/.test(n) ? 'Finding recipes for you' : 'Working on your recipe'; category = 'recipes'; }
  else if (/shopping|store_tabs/.test(n)) { label = /^(get_|list_)/.test(n) ? 'Checking your shopping list' : 'Updating your shopping list'; category = 'shopping'; }
  else if (/kitchen|item|check_in/.test(n)) { label = /^(get_|search_|recommend_)/.test(n) ? 'Checking your kitchen' : 'Updating your kitchen'; category = 'kitchen'; }
  else if (/dish/.test(n)) { label = 'Working on your dish log'; category = 'history'; }
  if (proposal) label = 'Preparing changes for your review';
  // A completed tool stage is not permission to claim a committed account change.
  if (state === 'completed') {
    if(proposal)label='Changes prepared for review';
    else if(name==='create_recipe'||name==='edit_recipe')label='Recipe ready';
    else if(/clarif/.test(n))label='Question ready';
    else if(/conversation_preferences/.test(n))label='Your choices noted for this chat';
    else if(/preferences|memory/.test(n) || (name==='read_trepo' && args?.resource==='preferences'))label='Food preferences checked';
    else if(name==='get_conversation_recipes')label='Chat recipes checked';
    else if(name==='search_web_recipes')label='Recipe search checked';
    else label=({kitchen:'Kitchen checked',shopping:'Shopping list checked',recipes:'Saved recipes checked',
      calendar:'Meal plan checked',history:'History checked'})[category] || 'Step checked';
  }
  if (state === 'failed') label = 'This step needs attention';
  return {type: 'progress', version: 1, id, stage: 'tool', label, category, state,
    ...(requestId ? {request_id: requestId} : {})};
}

export function updateActivities(previous = [], event) {
  if (!event || event.type !== 'progress' || event.version !== 1) return previous;
  const steps = previous.map(step => step.state === 'running' && step.stage !== 'tool' && step.id !== event.id
    ? {...step, state: 'completed'} : step);
  const index = steps.findIndex(step => step.id === event.id);
  const publicStep = Object.fromEntries(['id', 'stage', 'label', 'category', 'state'].map(k => [k, event[k]]));
  if (index < 0) steps.push(publicStep); else steps[index] = publicStep;
  return steps.slice(-8);
}
