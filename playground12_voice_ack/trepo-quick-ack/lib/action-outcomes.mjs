// Only durable tool-returned entities may supply a confirmation. Requested names,
// counts and quantities are intent, not evidence that a write committed.
const specs = [
  ['shopping_aisle', 'Moved to your shopping aisles', ['set_shopping_item_aisle']],
  ['shopping_add', 'Added to your shopping list', ['add_to_shopping_list','add_many_to_shopping_list','add_dish_ingredients_to_shopping_list','add_recipe_ingredients_to_shopping_list','add_saved_recipe_ingredients_to_shopping_list']],
  ['shopping_remove', 'Removed from your shopping list', ['remove_from_shopping_list','clear_shopping_list']],
  ['kitchen_add', 'Added to your kitchen', ['check_in_item','check_in_many_items']],
  ['kitchen_remove', 'Removed from your kitchen', ['discard_item','clear_kitchen_inventory']],
  ['kitchen_update', 'Updated in your kitchen', ['update_item_quantity','mark_item_opened','update_item_expiration','update_item_location','update_item_details']],
  ['recipe_save', 'Saved to your recipes', ['save_generated_recipe','save_recipe_from_tiktok']],
  ['recipe_remove', 'Removed from your recipes', ['remove_saved_recipe']],
  ['dishes', 'Logged in your dish log', ['log_dish_ingredients','log_dish_from_voice']],
  ['dishes', 'Updated in your dish log', ['update_recent_dish','append_to_recent_dish','mark_dish_consumed']],
  ['dishes', 'Removed from your dish log', ['delete_dish_log']],
  ['meal_calendar', 'Added to your meal calendar', ['add_recipe_to_meal_calendar','add_many_to_meal_calendar','add_generated_recipes_to_meal_calendar']],
  ['meal_calendar', 'Moved in your meal calendar', ['move_meal_calendar_entry']],
  ['meal_calendar', 'Removed from your meal calendar', ['remove_meal_calendar_entry']],
];
const byTool = new Map(specs.flatMap(([domain,label,tools]) => tools.map(name => [name,{domain,label}])));
const value = v => typeof v === 'string' || typeof v === 'number' ? String(v).trim() : '';

function entity(row, parent, toolName) {
  const data = row?.item || row?.dish || row?.recipe || row?.entry || row || {};
  const id = value(data.id || data._id || data.shopping_id || row?.entry_id || row?.removed_id || parent?.removed_id || row?.id);
  const name = value(data.item_name || data.product_name || data.dish_name || data.title || row?.title || row?.recipe_name || parent?.title);
  const quantity = value(data.remaining_quantity || data.quantity || (data.quantity_value != null ? `${data.quantity_value}${data.quantity_unit ? ' '+data.quantity_unit : ''}` : ''));
  const date = value(data.plan_date || row?.plan_date || parent?.new_date || parent?.plan_date);
  const slot = value(data.meal_slot || row?.meal_slot || parent?.new_meal_slot || parent?.meal_slot);
  const aisle = value(data.aisle_label);
  return {id,name,quantity,date,slot,aisle};
}

export function actionOutcomes(toolEvents = []) {
  return toolEvents.flatMap(event => {
    const spec = byTool.get(event?.toolName);
    if (!spec) return [];
    const result = event.result || {};
    // A denied unrequested tool never reached the executor. Do not replace a
    // useful read-only answer with a failure for work the user never requested.
    if (result.suppressed || (result.needs_clarification === true && result.toolResult?.inventory_changed === false)) return [];
    const body = result.toolResult;
    const base = {...spec,toolName:event.toolName,entities:[],failed:0,unverified:0};
    if (result.toolName && result.toolName !== event.toolName) return [{...base,state:'unverified',unverified:1}];
    if (!result.ok || Number(result.statusCode) >= 400 || body?.ok === false) {
      const error = result.error || body?.error || null;
      const uncertain = Number(result.statusCode) >= 500 || Number(result.statusCode) === 408 || /timeout|timed.?out|connection.*(?:reset|closed)/i.test(error || '');
      return [{...base,state:uncertain ? 'unverified' : 'failed',failed:uncertain ? 0 : 1,unverified:uncertain ? 1 : 0,error}];
    }
    if (!body || typeof body !== 'object') return [{...base,state:'unverified',unverified:1}];
    const rows = Array.isArray(body.affected_items) ? body.affected_items : Array.isArray(body.results) ? body.results : Array.isArray(body.items) ? body.items : [body];
    for (const row of rows) {
      if (!row || row.ok === false || row.error) {base.failed++;continue;}
      const item = entity(row, body, event.toolName);
      const requestedId = value(event.args?.item_id || event.args?.dish_id || event.args?.entry_id);
      if (!item.id || !item.name || (requestedId && requestedId !== item.id)) {base.unverified++;continue;}
      if (event.toolName === 'set_shopping_item_aisle') {
        const data = row?.item || row;
        if (!item.aisle || data.aisle_key !== event.args?.aisle_key ||
            item.name.toLocaleLowerCase('en-US') !== value(event.args?.item_name).toLocaleLowerCase('en-US')) {base.unverified++;continue;}
      }
      base.entities.push(item);
    }
    // A batch count alone cannot prove which targets changed.
    if (!rows.length) base.unverified++;
    return [{...base,state:base.entities.length ? (base.failed || base.unverified ? 'partial' : 'confirmed') : (base.failed ? 'failed' : 'unverified')}];
  });
}

export function actionOutcomeText(outcomes) {
  const sections=[];
  for (const outcome of outcomes) {
    if (outcome.entities.length) {
      const lines=outcome.entities.map(e => `${e.name}${e.aisle ? ` → ${e.aisle}` : ''}${e.quantity ? ` (${e.quantity})` : ''}${e.date ? ` · ${e.date}${e.slot ? ' '+e.slot : ''}` : ''}`);
      sections.push(`${outcome.label}:\n${lines.map(x => '- '+x).join('\n')}`);
    }
    if (outcome.failed) sections.push(outcome.error === 'ambiguous_match' ? 'More than one item matches. Choose the specific item before removing or changing it.' : "Some requested changes couldn't be completed.");
    if (outcome.unverified) sections.push("I couldn't confirm every change. Check the relevant list before trying again.");
  }
  return [...new Set(sections)].join('\n\n');
}

export function isActionTool(name) { return byTool.has(name); }
