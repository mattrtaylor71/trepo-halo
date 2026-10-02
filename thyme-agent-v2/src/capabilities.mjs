// One catalog drives discovery, dispatch and the parity audit. Read tools never write.
const str = { type: "string", minLength: 1, maxLength: 500 };
const list = { type: "array", items: str, maxItems: 100 };
const obj = (properties, required = []) => ({ type: "object", properties, required, additionalProperties: false });
const definition = (name, description, properties, required) => ({ type: "function", name, description,
  parameters: obj(properties, required), defer_loading: true });
export const READ_TOOLS = {
  get_shopping_list: "shopping", list_store_tabs: "stores", get_kitchen_overview: "kitchen",
  search_kitchen_item: "kitchen", get_recent_discards: "discards", get_recent_dishes: "dishes",
  get_dish_detail: "dishes", get_health_metrics: "health", get_recipe_suggestions: "suggestions",
  get_recipe_detail: "suggestions", get_saved_recipes: "saved_recipes", get_saved_recipe_detail: "saved_recipes",
  get_meal_plan: "meal_plan", get_meal_calendar: "calendar", list_recipe_categories: "recipe_categories",
  search_web_recipes: "web_recipes", recommend_items_to_buy: "buy_suggestions", recommend_items_to_use_up: "use_up_suggestions",
};
export const EXTRA_TOOLS = [
  definition("edit_saved_recipe", "Edit a recipe ALREADY saved in the app. Supply its exact recipe_id from get_saved_recipes. Only supplied fields change; other fields are preserved. Unlike edit_recipe this changes the library after approval.",
    { recipe_id: str, title: str, ingredients: list, instructions: list, notes: list }, ["recipe_id"]),
  definition("create_shopping_list", "Create a store/list group with initial items. The app derives lists from their items, so an empty list cannot be stored. All items go to this store.",
    { store: str, items: { type: "array", minItems: 1, maxItems: 50, items: obj({ item_name: str, quantity: str }, ["item_name"]) } }, ["store", "items"]),
  definition("rename_shopping_list", "Rename a store/list group by moving its exact current items to the new name. Other lists stay unchanged.",
    { store: str, new_name: str }, ["store", "new_name"]),
  definition("remove_shopping_list", "Remove one store/list group and all of its current items after explicit review. Does not remove other lists or kitchen items.",
    { store: str }, ["store"]),
  definition("update_shopping_item", "Edit a shopping item by exact item_id. Can change its name, amount, or store. Does not check it into the kitchen.",
    { item_id: str, new_name: str, quantity: str, store: str }, ["item_id"]),
  definition("update_dish", "Edit an exact personal Dish Log entry. Only supplied fields change. Nutrition is an estimate, never fabricated when unknown.",
    { dish_id: str, dish_name: str, serving_size: str, ingredients: list,
      calories: { type: "number", minimum: 0 }, protein_g: { type: "number", minimum: 0 },
      carbs_g: { type: "number", minimum: 0 }, fat_g: { type: "number", minimum: 0 } }, ["dish_id"]),
];
export const EXTRA_ACTIONS = EXTRA_TOOLS.map(t => t.name);
export const LEGACY_WRITES = [
  "clear_shopping_list", "clear_kitchen_inventory", "delete_recent_discard", "clear_recent_discards",
  "log_dish_from_voice", "log_dish_ingredients", "append_to_recent_dish", "update_recent_dish",
  "mark_dish_consumed", "delete_dish_log", "add_dish_ingredients_to_shopping_list",
  "add_recipe_ingredients_to_shopping_list", "add_saved_recipe_ingredients_to_shopping_list",
  "create_recipe_category", "move_recipe_to_category", "add_recipe_to_meal_calendar", "add_many_to_meal_calendar",
  "save_recipe_from_tiktok", "refresh_meal_plan",
];
export function catalog(raw, actions) {
  return [...raw, ...EXTRA_TOOLS].filter(t => actions.includes(t.name) || READ_TOOLS[t.name]).map(input => {
    const t=structuredClone(input);
    const idField = ["delete_recent_discard","remove_from_shopping_list","mark_shopping_item_bought","mark_shopping_item_unbought","update_shopping_item_store"].includes(t.name)?"item_id":
      ["remove_saved_recipe","get_saved_recipe_detail","add_saved_recipe_ingredients_to_shopping_list"].includes(t.name)?"recipe_id":null;
    if(idField) {
      const label=idField==="recipe_id"?"recipe_title":"item_name";
      t.parameters.properties[idField]=str;
      t.parameters.required=(t.parameters.required||[]).filter(k=>k!==label);
      t.parameters.anyOf=[{required:[idField]},{required:[label]}];
    }
    if(["get_recent_dishes","get_recent_discards","get_saved_recipes"].includes(t.name)) {
      Object.assign(t.parameters.properties,{query:str,limit:{type:"integer",minimum:1,maximum:100},offset:{type:"integer",minimum:0,maximum:100000}});
      t.description+=" Returns a page with count and next_offset; use offset to read remaining results. Use query to find a named entry.";
    }
    if(t.name==="update_many_shopping_item_stores") {
      const item=t.parameters.properties.items.items;item.properties.item_id=str;
      item.required=(item.required||[]).filter(k=>k!=="item_name");item.anyOf=[{required:["item_id"]},{required:["item_name"]}];
    }
    if(["add_generated_recipes_to_meal_calendar","add_many_to_meal_calendar"].includes(t.name))
      t.description="Prepare full dated recipes for the app meal calendar. Include ingredients, instructions and servings in notes. The review card is the confirmation; do not ask for an additional chat confirmation. Does not save a library copy or log a dish. The actual write only runs after Approve change.";
    return { ...t,
    name: READ_TOOLS[t.name] ? t.name : "request_" + t.name,
    description: READ_TOOLS[t.name] ? t.description : "Prepare exact changes for review; DOES NOT apply yet. " + t.description,
    defer_loading: true,
  };});
}
