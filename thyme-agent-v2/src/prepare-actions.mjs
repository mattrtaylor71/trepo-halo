import { Fault, hash } from "./core.mjs";
import { collection } from "./gateway.mjs";
import { dishPatch } from "./app-writes.mjs";
const norm=v=>String(v||"").trim().toLowerCase();
export const rowID=r=>String(r.id || r._id || r.shopping_id || r.entry_id || "");
export const storeName=value=>String(value||"").trim()||"Groceries";
export function exact(rows, id, label, nameKey) {
  if (!id && !norm(label)) throw new Fault("ambiguous_item","Choose the specific item to use.",409);
  const matches=rows.filter(r=>id ? rowID(r)===String(id) : norm(r[nameKey] || r.item_name || r.title || r.dish_name)===norm(label));
  if(matches.length!==1) throw new Fault("ambiguous_item","Read the current items and choose one exact item. Ask the user if more than one matches.",409);
  return matches[0];
}
const date=v=> {
  if(!/^\d{4}-\d{2}-\d{2}$/.test(v||"") || !Number.isFinite(new Date(v).getTime()) || new Date(v).toISOString().slice(0,10)!==v)
    throw new Fault("date","Choose an actual date in YYYY-MM-DD format.");
};
export async function prepareAction(gateway,a,name,args,before) {
  const rows=collection(before);
  if(name==="create_shopping_list") {
    args.items=args.items.map(x=>({...x,store:args.store}));
  } else if(["rename_shopping_list","remove_shopping_list","clear_shopping_list"].includes(name)) {
    const targets=name==="clear_shopping_list"?rows:rows.filter(r=>norm(storeName(r.store))===norm(storeName(args.store)));
    if(!targets.length) throw new Fault("missing","That list has no current items.",404);
    args.targets=targets.map(r=>({item_id:rowID(r),item_name:r.item_name,...(name==="rename_shopping_list"?{store:args.new_name}:{})}));
  } else if(["update_shopping_item","update_shopping_item_store","update_many_shopping_item_stores","mark_shopping_item_bought","mark_shopping_item_unbought","remove_from_shopping_list"].includes(name)) {
    for(const target of args.items || [args]) {
      const row=exact(rows,target.item_id,target.item_name);
      target.item_id=rowID(row);target.item_name=row.item_name;
    }
  } else if(["delete_recent_discard","clear_recent_discards"].includes(name)) {
    const targets=name==="clear_recent_discards"?rows:[exact(rows,args.item_id,args.item_name)];
    if(!targets.length) throw new Fault("nothing_to_remove","Discard history is already empty.");
    args.targets=targets.map(r=>({item_id:rowID(r),item_name:r.item_name}));
    if(name==="delete_recent_discard") args.item_id=rowID(targets[0]);
  } else if(["edit_saved_recipe","remove_saved_recipe"].includes(name)) {
    const row=exact(rows,args.recipe_id,args.recipe_title);
    args.recipe_id=rowID(row);
    if(name==="edit_saved_recipe" && !["title","ingredients","instructions","notes"].some(k=>args[k]!==undefined)) throw new Fault("empty_edit","Specify a recipe detail to change.");
  } else if(["append_to_recent_dish","update_recent_dish","update_dish","mark_dish_consumed","delete_dish_log"].includes(name)) {
    let row;
    if(["append_to_recent_dish","update_recent_dish"].includes(name)) {
      const candidates=rows.filter(r=>Date.now()-new Date(r.updated_at||r.created_at).getTime()<90*60000);
      if(candidates.length!==1) throw new Fault("ambiguous_item","Choose a specific dish from the Dish Log before editing it.",409);
      row=candidates[0];
    } else row=exact(rows,args.dish_id,args.dish_name,"dish_name");
    args.dish_id=rowID(row);
    if(name==="append_to_recent_dish") args.ingredients=[...(row.ingredients||[]),...args.ingredients];
    else if(args.add_ingredients || args.remove_ingredients) {
      if(args.ingredients) throw new Fault("ingredients","Choose a replacement list or specific additions/removals.");
      const remove=args.remove_ingredients||[];
      for(const value of remove) if(!(row.ingredients||[]).some(x=>norm(x)===norm(value))) throw new Fault("missing","An ingredient to remove was not found.");
      args.ingredients=(row.ingredients||[]).filter(x=>!remove.some(v=>norm(v)===norm(x))).concat(args.add_ingredients||[]);
      delete args.add_ingredients;delete args.remove_ingredients;
    }
    if(!["mark_dish_consumed","delete_dish_log"].includes(name) && !Object.keys(dishPatch(args)).length) throw new Fault("empty_edit","Specify a dish detail to change.");
  } else if(name.startsWith("log_dish_")) {
    if(!args.dish_name) args.dish_name=args.ingredients?.slice(0,3).join(", ");
    if(!args.dish_name) throw new Fault("dish_name","Name the dish to log.");
  } else if(["add_recipe_ingredients_to_shopping_list","add_saved_recipe_ingredients_to_shopping_list","add_dish_ingredients_to_shopping_list"].includes(name)) {
    let source;
    if(name.includes("dish")) source=exact(collection(await gateway.read(a,"dishes")),args.dish_id,args.dish_name,"dish_name");
    else if(name.includes("saved")) source=exact(collection(await gateway.read(a,"saved_recipes")),args.recipe_id,args.recipe_title);
    else {
      const data=await gateway.read(a,"suggestions");
      const candidates=[...(data.kitchen_only||[]),...(data.need_grocery||[]),...collection(data)];
      source=exact(candidates,null,args.recipe_title);
    }
    const missingOnly=name==="add_recipe_ingredients_to_shopping_list" && args.only_missing!==false;
    const ingredients=args.ingredients || (missingOnly ? source.missing_ingredients : source.ingredients);
    if(missingOnly && Array.isArray(ingredients) && !ingredients.length) throw new Fault("nothing_to_add","All required ingredients are already available; nothing needs adding.");
    if(!Array.isArray(ingredients) || !ingredients.length || ingredients.some(i=>typeof i!=="string")) throw new Fault("ingredients","Read and choose the exact ingredients to add.");
    args.items=ingredients.map(item_name=>({item_name,...(args.store?{store:args.store}:{})}));
    args.source_snapshot=hash(source);
  } else if(["add_recipe_to_meal_calendar","add_many_to_meal_calendar"].includes(name)) {
    const recipes=collection(await gateway.read(a,"saved_recipes"));
    args.entries=(args.entries||[args]).map(entry=> {
      const row=exact(recipes,entry.recipe_id,entry.recipe_name);
      return {plan_date:entry.plan_date,meal_slot:entry.meal_slot,title:row.title,ingredients:row.ingredients,
        instructions:row.instructions||row.steps||[],notes:row.notes||[]};
    });
  } else if(["move_meal_calendar_entry","remove_meal_calendar_entry"].includes(name)) {
    const filtered=rows.filter(r=>(!args.title||norm(r.title)===norm(args.title)) && (!(args.plan_date||args.from_date)||r.plan_date===(args.plan_date||args.from_date)) && (!(args.meal_slot||args.from_meal_slot)||r.meal_slot===(args.meal_slot||args.from_meal_slot)));
    const row=args.entry_id?exact(rows,args.entry_id):filtered.length===1?filtered[0]:null;
    if(!row) throw new Fault("ambiguous_item","Choose the exact meal calendar entry to change.",409);
    args.entry_id=rowID(row); args.title=row.title;
  } else if(name==="move_recipe_to_category") {
    const recipes=collection(await gateway.read(a,"saved_recipes"));
    const selected=args.recipe_names || [args.recipe_name];
    args.recipe_ids=selected.map(label=>rowID(exact(recipes,args.recipe_id,label)));
  }
  if(args.entries && name.includes("calendar")) for(const entry of args.entries) {
    date(entry.plan_date);
    if(!["breakfast","lunch","dinner","snack"].includes(entry.meal_slot)) throw new Fault("meal_slot","Choose breakfast, lunch, dinner, or snack.");
    if(!entry.ingredients?.length || !(entry.instructions||[]).length) throw new Fault("recipe_incomplete","A planned recipe needs ingredients and instructions.");
  }
  if(args.new_date) date(args.new_date);
  if(name==="clear_kitchen_inventory" && rows.length>100) throw new Fault("batch_size","Choose up to 100 specific kitchen items for each reviewed removal.");
  if(/^(update_item_|mark_item_)/.test(name) && args.remaining_quantity!==undefined && args.quantity_value===undefined && args.fill_percent===undefined)
    throw new Fault("amount","Ask for a numerical amount and unit before preparing this change.");
  if(/^(update_item_|mark_item_)/.test(name) && (args.quantity_value!==undefined || args.fill_percent!==undefined))
    delete args.remaining_quantity; // The app derives this display label from the approved amount.
  if(name==="move_recipe_to_category" || name==="create_recipe_category") {
    args.category_name=args.category_name?.trim();
    if(!args.category_name || args.category_name.length>64) throw new Fault("category_name","Choose a category name of 1 to 64 characters.");
  }
  if(name==="save_recipe_from_tiktok") {
    let u; try{u=new URL(args.url);}catch{throw new Fault("url","Provide a valid public recipe URL.");}
    // Use the existing social importer, not an arbitrary network tool.
    if(u.protocol!=="https:" || u.username || u.password || !/(^|\.)(tiktok\.com|instagram\.com|youtube\.com|youtu\.be)$/.test(u.hostname))
      throw new Fault("url","Use a supported TikTok, Instagram, or YouTube recipe link.");
  }
  return args;
}

// Keep date-scoped approvals/readback identical, including plans outside the default window.
export function proposalReadArgs(name,args) {
  if(!name.includes("calendar")) return {};
  if(args.entry_id) return {entry_id:args.entry_id};
  const values=[...(args.entries||[]).map(e=>e.plan_date),args.plan_date,args.from_date,args.new_date].filter(Boolean);
  values.forEach(date);
  if(!values.length) return {};
  const ordered=values.sort();
  if(new Date(ordered.at(-1))-new Date(ordered[0])>59*86400000) throw new Fault("date_range","Split plans spanning more than 60 days into separate requests.");
  return {start_date:ordered[0],end_date:ordered.at(-1)};
}
export function proposalSnapshot(name,args,data) {
  const rows=collection(data);
  // Only existing selected rows need a revision fence. Unrelated library or log
  // additions must not invalidate another reviewed save in the same conversation.
  if(name.startsWith("log_dish_")) return [];
  if(["edit_saved_recipe","remove_saved_recipe","update_dish","append_to_recent_dish","update_recent_dish","mark_dish_consumed","delete_dish_log"].includes(name))
    return rows.filter(r=>rowID(r)===String(args.recipe_id||args.dish_id));
  if(name==="save_generated_recipe") return rows.filter(r=>norm(r.title)===norm(args.title));
  return data;
}
export function pageData(data,args={}) {
  const rows=collection(data),query=norm(args.query);
  const filtered=query?rows.filter(r=>norm(r.item_name||r.dish_name||r.title).includes(query)):rows;
  const offset=Math.max(0,Math.floor(args.offset||0)),limit=Math.max(1,Math.min(100,Math.floor(args.limit||25)));
  if(!rows.length && !Array.isArray(data)) return data;
  return {items:filtered.slice(offset,offset+limit),count:filtered.length,next_offset:offset+limit<filtered.length?offset+limit:null};
}
