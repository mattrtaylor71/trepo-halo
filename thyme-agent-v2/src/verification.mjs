import { collection } from "./gateway.mjs";
import { hash } from "./core.mjs";
import { dishPatch } from "./app-writes.mjs";
const normalized = (x) =>
  String(x ?? "")
    .trim()
    .toLowerCase();
const name = (x) =>
  normalized(x.item_name || x.product_name || x.title || x.name || x.dish_name);
const identity = (x) =>
  String(
    x?.id ||
      x?._id ||
      x?.shopping_id ||
      x?.household_item_uuid ||
      x?.entry_id ||
      "",
  );
const matches = (items, ref) =>
  items.filter((x) =>
    ref.item_id || ref.dish_id || ref.entry_id || ref.recipe_id
      ? identity(x) === String(ref.item_id || ref.dish_id || ref.entry_id || ref.recipe_id)
      : name(x) ===
        normalized(
          ref.item_name ||
            ref.title ||
            ref.recipe_name ||
            ref.recipe_title ||
            ref.dish_name,
        ),
  );
export function verifyChange(action, args, before, after, result) {
  if (action === "create_shopping_list" || action.includes("ingredients_to_shopping_list"))
    return verifyChange("add_many_to_shopping_list",{items:args.items},before,after,result);
  if (["add_recipe_to_meal_calendar","add_many_to_meal_calendar"].includes(action))
    return verifyChange("add_generated_recipes_to_meal_calendar",{entries:args.entries},before,after,result);
  const b = collection(before),
    a = collection(after),
    targets = args.items || [args];
  let verified = false;
  if (action === "edit_saved_recipe") {
    const row=matches(a,args)[0];
    const fields=["title","ingredients","instructions","notes"].filter(k=>args[k]!==undefined);
    const old=matches(b,args)[0];
    verified=!!row && !!old && fields.length>0 &&
      ["title","ingredients","instructions","notes"].every(k=>hash(row[k] ?? [])===hash(args[k] ?? old[k] ?? []));
  } else if (action.startsWith("log_dish_")) {
    const id=result?.toolResult?.dish?.id;
    const row=a.find(x=>identity(x)===id);
    verified=!!id && !!row && !b.some(x=>identity(x)===id) && Object.entries(dishPatch(args)).every(([k,v])=>hash(row[k])===hash(v));
  } else if (["update_dish","append_to_recent_dish","update_recent_dish"].includes(action)) {
    const row=matches(a,args)[0],old=matches(b,args)[0],patch=dishPatch(args);
    verified=!!row && !!old && Object.keys(patch).length>0 &&
      ["dish_name","serving_size","ingredients","calories","protein","total_carbohydrates","total_fat","action"].every(k=>hash(row[k] ?? null)===hash(patch[k] ?? old[k] ?? null));
  } else if (action === "mark_dish_consumed") {
    const row=matches(a,args)[0]; verified=!!row && matches(b,args).length===1 && row.action==="OUT";
  } else if (["remove_shopping_list","rename_shopping_list"].includes(action)) {
    verified=args.targets?.length>0 && args.targets.every(t=>matches(b,t).length===1 && (action==="remove_shopping_list"?matches(a,t).length===0:matches(a,t).some(r=>r.store===t.store)));
    const ids=new Set((args.targets||[]).map(t=>t.item_id));
    verified=verified && b.filter(x=>!ids.has(identity(x))).every(x=>a.some(y=>hash(x)===hash(y)));
  } else if (action === "update_shopping_item") {
    const row=matches(a,args)[0];
    verified=!!row && ["new_name","quantity","store"].filter(k=>args[k]!==undefined).every(k=>row[k==="new_name"?"item_name":k]===args[k]);
  } else if (action === "create_recipe_category") {
    verified=(after.categories||[]).some(x=>normalized(x.name)===normalized(args.category_name));
  } else if (action === "move_recipe_to_category") {
    const category=(after.categories||[]).find(x=>normalized(x.name)===normalized(args.category_name));
    verified=!!category && args.recipe_ids?.length>0 && args.recipe_ids.every(id=>(after.assignments?.[id]||[]).includes(category.id) &&
      (before.assignments?.[id]||[]).every(old=>(after.assignments?.[id]||[]).includes(old)));
  } else if (action === "save_recipe_from_tiktok") {
    const id=result?.toolResult?.recipe?.id;
    verified=!!id && a.some(x=>identity(x)===id && x.status==="ready" && x.ingredients?.length>0 && (x.instructions||x.steps)?.length>0);
  } else if (action === "refresh_meal_plan") {
    verified=!!result?.toolResult?.async_update && after?.status==="ready" && !!after.plan &&
      new Date(after._updatedDate)>new Date(before._updatedDate||0) && new Date(after._updatedDate)>=new Date(result.requestedAt) &&
      (!args.focus || normalized(after.focus)===normalized(args.focus));
  } else if (action === "delete_recent_discard") {
    verified=matches(b,args).length===1 && matches(a,args).length===0;
  } else if (action === "clear_recent_discards") verified=args.targets?.length>0 && args.targets.every(t=>matches(b,t).length===1 && matches(a,t).length===0);
  else if (action === "update_food_memory")
    verified =
      after.memory?.revision > before.memory?.revision &&
      hash(after.memory?.explicit_preferences?.[args.key]?.value) ===
        hash(args.key === "max_prep_minutes" ? args.minutes : args.values);
  else if (["clear_shopping_list", "clear_kitchen_inventory"].includes(action))
    verified = a.length === 0;
  else if (
    [
      "add_to_shopping_list",
      "add_many_to_shopping_list",
      "check_in_item",
      "check_in_many_items",
    ].includes(action)
  ) {
    const used = new Set();
    verified = targets.every((t) => {
      const row = matches(a, t).find(
        (x) =>
          identity(x) &&
          !used.has(identity(x)) &&
          !b.some((y) => identity(y) === identity(x)) &&
          [
            "quantity",
            "quantity_value",
            "quantity_unit",
            "store",
            "brand",
            "location",
            "expiration_date",
            "is_opened",
            "fill_percent",
            "category",
          ].every((k) => t[k] === undefined || fieldMatches(x, k, t[k])),
      );
      if (row) used.add(identity(row));
      return !!row;
    });
  } else if (
    [
      "remove_from_shopping_list",
      "discard_item",
      "remove_saved_recipe",
      "delete_dish_log",
    ].includes(action)
  )
    verified = matches(b, args).length > 0 && matches(a, args).length === 0;
  else if (action === "mark_shopping_item_bought")
    verified =
      matches(a, args).length === 1 &&
      ["2", "in", "bought", "checked"].includes(
        normalized(matches(a, args)[0].action),
      );
  else if (action === "mark_shopping_item_unbought")
    verified =
      matches(a, args).length === 1 &&
      ["1", "out", "unbought", "unchecked", "added"].includes(
        normalized(matches(a, args)[0].action),
      );
  else if (
    ["update_shopping_item_store", "update_many_shopping_item_stores"].includes(
      action,
    )
  )
    verified = targets.every((t) =>
      matches(a, t).some((x) => normalized(x.store) === normalized(t.store)),
    );
  else if (action === "save_generated_recipe")
    verified = a.some(
      (x) =>
        name(x) === normalized(args.title) &&
        hash(x.ingredients) === hash(args.ingredients) &&
        hash(x.instructions || x.steps) === hash(args.steps) &&
        (args.notes === undefined || hash(x.notes) === hash(args.notes)),
    );
  else if (
    [
      "mark_item_opened",
      "update_item_quantity",
      "update_item_location",
      "update_item_expiration",
      "update_item_details",
    ].includes(action)
  ) {
    const old = matches(b, args);
    const current =
      old.length === 1 ? a.find((x) => identity(x) === identity(old[0])) : null;
    if (!current && args.quantity_value===0) {
      verified=old.length===1 && result?.toolResult?.items?.some(x=>x.removed===true && String(x.item?._id)===args.item_id);
    }
    if (current) {
      const expected = {
        ...args,
        ...(action === "mark_item_opened" ? { is_opened: true } : {}),
      };
      const map = { new_name: "item_name", location: "storage_location" };
      const fields = [
        "new_name",
        "quantity_value",
        "quantity_unit",
        "remaining_quantity",
        "fill_percent",
        "location",
        "is_opened",
        "brand",
        "category",
        "expiration_date",
      ].filter((k) => expected[k] !== undefined);
      verified =
        fields.length > 0 &&
        fields.every((k) => {
          const v = current[map[k] || k];
          return k === "is_opened"
            ? Boolean(v) === expected[k]
            : k === "expiration_date"
              ? String(v || "").slice(0, 10) ===
                String(expected[k]).slice(0, 10)
              : normalized(v) === normalized(expected[k]);
        });
    }
  } else if (action === "remove_meal_calendar_entry")
    verified =
      !!result?.toolResult?.removed_id &&
      b.some((x) => identity(x) === String(result.toolResult.removed_id)) &&
      (!args.entry_id ||
        String(result.toolResult.removed_id) === args.entry_id) &&
      !a.some((x) => identity(x) === String(result.toolResult.removed_id));
  else if (action === "move_meal_calendar_entry") {
    const entry = result?.toolResult?.entry;
    verified =
      !!entry &&
      a.some(
        (x) =>
          identity(x) === identity(entry) &&
          (!args.entry_id || identity(x) === args.entry_id) &&
          x.plan_date === args.new_date &&
          (!args.new_meal_slot || x.meal_slot === args.new_meal_slot),
      );
  } else if (action === "add_generated_recipes_to_meal_calendar") {
    const used = new Set();
    verified =
      args.entries?.length > 0 &&
      args.entries.every((t) => {
        const row = a.find(
          (x) =>
            identity(x) &&
            !used.has(identity(x)) &&
            !b.some((y) => identity(y) === identity(x)) &&
            x.title === t.title &&
            x.plan_date === t.plan_date &&
            x.meal_slot === t.meal_slot &&
            hash(x.ingredients) === hash(t.ingredients) &&
            hash(x.instructions) === hash(t.instructions || []) &&
            hash(x.notes || []) === hash(t.notes || []) &&
            (t.meal_category === undefined ||
              x.meal_category === t.meal_category),
        );
        if (row) used.add(identity(row));
        return !!row;
      });
  }
  // Unknown operations cannot earn a success badge from an unrelated timestamp change.
  return {
    verified,
    reason: verified
      ? "The requested change is present in current Trepo data."
      : "The requested change could not yet be verified from current data.",
  };
}

function fieldMatches(row, key, expected) {
  const map = { location: "storage_location" };
  const value =
    key === "quantity"
      ? (row.quantity ?? row.remaining_quantity)
      : row[map[key] || key];
  if (key === "is_opened") return [true, 1, "1"].includes(value) === expected;
  if (key === "expiration_date")
    return String(value || "").slice(0, 10) === String(expected).slice(0, 10);
  return normalized(value) === normalized(expected);
}
