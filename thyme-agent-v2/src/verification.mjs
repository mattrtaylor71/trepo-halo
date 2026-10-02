import { collection } from "./gateway.mjs";
import { hash } from "./core.mjs";
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
      x?.household_item_uuid ||
      x?.shopping_id ||
      x?.entry_id ||
      "",
  );
const matches = (items, ref) =>
  items.filter((x) =>
    ref.item_id || ref.dish_id || ref.entry_id
      ? identity(x) === String(ref.item_id || ref.dish_id || ref.entry_id)
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
  const b = collection(before),
    a = collection(after),
    targets = args.items || [args];
  let verified = false;
  if (action === "update_food_memory")
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
