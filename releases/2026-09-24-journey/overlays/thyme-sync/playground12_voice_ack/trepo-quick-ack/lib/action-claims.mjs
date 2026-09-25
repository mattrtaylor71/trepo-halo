// Current-turn action claims only. Narration is evidence to validate, never
// authorization to run a corrective write. Keep this pure for replay tests.
export const ACTION_CLAIM_RULES = [
  {
    domain: "shopping_aisle", expectedTool: "set_shopping_item_aisle",
    okTools: new Set(["set_shopping_item_aisle"]),
    failText: "I couldn’t confirm that aisle change. Check your list before trying again.",
    patterns: [/\b(?:moved|sorted|organized|organised|assigned|grouped|categorized|categorised)\b[^.?!\n]{0,100}\baisles?\b/i],
  },
  {
    domain: "shopping_remove", expectedTool: "remove_from_shopping_list",
    okTools: new Set(["remove_from_shopping_list", "clear_shopping_list"]),
    failText: "I couldn't confirm that removal from your shopping list.",
    patterns: [/\b(?:removed|deleted|cleared)\b[^.?!\n]{0,80}\b(?:shopping|grocery|list|cart)\b/i],
  },
  {
    domain: "recipe_remove", expectedTool: "remove_saved_recipe",
    okTools: new Set(["remove_saved_recipe"]),
    failText: "I couldn't confirm that recipe was removed.",
    patterns: [/\b(?:removed|deleted)\b[^.?!\n]{0,80}\b(?:saved\s+)?recipes?\b/i],
  },
  {
    domain: "kitchen_update", expectedTool: "update_item_quantity",
    okTools: new Set(["update_item_quantity", "mark_item_opened", "update_item_expiration", "update_item_location", "update_item_details"]),
    failText: "I couldn't confirm that kitchen update.",
    patterns: [/\b(?:updated|changed|reduced|adjusted)\b[^.?!\n]{0,80}\b(?:quantity|amount|remaining|kitchen|pantry|fridge|freezer|expiration|expiry)\b/i,
      /\brenamed\b[^.?!\n]{0,120}\bto\b/i],
  },
  {
    domain: "recipe_save",
    expectedTool: "save_generated_recipe",
    okTools: new Set(["save_generated_recipe", "save_recipe_from_tiktok"]),
    failText: "I couldn't confirm that recipe was saved. Check your saved recipes before trying again.",
    patterns: [/\bsaved\b[^.?!\n]{0,80}\brecipe\b/i, /\brecipe\b[^.?!\n]{0,60}\b(?:saved|added)\b/i],
  },
  {
    // Meal-calendar confirmations. Placed FIRST so a "scheduled/added ... to your
    // calendar" reply resolves to THIS domain (satisfied by a calendar write) and
    // never falls through to the dishes rule (which would demand a dish-log tool
    // and flag a real calendar action as a false confirmation). Planning a meal is
    // NOT eating one. Patterns are PAST-TENSE only, so capability answers ("I can
    // add things to your calendar") never match.
    domain: "meal_calendar",
    expectedTool: "add_recipe_to_meal_calendar",
    okTools: new Set([
      "add_recipe_to_meal_calendar",
      "add_many_to_meal_calendar",
      "add_generated_recipes_to_meal_calendar",
      "move_meal_calendar_entry",
      "remove_meal_calendar_entry"
    ]),
    failText: "I couldn't update your meal calendar just now — please try again, or check the recipe title and date.",
    patterns: [
      /\b(added|scheduled|planned|booked|slotted|penciled|pencilled)\b[^.?!\n]{0,60}\b(calendar|meal\s+plan|meal\s+calendar)\b/i,
      /\b(added|scheduled|planned|booked|slotted|put)\b[^.?!\n]{0,60}\b(?:for|to)\s+(?:your\s+)?(?:breakfast|lunch|dinner|snack)\b\s+(?:on\s+|for\s+)?(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|next|this|\d{4}-\d{2}-\d{2})/i,
      /\b(moved|rescheduled|shifted|bumped)\b[^.?!\n]{0,60}\b(?:to|on)\s+(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2}|next|this|tomorrow)\b/i,
      /\b(removed|took\s+off|taken\s+off|cleared|deleted|unscheduled)\b[^.?!\n]{0,60}\b(?:from\s+)?(?:your\s+)?(calendar|meal\s+plan|meal\s+calendar)\b/i,
    ],
  },
  {
    domain: "dishes",
    expectedTool: "log_dish_ingredients",
    okTools: new Set(["log_dish_ingredients", "log_dish_from_voice", "update_recent_dish", "append_to_recent_dish", "delete_dish_log", "mark_dish_consumed"]),
    failText: "I wasn't able to log that just now — please try again.",
    patterns: [
      /\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack)\b[^.?!\n]{0,40}\blogged\b/i,
      /\blogged\b[^.?!\n]{0,40}\b(meal|dish|food|breakfast|lunch|dinner|brunch|snack|it|that|your)\b/i,
      /\b(has|have|had|is|was|were|been)\s+logged\b/i,
      /\badded\b[^.?!\n]{0,30}\bto\s+your\s+(?:dish|meal)\s+log\b/i,
      // Sentence-initial completed-action confirmations that name the food
      // DIRECTLY ("Logged: two waffles") — the shape that slipped Brad's 07-09
      // turn (the meal-word patterns above require dish/meal/food/breakfast/etc.,
      // which "Eggo waffles"/"espresso" are not). Excludes "logged out/in/up" and
      // instructional/offer forms ("you can log a dish", "want me to log…") which
      // use present-tense "log" — so it coexists with the App Guide how-to text.
      /^\s*(?:(?:ok(?:ay)?|done|great|sure|perfect|got\s*it|all\s*set|no\s*problem|there|alright)[\s,!.—-]+)*(?:i(?:'ve|\s+have|\s+just)?\s+)?(logged|added|saved|recorded|noted)\b(?!\s+(?:out|in|up|off)\b)(?:\s*[:\-—]|\s+(?:your|the|a|an|two|one|both|it\b|that\b|\d))/i,
      /\bi(?:'ve|\s+have|\s+just)?\s+logged\b(?!\s+(?:out|in)\b)/i,
    ],
  },
  {
    domain: "shopping_add",
    expectedTool: "add_to_shopping_list",
    okTools: new Set(["add_to_shopping_list", "add_many_to_shopping_list", "add_dish_ingredients_to_shopping_list", "add_recipe_ingredients_to_shopping_list", "add_saved_recipe_ingredients_to_shopping_list"]),
    failText: "I wasn't able to add that to your shopping list just now — please try again.",
    patterns: [
      /\badded\b[^.?!\n]{0,40}\bto\s+your\s+(?:shopping\s+)?(?:list|cart)\b/i,
      /\bput\b[^.?!\n]{0,30}\bon\s+your\s+(?:shopping\s+)?list\b/i,
    ],
  },
  {
    domain: "kitchen_add",
    expectedTool: "check_in_item",
    okTools: new Set(["check_in_item", "check_in_many_items"]),
    failText: "I wasn't able to add that to your kitchen just now — please try again.",
    patterns: [
      /\badded\b[^.?!\n]{0,40}\bto\s+your\s+(kitchen|pantry|fridge)\b/i,
      // "checked in" AND "checked into" (into is one token, so \bin\b fails there).
      /\bchecked\s+in(?:to)?\b[^.?!\n]{0,50}\b(kitchen|pantry|fridge)\b/i,
      /\bchecked[\s-]*in\b[^.?!\n]{0,30}\b(kitchen|pantry|fridge|item|it|that)\b/i,
    ],
  },
  {
    // Removal claims were UNGUARDED — a "removed/discarded it" reply with no
    // discard_item call (e.g. a name-match miss that threw not-found) sailed
    // through as a green confirmation. That is why 0/3 of Zach's week-long
    // removal requests landed while Thyme confirmed every one.
    domain: "kitchen_remove",
    expectedTool: "discard_item",
    okTools: new Set(["discard_item", "clear_kitchen_inventory"]),
    failText: "I couldn't find that to remove — it may not be in your kitchen, or try naming it the way it's saved.",
    // The negative lookahead keeps SHOPPING-list removals ("removed X from your
    // shopping list") from matching the kitchen-remove rule — that domain uses a
    // different tool (remove_from_shopping_list) and has no rule of its own here.
    patterns: [
      /\b(removed|discarded|tossed|took\s+out|taken\s+out|threw\s+(?:it\s+)?(?:out|away)|thrown\s+(?:it\s+)?(?:out|away)|got(?:ten)?\s+rid\s+of|used\s+up)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,45}\b(kitchen|pantry|fridge|freezer|inventory|stock)\b/i,
      /\bi(?:'ve| have)\s+(removed|discarded|tossed|taken\s+out|thrown\s+(?:it\s+)?(?:out|away)|gotten\s+rid\s+of)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)/i,
      /\b(removed|discarded|tossed|took\s+out|threw\s+(?:it\s+)?(?:out|away)|got\s+rid\s+of)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,30}\b(it|that|them|those|the|your)\b/i,
      /\b(has|have|been)\s+(removed|discarded|taken\s+out|tossed)\b(?![^.?!\n]*\b(?:list|shopping|cart|grocer)\b)[^.?!\n]{0,30}\b(kitchen|pantry|fridge|from|it|that)\b/i,
    ],
  },
];

const COMPLETED = /\b(?:sorted|organized|organised|assigned|grouped|categorized|categorised|added|saved|logged|recorded|noted|put|checked|removed|discarded|tossed|taken|took|threw|thrown|got|gotten|used|scheduled|planned|booked|slotted|penciled|pencilled|moved|rescheduled|shifted|bumped|cleared|deleted|unscheduled|updated|changed|reduced|adjusted|renamed)\b/i;

function claimClauses(text) {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^\s*>.*$/gm, " ")
    .replace(/"[^"\n]*"|“[^”\n]*”|`[^`\n]*`/g, " ")
    .replace(/\*\*|__/g, "")
    .split(/(?:[.!?\n]+|;|\bbut\b|\band\s+(?=(?:I(?:'ve)?\s+)?(?:added|saved|logged|removed|discarded|checked|scheduled)\b))/i)
    .map(s => s.trim()).filter(Boolean);
}

function isCurrentAction(clause) {
  const match = COMPLETED.exec(clause);
  if (!match) return false;
  const prefix = clause.slice(0, match.index);
  // "This recipe has no added sugar" describes ingredients, not an app save.
  // Limit this to an adjacent negation so "No problem, I added milk" remains
  // guarded; clause splitting keeps a separate positive confirmation visible.
  if (/\b(?:no|without)\s+(?:any\s+)?$/i.test(prefix)) return false;
  if (/\b(?:not|never|haven't|hasn't|hadn't|wasn't|weren't|didn't|couldn't|cannot|can't)\b/i.test(prefix)) return false;
  if (/^(?:if|once|when|whenever|after|before|unless|items?\s+(?:that\s+)?you|you\b|the phrase\b|for example\b)/i.test(clause)) return false;
  if (/\b(?:can|could|would|should|will|may|might|want to|able to)\b/i.test(prefix)) return false;
  // Calendar dates describe the action's target, not when the write happened.
  // Past narration has explicit history cues or a past auxiliary plus a date.
  if (/\b(?:yesterday|last\s+(?:week|month|year|night)|previously|earlier|according to (?:your|the) history)\b/i.test(clause)) return false;
  if (/\b(?:was|were|had been)\b/i.test(prefix) && /\b(?:on\s+(?:mon|tue|wed|thu|fri|sat|sun)\w*|ago|\d{4}-\d{2}-\d{2})\b/i.test(clause)) return false;
  return true;
}

export function detectActionClaims(text) {
  const claims = [];
  for (const clause of claimClauses(text)) {
    if (!isCurrentAction(clause)) continue;
    // Explicit destination beats the generic sentence-initial dish fallback.
    const ordered = [...ACTION_CLAIM_RULES.filter(r => r.domain !== "dishes"), ...ACTION_CLAIM_RULES.filter(r => r.domain === "dishes")];
    const rule = ordered.find(r => r.patterns.some(pattern => pattern.test(clause)));
    if (rule) claims.push({ ...rule, clause });
  }
  return claims;
}

export function detectActionClaim(text) {
  return detectActionClaims(text)[0] || null;
}
