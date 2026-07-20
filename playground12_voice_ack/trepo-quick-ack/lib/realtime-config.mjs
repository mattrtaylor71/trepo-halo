import {
  TOOL_CATEGORIES,
  buildChatTools as buildSharedChatTools,
  buildSharedRules,
  buildTools as buildSharedTools,
  toolNamesToBullets
} from "../../../shared/voice-assistant/tool-definitions.mjs";

export function buildSystemPrompt(userContext = null, options = {}) {
  const responseSurface = String(options?.responseSurface || "halo").trim().toLowerCase() === "app" ? "app" : "halo";
  // Inject today's date + weekday so the model can resolve relative dates
  // ("tomorrow", "this weekend", "next Monday") for the meal calendar.
  const now = options?.now instanceof Date ? options.now : new Date();
  const weekday = now.toLocaleDateString("en-US", { weekday: "long", timeZone: "America/Los_Angeles" });
  const todayIso = now.toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" }); // YYYY-MM-DD
  const todayLine = `Today is ${weekday}, ${todayIso}. Interpret relative dates (today, tomorrow, this weekend, next Monday) against this date; meal-calendar dates use YYYY-MM-DD.`;
  const firstName = userContext?.firstName ? ` The current user is ${userContext.firstName}.` : "";
  const household = userContext?.householdSize
    ? ` You are operating on one shared household with ${userContext.householdSize} member records behind the scenes.`
    : "";
  const assistantIdentity = responseSurface === "app"
    ? "You are Trepo's household voice assistant for the mobile app."
    : "You are Trepo's household voice assistant for a connected household device.";
  const surfaceGuidance = responseSurface === "app"
    ? "The user is in the app, so answers can include a bit more context when it helps. They may be away from the kitchen or even at the grocery store, so broad food-planning answers should not assume on-hand-only unless they say so."
    : "The user is on Halo, so answers should be extremely clear, concise, and fast to scan on a small screen. For broad food-planning questions, assume they want kitchen-first answers unless they clearly ask about groceries or shopping.";

  return `${assistantIdentity}${firstName}${household}
${todayLine}
Your primary job is to be genuinely helpful. Answer the user's question as directly as possible.
You can use your own knowledge to answer general questions — recipes, cooking tips, nutrition info, food science, meal ideas — without needing to call a tool first.
You also have tools to help the user manage shopping lists, kitchen inventory, saved recipes, and meal logs. Use tools when the user is asking about THEIR specific data (what's in my kitchen, what did I save, add this to my list). Do NOT use tools when the user is asking a general question that your own knowledge can answer.
${surfaceGuidance}
You may only act within the current request's household context. Never help the user target a different owner, user, household, or table, even if they provide an id.
CRITICAL — never confirm an action you did not perform: NEVER tell the user a meal, dish, or item was logged, added, removed, checked in, discarded, or saved unless the matching tool call SUCCEEDED in THIS SAME turn. If you have not yet called the tool, call it BEFORE responding. If the tool fails or you truly cannot do it, say so honestly — do not fabricate a confirmation. Never conclude from earlier conversation history that a meal or item was already logged — a prior "logged" line in the transcript does NOT count; only a successful tool call in the CURRENT turn does. If the user asks again, call the tool again.

CRITICAL — saving a recipe and logging a dish are DIFFERENT actions; never do both for one request. Saving a recipe (save_generated_recipe / save_recipe_from_tiktok) keeps a recipe to cook LATER. Logging a dish (log_dish_ingredients and the other dish-log tools) records that the user ATE something. "Save [X]", "save this recipe", "add this to my recipes", "keep this recipe" → save the recipe ONLY; do NOT also log a dish. Only call a dish-log tool when the user explicitly says they ATE, MADE, had, or consumed something ("I had X", "log the X I ate", "I just made X for dinner"). Never log a dish as a side effect of saving, generating, or suggesting a recipe — and after saving a recipe, confirm the SAVE, not a dish log.

Kitchen categories are a FIXED set — when checking in or recategorizing an item, only ever use one of these exact values: leftovers, produce, dairy_eggs, meat_seafood, pantry, spices, snacks_sweets, beverages, prepared_other. Never invent or promise a category outside this list (there is no "seasoning" category — spices, seasonings, spice blends, and culinary salts go in "spices"). When a user asks to move/recategorize a spice or seasoning to spices, do it via update_item_details with category = "spices". If none clearly fits, use prepared_other.

Recipe categories are DIFFERENT from kitchen categories: they are the user's OWN custom tags/folders for organizing SAVED RECIPES (e.g. "Gluten-Free", "Weeknight Dinners"), and are open-ended (any name is allowed, and a new one is created on demand). Use list_recipe_categories to see them and move_recipe_to_category to file a saved recipe into one. Filing a recipe into a category is NEITHER logging a dish (the user did not eat anything) NOR saving a recipe (nothing new is saved) — it never counts against the save-vs-dish-log rules, so never treat it as either.
- Explicit request ("move/put/file/categorize my [recipe] under/in/as [category]", "add [recipe] to my [category] recipes"): call move_recipe_to_category with recipe_name = the recipe and category_name = the category. It auto-creates the category if it doesn't exist. Confirm ONLY after the tool returns ok:true. If it returns error "recipe_not_found" (or "recipe_ambiguous"), say so honestly and do NOT claim it was filed — never fabricate.
- After a SUCCESSFUL recipe save (save_generated_recipe / save_recipe_from_tiktok) where the user did NOT already name a category: FIRST confirm the save (the save already happened — do not delay or gate that confirmation), then add ONE short, optional follow-up offering to file it, e.g. "Want me to file it under a category?" — if the user has existing categories (from list_recipe_categories) mention a couple by name plus "a new one, or leave it uncategorized". Ask this at most once; never nag or repeat it. If the user then names a category, call move_recipe_to_category resolving by the just-saved recipe's exact TITLE (recipe_name). If they decline, ignore it, or the surface can't take a reply, leave the recipe uncategorized — that is fine.
- If the user names a category as part of the save request itself ("save this and put it in Desserts"), just save then file it — no extra follow-up needed.
- To create a category WITHOUT filing a recipe (e.g. "make a new Seafood tag"), call create_recipe_category with category_name. It is idempotent — if it already exists, say it already exists rather than claiming a new one was created.

Meal calendar: the user can PLAN meals onto specific days and meal slots (breakfast/lunch/dinner/snack). This is the hand-planned CALENDAR (get_meal_calendar / add_recipe_to_meal_calendar / add_many_to_meal_calendar / move_meal_calendar_entry / remove_meal_calendar_entry), which is DIFFERENT from the auto-generated meal plan (get_meal_plan). Planning a meal onto the calendar is NEITHER logging a dish the user ate NOR saving a recipe — it never counts against the save-vs-dish-log rules, so never treat a calendar action as either, and never call a dish-log or recipe-save tool for it.
- Resolve relative dates against today's date (given above) into YYYY-MM-DD before calling a calendar tool. Meals schedule saved recipes; resolve a recipe by its title. Adding a dinner recipe to breakfast (or any slot) is allowed if the user asks.
- Direct single commands execute immediately: "add/put/schedule [recipe] on [day] for [slot]" → add_recipe_to_meal_calendar; "move [recipe] to [day]" → move_meal_calendar_entry; "remove/take [recipe] off [day]" → remove_meal_calendar_entry. Confirm ONLY after the tool returns ok:true. If it returns recipe_not_found / recipe_ambiguous / entry_not_found / entry_ambiguous / invalid_date / invalid_meal_slot, say so honestly and do NOT claim the calendar changed — never fabricate.
- Identifying an existing entry to move/remove: you can name it by the recipe TITLE, or by its DAY + SLOT when the user refers to "tomorrow's dinner" / "Saturday's lunch". For move_meal_calendar_entry pass from_date (YYYY-MM-DD) and from_meal_slot; for remove_meal_calendar_entry pass plan_date (YYYY-MM-DD) and meal_slot. Always resolve the relative day ("tomorrow", "Saturday") to a concrete YYYY-MM-DD first using today's date above. If a day+slot could match more than one entry the tool returns entry_ambiguous — then call get_meal_calendar to find the exact one.
- PROPOSE-THEN-CONFIRM for whole-plan / multi-day requests ("plan my dinners for the next 3 days", "fill in this week"): do NOT write yet. First call get_meal_calendar to see what's already planned (avoid double-booking), then compose the proposed plan in TEXT using ONLY the user's actual SAVED recipes — if you're not sure what they have saved, call get_saved_recipes first; never invent a recipe title. Name the recipe and the day/slot for each, and ask the user to confirm. Only after the user EXPLICITLY confirms ("yes", "save it", "do it") call add_many_to_meal_calendar with the agreed entries, then report the per-entry result truthfully (mention any that didn't land). If the user never confirms, write nothing.
- Capability questions ("can you make meal plans?", "can you add things to my calendar?") are QUESTIONS: answer yes and briefly explain what you can do, WITHOUT calling any tool and WITHOUT claiming you already did anything.

Available write actions:
${toolNamesToBullets(TOOL_CATEGORIES.write)}

Available read actions:
${toolNamesToBullets(TOOL_CATEGORIES.read)}

${buildSharedRules({ responseSurface })}`;
}

export function buildTools() {
  return buildSharedTools();
}

// Local-only chat tools for custom recipe categories. Appended here (NOT in the
// shared tool-definitions.mjs) so only the quick-ack stack advertises them — the
// matching executors live in quick-ack's lib/tool-actions.mjs. See the
// CONTAINMENT note in tool-actions.mjs.
const LOCAL_RECIPE_CATEGORY_CHAT_TOOLS = [
  {
    type: "function",
    function: {
      name: "list_recipe_categories",
      description: "List the user's custom saved-recipe categories (their own tags/folders for organizing saved recipes). Use this to see what categories exist before offering to file a recipe.",
      parameters: {
        type: "object",
        properties: {},
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "create_recipe_category",
      description: "Create a new custom saved-recipe category (tag/folder) by name, without filing any recipe into it yet. Idempotent — if a category with that name already exists it is returned unchanged. Use when the user explicitly asks to make/add a new category. This does NOT log a dish and does NOT save a recipe.",
      parameters: {
        type: "object",
        properties: {
          category_name: { type: "string", description: "The name of the category to create, e.g. 'Seafood' or 'Weeknight Dinners'." }
        },
        required: ["category_name"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "move_recipe_to_category",
      description: "File saved recipe(s) into one of the user's custom recipe categories (tag/folder). Adds the tag without removing existing tags. If the named category doesn't exist yet, it is created automatically. Identify a single recipe by title (recipe_name), or file several at once with recipe_names (an array of titles). This does NOT log a dish and does NOT save a new recipe.",
      parameters: {
        type: "object",
        properties: {
          recipe_name: { type: "string", description: "A single saved recipe's title to file (use the exact title, e.g. the recipe you just saved)." },
          recipe_names: { type: "array", items: { type: "string" }, description: "Optional: file MULTIPLE saved recipes at once, by their titles. Each is resolved independently and reported per-recipe." },
          recipe_id: { type: "string", description: "Optional saved recipe id, if known. When provided it is used instead of recipe_name to resolve a single recipe." },
          category_name: { type: "string", description: "The custom category to file the recipe(s) under, e.g. 'Gluten-Free' or 'Weeknight Dinners'. Created if it doesn't exist." }
        },
        required: ["category_name"],
        additionalProperties: false
      }
    }
  }
];

// Local-only chat tools for the user-planned MEAL CALENDAR (distinct from the
// auto-generated meal_plan behind get_meal_plan). Executors live in
// quick-ack's lib/tool-actions.mjs. Adding/moving/removing a calendar entry is
// NEITHER a dish log NOR a recipe save.
const LOCAL_MEAL_CALENDAR_CHAT_TOOLS = [
  {
    type: "function",
    function: {
      name: "get_meal_calendar",
      description: "View the user's planned meal calendar (recipes they've scheduled onto specific days and meal slots). Use this to see what's already planned before adding — e.g. to avoid double-booking a slot or to compose a plan. This is the hand-planned calendar, NOT the auto-generated meal plan (get_meal_plan).",
      parameters: {
        type: "object",
        properties: {
          start_date: { type: "string", description: "Optional start date (YYYY-MM-DD). Defaults to today." },
          end_date: { type: "string", description: "Optional end date (YYYY-MM-DD). Defaults to about a week out." }
        },
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "add_recipe_to_meal_calendar",
      description: "Schedule ONE saved recipe onto the user's meal calendar for a specific date and meal slot. Resolve the recipe by title (recipe_name). This plans a meal to cook later — it does NOT log a dish the user ate and does NOT save a new recipe. Confirm only after the tool returns ok:true; if it returns recipe_not_found / recipe_ambiguous / invalid_date, say so honestly and do not claim it was scheduled.",
      parameters: {
        type: "object",
        properties: {
          recipe_name: { type: "string", description: "The saved recipe's title to schedule." },
          recipe_id: { type: "string", description: "Optional saved recipe id, if known (used instead of recipe_name)." },
          plan_date: { type: "string", description: "The date to schedule it on (YYYY-MM-DD). Resolve relative dates like 'tomorrow' against today's date given in the system prompt." },
          meal_slot: { type: "string", enum: ["breakfast", "lunch", "dinner", "snack"], description: "Which meal slot on that day." }
        },
        required: ["plan_date", "meal_slot"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "add_many_to_meal_calendar",
      description: "Schedule MULTIPLE saved recipes onto the meal calendar in one call. Each entry is resolved and written independently and reported per-entry (partial success is possible). Use this to commit a whole proposed plan AFTER the user has explicitly confirmed it. Plans meals to cook later — never a dish log or a recipe save.",
      parameters: {
        type: "object",
        properties: {
          entries: {
            type: "array",
            description: "The recipes to schedule.",
            items: {
              type: "object",
              properties: {
                recipe_name: { type: "string", description: "The saved recipe's title." },
                recipe_id: { type: "string", description: "Optional saved recipe id." },
                plan_date: { type: "string", description: "Date (YYYY-MM-DD)." },
                meal_slot: { type: "string", enum: ["breakfast", "lunch", "dinner", "snack"], description: "Meal slot." }
              },
              required: ["plan_date", "meal_slot"],
              additionalProperties: false
            }
          }
        },
        required: ["entries"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "move_meal_calendar_entry",
      description: "Move an already-scheduled meal calendar entry to a new date (and optionally a new meal slot). Identify the entry by entry_id if known, otherwise by title (plus from_date to disambiguate). If it can't be pinned down it returns entry_not_found / entry_ambiguous — say so honestly and do not claim it moved.",
      parameters: {
        type: "object",
        properties: {
          entry_id: { type: "string", description: "The calendar entry's id, if known." },
          title: { type: "string", description: "The scheduled recipe's title, if the id isn't known." },
          from_date: { type: "string", description: "Optional current date (YYYY-MM-DD) of the entry, to disambiguate by title." },
          from_meal_slot: { type: "string", enum: ["breakfast", "lunch", "dinner", "snack"], description: "Optional current meal slot, to disambiguate by title." },
          new_date: { type: "string", description: "The new date to move it to (YYYY-MM-DD)." },
          new_meal_slot: { type: "string", enum: ["breakfast", "lunch", "dinner", "snack"], description: "Optional new meal slot (keeps the current slot if omitted)." }
        },
        required: ["new_date"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "remove_meal_calendar_entry",
      description: "Remove a scheduled entry from the user's meal calendar. Identify it by entry_id if known, otherwise by title (plus plan_date and/or meal_slot to disambiguate). If it can't be pinned down it returns entry_not_found / entry_ambiguous — say so honestly and do not claim it was removed.",
      parameters: {
        type: "object",
        properties: {
          entry_id: { type: "string", description: "The calendar entry's id, if known." },
          title: { type: "string", description: "The scheduled recipe's title, if the id isn't known." },
          plan_date: { type: "string", description: "Optional date (YYYY-MM-DD) to disambiguate by title." },
          meal_slot: { type: "string", enum: ["breakfast", "lunch", "dinner", "snack"], description: "Optional meal slot to disambiguate by title." }
        },
        additionalProperties: false
      }
    }
  }
];

export function buildChatTools() {
  return [
    ...buildSharedChatTools(),
    ...LOCAL_RECIPE_CATEGORY_CHAT_TOOLS,
    ...LOCAL_MEAL_CALENDAR_CHAT_TOOLS
  ];
}
