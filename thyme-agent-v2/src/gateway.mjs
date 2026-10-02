import { saveCanonical } from "./canonical-save.mjs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { Fault, hash, stable } from "./core.mjs";
const SOFT = [
  "cuisines",
  "favorite_foods",
  "avoid_foods",
  "preferred_products",
  "cooking_style",
  "max_prep_minutes",
];
export const READS = [
  "kitchen",
  "shopping",
  "saved_recipes",
  "suggestions",
  "meal_plan",
  "calendar",
  "dishes",
  "discards",
  "preferences",
];
export const ACTIONS = [
  "add_to_shopping_list",
  "add_many_to_shopping_list",
  "update_shopping_item_store",
  "update_many_shopping_item_stores",
  "mark_shopping_item_bought",
  "mark_shopping_item_unbought",
  "remove_from_shopping_list",
  "check_in_item",
  "check_in_many_items",
  "update_item_quantity",
  "mark_item_opened",
  "update_item_expiration",
  "update_item_location",
  "update_item_details",
  "discard_item",
  "save_generated_recipe",
  "remove_saved_recipe",
  "add_generated_recipes_to_meal_calendar",
  "move_meal_calendar_entry",
  "remove_meal_calendar_entry",
];
export function resourceFor(name) {
  if (name.includes("shopping")) return "shopping";
  if (
    name.includes("kitchen") ||
    /^(check_in_|update_item_|mark_item_|discard_item)/.test(name)
  )
    return "kitchen";
  if (name.includes("calendar")) return "calendar";
  if (name.includes("meal_plan")) return "meal_plan";
  if (name.includes("dish")) return "dishes";
  if (name.includes("memory")) return "preferences";
  return "saved_recipes";
}
export function collection(x) {
  if (Array.isArray(x)) return x;
  for (const k of [
    "items",
    "recipes",
    "entries",
    "dishes",
    "saved_recipes",
    "results",
  ])
    if (Array.isArray(x?.[k])) return x[k];
  return [];
}
export function fingerprint(data) {
  return hash(data);
}
export class TrepoGateway {
  constructor({
    env = process.env,
    root = process.env.TREPO_TOOLS_ROOT ||
      resolve(import.meta.dirname, "../.."),
  } = {}) {
    this.env = env;
    this.root = root;
  }
  async init() {
    if (this.data) return;
    const lib = this.root + "/playground12_voice_ack/trepo-quick-ack/lib/";
    [this.data, this.mysql, this.contexts, this.actions, this.defs] =
      await Promise.all([
        import(pathToFileURL(lib + "data-access.mjs")),
        import(pathToFileURL(lib + "mysql.mjs")),
        import(pathToFileURL(lib + "user-context.mjs")),
        import(pathToFileURL(lib + "tool-actions.mjs")),
        import(pathToFileURL(lib + "realtime-config.mjs")),
      ]);
  }
  async actor(actor) {
    await this.init();
    if (!JSON.parse(this.env.PILOT_ACTORS || "[]").includes(actor))
      throw new Fault(
        "not_enrolled",
        "This account is not in the private pilot.",
        403,
      );
    const ctx = await this.contexts.lookupUserContextByOwnerId(actor, {
      env: this.env,
    });
    if (ctx.isFallbackContext || ctx.userId !== actor || !ctx.ownerId)
      throw new Fault(
        "membership",
        "Your household connection needs refreshing.",
        403,
      );
    return { actor, household: String(ctx.ownerId), name: ctx.firstName, ctx };
  }
  async check(a) {
    const current = await this.actor(a.actor);
    if (current.household !== a.household)
      throw new Fault(
        "membership",
        "Your household changed. Start a new conversation.",
        403,
      );
    return current;
  }
  async memory(a) {
    const current = await this.check(a);
    return this.mysql.withDbConnection(
      async (c) => {
        const [states] = await c.execute(
          `SELECT b.actor_id,b.household_id,b.facts,b.paused,b.revision,b.generation,b.learning_since FROM family_brain_subjects b JOIN new_users u ON u.user_id=b.actor_id AND COALESCE(NULLIF(u.owner_id,''),u.user_id)=b.household_id WHERE b.household_id=? AND b.enabled=1 ORDER BY b.actor_id LIMIT 2`,
          [current.household],
        );
        const allowed = JSON.parse(this.env.FAMILY_BRAIN_ACTORS || "[]"),
          own = states.find((s) => s.actor_id === a.actor);
        const explicit = {};
        for (const s of states
          .filter((s) => allowed.includes(s.actor_id))
          .sort(
            (x, y) =>
              (x.actor_id === a.actor ? 1 : 0) -
              (y.actor_id === a.actor ? 1 : 0),
          )) {
          const facts =
            typeof s.facts === "string" ? JSON.parse(s.facts) : s.facts || {};
          for (const [k, f] of Object.entries(facts)) {
            if (
              !SOFT.includes(k) ||
              !f ||
              (f.visibility !== "household" && s.actor_id !== a.actor) ||
              (f.expires_at && new Date(f.expires_at).getTime() <= Date.now())
            )
              continue;
            explicit[k] = { value: f.value, scope: f.visibility };
          }
        }
        const observations = {};
        if (own && !own.paused) {
          const since = new Date(
            Math.max(
              Date.now() - 90 * 86400000,
              new Date(own.learning_since).getTime(),
            ),
          );
          const [saved] = await c.execute(
            "SELECT LEFT(title,120) AS name FROM shared_saved_recipes WHERE owner_id=? AND status='ready' AND _createdDate>=? ORDER BY _createdDate DESC,_id LIMIT 8",
            [a.actor, since],
          );
          observations.saved_recipe_interest = saved.map((x) => x.name);
          const [dishes] = await c.execute(
            "SELECT LEFT(dish_name,120) AS name FROM shared_dishes WHERE user_id=? AND owner_id IN (?,?) AND action='IN' AND _createdDate>=? ORDER BY _createdDate DESC,_id LIMIT 8",
            [a.actor, a.actor, a.household, since],
          );
          observations.logged_dishes = dishes.map((x) => x.name);
        }
        return {
          explicit_preferences: explicit,
          observations,
          learningPaused: !!own?.paused,
          revision: own?.revision ?? null,
          generation: own?.generation ?? null,
          enrolled: !!own,
        };
      },
      { env: this.env },
    );
  }
  async read(a, resource, args = {}) {
    await this.check(a);
    const o = { env: this.env, strict: true, all: true };
    let result;
    switch (resource) {
      case "kitchen":
        result = await this.data.getKitchenItemsFull(a.ctx, o);
        break;
      case "shopping":
        result = await this.data.getShoppingItems(a.ctx, o);
        break;
      case "saved_recipes":
        result = await this.savedRecipes(a);
        break;
      case "suggestions":
        result = await this.data.getRecipeSuggestions(a.ctx, o);
        break;
      case "meal_plan":
        result = await this.data.getMealPlan(a.ctx, o);
        break;
      case "calendar":
        result = await this.data.getMealCalendar(
          a.ctx,
          {
            start_date: new Date(Date.now() - 7 * 86400000)
              .toISOString()
              .slice(0, 10),
            end_date: new Date(Date.now() + 52 * 86400000)
              .toISOString()
              .slice(0, 10),
            ...args,
          },
          o,
        );
        break;
      case "dishes":
        result = await this.data.getRecentDishes(a.ctx, { ...o, limit: 100 });
        break;
      case "discards":
        result = await this.data.getRecentDiscards(a.ctx, { ...o, limit: 100 });
        break;
      case "preferences":
        result = {
          dietary: await this.data.getDietaryPreferences(a.ctx, o),
          memory: await this.memory(a),
        };
        break;
      default:
        throw new Fault("unknown_tool", "That data source is unavailable.");
    }
    return result;
  }
  async dietary(a) {
    await this.check(a);
    return this.data.getDietaryPreferences(a.ctx, {
      env: this.env,
      strict: true,
    });
  }
  async savedRecipes(a) {
    await this.check(a);
    return this.mysql.withDbConnection(
      async (c) => {
        const [rows] = await c.execute(
          "SELECT _id AS id,title,ingredients,instructions,notes,source_type,source_url,image_url,status,_createdDate,_updatedDate FROM shared_saved_recipes WHERE owner_id=? ORDER BY COALESCE(_updatedDate,_createdDate) DESC,_id",
          [a.actor],
        );
        const decode = (x) => (typeof x === "string" ? JSON.parse(x) : x || []);
        return {
          recipes: rows.map((r) => ({
            ...r,
            ingredients: decode(r.ingredients),
            instructions: decode(r.instructions),
            steps: decode(r.instructions),
            notes: decode(r.notes),
          })),
          count: rows.length,
          scope: "own",
        };
      },
      { env: this.env },
    );
  }
  async normalizeProposal(name, args, a) {
    if (name.includes("shopping")) {
      await this.init();
      for (const item of args.items || [args])
        if (item.store)
          item.store = await this.mysql.withDbConnection(
            (c) => this.data.resolveShoppingStore(c, a.ctx, item.store),
            { env: this.env },
          );
    }
    if (!/^(check_in_|update_item_)/.test(name)) return;
    await this.init();
    for (const item of args.items || [args]) {
      if (item.category !== undefined)
        item.category = this.data.normalizeKitchenCategory(
          item.category,
          item.location,
          item.new_name || item.item_name,
        );
      if (item.quantity_unit !== undefined)
        item.quantity_unit = this.data.normalizeUnit(item.quantity_unit);
      for (const field of ["quantity_value", "fill_percent"])
        if (item[field] !== undefined) {
          if (
            item[field] < 0 ||
            (field === "fill_percent" && item[field] > 100)
          )
            throw new Fault("quantity", "Choose a valid amount or percentage.");
          item[field] =
            field === "fill_percent"
              ? Math.round(item[field])
              : Math.round(item[field] * 100) / 100;
        }
      if (
        item.expiration_date !== undefined &&
        (!/^\d{4}-\d{2}-\d{2}$/.test(item.expiration_date) ||
          !Number.isFinite(new Date(item.expiration_date).getTime()) ||
          new Date(item.expiration_date).toISOString().slice(0, 10) !==
            item.expiration_date)
      )
        throw new Fault(
          "date",
          "Resolve the expiry date to YYYY-MM-DD before preparing the change.",
        );
    }
  }
  async definitions() {
    await this.init();
    const raw = this.defs
      .buildChatTools()
      .map((t) => ({ type: "function", ...t.function }));
    return raw
      .filter((t) => ACTIONS.includes(t.name))
      .map((t) => ({
        ...t,
        name: "request_" + t.name,
        description:
          "Propose for user review; DOES NOT apply yet. " + t.description,
        defer_loading: true,
      }));
  }
  async mutate(a, name, args, operationId, expected) {
    await this.check(a);
    if (name === "save_generated_recipe")
      return this.mysql.withDbConnection(
        (c) => saveCanonical(c, a, args, operationId),
        { env: this.env },
      );
    if (name === "update_food_memory")
      return this.controlMemory(a, args, operationId, expected);
    if (!ACTIONS.includes(name))
      throw new Fault("tool_forbidden", "That action is not available.", 403);
    const result = await this.actions.executeToolAction({
      toolName: name,
      args,
      env: { ...this.env, ACTION_MODE: "real" },
      userContext: a.ctx,
      responseSurface: "app",
      mutationContext: { requestId: operationId, actor: a.actor },
      sourceTranscript:
        "User approved the exact changes in the private Thyme tester.",
    });
    if (!result?.ok)
      throw new Fault(
        "action_failed",
        result?.error || "The change could not be applied.",
        409,
      );
    return result;
  }
  async controlMemory(a, args, operationId, expected) {
    if (!SOFT.includes(args.key))
      throw new Fault(
        "memory_key",
        "Only soft food preferences can be saved here.",
      );
    const value = args.key === "max_prep_minutes" ? args.minutes : args.values;
    if (
      args.key === "max_prep_minutes"
        ? !Number.isInteger(value) || value < 5 || value > 240
        : !Array.isArray(value) ||
          value.length < 1 ||
          value.length > 8 ||
          value.some((v) => typeof v !== "string" || !v.trim() || v.length > 80)
    )
      throw new Fault("memory_invalid", "That food preference is not valid.");
    if (args.visibility && !["personal", "household"].includes(args.visibility))
      throw new Fault(
        "memory_invalid",
        "Choose personal or household visibility.",
      );
    return this.mysql.withDbConnection(
      async (c) => {
        await c.beginTransaction();
        try {
          const [members] = await c.execute(
            "SELECT owner_id FROM new_users WHERE user_id=? FOR UPDATE",
            [a.actor],
          );
          if (String(members[0]?.owner_id) !== a.household)
            throw new Fault("membership", "Your household changed.", 403);
          const [rows] = await c.execute(
            "SELECT * FROM family_brain_subjects WHERE actor_id=? FOR UPDATE",
            [a.actor],
          );
          const state = rows[0];
          if (
            !state ||
            !state.enabled ||
            String(state.household_id) !== a.household
          )
            throw new Fault(
              "memory_disabled",
              "Food memory is not enabled.",
              403,
            );
          const operation = {
            action: "set",
            key: args.key,
            value,
            visibility: args.visibility || "personal",
            operation_id: operationId,
          };
          const intent = hash(operation);
          const [old] = await c.execute(
            "SELECT request_hash,result_json FROM family_brain_controls WHERE actor_id=? AND operation_id=?",
            [a.actor, operationId],
          );
          if (old[0]) {
            if (old[0].request_hash !== intent)
              throw new Fault(
                "conflict",
                "This operation ID has already been used.",
                409,
              );
            await c.commit();
            return typeof old[0].result_json === "string"
              ? JSON.parse(old[0].result_json)
              : old[0].result_json;
          }
          if (Number(state.revision) !== expected?.memory?.revision)
            throw new Fault(
              "stale_approval",
              "Your preferences changed. Ask Thyme to prepare the change again.",
              409,
            );
          const facts =
            typeof state.facts === "string"
              ? JSON.parse(state.facts)
              : state.facts || {};
          facts[args.key] = {
            value,
            visibility: operation.visibility,
            expires_at: null,
          };
          const revision = Number(state.revision) + 1;
          await c.execute(
            "UPDATE family_brain_subjects SET facts=?,revision=?,updated_at=UTC_TIMESTAMP(6) WHERE actor_id=?",
            [stable(facts), revision, a.actor],
          );
          const result = { ok: true, revision, generation: state.generation };
          await c.execute(
            "INSERT INTO family_brain_controls(actor_id,operation_id,request_hash,result_json) VALUES(?,?,?,?)",
            [a.actor, operationId, intent, stable(result)],
          );
          await c.commit();
          return result;
        } catch (e) {
          await c.rollback();
          throw e;
        }
      },
      { env: this.env },
    );
  }
}
