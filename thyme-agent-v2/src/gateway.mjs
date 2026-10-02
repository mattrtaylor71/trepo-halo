import { READ_TOOLS, LEGACY_WRITES, EXTRA_ACTIONS, catalog } from "./capabilities.mjs";
import { personalWrite, shoppingWrite, discardWrite, categoryWrite, discardFields, operationUUID } from "./app-writes.mjs";
import { prepareAction, exact, rowID, storeName, pageData } from "./prepare-actions.mjs";
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
  "preferences", "stores", "recipe_categories", "health", "web_recipes", "buy_suggestions", "use_up_suggestions",
];
export const ACTIONS = [...LEGACY_WRITES, ...EXTRA_ACTIONS,
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
  if (name.includes("recipe_category") || name === "move_recipe_to_category") return "recipe_categories";
  if (name.includes("discard") && name !== "discard_item") return "discards";
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
    "categories",
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
    a = await this.check(a);
    const o = { env: this.env, strict: true, all: true };
    let result;
    switch (resource) {
      case "kitchen":
        result = (await this.kitchenRows(a)).map((row) => ({...this.data.mapKitchenRow(row), amount_revision:Number(row.amount_revision), reference_amount:row.reference_amount==null?null:Number(row.reference_amount)}));
        break;
      case "shopping":
        result = await this.mysql.withDbConnection(async (c) => {
          // The app's list is mirrored per member. Read this actor's current
          // rows once, not every member's copies or the removed history.
          const [rows] = await c.execute(
            "SELECT * FROM shared_shopping_list WHERE owner_id=? AND action IN ('ADDED','CHECKED') ORDER BY COALESCE(updated_at,created_at,_createdDate) DESC,_id",
            [a.actor],
          );
          return rows.map((row) => this.data.mapShoppingRow(row));
        }, o);
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
        if(args.entry_id) {result=await this.calendarEntry(a,args.entry_id);break;}
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
        result = await this.personalDishes(a);
        break;
      case "discards":
        result = await this.mysql.withDbConnection(async c=> {
          const table=a.actor+"_discards";
          if(!/^[a-zA-Z0-9_-]{1,64}$/.test(a.actor)) throw new Fault("identity","Invalid account.",403);
          if(!await this.mysql.tableExists(c,table)) return [];
          const [rows]=await c.execute(`SELECT * FROM \`${table}\` WHERE action='IN' ORDER BY _createdDate DESC,_id`);
          return rows.map(discardFields);
        },o);
        break;
      case "stores": {
        const items = await this.read(a, "shopping");
        result = { stores: [...new Set(items.map(r=>storeName(r.store)))] };
        break;
      }
      case "recipe_categories": result = await this.data.listRecipeCategories(a.ctx,o); break;
      case "health": result = await this.data.getHealthMetrics(a.ctx,o); break;
      case "web_recipes": result = await this.data.searchWebRecipes(a.ctx,args,o); break;
      case "buy_suggestions": {
        const [kitchen,shopping]=await Promise.all([this.read(a,"kitchen"),this.read(a,"shopping")]);
        const listed=new Set(shopping.map(r=>String(r.item_name).trim().toLowerCase()));
        result={items:kitchen.filter(r=>!listed.has(String(r.item_name).trim().toLowerCase()) &&
          (r.quantity_value===0 || (r.fill_percent!=null && r.fill_percent<=25))).slice(0,10).map(r=>({item_id:r.id,item_name:r.item_name,reason:"Recorded amount is low; check whether you want to restock."})),
          note:"These are measured low-stock candidates, not dietary recommendations. For recipe groceries compare the exact recipe with the fresh kitchen and shopping data; never assume an unrecorded amount is low."};
        break;
      }
      case "use_up_suggestions": {
        const kitchen=await this.read(a,"kitchen");
        result={items:kitchen.filter(r=>r.expiration_date || r.is_opened).sort((x,y)=>String(x.expiration_date||"9999").localeCompare(String(y.expiration_date||"9999"))).slice(0,10).map(r=>({item_id:r.id,item_name:r.item_name,expiration_date:r.expiration_date,is_opened:r.is_opened,
          reason:r.expiration_date?"Check the recorded date and condition before using.":"Opened item; check condition and storage before using."})),
          note:"Dates are estimates, not a guarantee of food safety. Do not recommend eating an item past its usable life."};
        break;
      }
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
  async kitchenRows(a) {
    return this.mysql.withDbConnection(async (c) => {
      const [members] = await c.execute(
        "SELECT user_id FROM new_users WHERE COALESCE(NULLIF(owner_id,''),user_id)=? ORDER BY user_id",
        [a.household],
      );
      const ids = [...new Set(members.map((m) => m.user_id))];
      if (!ids.includes(a.actor))
        throw new Fault("membership", "Your household changed. Refresh your kitchen.", 403);
      const [rows] = await c.execute(
        `SELECT k.*,COALESCE(e.revision,0) AS amount_revision,e.reference_amount FROM shared_kitchen k LEFT JOIN kitchen_item_edits e ON e.item_id=CONVERT(k._id USING utf8mb4) COLLATE utf8mb4_unicode_ci WHERE k.owner_id IN (${ids.map(() => '?').join(',')}) AND k.action='IN' ORDER BY COALESCE(k._updatedDate,k._createdDate) DESC,k._id,k.owner_id`,
        ids,
      );
      return rows;
    }, { env: this.env, strict: true });
  }
  async calendarEntry(a,id) {
    if(!/^[a-zA-Z0-9_-]{1,64}$/.test(a.actor)) throw new Fault("identity","Invalid account.",403);
    const dates=await this.mysql.withDbConnection(async c=> {
      const [members]=await c.execute("SELECT user_id FROM new_users WHERE COALESCE(NULLIF(owner_id,''),user_id)=?",[a.household]);
      const ids=[...new Set([a.actor,a.household,...members.map(x=>x.user_id)])],dates=[];
      if(await this.mysql.tableExists(c,"shared_meal_calendar")) {
        const [rows]=await c.execute(`SELECT plan_date FROM shared_meal_calendar WHERE _id=? AND owner_id IN (${ids.map(()=>"?").join(",")})`,[id,...ids]);dates.push(...rows.map(r=>r.plan_date));
      }
      const table=a.actor+"_meal_calendar";
      if(await this.mysql.tableExists(c,table)) {
        const [rows]=await c.execute(`SELECT plan_date FROM \`${table}\` WHERE _id=?`,[id]);dates.push(...rows.map(r=>r.plan_date));
      }
      return [...new Set(dates.map(d=>new Date(d).toISOString().slice(0,10)))];
    },{env:this.env});
    const entries=[];
    // SQL only locates dates. The app endpoint remains authoritative for which
    // entries this member may see; it also handles shared/per-user read modes.
    for(const date of dates) {
      const data=await this.data.getMealCalendar(a.ctx,{start_date:date,end_date:date},{env:this.env,strict:true});
      entries.push(...collection(data).filter(r=>rowID(r)===String(id)));
    }
    return {entries:[...new Map(entries.map(r=>[rowID(r),r])).values()]};
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
    if (name.includes("shopping") && !["remove_shopping_list","rename_shopping_list"].includes(name)) {
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
    return catalog(raw, ACTIONS);
  }
  async personalDishes(a) {
    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(a.actor)) throw new Fault("identity","Invalid account.",403);
    return this.mysql.withDbConnection(async c=> {
      const table=a.actor+"_dishes";
      if(!await this.mysql.tableExists(c,table)) return {items:[],count:0};
      const [rows]=await c.execute(`SELECT * FROM \`${table}\` ORDER BY COALESCE(_updatedDate,_createdDate) DESC,_id`);
      return {items:rows.map(r=>({...this.data.mapDishRow(r)})),count:rows.length};
    },{env:this.env});
  }
  async readTool(a,name,args) {
    const resource=READ_TOOLS[name];
    if(!resource) throw new Fault("unknown_tool","That read is unavailable.");
    const data=await this.read(a,resource,args);
    if(name==="search_kitchen_item") return {items:collection(data).filter(r=>args.item_id?rowID(r)===args.item_id:r.item_name.toLowerCase().includes(String(args.item_name||"").toLowerCase()))};
    if(name==="get_saved_recipe_detail") return {recipe:exact(collection(data),args.recipe_id,args.recipe_title)};
    if(name==="get_dish_detail") return {dish:exact(collection(data),args.dish_id,args.dish_name,"dish_name")};
    if(name==="get_recipe_detail") return {recipe:exact([...(data.kitchen_only||[]),...(data.need_grocery||[]),...collection(data)],null,args.recipe_title)};
    if(["get_recent_dishes","get_recent_discards","get_saved_recipes"].includes(name)) return pageData(data,args);
    if(args.limit && Array.isArray(data)) return data.slice(0,Math.min(100,args.limit));
    return data;
  }
  async prepare(a,name,args,before) { return prepareAction(this,a,name,args,before); }
  async kitchenEdit(a,name,args,operationId,expected) {
    const old=exact(collection(expected),args.item_id);
    const fields={};
    const map={new_name:"product_name",location:"storage_location",expiration_date:"product_expiration"};
    for(const k of ["new_name","brand","category","location","expiration_date","is_opened","quantity_value","quantity_unit"])
      if(args[k]!==undefined) fields[map[k]||k]=args[k];
    if(name==="mark_item_opened") fields.is_opened=true;
    if(args.fill_percent!==undefined) {
      if(!old.reference_amount) throw new Fault("amount","Set a measured amount before adjusting its percentage.");
      fields.quantity_value=Math.round(old.reference_amount*args.fill_percent)/100;
      fields.quantity_unit=args.quantity_unit||old.quantity_unit;
    }
    if(args.remaining_quantity!==undefined && fields.quantity_value===undefined)
      throw new Fault("amount","Specify a numerical amount and its unit.");
    if(fields.quantity_value!==undefined) fields.quantity_unit=fields.quantity_unit||old.quantity_unit;
    if(!Object.keys(fields).length || !Number.isSafeInteger(old.amount_revision)) throw new Fault("revision","Refresh this kitchen item before changing it.",409);
    let body={operation_id:operationUUID(a.actor,operationId),kind:"edit",items:[{item_id:args.item_id,revision:old.amount_revision,fields}]};
    if(fields.quantity_value===0) {
      if(Object.keys(fields).some(k=>!["quantity_value","quantity_unit"].includes(k))) throw new Fault("amount","Empty the item separately from changes to its details.");
      const known=old.quantity_value>0 && old.quantity_unit;
      body={operation_id:body.operation_id,kind:known?"consume":"discard",items:[{item_id:args.item_id,revision:old.amount_revision,
        ...(known?{amount:old.quantity_value,expected_amount:old.quantity_value,unit:old.quantity_unit}:{})}]};
    }
    const result=await this.appAPI(a,`/kitchen/${encodeURIComponent(a.actor)}?amount_operation=v1`,{method:"POST",body});
    if(result.operation_id!==body.operation_id || result.items?.length!==1 || result.items[0]?.item?._id!==args.item_id)
      throw new Fault("unconfirmed","The kitchen result could not be confirmed.",503);
    return {ok:true,toolResult:result};
  }
  async appAPI(a,path,{method="GET",body}={}) {
    await this.check(a);
    const base=new URL(this.env.HOUSEHOLD_API_BASE_URL || "https://7tn3gvwvh7.execute-api.us-east-1.amazonaws.com");
    if(base.protocol!=="https:" || base.username || base.password || base.search || base.hash) throw new Fault("configuration","App connection unavailable.",503);
    const response=await fetch(base.href.replace(/\/$/,"")+path,{method,redirect:"error",signal:AbortSignal.timeout(25000),headers:{"content-type":"application/json"},...(body?{body:JSON.stringify(body)}:{})});
    let result;try{result=await response.json();}catch{throw new Fault("unconfirmed","The app response could not be read.",503);}
    if(!response.ok) throw new Fault("app_request",result.error||result.message||"The app could not complete this request.",response.status);
    return result;
  }

  async mutate(a, name, args, operationId, expected) {
    a = await this.check(a);
    if (name === "save_generated_recipe")
      return this.mysql.withDbConnection(
        (c) => saveCanonical(c, a, args, operationId),
        { env: this.env },
      );
    if (name === "update_food_memory")
      return this.controlMemory(a, args, operationId, expected);
    if (!ACTIONS.includes(name))
      throw new Fault("tool_forbidden", "That action is not available.", 403);
    if (["edit_saved_recipe","remove_saved_recipe","log_dish_from_voice","log_dish_ingredients","append_to_recent_dish","update_recent_dish","update_dish","mark_dish_consumed","delete_dish_log"].includes(name))
      return this.mysql.withDbConnection(c=>personalWrite(c,a,name,args,operationId,collection(expected)),{env:this.env});
    if (["update_shopping_item","update_shopping_item_store","update_many_shopping_item_stores","mark_shopping_item_bought","mark_shopping_item_unbought","remove_from_shopping_list","rename_shopping_list","remove_shopping_list","clear_shopping_list"].includes(name))
      return this.mysql.withDbConnection(c=>shoppingWrite(c,a,name,args,collection(expected),operationId),{env:this.env});
    if (/^(update_item_|mark_item_)/.test(name)) return this.kitchenEdit(a,name,args,operationId,expected);
    if (["delete_recent_discard","clear_recent_discards"].includes(name))
      return this.mysql.withDbConnection(c=>discardWrite(c,a,args,collection(expected)),{env:this.env});
    if (["move_recipe_to_category","create_recipe_category"].includes(name))
      return this.mysql.withDbConnection(c=>categoryWrite(c,a,args),{env:this.env});
    if (["clear_kitchen_inventory","discard_item"].includes(name)) {
      const selected=name==="discard_item"?[exact(collection(expected),args.item_id)]:collection(expected);
      if(!selected.length || selected.some(r=>!Number.isSafeInteger(r.amount_revision))) throw new Fault("revision","Refresh the kitchen before removing items.",409);
      const body={operation_id:operationUUID(a.actor,operationId),kind:"discard",items:selected.map(r=>({item_id:rowID(r),revision:r.amount_revision}))};
      const result=await this.appAPI(a,`/kitchen/${encodeURIComponent(a.actor)}?amount_operation=v1`,{method:"POST",body});
      if(result.operation_id!==body.operation_id) throw new Fault("unconfirmed","The kitchen result could not be confirmed.",503);
      return {ok:true,toolResult:result};
    }
    if(name === "create_shopping_list" || name.includes("ingredients_to_shopping_list")) { name="add_many_to_shopping_list"; args={items:args.items}; }
    if(name === "add_recipe_to_meal_calendar" || name === "add_many_to_meal_calendar") { name="add_generated_recipes_to_meal_calendar"; args={entries:args.entries}; }
    const requestedAt=new Date().toISOString();
    const result = await this.actions.executeToolAction({
      toolName: name,
      args,
      env: { ...this.env, ACTION_MODE: "real" },
      userContext: a.ctx,
      responseSurface: "app",
      mutationContext: { requestId: operationId, actor: a.actor, operations:new Map(), editIntents:new Map() },
      sourceTranscript:
        "User approved the exact changes in the private Thyme tester." + (args.item_id ? " Item " + args.item_id : ""),
    });
    if (!result?.ok)
      throw new Fault(
        "action_failed",
        result?.error || "The change could not be applied.",
        409,
      );
    return {...result,requestedAt};
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
