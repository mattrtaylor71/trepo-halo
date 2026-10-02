import { Fault, hash } from "./core.mjs";
const decode = value => typeof value === "string" ? JSON.parse(value) : value;
export const operationUUID = (actor, operation) => {
  const h = hash([actor, operation]);
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20,32)}`;
};
const token = value => {
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(value)) throw new Fault("identity", "Invalid account.", 403);
  return value;
};
async function columns(c, table) {
  const [rows] = await c.execute("SELECT COLUMN_NAME FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=?", [table]);
  return new Set(rows.map(r => r.COLUMN_NAME));
}
export async function membership(c, a) {
  const [rows] = await c.execute("SELECT user_id,owner_id FROM new_users WHERE COALESCE(NULLIF(owner_id,''),user_id)=? ORDER BY user_id FOR UPDATE", [a.household]);
  if (!rows.some(r => r.user_id === a.actor)) throw new Fault("membership", "Your household changed. Review this again.", 403);
  return rows.map(r => token(r.user_id));
}
async function update(c, table, fields, where, values) {
  const available = await columns(c, table);
  if (!available.size) throw new Fault("schema", "This app library is not ready yet. Open that section in Trepo and try again.", 409);
  const names = Object.keys(fields);
  if (!names.every(n => available.has(n))) throw new Fault("schema", "This app library needs an update before it can be changed.", 409);
  const touch = available.has("_updatedDate") ? ",_updatedDate=UTC_TIMESTAMP()" : available.has("updated_at") ? ",updated_at=UTC_TIMESTAMP()" : "";
  await c.execute(`UPDATE \`${table}\` SET ${names.map(n => "`"+n+"`=?").join(",")}${touch} WHERE ${where}`,
    [...names.map(n => fields[n] instanceof Date ? fields[n] : typeof fields[n] === "object" && fields[n] !== null ? JSON.stringify(fields[n]) : fields[n]), ...values]);
}
async function insert(c, table, fields) {
  const available = await columns(c, table);
  if (!available.size) throw new Fault("schema", "Open the Dish Log in Trepo once before adding a dish here.", 409);
  const names = Object.keys(fields).filter(n => available.has(n));
  await c.execute(`INSERT INTO \`${table}\` (${names.map(n => "`"+n+"`").join(",")}) VALUES (${names.map(()=>"?").join(",")})`,
    names.map(n => fields[n] instanceof Date ? fields[n] : typeof fields[n] === "object" && fields[n] !== null ? JSON.stringify(fields[n]) : fields[n]));
}
const recipeFields = r => ({title:r.title, ingredients:decode(r.ingredients), instructions:decode(r.instructions || r.steps), notes:decode(r.notes) || []});
export function dishFields(r) {
  return { dish_name: r.dish_name, serving_size:r.serving_size ?? null, ingredients:decode(r.ingredients) || [],
    calories:r.calories == null ? null : Number(r.calories), protein:r.protein == null ? null : Number(r.protein),
    total_carbohydrates:r.total_carbohydrates == null ? null : Number(r.total_carbohydrates), total_fat:r.total_fat == null ? null : Number(r.total_fat), action:r.action || "IN" };
}
export function dishPatch(args) {
  const fields = {};
  const map = {protein_g:"protein",carbs_g:"total_carbohydrates",fat_g:"total_fat"};
  for (const k of ["dish_name","serving_size","ingredients","calories","protein_g","carbs_g","fat_g"])
    if (args[k] !== undefined) fields[map[k] || k] = args[k];
  return fields;
}
export async function personalWrite(c, a, action, args, operationId, before) {
  const actor = token(a.actor), dish = action.includes("dish"), suffix = dish ? "dishes" : "saved_recipes";
  const shared = "shared_" + suffix, legacy = actor + "_" + suffix;
  const id = args.dish_id || args.recipe_id || operationUUID(actor, operationId);
  await c.beginTransaction();
  try {
    await membership(c, a);
    // Dish Log is personal even if a historical shared mirror has a household owner.
    const [rows] = await c.execute(`SELECT * FROM \`${legacy}\` WHERE _id=? FOR UPDATE`, [id]);
    const row = rows[0];
    const scope = dish ? "_id=? AND (user_id=? OR (user_id IS NULL AND _owner=?)) AND owner_id IN (?,?)" : "_id=? AND owner_id=?";
    const scopeValues = dish ? [id,actor,actor,actor,a.household] : [id,actor];
    if (action.startsWith("log_dish_")) {
      const fields = { ...dishPatch(args), action:"IN" };
      if (row) {
        if (!Object.entries(fields).every(([k,v]) => hash(dishFields(row)[k])===hash(v))) throw new Fault("request_conflict","This request was already used for another dish.",409);
      } else {
        const payload = { _id:id,_owner:actor,_device:"thyme-pilot",owner_id:actor,user_id:actor,
          _createdDate:new Date().toISOString().slice(0,19).replace("T"," "), ...fields,
          explanation:"Logged as reviewed in Thyme. Nutrition, when supplied, is an estimate.", analysis_status:"complete" };
        await insert(c, legacy, payload);
        await insert(c, shared, payload);
      }
    } else {
      if (!row) throw new Fault("missing","That saved item no longer exists.",404);
      const old = before.find(r => String(r.id || r._id) === String(id));
      const snapshot = dish ? dishFields : recipeFields;
      if (!old || hash(snapshot(row)) !== hash(snapshot(old))) throw new Fault("stale_approval","That item changed after review. Prepare the change again.",409);
      if (["remove_saved_recipe","delete_dish_log"].includes(action)) {
        await c.execute(`DELETE FROM \`${legacy}\` WHERE _id=?`, [id]);
        await c.execute(`DELETE FROM \`${shared}\` WHERE ${scope}`, scopeValues);
      } else {
        const fields = dish ? (action === "mark_dish_consumed" ? {action:"OUT"} : dishPatch(args)) :
          Object.fromEntries(["title","ingredients","instructions","notes"].filter(k=>args[k]!==undefined).map(k=>[k,args[k]]));
        if (!Object.keys(fields).length) throw new Fault("empty_edit","Specify a detail to change.");
        await update(c, legacy, fields, "_id=?", [id]);
        // Shared mirrors must agree, not silently report success on a zero-row update.
        const [mirrors] = await c.execute(`SELECT _id FROM \`${shared}\` WHERE ${scope} FOR UPDATE`, scopeValues);
        if (!mirrors.length) {
          // A legacy personal row is authoritative. Repair an absent shared
          // mirror as part of the same edit; never overwrite another user's row.
          const [collision]=await c.execute(`SELECT _id FROM \`${shared}\` WHERE owner_id=? AND _id=? FOR UPDATE`,[actor,id]);
          if(collision.length) throw new Fault("mirror_missing","This item needs its app data refreshed before editing.",409);
          await insert(c,shared,{...row,owner_id:actor,_owner:actor,...(dish?{user_id:actor}:{})});
        }
        await update(c, shared, fields, scope, scopeValues);
      }
    }
    if (!dish) {
      if ((await columns(c,"owner_recipe_availability")).size)
        await c.execute("DELETE FROM owner_recipe_availability WHERE owner=? AND recipe_source='saved' AND recipe_id=?",[actor,id]);
      if (action==="remove_saved_recipe" && (await columns(c,"shared_recipe_category_map")).size)
        await c.execute("DELETE FROM shared_recipe_category_map WHERE owner_id=? AND recipe_id=?",[actor,id]);
    }
    await c.commit();
    return {ok:true,toolResult:{[dish?"dish":"recipe"]:{id},removed_id:action.startsWith("delete_")||action.startsWith("remove_")?id:undefined}};
  } catch(e) { await c.rollback(); throw e; }
}

// Exact IDs replace the legacy fuzzy-name resolver. Touch only current household
// mirrors with the same stable UUID. Missing legacy mirrors fail the transaction.
export async function shoppingWrite(c, a, action, args, before, operationId="") {
  await c.beginTransaction();
  try {
    const members = await membership(c,a);
    const targets = args.targets || args.items || [args];
    for (const target of targets) {
      const id = target.item_id;
      const old = before.find(r => String(r.id || r._id || r.shopping_id)===String(id));
      if (!old) throw new Fault("missing","Read the exact shopping item before changing it.",404);
      const [rows] = await c.execute("SELECT * FROM shared_shopping_list WHERE owner_id=? AND _id=? AND action IN ('ADDED','CHECKED') FOR UPDATE",[a.actor,id]);
      const row=rows[0];
      if (!row || row.product_name!==old.item_name || String(row.quantity||"")!==String(old.quantity||"") || row.store!==old.store || row.action!==old.action || (row.household_item_uuid||null)!==(old.household_item_uuid||null))
        throw new Fault("stale_approval","Your shopping list changed. Prepare the update again.",409);
      const uuid=row.household_item_uuid;
      // Legacy rows without a stable household UUID cannot safely match other members.
      if (!uuid && members.length>1) throw new Fault("legacy_identity","This older item needs to be refreshed in Trepo before changing it.",409);
      const remove=["remove_from_shopping_list","remove_shopping_list","clear_shopping_list"].includes(action);
      const fields=action==="mark_shopping_item_bought"?{action:"CHECKED"}:action==="mark_shopping_item_unbought"?{action:"ADDED"}:
        Object.fromEntries(Object.entries({product_name:target.new_name,quantity:target.quantity,store:target.store}).filter(([,v])=>v!==undefined));
      for(const member of members) {
        const match=uuid?"household_item_uuid=?":"_id=?", val=uuid||id;
        const table=member+"_new_list";
        const [legacy]=await c.execute(`SELECT _id FROM \`${table}\` WHERE ${match} FOR UPDATE`,[val]);
        const [sharedRows]=await c.execute(`SELECT _id FROM shared_shopping_list WHERE owner_id=? AND ${match} FOR UPDATE`,[member,val]);
        // Legacy adds create a shared actor row and one per-user row per member.
        // Newer app paths may also create shared member mirrors. Support both.
        if(legacy.length!==1 || sharedRows.length>1 || (member===a.actor && sharedRows.length!==1)) throw new Fault("mirror_changed","This list item needs a refresh before changing it.",409);
        if(remove) {
          const table=member+"_master_feed";
          if (!(await columns(c,table)).size) throw new Fault("history_unavailable","Open the app's activity history once before removing items here.",409);
          await insert(c,table,{_id:operationUUID(member,[operationId,action,id]),_owner:member,_device:"thyme-pilot",
            _createdDate:new Date().toISOString().slice(0,19).replace("T"," "),event_type:"shopping_remove",entity_type:"shopping_item",action:"OUT",title:row.product_name,
            item_id:String(id),user_id:a.actor,source_table:"shared_shopping_list",source_path:action,source_system:"thyme_pilot",metadata:{removed:true,household_item_uuid:uuid,snapshot:row}});
          await c.execute(`DELETE FROM \`${member}_new_list\` WHERE ${match}`,[val]);
          await c.execute(`DELETE FROM shared_shopping_list WHERE owner_id=? AND ${match}`,[member,val]);
        } else {
          if(!Object.keys(fields).length) throw new Fault("empty_edit","Specify a detail to change.");
          await update(c,table,fields,match,[val]);
          if(sharedRows.length) await update(c,"shared_shopping_list",fields,`owner_id=? AND ${match}`,[member,val]);
        }
      }
    }
    await c.commit(); return {ok:true,toolResult:{count:targets.length}};
  } catch(e) {await c.rollback();throw e;}
}

export function discardFields(row) {
  return {id:String(row.id||row._id),item_name:row.item_name||row.product_name||"Unknown item",brand:row.brand||null,
    category:row.category||null,discard_reason:row.discard_reason||null,action:row.action||"IN",
    created_at:new Date(row.created_at||row._createdDate).toISOString()};
}
export async function discardWrite(c,a,args,before) {
  await c.beginTransaction();
  try {
    const members=await membership(c,a),owners=[...new Set([...members,a.household])];
    for(const t of args.targets) {
      const old=before.find(r=>String(r.id)===t.item_id);
      const table=token(a.actor)+"_discards";
      const [rows]=await c.execute(`SELECT * FROM \`${table}\` WHERE _id=? AND action='IN' FOR UPDATE`,[t.item_id]);
      if(!old || rows.length!==1 || hash(discardFields(rows[0]))!==hash(discardFields(old))) throw new Fault("stale_approval","Discard history changed. Review the removal again.",409);
      for(const member of members) if((await columns(c,member+"_discards")).size)
        await update(c,member+"_discards",{action:"OUT"},"_id=? AND action='IN'",[t.item_id]);
      await update(c,"shared_discards",{action:"OUT"},`_id=? AND owner_id IN (${owners.map(()=>"?").join(",")}) AND action='IN'`,[t.item_id,...owners]);
    }
    await c.commit(); return {ok:true,toolResult:{count:args.targets.length}};
  }catch(e){await c.rollback();throw e;}
}

export async function categoryWrite(c,a,args) {
  await c.beginTransaction();
  try {
    await membership(c,a);
    const [categories]=await c.execute("SELECT category_id FROM shared_recipe_categories WHERE owner_id=? AND LOWER(name)=LOWER(?) FOR UPDATE",[a.actor,args.category_name]);
    let id=categories[0]?.category_id;
    if(!id) {
      const [all]=await c.execute("SELECT category_id,sort_order FROM shared_recipe_categories WHERE owner_id=? FOR UPDATE",[a.actor]);
      if(all.length>=100) throw new Fault("category_limit","The category limit has been reached.");
      id=operationUUID(a.actor,args.category_name.toLowerCase());
      await c.execute("INSERT INTO shared_recipe_categories(owner_id,category_id,name,sort_order) VALUES (?,?,?,?)",[a.actor,id,args.category_name,Math.max(-1,...all.map(x=>Number(x.sort_order)))+1]);
    }
    for(const recipeId of args.recipe_ids||[]) {
      const [recipes]=await c.execute("SELECT _id FROM shared_saved_recipes WHERE owner_id=? AND _id=? FOR UPDATE",[a.actor,recipeId]);
      if(recipes.length!==1) throw new Fault("missing","This saved recipe no longer exists.",409);
      // Add one tag without replacing other tags, including concurrent app edits.
      await c.execute("INSERT IGNORE INTO shared_recipe_category_map(owner_id,recipe_id,category_id) VALUES (?,?,?)",[a.actor,recipeId,id]);
    }
    await c.commit();return {ok:true,toolResult:{category:{id,name:args.category_name}}};
  }catch(e){await c.rollback();throw e;}
}
