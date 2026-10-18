// Read-only household food collections. Never use model-supplied member IDs.
const collections = {
  saved_recipes: ['shared_saved_recipes', 'COALESCE(r._updatedDate,r._createdDate)'],
  shopping: ['shared_shopping_list', 'COALESCE(r.updated_at,r.created_at,r._createdDate)'],
  meal_plan: ['shared_meal_plan', 'COALESCE(r._updatedDate,r._createdDate)'],
};
export async function readHouseholdFoodRows(connection, context, resource) {
  const spec=collections[resource];
  if(!spec) throw new Error('Unsupported household food collection');
  const actor=String(context?.userId || context?.tableOwnerId || context?.ownerId || '').trim();
  if(!actor || context?.isFallbackContext) throw new Error('Household membership unavailable.');
  const [table,recency]=spec;
  const [rows]=await connection.execute(`SELECT r.*,actor.user_id AS _scope_actor,
    COALESCE(NULLIF(actor.owner_id,''),actor.user_id) AS _scope_household,
    member.first_name AS _member_name
    FROM new_users actor
    LEFT JOIN new_users member ON member.user_id=actor.user_id
      OR (actor.owner_id IS NOT NULL AND actor.owner_id<>'' AND member.owner_id=actor.owner_id)
    LEFT JOIN ${table} r ON r.owner_id=CONVERT(member.user_id USING utf8mb4) COLLATE utf8mb4_unicode_ci
    WHERE actor.user_id=? ORDER BY ${recency} DESC,r._id,r.owner_id`,[actor]);
  if(!rows?.length || rows[0]._scope_actor!==actor) throw new Error('Household membership unavailable.');
  const household=String(rows[0]._scope_household);
  if(context.ownerId && ![actor,household].includes(String(context.ownerId)))throw new Error('Household membership changed.');
  return rows.filter(r=>r._id!=null).map(({_scope_actor,_scope_household,_member_name,...r})=>({...r,source_owner_id:r.owner_id,source_owner_name:_member_name || 'Housemate'}));
}
const decode=value=> {
  if(value==null || value==='')return [];
  const parsed=typeof value==='string'?JSON.parse(value):value;
  if(!Array.isArray(parsed))throw new Error('Invalid saved recipe content.');
  return parsed;
};
export function householdSavedRecipes(rows,actor) {
  const recipes=rows.map(r=>({...r,id:String(r._id),ingredients:decode(r.ingredients),instructions:decode(r.instructions),notes:decode(r.notes),saved_by:r.source_owner_id,saved_by_name:r.source_owner_name,can_edit:r.source_owner_id===actor}));
  // Saved recipes have separate owners and stable IDs, even for the same source.
  // Keep both so explicit partner IDs still resolve and ownership remains accurate.
  return recipes;
}
export function householdShoppingRows(rows,actor) {
  const groups=new Map();
  for(const row of rows){const key=row.household_item_uuid || row._id;const copies=groups.get(key)||[];copies.push(row);groups.set(key,copies);}
  const time=r=>new Date(r.updated_at||r.created_at||r._createdDate||0).getTime();
  const rank=r=>({REMOVED:3,'2':3,CHECKED:2,'1':2,ADDED:1,'0':1}[r.action]||0);
  const result=[];
  for(const copies of groups.values()){
    copies.sort((a,b)=>time(b)-time(a)||rank(b)-rank(a)||(b.owner_id===actor?1:0)-(a.owner_id===actor?1:0));
    const row=copies[0];
    // Choose latest state BEFORE filtering so a deleted copy cannot reappear.
    if(['ADDED','CHECKED','0','1'].includes(row.action))result.push(row);
  }
  return result.sort((a,b)=>time(b)-time(a)||String(a._id).localeCompare(String(b._id)));
}
export function householdMealPlans(rows){
  const seen=new Set();
  return rows.filter(r=>{const key=JSON.stringify([r.status,r.focus,r.plan,r.error_message]);if(seen.has(key))return false;seen.add(key);return true;});
}
