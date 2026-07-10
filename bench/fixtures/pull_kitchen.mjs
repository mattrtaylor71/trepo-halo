// Real kitchen inventories → fixtures/kitchen_*.json (read-only snapshot).
// Mirrors recipes_generator _get_kitchen_ingredients: shared_kitchen filtered by
// owner_id IN (household member_ids), household resolved via new_users.
import mysql from "mysql2/promise";
import { writeFileSync } from "node:fs";
const c = await mysql.createConnection({host:"database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com",user:"admin",password:"Nbmqyq17!",database:"mysqlTutorial",connectTimeout:15000});
async function members(owner){
  const [r] = await c.execute("SELECT owner_id FROM new_users WHERE user_id=? LIMIT 1",[owner]);
  const hh = r[0]?.owner_id;
  if(!hh) return [owner];
  const [m] = await c.execute("SELECT user_id FROM new_users WHERE owner_id=? ORDER BY created_at ASC, user_id ASC",[hh]);
  return m.map(x=>x.user_id);
}
const OWNERS = { testuser:"1f4db6b6-2f62-4558-aa40-c8e82527dc74", owner3ed:"3ed72da8-4bab-40c3-a547-9bd1c4d810f7" };
for (const [label, owner] of Object.entries(OWNERS)) {
  const mem = await members(owner);
  const ph = mem.map(()=>"?").join(",");
  const [rows] = await c.execute(
    `SELECT product_name, brand, variant, category, remaining_quantity, quantity_value, quantity_unit, is_opened, fill_percent, product_expiration, storage_location, product_description
     FROM shared_kitchen WHERE owner_id IN (${ph}) AND (action IS NULL OR action <> 'OUT') ORDER BY _createdDate DESC`, mem);
  const items = rows.map(r=>({
    product_name:r.product_name, brand:r.brand||null, variant:r.variant||null, category:r.category||null,
    quantity: r.remaining_quantity || (r.quantity_value!=null?`${r.quantity_value} ${r.quantity_unit||""}`.trim():null),
    is_opened:!!r.is_opened, fill_percent:r.fill_percent, expiration:r.product_expiration||null,
    storage_location:r.storage_location||null, description:r.product_description||null,
  })).filter(i=>i.product_name);
  writeFileSync(`fixtures/kitchen_${label}.json`, JSON.stringify({owner, household_members:mem, count:items.length, items}, null, 2));
  console.log(`${label}: household=${mem.length} members, ${items.length} kitchen items`);
}
await c.end();
