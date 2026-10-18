// Read-only kitchen scope. Membership and inventory are resolved in one statement
// so cached session member lists cannot grant access after a household change.
// A LEFT JOIN distinguishes a registered empty kitchen from an unknown actor.
export async function readSharedHouseholdKitchen(connection, context) {
  const actor = String(context?.userId || context?.tableOwnerId || context?.ownerId || '').trim();
  if (!actor || context?.isFallbackContext) throw new Error('Kitchen membership unavailable.');
  const [rows] = await connection.execute(`
    SELECT k.*, actor.user_id AS _kitchen_scope_actor,
           COALESCE(NULLIF(actor.owner_id,''),actor.user_id) AS _kitchen_scope_household
    FROM new_users actor
    LEFT JOIN shared_kitchen k ON k.action = 'IN' AND k.owner_id IN (
      SELECT CONVERT(member.user_id USING utf8mb4) COLLATE utf8mb4_unicode_ci FROM new_users member
      WHERE member.user_id = actor.user_id
         OR (actor.owner_id IS NOT NULL AND actor.owner_id <> '' AND member.owner_id = actor.owner_id)
    )
    WHERE actor.user_id = ?
    ORDER BY COALESCE(k._updatedDate,k._createdDate) DESC,k._id,k.owner_id`, [actor]);
  if (!rows?.length || rows[0]._kitchen_scope_actor !== actor) {
    throw new Error('Kitchen membership unavailable.');
  }
  const household = String(rows[0]._kitchen_scope_household);
  if (context?.ownerId && ![actor,household].includes(String(context.ownerId))) {
    throw new Error('Kitchen membership changed. Refresh your household.');
  }
  return rows.filter(row => row._id != null).map(({_kitchen_scope_actor, _kitchen_scope_household, ...row}) => row);
}
