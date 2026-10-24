// Scope comes from current server membership, never a model/cached member list.
function actorFor(context) {
  const actor = String(context?.userId || context?.tableOwnerId || context?.ownerId || '').trim();
  if (!actor || context?.isFallbackContext) throw new Error('Kitchen membership unavailable.');
  return actor;
}

export async function kitchenMutationScope(connection, context, lock = false) {
  const actor = actorFor(context);
  const suffix = lock ? ' LOCK IN SHARE MODE' : '';
  const [actors] = await connection.execute('SELECT user_id,owner_id FROM new_users WHERE user_id=?' + suffix, [actor]);
  if (actors.length !== 1) throw new Error('Kitchen membership unavailable.');
  const household = String(actors[0].owner_id || actor);
  if (context.ownerId && ![actor, household].includes(String(context.ownerId))) {
    throw new Error('Kitchen membership changed.');
  }
  const [members] = await connection.execute(
    'SELECT user_id FROM new_users WHERE user_id=? OR (owner_id=? AND owner_id IS NOT NULL AND owner_id<>\'\') ORDER BY user_id' + suffix,
    [actor, household]);
  const memberIds = [...new Set(members.map(row => String(row.user_id)))].sort();
  if (!memberIds.includes(actor)) throw new Error('Kitchen membership unavailable.');
  return { ...context, ownerId: household, userId: actor, tableOwnerId: actor, householdMemberIds: memberIds };
}

export async function withHouseholdKitchenMutation(connection, context, callback, prepare = async () => {}) {
  // Existing legacy table setup uses DDL, which implicitly commits in MySQL.
  // Complete it before taking membership/item locks; reject a transfer in between.
  const initial = await kitchenMutationScope(connection, context);
  await prepare(initial);
  await connection.beginTransaction();
  try {
    const current = await kitchenMutationScope(connection, context, true);
    if (initial.ownerId !== current.ownerId || JSON.stringify(initial.householdMemberIds) !== JSON.stringify(current.householdMemberIds)) {
      throw new Error('Kitchen membership changed.');
    }
    const result = await callback({ ...current, kitchenMutationLocked: true });
    await connection.commit();
    return result;
  } catch (error) {
    await connection.rollback();
    throw error;
  }
}

export async function findHouseholdKitchenRows(connection, context) {
  const scope = context.kitchenMutationLocked ? context : await kitchenMutationScope(connection, context);
  const members = scope.householdMemberIds;
  const [rows] = await connection.execute(
    `SELECT * FROM shared_kitchen WHERE owner_id IN (${members.map(() => '?').join(',')}) AND action='IN'
     ORDER BY COALESCE(_updatedDate,_createdDate) DESC,_id,owner_id${context.kitchenMutationLocked ? ' FOR UPDATE' : ''}`, members);
  return rows;
}

export async function updateHouseholdKitchenRow(connection, context, rowId, fields) {
  const actor = actorFor(context);
  const entries = Object.entries(fields);
  if (!entries.length || entries.some(([key]) => !/^[a-zA-Z_]+$/.test(key))) throw new Error('Invalid kitchen update.');
  // Revalidate membership in the UPDATE too, including callers outside a transaction.
  // The old empty-members fallback could omit the owner predicate entirely.
  const [result] = await connection.execute(`UPDATE \`shared_kitchen\` k
    JOIN new_users member ON k.owner_id=CONVERT(member.user_id USING utf8mb4) COLLATE utf8mb4_unicode_ci
    JOIN new_users actor ON actor.user_id=? AND (member.user_id=actor.user_id OR
      (COALESCE(NULLIF(member.owner_id,''),member.user_id)=COALESCE(NULLIF(actor.owner_id,''),actor.user_id)))
    SET ${entries.map(([key]) => 'k.`' + key + '`=?').join(',')}, k._updatedDate=NOW()
    WHERE k._id=? AND k.action='IN'
      AND (? IS NULL OR ?=actor.user_id OR ?=COALESCE(NULLIF(actor.owner_id,''),actor.user_id))`,
    [actor, ...entries.map(([, value]) => value), rowId, context.ownerId || null, context.ownerId || null, context.ownerId || null]);
  if (Number(result?.affectedRows || 0) < 1) throw new Error('Kitchen item changed or is no longer in your household.');
}
