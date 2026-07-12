import { tableExists, withDbConnection } from "./mysql.mjs";

function normalizeOwnerId(value) {
  return String(value || "").trim();
}

async function getHouseholdMemberIds(connection, ownerId, fallbackUserId) {
  const [memberRows] = await connection.execute(
    "SELECT user_id FROM new_users WHERE owner_id = ? ORDER BY created_at ASC, user_id ASC",
    [ownerId]
  );

  const memberIds = (memberRows || [])
    .map((row) => String(row.user_id || "").replace(/[^a-zA-Z0-9_-]/g, ""))
    .filter(Boolean);

  return memberIds.length > 0 ? Array.from(new Set(memberIds)) : [fallbackUserId];
}

async function resolveShoppingNamespace(connection, candidateIds) {
  if (!(await tableExists(connection, "users"))) {
    return null;
  }

  const cleanCandidateIds = Array.from(
    new Set(
      (candidateIds || [])
        .map((value) => String(value || "").trim())
        .filter(Boolean)
    )
  );

  if (cleanCandidateIds.length === 0) {
    return null;
  }

  const placeholders = cleanCandidateIds.map(() => "?").join(", ");
  const [rows] = await connection.execute(
    `SELECT web_id, auth0_sub
     FROM users
     WHERE auth0_sub IN (${placeholders})
     ORDER BY FIELD(auth0_sub, ${placeholders})
     LIMIT 1`,
    [...cleanCandidateIds, ...cleanCandidateIds]
  );

  if (!rows?.[0]?.web_id) {
    return null;
  }

  return {
    webId: String(rows[0].web_id),
    auth0Sub: String(rows[0].auth0_sub || "")
  };
}

export async function lookupUserContextByOwnerId(ownerId, options = {}) {
  const normalizedOwnerId = normalizeOwnerId(ownerId);
  if (!normalizedOwnerId) {
    const error = new Error("Missing x-owner-id header.");
    error.statusCode = 400;
    throw error;
  }

  return withDbConnection(async (connection) => {
    const [rows] = await connection.execute(
      `SELECT user_id, owner_id, first_name
       FROM new_users
       WHERE owner_id = ? OR user_id = ?
       ORDER BY CASE WHEN user_id = ? THEN 0 ELSE 1 END, created_at ASC
       LIMIT 1`,
      [normalizedOwnerId, normalizedOwnerId, normalizedOwnerId]
    );

    const row = rows?.[0] || null;
    const effectiveOwnerId = String(row?.owner_id || row?.user_id || normalizedOwnerId);
    const fallbackUserId = String(row?.user_id || effectiveOwnerId);
    const householdMemberIds = await getHouseholdMemberIds(connection, effectiveOwnerId, fallbackUserId);
    const shopping = await resolveShoppingNamespace(
      connection,
      [fallbackUserId, effectiveOwnerId, ...householdMemberIds]
    );

    return {
      ownerId: effectiveOwnerId,
      userId: fallbackUserId,
      tableOwnerId: fallbackUserId,
      firstName: String(row?.first_name || "").trim(),
      householdMemberIds,
      householdSize: householdMemberIds.length,
      shoppingNamespace: shopping?.webId || effectiveOwnerId,
      shoppingAuth0Sub: shopping?.auth0Sub || null,
      hasShoppingNamespace: Boolean(shopping?.webId || effectiveOwnerId),
      isFallbackContext: !row,
      contextSource: "device_owner_id"
    };
  }, options);
}
