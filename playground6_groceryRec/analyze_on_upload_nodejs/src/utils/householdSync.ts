import mysql from "mysql2/promise";
import crypto from "crypto";

function sanitizeUserId(userId: string): string {
  return String(userId || "").replace(/[^a-zA-Z0-9_-]/g, "");
}

export function stableHouseholdRowId(namespace: string, ...parts: Array<string | number>): string {
  const seed = `${namespace}:${parts.map((part) => String(part ?? "")).join(":")}`;
  const hash = crypto.createHash("sha256").update(seed).digest("hex").slice(0, 32);
  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20, 32)}`;
}

export async function getHouseholdMemberIds(
  connection: mysql.Connection,
  actingUserId: string
): Promise<string[]> {
  const safeUserId = sanitizeUserId(actingUserId);
  if (!safeUserId) {
    return [];
  }

  const [userRows] = await connection.execute<mysql.RowDataPacket[]>(
    "SELECT owner_id FROM new_users WHERE user_id = ? LIMIT 1",
    [safeUserId]
  );
  const householdId = Array.isArray(userRows) && userRows.length > 0
    ? String((userRows[0] as any).owner_id || "")
    : "";

  if (!householdId) {
    return [safeUserId];
  }

  const [memberRows] = await connection.execute<mysql.RowDataPacket[]>(
    "SELECT user_id FROM new_users WHERE owner_id = ? ORDER BY created_at ASC, user_id ASC",
    [householdId]
  );
  const memberIds = (memberRows || [])
    .map((row: any) => sanitizeUserId(row.user_id))
    .filter(Boolean);

  return memberIds.length > 0 ? Array.from(new Set(memberIds)) : [safeUserId];
}
