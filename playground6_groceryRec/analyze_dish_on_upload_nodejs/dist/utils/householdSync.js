"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.stableHouseholdRowId = stableHouseholdRowId;
exports.getHouseholdMemberIds = getHouseholdMemberIds;
const crypto_1 = __importDefault(require("crypto"));
function sanitizeUserId(userId) {
    return String(userId || "").replace(/[^a-zA-Z0-9_-]/g, "");
}
function stableHouseholdRowId(namespace, ...parts) {
    const seed = `${namespace}:${parts.map((part) => String(part ?? "")).join(":")}`;
    const hash = crypto_1.default.createHash("sha256").update(seed).digest("hex").slice(0, 32);
    return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20, 32)}`;
}
async function getHouseholdMemberIds(connection, actingUserId) {
    const safeUserId = sanitizeUserId(actingUserId);
    if (!safeUserId) {
        return [];
    }
    const [userRows] = await connection.execute("SELECT owner_id FROM new_users WHERE user_id = ? LIMIT 1", [safeUserId]);
    const householdId = Array.isArray(userRows) && userRows.length > 0
        ? String(userRows[0].owner_id || "")
        : "";
    if (!householdId) {
        return [safeUserId];
    }
    const [memberRows] = await connection.execute("SELECT user_id FROM new_users WHERE owner_id = ? ORDER BY created_at ASC, user_id ASC", [householdId]);
    const memberIds = (memberRows || [])
        .map((row) => sanitizeUserId(row.user_id))
        .filter(Boolean);
    return memberIds.length > 0 ? Array.from(new Set(memberIds)) : [safeUserId];
}
//# sourceMappingURL=householdSync.js.map