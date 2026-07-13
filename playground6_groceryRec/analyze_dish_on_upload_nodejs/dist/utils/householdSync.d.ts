import mysql from "mysql2/promise";
export declare function stableHouseholdRowId(namespace: string, ...parts: Array<string | number>): string;
export declare function getHouseholdMemberIds(connection: mysql.Connection, actingUserId: string): Promise<string[]>;
