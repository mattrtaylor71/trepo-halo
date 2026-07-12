import mysql from "mysql2/promise";

let pool = null;

export function getDbConfig(env = process.env) {
  const missing = ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"].filter((key) => !env?.[key]);
  if (missing.length > 0) {
    throw new Error(`Missing DB env vars: ${missing.join(", ")}`);
  }

  const port = Number(env.DB_PORT || 3306);
  if (!Number.isFinite(port)) {
    throw new Error(`Invalid DB_PORT value: ${env.DB_PORT}`);
  }

  return {
    host: env.DB_HOST,
    port,
    user: env.DB_USER,
    password: env.DB_PASS,
    database: env.DB_NAME,
    charset: "utf8mb4",
    waitForConnections: true,
    connectionLimit: Number(env.DB_POOL_CONNECTION_LIMIT || 20),
    queueLimit: Number(env.DB_POOL_QUEUE_LIMIT || 100),
    connectTimeout: Number(env.DB_CONNECT_TIMEOUT_MS || 5000)
  };
}

export function getDbPool(env = process.env) {
  if (!pool) {
    pool = mysql.createPool(getDbConfig(env));
  }

  return pool;
}

export async function withDbConnection(callback, options = {}) {
  const connection = await getDbPool(options.env).getConnection();

  try {
    return await callback(connection);
  } finally {
    connection.release();
  }
}

export function sanitizeIdentifier(value, label = "identifier") {
  const safeValue = String(value || "").replace(/[^a-zA-Z0-9_-]/g, "");
  if (!safeValue) {
    throw new Error(`Invalid ${label}.`);
  }

  return safeValue;
}

export async function tableExists(connection, tableName) {
  const [rows] = await connection.execute(
    `SELECT COUNT(*) AS count
     FROM information_schema.tables
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName]
  );

  return Boolean(rows?.[0]?.count);
}

export async function getTableColumns(connection, tableName) {
  const [rows] = await connection.execute(
    `SELECT column_name
     FROM information_schema.columns
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName]
  );

  return new Set(
    (rows || [])
      .map((row) => row.column_name || row.COLUMN_NAME)
      .filter(Boolean)
  );
}

export async function ensureColumn(connection, tableName, columnName, ddl) {
  const columns = await getTableColumns(connection, tableName);
  if (columns.has(columnName)) {
    return;
  }

  await connection.execute(`ALTER TABLE \`${tableName}\` ADD COLUMN ${ddl}`);
}

export function parseJsonColumn(value, fallback = []) {
  if (!value) {
    return fallback;
  }

  if (typeof value === "object") {
    return value;
  }

  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}
