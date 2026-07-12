import mysql from "mysql2/promise";

const ensuredTables = new Set();
let pool = null;

function hasDbConfig(env) {
  return Boolean(env?.DB_HOST && env?.DB_USER && env?.DB_PASS && env?.DB_NAME);
}

function getPool(env) {
  if (!pool) {
    pool = mysql.createPool({
      host: env.DB_HOST,
      port: env.DB_PORT ? parseInt(env.DB_PORT, 10) : 3306,
      user: env.DB_USER,
      password: env.DB_PASS,
      database: env.DB_NAME,
      waitForConnections: true,
      connectionLimit: 2,
      maxIdle: 1,
      idleTimeout: 60000,
      queueLimit: 0,
      charset: "utf8mb4",
    });
  }
  return pool;
}

function sanitizeOwner(owner) {
  return String(owner || "").replace(/[^a-zA-Z0-9_-]/g, "");
}

function triageTableName(owner) {
  const safe = sanitizeOwner(owner);
  if (!safe) return null;
  return `${safe}_triage`;
}

function isoToMysqlDateTime(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString().slice(0, 19).replace("T", " ");
}

function normalizeVoiceStatus(rawStatus) {
  const normalized = String(rawStatus || "").trim().toLowerCase();
  if (normalized === "processing") return "processing";
  if (normalized === "completed" || normalized === "done") return "completed";
  if (normalized === "failed") return "failed";
  return "accepted";
}

async function ensureTriageTable(env, tableName) {
  if (!hasDbConfig(env) || !tableName || ensuredTables.has(tableName)) return;
  const sql = `
    CREATE TABLE IF NOT EXISTS \`${tableName}\` (
      \`_id\` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      \`_owner\` VARCHAR(255) NOT NULL,
      \`_device\` VARCHAR(255) DEFAULT NULL,
      \`user_id\` VARCHAR(255) DEFAULT NULL,
      \`job_id\` VARCHAR(100) NOT NULL,
      \`source\` VARCHAR(32) NOT NULL,
      \`job_type\` VARCHAR(64) DEFAULT NULL,
      \`status_normalized\` VARCHAR(32) NOT NULL,
      \`raw_status\` VARCHAR(64) DEFAULT NULL,
      \`action\` VARCHAR(64) DEFAULT NULL,
      \`title\` VARCHAR(500) DEFAULT NULL,
      \`detail\` TEXT,
      \`error_message\` TEXT,
      \`s3_key\` VARCHAR(500) DEFAULT NULL,
      \`session_id\` VARCHAR(255) DEFAULT NULL,
      \`response_text\` TEXT,
      \`table_source\` VARCHAR(64) DEFAULT NULL,
      \`created_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      \`updated_at\` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      \`accepted_at\` DATETIME DEFAULT NULL,
      \`processing_started_at\` DATETIME DEFAULT NULL,
      \`completed_at\` DATETIME DEFAULT NULL,
      \`failed_at\` DATETIME DEFAULT NULL,
      \`metadata\` JSON DEFAULT NULL,
      PRIMARY KEY (\`_id\`),
      UNIQUE KEY \`uniq_job_id\` (\`job_id\`),
      KEY \`idx_owner_created\` (\`_owner\`, \`created_at\`),
      KEY \`idx_owner_updated\` (\`_owner\`, \`updated_at\`),
      KEY \`idx_status_updated\` (\`status_normalized\`, \`updated_at\`),
      KEY \`idx_source_updated\` (\`source\`, \`updated_at\`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `;
  await getPool(env).execute(sql);
  ensuredTables.add(tableName);
}

export function buildVoiceTriageRecord(job, overrides = {}) {
  if (!job?.job_id || !job?.owner_id) return null;
  const rawStatus = overrides.raw_status || job.status || "accepted";
  const normalizedStatus = overrides.status_normalized || normalizeVoiceStatus(rawStatus);
  const createdAt = overrides.created_at || job.accepted_at || job.updated_at || null;
  const updatedAt = overrides.updated_at || job.updated_at || createdAt;
  const transcript = typeof job.transcript === "string" ? job.transcript.trim() : "";
  const responseText = typeof job.response_text === "string" ? job.response_text.trim() : "";
  const detail = overrides.detail || [
    transcript ? `transcript: ${transcript}` : null,
    responseText && responseText !== transcript ? `response: ${responseText}` : null,
  ].filter(Boolean).join(" | ");

  return {
    owner: job.owner_id,
    device_id: overrides.device_id || job.device_id || null,
    user_id: overrides.user_id || null,
    job_id: job.job_id,
    source: "voice",
    job_type: "voice",
    status_normalized: normalizedStatus,
    raw_status: rawStatus,
    action: overrides.action || null,
    title: overrides.title || transcript || responseText || "Voice request",
    detail: detail || null,
    error_message: overrides.error_message || job.last_error || null,
    s3_key: overrides.s3_key || job.s3_key || null,
    session_id: overrides.session_id || job.session_id || null,
    response_text: overrides.response_text || responseText || null,
    table_source: overrides.table_source || "voice_async",
    created_at: createdAt,
    updated_at: updatedAt,
    accepted_at: overrides.accepted_at || job.accepted_at || createdAt,
    processing_started_at: overrides.processing_started_at || job.processing_started_at || null,
    completed_at: overrides.completed_at || (normalizedStatus === "completed" ? (job.completed_at || updatedAt) : null),
    failed_at: overrides.failed_at || (normalizedStatus === "failed" ? (job.failed_at || updatedAt) : null),
    metadata: {
      response_surface: job.response_surface || null,
      content_type: job.content_type || null,
      audio_format: job.audio_format || null,
      audio_sample_rate: job.audio_sample_rate || null,
      audio_bytes: job.audio_bytes || null,
      enqueue_count: job.enqueue_count || 0,
      worker_attempts: job.worker_attempts || 0,
    },
  };
}

export async function upsertVoiceTriageRecord(env, record) {
  if (!record || !hasDbConfig(env)) return;
  const tableName = triageTableName(record.owner);
  if (!tableName) return;
  await ensureTriageTable(env, tableName);
  const sql = `
    INSERT INTO \`${tableName}\` (
      _owner, _device, user_id, job_id, source, job_type, status_normalized, raw_status,
      action, title, detail, error_message, s3_key, session_id, response_text, table_source,
      created_at, updated_at, accepted_at, processing_started_at, completed_at, failed_at, metadata
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON DUPLICATE KEY UPDATE
      _device = COALESCE(VALUES(_device), _device),
      user_id = COALESCE(VALUES(user_id), user_id),
      source = VALUES(source),
      job_type = COALESCE(VALUES(job_type), job_type),
      status_normalized = VALUES(status_normalized),
      raw_status = COALESCE(VALUES(raw_status), raw_status),
      action = COALESCE(VALUES(action), action),
      title = COALESCE(VALUES(title), title),
      detail = COALESCE(VALUES(detail), detail),
      error_message = COALESCE(VALUES(error_message), error_message),
      s3_key = COALESCE(VALUES(s3_key), s3_key),
      session_id = COALESCE(VALUES(session_id), session_id),
      response_text = COALESCE(VALUES(response_text), response_text),
      table_source = COALESCE(VALUES(table_source), table_source),
      updated_at = COALESCE(VALUES(updated_at), updated_at),
      accepted_at = COALESCE(VALUES(accepted_at), accepted_at),
      processing_started_at = COALESCE(VALUES(processing_started_at), processing_started_at),
      completed_at = COALESCE(VALUES(completed_at), completed_at),
      failed_at = COALESCE(VALUES(failed_at), failed_at),
      metadata = COALESCE(VALUES(metadata), metadata)
  `;
  await getPool(env).execute(sql, [
    record.owner,
    record.device_id,
    record.user_id,
    record.job_id,
    record.source,
    record.job_type,
    record.status_normalized,
    record.raw_status,
    record.action,
    record.title,
    record.detail,
    record.error_message,
    record.s3_key,
    record.session_id,
    record.response_text,
    record.table_source,
    isoToMysqlDateTime(record.created_at),
    isoToMysqlDateTime(record.updated_at),
    isoToMysqlDateTime(record.accepted_at),
    isoToMysqlDateTime(record.processing_started_at),
    isoToMysqlDateTime(record.completed_at),
    isoToMysqlDateTime(record.failed_at),
    record.metadata ? JSON.stringify(record.metadata) : null,
  ]);
}

export async function syncVoiceTriage(env, job, overrides = {}) {
  const record = buildVoiceTriageRecord(job, overrides);
  if (!record) return;
  await upsertVoiceTriageRecord(env, record);
}
