CREATE TABLE IF NOT EXISTS meal_plan_generation_jobs (
  scope_id VARCHAR(64) NOT NULL PRIMARY KEY,
  run_token CHAR(36) NULL,
  state VARCHAR(16) NOT NULL,
  payload JSON NULL,
  input_hash CHAR(64) NULL,
  attempts INT NOT NULL DEFAULT 0,
  due_at DATETIME NULL,
  lease_until DATETIME NULL,
  last_error VARCHAR(300) NULL,
  updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  INDEX retry_due (state,due_at),
  INDEX expired_lease (state,lease_until)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS meal_plan_publications (
  owner_id VARCHAR(64) NOT NULL PRIMARY KEY,
  household_id VARCHAR(64) NOT NULL,
  members_hash CHAR(64) NOT NULL,
  dietary_hash CHAR(64) NOT NULL,
  plan_hash CHAR(64) NOT NULL,
  published_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
