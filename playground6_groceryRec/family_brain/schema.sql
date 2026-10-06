-- Additive founder pilot only. Run once with the deployment operator, never on an app request.
CREATE TABLE IF NOT EXISTS family_brain_subjects (
 actor_id VARCHAR(64) PRIMARY KEY,
 household_id VARCHAR(64) NOT NULL,
 enabled BOOLEAN NOT NULL DEFAULT FALSE,
 paused BOOLEAN NOT NULL DEFAULT FALSE,
 generation BIGINT NOT NULL DEFAULT 1,
 revision BIGINT NOT NULL DEFAULT 1,
 learning_since DATETIME(6) NOT NULL,
 facts JSON NOT NULL,
 updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 INDEX idx_brain_household (household_id, enabled)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS family_brain_controls (
 actor_id VARCHAR(64) NOT NULL,
 operation_id VARCHAR(64) NOT NULL,
 request_hash CHAR(64) NOT NULL,
 result_json JSON NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY (actor_id, operation_id)
) ENGINE=InnoDB;
