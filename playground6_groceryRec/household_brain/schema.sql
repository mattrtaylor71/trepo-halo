-- Additive only. Installation does not enable capture or serving.
CREATE TABLE IF NOT EXISTS household_brain_config (
 singleton TINYINT PRIMARY KEY,
 capture_enabled BOOLEAN NOT NULL DEFAULT FALSE,
 default_serving BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE=InnoDB;
INSERT IGNORE INTO household_brain_config(singleton) VALUES(1);
CREATE TABLE IF NOT EXISTS household_brain_state (
 household_id VARCHAR(64) PRIMARY KEY,
 input_revision BIGINT NOT NULL DEFAULT 1,
 generation BIGINT NOT NULL DEFAULT 1,
 enabled BOOLEAN NOT NULL DEFAULT FALSE,
 paused BOOLEAN NOT NULL DEFAULT FALSE,
 learning_since DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 timezone VARCHAR(64) NOT NULL DEFAULT 'UTC',
 dirty BOOLEAN NOT NULL DEFAULT TRUE,
 updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 INDEX hb_dirty (dirty,updated_at,household_id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS household_brain_contexts (
 household_id VARCHAR(64) NOT NULL,
 audience VARCHAR(64) NOT NULL,
 input_revision BIGINT NOT NULL,
 generation BIGINT NOT NULL,
 body JSON NOT NULL,
 expires_at DATETIME(6) NOT NULL,
 PRIMARY KEY(household_id,audience),
 INDEX hb_expiry(expires_at)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS household_brain_suppressions (
 household_id VARCHAR(64) NOT NULL,
 audience VARCHAR(64) NOT NULL,
 product_key CHAR(64) NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY(household_id,audience,product_key)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS household_brain_events (
 producer VARCHAR(40) NOT NULL,
 event_id VARCHAR(160) NOT NULL,
 household_id VARCHAR(64) NOT NULL,
 actor_id VARCHAR(64) NULL,
 generation BIGINT NOT NULL,
 kind VARCHAR(40) NOT NULL,
 entity_id VARCHAR(200) NOT NULL,
 occurred_at DATETIME(6) NOT NULL,
 payload JSON NOT NULL,
 request_hash CHAR(64) NOT NULL,
 reversed_by VARCHAR(160) NULL,
 received_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY(producer,event_id),
 INDEX hb_events(household_id,generation,occurred_at),
 INDEX hb_actor_events(actor_id,occurred_at)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS household_brain_audit (
 operation_id VARCHAR(64) PRIMARY KEY,
 operator_id VARCHAR(160) NOT NULL,
 household_id VARCHAR(64) NOT NULL,
 action VARCHAR(40) NOT NULL,
 request_hash CHAR(64) NOT NULL,
 request JSON NOT NULL,
 result JSON NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 INDEX hb_audit_household(household_id,created_at)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS household_brain_reversals (
 producer VARCHAR(40) NOT NULL,
 event_id VARCHAR(160) NOT NULL,
 household_id VARCHAR(64) NOT NULL,
 actor_id VARCHAR(64) NOT NULL,
 reversal_id VARCHAR(160) NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY(producer,event_id)
) ENGINE=InnoDB;
