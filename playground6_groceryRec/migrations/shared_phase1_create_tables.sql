-- Shared-table migration Phase 1 — CREATE the 4 shared targets (zero-risk, empty).
-- Schemas mirror the REAL per-user tables (saved_recipes_api._ensure + twilio
-- ensureUserTables) + owner_id tenancy key. owner_id standardized to
-- utf8mb4_0900_ai_ci for join-safety. Reads untouched; dual-writes added later
-- behind per-family flags. saved_recipes: per-owner dedupe unique. recipes /
-- meal_plan: one-row-per-user caches (_id was 'current' for all users) → owner_id
-- becomes the key. metrics: snapshot log (append), owner_id indexed not unique.

CREATE TABLE IF NOT EXISTS `shared_saved_recipes` (
  `_id` VARCHAR(36) NOT NULL,
  `owner_id` VARCHAR(36) COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `_owner` VARCHAR(36) NOT NULL,
  `source_type` VARCHAR(32) NOT NULL DEFAULT 'tiktok',
  `source_url` VARCHAR(1000) NOT NULL,
  `resolved_url` VARCHAR(1000) NOT NULL,
  `resolved_url_hash` CHAR(64) NOT NULL,
  `title` VARCHAR(255) NOT NULL,
  `image_url` VARCHAR(1000) NULL,
  `image_urls` JSON NULL,
  `source_image_url` VARCHAR(1000) NULL,
  `source_image_urls` JSON NULL,
  `image_storage_key` VARCHAR(1000) NULL,
  `ingredients` JSON,
  `instructions` JSON,
  `notes` JSON,
  `raw_caption` TEXT,
  `raw_content` TEXT NULL,
  `extraction_source` VARCHAR(32),
  `author_name` VARCHAR(255),
  `caption_field` VARCHAR(64) NULL,
  `status` ENUM('ready','failed') NOT NULL DEFAULT 'ready',
  `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`_id`),
  UNIQUE KEY `uniq_owner_url_hash` (`owner_id`, `resolved_url_hash`),
  KEY `ix_owner` (`owner_id`),
  KEY `ix_owner_created` (`owner_id`, `_createdDate`),
  KEY `idx_title` (`title`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `shared_recipes` (
  `owner_id` VARCHAR(36) COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `_id` VARCHAR(36) NOT NULL DEFAULT 'current',
  `_owner` VARCHAR(36) NOT NULL,
  `status` ENUM('ready','regenerating','failed','empty') NOT NULL DEFAULT 'empty',
  `kitchen_only` JSON DEFAULT NULL,
  `need_grocery` JSON DEFAULT NULL,
  `error_message` TEXT,
  `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`owner_id`),
  KEY `ix_owner_created` (`owner_id`, `_createdDate`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `shared_meal_plan` (
  `owner_id` VARCHAR(36) COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `_id` VARCHAR(36) NOT NULL DEFAULT 'current',
  `_owner` VARCHAR(36) NOT NULL,
  `status` ENUM('ready','regenerating','failed','empty') NOT NULL DEFAULT 'empty',
  `focus` TEXT,
  `explanation_title` VARCHAR(255) DEFAULT NULL,
  `explanation_paragraph` TEXT,
  `plan` JSON DEFAULT NULL,
  `error_message` TEXT,
  `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `_updatedDate` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`owner_id`),
  KEY `ix_owner_created` (`owner_id`, `_createdDate`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `shared_metrics` (
  `_id` VARCHAR(36) NOT NULL,
  `owner_id` VARCHAR(36) COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `_owner` VARCHAR(36) NOT NULL,
  `_createdDate` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `IQ` INT NOT NULL DEFAULT 75,
  `Points` BIGINT NOT NULL DEFAULT 0,
  `UPF` DECIMAL(5,2) NOT NULL DEFAULT 0.00,
  `harmful_ingredients` INT NOT NULL DEFAULT 0,
  `IQ_what` TEXT,
  `IQ_suggestions` JSON DEFAULT NULL,
  `UPF_what` TEXT,
  `UPF_suggestions` JSON DEFAULT NULL,
  `harmful_ingredients_what` TEXT,
  `harmful_ingredients_suggestions` JSON DEFAULT NULL,
  `kitchen_analysis_status` VARCHAR(32) DEFAULT NULL,
  `kitchen_analysis_content` MEDIUMTEXT,
  `kitchen_analysis_generated_at` DATETIME NULL,
  `kitchen_analysis_error` TEXT,
  PRIMARY KEY (`_id`),
  KEY `ix_owner` (`owner_id`),
  KEY `ix_owner_created` (`owner_id`, `_createdDate`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
