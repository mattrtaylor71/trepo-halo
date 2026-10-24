#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RELIABILITY_MYSQL_PORT:?Set to the disposable local MySQL port 33387}"
[ "$RELIABILITY_MYSQL_PORT" = 33387 ] || { echo 'Refusing a non-fixture database port'; exit 1; }
.venv/bin/python -m pytest -q \
 playground6_groceryRec/meal_plan_generator/tests/test_generation_jobs.py \
 playground6_groceryRec/meal_plan_generator/tests/test_handler_reliability.py \
 playground6_groceryRec/meal_plan_generator/tests/test_publication_guard.py \
 playground6_groceryRec/meal_plan_api/tests/test_plan_visibility.py \
 playground6_groceryRec/family_brain/test_recipe_safety_schema.py \
 playground6_groceryRec/tests/test_recipe_status_readonly.py \
 playground6_groceryRec/tests/test_recipe_status_cold_start.py \
 playground6_groceryRec/saved_recipes_api/tests/test_recovery_reliability.py \
 playground6_groceryRec/kitchen_api/tests/test_deadlock_retry.py \
 obs/ux_failure_monitor/tests/test_outcomes.py \
 ops/cloud-reliability/icons/tests ops/cloud-reliability/release/test_build.py
(cd playground12_voice_ack/trepo-quick-ack && npm test)
node --test ops/cloud-reliability/icons/tests/edge.test.mjs
node --experimental-test-module-mocks --test ops/cloud-reliability/context-projection/reliability.test.mjs
