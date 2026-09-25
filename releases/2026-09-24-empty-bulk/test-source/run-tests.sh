#!/bin/bash
set -euo pipefail
test_bundle_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export NODE_PATH="$test_bundle_dir/test-dependencies:$test_bundle_dir/candidates/capture/node_modules"
cd "$test_bundle_dir/candidates/capture"
exec node --experimental-test-module-mocks --test --test-reporter=tap test/*.test.cjs test/*.test.mjs ../../tests/*.test.cjs
