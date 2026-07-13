#!/usr/bin/env bash
# Drift guard for the recipe-inventory matcher.
#
# The AI matcher `recipe_inventory_llm.py` is deployed as a Lambda layer in TWO stacks
# (playground6_groceryRec + playground_explore). They MUST stay byte-identical so a fix
# lands on every recipe surface at once. This check fails if they diverge.
#
# CANONICAL = playground6_groceryRec copy. To change the matcher: edit the canonical,
# then `cp` it over the explore copy, then re-run this check. Wire this into the deploy
# path / CI so drift can never silently return (it was two drifted copies before 2026-07-13).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANON="$ROOT/playground6_groceryRec/recipe_inventory_layer/python/recipe_inventory_llm.py"
EXPLORE="$ROOT/playground_explore/recipe_inventory_layer/python/recipe_inventory_llm.py"

for f in "$CANON" "$EXPLORE"; do
  [ -f "$f" ] || { echo "MISSING matcher copy: $f"; exit 2; }
done

if diff -q "$CANON" "$EXPLORE" >/dev/null; then
  echo "OK: recipe_inventory_llm.py is in sync across both layers ($(wc -l < "$CANON") lines)."
  exit 0
else
  echo "DRIFT DETECTED: the two recipe_inventory_llm.py copies differ." >&2
  echo "  canonical: $CANON" >&2
  echo "  explore:   $EXPLORE" >&2
  echo "Fix: edit the canonical, then: cp \"$CANON\" \"$EXPLORE\"" >&2
  echo "--- diff (canonical vs explore) ---" >&2
  diff "$CANON" "$EXPLORE" >&2 || true
  exit 1
fi
