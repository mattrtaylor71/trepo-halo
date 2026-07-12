"""Characterization tests for the per-user WRITE paths of the four families that
get a shared_* target in the migration (saved_recipes, recipes, meal_plan,
metrics). These PIN the exact column contract + fixed-value semantics of the
current per-user INSERTs so that the Phase-1 dual-write (writing the SAME values
to shared_* + owner_id) is provably behavior-preserving. If a per-user INSERT
changes, these fail — forcing the dual-write to be updated in lock-step.

Value assertions, not target-only (the wave-1 lesson): we assert the ordered
column list AND the fixed literals ('current' cache id, status defaults).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _insert_columns(sql_text, table_marker):
    """Extract the ordered column list from the first INSERT INTO `{table...}`
    matching table_marker in sql_text."""
    # Find "INSERT INTO `{...}` ( col, col, ... )"
    pat = re.compile(r"INSERT INTO\s+`\{" + re.escape(table_marker) + r"[^}]*\}`\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)
    m = pat.search(sql_text)
    assert m, f"no INSERT INTO `{{{table_marker}...}}` found"
    cols = [c.strip().strip("`") for c in m.group(1).split(",") if c.strip()]
    return cols


# ---- saved_recipes: full 21-column recipe row, per-owner dedupe on resolved_url_hash
SAVED_RECIPES_COLUMNS = [
    "_id", "_owner", "source_type", "source_url", "resolved_url", "resolved_url_hash",
    "title", "image_url", "image_urls", "source_image_url", "source_image_urls", "image_storage_key",
    "ingredients", "instructions", "notes", "raw_caption", "raw_content",
    "extraction_source", "author_name", "caption_field", "status",
]


def test_saved_recipes_insert_column_contract():
    src = (ROOT / "saved_recipes_api" / "app.py").read_text()
    cols = _insert_columns(src, "table")
    assert cols == SAVED_RECIPES_COLUMNS, f"saved_recipes INSERT columns drifted: {cols}"
    # resolved_url_hash carries the dedupe identity — dual-write must key on it per-owner.
    assert "resolved_url_hash" in cols
    # status literal defaults to 'ready' in the VALUES tail.
    assert re.search(r"caption_field, status\s*\)\s*VALUES.*'ready'\)", src, re.DOTALL)


# ---- recipes: one-row-per-user cache, _id literally 'current', status 'regenerating'
def test_recipes_is_current_row_cache():
    src = (ROOT / "recipes_generator" / "app.py").read_text()
    assert re.search(r"INSERT INTO\s+`\{table\}`\s*\(_id, _owner, status\)\s*VALUES \('current', %s, 'regenerating'\)", src), \
        "recipes cache INSERT contract drifted (expected _id='current', status='regenerating')"


# ---- meal_plan: one-row-per-user cache, _id 'current', status 'regenerating', + focus
def test_meal_plan_is_current_row_cache():
    src = (ROOT / "meal_plan_generator" / "app.py").read_text()
    assert re.search(r"INSERT INTO\s+`\{table\}`\s*\(_id, _owner, status, focus\)\s*VALUES \('current', %s, 'regenerating', %s\)", src), \
        "meal_plan cache INSERT contract drifted (expected _id='current', status='regenerating', focus)"


# ---- metrics: denormalized snapshot; pin the 17-column contract
METRICS_COLUMNS = [
    "_id", "_owner", "_createdDate", "IQ", "Points", "UPF", "harmful_ingredients",
    "IQ_what", "IQ_suggestions", "UPF_what", "UPF_suggestions",
    "harmful_ingredients_what", "harmful_ingredients_suggestions",
    "kitchen_analysis_status", "kitchen_analysis_content", "kitchen_analysis_generated_at",
    "kitchen_analysis_error",
]


def test_metrics_insert_column_contract():
    src = (ROOT / "kitchen_analysis_generator" / "app.py").read_text()
    cols = _insert_columns(src, "table_name")
    assert cols == METRICS_COLUMNS, f"metrics INSERT columns drifted: {cols}"
    # _createdDate written via NOW() literal in the VALUES tail.
    assert re.search(r"VALUES \(%s, %s, NOW\(\)", src)
