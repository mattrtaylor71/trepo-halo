"""Tests for the kitchen_api category normalizer (Python port of the JS map).
Kept behavior-equivalent to data-access.mjs normalizeKitchenCategory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitchen_api"))
from category_normalizer import normalize_kitchen_category as N, KITCHEN_CATEGORY_ENUM


def test_exact_enum_passthrough():
    for v in KITCHEN_CATEGORY_ENUM:
        assert N(v) == v


def test_freetext_category_maps_to_enum():
    assert N("Condiment") == "pantry"
    assert N("Dairy") == "dairy_eggs"
    assert N("Beverage") == "beverages"
    assert N("Snacks") == "snacks_sweets"
    assert N("Meat") == "meat_seafood"
    assert N("Baking") == "pantry"
    assert N("Bulk grocery scan") == "prepared_other"   # no keyword -> default
    assert N("Frozen") == "prepared_other"
    assert N("Condiments_Sauces") == "pantry"


def test_case_variants_lowercase():
    assert N("Pantry") == "pantry"
    assert N("Produce") == "produce"
    assert N("Leftovers") == "leftovers"


def test_name_inference_when_category_null():
    assert N(None, None, "Jasmine Rice") == "pantry"
    assert N(None, None, "Mild Cheddar") == "dairy_eggs"
    assert N(None, None, "Ground Turkey") == "meat_seafood"
    assert N(None, "fridge", "Mystery") == "prepared_other"


def test_seasonings_are_pantry_not_meat():
    assert N("Seasoning", "pantry", "Cajun Seasoning") == "pantry"
    assert N(None, None, "Fish Sauce") == "pantry"
    assert N(None, None, "Chicken Bouillon") == "pantry"


def test_storage_fallback_and_default():
    assert N(None, "pantry", None) == "pantry"
    assert N(None, "snacks", None) == "snacks_sweets"
    assert N(None, None, None) == "prepared_other"
    assert N("", "", "") == "prepared_other"
