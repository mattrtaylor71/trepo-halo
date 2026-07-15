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


def test_seasonings_are_spices_not_meat():
    # Seasonings/spice-blends route to the `spices` enum (added in the spices split;
    # scanned before pantry). Previously this asserted "pantry" — stale since the
    # spices category was introduced. The point is they must never be meat.
    assert N("Seasoning", "pantry", "Cajun Seasoning") == "spices"
    assert N(None, None, "Fish Sauce") == "pantry"
    assert N(None, None, "Chicken Bouillon") == "pantry"


def test_raw_steak_cuts_are_meat_not_beverages():
    # THE live bug: the Gemini bulk path mislabeled raw steaks as 'beverages'; the
    # exact-enum value passed straight through. The strong-meat override must rescue
    # it — even when the AI category guess is 'beverages'/'Beverage'.
    assert N("beverages", "fridge", "Ribeye Steak") == "meat_seafood"
    assert N("Beverage", "fridge", "Steak") == "meat_seafood"
    assert N("beverages", None, "NY Strip Steak") == "meat_seafood"
    assert N("beverages", None, "Sirloin") == "meat_seafood"
    assert N("beverages", None, "Pork steak") == "meat_seafood"
    assert N("beverages", None, "Ham Steaks") == "meat_seafood"
    assert N("beverages", None, "Skirt Steak") == "meat_seafood"
    # name-derived (raw category null) still lands on meat via keyword + override
    assert N(None, None, "Steak") == "meat_seafood"
    assert N(None, None, "Sirloin") == "meat_seafood"
    assert N(None, None, "Beef Brisket") == "meat_seafood"


def test_steak_controls_do_not_regress():
    # Sauces/seasonings must NOT be pulled into meat (guard + keyword order).
    assert N(None, None, "Steak Seasoning") == "spices"
    assert N(None, None, "Steak Sauce") == "pantry"
    assert N("spices", None, "Steak Seasoning") == "spices"
    # Beverages/broths without a cut word stay put — override must not over-fire.
    assert N("beverages", None, "Protein Shake") == "beverages"
    assert N(None, None, "Chicken Broth") == "pantry"
    assert N("beverages", None, "Beef Broth") == "beverages"   # broth guard -> no override; trusts enum
    # A leftover steak dish stays leftovers (leftovers wins over the meat override).
    assert N("leftovers", None, "Leftover Ribeye Steak") == "leftovers"


def test_storage_fallback_and_default():
    assert N(None, "pantry", None) == "pantry"
    assert N(None, "snacks", None) == "snacks_sweets"
    assert N(None, None, None) == "prepared_other"
    assert N("", "", "") == "prepared_other"
