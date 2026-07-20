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


# The team lead's MUST-PASS positive set (exact product names). Each is a spice the
# model mislabeled with a VALID enum ('prepared_other'/'pantry') — the passthrough
# case the hard-override rescues BEFORE the enum check.
SPICE_MUST_PASS_POSITIVE = [
    "Black Salt", "Garam Masala", "Chaat Masala", "Coarse Salt", "White Pepper",
    "Black Pepper", "Sichuan Peppers", "Morton Salt", "Kosher Salt", "Himalayan Salt",
    "Diamond Crystal Iodized Salt", "Good & Gather Ground Cumin", "Turmeric",
    "Smoked Paprika", "Badia Cayenne Pepper", "Stonemill Black Pepper", "Nutmeg",
    "Chili Powder", "Kinder's Seasoning Blend",
]

# The 10 real fleet FALSE POSITIVES the greedy version produced — spice/salt/pepper
# words used as flavor DESCRIPTORS on prepared foods. These must NEVER become spices.
SPICE_MUST_PASS_NEGATIVE = [
    "Great Value Black Beans No Salt Added",
    "Ithaca Hummus Olive Oil Sea Salt",
    "Bumble Bee Wild Caught Tuna Lemon Pepper",
    "Trader Joe's Organic Garbanzo Beans No Salt",
    "Whole Kernel Golden Corn No Salt Added",
    "Del Monte Sweet Corn Cream Style No Salt Added",
    "365 Organic Cannellini Beans No Salt",
    "Vigo Yellow Rice Saffron",
    "Aldi Greek Chickpeas With Parsley & Cumin",
    "Kinder's Crispy Fried Onions",
]


def test_spices_hard_override_passthrough_positive():
    # Each must resolve to 'spices' even though the model emitted a VALID passthrough
    # enum ('prepared_other'/'pantry') — the exact bug class from owner d303a754.
    for name in SPICE_MUST_PASS_POSITIVE:
        assert N("prepared_other", None, name) == "spices", f"{name!r} (raw=prepared_other) should be spices"
        assert N("pantry", None, name) == "spices", f"{name!r} (raw=pantry) should be spices"


def test_spices_override_precision_no_false_positives():
    # PRECISION-FIRST: the 10 real fleet false positives must NOT be reclassified as
    # spices. With a valid passthrough enum the override must DECLINE, leaving the row
    # on its stored category (here 'prepared_other'), never 'spices'.
    for name in SPICE_MUST_PASS_NEGATIVE:
        got = N("prepared_other", None, name)
        assert got != "spices", f"{name!r} must NOT become spices (got {got})"
        assert got == "prepared_other", f"{name!r} should stay on its passthrough enum (got {got})"


# Round-3 fleet FALSE POSITIVES (the ~5% tail on 692 real names) — each must NOT be
# reclassified to spices: garlic-clove produce, ready-meal masalas, and spice/salt words
# used as descriptors/ingredients on prepared foods (fries/carrots/peas/seeds/couscous/
# broth/vinegar/syrup/alfredo/spread/ghee/thins/seaweed/chick peas), incl. "No Added Salt".
SPICE_ROUND3_NEGATIVE = [
    # garlic cloves = produce (bare/ground/whole cloves stay spices — see positives)
    "Garlic Cloves", "Garlic clove", "Pickled Garlic Cloves", "Frozen garlic cloves",
    # ready-meal masalas / masala dishes (bread/tea/noodles) — NOT the spice
    "Trader Joe's Paneer Tikka Masala", "Maya Kaimal Tikka Masala",
    "Trader Joe's Vegan Tikka Masala", "Tasty Bite Organic Channa Masala",
    "Trader Joe's Channa Masala", "Masala Noodles", "Masala Roti", "Masala Chai",
    # food nouns carrying salt/spice as a descriptor
    "Sweet Potato Fries Sea Salt", "Sliced Carrots with Sea Salt",
    "Green Peas No added salt", "Pumpkin Seeds Sea Salt", "Pepitas with Sea Salt",
    "Seaweed Snacks Sea Salt", "Ghee Himalayan Pink Salt", "Good Thins Simply Salt",
    "Turmeric Pearl Couscous", "Maple Syrup Cardamom", "Alfredo Sauce Paprika",
    "Orange & Cloves Spread", "Bone Broth with Turmeric",
    "Apple Cider Vinegar with Turmeric", "Sliced Carrots No Added Salt",
    "Goya Chick Peas with Sea Salt",
]

# These spice forms MUST survive the round-3 vetoes (the veto must not over-reach).
SPICE_ROUND3_STILL_SPICES = [
    "Ground Cloves", "Whole Cloves", "Cloves", "Garam Masala", "Chaat Masala",
    "Tandoori Masala", "Tikka Masala Seasoning", "Tikka Masala Spice Blend",
    "Chai Spice Blend",
]


def test_spices_override_round3_no_false_positives():
    # The ~5% real-fleet FP tail. Override must DECLINE, leaving the stored enum intact.
    for name in SPICE_ROUND3_NEGATIVE:
        got = N("prepared_other", None, name)
        assert got != "spices", f"{name!r} must NOT become spices (got {got})"
        assert got == "prepared_other", f"{name!r} should stay on its passthrough enum (got {got})"


def test_spices_override_round3_real_spices_survive():
    # Bare/ground/whole cloves, garam/chaat/tandoori masala, and masala SEASONING/BLEND
    # (re-qualified) must still resolve to spices despite the new vetoes.
    for name in SPICE_ROUND3_STILL_SPICES:
        assert N("prepared_other", None, name) == "spices", f"{name!r} should still be spices"


def test_spices_override_prior_negatives_still_hold():
    # All earlier negatives must keep passing (share a token with a spice/salt but are
    # not spices): salted dairy/sweets, fresh chiles, cured meat, tea, cheese, sauce.
    assert N("dairy_eggs", None, "Salted Butter") == "dairy_eggs"
    assert N("snacks_sweets", None, "Salted Caramel") == "snacks_sweets"
    assert N("produce", None, "Bell Pepper") == "produce"
    assert N("produce", None, "Jalapeno") == "produce"
    assert N("produce", None, "Jalapeño") == "produce"
    assert N("meat_seafood", None, "Pepperoni") == "meat_seafood"
    assert N("beverages", None, "Peppermint Tea") == "beverages"
    assert N("dairy_eggs", None, "Pepper Jack Cheese") == "dairy_eggs"
    assert N("pantry", None, "Hot Sauce") == "pantry"
    # salt/sea-salt only counts as a spice when it's the trailing HEAD noun, so these
    # salted-snack names (salt mid-name) never fire:
    assert N("snacks_sweets", None, "Sea Salt Chocolate Almonds") == "snacks_sweets"
    assert N("snacks_sweets", None, "Sea Salt Crackers") == "snacks_sweets"
    assert N("snacks_sweets", None, "Salt Water Taffy") == "snacks_sweets"
    assert N("produce", None, "Poblano Pepper") == "produce"


def test_spices_override_respects_leftovers_and_meat_order():
    # Leftovers still win over the spice override.
    assert N("leftovers", None, "Leftover Garam Masala Chicken") == "leftovers"
    # Meat override runs first: a peppercorn-crusted steak is meat, not spices.
    assert N("prepared_other", None, "Peppercorn Crusted Ribeye Steak") == "meat_seafood"
    assert N("beverages", None, "Pepper Steak") == "meat_seafood"


def test_storage_fallback_and_default():
    assert N(None, "pantry", None) == "pantry"
    assert N(None, "snacks", None) == "snacks_sweets"
    assert N(None, None, None) == "prepared_other"
    assert N("", "", "") == "prepared_other"
