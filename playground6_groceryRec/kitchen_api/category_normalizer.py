"""Kitchen category normalizer — Python port of the JS
`normalizeKitchenCategory` in
playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs (commit 87eb7c9+).

The image/ios-app capture path writes the identify model's free-text category
guess (e.g. 'Condiment', 'Pantry', 'Bulk grocery scan') straight into
shared_kitchen. The iOS app groups by an EXACT lowercase enum, so free-text /
Title-Case values are invisible. This clamps every kitchen_api write onto the
enum, mirroring the voice path.

⚠️ DUPLICATION: the keyword map below is kept BYTE-EQUIVALENT to the JS map in
data-access.mjs. If you change one, change both. (Follow-up: hoist to a shared
layer so there is a single source of truth.)
"""
import re

KITCHEN_CATEGORY_ENUM = {
    "leftovers", "produce", "dairy_eggs", "meat_seafood",
    "pantry", "spices", "snacks_sweets", "beverages", "prepared_other",
}

# Order matters: PANTRY + SNACKS_SWEETS are scanned before produce/meat/dairy so
# shelf-stable bakery + baked sweets win over incidental produce/meat nouns
# ("Potato Bread"->pantry not produce, "Blueberry Muffins"->snacks not produce,
# "Steak Blend Seasoning"->pantry not meat, "Fish Sauce"->pantry not meat).
# NB: bare "sweet" was REMOVED from snacks (it wrongly caught produce: "Sweet
# Potato/Corn/Onion"); bakery uses "buns" not "bun" (avoids "bunch") and omits bare
# "pie" (avoids "pieces"). Keep byte-equivalent with the JS map.
KITCHEN_CATEGORY_KEYWORDS = [
    (["leftover"], "leftovers"),
    # Spices/seasonings scanned before pantry so "Steak Blend Seasoning" -> spices and
    # "Ground Cinnamon" -> spices (were previously folded into pantry).
    (["spice", "spices", "seasoning", "seasonings", "spice blend", "seasoning blend", "spice rub",
      "rub", "cinnamon", "cumin", "paprika", "oregano", "turmeric", "nutmeg", "cardamom", "cayenne",
      "peppercorn", "chili powder", "garlic powder", "onion powder", "curry powder", "bay leaf"], "spices"),
    (["pantry", "condiment", "broth", "stock", "bouillon",
      "sauce", "marinara", "salsa", "ketchup", "mustard", "mayo", "dressing", "oil", "vinegar",
      "syrup", "honey", "jam", "jelly", "peanut butter", "baking", "flour", "sugar", "rice",
      "pasta", "noodle", "grain", "oat", "cereal", "bean", "lentil", "canned",
      "bread", "bagel", "tortilla", "buns", "roll", "pita", "naan", "english muffin", "wrap"], "pantry"),
    (["dairy", "creamer", "cheese", "cheddar", "mozzarella", "parmesan", "milk", "yogurt",
      "yoghurt", "butter", "cream", "egg"], "dairy_eggs"),
    (["snack", "dessert", "candy", "chocolate", "chip", "cookie", "cracker",
      "pretzel", "popcorn", "granola",
      "muffin", "cake", "pastry", "brownie", "donut", "doughnut", "croissant", "biscuit", "waffle", "pancake"], "snacks_sweets"),
    (["beverage", "drink", "juice", "soda", "coffee", "tea", "water", "kombucha", "lemonade"], "beverages"),
    (["produce", "fruit", "vegetable", "veggie", "lettuce", "spinach", "tomato", "onion",
      "potato", "apple", "banana", "berry", "berries", "corn", "herb", "cilantro"], "produce"),
    (["meat", "seafood", "fish", "poultry", "beef", "pork", "bacon", "sausage", "chicken",
      "turkey", "ham", "salmon", "shrimp", "tuna", "deli"], "meat_seafood"),
    (["prepared", "meal", "entree", "other", "misc"], "prepared_other"),
]

KITCHEN_STORAGE_CATEGORY_FALLBACK = {
    "pantry": "pantry", "produce": "produce", "snacks": "snacks_sweets",
}


def _scan_kitchen_category_keywords(value):
    cleaned = re.sub(r"[^a-z]+", " ", ("" if value is None else str(value)).lower())
    if not cleaned.strip():
        return None
    for keys, category in KITCHEN_CATEGORY_KEYWORDS:
        if any(k in cleaned for k in keys):
            return category
    return None


def normalize_kitchen_category(raw_category, storage_location=None, product_name=None):
    """Map a raw category guess onto the app's category enum, using in order: the
    category guess (exact enum, then keywords), then keywords in the PRODUCT NAME,
    then the storage location, then 'prepared_other'. Never returns None/free-text."""
    raw = ("" if raw_category is None else str(raw_category)).strip().lower()
    # Hard override: tofu/tempeh/seitan/plant-based proteins are a processed soy/plant
    # product — NOT produce/a vegetable and NOT meat. Pin them to prepared_other even
    # when the model confidently guesses 'produce' (user feedback: "this is not a
    # vegetable"). Leftovers (a home dish) still win.
    pname = ("" if product_name is None else str(product_name)).lower()
    if raw != 'leftovers' and re.search(r'\b(tofu|tempeh|seitan)\b', pname):
        return 'prepared_other'
    if raw in KITCHEN_CATEGORY_ENUM:
        return raw
    return (
        _scan_kitchen_category_keywords(raw)
        or _scan_kitchen_category_keywords(product_name)
        or KITCHEN_STORAGE_CATEGORY_FALLBACK.get(("" if storage_location is None else str(storage_location)).strip().lower())
        or "prepared_other"
    )
