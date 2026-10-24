"""Exact concept/alias matching; never fuzzy-match products to similar words."""
import hashlib
import re
import unicodedata

STYLE = "trepo-grocery-v1"
APP_STYLE = "trepo-app-bold-outline-v1"
SUPPORTED_STYLES = (STYLE, APP_STYLE)

def validate_style(style):
    if style not in SUPPORTED_STYLES: raise ValueError("Unsupported icon style")
    return style

MODEL_REVISION = "flux2-klein-4b-4bit-860e8718"

# These names reuse the app's existing approved artwork. New aliases can be
# added without changing an image. Style changes require a new STYLE namespace.
SEEDS = {
    "banana": ("gi_bananas", ["bananas"]),
    "carrot": ("gi_carrot", ["carrots"]),
    "egg": ("gi_eggs", ["eggs"]),
    "milk": ("gi_milk", []),
    "bread": ("gi_bread", []),
    "apple": ("gi_apple", ["apples"]),
    "broccoli": ("gi_broccoli", []),
    "cheese": ("gi_cheese", []),
    "fish": ("gi_fish", []),
    "tomato": ("gi_tomato", ["tomatoes"]),
    "avocado": ("gi_avocado", ["avocados"]),
    "cereal": ("gi_cereal", []),
    "yogurt": ("gi_yogurt", ["yoghurt"]),
    "olive oil": ("gi_olive_oil", []),
    "mushroom": ("gi_mushroom", ["mushrooms"]),
    "onion": ("gi_onion", ["onions"]),
    "bell pepper": ("gi_pepper", ["bell peppers", "capsicum"]),
    "grape": ("gi_grapes", ["grapes"]),
    "berry": ("gi_berries", ["berries"]),
}
ALIASES = {alias: term for term, (_, aliases) in SEEDS.items() for alias in [term, *aliases]}
ALIASES.update({"paper towels": "paper towel", "kitchen roll": "paper towel",
                "dishwashing liquid": "dish soap", "dishwashing soap": "dish soap",
                "lemons": "lemon"})


def term(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        raise ValueError("An item name must be between 1 and 240 characters")
    value = unicodedata.normalize("NFKC", value).casefold()
    if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in value):
        raise ValueError("Invalid item name")
    value = re.sub(r"\s+", " ", value.replace("-", " ")).strip()
    if not value:
        raise ValueError("An item name is required")
    return value


def concept(value):
    value = term(value)
    # Only a short generic English noun phrase may enter the image prompt or
    # public metadata; raw product titles, instructions and URLs never do.
    if not re.fullmatch(r"[a-z]+(?: [a-z]+){0,5}", value) or len(value) > 64:
        raise ValueError("Invalid generic concept")
    if value in {"unknown", "item", "food", "grocery", "groceries", "leftovers", "none"}:
        raise ValueError("The item has not been identified yet")
    return ALIASES.get(value, value)


def key(kind, value, style=STYLE):
    return f"{kind}#{style}#{hashlib.sha256(term(value).encode()).hexdigest()}"


def asset_key(value, image_sha, style=STYLE):
    # Content-addressed as well as style-addressed: a published URL is immutable.
    return f"icons/{style}/{hashlib.sha256(concept(value).encode()).hexdigest()}/{image_sha}.png"
