# Existing deployed recipe eligibility rules, extracted without semantic changes.
# Source RecipesGeneratorFunction package baseline recorded in founder-brain evidence.
import re
import json

_DIETARY_PREF_KEYS = ('allergies', 'diets', 'religious', 'health', 'custom')

_PLANT_BUTTER = r"(peanut|almond|cashew|sunflower|seed|nut|soy|plant|vegan|coconut)\s*butter"

_PLANT_MILK = r"(coconut|almond|oat|soy|cashew|rice|hemp|plant|non-?dairy|nut)\s*milk"

def _scrub_dairy(b):
    b = re.sub(_PLANT_BUTTER, "", b)
    b = re.sub(_PLANT_MILK, "", b)
    for kill in ("vegan cheese", "dairy-free cheese", "vegan yogurt", "vegan butter",
                 "dairy-free", "plant-based cheese"):
        b = b.replace(kill, "")
    for w in ["cheese", "cheddar", "mozzarella", "parmesan", "parmigiano", "pecorino",
              "feta", "ricotta", "mascarpone", "halloumi", "paneer", "burrata", "brie",
              "gruyere", "gouda", "yogurt", "yoghurt", "ghee", "whey", "casein", "custard",
              "creme fraiche", "crème fraîche", "clotted cream", "condensed milk",
              "evaporated milk", "heavy cream", "double cream", "sour cream",
              "cream cheese", "ice cream", "kefir", "quark", "curd", " milk", " butter",
              "buttermilk"]:
        if w in b:
            return w.strip()
    return None

def _scrub_egg(b):
    return "egg" if (re.search(r"\begg", b) and "eggplant" not in b) else None

_FISH_STEAK = re.compile(
    r'\b(tuna|ahi|salmon|swordfish|halibut|mahi[\s-]?mahi|cod|marlin|shark|fish)\s+steak', re.I)

def _scrub_meat(b):
    b = _FISH_STEAK.sub(' ', b)
    for w in ["chicken", "beef", "bacon", " pork", "sausage", "turkey", "lamb", "steak",
              "ham ", "salami", "pepperoni", "chorizo", "meatball", "prosciutto", "veal"]:
        if w in b:
            return w.strip()
    return None

def _scrub_seafood(b):
    for w in ["shrimp", "salmon", "fish", "tuna", "cod", "crab", "lobster", "clam", "oyster",
              "anchovy", "tilapia", "scallop", "mussel", "sardine", "prawn", "halibut"]:
        if w in b:
            return w
    return None

def _scrub_fish(b):
    for w in ["salmon", "tuna", "cod", "tilapia", "halibut", "anchovy", "sardine", "trout",
              "bass", "mackerel", "fish"]:
        if w in b:
            return w
    return None

def _scrub_shellfish(b):
    for w in ["shrimp", "crab", "lobster", "clam", "oyster", "mussel", "scallop", "prawn", "crawfish"]:
        if w in b:
            return w
    return None

_ALLERGEN_NEGATIONS = (
    "peanut-free", "peanut free", "nut-free", "nut free", "dairy-free", "dairy free",
    "gluten-free", "gluten free", "egg-free", "egg free", "soy-free", "soy free",
    "sesame-free", "sesame free", "wheat-free", "wheat free", "no peanuts", "no nuts",
)

def _strip_allergen_negations(b):
    """Remove 'peanut-free'/'nut free'/etc BEFORE matching, so a label advertising the
    ABSENCE of an allergen is not read as its presence. Without this, 'peanut-free
    chocolate' is flagged as containing peanuts and a safe recipe is hidden."""
    for phrase in _ALLERGEN_NEGATIONS:
        b = b.replace(phrase, " ")
    return b

def _scrub_peanut(b):
    # "peanut" alone missed the names peanuts actually travel under: groundnut (UK/India/
    # Africa), arachis (the botanical name used on labels and in oils), satay and its
    # spellings, and monkey nut. Every one of these is a real ingredient line on real
    # recipe sites, and each is a hospital visit for an allergic user.
    b = _strip_allergen_negations(b)
    for w in ("peanut", "groundnut", "ground nut", "arachis", "satay", "sate sauce",
              "monkey nut", "beer nut", "goober"):
        if w in b:
            return w
    return None

def _scrub_treenut(b):
    # Beyond the bare nut names, tree nuts hide inside prepared ingredients that never
    # say "nut": marzipan and frangipane are almond, praline/nougat/gianduja/nutella are
    # hazelnut, amaretto is apricot-kernel/almond, and pesto is pine nut unless it
    # explicitly says otherwise. These are the forms that reach a recipe page.
    b = _strip_allergen_negations(b)
    # A peanut is a legume, not a tree nut. Remove peanut forms before the generic
    # "nut butter"/"nut flour" sweep, or "Spicy Peanut Butter Noodles" is withheld from
    # someone whose only allergy is tree nuts. Peanuts remain fully covered by
    # _scrub_peanut for anyone who declares that allergy.
    b = b.replace("peanut butter", " ").replace("peanut flour", " ").replace("peanut", " ")
    for w in ("almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut", "macadamia",
              "brazil nut", "pine nut", "pignoli", "marzipan", "frangipane", "praline",
              "nougat", "gianduja", "nutella", "amaretto", "amaretti", "filbert",
              "chestnut", "nut butter", "nut flour", "nut meal"):
        if w in b:
            return w
    # Pesto is pine nut by default. A pesto that names a different nut is caught above;
    # one that says nut-free has already been stripped by the negation pass.
    if "pesto" in b:
        return "pesto(pine nut)"
    return None

def _scrub_sesame(b):
    for w in ["sesame", "tahini"]:
        if w in b:
            return w
    return None

def _scrub_soy(b):
    b = b.replace("soy-free", "")
    for w in ["soy sauce", "soybean", " soy ", "tofu", "edamame", "tempeh", "miso"]:
        if w in b:
            return w.strip()
    return None

def _scrub_wheat(b):
    hits = []
    for w in ["wheat", "bread", "flour", "spaghetti", "couscous", "breadcrumb", "cracker", " bun "]:
        if w in b and "gluten-free" not in b:
            hits.append(w.strip())
    if "pasta" in b and "gluten-free" not in b and "rice pasta" not in b:
        hits.append("pasta")
    if "noodle" in b and "rice noodle" not in b and "gluten-free" not in b:
        hits.append("noodle")
    if "tortilla" in b and "corn tortilla" not in b:
        hits.append("tortilla")
    return hits[0] if hits else None

def _scrub_gluten(b):
    # "noodle" alone is not gluten: zucchini noodles, rice noodles, glass/mung-bean
    # noodles and shirataki are all gluten-free staples of exactly the recipes a
    # gluten-free user wants. Measured: it silently dropped "Avocado Pesto Zucchini
    # Noodles" from a gluten-free feed. Remove the gluten-free noodle forms before the
    # sweep, the same way plant milks are removed before the dairy sweep.
    for gf in ("zucchini noodle", "zoodle", "rice noodle", "glass noodle", "shirataki",
               "sweet potato noodle", "kelp noodle", "veggie noodle", "vegetable noodle",
               "chickpea noodle", "lentil noodle", "buckwheat noodle", "gluten-free noodle",
               "gluten free noodle", "gluten-free pasta", "gluten free pasta",
               "chickpea pasta", "lentil pasta", "rice pasta", "corn pasta"):
        b = b.replace(gf, " ")
    w = _scrub_wheat(b)
    if w:
        return w
    for x in ["barley", "rye", "malt", "seitan", "farro", "bulgur"]:
        if x in b:
            return x
    if "soy sauce" in b and "tamari" not in b and "gluten-free soy" not in b:
        return "soy sauce(wheat)"
    return None

def _scrub_honey(b):
    return "honey" if "honey" in b else None

def _scrub_pork(b):
    for w in ["pork", "bacon", "ham ", "prosciutto", "pancetta", "lard", "gelatin"]:
        if w in b:
            return w.strip()
    return None

def _scrub_alcohol(b):
    for w in ["wine", "beer", " rum", "vodka", "bourbon", "sake", "mirin", "sherry",
              "brandy", "liqueur", "whiskey"]:
        if w in b:
            return w.strip()
    return None

_ALLERGEN_DETECTORS = [
    ('dairy', _scrub_dairy), ('milk', _scrub_dairy),
    ('egg', _scrub_egg),
    ('peanut', _scrub_peanut),
    ('tree nut', _scrub_treenut), ('treenut', _scrub_treenut),
    ('shellfish', _scrub_shellfish),
    ('fish', _scrub_fish),
    ('wheat', _scrub_wheat),
    ('soy', _scrub_soy),
    ('sesame', _scrub_sesame),
    ('gluten', _scrub_gluten),
]

_PROTEIN_SOURCES = (
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'veal', 'venison', 'bacon',
    'ham', 'sausage', 'steak', 'mince', 'ground meat', 'prosciutto', 'chorizo',
    'salmon', 'tuna', 'cod', 'haddock', 'halibut', 'trout', 'sardine', 'anchov',
    'mackerel', 'tilapia', 'shrimp', 'prawn', 'crab', 'lobster', 'scallop', 'mussel',
    'clam', 'squid', 'octopus', 'fish',
    'egg', 'tofu', 'tempeh', 'seitan', 'edamame', 'soy curl',
    'lentil', 'chickpea', 'garbanzo', 'black bean', 'kidney bean', 'pinto bean',
    'white bean', 'cannellini', 'navy bean', 'butter bean', 'split pea', 'bean',
    'greek yogurt', 'yoghurt', 'yogurt', 'cottage cheese', 'ricotta', 'paneer',
    'halloumi', 'cheese', 'milk', 'whey', 'protein powder', 'protein shake',
    'quinoa', 'peanut butter', 'almond butter', 'almond', 'cashew', 'walnut',
    'peanut', 'pumpkin seed', 'hemp seed', 'chia',
)

def _scrub_protein_source(b):
    """Return the protein source found, or None. Inverse of the allergen scrubbers: here
    finding something is GOOD."""
    for term in _PROTEIN_SOURCES:
        if term in b:
            return term
    return None

def _lacks_protein_for_high_protein(recipe, prefs):
    """Reason string if a high-protein preference is set but the recipe has no protein
    source at all, else None.

    Deliberately a floor, not a threshold. Without grams we cannot say a dish is high in
    protein; we can say with confidence that one containing no protein source is not.
    """
    wanted = [str(x).strip().lower() for v in ('health', 'diets')
              for x in (prefs.get(v) or [])]
    if not any('high-protein' in w or 'high protein' in w for w in wanted):
        return None
    b = _recipe_blob(recipe)
    return None if _scrub_protein_source(b) else 'high-protein:no protein source'

_DIET_DETECTORS = {
    'vegetarian': [_scrub_meat, _scrub_seafood],
    'vegan': [_scrub_meat, _scrub_seafood, _scrub_dairy, _scrub_egg, _scrub_honey],
    'pescatarian': [_scrub_meat],
    'dairy-free': [_scrub_dairy],
    'dairy free': [_scrub_dairy],
    'gluten-free': [_scrub_gluten],
    'gluten free': [_scrub_gluten],
}

def _recipe_blob(recipe):
    """Lowercased title + ingredients + missing_ingredients (+ UWIH subtitle and
    detail.matched/missing) for a recipe dict. Robust to missing keys."""
    parts = [str(recipe.get('title', '') or ''), str(recipe.get('name', '') or ''),
             str(recipe.get('subtitle', '') or '')]
    for key in ('ingredients', 'missing_ingredients', 'matched_ingredients'):
        v = recipe.get(key)
        if isinstance(v, list):
            parts += [str(x) for x in v]
    detail = recipe.get('detail')
    if isinstance(detail, dict):
        for key in ('matched_ingredients', 'missing_ingredients'):
            v = detail.get(key)
            if isinstance(v, list):
                parts += [str(x) for x in v]
    return ' '.join(parts).lower()

_SERVING_SUGGESTION_RE = re.compile(
    r'\b(serve[sd]?\s+(with|over|alongside)|serving\s+suggestion|garnish(e[sd])?\s+with|'
    r'for\s+serving|to\s+serve|top\s+with\s+optional|optional[:,]?)\b.*', re.I)

def _instructions_blob(recipe):
    """Instruction text only, with serving suggestions removed."""
    # Meal plans emit `steps`; saved recipes emit `instructions`. Validate both
    # when present so an empty legacy field cannot hide newer generated steps.
    steps = []
    for key in ('instructions', 'steps'):
        value = recipe.get(key)
        if isinstance(value, list):
            steps.extend(str(step) for step in value)
        elif isinstance(value, str):
            steps.append(value)
    return ' \n '.join(_SERVING_SUGGESTION_RE.sub(' ', s) for s in steps).lower()

def _instruction_only_hit(detector, recipe):
    """Run one detector over the INSTRUCTIONS and report a hit only when the instructions
    introduce a food the ingredient list never mentions.

    Why so narrow. Steps refer to ingredients by short name: a recipe listing "vegan
    butter" or "flat rice noodles" says "spread butter" / "cook noodles" in the method.
    The qualifier stripping that keeps those safe works on the PHRASE, so the bare word in
    a step sails past it and the recipe gets dropped - a vegan recipe pulled from a vegan
    user, Pad Thai pulled from a gluten-free user. Both measured on real saved recipes.
    Requiring the food to be absent from the ingredient list keeps the case this was built
    for ("cook pork belly" in a dish whose ingredients list no pork) and discards the rest.

    The whole-word test also fixes a real substring bug: the dairy detector matches bare
    "brie", so the step "dip the rice paper briefly" read as cheese.
    """
    ins = _instructions_blob(recipe)
    if not ins:
        return None
    token = detector(ins)
    if not token:
        return None
    token = token.strip()
    if not token:
        return None
    # Already accounted for by the ingredient list (possibly in a qualified, allowed form).
    if token in _recipe_blob(recipe):
        return None
    if not re.search(r'\b' + re.escape(token) + r'\b', ins):
        return None
    return token

def _pref_names_allergen(pref_text, token):
    """Does this preference actually name this allergen?

    `token in pref_text` looks right and is badly wrong: "fish" is a substring of
    "shellfish", so every shellfish-allergic user also ran the fin-fish detector and lost
    salmon, cod, tuna, halibut and sardines. Measured on a live run: a pescatarian with a
    shellfish allergy had 8 of 15 candidates dropped that way and got an almost entirely
    vegetarian feed.

    Anchoring at a word START keeps plurals and phrasings working ("tree nuts",
    "peanuts", "milk/dairy") while refusing to find "fish" inside "shellfish".
    """
    return re.search(r'\b' + re.escape(token) + r'\w*', pref_text) is not None

def _recipe_violates(recipe, prefs):
    """Return a reason string if the recipe violates any HARD dietary exclusion in
    prefs, else None. prefs = {allergies, diets, religious, health, custom}."""
    b = _recipe_blob(recipe)
    prefs = prefs or {}

    # Allergies (safety-critical hard exclusions)
    for name in prefs.get('allergies', []) or []:
        low = str(name).strip().lower()
        for token, fn in _ALLERGEN_DETECTORS:
            if _pref_names_allergen(low, token):
                hit = fn(b)
                if hit:
                    return f'allergy:{name}={hit}'
                # An allergen named only in a step ("stir in the peanut butter") was
                # invisible to every check, because the blob stops at the ingredient list.
                # Narrow by construction - see _instruction_only_hit.
                hit = _instruction_only_hit(fn, recipe)
                if hit:
                    return f'allergy:{name}={hit}(step)'

    # Exclusion diets
    for name in prefs.get('diets', []) or []:
        low = str(name).strip().lower()
        for diet_key, fns in _DIET_DETECTORS.items():
            if diet_key == low or diet_key in low.replace(' ', '-'):
                for fn in fns:
                    hit = fn(b)
                    if hit:
                        return f'diet:{name}={hit}'
                    hit = _instruction_only_hit(fn, recipe)
                    if hit:
                        return f'diet:{name}={hit}(step)'
                break

    # Religious rules
    for name in prefs.get('religious', []) or []:
        low = str(name).strip().lower()
        if 'kosher' in low:
            p = _scrub_pork(b)
            if p:
                return f'kosher:pork={p}'
            s = _scrub_shellfish(b)
            if s:
                return f'kosher:shellfish={s}'
            m = _scrub_meat(b)
            d = _scrub_dairy(b)
            if m and d:
                return f'kosher:meat+dairy={m}+{d}'
        if 'halal' in low:
            p = _scrub_pork(b)
            if p:
                return f'halal:pork={p}'
            a = _scrub_alcohol(b)
            if a:
                return f'halal:alcohol={a}'

    # Quantitative preferences: only the high-protein FLOOR is checkable without
    # nutrition data. low-sodium / low-sugar / low-carb / keto stay prompt-only, because
    # guessing at them from an ingredient list would drop good recipes on a hunch.
    lacks = _lacks_protein_for_high_protein(recipe, prefs)
    if lacks:
        return lacks

    # Custom "no X" / "avoid X" / "no more X" -> substring match on X
    for entry in prefs.get('custom', []) or []:
        term = str(entry).strip().lower()
        for prefix in ('no more ', 'avoid ', 'no '):
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                break
        if term and term in b:
            return f'custom:{entry}={term}'

    # Health entries are soft ("tailor toward"), not hard exclusions -> skip.
    return None

def _format_preferences_block(prefs):
    """Render the DIETARY CONSTRAINTS prompt block. Empty categories are omitted; returns ''
    when there are NO preferences at all (so the prompt is byte-for-byte unchanged)."""
    prefs = prefs or {}
    if not any(prefs.get(k) for k in _DIETARY_PREF_KEYS):
        return ''
    lines = ['DIETARY CONSTRAINTS (every recipe MUST comply):']
    if prefs.get('allergies'):
        lines.append('- ALLERGIES — NEVER include these or any derivative/trace '
                     '(safety-critical): ' + ', '.join(prefs['allergies']))
    if prefs.get('diets'):
        lines.append('- DIET — recipes MUST be: ' + ', '.join(prefs['diets']))
    if prefs.get('religious'):
        lines.append('- RELIGIOUS: ' + ', '.join(prefs['religious']))
        _religious_low = ' '.join(str(r).lower() for r in prefs['religious'])
        if 'kosher' in _religious_low:
            lines.append('  For KOSHER: no pork or shellfish, and never combine meat and '
                         'dairy in the same recipe.')
        if 'halal' in _religious_low:
            lines.append('  For HALAL: no pork or pork derivatives (bacon, ham, lard, '
                         'gelatin) and no alcohol including wine, beer, or cooking wine.')
    if prefs.get('health'):
        lines.append('- HEALTH — tailor toward (informational, not medical advice): '
                     + ', '.join(prefs['health']))
    if prefs.get('custom'):
        lines.append("- ALSO AVOID/HONOR (user's own words): " + ', '.join(prefs['custom']))
    lines.append(
        'INGREDIENT-LEVEL CHECK: verify EVERY ingredient in each recipe against the '
        'constraints above, including staples pulled from the kitchen. Do not assume a '
        'staple is compliant just because it is common. Butter, milk, cream, cheese, '
        'yogurt, and eggs are NOT vegan or dairy-free (use plant milk, vegan butter, or '
        'omit them); honey is not vegan; regular soy sauce, teriyaki, most bread, pasta, '
        'flour, and breadcrumbs contain gluten (use tamari or certified gluten-free '
        'versions). Substitute any non-compliant ingredient with a compliant alternative '
        'or leave it out.')
    lines.append('Any recipe that violates an ALLERGY or DIET is unacceptable — omit it '
                 'entirely and generate a compliant one instead.')
    return '\n'.join(lines)
