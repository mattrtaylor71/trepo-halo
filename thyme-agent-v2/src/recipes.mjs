import { Fault, hash } from "./core.mjs";
const text = (x, max = 3000) => {
  if (typeof x !== "string" || !x.trim() || x.length > max)
    throw new Fault("recipe_invalid", "Recipe text is missing or too long.");
  return x.trim();
};
export function createRecipe(args, recipeId) {
  if (
    !Number.isInteger(args.servings) ||
    args.servings < 1 ||
    args.servings > 40
  )
    throw new Fault("recipe_invalid", "Choose 1–40 servings.");
  if (
    !Array.isArray(args.ingredients) ||
    !args.ingredients.length ||
    args.ingredients.length > 60 ||
    !Array.isArray(args.steps) ||
    !args.steps.length ||
    args.steps.length > 40
  )
    throw new Fault(
      "recipe_invalid",
      "A recipe needs a complete ingredient list and method.",
    );
  return {
    id: recipeId,
    title: text(args.title, 180),
    servings: args.servings,
    revision: 1,
    ingredients: args.ingredients.map((x, i) => ({
      id: `ingredient-${i + 1}`,
      text: text(x, 500),
    })),
    steps: args.steps.map((x, i) => ({ id: `step-${i + 1}`, text: text(x) })),
    changes: ["Created recipe"],
    contentHash: hash(args),
  };
}
export function patchRecipe(recipe, args) {
  if (args.expected_revision !== recipe.revision)
    throw new Fault(
      "recipe_conflict",
      "The recipe changed. Read the latest version before editing.",
      409,
    );
  if (
    !Array.isArray(args.changes) ||
    !args.changes.length ||
    args.changes.length > 80
  )
    throw new Fault("recipe_invalid", "Choose specific recipe changes.");
  const r = structuredClone(recipe),
    notes = [];
  const touched = new Set();
  for (const c of args.changes) {
    const target = c.kind + ":" + (c.id || "");
    if (touched.has(target) && !c.kind.startsWith("add_"))
      throw new Fault(
        "recipe_invalid",
        "An ingredient or step can only change once per edit.",
      );
    touched.add(target);
    if (c.kind === "title") {
      r.title = text(c.text, 180);
      notes.push("Updated title");
    } else if (c.kind === "servings") {
      if (!Number.isInteger(c.servings) || c.servings < 1 || c.servings > 40)
        throw new Fault("recipe_invalid", "Choose 1–40 servings.");
      r.servings = c.servings;
      notes.push(`Servings: ${recipe.servings} → ${c.servings}`);
    } else if (
      [
        "replace_ingredient",
        "remove_ingredient",
        "replace_step",
        "remove_step",
      ].includes(c.kind)
    ) {
      const list = c.kind.endsWith("ingredient") ? r.ingredients : r.steps,
        at = list.findIndex((x) => x.id === c.id);
      if (at < 0)
        throw new Fault(
          "recipe_invalid",
          "That recipe line no longer exists.",
          409,
        );
      notes.push(
        `${list[at].text}${c.kind.startsWith("remove") ? " → removed" : " → " + text(c.text)}`,
      );
      if (c.kind.startsWith("remove")) list.splice(at, 1);
      else
        list[at] = {
          ...list[at],
          text: text(c.text, c.kind.endsWith("ingredient") ? 500 : 3000),
        };
    } else if (["add_ingredient", "add_step"].includes(c.kind)) {
      const list = c.kind.endsWith("ingredient") ? r.ingredients : r.steps;
      list.push({
        id: `${c.kind}-${hash([recipe.id, recipe.revision, list.length, c.text]).slice(0, 10)}`,
        text: text(c.text),
      });
      notes.push("Added " + c.text);
    } else throw new Fault("recipe_invalid", "Unknown recipe change.");
  }
  if (!r.ingredients.length || !r.steps.length)
    throw new Fault("recipe_invalid", "Keep at least one ingredient and step.");
  if (
    r.servings !== recipe.servings &&
    !recipe.ingredients.every((i) =>
      args.changes.some(
        (x) =>
          ["replace_ingredient", "remove_ingredient"].includes(x.kind) &&
          x.id === i.id,
      ),
    )
  )
    throw new Fault(
      "recipe_invalid",
      "Changing servings requires explicit ingredient quantities; keep the method consistent too.",
    );
  r.revision++;
  r.changes = notes;
  r.contentHash = hash([r.title, r.servings, r.ingredients, r.steps]);
  return r;
}
// Conservative, explicit allergy screening; model also receives complete constraints.
const ALLERGENS = {
  peanut: ["peanut", "groundnut"],
  "tree nut": [
    "almond",
    "walnut",
    "cashew",
    "pecan",
    "pistachio",
    "hazelnut",
    "macadamia",
    "brazil nut",
  ],
  dairy: [
    "milk",
    "butter",
    "cream",
    "cheese",
    "yogurt",
    "yoghurt",
    "whey",
    "ghee",
  ],
  milk: ["milk", "butter", "cream", "cheese", "yogurt", "whey", "ghee"],
  egg: ["egg", "mayonnaise"],
  wheat: ["wheat", "flour", "bread", "pasta", "couscous", "bulgur"],
  gluten: [
    "wheat",
    "flour",
    "bread",
    "pasta",
    "couscous",
    "bulgur",
    "barley",
    "rye",
  ],
  soy: ["soy", "tofu", "tempeh", "edamame", "miso"],
  sesame: ["sesame", "tahini"],
  shellfish: [
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "clam",
    "mussel",
    "scallop",
    "oyster",
  ],
  fish: ["salmon", "tuna", "cod", "anchovy", "sardine", "fish"],
};
export function checkRecipe(recipe, prefs = {}) {
  const lines = recipe.ingredients.map((x) => x.text.toLowerCase());
  for (const allergy of prefs.allergies || []) {
    const key = String(allergy)
      .toLowerCase()
      .replace(/ allerg(?:y|ies)$/, "")
      .replace(/s$/, "")
      .trim();
    const terms = ALLERGENS[key] || [key];
    for (const original of lines) {
      // Exempt only the precise dairy replacement phrase; never exempt another allergen on the line.
      let line = original;
      if (["milk", "dairy"].includes(key))
        line = line
          .replace(
            /\b(?:oat|almond|coconut|soy|rice|cashew|hemp)[ -]milk\b/g,
            "plant drink",
          )
          .replace(
            /\b(?:peanut|almond|cashew|sunflower|seed)[ -]butter\b/g,
            "spread",
          );
      for (const term of terms) {
        const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        // "Milk-free" is a descriptor, while "milk, dairy-free butter" still contains milk.
        const candidate = line.replace(
          new RegExp("\\b" + escaped + "[- ]free\\b", "g"),
          "",
        );
        if (new RegExp("\\b" + escaped + "s?\\b").test(candidate))
          throw new Fault(
            "dietary_conflict",
            "This recipe conflicts with recorded allergies. Choose safe alternatives before presenting it.",
          );
      }
    }
  }
  return recipe;
}
