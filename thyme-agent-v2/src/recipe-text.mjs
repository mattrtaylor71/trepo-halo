// Conservative rescue for a complete recipe returned in chat instead of its tool.
// Copies source lines verbatim; never guesses portions, fills gaps or rewrites a method.
export function recipeFromText(text) {
  const lines = text.split(/\r?\n/);
  const clean = (x) =>
    x
      .trim()
      .replace(/^#{1,6}\s*/, "")
      .replace(/\*\*/g, "")
      .replace(/:$/, "")
      .trim();
  const ingredientHeads = lines
    .map((x, i) => (/^ingredients(?:\s*\([^)]*\))?$/i.test(clean(x)) ? i : -1))
    .filter((i) => i >= 0);
  const methodHeads = lines
    .map((x, i) =>
      /^(method|instructions|directions|steps)$/i.test(clean(x)) ? i : -1,
    )
    .filter((i) => i >= 0);
  if (ingredientHeads.length !== 1 || methodHeads.length !== 1) return null;
  const a = ingredientHeads[0],
    b = methodHeads[0];
  if (b <= a) return null;
  const heading =
    lines.slice(0, a).find((x) => /^#{1,6}\s/.test(x.trim())) ||
    lines.slice(0, a).find((x) => /^\*\*[^*]+\*\*/.test(x.trim()));
  if (!heading) return null;
  const count = lines
    .slice(0, a + 1)
    .join("\n")
    .match(/(?:serves|servings\s*:?|for)\s*(\d{1,2})(?:\b)/i);
  if (!count) return null;
  const title = clean(heading)
    .replace(/\s*\((?:serves|servings\s*:?)\s*\d+\)\s*$/i, "")
    .trim();
  const ingredients = [];
  for (const line of lines.slice(a + 1, b)) {
    if (!line.trim()) continue;
    const m = line.match(/^\s*[-*•]\s+(.+)$/);
    if (!m) return null;
    ingredients.push(m[1].trim());
  }
  const steps = [];
  let end = b + 1;
  for (let i = b + 1; i < lines.length; i++) {
    if (!lines[i].trim()) {
      end = i + 1;
      continue;
    }
    const m = lines[i].match(/^\s*(\d+)[.)]\s+(.+)$/);
    if (!m) {
      if(!/^(?:#{1,6}\s*)?(?:\*\*)?(?:missing|notes?|tips?|you |your |nothing |check |no (?:items|changes))\b/i.test(lines[i].trim())) return null;
      if(lines.slice(i+1).some(x=>/^\s*\d+[.)]\s/.test(x))) return null;
      break;
    }
    if (Number(m[1]) !== steps.length + 1) return null;
    steps.push(m[2].trim());
    end = i + 1;
  }
  if (!ingredients.length || !steps.length) return null;
  return {
    recipe: { title, servings: Number(count[1]), ingredients, steps },
    tail: lines.slice(end).join("\n").trim(),
  };
}
