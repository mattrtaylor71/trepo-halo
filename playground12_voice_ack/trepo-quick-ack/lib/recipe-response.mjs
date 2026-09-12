// Keep recipe text compatible with shipped iOS parsers. This is formatting only:
// no tools, writes, schema changes, or AI calls. Ambiguous/incomplete text is left alone.
export function formatRecipeResponse(text) {
  if (typeof text !== 'string') return null;
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  const clean = value => value.trim().replace(/^(?:#{1,6}\s*|>\s*)/, '').replace(/\*\*|__/g, '').trim();
  const section = line => {
    const m = clean(line).replace(/^(?:[-*•]\s+|\d+[.)]\s+)/, '').match(/^(ingredients|steps|instructions|directions|method|preparation|notes|tips|nutrition|servings|prep time|cook time|total time)\s*(?::\s*(.*)|$)/i);
    if (!m) return null;
    return { kind: /^ingredients$/i.test(m[1]) ? 'ingredients' : /^(steps|instructions|directions|method|preparation)$/i.test(m[1]) ? 'steps' : 'other', tail: m[2] || '' };
  };
  const title = line => {
    if (section(line)) return null;
    const value = clean(line).replace(/^(?:recipe\s+)?\d+[.):]\s+/i, '').replace(/^[: #*]+|[: #*]+$/g, '');
    return value && value.length <= 120 && !/[.!?]$/.test(value) && !/^[-•]\s/.test(value) && /\p{L}/u.test(value) ? value : null;
  };
  const rank = line => /^#/.test(line.trim()) ? 2 : /^(?:\*\*|__|\d+[.)]\s+(?:\*\*|__))/.test(line.trim()) ? 1 : 0;
  const explicit = line => rank(line) > 0;
  const marker = line => line.match(/^(?:[-*•]\s+|\d+[.)]\s+)(.+)$/)?.[1]?.trim();
  const indices = lines.flatMap((line, index) => section(line)?.kind === 'ingredients' ? [index] : []);
  if (!indices.length) return null;
  const starts = [];
  let lower = 0;
  for (const index of indices) {
    const candidates = [];
    for (let i = lower; i < index; i++) if (title(lines[i])) candidates.push(i);
    const bestRank = Math.max(0, ...candidates.map(i => rank(lines[i])));
    const titleIndex = candidates.filter(i => rank(lines[i]) === bestRank).at(-1);
    if (titleIndex === undefined) return null;
    starts.push({ titleIndex, ingredientIndex: index, name: title(lines[titleIndex]) });
    lower = index + 1;
  }
  const recipes = [];
  const descriptions = [];
  for (let offset = 0; offset < starts.length; offset++) {
    const start = starts[offset];
    const end = starts[offset + 1]?.titleIndex ?? lines.length;
    if (end <= start.ingredientIndex) return null;
    const description = lines.slice(start.titleIndex + 1, start.ingredientIndex).join('\n').trim();
    if (description) descriptions.push(`${start.name}: ${description}`);
    let kind = 'other';
    const content = { ingredients: [], steps: [], other: [] };
    for (let i = start.ingredientIndex; i < end; i++) {
      const header = section(lines[i]);
      if (header) kind = header.kind;
      content[kind].push(clean(header && kind !== 'other' ? header.tail : lines[i]));
    }
    const extras = content.other.filter(value => value && !/^notes:?$/i.test(value));
    const items = values => {
      if (!values.some(marker)) return values.filter(v => v && !/^[-*_]{3,}$/.test(v));
      const result = [];
      let mayContinue = false;
      for (const value of values) {
        const item = marker(value);
        if (item) { result.push(item); mayContinue = true; }
        else if (!value) mayContinue = false;
        else if (mayContinue && result.length && !explicit(value)) result[result.length - 1] += ` ${value}`;
        else if (value && !/^[-*_]{3,}$/.test(value)) extras.push(value);
      }
      return result;
    };
    const ingredients = items(content.ingredients), steps = items(content.steps);
    if (!ingredients.length || !steps.length) return null;
    recipes.push(`${start.name}\nIngredients:\n${ingredients.map(v => `- ${v}`).join('\n')}\n\nSteps:\n${steps.map((v, i) => `${i + 1}. ${v}`).join('\n')}${extras.length ? `\n\nNotes:\n${extras.join('\n')}` : ''}`);
  }
  const intro = lines.slice(0, starts[0].titleIndex).join('\n').trim();
  // Retain descriptions before the card content, never between a dish name and its
  // Ingredients heading (older clients take that intervening line as the name).
  return [intro, ...descriptions, ...recipes].filter(Boolean).join('\n\n');
}
