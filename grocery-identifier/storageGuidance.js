const OpenAI = require("openai");

let _openaiClient = null;

function _getClient() {
  if (!_openaiClient) {
    if (!process.env.OPENAI_API_KEY) return null;
    _openaiClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return _openaiClient;
}

function _getModel() {
  return process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";
}

function clean(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function containsAny(text, terms) {
  return terms.some((term) => text.includes(term));
}

function buildHaystack(item) {
  return [
    clean(item?.product_name),
    clean(item?.item_name),
    clean(item?.brand),
    clean(item?.variant),
    clean(item?.category),
    clean(item?.product_description),
    clean(item?.explanation),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function estimateStorageGuidance(item) {
  const haystack = buildHaystack(item);

  if (!haystack) {
    return {
      summary: "Check the label for storage tips. Once you open it, try to use it up within a week or two.",
      min_days: null,
      max_days: null,
      timing_start: "after_opening",
      storage_zone: "mixed",
      confidence: 0.2,
      source: "heuristic",
    };
  }

  // Alcohol — check first (rare false positives)
  if (containsAny(haystack, ["bitters", "liqueur", "liquor", "spirit", "vodka", "gin", "rum", "tequila", "whiskey", "whisky", "bourbon", "alcohol", "wine", "beer", "hard seltzer"])) {
    return {
      summary: "Store in a cool, dark spot in the pantry. Sealed bottles last practically forever. Once opened, it'll stay good for 6 months to a year — just keep the cap on tight.",
      min_days: 180,
      max_days: 365,
      timing_start: "after_opening",
      storage_zone: "pantry",
      confidence: 0.8,
      source: "heuristic",
    };
  }

  // Beverages — check BEFORE produce to avoid fruit-flavored drinks matching produce keywords
  if (containsAny(haystack, ["soda", "sparkling water", "seltzer", "juice", "tea", "coffee", "energy drink", "kombucha", "drink", "beverage", "water bottle", "lemonade", "iced tea", "sports drink", "gatorade", "spindrift"])) {
    return {
      summary: "Keep these in the fridge for best taste. Unopened, they'll last weeks or months. Once you crack one open, finish it within a day or two — the fizz and flavor drop off fast.",
      min_days: 1,
      max_days: 7,
      timing_start: "after_opening",
      storage_zone: "refrigerated",
      confidence: 0.62,
      source: "heuristic",
    };
  }

  // Delicate produce
  if (containsAny(haystack, ["berry", "berries", "spinach", "lettuce", "salad", "greens", "herb", "cilantro", "parsley", "mushroom"])) {
    return {
      summary: "Pop these in the fridge as soon as you can. They're delicate — plan to use them within 3 to 7 days. Don't wash until you're ready to eat, since moisture speeds up spoilage.",
      min_days: 3,
      max_days: 7,
      timing_start: "from_check_in",
      storage_zone: "refrigerated",
      confidence: 0.72,
      source: "heuristic",
    };
  }

  // Hardy whole produce
  if (containsAny(haystack, ["apple", "orange", "lemon", "lime", "grape", "cabbage", "carrot"])) {
    return {
      summary: "These do great in the fridge and will keep for 2 to 4 weeks when stored whole. Once you cut into them, wrap the rest tightly and try to use it within about a week.",
      min_days: 14,
      max_days: 30,
      timing_start: "from_check_in",
      storage_zone: "refrigerated",
      confidence: 0.7,
      source: "heuristic",
    };
  }

  // Counter-ripening produce
  if (containsAny(haystack, ["banana", "avocado", "tomato", "peach", "pear", "plum", "mango", "kiwi", "nectarine"])) {
    return {
      summary: "Leave on the counter to ripen — they'll be at their best in 2 to 5 days. Once they're ripe, you can move them to the fridge to buy a couple extra days.",
      min_days: 2,
      max_days: 7,
      timing_start: "from_check_in",
      storage_zone: "counter",
      confidence: 0.68,
      source: "heuristic",
    };
  }

  // Dairy - milk/cream
  if (containsAny(haystack, ["milk", "half and half", "cream", "creamer"])) {
    return {
      summary: "Keep this in the fridge — the colder the better, so avoid the door shelf. Unopened, check the best-by date. Once opened, aim to use it within 5 to 7 days.",
      min_days: 5,
      max_days: 7,
      timing_start: "after_opening",
      storage_zone: "refrigerated",
      confidence: 0.76,
      source: "heuristic",
    };
  }

  // Dairy - cultured
  if (containsAny(haystack, ["yogurt", "yoghurt", "cottage cheese", "sour cream", "dip"])) {
    return {
      summary: "Store in the fridge and keep the lid sealed tight. Unopened, it'll last until the date on the package. Once opened, try to finish it within about 5 to 10 days.",
      min_days: 5,
      max_days: 10,
      timing_start: "after_opening",
      storage_zone: "refrigerated",
      confidence: 0.72,
      source: "heuristic",
    };
  }

  // Eggs
  if (containsAny(haystack, ["egg", "eggs"])) {
    return {
      summary: "Keep eggs in the fridge in their original carton — it protects them from picking up odors. They'll stay fresh for 3 to 5 weeks from the purchase date.",
      min_days: 21,
      max_days: 35,
      timing_start: "from_check_in",
      storage_zone: "refrigerated",
      confidence: 0.75,
      source: "heuristic",
    };
  }

  // Cheese/butter
  if (containsAny(haystack, ["cheese", "butter"])) {
    return {
      summary: "Keep in the fridge. Unopened, hard cheeses last weeks; softer ones about 1 to 2 weeks. Once opened, wrap tightly and use within 1 to 3 weeks depending on the type.",
      min_days: 7,
      max_days: 21,
      timing_start: "after_opening",
      storage_zone: "refrigerated",
      confidence: 0.7,
      source: "heuristic",
    };
  }

  // Meat/seafood
  if (containsAny(haystack, ["chicken", "beef", "pork", "turkey", "fish", "salmon", "shrimp", "meat", "seafood"])) {
    return {
      summary: "Get this into the fridge right away — raw meat and seafood are only good for 1 to 3 days refrigerated. If you're not cooking it soon, throw it in the freezer.",
      min_days: 1,
      max_days: 3,
      timing_start: "from_check_in",
      storage_zone: "refrigerated",
      confidence: 0.8,
      source: "heuristic",
    };
  }

  // Snacks
  if (containsAny(haystack, ["chips", "cracker", "cookie", "cookies", "pretzel", "popcorn", "granola bar", "snack"])) {
    return {
      summary: "Keep in the pantry — a cool, dry spot is perfect. Sealed, they'll last for months. Once you open the bag, clip it shut or transfer to a container and finish within 1 to 3 weeks.",
      min_days: 7,
      max_days: 21,
      timing_start: "after_opening",
      storage_zone: "pantry",
      confidence: 0.7,
      source: "heuristic",
    };
  }

  // Sauces/condiments
  if (containsAny(haystack, ["sauce", "salsa", "dressing", "ketchup", "mustard", "mayo", "mayonnaise", "broth", "stock"])) {
    return {
      summary: "Unopened, these are fine in the pantry. Once opened, move to the fridge and use within 1 to 2 months — though fresh salsas and broths should be used within a week.",
      min_days: 7,
      max_days: 60,
      timing_start: "after_opening",
      storage_zone: "mixed",
      confidence: 0.56,
      source: "heuristic",
    };
  }

  // Pantry staples
  if (containsAny(haystack, ["pasta", "rice", "bean", "lentil", "flour", "oat", "granola", "cereal", "oil", "vinegar", "spice", "seasoning", "canned"])) {
    return {
      summary: "These are pantry staples — store in a cool, dry spot and they'll keep for months. Once opened, seal the package well and aim to use within a few months for the best quality.",
      min_days: 30,
      max_days: 180,
      timing_start: "after_opening",
      storage_zone: "pantry",
      confidence: 0.55,
      source: "heuristic",
    };
  }

  // Fallback
  return {
    summary: "Check the label for the best storage spot. Once opened, most items are best within 1 to 2 weeks — keep it sealed and stored properly.",
    min_days: 5,
    max_days: 14,
    timing_start: "after_opening",
    storage_zone: "mixed",
    confidence: 0.35,
    source: "heuristic",
  };
}

function _clampDays(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const rounded = Math.round(value);
  if (rounded < 0) return 0;
  if (rounded > 730) return 730;
  return rounded;
}

function _clampConfidence(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.max(0, Math.min(1, value));
}

function _finalizeGuidance(parsed, fallback) {
  const timingStart =
    parsed.timing_start === "after_opening" || parsed.timing_start === "from_check_in"
      ? parsed.timing_start
      : fallback.timing_start;
  const storageZone = ["counter", "refrigerated", "pantry", "frozen", "mixed"].includes(parsed.storage_zone)
    ? parsed.storage_zone
    : fallback.storage_zone;

  let minDays = _clampDays(parsed.min_days);
  let maxDays = _clampDays(parsed.max_days);
  if (minDays == null && maxDays == null) {
    minDays = fallback.min_days;
    maxDays = fallback.max_days;
  } else if (minDays == null && maxDays != null) {
    minDays = Math.max(0, Math.min(maxDays, fallback.min_days ?? maxDays));
  } else if (minDays != null && maxDays == null) {
    maxDays = Math.max(minDays, fallback.max_days ?? minDays);
  }
  if (minDays != null && maxDays != null && minDays > maxDays) {
    const swap = minDays;
    minDays = maxDays;
    maxDays = swap;
  }

  const summary = (typeof parsed.summary === "string" && parsed.summary.trim()) ? parsed.summary.trim() : fallback.summary;
  const confidence = _clampConfidence(parsed.confidence) ?? fallback.confidence;

  return {
    summary,
    min_days: minDays,
    max_days: maxDays,
    timing_start: timingStart,
    storage_zone: storageZone,
    confidence,
    source: "ai",
  };
}

async function estimateStorageGuidanceAI(item) {
  const fallback = estimateStorageGuidance(item);

  const client = _getClient();
  if (!client) return fallback;

  try {
    const response = await client.responses.create({
      model: _getModel(),
      reasoning: { effort: "low" },
      max_output_tokens: 250,
      text: {
        format: {
          type: "json_schema",
          name: "storage_guidance",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["summary", "min_days", "max_days", "timing_start", "storage_zone", "confidence"],
            properties: {
              summary: { type: "string" },
              min_days: { type: ["number", "null"] },
              max_days: { type: ["number", "null"] },
              timing_start: { type: "string", enum: ["from_check_in", "after_opening"] },
              storage_zone: { type: "string", enum: ["counter", "refrigerated", "pantry", "frozen", "mixed"] },
              confidence: { type: ["number", "null"] },
            },
          },
        },
      },
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text: "You're giving friendly, practical storage advice for a grocery item someone just brought home. Write like you're talking to a friend — casual, warm, and specific to the product (name it, don't say 'this item' or 'whole produce like this'). Cover three things in 2-3 sentences: (1) where to store it and roughly how long it lasts sealed/whole, (2) how long it's good once opened or cut into, and (3) any quick tip if relevant. For fresh produce, meat, seafood, bread, deli, and prepared foods, estimate from check-in (just brought home, stored properly). For packaged pantry or refrigerated items, focus on how long quality lasts once opened. Be conservative — practical consumer guidance, not extreme maximums. Return only valid JSON.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: JSON.stringify({
                product_name: clean(item?.product_name),
                brand: clean(item?.brand),
                variant: clean(item?.variant),
                category: clean(item?.category),
                product_description: clean(item?.product_description),
                explanation: clean(item?.explanation),
              }),
            },
          ],
        },
      ],
    });

    const outputText = response?.output_text;
    if (!outputText || typeof outputText !== "string") return fallback;

    const parsed = JSON.parse(outputText);
    return _finalizeGuidance(parsed, fallback);
  } catch (error) {
    console.warn("[storage-guidance] AI fallback to heuristic:", error?.message || error);
    return fallback;
  }
}

// Map a storage_guidance `storage_zone` to the app's `storage_location` enum.
// The app's location options are exactly "fridge"/"freezer"/"pantry" (lowercase)
// and it renders "fridge" when the value is null. We map counter -> "pantry"
// (the app has no counter bucket; counter items like onions belong OUT of the
// fridge and pantry is the closest option) and leave "mixed"/unknown UNSET (null)
// so we never guess when the zone is genuinely ambiguous.
function storageZoneToLocation(zone) {
  switch (zone) {
    case "refrigerated":
      return "fridge";
    case "frozen":
      return "freezer";
    case "pantry":
      return "pantry";
    case "counter":
      return "pantry";
    case "mixed":
    default:
      return null;
  }
}

module.exports = {
  estimateStorageGuidance,
  estimateStorageGuidanceAI,
  storageZoneToLocation,
};
