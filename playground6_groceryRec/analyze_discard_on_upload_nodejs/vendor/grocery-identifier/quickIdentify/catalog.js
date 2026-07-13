const fs = require("fs");

const DEFAULT_CATALOG = [
  {
    canonical_name: "Barilla Penne Rigate Pasta",
    brand: "Barilla",
    category: "Pasta",
    item_type: "packaged",
    aliases: ["barilla penne rigate pasta", "penne rigate pasta", "barilla penne rigate", "penne rigate"],
    visible_text_aliases: ["BARILLA", "PENNE RIGATE"],
  },
  {
    canonical_name: "Rolled Oats Canister",
    brand: null,
    category: "Rolled Oats",
    item_type: "packaged",
    aliases: ["rolled oats canister", "rolled oats", "old fashioned oats"],
    visible_text_aliases: ["ROLLED OATS", "OLD FASHIONED OATS"],
  },
  {
    canonical_name: "Large Eggs Carton",
    brand: null,
    category: "Eggs",
    item_type: "packaged",
    aliases: ["large eggs carton", "eggs carton", "large eggs", "brown eggs"],
    visible_text_aliases: ["LARGE EGGS", "LARGE BROWN", "GRADE A", "12"],
  },
  {
    canonical_name: "Whole Milk Gallon Jug",
    brand: null,
    category: "Milk",
    item_type: "packaged",
    aliases: ["whole milk gallon jug", "vitamin d milk gallon jug", "milk gallon jug", "whole milk", "milk"],
    visible_text_aliases: ["MILK", "VITAMIN D"],
  },
  {
    canonical_name: "Shredded Medium Cheddar Cheese",
    brand: null,
    category: "Shredded Cheese",
    item_type: "packaged",
    aliases: ["shredded medium cheddar cheese", "medium cheddar shredded cheese", "thick cut medium cheddar", "shredded cheese"],
    visible_text_aliases: ["THICK CUT", "MEDIUM CHEDDAR", "SHREDS"],
  },
  {
    canonical_name: "Flour Tortillas",
    brand: null,
    category: "Flour Tortillas",
    item_type: "packaged",
    aliases: ["flour tortillas", "tortillas"],
    visible_text_aliases: ["FLOUR TORTILLAS", "TORTILLAS"],
  },
  {
    canonical_name: "Peanut Butter",
    brand: null,
    category: "Peanut Butter",
    item_type: "packaged",
    aliases: ["peanut butter", "natural peanut butter", "creamy peanut butter"],
    visible_text_aliases: ["PEANUT BUTTER", "CREAMY", "NO STIR", "ALL NATURAL"],
  },
  {
    canonical_name: "Rice Cakes",
    brand: null,
    category: "Rice Cakes",
    item_type: "packaged",
    aliases: ["rice cakes", "rice cake snacks"],
    visible_text_aliases: ["RICE CAKES", "RICE", "CAKES"],
  },
  {
    canonical_name: "Mac And Cheese Box",
    brand: null,
    category: "Boxed Mac And Cheese",
    item_type: "packaged",
    aliases: ["mac and cheese box", "mac cheese box", "macaroni and cheese", "boxed macaroni and cheese"],
    visible_text_aliases: ["MAC", "CHEESE", "DINNER"],
  },
  {
    canonical_name: "Canned Beans",
    brand: null,
    category: "Canned Beans",
    item_type: "packaged",
    aliases: ["canned beans", "kidney beans", "black beans", "beans can"],
    visible_text_aliases: ["BEANS", "BUSH'S", "KIDNEY", "BLACK"],
  },
  {
    canonical_name: "Tater Tots Bag",
    brand: null,
    category: "Frozen Potato Products",
    item_type: "packaged",
    aliases: ["tater tots bag", "potato bites bag", "frozen potato products", "frozen potato bites"],
    visible_text_aliases: ["TATER", "TOTS", "POTATO", "BITES"],
  },
  {
    canonical_name: "Sugar-Free Chewing Gum",
    brand: "Eclipse",
    category: "Gum",
    item_type: "packaged",
    aliases: [
      "sugar free chewing gum",
      "spearmint chewing gum",
      "eclipse gum",
      "wrigley gum",
      "wrigley s eclipse gum",
    ],
    visible_text_aliases: ["ECLIPSE", "WRIGLEY", "WRIGLEY'S", "CHEWING GUM", "SPEARMINT", "SUGARFREE", "SUGAR FREE"],
  },
  {
    canonical_name: "Sugar-Free Mints",
    brand: null,
    category: "Mints",
    item_type: "packaged",
    aliases: ["sugar free mints", "peppermint mints", "mint container", "mint candy"],
    visible_text_aliases: ["MINT", "MINTS", "PEPPERMINT", "MENTOS"],
  },
  {
    canonical_name: "Yogurt Multipack",
    brand: null,
    category: "Yogurt",
    item_type: "packaged",
    aliases: ["yogurt multipack", "yogurt cups", "yogurt multi pack", "yogurt"],
    visible_text_aliases: ["YOGURT"],
  },
  {
    canonical_name: "Cafe Bustelo Espresso Coffee",
    brand: "Cafe Bustelo",
    category: "Coffee",
    item_type: "packaged",
    aliases: ["cafe bustelo espresso coffee", "cafe bustelo", "espresso coffee"],
    visible_text_aliases: ["CAFE BUSTELO", "ESPRESSO"],
  },
];

const PRODUCE_CANONICALS = [
  { canonical_name: "Bananas", category: "Produce", aliases: ["banana", "bananas", "bunch of bananas"] },
  { canonical_name: "Green Grapes", category: "Produce", aliases: ["green grapes", "seedless grapes", "grapes"] },
  { canonical_name: "Spinach", category: "Packaged Greens", aliases: ["spinach", "baby spinach", "organic spinach"] },
  { canonical_name: "Broccoli Florets", category: "Produce", aliases: ["broccoli", "broccoli florets", "florets"] },
  { canonical_name: "Mandarins", category: "Produce", aliases: ["mandarins", "mandarin oranges", "clementines", "small oranges"] },
  { canonical_name: "Apples", category: "Produce", aliases: ["apples", "apple"] },
  { canonical_name: "Avocados", category: "Produce", aliases: ["avocados", "avocado"] },
  { canonical_name: "Potatoes", category: "Produce", aliases: ["potatoes", "potato"] },
];

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function tokenize(value) {
  return normalizeText(value)
    .split(" ")
    .map((token) => token.trim())
    .filter(Boolean);
}

function overlapScore(a, b) {
  const aTokens = new Set(tokenize(a));
  const bTokens = new Set(tokenize(b));
  let score = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) {
      score += 1;
    }
  }
  return score;
}

let cachedCatalog = null;

function loadExternalCatalog() {
  const catalogPath = process.env.GROCERY_CATALOG_PATH;
  if (!catalogPath) {
    return [];
  }

  try {
    const raw = fs.readFileSync(catalogPath, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn("[QuickIdentify] Failed to load external grocery catalog:", error);
    return [];
  }
}

function getCatalogEntries() {
  if (!cachedCatalog) {
    cachedCatalog = [...DEFAULT_CATALOG, ...loadExternalCatalog()];
  }
  return cachedCatalog;
}

function classifyItemType(item) {
  if (item?.item_type && ["produce", "packaged", "unknown"].includes(item.item_type)) {
    return item.item_type;
  }

  const combined = [
    item?.item_name,
    item?.brand,
    item?.category,
    ...(Array.isArray(item?.visible_text) ? item.visible_text : []),
  ]
    .filter(Boolean)
    .join(" ");

  const produceTokens = [
    "banana",
    "bananas",
    "grape",
    "grapes",
    "spinach",
    "broccoli",
    "produce",
    "greens",
    "avocado",
    "apple",
    "mandarin",
    "clementine",
    "orange",
  ];
  const packagedTokens = [
    "box",
    "bag",
    "carton",
    "jar",
    "can",
    "bottle",
    "milk",
    "cheese",
    "tortillas",
    "pasta",
    "oats",
    "rice cakes",
    "peanut butter",
    "beans",
    "espresso",
    "yogurt",
    "gum",
    "mints",
    "mint",
    "candy",
  ];

  const combinedNormalized = normalizeText(combined);
  const produceScore = produceTokens.reduce((sum, token) => sum + (combinedNormalized.includes(token) ? 1 : 0), 0);
  const packagedScore = packagedTokens.reduce((sum, token) => sum + (combinedNormalized.includes(token) ? 1 : 0), 0);

  if (produceScore > packagedScore) {
    return "produce";
  }
  if (packagedScore > 0) {
    return "packaged";
  }
  return "unknown";
}

function canonicalizeProduceItem(item) {
  const combined = [
    item?.item_name,
    item?.brand,
    item?.category,
    ...(Array.isArray(item?.visible_text) ? item.visible_text : []),
  ]
    .filter(Boolean)
    .join(" ");

  let best = null;
  let bestScore = 0;
  for (const entry of PRODUCE_CANONICALS) {
    const score = entry.aliases.reduce((sum, alias) => sum + overlapScore(alias, combined), 0);
    if (score > bestScore) {
      best = entry;
      bestScore = score;
    }
  }

  if (!best || bestScore < 1) {
    return {
      ...item,
      item_type: "produce",
    };
  }

  return {
    ...item,
    item_name: best.canonical_name,
    category: item.category || best.category,
    item_type: "produce",
  };
}

function matchCatalogCandidate(item) {
  const combinedText = [
    item?.item_name,
    item?.brand,
    item?.category,
    ...(Array.isArray(item?.visible_text) ? item.visible_text : []),
  ]
    .filter(Boolean)
    .join(" ");

  let best = null;
  let bestScore = 0;
  for (const entry of getCatalogEntries()) {
    let score = 0;
    for (const alias of entry.aliases || []) {
      score += overlapScore(alias, combinedText) * 3;
      if (normalizeText(combinedText).includes(normalizeText(alias))) {
        score += 4;
      }
    }
    for (const textAlias of entry.visible_text_aliases || []) {
      score += overlapScore(textAlias, combinedText) * 2;
      if (normalizeText(combinedText).includes(normalizeText(textAlias))) {
        score += 3;
      }
    }
    if (normalizeText(item?.brand) && normalizeText(entry.brand) === normalizeText(item.brand)) {
      score += 4;
    }
    if (score > bestScore) {
      best = entry;
      bestScore = score;
    }
  }

  if (!best || bestScore < 4) {
    return null;
  }

  return {
    entry: best,
    score: bestScore,
  };
}

function applyCatalogMatch(item) {
  const itemType = classifyItemType(item);
  if (itemType === "produce") {
    return canonicalizeProduceItem(item);
  }

  const match = matchCatalogCandidate(item);
  if (!match) {
    return {
      ...item,
      item_type: itemType,
    };
  }

  return {
    ...item,
    item_name: match.entry.canonical_name,
    brand: item.brand || match.entry.brand || null,
    category: item.category || match.entry.category || null,
    item_type: match.entry.item_type || itemType,
    catalog_match: {
      canonical_name: match.entry.canonical_name,
      score: match.score,
      source: "local_catalog",
    },
  };
}

function appendUncertaintySignals(item) {
  const reasons = Array.isArray(item?.uncertainty_reasons) ? [...item.uncertainty_reasons] : [];
  if ((item?.confidence || 0) < 0.7) {
    reasons.push("low_confidence");
  }
  if (!item?.brand && item?.item_type === "packaged") {
    reasons.push("missing_brand");
  }
  if (!item?.visible_text || item.visible_text.length === 0) {
    reasons.push("no_visible_text");
  }
  if (item?.item_type === "unknown") {
    reasons.push("unknown_item_type");
  }

  const uniqueReasons = Array.from(new Set(reasons));
  return {
    ...item,
    needs_review: uniqueReasons.length >= 2 || uniqueReasons.includes("low_confidence"),
    uncertainty_reasons: uniqueReasons,
  };
}

module.exports = {
  appendUncertaintySignals,
  applyCatalogMatch,
  canonicalizeProduceItem,
  classifyItemType,
  DEFAULT_CATALOG,
  PRODUCE_CANONICALS,
  getCatalogEntries,
  matchCatalogCandidate,
};
