import OpenAI from "openai";
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";

let openaiClient = null;
let s3Client = null;

const GENERIC_PRODUCT_TERMS = new Set([
  "food",
  "drink",
  "snack",
  "meal",
  "dish",
  "item",
  "items",
  "milk",
  "rice",
  "bread",
  "eggs",
  "egg",
  "juice",
  "water",
  "cheese",
  "yogurt",
  "butter",
  "oil",
  "chicken",
  "beef",
  "bison",
  "fish",
  "fruit",
  "vegetable",
  "veggies",
  "banana",
  "apple",
  "berries",
  "strawberries",
  "blueberries",
  "grapes",
  "lettuce",
  "spinach",
  "tomatoes",
  "tomato"
]);

function getOpenAIClient(env = process.env) {
  if (!openaiClient) {
    if (!env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY environment variable is required");
    }
    openaiClient = new OpenAI({
      apiKey: env.OPENAI_API_KEY
    });
  }
  return openaiClient;
}

function getS3Client(env = process.env) {
  if (!s3Client) {
    s3Client = new S3Client({
      region: env.AWS_REGION || env.AWS_DEFAULT_REGION || "us-east-1"
    });
  }
  return s3Client;
}

function trimString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function normalizeName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function uniqueStrings(values = []) {
  const seen = new Set();
  const items = [];
  for (const value of values) {
    const cleaned = trimString(value);
    if (!cleaned) {
      continue;
    }
    const key = normalizeName(cleaned);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push(cleaned);
  }
  return items;
}

function toStringArray(value) {
  if (Array.isArray(value)) {
    return value
      .map((entry) => trimString(entry))
      .filter(Boolean);
  }
  const cleaned = trimString(value);
  if (!cleaned) {
    return [];
  }
  return cleaned
    .split(/[,;|]/)
    .map((entry) => trimString(entry))
    .filter(Boolean);
}

function buildProductDescription(item = {}) {
  const parts = [];
  const brand = trimString(item.brand);
  const productName = trimString(item.product_name || item.item_name);
  const variant = trimString(item.variant);
  const category = trimString(item.category);

  if (brand && productName) {
    parts.push(`${brand} ${productName}`);
  } else if (productName) {
    parts.push(productName);
  } else if (brand) {
    parts.push(brand);
  }

  if (variant) {
    parts.push(variant);
  }
  if (category) {
    parts.push(`(${category})`);
  }

  return parts.join(" ") || "grocery product";
}

function buildIngredientHints(item = {}) {
  const ingredients = toStringArray(item.ingredients);
  const ingredientsText = ingredients.join(", ").toLowerCase();
  const hints = [];
  if (!ingredientsText) {
    return hints;
  }
  if (ingredientsText.includes("chocolate")) hints.push("chocolate");
  if (ingredientsText.includes("strawberry") || ingredientsText.includes("berry")) hints.push("berry");
  if (ingredientsText.includes("vanilla")) hints.push("vanilla");
  if (ingredientsText.includes("mint")) hints.push("mint");
  if (ingredientsText.includes("lemon") || ingredientsText.includes("citrus")) hints.push("citrus");
  if (ingredientsText.includes("apple")) hints.push("apple");
  if (ingredientsText.includes("banana")) hints.push("banana");
  if (ingredientsText.includes("orange")) hints.push("orange");
  if (ingredientsText.includes("peanut")) hints.push("peanut");
  return uniqueStrings(hints);
}

function buildDishDescription(dish = {}) {
  const parts = [];
  const dishName = trimString(dish.dish_name);
  const category = trimString(dish.category);
  const cuisineType = trimString(dish.cuisine_type);

  if (dishName) {
    parts.push(dishName);
  }
  if (category) {
    parts.push(`(${category})`);
  }
  if (cuisineType) {
    parts.push(`- ${cuisineType} cuisine`);
  }
  return parts.join(" ") || "food dish";
}

function buildDishIngredientHints(dish = {}) {
  const ingredients = toStringArray(dish.ingredients);
  const ingredientsText = ingredients.join(", ").toLowerCase();
  const hints = [];
  if (!ingredientsText) {
    return hints;
  }
  if (ingredientsText.includes("chicken")) hints.push("chicken");
  if (ingredientsText.includes("beef") || ingredientsText.includes("steak")) hints.push("beef");
  if (ingredientsText.includes("bison")) hints.push("bison");
  if (ingredientsText.includes("fish") || ingredientsText.includes("salmon") || ingredientsText.includes("tuna")) hints.push("seafood");
  if (ingredientsText.includes("pasta") || ingredientsText.includes("spaghetti") || ingredientsText.includes("noodles")) hints.push("pasta");
  if (ingredientsText.includes("rice")) hints.push("rice");
  if (ingredientsText.includes("cheese")) hints.push("cheese");
  if (ingredientsText.includes("tomato")) hints.push("tomato");
  if (ingredientsText.includes("lettuce") || ingredientsText.includes("salad")) hints.push("fresh greens");
  if (ingredientsText.includes("chocolate")) hints.push("chocolate");
  if (ingredientsText.includes("berry") || ingredientsText.includes("strawberry")) hints.push("berries");
  if (ingredientsText.includes("banana")) hints.push("banana");
  return uniqueStrings(hints);
}

async function generateImageBuffer(prompt, env = process.env) {
  const response = await getOpenAIClient(env).images.generate({
    model: "gpt-image-1",
    prompt,
    size: "1024x1024",
    quality: "medium",
    output_format: "jpeg",
    output_compression: 75,
    n: 1
  });
  const imageBase64 = response?.data?.[0]?.b64_json;
  return imageBase64 ? Buffer.from(imageBase64, "base64") : null;
}

async function uploadBufferToS3(buffer, { bucketName, key, contentType = "image/jpeg", env = process.env }) {
  if (!buffer || !bucketName || !key) {
    return null;
  }
  await getS3Client(env).send(new PutObjectCommand({
    Bucket: bucketName,
    Key: key,
    Body: buffer,
    ContentType: contentType
  }));
  return {
    url: `https://${bucketName}.s3.amazonaws.com/${key}`,
    key
  };
}

function inferImageExtension(contentType, imageUrl) {
  const normalizedType = String(contentType || "").toLowerCase();
  if (normalizedType.includes("png")) return "png";
  if (normalizedType.includes("webp")) return "webp";
  if (normalizedType.includes("gif")) return "gif";
  if (normalizedType.includes("jpeg") || normalizedType.includes("jpg")) return "jpg";
  try {
    const pathname = new URL(imageUrl).pathname.toLowerCase();
    if (pathname.endsWith(".png")) return "png";
    if (pathname.endsWith(".webp")) return "webp";
    if (pathname.endsWith(".gif")) return "gif";
  } catch {
    return "jpg";
  }
  return "jpg";
}

function looksLikeProductImage(candidate = {}) {
  const title = String(candidate.title || "").toLowerCase();
  const source = String(candidate.source || candidate.link || candidate.original || "").toLowerCase();
  if (!candidate.original && !candidate.link && !candidate.thumbnail) {
    return false;
  }
  const blocked = ["logo", "emblem", "symbol", "trademark", "barcode"];
  return !blocked.some((word) => title.includes(word) || source.includes(word));
}

async function mirrorRemoteImageToS3(imageUrl, { bucketName, keyPrefix, env = process.env }) {
  if (!imageUrl) {
    return null;
  }
  const response = await fetch(imageUrl, {
    headers: {
      "user-agent": "Mozilla/5.0"
    }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch remote image: HTTP ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "image/jpeg";
  if (!contentType.startsWith("image/")) {
    throw new Error(`Unexpected remote image content type: ${contentType}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  if (!bucketName) {
    return {
      url: imageUrl,
      key: null
    };
  }
  const ext = inferImageExtension(contentType, imageUrl);
  const key = `${keyPrefix}.${ext}`;
  return uploadBufferToS3(buffer, {
    bucketName,
    key,
    contentType,
    env
  });
}

function isLikelyGenericProduct(item = {}) {
  const brand = trimString(item.brand);
  const variant = trimString(item.variant);
  if (brand || variant) {
    return false;
  }
  const normalized = normalizeName(item.product_name || item.item_name);
  if (!normalized) {
    return true;
  }
  const tokens = normalized.split(/\s+/).filter(Boolean);
  const nonGenericTokens = tokens.filter((token) => !GENERIC_PRODUCT_TERMS.has(token));
  return nonGenericTokens.length === 0;
}

function buildStockQueries(item = {}) {
  const brand = trimString(item.brand);
  const productName = trimString(item.product_name || item.item_name);
  const variant = trimString(item.variant);
  const category = trimString(item.category);
  return uniqueStrings([
    [brand, productName, variant].filter(Boolean).join(" "),
    [productName, variant].filter(Boolean).join(" "),
    [brand, productName].filter(Boolean).join(" "),
    [productName, category].filter(Boolean).join(" ")
  ]).map((query) => `${query} grocery product`);
}

async function searchSerpApiImages(query, env = process.env) {
  const apiKey = trimString(env.SERPAPI_API_KEY);
  if (!apiKey || !query) {
    return [];
  }
  const url = `https://serpapi.com/search.json?q=${encodeURIComponent(query)}&tbm=isch&imgtype=photo&api_key=${encodeURIComponent(apiKey)}&num=10`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`SerpAPI request failed with HTTP ${response.status}`);
  }
  const payload = await response.json();
  return Array.isArray(payload?.images_results) ? payload.images_results : [];
}

async function resolveStockProductImage(item, env = process.env) {
  if (isLikelyGenericProduct(item)) {
    return null;
  }
  for (const query of buildStockQueries(item)) {
    try {
      const candidates = await searchSerpApiImages(query, env);
      const match = candidates.find((candidate) => looksLikeProductImage(candidate));
      const imageUrl = match?.original || match?.link || match?.thumbnail || null;
      if (imageUrl) {
        console.log("[voice-image] stock image candidate selected:", JSON.stringify({
          query,
          title: match?.title || null,
          source: match?.source || null,
          imageUrl
        }));
        return {
          imageUrl,
          query
        };
      }
    } catch (error) {
      console.warn("[voice-image] stock image lookup failed:", error?.message || error);
    }
  }
  return null;
}

// Emoji lookup for dish types based on name, category, and ingredients
const DISH_EMOJI_MAP = [
  // Specific dishes
  { keywords: ["pizza"], emoji: "🍕" },
  { keywords: ["burger", "hamburger"], emoji: "🍔" },
  { keywords: ["taco"], emoji: "🌮" },
  { keywords: ["burrito", "wrap"], emoji: "🌯" },
  { keywords: ["sushi", "sashimi"], emoji: "🍣" },
  { keywords: ["ramen", "pho", "noodle soup"], emoji: "🍜" },
  { keywords: ["pasta", "spaghetti", "linguine", "penne", "fettuccine", "mac and cheese"], emoji: "🍝" },
  { keywords: ["curry"], emoji: "🍛" },
  { keywords: ["sandwich", "sub", "panini"], emoji: "🥪" },
  { keywords: ["salad"], emoji: "🥗" },
  { keywords: ["soup", "stew", "chili", "chowder"], emoji: "🍲" },
  { keywords: ["stir fry", "stir-fry", "fried rice", "wok"], emoji: "🥘" },
  { keywords: ["pancake", "waffle", "french toast"], emoji: "🥞" },
  { keywords: ["egg", "omelette", "omelet", "frittata", "scramble"], emoji: "🍳" },
  { keywords: ["toast", "bread", "bagel", "croissant"], emoji: "🍞" },
  { keywords: ["cereal", "oatmeal", "granola", "porridge"], emoji: "🥣" },
  { keywords: ["ice cream", "gelato", "frozen yogurt"], emoji: "🍨" },
  { keywords: ["cake", "cupcake", "brownie", "muffin"], emoji: "🍰" },
  { keywords: ["cookie", "biscuit"], emoji: "🍪" },
  { keywords: ["donut", "doughnut"], emoji: "🍩" },
  { keywords: ["pie"], emoji: "🥧" },
  { keywords: ["fries", "french fries", "chips"], emoji: "🍟" },
  { keywords: ["hot dog"], emoji: "🌭" },
  // Proteins
  { keywords: ["chicken", "wing", "tender", "nugget"], emoji: "🍗" },
  { keywords: ["steak", "beef", "brisket", "ribs"], emoji: "🥩" },
  { keywords: ["fish", "salmon", "tuna", "cod", "tilapia", "seafood", "shrimp"], emoji: "🐟" },
  { keywords: ["pork", "bacon", "ham", "sausage"], emoji: "🥓" },
  // Drinks
  { keywords: ["coffee", "espresso", "latte", "cappuccino"], emoji: "☕" },
  { keywords: ["tea", "matcha", "chai"], emoji: "🍵" },
  { keywords: ["smoothie", "shake", "protein shake"], emoji: "🥤" },
  { keywords: ["juice", "lemonade"], emoji: "🧃" },
  { keywords: ["beer", "ale", "ipa"], emoji: "🍺" },
  { keywords: ["wine"], emoji: "🍷" },
  { keywords: ["cocktail", "margarita", "mojito"], emoji: "🍹" },
  { keywords: ["soda", "coke", "pepsi", "sprite", "red bull", "energy drink"], emoji: "🥤" },
  { keywords: ["water", "sparkling"], emoji: "💧" },
  // General categories
  { keywords: ["fruit", "apple", "banana", "berry", "mango"], emoji: "🍎" },
  { keywords: ["rice", "grain bowl", "bowl"], emoji: "🍚" },
  { keywords: ["snack", "chips", "crackers", "nuts", "trail mix"], emoji: "🥜" },
  { keywords: ["yogurt", "parfait"], emoji: "🥛" },
  { keywords: ["cheese"], emoji: "🧀" },
];

function pickDishEmoji(dish = {}) {
  const dishName = (dish.dish_name || "").toLowerCase();
  const category = (dish.category || "").toLowerCase();
  const ingredients = (dish.ingredients || []).map(i => String(i).toLowerCase()).join(" ");
  const searchText = `${dishName} ${category} ${ingredients}`;

  for (const entry of DISH_EMOJI_MAP) {
    if (entry.keywords.some(kw => searchText.includes(kw))) {
      return entry.emoji;
    }
  }

  // Fallback by meal category
  if (category.includes("breakfast")) return "🍳";
  if (category.includes("lunch")) return "🥗";
  if (category.includes("dinner") || category.includes("supper")) return "🍽️";
  if (category.includes("snack")) return "🍿";
  if (category.includes("dessert") || category.includes("sweet")) return "🍰";
  if (category.includes("drink") || category.includes("beverage")) return "🥤";

  return "🍽️"; // generic fallback
}

export async function buildDishImageAsset({ dish, rowId, userId, bucketName, env = process.env }) {
  // Assign an emoji instead of generating an AI image
  const emoji = pickDishEmoji(dish);
  console.log("[dish-emoji] Assigned emoji:", { dish_name: dish?.dish_name, emoji });
  return {
    url: `emoji:${emoji}`,
    key: null,
    source: "emoji"
  };
}

export async function buildKitchenProductImageAsset({ item, rowId, userId, bucketName, env = process.env }) {
  const stockMatch = await resolveStockProductImage(item, env);
  if (stockMatch?.imageUrl) {
    const mirrored = await mirrorRemoteImageToS3(stockMatch.imageUrl, {
      bucketName,
      keyPrefix: `product-images/${userId || "voice-assistant"}/voice-assistant/${rowId}`,
      env
    });
    if (mirrored) {
      return {
        ...mirrored,
        source: "stock"
      };
    }
  }

  if (!bucketName) {
    return null;
  }

  const productDescription = buildProductDescription(item);
  const ingredientHints = buildIngredientHints(item);
  let prompt = `A simple, minimal flat icon of ${productDescription}`;
  if (ingredientHints.length > 0) {
    prompt += ` with ${ingredientHints.join(" and ")} characteristics`;
  }
  prompt += `. 

Style: Clean vector-like icon, minimal detail, bold simple shapes, limited color palette.
Centered on a square canvas, high contrast, recognizable at small sizes.
Background: solid light neutral color.
NO TEXT. NO WORDS. NO LABELS. NO BRAND NAMES. NO BARCODES. NO WRITING OF ANY KIND.
Icon-only, not a photo.`;

  const imageBuffer = await generateImageBuffer(prompt, env);
  if (!imageBuffer) {
    return null;
  }
  const uploaded = await uploadBufferToS3(imageBuffer, {
    bucketName,
    key: `product-images/${userId || "voice-assistant"}/voice-assistant/${rowId}.jpg`,
    env
  });
  return uploaded
    ? {
        ...uploaded,
        source: "generated_icon"
      }
    : null;
}
