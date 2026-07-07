import fetch from "node-fetch";
import {
  getVerifiedProductPageUrls,
  resolveStockImage,
  StockImageMode,
  StockImageResult,
} from "../images/resolveStockImage";
import { getOpenAIModel } from "../openai/client";
import { getStoreAvailability } from "../openai/getStoreAvailability";
import { GroceryItem, identifyGroceryItem, isLikelyNonGroceryItem } from "../openai/identifyGrocery";
import { PurchaseOption } from "../search/findRetailerLinks";

export type AnalyzeProductInput =
  | { imageBuffer: Buffer; imageUrl?: never }
  | { imageUrl: string; imageBuffer?: never };

export type StockImageResponse = {
  status: "verified" | "pending_deep_search" | "not_found";
  url: string | null;
  source: StockImageResult["source"] | null;
  source_page_url: string | null;
  license: string | null;
  match_confidence: number | null;
  match_reason: string | null;
};

export type AnalyzeProductOptions = {
  stockImageMode?: StockImageMode;
  userHint?: string | null;
  leftovers?: boolean;
};

export type AnalyzeProductResult = {
  groceryItem: GroceryItem;
  purchase_options: PurchaseOption[];
  store_availability: PurchaseOption[];
  stock_image: StockImageResponse;
  debug: {
    total_ms: number;
    identify_ms: number;
    retailer_search_ms: number;
    stock_image_ms: number;
    model: string;
    retailer_link_source: string;
  };
};

async function assertImageUrl(imageUrl: string): Promise<void> {
  const response = await fetch(imageUrl, {
    headers: { "user-agent": "Mozilla/5.0" },
  });

  if (!response.ok) {
    throw new Error(`Failed to download image from URL: HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.startsWith("image/")) {
    throw new Error("URL does not point to an image file");
  }
}

function buildTargetImageUrl(input: AnalyzeProductInput): string | null {
  if ("imageUrl" in input && input.imageUrl) {
    return input.imageUrl;
  }

  if ("imageBuffer" in input && input.imageBuffer) {
    return `data:image/jpeg;base64,${input.imageBuffer.toString("base64")}`;
  }

  return null;
}

function getStockImageTimeoutMs(mode: StockImageMode): number {
  const fallback = mode === "deep" ? 45000 : 10000;
  const modeSpecificName = mode === "deep" ? "STOCK_IMAGE_DEEP_TIMEOUT_MS" : "STOCK_IMAGE_TIMEOUT_MS";
  const raw = parseInt(process.env[modeSpecificName] || process.env.STOCK_IMAGE_TIMEOUT_MS || "", 10);
  return Number.isFinite(raw) ? raw : fallback;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, errorMessage: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(errorMessage)), timeoutMs);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

function toStockImageResponse(result: StockImageResult | null, mode: StockImageMode): StockImageResponse {
  if (result) {
    return {
      status: "verified",
      url: result.url,
      source: result.source,
      source_page_url: result.source_page_url,
      license: result.license,
      match_confidence: result.match_confidence,
      match_reason: result.match_reason,
    };
  }

  return {
    status: mode === "fast" ? "pending_deep_search" : "not_found",
    url: null,
    source: null,
    source_page_url: null,
    license: null,
    match_confidence: null,
    match_reason: null,
  };
}

export async function analyzeProduct(
  input: AnalyzeProductInput,
  options: AnalyzeProductOptions = {}
): Promise<AnalyzeProductResult> {
  const startedAt = Date.now();
  const stockImageMode = options.stockImageMode || "fast";
  const targetImageUrl = buildTargetImageUrl(input);

  if ("imageUrl" in input && input.imageUrl) {
    await assertImageUrl(input.imageUrl);
  }

  const identifyStartedAt = Date.now();
  const groceryItem = await identifyGroceryItem(input, { userHint: options.userHint, leftovers: options.leftovers });
  const identify_ms = Date.now() - identifyStartedAt;

  // Retail search and stock image lookup removed — they added ~45s median
  // and produced 0% useful results. Emoji assignment handles product
  // images separately in app.js.
  const disabledStockImage: StockImageResponse = {
    status: "not_found",
    url: null,
    source: null,
    source_page_url: null,
    license: null,
    match_confidence: null,
    match_reason: null,
  };

  return {
    groceryItem,
    purchase_options: [],
    store_availability: [],
    stock_image: disabledStockImage,
    debug: {
      total_ms: Date.now() - startedAt,
      identify_ms,
      retailer_search_ms: 0,
      stock_image_ms: 0,
      model: getOpenAIModel(),
      retailer_link_source: "disabled",
    },
  };
}
