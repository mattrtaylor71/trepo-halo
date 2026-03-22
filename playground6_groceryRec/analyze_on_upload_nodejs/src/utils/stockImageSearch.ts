import fetch from "node-fetch";
import OpenAI from "openai";
import { GroceryItem } from "../openai/identifyGrocery";

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY environment variable is required");
    }
    client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
  return client;
}

function parseJsonResponse<T>(response: any): T {
  const outputText = response?.output_text;
  if (typeof outputText !== "string" || !outputText.trim()) {
    throw new Error("No structured output returned from OpenAI");
  }
  return JSON.parse(outputText) as T;
}

const imageVerificationJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["matches", "confidence", "reason"],
  properties: {
    matches: { type: "boolean" },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    reason: { type: "string" },
  },
};

export type StockImageResult = {
  url: string;
  source: "serpapi" | "wikimedia";
  source_page_url?: string | null;
  license?: string | null;
};

export type StockImageSearchOptions = {
  targetImageUrl?: string | null;
  targetDomain?: string | null;
  requireTargetMatch?: boolean;
};

type SearchContext = {
  targetImageUrl?: string | null;
  targetDomain?: string | null;
  requireTargetMatch: boolean;
  seenUrls: Set<string>;
};

function isProductImage(img: { title?: string; link?: string; source?: string }): boolean {
  const title = (img.title || "").toLowerCase();
  const link = (img.link || img.source || "").toLowerCase();

  const logoKeywords = ["logo", "emblem", "symbol", "trademark", "brand"];
  for (const keyword of logoKeywords) {
    if (title.includes(keyword) || link.includes(keyword)) {
      return false;
    }
  }

  return true;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function getVerifyAttempts(): number {
  const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_ATTEMPTS || "5", 10);
  if (!Number.isFinite(raw)) return 5;
  return clampNumber(raw, 1, 10);
}

function getVerifyMinMatches(attempts: number): number {
  const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_MIN_MATCHES || "", 10);
  if (Number.isFinite(raw) && raw > 0) {
    return clampNumber(raw, 1, attempts);
  }
  return Math.max(1, Math.ceil(attempts * 0.6));
}

function getVerifyMinConfidence(): number {
  const raw = parseFloat(process.env.OPENAI_IMAGE_VERIFY_MIN_CONFIDENCE || "0.65");
  if (!Number.isFinite(raw)) return 0.65;
  return clampNumber(raw, 0, 1);
}

function getVerifyParallelism(): number {
  const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_PARALLELISM || "3", 10);
  if (!Number.isFinite(raw)) return 3;
  return clampNumber(raw, 1, 6);
}

function getMaxCandidates(targetImageUrl?: string | null): number {
  const raw = parseInt(process.env.STOCK_IMAGE_MAX_CANDIDATES || "8", 10);
  if (!Number.isFinite(raw)) return 8;
  let max = clampNumber(raw, 1, 20);
  if (targetImageUrl) {
    const targetRaw = parseInt(process.env.STOCK_IMAGE_MAX_CANDIDATES_TARGET || "", 10);
    if (Number.isFinite(targetRaw)) {
      max = clampNumber(targetRaw, max, 20);
    }
  }
  return max;
}

function getSerpApiResultCount(targetImageUrl?: string | null): number {
  const raw = parseInt(process.env.STOCK_IMAGE_SERPAPI_NUM || "50", 10);
  if (!Number.isFinite(raw)) return 50;
  let max = clampNumber(raw, 10, 100);
  if (targetImageUrl) {
    const targetRaw = parseInt(process.env.STOCK_IMAGE_SERPAPI_NUM_TARGET || "", 10);
    if (Number.isFinite(targetRaw)) {
      max = clampNumber(targetRaw, max, 100);
    }
  }
  return max;
}

function getReversePageCandidateLimit(): number {
  const raw = parseInt(process.env.STOCK_IMAGE_REVERSE_PAGE_CANDIDATES || "3", 10);
  if (!Number.isFinite(raw)) return 3;
  return clampNumber(raw, 1, 10);
}

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

const SUPPORTED_IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif"]);
const UNSUPPORTED_IMAGE_EXTENSIONS = new Set([
  ".svg",
  ".bmp",
  ".tif",
  ".tiff",
  ".heic",
  ".heif",
  ".webm",
  ".mp4",
  ".mov",
  ".avi",
  ".mkv",
  ".ogg",
  ".ogv",
  ".pdf",
]);

function getUrlExtension(url: string): string {
  if (!url) return "";
  if (url.startsWith("data:image/")) {
    const match = url.match(/^data:image\/([a-z0-9+.-]+);/i);
    return match ? `.${match[1].toLowerCase()}` : "";
  }
  try {
    const parsed = new URL(url);
    const pathname = parsed.pathname || "";
    const lastDot = pathname.lastIndexOf(".");
    if (lastDot === -1) return "";
    return pathname.slice(lastDot).toLowerCase();
  } catch {
    return "";
  }
}

function isSupportedImageUrl(url: string): boolean {
  if (!url) return false;
  if (url.startsWith("data:image/")) return true;
  const ext = getUrlExtension(url);
  if (!ext) return true;
  if (UNSUPPORTED_IMAGE_EXTENSIONS.has(ext)) return false;
  return SUPPORTED_IMAGE_EXTENSIONS.has(ext);
}

function isHttpUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  return url.startsWith("http://") || url.startsWith("https://");
}

function getMaxImageBytes(): number {
  const raw = parseInt(process.env.STOCK_IMAGE_MAX_BYTES || "4000000", 10);
  if (!Number.isFinite(raw)) return 4_000_000;
  return clampNumber(raw, 100_000, 10_000_000);
}

async function fetchImageAsDataUrl(url: string): Promise<string | null> {
  try {
    if (!isHttpUrl(url)) return null;
    const controller = new AbortController();
    const timeoutMs = parseInt(process.env.STOCK_IMAGE_FETCH_TIMEOUT_MS || "4000", 10);
    const timeout = Number.isFinite(timeoutMs) ? timeoutMs : 4000;
    const timer = setTimeout(() => controller.abort(), timeout);

    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;

    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (!contentType.startsWith("image/")) return null;

    const contentLength = parseInt(res.headers.get("content-length") || "", 10);
    const maxBytes = getMaxImageBytes();
    if (Number.isFinite(contentLength) && contentLength > maxBytes) return null;

    const buffer = Buffer.from(await res.arrayBuffer());
    if (buffer.length > maxBytes) return null;

    const base64 = buffer.toString("base64");
    return `data:${contentType};base64,${base64}`;
  } catch {
    return null;
  }
}

async function resolveCandidateImageUrl(
  url: string,
  requireTargetMatch: boolean
): Promise<string | null> {
  if (url.startsWith("data:image/")) return url;
  const resolved = await fetchImageAsDataUrl(url);
  if (resolved) return resolved;
  return requireTargetMatch ? null : url;
}

function normalizeUrlForMatch(value: string): string {
  return value.split("?")[0].toLowerCase();
}

function isLikelyUserGeneratedImageUrl(value?: string | null): boolean {
  if (!value || !isHttpUrl(value)) return false;
  try {
    const host = new URL(value).hostname.toLowerCase();
    if (host.includes("bazaarvoice")) return true;
    if (host.includes("ugc")) return true;
    if (host.includes("review")) return true;
    return false;
  } catch {
    return false;
  }
}

function extractMetaImageUrl(html: string, pageUrl: string): string | null {
  const patterns = [
    /property=["']og:image["'][^>]*content=["']([^"']+)["']/i,
    /property=["']og:image:secure_url["'][^>]*content=["']([^"']+)["']/i,
    /name=["']twitter:image["'][^>]*content=["']([^"']+)["']/i,
  ];
  for (const pattern of patterns) {
    const match = html.match(pattern);
    if (match?.[1]) {
      try {
        return new URL(match[1], pageUrl).toString();
      } catch {
        return match[1];
      }
    }
  }
  return null;
}

async function fetchPageImageUrl(pageUrl: string): Promise<string | null> {
  try {
    if (!isHttpUrl(pageUrl)) return null;
    const controller = new AbortController();
    const timeoutMs = parseInt(process.env.STOCK_IMAGE_PAGE_FETCH_TIMEOUT_MS || "4000", 10);
    const timeout = Number.isFinite(timeoutMs) ? timeoutMs : 4000;
    const timer = setTimeout(() => controller.abort(), timeout);
    const res = await fetch(pageUrl, {
      signal: controller.signal,
      headers: { "user-agent": "Mozilla/5.0" },
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (!contentType.includes("text/html")) return null;
    const html = await res.text();
    const metaImage = extractMetaImageUrl(html, pageUrl);
    return metaImage && isSupportedImageUrl(metaImage) ? metaImage : null;
  } catch {
    return null;
  }
}

function getTargetDomain(targetImageUrl?: string | null): string | null {
  if (!targetImageUrl) return null;
  try {
    const url = new URL(targetImageUrl);
    return url.hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

function getTargetKeywords(targetImageUrl?: string | null): string[] {
  if (!targetImageUrl) return [];
  try {
    const url = new URL(targetImageUrl);
    const filename = url.pathname.split("/").pop() || "";
    return filename
      .toLowerCase()
      .replace(/\.(jpg|jpeg|png|webp|gif)$/i, "")
      .split(/[^a-z0-9]+/g)
      .filter(token => token.length > 2);
  } catch {
    return [];
  }
}

function scoreCandidate(
  candidateText: string,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  variant?: string | null,
  targetDomain?: string | null
): number {
  const text = normalizeText(candidateText);
  let score = 0;

  const negativeKeywords = ["logo", "vector", "clipart", "icon", "nutrition", "recipe", "review"];
  for (const keyword of negativeKeywords) {
    if (text.includes(keyword)) score -= 2;
  }

  if (brand) {
    const b = normalizeText(brand);
    if (text.includes(b)) score += 4;
  }
  if (productName) {
    const p = normalizeText(productName);
    if (text.includes(p)) score += 5;
    const parts = p.split(" ").filter(Boolean);
    const matchedParts = parts.filter(part => text.includes(part));
    score += Math.min(3, matchedParts.length);
  }
  if (variant) {
    const v = normalizeText(variant);
    if (text.includes(v)) score += 2;
  }
  if (category) {
    const c = normalizeText(category);
    if (text.includes(c)) score += 1;
  }
  if (targetDomain) {
    const d = normalizeText(targetDomain);
    if (text.includes(d)) score += 2;
  }

  return score;
}

function buildProductDescription(
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null
): string {
  const parts: string[] = [];
  if (brand) parts.push(`brand: ${brand}`);
  if (productName) parts.push(`product: ${productName}`);
  if (category) parts.push(`category: ${category}`);
  if (productDescription) parts.push(`description: ${productDescription}`);
  return parts.join(", ");
}

async function runVisionCheck(
  imageUrl: string,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null,
  targetImageUrl?: string | null
): Promise<{ matches: boolean; confidence: number; reason: string }> {
  const description = buildProductDescription(productName, brand, category, productDescription);
  const hasTarget = !!targetImageUrl;
  if (!description && !hasTarget) {
    return { matches: false, confidence: 0, reason: "No product details or target image to verify" };
  }
  if (targetImageUrl && normalizeUrlForMatch(targetImageUrl) === normalizeUrlForMatch(imageUrl)) {
    return { matches: true, confidence: 1, reason: "Exact URL match" };
  }

  const openai = getClient();
  const model = process.env.OPENAI_IMAGE_VERIFY_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-2026-03-05";

  try {
    const userText = hasTarget
      ? `Compare the reference image (Image A) with the candidate image (Image B).
Return JSON: { "matches": true/false, "confidence": 0.0-1.0, "reason": "..." }.
matches=true ONLY if the images show the same product packaging and appear to be the same stock photo or near-identical.
${description ? `Product details to consider: ${description}` : ""}`
      : `Does this image show: ${description}?
Return JSON: { "matches": true/false, "confidence": 0.0-1.0, "reason": "..." }`;
    const response = await openai.responses.create({
      model,
      reasoning: { effort: "low" },
      max_output_tokens: 120,
      text: {
        format: {
          type: "json_schema",
          name: "image_verification",
          strict: true,
          schema: imageVerificationJsonSchema,
        },
      } as any,
      input: [
        {
          role: "system",
          content: [{
            type: "input_text",
            text: `You are a strict image verification expert. Determine if an image matches a specific grocery product description or reference image.
Return ONLY valid JSON with: { "matches": boolean, "confidence": number (0-1), "reason": string }.
- matches: true ONLY if the image clearly shows the same product/brand/packaging or is a near-identical stock photo match.
- If it is a different product, a logo, unrelated, or uncertain, return matches=false.`,
          }],
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: userText,
            },
            ...(hasTarget
              ? [
                  {
                    type: "input_image" as const,
                    image_url: targetImageUrl!,
                  },
                ]
              : []),
            {
              type: "input_image",
              image_url: imageUrl,
            },
          ],
        },
      ],
    } as any);

    const parsed = parseJsonResponse<{ matches?: boolean; confidence?: number; reason?: string }>(response);
    return {
      matches: parsed.matches === true,
      confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0,
      reason: typeof parsed.reason === "string" ? parsed.reason : "No reason provided",
    };
  } catch (error) {
    console.error("[stock-image] Verification error:", error);
    return { matches: false, confidence: 0, reason: "Verification failed" };
  }
}

async function verifyImageMatch(
  imageUrl: string,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null,
  targetImageUrl?: string | null
): Promise<{ matches: boolean; confidence: number; reason: string }> {
  const attempts = getVerifyAttempts();
  const minMatches = getVerifyMinMatches(attempts);
  const minConfidence = getVerifyMinConfidence();
  const parallelism = getVerifyParallelism();
  let matchCount = 0;
  let totalConfidence = 0;
  let lastReason = "No verification attempts";
  let attemptsUsed = 0;

  for (let i = 0; i < attempts; i += parallelism) {
    const batchSize = Math.min(parallelism, attempts - i);
    const batch = await Promise.all(
      Array.from({ length: batchSize }).map(() =>
        runVisionCheck(
          imageUrl,
          productName,
          brand,
          category,
          productDescription,
          targetImageUrl
        )
      )
    );

    for (const result of batch) {
      attemptsUsed += 1;
      lastReason = result.reason;
      totalConfidence += result.confidence;
      if (result.matches) {
        matchCount += 1;
      }
    }

    const remaining = attempts - attemptsUsed;
    const avgConfidence = totalConfidence / attemptsUsed;
    if (matchCount >= minMatches && avgConfidence >= minConfidence) {
      return { matches: true, confidence: avgConfidence, reason: lastReason };
    }
    if (matchCount + remaining < minMatches) {
      break;
    }
  }

  const avgConfidence = attemptsUsed > 0 ? totalConfidence / attemptsUsed : 0;
  return { matches: false, confidence: avgConfidence, reason: lastReason };
}

async function searchSerpAPI(
  query: string,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null,
  variant?: string | null,
  context?: SearchContext
): Promise<StockImageResult | null> {
  if (!process.env.SERPAPI_API_KEY) return null;

  const enhancedQuery = `${query} packaging product -logo -emblem`;
  const resultCount = getSerpApiResultCount(context?.targetImageUrl);
  const url =
    `https://serpapi.com/search.json?q=${encodeURIComponent(
      enhancedQuery
    )}&tbm=isch&imgtype=photo&api_key=${process.env.SERPAPI_API_KEY}&num=${resultCount}`;

  const res = await fetch(url).then((r: any) => r.json());
  if (!res.images_results?.length) return null;

  const candidateImages = res.images_results
    .map((img: any) => ({
      url: img.original || img.thumbnail || img.image || null,
      title: img.title,
      link: img.link,
      source: img.source,
      score: scoreCandidate(
        `${img.title || ""} ${img.link || ""} ${img.source || ""}`,
        productName,
        brand,
        category,
        variant,
        context?.targetDomain || null
      ),
    }))
    .filter((img: any) => {
      if (!img.url || !isProductImage(img)) return false;
      if (!isSupportedImageUrl(img.url)) return false;
      if (!context?.seenUrls) return true;
      const normalized = normalizeUrlForMatch(img.url);
      if (context.seenUrls.has(normalized)) return false;
      context.seenUrls.add(normalized);
      return true;
    })
    .sort((a: any, b: any) => b.score - a.score)
    .slice(0, getMaxCandidates(context?.targetImageUrl));
  if (candidateImages.length === 0) return null;

  for (const img of candidateImages) {
    try {
      if (
        context?.targetImageUrl &&
        normalizeUrlForMatch(context.targetImageUrl) === normalizeUrlForMatch(img.url)
      ) {
        return {
          url: img.url,
          source: "serpapi",
          source_page_url: img.link || img.source || null,
          license: null,
        };
      }
      const candidateUrl = await resolveCandidateImageUrl(
        img.url,
        context?.requireTargetMatch ?? false
      );
      if (!candidateUrl) {
        continue;
      }
      const verification = await verifyImageMatch(
        candidateUrl,
        productName,
        brand,
        category,
        productDescription,
        context?.targetImageUrl || null
      );
      if (verification.matches) {
        return {
          url: img.url,
          source: "serpapi",
          source_page_url: img.link || img.source || null,
          license: null,
        };
      }
    } catch (error) {
      console.warn("[stock-image] SerpAPI verification error:", error);
    }
  }

  return null;
}

async function searchSerpAPIReverseImage(
  targetImageUrl: string | null,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null,
  variant?: string | null,
  context?: SearchContext
): Promise<StockImageResult | null> {
  if (!process.env.SERPAPI_API_KEY || !targetImageUrl || !isHttpUrl(targetImageUrl)) {
    return null;
  }

  const url =
    `https://serpapi.com/search.json?engine=google_reverse_image&image_url=${encodeURIComponent(
      targetImageUrl
    )}&api_key=${process.env.SERPAPI_API_KEY}`;

  const res = await fetch(url).then((r: any) => r.json());
  const inlineImages = Array.isArray(res.inline_images) ? res.inline_images : [];
  const imageResults = Array.isArray(res.image_results) ? res.image_results : [];

  const pageCandidates = [
    ...imageResults.map((img: any) => img.link || img.redirect_link || null),
    ...inlineImages.map((img: any) => img.source || img.link || null),
  ]
    .filter((link: any) => typeof link === "string" && isHttpUrl(link))
    .filter((link: string) => {
      if (!context?.seenUrls) return true;
      const normalized = normalizeUrlForMatch(link);
      if (context.seenUrls.has(normalized)) return false;
      context.seenUrls.add(normalized);
      return true;
    })
    .slice(0, getReversePageCandidateLimit());

  for (const pageUrl of pageCandidates) {
    const pageImageUrl = await fetchPageImageUrl(pageUrl);
    if (!pageImageUrl) continue;
    if (
      context?.targetImageUrl &&
      normalizeUrlForMatch(context.targetImageUrl) === normalizeUrlForMatch(pageImageUrl)
    ) {
      return {
        url: pageImageUrl,
        source: "serpapi",
        source_page_url: pageUrl,
        license: null,
      };
    }
    if (!context?.requireTargetMatch || isLikelyUserGeneratedImageUrl(context?.targetImageUrl)) {
      return {
        url: pageImageUrl,
        source: "serpapi",
        source_page_url: pageUrl,
        license: null,
      };
    }
    const candidateUrl = await resolveCandidateImageUrl(
      pageImageUrl,
      context?.requireTargetMatch ?? false
    );
    if (!candidateUrl) {
      continue;
    }
    const verification = await verifyImageMatch(
      candidateUrl,
      productName,
      brand,
      category,
      productDescription,
      context?.targetImageUrl || null
    );
    if (verification.matches) {
      return {
        url: pageImageUrl,
        source: "serpapi",
        source_page_url: pageUrl,
        license: null,
      };
    }
  }

  const candidates = [
    ...inlineImages.map((img: any) => ({
      url: img.original || img.thumbnail || null,
      title: img.title,
      link: img.source || img.link,
      source: img.source,
      score: scoreCandidate(
        `${img.title || ""} ${img.source || ""} ${img.link || ""}`,
        productName,
        brand,
        category,
        variant,
        context?.targetDomain || null
      ),
    })),
    ...imageResults.map((img: any) => ({
      url: img.thumbnail || img.image || null,
      title: img.title,
      link: img.link || img.redirect_link,
      source: img.source,
      score: scoreCandidate(
        `${img.title || ""} ${img.source || ""} ${img.link || ""}`,
        productName,
        brand,
        category,
        variant,
        context?.targetDomain || null
      ),
    })),
  ]
    .filter((img: any) => img.url && isSupportedImageUrl(img.url))
    .filter((img: any) => {
      if (!context?.seenUrls) return true;
      const normalized = normalizeUrlForMatch(img.url);
      if (context.seenUrls.has(normalized)) return false;
      context.seenUrls.add(normalized);
      return true;
    })
    .sort((a: any, b: any) => b.score - a.score)
    .slice(0, getMaxCandidates(context?.targetImageUrl));

  for (const img of candidates) {
    if (
      context?.targetImageUrl &&
      normalizeUrlForMatch(context.targetImageUrl) === normalizeUrlForMatch(img.url)
    ) {
      return {
        url: img.url,
        source: "serpapi",
        source_page_url: img.link || img.source || null,
        license: null,
      };
    }
    const candidateUrl = await resolveCandidateImageUrl(
      img.url,
      context?.requireTargetMatch ?? false
    );
    if (!candidateUrl) {
      continue;
    }
    const verification = await verifyImageMatch(
      candidateUrl,
      productName,
      brand,
      category,
      productDescription,
      context?.targetImageUrl || null
    );
    if (verification.matches) {
      return {
        url: img.url,
        source: "serpapi",
        source_page_url: img.link || img.source || null,
        license: null,
      };
    }
  }

  return null;
}

async function searchWikimedia(
  query: string,
  productName?: string | null,
  brand?: string | null,
  category?: string | null,
  productDescription?: string | null,
  context?: SearchContext
): Promise<StockImageResult | null> {
  const enhancedQuery = `${query} packaging product -logo -brand`;

  const searchUrl =
    `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(
      enhancedQuery
    )}&format=json&srlimit=10`;

  const searchRes = await fetch(searchUrl).then((r: any) => r.json());
  if (!searchRes.query?.search?.length) return null;

  const pages = searchRes.query.search.slice(0, 5);
  for (const pageResult of pages) {
    const pageTitle = pageResult.title;
    const titleLower = pageTitle.toLowerCase();
    if (titleLower.includes("logo") || titleLower.includes("brand") || titleLower.includes("trademark")) {
      continue;
    }

    const imageUrl =
      `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(
        pageTitle
      )}&prop=pageimages|images&pithumbsize=1000&format=json`;

    const imageRes = await fetch(imageUrl).then((r: any) => r.json());
    const page = Object.values(imageRes.query.pages)[0] as any;

    if (page.thumbnail?.source && isSupportedImageUrl(page.thumbnail.source)) {
      const normalized = normalizeUrlForMatch(page.thumbnail.source);
      if (context?.seenUrls && context.seenUrls.has(normalized)) {
        continue;
      }
      context?.seenUrls?.add(normalized);
      if (
        context?.targetImageUrl &&
        normalizeUrlForMatch(context.targetImageUrl) === normalized
      ) {
        return {
          url: page.thumbnail.source,
          source: "wikimedia",
          source_page_url: `https://en.wikipedia.org/wiki/${pageTitle.replace(/ /g, "_")}`,
          license: "Wikimedia Commons",
        };
      }
      const candidateUrl = await resolveCandidateImageUrl(
        page.thumbnail.source,
        context?.requireTargetMatch ?? false
      );
      if (!candidateUrl) {
        continue;
      }
      const verification = await verifyImageMatch(
        candidateUrl,
        productName,
        brand,
        category,
        productDescription,
        context?.targetImageUrl || null
      );
      if (verification.matches) {
        return {
          url: page.thumbnail.source,
          source: "wikimedia",
          source_page_url: `https://en.wikipedia.org/wiki/${pageTitle.replace(/ /g, "_")}`,
          license: "Wikimedia Commons",
        };
      }
    }

    if (page.images && Array.isArray(page.images)) {
      for (const img of page.images.slice(0, 5)) {
        const imgTitle = img.title;
        if (
          imgTitle &&
          !imgTitle.toLowerCase().includes("logo") &&
          isSupportedImageUrl(imgTitle)
        ) {
          const imgInfoUrl = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(
            imgTitle
          )}&prop=imageinfo&iiprop=url&format=json`;
          const imgInfoRes = await fetch(imgInfoUrl).then((r: any) => r.json());
          const imgPage = Object.values(imgInfoRes.query.pages)[0] as any;
          if (imgPage.imageinfo?.[0]?.url) {
            const imgUrl = imgPage.imageinfo[0].url;
            if (!isSupportedImageUrl(imgUrl)) {
              continue;
            }
            const normalized = normalizeUrlForMatch(imgUrl);
            if (context?.seenUrls && context.seenUrls.has(normalized)) {
              continue;
            }
            context?.seenUrls?.add(normalized);
            if (
              context?.targetImageUrl &&
              normalizeUrlForMatch(context.targetImageUrl) === normalized
            ) {
              return {
                url: imgUrl,
                source: "wikimedia",
                source_page_url: `https://commons.wikimedia.org/wiki/${imgTitle.replace(/ /g, "_")}`,
                license: "Wikimedia Commons",
              };
            }
            const candidateUrl = await resolveCandidateImageUrl(
              imgUrl,
              context?.requireTargetMatch ?? false
            );
            if (!candidateUrl) {
              continue;
            }
            const verification = await verifyImageMatch(
              candidateUrl,
              productName,
              brand,
              category,
              productDescription,
              context?.targetImageUrl || null
            );
            if (verification.matches) {
              return {
                url: imgUrl,
                source: "wikimedia",
                source_page_url: `https://commons.wikimedia.org/wiki/${imgTitle.replace(/ /g, "_")}`,
                license: "Wikimedia Commons",
              };
            }
          }
        }
      }
    }
  }

  return null;
}

function buildQueries(
  groceryItem: GroceryItem,
  options?: { targetDomain?: string | null; targetKeywords?: string[] }
): string[] {
  const queries: string[] = [];
  const productName = groceryItem.product_name?.trim();
  const brand = groceryItem.brand?.trim();
  const category = groceryItem.category?.trim();
  const variant = groceryItem.variant?.trim();
  const description = groceryItem.product_description?.trim();
  const targetDomain = options?.targetDomain || null;
  const targetKeywords = options?.targetKeywords || [];
  const storeSites = [
    "walmart.com",
    "target.com",
    "kroger.com",
    "instacart.com",
    "amazon.com",
    "costco.com",
    "wholefoodsmarket.com",
  ];

  if (targetKeywords.length > 0) {
    const keywordPhrase = targetKeywords.slice(0, 5).join(" ");
    if (keywordPhrase) {
      queries.push(`${keywordPhrase} packaging`);
      if (productName) {
        queries.push(`${productName} ${keywordPhrase}`);
      }
    }
  }
  if (brand && productName && variant) {
    queries.push(`${brand} ${productName} ${variant} packaging`);
  }
  if (brand && productName) {
    queries.push(`${brand} ${productName} packaging`);
    queries.push(`${brand} ${productName}`);
    queries.push(`${brand} ${productName} product photo`);
  }
  if (productName) {
    queries.push(`${productName} packaging product`);
    queries.push(`${productName} product photo`);
  }
  if (description) {
    queries.push(`${description} packaging`);
    queries.push(`${description} product photo`);
  }
  if (category) {
    queries.push(`${category} packaging`);
    queries.push(`${category} product`);
  }
  if (productName) {
    const productWords = productName.toLowerCase().split(/\s+/);
    const mainProduct = productWords[productWords.length - 1];
    if (mainProduct && mainProduct.length > 3) {
      queries.push(`${mainProduct} packaging`);
      queries.push(`${mainProduct} product`);
    }
  }
  if (productName) {
    const storeQueryBase = brand ? `${brand} ${productName}` : productName;
    for (const site of storeSites) {
      queries.push(`${storeQueryBase} site:${site}`);
    }
  }
  if (targetDomain) {
    const storeQueryBase = brand && productName ? `${brand} ${productName}` : productName;
    if (storeQueryBase) {
      queries.push(`${storeQueryBase} site:${targetDomain}`);
    }
    if (targetKeywords.length > 0) {
      const keywordPhrase = targetKeywords.slice(0, 5).join(" ");
      if (keywordPhrase) {
        queries.push(`${keywordPhrase} site:${targetDomain}`);
      }
    }
  }

  return Array.from(new Set(queries.map(q => q.trim()).filter(Boolean)));
}

export async function findStockImage(
  groceryItem: GroceryItem,
  options: StockImageSearchOptions = {}
): Promise<StockImageResult | null> {
  const hasIdentity = !!(
    groceryItem.product_name ||
    groceryItem.brand ||
    groceryItem.product_description
  );
  if (!hasIdentity) {
    return null;
  }

  const targetImageUrl = options.targetImageUrl?.trim() || null;
  const targetDomain = options.targetDomain || getTargetDomain(targetImageUrl);
  const targetKeywords = getTargetKeywords(targetImageUrl);
  const requireTargetMatch =
    options.requireTargetMatch ??
    (!!targetImageUrl && process.env.STOCK_IMAGE_REQUIRE_TARGET_MATCH === "true");
  const context: SearchContext = {
    targetImageUrl,
    targetDomain,
    requireTargetMatch,
    seenUrls: new Set<string>(),
  };
  const queries = buildQueries(groceryItem, { targetDomain, targetKeywords });

  if (targetImageUrl) {
    try {
      const reverseResult = await searchSerpAPIReverseImage(
        targetImageUrl,
        groceryItem.product_name,
        groceryItem.brand,
        groceryItem.category,
        groceryItem.product_description,
        groceryItem.variant,
        context
      );
      if (reverseResult) return reverseResult;
    } catch (error) {
      console.warn("[stock-image] SerpAPI reverse search failed:", error);
    }
  }

  for (const query of queries) {
    try {
      const result = await searchSerpAPI(
        query,
        groceryItem.product_name,
        groceryItem.brand,
        groceryItem.category,
        groceryItem.product_description,
        groceryItem.variant,
        context
      );
      if (result) return result;
    } catch (error) {
      console.warn("[stock-image] SerpAPI search failed:", error);
    }
  }

  const wikimediaQueries = targetImageUrl ? queries.slice(0, 5) : queries.slice(0, 2);
  for (const query of wikimediaQueries) {
    try {
      const result = await searchWikimedia(
        query,
        groceryItem.product_name,
        groceryItem.brand,
        groceryItem.category,
        groceryItem.product_description,
        context
      );
      if (result) return result;
    } catch (error) {
      console.warn("[stock-image] Wikimedia search failed:", error);
    }
  }

  return null;
}
