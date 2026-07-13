"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.findStockImage = findStockImage;
const node_fetch_1 = __importDefault(require("node-fetch"));
const openai_1 = __importDefault(require("openai"));
let client = null;
function getClient() {
    if (!client) {
        if (!process.env.OPENAI_API_KEY) {
            throw new Error("OPENAI_API_KEY environment variable is required");
        }
        client = new openai_1.default({
            apiKey: process.env.OPENAI_API_KEY,
        });
    }
    return client;
}
function isProductImage(img) {
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
function clampNumber(value, min, max) {
    return Math.min(max, Math.max(min, value));
}
function getVerifyAttempts() {
    const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_ATTEMPTS || "5", 10);
    if (!Number.isFinite(raw))
        return 5;
    return clampNumber(raw, 1, 10);
}
function getVerifyMinMatches(attempts) {
    const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_MIN_MATCHES || "", 10);
    if (Number.isFinite(raw) && raw > 0) {
        return clampNumber(raw, 1, attempts);
    }
    return Math.max(1, Math.ceil(attempts * 0.6));
}
function getVerifyMinConfidence() {
    const raw = parseFloat(process.env.OPENAI_IMAGE_VERIFY_MIN_CONFIDENCE || "0.65");
    if (!Number.isFinite(raw))
        return 0.65;
    return clampNumber(raw, 0, 1);
}
function getVerifyParallelism() {
    const raw = parseInt(process.env.OPENAI_IMAGE_VERIFY_PARALLELISM || "3", 10);
    if (!Number.isFinite(raw))
        return 3;
    return clampNumber(raw, 1, 6);
}
function getMaxCandidates(targetImageUrl) {
    const raw = parseInt(process.env.STOCK_IMAGE_MAX_CANDIDATES || "8", 10);
    if (!Number.isFinite(raw))
        return 8;
    let max = clampNumber(raw, 1, 20);
    if (targetImageUrl) {
        const targetRaw = parseInt(process.env.STOCK_IMAGE_MAX_CANDIDATES_TARGET || "", 10);
        if (Number.isFinite(targetRaw)) {
            max = clampNumber(targetRaw, max, 20);
        }
    }
    return max;
}
function getSerpApiResultCount(targetImageUrl) {
    const raw = parseInt(process.env.STOCK_IMAGE_SERPAPI_NUM || "50", 10);
    if (!Number.isFinite(raw))
        return 50;
    let max = clampNumber(raw, 10, 100);
    if (targetImageUrl) {
        const targetRaw = parseInt(process.env.STOCK_IMAGE_SERPAPI_NUM_TARGET || "", 10);
        if (Number.isFinite(targetRaw)) {
            max = clampNumber(targetRaw, max, 100);
        }
    }
    return max;
}
function getReversePageCandidateLimit() {
    const raw = parseInt(process.env.STOCK_IMAGE_REVERSE_PAGE_CANDIDATES || "3", 10);
    if (!Number.isFinite(raw))
        return 3;
    return clampNumber(raw, 1, 10);
}
function normalizeText(value) {
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
function getUrlExtension(url) {
    if (!url)
        return "";
    if (url.startsWith("data:image/")) {
        const match = url.match(/^data:image\/([a-z0-9+.-]+);/i);
        return match ? `.${match[1].toLowerCase()}` : "";
    }
    try {
        const parsed = new URL(url);
        const pathname = parsed.pathname || "";
        const lastDot = pathname.lastIndexOf(".");
        if (lastDot === -1)
            return "";
        return pathname.slice(lastDot).toLowerCase();
    }
    catch {
        return "";
    }
}
function isSupportedImageUrl(url) {
    if (!url)
        return false;
    if (url.startsWith("data:image/"))
        return true;
    const ext = getUrlExtension(url);
    if (!ext)
        return true;
    if (UNSUPPORTED_IMAGE_EXTENSIONS.has(ext))
        return false;
    return SUPPORTED_IMAGE_EXTENSIONS.has(ext);
}
function isHttpUrl(url) {
    if (!url)
        return false;
    return url.startsWith("http://") || url.startsWith("https://");
}
function getMaxImageBytes() {
    const raw = parseInt(process.env.STOCK_IMAGE_MAX_BYTES || "4000000", 10);
    if (!Number.isFinite(raw))
        return 4000000;
    return clampNumber(raw, 100000, 10000000);
}
async function fetchImageAsDataUrl(url) {
    try {
        if (!isHttpUrl(url))
            return null;
        const controller = new AbortController();
        const timeoutMs = parseInt(process.env.STOCK_IMAGE_FETCH_TIMEOUT_MS || "4000", 10);
        const timeout = Number.isFinite(timeoutMs) ? timeoutMs : 4000;
        const timer = setTimeout(() => controller.abort(), timeout);
        const res = await (0, node_fetch_1.default)(url, { signal: controller.signal });
        clearTimeout(timer);
        if (!res.ok)
            return null;
        const contentType = (res.headers.get("content-type") || "").toLowerCase();
        if (!contentType.startsWith("image/"))
            return null;
        const contentLength = parseInt(res.headers.get("content-length") || "", 10);
        const maxBytes = getMaxImageBytes();
        if (Number.isFinite(contentLength) && contentLength > maxBytes)
            return null;
        const buffer = Buffer.from(await res.arrayBuffer());
        if (buffer.length > maxBytes)
            return null;
        const base64 = buffer.toString("base64");
        return `data:${contentType};base64,${base64}`;
    }
    catch {
        return null;
    }
}
async function resolveCandidateImageUrl(url, requireTargetMatch) {
    if (url.startsWith("data:image/"))
        return url;
    const resolved = await fetchImageAsDataUrl(url);
    if (resolved)
        return resolved;
    return requireTargetMatch ? null : url;
}
function normalizeUrlForMatch(value) {
    return value.split("?")[0].toLowerCase();
}
function isLikelyUserGeneratedImageUrl(value) {
    if (!value || !isHttpUrl(value))
        return false;
    try {
        const host = new URL(value).hostname.toLowerCase();
        if (host.includes("bazaarvoice"))
            return true;
        if (host.includes("ugc"))
            return true;
        if (host.includes("reviews"))
            return true;
        if (host.includes("review"))
            return true;
        if (host.includes("usercontent"))
            return true;
        return false;
    }
    catch {
        return false;
    }
}
function extractMetaImageUrl(html, pageUrl) {
    if (!html)
        return null;
    const ogImageMatch = html.match(/property=["']og:image["'][^>]*content=["']([^"']+)["']/i);
    if (ogImageMatch && ogImageMatch[1]) {
        const value = ogImageMatch[1].trim();
        try {
            return new URL(value, pageUrl).toString();
        }
        catch {
            return value;
        }
    }
    const twitterMatch = html.match(/name=["']twitter:image["'][^>]*content=["']([^"']+)["']/i);
    if (twitterMatch && twitterMatch[1]) {
        const value = twitterMatch[1].trim();
        try {
            return new URL(value, pageUrl).toString();
        }
        catch {
            return value;
        }
    }
    return null;
}
async function fetchPageImageUrl(pageUrl) {
    try {
        if (!pageUrl || !isHttpUrl(pageUrl))
            return null;
        const controller = new AbortController();
        const timeoutMs = parseInt(process.env.STOCK_IMAGE_PAGE_FETCH_TIMEOUT_MS || "4000", 10);
        const timeout = Number.isFinite(timeoutMs) ? timeoutMs : 4000;
        const timer = setTimeout(() => controller.abort(), timeout);
        const res = await (0, node_fetch_1.default)(pageUrl, {
            signal: controller.signal,
            headers: {
                "user-agent": "Mozilla/5.0 (stock-image-bot)",
            },
        });
        clearTimeout(timer);
        if (!res.ok)
            return null;
        const contentType = (res.headers.get("content-type") || "").toLowerCase();
        if (!contentType.includes("text/html"))
            return null;
        const html = await res.text();
        return extractMetaImageUrl(html, pageUrl);
    }
    catch {
        return null;
    }
}
function buildQueries(groceryItem, targetDomain) {
    const parts = [];
    if (groceryItem.brand && groceryItem.product_name) {
        parts.push(`${groceryItem.brand} ${groceryItem.product_name}`);
    }
    else if (groceryItem.product_name) {
        parts.push(groceryItem.product_name);
    }
    else if (groceryItem.brand) {
        parts.push(groceryItem.brand);
    }
    if (groceryItem.variant)
        parts.push(groceryItem.variant);
    if (groceryItem.category)
        parts.push(groceryItem.category);
    if (groceryItem.ingredients && groceryItem.ingredients.length > 0) {
        const ingredients = groceryItem.ingredients.slice(0, 3);
        parts.push(ingredients.join(" "));
    }
    if (groceryItem.nutrition) {
        const nutrition = [];
        if (groceryItem.nutrition.calories)
            nutrition.push(`${groceryItem.nutrition.calories} calories`);
        if (groceryItem.nutrition.protein)
            nutrition.push(`${groceryItem.nutrition.protein} protein`);
        if (groceryItem.nutrition.total_fat)
            nutrition.push(`${groceryItem.nutrition.total_fat} fat`);
        if (groceryItem.nutrition.total_carbohydrates)
            nutrition.push(`${groceryItem.nutrition.total_carbohydrates} carbs`);
        if (nutrition.length > 0)
            parts.push(nutrition.join(" "));
    }
    const baseQuery = parts.join(" ").trim();
    const queries = [];
    if (baseQuery) {
        queries.push(baseQuery);
        queries.push(`${baseQuery} product image`);
        queries.push(`${baseQuery} package`);
        queries.push(`${baseQuery} packaging`);
        queries.push(`${baseQuery} label`);
    }
    if (targetDomain) {
        const cleanDomain = targetDomain.replace(/^www\./i, "");
        queries.push(`${baseQuery} site:${cleanDomain}`.trim());
    }
    return Array.from(new Set(queries.map((q) => q.trim()).filter(Boolean)));
}
async function searchSerpAPI(query, context) {
    const apiKey = process.env.SERPAPI_API_KEY;
    if (!apiKey)
        return [];
    const params = new URLSearchParams({
        engine: "google_images",
        q: query,
        api_key: apiKey,
        num: String(getSerpApiResultCount(context?.targetImageUrl)),
    });
    try {
        const res = await (0, node_fetch_1.default)(`https://serpapi.com/search.json?${params.toString()}`);
        if (!res.ok)
            return [];
        const data = await res.json();
        const results = Array.isArray(data.images_results)
            ? data.images_results
            : Array.isArray(data.inline_images)
                ? data.inline_images
                : [];
        return results
            .map((img) => ({
            url: img.original || img.link || img.source || img.thumbnail,
            source: "serpapi",
            source_page_url: img.link || img.source || null,
            title: img.title || img.snippet || null,
        }))
            .filter((img) => !!img.url && isHttpUrl(img.url));
    }
    catch {
        return [];
    }
}
async function searchSerpAPIReverseImage(imageUrl, context) {
    const apiKey = process.env.SERPAPI_API_KEY;
    if (!apiKey)
        return [];
    const params = new URLSearchParams({
        engine: "google_reverse_image",
        image_url: imageUrl,
        api_key: apiKey,
        num: String(getSerpApiResultCount(context?.targetImageUrl)),
    });
    try {
        const res = await (0, node_fetch_1.default)(`https://serpapi.com/search.json?${params.toString()}`);
        if (!res.ok)
            return [];
        const data = await res.json();
        const inlineImages = Array.isArray(data.inline_images) ? data.inline_images : [];
        const imageResults = Array.isArray(data.image_results) ? data.image_results : [];
        const pageCandidates = [
            ...imageResults.map((img) => img.link || img.redirect_link || null),
            ...inlineImages.map((img) => img.source || img.link || null),
        ]
            .filter((link) => typeof link === "string" && isHttpUrl(link))
            .filter((link) => {
            if (!context?.seenUrls)
                return true;
            const normalized = normalizeUrlForMatch(link);
            if (context.seenUrls.has(normalized))
                return false;
            context.seenUrls.add(normalized);
            return true;
        })
            .slice(0, getReversePageCandidateLimit());
        for (const pageUrl of pageCandidates) {
            const pageImageUrl = await fetchPageImageUrl(pageUrl);
            if (!pageImageUrl)
                continue;
            if (context?.targetImageUrl &&
                normalizeUrlForMatch(context.targetImageUrl) === normalizeUrlForMatch(pageImageUrl)) {
                return [
                    {
                        url: pageImageUrl,
                        source: "serpapi",
                        source_page_url: pageUrl,
                        license: null,
                    },
                ];
            }
            if (!context?.requireTargetMatch || isLikelyUserGeneratedImageUrl(context?.targetImageUrl)) {
                return [
                    {
                        url: pageImageUrl,
                        source: "serpapi",
                        source_page_url: pageUrl,
                        license: null,
                    },
                ];
            }
            const resolved = await resolveCandidateImageUrl(pageImageUrl, context.requireTargetMatch);
            if (!resolved)
                continue;
            const verified = await verifyImageMatch(resolved, context.targetImageUrl || "");
            if (verified) {
                return [
                    {
                        url: pageImageUrl,
                        source: "serpapi",
                        source_page_url: pageUrl,
                        license: null,
                    },
                ];
            }
        }
        const results = [
            ...imageResults.map((img) => ({
                url: img.thumbnail || img.original || img.link,
                source: "serpapi",
                source_page_url: img.link || img.redirect_link || null,
                title: img.title || img.snippet || null,
            })),
            ...inlineImages.map((img) => ({
                url: img.thumbnail || img.original || img.link,
                source: "serpapi",
                source_page_url: img.source || img.link || null,
                title: img.title || null,
            })),
        ]
            .filter((img) => !!img.url && isHttpUrl(img.url));
        return results;
    }
    catch {
        return [];
    }
}
async function searchWikimedia(query, context) {
    const params = new URLSearchParams({
        action: "query",
        format: "json",
        prop: "imageinfo",
        generator: "search",
        gsrsearch: query,
        gsrlimit: "10",
        iiprop: "url|extmetadata",
        origin: "*",
    });
    try {
        const res = await (0, node_fetch_1.default)(`https://commons.wikimedia.org/w/api.php?${params.toString()}`);
        if (!res.ok)
            return [];
        const data = await res.json();
        const pages = data?.query?.pages;
        if (!pages)
            return [];
        return Object.values(pages)
            .map((page) => {
            const info = page.imageinfo?.[0];
            if (!info?.url)
                return null;
            return {
                url: info.url,
                source: "wikimedia",
                source_page_url: page.fullurl || null,
                license: info.extmetadata?.LicenseShortName?.value || null,
                title: page.title || null,
            };
        })
            .filter((item) => !!item && isHttpUrl(item.url));
    }
    catch {
        return [];
    }
}
async function runVisionCheck(candidateUrl, targetImageUrl) {
    try {
        const client = getClient();
        const prompt = `You are verifying whether two images are the same product image. 
Answer with a JSON object {"match": true/false, "confidence": 0-1, "reason": "short explanation"}.
Only set match=true if the images appear to be the SAME exact product image or near-identical product packaging photo.
If they are clearly different, match=false.`;
        const res = await client.chat.completions.create({
            model: "gpt-5.2-mini",
            messages: [
                { role: "system", content: prompt },
                {
                    role: "user",
                    content: [
                        { type: "text", text: "Compare these images." },
                        { type: "image_url", image_url: { url: candidateUrl } },
                        { type: "image_url", image_url: { url: targetImageUrl } },
                    ],
                },
            ],
            response_format: { type: "json_object" },
            max_completion_tokens: 250,
            temperature: 0.1,
        });
        const content = res.choices?.[0]?.message?.content || "{}";
        const parsed = JSON.parse(content);
        return {
            match: !!parsed.match,
            confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0,
            reason: parsed.reason || "",
        };
    }
    catch (error) {
        console.warn("[vision] Verification failed:", error?.message || error);
        return null;
    }
}
async function verifyImageMatch(candidateUrl, targetImageUrl) {
    if (!targetImageUrl)
        return true;
    if (!candidateUrl)
        return false;
    if (!isSupportedImageUrl(candidateUrl))
        return false;
    if (!isSupportedImageUrl(targetImageUrl))
        return false;
    const attempts = getVerifyAttempts();
    const minMatches = getVerifyMinMatches(attempts);
    const minConfidence = getVerifyMinConfidence();
    const parallelism = getVerifyParallelism();
    const checks = new Array(attempts).fill(null).map(() => runVisionCheck(candidateUrl, targetImageUrl));
    let matches = 0;
    let completed = 0;
    for (let i = 0; i < checks.length; i += parallelism) {
        const chunk = checks.slice(i, i + parallelism);
        const results = await Promise.all(chunk);
        for (const result of results) {
            completed += 1;
            if (result && result.match && result.confidence >= minConfidence) {
                matches += 1;
            }
            if (matches >= minMatches) {
                return true;
            }
            if (completed - matches > attempts - minMatches) {
                return false;
            }
        }
    }
    return matches >= minMatches;
}
function normalizeUrl(url) {
    if (!url)
        return "";
    try {
        const parsed = new URL(url);
        parsed.hash = "";
        return parsed.toString();
    }
    catch {
        return url;
    }
}
function getTargetDomain(targetImageUrl) {
    if (!targetImageUrl)
        return null;
    try {
        const parsed = new URL(targetImageUrl);
        return parsed.hostname;
    }
    catch {
        return null;
    }
}
function scoreCandidate(candidate, queryTokens, targetDomain) {
    let score = 0;
    if (candidate.source === "wikimedia")
        score += 1;
    if (candidate.source_page_url) {
        const sourceText = normalizeText(candidate.source_page_url);
        for (const token of queryTokens) {
            if (sourceText.includes(token))
                score += 1;
        }
        if (targetDomain && sourceText.includes(targetDomain))
            score += 2;
    }
    if (candidate.title) {
        const titleText = normalizeText(candidate.title);
        for (const token of queryTokens) {
            if (titleText.includes(token))
                score += 1;
        }
    }
    return score;
}
async function findBestCandidate(candidates, context) {
    const maxCandidates = getMaxCandidates(context?.targetImageUrl);
    const queryTokens = context?.targetDomain
        ? [context.targetDomain.replace(/^www\./i, "")]
        : [];
    const scored = candidates
        .filter((candidate) => !!candidate.url && isProductImage(candidate))
        .filter((candidate) => isSupportedImageUrl(candidate.url))
        .map((candidate) => ({
        candidate,
        score: scoreCandidate(candidate, queryTokens, context?.targetDomain),
    }))
        .sort((a, b) => b.score - a.score)
        .slice(0, maxCandidates)
        .map((item) => item.candidate);
    for (const candidate of scored) {
        if (context?.targetImageUrl && context.requireTargetMatch) {
            if (normalizeUrlForMatch(candidate.url) === normalizeUrlForMatch(context.targetImageUrl)) {
                return candidate;
            }
            const resolved = await resolveCandidateImageUrl(candidate.url, context.requireTargetMatch);
            if (!resolved)
                continue;
            const match = await verifyImageMatch(resolved, context.targetImageUrl);
            if (!match)
                continue;
        }
        return candidate;
    }
    return null;
}
async function searchStockImages(groceryItem, context) {
    const queries = buildQueries(groceryItem, context?.targetDomain || null);
    const results = [];
    for (const query of queries) {
        const serpResults = await searchSerpAPI(query, context);
        results.push(...serpResults);
    }
    const seen = new Set();
    const uniqueResults = results.filter((item) => {
        if (!item.url)
            return false;
        const normalized = normalizeUrl(item.url);
        if (seen.has(normalized))
            return false;
        seen.add(normalized);
        return true;
    });
    return uniqueResults;
}
async function findStockImage(groceryItem, options) {
    const context = {
        targetImageUrl: options?.targetImageUrl || null,
        targetDomain: options?.targetDomain || getTargetDomain(options?.targetImageUrl || ""),
        requireTargetMatch: options?.requireTargetMatch ?? false,
        seenUrls: new Set(),
    };
    if (context.targetImageUrl) {
        context.seenUrls.add(normalizeUrlForMatch(context.targetImageUrl));
    }
    let candidates = await searchStockImages(groceryItem, context);
    if (context.targetImageUrl) {
        const reverseResults = await searchSerpAPIReverseImage(context.targetImageUrl, context);
        candidates = [...reverseResults, ...candidates];
    }
    const wikiResults = await searchWikimedia(buildQueries(groceryItem, context.targetDomain)[0] || "");
    candidates.push(...wikiResults);
    if (candidates.length === 0) {
        return null;
    }
    const best = await findBestCandidate(candidates, context);
    if (!best)
        return null;
    if (!best.url)
        return null;
    return {
        url: best.url,
        source: best.source,
        source_page_url: best.source_page_url || null,
        license: best.license || null,
    };
}
