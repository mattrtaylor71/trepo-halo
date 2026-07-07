"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.analyzeProduct = analyzeProduct;
const node_fetch_1 = __importDefault(require("node-fetch"));
const client_1 = require("../openai/client");
const identifyGrocery_1 = require("../openai/identifyGrocery");
async function assertImageUrl(imageUrl) {
    const response = await (0, node_fetch_1.default)(imageUrl, {
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
function buildTargetImageUrl(input) {
    if ("imageUrl" in input && input.imageUrl) {
        return input.imageUrl;
    }
    if ("imageBuffer" in input && input.imageBuffer) {
        return `data:image/jpeg;base64,${input.imageBuffer.toString("base64")}`;
    }
    return null;
}
function getStockImageTimeoutMs(mode) {
    const fallback = mode === "deep" ? 45000 : 10000;
    const modeSpecificName = mode === "deep" ? "STOCK_IMAGE_DEEP_TIMEOUT_MS" : "STOCK_IMAGE_TIMEOUT_MS";
    const raw = parseInt(process.env[modeSpecificName] || process.env.STOCK_IMAGE_TIMEOUT_MS || "", 10);
    return Number.isFinite(raw) ? raw : fallback;
}
function withTimeout(promise, timeoutMs, errorMessage) {
    return new Promise((resolve, reject) => {
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
function toStockImageResponse(result, mode) {
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
async function analyzeProduct(input, options = {}) {
    const startedAt = Date.now();
    const stockImageMode = options.stockImageMode || "fast";
    const targetImageUrl = buildTargetImageUrl(input);
    if ("imageUrl" in input && input.imageUrl) {
        await assertImageUrl(input.imageUrl);
    }
    const identifyStartedAt = Date.now();
    const groceryItem = await (0, identifyGrocery_1.identifyGroceryItem)(input, { userHint: options.userHint, leftovers: options.leftovers });
    const identify_ms = Date.now() - identifyStartedAt;
    // Retail search and stock image lookup removed — they added ~45s median
    // and produced 0% useful results. Emoji assignment handles product
    // images separately in app.js.
    const disabledStockImage = {
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
            model: (0, client_1.getOpenAIModel)(),
            retailer_link_source: "disabled",
        },
    };
}
//# sourceMappingURL=analyzeProduct.js.map