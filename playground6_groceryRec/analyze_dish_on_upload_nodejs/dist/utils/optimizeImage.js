"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.optimizeToWebp = optimizeToWebp;
// Lazy-load sharp so Lambda can start even when sharp native bindings are missing (wrong platform)
function getSharp() {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require("sharp");
}
/**
 * Resize and re-encode image to WebP for smaller file size.
 * Used after OpenAI image generation to reduce stored bytes.
 *
 * @param input - Raw image buffer (e.g. from OpenAI b64 decode)
 * @param maxSizePx - Max width/height in pixels (fit inside, no upscale). Hero: 768, thumbnail: 512
 * @param quality - WebP quality 0–100. Hero: 70–75, thumbnail: 65–72
 */
async function optimizeToWebp(input, maxSizePx = 768, quality = 72) {
    const sharp = getSharp();
    return sharp(input)
        .resize(maxSizePx, maxSizePx, { fit: "inside", withoutEnlargement: true })
        .webp({ quality })
        .toBuffer();
}
//# sourceMappingURL=optimizeImage.js.map