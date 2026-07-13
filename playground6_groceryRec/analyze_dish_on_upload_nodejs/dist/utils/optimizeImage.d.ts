/**
 * Resize and re-encode image to WebP for smaller file size.
 * Used after OpenAI image generation to reduce stored bytes.
 *
 * @param input - Raw image buffer (e.g. from OpenAI b64 decode)
 * @param maxSizePx - Max width/height in pixels (fit inside, no upscale). Hero: 768, thumbnail: 512
 * @param quality - WebP quality 0–100. Hero: 70–75, thumbnail: 65–72
 */
export declare function optimizeToWebp(input: Buffer, maxSizePx?: number, quality?: number): Promise<Buffer>;
