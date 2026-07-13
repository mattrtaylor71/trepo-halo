const fetch = require("node-fetch");
const { Jimp } = require("jimp");

const GOOGLE_VISION_TIMEOUT_MS = Number(process.env.GOOGLE_VISION_TIMEOUT_MS || 15000);
const GOOGLE_VISION_MAX_DIMENSION = Number(process.env.GOOGLE_VISION_MAX_DIMENSION || 2048);
const GOOGLE_VISION_MAX_LINES = Number(process.env.GOOGLE_VISION_MAX_LINES || 80);
const GOOGLE_VISION_MAX_OBJECTS = Number(process.env.GOOGLE_VISION_MAX_OBJECTS || 24);
const GOOGLE_VISION_MAX_PROMPT_CHARS = Number(process.env.GOOGLE_VISION_MAX_PROMPT_CHARS || 4000);

function normalizeLine(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function uniqueLines(values) {
  const seen = new Set();
  const deduped = [];

  for (const value of values) {
    const normalized = normalizeLine(value);
    if (!normalized) {
      continue;
    }

    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    deduped.push(normalized);
  }

  return deduped;
}

async function resizeForVision(buffer) {
  try {
    const image = await Jimp.read(buffer);
    const width = image.bitmap?.width || 0;
    const height = image.bitmap?.height || 0;

    if (!width || !height) {
      return buffer;
    }

    const maxDimension = Math.max(width, height);
    if (maxDimension <= GOOGLE_VISION_MAX_DIMENSION) {
      return buffer;
    }

    const scale = GOOGLE_VISION_MAX_DIMENSION / maxDimension;
    image.scale(scale);
    return await image.getBuffer("image/jpeg");
  } catch (_error) {
    return buffer;
  }
}

function extractOcrLines(annotation) {
  const fullText = annotation?.fullTextAnnotation?.text || "";
  const textAnnotations = Array.isArray(annotation?.textAnnotations) ? annotation.textAnnotations.slice(1) : [];
  const fullTextLines = fullText.split("\n");
  const annotationLines = textAnnotations.map((entry) => entry?.description || "");
  return uniqueLines([...fullTextLines, ...annotationLines]).slice(0, GOOGLE_VISION_MAX_LINES);
}

function truncatePromptText(text) {
  if (text.length <= GOOGLE_VISION_MAX_PROMPT_CHARS) {
    return text;
  }

  return `${text.slice(0, GOOGLE_VISION_MAX_PROMPT_CHARS)}\n...[truncated OCR text]`;
}

async function extractGoogleVisionContext(imageAsset) {
  if (!process.env.GOOGLE_VISION_API_KEY) {
    return null;
  }

  const preparedBuffer = await resizeForVision(imageAsset.buffer);
  const body = {
    requests: [
      {
        image: {
          content: preparedBuffer.toString("base64"),
        },
        features: [
          { type: "TEXT_DETECTION", maxResults: 1 },
          { type: "OBJECT_LOCALIZATION", maxResults: GOOGLE_VISION_MAX_OBJECTS },
        ],
      },
    ],
  };

  const response = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_VISION_API_KEY}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    timeout: GOOGLE_VISION_TIMEOUT_MS,
  });

  if (!response.ok) {
    throw new Error(`Google Vision OCR failed: HTTP ${response.status}`);
  }

  const payload = await response.json();
  const annotation = payload?.responses?.[0] || {};
  const lines = extractOcrLines(annotation);
  const objectNames = uniqueLines(
    (annotation?.localizedObjectAnnotations || []).map((item) => item?.name || "")
  ).slice(0, GOOGLE_VISION_MAX_OBJECTS);
  const fullText = uniqueLines(lines).join("\n");
  const promptText = truncatePromptText(fullText);

  return {
    fullText,
    promptText,
    lines,
    objectNames,
    lineCount: lines.length,
    objectCount: objectNames.length,
    usedGoogleVision: true,
  };
}

module.exports = {
  extractGoogleVisionContext,
};
