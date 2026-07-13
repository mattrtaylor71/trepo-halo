import { getOpenAIClient, getOpenAIModel } from "./client";

// Mirrors analyze_on_upload's assignEmoji (the scan path) so text-added items get
// the same emoji treatment. The emoji is stored as the `emoji:<x>` product_image_url
// placeholder the app already understands.
const EMOJI_SYSTEM_PROMPT = `You assign a single emoji to grocery/kitchen items. Rules:
- Pick the MOST specific emoji available for each item.
- For branded products, pick the emoji for what the product IS (e.g. chips → 🍿, soda → 🥤, hummus → 🫕, hot sauce → 🌶️, olive oil → 🫒, yogurt → 🥛).
- For produce, use the specific fruit/vegetable emoji if one exists (🍌🍎🥑🍋🥕🧅🥦🫐🍇🍊🌽🥒🍑🍒🍓🥭🍍🍈🫑🧄).
- For meat/seafood: 🥩🍗🐟🦐🥓🍖
- For dairy: 🧀🥛🧈🥚
- For drinks: ☕🍵🧃🥤🍺🍷
- For baked goods: 🍞🥐🧁🍪🎂
- For snacks: 🍫🍬🍿🥨🍩
- For spices/seasonings: 🧂🌶️
- For condiments/sauces: 🫙🧂🍯🫒
- For prepared foods: 🍕🌮🍜🍱🥗🍝
- Fallback: 🍽️ (only if nothing else fits)
Return ONLY the JSON.`;

export interface EmojiItem {
  product_name?: string | null;
  brand?: string | null;
  category?: string | null;
}

/**
 * Assigns a single emoji to a grocery item using the identify model. Returns the
 * emoji string (e.g. "🧄"), or "🍽️" as a non-throwing fallback.
 */
export async function assignEmojiToItem(item: EmojiItem): Promise<string> {
  try {
    const openai = getOpenAIClient();
    const model = getOpenAIModel();
    const itemDesc = {
      name: item.product_name || "unknown",
      brand: item.brand || "",
      category: item.category || "",
    };

    const request: any = {
      model,
      // Reasoning models can burn a tiny budget before emitting; keep effort low and
      // leave generous headroom so the JSON actually comes out.
      max_output_tokens: 2000,
      input: [
        { role: "system", content: EMOJI_SYSTEM_PROMPT },
        { role: "user", content: JSON.stringify(itemDesc) },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "emoji_assignment",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: { emoji: { type: "string" } },
            required: ["emoji"],
          },
        },
      },
    };
    if (/^(o[1-9]|gpt-5)/.test(model)) {
      request.reasoning = { effort: "low" };
    }

    const response: any = await openai.responses.create(request);
    const text = response?.output_text;
    if (typeof text === "string" && text.trim()) {
      const parsed = JSON.parse(text.trim());
      if (parsed && typeof parsed.emoji === "string" && parsed.emoji.trim()) {
        return parsed.emoji.trim();
      }
    }
  } catch {
    // non-fatal — fall through to the generic fallback
  }
  return "🍽️";
}
