// ✅ correct
import { RealtimeAgent, RealtimeSession } from "@openai/agents-realtime";


export const agent = new RealtimeAgent({
  name: "Greeter",
  // High-level system prompt/instructions:
  instructions:
    "You are a concise, friendly voice assistant. Keep replies short and helpful.",
});
