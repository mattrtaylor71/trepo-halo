// ✅ correct
import { RealtimeAgent, RealtimeSession } from "@openai/agents-realtime";
import { agent } from "./agent";

// IMPORTANT: use a client API key in development only.
// For production, proxy a server-side token instead.
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY as string;

const connectBtn = document.getElementById("connect") as HTMLButtonElement;
const disconnectBtn = document.getElementById("disconnect") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLParagraphElement;

let session: RealtimeSession | null = null;

function setStatus(text: string) {
  statusEl.textContent = `Status: ${text}`;
}

connectBtn.onclick = async () => {
  connectBtn.disabled = true;
  setStatus("requesting mic…");

  // Ask for mic permission early so WebRTC can attach the track
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });

  // Create the Realtime session
  session = new RealtimeSession(agent, {
    model: "gpt-realtime", // default; keep explicit
    // Configure voice + turn detection (defaults work fine)
    config: {
      voice: "marin", // pick any supported voice
      turnDetection: {
        type: "semantic_vad",
        eagerness: "medium",
        createResponse: true,
        interruptResponse: true,
      },
    },
  });

  // Hook up basic events for visibility
  session.on("connected", () => setStatus("connected (listening)"));
  session.on("disconnected", () => setStatus("disconnected"));
  session.on("error", (e) => setStatus(`error: ${String(e)}`));

  // Connect using WebRTC with your key (development only)
  await session.connect({ apiKey: OPENAI_API_KEY, mediaStream: mic });

  disconnectBtn.disabled = false;
};

disconnectBtn.onclick = async () => {
  disconnectBtn.disabled = true;
  if (session) {
    await session.disconnect();
    session = null;
  }
  connectBtn.disabled = false;
};
