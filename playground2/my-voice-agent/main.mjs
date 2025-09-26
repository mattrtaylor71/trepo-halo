// main.mjs

// ====== UI refs ======
const statusEl = document.getElementById('status');
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const remoteAudio = document.getElementById('remote');
const transcriptEl = document.getElementById('transcript');

// ====== Optional: stored Prompt ID / version / variables ======
const PROMPT_ID = null;
const PROMPT_VERSION = null;
const PROMPT_VARS = {};

// ====== State ======
let audioCtx, micStream, pc, dc;

// ====== Helpers ======
const setStatus = (m) => { statusEl.textContent = `Status: ${m}`; console.log("[status]", m); };

async function resumeAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
}

async function requestMic() {
  setStatus('requesting mic…');
  await resumeAudioCtx();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: { ideal: 1 },
      sampleRate:   { ideal: 24000 },
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false
    },
    video: false
  });
  setStatus('mic granted');
  return stream;
}

async function getEphemeralToken() {
  const r = await fetch("http://localhost:5174/session", { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  const token = j?.client_secret?.value ?? j?.value ?? j?.client_secret;
  if (!token) throw new Error("No client_secret.value in session response");
  return token;
}

// ====== Browser-side tools (must match server schemas) =========
async function tool_list_discarded_14d(args) {
  const q = new URLSearchParams();
  if (Number.isFinite(args?.limit)) q.set("limit", String(args.limit));
  const r = await fetch(`http://localhost:5174/groceries/discarded14?${q}`);
  return await r.json();
}
async function tool_list_kitchen_14d(args) {
  const q = new URLSearchParams();
  if (Number.isFinite(args?.limit)) q.set("limit", String(args.limit));
  const r = await fetch(`http://localhost:5174/groceries/kitchen14?${q}`);
  return await r.json();
}
async function tool_compute_needs(args) {
  const q = new URLSearchParams();
  if (Number.isFinite(args?.limit)) q.set("limit", String(args.limit));
  const r = await fetch(`http://localhost:5174/groceries/needs?${q}`);
  return await r.json();
}
async function tool_infer_preferences() {
  const r = await fetch(`http://localhost:5174/groceries/preferences`);
  return await r.json();
}
async function tool_suggest_from_history(args) {
  const q = new URLSearchParams();
  if (Number.isFinite(args?.limit)) q.set("limit", String(args.limit));
  const r = await fetch(`http://localhost:5174/groceries/suggest?${q}`);
  return await r.json();
}
// For “when was X added?”
async function tool_get_item_age(args) {
  const name = (args?.name || "").trim();
  if (!name) return { ok: false, error: "missing name" };
  // Pull from kitchen and compute locally to keep latency low
  const r = await fetch(`http://localhost:5174/groceries/kitchen14?limit=300`);
  const d = await r.json();
  if (!d?.ok) return { ok: false, error: "kitchen fetch failed" };
  const items = Array.isArray(d.items) ? d.items : [];
  const q = name.toLowerCase();
  const matches = items
    .map(it => ({ it, t: (it.title || "").toLowerCase() }))
    .filter(x => x.t.includes(q));
  const pick = matches.length
    ? matches.sort((a,b)=> new Date(b.it.createdDate) - new Date(a.it.createdDate))[0].it
    : items.find(it => (it.title || "").toLowerCase().startsWith(q));
  if (!pick) return { ok: false, error: "not found" };
  const days = (() => {
    const ts = new Date(pick.createdDate).getTime();
    if (!Number.isFinite(ts)) return null;
    return Math.max(0, Math.floor((Date.now() - ts) / 86400000));
  })();
  return { ok: true, title: pick.title, createdDate: pick.createdDate, days };
}

async function runTool(name, args) {
  console.log("[tool:dispatch]", name, args);
  if (name === "list_discarded_14d")     return tool_list_discarded_14d(args);
  if (name === "list_kitchen_14d")       return tool_list_kitchen_14d(args);
  if (name === "compute_needs")          return tool_compute_needs(args);
  if (name === "infer_preferences")      return tool_infer_preferences(args);
  if (name === "suggest_from_history")   return tool_suggest_from_history(args);
  if (name === "get_item_age")           return tool_get_item_age(args);
  return { error: `unknown tool ${name}` };
}

// ====== Realtime connect (WebRTC) ======
async function connectRealtime() {
  setStatus('preparing Realtime…');
  const token = await getEphemeralToken();

  const pcLocal = new RTCPeerConnection();
  pc = pcLocal;

  pc.oniceconnectionstatechange = () => {
    console.log("ICE:", pc.iceConnectionState);
    setStatus(`ice: ${pc.iceConnectionState}`);
  };
  pc.onconnectionstatechange = () => {
    console.log("PC state:", pc.connectionState);
    if (["disconnected","failed","closed"].includes(pc.connectionState)) {
      setStatus("connection lost — click Connect to start a new session");
      disconnectBtn.disabled = true;
      connectBtn.disabled = false;
    }
  };

  // Data channel (events, session.update, function calling)
  const dcLocal = pc.createDataChannel("oai-events");
  dc = dcLocal;
  dc.onopen = () => {
    console.log("datachannel open");

    // Strong, situation-aware instructions that override server defaults
    const sessionUpdate = {
      type: "session.update",
      session: {
        type: "realtime",
        model: "gpt-realtime",
        temperature: 0.3,
        output_modalities: ["audio"],
        audio: {
          input: { format: { type: "audio/pcm", rate: 24000 }, turn_detection: { type: "semantic_vad" } },
          output: { format: { type: "audio/pcm" }, voice: "marin" }
        },
        instructions: [
          "You are a quick grocery co-pilot. Assume the user is planning, driving to, or in a store—or standing in their kitchen.",
          "",
          "INTENT:",
          "- 'what do I need' / 'what should I buy' / 'shopping list' → call compute_needs first; then call suggest_from_history for 3–5 tasteful extras.",
          "- 'what's in my kitchen' → list_kitchen_14d.",
          "- 'what did I toss / discard' → list_discarded_14d.",
          "- 'recommendations' → suggest_from_history (+ infer_preferences if helpful).",
          "- 'recipes' → prefer kitchen ingredients; if user mentions 'store', allow missing ingredients. Offer 2–3 ideas max.",
          "",
          "STYLE:",
          "- 2–4 concise sentences or a tight bullet list. No filler.",
          "- Include one budget or health-oriented tip when relevant.",
          "- If asked 'when/age', call get_item_age.",
          "- Briefly paraphrase any tool 'note'.",
        ].join("\n")
      }
    };
    dc.send(JSON.stringify(sessionUpdate));
  };

  // Handle model events (function calling + streamed text)
  dc.onmessage = async (ev) => {
    let data = ev.data;
    try { data = JSON.parse(ev.data); } catch {}
    // Show text stream if present
    if (data?.type === "response.output_text.delta" && typeof data.delta === "string") {
      transcriptEl.textContent += data.delta;
    }

    // Newer pattern: function calls appear inside response.done
    if (data?.type === "response.done" && data?.response?.output?.length) {
      for (const item of data.response.output) {
        if (item?.type === "function_call") {
          const { name, call_id, arguments: argStr } = item;
          let args = {};
          try { args = typeof argStr === "string" ? JSON.parse(argStr || "{}") : (argStr || {}); } catch {}
          const result = await runTool(name, args);
          dc.send(JSON.stringify({
            type: "conversation.item.create",
            item: { type: "function_call_output", call_id, output: JSON.stringify(result) }
          }));
          dc.send(JSON.stringify({ type: "response.create" }));
        }
      }
      return;
    }

    // Back-compat: explicit tool/function call events
    const isToolCall =
      data?.type === "tool.call" ||
      data?.type === "response.tool_call" ||
      data?.type === "response.function_call" ||
      data?.type === "function.call" ||
      data?.type === "function_call";

    if (isToolCall || data?.tool_call || data?.function_call) {
      const call = data.tool_call || data.function_call || data;
      const id = call.id || data.id;
      const name = call.name;
      const rawArgs = call.arguments;
      let args = {};
      try { args = typeof rawArgs === "string" ? JSON.parse(rawArgs || "{}") : (rawArgs || {}); } catch {}
      const result = await runTool(name, args);

      // New style
      dc.send(JSON.stringify({
        type: "conversation.item.create",
        item: { type: "function_call_output", call_id: id, output: JSON.stringify(result) }
      }));
      dc.send(JSON.stringify({ type: "response.create" }));

      // Legacy reply (harmless if ignored)
      dc.send(JSON.stringify({ type: "tool.result", tool_call_id: id, output: result }));
    }
  };

  // Media: add mic track, receive remote audio
  micStream.getTracks().forEach(t => pc.addTrack(t, micStream));
  pc.ontrack = (e) => {
    remoteAudio.srcObject = e.streams[0];
    remoteAudio.play().catch(err => console.warn("remote play blocked:", err));
  };

  // Offer/answer with the ephemeral token
  const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
  await pc.setLocalDescription(offer);

  const resp = await fetch("https://api.openai.com/v1/realtime?model=gpt-realtime", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/sdp",
      "OpenAI-Beta": "realtime=v1",
    },
    body: offer.sdp
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Realtime SDP error ${resp.status}: ${txt}`);
  }

  const answerSdp = await resp.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  setStatus("connected — speak and pause; it should reply.");
}

function disconnectRealtime() {
  try { if (dc && dc.readyState === "open") dc.close(); } catch {}
  try { if (pc) pc.getSenders().forEach(s => s.track && s.track.stop()); } catch {}
  try { if (pc) pc.close(); } catch {}
  dc = null; pc = null;
}

// ====== UI wiring ======
connectBtn.addEventListener('click', async () => {
  connectBtn.disabled = true;
  transcriptEl.textContent = "";
  try {
    micStream = await requestMic();
    await connectRealtime();
    disconnectBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus(`error: ${e.message}`);
    connectBtn.disabled = false;
    disconnectRealtime();
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  }
});

disconnectBtn.addEventListener('click', () => {
  disconnectBtn.disabled = true;
  disconnectRealtime();
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  setStatus('disconnected');
  connectBtn.disabled = false;
});
