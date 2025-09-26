const statusEl = document.getElementById('status');
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const vu = document.getElementById('vu');

let micStream = null;
let audioCtx = null;
let vuNode = null;
let vuRAF = 0;

function setStatus(msg) {
  statusEl.textContent = `Status: ${msg}`;
  console.log(`[voice-agent] ${msg}`);
}

async function resumeAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') await audioCtx.resume();
}

async function requestMic() {
  setStatus('requesting mic…');
  // *Must* be called from a user gesture (button click)
  await resumeAudioCtx();

  // Relaxed constraints (avoid Overconstrained hang); localhost is allowed for gUM.
  const constraints = {
    audio: {
      channelCount: { ideal: 1 },
      sampleRate:   { ideal: 24000 },   // prefer 24k; allow browser to choose
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false
    },
    video: false
  };

  // DEBUG: permissions check (helps spot “denied” that won’t re-prompt)
  try {
    const perm = await navigator.permissions.query({ name: 'microphone' });
    console.log('microphone permission:', perm.state);
  } catch {}

  // Actually get the stream; make sure we *await* it.
  const stream = await navigator.mediaDevices.getUserMedia(constraints);

  // Sanity: list devices (labels appear only after permission granted)
  try {
    console.log('devices:', await navigator.mediaDevices.enumerateDevices());
  } catch {}

  // Optional: set up a VU meter so we *see* input arriving
  const src = audioCtx.createMediaStreamSource(stream);
  // analyser for level
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);

  vu.style.display = 'inline-block';
  const data = new Uint8Array(analyser.frequencyBinCount);

  const tick = () => {
    analyser.getByteTimeDomainData(data);
    // simple RMS-ish
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length); // 0..~1
    vu.value = Math.min(1, rms * 3); // scale for visibility
    vuRAF = requestAnimationFrame(tick);
  };
  vuRAF = requestAnimationFrame(tick);

  // Expose nodes to caller
  vuNode = analyser;
  return stream;
}

function stopMic() {
  if (vuRAF) cancelAnimationFrame(vuRAF);
  vuRAF = 0;
  vu.value = 0;
  vu.style.display = 'none';

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (audioCtx && audioCtx.state !== 'closed') {
    // keep context open if you also play TTS; otherwise close:
    // audioCtx.close();
  }
}

async function onConnect() {
  connectBtn.disabled = true;
  try {
    micStream = await requestMic();               // <-- if this resolves, you HAVE the mic
    setStatus('mic granted');

    // Now proceed to your Realtime/WebRTC connect with `micStream`.
    // Example:
    // await connectRealtime({ micStream, audioCtx, vuNode });

    disconnectBtn.disabled = false;
  } catch (err) {
    console.error('getUserMedia failed:', err);
    let msg = 'mic error';
    if (err?.name === 'NotAllowedError') msg = 'permission denied (allow mic in site settings)';
    else if (err?.name === 'NotFoundError') msg = 'no microphone found';
    else if (err?.name === 'NotReadableError') msg = 'mic busy (close Zoom/Meet/etc.)';
    else if (err?.name === 'OverconstrainedError') msg = 'unsupported constraint (relax sampleRate/channelCount)';
    setStatus(msg);
    connectBtn.disabled = false;
  }
}

function onDisconnect() {
  disconnectBtn.disabled = true;
  stopMic();
  setStatus('disconnected');
  connectBtn.disabled = false;
}

connectBtn.addEventListener('click', onConnect);
disconnectBtn.addEventListener('click', onDisconnect);
