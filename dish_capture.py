# dish_capture.py — resilient preview with Picamera2 -> fallback to rpicam-still
import os, time, base64, json, logging, subprocess, shutil
from collections import deque
import numpy as np
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")

# ---- CONFIG ----
SHOW_PREVIEW      = True                     # False if fully headless
WRITE_HUD_FILE    = True                     # saves /tmp/hud.jpg for headless peek
PREVIEW_SIZE      = (960, 540)               # preview size
CAPTURE_SIZE      = (2028, 1520)             # full-res capture
STABILITY_MS      = 450
AREA_THRESH       = 15000
DRIFT_PX          = 15
BURST_N           = 4
JPEG_QUALITY      = 90
REFRACTORY_MS     = 700

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("halo")

# ---- CAMERA BACKENDS ----
class CamBackend:
    def preview_frame(self): ...
    def capture_best(self): ...

class Picam2Backend(CamBackend):
    def __init__(self):
        from picamera2 import Picamera2, Preview
        self.Preview = Preview
        self.picam = Picamera2()
        # Force RGB888 to avoid format surprises
        self.video_cfg  = self.picam.create_video_configuration(main={"size": PREVIEW_SIZE, "format":"RGB888"})
        self.still_cfg  = self.picam.create_still_configuration(main={"size": CAPTURE_SIZE, "format":"RGB888"})
        self.picam.configure(self.video_cfg)
        # No GL window; we render ourselves
        self.picam.start_preview(self.Preview.NULL)
        self.picam.start()
        time.sleep(0.6)  # AE/AWB warmup

    def preview_frame(self):
        # returns RGB frame or None
        try:
            fr = self.picam.capture_array("main")
            if fr is None or fr.size == 0: return None
            return fr  # RGB
        except Exception as e:
            log.warning(f"Picamera2 preview error: {e}")
            return None

    def capture_best(self):
        # switch to still, burst, pick sharpest; returns (jpeg_bytes, sharpness)
        def lap_var(rgb): return cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        t0 = time.time()
        try:
            self.picam.switch_mode_and_capture_file(self.still_cfg, "/tmp/_warm.jpg")
            time.sleep(0.15)
            frames, scores = [], []
            for _ in range(BURST_N):
                fr = self.picam.capture_array("main")
                sc = lap_var(fr); frames.append(fr); scores.append(sc)
            bi = int(np.argmax(scores)); best = frames[bi]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(best, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok: raise RuntimeError("JPEG encode failed")
            jpeg = buf.tobytes()
            with open("/tmp/capture_best.jpg","wb") as f: f.write(jpeg)
            log.info(f"[picam2] burst {BURST_N}, sharpest={scores[bi]:.1f}, {(time.time()-t0)*1000:.0f} ms")
            return jpeg, scores[bi]
        except Exception as e:
            log.warning(f"Picamera2 capture error, falling back later: {e}")
            return None, None

    def close(self):
        try: self.picam.stop()
        except: pass

class RpiCamStillBackend(CamBackend):
    def __init__(self):
        if not shutil.which("rpicam-still"):
            raise RuntimeError("rpicam-still not found")
        log.info("Using rpicam-still backend")

    def _grab_jpeg(self, w, h, quality, preview=False):
        # preview: lower quality/size; capture: full size
        cmd = [
            "rpicam-still",
            "-n",                    # no on-screen preview
            "--width", str(w),
            "--height", str(h),
            "--quality", str(quality),
            "-o", "-"               # write JPEG to stdout
        ]
        # For preview, speed up exposure/AF a bit
        if preview:
            cmd += ["--shutter", "0", "--gain", "0"]  # let AE pick; minimal overhead
        out = subprocess.run(cmd, capture_output=True, timeout=5)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.decode("utf-8","ignore"))
        return out.stdout  # JPEG bytes

    def preview_frame(self):
        try:
            jpg = self._grab_jpeg(PREVIEW_SIZE[0], PREVIEW_SIZE[1], 70, preview=True)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as e:
            log.warning(f"rpicam-still preview error: {e}")
            return None

    def capture_best(self):
        # For simplicity, one high-quality shot (still fast); could do a mini-burst if needed
        t0 = time.time()
        jpg = self._grab_jpeg(CAPTURE_SIZE[0], CAPTURE_SIZE[1], JPEG_QUALITY, preview=False)
        with open("/tmp/capture_best.jpg","wb") as f: f.write(jpg)
        log.info(f"[rpicam-still] capture done in {(time.time()-t0)*1000:.0f} ms")
        return jpg, None  # no sharpness metric from burst

    def close(self): pass

# Try Picamera2, fall back to rpicam-still
def pick_backend():
    try:
        be = Picam2Backend()
        # sanity: get one frame
        fr = be.preview_frame()
        if fr is None or fr.mean() == 0:
            raise RuntimeError("Picamera2 returned empty/black frame")
        log.info("Using Picamera2 backend")
        return be
    except Exception as e:
        log.warning(f"Picamera2 unavailable ({e}); falling back to rpicam-still.")
        return RpiCamStillBackend()

backend = pick_backend()

# ---- detection / HUD ----
def lap_var_rgb(rgb):
    return cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def detect_stable(prev_rgb, curr_rgb, area_thresh=AREA_THRESH, drift_px=DRIFT_PX):
    g1 = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(g1, g2), (7,7), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False, None, 0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < area_thresh: return False, None, area
    x,y,w,h = cv2.boundingRect(c)
    return True, (x,y,w,h), area

def draw_hud(rgb, box, status, sharp=None, fps=None):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if box is not None:
        x,y,w,h = box
        cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
    text = status
    if sharp is not None: text += f" | sharp:{sharp:.0f}"
    if fps is not None:   text += f" | {fps:.1f} FPS"
    cv2.putText(bgr, text, (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return bgr

# ---- OpenAI call ----
def call_openai_vision(jpeg_bytes):
    assert OPENAI_API_KEY, "OPENAI_API_KEY missing (.env)"
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = {"role":"user","content":[
        {"type":"input_text","text":"You are a nutritionist. From the image, infer ingredients and a reasonable single-plate portion. Output valid JSON only with: ingredients[], est_portion_g, est_macros {kcal, protein_g, carbs_g, fat_g}, confidence (0-1), notes[]. Use ranges if unsure."},
        {"type":"input_image","image":{"data":b64,"media_type":"image/jpeg"}}
    ]}
    t0 = time.time()
    r = requests.post("https://api.openai.com/v1/responses",
        headers={"Authorization":f"Bearer {OPENAI_API_KEY}"},
        json={"model":MODEL,"input":[prompt]}, timeout=60)
    r.raise_for_status()
    text = r.json().get("output_text","").strip()
    log.info(f"OpenAI latency {(time.time()-t0)*1000:.0f} ms, {len(text)} chars")
    try: return json.loads(text)
    except Exception:
        s,e = text.find("{"), text.rfind("}")
        if s!=-1 and e!=-1: return json.loads(text[s:e+1])
        raise ValueError("Model did not return JSON")

# ---- main loop ----
def main_loop():
    log.info("Dish detector running. Hold a plate steady ~0.5s.")
    prev = backend.preview_frame()
    while prev is None:
        log.info("Waiting for first frame…")
        time.sleep(0.1)
        prev = backend.preview_frame()

    stable_since = None
    last_trigger_ms = 0
    fps_clock = deque(maxlen=30)

    while True:
        t0 = time.time()
        frame = backend.preview_frame()
        if frame is None:
            log.warning("No frame from backend; retrying…")
            time.sleep(0.05)
            continue

        ok, box, area = detect_stable(prev, frame); prev = frame
        now_ms = time.time()*1000; status = "idle"

        if ok:
            if stable_since is None:
                stable_since = now_ms; log.info(f"Motion; area={int(area)}; box={box}")
            elapsed = now_ms - stable_since; status = f"stable {int(elapsed)}ms"
            if elapsed > STABILITY_MS and (now_ms - last_trigger_ms) > REFRACTORY_MS:
                log.info("→ Trigger: capturing")
                try:
                    jpeg, sharp = backend.capture_best()
                    result = call_openai_vision(jpeg)
                    log.info("Nutrition JSON:\n" + json.dumps(result, indent=2))
                    status = f"captured; kcal≈{result.get('est_macros',{}).get('kcal','?')}"
                except Exception as e:
                    log.exception(f"Capture/analysis error: {e}")
                last_trigger_ms = now_ms; stable_since = None
        else:
            if stable_since is not None: log.info("Stability lost; reset")
            stable_since = None

        # FPS & HUD
        fps_clock.append(time.time())
        fps = ((len(fps_clock)-1)/(fps_clock[-1]-fps_clock[0])) if len(fps_clock)>1 else None
        hud = draw_hud(frame, box if ok else None, status, sharp=None, fps=fps)

        if WRITE_HUD_FILE:
            cv2.imwrite("/tmp/hud.jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        if SHOW_PREVIEW and os.environ.get("DISPLAY"):
            cv2.imshow("Trepo Halo — Dish Detector", hud)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("Quit (q)"); break

        # tiny sleep to avoid 100% CPU
        dt = time.time()-t0
        if dt < 0.01: time.sleep(0.01-dt)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except: pass
        try:
            backend.close()
        except: pass
