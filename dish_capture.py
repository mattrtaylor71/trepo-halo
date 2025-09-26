# dish_capture.py — Minimal "dish → nutrition" pipeline (no gestures)
import os, time, json, base64, logging, subprocess, shutil
from collections import deque
import numpy as np
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")   # any vision-capable model

# ---------- Config ----------
SHOW_PREVIEW    = True                   # False = headless; HUD written to /tmp/hud.jpg
WRITE_HUD_FILE  = True
PREVIEW_SIZE    = (960, 540)
CAPTURE_SIZE    = (2028, 1520)          # adjust to your sensor
STABILITY_MS    = 450                   # how long motion must be low
AREA_THRESH     = 14000                 # frame-diff area to consider “something’s there”
BURST_N         = 4
JPEG_QUALITY    = 90
REFRACTORY_MS   = 800                   # pause after a capture
THRESH_DIFF     = 18                    # frame diff threshold (0–255)
HUD_WINDOW      = "Trepo Halo — Dish Detector"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("halo")

# ---------- Camera backends ----------
class CamBackend:
    def preview_frame(self): ...
    def capture_best(self): ...

class Picam2Backend(CamBackend):
    def __init__(self):
        from picamera2 import Picamera2, Preview
        self.picam = Picamera2()
        self.video_cfg = self.picam.create_video_configuration(main={"size": PREVIEW_SIZE, "format":"RGB888"})
        self.still_cfg = self.picam.create_still_configuration(main={"size": CAPTURE_SIZE, "format":"RGB888"})
        self.picam.configure(self.video_cfg)
        self.picam.start_preview(Preview.NULL)
        self.picam.start()
        time.sleep(0.6)  # warmup

    def preview_frame(self):
        try:
            fr = self.picam.capture_array("main")  # RGB
            if fr is None or fr.size == 0:
                return None
            return fr
        except Exception as e:
            log.warning(f"Picamera2 preview error: {e}")
            return None

    def capture_best(self):
        def lap(rgb): return cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        t0 = time.time()
        try:
            self.picam.switch_mode_and_capture_file(self.still_cfg, "/tmp/_warm.jpg")
            time.sleep(0.15)
            frames, scores = [], []
            for _ in range(BURST_N):
                fr = self.picam.capture_array("main")
                frames.append(fr); scores.append(lap(fr))
            bi = int(np.argmax(scores)); best = frames[bi]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(best, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok: raise RuntimeError("JPEG encode failed")
            jpeg = buf.tobytes()
            with open("/tmp/capture_best.jpg","wb") as f: f.write(jpeg)
            log.info(f"[picam2] burst {BURST_N}; sharp={scores[bi]:.1f}; {(time.time()-t0)*1000:.0f} ms; saved /tmp/capture_best.jpg")
            return jpeg
        except Exception as e:
            log.warning(f"Picamera2 capture error: {e}")
            return None

    def close(self):
        try: self.picam.stop()
        except: pass

class RpiCamStillBackend(CamBackend):
    def __init__(self):
        if not shutil.which("rpicam-still"):
            raise RuntimeError("rpicam-still not found")
        log.info("Using rpicam-still backend")

    def _grab_jpeg(self, w, h, q):
        cmd = ["rpicam-still", "-n", "--width", str(w), "--height", str(h), "--quality", str(q), "-o", "-"]
        out = subprocess.run(cmd, capture_output=True, timeout=5)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.decode("utf-8","ignore"))
        return out.stdout

    def preview_frame(self):
        try:
            jpg = self._grab_jpeg(PREVIEW_SIZE[0], PREVIEW_SIZE[1], 75)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            log.warning(f"rpicam-still preview error: {e}")
            return None

    def capture_best(self):
        t0 = time.time()
        jpg = self._grab_jpeg(CAPTURE_SIZE[0], CAPTURE_SIZE[1], JPEG_QUALITY)
        with open("/tmp/capture_best.jpg","wb") as f: f.write(jpg)
        log.info(f"[rpicam-still] capture {(time.time()-t0)*1000:.0f} ms; saved /tmp/capture_best.jpg")
        return jpg

    def close(self): pass

def pick_backend():
    try:
        be = Picam2Backend()
        probe = be.preview_frame()
        if probe is None or probe.mean() == 0:
            raise RuntimeError("Picamera2 returned empty/black frame")
        log.info("Using Picamera2 backend")
        return be
    except Exception as e:
        log.warning(f"Picamera2 unavailable ({e}); falling back to rpicam-still.")
        return RpiCamStillBackend()

backend = pick_backend()

# ---------- Detection ----------
def frame_diff_ok(prev_rgb, curr_rgb):
    g1 = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(g1, g2), (7,7), 0)
    _, mask = cv2.threshold(diff, THRESH_DIFF, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    area = int(mask.sum() / 255)
    return area >= AREA_THRESH, area

def draw_hud(rgb, status, area, fps=None):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    txt = f"{status} | area:{area}"
    if fps is not None: txt += f" | {fps:.1f} FPS"
    cv2.putText(bgr, txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return bgr

# ---------- OpenAI ----------
def call_openai_vision(jpeg_bytes):
    assert OPENAI_API_KEY, "OPENAI_API_KEY missing (.env)"
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = {
      "role": "user",
      "content": [
        {"type":"input_text","text":
         "You are a nutritionist. From the image, infer ingredients and a reasonable single-plate portion."
         " Output valid JSON only with: ingredients[], est_portion_g, est_macros {kcal, protein_g, carbs_g, fat_g},"
         " confidence (0-1), notes[]. Use ranges if unsure."
        },
        {"type":"input_image","image": {"data": b64, "media_type": "image/jpeg"}}
      ]
    }
    t0 = time.time()
    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": MODEL, "input": [prompt]},
        timeout=60
    )
    r.raise_for_status()
    text = r.json().get("output_text","").strip()
    log.info(f"OpenAI latency {(time.time()-t0)*1000:.0f} ms, {len(text)} chars")
    try:
        return json.loads(text)
    except Exception:
        s,e = text.find("{"), text.rfind("}")
        if s!=-1 and e!=-1: return json.loads(text[s:e+1])
        raise ValueError("Model did not return JSON")

# ---------- Main ----------
def main():
    log.info("Dish detector: show item and hold steady ~0.5s. Keys: [c]=force capture, [q]=quit")
    prev = backend.preview_frame()
    while prev is None:
        time.sleep(0.05)
        prev = backend.preview_frame()

    stable_since = None
    last_capture_ms = 0
    fpsq = deque(maxlen=30)

    while True:
        t0 = time.time()
        frame = backend.preview_frame()
        if frame is None:
            log.warning("No frame, retrying…")
            time.sleep(0.05); continue

        motion_ok, area = frame_diff_ok(prev, frame)
        prev = frame

        status = "idle"
        now_ms = time.time()*1000
        if motion_ok:
            stable_since = stable_since or now_ms
            status = f"stable {int(now_ms - stable_since)}ms"
            if (now_ms - stable_since) > STABILITY_MS and (now_ms - last_capture_ms) > REFRACTORY_MS:
                log.info("→ Auto capture")
                jpeg = backend.capture_best()
                try:
                    result = call_openai_vision(jpeg)
                    log.info("Nutrition JSON:\n" + json.dumps(result, indent=2))
                    status = f"captured; kcal≈{result.get('est_macros',{}).get('kcal','?')}"
                except Exception as e:
                    log.exception(f"OpenAI error: {e}")
                last_capture_ms = now_ms
                stable_since = None
        else:
            stable_since = None

        # hotkeys (desktop only)
        if SHOW_PREVIEW and os.environ.get("DISPLAY"):
            hud = draw_hud(frame, status, area, fps=None)
            cv2.imshow(HUD_WINDOW, hud)
            if WRITE_HUD_FILE:
                cv2.imwrite("/tmp/hud.jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                log.info("Quit (q)"); break
            if k == ord('c'):
                log.info("→ Manual capture (c)")
                jpeg = backend.capture_best()
                try:
                    result = call_openai_vision(jpeg)
                    log.info("Nutrition JSON:\n" + json.dumps(result, indent=2))
                except Exception as e:
                    log.exception(f"OpenAI error: {e}")
                last_capture_ms = now_ms
                stable_since = None
        else:
            # headless HUD file
            if WRITE_HUD_FILE:
                hud = draw_hud(frame, status, area, fps=None)
                cv2.imwrite("/tmp/hud.jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        # simple pacing
        dt = time.time()-t0
        if dt < 0.01: time.sleep(0.01-dt)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        try: cv2.destroyAllWindows()
        except: pass
        try: backend.close()
        except: pass
