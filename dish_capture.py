# dish_capture.py  — preview + richer logging
import os, time, base64, json, io, sys
import logging
from collections import deque
from dotenv import load_dotenv

load_dotenv()

from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import requests

# ------------ config ------------
SHOW_PREVIEW     = True          # Set False if headless
PREVIEW_SIZE     = (960, 540)    # fast preview size
CAPTURE_SIZE     = (2028, 1520)  # adjust to your sensor
STABILITY_MS     = 450           # how long bbox must be steady
AREA_THRESH      = 15000         # min moving area (tune for your setup)
DRIFT_PX         = 15            # bbox centroid drift tolerance
BURST_N          = 4             # frames in burst; pick sharpest
JPEG_QUALITY     = 90
REFRACTORY_MS    = 700
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
MODEL            = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")

# ------------ logging ------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("halo")

# ------------ camera setup ------------
picam = Picamera2()
video_cfg   = picam.create_video_configuration(main={"size": PREVIEW_SIZE})
still_cfg   = picam.create_still_configuration(main={"size": CAPTURE_SIZE})
picam.configure(video_cfg)
if SHOW_PREVIEW:
    # If Qt preview isn’t available, we’ll use OpenCV’s window instead (see loop)
    try:
        picam.start_preview(Preview.NULL)  # keep ISP warm; we draw our own with OpenCV
    except Exception:
        pass
picam.start()
time.sleep(0.1)

# ------------ utils ------------
def lap_var(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def detect_stable(prev, curr, area_thresh=AREA_THRESH, drift_px=DRIFT_PX):
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (7,7), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False, None, 0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < area_thresh: return False, None, area
    x,y,w,h = cv2.boundingRect(c)
    return True, (x,y,w,h), area

def draw_hud(frame, box, status, sharp=None, fps=None):
    overlay = frame.copy()
    if box is not None:
        x,y,w,h = box
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
    text = f"{status}"
    if sharp is not None:
        text += f" | sharp:{sharp:.0f}"
    if fps is not None:
        text += f" | {fps:.1f} FPS"
    cv2.putText(overlay, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return overlay

def capture_best():
    # switch to still for a moment to lock AE/AWB/AF then burst
    t0 = time.time()
    picam.switch_mode_and_capture_file(still_cfg, "/tmp/_warm.jpg")
    time.sleep(0.15)
    frames, scores = [], []
    for _ in range(BURST_N):
        frame = picam.capture_array()
        sc = lap_var(frame)
        frames.append(frame)
        scores.append(sc)
    best_idx = int(np.argmax(scores))
    best = frames[best_idx]
    ok, buf = cv2.imencode(".jpg", best, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    jpeg = buf.tobytes() if ok else None
    if jpeg:
        with open("/tmp/capture_best.jpg", "wb") as f:
            f.write(jpeg)
    dt = (time.time() - t0) * 1000
    log.info(f"Capture burst {BURST_N} frames; sharpest={scores[best_idx]:.1f}; time={dt:.0f} ms; saved /tmp/capture_best.jpg")
    return jpeg, scores[best_idx]

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
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={ "model": MODEL, "input": [prompt] },
        timeout=60
    )
    resp.raise_for_status()
    text = resp.json().get("output_text","").strip()
    dt = (time.time() - t0) * 1000
    log.info(f"OpenAI responded in {dt:.0f} ms; {len(text)} chars")
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise ValueError("Model did not return JSON")

def main_loop():
    log.info("Dish detector running. Hold a plate steady ~0.5s.")
    prev = picam.capture_array()
    stable_since = None
    last_trigger_ms = 0
    fps_clock = deque(maxlen=30)

    while True:
        t_loop = time.time()
        frame = picam.capture_array()
        ok, box, area = detect_stable(prev, frame)
        prev = frame

        status = "idle"
        now_ms = time.time() * 1000
        if ok:
            if stable_since is None:
                stable_since = now_ms
                log.info(f"Motion detected; area={int(area)}; box={box}")
            elapsed = now_ms - stable_since
            status = f"stable {int(elapsed)}ms"
            if elapsed > STABILITY_MS and (now_ms - last_trigger_ms) > REFRACTORY_MS:
                log.info("→ Trigger: capturing still")
                try:
                    jpeg, sharp = capture_best()
                    result = call_openai_vision(jpeg)
                    log.info("Nutrition JSON:\n" + json.dumps(result, indent=2))
                    status = f"captured; kcal≈{result.get('est_macros',{}).get('kcal','?')}"
                except Exception as e:
                    log.exception(f"Capture/analysis error: {e}")
                last_trigger_ms = now_ms
                stable_since = None
        else:
            if stable_since is not None:
                log.info("Stability lost; resetting")
            stable_since = None

        # FPS calc
        fps_clock.append(time.time())
        fps = None
        if len(fps_clock) >= 2:
            dt = fps_clock[-1] - fps_clock[0]
            fps = (len(fps_clock)-1) / dt if dt > 0 else None

        # Show preview (OpenCV window with HUD)
        if SHOW_PREVIEW:
            hud = draw_hud(frame, box if ok else None, status, sharp=None, fps=fps)
            cv2.imshow("Trepo Halo — Dish Detector", hud)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("Quit requested (q).")
                break

        # gentle pacing
        # (We’re in video config already; no need to sleep much)
        # But avoid 100% CPU spin:
        time.sleep(0.01)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            picam.stop()
        except Exception:
            pass
