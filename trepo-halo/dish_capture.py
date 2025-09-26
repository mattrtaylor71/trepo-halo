# dish_capture.py
import os, time, base64, json, io
from dotenv import load_dotenv
load_dotenv()

from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")  # any vision-capable model

# --- camera setup ---
picam = Picamera2()
preview_cfg = picam.create_preview_configuration(main={"size": (640, 480)})
capture_cfg = picam.create_still_configuration(main={"size": (2028, 1520)})  # adjust to your sensor
picam.configure(preview_cfg)
picam.start()

def lap_var(img):  # sharpness
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def detect_stable(prev, curr, area_thresh=15000, drift_px=15):
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (7,7), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < area_thresh: return False, None
    x,y,w,h = cv2.boundingRect(c)
    return True, (x,y,w,h)

def capture_best():
    # lock AE/AWB/AF by switching to still config and sleeping a hair
    picam.switch_mode_and_capture_file(capture_cfg, "/tmp/_tmp.jpg")  # pre-warm still
    time.sleep(0.15)
    frames, scores = [], []
    for _ in range(4):
        frame = picam.capture_array()  # full-res still
        frames.append(frame)
        scores.append(lap_var(frame))
    best = frames[int(np.argmax(scores))]
    ok, buf = cv2.imencode(".jpg", best, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None

def call_openai_vision(jpeg_bytes):
    # Send via Responses API with an image input
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
    # Minimal Responses API call
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
          "model": MODEL,
          "input": [prompt],
          # You can also use structured outputs here; keeping generic for brevity.
        },
        timeout=60
    )
    resp.raise_for_status()
    text = resp.json()["output_text"]  # Responses API returns text/output fields
    # Be defensive: strip non-JSON if model speaks; but we asked for "JSON only"
    try:
        return json.loads(text)
    except Exception:
        # last resort: find JSON block
        start = text.find("{")
        end   = text.rfind("}")
        return json.loads(text[start:end+1])

def main_loop():
    print("Dish detector running. Hold a plate steady ~0.5s in front of camera.")
    ok, prev = True, picam.capture_array()
    stable_since = None
    while True:
        frame = picam.capture_array()
        ok, box = detect_stable(prev, frame)
        prev = frame
        now = time.time()
        if ok:
            stable_since = stable_since or now
            if now - stable_since > 0.45:  # stability window
                print("→ Capturing…")
                jpeg = capture_best()
                result = call_openai_vision(jpeg)
                print("Nutrition:", json.dumps(result, indent=2))
                # TODO: speak/display & save locally
                stable_since = None
                time.sleep(0.7)  # brief refractory
        else:
            stable_since = None
        time.sleep(0.06)

if __name__ == "__main__":
    try:
        main_loop()
    finally:
        picam.stop()
