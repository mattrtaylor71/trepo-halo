#!/usr/bin/env python3

import sys, os, time
from PIL import Image, ImageDraw, ImageFont
import boto3
from botocore.exceptions import NoCredentialsError
import textwrap

# ---------------------------------------------------------------------------
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")
)
from waveshare_epd import epd2in13_V4
# ---------------------------------------------------------------------------

# --- E-PAPER SETUP ---
epd = epd2in13_V4.EPD()
epd.init()
epd.Clear(0xFF)
epd.init()

def update_display(title_text, health_text):
    W, H = epd.height, epd.width
    canvas = Image.new("1", (W, H), 255)
    draw = ImageDraw.Draw(canvas)

    # --- Load two font sizes ---
    try:
        font_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pic", "Font.ttc"
        )
        font_title = ImageFont.truetype(font_path, 28)  # Big title font
        font_health = ImageFont.truetype(font_path, 20)  # Smaller health font
    except IOError:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_health = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    # --- Draw Title ---
    w_title, h_title = draw.textsize(title_text, font=font_title)
    y = 5  # Top padding
    draw.text(((W - w_title) // 2, y), title_text, font=font_title, fill=0)
    y += h_title + 5  # Space between title and health text

    # --- Wrap and Draw Health Text ---
    wrap_width = 25  # You can tweak this value
    health_lines = textwrap.wrap(health_text, width=wrap_width)

    for line in health_lines[:3]:  # Max 2 lines
        w_line, h_line = draw.textsize(line, font=font_health)
        draw.text(((W - w_line) // 2, y), line, font=font_health, fill=0)
        y += h_line + 2  # Space between lines

    epd.displayPartial(epd.getbuffer(canvas))


# --- S3 SETUP ---
bucket_name = 'demopics-1711'

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAYH4AGUCDJSJD6NEK',
    aws_secret_access_key='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'
)

def parse_result(result_text):
    title = "Unknown Item"
    health = "No summary."

    for line in result_text.splitlines():
        if line.lower().startswith("title:"):
            title = line.partition(":")[2].strip()
        elif line.lower().startswith("health:"):
            health = line.partition(":")[2].strip()

    # Return two parts: title and health (possibly wrapped into 2 lines)
    health_lines = textwrap.wrap(health, width=30)
    return title, health_lines[:3]  # Max 2 lines


def upload_to_s3(file_name, object_name):
    try:
        s3.upload_file(file_name, bucket_name, object_name)
        print(f"Upload successful: {object_name}")
        return True
    except FileNotFoundError:
        print("File not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def wait_for_result(capture_key):
    result_key = capture_key.replace("captures/", "captures_results/") + ".txt"
    print(f"Waiting for result: {result_key}")

    for attempt in range(90):  # Check for up to ~90 x 5 seconds = 7.5 minutes
        try:
            response = s3.get_object(Bucket=bucket_name, Key=result_key)
            description = response['Body'].read().decode('utf-8')
            print("Result found:", description)
            return description
        except s3.exceptions.NoSuchKey:
            print(f"Result not ready yet... (attempt {attempt+1})")
            time.sleep(5)

    print("No result found after waiting.")
    return "No result available."

# --- VCNL4040 SETUP ---
import board
import busio
import adafruit_vcnl4040

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_vcnl4040.VCNL4040(i2c)
sensor.led_current_mA = 120

THRESHOLD = 3

# --- CAMERA SETUP ---
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(
    main={"size": (1024, 768)}
))
picam2.start()
time.sleep(2)

# --- Improve sharpness: fast shutter + ISO boost ---
picam2.set_controls({
    "ExposureTime": 10000,   # 1/100 sec
    "AnalogueGain": 12        # ISO ~800 (depends on camera model)
})
time.sleep(0.3)  # Let exposure settle


update_display("Hi! I'm Trepo :)", "Wave to begin")
print("Wave detection armed. Ctrl+C to stop.")

# --- MAIN LOOP ---
while True:
    try:
        proximity = sensor.proximity
        #print("Proximity:", proximity)

        if proximity >= THRESHOLD:
            print("Wave detected! Taking image...")
            update_display("Let's take a look", "Hold still...")
            time.sleep(.3)
            picam2.capture_file("capture.jpg")
            update_display("Analyzing...", "Please wait...")

            capture_key = f"captures/capture_{int(time.time())}.jpg"
            upload_to_s3("capture.jpg", capture_key)

            # --- Poll S3 for result ---
            #update_display(["Analyzing...", "Please wait..."])
            result_text = wait_for_result(capture_key)

            # Display result
            title, health_lines = parse_result(result_text)
            health_text = " ".join(health_lines)
            update_display(title, health_text)


        time.sleep(0.1)

    except KeyboardInterrupt:
        update_display("Stopped by user.")
        break
