import boto3
import time
import urllib
from openai import OpenAI

# --- CONFIGURATION ---
bucket_name = "demopics-1711"
captures_prefix = "captures/"
results_prefix = "captures_results/"

openai_client = OpenAI(api_key="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")

s3 = boto3.client("s3")

# --- HELPERS ---
def list_captures():
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=captures_prefix)
    files = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".jpg"):
                files.append(key)
    return files

def result_exists(capture_key):
    result_key = capture_key.replace(captures_prefix, results_prefix) + ".txt"
    try:
        s3.head_object(Bucket=bucket_name, Key=result_key)
        return True
    except s3.exceptions.ClientError:
        return False

def run_gpt_vision(image_key):
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{urllib.parse.quote(image_key)}"
    print(f"Running GPT Vision on {s3_url}...")

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":
                        "You are acting as a health advisor for a smart camera device. "
                        "The user will hold an item such as food, beverage, or consumer good product in front of the camera. "
                        "Your task is to identify the item and provide a brief health recommendation."
                        "\n\nRespond ONLY in the following format:\n"
                        "Title: (2 to 3 word description of the item)\n"
                        "Health: (a very short health summary or recommendation, 1 line). "
                        "If the item contains potentially harmful ingredients (e.g. high sugar, high sodium, processed oils, additives), highlight this in the response."
                        "\n\nIf the image is unclear or you can't confidently identify the item, reply exactly with:\n"
                        "Title: Unknown\nHealth: Unable to determine."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": s3_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

    )


    description = response.choices[0].message.content.strip()
    print(f"AI description: {description}")
    return description

def save_result(capture_key, description):
    result_key = capture_key.replace(captures_prefix, results_prefix) + ".txt"
    s3.put_object(Body=description, Bucket=bucket_name, Key=result_key)
    print(f"Saved result to {result_key}")

# --- MAIN LOOP ---
print("Polling S3 for new images...")

while True:
    captures = list_captures()

    for capture_key in captures:
        if not result_exists(capture_key):
            print(f"New image found: {capture_key}")
            try:
                description = run_gpt_vision(capture_key)
                save_result(capture_key, description)
            except Exception as e:
                print(f"Error processing {capture_key}: {e}")

    time.sleep(.2)  # Poll every 10 seconds
