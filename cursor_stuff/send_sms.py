import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def send_sms_via_klaviyo(phone_number, message):
    """
    Send an SMS message using Klaviyo API by tracking a Sent SMS metric
    """
    # Get API key from environment variable
    api_key = os.getenv('KLAVIYO_API_KEY')
    
    if not api_key:
        raise ValueError("KLAVIYO_API_KEY not found in environment variables")
    
    # Klaviyo API endpoint for tracking events
    url = "https://a.klaviyo.com/api/track"
    
    # Prepare the payload
    payload = {
        "token": api_key,
        "event": "Sent SMS",
        "customer_properties": {
            "$phone_number": phone_number,
            "$consent": ["sms"]
        },
        "properties": {
            "$message": message
        }
    }
    
    # Send the request
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    if response.status_code in [200, 201, 202]:
        print("SMS trigger sent successfully!")
        return True
    else:
        print(f"Failed to trigger SMS. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

if __name__ == "__main__":
    # Example usage
    phone_number = "+18582321987"  # Your phone number
    message = "matt testing code!"  # Matching the message in your flow
    
    try:
        send_sms_via_klaviyo(phone_number, message)
    except Exception as e:
        print(f"Error: {str(e)}") 