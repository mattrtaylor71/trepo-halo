import json
import os
import http.client

# Retrieve API key from environment variables
INSTACART_API_KEY = os.getenv("INSTACART_API_KEY")  # Set this in AWS Lambda settings

INSTACART_HOST = "connect.dev.instacart.tools"  # Keep in sandbox until production ready

def lambda_handler(event, context):
    """
    AWS Lambda function to generate an Instacart shopping list link.
    
    Expected event JSON:
    {
        "items": ["apples", "milk", "bread"]
    }
    """
    try:
        body = json.loads(event["body"])  # Parse request body
        items = body.get("items", [])
        
        if not items:
            return {"statusCode": 400, "body": json.dumps({"error": "No items provided."})}

        # Format line items for Instacart API request
        line_items = [{"name": item, "quantity": 1, "unit": "each", "display_text": item} for item in items]

        payload = json.dumps({
            "title": "My Instacart Shopping List",
            "link_type": "shopping_list",
            "expires_in": 30,  # Expires in 30 days
            "line_items": line_items,
            "landing_page_configuration": {
                "partner_linkback_url": "https://www.instacart.com",
                "enable_pantry_items": False
            }
        })

        headers = {
            'Accept': "application/json",
            'Content-Type': "application/json",
            'Authorization': f"Bearer {INSTACART_API_KEY}"
        }

        # Make API request
        conn = http.client.HTTPSConnection(INSTACART_HOST)
        conn.request("POST", "/idp/v1/products/products_link", payload, headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()

        response_data = json.loads(data.decode("utf-8"))
        if "products_link_url" in response_data:
            return {
                "statusCode": 200,
                "body": json.dumps({"shopping_list_url": response_data["products_link_url"]})
            }
        else:
            return {
                "statusCode": res.status,
                "body": json.dumps({"error": response_data})
            }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
