import os
import json
import pymysql
import openai

# Set API key securely
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing!")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Database connection details from environment variables (recommended for security)
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

def lambda_handler(event, context):
    try:
        # Parse input request
        body = json.loads(event["body"])
        user_id = body.get("user_id")
        additional_request = body.get("additional_request", "")

        # Connect to MySQL database using pymysql
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor  # Return results as dictionaries
        )
        
        with conn.cursor() as cursor:
            # Fetch the user's discarded items
            sql_query = "SELECT title, _createdDate FROM items WHERE _owner = %s"
            cursor.execute(sql_query, (user_id,))
            items = cursor.fetchall()

        conn.close()

        # Format the data for OpenAI prompt
        titles_with_dates = [f"{item['title']} (thrown away on {item['_createdDate']})" for item in items]

        prompt = (
            f"Here is a list of all the items that a user consumes along with the dates they were discarded: {', '.join(titles_with_dates)}. "
            "You are to analyze the user's consumption, understand their preferences, eating patterns, and style of foods they like. "
            "Given this information, suggest a grocery list (full of a week's worth of products) that the user might prefer to buy. "
            "For each product, attach 1 sentence explaining why this product is great for them. "
            "Note that the list of items is not a list of ingredients that they have on hand, rather it is a list of items that they have thrown away in the past. "
            "You should especially consider what the user has thrown out in the past week, as they likely need these items replenished. Factor this into your decision. "
            "However, you should also subtly mix in some new products to try, but make sure they tie into the user's preferences. "
            "Respond with a maximum of 20 items. These items should be actual real products found on the internet, not abstract ones."
            f"Additionally, the user has requested: '{additional_request}'. Make sure to incorporate this preference into your list."
        )

        # Request a grocery list from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
        )

        grocery_list = completion.choices[0].message.content

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"grocery_list": grocery_list})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
