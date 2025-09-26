import json
import pymysql
import os

# Database connection config
db_config = {
    "host": os.getenv("DB_HOST", "database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASS", "Nbmqyq17!"),
    "database": os.getenv("DB_NAME", "mysqlTutorial"),
    "cursorclass": pymysql.cursors.DictCursor
}

def lambda_handler(event, context):
    try:
        # Parse request body
        body = json.loads(event["body"])
        user_name = body.get("user_name", "")

        if not user_name:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing user_name"})}

        # Connect to database
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        # ✅ Step 1: Fetch shopping list items for user
        cursor.execute("""
            SELECT item_id, source 
            FROM shopping_list 
            WHERE user_name = %s
        """, (user_name,))
        shopping_items = cursor.fetchall()

        if not shopping_items:
            return {"statusCode": 200, "body": json.dumps({"items": [], "whole_foods": []})}

        # Separate items by source
        main_item_ids = [item["item_id"] for item in shopping_items if item["source"] == "main"]
        wf_item_ids = [item["item_id"] for item in shopping_items if item["source"] == "whole_foods"]

        # ✅ Step 2: Fetch product data from "items" table
        items = []
        if main_item_ids:
            format_strings = ', '.join(['%s'] * len(main_item_ids))
            cursor.execute(f"""
                SELECT _id, title, brand, images, simplified_category 
                FROM items 
                WHERE _id IN ({format_strings})
            """, main_item_ids)
            items = cursor.fetchall()

        # ✅ Step 3: Fetch product data from "whole_foods" table
        whole_foods = []
        if wf_item_ids:
            format_strings = ', '.join(['%s'] * len(wf_item_ids))
            cursor.execute(f"""
                SELECT _id, name AS title, brand, image_urls AS images, 'Whole Foods' AS source 
                FROM whole_foods 
                WHERE _id IN ({format_strings})
            """, wf_item_ids)
            whole_foods = cursor.fetchall()

        cursor.close()
        conn.close()

        # ✅ Step 4: Return structured response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"items": items, "whole_foods": whole_foods}),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": "Internal server error"})}
