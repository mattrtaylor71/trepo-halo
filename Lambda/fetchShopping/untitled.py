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
        print("🔹 Received event:", json.dumps(event))

        # Parse request body
        body = json.loads(event.get("body", "{}"))
        user_name = body.get("user_name", "").strip()

        print(f"🔹 Extracted user_name: {user_name}")

        if not user_name:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing user_name"})}

        # Connect to database
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        # ✅ Step 1: Fetch `web_id` from `users` table
        cursor.execute("SELECT web_id FROM users WHERE auth0_sub = %s", (user_name,))
        user_result = cursor.fetchone()
        print(f"🔹 User result: {user_result}")

        if not user_result:
            return {"statusCode": 404, "body": json.dumps({"error": "User not found"})}

        web_id = user_result["web_id"]
        shopping_table = f"{web_id}-shopping"
        items_table = f"{web_id}-items"

        print(f"🔹 Fetching from shopping table: {shopping_table}")

        # ✅ Step 2: Fetch shopping list items (_id, item_id, source, action)
        cursor.execute(f"""
            SELECT _id, item AS item_id, source, action
            FROM `{shopping_table}`
        """)
        shopping_items = cursor.fetchall()


        print(f"🔹 Shopping items fetched: {shopping_items}")

        if not shopping_items:
            return {"statusCode": 200, "body": json.dumps({"items": [], "whole_foods": []})}

        # Separate items by source
        # ✅ Separate items by source
        manual_items = [item for item in shopping_items if item["source"] == "manual"]
        main_item_ids = [item["item_id"] for item in shopping_items if item["source"] == "main"]
        wf_item_ids = [item["item_id"] for item in shopping_items if item["source"] == "whole_foods"]

        print(f"🔹 Manual items: {manual_items}")  # ✅ Debug manual items


        print(f"🔹 Main item IDs (from {shopping_table}): {main_item_ids}")
        print(f"🔹 Whole Foods item IDs: {wf_item_ids}")

        # ✅ Step 3: Fetch real item `_id` from `<web_id>-items`
        real_item_ids = []
        # Fetch item IDs from the intermediate `-items` table
        if main_item_ids:
            format_strings = ', '.join(['%s'] * len(main_item_ids))
            query = f"""
                SELECT _id, item
                FROM `{items_table}`
                WHERE _id IN ({format_strings})
            """
            cursor.execute(query, tuple(main_item_ids))
            mapped_items = cursor.fetchall()

            # Extract the correct `_id` from the intermediate table
            real_item_ids = [item["item"] for item in mapped_items]


        # ✅ Step 4: Fetch product data from `items` table using real `_id`
        items = []
        if real_item_ids:
            format_strings = ', '.join(['%s'] * len(real_item_ids))
            query = f"""
                SELECT _id, title, brand, images, simplified_category, simplified_title 
                FROM items 
                WHERE _id IN ({format_strings})
            """
            print(f"🔹 Executing `items` query: {query} with values {real_item_ids}")
            cursor.execute(query, tuple(real_item_ids))
            items = cursor.fetchall()

            if not items:
                print("⚠️ No matching items found in `items` table!")
            else:
                print(f"✅ Items fetched: {items}")

        # ✅ Step 5: Fetch product data from `whole_foods` table
        whole_foods = []
        if wf_item_ids:
            format_strings = ', '.join(['%s'] * len(wf_item_ids))
            query = f"""
                SELECT _id, name AS title, brand, image_urls AS images, 'Whole Foods' AS source 
                FROM whole_foods 
                WHERE _id IN ({format_strings})
            """
            print(f"🔹 Executing whole_foods query: {query} with values {wf_item_ids}")
            cursor.execute(query, tuple(wf_item_ids))
            whole_foods = cursor.fetchall()

            if not whole_foods:
                print("⚠️ No matching Whole Foods items found!")
            else:
                print(f"✅ Whole Foods items fetched: {whole_foods}")

        print(f"🔹 Shopping Items: {shopping_items}")  # ✅ Debug shopping items
        print(f"🔹 Intermediate Item Mappings: {mapped_items}")  # ✅ Debug mapping step
        print(f"🔹 Resolved real item IDs: {real_item_ids}")  # ✅ Debug resolved IDs

        # ✅ Step 6: Create a mapping of `real_item_id` (final item) to `shopping_id`
        item_map = {item["item"]: item["_id"] for item in mapped_items}

        # ✅ Append manual items directly
        for manual in manual_items:
            items.append({
                "_id": manual["_id"],  # ✅ Unique ID
                "title": manual["item"],  # ✅ Use `item` as `title`
                "brand": "Manual Entry",  # ✅ Placeholder brand
                "images": "",  # ✅ No image for manual items
                "simplified_category": "Other",  # ✅ Default category
                "shopping_id": manual["_id"],  # ✅ Attach correct shopping_id
                "action": manual.get("action", "1"),  # ✅ Default action to "1"
                "source": "manual"  # ✅ Mark as manual
            })

        print(f"✅ Final shopping items (including manual): {items}")  # ✅ Debug final item list



        for item in items:
            for mapped_item in mapped_items:
                if mapped_item["item"] == item["_id"]:  # ✅ Correct mapping from web_id-items
                    for shopping_item in shopping_items:
                        if shopping_item["item_id"] == mapped_item["_id"]:  # ✅ Correct mapping from shopping table
                            item["shopping_id"] = shopping_item["_id"]  # ✅ Assign shopping table _id
                            item["action"] = shopping_item.get("action", "1")  # ✅ Default to "1" if missing
                            break  # Stop searching once found
                    break  # Stop searching in mapped_items

        for wf_item in whole_foods:
            for shopping_item in shopping_items:
                if shopping_item["item_id"] == wf_item["_id"] and shopping_item["source"] == "whole_foods":
                    wf_item["shopping_id"] = shopping_item["_id"]  # ✅ Assign correct `shopping_id`
                    wf_item["action"] = shopping_item.get("action", "1")  # ✅ Ensure `action` is included
                    break  # No need to check further


        # ✅ Step 7: Return structured response
        response_body = {"items": items, "whole_foods": whole_foods}
        print(f"🔹 Response body: {response_body}")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_body),
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": "Internal server error"})}
