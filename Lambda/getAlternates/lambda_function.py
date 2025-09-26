import json
import mysql.connector
import os

# Database connection config
db_config = {
    "host": os.getenv("DB_HOST", "database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASS", "Nbmqyq17!"),
    "database": os.getenv("DB_NAME", "mysqlTutorial"),
}

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        alt_ids = body.get("alternates", [])

        if not alt_ids:
            return {"statusCode": 400, "body": json.dumps({"error": "No alternates provided"})}

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        query = f"SELECT _id, name, brand, image_urls FROM whole_foods WHERE _id IN ({', '.join(['%s'] * len(alt_ids))})"
        cursor.execute(query, alt_ids)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": "Internal server error"})}
