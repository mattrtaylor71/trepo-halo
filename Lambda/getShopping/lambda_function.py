import json
import pymysql
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

def default_serializer(obj):
    import datetime
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    method = event['requestContext']['http']['method']
    print("HTTP Method:", method)

    connection = pymysql.connect(
        host='database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',
        user='admin',
        password='Nbmqyq17!',
        database='mysqlTutorial',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        if method == 'POST':
            return get_shopping_list(event, connection)
        elif method == 'PUT':
            return update_shopping_list(event, connection)
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({"error": "Method not allowed"}),
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
            }
    except Exception as e:
        print("Error occurred:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }
    finally:
        connection.close()

def get_shopping_list(event, connection):
    body = json.loads(event.get("body", "{}"))  
    user_name = body.get("user_name")
    print("Parsed user_name:", user_name)

    if not user_name:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "user_name is required"}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    with connection.cursor() as cursor:
        sql = "SELECT web_id FROM users WHERE auth0_sub = %s"
        cursor.execute(sql, (user_name,))
        result = cursor.fetchone()

        if not result:
            return {
                'statusCode': 404,
                'body': json.dumps({"error": "User not found"}),
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
            }

        web_id = result["web_id"]
        table_name = f"{web_id}-shopping"

        print(f"Fetching data from table: {table_name}")

        sql = f"""
            SELECT _id, item, source, _createdDate, _updatedDate 
            FROM `{table_name}`
        """
        cursor.execute(sql)
        shopping_items = cursor.fetchall()

        print("Fetched shopping list:", shopping_items)

        return {
            'statusCode': 200,
            'body': json.dumps(shopping_items, default=default_serializer),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

def update_shopping_list(event, connection):
    body = json.loads(event.get("body", "{}"))  
    user_name = body.get("user_name")
    item_id = body.get("item_id")
    action = body.get("action")  # 'add' or 'remove'
    source = body.get("source")

    print("User name:", user_name, "Item ID:", item_id, "Action:", action, "Source:", source)

    if not user_name or not item_id or not action or not source:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "user_name, item_id, action, and source are required"}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    with connection.cursor() as cursor:
        sql = "SELECT web_id FROM users WHERE auth0_sub = %s"
        cursor.execute(sql, (user_name,))
        result = cursor.fetchone()

        if not result:
            return {
                'statusCode': 404,
                'body': json.dumps({"error": "User not found"}),
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
            }

        web_id = result["web_id"]
        table_name = f"{web_id}-shopping"

        if action == "add":
            new_uuid = str(uuid.uuid1())
            sql_insert = f"""
                INSERT INTO `{table_name}` (_id, item, source, _createdDate, _updatedDate)
                VALUES (%s, %s, %s, NOW(), NOW())
            """
            cursor.execute(sql_insert, (new_uuid, item_id, source))
        elif action == "remove":
            sql_delete = f"DELETE FROM `{table_name}` WHERE item = %s AND source = %s"
            cursor.execute(sql_delete, (item_id, source))
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Invalid action. Must be 'add' or 'remove'"}),
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
            }

        connection.commit()

    return {
        'statusCode': 200,
        'body': json.dumps({"message": "Shopping list updated successfully"}),
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
    }
