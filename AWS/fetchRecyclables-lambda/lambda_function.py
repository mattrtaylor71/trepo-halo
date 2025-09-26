import json
import pymysql
from datetime import datetime, timedelta
from decimal import Decimal

def default_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()  # Convert datetime to ISO format string
    if isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float
    raise TypeError(f"Type {type(obj)} not serializable")

def lambda_handler(event, context):
    # Log the incoming event for debugging
    print("Received event:", json.dumps(event))  # Log the entire event
    
    # Determine the HTTP method from the requestContext
    method = event['requestContext']['http']['method']
    print("HTTP Method:", method)  # Log the HTTP method

    # Database connection details
    connection = pymysql.connect(
        host='database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',  # Replace with your RDS endpoint
        user='admin',      # Replace with your RDS username
        password='Nbmqyq17!',  # Replace with your RDS password
        database='mysqlTutorial'
    )
    
    try:
        if method == 'POST':
            return get_recyclables(event, connection)
        elif method == 'PUT':
            return update_inventory(event, connection)
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({"error": "Method not allowed"}),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                }
            }
    except Exception as e:
        print("Error occurred:", str(e))  # Log the error message
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            }
        }
    finally:
        connection.close()

def get_recyclables(event, connection):
    # Parse the incoming request body to get the user name
    body = json.loads(event.get("body", "{}"))  # Safely parse the JSON body
    user_name = body.get("user_name")
    print("Parsed user_name:", user_name)  # Log the parsed user_name
    
    if not user_name:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "user_name is required"}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            }
        }

    with connection.cursor() as cursor:
        # Fetch the web_id for the provided user_name (auth0_sub)
        sql = "SELECT web_id FROM users WHERE auth0_sub = %s"
        cursor.execute(sql, (user_name,))
        result = cursor.fetchone()
        
        if not result:
            return {
                'statusCode': 404,
                'body': json.dumps({"error": "User not found"}),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                }
            }
        
        web_id = result[0]
        table_name = f"{web_id}-raw"
        fourteen_days_ago = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        
        # Log the table name for debugging
        print(f"Fetching data from table: {table_name}")
        
        # Execute query to fetch items created in the last 14 days
        sql = f"SELECT * FROM `{table_name}` WHERE _createdDate >= %s"
        cursor.execute(sql, (fourteen_days_ago,))
        items = cursor.fetchall()

        # Fetch the column names dynamically
        columns = [desc[0] for desc in cursor.description]
        print("Fetched columns:", columns)

        # Format the result as a list of dictionaries with all fields
        items = [dict(zip(columns, row)) for row in items]
        print("Items fetched:", items)

    # Use the default_serializer to handle datetime and decimal objects
    return {
        'statusCode': 200,
        'body': json.dumps(items, default=default_serializer),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Enable CORS
        }
    }

# The update_inventory function remains unchanged, assuming it follows a similar pattern
def update_inventory(event, connection):
    # Parse the incoming request body
    body = json.loads(event['body'])  # Get the JSON data from the request
    user_name = body.get("user_name")  # Retrieve the user_name
    updates = body.get("items", [])  # Extract the items to update

    print("Received updates:", updates)  # Log the updates received
    print("User name:", user_name)  # Log the user name received

    if not user_name or not updates:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "user_name and items are required"}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # Enable CORS
            }
        }

    with connection.cursor() as cursor:
        # Fetch the web_id for the provided user_name
        cursor.execute("SELECT web_id FROM users WHERE auth0_sub = %s", (user_name,))
        result = cursor.fetchone()

        if not result:
            return {
                'statusCode': 404,
                'body': json.dumps({"error": "User not found"}),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                }
            }
        
        web_id = result[0]
        table_name = f"{web_id}-raw"  # Create the dynamic table name

        for update in updates:
            item_id = update['id']  # Using the _id field as the unique identifier
            inventory_status = update['inventory']
            print(f"Updating item ID {item_id} with inventory status: {inventory_status}")  # Log each update
            
            # Update the inventory field in the user's specific table
            sql = f"UPDATE `{table_name}` SET inventory = %s WHERE _id = %s"
            cursor.execute(sql, (inventory_status, item_id))
        
        # Commit the changes to the database
        connection.commit()

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Inventory updated successfully'}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Enable CORS
        }
    }
