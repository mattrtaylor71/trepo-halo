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
    # Log the incoming event
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
        if method == 'GET':
            return get_recyclables(connection)
        elif method == 'POST':
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

def get_recyclables(connection):
    with connection.cursor() as cursor:
        # Calculate the date 14 days ago from today
        fourteen_days_ago = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        print("Fetching recyclables created after:", fourteen_days_ago)  # Log the date

        # Execute query to fetch all fields from items created in the last 14 days
        sql = """
            SELECT * FROM `0b358632-5166-4d17-930d-2250474aa709-raw`
            WHERE _createdDate >= %s
        """
        cursor.execute(sql, (fourteen_days_ago,))
        result = cursor.fetchall()

        # Fetch the column names dynamically
        columns = [desc[0] for desc in cursor.description]
        print("Fetched columns:", columns)  # Log fetched column names

        # Format the result as a list of dictionaries with all fields
        items = [dict(zip(columns, row)) for row in result]
        print("Items fetched:", items)  # Log the fetched items

    # Use the default_serializer to handle datetime and decimal objects
    return {
        'statusCode': 200,
        'body': json.dumps(items, default=default_serializer),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Enable CORS
        }
    }

def update_inventory(event, connection):
    # Parse the incoming request body
    updates = json.loads(event['body'])  # Get the JSON data from the request
    print("Received updates:", updates)  # Log the updates received
    
    # Ensure 'updates' is a list, even if it's a single object
    if isinstance(updates, dict):
        updates = [updates]  # Convert single object to a list

    print("Processing updates:", updates)  # Log the updates to be processed

    with connection.cursor() as cursor:
        for update in updates:
            item_id = update['id']  # Using the _id field as the unique identifier
            inventory_status = update['inventory']
            print(f"Updating item ID {item_id} with inventory status: {inventory_status}")  # Log each update
            
            # Update the inventory field in the database
            sql = "UPDATE `0b358632-5166-4d17-930d-2250474aa709-raw` SET inventory = %s WHERE _id = %s"
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

