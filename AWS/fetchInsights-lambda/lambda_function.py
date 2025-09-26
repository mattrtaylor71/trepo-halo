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

    if method != 'GET':
        return {
            'statusCode': 405,
            'body': json.dumps({"error": "Method not allowed"}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            }
        }
    
    # Parse the web_id from query parameters
    web_id = event['queryStringParameters'].get('web_id')
    print("Received web_id:", web_id)  # Log the received web_id
    
    if not web_id:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "web_id is required"}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            }
        }

    # Database connection details
    connection = pymysql.connect(
        host='database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',  # Replace with your RDS endpoint
        user='admin',      # Replace with your RDS username
        password='Nbmqyq17!',  # Replace with your RDS password
        database='mysqlTutorial'
    )
    
    try:
        with connection.cursor() as cursor:
            # Query the insights table for the given web_id
            sql = "SELECT * FROM insights WHERE _owner = %s"
            cursor.execute(sql, (web_id,))
            insights = cursor.fetchall()

            # Fetch the column names dynamically
            columns = [desc[0] for desc in cursor.description]
            print("Fetched columns:", columns)

            # Format the result as a list of dictionaries with all fields
            insights = [dict(zip(columns, row)) for row in insights]
            print("Insights fetched:", insights)

        # Use the default_serializer to handle datetime and decimal objects
        return {
            'statusCode': 200,
            'body': json.dumps(insights, default=default_serializer),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # Enable CORS
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
