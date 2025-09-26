import json
import pymysql
import os
import logging
import uuid

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# RDS settings
rds_host = os.environ['DB_HOST']
name = os.environ['DB_USER']
password = os.environ['DB_PASSWORD']
db_name = os.environ['DB_NAME']

# Establish a connection to the RDS database
try:
    conn = pymysql.connect(host=rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
    logger.info("Connection to RDS successful")
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    raise

def lambda_handler(event, context):
    try:
        # Print the entire event object for debugging
        logger.info(f"Event: {json.dumps(event)}")
        
        # Extract barcode and owner from the event object
        if 'barcode' in event and 'owner' in event:
            barcode = event['barcode']
            owner = event['owner']
            logger.info(f"Received barcode: {barcode}")
            logger.info(f"Received owner: {owner}")

            with conn.cursor() as cursor:
                sql = "INSERT INTO recyclables (_id, _owner, barcode_number) VALUES (%s, %s, %s)"
                cursor.execute(sql, (str(uuid.uuid1()), owner, barcode))
                conn.commit()
                logger.info("Data inserted successfully")

            return {
                'statusCode': 200,
                'body': json.dumps('Data inserted successfully')
            }
        else:
            logger.error("Error: 'barcode' or 'owner' key not found in event")
            return {
                'statusCode': 400,
                'body': json.dumps("Error: 'barcode' or 'owner' key not found in event")
            }
    except Exception as e:
        logger.error("Error processing the event")
        logger.error(e)
        return {
            'statusCode': 500,
            'body': json.dumps('Error processing the event')
        }
