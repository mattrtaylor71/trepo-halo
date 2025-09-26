import json
import pymysql
import bcrypt

# Database configuration
db_config = {
    'host': 'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',
    'user': 'admin',
    'password': 'Nbmqyq17',
    'database': 'mysqlTutorial',
}

# Connect to the database
def get_db_connection():
    return pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        cursorclass=pymysql.cursors.DictCursor
    )

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Compare password with hash
def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# Lambda handler function
def lambda_handler(event, context):
    body = json.loads(event['body'])
    email = body.get('email')
    password = body.get('password')
    signup_code = body.get('signup_code')  # Only used for signup

    is_signup = bool(signup_code)

    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            if is_signup:
                return handle_signup(cursor, email, password, signup_code, connection)
            else:
                return handle_login(cursor, email, password)

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': str(e)}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    finally:
        connection.close()

# Handle the user signup
def handle_signup(cursor, email, password, signup_code, connection):
    # Check if the email already exists
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    
    if user:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'User already exists!'}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    # Validate the signup code
    cursor.execute('SELECT * FROM signup_codes WHERE code = %s', (signup_code,))
    valid_code = cursor.fetchone()
    
    if not valid_code:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Invalid signup code!'}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    # Hash the password
    hashed_password = hash_password(password)
    
    # Insert the new user into the database
    cursor.execute(
        'INSERT INTO users (email, password, signup_code, _createdDate) VALUES (%s, %s, %s, NOW())',
        (email, hashed_password, signup_code)
    )
    connection.commit()

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Signup successful!'}),
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
    }

# Handle the user login
def handle_login(cursor, email, password):
    # Retrieve the user by email
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    
    if not user or not check_password(password, user['password']):
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Invalid email or password!'}),
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
        }

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Login successful!'}),
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}
    }
