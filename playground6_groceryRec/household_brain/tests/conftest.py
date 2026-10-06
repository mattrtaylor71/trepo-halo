import os
import uuid
from datetime import datetime,timedelta,timezone
import pymysql
import pytest
from household_brain import migrate,operator

@pytest.fixture
def db():
    if os.getenv('HOUSEHOLD_BRAIN_TEST_MYSQL')!='1':
        pytest.skip('requires isolated synthetic localhost MySQL on 33826')
    conn=pymysql.connect(host='127.0.0.1',port=33826,user='root',autocommit=True,
                         cursorclass=pymysql.cursors.DictCursor,charset='utf8mb4')
    name='hb_test_'+uuid.uuid4().hex
    with conn.cursor() as q:
        q.execute('CREATE DATABASE '+name+' CHARACTER SET utf8mb4');q.execute('USE '+name)
        q.execute('CREATE TABLE new_users(user_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64))')
        inventory='''_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),product_name VARCHAR(255),brand VARCHAR(160),variant VARCHAR(160),barcode VARCHAR(160),
          _createdDate DATETIME(6),quantity_value DECIMAL(12,3),fill_percent DECIMAL(8,3),product_expiration DATE,action VARCHAR(32),
          needs_review BOOLEAN,analysis_status VARCHAR(32),provisional_payload JSON,job_id VARCHAR(64)'''
        for table in ['shared_kitchen','shared_archive_kitchen']:q.execute('CREATE TABLE '+table+'('+inventory+')')
        q.execute('CREATE TABLE shared_saved_recipes(_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),title VARCHAR(255),status VARCHAR(32),_createdDate DATETIME(6))')
        q.execute('CREATE TABLE shared_dishes(_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),user_id VARCHAR(64),dish_name VARCHAR(255),action VARCHAR(32),analysis_status VARCHAR(32),_createdDate DATETIME(6))')
        q.execute('CREATE TABLE shared_meal_calendar(_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),title VARCHAR(255),_createdDate DATETIME(6))')
        q.execute('CREATE TABLE shared_shopping_list(_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),product_name VARCHAR(255),action VARCHAR(32))')
        q.execute('CREATE TABLE kitchen_amount_operations(actor VARCHAR(64),operation_id VARCHAR(64),created_at DATETIME(6),undone_by VARCHAR(64),changes JSON,PRIMARY KEY(actor,operation_id))')
        q.execute('CREATE TABLE user_dietary_preferences(owner_id VARCHAR(64) PRIMARY KEY,allergies JSON,diets JSON,religious JSON,health JSON,custom JSON)')
    migrate.install(conn)
    with conn.cursor() as q:
        q.execute('UPDATE household_brain_config SET capture_enabled=1')
        q.execute("INSERT INTO new_users VALUES ('a','h'),('b','h'),('foreign','other')")
        q.execute("UPDATE household_brain_state SET enabled=1,learning_since=UTC_TIMESTAMP(6)-INTERVAL 90 DAY")
    yield conn
    with conn.cursor() as q:q.execute('DROP DATABASE '+name)
    conn.close()

def sql(conn,query,args=()):
    with conn.cursor() as q:
        q.execute(query,args)
        return q.fetchall()

def edit(conn,action,**kwargs):
    revision=sql(conn,"SELECT input_revision FROM household_brain_state WHERE household_id='h'")[0]['input_revision']
    return operator.change(conn,'arn:test:operator',{'action':action,'household_id':'h','reason':'Synthetic test correction',
        'operation_id':uuid.uuid4().hex,'expected_revision':revision,**kwargs})

def item(conn,name='Milk',id='milk',**fields):
    values={'_id':id,'owner_id':'a','product_name':name,'_createdDate':datetime.now(timezone.utc).replace(tzinfo=None)-timedelta(days=1),
            'action':'IN','analysis_status':'ready','needs_review':0,**fields}
    sql(conn,'INSERT INTO shared_kitchen('+','.join(values)+') VALUES('+','.join(['%s']*len(values))+')',list(values.values()))

def recipe(conn,id='r',actor='a',title='Bean tacos'):
    sql(conn,'INSERT INTO shared_saved_recipes VALUES(%s,%s,%s,\'ready\',UTC_TIMESTAMP(6)-INTERVAL 1 DAY)',[id,actor,title])
