import sys,uuid,json,threading,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pymysql,pytest
sys.path.insert(0,os.environ.get('TREPO_KITCHEN_TEST_SOURCE',str(Path(__file__).resolve().parents[1]/'kitchen_api')))
import amount_operations as a
O='owner';M='member';X='stranger'
@pytest.fixture
def db():
 name='quantity_fixture_'+uuid.uuid4().hex
 root=pymysql.connect(host='127.0.0.1',port=int(os.environ.get("TREPO_AMOUNT_TEST_PORT", "33316")),user='root',password='',autocommit=True)
 with root.cursor() as c:c.execute('CREATE DATABASE '+name)
 def connect():return pymysql.connect(host='127.0.0.1',port=int(os.environ.get("TREPO_AMOUNT_TEST_PORT", "33316")),user='root',password='',database=name,autocommit=True,cursorclass=pymysql.cursors.DictCursor)
 conn=connect()
 with conn.cursor() as c:
  cols='_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),_updatedDate DATETIME, action VARCHAR(10)'
  for f,t in a.FIELDS.items():cols+=',`'+f+'` '+({'decimal':'DECIMAL(10,2)','int':'INT','bool':'BOOLEAN','date':'DATE','list':'JSON'}.get(t,'TEXT'))
  c.execute('CREATE TABLE shared_kitchen ('+cols+') ENGINE=InnoDB')
  c.execute('CREATE TABLE shared_archive_kitchen LIKE shared_kitchen')
  c.execute('ALTER TABLE shared_archive_kitchen ADD archived_at DATETIME, ADD archived_reason TEXT, ADD archived_from_table TEXT')
  for sql in a.schema_sql():c.execute(sql)
  for i,owner in [('beef',O),('eggs',M),('private',X)]:c.execute("INSERT INTO shared_kitchen(_id,owner_id,product_name,quantity_value,quantity_unit,action) VALUES(%s,%s,%s,25,'oz','IN')",(i,owner,i))
 yield conn,connect
 conn.close()
 with root.cursor() as c:c.execute('DROP DATABASE '+name)
 root.close()
def request(q='12.5',item='beef',rev=0):return {'operation_id':str(uuid.uuid4()),'kind':'consume','items':[{'item_id':item,'revision':rev,'amount':q,'unit':'oz','expected_amount':'25' if rev==0 else '12.5'}]}
def run(db,b):return a.execute(db[0],O,b,[O,M])
def row(db,item='beef'):
 with db[0].cursor() as c:c.execute('SELECT * FROM shared_kitchen WHERE _id=%s',(item,));return c.fetchone()
def test_half_retry_and_quarter(db):
 b=request();r=run(db,b);assert str(r['items'][0]['item']['quantity_value'])=='12.50';assert run(db,b)==r;assert row(db)['quantity_value']==a.amount('12.5')
 r2=run(db,request('6.25',rev=r['items'][0]['item']['amount_revision']));assert a.amount(r2['items'][0]['item']['quantity_value'])==a.amount('6.25')
 assert r2['items'][0]['item']['fill_percent']==25
 assert r2['items'][0]['item']['remaining_quantity']=='6.25 oz'

@pytest.mark.parametrize('unit',['cloves','heads'])
def test_explicit_garlic_units_persist_and_consume_without_guessing_count_conversion(db,unit):
 set_body={'operation_id':str(uuid.uuid4()),'kind':'set','items':[{'item_id':'beef','revision':0,'fields':{'quantity_value':4,'quantity_unit':unit}}]}
 saved=run(db,set_body);item=saved['items'][0]['item'];canonical='clove' if unit=='cloves' else 'head'
 assert item['quantity_unit']==canonical
 consume={'operation_id':str(uuid.uuid4()),'kind':'consume','items':[{'item_id':'beef','revision':item['amount_revision'],'amount':1,'unit':canonical,'expected_amount':4}]}
 result=run(db,consume);assert a.amount(result['items'][0]['item']['quantity_value'])==3
 assert run(db,consume)==result
def test_duplicate_id_and_payload_reuse(db):
 b=request();run(db,b);b['items'][0]['amount']='1'
 with pytest.raises(a.Rejected):run(db,b)
 b=request();b['items']*=2
 with pytest.raises(a.Rejected):run(db,b)
def test_batch_rolls_back_on_private_or_stale(db):
 for id in ['private','missing']:
  b=request();b['items']+=request(item=id)['items']
  with pytest.raises(a.Rejected):run(db,b)
  assert row(db)['quantity_value']==25
 b=request();b['items']+=request(item='eggs',rev=99)['items']
 with pytest.raises(a.Rejected):run(db,b)
 assert row(db)['quantity_value']==25
 assert row(db,'eggs')['quantity_value']==25
@pytest.mark.parametrize('q',['NaN','Infinity','-1','0','25.01','0.001',True])
def test_bad_amounts(db,q):
 with pytest.raises(a.Rejected):run(db,request(q))
 assert row(db)['quantity_value']==25

def test_undo_preserves_subsequent_use_and_once(db):
 first=run(db,request());run(db,request('2.5',rev=first['items'][0]['item']['amount_revision']))
 undo={'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':first['operation_id']};run(db,undo);assert row(db)['quantity_value']==a.amount('22.5');run(db,undo);assert row(db)['quantity_value']==a.amount('22.5')
 undo['operation_id']=str(uuid.uuid4())
 with pytest.raises(a.Rejected):run(db,undo)
def test_empty_and_restore(db):
 first=run(db,request('25'));assert row(db) is None
 run(db,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':first['operation_id']});assert row(db)['quantity_value']==25

def test_edit_protection_clear_and_analysis_merge(db):
 fields={'product_name':'My beef','brand':None,'ingredients':['Beef'],'storage_location':'freezer','quantity_value':'12.5','quantity_unit':'oz'}
 r=run(db,{'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'beef','revision':0,'fields':fields}]})
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET product_name='AI beef',brand='AI brand',ingredients=JSON_ARRAY('Wrong'),quantity_value=25,storage_location='fridge',nutrition_summary='new analysis' WHERE _id='beef'")
 v=row(db);assert v['product_name']=='My beef';assert v['brand'] is None;assert json.loads(v['ingredients'])==['Beef'];assert v['quantity_value']==a.amount('12.5');assert v['storage_location']=='freezer';assert v['nutrition_summary']=='new analysis'
 with pytest.raises(a.Rejected):run(db,request(rev=r['items'][0]['item']['amount_revision']))

def test_concurrent_requests_only_one_deducts(db):
 barrier=threading.Barrier(2)
 def worker():
  c=db[1]();barrier.wait()
  try:a.execute(c,O,request(),[O,M]);return 'ok'
  except a.Rejected:return 'conflict'
  finally:c.close()
 with ThreadPoolExecutor(2) as ex:results=list(ex.map(lambda _:worker(),range(2)))
 assert sorted(results)==['conflict','ok'];assert row(db)['quantity_value']==a.amount('12.5')
def test_unknown_and_units_not_guessed(db):
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET quantity_unit=NULL,quantity_value=NULL WHERE _id='beef'")
 with pytest.raises(a.Rejected):run(db,request())
 r=run(db,{'operation_id':str(uuid.uuid4()),'kind':'set','items':[{'item_id':'beef','revision':0,'fields':{'quantity_value':'25','quantity_unit':'oz'}}]})
 b=request(rev=r['items'][0]['item']['amount_revision']);b['items'][0]['unit']='fl oz'
 with pytest.raises(a.Rejected):run(db,b)

def test_first_use_rejects_analysis_amount_change(db):
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET quantity_value=30 WHERE _id='beef'")
 with pytest.raises(a.Rejected):run(db,request())
 assert row(db)['quantity_value']==30

def test_legacy_rename_storage_and_stale_quantity(db,monkeypatch):
 import app
 r=run(db,request())
 monkeypatch.setattr(app,'_mysql_conn',lambda:db[0]);monkeypatch.setattr(app,'_get_household_member_ids',lambda *args:[O,M]);monkeypatch.setattr(app,'_resolve_kitchen_table',lambda *args:('shared_kitchen','owner_id IN (%s,%s)',[O,M]));monkeypatch.setattr(app,'_mark_recipe_refresh_needed_for_owners',lambda *args:None);monkeypatch.setattr(app,'_refresh_shelf_life_cache',lambda *args:None)
 assert app._update_kitchen_item(O,'beef',{'product_name':'Old client rename','storage_location':'freezer'})['statusCode']==200
 assert row(db)['quantity_value']==a.amount('12.5');assert row(db)['product_name']=='Old client rename'
 assert app._update_kitchen_item(O,'beef',{'quantity_value':25})['statusCode']==409
 assert row(db)['quantity_value']==a.amount('12.5')
 assert app._update_kitchen_item(O,'beef',{'quantity_value':12.5,'brand':'Old brand'})['statusCode']==200
 assert row(db)['brand']=='Old brand'
 unchanged=app._update_kitchen_item(O,'beef',{'quantity_value':12.5})
 assert unchanged['statusCode']==200 and json.loads(unchanged['body'])['item']['_id']=='beef'
 # Existing full-row archive/delete still works with the sidecar and triggers.
 app._shared_kitchen_delete(db[0],O,'beef','deleted');assert row(db) is None
 with pytest.raises(a.Rejected):run(db,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':r['operation_id']})

def test_released_enrichment_write_on_unmanaged_item_unchanged(db):
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET quantity_value=2,product_name='Analyzed beef' WHERE _id='beef'")
 assert row(db)['quantity_value']==2;assert row(db)['product_name']=='Analyzed beef'

@pytest.mark.parametrize('u',['gallon','liter','cups','bottle','pack'])
def test_legacy_units_keep_their_dimension(db,u):
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET quantity_value=1,quantity_unit=%s WHERE _id='beef'",(u,))
 b=request('.5');b['items'][0].update(unit=u,expected_amount=1)
 r=run(db,b);assert a.amount(r['items'][0]['item']['quantity_value'])==a.amount('.5');assert r['items'][0]['item']['quantity_unit']==a.unit(u)


def test_explicit_correction_replaces_description_but_keeps_used_amount(db):
 first=run(db,request())
 edited=run(db,{'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'beef','revision':first['items'][0]['item']['amount_revision'],'fields':{'product_name':'My beef','ingredients':['Beef']}}]})
 revision=edited['items'][0]['item']['amount_revision']
 a.apply_explicit_correction(db[0],O,'beef',{'product_name':'Ground turkey','ingredients':'["Turkey"]'},revision,[O,M])
 item=row(db);assert item['product_name']=='Ground turkey' and json.loads(item['ingredients'])==['Turkey'];assert item['quantity_value']==a.amount('12.5')
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET product_name='Late beef',ingredients=JSON_ARRAY('Beef'),quantity_value=25 WHERE _id='beef'")
 assert row(db)['product_name']=='Ground turkey' and row(db)['quantity_value']==a.amount('12.5')
 with pytest.raises(a.Rejected):a.apply_explicit_correction(db[0],O,'beef',{'product_name':'Stale correction'},revision,[O,M])
 assert row(db)['product_name']=='Ground turkey'
 with pytest.raises(a.Rejected):a.apply_explicit_correction(db[0],O,'private',{'product_name':'Forbidden'},0,[O,M])
 assert row(db,'private')['product_name']=='private'

def test_explicit_correction_handler_preserves_amount(db,monkeypatch):
 import app
 run(db,request())
 with db[0].cursor() as c:
  for column in ['upf','harmful_ingredients','healthier_alternatives','analysis_stage']:c.execute('ALTER TABLE shared_kitchen ADD '+column+' TEXT')
 monkeypatch.setattr(app,'USE_SHARED_TABLES',True)
 monkeypatch.setattr(app,'_get_household_member_ids',lambda *args:[O,M]);monkeypatch.setattr(app,'_resolve_kitchen_table',lambda *args:('shared_kitchen','owner_id IN (%s,%s)',[O,M]))
 for name in ['_mark_recipe_refresh_needed_for_owners','_record_master_feed_event','_invoke_kitchen_analysis_generator','_refresh_shelf_life_cache']:monkeypatch.setattr(app,name,lambda *args:None)
 monkeypatch.setattr(app,'_correct_kitchen_item_with_llm',lambda *args,**kwargs:{'product_name':'Ground turkey','ingredients':['Turkey'],'quantity_value':25})
 response=app._handle_correct_kitchen_item(O,'beef',{'correction':'Actually turkey'},db[0])
 assert response['statusCode']==200;assert row(db)['product_name']=='Ground turkey';assert row(db)['quantity_value']==a.amount('12.5')


def discard(item='beef',revision=0):
 return {'operation_id':str(uuid.uuid4()),'kind':'discard','items':[{'item_id':item,'revision':revision}]}

def test_clean_discard_restore_unknown_amount_and_retry(db):
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET quantity_value=NULL,quantity_unit=NULL,brand='My brand',product_expiration='2026-09-20' WHERE _id='beef'")
 before=row(db);body=discard();result=run(db,body)
 assert row(db) is None and result['can_undo']
 assert run(db,body)==result
 undo={'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':body['operation_id']}
 restored=run(db,undo)
 assert row(db)==before and not restored['items'][0]['removed']
 assert run(db,undo)==restored and row(db)==before
 with pytest.raises(a.Rejected):run(db,{**undo,'operation_id':str(uuid.uuid4())})

def test_clean_conflict_and_household_ownership(db):
 for body in [discard('private'),discard('beef',99)]:
  with pytest.raises(a.Rejected):run(db,body)
  assert row(db) is not None
 result=run(db,discard('eggs'));assert row(db,'eggs') is None
 with pytest.raises(a.Rejected):a.execute(db[0],X,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':result['operation_id']},[X])
 run(db,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':result['operation_id']});assert row(db,'eggs')['owner_id']==M

def test_clean_undo_cannot_restore_after_new_archive_operation(db):
 result=run(db,discard())
 with db[0].cursor() as c:c.execute("UPDATE shared_archive_kitchen SET archived_reason='deleted' WHERE _id='beef'")
 with pytest.raises(a.Rejected):run(db,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':result['operation_id']})
 assert row(db) is None

def test_clean_discard_preserves_human_edits_after_restore(db):
 edited=run(db,{'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'beef','revision':0,'fields':{'brand':'Mine','quantity_value':'12.5','quantity_unit':'oz'}}]})
 result=run(db,discard(revision=edited['items'][0]['item']['amount_revision']))
 run(db,{'operation_id':str(uuid.uuid4()),'kind':'undo','undo_of':result['operation_id']})
 with db[0].cursor() as c:c.execute("UPDATE shared_kitchen SET brand='Late AI',quantity_value=25 WHERE _id='beef'")
 assert row(db)['brand']=='Mine' and row(db)['quantity_value']==a.amount('12.5')


def test_state_clocks_record_transitions_and_preserve_repeat_edits(db):
 first=run(db,{'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'beef','revision':0,'fields':{'is_opened':True,'storage_location':'freezer'}}]})
 item=first['items'][0]['item'];assert item['opened_at'] and item['storage_started_at']
 second=run(db,{'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'beef','revision':item['amount_revision'],'fields':{'is_opened':True,'brand':'Other'}}]})['items'][0]['item']
 assert second['opened_at']==item['opened_at'] and second['storage_started_at']==item['storage_started_at']
 # Older clients' explicit toggle also records the same metadata contract.
 import app
 # Direct operation replay remains idempotent, including clock values.
 assert run(db,{'operation_id':first['operation_id'],'kind':'edit','items':[{'item_id':'beef','revision':0,'fields':{'is_opened':True,'storage_location':'freezer'}}]})==first


@pytest.mark.parametrize('fields',[{'quantity_value':12.5},{'quantity_unit':'g'},{'quantity_value':12.5,'quantity_unit':'oz'}])
def test_legacy_quantity_edit_invalidates_all_household_recipe_versions(db,monkeypatch,fields):
 import app
 with db[0].cursor() as c:
  c.execute('CREATE TABLE owner_prod_kitchen (_updatedDate DATETIME)')
  c.execute('ALTER TABLE shared_kitchen ADD analysis_updated_at DATETIME')
 monkeypatch.setattr(app,'_mysql_conn',lambda:db[0])
 monkeypatch.setattr(app,'USE_SHARED_TABLES',True)
 monkeypatch.setattr(app,'_get_household_member_ids',lambda *args:[O,M])
 monkeypatch.setattr(app,'_resolve_kitchen_table',lambda *args:('shared_kitchen','owner_id IN (%s,%s)',[O,M]))
 for name in ['_ensure_prod_kitchen_table','_record_master_feed_event','_invoke_kitchen_analysis_generator','_refresh_shelf_life_cache']:
  monkeypatch.setattr(app,name,lambda *args:None)
 app._ensure_owner_kitchen_state_table(db[0])
 with db[0].cursor() as c:
  c.executemany('INSERT INTO owner_kitchen_state(owner,kitchen_version) VALUES(%s,7)',[(O,),(M,)])
 response=app._update_kitchen_item(O,'beef',fields)
 assert response['statusCode']==200
 saved=row(db)
 for key,value in fields.items(): assert saved[key]==value
 with db[0].cursor() as c:
  c.execute('SELECT owner,kitchen_version FROM owner_kitchen_state ORDER BY owner')
  versions={r['owner']:r['kitchen_version'] for r in c.fetchall()}
 assert versions=={O:8,M:8},'Old request must invalidate availability for every household member'
