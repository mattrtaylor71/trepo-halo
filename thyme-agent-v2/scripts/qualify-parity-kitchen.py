"""Exercise the pinned serving canonical kitchen implementation in isolated MySQL."""
import sys,os,json,uuid,pathlib
sys.path.insert(0,os.environ['THYME_CANONICAL_ROOT'])
import amount_operations as m
import pymysql
c=pymysql.connect(host='127.0.0.1',port=33317,user='fixture',password='fixture',database='thyme_test_kitchen_parity_20261002',cursorclass=pymysql.cursors.DictCursor,autocommit=True)
report={'environment':'isolated MySQL with serving canonical kitchen module','customerWrites':0,'tests':[]}
def passed(x):report['tests'].append(x);print(x)
def query(q,args=()):
 with c.cursor() as cur:cur.execute(q,args);return cur.fetchall()
def read(id):return m.annotate(c,query('SELECT * FROM shared_kitchen WHERE _id=%s',(id,)))[0]
def execute(kind,items):return m.execute(c,'actor',{'operation_id':str(uuid.uuid4()),'kind':kind,'items':items},['actor','member'])
try:
 fields=[]
 for name,kind in m.FIELDS.items():fields.append('`'+name+'` '+{'text':'TEXT','list':'JSON','bool':'TINYINT','decimal':'DECIMAL(10,2)','date':'DATE','int':'INT'}[kind])
 query('CREATE TABLE IF NOT EXISTS shared_kitchen(_id VARCHAR(64) PRIMARY KEY,owner_id VARCHAR(64),action VARCHAR(20),_createdDate DATETIME,_updatedDate DATETIME,'+','.join(fields)+') ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci')
 query('CREATE TABLE IF NOT EXISTS shared_archive_kitchen LIKE shared_kitchen')
 cols={r['Field'] for r in query('SHOW COLUMNS FROM shared_archive_kitchen')}
 for col,typ in [('archived_at','DATETIME'),('archived_reason','VARCHAR(50)'),('source_table','VARCHAR(64)')]:
  if col not in cols:query('ALTER TABLE shared_archive_kitchen ADD '+col+' '+typ)
 for q in m.schema_sql():
  if "CREATE TRIGGER" in q and query("SELECT TRIGGER_NAME FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=DATABASE() AND TRIGGER_NAME=%s",(q.split()[2],)):continue
  try:query(q)
  except pymysql.err.OperationalError as e:
   if e.args[0]!=1359:raise
 query('DELETE FROM shared_kitchen');query('DELETE FROM shared_archive_kitchen');query('DELETE FROM kitchen_item_edits');query('DELETE FROM kitchen_amount_operations')
 for id,owner in [('eggs','actor'),('rice','member'),('foreign','other')]:query("INSERT INTO shared_kitchen(_id,owner_id,action,product_name,quantity_value,quantity_unit,storage_location,is_opened) VALUES(%s,%s,'IN',%s,6,'count','fridge',0)",(id,owner,id))
 for fields in [{'quantity_value':4,'quantity_unit':'count'},{'product_name':'Organic eggs'},{'storage_location':'freezer'},{'is_opened':True},{'product_expiration':'2026-12-01'}]:
  old=read('eggs');body={'operation_id':str(uuid.uuid4()),'kind':'edit','items':[{'item_id':'eggs','revision':old['amount_revision'],'fields':fields}]};r=m.execute(c,'actor',body,['actor','member']);new=read('eggs')
  for k,v in fields.items():assert str(new[k])==str(v) or (k=='is_opened' and bool(new[k])==v) or (k=='quantity_value' and float(new[k])==v),(k,new[k],v)
  assert new['amount_revision']>old['amount_revision'];assert m.execute(c,'actor',body,['actor','member'])==r
 passed('Quantity, name, storage, opened state and explicit expiry persist; duplicate operation does not repeat')
 new=read('eggs');assert new['opened_at'] and new['storage_started_at'] and new['expiration_provenance']['source']=='item_edit';passed('Shelf-life event timestamps and explicit expiration provenance retained')
 query("UPDATE shared_kitchen SET product_name='Late model overwrite',quantity_value=99 WHERE _id='eggs'");new=read('eggs');assert new['product_name']=='Organic eggs' and float(new['quantity_value'])==4;passed('Delayed enrichment cannot overwrite approved name/amount')
 try:execute('edit',[{'item_id':'eggs','revision':0,'fields':{'quantity_value':2,'quantity_unit':'count'}}]);assert False
 except m.Rejected as e:assert e.status==409
 passed('Stale revision fails without changing quantity')
 old=read('eggs')
 try:execute('discard',[{'item_id':'eggs','revision':old['amount_revision']},{'item_id':'foreign','revision':0}]);assert False
 except m.Rejected as e:assert e.status==404
 assert read('eggs');passed('Foreign household target rolls back entire batch')
 old=read('eggs');r=execute('consume',[{'item_id':'eggs','revision':old['amount_revision'],'amount':4,'expected_amount':4,'unit':'count'}]);assert r['items'][0]['removed'];assert not query("SELECT * FROM shared_kitchen WHERE _id='eggs'");assert query("SELECT * FROM shared_archive_kitchen WHERE _id='eggs'");passed('Set-to-zero adapter consume archives and removes the exact item')
 r=execute('discard',[{'item_id':'rice','revision':0}]);assert r['items'][0]['removed'];passed('Current household member item can be removed by exact revision')
 report['passed']=True
 pathlib.Path(os.environ['THYME_EVIDENCE']).write_text(json.dumps(report,indent=2)+'\n')
finally:c.close()
