from expiration_metadata import expiration_provenance
"""Additive, transactional Kitchen operations. No DDL in request handlers.

The sidecar avoids changing shared_kitchen's column order (legacy archive writers
use INSERT SELECT *). A database trigger preserves explicit human values against
all delayed enrichment writers. Existing unedited rows retain their old behavior.
"""
import hashlib
import json
import uuid
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import date, datetime, timezone

FIELDS = {
 'product_name':'text','brand':'text','variant':'text','category':'text',
 'storage_location':'text','is_opened':'bool','product_expiration':'date',
 'product_description':'text','ingredients':'list','nutrition_summary':'text',
 'estimated_price':'text','barcode':'text','country_guess':'text','product_image_url':'text',
 'quantity_value':'decimal','quantity_unit':'text','remaining_quantity':'text','fill_percent':'int',
}
UNITS = {'count','oz','lb','g','kg','ml','l','fl oz','cup','tbsp','tsp','pack','gallon','quart','pint','can','jar','bottle','bag','slice','serving','bunch','container','clove','head'}
ALIASES = {'lbs':'lb','pound':'lb','pounds':'lb','ounces':'oz','ounce':'oz','grams':'g','kilograms':'kg','liter':'l','litre':'l','liters':'l','pieces':'count','each':'count','packs':'pack','gallons':'gallon','quarts':'quart','pints':'pint','cups':'cup','tablespoon':'tbsp','tablespoons':'tbsp','teaspoon':'tsp','teaspoons':'tsp','cans':'can','jars':'jar','bottles':'bottle','bags':'bag','slices':'slice','servings':'serving','containers':'container','package':'pack','packages':'pack','cloves':'clove','heads':'head'}
class Rejected(Exception):
 def __init__(self,message,status=400):self.message=message;self.status=status

def dumps(value):
 return json.dumps(value,default=lambda x:str(x) if isinstance(x,Decimal) else x.isoformat() if isinstance(x,(date,datetime)) else str(x),sort_keys=True,separators=(',',':'),allow_nan=False)
def amount(value):
 if isinstance(value,bool):raise Rejected('Enter a valid amount.')
 try:d=Decimal(str(value))
 except (InvalidOperation,ValueError,TypeError):raise Rejected('Enter a valid amount.')
 if not d.is_finite() or d<0 or d>Decimal('99999999.99'):raise Rejected('Amount must be between 0 and 99,999,999.99.')
 if d != d.quantize(Decimal('.01')):raise Rejected('Use no more than two decimal places.')
 return d

def unit(value):
 s=str(value or '').strip().lower();s=ALIASES.get(s,s)
 if s not in UNITS:raise Rejected('Set an amount and unit before using this item.')
 return s

def schema_sql():
 yield '''CREATE TABLE IF NOT EXISTS kitchen_item_edits (
 item_id VARCHAR(64) PRIMARY KEY, revision BIGINT NOT NULL DEFAULT 0,
 reference_amount DECIMAL(10,2) NULL, overrides JSON NOT NULL,
 updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
 yield '''CREATE TABLE IF NOT EXISTS kitchen_amount_operations (
 actor VARCHAR(64) NOT NULL, operation_id VARCHAR(64) NOT NULL, request_hash CHAR(64) NOT NULL,
 result JSON NULL, changes JSON NULL, undone_by VARCHAR(64) NULL,
 created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6), PRIMARY KEY(actor,operation_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'''
 parts=[]
 for field,kind in FIELDS.items():
  path='$."'+field+'"';extract="JSON_EXTRACT(saved, '%s')"%path
  value=extract if kind=='list' else 'JSON_UNQUOTE('+extract+')'
  parts.append("IF JSON_CONTAINS_PATH(saved, 'one', '%s') THEN SET NEW.`%s` = IF(JSON_TYPE(%s)='NULL', NULL, %s); END IF;"%(path,field,extract,value))
 yield '''CREATE TRIGGER trepo_preserve_item_edits BEFORE UPDATE ON shared_kitchen FOR EACH ROW
 BEGIN
 DECLARE saved JSON;
 SET saved=(SELECT overrides FROM kitchen_item_edits WHERE item_id=CONVERT(OLD._id USING utf8mb4) COLLATE utf8mb4_unicode_ci);
 IF saved IS NOT NULL THEN
 '''+'\n'.join(parts)+'''
 UPDATE kitchen_item_edits SET revision=revision+1 WHERE item_id=CONVERT(OLD._id USING utf8mb4) COLLATE utf8mb4_unicode_ci;
 END IF;
 END'''
 yield '''CREATE TRIGGER trepo_track_item_delete BEFORE DELETE ON shared_kitchen FOR EACH ROW
 UPDATE kitchen_item_edits SET revision=revision+1 WHERE item_id=CONVERT(OLD._id USING utf8mb4) COLLATE utf8mb4_unicode_ci'''



def annotate(conn, rows):
 if not rows:return rows
 with conn.cursor() as cur:
  cur.execute('SELECT item_id, revision, reference_amount, overrides FROM kitchen_item_edits WHERE item_id IN ('+','.join(['%s']*len(rows))+')',[r['_id'] for r in rows]);meta={r['item_id']:r for r in cur.fetchall()}
 for row in rows:
  m=meta.get(row['_id'],{});row['amount_revision']=m.get('revision',0);row['reference_amount']=m.get('reference_amount')
  saved=json.loads(m['overrides']) if isinstance(m.get('overrides'),str) else m.get('overrides',{})
  row['opened_at']=saved.get('__opened_at');row['storage_started_at']=saved.get('__storage_started_at')
  row['expiration_provenance']=expiration_provenance(saved,row.get('product_expiration'))
 return rows

def validate_fields(fields):
 if not isinstance(fields,dict) or not fields or set(fields)-set(FIELDS):raise Rejected('Unsupported item fields.')
 result={}
 for key,v in fields.items():
  kind=FIELDS[key]
  if key in ('remaining_quantity','fill_percent'):raise Rejected('Remaining labels are calculated from the amount.')
  if v is None:result[key]=None;continue
  if kind=='text':
   limit={'product_name':500,'brand':500,'variant':500,'barcode':100,'country_guess':100,'estimated_price':100,'quantity_unit':50,'storage_location':50,'category':255,'product_image_url':2048}.get(key,10000)
   if not isinstance(v,str) or len(v)>limit:raise Rejected('Invalid '+key)
   v=v.strip() or None
  elif kind=='bool':
   if not isinstance(v,bool):raise Rejected('Invalid opened state.')
   v=int(v)
  elif kind=='date':
   try:v=datetime.strptime(v,'%Y-%m-%d').strftime('%Y-%m-%d')
   except (TypeError,ValueError):raise Rejected('Use a valid date.')
  elif kind=='list':
   if not isinstance(v,list) or len(v)>200 or any(not isinstance(s,str) or len(s)>500 for s in v):raise Rejected('Enter a list of ingredients.')
  elif kind=='decimal':v=amount(v)
  result[key]=v
 if 'product_name' in result and not result['product_name']:raise Rejected('Name cannot be empty.')
 if 'storage_location' in result and result['storage_location'] not in (None,'fridge','freezer','pantry'):raise Rejected('Choose a storage location.')
 return result

def _save(cur,row,meta,fields,reference=None):
 saved=json.loads(meta['overrides']) if isinstance(meta['overrides'],str) else dict(meta['overrides'])
 # Record only an observed state transition. Existing opened items without a
 # known start remain unknown; enrichment timestamps are not opening dates.
 now=datetime.now(timezone.utc).isoformat()
 if 'product_expiration' in fields:
  prior=expiration_provenance(saved,row.get('product_expiration'))
  if not prior or prior['value']!=fields['product_expiration']:
   saved['__expiration_provenance']={'version':1,'source':'item_edit',
       'value':fields['product_expiration'],'observed_at':now}
 if 'is_opened' in fields and bool(fields['is_opened']) != bool(row.get('is_opened')):
  saved['__opened_at']=now if fields['is_opened'] else None
 if 'storage_location' in fields and fields['storage_location'] != row.get('storage_location'):
  saved['__storage_started_at']=now
 saved.update(fields)
 cur.execute('UPDATE kitchen_item_edits SET overrides=%s, reference_amount=%s WHERE item_id=%s',(dumps(saved),reference if reference is not None else meta['reference_amount'],row['_id']))
 sets=','.join('`'+k+'`=%s' for k in fields)
 vals=[dumps(v) if isinstance(v,list) else v for v in fields.values()]
 cur.execute('UPDATE shared_kitchen SET '+sets+', `_updatedDate`=NOW() WHERE _id=%s',vals+[row['_id']])

def _quantity_fields(left,u,reference):
 fill=int((left/reference*100).quantize(Decimal('1'),rounding=ROUND_HALF_UP)) if reference and reference>0 else None
 return {'quantity_value':left,'quantity_unit':u,'remaining_quantity':format(left.normalize(),'f')+' '+u,'fill_percent':min(100,fill) if fill is not None else None}

def _verify_saved_fields(row,fields):
 """Compare authoritative storage before committing or creating a success receipt.

 MySQL may accept UPDATE while a trigger changes its values. Normalize database
 representations only; do not forgive missing fields or different user content.
 """
 if not row:raise Rejected('Could not confirm the saved change. Refresh and retry.',409)
 for key,expected in fields.items():
  actual=row.get(key);kind=FIELDS[key]
  try:
   if actual is not None and expected is not None:
    if kind=='decimal':actual=Decimal(str(actual));expected=Decimal(str(expected))
    elif kind in ('bool','int'):actual=int(actual);expected=int(expected)
    elif kind=='date' and isinstance(actual,(datetime,date)):actual=actual.strftime('%Y-%m-%d')
    elif kind=='list' and isinstance(actual,str):actual=json.loads(actual)
   matches=key in row and actual==expected
  except (ValueError,TypeError,InvalidOperation):matches=False
  if not matches:raise Rejected('Could not confirm the saved change. Refresh and retry.',409)

def execute(conn,actor,body,members):
 try:op=str(uuid.UUID(body.get('operation_id','')))
 except (ValueError,TypeError,AttributeError):raise Rejected('Missing operation ID.')
 kind=body.get('kind');request_hash=hashlib.sha256(dumps(body).encode()).hexdigest()
 if kind not in ('consume','set','edit','undo','discard'):raise Rejected('Unsupported operation.')
 conn.begin()
 try:
  with conn.cursor() as cur:
   cur.execute('INSERT IGNORE INTO kitchen_amount_operations(actor,operation_id,request_hash) VALUES(%s,%s,%s)',(actor,op,request_hash))
   cur.execute('SELECT * FROM kitchen_amount_operations WHERE actor=%s AND operation_id=%s FOR UPDATE',(actor,op));prior=cur.fetchone()
   if prior['request_hash']!=request_hash:raise Rejected('This request ID was already used.',409)
   if prior['result']:
    conn.commit();return json.loads(prior['result']) if isinstance(prior['result'],str) else prior['result']
   undo_source=None
   if kind=='undo':
    undo_source=str(body.get('undo_of',''))
    cur.execute('SELECT * FROM kitchen_amount_operations WHERE actor=%s AND operation_id=%s FOR UPDATE',(actor,undo_source));source=cur.fetchone()
    if not source or not source['result'] or source['undone_by']:raise Rejected('This change cannot be undone again.',409)
    changes=json.loads(source['changes']) if isinstance(source['changes'],str) else source['changes']
    if not changes or any(c.get('mode') not in ('consume','discard') for c in changes):raise Rejected('This change cannot be undone.',409)
    entries=[dict(c, item_id=c['snapshot']['_id']) for c in changes]
   else:entries=body.get('items')
   if not isinstance(entries,list) or not 1<=len(entries)<=100:raise Rejected('Choose between 1 and 100 items.')
   ids=[e.get('item_id') for e in entries if isinstance(e,dict)]
   if len(ids)!=len(entries) or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise Rejected('Each item must appear once.')
   results=[];changes=[]
   for entry in sorted(entries,key=lambda x:x['item_id']):
    item_id=entry['item_id']
    cur.execute('SELECT * FROM shared_kitchen WHERE _id=%s FOR UPDATE',(item_id,));row=cur.fetchone()
    restoring=False
    if kind=='undo' and row is None:
     cur.execute('SELECT * FROM shared_archive_kitchen WHERE _id=%s FOR UPDATE',(item_id,));arch=cur.fetchone()
     cur.execute('SELECT overrides FROM kitchen_item_edits WHERE item_id=%s FOR UPDATE',(item_id,));saved_meta=cur.fetchone()
     saved_overrides=json.loads(saved_meta['overrides']) if saved_meta and isinstance(saved_meta['overrides'],str) else (saved_meta or {}).get('overrides',{})
     reasons = ('clean_discard',) if entry.get('mode') == 'discard' else ('consumed_manual','consumed_recipe')
     if not arch or arch.get('archived_reason') not in reasons or saved_overrides.get('__empty_operation')!=undo_source:raise Rejected('This item changed after use. Refresh your kitchen.',409)
     row=dict(entry['snapshot']);restoring=True
    if not row or row.get('owner_id') not in members:raise Rejected('Item is no longer in your household kitchen.',404)
    if not restoring and row.get('action')!='IN':raise Rejected('Item is no longer available.',409)
    cur.execute("INSERT IGNORE INTO kitchen_item_edits(item_id,overrides) VALUES(%s,JSON_OBJECT())",(item_id,))
    cur.execute('SELECT * FROM kitchen_item_edits WHERE item_id=%s FOR UPDATE',(item_id,));meta=cur.fetchone()
    if kind!='undo' and entry.get('revision')!=meta['revision']:raise Rejected('This item changed. Refresh and review the amount before saving.',409)
    if kind == 'discard' or (kind == 'undo' and entry.get('mode') == 'discard'):
     snapshot = dict(row)
     if kind == 'discard':
      cur.execute("INSERT INTO shared_archive_kitchen SELECT *, NOW(), 'clean_discard', 'shared_kitchen' FROM shared_kitchen WHERE _id=%s ON DUPLICATE KEY UPDATE archived_at=NOW(), archived_reason=VALUES(archived_reason)",(item_id,))
      cur.execute("UPDATE kitchen_item_edits SET overrides=JSON_SET(overrides,'$.__empty_operation',%s) WHERE item_id=%s",(op,item_id))
      cur.execute('DELETE FROM shared_kitchen WHERE _id=%s',(item_id,))
     else:
      if not restoring:raise Rejected('This item has already been restored or changed.',409)
      cur.execute('SHOW COLUMNS FROM shared_kitchen');cols=[c['Field'] for c in cur.fetchall()]
      cur.execute('INSERT INTO shared_kitchen ('+','.join('`'+c+'`' for c in cols)+') VALUES ('+','.join(['%s']*len(cols))+')',[row.get(c) for c in cols])
      cur.execute('DELETE FROM shared_archive_kitchen WHERE _id=%s',(item_id,))
      cur.execute("UPDATE kitchen_item_edits SET overrides=JSON_REMOVE(overrides,'$.__empty_operation'), revision=revision+1 WHERE item_id=%s",(item_id,))
     cur.execute('SELECT revision,reference_amount FROM kitchen_item_edits WHERE item_id=%s',(item_id,));m=cur.fetchone()
     updated=dict(row);updated.update(amount_revision=m['revision'],reference_amount=m['reference_amount'])
     results.append({'item':updated,'removed':kind=='discard'})
     changes.append({'mode':kind,'snapshot':snapshot})
     continue
    before=amount(row['quantity_value']) if row.get('quantity_value') is not None else None
    if kind in ('consume','undo'):
     u=unit(row.get('quantity_unit'))
     if u!=unit(entry.get('unit')) or before is None:raise Rejected('Item amount or unit changed. Review it again.',409)
     if kind=='consume' and amount(entry.get('expected_amount'))!=before:raise Rejected('The available amount changed. Refresh your kitchen.',409)
     delta=amount(entry['amount'])
     if delta<=0:raise Rejected('Choose an amount greater than zero.')
     if kind=='consume' and delta>before:raise Rejected('There is not enough left. Refresh your kitchen.',409)
     left=amount(before+delta if kind=='undo' and not restoring else before if restoring else before-delta)
     reference=meta['reference_amount'] or before
     fields=_quantity_fields(left,u,reference)
    else:
     fields=validate_fields(entry.get('fields'))
     reference=meta['reference_amount']
     if 'quantity_value' in fields or 'quantity_unit' in fields:
      left=amount(fields.get('quantity_value',row.get('quantity_value')))
      if left<=0:raise Rejected('Use all to empty an item.')
      u=unit(fields.get('quantity_unit',row.get('quantity_unit')))
      reference=left if u!=row.get('quantity_unit') or not reference or left>reference else reference
      fields.update(_quantity_fields(left,u,reference))
     if 'category' in fields:
      from category_normalizer import normalize_kitchen_category
      fields['category']=normalize_kitchen_category(fields['category'],fields.get('storage_location',row.get('storage_location')),fields.get('product_name',row.get('product_name')))
    snapshot=dict(row)
    if restoring:
     cur.execute('SHOW COLUMNS FROM shared_kitchen');cols=[c['Field'] for c in cur.fetchall()]
     cur.execute('INSERT INTO shared_kitchen ('+','.join('`'+c+'`' for c in cols)+') VALUES ('+','.join(['%s']*len(cols))+')',[row.get(c) for c in cols])
     cur.execute('DELETE FROM shared_archive_kitchen WHERE _id=%s',(item_id,))
    _save(cur,row,meta,fields,reference)
    cur.execute('SELECT * FROM shared_kitchen WHERE _id=%s',(item_id,));updated=cur.fetchone()
    _verify_saved_fields(updated,fields)
    emptied=updated.get('quantity_value') is not None and amount(updated['quantity_value'])==0
    if emptied:
     cur.execute("INSERT INTO shared_archive_kitchen SELECT *, NOW(), %s, 'shared_kitchen' FROM shared_kitchen WHERE _id=%s ON DUPLICATE KEY UPDATE archived_at=NOW(), archived_reason=VALUES(archived_reason)",('consumed_recipe' if body.get('recipe_title') else 'consumed_manual',item_id))
     cur.execute("UPDATE kitchen_item_edits SET overrides=JSON_SET(overrides,'$.__empty_operation',%s) WHERE item_id=%s",(op,item_id))
     cur.execute('DELETE FROM shared_kitchen WHERE _id=%s',(item_id,))
    cur.execute('SELECT revision,reference_amount FROM kitchen_item_edits WHERE item_id=%s',(item_id,));m=cur.fetchone();updated.update(amount_revision=m['revision'],reference_amount=m['reference_amount'])
    results.append({'item':updated,'removed':emptied})
    changes.append({'mode':'consume' if kind=='consume' else kind,'snapshot':snapshot,'amount':entry.get('amount'),'unit':entry.get('unit'),'recipe_title':body.get('recipe_title')})
   annotate(conn,[entry['item'] for entry in results])
   result={'operation_id':op,'items':results,'can_undo':kind in ('consume','discard')}
   cur.execute('UPDATE kitchen_amount_operations SET result=%s,changes=%s WHERE actor=%s AND operation_id=%s',(dumps(result),dumps(changes),actor,op))
   if undo_source:cur.execute('UPDATE kitchen_amount_operations SET undone_by=%s WHERE actor=%s AND operation_id=%s',(op,actor,undo_source))
  conn.commit();return json.loads(dumps(result))
 except Exception:
  conn.rollback();raise

def apply_explicit_correction(conn, actor, item_id, fields, revision, members):
 """An explicit correction may replace human descriptions, never used amounts.

 Revision is captured before the model call. A concurrent user change wins;
 delayed background enrichment continues through the ordinary protected writer.
 """
 allowed={'product_name','brand','variant','category','product_description','ingredients','nutrition_summary','upf','harmful_ingredients','healthier_alternatives','analysis_stage'}
 if not fields or set(fields)-allowed:raise Rejected('Unsupported correction fields.')
 fields=dict(fields)
 if 'category' in fields:
  from category_normalizer import normalize_kitchen_category
  fields['category']=normalize_kitchen_category(fields['category'],None,fields.get('product_name'))
 conn.begin()
 try:
  with conn.cursor() as cur:
   cur.execute('SELECT * FROM shared_kitchen WHERE _id=%s FOR UPDATE',(item_id,));row=cur.fetchone()
   if not row or row.get('owner_id') not in members or row.get('action')!='IN':raise Rejected('Item is no longer in your household kitchen.',404)
   cur.execute("INSERT IGNORE INTO kitchen_item_edits(item_id,overrides) VALUES(%s,JSON_OBJECT())",(item_id,))
   cur.execute('SELECT * FROM kitchen_item_edits WHERE item_id=%s FOR UPDATE',(item_id,));meta=cur.fetchone()
   if meta['revision']!=revision:raise Rejected('This item changed during correction. Review it and try again.',409)
   saved=json.loads(meta['overrides']) if isinstance(meta['overrides'],str) else dict(meta['overrides'])
   for key,value in fields.items():
    if key in FIELDS:saved[key]=json.loads(value) if FIELDS[key]=='list' and isinstance(value,str) else value
   cur.execute('UPDATE kitchen_item_edits SET overrides=%s WHERE item_id=%s',(dumps(saved),item_id))
   cur.execute('UPDATE shared_kitchen SET '+','.join('`'+k+'`=%s' for k in fields)+', `_updatedDate`=NOW() WHERE _id=%s',[dumps(v) if isinstance(v,list) else v for v in fields.values()]+[item_id])
  conn.commit()
 except Exception:
  conn.rollback();raise

def route(api,owner,body):
 conn=api._mysql_conn()
 try:
  result=execute(conn,owner,body,api._get_household_member_ids(conn,owner))
  api._mark_recipe_refresh_needed_for_owners(conn,api._get_household_member_ids(conn,owner))
  api._refresh_shelf_life_cache(owner)
  return api._success_response(result)
 except Rejected as e:return api._error_response(e.status,e.message)
 except Exception:
  api._report_backend_error('amount_operation',owner_id=owner,code='transaction_failed')
  return api._error_response(503,'Could not confirm the change. Retry this same request.')
