import sys, unittest, json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amount_operations as amounts
import shelf_dates
class Cursor:
 def __init__(self):self.calls=[];self.connection=self
 def execute(self,sql,args=()):self.calls.append((sql,args))
 def fetchone(self):return {'overrides':'{}','_id':'milk','owner_id':'member','product_name':'Milk'}
 def fetchall(self):return [{'item_id':'milk','revision':3,'reference_amount':1,'overrides':json.dumps({'__acquired_on':None})}]
 def cursor(self,*a):return self
 def __enter__(self):return self
 def __exit__(self,*a):pass
class BoughtDateTests(unittest.TestCase):
 def test_empty_edit_requires_date_contract(self):
  with self.assertRaises(amounts.Rejected):amounts.validate_fields({})
  self.assertEqual(amounts.validate_fields({},allow_empty=True),{})
  with self.assertRaises(amounts.Rejected):amounts.validate_fields({'_createdDate':'2026-10-19'},allow_empty=True)
 def test_date_only_update_has_valid_sql_and_preserves_fields(self):
  cur=Cursor()
  with patch('shelf_lifecycle_store.enabled_for',return_value=False):amounts._save(cur,{'_id':'milk','owner_id':'member'},{'overrides':'{}','reference_amount':1},{})
  updates=[c for c in cur.calls if c[0].startswith('UPDATE shared_kitchen')]
  self.assertEqual(updates,[('UPDATE shared_kitchen SET `_updatedDate`=NOW() WHERE _id=%s',['milk'])])
 def test_bought_only_available_outside_pilot(self):
  cur=Cursor()
  with patch.object(shelf_dates,'enabled_for',return_value=False),patch.object(shelf_dates,'materialize',return_value={'milk':{'time_zone':'America/Los_Angeles'}}):
   shelf_dates.apply(cur,cur,'member','request',{'_id':'milk','owner_id':'member'},{'acquired_on':'2026-10-19'})
   saved=json.loads(next(args[0] for sql,args in cur.calls if 'SET overrides=' in sql))
   self.assertEqual(saved['__acquired_on'],'2026-10-19')
   self.assertTrue(any('kitchen_shelf_events' in s for s,a in cur.calls))
   self.assertTrue(any('needs_work=TRUE' in s for s,a in cur.calls))
 def test_other_clocks_still_require_pilot(self):
  with patch.object(shelf_dates,'enabled_for',return_value=False):
   for values in [{'opened_on':'2026-10-19'},{'acquired_on':'2026-10-19','frozen_on':'2026-10-19'}]:
    with self.assertRaises(amounts.Rejected):shelf_dates.apply(None,None,'member','request',{},values)
 def test_bad_dates_rejected_without_writes(self):
  for value in ['2026-02-30','2999-01-01',1,True,'10/19/2026']:
   cur=Cursor()
   with patch.object(shelf_dates,'materialize',return_value={'milk':{'time_zone':'America/Los_Angeles'}}):
    with self.assertRaises(amounts.Rejected):shelf_dates.apply(cur,cur,'member','request',{'_id':'milk'},{'acquired_on':value})
   self.assertEqual(cur.calls,[])
 def test_readback_preserves_explicit_unknown(self):
  rows=amounts.annotate(Cursor(),[{'_id':'milk','_createdDate':'2026-10-26'}])
  self.assertIn('acquired_on',rows[0]);self.assertIsNone(rows[0]['acquired_on'])
if __name__=='__main__':unittest.main()
