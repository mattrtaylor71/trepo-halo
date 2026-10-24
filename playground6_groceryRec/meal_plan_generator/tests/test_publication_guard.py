"""Model-boundary regressions: harmless activity vs actual food-input changes."""
import importlib.util
from pathlib import Path
import sys,types,unittest,json,hashlib
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
# Isolated pure tests don't require cloud credentials or optional app dependencies.
brain=types.ModuleType('family_brain.store');brain.require_publishable=Mock();brain.is_current=Mock(return_value=False)
fam=types.ModuleType('family_brain');fam.store=brain
core=types.ModuleType('household_brain.core');core.digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();core.obj=lambda x:x;core.stamp=lambda x:x
saved_modules=sys.modules.copy()
sys.modules.update({'family_brain':fam,'family_brain.store':brain,'household_brain':types.ModuleType('household_brain'),'household_brain.core':core})
spec=importlib.util.spec_from_file_location('guard',ROOT/'recipe_generation_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
for name in ('family_brain','family_brain.store','household_brain','household_brain.core'):
 if name in saved_modules:sys.modules[name]=saved_modules[name]
 else:sys.modules.pop(name,None)
class GuardTests(unittest.TestCase):
 def setUp(self):
  self.conn=Mock();self.data=dict(household='home',members=['a','b'],dietary=core.digest(guard.normalize({})),versions={'a':1,'b':1},inventory='stock',controls='explicit');brain.require_publishable.reset_mock();brain.is_current.reset_mock()
 def capture(self):
  with patch.object(guard,'_dependencies',return_value=self.data.copy()):return guard.capture(self.conn,owner='a',recipients=['a','b'],dietary={},memory={'fence':{'schema':'household-brain-v2','audience':None,'revision':1,'expires':1}},inventory_table='shared_kitchen')
 def test_activity_revision_and_cache_ttl_do_not_cancel_completed_plan(self):
  snapshot=self.capture()
  with patch.object(guard,'_dependencies',return_value=self.data.copy()):
   with guard.publication(self.conn,snapshot=snapshot):pass
  brain.require_publishable.assert_called_once();brain.is_current.assert_not_called();self.conn.commit.assert_called()
 def test_each_actual_dependency_change_rolls_back_without_publication(self):
  snapshot=self.capture()
  for field,value in [('household','other'),('members',['a']),('dietary','new'),('versions',{'a':2,'b':1}),('inventory','different'),('controls','paused')]:
   with self.subTest(field=field),patch.object(guard,'_dependencies',return_value={**self.data,field:value}):
    with self.assertRaises(guard.GenerationInputsChanged):
     with guard.publication(self.conn,snapshot=snapshot):self.fail('must not write')
  self.conn.rollback.assert_called()
 def test_expired_job_is_rejected(self):
  snapshot=self.capture();snapshot['deadline']=0
  with self.assertRaisesRegex(guard.GenerationInputsChanged,'deadline'):guard.require_current(self.conn,snapshot)
 def test_personal_memory_is_rejected_for_shared_plan(self):
  with self.assertRaisesRegex(guard.GenerationInputsChanged,'personal_context'):
   guard.capture(self.conn,owner='a',recipients=['a','b'],dietary={},memory={'fence':{'audience':'a'}})
 def test_initial_freshness_is_still_required(self):
  with patch.object(brain,'require_publishable',side_effect=ValueError('Food preferences changed during generation')):
   with self.assertRaisesRegex(guard.GenerationInputsChanged,'inputs_changed_before'):self.capture()
 def test_generator_uses_guard_for_publish_and_retains_dietary_recipe_check(self):
  # Prevent reverting to the global activity/TTL fence in this consumer.
  source=(ROOT/'app.py').read_text();self.assertNotIn('brain_store.is_current',source);self.assertIn('recipe_generation_guard.capture',source);self.assertIn('recipe_generation_guard.publication(conn, snapshot=snapshot)',source);self.assertIn('_recipe_violates(recipe,prefs or {})',source)
class SemanticInventoryTests(unittest.TestCase):
 def digest(self,stage='fast',name='Milk',quantity=1):
  cur=Mock()
  row={'_id':'one','owner_id':'a','product_name':name,'action':'IN','analysis_stage':stage,'analysis_status':'ready','quantity':quantity}
  cur.fetchall.side_effect=[[{'COLUMN_NAME':key} for key in row],[row]]
  return guard._inventory(cur,'shared_kitchen',['a'])
 def test_cosmetic_stage_change_preserves_hash(self):
  self.assertEqual(self.digest('fast'),self.digest('final'))
 def test_actual_name_and_quantity_still_change_hash(self):
  self.assertNotEqual(self.digest(),self.digest(name='Almond milk'))
  self.assertNotEqual(self.digest(),self.digest(quantity=2))

if __name__=='__main__':unittest.main()
