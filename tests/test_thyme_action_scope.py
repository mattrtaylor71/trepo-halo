"""Run the actual serving functions with synthetic SQL rows; never connect to prod."""
import ast, copy, json, os, types, unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'playground6_groceryRec/household_brain/store.py'
class ActionScopeTests(unittest.TestCase):
 def setUp(self):
  self.state={'enabled':1,'input_revision':10,'generation':2};self.home='home';self.members=['actor'];self.hard={'allergies':[]};self.states=[{'actor_id':'actor','generation':3,'revision':4}];self.enabled=True
  outer=self
  class Cursor:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def execute(self,sql,args):
    assert 'LOCK IN SHARE MODE' in sql;assert args==['home']
   def fetchall(self):return outer.states
  self.conn=types.SimpleNamespace(cursor=Cursor)
  self.g={'datetime':datetime,'timezone':timezone,'active':lambda cur:self.enabled,'state_for':lambda *a,**kw:self.state,
    'sources':types.SimpleNamespace(membership=lambda *a,**kw:self.home,members=lambda *a:self.members),
    'digest':lambda x:json.dumps(x,sort_keys=True),'dietary':lambda *a,**kw:self.hard}
  tree=ast.parse(SOURCE.read_text());fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'is_current','is_action_scope_current'}]
  exec(compile(ast.Module(body=fns,type_ignores=[]),str(SOURCE),'exec'),self.g)
  now=datetime.now(timezone.utc).timestamp();self.fence={'schema':'household-brain-v2','actor':'actor','household':'home','audience':'actor','revision':10,'generation':2,'states':[['actor',3,4]],'members_hash':self.g['digest'](self.members),'dietary_hash':self.g['digest'](self.hard),'expires':now+120,'turn_expires':now+500}
 def check(self):return self.g['is_action_scope_current'](self.conn,self.fence)
 def test_own_write_keeps_receipt_scope_but_not_old_answer(self):
  self.assertTrue(self.g['is_current'](self.conn,self.fence));self.state['input_revision']+=1
  self.assertFalse(self.g['is_current'](self.conn,self.fence));self.assertTrue(self.check())
 def test_content_expiry_does_not_erase_receipt(self):
  self.fence['expires']=0;self.assertFalse(self.g['is_current'](self.conn,self.fence));self.assertTrue(self.check())
 def test_actor_move(self):self.home='other';self.assertFalse(self.check())
 def test_member_leaves(self):self.members=[];self.assertFalse(self.check())
 def test_allergy_change(self):self.hard={'allergies':['peanuts']};self.assertFalse(self.check())
 def test_control_change(self):self.states[0]['revision']+=1;self.assertFalse(self.check())
 def test_personal_forget(self):self.states[0]['generation']+=1;self.assertFalse(self.check())
 def test_subject_disabled(self):self.states=[];self.assertFalse(self.check())
 def test_household_reset(self):self.state['generation']+=1;self.assertFalse(self.check())
 def test_service_disabled(self):self.enabled=False;self.assertFalse(self.check())
 def test_household_disabled(self):self.state['enabled']=0;self.assertFalse(self.check())
 def test_expired_turn(self):self.fence['turn_expires']=0;self.assertFalse(self.check())
 def test_unbounded_turn(self):self.fence['turn_expires']+=10000;self.assertFalse(self.check())
 def test_wrong_audience(self):self.fence['audience']='other';self.assertFalse(self.check())

class ApiTests(unittest.TestCase):
 def setUp(self):
  self.scope=True;self.strict=True;self.ticket=None;self.reads=0;self.fence={'actor':'actor','household':'home','turn_expires':500}
  def read(*a,**k):self.reads+=1;return {'fence':dict(self.fence),'prompt':'fresh'}
  store=types.SimpleNamespace(enabled_owner=lambda a:True,read_context=read,is_current=lambda c,f:self.strict)
  self.g={'os':os,'time':__import__('time'),'ticket_actor':lambda t:self.ticket,'store':store}
  source=ROOT/'playground6_groceryRec/family_brain/api.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='handle')
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),self.g)
  module=types.ModuleType('household_brain.store');module.is_action_scope_current=lambda c,f:self.scope
  self.patcher=patch.dict('sys.modules',{'household_brain.store':module});self.patcher.start();self.addCleanup(self.patcher.stop)
  self.env=patch.dict(os.environ,{'HOUSEHOLD_BRAIN_V2':'1'});self.env.start();self.addCleanup(self.env.stop)
 def call(self,op,**extra):return self.g['handle'](object(),{'owner':'actor','authorization':'Bearer fake','operation':op,'fence':self.fence,**extra},lambda t:{'user_id':'actor'})
 def test_validate_does_not_rebuild_context(self):self.assertTrue(self.call('validate')['current']);self.assertEqual(self.reads,0)
 def test_receipt_does_not_rebuild_context(self):self.strict=False;self.assertTrue(self.call('validate_receipt')['current']);self.assertEqual(self.reads,0)
 def test_authenticated_owner_required(self):self.assertEqual(self.call('validate_receipt',owner='other')['statusCode'],403)
 def test_fence_actor_required(self):self.assertFalse(self.call('validate_receipt',fence={'actor':'other'})['current'])
 def test_refresh_retains_original_deadline(self):r=self.call('refresh_turn');self.assertTrue(r['current']);self.assertEqual(r['fence']['turn_expires'],500)
 def test_refresh_scope_denied(self):self.scope=False;self.assertFalse(self.call('refresh_turn')['current']);self.assertEqual(self.reads,0)
 def test_async_validates_current_request_fence_after_own_write(self):self.ticket={'actor':'actor','fence':{}};self.assertTrue(self.call('validate')['current'])
 def test_async_scope_revoked(self):self.ticket={'actor':'actor','fence':{}};self.scope=False;self.assertFalse(self.call('validate')['current'])
 def test_async_cannot_mint_ticket(self):self.ticket={'actor':'actor','fence':{}};self.assertEqual(self.call('authorize')['statusCode'],403)
 def test_stale_general_answer_remains_denied(self):self.strict=False;self.assertFalse(self.call('validate')['current'])

if __name__=='__main__':unittest.main()
