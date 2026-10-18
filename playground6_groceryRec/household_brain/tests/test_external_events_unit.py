import unittest
from unittest.mock import patch
from boto3.dynamodb.types import TypeSerializer
from household_brain import external_queue as queue
S=TypeSerializer();encode=lambda row:{k:S.serialize(v) for k,v in row.items()}
STREAM='arn:verified-native';KEY={'pk':'A#a#H#h','sk':'S#s'}
def event(name='MODIFY',old=None,new=None,sk='S#s'):
 return {'eventName':name,'eventSource':'aws:dynamodb','eventSourceARN':STREAM,'dynamodb':{'Keys':encode({**KEY,'sk':sk}),'OldImage':encode(old or {}),'NewImage':encode(new or {})}}
SESSION={'type':'session','id':'s','actor':'a','household':'h','expires':9999999999,'status':'running','progress':'Checking kitchen','version':2}
MESSAGE={'type':'message','id':'m','role':'user','text':'What should we cook?','order':100,'expires':9999999999}
class NativeHints(unittest.TestCase):
 def test_progress_does_not_retract_memory_while_answering(self):
  for patch in [{'progress':'Putting your answer together'},{'status':'completed'},{'providerId':'provider'},{'activeRequest':'r2'},{'version':3,'updatedAt':'now'}]:
   self.assertEqual(queue.hints(event(old=SESSION,new={**SESSION,**patch}),'chat_native'),[])
 def test_deleting_session_still_retracts(self):
  self.assertEqual(queue.hints(event('REMOVE',old=SESSION),'chat_native'),[('a','h',None)])
 def test_session_identity_or_retention_change_still_retracts(self):
  for patch in [{'actor':'b'},{'household':'other'},{'expires':1}]:self.assertEqual(len(queue.hints(event(old=SESSION,new={**SESSION,**patch}),'chat_native')),1)
 def test_real_user_message_correction_and_deletion_still_retract(self):
  for record in [event(old=MESSAGE,new={**MESSAGE,'text':'No peanuts'},sk='S#s#M#m'),event('REMOVE',old=MESSAGE,sk='S#s#M#m')]:self.assertEqual(len(queue.hints(record,'chat_native')),1)
 def test_new_user_message_schedules_non_retracting_refresh(self):
  record=event('INSERT',new=MESSAGE,sk='S#s#M#m')
  with patch.object(queue,'enqueue',return_value=True) as enqueue:
   self.assertEqual(queue.stream(None,{'Records':[record]},{STREAM:'chat_native'}),{'checked':1});self.assertFalse(enqueue.call_args.kwargs['retract'])
 def test_assistant_answer_and_recipe_proposal_records_are_not_user_history(self):
  self.assertEqual(queue.hints(event('INSERT',new={**MESSAGE,'role':'assistant'},sk='S#s#M#m'),'chat_native'),[])
  for sk in ['S#s#P#p','S#s#R#r','Q#request','S#s#C#settings']:self.assertEqual(queue.hints(event(new=MESSAGE,sk=sk),'chat_native'),[])
 def test_message_version_only_change_is_not_an_edit(self):
  self.assertEqual(queue.hints(event(old=MESSAGE,new={**MESSAGE,'version':2},sk='S#s#M#m'),'chat_native'),[])
 def test_role_change_away_from_user_retracts_prior_evidence(self):
  self.assertEqual(len(queue.hints(event(old=MESSAGE,new={**MESSAGE,'role':'assistant'},sk='S#s#M#m'),'chat_native')),1)
 def test_unknown_delete_and_missing_old_image_fail_conservatively(self):
  self.assertEqual(len(queue.hints(event('REMOVE',sk='S#s#M#m'),'chat_native')),1)
  self.assertEqual(len(queue.hints(event(new=SESSION),'chat_native')),1)
 def test_mixed_batch_retracts_only_for_real_edit(self):
  progress=event(old=SESSION,new={**SESSION,'progress':'New progress'})
  insert=event('INSERT',new=MESSAGE,sk='S#s#M#m')
  with patch.object(queue,'enqueue',return_value=True) as enqueue:
   queue.stream(None,{'Records':[progress,insert]},{STREAM:'chat_native'});enqueue.assert_called_once();self.assertFalse(enqueue.call_args.kwargs['retract'])
   enqueue.reset_mock();queue.stream(None,{'Records':[progress,insert,event(old=MESSAGE,new={**MESSAGE,'text':'Corrected'},sk='S#s#M#m')]},{STREAM:'chat_native'});enqueue.assert_called_once();self.assertTrue(enqueue.call_args.kwargs['retract'])
if __name__=='__main__':unittest.main()
