import importlib.util,json
from pathlib import Path
from unittest.mock import Mock
import pytest
spec=importlib.util.spec_from_file_location('ux',Path(__file__).resolve().parents[1]/'app.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_meal_failures_deduplicate_households_and_count_missing_plan(monkeypatch):
    monkeypatch.setattr(m,'_q',lambda *a:[
        {'owner_id':'a','household':'home','plan':'[]'},
        {'owner_id':'b','household':'home','plan':[{'recipe':'x'}]},
        {'owner_id':'c','household':'other','plan':None},
        {'owner_id':'test-x','household':'fixture','plan':None}])
    n,rows=m.check_meal_plan_failures(None)
    assert n==2 and sum(r['without_plan'] for r in rows)==1

def test_failed_import_attempts_do_not_multiply_records(monkeypatch):
    monkeypatch.setattr(m,'_q',lambda *a:[{'owner_id':'a','recipe_id':'x'},{'owner_id':'a','recipe_id':'x'},{'owner_id':'b','recipe_id':'x'}])
    assert m.check_terminal_imports(None)[0]==2

def test_partial_actions_come_from_structured_receipt(monkeypatch):
    monkeypatch.setenv('VOICE_OPERATION_TABLE','fixture')
    monkeypatch.setattr(m,'_scan_all',lambda *a,**kw:[
        {'operation_id':'one','response_body':{'text':'Done!','outcome':{'actions':'partial'}}},
        {'operation_id':'two','response_body':{'text':'Sorry','outcome':{'actions':'confirmed'}}}])
    assert m.check_partial_actions()==(1,[{'operation_id':'one','actions':'partial'}])

def test_unsupported_receipts_report_coverage_failure(monkeypatch):
    monkeypatch.setenv('VOICE_OPERATION_TABLE','fixture');monkeypatch.setattr(m,'_scan_all',lambda *a,**kw:[{'response_body':{'text':'Done!'}}])
    with pytest.raises(ValueError):m.check_partial_actions()

def test_failed_reads_never_emit_false_zero_and_coverage_alarm_is_explicit(monkeypatch):
    monkeypatch.setattr(m,'_conn',Mock(side_effect=TimeoutError))
    monkeypatch.setattr(m,'check_capture_not_analyzed',lambda:(0,[]))
    monkeypatch.setattr(m,'check_recipes_not_analyzed',lambda:(1,[]))
    monkeypatch.setattr(m,'check_partial_actions',Mock(side_effect=PermissionError))
    monkeypatch.setattr(m,'check_capacity_rejections',lambda:0)
    monkeypatch.setattr(m,'check_halo_voice_outcomes',lambda:(0,[]))
    monkeypatch.setattr(m,'check_missing_halo_media',lambda:(0,[]))
    client=Mock();monkeypatch.setattr(m.boto3,'client',lambda _:client)
    result=m.handler({},None)
    assert result['check_errors']==6 and result['metrics']['CoverageComplete']==0
    assert 'MealPlanFailures' not in result['metrics'] and 'PartialActions' not in result['metrics']
    assert result['metrics']['CaptureNotAnalyzed']==0

def test_metric_delivery_failure_cannot_be_reported_healthy(monkeypatch):
    monkeypatch.setattr(m,'_conn',Mock(side_effect=TimeoutError))
    for name in ('check_capture_not_analyzed','check_recipes_not_analyzed','check_partial_actions','check_capacity_rejections','check_halo_voice_outcomes','check_missing_halo_media'):
        monkeypatch.setattr(m,name,lambda:0)
    client=Mock();client.put_metric_data.side_effect=TimeoutError
    monkeypatch.setattr(m.boto3,'client',lambda _:client)
    with pytest.raises(TimeoutError):m.handler({},None)


def test_halo_partial_success_uses_saved_tool_outcome(monkeypatch):
    monkeypatch.setattr(m,'_scan_all',lambda *a,**kw:[{'job_id':'one','status':'completed','response_text':'Done!',
        'outcome':{'actions':'partial'}}])
    assert m.check_halo_voice_outcomes()[0]==1

def test_missing_media_and_late_arrival_are_reconciled_without_writes(monkeypatch):
    from botocore.exceptions import ClientError
    monkeypatch.setenv('HALO_MEDIA_BUCKETS','["images","discards"]')
    monkeypatch.setattr(m,'_scan_all',lambda *a,**kw:[{'job_id':'one','device_id':'halo-one','owner':'a','type':'grocery',
        'created_at':m._iso_ago(hours=1),'s3_key':'images/a/halo-one/one.jpg'}])
    client=Mock();client.head_object.side_effect=ClientError({'Error':{'Code':'404'}},'HeadObject')
    monkeypatch.setattr(m.boto3,'client',lambda _:client)
    assert m.check_missing_halo_media()[0]==1
    client.head_object.side_effect=None;client.head_object.return_value={'ContentLength':2000}
    assert m.check_missing_halo_media()[0]==0
    client.head_object.side_effect=ClientError({'Error':{'Code':'403'}},'HeadObject')
    with pytest.raises(ClientError):m.check_missing_halo_media()
