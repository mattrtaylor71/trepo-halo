"""Actual handler POST -> stored event/latest -> analytics GET regressions."""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
os.environ.update(AWS_EC2_METADATA_DISABLED='true',AWS_ACCESS_KEY_ID='offline-test',AWS_SECRET_ACCESS_KEY='offline-test',AWS_DEFAULT_REGION='us-east-1',OTA_REPORT_EVENTS_TABLE='events',OTA_DEVICE_LATEST_TABLE='latest')
import app
import analytics as a
from botocore.exceptions import ClientError
NOW=1789100000
SECRET='offline-test-key-never-deployed-123456'
BASE=dict(device_id='halo-raw-numeric-qa',device_type='sense',fw='6.4.117',build='qa-offline',report_type='health',ts_epoch=NOW)

class Table:
    def __init__(self):self.items={};self.puts=0;self.updates=0
    def key(self,item):return (item['device_id'],item.get('event_ts_key'))
    def put_item(self,Item):self.items[self.key(Item)]=copy.deepcopy(Item);self.puts+=1;return {}
    def update_item(self,**kw):
        self.updates+=1;key=self.key(kw['Key']);item=self.items.setdefault(key,dict(kw['Key']));values=kw['ExpressionAttributeValues'];names=kw.get('ExpressionAttributeNames',{})
        field,value=('last_seen_epoch',values.get(':last_seen_epoch')) if ':last_seen_epoch' in values else ('analytics_received_at_epoch',values[':received'])
        if item.get(field,-1)>value:raise ClientError({'Error':{'Code':'ConditionalCheckFailedException'}},'UpdateItem')
        expression=kw['UpdateExpression'];assert expression.startswith('SET ') and 'REMOVE ' not in expression
        for assignment in expression[4:].split(','):
            target,source=assignment.strip().split(' = ');item[names.get(target,target)]=copy.deepcopy(values[source])
        return {}
    def get_item(self,Key,**kw):
        item=self.items.get(self.key(Key));return {'Item':copy.deepcopy(item)} if item else {}
    def query(self,**kw):return {'Items':copy.deepcopy(list(self.items.values())),'Count':len(self.items),'ScannedCount':len(self.items)}


def diagnostic():
    context=bytearray(256);context[:8]=b'HDG1'+bytes([3,1,0,0]);context[8:16]=(1).to_bytes(8,'little');context[20:22]=(212).to_bytes(2,'little')
    context[24:40]=bytes(range(1,17));context[40:56]=b'c'*16;context[56:88]=b'h'*32
    context[88:95]=b'6.4.117';context[120:130]=b'qa-offline';context[184:190]=bytes([1,2,3,4,5,6]);context[190:192]=bytes([1,1]);context[192:196]=(2621440).to_bytes(4,'little')
    context_crc=zlib.crc32(context[:252])&0xffffffff;context[252:]=context_crc.to_bytes(4,'little')
    record=bytearray(256);record[:8]=b'HDG1'+bytes([3,2,1,0]);record[8:16]=(2).to_bytes(8,'little');record[16:20]=context_crc.to_bytes(4,'little');record[20:22]=(228).to_bytes(2,'little')
    record[24:40]=b'c'*16;record[40:48]=(7).to_bytes(8,'little');record[48:56]=NOW.to_bytes(8,'little');record[60:62]=(1).to_bytes(2,'little');record[72:76]=(2621440).to_bytes(4,'little')
    record[96:106]=b'qa-offline';record[240:247]=b'6.4.117'
    crc=zlib.crc32(record[:252])&0xffffffff;record[252:]=crc.to_bytes(4,'little')
    return dict(BASE,device_id='halo-0304-0506',report_type='heartbeat',diag_export=1,diag_schema=3,diag_slot=1,diag_sequence=2,diag_crc=crc,diag_context_b64=base64.b64encode(context).decode(),diag_record_b64=base64.b64encode(record).decode(),request_id='d3-'+context[24:40].hex()+'-0000000000000002-'+format(crc,'08x'),boot_count=True)

class RawNumericHandlerTests(unittest.TestCase):
    def setUp(self):
        self.events=Table();self.latest=Table();self.env=patch.dict(os.environ,ANALYTICS_READ_KEY=SECRET);self.env.start()
        self.clock=patch.object(app.time,'time',return_value=NOW);self.clock.start();self.log=patch.object(app,'_log');self.log.start()
        self.tables=patch.object(app.dynamodb,'Table',side_effect=lambda name:self.events if name=='events' else self.latest);self.tables.start()
    def tearDown(self):self.tables.stop();self.log.stop();self.clock.stop();self.env.stop()
    def invoke(self,method,path,body=None,query=None):
        event={'requestContext':{'http':{'method':method,'path':path}},'headers':{'X-Analytics-Key':SECRET}}
        if body is not None:event['body']=json.dumps(body)
        if query is not None:event['queryStringParameters']=query
        out=app.handler(event,None);return out['statusCode'],json.loads(out['body'])
    def post(self,**changes):return self.invoke('POST','/ota/report',dict(BASE,**changes))
    def view(self,kind='latest',device='halo-raw-numeric-qa'):
        status,out=self.invoke('GET','/ota/report/'+kind,query={'view':'analytics','device_id':device});self.assertEqual(status,200);return out['items']
    def test_clock_terminal_reasons_survive_real_handler_latest_and_history(self):
        for reason in ('clock_unconfirmed', 'clock_unconfirmed_defer'):
            with self.subTest(reason=reason):
                self.events.items.clear();self.latest.items.clear()
                status,_=self.post(fw='6.4.127',build='6.4.127-clock-regression',report_type='pre_sleep',ota_result=reason)
                self.assertEqual(status,200)
                for kind in ('latest','events'):
                    self.assertEqual(self.view(kind)[0]['metrics']['ota']['result'],reason)
    def test_unknown_clock_terminal_text_remains_outside_analytics_allowlist(self):
        self.assertEqual(self.post(ota_result='clock_unconfirmed-private-sentinel')[0],200)
        for kind in ('latest','events'):
            row=self.view(kind)[0]
            self.assertNotIn('result',row['metrics'].get('ota',{}))
            self.assertNotIn('private-sentinel',json.dumps(row))
    def test_actual_handler_omits_bool_counter_and_fractional_values_both_views(self):
        status,_=self.post(boot_count=True,uptime_ms=1.5,rssi=-61.75);self.assertEqual(status,200)
        stored=next(iter(self.events.items.values()));self.assertEqual(stored['payload']['boot_count'],1);self.assertEqual(stored['payload']['uptime_ms'],1);self.assertEqual(stored['payload']['rssi'],-61)
        self.assertEqual(self.view()[0]['metrics'],{});self.assertEqual(self.view('events')[0]['metrics'],{})
        self.assertEqual(stored['analytics_raw_numeric_rejections'],['boot_count','rssi','uptime_ms'])
    def test_valid_bool_flags_and_integer_zero_still_export(self):
        self.assertEqual(self.post(wifi=True,maintenance_mode=True,boot_count=0,uptime_ms=0,rssi=-61)[0],200)
        for kind in ('latest','events'):
            metrics=self.view(kind)[0]['metrics'];self.assertEqual(metrics['system']['boot_count'],0);self.assertEqual(metrics['network']['wifi_connected'],1);self.assertEqual(metrics['sleep']['maintenance_mode'],1)
    def test_valid_numeric_strings_and_wifi_enum_are_preserved(self):
        self.post(boot_count=' 42 ',rssi='-61',wifi='connected')
        for kind in ('latest','events'):
            metrics=self.view(kind)[0]['metrics'];self.assertEqual(metrics['system']['boot_count'],42);self.assertEqual(metrics['network']['rssi_dbm'],-61);self.assertEqual(metrics['network']['wifi_state'],'connected')
    def test_invalid_raw_timestamp_remains_unknown_in_analytics_only(self):
        for stamp in (True,1.5):
            self.events.items.clear();self.latest.items.clear();self.assertEqual(self.post(ts_epoch=stamp)[0],200)
            for kind in ('latest','events'):
                row=self.view(kind)[0];self.assertIsNone(row['device_reported_at_epoch']);self.assertEqual(row['clock_status'],'unknown');self.assertEqual(row['age_basis'],'server_received_at')
            self.assertEqual(next(iter(self.events.items.values()))['payload']['ts_epoch'],1)
    def test_negative_out_of_range_and_invalid_numeric_text_are_omitted(self):
        self.post(boot_count=-1,rssi=-999,uptime_ms='private-sentinel')
        for kind in ('latest','events'):
            row=self.view(kind)[0];self.assertEqual(row['metrics'],{});self.assertNotIn('private-sentinel',json.dumps(row))
    def test_client_cannot_forge_top_level_rejections(self):
        self.post(boot_count=7,analytics_raw_numeric_rejections=['boot_count','device_id'])
        self.assertEqual(next(iter(self.events.items.values()))['analytics_raw_numeric_rejections'],[])
        self.assertEqual(self.view()[0]['metrics']['system']['boot_count'],7)
    def test_latest_snapshot_does_not_inherit_old_legacy_mask_after_clock_regression(self):
        self.post(ts_epoch=NOW+100,boot_count=True)
        status,out=self.post(ts_epoch=NOW-100,boot_count=7);self.assertEqual(status,200);self.assertFalse(out['latest_updated']);self.assertTrue(out['analytics_latest_updated'])
        legacy=next(iter(self.latest.items.values()));self.assertEqual(legacy['analytics_raw_numeric_rejections'],['boot_count'])
        self.assertEqual(self.view()[0]['metrics']['system']['boot_count'],7)
    def test_failed_optional_snapshot_uses_safe_legacy_fallback_mask(self):
        with patch.object(a,'update_latest',side_effect=RuntimeError('offline failure')):
            status,out=self.post(boot_count=True,rssi=-61.2)
        self.assertEqual(status,200);self.assertFalse(out['analytics_latest_updated']);self.assertEqual(self.view()[0]['metrics'],{})
    def test_real_crc_valid_d3_handler_preserves_retained_source_and_latest_exclusion(self):
        payload=diagnostic();self.assertTrue(app._validated_diagnostic_export(payload))
        status,out=self.invoke('POST','/ota/report',payload);self.assertEqual(status,200);self.assertFalse(out['latest_updated']);self.assertEqual(self.latest.updates,0)
        row=self.view('events',payload['device_id'])[0];self.assertEqual(row['source'],'retained_diagnostic');self.assertNotIn('boot_count',row['metrics'].get('system',{}))
    def test_bad_d3_crc_is_still_rejected_before_write(self):
        payload=diagnostic();raw=bytearray(base64.b64decode(payload['diag_record_b64']));raw[30]^=1;payload['diag_record_b64']=base64.b64encode(raw).decode()
        self.assertEqual(self.invoke('POST','/ota/report',payload)[0],400);self.assertEqual(self.events.puts,0);self.assertEqual(self.latest.updates,0)
    def test_historical_rows_are_not_speculatively_rewritten(self):
        historical=dict(device_id=BASE['device_id'],event_ts_key='legacy',payload=dict(BASE,boot_count=1),ingested_at_epoch=NOW)
        self.assertEqual(a.from_item(historical)['metrics']['system']['boot_count'],1)
    def test_rejection_metadata_cannot_remove_identity_or_report_strings(self):
        row=a.from_item(dict(device_id=BASE['device_id'],event_ts_key='x',payload=dict(BASE,boot_count=3),ingested_at_epoch=NOW,analytics_raw_numeric_rejections=['device_id','fw',{'boot_count':True}]))
        self.assertEqual(row['device_id'],BASE['device_id']);self.assertEqual(row['fw'],BASE['fw']);self.assertEqual(row['metrics']['system']['boot_count'],3)
    def test_legacy_normalizer_and_diagnostic_implementation_ast_unchanged(self):
        expected={'_normalize_payload': '0c3d79d17f23642d7fea43fc3d278ad3b98c143477bfe5a80fee276f13a7f5be', '_validate_payload': '5d914485bd95612b12d5ea77c81389016512f221f273768b1f89a1e4034111fc', '_safe_int': '757f40b42509d644c1b9d99bff1512b399b5543b1e79864195e46a70d52c4538', '_build_latest_item': '751192e5cfbf7ea2495bd90fcadce06e5d1e7b660e9b934fbae53b5a3b0fc35d', '_update_latest_if_newer': 'ec6057427714813b50a97723cda622ed616b0883a7e5475ee92b0a1f94423643', '_get_latest_state': '1ed93f3ff500c53a681b038fcd14233b2bd7da226ffebe08d50498bc7f04efa6', '_get_events': 'e5367ffd0e1447850c0492b5a22db30bce45161a27bfc03ffe5e174745c46102', '_diagnostic_blob': 'e2ea46843a24bc674b47583fc2713c8d68ff6a3e081ee827b90bb97317341855', '_diagnostic_text': 'dfc411994128df2abcca4a71125d3093d8bd3f0bff300ac42688cdb618292302', '_validated_diagnostic_export': 'cee12517d956fb1decc76f13ab5b1a0014cd7c11b630b79752aabd70908173b6'}
        current=ast.parse((ROOT/"app.py").read_text())
        for name,digest in expected.items():
            node=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(hashlib.sha256(ast.dump(node).encode()).hexdigest(),digest,name)

if __name__=='__main__':unittest.main()
