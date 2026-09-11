"""Safe analytics projection of existing reports. No device control or media fields.

Legacy payload storage/query behavior is owned by app.py. Analytics exposes only
this explicit schema, and records received time independently of the device clock.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import time
from decimal import Decimal

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

SCHEMA = 1
REPORT_TYPES = frozenset('boot maintenance_enter ota_result pre_sleep manifest_ok manifest_err maintenance_done lcd_ota_result heartbeat health action_result'.split())
IDENTIFIER = re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z')
VERSION = re.compile(r'[A-Za-z0-9_.+-]{1,96}\Z')
# group/output -> (wire names, minimum, maximum). Unknown values stay absent.
NUMBERS = {}
def fields(group, names, maximum=4294967295):
    for name in names.split():
        NUMBERS[(group, name)] = ((name,), 0, maximum)
def aliases(group, name, wire, minimum=0, maximum=4294967295):
    NUMBERS[(group, name)] = (tuple(wire.split()), minimum, maximum)

aliases('network', 'rssi_dbm', 'rssi rssi_dbm', -127, 0)
aliases('network', 'wifi_connected', 'wifi wifi_connected', 0, 1)
aliases('network', 'disconnect_count', 'wifi_disconnects disconnect_count')
aliases('network', 'http_status', 'sched_fetch_http http_status', -1000, 599)
fields('network', 'sched_fetch_age_s sched_last_event_age_s idle_net_left_ms')
aliases('memory', 'free_heap_bytes', 'heap_free heap_free_bytes free_heap_bytes', 0, 1073741824)
aliases('memory', 'min_free_heap_bytes', 'heap_min heap_min_free heap_min_bytes min_free_heap_bytes', 0, 1073741824)
aliases('memory', 'largest_free_block_bytes', 'heap_largest heap_largest_free largest_free_block_bytes', 0, 1073741824)
aliases('memory', 'psram_free_bytes', 'psram_free psram_free_bytes', 0, 1073741824)
fields('memory', 'loop_stack_min_free_bytes loop_stack_sample_ms upload_stack_min_free_bytes op_stack_min_free_bytes uart_stack_min_free_bytes')
fields('system', 'uptime_ms boot_count wake_cause last_action_age_ms')
aliases('system', 'reboot_loop_detected', 'reboot_loop_detected', 0, 1)
fields('link', 'uart_rx_count uart_tx_count uart_crc_errors uart_frame_errors uart_ack_retries uart_ack_failures uart_dropped_frames lcd_diag_age_ms')
fields('link', 'lcd_maint_ack_age_s lcd_maint_ack_remaining_s lcd_maint_ack_wake_in_s lcd_ota_window_remaining_s')
for name in 'lcd_maint_ack lcd_maint_ack_persisted lcd_ota_done lcd_ota_request_active'.split(): aliases('link', name, name, 0, 1)
fields('sleep', 'last_sleep_timer_selected_epoch last_sleep_timer_selected_s maint_wake_epoch maint_start_epoch maint_end_epoch maint_last_idle_ms maint_min_idle_min maint_last_tx_age_s maint_last_tx_remaining_s maint_last_tx_wake_in_s maint_followup_retry_attempts maint_sync_attempts')
for name in 'maint_scheduled maintenance_in_window maintenance_mode maint_sync_pending maint_followup_retry_active maint_last_idle_gate_bypassed maint_last_idle_gate_hit maint_last_idle_http_window maint_last_tx_clear maint_last_tx_link_recent'.split(): aliases('sleep', name, name, 0, 1)
for name, wire in {'phase':'ota_policy_phase','network_windows':'ota_policy_network_windows','apply_attempts':'ota_policy_apply_attempts','reserved_work_ms':'ota_policy_reserved_ms','work_remaining_ms':'ota_policy_work_remaining_ms'}.items(): aliases('ota', name, wire)
fields('ota', 'ota_policy_storage ota_policy_generation ota_policy_day ota_policy_attempt_ordinal ota_policy_lcd_begins ota_policy_sense_begins ota_policy_sense_boot ota_policy_target_bytes ota_policy_next_normal ota_policy_retry_due ota_policy_retry_expiry ota_ready_due ota_ready_epoch ota_ready_left_ms ota_ready_ms')
for name in 'ota_clock_fresh ota_ready_fresh ota_ready_origin'.split(): aliases('ota', name, name, 0, 1)
fields('ota', 'ota_transfer_bytes ota_transfer_total_bytes ota_transfer_duration_ms ota_transfer_retries ota_transfer_stalls')
fields('diagnostics', 'diag_s_qualification diag_l_qualification diag_export_result diag_export_slot diag_export_crc diag_schema diag_discovery_schema diag_export_left_ms diag_l_qualification_boot diag_l_qualification_ms diag_queue_depth diag_queue_dropped diag_queue_acked diag_queue_pending diag_export_attempts diag_export_successes diag_export_failures')
fields('diagnostics', 'diag_export_seq', 18446744073709551615)
aliases('diagnostics', 'diag_export_http', 'diag_export_http', -1000, 599)
for name in 'diag_export_handoff diag_export_attempted diag_s_ready diag_s_uncertain'.split(): aliases('diagnostics', name, name, 0, 1)
fields('action', 'action_boot_id list_attempts list_successes list_failures list_last_duration_ms capture_attempts capture_successes capture_failures voice_attempts voice_successes voice_failures')
aliases('action', 'duration_ms', 'action_duration_ms')
aliases('action', 'count', 'action_count')
aliases('action', 'failure_count', 'action_failure_count action_failures')

HEAP_POINTS = ('before_dma','after_dma','after_mqtt','after_sense_manifest','after_lcd_manifest','after_lcd_scope','before_apply')
UART_TYPES = 'FW_INFO IMG_XFER_BEGIN INPUT_ACK INPUT_DELETE INPUT_DISCARD_OPTIONS INPUT_EXPIRY_DATE INPUT_FW_INFO INPUT_LONG_PRESS_END INPUT_LONG_PRESS_START INPUT_MAINT_TEST INPUT_MENU_PRESS INPUT_MENU_SELECT INPUT_OTA_CHECK INPUT_PING INPUT_RESET_WIFI INPUT_RETRY INPUT_SCROLL INPUT_SENSE_FW INPUT_SLEEP INPUT_TEST_ERRORS INPUT_TOUCH INPUT_WAKE INPUT_WIFI_SCAN INPUT_WIFI_TEST LCD_DIAG LCD_OTA_ABORT LCD_OTA_BEGIN LCD_OTA_BEGIN_ACK LCD_OTA_DONE LCD_OTA_END_ACK LCD_OTA_QUERY LCD_OTA_QUERY_RESP LCD_OTA_STATUS LINK_HB LIST_ACTIVE MAINT_KEEPALIVE MAINT_WINDOW MAINT_WINDOW_ACK OTA_BENCH OTA_CHECK OTA_CHECK_ACK OTA_CHECK_RESULT OTA_DIAG OTA_DIAG_AUTH OTA_DIAG_AUTH_ACK OTA_IDLE_NET_ACK OTA_ONE_SHOT OTA_PEER_READY OTA_UNLOCK PONG PROVISION_QR PROVISION_STATUS RELEASE_WAKE SENSE_DIAG SENSE_SLEEP_INTENT SLEEP_DENY SLEEP_READY SPOOL_DELETE SPOOL_FETCH SPOOL_FETCH_READY SPOOL_LIST SPOOL_LIST_REQ SYNC SYNC_ACK UI_LIST UI_STATUS UI_TOAST UI_VOICE_RESPONSE WIFI_CREDS WIFI_CREDS_ACK WIFI_DIAG_SUMMARY WIFI_ON WIFI_ON_ACK WIFI_STATUS'
fields('memory','ota_heap_schema ota_heap_mask ota_fail_heap ota_fail_largest ota_fail_minimum')
fields('ota','ota_fail_offset ota_fail_expected')
aliases('ota','ota_fail_sdk','ota_fail_sdk',-2147483648,2147483647)
aliases('ota','ota_lcd_proxy_invoked','ota_lcd_proxy_invoked',0,1)
ENUMS = {
    ('network','wifi_state'): ('wifi','connected disconnected failed unknown'),
    ('system','wake_cause_str'): ('wake_cause_str','EXT0 EXT1 TIMER TOUCHPAD TOUCH ULP GPIO UART WIFI COCPU COCPU_TRAP BT UNDEFINED OTHER'),
    ('link','last_uart_tx_type'): ('last_uart_tx_type',UART_TYPES),
    ('link','last_uart_rx_type'): ('last_uart_rx_type',UART_TYPES),
    ('link','lcd_diag_last_tx'): ('lcd_diag_last_tx',UART_TYPES),
    ('link','lcd_diag_last_rx'): ('lcd_diag_last_rx',UART_TYPES),
    ('system', 'reset_reason'): ('reset_reason', 'UNKNOWN POWERON EXT EXTERNAL SW PANIC INT_WDT TASK_WDT WDT DEEPSLEEP BROWNOUT SDIO USB JTAG EFUSE PWR_GLITCH CPU_LOCKUP'),
    ('system', 'running_partition'): ('part', 'app0 app1 factory unknown'),
    ('system', 'boot_partition'): ('boot', 'app0 app1 factory unknown'),
    ('ota', 'result'): ('ota_result', 'unknown none pending started checking noop no_update up_to_date success succeeded ok failed error deferred aborted rolled_back rollback policy_daily_limit peer_unavailable clock_unconfirmed_defer policy_target_valid check_begin manifest_url_invalid manifest_fetch_fail policy_target_mismatch retry_target_changed board_mismatch bin_url_disallowed downgrade_blocked rollout_min_version rollout_skip lcd_proxy_failed_defer lcd_arm_unverified_defer paired_deadline target_storage_unverified apply_success SUCCESS FAILED_DOWNLOAD FAILED_SHA256_MISMATCH FAILED_WRITE FAILED_VERIFY FAILED_INVALID_MANIFEST PARTITION_TOO_SMALL HARD_DEADLINE_TIMEOUT NO_PROGRESS_TIMEOUT INCOMPLETE SIZE_MISMATCH NO_STREAM HTTP_BEGIN_FAIL HTTP_GET_FAIL STALL_RETRY_EXHAUSTED SET_BOOT_FAIL MARKER_NOT_FOUND MARKER_VERSION_MISMATCH POLICY_DEFERRED'),
    ('action', 'kind'): ('action_kind', 'check_in eat discard voice shopping_refresh shopping_add shopping_remove manual_ota scheduled_ota unknown'),
    ('action', 'result'): ('action_result', 'started accepted completed failed cancelled deferred noop unknown'),
    ('action', 'scope'): ('action_scope', 'boot_v1'),
    ('action', 'list_last_result'): ('list_last_result', 'pending ok wifi_timeout wifi_unavailable provisioning host network heap parse http dns timeout failed'),
    ('ota', 'ota_fail_stage'): ('ota_fail_stage','none heap_start partition begin_initial begin_restart begin_range write_marker write_probe heap_stream begin_same_offset begin_no_progress write_stream end sha_check unknown'),
    ('ota', 'ota_ready_decision'): ('ota_ready_decision', 'unobserved local_not_valid deadline storage_wait no_record clock_wait wait_due not_due before_due expired deferred ready'),
    ('diagnostics', 'diag_export_ack'): ('diag_export_ack', 'committed not_committed'),
    ('sleep', 'maint_sync_resolution'): ('maint_sync_resolution', 'idle acked ack_mismatch pending request_changed tx_sent timeout_no_ack sent_no_recent_link deferred_time_invalid skip_empty_window waiting_for_ack'),
    ('network', 'idle_net_decision'): ('idle_net_decision', 'allow allowed denied busy skipped deadline expired unknown ready'),
    ('network', 'idle_net_op'): ('idle_net_op', 'none manifest report diagnostic handoff unknown'),
    ('diagnostics', 'diag_export_decision'): ('diag_export_decision', 'not_called pre_sleep_budget deadline pending_boot busy wifi local_unavailable no_item owner envelope post_return ack peer_query'),
}
HEALTH = {'ok':'healthy', 'healthy':'healthy', 'ready':'healthy', 'degraded':'degraded', 'warning':'degraded', 'warn':'degraded', 'error':'error', 'failed':'error', 'unknown':'unknown'}

def integer(value, low=0, high=253402300799):
    if type(value) is bool or value is None:
        return None
    if isinstance(value, str):
        if not re.fullmatch(r'-?[0-9]{1,20}', value): return None
    elif not isinstance(value, (int, Decimal)):
        return None
    try:
        n = int(value)
        if n != value and not isinstance(value, str): return None
        return n if low <= n <= high else None
    except (ValueError, OverflowError):
        return None

def safe_text(value, pattern=IDENTIFIER):
    return value if isinstance(value, str) and pattern.fullmatch(value) else None

NUMERIC_WIRE_FIELDS = frozenset(key for keys, _, _ in NUMBERS.values() for key in keys) | {'ts_epoch'}


def raw_numeric_rejections(raw):
    """Remember raw type/bound failures before legacy INT_FIELDS coercion loses them."""
    specs = {}
    for keys, low, high in NUMBERS.values():
        for key in keys: specs.setdefault(key, []).append((low, high))
    specs['ts_epoch'] = [(0, 253402300799)]
    rejected = []
    for key, bounds in specs.items():
        if key not in raw: continue
        value = raw[key].strip() if isinstance(raw[key], str) else raw[key]
        # A few wire names, such as wifi, also have a supported string enum.
        if isinstance(value, str) and any(key == wire and value in values.split() for wire, values in ENUMS.values()): continue
        accepted = any(integer(int(value) if type(value) is bool and (low, high) == (0, 1) else value, low, high) is not None for low, high in bounds)
        if not accepted: rejected.append(key)
    return sorted(rejected)


def without_rejected_numerics(payload, rejected):
    # Only server-derived, allowlisted field names can affect this projection.
    blocked = {key for key in rejected if isinstance(key, str) and key in NUMERIC_WIRE_FIELDS} if isinstance(rejected, list) else set()
    return {key: value for key, value in payload.items() if key not in blocked}


def retention_days():
    return integer(os.getenv('ANALYTICS_RETENTION_DAYS', '90'), 7, 365) or 90

def heap_samples(payload):
    # OtaHeapTrace.h uses exactly 14 scalars; unsampled pairs are not zero observations.
    mask=integer(payload.get('ota_heap_mask'),1,127)
    pairs=payload.get('ota_heap_pairs')
    if payload.get('ota_heap_schema')!=1 or type(payload.get('ota_heap_schema')) is bool or mask is None or not isinstance(pairs,list) or len(pairs)!=14:return None
    values=[integer(v,0,1073741824) for v in pairs]
    if any(v is None for v in values):return None
    return mask,values

def normalize(payload, received_at_epoch, event_id, diagnostic=False):
    """No arbitrary strings/nested structures survive this allowlist."""
    report_type = payload.get('report_type')
    health = payload.get('health')
    row = {'schema_version':SCHEMA, 'device_id':safe_text(payload.get('device_id')),
           'device_type':payload.get('device_type') if payload.get('device_type') in ('sense','lcd') else 'unknown',
           'fw':safe_text(payload.get('fw'), VERSION), 'build':safe_text(payload.get('build'), VERSION),
           'board':safe_text(payload.get('board')), 'channel':payload.get('channel') if payload.get('channel') in ('prod','dev','beta','canary') else None,
           'report_type':report_type if isinstance(report_type, str) and report_type in REPORT_TYPES else 'unknown',
           'event_id':event_id, 'received_at_epoch':integer(received_at_epoch),
           'device_reported_at_epoch':integer(payload.get('ts_epoch')),
           'source':'retained_diagnostic' if diagnostic else 'device_report',
           'health':HEALTH.get(health, 'unknown') if isinstance(health,str) else 'unknown', 'metrics':{}}
    for (group, name), (keys, low, high) in NUMBERS.items():
        for key in keys:
            value = payload.get(key)
            if type(value) is bool and (low,high)==(0,1): value=int(value)
            number = integer(value, low, high)
            if number is not None:
                row['metrics'].setdefault(group,{})[name]=number
                break
    for (group,name),(wire,values) in ENUMS.items():
        value=payload.get(wire)
        if isinstance(value,str) and value in values.split(): row['metrics'].setdefault(group,{})[name]=value
    wifi=row['metrics'].get('network',{}).get('wifi_state')
    if wifi in ('connected','disconnected','failed'):
        row['metrics'].setdefault('network',{})['wifi_connected']=int(wifi=='connected')
    heap=heap_samples(payload)
    if heap:
        mask,pairs=heap
        for i,point in enumerate(HEAP_POINTS):
            if mask&(1<<i):
                row['metrics'].setdefault('memory',{})['internal_free_bytes_'+point]=pairs[i*2]
                row['metrics']['memory']['internal_largest_bytes_'+point]=pairs[i*2+1]
    action=row['metrics'].get('action',{})
    scoped=('action_boot_id','list_attempts','list_successes','list_failures','list_last_result','list_last_duration_ms','scope')
    if action.get('scope')!='boot_v1' or not action.get('action_boot_id'):
        for key in scoped: action.pop(key,None)
    elif not action.get('list_attempts'):
        action.pop('list_last_result',None);action.pop('list_last_duration_ms',None)
    elif action.get('list_last_result')=='pending':
        action.pop('list_last_duration_ms',None)
    if not action: row['metrics'].pop('action',None)
    target=safe_text(payload.get('ota_policy_target') or payload.get('manifest_version'), VERSION)
    if target: row['metrics'].setdefault('ota',{})['target_version']=target
    peer_fw=safe_text(payload.get('lcd_fw'), VERSION)
    if row['device_type']=='sense' and peer_fw and peer_fw not in ('-', 'unknown'):
        peer={'source':'sense_peer_report','reported_by':row['device_id'],'fw':peer_fw,
              'report_received_at_epoch':row['received_at_epoch'],
              'peer_observation_age_s':integer(payload.get('lcd_fw_age_s'),0,31536000),
              'direct_presence_confirmed':False}
        outcome=payload.get('lcd_ota_result')
        if outcome in ENUMS[('ota','result')][1].split(): peer['ota_result']=outcome
        row['paired_lcd']=peer
    return row

def add_age(row, now, stale_after):
    row=dict(row);received=integer(row.get('received_at_epoch'))
    valid=received is not None and 0<received<=now+5
    row.update(age_basis='server_received_at' if valid else 'unknown',age_s=max(0,now-received) if valid else None,stale_after_s=stale_after)
    row['stale']=row['age_s']>=stale_after if valid else None
    if not valid: row['received_at_epoch']=None
    device=integer(row.get('device_reported_at_epoch'))
    skew=device-received if valid and device and device>=1577836800 else None
    row['clock_skew_s']=skew
    row['clock_status']='unknown' if skew is None else ('device_ahead' if skew>300 else 'device_behind' if skew< -300 else 'aligned')
    return row

def from_item(item, latest=False):
    payload=item.get('last_payload' if latest else 'payload') or {}
    if not isinstance(payload,dict): payload={}
    rejected=item.get('analytics_raw_numeric_rejections',[])
    received=item.get('updated_at_epoch' if latest else 'ingested_at_epoch')
    event_id=str(item.get('device_id') or '')+':'+str(item.get('last_event_ts_key' if latest else 'event_ts_key') or '')
    # Project stored normalized rows AGAIN, rather than trust arbitrary database maps.
    if latest and isinstance(item.get('analytics_latest_payload'),dict):
        payload=item['analytics_latest_payload'];received=item.get('analytics_received_at_epoch');event_id=item.get('analytics_event_id') or event_id
        rejected=[]  # This independent snapshot already omitted its own raw failures.
    payload=without_rejected_numerics(payload,rejected)
    diagnostic=bool(payload.get('diag_export') or payload.get('diag_handoff'))
    return normalize(payload,received,event_id,diagnostic)

def safe_payload(payload):
    """Minimal flattened snapshot for independent analytics latest (no owner/media)."""
    row=normalize(payload,None,'');allowed={}
    for key in ('device_id','fw','build','board','channel'):
        if row.get(key) is not None: allowed[key]=row[key]
    for key in ('device_type','report_type'):
        if row[key]!='unknown': allowed[key]=row[key]
    if row['device_reported_at_epoch'] is not None: allowed['ts_epoch']=row['device_reported_at_epoch']
    if row['health']!='unknown': allowed['health']=row['health']
    for (group,name),(keys,_,_) in NUMBERS.items():
        if name in row['metrics'].get(group,{}): allowed[keys[0]]=row['metrics'][group][name]
    for (group,name),(wire,_) in ENUMS.items():
        if name in row['metrics'].get(group,{}): allowed[wire]=row['metrics'][group][name]
    if 'target_version' in row['metrics'].get('ota',{}): allowed['ota_policy_target']=row['metrics']['ota']['target_version']
    if 'paired_lcd' in row:
        peer=row['paired_lcd'];allowed['lcd_fw']=peer['fw']
        if peer['peer_observation_age_s'] is not None: allowed['lcd_fw_age_s']=peer['peer_observation_age_s']
        if 'ota_result' in peer: allowed['lcd_ota_result']=peer['ota_result']
    heap=heap_samples(payload)
    if heap:
        mask,pairs=heap
        allowed['ota_heap_pairs']=[value if mask&(1<<(i//2)) else 0 for i,value in enumerate(pairs)]
    return allowed

def update_latest(table,payload,event_item,received):
    # Exported retained diagnostics do not represent the current boot and never
    # change latest contact/state (preserves the deployed D3/H4 exclusion).
    snapshot=safe_payload(payload)
    try:
        table.update_item(Key={'device_id':payload['device_id']},
            UpdateExpression='SET analytics_latest_payload = :payload, analytics_received_at_epoch = :received, analytics_event_id = :event',
            ConditionExpression='attribute_not_exists(analytics_received_at_epoch) OR analytics_received_at_epoch <= :received',
            ExpressionAttributeValues={':payload':snapshot,':received':received,':event':payload['device_id']+':'+event_item['event_ts_key']})
        return True
    except ClientError as exc:
        if exc.response.get('Error',{}).get('Code')=='ConditionalCheckFailedException':return False
        raise

def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=lambda x:int(x) if isinstance(x,Decimal) and x==int(x) else str(x)).encode()

def _b64(data): return base64.urlsafe_b64encode(data).rstrip(b'=').decode()
def _unb64(text): return base64.urlsafe_b64decode(text+'='*((-len(text))%4))

def cursor_encode(key,scope,secret,now):
    raw=_json({'v':1,'expires':now+900,'scope':scope,'key':key})
    return _b64(raw)+'.'+_b64(hmac.new(secret.encode(),raw,hashlib.sha256).digest())

def cursor_decode(token,scope,secret,now):
    try:
        if not isinstance(token,str) or len(token)>4096: raise ValueError()
        body,sig=token.split('.');raw=_unb64(body)
        if not hmac.compare_digest(_unb64(sig),hmac.new(secret.encode(),raw,hashlib.sha256).digest()): raise ValueError()
        d=json.loads(raw)
        if d['v']!=1 or d['scope']!=scope or type(d['expires']) is not int or d['expires']<now or not isinstance(d['key'],dict):raise ValueError()
        return d['key']
    except Exception:
        raise ValueError('invalid_query_or_cursor') from None

def serve(event,events_table,latest_table,response):
    """One bounded DB operation per request. Legacy GET routes never call here."""
    secret=os.getenv('ANALYTICS_READ_KEY','')
    if len(secret)<32:return response(503,{'ok':False,'error':'analytics_not_configured'})
    headers={str(k).lower():v for k,v in (event.get('headers') or {}).items()}
    provided=headers.get('x-analytics-key')
    if not isinstance(provided,str) or not hmac.compare_digest(provided.encode(),secret.encode()):return response(403,{'ok':False,'error':'forbidden'})
    try:
        query=event.get('queryStringParameters') or {};now=int(time.time())
        path=(event.get('rawPath') or event.get('requestContext',{}).get('http',{}).get('path','')).rstrip('/')
        route='latest' if path.endswith('/ota/report/latest') else 'events' if path.endswith('/ota/report/events') else None
        if not route:raise ValueError()
        allowed={'view','limit','cursor','stale_after_s','device_id','owner_id'} if route=='latest' else {'view','limit','cursor','stale_after_s','device_id','report_type'}
        if set(query)-allowed:raise ValueError()
        limit=integer(query.get('limit','50'),1,200);stale_after=integer(query.get('stale_after_s','900'),60,604800)
        if limit is None or stale_after is None:raise ValueError()
        device=safe_text(query.get('device_id'));owner=safe_text(query.get('owner_id'));kind=query.get('report_type')
        for key in ('device_id','owner_id'):
            if key in query and safe_text(query[key]) is None:raise ValueError()
        if route=='latest' and device and owner:raise ValueError()
        if route=='events' and (not device and not kind or device and kind):raise ValueError()
        if kind is not None and (not isinstance(kind,str) or kind not in REPORT_TYPES):raise ValueError()
        scope={'route':route,'device_id':device,'owner_id':owner,'report_type':kind,'stale_after_s':stale_after,'limit':limit}
        token=query.get('cursor');key=cursor_decode(token,scope,secret,now) if token else None
        if route=='latest' and device:
            if key:raise ValueError()
            result=latest_table.get_item(Key={'device_id':device},ConsistentRead=True);items=[result['Item']] if result.get('Item') else [];evaluated=len(items);next_key=None
        else:
            kwargs={'Limit':limit}
            if key:kwargs['ExclusiveStartKey']=key
            if route=='latest' and owner:
                result=latest_table.query(IndexName='OwnerLatestIndex',KeyConditionExpression=Key('owner_id').eq(owner),ScanIndexForward=False,**kwargs)
            elif route=='latest':result=latest_table.scan(**kwargs)
            elif device:result=events_table.query(KeyConditionExpression=Key('device_id').eq(device),ScanIndexForward=False,**kwargs)
            else:result=events_table.query(IndexName='ReportTypeTsIndex',KeyConditionExpression=Key('report_type').eq(kind),ScanIndexForward=False,**kwargs)
            items=result.get('Items',[]);evaluated=result.get('ScannedCount',result.get('Count',len(items)));next_key=result.get('LastEvaluatedKey')
        rows=[];cutoff=now-retention_days()*86400
        for item in items:
            row=from_item(item,route=='latest')
            if route=='events' and (row['received_at_epoch'] is None or row['received_at_epoch']<cutoff):continue
            rows.append(add_age(row,now,stale_after))
        rows.sort(key=lambda row:(row.get('received_at_epoch') or 0,row['event_id']),reverse=True)
        return response(200,{'ok':True,'schema_version':SCHEMA,'server_time_epoch':now,'count':len(rows),'items':rows,
            'next_cursor':cursor_encode(next_key,scope,secret,now) if next_key else None,
            'page':{'scope':'bounded_page','evaluated_count':evaluated,'returned_count':len(rows),'has_more':bool(next_key),'ordering':'page_received_at_desc','global_order_guaranteed':False},
            'retention':{'events_window_days':retention_days(),'storage_expiry':'unchanged; analytics does not set expiry attributes or change DynamoDB TTL'}})
    except ValueError:return response(400,{'ok':False,'error':'invalid_query_or_cursor'})
    except Exception:return response(500,{'ok':False,'error':'analytics_unavailable'})
