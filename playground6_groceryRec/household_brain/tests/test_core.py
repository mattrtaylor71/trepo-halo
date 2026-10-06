from datetime import datetime,timedelta,timezone
from household_brain.core import project,encode,MAX_CONTEXT_BYTES,product_key

NOW=datetime(2026,10,6,18,tzinfo=timezone.utc)
STATE={'household_id':'h','generation':1,'paused':False,'learning_since':NOW-timedelta(days=90),'timezone':'America/Los_Angeles'}

def base(**extra):return {'members':['a','b'],'subjects':[],**extra}

def test_cadence_requires_three_distinct_confirmed_purchase_days():
    def purchase(i,day):
        return {'producer':'orders','event_id':str(i),'actor_id':'a','generation':1,'kind':'purchase',
            'payload':{'name':'Milk','visibility':'household'},'occurred_at':NOW-timedelta(days=day)}
    events=[purchase(1,16),purchase(2,9),purchase(3,2)]
    context=project(base(events=events),STATE,'a',NOW)
    assert context['products'][0]['cadence']['median_days']==7
    assert project(base(events=events[:2]),STATE,'a',NOW)['products'][0]['cadence'] is None
    duplicated=[purchase(i,2) for i in range(10)]
    assert project(base(events=duplicated),STATE,'a',NOW)['products'][0]['cadence'] is None

def test_unicode_bounded_and_variants_not_merged():
    states=[{'household_id':'h','actor_id':'a','enabled':True,'facts':{key:{'value':['🍎'*80]*8,'visibility':'personal'}
        for key in ['cuisines','favorite_foods','avoid_foods','preferred_products','cooking_style']}}]
    context=project(base(subjects=states),STATE,'a',NOW)
    assert len(encode(context).encode())<=MAX_CONTEXT_BYTES
    assert product_key('Milk','Brand','whole')!=product_key('Milk','Brand','skim')

def test_unknown_stock_future_evidence_and_expired_dates_are_not_inferred():
    kitchen=[{'_id':'i','product_name':'Milk','action':'IN','_createdDate':NOW,'product_expiration':'2026-10-05'},
             {'_id':'f','product_name':'Future','action':'IN','_createdDate':NOW+timedelta(days=1)}]
    context=project(base(kitchen=kitchen),STATE,'a',NOW)
    assert context['check_dates'][0]['name']=='Milk' and not context['use_soon']
    assert not context['groceries']
    assert all(p['name']!='Future' for p in context['products'])

def test_receipt_calendar_date_is_interpreted_in_household_timezone():
    rows=[{'_id':str(day),'product_name':'Milk','action':'IN','_createdDate':NOW,'provisional_payload':{
        'scan_source':'receipt_scan','source_job_id':str(day),'receipt_purchase_date':f'2026-09-{day:02d}'}}
        for day in [16,23,30]]
    context=project(base(kitchen=rows),STATE,'a',NOW)
    assert context['products'][0]['cadence']['next_window_start']=='2026-10-07'
