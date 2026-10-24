import importlib.util,json,sys,time,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import link_recipe_recovery as r

TEXT='Ingredients:\n1 cup yogurt\n1 cup oats\nHow to make:\n1. Mix yogurt and oats in a bowl.\n2. Serve the mixture.'
CANDIDATE={'is_recipe':True,'title':'Yogurt oats','ingredients':[
    {'name':'yogurt','amount':'1 cup','amount_quote':'1 cup yogurt','evidence_ids':['source_text']},
    {'name':'oats','amount':'1 cup','amount_quote':'1 cup oats','evidence_ids':['source_text']}],
    'steps':[{'instruction':'Mix yogurt and oats in a bowl.','source_quote':'Mix yogurt and oats in a bowl.','evidence_ids':['source_text']},
    {'instruction':'Serve the mixture.','source_quote':'Serve the mixture.','evidence_ids':['source_text']}], 'missing_information':[]}
REVIEW={'is_recipe':True,'single_coherent_recipe':True,'essential_components_identified':True,
    'ingredients':[{'index':i,'supported':True,'amount_supported':True} for i in range(2)],
    'steps':[{'index':i,'supported':True} for i in range(2)]}
class Table:
    def __init__(self):self.rows={}
    def get_item(self,Key,**kw):return {'Item':copy.deepcopy(self.rows.get(Key['job_id'],{}))}
    def put_item(self,Item,**kw):self.rows[Item['job_id']]=copy.deepcopy(Item)
class Budget:
    input_tokens=0;output_tokens=0
    def __init__(self,table=None,fail=False):
        self.calls=[];self.fail=fail
        if table:self.checkpoint=r.ExtractionCheckpoint(table,'owner','source')
    def call(self,prompt,parts,schema,name):
        self.calls.append(name)
        if name=='recipe_verification' and self.fail:raise TimeoutError()
        return copy.deepcopy(CANDIDATE if name=='recipe_extraction' else REVIEW)

@pytest.mark.parametrize('heading',['How to make','Method','Instructions','Directions','做法','步驟'])
def test_written_captions_select_text_path_even_when_short(heading):
    assert r.has_written_method({'content':TEXT.replace('How to make',heading)})

def test_numbered_cooking_steps_and_insufficient_caption():
    assert r.has_written_method({'content':TEXT.replace('How to make:','')})
    assert not r.has_written_method({'content':'Ingredients: yogurt. Follow for more recipes! '*3})

def test_caption_route_does_not_download_media(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'public_download',Mock(side_effect=AssertionError('unnecessary media')))
    result,metrics=r.extract_source({'content':TEXT,'video_url':'https://example.org/video.mp4'},tmp_path,Budget())
    assert result['title']=='Yogurt oats' and metrics['reason']=='verified'

def test_verifier_timeout_reuses_extraction_but_still_verifies():
    table=Table();first=Budget(table,True)
    with pytest.raises(TimeoutError):r.extract_evidence([{'id':'source_text','text':TEXT}],[],first)
    second=Budget(table)
    result,_=r.extract_evidence([{'id':'source_text','text':TEXT}],[],second)
    assert result and second.calls==['recipe_verification']

@pytest.mark.parametrize('change',['text','version','correction','owner','expired'])
def test_checkpoint_never_crosses_evidence_owner_version_or_expiry(monkeypatch,change):
    table=Table();first=Budget(table)
    r.extract_evidence([{'id':'source_text','text':TEXT}],[],first)
    second=Budget(table);text=TEXT;correction=None
    if change=='text':text+=' New source note.'
    if change=='version':monkeypatch.setattr(r,'VERSION','new-validator')
    if change=='correction':correction='unquoted_instruction_fact'
    if change=='owner':second.checkpoint.owner='other'
    if change=='expired':
        for item in table.rows.values():item['expires_epoch']=0
    r.extract_evidence([{'id':'source_text','text':text}],[],second,correction=correction)
    assert second.calls==['recipe_extraction','recipe_verification']

def test_unsupported_temperature_is_still_rejected():
    candidate=copy.deepcopy(CANDIDATE)
    candidate['steps'][0]['instruction']='Bake at 400 degrees F.'
    result=r.verified_recipe(candidate,REVIEW,[{'id':'source_text','text':TEXT}],set(),{})
    assert result is None

@pytest.mark.parametrize('name,amount,expected',[
    ('yogurt','1 cup yogurt','1 cup yogurt'),
    ('yogurt','1 cup or 150g yogurt','1 cup or 150g yogurt'),
    ('yogurt (for topping)','1 cup yogurt','1 cup yogurt (for topping)'),
    ('1 cup yogurt','1 cup','1 cup yogurt'),
    ('corn','2 cups corn or 240g kernels','2 cups corn or 240g kernels'),
])
def test_quoted_measurement_is_preserved_without_duplicate_food(name,amount,expected):
    candidate=copy.deepcopy(CANDIDATE)
    quote=amount if 'yogurt' in amount or 'corn' in amount else amount+' yogurt'
    candidate['ingredients'][0].update(name=name,amount=amount,amount_quote=quote)
    text=TEXT+'\n'+quote
    result=r.verified_recipe(candidate,REVIEW,[{'id':'source_text','text':text}],set(),{})
    assert result['ingredients'][0]==expected

def test_interleaved_measurement_uses_literal_source_line_not_reconstructed_fields():
    candidate=copy.deepcopy(CANDIDATE)
    candidate['ingredients'][0].update(name='corn kernels',amount='2 cups corn or 240g kernels',amount_quote='2 cups corn or 240g kernels')
    result=r.verified_recipe(candidate,REVIEW,[{'id':'source_text','text':TEXT+'\n2 cups corn or 240g kernels'}],set(),{})
    assert result['ingredients'][0]=='2 cups corn or 240g kernels'
    candidate['ingredients'][0].update(name='dry jasmine rice',amount='1 cup (185g)',amount_quote='1 cup dry jasmine rice (185g)')
    result=r.verified_recipe(candidate,REVIEW,[{'id':'source_text','text':TEXT+'\n1 cup dry jasmine rice (185g)'}],set(),{})
    assert result['ingredients'][0]=='1 cup dry jasmine rice (185g)'

def test_invented_measurement_requires_correction_not_silent_omission():
    candidate=copy.deepcopy(CANDIDATE)
    candidate['ingredients'][0]['amount']='3 cups'
    diagnostic={}
    assert r.verified_recipe(candidate,REVIEW,[{'id':'source_text','text':TEXT}],set(),diagnostic) is None
    assert diagnostic['rejection_code']=='unverified_ingredient_amount'
    assert diagnostic['rejection_code'] in r.CORRECTABLE_GUARDS

@pytest.mark.parametrize('url',['https://tiktok.com/@x/video/1','https://instagram.com/p/1','https://example.org/recipe'])
def test_retained_evidence_precedes_network_for_all_sources(monkeypatch,url):
    table=Table();app=SimpleNamespace(_jobs_table=lambda:table,_resolved_url_hash=lambda u:u,_log_event=Mock())
    monkeypatch.setattr(r,'cache_claim',lambda *a:('claimed',None));monkeypatch.setattr(r,'cache_finish',Mock())
    monkeypatch.setattr(r,'ModelBudget',Budget);monkeypatch.setattr(r,'source_for',Mock(side_effect=AssertionError('should reuse source')))
    result=r.recover(app,'owner',url,retained_document=TEXT)
    assert result['recipe']['title']=='Yogurt oats'

def test_incomplete_retained_source_fetches_requested_source(monkeypatch):
    table=Table();app=SimpleNamespace(_jobs_table=lambda:table,_resolved_url_hash=lambda u:u,_log_event=Mock())
    monkeypatch.setattr(r,'cache_claim',lambda *a:('claimed',None));monkeypatch.setattr(r,'cache_finish',Mock())
    monkeypatch.setattr(r,'ModelBudget',Budget);fetch=Mock(return_value={'content':TEXT})
    monkeypatch.setattr(r,'source_for',fetch);monkeypatch.setattr(r,'prefer_linked_publisher',lambda app,s:s)
    result=r.recover(app,'owner','https://example.org/recipe',retained_document='My lunch today!')
    assert result['recipe'] and fetch.call_count==1
