"""A status read must work without loading extraction/model runtimes."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_process_status_read_does_not_import_media_or_model_libraries():
    script = r'''
import importlib.abc, json, sys
class ExtractionImportTrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai','bs4','yt_dlp','recipe_inventory_llm'}:
            raise AssertionError('Unexpected extraction import during status read: '+fullname)
sys.meta_path.insert(0, ExtractionImportTrap())
import app
app._get_saved_recipe_batch_job = lambda _: {'job_id':'fixture','owner':'alice','status':'COMPLETED','results':[{'recipe_id':'rice'}]}
app._mysql_conn = lambda: object()
app._fetch_saved_recipe_by_id = lambda conn, owner, rid: {'_id':rid,'_owner':owner,'title':'Rice','status':'ready','ingredients':['1 cup rice'],'instructions':['Cook the rice.']}
response = app._get_saved_recipe_batch_job_status('alice','fixture')
assert response['statusCode'] == 200, response
body = json.loads(response['body'])
assert body['recipes'][0]['title'] == 'Rice'
assert body['recipes'][0]['instructions'] == ['Cook the rice.']
assert app._get_saved_recipe_batch_job_status('foreign','fixture')['statusCode'] == 404
print('PASS cold status read, current content, owner isolation, no extraction imports')
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(ROOT/'saved_recipe_status'), str(ROOT/'saved_recipes_api')]),
        AWS_EC2_METADATA_DISABLED='true', AWS_DEFAULT_REGION='us-east-1',
        AWS_ACCESS_KEY_ID='fixture', AWS_SECRET_ACCESS_KEY='fixture')
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
