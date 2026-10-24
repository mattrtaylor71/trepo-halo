import sys,io,hashlib,importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from icon_service import assets,delivery
PATH='/icons/trepo-app-bold-outline-v1/'+'a'*64+'/'+'b'*64+'.png'

def test_cdn_rewrites_existing_catalog_urls_without_changing_key(monkeypatch):
    monkeypatch.setenv('ICON_CDN_HOST','dexample.cloudfront.net')
    assert delivery.transport_url('https://origin.example'+PATH,delivery.APP_STYLE)=='https://dexample.cloudfront.net'+PATH

@pytest.mark.parametrize('host',['https://bad.com','evil.example','dgood.cloudfront.net/path','dgood.cloudfront.net:443'])
def test_invalid_cdn_config_fails_closed(monkeypatch,host):
    monkeypatch.setenv('ICON_CDN_HOST',host)
    with pytest.raises(ValueError):delivery.transport_url('https://origin.example'+PATH,delivery.APP_STYLE)

def test_origin_hash_check_remains_required_and_corrupt_bytes_never_cache():
    client=Mock();client.get_object.return_value={'ContentLength':3,'Body':io.BytesIO(b'bad')}
    response=assets.handle({'rawPath':PATH},client,'private-bucket')
    assert response['statusCode']==404 and 'Cache-Control' not in response.get('headers',{})

def test_valid_immutable_origin_bytes_keep_digest_and_cache_control():
    data=b'fixture-png';digest=hashlib.sha256(data).hexdigest()
    client=Mock();client.get_object.return_value={'ContentLength':len(data),'Body':io.BytesIO(data)}
    path=PATH.replace('b'*64,digest)
    response=assets.handle({'rawPath':path},client,'private-bucket')
    assert response['statusCode']==200 and 'immutable' in response['headers']['Cache-Control']
    assert response['headers']['ETag']=='"'+digest+'"'
