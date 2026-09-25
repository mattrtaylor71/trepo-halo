"""Keep TikTok video-only CDN variants out of transcription, without changing other sites."""
import ast,os,socket
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
import pytest,yt_dlp,recipe_work_budget


def actual_options(url,download=True):
    source=(Path(os.environ['RECIPE_CANDIDATE_SOURCE'])/'app.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_extract_ytdlp_info')
    options=[]
    class Capture:
        def __init__(self,value):options.append(value)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def extract_info(self,*args,**kwargs):return {}
    namespace={'recipe_work_budget':recipe_work_budget,'yt_dlp':SimpleNamespace(YoutubeDL=Capture),'_REQUEST_TIMEOUT_SECONDS':15,'_USER_AGENT':'fixture',
      '_resolve_ytdlp_cookiefile':lambda:None,'_YTDLP_MAX_ATTEMPTS':1,'urlparse':urlparse,'_VALID_TIKTOK_HOSTS':{'tiktok.com','www.tiktok.com','vm.tiktok.com'},'os':os}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual extraction helper>','exec'),namespace)
    namespace['_extract_ytdlp_info'](url,download=download)
    return options[0]


def select(options,formats):
    with yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True}) as downloader:
        selection=downloader.build_format_selector(options['format'])
        return list(selection({'formats':formats,'has_merged_format':False,'incomplete_formats':False}))


def video(ident,path,**extra):
    return dict(format_id=ident,url='https://cdn.example/'+path,ext='mp4',vcodec='h264',acodec='aac',**extra)


def test_actual_tiktok_selection_avoids_mislabelled_silent_hevc():
    formats=[video('play','media-video-avc1/'),video('bytevc1','media-video-hvc1/')]
    chosen=select(actual_options('https://www.tiktok.com/@fixture/video/123'),formats)
    assert [f['format_id'] for f in chosen]==['play']

@pytest.mark.parametrize('url',['https://www.instagram.com/reel/fixture/','https://www.facebook.com/reel/123','https://not-tiktok.com/video/123'])
def test_other_platforms_retain_their_existing_selection(url):
    assert actual_options(url)['format']=='bestaudio/best'

@pytest.mark.parametrize('host',['tiktok.com','www.tiktok.com','vm.tiktok.com','m.tiktok.com'])
def test_tiktok_audio_only_and_muxed_fallbacks_still_work(host):
    options=actual_options('https://'+host+'/video/123')
    audio=dict(format_id='audio',url='https://cdn.example/voice.m4a',ext='m4a',vcodec='none',acodec='aac')
    assert select(options,[audio,video('play','video.mp4')])[0]['format_id']=='audio'
    assert select(options,[video('play','video.mp4')])[0]['format_id']=='play'
    assert select(options,[video('silent','media-video-hvc1/')])==[]


def test_caption_metadata_extraction_does_not_apply_audio_filter():
    assert 'format' not in actual_options('https://www.tiktok.com/@fixture/video/123',download=False)
