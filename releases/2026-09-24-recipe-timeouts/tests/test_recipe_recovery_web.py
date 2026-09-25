import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('recovery_web',Path(__file__).resolve().parents[1]/'saved_recipes_api/recovery_web.py')
web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)

def dns(host,port,**kwargs):
    return [(None,None,None,None,('93.184.216.34' if host=='public.example' else '127.0.0.1',port))]

@pytest.mark.parametrize('url',['file:///etc/passwd','http://localhost','http://user:pass@public.example','http://public.example:8080','http://127.0.0.1','http://169.254.169.254'])
def test_recovery_rejects_nonpublic_targets(url):
    with pytest.raises(ValueError):web.public_target(url,dns)


def test_checked_address_is_pinned_with_original_tls_and_host():
    calls=[]
    class Pool:
        def __init__(self,**kwargs):calls.append(kwargs)
        def urlopen(self,method,path,**kwargs):
            calls.append(kwargs)
            assert kwargs['headers']['Host']=='public.example'
            class Response:
                status=200;headers={'Content-Type':'text/html'}
                def read(self,*a,**k):return b'<html>Recipe</html>'
                def close(self):pass
            return Response()
        def close(self):pass
    assert web.fetch_page('https://public.example/recipe',dns,Pool)[0]=='<html>Recipe</html>'
    assert calls[0]['host']=='93.184.216.34'
    assert calls[0]['server_hostname']==calls[0]['assert_hostname']=='public.example'
    assert calls[1]['redirect'] is False


def test_redirect_to_internal_address_stops_before_second_request():
    requests=[]
    class Pool:
        def __init__(self,**kwargs):requests.append(kwargs)
        def urlopen(self,*a,**k):
            class Response:
                status=302;headers={'Location':'http://localhost/private'}
                def close(self):pass
            return Response()
        def close(self):pass
    with pytest.raises(ValueError):web.fetch_page('https://public.example/recipe',dns,Pool)
    assert len(requests)==1


def test_mixed_public_and_private_dns_answer_is_not_trusted():
    def mixed(host,port,**kwargs):return dns('public.example',port)+dns('localhost',port)
    with pytest.raises(ValueError):web.public_target('https://public.example',mixed)


def test_deadline_interrupts_slow_dns_and_restores_signal_handler():
    import signal
    import time
    import threading
    original = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    release = threading.Event()
    def slow(*a, **k):
        release.wait(1)
        return dns(*a, **k)
    try:
        with pytest.raises(ValueError, match='has not changed'):
            with web.recovery_budget(0.04):
                web.public_target('https://public.example', slow)
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == 'recipe-recovery-dns': thread.join(0.5)
    assert time.monotonic() - started < 0.5
    assert signal.getsignal(signal.SIGALRM) == original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_native_dns_cannot_hold_request_or_accumulate_workers():
    import ctypes
    import threading
    import time
    def blocked(*a, **k):
        ctypes.CDLL(None).usleep(300000)
        return dns(*a, **k)
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match='lookup timed out'):
            web._bounded_dns('public.example', 443, blocked, seconds=0.03)
        assert time.monotonic() - started < 0.2
        for _ in range(10):
            with pytest.raises(ValueError, match='lookup is busy'):
                web._bounded_dns('public.example', 443, dns, seconds=0.03)
        assert sum(t.name == 'recipe-recovery-dns' for t in threading.enumerate()) == 1
    finally:
        for thread in threading.enumerate():
            if thread.name == 'recipe-recovery-dns': thread.join(1)


def test_deadline_escapes_provider_exception_retry_without_background_work():
    import time
    attempts = []
    with pytest.raises(ValueError, match='has not changed'):
        with web.recovery_budget(0.04):
            for attempt in range(2):
                try:
                    attempts.append(attempt)
                    time.sleep(1)
                except Exception:
                    continue
    assert attempts == [0]


def test_deadline_stops_slow_body_and_closes_connection():
    import time
    closed = []
    class Response:
        status = 200
        headers = {'Content-Type':'text/html'}
        def read(self, *a, **k): time.sleep(1)
        def close(self): closed.append('response')
    class Pool:
        def __init__(self, **kwargs): pass
        def urlopen(self, *a, **k): return Response()
        def close(self): closed.append('pool')
    with pytest.raises(ValueError, match='has not changed'):
        with web.recovery_budget(0.04):
            web.fetch_page('https://public.example', dns, Pool)
    assert closed == ['response','pool']
