"""Small, pinned-address fetcher used only for explicit recipe source recovery."""
import ipaddress
import socket
import signal
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlsplit, urljoin
import urllib3

MAX_BYTES = 2 * 1024 * 1024
_DNS_SLOT = threading.BoundedSemaphore(1)


def _bounded_dns(host, port, resolver, seconds=2):
    # getaddrinfo can remain in native code despite SIGALRM. One daemon worker
    # isolates that call; its only side effect is DNS, never network extraction
    # or persistence. A stuck resolver cannot accumulate additional workers.
    if not _DNS_SLOT.acquire(blocking=False):
        raise ValueError('Recipe lookup is busy. Please try again.')
    done, result = threading.Event(), {}
    def resolve():
        try:
            result['addresses'] = resolver(host, port, type=socket.SOCK_STREAM)
        except Exception as exc:
            result['error'] = exc
        finally:
            _DNS_SLOT.release()
            done.set()
    try:
        threading.Thread(target=resolve, daemon=True, name='recipe-recovery-dns').start()
    except RuntimeError:
        _DNS_SLOT.release()
        raise ValueError('Recipe lookup is unavailable. Please try again.') from None
    if not done.wait(seconds):
        raise ValueError('The recipe site lookup timed out. Try again or paste its recipe text.')
    if 'error' in result:
        raise ValueError('The recipe site could not be found. Check its URL or paste its recipe text.') from result['error']
    return result['addresses']


class _RecoveryDeadline(BaseException):
    """Escape provider retry handlers; caught only at the recovery boundary."""


@contextmanager
def recovery_budget(seconds=18):
    """Bound synchronous Lambda recovery, including DNS and trickling responses.

    Only extraction runs inside this boundary. Database writes start afterwards.
    A socket read timeout alone restarts on each byte and is not a wall deadline.
    Fail closed outside the Lambda main thread rather than leave background work.
    """
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        raise ValueError('Recipe recovery is unavailable. Please try again.')
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    budget = min(seconds, previous_timer[0]) if previous_timer[0] else seconds

    def expired(*_):
        raise _RecoveryDeadline()

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        yield
    except _RecoveryDeadline:
        raise ValueError('This is taking too long. Your saved recipe has not changed. Try again or paste its recipe text.') from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


def public_target(url, resolver=socket.getaddrinfo):
    parsed = urlsplit(url)
    if (parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username
            or parsed.password or parsed.port not in (None, 80, 443)
            or any(c.isspace() for c in parsed.netloc)):
        raise ValueError('Use a public recipe page URL.')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    addresses = list(dict.fromkeys(entry[4][0] for entry in _bounded_dns(parsed.hostname, port, resolver)))
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('Use a public recipe page URL.')
    return parsed, port, addresses[0]


def fetch_page(url, resolver=socket.getaddrinfo, pool_factory=None):
    with recovery_budget(8):
        return _fetch_page(url, resolver, pool_factory)


def _fetch_page(url, resolver, pool_factory):
    """Validate every redirect and pin the checked IP so DNS rebinding cannot win."""
    for _ in range(4):
        parsed, port, address = public_target(url, resolver)
        kwargs = {'host': address, 'port': port, 'timeout': urllib3.Timeout(connect=4, read=8), 'retries':False}
        if parsed.scheme == 'https':
            kwargs.update(server_hostname=parsed.hostname, assert_hostname=parsed.hostname)
        factory = pool_factory or (urllib3.HTTPSConnectionPool if parsed.scheme == 'https' else urllib3.HTTPConnectionPool)
        pool = factory(**kwargs)
        response = None
        try:
            path = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
            response = pool.urlopen('GET', path, headers={'Host':parsed.netloc, 'Accept':'text/html,application/xhtml+xml',
                'Accept-Encoding':'identity','User-Agent':'Trepo Recipe Import'}, redirect=False, preload_content=False)
            if response.status in (301,302,303,307,308):
                location = response.headers.get('Location')
                if not location: raise ValueError('The source redirect could not be followed.')
                url = urljoin(url, location)
                continue
            if response.status != 200: raise ValueError('The recipe page is unavailable. Paste its recipe text instead.')
            content_type = response.headers.get('Content-Type','').lower()
            if not any(t in content_type for t in ('text/html','application/xhtml+xml','text/plain')):
                raise ValueError('Use a recipe page, or paste its recipe text.')
            body = response.read(MAX_BYTES + 1, decode_content=True)
            if len(body) > MAX_BYTES: raise ValueError('This page is too large. Paste the recipe text instead.')
            return body.decode('utf-8', errors='replace'), url
        finally:
            if response: response.close()
            pool.close()
    raise ValueError('Too many redirects. Paste the recipe text instead.')
