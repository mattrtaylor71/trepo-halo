"""Resolve one public Pinterest pin to its publisher; never turn a caption into a recipe."""
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

PIN_DOMAINS = ('pinterest.com', 'pinterest.co.uk', 'pinterest.ca', 'pinterest.de',
               'pinterest.fr', 'pinterest.es', 'pinterest.it', 'pinterest.com.au',
               'pinterest.com.mx', 'pinterest.com.br', 'pinterest.jp')
MESSAGE = "We couldn't find this Pin's recipe website. Open the recipe on Pinterest and share its website link, or paste the recipe text."
MAX_BYTES = 8 * 1024 * 1024

class PinterestSourceError(ValueError):
    pass


def is_pinterest(url):
    host = (urlparse(url).hostname or '').lower().rstrip('.')
    return host == 'pin.it' or any(host == d or host.endswith('.' + d) for d in PIN_DOMAINS)


def pin_id(url):
    if not is_pinterest(url):
        return None
    match = re.fullmatch(r'/pin/(?:[\w-]+--)?(\d+)/(?:sent/)?', urlparse(url).path.rstrip('/') + '/')
    return match.group(1) if match else None


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            attrs = dict(attrs)
            self.active = (attrs.get('type') == 'application/json'
                           or attrs.get('data-relay-completed-request') == 'true')
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.blocks.append(''.join(self.parts).strip())
            self.active = False


def _nodes(value):
    # Iterative walk: untrusted deeply nested data must not exhaust the Python stack.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            yield item
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def source_from_html(html, requested_id):
    if not requested_id or len(html) > MAX_BYTES:
        raise PinterestSourceError(MESSAGE)
    parser = _Scripts()
    parser.feed(html)
    decoder = json.JSONDecoder()
    candidates = set()
    for block in parser.blocks:
        try:
            prefix = 'window.__PWS_RELAY_REGISTER_COMPLETED_REQUEST__('
            if block.startswith(prefix):
                # Decode JSON arguments, never eval Pinterest's JavaScript.
                _, end = decoder.raw_decode(block[len(prefix):])
                rest = block[len(prefix) + end:].lstrip()
                if not rest.startswith(','):
                    continue
                value, _ = decoder.raw_decode(rest[1:].lstrip())
            else:
                value = json.loads(block)
        except (ValueError, RecursionError):
            continue
        for node in _nodes(value):
            identity = str(node.get('entityId') or node.get('id') or '')
            if identity == requested_id and isinstance(node.get('link'), str) and node['link'].strip():
                parsed = urlparse(node['link'].strip())
                candidates.add(urlunparse(parsed._replace(fragment='')))
    if len(candidates) != 1:
        raise PinterestSourceError(MESSAGE)
    source = candidates.pop()
    if is_pinterest(source):
        raise PinterestSourceError(MESSAGE)
    return source


def _public_url(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ''
        if (parsed.scheme not in ('https', 'http') or not host or parsed.username or parsed.password
                or parsed.port not in (None, 80, 443) or '\\' in url
                or any(ord(c) < 33 for c in url)):
            raise ValueError('Unsafe URL')
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == 'https' else 80), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ValueError('Non-public address')
    except (ValueError, OSError):
        raise PinterestSourceError(MESSAGE) from None
    return urlunparse(parsed._replace(fragment=''))


def _fetch(url):
    import requests
    # Pin destinations are untrusted URLs. Validate every redirect BEFORE fetching it.
    # No credentials/cookies, automatic redirects, or unbounded download.
    for _ in range(6):
        url = _public_url(url)
        with requests.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'text/html'},
                          allow_redirects=False, timeout=(5, 10), stream=True) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                destination = response.headers.get('Location')
                if not destination:
                    raise PinterestSourceError(MESSAGE)
                url = urljoin(url, destination)
                continue
            if response.status_code != 200:
                raise PinterestSourceError(MESSAGE)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' not in content_type:
                raise PinterestSourceError(MESSAGE)
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise PinterestSourceError(MESSAGE)
                chunks.append(chunk)
            return url, b''.join(chunks).decode('utf-8', errors='replace')
    raise PinterestSourceError(MESSAGE)


def resolve(url):
    """Fetch pin + exact publisher once each; callers parse the returned publisher HTML."""
    try:
        if not is_pinterest(url):
            raise PinterestSourceError(MESSAGE)
        requested_id = pin_id(url)
        pin_url, pin_html = _fetch(url)
        resolved_id = pin_id(pin_url)
        if not resolved_id or (requested_id and requested_id != resolved_id):
            raise PinterestSourceError(MESSAGE)
        source = source_from_html(pin_html, resolved_id)
        final_url, html = _fetch(source)
        if is_pinterest(final_url):
            raise PinterestSourceError(MESSAGE)
        return final_url, html
    except PinterestSourceError:
        raise
    except Exception as exc:
        # No upstream response, share token, or parser internals in user-facing errors.
        raise PinterestSourceError(MESSAGE) from exc
