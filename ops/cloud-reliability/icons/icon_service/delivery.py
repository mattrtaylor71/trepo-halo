"""Stable display contract. Renderer recipes and queues remain server concerns."""
import re
import os
from urllib.parse import urlsplit, urlunsplit
from .catalog import APP_STYLE, STYLE, ALIASES, key, term, validate_style

# These wire namespaces are contracts with installed clients, not model versions.
PATH = re.compile(r'/icons/(trepo-grocery-v1|trepo-app-bold-outline-v1)/([0-9a-f]{64})/([0-9a-f]{64})\.png')


def transport_url(url, requested_style):
    validate_style(requested_style)
    parsed = urlsplit(url)
    match = PATH.fullmatch(parsed.path)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment or not match:
        raise ValueError('Invalid catalog asset URL')
    cdn = os.getenv('ICON_CDN_HOST','').strip()
    if cdn:
        if not re.fullmatch(r'd[a-z0-9]+\.cloudfront\.net',cdn):
            raise ValueError('Invalid icon CDN host')
        parsed = parsed._replace(netloc=cdn)
    return urlunsplit(parsed._replace(path='/icons/' + requested_style + '/' + match[2] + '/' + match[3] + '.png'))


def lookup_many(store, names, style=APP_STYLE):
    """Two bounded batch reads instead of two network round trips per item."""
    normalized = list(dict.fromkeys(term(name) for name in names))
    aliases = store.get_many([key('alias', name, style) for name in normalized])
    concepts = {name: (aliases.get(key('alias', name, style)) or {}).get('concept', ALIASES.get(name, name)) for name in normalized}
    icons = store.get_many([key('icon', value, style) for value in concepts.values()])
    result = {}
    for name, canonical in concepts.items():
        icon = icons.get(key('icon', canonical, style))
        if icon and icon.get('status') == 'ready':
            result[name] = {'status': 'ready', 'concept': canonical, 'url': transport_url(icon['url'], APP_STYLE)}
    return result
