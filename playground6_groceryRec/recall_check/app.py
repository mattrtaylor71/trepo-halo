# recall_check/app.py
# Standalone Lambda: match a user's checked-in kitchen items against USDA FSIS
# meat/poultry/egg recalls.
#
#   GET /recall-check/{owner}          -> run the check for that owner's LIVE kitchen
#   GET /recall-check/{owner}?refresh=1  force-refresh the cached FSIS feed
#   GET /recall-check/{owner}?probe=1    return raw sample recall records (debug/field discovery)
#   GET /recall-check/_feed              return the compact ACTIVE recall list (fleet-scan helper)
#
# FSIS covers MEAT, POULTRY, and EGG products only. Packaged/produce/seafood/FDA-
# regulated foods are NOT in this feed (the FDA food-enforcement API is the complement).
#
# Design notes:
#  - The FSIS feed is fetched with urllib (stdlib) on cold start and cached in a module
#    global with a 6h TTL. `?refresh=1` forces a re-fetch.
#  - Matching is two-tier ("likely" = brand/product alignment, "possible" = generic
#    item type appears in an active recall) via ONE batched gpt-4.1-mini call.
#  - A deterministic pre-filter (recent recalls + token overlap) runs first so the LLM
#    only sees plausible candidates and most checks skip the LLM entirely.

import os
import re
import json
import sys
import time
import html
import urllib.request
import urllib.error

try:
    import pymysql
except ImportError as exc:  # pragma: no cover
    pymysql = None
    _PYMYSQL_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FSIS_FEED_URL = os.getenv(
    'FSIS_FEED_URL', 'https://www.fsis.usda.gov/fsis/api/recall/v/1')
FEED_TTL_SECONDS = int(os.getenv('RECALL_FEED_TTL_SECONDS', str(6 * 3600)))
FEED_FETCH_TIMEOUT = int(os.getenv('RECALL_FEED_FETCH_TIMEOUT', '20'))
RECALL_LOOKBACK_DAYS = int(os.getenv('RECALL_LOOKBACK_DAYS', '365'))

DB_HOST = os.getenv('DB_HOST', 'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASS = os.getenv('DB_PASS', '')
DB_NAME = os.getenv('DB_NAME', 'mysqlTutorial')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_CONNECT_TIMEOUT = int(os.getenv('DB_CONNECT_TIMEOUT_SECONDS', '5'))
DB_READ_TIMEOUT = int(os.getenv('DB_READ_TIMEOUT_SECONDS', '10'))
KITCHEN_TABLE = os.getenv('SHARED_KITCHEN_TABLE', 'shared_kitchen')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MATCH_MODEL = os.getenv('RECALL_MATCH_MODEL', 'gpt-4.1-mini')
OPENAI_TIMEOUT_SECONDS = int(os.getenv('RECALL_OPENAI_TIMEOUT', '40'))
# Only send the LLM recalls that share a token with SOME kitchen item, and only items
# that overlap SOME recall. Belt-and-suspenders context bounds.
MAX_RECALLS_TO_LLM = int(os.getenv('RECALL_MAX_TO_LLM', '60'))

# Module-global feed cache (survives warm invocations).
_FEED_CACHE = {'fetched_at': 0.0, 'recalls': [], 'raw_sample': [], 'source_status': None}

# Tokens that carry no product-identity signal — dropped before overlap matching.
_NOISE_TOKENS = {
    'the', 'and', 'with', 'for', 'from', 'inc', 'llc', 'co', 'company', 'corp', 'ltd',
    'brand', 'brands', 'foods', 'food', 'product', 'products', 'item', 'items',
    'fresh', 'organic', 'natural', 'frozen', 'raw', 'cooked', 'ready', 'eat',
    'oz', 'lb', 'lbs', 'count', 'ct', 'pack', 'package', 'bag', 'box', 'case',
    'all', 'new', 'original', 'classic', 'style', 'flavor', 'size', 'net', 'wt',
    'usda', 'establishment', 'est', 'number', 'various', 'assorted',
}


# ---------------------------------------------------------------------------
# OpenAI helper (param-drop self-heal, copied from recipes_generator style)
# ---------------------------------------------------------------------------
def _create_chat(client, **kwargs):
    """chat.completions.create with self-healing param fallback: if a model rejects
    a param (max_tokens / custom temperature) with a 400 naming it, drop that exact
    param and retry so future model families self-heal."""
    for _ in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            param = None
            body = getattr(exc, 'body', None)
            if isinstance(body, dict):
                err = body.get('error') if isinstance(body.get('error'), dict) else body
                param = err.get('param') if isinstance(err, dict) else None
            if not param:
                param = getattr(exc, 'param', None)
            if not param:
                m = re.search(r"'([A-Za-z_]+)'", str(exc))
                if m:
                    param = m.group(1)
            if not param or param not in kwargs:
                raise
            print(json.dumps({'evt': 'openai_param_dropped',
                              'service': 'recall_check', 'param': param}))
            kwargs.pop(param, None)
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# FSIS feed fetch + normalization
# ---------------------------------------------------------------------------
def _strip_html(s):
    # Several FSIS fields are arrays (field_establishment, field_recall_reason,
    # field_product_items, field_states, field_processing); flatten to a string.
    if s is None:
        return ''
    if isinstance(s, (list, tuple)):
        s = '; '.join(_strip_html(x) for x in s if x not in (None, ''))
    s = re.sub(r'<[^>]+>', ' ', str(s))
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()


def _first(rec, *keys):
    """Return the first non-empty value among candidate keys (FSIS field naming
    varies across fields; be defensive)."""
    for k in keys:
        v = rec.get(k)
        if v not in (None, '', [], {}):
            return v
    return ''


def _is_active(rec):
    """FSIS marks live recalls via field_active_notice == 'True' (string). Fall back to
    an empty closed date."""
    act = str(_first(rec, 'field_active_notice')).strip().lower()
    if act in ('true', '1', 'yes'):
        return True
    if act in ('false', '0', 'no'):
        return False
    # Fallback: no closed date => still active.
    closed = _strip_html(_first(rec, 'field_closed_date', 'field_closed_date_1'))
    return closed == ''


def _recall_date(rec):
    raw = _strip_html(_first(rec, 'field_recall_date', 'field_recall_date_1',
                             'field_year', 'created', 'changed'))
    # Try to pull a YYYY-MM-DD.
    m = re.search(r'(\d{4}-\d{2}-\d{2})', raw)
    return m.group(1) if m else raw


def normalize_recall(rec):
    """Map a raw FSIS record onto our stable shape."""
    title = _strip_html(_first(rec, 'field_title', 'title'))
    product = _strip_html(_first(rec, 'field_product_items', 'field_products',
                                 'field_summary'))
    reason = _strip_html(_first(rec, 'field_recall_reason', 'field_reason',
                                'field_recall_reason_id'))
    classification = _strip_html(_first(rec, 'field_recall_classification',
                                        'field_risk_level', 'field_recall_type'))
    establishment = _strip_html(_first(rec, 'field_establishment', 'field_company',
                                       'field_company_media_contact',
                                       'field_processing'))
    states = _strip_html(_first(rec, 'field_states'))
    number = _strip_html(_first(rec, 'field_recall_number', 'field_recall_number_2'))
    link = _first(rec, 'field_press_release', 'field_recall_url', 'field_url', 'url')
    link = _strip_html(link)
    if link and link.startswith('/'):
        link = 'https://www.fsis.usda.gov' + link
    active = _is_active(rec)
    return {
        'title': title,
        'product_description': product,
        'reason': reason,
        'classification': classification,
        'status': 'Active' if active else 'Closed',
        'active': active,
        'date': _recall_date(rec),
        'link': link,
        'establishment': establishment,
        'states': states,
        'recall_number': number,
    }


_BROWSER_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'identity',
    'Referer': 'https://www.fsis.usda.gov/recalls',
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
}


def _http_get_json(url):
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=FEED_FETCH_TIMEOUT) as resp:
        data = resp.read()
    return json.loads(data.decode('utf-8', 'replace'))


def fetch_feed(force=False):
    """Fetch + cache the FSIS feed. Returns (recalls, meta)."""
    now = time.time()
    age = now - _FEED_CACHE['fetched_at']
    if not force and _FEED_CACHE['recalls'] and age < FEED_TTL_SECONDS:
        return _FEED_CACHE['recalls'], {'cache': 'hit', 'age_s': round(age, 1),
                                        'fetch_ms': 0}
    t0 = time.time()
    raw = _http_get_json(FSIS_FEED_URL)
    fetch_ms = int((time.time() - t0) * 1000)
    if isinstance(raw, dict):
        # Some Drupal JSON endpoints wrap rows in a top-level key.
        for k in ('results', 'data', 'rows', 'items'):
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
        else:
            raw = [raw]
    recalls = [normalize_recall(r) for r in raw if isinstance(r, dict)]
    _FEED_CACHE['fetched_at'] = now
    _FEED_CACHE['recalls'] = recalls
    _FEED_CACHE['raw_sample'] = [r for r in raw[:3] if isinstance(r, dict)]
    _FEED_CACHE['source_status'] = 'ok'
    return recalls, {'cache': 'miss', 'fetch_ms': fetch_ms, 'count': len(recalls)}


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def _connect():
    if pymysql is None:
        raise RuntimeError('pymysql unavailable: %r' % (_PYMYSQL_IMPORT_ERROR,))
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT,
        connect_timeout=DB_CONNECT_TIMEOUT, read_timeout=DB_READ_TIMEOUT,
        cursorclass=pymysql.cursors.DictCursor, autocommit=True)


def read_live_kitchen(owner):
    """Live rows follow the kitchen_api convention: action = 'IN'."""
    sql = ("SELECT _id, product_name, brand, variant, category "
           "FROM `%s` WHERE owner_id=%%s AND `action`='IN' "
           "ORDER BY `_createdDate` DESC" % KITCHEN_TABLE)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (owner,))
            return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _tokens(*parts):
    text = ' '.join(str(p) for p in parts if p)
    toks = re.findall(r'[a-z0-9]+', text.lower())
    return {t for t in toks if len(t) > 2 and t not in _NOISE_TOKENS}


def _recall_tokens(r):
    return _tokens(r['title'], r['product_description'], r['establishment'], r['reason'])


def prefilter(items, recalls):
    """Deterministic narrowing before the LLM:
      - keep only ACTIVE recalls within the lookback window,
      - keep only (item, recall) pairs with >=1 shared token,
      - return the surviving items and the union of candidate recalls.
    """
    cutoff = time.time() - RECALL_LOOKBACK_DAYS * 86400
    active = []
    for r in recalls:
        if not r['active']:
            continue
        d = r['date']
        keep = True
        if re.match(r'\d{4}-\d{2}-\d{2}', d or ''):
            try:
                keep = time.mktime(time.strptime(d[:10], '%Y-%m-%d')) >= cutoff
            except Exception:
                keep = True
        if keep:
            active.append(r)
    r_toks = [(_recall_tokens(r), r) for r in active]

    cand_items = []
    cand_recall_ids = set()
    for it in items:
        it_toks = _tokens(it.get('product_name'), it.get('brand'), it.get('variant'))
        if not it_toks:
            continue
        overlaps = [i for i, (rt, _) in enumerate(r_toks) if it_toks & rt]
        if overlaps:
            cand_items.append(it)
            cand_recall_ids.update(overlaps)
    cand_recalls = [r_toks[i][1] for i in sorted(cand_recall_ids)][:MAX_RECALLS_TO_LLM]
    return cand_items, cand_recalls, len(active)


_MATCH_SYSTEM = (
    "You are a food-safety matcher for a kitchen app. You are given a USER'S KITCHEN "
    "ITEMS and a list of ACTIVE USDA FSIS meat/poultry/egg RECALLS. For each kitchen "
    "item, decide whether it could be the SAME PRODUCT that was recalled.\n\n"
    "The core test is PRODUCT-FORM identity, not a shared word. The recalled product has "
    "a specific form/preparation (e.g. beef JERKY, ground beef, chicken nuggets, "
    "summer sausage, deli ham). A kitchen item only matches if it is that SAME specific "
    "form. Sharing a protein word like 'beef', 'chicken', or 'pork' is NOT enough: "
    "ground beef, steak, roast, carne asada, and beef sausage are DIFFERENT products "
    "from beef jerky and must NOT match a beef-jerky recall.\n\n"
    "Assign exactly one level per item:\n"
    "  \"likely\"  = the item's BRAND (or establishment) matches the recalling company/"
    "brand AND it is the same product form. Strong, specific match.\n"
    "  \"possible\" = the item is a GENERIC/unbranded (or different-brand) version of the "
    "SAME product form as a recalled product. Only use this when the form genuinely "
    "matches (e.g. unbranded 'beef jerky' vs a beef-jerky recall; unbranded 'ground "
    "beef' vs a ground-beef recall). 'Worth double-checking'.\n"
    "  \"none\"     = anything else: a different product form, a different food, a "
    "seasoning/marinade/packaging/non-food item, or only an incidental shared word.\n\n"
    "Worked examples (recall = 'STREET\\'S BEEF Jerky Teriyaki', a beef JERKY product):\n"
    "  - {name:'Teriyaki Beef Jerky', brand:'Street\\'s Beef'} -> likely (brand + jerky).\n"
    "  - {name:'Beef Jerky', brand:''} -> possible (same form: jerky, brand unknown).\n"
    "  - {name:'Jack Link\\'s Beef Jerky', brand:'Jack Link\\'s'} -> possible "
    "(jerky, but a different brand than the recall).\n"
    "  - {name:'Ground Beef'} -> none (ground beef is NOT jerky).\n"
    "  - {name:'Beef Tenderloin Steak'} -> none (steak is NOT jerky).\n"
    "  - {name:'Beef Smoked Sausage'} -> none (sausage is NOT jerky).\n"
    "  - {name:'Jerky Seasoning'} -> none (a seasoning, not a jerky product).\n"
    "  - {name:'Plastic Wrap'} -> none (not a food).\n\n"
    "Return STRICT JSON: {\"verdicts\":[{\"item_id\":\"...\","
    "\"level\":\"none|possible|likely\",\"recall_index\":<int or null>}]}. "
    "Include an entry for EVERY kitchen item. recall_index is the 0-based index into the "
    "RECALLS list for the matched recall, or null when level is none."
)


def run_llm_match(items, recalls):
    """One batched gpt-4.1-mini call. Returns {item_id: {level, recall_index}}."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)
    item_lines = [{
        'item_id': it['_id'],
        'name': it.get('product_name') or '',
        'brand': it.get('brand') or '',
        'variant': it.get('variant') or '',
    } for it in items]
    recall_lines = [{
        'index': i,
        'product': (r['product_description'] or r['title'])[:300],
        'establishment': r['establishment'][:120],
        'reason': r['reason'][:120],
    } for i, r in enumerate(recalls)]
    user = ("KITCHEN ITEMS:\n" + json.dumps(item_lines, ensure_ascii=False) +
            "\n\nACTIVE RECALLS:\n" + json.dumps(recall_lines, ensure_ascii=False))
    resp = _create_chat(
        client,
        model=OPENAI_MATCH_MODEL,
        temperature=0,
        response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': _MATCH_SYSTEM},
                  {'role': 'user', 'content': user}],
    )
    content = resp.choices[0].message.content or '{}'
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {}
    out = {}
    for v in (parsed.get('verdicts') or []):
        iid = v.get('item_id')
        lvl = str(v.get('level') or 'none').lower()
        if iid and lvl in ('possible', 'likely'):
            out[iid] = {'level': lvl, 'recall_index': v.get('recall_index')}
    return out


def check_owner(owner, recalls, item_override=None):
    """Core check for one owner. Returns (matches, stats). Read-only."""
    items = item_override if item_override is not None else read_live_kitchen(owner)
    cand_items, cand_recalls, active_count = prefilter(items, recalls)
    llm_ms = 0
    verdicts = {}
    if cand_items and cand_recalls:
        t0 = time.time()
        verdicts = run_llm_match(cand_items, cand_recalls)
        llm_ms = int((time.time() - t0) * 1000)
    by_id = {it['_id']: it for it in items}
    matches = []
    for iid, v in verdicts.items():
        it = by_id.get(iid)
        if not it:
            continue
        ri = v.get('recall_index')
        if not isinstance(ri, int) or ri < 0 or ri >= len(cand_recalls):
            continue
        r = cand_recalls[ri]
        matches.append({
            'kitchen_item_id': iid,
            'kitchen_item_name': it.get('product_name'),
            'brand': it.get('brand'),
            'match_level': v['level'],
            'recall': {
                'title': r['title'],
                'product_description': r['product_description'],
                'reason': r['reason'],
                'classification': r['classification'],
                'status': r['status'],
                'date': r['date'],
                'link': r['link'],
                'establishment': r['establishment'],
            },
        })
    # likely first, then possible
    matches.sort(key=lambda m: 0 if m['match_level'] == 'likely' else 1)
    stats = {
        'kitchen_items': len(items),
        'candidate_items': len(cand_items),
        'candidate_recalls': len(cand_recalls),
        'active_recalls': active_count,
        'llm_called': bool(cand_items and cand_recalls),
        'llm_ms': llm_ms,
    }
    return matches, stats


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def _resp(code, body):
    return {'statusCode': code,
            'headers': {'content-type': 'application/json',
                        'access-control-allow-origin': '*'},
            'body': json.dumps(body, ensure_ascii=False, default=str)}


def _iso(ts):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))


def lambda_handler(event, context):
    t_start = time.time()
    params = (event or {}).get('pathParameters') or {}
    qs = (event or {}).get('queryStringParameters') or {}
    owner = params.get('owner')
    force = str(qs.get('refresh', '')).lower() in ('1', 'true', 'yes')

    try:
        recalls, feed_meta = fetch_feed(force=force)
    except urllib.error.HTTPError as e:
        return _resp(502, {'error': 'fsis_fetch_failed', 'http_status': e.code,
                           'detail': str(e)})
    except Exception as e:
        return _resp(502, {'error': 'fsis_fetch_failed', 'detail': str(e)})

    # Debug: raw sample of records (field discovery / reachability probe).
    if str(qs.get('probe', '')).lower() in ('1', 'true', 'yes'):
        return _resp(200, {
            'feed_meta': feed_meta,
            'total_records': len(recalls),
            'active_records': sum(1 for r in recalls if r['active']),
            'raw_keys': sorted(_FEED_CACHE['raw_sample'][0].keys())
            if _FEED_CACHE['raw_sample'] else [],
            'raw_sample': _FEED_CACHE['raw_sample'][:2],
            'normalized_sample': [normalize_recall(r)
                                  for r in _FEED_CACHE['raw_sample'][:2]],
        })

    # Fleet-scan helper: return compact ACTIVE recall list (no owner needed).
    if owner in (None, '_feed'):
        active = [r for r in recalls if r['active']]
        return _resp(200, {'recall_feed_date': _iso(_FEED_CACHE['fetched_at']),
                           'active_recalls': len(active), 'recalls': active})

    try:
        matches, stats = check_owner(owner, recalls)
    except Exception as e:
        return _resp(500, {'error': 'check_failed', 'detail': str(e)})

    return _resp(200, {
        'checked_at': _iso(t_start),
        'recall_feed_date': _iso(_FEED_CACHE['fetched_at']),
        'matches': matches,
        'feed_stats': {'active_recalls': stats['active_recalls']},
        'timing': {
            'feed_fetch_ms': feed_meta.get('fetch_ms', 0),
            'feed_cache': feed_meta.get('cache'),
            'llm_ms': stats['llm_ms'],
            'llm_called': stats['llm_called'],
            'total_ms': int((time.time() - t_start) * 1000),
        },
        'debug': stats,
    })
