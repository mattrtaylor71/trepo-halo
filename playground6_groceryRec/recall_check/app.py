# recall_check/app.py
# Standalone Lambda: match a user's checked-in kitchen items against a MERGED,
# full-US food-recall feed (USDA FSIS + FDA), persisted in MySQL (`recall_feed`).
#
#   GET  /recall-check/{owner}            -> run the check for that owner's LIVE kitchen
#   GET  /recall-check/{owner}?probe=1    -> feed diagnostics (per-source + merge stats)
#   GET  /recall-check/_feed              -> compact ACTIVE recall list (fleet-scan helper)
#   EventBridge/manual {action:"refresh_feed"} -> refresh + re-merge the persisted feed
#
# SOURCES (merged + de-duplicated):
#  - FSIS recall API           meat/poultry/egg, same-day-ish; needs browser headers
#                              (Akamai 403s default UAs). COMPLETENESS for USDA products.
#  - openFDA food/enforcement  FDA-regulated food/supplements; weekly, post-classification.
#                              COMPLETENESS layer (FDA says don't use as the alert trigger).
#  - FDA press-release XLSX     same-day FDA announcements. FAST/freshness layer.
#  - CDC combined food-safety RSS  aggregates FDA+FSIS, rolling ~20 items. FAST layer only.
#
# Design notes:
#  - The merged feed lives in the `recall_feed` table (built by refresh_feed); the matcher
#    reads it. openFDA is only re-pulled when its export_date advances (~weekly); other
#    sources are pulled every refresh. A stale table (>26h) triggers an inline refresh.
#  - DEDUPE: union-find on normalized recall_number, else fuzzy (company + product tokens
#    within a date window). The richest member is kept and back-filled; sources tracked.
#  - Matching is two-tier ("likely" = brand+form, "possible" = generic same-form) via ONE
#    batched gpt-4.1-mini call, after a deterministic recent+token-overlap pre-filter.

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
OPENFDA_URL = os.getenv('OPENFDA_URL', 'https://api.fda.gov/food/enforcement.json')
OPENFDA_DOWNLOAD_URL = os.getenv('OPENFDA_DOWNLOAD_URL', 'https://api.fda.gov/download.json')
OPENFDA_API_KEY = os.getenv('OPENFDA_API_KEY', '')  # optional; keyless works
FDA_XLSX_URL = os.getenv(
    'FDA_XLSX_URL',
    'https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/datatables-data?_format=xlsx')
CDC_RSS_URL = os.getenv('CDC_RSS_URL', 'https://tools.cdc.gov/api/v2/resources/media/316422.rss')

FEED_TTL_SECONDS = int(os.getenv('RECALL_FEED_TTL_SECONDS', str(6 * 3600)))
FEED_FETCH_TIMEOUT = int(os.getenv('RECALL_FEED_FETCH_TIMEOUT', '30'))
RECALL_LOOKBACK_DAYS = int(os.getenv('RECALL_LOOKBACK_DAYS', '365'))
# How far back to ingest FDA sources into the feed table (matcher still applies its own
# RECALL_LOOKBACK_DAYS window). openFDA/XLSX carry years of history; cap ingestion.
FEED_INGEST_DAYS = int(os.getenv('RECALL_FEED_INGEST_DAYS', '550'))  # ~18 months
FEED_STALE_SECONDS = int(os.getenv('RECALL_FEED_STALE_SECONDS', str(26 * 3600)))
FEED_TABLE = os.getenv('RECALL_FEED_TABLE', 'recall_feed')
FEED_META_TABLE = os.getenv('RECALL_FEED_META_TABLE', 'recall_feed_meta')

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
# Per-call timeout. A single batched call over a big kitchen (100+ items) generating a
# verdict per item is SUPERLINEAR and hung past 60s -> we now chunk items and run the
# batches concurrently, with retries OFF so one slow batch can't blow the Lambda budget.
OPENAI_TIMEOUT_SECONDS = int(os.getenv('RECALL_OPENAI_TIMEOUT', '22'))
OPENAI_MAX_RETRIES = int(os.getenv('RECALL_OPENAI_MAX_RETRIES', '0'))
# Items per LLM call (latency is ~linear in item count: ~7s@15, ~11s@25, ~19s@40; 104
# in one call hung >120s). Small batches keep each call fast; batches run concurrently.
LLM_ITEMS_PER_BATCH = int(os.getenv('RECALL_LLM_ITEMS_PER_BATCH', '14'))
LLM_MAX_WORKERS = int(os.getenv('RECALL_LLM_MAX_WORKERS', '8'))
# Hard wall-clock ceiling for the whole matching step; batches still pending when this
# elapses are abandoned and we return partial results (never hang the request).
LLM_TOTAL_DEADLINE_S = int(os.getenv('RECALL_LLM_TOTAL_DEADLINE_S', '35'))
# Per-BATCH recall cap (each batch is sent only the recalls its own items overlap).
MAX_RECALLS_TO_LLM = int(os.getenv('RECALL_MAX_TO_LLM', '40'))
# Global cap on the candidate recall pool the prefilter hands to the matcher (kept high
# so no relevant recall is dropped; per-batch subsetting controls actual payload size).
MAX_CANDIDATE_RECALLS = int(os.getenv('RECALL_MAX_CANDIDATE_RECALLS', '250'))
# Absolute cap on candidate items sent to the LLM in one request (protects against a
# pathologically large kitchen). Most-recently-added items win (prefilter preserves order).
MAX_CANDIDATE_ITEMS = int(os.getenv('RECALL_MAX_CANDIDATE_ITEMS', '120'))

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


def _record(source, source_id, title='', product_description='', reason='',
            classification='', status='', active=True, date='', link='',
            establishment='', brand='', states='', product_type='',
            recall_number='', raw=None):
    """Build the common cross-source recall record shape (matcher + table use this)."""
    return {
        'source': source,
        'source_id': str(source_id or '')[:120],
        'recall_number': str(recall_number or '')[:120],
        'brand': brand or '',
        'title': title or product_description,
        'product_description': product_description,
        'reason': reason,
        'classification': classification,
        'status': status,
        'active': bool(active),
        'date': date or '',
        'link': link or '',
        'establishment': establishment or '',
        'states': states or '',
        'product_type': product_type or '',
        'sources': [source],
        'raw': raw if raw is not None else {},
    }


def normalize_recall(rec):
    """Map a raw FSIS record onto the common shape (source='FSIS')."""
    number = _strip_html(_first(rec, 'field_recall_number', 'field_recall_number_2'))
    link = _strip_html(_first(rec, 'field_recall_url', 'field_press_release',
                              'field_url', 'url'))
    if link and link.startswith('/'):
        link = 'https://www.fsis.usda.gov' + link
    active = _is_active(rec)
    return _record(
        source='FSIS',
        source_id=number or _strip_html(_first(rec, 'field_title'))[:120],
        recall_number=number,
        title=_strip_html(_first(rec, 'field_title', 'title')),
        product_description=_strip_html(_first(rec, 'field_product_items',
                                               'field_products', 'field_summary')),
        reason=_strip_html(_first(rec, 'field_recall_reason', 'field_reason')),
        classification=_strip_html(_first(rec, 'field_recall_classification',
                                          'field_risk_level', 'field_recall_type')),
        establishment=_strip_html(_first(rec, 'field_establishment', 'field_company',
                                         'field_company_media_contact')),
        states=_strip_html(_first(rec, 'field_states')),
        status='Active' if active else 'Closed',
        active=active,
        date=_recall_date(rec),
        link=link,
        product_type='Meat/Poultry/Egg',
        raw={'establishment': _strip_html(_first(rec, 'field_establishment'))},
    )


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


def _http_get(url, headers=None, timeout=None):
    req = urllib.request.Request(url, headers=headers or _BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout or FEED_FETCH_TIMEOUT) as resp:
        return resp.read()


def _http_get_json(url, headers=None):
    return json.loads(_http_get(url, headers=headers).decode('utf-8', 'replace'))


def _ingest_cutoff():
    return time.time() - FEED_INGEST_DAYS * 86400


def _date_ok(date_str):
    """True if date_str (YYYY-MM-DD) is within the ingest window (or undated)."""
    if not date_str or not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return True
    try:
        return time.mktime(time.strptime(date_str[:10], '%Y-%m-%d')) >= _ingest_cutoff()
    except Exception:
        return True


# ---- Source: FSIS ---------------------------------------------------------
def fetch_fsis():
    raw = _http_get_json(FSIS_FEED_URL)
    if isinstance(raw, dict):
        for k in ('results', 'data', 'rows', 'items'):
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
        else:
            raw = [raw]
    return [normalize_recall(r) for r in raw if isinstance(r, dict)]


# ---- Source: openFDA food enforcement -------------------------------------
def _openfda_export_date():
    """The export_date on the food/enforcement partition — advances ~weekly."""
    try:
        d = _http_get_json(OPENFDA_DOWNLOAD_URL)
        return d['results']['food']['enforcement'].get('export_date', '')
    except Exception:
        return ''


def _yyyymmdd_to_iso(s):
    s = re.sub(r'\D', '', str(s or ''))
    return '%s-%s-%s' % (s[0:4], s[4:6], s[6:8]) if len(s) >= 8 else ''


def fetch_openfda():
    """Recent + Ongoing FDA food/supplement enforcement reports, paginated."""
    start = time.strftime('%Y%m%d', time.gmtime(_ingest_cutoff()))
    end = time.strftime('%Y%m%d', time.gmtime(time.time() + 86400))
    out = []
    skip = 0
    limit = 1000
    for _ in range(20):  # hard page cap (<=20k rows)
        params = ('?search=report_date:[%s+TO+%s]+AND+status:Ongoing'
                  '&limit=%d&skip=%d' % (start, end, limit, skip))
        if OPENFDA_API_KEY:
            params += '&api_key=' + OPENFDA_API_KEY
        try:
            d = _http_get_json(OPENFDA_URL + params)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # openFDA returns 404 when skip past the end
                break
            raise
        results = d.get('results') or []
        for r in results:
            iso = _yyyymmdd_to_iso(r.get('report_date'))
            out.append(_record(
                source='openFDA',
                source_id=r.get('recall_number') or r.get('event_id'),
                recall_number=r.get('recall_number', ''),
                product_description=r.get('product_description', ''),
                reason=r.get('reason_for_recall', ''),
                classification=r.get('classification', ''),
                establishment=r.get('recalling_firm', ''),
                states=r.get('distribution_pattern', ''),
                status=r.get('status', ''),
                active=str(r.get('status', '')).strip().lower() == 'ongoing',
                date=iso,
                link='',
                product_type=r.get('product_type', 'Food'),
                raw={'code_info': r.get('code_info', '')},
            ))
        if len(results) < limit:
            break
        skip += limit
    return out


# ---- Source: FDA press-release XLSX ---------------------------------------
def _parse_xlsx_rows(data):
    """Minimal .xlsx reader (avoids the openpyxl dep): unzip, read sharedStrings +
    sheet1, yield rows of cell strings."""
    import zipfile
    import io
    import xml.etree.ElementTree as ET
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            st = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in st.findall('%ssi' % ns):
                shared.append(''.join(t.text or '' for t in si.iter('%st' % ns)))
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        for row in sheet.iter('%srow' % ns):
            cells = {}
            maxc = 0
            for c in row.findall('%sc' % ns):
                ref = c.get('r', '')
                col = re.match(r'[A-Z]+', ref)
                ci = 0
                for ch in (col.group(0) if col else 'A'):
                    ci = ci * 26 + (ord(ch) - 64)
                ci -= 1
                v = c.find('%sv' % ns)
                text = ''
                if v is not None and v.text is not None:
                    text = shared[int(v.text)] if c.get('t') == 's' else v.text
                cells[ci] = text
                maxc = max(maxc, ci)
            yield [cells.get(i, '') for i in range(maxc + 1)]


def fetch_fda_xlsx():
    data = _http_get(FDA_XLSX_URL, headers={'User-Agent': _BROWSER_HEADERS['User-Agent'],
                                            'Accept': '*/*'})
    rows = list(_parse_xlsx_rows(data))
    if not rows:
        return []
    hdr = [str(h).strip().lower() for h in rows[0]]

    def col(*names):
        for n in names:
            if n in hdr:
                return hdr.index(n)
        return -1
    i_date = col('date')
    i_brand = col('brand-names', 'brand names', 'brand-name(s)')
    i_prod = col('product-description', 'product description')
    i_type = col('product-types', 'product type', 'product-type')
    i_reason = col('recall-reason-description', 'recall reason description')
    i_co = col('company-name', 'company name')
    i_term = col('terminated recall', 'terminated')
    out = []

    def g(r, i):
        return (r[i].strip() if 0 <= i < len(r) and r[i] else '')
    for r in rows[1:]:
        ptype = g(r, i_type)
        if not ('food' in ptype.lower() or 'dietary' in ptype.lower()):
            continue  # skip pet food / devices / drugs / cosmetics
        # Date is MM/DD/YYYY.
        iso = ''
        m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', g(r, i_date))
        if m:
            iso = '%s-%02d-%02d' % (m.group(3), int(m.group(1)), int(m.group(2)))
        if not _date_ok(iso):
            continue
        terminated = g(r, i_term).lower().startswith('terminat')
        brand = g(r, i_brand)
        prod = g(r, i_prod)
        out.append(_record(
            source='FDA-press',
            source_id=(brand + '|' + prod + '|' + iso)[:120],
            brand=brand,
            product_description=prod,
            reason=g(r, i_reason),
            establishment=g(r, i_co),
            status='Terminated' if terminated else 'Ongoing',
            active=not terminated,
            date=iso,
            product_type=ptype,
            link='https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts',
        ))
    return out


# ---- Source: CDC combined food-safety RSS ---------------------------------
def fetch_cdc_rss():
    import xml.etree.ElementTree as ET
    data = _http_get(CDC_RSS_URL, headers={'User-Agent': _BROWSER_HEADERS['User-Agent'],
                                           'Accept': '*/*'})
    root = ET.fromstring(data)
    out = []
    for it in root.iter('item'):
        def t(tag):
            e = it.find(tag)
            return (e.text or '').strip() if e is not None else ''
        title = t('title')
        iso = ''
        pd = t('pubDate')  # e.g. 'Mon, 13 Jul 2026 21:46:00 GMT'
        m = re.search(r'(\d{1,2})\s+(\w{3})\s+(\d{4})', pd)
        if m:
            mm = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7,
                  'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}.get(
                      m.group(2).lower(), 0)
            if mm:
                iso = '%s-%02d-%02d' % (m.group(3), mm, int(m.group(1)))
        # RSS titles are "<Company> Issues Recall of <Product> Due to <Reason>".
        est = title.split(' Issues')[0].split(' Recalls')[0].strip()
        out.append(_record(
            source='CDC-RSS',
            source_id=t('guid') or title[:120],
            title=title,
            product_description=t('description') or title,
            establishment=est,
            status='Announced',
            active=True,
            date=iso,
            link=t('link'),
            product_type='Food',
        ))
    return out


# ---------------------------------------------------------------------------
# Non-human-food filter (openFDA food/enforcement includes pet/animal feed; drop it —
# users' kitchens are human food, and "Ground Beef for Dogs" was matching human meat).
# ---------------------------------------------------------------------------
_PETFOOD_RE = re.compile(
    r'\b(for dogs|for cats|dog food|cat food|pet food|puppy|kitten|canine|feline|'
    r'dog treat|cat treat|pet treat|animal feed|livestock|equine|poultry feed|'
    r'wild bird|birdseed|bird seed|aquarium|veterinary)\b', re.I)


def _is_non_human_food(r):
    ptype = (r.get('product_type') or '').lower()
    if 'animal' in ptype or 'veterinary' in ptype or 'pet' in ptype:
        return True
    blob = (r.get('product_description', '') + ' ' + r.get('title', '') + ' ' +
            r.get('reason', ''))
    return bool(_PETFOOD_RE.search(blob))


# ---------------------------------------------------------------------------
# Merge / dedupe
# ---------------------------------------------------------------------------
def _norm_num(n):
    return re.sub(r'[^a-z0-9]', '', str(n or '').lower())


def _company_tokens(r):
    return _tokens(r.get('brand'), r.get('establishment'))


def _product_tokens(r):
    return _tokens(r.get('product_description'), r.get('title'))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _date_diff_days(a, b):
    if not a or not b:
        return 9999
    try:
        ta = time.mktime(time.strptime(a[:10], '%Y-%m-%d'))
        tb = time.mktime(time.strptime(b[:10], '%Y-%m-%d'))
        return abs(ta - tb) / 86400.0
    except Exception:
        return 9999


def _fuzzy_same(x, y):
    """Same recall across sources w/o a shared number: strong company + product token
    overlap. Press-release vs post-classification dates can be weeks apart, so a very
    strong match tolerates a wider window; a moderate match needs the recall within
    ~3 days (per spec)."""
    cj = _jaccard(_company_tokens(x), _company_tokens(y))
    pj = _jaccard(_product_tokens(x), _product_tokens(y))
    if cj >= 0.6 and pj >= 0.6:
        return _date_diff_days(x['date'], y['date']) <= 60
    if cj >= 0.5 and pj >= 0.4:
        return _date_diff_days(x['date'], y['date']) <= 3
    return False


def _richness(r):
    return sum(1 for k in ('recall_number', 'brand', 'classification', 'reason',
                           'link', 'establishment', 'states') if r.get(k))


def merge_recalls(records):
    """Union-find dedupe across sources. Returns merged records with a `sources` list
    and back-filled fields (richest member kept as the base)."""
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Pass 1: identical recall_number.
    bynum = {}
    for i, r in enumerate(records):
        nn = _norm_num(r.get('recall_number'))
        if nn:
            bynum.setdefault(nn, []).append(i)
    for idxs in bynum.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # Pass 2: fuzzy company+product+date, bucketed by a company token to bound work.
    buckets = {}
    for i, r in enumerate(records):
        for tkn in list(_company_tokens(r))[:4]:
            buckets.setdefault(tkn, []).append(i)
    for idxs in buckets.values():
        if len(idxs) > 400:      # skip pathological mega-buckets (a common word)
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if find(i) != find(j) and _fuzzy_same(records[i], records[j]):
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        recs = [records[i] for i in members]
        base = dict(max(recs, key=_richness))
        srcs = []
        for r in recs:
            for s in r['sources']:
                if s not in srcs:
                    srcs.append(s)
        base['sources'] = srcs
        base['active'] = any(r['active'] for r in recs)  # active if ANY source active
        # Back-fill empty fields from other members; prefer non-empty recall_number/link.
        for k in ('recall_number', 'brand', 'link', 'classification', 'reason',
                  'establishment', 'states', 'product_type', 'date'):
            if not base.get(k):
                for r in recs:
                    if r.get(k):
                        base[k] = r[k]
                        break
        # Preserve ALL distinct product text across members (a single recall_number can
        # cover several product lines) so matching signal is not lost to the collapse.
        if len(recs) > 1:
            seen_p = set()
            parts = []
            for r in recs:
                p = (r.get('product_description') or '').strip()
                key = p.lower()[:60]
                if p and key not in seen_p:
                    seen_p.add(key)
                    parts.append(p)
            if parts:
                base['product_description'] = ' | '.join(parts)[:2000]
        merged.append(base)
    return merged


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
# Feed persistence (recall_feed table) + refresh orchestration
# ---------------------------------------------------------------------------
def _meta_get(cur, k):
    cur.execute("SELECT v FROM `%s` WHERE k=%%s" % FEED_META_TABLE, (k,))
    row = cur.fetchone()
    return row['v'] if row else None


def _meta_set(cur, k, v):
    cur.execute(
        "INSERT INTO `%s` (k, v, updated_at) VALUES (%%s, %%s, UTC_TIMESTAMP()) "
        "ON DUPLICATE KEY UPDATE v=VALUES(v), updated_at=VALUES(updated_at)"
        % FEED_META_TABLE, (k, str(v)[:255]))


def _feed_id(r):
    """Stable primary key for a merged record: recall_number if present, else a hash of
    company+product tokens (source_id as last resort)."""
    nn = _norm_num(r.get('recall_number'))
    if nn:
        return ('rn:' + nn)[:80]
    import hashlib
    sig = '|'.join(sorted(_company_tokens(r))) + '#' + '|'.join(sorted(_product_tokens(r)))
    if sig.strip('|#'):
        return 'fz:' + hashlib.sha1(sig.encode('utf-8')).hexdigest()[:60]
    return ('sid:' + r['source'] + ':' + r['source_id'])[:80]


def write_feed_table(merged):
    """Full-replace the recall_feed table transactionally (feed is small)."""
    conn = _connect()
    conn.autocommit(False)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM `%s`" % FEED_TABLE)
            rows = []
            seen = set()
            for r in merged:
                fid = _feed_id(r)
                if fid in seen:      # collapse any residual id collisions
                    continue
                seen.add(fid)
                rd = r['date'] if re.match(r'\d{4}-\d{2}-\d{2}', r.get('date') or '') else None
                rows.append((
                    fid, ','.join(r['sources'])[:255], r.get('recall_number', '')[:120],
                    (r.get('brand') or '')[:500], r.get('product_description', ''),
                    (r.get('reason') or '')[:1200], (r.get('classification') or '')[:120],
                    (r.get('status') or '')[:60], 1 if r['active'] else 0, rd,
                    (r.get('link') or '')[:1200], (r.get('establishment') or '')[:600],
                    (r.get('states') or '')[:600], (r.get('product_type') or '')[:255],
                    json.dumps(r.get('raw') or {})[:60000],
                ))
            cur.executemany(
                "INSERT INTO `%s` (id, sources, recall_number, brand, product_description,"
                " reason, classification, status, active, recall_date, link, establishment,"
                " states, product_type, raw, updated_at) VALUES "
                "(%s)" % (FEED_TABLE, ','.join(['%s'] * 15) + ',UTC_TIMESTAMP()'), rows)
            _meta_set(cur, 'last_refresh_ts', str(int(time.time())))
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_feed_table(active_only=True, recent_days=None):
    """Load merged recalls from the table into the common shape."""
    where = []
    if active_only:
        where.append('active=1')
    if recent_days:
        cutoff = time.strftime('%Y-%m-%d',
                               time.gmtime(time.time() - recent_days * 86400))
        where.append("(recall_date IS NULL OR recall_date >= '%s')" % cutoff)
    sql = "SELECT * FROM `%s`" % FEED_TABLE
    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            out = []
            for row in cur.fetchall():
                d = row['recall_date']
                out.append({
                    'source': (row['sources'] or '').split(',')[0],
                    'sources': (row['sources'] or '').split(',') if row['sources'] else [],
                    'recall_number': row['recall_number'] or '',
                    'brand': row['brand'] or '',
                    'title': row['product_description'] or '',
                    'product_description': row['product_description'] or '',
                    'reason': row['reason'] or '',
                    'classification': row['classification'] or '',
                    'status': row['status'] or '',
                    'active': bool(row['active']),
                    'date': d.strftime('%Y-%m-%d') if d else '',
                    'link': row['link'] or '',
                    'establishment': row['establishment'] or '',
                    'states': row['states'] or '',
                    'product_type': row['product_type'] or '',
                })
            return out
    finally:
        conn.close()


def _feed_last_refresh_age():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            ts = _meta_get(cur, 'last_refresh_ts')
            cur.execute("SELECT COUNT(*) AS c FROM `%s`" % FEED_TABLE)
            count = cur.fetchone()['c']
        return (time.time() - float(ts)) if ts else None, count
    finally:
        conn.close()


_SOURCE_FETCHERS = [('FSIS', fetch_fsis), ('openFDA', fetch_openfda),
                    ('FDA-press', fetch_fda_xlsx), ('CDC-RSS', fetch_cdc_rss)]


def refresh_feed(force_openfda=False):
    """Fetch every source, merge/dedupe, and persist. openFDA is only re-pulled when its
    export_date advances (else its rows are carried over from the table). Returns stats."""
    per_source = {}
    latency = {}
    records = []

    # openFDA gating by export_date.
    prev_export = None
    conn = _connect()
    try:
        with conn.cursor() as cur:
            prev_export = _meta_get(cur, 'openfda_export_date')
    finally:
        conn.close()
    cur_export = _openfda_export_date()
    pull_openfda = force_openfda or (cur_export and cur_export != prev_export)

    for name, fn in _SOURCE_FETCHERS:
        if name == 'openFDA' and not pull_openfda:
            carried = [r for r in read_feed_table(active_only=False)
                       if 'openFDA' in r.get('sources', [])]
            for r in carried:
                r['sources'] = ['openFDA']
                r['source'] = 'openFDA'
                r['raw'] = {}
            records.extend(carried)
            per_source[name] = len(carried)
            latency[name] = 0
            continue
        t0 = time.time()
        try:
            recs = fn()
            per_source[name] = len(recs)
            records.extend(recs)
        except Exception as e:
            per_source[name] = 'ERROR: %s' % (str(e)[:120],)
            print(json.dumps({'evt': 'recall_source_error', 'source': name,
                              'error': str(e)[:300]}), file=sys.stderr)
        latency[name] = int((time.time() - t0) * 1000)

    pre = len(records)
    records = [r for r in records if not _is_non_human_food(r)]
    dropped_nonfood = pre - len(records)
    merged = merge_recalls(records)
    written = write_feed_table(merged)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            if pull_openfda and cur_export:
                _meta_set(cur, 'openfda_export_date', cur_export)
    finally:
        conn.close()

    active = sum(1 for r in merged if r['active'])
    stats = {'per_source': per_source, 'latency_ms': latency, 'pre_merge': pre,
             'dropped_non_human_food': dropped_nonfood,
             'merged_total': written, 'deduped': (pre - dropped_nonfood) - written,
             'active': active, 'openfda_pulled': bool(pull_openfda),
             'openfda_export_date': cur_export}
    print(json.dumps({'evt': 'recall_feed_refreshed', **{
        'src_' + k: v for k, v in per_source.items()},
        'merged_total': written, 'deduped': pre - written, 'active': active}))
    return stats


# Warm-container guard so a burst of GETs during a stale window fires at most one
# self-invoke per container per throttle window.
_LAST_ASYNC_TRIGGER = {'ts': 0.0}
ASYNC_TRIGGER_THROTTLE_S = int(os.getenv('RECALL_ASYNC_TRIGGER_THROTTLE_S', '600'))


def _async_self_refresh(context):
    """Fire-and-forget: invoke THIS function asynchronously with {action:refresh_feed}
    so a user GET never blocks on a multi-source fetch. Throttled + best-effort; a
    failure here must never affect the GET response."""
    now = time.time()
    if now - _LAST_ASYNC_TRIGGER['ts'] < ASYNC_TRIGGER_THROTTLE_S:
        return
    fn_name = (getattr(context, 'function_name', None)
               or os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    if not fn_name:
        return
    _LAST_ASYNC_TRIGGER['ts'] = now
    try:
        import boto3
        boto3.client('lambda').invoke(
            FunctionName=fn_name, InvocationType='Event',
            Payload=json.dumps({'action': 'refresh_feed', 'trigger': 'stale_get'}).encode())
        print(json.dumps({'evt': 'recall_feed_async_refresh_fired', 'fn': fn_name}))
    except Exception as e:
        print(json.dumps({'evt': 'recall_feed_async_refresh_failed',
                          'error': str(e)[:300]}), file=sys.stderr)


def read_active_feed_nonblocking(context):
    """The feed the GET path reads. STRICTLY READ-ONLY: never fetches sources inline.
    If the table is empty or stale (>26h) it serves whatever is present (possibly stale)
    and fires an async self-invoke to rebuild in the background.
    Returns (recalls, age_seconds_or_None, total_row_count)."""
    age, count = _feed_last_refresh_age()
    if count == 0 or age is None or age > FEED_STALE_SECONDS:
        _async_self_refresh(context)
    recalls = read_feed_table(active_only=True, recent_days=RECALL_LOOKBACK_DAYS)
    return recalls, age, count


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
    cand_items = cand_items[:MAX_CANDIDATE_ITEMS]  # bound worst-case LLM work
    cand_recalls = [r_toks[i][1] for i in sorted(cand_recall_ids)][:MAX_CANDIDATE_RECALLS]
    return cand_items, cand_recalls, len(active)


_MATCH_SYSTEM = (
    "You are a food-safety matcher for a kitchen app. You are given a USER'S KITCHEN "
    "ITEMS and a list of ACTIVE US FOOD RECALLS (USDA FSIS meat/poultry/egg + FDA food, "
    "produce, seafood, packaged & supplements). For each kitchen item, decide whether it "
    "could be the SAME PRODUCT that was recalled.\n\n"
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
    "Each kitchen item has a small integer \"i\" field — echo that SAME integer in your "
    "verdict (do not invent ids or repeat names). "
    "Return STRICT JSON: {\"verdicts\":[{\"i\":<the item's i>,"
    "\"level\":\"none|possible|likely\",\"recall_index\":<int or null>}]}. "
    "Include an entry for EVERY kitchen item (by its \"i\"). recall_index is the 0-based "
    "index into the RECALLS list for the matched recall, or null when level is none."
)


def _match_one_batch(client, item_batch, recalls, recall_toks):
    """One LLM call over a SMALL batch of items. Each batch is sent ONLY the recalls its
    own items token-overlap (keeps the payload small and never drops a relevant recall to
    a global cap). Returns {item_id: {level, recall_index(GLOBAL)}}; empty on any error so
    a failed batch cannot sink the request."""
    # Global recall indices relevant to THIS batch, collected ROUND-ROBIN across items so
    # every item's top matches survive the per-batch cap (item-by-item order would let a
    # few early items exhaust the cap and starve later items of their own recall).
    import itertools
    per_item = []
    for it in item_batch:
        it_toks = _tokens(it.get('product_name'), it.get('brand'), it.get('variant'))
        per_item.append([gi for gi, rt in enumerate(recall_toks) if it_toks & rt])
    rel = []
    seen = set()
    for tier in itertools.zip_longest(*per_item):
        for gi in tier:
            if gi is not None and gi not in seen:
                seen.add(gi)
                rel.append(gi)
    rel = rel[:MAX_RECALLS_TO_LLM]
    if not rel:
        return {}
    local_to_global = rel
    recall_lines = [{
        'index': li,
        'product': (recalls[gi]['product_description'] or recalls[gi]['title'])[:220],
        'brand': (recalls[gi].get('brand') or '')[:100],
        'establishment': (recalls[gi].get('establishment') or '')[:100],
        'reason': (recalls[gi].get('reason') or '')[:100],
    } for li, gi in enumerate(local_to_global)]
    # Reference items by a small per-batch integer "i" (0..len-1) instead of the 36-char
    # _id UUID. The LLM echoes "i" per verdict; without this ~half the completion tokens
    # (and thus latency — this call is OUTPUT-token-bound) were spent re-emitting UUIDs.
    # Map "i" back to the real _id locally.
    item_lines = [{
        'i': idx,
        'name': it.get('product_name') or '',
        'brand': it.get('brand') or '',
        'variant': it.get('variant') or '',
    } for idx, it in enumerate(item_batch)]
    user = ("KITCHEN ITEMS:\n" + json.dumps(item_lines, ensure_ascii=False) +
            "\n\nACTIVE RECALLS:\n" + json.dumps(recall_lines, ensure_ascii=False))
    try:
        resp = _create_chat(
            client,
            model=OPENAI_MATCH_MODEL,
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': _MATCH_SYSTEM},
                      {'role': 'user', 'content': user}],
        )
        parsed = json.loads(resp.choices[0].message.content or '{}')
    except Exception as e:
        print(json.dumps({'evt': 'recall_llm_batch_error', 'n_items': len(item_batch),
                          'error': str(e)[:200]}), file=sys.stderr)
        return {}
    out = {}
    for v in (parsed.get('verdicts') or []):
        # Accept the compact "i" index; tolerate a stray "item_id" echo just in case.
        ref = v.get('i')
        if ref is None:
            ref = v.get('item_id')
        try:
            bi = int(ref)
        except (TypeError, ValueError):
            continue
        if not (0 <= bi < len(item_batch)):
            continue
        lvl = str(v.get('level') or 'none').lower()
        li = v.get('recall_index')
        if lvl in ('possible', 'likely') and isinstance(li, int) \
                and 0 <= li < len(local_to_global):
            out[item_batch[bi]['_id']] = {'level': lvl, 'recall_index': local_to_global[li]}
    return out


def run_llm_match(items, recalls):
    """Chunk candidate items into small batches and run the batches CONCURRENTLY, each a
    bounded (retries-off, short-timeout) gpt-4.1-mini call. Returns {item_id: {level,
    recall_index}} with recall_index into `recalls`. A single batched call over 100+ items
    is superlinear and hung past the Lambda timeout; chunking bounds each call and
    parallelism bounds wall-clock. Any batch still pending at the total deadline is
    abandoned (partial results returned)."""
    from concurrent.futures import (ThreadPoolExecutor, as_completed,
                                    TimeoutError as _FuturesTimeout)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS,
                    max_retries=OPENAI_MAX_RETRIES)
    recall_toks = [_recall_tokens(r) for r in recalls]
    batches = [items[i:i + LLM_ITEMS_PER_BATCH]
               for i in range(0, len(items), LLM_ITEMS_PER_BATCH)]
    out = {}
    ex = ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS)
    try:
        futs = [ex.submit(_match_one_batch, client, b, recalls, recall_toks)
                for b in batches]
        try:
            for f in as_completed(futs, timeout=LLM_TOTAL_DEADLINE_S):
                try:
                    out.update(f.result())
                except Exception:
                    pass
        except _FuturesTimeout:
            done = sum(1 for f in futs if f.done())
            print(json.dumps({'evt': 'recall_llm_deadline', 'batches': len(batches),
                              'completed': done}), file=sys.stderr)
    finally:
        ex.shutdown(wait=False)
    return out


def _validate_match(it, r, level):
    """Deterministic guard against LLM hallucination: the CHOSEN recall must actually
    share a product-identity token with the item (kills 'tuna -> ice cream'). And
    'likely' additionally requires real brand/establishment overlap, else it is
    downgraded to 'possible'. Returns the effective level, or None to reject."""
    it_toks = _tokens(it.get('product_name'), it.get('variant'))
    r_prod = _tokens(r.get('product_description'), r.get('title'))
    # A shared token that is ONLY a generic prep/packaging word is not product identity.
    shared = it_toks & r_prod
    if not shared or shared <= {'ground', 'sliced', 'frozen', 'fresh', 'cooked',
                                'roasted', 'smoked', 'dried', 'canned', 'mini'}:
        return None
    if level == 'likely':
        it_brand = _tokens(it.get('brand'))
        r_brand = _tokens(r.get('brand'), r.get('establishment'))
        if not (it_brand and r_brand and (it_brand & r_brand)):
            return 'possible'  # no genuine brand alignment -> soften
    return level


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
        level = _validate_match(it, r, v['level'])
        if level is None:
            continue  # LLM hallucination — item and chosen recall don't share a product
        matches.append({
            'kitchen_item_id': iid,
            'kitchen_item_name': it.get('product_name'),
            'brand': it.get('brand'),
            'match_level': level,
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


def _source_counts(recalls):
    counts = {}
    for r in recalls:
        for s in (r.get('sources') or [r.get('source')]):
            if s:
                counts[s] = counts.get(s, 0) + 1
    return counts


def lambda_handler(event, context):
    t_start = time.time()
    event = event or {}
    params = event.get('pathParameters') or {}
    qs = event.get('queryStringParameters') or {}
    owner = params.get('owner')

    # --- Feed refresh (EventBridge cron or manual). Not an HTTP route. ---
    action = event.get('action') or (event.get('detail') or {}).get('action')
    if action == 'refresh_feed':
        force_openfda = bool(event.get('force_openfda'))
        try:
            stats = refresh_feed(force_openfda=force_openfda)
            return {'ok': True, 'stats': stats}
        except Exception as e:
            print(json.dumps({'evt': 'recall_feed_refresh_failed',
                              'error': str(e)[:400]}), file=sys.stderr)
            return {'ok': False, 'error': str(e)[:400]}

    # `?refresh=1` is a convenience trigger — it fires the SAME async self-invoke as the
    # stale fallback (never a synchronous multi-source fetch on the HTTP path).
    if str(qs.get('refresh', '')).lower() in ('1', 'true', 'yes'):
        _async_self_refresh(context)

    # STRICTLY READ-ONLY feed load; serves stale + async-rebuilds if the table lapsed.
    try:
        recalls, age, total_rows = read_active_feed_nonblocking(context)
    except Exception as e:
        return _resp(502, {'error': 'feed_read_failed', 'detail': str(e)})

    feed_date = _iso(time.time() - age) if age is not None else ''

    # Diagnostics: per-source + merge stats.
    if str(qs.get('probe', '')).lower() in ('1', 'true', 'yes'):
        all_rows = read_feed_table(active_only=False)
        return _resp(200, {
            'feed_table_rows': total_rows,
            'active_recent_records': len(recalls),
            'feed_age_hours': round(age / 3600, 1) if age is not None else None,
            'source_counts_all': _source_counts(all_rows),
            'source_counts_active_recent': _source_counts(recalls),
            'multi_source_merged': sum(1 for r in all_rows if len(r.get('sources', [])) > 1),
            'sample': recalls[:3],
        })

    # Fleet-scan helper: compact ACTIVE recall list (no owner needed).
    if owner in (None, '_feed'):
        return _resp(200, {'recall_feed_date': feed_date,
                           'active_recalls': len(recalls),
                           'source_counts': _source_counts(recalls),
                           'recalls': recalls})

    try:
        matches, stats = check_owner(owner, recalls)
    except Exception as e:
        return _resp(500, {'error': 'check_failed', 'detail': str(e)})

    return _resp(200, {
        'checked_at': _iso(t_start),
        'recall_feed_date': feed_date,
        'matches': matches,
        'feed_stats': {'active_recalls': stats['active_recalls'],
                       'sources': _source_counts(recalls)},
        'timing': {
            'llm_ms': stats['llm_ms'],
            'llm_called': stats['llm_called'],
            'total_ms': int((time.time() - t_start) * 1000),
        },
        'debug': stats,
    })
