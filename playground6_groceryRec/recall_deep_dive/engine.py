# engine.py — the "deep dive" per-item recall/outbreak investigation.
#
# One kitchen item  ->  one web-search-enabled reasoning call  ->  one verdict.
#
# ENGINE CHOICE: OpenAI Responses API (`/v1/responses`) with the server-side
# `web_search` tool, driven by a GPT-5 reasoning model. Rationale:
#   * The OpenAI key is ALREADY provisioned in this stack (recall_check/app.py uses
#     OPENAI_API_KEY / gpt-4.1-mini) — no new secret to stand up. No Anthropic key is
#     configured anywhere in this repo today, so Claude would mean provisioning a brand
#     new secret for zero quality delta on this task.
#   * A single Responses call does the whole agent loop SERVER-SIDE at OpenAI: it issues
#     real web searches, reads the actual recall/press/news pages, reasons about
#     brand/product/lot/outbreak scope, and returns citations (url_citation annotations).
#     The Lambda just makes one HTTPS request per item — no crawler, no egress beyond
#     reaching api.openai.com (RDS is public so the fn already has internet egress).
#   * GPT-5's reasoning is what catches the hard cases this feature exists for: GENERIC /
#     category recalls and ACTIVE OUTBREAKS ("all frozen blueberries distributed by X since
#     June") while NOT false-positiving a *different* brand of the same food.
#
# We speak the REST API directly over urllib (no openai SDK dependency) so the verdict
# path is immune to SDK-version drift and the deploy package stays tiny (pymysql only).

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
# GPT-5 reasoning model with web search. Overridable; falls back on 400 (see _create).
DEEP_MODEL = os.getenv('RECALL_DEEP_MODEL', 'gpt-5')
DEEP_MODEL_FALLBACK = os.getenv('RECALL_DEEP_MODEL_FALLBACK', 'gpt-4.1')
# Reasoning effort for GPT-5. 'low' keeps latency/cost bounded; the web_search results (not
# deep chain-of-thought) carry this task, and 'low' still nails brand/outbreak scope in
# testing. Bump to 'medium' only if research quality regresses.
DEEP_EFFORT = os.getenv('RECALL_DEEP_EFFORT', 'low')
# Per-socket-op read timeout (a truly stalled connection raises -> one retry).
DEEP_ITEM_TIMEOUT = int(os.getenv('RECALL_DEEP_ITEM_TIMEOUT', '120'))
# Cap the search tool's context to bound cost/latency ('low'|'medium'|'high').
DEEP_SEARCH_CONTEXT = os.getenv('RECALL_DEEP_SEARCH_CONTEXT', 'low')
# HARD BOUND on the agent loop: max built-in (web_search) tool calls per item. Without this
# GPT-5 can spiral into 20-minute research loops (observed 1330s). ~5 searches is plenty to
# confirm/deny a recall while keeping each item to ~1-2 min.
DEEP_MAX_TOOL_CALLS = int(os.getenv('RECALL_DEEP_MAX_TOOL_CALLS', '5'))
# Bound the final answer size (the verdict is small; reasoning is 1-2 sentences).
DEEP_MAX_OUTPUT_TOKENS = int(os.getenv('RECALL_DEEP_MAX_OUTPUT_TOKENS', '1200'))
# Transient-network retries. The Responses call can run for minutes; a mid-stream
# ConnectionReset / read timeout is worth ONE retry before we give up on the item.
DEEP_MAX_RETRIES = int(os.getenv('RECALL_DEEP_MAX_RETRIES', '1'))

VALID_VERDICTS = ('recalled', 'possible', 'clear')

_SYSTEM = (
    "You are a meticulous US food-safety recall investigator. For ONE grocery product a "
    "user has in their kitchen, use web search to determine whether it is affected by a "
    "CURRENT or RECENT (last ~12 months, still relevant) food recall OR an ACTIVE "
    "foodborne-illness outbreak.\n"
    "\n"
    "You MUST consider three ways a product can be affected:\n"
    "  1. BRAND-SPECIFIC recall: this exact brand + product is named in a recall notice.\n"
    "  2. GENERIC / CATEGORY recall: a recall or advisory covering a whole category or a "
    "distributor/manufacturer that supplies many brands (store brands, private label, "
    "'sold at retailers nationwide', a co-packer) where THIS product plausibly falls in "
    "scope.\n"
    "  3. ACTIVE OUTBREAK: an ongoing CDC/FDA/FSIS outbreak investigation linked to this "
    "type of food (e.g. 'all cantaloupe', 'frozen blueberries from distributor X') where "
    "this product plausibly falls in the affected window/scope.\n"
    "\n"
    "CRITICAL — avoid false alarms: do NOT flag a product just because a DIFFERENT brand of "
    "the same food was recalled. A recall of 'Brand A frozen blueberries' does NOT implicate "
    "'Brand B frozen blueberries' unless there is a shared supplier/co-packer/distributor or "
    "a category-wide advisory. Be specific about WHY this product is or isn't in scope.\n"
    "\n"
    "Read the ACTUAL recall/press-release/outbreak page before concluding. Prefer official "
    "sources (fsis.usda.gov, fda.gov, cdc.gov, the company press release) over aggregators.\n"
    "\n"
    "Return your answer by calling the `report_verdict` function EXACTLY once."
)

# We force a single structured tool call so parsing is deterministic. The web_search tool
# and this function tool coexist in the same Responses call; the model searches, then emits
# the function call as its final action.
_VERDICT_TOOL = {
    "type": "function",
    "name": "report_verdict",
    "description": "Report the recall/outbreak investigation result for this product.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["recalled", "possible", "clear"],
                "description": (
                    "recalled = this product IS affected by a specific active recall/outbreak; "
                    "possible = could be affected (generic/category recall, outbreak scope "
                    "overlap, or same-supplier ambiguity) and is worth the user checking; "
                    "clear = nothing current/relevant found for this product."),
            },
            "confidence": {
                "type": "number",
                "description": "0..1 confidence in the verdict.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "ONE or TWO short human sentences a shopper can read, naming the recall/"
                    "outbreak and why this specific product is (or isn't) in scope."),
            },
            "source_url": {
                "type": "string",
                "description": (
                    "The single best REAL URL: the official recall notice / press release / "
                    "outbreak page you relied on. Empty string if verdict is clear."),
            },
            "recall_title": {
                "type": "string",
                "description": "Short title of the recall/outbreak, or empty string if clear.",
            },
        },
        "required": ["verdict", "confidence", "reasoning", "source_url", "recall_title"],
    },
}


def _item_query(item):
    brand = (item.get('brand') or '').strip()
    name = (item.get('product_name') or '').strip()
    variant = (item.get('variant') or '').strip()
    category = (item.get('category') or '').strip()
    bits = []
    if brand:
        bits.append("Brand: %s" % brand)
    bits.append("Product: %s" % (name or '(unnamed)'))
    if variant:
        bits.append("Variant/size: %s" % variant)
    if category:
        bits.append("Category: %s" % category)
    return "\n".join(bits)


_TRANSIENT = (ConnectionResetError, socket.timeout, TimeoutError, urllib.error.URLError)


def _http_post(path, payload, timeout):
    data = json.dumps(payload).encode('utf-8')
    url = OPENAI_BASE.rstrip('/') + path
    last = None
    for attempt in range(DEEP_MAX_RETRIES + 1):
        req = urllib.request.Request(
            url, data=data, method='POST',
            headers={'Authorization': 'Bearer ' + OPENAI_API_KEY,
                     'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError:
            raise  # 4xx/5xx are handled/inspected by the caller, not retried blindly here
        except _TRANSIENT as e:
            last = e
            if attempt < DEEP_MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last  # pragma: no cover


def _create(model, item, timeout):
    payload = {
        "model": model,
        "instructions": _SYSTEM,
        "input": (
            "Investigate this single kitchen product for any current recall or active "
            "outbreak affecting it:\n\n" + _item_query(item)),
        "tools": [
            {"type": "web_search", "search_context_size": DEEP_SEARCH_CONTEXT},
            _VERDICT_TOOL,
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tool_calls": DEEP_MAX_TOOL_CALLS,
        "max_output_tokens": DEEP_MAX_OUTPUT_TOKENS,
        "store": False,
    }
    # GPT-5 reasoning knob (ignored/400 on non-reasoning models -> retry without it).
    if model.startswith('gpt-5') or model.startswith('o'):
        payload["reasoning"] = {"effort": DEEP_EFFORT}
    try:
        return _http_post('/responses', payload, timeout)
    except urllib.error.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', 'replace')
        except Exception:
            pass
        # Drop a param the API named as unsupported and retry once (mirrors recall_check's
        # _create_chat tolerance). Also fall back web_search -> web_search_preview.
        low = body.lower()
        if 'web_search' in low and 'web_search_preview' not in low:
            payload["tools"][0] = {"type": "web_search_preview",
                                   "search_context_size": DEEP_SEARCH_CONTEXT}
            return _http_post('/responses', payload, timeout)
        # Drop any optional param the API names as unsupported, then retry once.
        for opt in ('reasoning', 'max_tool_calls', 'max_output_tokens', 'parallel_tool_calls'):
            if opt in low and opt in payload:
                payload.pop(opt, None)
                return _http_post('/responses', payload, timeout)
        raise


def _extract(resp):
    """Pull (verdict_dict_or_None, output_text, citations[]) out of a Responses payload."""
    verdict = None
    text_parts = []
    citations = []
    for item in resp.get('output') or []:
        itype = item.get('type')
        if itype == 'function_call' and item.get('name') == 'report_verdict':
            try:
                verdict = json.loads(item.get('arguments') or '{}')
            except Exception:
                verdict = None
        elif itype == 'message':
            for c in item.get('content') or []:
                if c.get('type') in ('output_text', 'text'):
                    text_parts.append(c.get('text') or '')
                for ann in c.get('annotations') or []:
                    url = ann.get('url') or (ann.get('url_citation') or {}).get('url')
                    if url:
                        citations.append(url)
    return verdict, "\n".join(text_parts), citations


def _salvage_json(text):
    """If the model answered in text instead of the tool, dig a JSON object out of it."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _is_generic_index_url(url):
    """A bare source landing page is not a specific notice -> treat as non-'official'."""
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return True
    host = (p.netloc or '').lower()
    path = (p.path or '/').rstrip('/')
    if not host:
        return True
    # Root or shallow index of a known aggregator/source == not a specific recall page.
    return path in ('', '/') or path.count('/') <= 1 and not p.query


def _google(item):
    brand = (item.get('brand') or '').strip()
    name = (item.get('product_name') or '').strip()
    q = ("%s %s recall" % (brand, name)).strip()
    return 'https://www.google.com/search?q=' + urllib.parse.quote_plus(q)


def investigate_item(item):
    """Run one deep investigation. Returns a normalized dict:
        {verdict, confidence, reasoning, source_url, link_type, recall_title, engine_ms}
    Never raises for a model/verdict problem — returns a 'clear'/error-tagged dict so one
    bad item can't sink the whole run. (Network/timeout DOES raise so the pool can log it.)"""
    t0 = time.time()
    resp = _create(DEEP_MODEL, item, DEEP_ITEM_TIMEOUT)
    verdict, text, citations = _extract(resp)
    if not verdict:
        verdict = _salvage_json(text)
    if not isinstance(verdict, dict):
        # Model produced neither a tool call nor parseable JSON — treat conservatively.
        return {'verdict': 'clear', 'confidence': 0.0,
                'reasoning': 'Investigation returned no structured verdict.',
                'source_url': '', 'link_type': 'search', 'recall_title': '',
                'engine_ms': int((time.time() - t0) * 1000)}

    v = str(verdict.get('verdict', '')).strip().lower()
    if v not in VALID_VERDICTS:
        v = 'possible' if v else 'clear'
    try:
        conf = float(verdict.get('confidence'))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reasoning = (verdict.get('reasoning') or '').strip()
    title = (verdict.get('recall_title') or '').strip()

    src = (verdict.get('source_url') or '').strip()
    if not src and citations:
        src = citations[0].strip()
    # Guaranteed-tappable link, same convention as recall_check: a specific real page is
    # 'official'; otherwise a Google search targeted at this product is 'search'.
    if src and src.startswith('http') and not _is_generic_index_url(src):
        link_type = 'official'
    else:
        src = _google(item)
        link_type = 'search'

    return {'verdict': v, 'confidence': conf, 'reasoning': reasoning,
            'source_url': src, 'link_type': link_type, 'recall_title': title,
            'engine_ms': int((time.time() - t0) * 1000)}
