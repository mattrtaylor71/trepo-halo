import ai_consent_guard as _ai_consent
import ai_consent_guard as _ai_consent
"""A generation job shares one wall-clock budget across all provider work.

The scope is local to an invocation. Explicit propagation to executor workers
prevents nested matching/substitution calls from receiving a fresh timeout.
Without a scope, existing consumers retain their original transport behavior.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import io
import time
import urllib.error
import urllib.request

_deadline = ContextVar('recipe_generation_deadline', default=None)


class GenerationDeadlineExceeded(TimeoutError):
    pass


@contextmanager
def scope(seconds, context=None, reserve_seconds=20):
    allowed = float(seconds)
    if context and callable(getattr(context, 'get_remaining_time_in_millis', None)):
        allowed = min(allowed, context.get_remaining_time_in_millis()/1000 - reserve_seconds)
    deadline = time.monotonic() + max(0, allowed)
    parent = _deadline.get()
    token = _deadline.set(min(deadline, parent) if parent is not None else deadline)
    try:
        remaining()
        yield
    finally:
        _deadline.reset(token)


def remaining(cap=None):
    deadline = _deadline.get()
    value = max(0, deadline-time.monotonic()) if deadline is not None else float('inf')
    if value <= 0:
        raise GenerationDeadlineExceeded('recipe_generation_deadline')
    return min(value, float(cap)) if cap is not None else value


def submit(executor, function, *args, **kwargs):
    context = copy_context()
    return _ai_consent.submit(executor, context.run, function, *args, **kwargs)


def sleep(seconds):
    time.sleep(remaining(seconds))
    remaining()


def _run(operation, cap):
    limit = remaining(cap)
    async def bounded():
        try:
            return await asyncio.wait_for(operation(), timeout=limit)
        except asyncio.TimeoutError as error:
            raise GenerationDeadlineExceeded('recipe_provider_deadline') from error
    return asyncio.run(bounded())


def chat(client, **kwargs):
    import ai_consent_guard
    ai_consent_guard.require(ai_consent_guard.actor_context.get())
    if _deadline.get() is None:
        return _ai_consent.provider_call(client.chat.completions.create)(**kwargs)
    from openai import AsyncOpenAI
    timeout = getattr(client, 'timeout', 60)
    cap = timeout if isinstance(timeout, (int, float)) else getattr(timeout, 'read', None) or 60
    cap = min(float(cap), remaining())
    async def request():
        # Retries belong to the caller's bounded fallback ladder. SDK retries
        # must not multiply every generation or substitution timeout.
        async with _ai_consent.provider_client(AsyncOpenAI)(api_key=client.api_key, base_url=client.base_url,
                               organization=client.organization, project=client.project,
                               timeout=cap, max_retries=0) as provider:
            return await _ai_consent.provider_call(provider.chat.completions.create)(**kwargs)
    return _run(request, cap)


class _Response(io.BytesIO):
    def __init__(self, body, status, headers, url):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers, url


def urlopen(request, timeout, headers_only=False, max_bytes=2*1024*1024):
    if _deadline.get() is None:
        return urllib.request.urlopen(request, timeout=timeout)
    import httpx
    cap = remaining(timeout)
    async def fetch():
        async with httpx.AsyncClient(timeout=cap, follow_redirects=True) as client:
            async with client.stream(request.get_method(), request.full_url,
                                     headers=dict(request.header_items()), content=request.data) as response:
                body = bytearray()
                if not headers_only or response.status_code >= 400:
                    async for chunk in response.aiter_bytes():
                        if len(body)+len(chunk)>max_bytes:
                            raise ValueError('recipe_provider_response_too_large')
                        body.extend(chunk)
                result = _Response(bytes(body), response.status_code, response.headers, str(response.url))
                if response.status_code >= 400:
                    raise urllib.error.HTTPError(str(response.url), response.status_code,
                                                 response.reason_phrase, response.headers, result)
                return result
    return _run(fetch, cap)
