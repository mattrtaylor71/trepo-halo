"""Bound URL provider work before Lambda expiry, leaving time to retain the link.

Timers cover network work only. Persistence runs after these boundaries unwind;
no background extraction can outlive a deadline and later overwrite a recipe.
"""
import contextvars
import functools
import signal
import threading
import time
from contextlib import contextmanager

_state = contextvars.ContextVar('recipe_url_work_budget', default=None)
DOWNLOAD_SECONDS = 30.0
TRANSCRIBE_SECONDS = 45.0
MAX_MEDIA_BYTES = 24_000_000  # leave multipart headroom below the 25 MB API limit


class WorkLimit(RuntimeError):
    def __init__(self, reason='deadline'):
        self.reason = reason
        super().__init__('Recipe source work stopped: ' + reason)


class _Expired(BaseException):
    """Escape requests/SDK/provider retry handlers at the timed boundary."""


@contextmanager
def invocation(context, is_sync):
    get_remaining = getattr(context, 'get_remaining_time_in_millis', None)
    native = max(0.0, float(get_remaining()) / 1000.0) if callable(get_remaining) else 120.0
    # Existing image checkpointing keeps its separate stage budget. This state is
    # consulted only by the URL extraction/refinement boundaries below.
    seconds = min(native - 15.0, 24.0) if is_sync else native - 15.0
    token = _state.set({'deadline': time.monotonic() + max(0.0, seconds),
                        'is_sync': is_sync, 'reason': None})
    try:
        yield
    finally:
        _state.reset(token)


def remaining():
    state = _state.get()
    return None if state is None else max(0.0, state['deadline'] - time.monotonic())


def stop(reason):
    state = _state.get()
    if state is not None:
        state['reason'] = reason
    raise WorkLimit(reason)


def terminal_reason():
    state = _state.get()
    if not state:
        return None
    if state['reason'] == 'media_too_large':
        return state['reason']
    # A gateway-limited save can still use one longer async repair attempt.
    if not state['is_sync'] and (state['reason'] or remaining() <= 0):
        return state['reason'] or 'deadline'
    return None


def note_rung_skip():
    state = _state.get()
    if state is not None and not state['is_sync'] and not state['reason']:
        state['reason'] = 'deadline'


def phase_seconds(max_seconds=None, reserve=0):
    available = remaining()
    available = 105.0 if available is None else available
    return max(0.0, min(available - reserve, max_seconds if max_seconds is not None else available))


@contextmanager
def phase(max_seconds=None, reserve=0):
    seconds = phase_seconds(max_seconds, reserve)
    if seconds <= 0:
        stop('deadline')
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        stop('deadline_unavailable')
    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    budget = min(seconds, prior_timer[0]) if prior_timer[0] else seconds

    def expire(*_):
        raise _Expired()

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        yield
    except _Expired:
        stop('deadline')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, prior_timer[0] - (time.monotonic() - started)), prior_timer[1])


def bounded(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with phase():
            return function(*args, **kwargs)
    return call


def check_bytes(count, limit=MAX_MEDIA_BYTES):
    if count and count > limit:
        stop('media_too_large')


def download_progress(progress):
    check_bytes(progress.get('downloaded_bytes'))
    check_bytes(progress.get('total_bytes'))
    # Estimates are advisory: rejecting an overestimate loses valid spoken recipes.
    if remaining() is not None and remaining() <= 0:
        stop('deadline')
