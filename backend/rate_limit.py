"""Per-client-IP rate limiting for the endpoints that cost money or compute
to serve: POST /ask and POST /datasets/{id}/select (an LLM/embedding call),
POST /datasets/upload (a CSV->SQLite conversion), and GET /debug/retrieve
(a dev-only endpoint, disabled by default — see backend/main.py).

Deliberately in-process, in-memory — no Redis, no external store. This app
is meant to run as a SINGLE instance with a SINGLE worker process; see the
module-level note below `SlidingWindowRateLimiter` for exactly what that
assumption buys us and what breaks if it's ever violated. Same "no external
infra needed" philosophy as backend/sql_guard.py and backend/db.py.

Getting the client's IP right behind a reverse proxy is the one subtle
part — see get_client_ip() below, and read its docstring before trusting it
in a new deployment.
"""
import threading
import time

from fastapi import HTTPException, Request

from backend.config import settings


def get_client_ip(request: Request) -> str:
    """The client IP used as the rate-limiter key.

    By default (settings.TRUST_PROXY_HEADERS=False) this ONLY ever returns
    the raw TCP peer (request.client.host) — it never reads a
    client-supplied header at all, so there is nothing for a client to
    spoof by sending its own X-Forwarded-For. That's the safe default for
    local development and for any deployment whose network topology in
    front of this app isn't a known, trusted quantity.

    Set TRUST_PROXY_HEADERS=true ONLY when this app runs behind exactly one
    reverse proxy you trust to set X-Forwarded-For correctly for the
    connection it actually received (e.g. intended for Render). Even then,
    this reads only the LAST entry in that header: the reasoning is that a
    single trusted proxy hop APPENDS to (rather than replaces) whatever a
    client already sent, so the last entry is the one hop that could not
    have been written by the client itself, while everything before it
    could have been.

    IMPORTANT: that reasoning about Render specifically is based on
    Render community-support reports at the time this was written, NOT
    confirmed against Render's own primary documentation. Treat
    TRUST_PROXY_HEADERS=true as an explicit, documented assumption to
    verify against a real deployment (e.g. log the resolved IP for a few
    live requests and compare it against the actual caller) before relying
    on it for anything beyond coarse abuse-rate-limiting — never for
    anything security-critical.
    """
    if settings.TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            last_hop = forwarded_for.split(",")[-1].strip()
            if last_hop:
                return last_hop
    return request.client.host if request.client else "unknown"


class SlidingWindowRateLimiter:
    """At most `max_requests` calls to allow() per key in any trailing
    `window_seconds` window.

    A plain {key: [timestamp, ...]} dict behind one lock — not a token
    bucket or fixed window — because a sliding log needs no separate
    refill/reset logic to get subtly wrong, and the request volumes a
    single free-tier instance will ever see make an O(hits-in-window) prune
    on every call cheap enough not to matter.

    Deployment assumption, stated explicitly: this holds its state in this
    ONE process's memory. It is only correct for a SINGLE instance running
    a SINGLE worker process.
      - Multiple instances (horizontal scaling) would each enforce the
        limit independently, with no shared view — the effective limit
        becomes (configured limit) x (instance count).
      - Multiple worker processes on one instance (e.g. `uvicorn --workers
        N`) have the same problem for the same reason: each worker gets
        its own copy of this object, since module-level state is per
        process, not shared. Run with a single worker.
    Neither case is detected or guarded against here — there is no
    cross-process coordination in this file at all. If this ever needs to
    run as more than one process, this needs a shared store (e.g. Redis),
    not a bigger version of this class.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            self._prune_expired(cutoff)
            hits = self._hits.get(key, [])
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def _prune_expired(self, cutoff: float) -> None:
        """Drop every hit at or before `cutoff`, for EVERY tracked key —
        not just the one key allow() was called for — and remove any key
        left with no hits at all. Caller must already hold self._lock.

        Doing this for every key on every call (not just the current one)
        is what keeps this dict from growing forever: an IP that hits the
        API once and never comes back would otherwise sit in here forever,
        since nothing else would ever revisit that key to prune it. O(total
        tracked keys) per call — acceptable at the traffic volumes a single
        free-tier instance will ever see; the first thing to revisit if
        that stops being true.
        """
        for existing_key in list(self._hits):
            remaining = [t for t in self._hits[existing_key] if t > cutoff]
            if remaining:
                self._hits[existing_key] = remaining
            else:
                del self._hits[existing_key]


ask_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_ASK_PER_MINUTE, window_seconds=60.0
)
upload_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_UPLOAD_PER_MINUTE, window_seconds=60.0
)
select_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_SELECT_PER_MINUTE, window_seconds=60.0
)
debug_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_DEBUG_PER_MINUTE, window_seconds=60.0
)


def _too_many_requests(max_requests: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            f"Rate limit exceeded: this endpoint allows at most {max_requests} "
            "requests per minute per client. Please wait a moment and try again."
        ),
    )


def enforce_ask_rate_limit(request: Request) -> None:
    """FastAPI dependency: raises 429 once the caller's IP has made too many
    /ask requests in the last minute. Wired into the endpoint via Depends()
    so the limit check runs before the (expensive) pipeline does."""
    if not ask_limiter.allow(get_client_ip(request)):
        raise _too_many_requests(ask_limiter.max_requests)


def enforce_upload_rate_limit(request: Request) -> None:
    """Same as enforce_ask_rate_limit, for POST /datasets/upload."""
    if not upload_limiter.allow(get_client_ip(request)):
        raise _too_many_requests(upload_limiter.max_requests)


def enforce_select_rate_limit(request: Request) -> None:
    """Same as enforce_ask_rate_limit, for POST /datasets/{id}/select."""
    if not select_limiter.allow(get_client_ip(request)):
        raise _too_many_requests(select_limiter.max_requests)


def enforce_debug_rate_limit(request: Request) -> None:
    """Same as enforce_ask_rate_limit, for GET /debug/retrieve (only
    reachable at all when settings.ENABLE_DEBUG_ENDPOINTS is true)."""
    if not debug_limiter.allow(get_client_ip(request)):
        raise _too_many_requests(debug_limiter.max_requests)
