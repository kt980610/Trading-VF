"""News provider adapters.

A provider turns a (query, UTC day) request into a list of raw provider article
dicts. The HTTP transport is injectable so the logic (pagination, retry/backoff,
429 handling, plan/truncation detection) is fully unit-testable without network.

Design rules enforced here:

* The API key is sent via an HTTP header (``X-Api-Key``) so it never appears in a
  URL, and it is NEVER printed or logged.
* A failed day/query (network error, exhausted retries, plan rejection, or result
  truncation) raises :class:`NewsProviderError`. Callers must record this as
  ``coverage_status=failed`` and must NOT treat it as "0 news".
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Callable, Dict, List, Optional, Tuple


class NewsProviderError(Exception):
    """Raised when a provider request fails in a way that is NOT 'zero news'.

    ``reason`` is a stable machine-readable category:

    * ``rate_limited``         - 429 / provider rateLimited after retries exhausted
    * ``http_error``           - non-retryable HTTP status
    * ``network_error``        - transport/connection failure after retries
    * ``provider_error``       - provider returned status=error (e.g. bad key, plan)
    * ``result_truncated``     - more results exist than the plan let us page through
    * ``empty_response``       - 2xx with an empty body after retries
    * ``non_json_response``    - 2xx with an HTML/WAF/transient non-JSON body
    * ``invalid_json_response``- 2xx with malformed JSON after retries
    * ``invalid_request``      - provider rejected the query/params (NOT retryable)

    The structured attributes (``error_code``/``http_status``/``content_type``/
    ``response_excerpt``/``retry_after_seconds``) are persisted to the coverage
    manifest so a failed cell can be diagnosed without re-running. They NEVER carry
    the request URL, query, API key or request headers.
    """

    def __init__(
        self,
        reason: str,
        detail: str = "",
        *,
        error_code: Optional[str] = None,
        http_status: Optional[int] = None,
        content_type: Optional[str] = None,
        response_excerpt: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
    ):
        self.reason = reason
        self.error_code = error_code or reason
        self.http_status = http_status
        self.content_type = content_type
        self.response_excerpt = response_excerpt
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class HttpResponse:
    status: int
    body: dict
    content_type: str = ""
    # Raw response text (kept for diagnostics; sanitised before logging).
    text: str = ""
    # False when the body could not be parsed into a JSON object.
    json_ok: bool = True
    # Parsed ``Retry-After`` (seconds), when the server supplied one.
    retry_after: Optional[float] = None


# A transport takes (url, headers, timeout) and returns an HttpResponse. It must
# raise on a connection-level failure (the provider handles retry/backoff).
Transport = Callable[[str, Dict[str, str], float], HttpResponse]

# Some endpoints (notably GDELT) return an HTML error / empty body instead of JSON
# when no User-Agent is sent; always present a real one.
_DEFAULT_USER_AGENT = "trading-vf-news/1.0 (+https://github.com/)"

_SECRET_EXCERPT_RE = re.compile(
    r"(?i)(api[_-]?key|apikey|token|authorization|x-api-key)\s*[=:]\s*[^\s&\"']+"
)
# Coverage-manifest excerpts are capped small; they are diagnostics, not payloads.
_MAX_EXCERPT_CHARS = 500

# Body-content signatures used to classify a 2xx GDELT response that is NOT valid
# ArtList JSON. GDELT returns query/parameter problems as a short message (often
# with a ``text/html`` content type) -- these are PERMANENT for the given query
# and must be classified as ``invalid_request`` and never retried, regardless of
# the content type. Checked before the HTML/JSON-shape heuristics.
_GDELT_VALIDATION_MARKERS = (
    "searches may only be used",          # "...around OR'd statements."
    "or'd statements",
    "you did not specify any search terms",
    "did not specify any search",
    "your query was too short",
    "your query was too long",
    "query was too short",
    "query was too long",
    "please make it longer",
    "please make it shorter",
    "specified phrase",
    "specified search string was too short",
    "could not parse",
    "is not a valid",
    "invalid date",
    "requires at least",
    "must specify",
    "unrecognized",
)
# Plain-text bodies that DO indicate a transient condition worth retrying.
_GDELT_TRANSIENT_TEXT = (
    "rate limit",
    "too many requests",
    "temporarily",
    "try again",
    "timeout",
    "timed out",
    "overloaded",
    "please wait",
    "service unavailable",
    "server is busy",
)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if not value:
        return None
    value = value.strip()
    try:
        secs = float(value)
        return max(0.0, secs)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(value)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


def _sanitize_excerpt(text: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    """Truncate a response body to <= ``limit`` chars and redact secret-looking
    ``key=value`` fragments. Response bodies should not carry secrets, but this is
    a defence-in-depth pass so a diagnostic excerpt is always safe to log."""
    if not text:
        return ""
    snippet = text[:limit]
    snippet = _SECRET_EXCERPT_RE.sub(r"\1=<redacted>", snippet)
    # Collapse to a compact single-ish line so multi-KB HTML stays readable.
    snippet = " ".join(snippet.split())
    if len(text) > limit:
        snippet += f" ...[truncated {len(text) - limit} chars]"
    return snippet


def _response_diagnostic(resp: "HttpResponse") -> str:
    """Safe, secret-free one-line diagnostic: status, content type, body excerpt.

    Deliberately omits the request URL/query and headers (which can carry an API
    key for key-based providers); only the server's own response is described.
    """
    return (
        f"status={resp.status} content_type={resp.content_type or 'unknown'!r} "
        f"body={_sanitize_excerpt(resp.text)!r}"
    )


def _looks_like_html(resp: "HttpResponse") -> bool:
    if "html" in (resp.content_type or "").lower():
        return True
    head = (resp.text or "").lstrip()[:64].lower()
    return head.startswith("<!doctype") or head.startswith("<html") or head.startswith("<")


def _looks_like_json(resp: "HttpResponse", stripped: str) -> bool:
    if "json" in (resp.content_type or "").lower():
        return True
    return stripped[:1] in ("{", "[")


def _parse_json_body(raw: str):
    """Return ``(body_dict, json_ok)`` without ever raising on bad input.

    Non-JSON / empty input yields ``({}, False)`` so callers can classify the
    failure instead of crashing on an uncaught :class:`json.JSONDecodeError`.
    """
    if raw is None or not raw.strip():
        return {}, False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, False
    if isinstance(parsed, dict):
        return parsed, True
    # A JSON array/scalar is valid JSON but not the object shape providers expect;
    # wrap it so ``.get()`` stays safe while still marking it parseable.
    return {"_json": parsed}, True


def _urllib_transport(url: str, headers: Dict[str, str], timeout: float) -> HttpResponse:
    hdrs = dict(headers)
    hdrs.setdefault("User-Agent", _DEFAULT_USER_AGENT)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
            ctype = resp.headers.get("Content-Type", "") if resp.headers else ""
            retry_after = _parse_retry_after(
                resp.headers.get("Retry-After") if resp.headers else None
            )
            body, ok = _parse_json_body(raw)
            status = getattr(resp, "status", None) or 200
            return HttpResponse(
                status=status, body=body, content_type=ctype, text=raw,
                json_ok=ok, retry_after=retry_after,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        ctype = exc.headers.get("Content-Type", "") if exc.headers else ""
        retry_after = _parse_retry_after(
            exc.headers.get("Retry-After") if exc.headers else None
        )
        body, ok = _parse_json_body(raw)
        return HttpResponse(
            status=exc.code, body=body, content_type=ctype, text=raw,
            json_ok=ok, retry_after=retry_after,
        )


def _day_bounds(day: date) -> Tuple[str, str]:
    """Inclusive UTC ISO-8601 [start, end] covering a single calendar day."""
    start = datetime.combine(day, dtime(0, 0, 0), tzinfo=timezone.utc)
    end = datetime.combine(day, dtime(23, 59, 59), tzinfo=timezone.utc)
    return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")


class NewsProvider:
    """Base provider interface."""

    name = "base"

    def fetch_day(self, query: str, day: date, language: str = "en") -> List[dict]:
        raise NotImplementedError


class NewsApiProvider(NewsProvider):
    """Adapter for https://newsapi.org `/v2/everything`.

    Day-partitioned querying avoids silent truncation: each call covers exactly
    one UTC day. If a single day still returns more results than the plan can
    page through, that is surfaced as ``result_truncated`` (never silently
    dropped).
    """

    name = "newsapi"
    DEFAULT_BASE_URL = "https://newsapi.org/v2/everything"
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        transport: Optional[Transport] = None,
        base_url: str = DEFAULT_BASE_URL,
        page_size: int = 100,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise NewsProviderError("provider_error", "missing api key")
        self._api_key = api_key
        self._transport = transport or _urllib_transport
        self.base_url = base_url
        self.page_size = max(1, min(int(page_size), 100))
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.timeout = float(timeout)
        self._sleep = sleep

    def _headers(self) -> Dict[str, str]:
        # Key travels in a header (never in the URL) and is never logged.
        return {
            "X-Api-Key": self._api_key,
            "Accept": "application/json",
            "User-Agent": _DEFAULT_USER_AGENT,
        }

    def _build_url(self, query: str, frm: str, to: str, language: str, page: int) -> str:
        params = {
            "q": query,
            "from": frm,
            "to": to,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": self.page_size,
            "page": page,
        }
        return f"{self.base_url}?{urllib.parse.urlencode(params)}"

    def _request(self, url: str) -> dict:
        """Single page request with retry/backoff on 429 / 5xx / network errors."""
        last_detail = ""
        last_reason = "rate_limited"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._transport(url, self._headers(), self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_reason = "network_error"
                last_detail = f"network: {type(exc).__name__}"
                self._backoff(attempt)
                continue

            if resp.status == 200 and resp.json_ok and resp.body.get("status") != "error":
                return resp.body

            # A 200 with an empty / HTML / malformed body is a transient upstream
            # glitch, not "0 news": retry, then surface a precise, secret-free error.
            if resp.status == 200 and not resp.json_ok:
                if not (resp.text or "").strip():
                    last_reason, last_detail = "empty_response", _response_diagnostic(resp)
                elif _looks_like_html(resp):
                    last_reason, last_detail = "non_json_response", _response_diagnostic(resp)
                else:
                    last_reason, last_detail = "invalid_json_response", _response_diagnostic(resp)
                self._backoff(attempt)
                continue

            # Provider-level error body (NewsAPI returns 200/4xx with status=error).
            code = str(resp.body.get("code", "")) if isinstance(resp.body, dict) else ""
            if resp.status in self._RETRYABLE_STATUS or code == "rateLimited":
                last_reason = "rate_limited" if (resp.status == 429 or code == "rateLimited") else "http_error"
                last_detail = f"status={resp.status} code={code}"
                self._backoff(attempt, retry_after=self._retry_after(resp))
                continue
            if code == "maximumResultsReached":
                raise NewsProviderError("result_truncated", "maximumResultsReached")
            # Non-retryable provider error (bad key, plan/date not allowed, etc.).
            # NOTE: never include the response body verbatim if it could echo the
            # key; NewsAPI does not, and we only pass code + status.
            raise NewsProviderError("provider_error", f"status={resp.status} code={code}")

        raise NewsProviderError(last_reason, last_detail or "retries exhausted")

    @staticmethod
    def _retry_after(resp: HttpResponse) -> Optional[float]:
        try:
            ra = resp.body.get("retry_after") if isinstance(resp.body, dict) else None
            return float(ra) if ra is not None else None
        except (TypeError, ValueError):
            return None

    def _backoff(self, attempt: int, retry_after: Optional[float] = None) -> None:
        if retry_after is not None:
            self._sleep(min(retry_after, self.backoff_cap))
            return
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_cap)
        self._sleep(delay)

    def fetch_day(self, query: str, day: date, language: str = "en") -> List[dict]:
        frm, to = _day_bounds(day)
        collected: List[dict] = []
        page = 1
        total_results: Optional[int] = None
        while True:
            url = self._build_url(query, frm, to, language, page)
            body = self._request(url)
            if total_results is None:
                total_results = int(body.get("totalResults", 0) or 0)
            articles = body.get("articles") or []
            collected.extend(articles)
            if not articles:
                break
            if total_results is not None and len(collected) >= total_results:
                break
            if len(collected) >= page * self.page_size:
                page += 1
                # Guard: NewsAPI dev plan blocks beyond 100 results. If more exist
                # but we've hit the page ceiling, that is truncation, not "done".
                if page * self.page_size > 100 and len(collected) < (total_results or 0):
                    raise NewsProviderError(
                        "result_truncated",
                        f"have={len(collected)} total={total_results}",
                    )
            else:
                break
        return collected


def _gdelt_window_bounds(start: datetime, end: datetime) -> Tuple[str, str]:
    """GDELT DOC 2.0 compact UTC datetime bounds ``YYYYMMDDHHMMSS``."""
    return (
        start.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S"),
        end.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S"),
    )


def _parse_gdelt_seendate(raw) -> Optional[str]:
    """Parse a GDELT ``seendate`` to ISO-8601 UTC, or ``None`` if unparseable.

    GDELT emits compact instants such as ``20220101T120000Z`` or
    ``20220101120000``; both pin an exact UTC second.
    """
    if raw is None:
        return None
    s = str(raw).strip().replace("T", "").replace("Z", "")
    if len(s) < 14 or not s[:14].isdigit():
        return None
    try:
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class GlobalRateLimiter:
    """Serialises ALL requests through one paced gate.

    A single instance is shared by every query in a fetch run, so there is no
    per-query burst: each ``acquire`` waits until the next allowed slot (>=
    ``min_interval`` after the previous request). ``penalize`` pushes the next
    slot further into the future, which is how a 429 / adaptive cooldown forces
    *every* subsequent query to wait, not just the one that hit the limit.

    The clock and sleeper are injectable so tests never sleep for real.
    """

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: Optional[float] = None

    def acquire(self) -> float:
        """Block until the next slot; return the seconds actually waited."""
        now = self._clock()
        waited = 0.0
        if self._next_allowed is not None and now < self._next_allowed:
            waited = self._next_allowed - now
            self._sleep(waited)
            now = self._clock()
        self._next_allowed = now + self.min_interval
        return waited

    def penalize(self, cooldown_seconds: float) -> None:
        """Delay the next slot by a global cooldown (affects all later queries)."""
        cooldown = max(0.0, float(cooldown_seconds or 0.0))
        now = self._clock()
        base = self._next_allowed if self._next_allowed is not None else now
        self._next_allowed = max(base, now + cooldown)


@dataclass
class _Classified:
    """Outcome of inspecting a single HTTP response."""

    ok: bool
    body: dict
    reason: str
    error_code: str
    retryable: bool
    retry_after_seconds: Optional[float]
    http_status: Optional[int]
    content_type: Optional[str]
    response_excerpt: str


class GdeltProvider(NewsProvider):
    """Adapter for the free GDELT DOC 2.0 API (``/api/v2/doc/doc``).

    GDELT requires no API key. Each ``fetch_day`` covers one UTC day and, to
    avoid silent truncation on high-volume days, recursively bisects the time
    window whenever a window returns the maximum record count. A window that is
    still saturated at the minimum granularity raises ``result_truncated`` so the
    day/query is recorded as ``failed`` (never a false "0 news").

    The DOC ``ArtList`` response gives ``url``, ``title`` and ``seendate`` (an
    exact UTC instant) but no article body; an optional ``tone`` field (present in
    GKG-derived transports) is carried through as ``gdelt_tone`` and is NEVER
    treated as a FinBERT sentiment score.
    """

    name = "gdelt"
    DEFAULT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    # GDELT ``sourcelang`` expects the language NAME (e.g. ``english``), not an
    # ISO-639-1 code. Map common ISO-639-1 / ISO-639-2 / names -> GDELT token.
    _LANG_TOKEN = {
        "en": "english",
        "eng": "english",
        "english": "english",
    }

    def __init__(
        self,
        api_key: str = "",  # accepted but unused (GDELT is keyless)
        transport: Optional[Transport] = None,
        base_url: str = DEFAULT_BASE_URL,
        max_records: int = 250,
        min_window_seconds: int = 900,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        timeout: float = 30.0,
        min_request_interval: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport or _urllib_transport
        self.base_url = base_url
        self.max_records = max(1, min(int(max_records), 250))
        self.min_window_seconds = max(1, int(min_window_seconds))
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.timeout = float(timeout)
        self._sleep = sleep
        self.min_request_interval = max(0.0, float(min_request_interval))
        # One global limiter for the whole run -> paced, no per-query bursts.
        self._limiter = GlobalRateLimiter(self.min_request_interval, clock=clock, sleep=sleep)
        # Adaptive cooldown applied on 429s without a Retry-After; grows on repeat
        # throttling and resets after a clean success.
        self._cooldown_base = max(self.backoff_base, self.min_request_interval)
        self._adaptive_cooldown = self._cooldown_base

    @staticmethod
    def _headers() -> Dict[str, str]:
        return {"Accept": "application/json", "User-Agent": _DEFAULT_USER_AGENT}

    @classmethod
    def _sourcelang_token(cls, language: str) -> Optional[str]:
        """Normalise a language to a GDELT ``sourcelang`` token (fail-fast).

        ISO-639-1 ``en`` is written as ``sourcelang:english`` (GDELT does NOT
        accept the bare ISO code). An empty language means "no language filter";
        an unrecognised code raises ``ValueError`` so a misconfiguration is caught
        immediately instead of silently producing an unfiltered or invalid query.
        """
        key = (language or "").strip().lower()
        if not key:
            return None
        token = cls._LANG_TOKEN.get(key)
        if token is None:
            raise ValueError(
                f"unsupported news language {language!r}; expected one of "
                f"{sorted(set(cls._LANG_TOKEN))}"
            )
        return token

    @staticmethod
    def _format_query_term(query: str) -> str:
        """Format the keyword term per GDELT DOC syntax.

        GDELT rejects a parenthesised single term/phrase ("Searches may only be
        used around OR'd statements."). So parentheses are used ONLY for a real
        alias/OR list (>= 2 operands joined by an uppercase ``OR``):

        * single keyword         -> ``bitcoin``
        * single multi-word phrase-> ``"jobs report"`` (quoted, never parenthesised)
        * alias/OR list          -> ``(bitcoin OR BTC OR cryptocurrency)``
        """
        q = (query or "").strip()
        if not q:
            return q
        # Trust an expression the caller already grouped or quoted.
        if (q.startswith("(") and q.endswith(")")) or (q.startswith('"') and q.endswith('"')):
            return q
        # A genuine OR alias list is the only case allowed to be parenthesised.
        operands = re.split(r"\s+OR\s+", q)
        if len(operands) >= 2 and all(p.strip() for p in operands):
            return f"({q})"
        # Single multi-word phrase -> quote it (never parenthesise).
        if " " in q:
            return f'"{q}"'
        # Single keyword -> bare.
        return q

    def build_query(self, query: str, language: str) -> str:
        """The exact, secret-free GDELT query string (keyword + language filter)."""
        token = self._sourcelang_token(language)
        term = self._format_query_term(query)
        return f"{term} sourcelang:{token}" if token else term

    def _build_url(self, query: str, frm: str, to: str, language: str) -> str:
        params = {
            "query": self.build_query(query, language),
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": self.max_records,
            "startdatetime": frm,
            "enddatetime": to,
        }
        return f"{self.base_url}?{urllib.parse.urlencode(params)}"

    def _classify(self, resp: HttpResponse) -> _Classified:
        """Map one HTTP response to a precise outcome. Never raises.

        Classification is body-aware: a 2xx plain-text GDELT validation message is
        a *permanent* query problem (``invalid_request``, not retryable), while an
        HTML/WAF/empty body is treated as transient. This is what stops a bad query
        from being hammered as if it were a generic rate-limit.
        """
        excerpt = _sanitize_excerpt(resp.text, _MAX_EXCERPT_CHARS)
        ctype = resp.content_type or None

        if resp.status in self._RETRYABLE_STATUS:
            reason = "rate_limited" if resp.status == 429 else "http_error"
            return _Classified(
                False, {}, reason, f"http_{resp.status}", True,
                resp.retry_after, resp.status, ctype, excerpt,
            )
        if resp.status != 200:
            return _Classified(
                False, {}, "http_error", f"http_{resp.status}", False,
                resp.retry_after, resp.status, ctype, excerpt,
            )

        # --- status 200 -----------------------------------------------------
        # ``json_ok`` is the authoritative parse flag; a parsed body is success
        # even if the transport supplied no raw text (injected/mock transports).
        if resp.json_ok:
            body = resp.body if isinstance(resp.body, dict) else {}
            return _Classified(True, body, "ok", "ok", False, None, 200, ctype, excerpt)

        stripped = (resp.text or "").strip()
        if not stripped:
            return _Classified(
                False, {}, "empty_response", "empty_body", True,
                resp.retry_after, 200, ctype, excerpt,
            )

        low = stripped.lower()
        if any(m in low for m in _GDELT_VALIDATION_MARKERS):
            # Permanent query/parameter problem -> never retried.
            return _Classified(
                False, {}, "invalid_request", "gdelt_query_validation", False,
                None, 200, ctype, excerpt,
            )
        if any(m in low for m in _GDELT_TRANSIENT_TEXT):
            return _Classified(
                False, {}, "rate_limited", "text_rate_limit", True,
                resp.retry_after, 200, ctype, excerpt,
            )
        if _looks_like_json(resp, stripped):
            # Looks like JSON but failed to parse (e.g. truncated) -> transient.
            return _Classified(
                False, {}, "invalid_json_response", "malformed_json", True,
                resp.retry_after, 200, ctype, excerpt,
            )
        if _looks_like_html(resp):
            # HTML / WAF / interstitial -> transient overload.
            return _Classified(
                False, {}, "non_json_response", "html_or_waf_body", True,
                resp.retry_after, 200, ctype, excerpt,
            )
        # Unknown non-JSON plain text: treat as a (non-retryable) request problem
        # rather than retrying blindly like a transient error.
        return _Classified(
            False, {}, "invalid_request", "unrecognized_non_json", False,
            None, 200, ctype, excerpt,
        )

    @staticmethod
    def _error(cls: _Classified) -> NewsProviderError:
        detail = (
            f"status={cls.http_status} content_type={(cls.content_type or 'unknown')!r} "
            f"code={cls.error_code} body={cls.response_excerpt!r}"
        )
        return NewsProviderError(
            cls.reason,
            detail,
            error_code=cls.error_code,
            http_status=cls.http_status,
            content_type=cls.content_type,
            response_excerpt=cls.response_excerpt,
            retry_after_seconds=cls.retry_after_seconds,
        )

    def _on_success(self) -> None:
        self._adaptive_cooldown = self._cooldown_base

    def _apply_throttle(self, cls: _Classified) -> None:
        """Push the global gate forward after throttling so EVERY later query waits."""
        cooldown = cls.retry_after_seconds if cls.retry_after_seconds else self._adaptive_cooldown
        self._limiter.penalize(cooldown)
        if not cls.retry_after_seconds:
            self._adaptive_cooldown = min(self._adaptive_cooldown * 2.0, self.backoff_cap)

    def _request(self, url: str) -> dict:
        """One logical GDELT request, paced by the global limiter and retried only
        for genuinely transient outcomes. Permanent problems (bad query/params,
        non-retryable HTTP) raise immediately with full, secret-free diagnostics;
        a malformed/empty/HTML body is NEVER an uncaught ``JSONDecodeError``."""
        last: Optional[_Classified] = None
        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._transport(url, self._headers(), self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = _Classified(
                    False, {}, "network_error", f"network_{type(exc).__name__}",
                    True, None, None, None, "",
                )
                self._backoff(attempt)
                continue

            cls = self._classify(resp)
            if cls.ok:
                self._on_success()
                return cls.body

            last = cls
            if not cls.retryable:
                raise self._error(cls)
            # Transient: 429/text-rate-limit drive the global cooldown; other
            # transient bodies use plain exponential backoff.
            if cls.reason == "rate_limited":
                self._apply_throttle(cls)
            else:
                self._backoff(attempt)

        raise self._error(last) if last else NewsProviderError("rate_limited", "retries exhausted")

    def probe(self, query: str, day: date, language: str = "en") -> dict:
        """Single query, single day, single HTTP call (NO retries, NO bisection).

        Returns a safe, secret-free classification dict suitable for writing
        straight into the coverage manifest. Used to gate a real fetch: if the
        probe is not ``ok``, do not run a day-level fetch.
        """
        start = datetime.combine(day, dtime(0, 0, 0), tzinfo=timezone.utc)
        end = datetime.combine(day, dtime(23, 59, 59), tzinfo=timezone.utc)
        frm, to = _gdelt_window_bounds(start, end)
        url = self._build_url(query, frm, to, language)
        self._limiter.acquire()
        try:
            resp = self._transport(url, self._headers(), self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "ok": False,
                "response_class": "network_error",
                "error_code": f"network_{type(exc).__name__}",
                "http_status": None,
                "content_type": None,
                "response_excerpt": "",
                "retry_after_seconds": None,
                "article_count": 0,
            }
        cls = self._classify(resp)
        article_count = len(cls.body.get("articles") or []) if cls.ok else 0
        return {
            "ok": cls.ok,
            "response_class": "ok" if cls.ok else cls.reason,
            "error_code": cls.error_code,
            "http_status": cls.http_status,
            "content_type": cls.content_type,
            "response_excerpt": cls.response_excerpt,
            "retry_after_seconds": cls.retry_after_seconds,
            "article_count": article_count,
        }

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(self.backoff_base * (2 ** attempt), self.backoff_cap))

    @staticmethod
    def _normalize_article(art: dict) -> Optional[dict]:
        # ``seendate`` is GDELT's OBSERVATION time (when the article was first seen
        # in the global stream), NOT a verified publisher publish time. It is
        # surfaced as ``source_seen_at`` and never as ``published_at``.
        seen = _parse_gdelt_seendate(art.get("seendate"))
        if seen is None:
            return None
        tone = art.get("tone")
        try:
            tone_val = float(tone) if tone is not None and str(tone) != "" else None
        except (TypeError, ValueError):
            tone_val = None
        return {
            "source_seen_at": seen,
            # GDELT DOC ArtList exposes no verified publish time.
            "published_at": None,
            "url": art.get("url") or "",
            "title": art.get("title") or "",
            # DOC ArtList carries no body/summary; never invent one.
            "description": art.get("body") or "",
            "source": {"name": art.get("domain") or ""},
            "gdelt_tone": tone_val,
        }

    def _fetch_window(
        self, query: str, start: datetime, end: datetime, language: str
    ) -> List[dict]:
        frm, to = _gdelt_window_bounds(start, end)
        body = self._request(self._build_url(query, frm, to, language))
        raw_articles = body.get("articles") or []
        normalized = [
            n for n in (self._normalize_article(a) for a in raw_articles) if n is not None
        ]
        if len(raw_articles) < self.max_records:
            return normalized
        # Saturated window: bisect to recover the full set without truncation.
        span = (end - start).total_seconds()
        if span <= self.min_window_seconds:
            raise NewsProviderError(
                "result_truncated", f"window={int(span)}s hit max_records={self.max_records}"
            )
        mid = start + (end - start) / 2
        merged: Dict[str, dict] = {}
        for art in self._fetch_window(query, start, mid, language) + self._fetch_window(
            query, mid, end, language
        ):
            merged[art["url"] or (art["source_seen_at"] + art["title"])] = art
        return list(merged.values())

    def fetch_day(self, query: str, day: date, language: str = "en") -> List[dict]:
        start = datetime.combine(day, dtime(0, 0, 0), tzinfo=timezone.utc)
        end = datetime.combine(day, dtime(23, 59, 59), tzinfo=timezone.utc)
        return self._fetch_window(query, start, end, language)


def get_provider(
    name: str,
    api_key: str,
    transport: Optional[Transport] = None,
    **kwargs,
) -> NewsProvider:
    """Factory. Add new providers here; the name is config-driven."""
    key = (name or "").strip().lower()
    if key == "newsapi":
        return NewsApiProvider(api_key=api_key, transport=transport, **kwargs)
    if key == "gdelt":
        return GdeltProvider(api_key=api_key, transport=transport, **kwargs)
    raise NewsProviderError("provider_error", f"unknown provider: {name}")


def provider_requires_api_key(name: str) -> bool:
    """Whether a provider needs an API key (GDELT is free/keyless)."""
    return (name or "").strip().lower() != "gdelt"
