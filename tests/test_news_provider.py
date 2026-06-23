"""Unit tests for the NewsAPI provider adapter (no network)."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.news_provider import (
    HttpResponse,
    NewsApiProvider,
    NewsProviderError,
    get_provider,
)

API_KEY = "SECRET-KEY-DO-NOT-LEAK-123"


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(articles, total):
    return HttpResponse(status=200, body={"status": "ok", "totalResults": total, "articles": articles})


def _article(i):
    return {
        "source": {"name": f"src{i}"},
        "title": f"title {i}",
        "description": f"desc {i}",
        "url": f"https://example.com/{i}",
        "publishedAt": "2022-01-01T12:00:00Z",
    }


def test_pagination_collects_all_pages():
    transport = FakeTransport(
        [
            _ok([_article(1), _article(2)], total=3),
            _ok([_article(3)], total=3),
        ]
    )
    provider = NewsApiProvider(API_KEY, transport=transport, page_size=2)
    out = provider.fetch_day("bitcoin", date(2022, 1, 1))
    assert len(out) == 3
    assert len(transport.calls) == 2
    # page numbers advanced
    assert "page=1" in transport.calls[0]["url"]
    assert "page=2" in transport.calls[1]["url"]


def test_429_is_retried_then_succeeds():
    sleeps = []
    transport = FakeTransport(
        [
            HttpResponse(status=429, body={"status": "error", "code": "rateLimited"}),
            HttpResponse(status=429, body={"status": "error", "code": "rateLimited"}),
            _ok([_article(1)], total=1),
        ]
    )
    provider = NewsApiProvider(
        API_KEY, transport=transport, page_size=100, sleep=lambda s: sleeps.append(s)
    )
    out = provider.fetch_day("bitcoin", date(2022, 1, 1))
    assert len(out) == 1
    assert len(transport.calls) == 3
    assert len(sleeps) == 2  # backed off twice before success


def test_429_exhausted_raises_rate_limited():
    transport = FakeTransport(
        [HttpResponse(status=429, body={"status": "error", "code": "rateLimited"})] * 10
    )
    provider = NewsApiProvider(
        API_KEY, transport=transport, max_retries=2, sleep=lambda s: None
    )
    with pytest.raises(NewsProviderError) as ei:
        provider.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "rate_limited"


def test_result_truncation_raises():
    transport = FakeTransport([_ok([_article(i) for i in range(100)], total=150)])
    provider = NewsApiProvider(API_KEY, transport=transport, page_size=100, sleep=lambda s: None)
    with pytest.raises(NewsProviderError) as ei:
        provider.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "result_truncated"


def test_provider_error_is_not_retried():
    transport = FakeTransport(
        [HttpResponse(status=401, body={"status": "error", "code": "apiKeyInvalid"})]
    )
    provider = NewsApiProvider(API_KEY, transport=transport, sleep=lambda s: None)
    with pytest.raises(NewsProviderError) as ei:
        provider.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "provider_error"
    assert len(transport.calls) == 1


def test_api_key_is_in_header_not_url():
    transport = FakeTransport([_ok([_article(1)], total=1)])
    provider = NewsApiProvider(API_KEY, transport=transport)
    provider.fetch_day("bitcoin", date(2022, 1, 1))
    call = transport.calls[0]
    assert API_KEY not in call["url"]
    assert call["headers"].get("X-Api-Key") == API_KEY


def test_factory_unknown_provider():
    with pytest.raises(NewsProviderError):
        get_provider("nope", API_KEY)
