"""Unit tests for FinBERT scoring + content-hash cache (no model download)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import news_scoring as ns
from src.news_scoring import SentimentScorerError


class FakeScorer:
    model_name = "fake-finbert"
    model_version = "v1"

    def __init__(self):
        self.calls = []

    def score_texts(self, texts):
        self.calls.append(list(texts))
        # Deterministic per content so identical texts get identical scores.
        return [round(0.01 * len(t), 6) for t in texts]


def test_content_hash_is_deterministic_and_content_sensitive():
    h1 = ns.content_hash("Title", "Body")
    h2 = ns.content_hash("Title", "Body")
    h3 = ns.content_hash("Title", "Other")
    assert h1 == h2
    assert h1 != h3


def test_score_records_maps_cache_rows_to_articles_by_hash():
    cache = {}
    scorer = FakeScorer()
    recs = [
        {"title": "A", "body": "x"},
        {"title": "B", "body": "yy"},
        {"title": "A", "body": "x"},  # duplicate content of rec[0]
    ]
    new = ns.score_records(recs, scorer, cache)
    assert new == 2  # only two unique contents scored

    # Duplicate content -> same hash and same score.
    assert recs[0]["content_hash"] == recs[2]["content_hash"]
    assert recs[0]["sentiment_score"] == recs[2]["sentiment_score"]

    # Every article's score is provably its cache row (hash + model identity).
    for r in recs:
        row = cache[r["content_hash"]]
        assert row["content_hash"] == r["content_hash"]
        assert row["sentiment_score"] == r["sentiment_score"]
        assert row["model_name"] == "fake-finbert"
        assert row["model_version"] == "v1"


def test_cache_hit_skips_rescoring():
    cache = {}
    scorer = FakeScorer()
    recs = [{"title": "A", "body": "x"}]
    ns.score_records(recs, scorer, cache)
    assert len(scorer.calls) == 1

    # Re-running with the same content must not call the scorer again.
    recs2 = [{"title": "A", "body": "x"}]
    new = ns.score_records(recs2, scorer, cache)
    assert new == 0
    assert len(scorer.calls) == 1
    assert recs2[0]["sentiment_score"] == recs[0]["sentiment_score"]


def test_score_only_written_after_successful_score():
    cache = {}
    scorer = FakeScorer()
    rec = {"title": "A", "body": "x"}
    assert "sentiment_score" not in rec
    ns.score_records([rec], scorer, cache)
    assert "sentiment_score" in rec


def test_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    cache = {"abc": {"content_hash": "abc", "sentiment_score": 0.5,
                     "model_name": "fake-finbert", "model_version": "v1"}}
    ns.save_cache(cache, path)
    loaded = ns.load_cache(path)
    assert loaded["abc"]["sentiment_score"] == 0.5


def test_finbert_unavailable_raises_explicit_error():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    if installed:
        pytest.skip("transformers/torch installed; the unavailable path is not exercised")
    with pytest.raises(SentimentScorerError):
        ns.FinbertScorer()
