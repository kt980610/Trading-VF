"""FinBERT sentiment scoring with a content-addressed cache.

Scoring model
-------------
Default model: ``ProsusAI/finbert`` (3-class: positive / negative / neutral).
For each text we compute softmax probabilities and map to a single signed score::

    sentiment_score = P(positive) - P(negative)   in [-1.0, +1.0]

so neutral articles land near 0.0, strongly positive near +1.0 and strongly
negative near -1.0.

Cache
-----
Every scored text is keyed by a SHA-256 hash of ``title + "\\n" + body``. Each
cache row records ``content_hash``, ``sentiment_score``, ``model_name`` and
``model_version`` so the score for any article is always provable and
reproducible. A cached score is only reused when BOTH the model name and version
match the active scorer.

There is NO silent lexicon fallback: if FinBERT (transformers/torch or the model
weights) cannot be loaded, scoring raises :class:`SentimentScorerError`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

FINBERT_MODEL_NAME = "ProsusAI/finbert"
SCORER_VERSION = "finbert-1"


class SentimentScorerError(Exception):
    """Raised when the requested sentiment backend cannot be used."""


def content_hash(title: Optional[str], body: Optional[str]) -> str:
    base = f"{title or ''}\n{body or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def text_for_scoring(record: dict) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("body") or "").strip()
    if title and body:
        return f"{title}. {body}"
    return title or body


class FinbertScorer:
    """Lazy FinBERT scorer. Construction loads the model or raises (no fallback)."""

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.model_version = SCORER_VERSION
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise SentimentScorerError(
                "finbert backend unavailable: install 'transformers' and 'torch' "
                "(no silent lexicon fallback)"
            ) from exc

        try:
            self._torch = __import__("torch")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            self._device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            # Map model label order -> indices for positive/negative.
            self._labels = {
                v.lower(): k for k, v in self._model.config.id2label.items()
            }
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - needs weights
            raise SentimentScorerError(
                f"failed to load FinBERT model '{model_name}': {exc}"
            ) from exc

    def score_texts(self, texts: List[str]) -> List[float]:  # pragma: no cover - needs weights
        if not texts:
            return []
        torch = self._torch
        pos_idx = self._labels.get("positive")
        neg_idx = self._labels.get("negative")
        if pos_idx is None or neg_idx is None:
            raise SentimentScorerError("FinBERT model missing positive/negative labels")
        out: List[float] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=256, padding=True
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
            for row in probs:
                out.append(float(row[pos_idx].item() - row[neg_idx].item()))
        return out


# ---------------------------------------------------------------------------
# Cache I/O.
# ---------------------------------------------------------------------------
def load_cache(path: str) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not os.path.isfile(path):
        return cache
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            h = row.get("content_hash")
            if h:
                cache[h] = row
    return cache


def save_cache(cache: Dict[str, dict], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in cache.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def has_scorable_text(record: dict) -> bool:
    """True only when the record carries real title/body text to score.

    A record with no title and no body (e.g. a GDELT ``ArtList`` hit reduced to a
    URL + tone) is NOT scorable: FinBERT must never be run on empty text and no
    sentiment score may be fabricated for it.
    """
    return bool(text_for_scoring(record).strip())


def score_records(records: List[dict], scorer, cache: Dict[str, dict]) -> int:
    """Attach ``sentiment_score`` to records, scoring uncached texts via ``scorer``.

    ``sentiment_score`` is written ONLY for records with real text, and only after
    a score is successfully produced. Text-less records (no title/body) get an
    explicit ``sentiment_score=None`` and are never sent to FinBERT (their GDELT
    tone, if any, is preserved separately upstream). Returns the number of
    newly-scored (cache-miss) texts. ``cache`` is mutated.
    """
    to_score: Dict[str, str] = {}
    for rec in records:
        if not has_scorable_text(rec):
            # No text -> no content hash, no FinBERT, no fabricated sentiment.
            rec["content_hash"] = None
            continue
        h = content_hash(rec.get("title"), rec.get("body"))
        rec["content_hash"] = h
        cached = cache.get(h)
        if (
            cached is not None
            and cached.get("model_name") == scorer.model_name
            and cached.get("model_version") == scorer.model_version
        ):
            continue
        to_score[h] = text_for_scoring(rec)

    new_count = 0
    if to_score:
        hashes = list(to_score)
        scores = scorer.score_texts([to_score[h] for h in hashes])
        if len(scores) != len(hashes):
            raise SentimentScorerError("scorer returned wrong number of scores")
        for h, s in zip(hashes, scores):
            cache[h] = {
                "content_hash": h,
                "sentiment_score": float(s),
                "model_name": scorer.model_name,
                "model_version": scorer.model_version,
            }
            new_count += 1

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for rec in records:
        h = rec.get("content_hash")
        if not h:
            # Text-less record: explicit "no sentiment" (not a neutral score).
            rec["sentiment_score"] = None
            if not rec.get("scored_at"):
                rec["scored_at"] = now_iso
            continue
        row = cache[h]
        rec["sentiment_score"] = row["sentiment_score"]
        rec["sentiment_model"] = row["model_name"]
        rec["sentiment_model_version"] = row["model_version"]
        # ``scored_at`` is the live availability time used by the as-of join. Stamp
        # it only once (first successful scoring) so it persists across runs.
        if not rec.get("scored_at"):
            rec["scored_at"] = now_iso
    return new_count
