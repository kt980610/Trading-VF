"""CLI: score canonical raw news with FinBERT, writing back ``sentiment_score``.

Example::

    python -m src.main_score_news \
        --config config/distribution_config.yaml \
        --backend finbert

Scores title + description with FinBERT (``P(positive) - P(negative)`` in
[-1, 1]) and caches every score by a SHA-256 content hash so each article's score
is provable and reused across runs. If FinBERT cannot be loaded, this fails
loudly (no silent lexicon fallback). ``sentiment_score`` is written into the raw
partitions only after a score is successfully produced.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src import news_ingest as ni
    from src import news_scoring as ns
else:
    from .config import load_config
    from . import news_ingest as ni
    from . import news_scoring as ns


def _build_scorer(backend: str):
    backend = (backend or "").strip().lower()
    if backend == "finbert":
        return ns.FinbertScorer()
    raise SystemExit(f"unsupported sentiment backend for scoring: {backend!r} (use finbert)")


def _has_unscored(records) -> bool:
    for r in records:
        # Text-less records (e.g. GDELT URL-only hits) are never scorable, so a
        # missing score for them is NOT outstanding work and must not force a
        # FinBERT reload every incremental run.
        if not ns.has_scorable_text(r):
            continue
        score = r.get("sentiment_score")
        if score is None or (isinstance(score, float) and score != score):
            return True
    return False


def run(config_path: str, backend: str = "finbert", scorer=None, incremental: bool = False) -> dict:
    config = load_config(config_path)
    news = config.news
    raw_path = config.resolve(news.raw_path)
    cache_path = config.resolve(news.sentiment_cache_path)

    cache = ns.load_cache(cache_path)

    files = ni.list_partition_files(raw_path)
    if not files:
        raise SystemExit(f"no raw news found under {ni.partition_dir(raw_path)} or {raw_path}")

    # Lazily build the scorer so an incremental run with nothing to do never
    # pays the cost of loading FinBERT.
    _scorer = scorer

    def get_scorer():
        nonlocal _scorer
        if _scorer is None:
            _scorer = _build_scorer(backend or news.sentiment_backend)
        return _scorer

    total_records = 0
    total_new = 0
    for path in files:
        records = ni.read_jsonl(path)
        if not records:
            continue
        # In incremental mode skip partitions that are already fully scored; the
        # content-hash cache still guarantees no article is ever rescored.
        if incremental and not _has_unscored(records):
            total_records += len(records)
            continue
        total_new += ns.score_records(records, get_scorer(), cache)
        ni.write_jsonl_atomic(records, path)
        total_records += len(records)

    scorer = _scorer or scorer
    ns.save_cache(cache, cache_path)
    summary = {
        "files": len(files),
        "records": total_records,
        "newly_scored": total_new,
        "cache_rows": len(cache),
        "model_name": getattr(scorer, "model_name", backend or news.sentiment_backend),
        "model_version": getattr(scorer, "model_version", "n/a"),
        "cache_path": cache_path,
    }
    print(
        "scored: files={files} records={records} newly_scored={newly_scored} "
        "cache_rows={cache_rows} model={model_name}/{model_version}".format(**summary)
    )
    print(f"sentiment cache -> {summary['cache_path']}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score raw news with FinBERT.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument("--backend", default="finbert", help="sentiment backend (finbert)")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="only score partitions that still have unscored articles (live worker)",
    )
    args = parser.parse_args(argv)
    run(args.config, backend=args.backend, incremental=args.incremental)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
