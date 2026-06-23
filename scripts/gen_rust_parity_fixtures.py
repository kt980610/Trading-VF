"""Generate Python->Rust parity fixtures for the live close classifier.

Outputs (under rust_live/tests/fixtures/):

* ``rf_close_classifier.json`` - a REAL sklearn RandomForestClassifier exported
  in the exact `src/rf_classifier.export_model_json` format (leaf P(close), median
  imputer, RobustScaler center/scale, scale/passthrough split).
* ``parity_rows.json`` - >=15 feature rows with `p_close` taken straight from
  `clf.predict_proba()[:,1]` and the close/continue decision at the threshold.
* ``halving_cases.json`` - season selections (incl. an equidistant SHA tie case)
  produced by `src/halving_season.select_season`.

The Rust test `close_classifier_tests.rs` loads these and asserts byte-for-byte
numeric parity (1e-9) for p_close and identical season selection.

Run once (requires numpy + scikit-learn):

    python scripts/gen_rust_parity_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys

# Make the repo root importable so `from src...` works when this script is run
# directly (e.g. `python scripts/gen_rust_parity_fixtures.py`) without requiring
# PYTHONPATH to be set externally.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from src import news_features as nf
from src.halving_season import parse_halving_dates, select_season
from src.portfolio import weights_writer as ww
from src.portfolio.config import PortfolioConfig
from src.portfolio.mvo import MVOStats

FIX_DIR = os.path.join("rust_live", "tests", "fixtures")
MODEL_VERSION = "rf_close_classifier_v1"

SCALE_COLS = [
    "LongEdge_Return",
    "ShortEdge_Return",
    "mdp_return_integral_updated",
    "intraday_volume_so_far",
    "current_volume_to_predicted_daily_volume",
]
PASSTHROUGH_COLS = [
    "side_code",
    "mode_code",
    "season_pre_halving_2y",
    "season_post_halving_2y",
    "season_unknown",
    "halving_cycle_id",
]
FINAL_ORDER = SCALE_COLS + PASSTHROUGH_COLS


def _export_tree(est, class1_idx: int) -> dict:
    t = est.tree_
    value = []
    for node in range(t.node_count):
        counts = t.value[node][0]
        total = float(counts.sum())
        value.append(float(counts[class1_idx] / total) if total > 0 else 0.0)
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": value,
    }


def build_model():
    rng = np.random.default_rng(7)
    n = 600
    x_scale = rng.normal(size=(n, len(SCALE_COLS)))
    # Passthrough categorical-ish columns.
    side = rng.integers(0, 2, size=n).astype(float)
    mode = rng.integers(0, 3, size=n).astype(float)
    # One-hot season (exactly one of three set).
    season = np.zeros((n, 3))
    pick = rng.integers(0, 3, size=n)
    season[np.arange(n), pick] = 1.0
    cycle = rng.integers(0, 4, size=n).astype(float)
    x_pass = np.column_stack([side, mode, season, cycle])

    # Label depends on a few features so the trees actually split.
    logits = x_scale[:, 0] + 0.6 * x_scale[:, 2] - 0.4 * side + 0.3 * (pick == 0)
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(int)

    medians = {c: float(np.median(x_scale[:, j])) for j, c in enumerate(SCALE_COLS)}
    scaler = RobustScaler().fit(x_scale)
    x_scaled = scaler.transform(x_scale)
    x_mat = np.hstack([x_scaled, x_pass])

    clf = RandomForestClassifier(n_estimators=11, max_depth=5, random_state=0)
    clf.fit(x_mat, y)
    class1 = list(clf.classes_).index(1) if 1 in list(clf.classes_) else len(clf.classes_) - 1

    export = {
        "type": "random_forest_classifier",
        "model_version": MODEL_VERSION,
        "symbol": "BTCUSDT",
        "threshold": 0.5,
        "final_feature_order": FINAL_ORDER,
        "scale_cols": SCALE_COLS,
        "passthrough_cols": PASSTHROUGH_COLS,
        "imputer": {"strategy": "median", "medians": medians},
        "scaler": {
            "type": "robust",
            "center": [float(c) for c in scaler.center_],
            "scale": [float(s) for s in scaler.scale_],
        },
        "trees": [_export_tree(est, class1) for est in clf.estimators_],
    }

    # Parity rows straight from predict_proba.
    proba = clf.predict_proba(x_mat)[:, class1]
    rows = []
    for i in range(20):
        feat = {c: float(x_scale[i, j]) for j, c in enumerate(SCALE_COLS)}
        for j, c in enumerate(PASSTHROUGH_COLS):
            feat[c] = float(x_pass[i, j])
        p = float(proba[i])
        rows.append({"features": feat, "p_close": p, "decision": "CLOSE" if p >= 0.5 else "CONTINUE"})

    return export, rows


def build_halving_cases():
    cases = []
    default_h = parse_halving_dates()
    default_iso = [t.isoformat() for t in default_h]
    for ts in [
        "2024-05-01T00:00:00+00:00",
        "2020-06-01T00:00:00+00:00",
        "2017-01-01T00:00:00+00:00",
        "2005-01-01T00:00:00+00:00",  # unknown
    ]:
        label, cid = select_season("BTCUSDT", pd.Timestamp(ts), default_h, 0)
        cases.append(
            {
                "symbol": "BTCUSDT",
                "timestamp": ts,
                "seed": 0,
                "halvings": default_iso,
                "season": label,
                "cycle_id": int(cid),
            }
        )

    # Equidistant tie: two halvings 1000 days apart, the midpoint is in BOTH
    # windows and equidistant -> SHA-256 tie-break must match Rust.
    h0 = pd.Timestamp("2020-01-01T00:00:00+00:00")
    h1 = h0 + pd.Timedelta(days=1000)
    tie_iso = [h0.isoformat(), h1.isoformat()]
    for (sym, seed) in [("BTCUSDT", 0), ("ETHUSDT", 0), ("BTCUSDT", 42), ("SOLUSDT", 7)]:
        mid = h0 + pd.Timedelta(days=500)
        label, cid = select_season(sym, mid, [h0, h1], seed)
        cases.append(
            {
                "symbol": sym,
                "timestamp": mid.isoformat(),
                "seed": seed,
                "halvings": tie_iso,
                "season": label,
                "cycle_id": int(cid),
            }
        )
    return cases


def build_intraday_news():
    """As-of intraday news fixture proving the lag-at-selection semantics.

    The artifact records carry lag-free content as of their run time. A decision
    at ``D`` must select the newest record with ``asof <= D - safety_lag``; that
    record's features must equal the training join ``asof_news_provider(D)``.
    """
    universe = ["BTCUSDT", "ETHUSDT"]
    lag = 300
    # GDELT rows: the instant is an OBSERVATION time (seendate) -> observed_utc,
    # treated identically to exact_utc for the window but never relabelled as a
    # verified publish time.
    obs = nf.TS_OBSERVED
    raw = pd.DataFrame(
        [
            {"timestamp": "2026-06-20T08:00:00Z", "category": "symbol_specific", "symbol": "BTCUSDT", "title": "", "body": "", "sentiment_score": 0.4, "gdelt_tone": 1.5, "timestamp_quality": obs},
            {"timestamp": "2026-06-20T08:30:00Z", "category": "symbol_specific", "symbol": "ETHUSDT", "title": "", "body": "", "sentiment_score": -0.6, "gdelt_tone": -2.0, "timestamp_quality": obs},
            {"timestamp": "2026-06-20T09:00:00Z", "category": "macro", "symbol": None, "title": "", "body": "", "sentiment_score": 0.5, "gdelt_tone": 0.5, "timestamp_quality": obs},
            {"timestamp": "2026-06-20T09:30:00Z", "category": "crypto_market", "symbol": None, "title": "", "body": "", "sentiment_score": -0.2, "gdelt_tone": -1.0, "timestamp_quality": obs},
            # Seen between the two snapshots -> only affects the later decision.
            {"timestamp": "2026-06-20T12:30:00Z", "category": "macro", "symbol": None, "title": "", "body": "", "sentiment_score": -1.0, "gdelt_tone": -3.0, "timestamp_quality": obs},
        ]
    )
    news_df = nf.prepare_news(raw, backend="lexicon")
    # GDELT provenance/coverage features are part of the live + training schema.
    news_source = "gdelt"

    # Worker run times (lag-free content snapshots, safety_lag=0). Each carries
    # provenance (all exact_utc here -> intraday_asof / null source date).
    run_times = ["2026-06-20T11:55:00Z", "2026-06-20T13:55:00Z"]
    artifact = []
    for asof in run_times:
        for symbol in universe:
            feats, prov = nf.asof_news_for_decision(
                news_df, symbol, universe, pd.Timestamp(asof),
                corr_provider=None, lookback_hours=24, safety_lag_seconds=0,
                news_source=news_source,
            )
            observed = prov["timestamp_quality"] == nf.TS_OBSERVED
            rec = {
                "asof_timestamp": asof,
                "symbol": symbol,
                "feature_version": "news_v1",
                "published_at": None if observed else prov["published_at"],
                "source_seen_at": prov["published_at"] if observed else None,
                "available_at": prov["available_at"],
                "news_mode": prov["news_mode"],
                "news_source": prov["news_source"],
                "timestamp_quality": prov["timestamp_quality"],
                "source_feature_date": prov["source_feature_date"],
            }
            rec.update(feats)
            artifact.append(rec)

    # Decisions: selected asof == D - lag (a run time exists exactly there). The
    # expected features come from the SHARED training join so live == train.
    expected = []
    for decision, exp_asof in [
        ("2026-06-20T12:00:00Z", "2026-06-20T11:55:00Z"),
        ("2026-06-20T14:00:00Z", "2026-06-20T13:55:00Z"),
    ]:
        for symbol in universe:
            feats, prov = nf.asof_news_for_decision(
                news_df, symbol, universe, pd.Timestamp(decision),
                corr_provider=None, lookback_hours=24, safety_lag_seconds=lag,
                news_source=news_source,
            )
            expected.append(
                {"decision_timestamp": decision, "safety_lag_seconds": lag,
                 "symbol": symbol, "asof_timestamp": exp_asof,
                 "news_mode": prov["news_mode"],
                 "news_source": prov["news_source"],
                 "timestamp_quality": prov["timestamp_quality"],
                 "source_feature_date": prov["source_feature_date"],
                 "features": {k: float(v) for k, v in feats.items()}}
            )
    return artifact, expected


def build_sizing_parity():
    """Portfolio-weight sizing parity: the Python MVO artifact plus the canonical
    targets (``target_margin = total_equity * weight``, ``target_notional =
    target_margin * leverage``) the Rust ``sizing::allocate_symbol`` must match.
    """
    symbols = ["BTCUSDT", "ETHUSDT"]
    config = PortfolioConfig(
        symbols=symbols,
        weight_steps={"BTCUSDT": 0.01, "ETHUSDT": 0.01},
        lookback_days=30,
        integer_tolerance=1e-9,
    )
    stats = MVOStats(
        symbols=symbols,
        mu=np.array([0.002, 0.0015]),
        variance=np.array([0.01, 0.02]),
        sigma_reg=np.array([[0.01, 0.0], [0.0, 0.02]]),
    )
    continuous_w = np.array([0.35, 0.25])
    discrete_k = np.array([35, 25])
    discrete_w = np.array([0.35, 0.25])
    payload = ww.build_payload(
        config, "2026-06-20", stats, symbols, {}, continuous_w, discrete_k, discrete_w
    )

    total_equity = 12_500.0
    leverage = {"BTCUSDT": 5.0, "ETHUSDT": 3.0}
    expected = []
    for symbol in symbols:
        weight = float(payload["weights"][symbol])
        lev = leverage[symbol]
        target_margin = total_equity * weight
        expected.append(
            {
                "symbol": symbol,
                "weight": weight,
                "leverage": lev,
                "target_margin": target_margin,
                "target_notional": target_margin * lev,
            }
        )
    meta = {"total_equity": total_equity, "expected": expected}
    return payload, meta


def main():
    os.makedirs(FIX_DIR, exist_ok=True)
    export, rows = build_model()
    with open(os.path.join(FIX_DIR, "rf_close_classifier.json"), "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2)
    with open(os.path.join(FIX_DIR, "parity_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(FIX_DIR, "halving_cases.json"), "w", encoding="utf-8") as fh:
        json.dump(build_halving_cases(), fh, indent=2)

    art, expected = build_intraday_news()
    with open(os.path.join(FIX_DIR, "intraday_news.jsonl"), "w", encoding="utf-8") as fh:
        for rec in art:
            fh.write(json.dumps(rec) + "\n")
    with open(os.path.join(FIX_DIR, "intraday_news_expected.json"), "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=2)

    sizing_weights, sizing_meta = build_sizing_parity()
    with open(os.path.join(FIX_DIR, "sizing_weights.json"), "w", encoding="utf-8") as fh:
        json.dump(sizing_weights, fh, indent=2)
    with open(os.path.join(FIX_DIR, "sizing_parity_expected.json"), "w", encoding="utf-8") as fh:
        json.dump(sizing_meta, fh, indent=2)

    print(
        f"wrote fixtures to {FIX_DIR}: rf_close_classifier.json, parity_rows.json, "
        "halving_cases.json, intraday_news.jsonl, intraday_news_expected.json, "
        "sizing_weights.json, sizing_parity_expected.json"
    )
    print(
        f"parity rows: {len(rows)} intraday cases: {len(expected)} "
        f"sizing cases: {len(sizing_meta['expected'])}"
    )


if __name__ == "__main__":
    main()
