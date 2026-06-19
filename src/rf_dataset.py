"""Per-symbol RF training dataset schema and assembly (spec sections 10 & 11).

Every symbol gets its OWN dataset; rows from one symbol must never leak into
another symbol's training set. The label is the policy-improvement target::

    target_policy_improvement = PnL_if_continue_from_this_minute - PnL_if_close_now

(`continue` minus `close`; with the default 0 threshold the policy continues
when continuing is expected to beat closing.)
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

INTEGRAL_EDGE_FEATURES = [
    "LongEdge_Return",
    "LongEdge_Mean",
    "LongEdge_Var",
    "LongEdge_MeanOfMean",
    "LongEdge_VarOfMean",
    "ShortEdge_Return",
    "ShortEdge_Mean",
    "ShortEdge_Var",
    "ShortEdge_MeanOfMean",
    "ShortEdge_VarOfMean",
]

OPTIONAL_COMPONENT_FEATURES = [
    "LongRightPnL_Return", "LongLeftPnL_Return",
    "LongRightPnL_Mean", "LongLeftPnL_Mean",
    "LongRightPnL_Var", "LongLeftPnL_Var",
    "LongRightPnL_MeanOfMean", "LongLeftPnL_MeanOfMean",
    "LongRightPnL_VarOfMean", "LongLeftPnL_VarOfMean",
    "ShortLeftPnL_Return", "ShortRightPnL_Return",
    "ShortLeftPnL_Mean", "ShortRightPnL_Mean",
    "ShortLeftPnL_Var", "ShortRightPnL_Var",
    "ShortLeftPnL_MeanOfMean", "ShortRightPnL_MeanOfMean",
    "ShortLeftPnL_VarOfMean", "ShortRightPnL_VarOfMean",
]

STATE_FEATURES = [
    "side",
    "mode",
    "CurrentPrice",
    "EntryPrice",
    "y",
    "current_pnl",
    "distance_to_liq",
    "distance_to_liq_pct",
    "distance_to_first_liq",
    "remaining_balance",
    "N_open",
    "M_open_current",
    "liquidation_cutoff",
    "hour_of_day",
    "day_of_week",
]

VOLUME_FEATURES = [
    "predicted_daily_volume",
    "previous_day_real_volume",
    "predicted_volume_change_pct",
    "current_minute_volume",
    "intraday_volume_so_far",
    "intraday_volume_pct_of_predicted",
    "last_5m_volume",
    "last_15m_volume",
    "last_60m_volume",
]

NEWS_FEATURES = [
    "macro_news_sentiment",
    "policy_news_sentiment",
    "stock_market_news_sentiment",
    "crypto_market_news_sentiment",
    "symbol_specific_news_sentiment",
    "macro_news_count",
    "policy_news_count",
    "stock_market_news_count",
    "crypto_market_news_count",
    "symbol_specific_news_count",
    # Correlation-weighted cross-coin aggregates (spec sections 4-5). Per-source
    # `news_sentiment_from_<SYMBOL>_weighted` columns are discovered dynamically.
    "weighted_symbol_news_sentiment",
    "weighted_symbol_news_count",
    "weighted_positive_symbol_news_count",
    "weighted_negative_symbol_news_count",
]

# Raw price levels differ in scale across coins and must NOT be fed to the ML
# model directly; use normalized price-derived features (y, returns, distances).
PRICE_EXCLUDED_FROM_ML = ["CurrentPrice", "EntryPrice"]

SEASON_FEATURES = [
    "days_since_last_halving",
    "days_to_next_halving",
    "halving_cycle_progress",
    "halving_sin",
    "halving_cos",
]

SIDE_ENCODING = {"long": 0, "short": 1, "both": 2}
MODE_ENCODING = {
    "HEDGED_BOTH_ACTIVE": 0,
    "LONG_ONLY_AFTER_SHORT_LIQ": 1,
    "SHORT_ONLY_AFTER_LONG_LIQ": 2,
}

TARGET_COLUMN = "target_policy_improvement"
META_COLUMNS = ["symbol", "timestamp", "pnl_if_continue", "pnl_if_close"]


def policy_improvement(pnl_if_continue: float, pnl_if_close: float) -> float:
    """Label: PnL_if_continue - PnL_if_close (spec section 10)."""
    return float(pnl_if_continue) - float(pnl_if_close)


def base_feature_columns(include_optional: bool = False) -> List[str]:
    cols = list(INTEGRAL_EDGE_FEATURES)
    if include_optional:
        cols += list(OPTIONAL_COMPONENT_FEATURES)
    cols += STATE_FEATURES + VOLUME_FEATURES + NEWS_FEATURES + SEASON_FEATURES
    return cols


def make_row(
    symbol: str,
    timestamp,
    features: Dict[str, float],
    pnl_if_continue: float,
    pnl_if_close: float,
) -> Dict[str, object]:
    """Build a single dataset row dict with the policy-improvement label."""
    row: Dict[str, object] = dict(features)
    row["symbol"] = symbol
    row["timestamp"] = timestamp
    row["pnl_if_continue"] = float(pnl_if_continue)
    row["pnl_if_close"] = float(pnl_if_close)
    row[TARGET_COLUMN] = policy_improvement(pnl_if_continue, pnl_if_close)
    return row


def to_frame(rows: Iterable[Dict[str, object]], include_optional: bool = False) -> pd.DataFrame:
    """Assemble rows into a frame with a stable column ordering and encodings."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        cols = base_feature_columns(include_optional) + META_COLUMNS + [TARGET_COLUMN]
        return pd.DataFrame(columns=cols)

    if "side" in df.columns:
        df["side_code"] = df["side"].map(SIDE_ENCODING).fillna(-1).astype(int)
    if "mode" in df.columns:
        df["mode_code"] = df["mode"].map(MODE_ENCODING).fillna(-1).astype(int)

    feature_cols = base_feature_columns(include_optional)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def numeric_feature_columns(include_optional: bool = False) -> List[str]:
    """Feature columns usable directly by the RF (side/mode replaced by codes).

    Raw price levels are excluded (section 12); side/mode become numeric codes.
    """
    cols = base_feature_columns(include_optional)
    excluded = {"side", "mode", *PRICE_EXCLUDED_FROM_ML}
    cols = [c for c in cols if c not in excluded]
    return cols + ["side_code", "mode_code"]


def news_like_columns(df: pd.DataFrame) -> List[str]:
    """Discover correlation-weighted / per-source news columns present in a frame."""
    skip = set(META_COLUMNS) | {TARGET_COLUMN, "side", "mode"}

    def is_news(col: str) -> bool:
        return (
            col.endswith("_news_sentiment")
            or col.endswith("_news_count")
            or col.endswith("_weighted")
        )

    return [c for c in df.columns if is_news(c) and c not in skip]


def model_feature_columns(df: pd.DataFrame, include_optional: bool = False) -> List[str]:
    """Final RF feature order: static numeric columns + discovered news columns.

    Only columns present in ``df`` are kept so training and the metadata stay in
    sync, and the Rust live engine consumes the exact same order.
    """
    base = [c for c in numeric_feature_columns(include_optional) if c in df.columns]
    extras = [c for c in news_like_columns(df) if c not in base]
    return base + extras


def split_by_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return only rows for ``symbol`` (spec: never mix symbols)."""
    if "symbol" not in df.columns:
        return df.iloc[0:0]
    return df[df["symbol"] == symbol].reset_index(drop=True)


def dataset_path(rf_dataset_dir: str, symbol: str) -> str:
    return os.path.join(rf_dataset_dir, f"rf_decision_dataset_{symbol}.parquet")


def write_dataset(df: pd.DataFrame, rf_dataset_dir: str, symbol: str) -> str:
    os.makedirs(rf_dataset_dir, exist_ok=True)
    path = dataset_path(rf_dataset_dir, symbol)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Parquet engine (pyarrow/fastparquet) unavailable -> CSV fallback.
        path = path.replace(".parquet", ".csv")
        df.to_csv(path, index=False)
    return path


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
