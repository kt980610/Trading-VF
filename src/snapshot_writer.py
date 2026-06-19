"""Assembling and writing the distribution snapshot JSON."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_snapshot(method: str, bins: int, symbols: dict) -> dict:
    return {
        "created_at": utc_now_iso(),
        "method": method,
        "bins": bins,
        "symbols": symbols,
    }


def write_snapshot(snapshot: dict, output_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, default=_json_default, indent=2)
    return output_path


def load_snapshot(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
