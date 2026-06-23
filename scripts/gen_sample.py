"""Generate a small synthetic daily CSV for smoke-testing the pipeline."""

import os

import numpy as np
import pandas as pd

out_dir = os.path.join("config", "data", "raw")
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(42)
n = 400
rets = rng.normal(0.0008, 0.02, n)
close = 100.0 * np.cumprod(1.0 + rets)
days = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
df = pd.DataFrame(
    {
        "timestamp": days,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1000, 5000, n),
    }
)
df.to_csv(os.path.join(out_dir, "BTCUSDT_daily.csv"), index=False)
print("wrote", len(df), "rows to", out_dir)
