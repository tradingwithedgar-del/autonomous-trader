"""Synthetic markets for tests and the honesty check.

`random_walk` has NO edge by construction: any strategy that "passes" on it is a false
discovery. `planted_breakout` hides a real (known) momentum edge after channel breakouts,
so we can check the search can find something that is really there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..broker.base import TIMEFRAME_SECONDS


def _bars_from_path(closes: np.ndarray, sig: np.ndarray, rng: np.random.Generator, start: str, timeframe: str,
                    price0: float) -> pd.DataFrame:
    n = len(closes)
    opens = np.empty(n)
    opens[0] = price0
    opens[1:] = closes[:-1]
    wick_up = np.abs(rng.normal(0, 0.6, n)) * sig * closes
    wick_dn = np.abs(rng.normal(0, 0.6, n)) * sig * closes
    highs = np.maximum(opens, closes) + wick_up
    lows = np.minimum(opens, closes) - wick_dn
    idx = pd.date_range(start, periods=n, freq=pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe]), tz="UTC")
    vol = rng.integers(50, 500, n).astype(float)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx)


def _vol_path(n: int, base: float, rng: np.random.Generator, timeframe: str) -> np.ndarray:
    # slowly drifting volatility regime + intraday seasonality (busier around 13-17 UTC)
    regime = np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.5)
    regime = regime / regime.mean()
    secs = TIMEFRAME_SECONDS[timeframe]
    hours = (np.arange(n) * secs / 3600.0) % 24
    season = 0.8 + 0.5 * np.exp(-((hours - 15) ** 2) / 8)
    return base * regime * season


def random_walk(n: int = 6000, timeframe: str = "15m", seed: int = 0, price0: float = 100.0,
                vol: float = 0.0015, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sig = _vol_path(n, vol, rng, timeframe)
    rets = rng.standard_t(5, n) / np.sqrt(5 / 3) * sig
    closes = price0 * np.exp(np.cumsum(rets))
    return _bars_from_path(closes, sig, rng, start, timeframe, price0)


def planted_breakout(n: int = 6000, timeframe: str = "15m", seed: int = 0, price0: float = 100.0,
                     vol: float = 0.0015, edge: float = 0.25, lookback: int = 20, hold: int = 12,
                     start: str = "2025-01-01") -> pd.DataFrame:
    """After the close breaks the prior `lookback`-bar high (low), the next `hold` bars drift up (down)
    by `edge` x volatility per bar. A real, exploitable momentum effect."""
    rng = np.random.default_rng(seed)
    sig = _vol_path(n, vol, rng, timeframe)
    noise = rng.standard_t(5, n) / np.sqrt(5 / 3) * sig
    closes = np.empty(n)
    p = price0
    drift_left, drift_dir = 0, 0
    for i in range(n):
        r = noise[i] + (drift_dir * edge * sig[i] if drift_left > 0 else 0.0)
        drift_left = max(0, drift_left - 1)
        p *= np.exp(r)
        closes[i] = p
        if i > lookback and drift_left == 0:
            window = closes[i - lookback:i]
            if p > window.max():
                drift_left, drift_dir = hold, 1
            elif p < window.min():
                drift_left, drift_dir = hold, -1
    return _bars_from_path(closes, sig, rng, start, timeframe, price0)
