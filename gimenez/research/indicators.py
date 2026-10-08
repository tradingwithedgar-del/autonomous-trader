"""Indicators on numpy arrays. Every value at index i uses bars <= i only (no look-ahead).
`Data` caches results so a search that evaluates thousands of ideas computes each indicator once."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..broker.base import TIMEFRAME_SECONDS


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1)
    return pd.Series(x).ewm(alpha=a, adjust=False, min_periods=n).mean().to_numpy()


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def rstd(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def rmax(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def rmin(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k] if k > 0 else x
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    return pd.Series(true_range(h, l, c)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = np.array((100 - 100 / (1 + rs)).to_numpy(), dtype=float)
    out[(dn.to_numpy() == 0) & (up.to_numpy() > 0)] = 100.0
    return out


def efficiency_ratio(c: np.ndarray, n: int) -> np.ndarray:
    """Kaufman: net move / path length over n bars. ~1 = clean trend, ~0 = chop."""
    net = np.abs(c - shift(c, n))
    path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(path > 0, net / path, 0.0)


def pct_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Percentile (0-1) of the current value within the last n values."""
    s = pd.Series(x)
    return s.rolling(n, min_periods=max(20, n // 4)).rank(pct=True).to_numpy()


def pivots(h: np.ndarray, l: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Last *confirmed* swing high/low. A pivot of strength k is only known k bars after it formed."""
    roll_hi = pd.Series(h).rolling(2 * k + 1, center=True).max().to_numpy()
    roll_lo = pd.Series(l).rolling(2 * k + 1, center=True).min().to_numpy()
    piv_hi = pd.Series(np.where(h == roll_hi, h, np.nan)).shift(k).ffill().to_numpy()
    piv_lo = pd.Series(np.where(l == roll_lo, l, np.nan)).shift(k).ffill().to_numpy()
    return piv_hi, piv_lo


class Data:
    """One market's bars at one timeframe, plus a cache of indicators."""

    def __init__(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "15m") -> None:
        self.df = df
        self.symbol, self.tf = symbol, timeframe
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.v = df["volume"].to_numpy(dtype=float) if "volume" in df else np.zeros(len(df))
        self.index = df.index
        self.n = len(df)
        self.secs = TIMEFRAME_SECONDS.get(timeframe, 900)
        self.hour = np.asarray(df.index.hour, dtype=int) if self.n else np.zeros(0, int)
        self.minute = np.asarray(df.index.minute, dtype=int) if self.n else np.zeros(0, int)
        self.dow = np.asarray(df.index.dayofweek, dtype=int) if self.n else np.zeros(0, int)
        self._cache: dict = {}

    def slice(self, start: int, end: int) -> "Data":
        return Data(self.df.iloc[start:end], self.symbol, self.tf)

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # common indicators
    def atr(self, n: int = 14) -> np.ndarray:
        return self.get(("atr", n), lambda: atr(self.h, self.l, self.c, n))

    def ema(self, n: int) -> np.ndarray:
        return self.get(("ema", n), lambda: ema(self.c, n))

    def sma(self, n: int) -> np.ndarray:
        return self.get(("sma", n), lambda: sma(self.c, n))

    def std(self, n: int) -> np.ndarray:
        return self.get(("std", n), lambda: rstd(self.c, n))

    def rsi(self, n: int) -> np.ndarray:
        return self.get(("rsi", n), lambda: rsi(self.c, n))

    def hh(self, n: int) -> np.ndarray:
        """Highest high of the n bars BEFORE this one."""
        return self.get(("hh", n), lambda: shift(rmax(self.h, n)))

    def ll(self, n: int) -> np.ndarray:
        return self.get(("ll", n), lambda: shift(rmin(self.l, n)))

    def er(self, n: int) -> np.ndarray:
        return self.get(("er", n), lambda: efficiency_ratio(self.c, n))

    def atr_rank(self, n: int = 500) -> np.ndarray:
        return self.get(("atr_rank", n), lambda: pct_rank(self.atr(14) / self.c, n))

    def pivots(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get(("piv", k), lambda: pivots(self.h, self.l, k))

    def htf_ema(self, factor: int, n: int) -> np.ndarray:
        """EMA(n) of a higher timeframe (factor x this one), using only CLOSED higher-timeframe bars."""
        def calc():
            rule = pd.Timedelta(seconds=self.secs * factor)
            closes = pd.Series(self.c, index=self.index).resample(rule, label="left", closed="left").last().dropna()
            e = closes.ewm(span=n, adjust=False, min_periods=n).mean().shift(1)   # last closed HTF bar only
            return e.reindex(self.index, method="ffill").to_numpy()
        return self.get(("htf", factor, n), calc)

    def range_rank(self, n: int) -> np.ndarray:
        """True if this bar's range is the narrowest of the last n (NR-n)."""
        def calc():
            r = self.h - self.l
            return r <= rmin(r, n) + 1e-12
        return self.get(("nr", n), calc)
