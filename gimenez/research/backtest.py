"""Backtester with deliberately pessimistic fills.

* Signal on bar i's close -> market entry at bar i+1's open, paying half the spread + slippage.
* Bars are treated as mid prices: longs exit on the bid (mid - half spread), shorts on the ask.
* A stop is a market order: it slips, and if the bar OPENS beyond the stop (a gap) it fills at the open.
* A target is a limit order: no slippage, and no price improvement.
* If one bar touches both the stop and the target we assume the stop came first.
* Holding costs (swap) are charged per day held.
* Trailing stops only use bars that already closed (no peeking inside the current bar).

`ExitState` is the same exit logic one bar at a time; it runs shadow and real trades live,
and a test checks it matches the vectorised backtest exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .genome import Genome
from .indicators import Data

WARMUP = 300
DEFAULT_SPREAD_FRAC = {"fx": 0.00012, "index": 0.00012, "metal": 0.00025, "energy": 0.0005, "crypto": 0.0008,
                       "stock": 0.0005, "other": 0.0008}


@dataclass
class Costs:
    spread: float              # typical spread in price units (already multiplied by the safety factor)
    slip_atr: float = 0.02     # slippage on market fills, in ATRs
    swap_per_day: float = 0.0002   # fraction of price per day held

    @classmethod
    def for_market(cls, price: float, asset_class: str, recorded_spread: float | None, cfg) -> "Costs":
        base = recorded_spread if recorded_spread and recorded_spread > 0 else DEFAULT_SPREAD_FRAC.get(asset_class, 0.0008) * price
        return cls(base * cfg.spread_multiplier, cfg.slippage_atr, cfg.swap_per_day)


@dataclass
class Trades:
    side: list = field(default_factory=list)       # +1 long, -1 short
    entry_i: list = field(default_factory=list)
    exit_i: list = field(default_factory=list)
    entry: list = field(default_factory=list)
    exit: list = field(default_factory=list)
    stop: list = field(default_factory=list)
    target: list = field(default_factory=list)
    r: list = field(default_factory=list)
    mfe: list = field(default_factory=list)
    mae: list = field(default_factory=list)
    reason: list = field(default_factory=list)

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.r, dtype=float)

    def __len__(self) -> int:
        return len(self.r)

    def select(self, lo: int, hi: int) -> np.ndarray:
        """R of trades whose ENTRY bar is in [lo, hi)."""
        e = np.asarray(self.entry_i)
        return self.R[(e >= lo) & (e < hi)] if len(e) else np.zeros(0)


def initial_stop(d: Data, i: int, side: int, entry: float, ex: dict, atr_i: float) -> float | None:
    if ex["stop_mode"] == "swing":
        n = int(ex["swing_n"])
        lo = max(0, i - n + 1)
        stop = (d.l[lo:i + 1].min() - 0.2 * atr_i) if side > 0 else (d.h[lo:i + 1].max() + 0.2 * atr_i)
    else:
        stop = entry - side * ex["sl_atr"] * atr_i
    dist = (entry - stop) * side
    if not np.isfinite(dist) or dist < 0.5 * atr_i or dist > 8 * atr_i:
        return None   # too tight (latency-scalping territory) or absurdly wide
    return float(stop)


class ExitState:
    """Manages one open trade bar by bar, exactly like the backtest does."""

    def __init__(self, side: int, entry: float, stop: float, target: float | None, ex: dict, atr_i: float,
                 half_spread: float, slip: float, swap_per_bar: float) -> None:
        self.side, self.entry, self.stop0 = side, entry, stop
        self.risk = abs(entry - stop)
        self.target = target
        self.ex, self.atr = ex, atr_i
        self.half, self.slip, self.swap_per_bar = half_spread, slip, swap_per_bar
        self.stop_now = stop
        self.best = -np.inf          # best favourable (exit-side) price seen on CLOSED bars
        self.bars = 0
        self.mfe = 0.0
        self.mae = 0.0

    def _fav(self, h, l):
        return (h - self.half) if self.side > 0 else -(l + self.half)

    def update(self, o: float, h: float, l: float, c: float) -> tuple[float, str] | None:
        """Process one closed bar. Returns (exit_price, reason) if the trade ended in it."""
        s, half = self.side, self.half
        self.bars += 1
        ob, hb, lb, cb = (o - half, h - half, l - half, c - half) if s > 0 else (o + half, h + half, l + half, c + half)
        stop_hit = (lb <= self.stop_now) if s > 0 else (hb >= self.stop_now)
        tgt_hit = self.target is not None and ((hb >= self.target) if s > 0 else (lb <= self.target))
        gap_tgt = self.target is not None and ((ob >= self.target) if s > 0 else (ob <= self.target))
        fav = (hb - self.entry) * s if s > 0 else (self.entry - lb)
        adv = (self.entry - lb) if s > 0 else (hb - self.entry)
        self.mfe = max(self.mfe, fav / self.risk)
        self.mae = max(self.mae, adv / self.risk)
        result = None
        if stop_hit and not gap_tgt:
            px = min(self.stop_now, ob) if s > 0 else max(self.stop_now, ob)
            result = (px - s * self.slip, "stop" if (self.stop_now - self.stop0) * s <= 0 else "trail")
        elif tgt_hit:
            result = (self.target, "target")
        elif self.bars >= self.ex["max_bars"]:
            result = (cb - s * self.slip, "time")
        if result:
            return result
        # tighten the stop for the NEXT bar using this closed bar
        self.best = max(self.best, self._fav(h, l))
        best_px = self.best if s > 0 else -self.best
        st = self.stop_now
        if self.ex["trail"] == "chandelier":
            cand = best_px - s * self.ex["trail_atr"] * self.atr
            st = max(st, cand) if s > 0 else min(st, cand)
        elif self.ex["trail"] == "breakeven":
            if (best_px - self.entry) * s >= self.ex["be_r"] * self.risk:
                st = max(st, self.entry) if s > 0 else min(st, self.entry)
        self.stop_now = st
        return None

    def r_multiple(self, exit_price: float) -> float:
        return ((exit_price - self.entry) * self.side - self.swap_per_bar * self.bars) / self.risk


def run(genome: Genome, d: Data, costs: Costs, start: int = 0, end: int | None = None,
        signals: tuple[np.ndarray, np.ndarray] | None = None) -> Trades:
    """Backtest one genome on bars [start, end). Indicators use all earlier bars as warm-up."""
    end = d.n if end is None else min(end, d.n)
    long_, short = signals if signals is not None else genome.signals(d)
    a = d.atr(14)
    ex = genome.exit
    half = costs.spread / 2
    swap_per_bar = costs.swap_per_day * d.secs / 86400.0
    out = Trades()
    sig_idx = np.flatnonzero(long_[:end - 1] | short[:end - 1])
    sig_idx = sig_idx[sig_idx >= max(start, WARMUP)]
    next_free = 0
    max_bars = int(ex["max_bars"])
    for i in sig_idx:
        if i < next_free or i + 1 >= d.n:
            continue
        atr_i = a[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        side = 1 if long_[i] else -1
        slip = costs.slip_atr * atr_i
        entry = d.o[i + 1] + side * (half + slip)
        stop = initial_stop(d, i, side, entry, ex, atr_i)
        if stop is None:
            continue
        risk = abs(entry - stop)
        target = entry + side * ex["rr"] * risk if ex["rr"] else None
        st = ExitState(side, entry, stop, target, ex, atr_i, half, slip, swap_per_bar * entry)
        j_end = min(d.n, i + 1 + max_bars)
        res = _vector_exit(st, d, i + 1, j_end)
        if res is None:   # ran out of data: close at the last close
            j = j_end - 1
            px = d.c[j] - side * (half + slip)
            res = (j, px, "end")
        j, px, why = res
        st.bars = j - i
        out.side.append(side); out.entry_i.append(int(i + 1)); out.exit_i.append(int(j))
        out.entry.append(entry); out.exit.append(float(px)); out.stop.append(stop)
        out.target.append(target if target is not None else np.nan)
        out.r.append(float(st.r_multiple(px))); out.reason.append(why)
        out.mfe.append(st.mfe); out.mae.append(st.mae)
        next_free = j + 1
    return out


def _vector_exit(st: ExitState, d: Data, j0: int, j1: int):
    """Vectorised version of ExitState.update over bars [j0, j1)."""
    s, half, ex = st.side, st.half, st.ex
    if j1 <= j0:
        return None
    o, h, l, c = d.o[j0:j1], d.h[j0:j1], d.l[j0:j1], d.c[j0:j1]
    if s > 0:
        ob, hb, lb, cb = o - half, h - half, l - half, c - half
        fav = hb
    else:
        ob, hb, lb, cb = o + half, h + half, l + half, c + half
        fav = -lb
    prev_best = np.concatenate(([-np.inf], np.maximum.accumulate(fav)[:-1]))
    best_px = prev_best if s > 0 else -prev_best
    stop = np.full(len(o), st.stop0)
    if ex["trail"] == "chandelier":
        cand = best_px - s * ex["trail_atr"] * st.atr
        stop = np.maximum(stop, np.where(np.isfinite(cand), cand, -np.inf)) if s > 0 else \
            np.minimum(stop, np.where(np.isfinite(cand), cand, np.inf))
        # stops only ever tighten
        stop = np.maximum.accumulate(stop) if s > 0 else np.minimum.accumulate(stop)
    elif ex["trail"] == "breakeven":
        reached = (best_px - st.entry) * s >= ex["be_r"] * st.risk
        reached = np.maximum.accumulate(reached)
        stop = np.where(reached, (np.maximum(stop, st.entry) if s > 0 else np.minimum(stop, st.entry)), stop)
    stop_hit = (lb <= stop) if s > 0 else (hb >= stop)
    if st.target is not None:
        tgt_hit = (hb >= st.target) if s > 0 else (lb <= st.target)
        gap_tgt = (ob >= st.target) if s > 0 else (ob <= st.target)
    else:
        tgt_hit = gap_tgt = np.zeros(len(o), bool)
    stop_hit = stop_hit & ~gap_tgt
    any_exit = stop_hit | tgt_hit
    k_time = int(ex["max_bars"]) - 1
    k = int(np.argmax(any_exit)) if any_exit.any() else None
    if k is None or k > k_time:
        if k_time < len(o):
            k, kind = k_time, "time"
        else:
            _mfe_mae(st, hb, lb, len(o))
            return None
    else:
        kind = "stop" if stop_hit[k] else "target"
    _mfe_mae(st, hb, lb, k + 1)
    if kind == "stop":
        px = min(stop[k], ob[k]) if s > 0 else max(stop[k], ob[k])
        return j0 + k, px - s * st.slip, ("stop" if (stop[k] - st.stop0) * s <= 0 else "trail")
    if kind == "target":
        return j0 + k, st.target, "target"
    return j0 + k, cb[k] - s * st.slip, "time"


def _mfe_mae(st: ExitState, hb, lb, k: int) -> None:
    if k <= 0:
        return
    if st.side > 0:
        st.mfe = max(0.0, float((hb[:k].max() - st.entry) / st.risk))
        st.mae = max(0.0, float((st.entry - lb[:k].min()) / st.risk))
    else:
        st.mfe = max(0.0, float((st.entry - lb[:k].min()) / st.risk))
        st.mae = max(0.0, float((hb[:k].max() - st.entry) / st.risk))
