"""Strategy building blocks. A strategy = one ENTRY block + up to two FILTER blocks + an EXIT plan
+ which sides it trades. The search combines and tunes these; it never sees future bars.

Each entry returns (long, short) boolean arrays: True at bar i means "signal on the close of bar i",
and the trade is entered at the next bar's open.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .indicators import Data, shift


@dataclass
class Param:
    kind: str            # "int" | "float" | "choice"
    lo: float = 0
    hi: float = 0
    step: float = 1
    choices: tuple = ()

    def sample(self, rng: np.random.Generator):
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            return int(rng.integers(int(self.lo), int(self.hi) + 1))
        v = rng.uniform(self.lo, self.hi)
        return round(round(v / self.step) * self.step, 6)

    def mutate(self, v, rng: np.random.Generator, scale: float = 0.2):
        if self.kind == "choice":
            return self.sample(rng)
        span = (self.hi - self.lo) * scale
        nv = v + rng.normal(0, span)
        nv = min(self.hi, max(self.lo, nv))
        if self.kind == "int":
            nv = int(round(nv))
            if nv == v:
                nv = int(min(self.hi, max(self.lo, v + rng.choice([-1, 1]))))
            return nv
        return round(round(nv / self.step) * self.step, 6)


@dataclass
class Block:
    name: str
    family: str
    params: dict[str, Param]
    fn: Callable
    describe: Callable
    overlay: Callable | None = None


def _cross_up(a, b):
    return (a > b) & (shift(a) <= shift(b))


def _cross_dn(a, b):
    return (a < b) & (shift(a) >= shift(b))


def _nan0(x):
    return np.nan_to_num(x, nan=0.0).astype(bool)


# --- ENTRY BLOCKS -------------------------------------------------------------------------
def donchian(d: Data, p):
    n = p["n"]
    return _nan0(d.c > d.hh(n)), _nan0(d.c < d.ll(n))


def ma_cross(d: Data, p):
    f, s = d.ema(p["fast"]), d.ema(max(p["slow"], p["fast"] + 5))
    return _nan0(_cross_up(f, s)), _nan0(_cross_dn(f, s))


def rsi_revert(d: Data, p):
    r = d.rsi(p["n"])
    lo = p["level"]
    return _nan0(_cross_up(r, np.full_like(r, lo))), _nan0(_cross_dn(r, np.full_like(r, 100 - lo)))


def band_revert(d: Data, p):
    m, s = d.sma(p["n"]), d.std(p["n"])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (d.c - m) / s
    k = p["k"]
    return _nan0(_cross_up(z, np.full_like(z, -k))), _nan0(_cross_dn(z, np.full_like(z, k)))


def squeeze_break(d: Data, p):
    n = p["n"]
    m, s = d.sma(n), d.std(n)
    width = d.get(("bbw", n), lambda: (4 * s) / m)
    from .indicators import pct_rank
    rank = d.get(("bbw_rank", n), lambda: pct_rank(width, 300))
    was_tight = _nan0(shift(rank) < p["q"])
    up, dn = m + 2 * s, m - 2 * s
    return was_tight & _nan0(d.c > up), was_tight & _nan0(d.c < dn)


def trend_pullback(d: Data, p):
    f, s = d.ema(p["fast"]), d.ema(max(p["slow"], p["fast"] + 10))
    up = (f > s) & (s > shift(s, 5))
    dn = (f < s) & (s < shift(s, 5))
    long_ = up & (d.l <= f) & (d.c > f) & (d.c > d.o)
    short = dn & (d.h >= f) & (d.c < f) & (d.c < d.o)
    return _nan0(long_), _nan0(short)


def momentum(d: Data, p):
    n = p["n"]
    roc = (d.c - shift(d.c, n)) / d.atr(14)
    thr = p["thr"]
    return _nan0(_cross_up(roc, np.full_like(roc, thr))), _nan0(_cross_dn(roc, np.full_like(roc, -thr)))


def structure_break(d: Data, p):
    ph, pl = d.pivots(p["k"])
    return _nan0(_cross_up(d.c, ph)), _nan0(_cross_dn(d.c, pl))


def nr_break(d: Data, p):
    nr = shift(d.range_rank(p["n"]).astype(float)) > 0.5
    return _nan0(nr & (d.c > shift(d.h))), _nan0(nr & (d.c < shift(d.l)))


def sweep_reversal(d: Data, p):
    """Liquidity sweep: price pokes beyond the n-bar extreme then closes back inside."""
    n = p["n"]
    lo, hi = d.ll(n), d.hh(n)
    rng_ok = (d.h - d.l) > p["min_range"] * d.atr(14)
    long_ = (d.l < lo) & (d.c > lo) & rng_ok
    short = (d.h > hi) & (d.c < hi) & rng_ok
    return _nan0(long_), _nan0(short)


def range_expansion(d: Data, p):
    """Volatility expansion bar closing near its extreme: continuation."""
    a = d.atr(14)
    big = (d.h - d.l) > p["k"] * shift(a)
    pos = (d.c - d.l) / np.maximum(d.h - d.l, 1e-12)
    return _nan0(big & (pos > 0.8)), _nan0(big & (pos < 0.2))


def opening_range(d: Data, p):
    """Breakout of the first `bars` bars after a session opens at `hour` UTC."""
    hour, nb = p["hour"], p["bars"]
    n = d.n
    key = ("orb", hour, nb)

    def calc():
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        start = (d.hour == hour) & (d.minute == 0)
        cur_h = cur_l = np.nan
        count = -1
        day_end = 0
        for i in range(n):
            if start[i]:
                cur_h, cur_l, count = d.h[i], d.l[i], 1
                day_end = i + max(1, int(6 * 3600 / d.secs))   # trade the break for ~6 hours
                continue
            if 0 < count < nb:
                cur_h, cur_l = max(cur_h, d.h[i]), min(cur_l, d.l[i])
                count += 1
                continue
            if count >= nb and i <= day_end:
                hi[i], lo[i] = cur_h, cur_l
        return hi, lo
    hi, lo = d.get(key, calc)
    return _nan0(_cross_up(d.c, hi)), _nan0(_cross_dn(d.c, lo))


def time_of_day(d: Data, p):
    """Pure seasonality: enter at a given hour (session effect). The side gene picks direction."""
    at = (d.hour == p["hour"]) & (d.minute == 0)
    return at, at.copy()


ENTRIES: dict[str, Block] = {b.name: b for b in [
    Block("donchian", "breakout", {"n": Param("int", 10, 120)}, donchian,
          lambda p: f"close breaks the {p['n']}-bar high/low",
          lambda d, p: {"channel high": d.hh(p["n"]), "channel low": d.ll(p["n"])}),
    Block("ma_cross", "trend", {"fast": Param("int", 5, 50), "slow": Param("int", 20, 250)}, ma_cross,
          lambda p: f"EMA{p['fast']} crosses EMA{max(p['slow'], p['fast'] + 5)}",
          lambda d, p: {f"EMA{p['fast']}": d.ema(p["fast"]), f"EMA{max(p['slow'], p['fast'] + 5)}": d.ema(max(p["slow"], p["fast"] + 5))}),
    Block("rsi_revert", "mean_reversion", {"n": Param("int", 2, 21), "level": Param("float", 5, 35, 1)}, rsi_revert,
          lambda p: f"RSI{p['n']} turns back from {p['level']:.0f}/{100 - p['level']:.0f}", None),
    Block("band_revert", "mean_reversion", {"n": Param("int", 10, 100), "k": Param("float", 1.5, 3.5, 0.1)}, band_revert,
          lambda p: f"close returns inside {p['k']:.1f}-sigma band of SMA{p['n']}",
          lambda d, p: {f"SMA{p['n']}": d.sma(p["n"]), "upper band": d.sma(p["n"]) + p["k"] * d.std(p["n"]),
                        "lower band": d.sma(p["n"]) - p["k"] * d.std(p["n"])}),
    Block("squeeze_break", "volatility", {"n": Param("int", 10, 60), "q": Param("float", 0.05, 0.35, 0.01)}, squeeze_break,
          lambda p: f"Bollinger({p['n']}) squeeze (width < {p['q']:.0%} pct) then breakout",
          lambda d, p: {"upper band": d.sma(p["n"]) + 2 * d.std(p["n"]), "lower band": d.sma(p["n"]) - 2 * d.std(p["n"])}),
    Block("trend_pullback", "trend", {"fast": Param("int", 10, 50), "slow": Param("int", 50, 300)}, trend_pullback,
          lambda p: f"pullback to EMA{p['fast']} in an EMA{p['fast']}/{max(p['slow'], p['fast'] + 10)} trend",
          lambda d, p: {f"EMA{p['fast']}": d.ema(p["fast"]), f"EMA{max(p['slow'], p['fast'] + 10)}": d.ema(max(p["slow"], p["fast"] + 10))}),
    Block("momentum", "momentum", {"n": Param("int", 3, 60), "thr": Param("float", 0.5, 4.0, 0.1)}, momentum,
          lambda p: f"{p['n']}-bar move exceeds {p['thr']:.1f} ATR", None),
    Block("structure_break", "structure", {"k": Param("int", 2, 10)}, structure_break,
          lambda p: f"break of the last confirmed swing (strength {p['k']})",
          lambda d, p: {"swing high": d.pivots(p["k"])[0], "swing low": d.pivots(p["k"])[1]}),
    Block("nr_break", "volatility", {"n": Param("int", 4, 12)}, nr_break,
          lambda p: f"break of an NR{p['n']} (narrowest range in {p['n']}) bar", None),
    Block("sweep_reversal", "structure", {"n": Param("int", 10, 100), "min_range": Param("float", 0.5, 2.5, 0.1)}, sweep_reversal,
          lambda p: f"sweep of the {p['n']}-bar extreme that closes back inside",
          lambda d, p: {"prior high": d.hh(p["n"]), "prior low": d.ll(p["n"])}),
    Block("range_expansion", "volatility", {"k": Param("float", 1.2, 3.5, 0.1)}, range_expansion,
          lambda p: f"bar range > {p['k']:.1f} ATR closing at its extreme", None),
    Block("opening_range", "session", {"hour": Param("int", 0, 23), "bars": Param("int", 1, 6)}, opening_range,
          lambda p: f"breakout of the first {p['bars']} bar(s) after {p['hour']:02d}:00 UTC", None),
    Block("time_of_day", "session", {"hour": Param("int", 0, 23)}, time_of_day,
          lambda p: f"enters at {p['hour']:02d}:00 UTC (time-of-day effect)", None),
]}


# --- FILTERS --------------------------------------------------------------------------------
def f_trend(d: Data, p):
    e = d.ema(p["n"])
    up, dn = d.c > e, d.c < e
    return (up, dn) if p["mode"] == "with" else (dn, up)


def f_htf(d: Data, p):
    e = d.htf_ema(p["factor"], p["n"])
    up, dn = _nan0(d.c > e), _nan0(d.c < e)
    return (up, dn) if p["mode"] == "with" else (dn, up)


def f_vol(d: Data, p):
    r = d.atr_rank(500)
    lo, hi = sorted((p["lo"], p["hi"]))
    ok = _nan0((r >= lo) & (r <= max(hi, lo + 0.15)))
    return ok, ok


def f_er(d: Data, p):
    e = d.er(p["n"])
    ok = (e > p["thr"]) if p["mode"] == "trending" else (e < p["thr"])
    return ok, ok


def f_session(d: Data, p):
    hrs = (d.hour - p["start"]) % 24
    ok = hrs < p["length"]
    return ok, ok


def f_weekday(d: Data, p):
    ok = d.dow != p["skip"]
    return ok, ok


FILTERS: dict[str, Block] = {b.name: b for b in [
    Block("trend", "filter", {"n": Param("int", 50, 300), "mode": Param("choice", choices=("with", "against"))}, f_trend,
          lambda p: f"{'with' if p['mode'] == 'with' else 'against'} the EMA{p['n']} trend",
          lambda d, p: {f"EMA{p['n']}": d.ema(p["n"])}),
    Block("htf_trend", "filter", {"factor": Param("choice", choices=(4, 12, 24)), "n": Param("int", 20, 100),
                                  "mode": Param("choice", choices=("with", "against"))}, f_htf,
          lambda p: f"{p['mode']} the higher-timeframe (x{p['factor']}) EMA{p['n']}",
          lambda d, p: {f"HTF EMA{p['n']}": d.htf_ema(p["factor"], p["n"])}),
    Block("vol_regime", "filter", {"lo": Param("float", 0.0, 0.8, 0.05), "hi": Param("float", 0.3, 1.0, 0.05)}, f_vol,
          lambda p: f"volatility percentile between {min(p['lo'], p['hi']):.0%} and {max(p['lo'], p['hi']):.0%}", None),
    Block("efficiency", "filter", {"n": Param("int", 10, 60), "thr": Param("float", 0.1, 0.6, 0.05),
                                   "mode": Param("choice", choices=("trending", "ranging"))}, f_er,
          lambda p: f"market {p['mode']} (efficiency ratio{p['n']} {'>' if p['mode'] == 'trending' else '<'} {p['thr']:.2f})", None),
    Block("session", "filter", {"start": Param("int", 0, 23), "length": Param("int", 3, 14)}, f_session,
          lambda p: f"only {p['start']:02d}:00-{(p['start'] + p['length']) % 24:02d}:00 UTC", None),
    Block("weekday", "filter", {"skip": Param("choice", choices=(0, 1, 2, 3, 4))}, f_weekday,
          lambda p: f"skip {['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][p['skip']]}", None),
]}


# --- EXIT PLAN ----------------------------------------------------------------------------
EXIT_PARAMS: dict[str, Param] = {
    "stop_mode": Param("choice", choices=("atr", "swing")),
    "sl_atr": Param("float", 0.8, 4.0, 0.1),        # ATR multiple (or buffer beyond the swing for "swing")
    "swing_n": Param("int", 3, 30),
    "rr": Param("choice", choices=(0.0, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)),   # 0 = no fixed target
    "trail": Param("choice", choices=("none", "chandelier", "breakeven")),
    "trail_atr": Param("float", 1.0, 5.0, 0.1),
    "be_r": Param("float", 0.5, 2.0, 0.1),
    "max_bars": Param("int", 4, 300),
}
SIDES = ("long", "short", "both")


def describe_exit(x: dict) -> str:
    stop = f"stop {x['sl_atr']:.1f} ATR" if x["stop_mode"] == "atr" else f"stop beyond {x['swing_n']}-bar swing"
    tgt = f"target {x['rr']:g}R" if x["rr"] else "no fixed target"
    trail = {"none": "", "chandelier": f", trail {x['trail_atr']:.1f} ATR", "breakeven": f", break-even at {x['be_r']:.1f}R"}[x["trail"]]
    return f"{stop}, {tgt}{trail}, max {x['max_bars']} bars"
