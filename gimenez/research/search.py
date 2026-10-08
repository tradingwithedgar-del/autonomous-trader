"""Evolutionary strategy search + the validation gauntlet.

Data for one market/timeframe is split in time:

    [ warm-up | in-sample: fold1 fold2 fold3 fold4 | HOLDOUT (most recent 30%) ]

1. The genetic search only ever sees the in-sample part. Fitness rewards a statistically strong
   AND consistent edge (t-stat x share of profitable folds), with a penalty for complexity.
2. The best few candidates then face, in order (cheapest first, holdout last):
   a. minimum trades and positive in every-fold consistency
   b. deflated Sharpe: beats the best Sharpe expected from luck given ALL ideas tried so far.
      The luck benchmark uses each trial's sampling variance under "no edge" (about 1/trades),
      not the spread of the trials' Sharpes, which real edges would inflate.
   c. parameter plateau: most nearby parameter sets must be profitable too (no knife-edge optimum)
   d. walk-forward: re-pick parameters on past data only, trade the next fold, repeat
   e. sibling markets: the same idea must not lose money on similar instruments
   f. the untouched holdout, with a significance threshold split (Bonferroni) across every
      look ever taken at that holdout, and minimum trades / expectancy / profit factor.
3. Survivors go to shadow (virtual) trading on live data before any real money is risked.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import ResearchConfig
from . import backtest as bt
from .genome import Genome
from .indicators import Data
from .stats import deflated_sharpe, summarize

log = logging.getLogger(__name__)


@dataclass
class Market:
    data: Data
    costs: bt.Costs
    asset_class: str = "other"


def split_points(n: int, cfg: ResearchConfig) -> tuple[int, list[tuple[int, int]]]:
    """Holdout start and the in-sample folds."""
    is_end = int(n * (1 - cfg.holdout_fraction))
    start = bt.WARMUP
    edges = np.linspace(start, is_end, cfg.is_folds + 1).astype(int)
    return is_end, [(int(edges[k]), int(edges[k + 1])) for k in range(cfg.is_folds)]


@dataclass
class Evaluation:
    genome: Genome
    fitness: float
    stats: dict
    folds: list[dict]
    trades: bt.Trades | None = None


def evaluate(g: Genome, m: Market, lo: int, hi: int, folds: list[tuple[int, int]], cfg: ResearchConfig,
             min_trades: int | None = None) -> Evaluation:
    try:
        tr = bt.run(g, m.data, m.costs, start=lo, end=hi)
    except Exception as e:  # a broken idea is just a bad idea
        log.debug("genome %s failed: %s", g.id, e)
        return Evaluation(g, -99.0, summarize(np.zeros(0)), [])
    r = tr.select(lo, hi)
    st = summarize(r)
    fs = [summarize(tr.select(a, b)) for a, b in folds]
    need = cfg.min_trades_is if min_trades is None else min_trades
    if st["n"] < need:
        fit = -5.0 + st["n"] / max(1, need)
    else:
        good = sum(1 for f in fs if f["n"] >= max(3, cfg.min_trades_fold * len(fs) // max(1, cfg.is_folds)) and f["mean"] > 0)
        frac = good / max(1, len(fs))
        fit = st["t"] * (0.5 + 0.5 * frac) - 0.15 * g.complexity()
    return Evaluation(g, float(fit), st, fs, tr)


@dataclass
class SearchResult:
    symbol: str
    tf: str
    trials: int = 0
    inv_n: list = field(default_factory=list)   # 1/trades of each trial: the null variance of its Sharpe
    best: list = field(default_factory=list)          # Evaluations, best first
    seconds: float = 0.0


def evolve(m: Market, cfg: ResearchConfig, rng: np.random.Generator, deadline: float | None = None,
           hints: dict | None = None, seeds: list[Genome] | None = None) -> SearchResult:
    d = m.data
    is_end, folds = split_points(d.n, cfg)
    res = SearchResult(d.symbol, d.tf)
    t0 = time.monotonic()
    seen: dict[str, Evaluation] = {}

    def ev(g: Genome) -> Evaluation:
        if g.id in seen:
            return seen[g.id]
        e = evaluate(g, m, bt.WARMUP, is_end, folds, cfg)
        seen[g.id] = e
        res.trials += 1
        if e.stats["n"] >= 20:
            res.inv_n.append(1.0 / e.stats["n"])
        e.trades = None   # free memory; re-run for survivors
        return e

    # diverse start: every entry family appears
    from .blocks import ENTRIES
    pop = [Genome.random(d.symbol, d.tf, rng, et, hints) for et in ENTRIES]
    pop += [s.for_symbol(d.symbol) for s in (seeds or [])]
    while len(pop) < cfg.population:
        pop.append(Genome.random(d.symbol, d.tf, rng, hints=hints))
    scored = [ev(g) for g in pop]
    for gen in range(cfg.generations):
        if deadline and time.monotonic() > deadline:
            break
        scored.sort(key=lambda e: e.fitness, reverse=True)
        nxt = [e.genome for e in scored[:cfg.elite]]
        while len(nxt) < cfg.population:
            a, b = _tournament(scored, rng), _tournament(scored, rng)
            child = a.genome.crossover(b.genome, rng) if rng.random() < 0.5 else a.genome
            if rng.random() < 0.9:
                child = child.mutate(rng, cfg.mutation_rate)
            if rng.random() < 0.1:
                child = Genome.random(d.symbol, d.tf, rng, hints=hints)   # fresh blood
            nxt.append(child)
        scored = [ev(g) for g in nxt]
    allev = sorted(seen.values(), key=lambda e: e.fitness, reverse=True)
    # keep the best of each entry family so survivors aren't all near-copies of one idea
    best, fams = [], set()
    for e in allev:
        key = (e.genome.entry["type"], e.genome.side)
        if key in fams:
            continue
        fams.add(key)
        best.append(e)
    res.best = best
    res.seconds = time.monotonic() - t0
    return res


def _tournament(scored: list[Evaluation], rng: np.random.Generator, k: int = 3) -> Evaluation:
    picks = [scored[int(rng.integers(len(scored)))] for _ in range(k)]
    return max(picks, key=lambda e: e.fitness)


# --- validation -------------------------------------------------------------------------------
@dataclass
class Verdict:
    genome: Genome
    passed: bool
    stage_reached: str
    reasons: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def validate(e: Evaluation, m: Market, cfg: ResearchConfig, rng: np.random.Generator, n_trials: int,
             var_trials: float, peers: list[Market], register_look: Callable[[], int]) -> Verdict:
    g, d = e.genome, m.data
    is_end, folds = split_points(d.n, cfg)
    rep: dict = {"in_sample": _clean(e.stats), "folds": [_clean(f) for f in e.folds]}
    v = Verdict(g, False, "in-sample", report=rep)

    # a. enough trades, consistent across time
    st = e.stats
    if st["n"] < cfg.min_trades_is or st["mean"] <= 0:
        v.reasons.append(f"in-sample: {st['n']} trades, {st['mean']:+.3f}R each - not enough or not positive")
        return v
    good = sum(1 for f in e.folds if f["mean"] > 0)
    if good < len(e.folds) - 1:
        v.reasons.append(f"inconsistent: profitable in only {good}/{len(e.folds)} time periods")
        return v

    # b. deflated Sharpe
    dsr = deflated_sharpe(st["sharpe"], st["n"], st["skew"], st["kurt"], n_trials, var_trials)
    rep["dsr"] = dsr
    rep["trials_so_far"] = n_trials
    v.stage_reached = "deflated-sharpe"
    if dsr < cfg.min_dsr:
        v.reasons.append(f"deflated Sharpe {dsr:.2f} < {cfg.min_dsr}: after {n_trials} ideas, this looks like luck")
        return v

    # c. parameter plateau
    neigh = g.neighbors(rng, 16)
    nev = [evaluate(x, m, bt.WARMUP, is_end, folds, cfg, min_trades=cfg.min_trades_is // 2) for x in neigh]
    plateau = float(np.mean([x.stats["mean"] > 0 for x in nev]))
    rep["plateau"] = plateau
    v.stage_reached = "plateau"
    if plateau < cfg.min_plateau:
        v.reasons.append(f"knife-edge: only {plateau:.0%} of nearby parameter sets are profitable")
        return v

    # d. walk-forward (re-pick params on the past, trade the next fold)
    oos: list[np.ndarray] = []
    pool = [g] + neigh[:8]
    for k in range(1, len(folds) - 1):
        train_end = folds[k][1]
        test_lo, test_hi = folds[k + 1]
        need = max(10, int(cfg.min_trades_is * (train_end - bt.WARMUP) / max(1, is_end - bt.WARMUP)))
        ranked = sorted(pool, key=lambda x: evaluate(x, m, bt.WARMUP, train_end, folds[:k + 1], cfg, need).fitness,
                        reverse=True)
        tr = bt.run(ranked[0], d, m.costs, start=test_lo, end=test_hi)
        oos.append(tr.select(test_lo, test_hi))
    wf = summarize(np.concatenate(oos) if oos else np.zeros(0))
    eff = wf["mean"] / st["mean"] if st["mean"] > 0 else 0.0
    rep["walk_forward"] = _clean(wf) | {"efficiency": eff}
    v.stage_reached = "walk-forward"
    if wf["n"] < 10 or wf["mean"] <= 0 or eff < cfg.min_walkforward_efficiency:
        v.reasons.append(f"walk-forward failed: {wf['n']} trades, {wf['mean']:+.3f}R, efficiency {eff:.0%}")
        return v

    # e. sibling markets
    peer_means = []
    for p in peers[:3]:
        p_is, p_folds = split_points(p.data.n, cfg)
        pe = evaluate(g.for_symbol(p.data.symbol), p, bt.WARMUP, p_is, p_folds, cfg, min_trades=20)
        if pe.stats["n"] >= 20:
            peer_means.append((p.data.symbol, pe.stats["mean"], pe.stats["n"]))
    rep["peers"] = peer_means
    v.stage_reached = "peers"
    if peer_means and np.mean([x[1] for x in peer_means]) < cfg.min_peer_expectancy:
        v.reasons.append("does not generalise: loses on similar markets " +
                         ", ".join(f"{s} {mu:+.2f}R" for s, mu, _ in peer_means))
        return v

    # f. the holdout - every look is counted and makes the next look stricter
    looks = register_look()
    tr = bt.run(g, d, m.costs, start=is_end, end=d.n)
    ho = summarize(tr.select(is_end, d.n))
    alpha = cfg.alpha / max(1, looks)
    rep["holdout"] = _clean(ho) | {"alpha": alpha, "looks": looks}
    v.stage_reached = "holdout"
    fails = []
    if ho["n"] < cfg.min_trades_holdout:
        fails.append(f"{ho['n']} trades < {cfg.min_trades_holdout}")
    if ho["mean"] < cfg.min_holdout_expectancy:
        fails.append(f"expectancy {ho['mean']:+.3f}R")
    if ho["pf"] < cfg.min_holdout_pf:
        fails.append(f"profit factor {ho['pf']:.2f}")
    if ho["p"] > alpha:
        fails.append(f"p={ho['p']:.4f} > {alpha:.4f} (look #{looks})")
    if fails:
        v.reasons.append("holdout failed: " + ", ".join(fails))
        return v
    # What we expect live: walk-forward + holdout trades (both out-of-sample).
    expect = np.concatenate([np.concatenate(oos) if oos else np.zeros(0), tr.select(is_end, d.n)])
    rep["expected"] = _clean(summarize(expect))
    rep["expected_r"] = [round(float(x), 4) for x in expect[-400:]]
    rep["regimes"] = regime_table(g, m, tr)
    v.passed, v.stage_reached = True, "validated"
    return v


def regime_table(g: Genome, m: Market, tr: bt.Trades) -> dict:
    """Out-of-sample expectancy by volatility regime and by session, for the 'what works when' view."""
    d = m.data
    if not len(tr):
        return {}
    rank = d.atr_rank(500)
    out: dict = {"vol": {}, "session": {}}
    for i, r in zip(tr.entry_i, tr.r):
        k = i - 1
        vol = vol_bucket(rank[k])
        ses = session_of(int(d.hour[k]))
        for dim, key in (("vol", vol), ("session", ses)):
            cell = out[dim].setdefault(key, [0, 0.0])
            cell[0] += 1
            cell[1] += r
    return {dim: {k: {"n": n, "mean": s / n} for k, (n, s) in cells.items()} for dim, cells in out.items()}


def vol_bucket(rank: float) -> str:
    if not np.isfinite(rank):
        return "unknown"
    return "low_vol" if rank < 0.33 else ("high_vol" if rank > 0.67 else "normal_vol")


def session_of(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "ny_overlap" if hour < 16 else "new_york"
    if 17 <= hour < 21:
        return "new_york"
    return "late"


def _clean(st: dict) -> dict:
    return {k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
            for k, v in st.items()}
