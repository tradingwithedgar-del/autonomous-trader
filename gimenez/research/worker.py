"""Background research process: picks a market/timeframe, evolves ideas, puts the best through the
validation gauntlet and hands survivors to shadow trading. Runs at the lowest CPU priority and only
reads price history from disk, so it can never slow down or rate-limit the live trader."""
from __future__ import annotations

import gc
import logging
import os
import time

import numpy as np
import pandas as pd

from ..config import Settings
from ..data.store import BarStore
from ..journal import STAGES_RUNNING, Journal
from . import backtest as bt
from .genome import Genome
from .indicators import Data
from .search import Market, evolve, split_points, validate

log = logging.getLogger(__name__)
MIN_BARS = 3000


def load_market(store: BarStore, j: Journal, symbol: str, tf: str, s: Settings) -> Market | None:
    df = store.load(symbol, tf, limit=s.research.max_bars_per_dataset)
    if len(df) < MIN_BARS:
        return None
    wl = j.query("SELECT asset_class FROM watchlist WHERE symbol=?", (symbol,))
    cls = wl[0]["asset_class"] if wl else "other"
    costs = bt.Costs.for_market(float(df["close"].iloc[-1]), cls, store.typical_spread(symbol), s.research)
    return Market(Data(df, symbol, tf), costs, cls)


def pick_dataset(store: BarStore, j: Journal, s: Settings) -> tuple[str, str] | None:
    chosen = {r["symbol"] for r in j.query("SELECT symbol FROM watchlist WHERE chosen=1")}
    cands = [(sym, tf) for sym, tf, n in store.datasets()
             if n >= MIN_BARS and tf in s.research.timeframes and (not chosen or sym in chosen)]
    if not cands:
        return None
    last = {k: j.get(f"research_last:{k[0]}:{k[1]}", "") for k in cands}
    return min(cands, key=lambda k: last[k])


def research_once(s: Settings, store: BarStore, j: Journal, rng: np.random.Generator, budget_s: float | None = None,
                  dataset: tuple[str, str] | None = None) -> dict | None:
    ds = dataset or pick_dataset(store, j, s)
    if not ds:
        return None
    symbol, tf = ds
    j.set(f"research_last:{symbol}:{tf}", j.clock())
    m = load_market(store, j, symbol, tf, s)
    if m is None:
        return None
    cfg = s.research
    hints = j.get("research_hints", {})
    seeds = [Genome.from_dict(x["genome"]) for x in j.strategies(STAGES_RUNNING) if x["tf"] == tf][:6]
    budget = budget_s if budget_s is not None else cfg.cpu_seconds_per_cycle
    t0 = time.monotonic()
    res = evolve(m, cfg, rng, deadline=t0 + budget * 0.7, hints=hints, seeds=seeds)
    n_trials, sum_inv = j.add_trials(symbol, tf, res.trials, float(np.sum(res.inv_n)) if res.inv_n else 0.0)
    var = sum_inv / n_trials if n_trials else 0.0
    is_end, _ = split_points(m.data.n, cfg)
    window = str(m.data.index[is_end])[:7]   # holdout window id (its start month)
    peers = []
    peer_syms = [r["symbol"] for r in j.query("SELECT symbol FROM watchlist WHERE asset_class=? AND symbol!=?",
                                               (m.asset_class, symbol))]
    for ps in peer_syms[:3]:
        pm = load_market(store, j, ps, tf, s)
        if pm:
            peers.append(pm)
    looks_used, passed, tried, reasons = 0, [], 0, []
    for e in res.best:
        if looks_used >= cfg.max_holdout_candidates_per_run or time.monotonic() - t0 > budget:
            break
        if e.fitness <= 0:
            break
        tried += 1

        def look() -> int:
            nonlocal looks_used
            looks_used += 1
            return j.register_look(symbol, tf, window)
        v = validate(e, m, cfg, rng, n_trials, var, peers, look)
        reasons.append({"id": e.genome.id, "name": e.genome.short_name(), "stage": v.stage_reached,
                        "why": v.reasons[:1], "fitness": round(e.fitness, 2)})
        if v.passed:
            passed.append(v)
    for v in passed:
        g = v.genome
        n_shadow = len(j.strategies(("shadow",)))
        stage = "shadow" if n_shadow < s.pipeline.max_shadow else "validated"
        j.add_strategy(g.id, g.short_name(), g.symbol, g.tf, g.family, g.describe(), g.to_dict(), stage, v.report)
        ho = v.report["holdout"]
        j.event("discovery", f"New strategy passed every test: {g.short_name()} - holdout {ho['n']} trades, "
                f"{ho['mean']:+.2f}R/trade, PF {ho['pf']:.2f}, p={ho['p']:.4f}. Now in {stage}.",
                {"strategy": g.id})
    summary = {"top": reasons, "costs": {"spread": m.costs.spread}, "trials_total": n_trials}
    j.insert("research_runs", {"ts": j.clock(), "symbol": symbol, "tf": tf, "bars": m.data.n, "trials": res.trials,
                               "candidates": tried, "holdout_looks": looks_used, "passed": len(passed),
                               "seconds": round(time.monotonic() - t0, 1), "summary": summary})
    log.info("research %s %s: %d ideas, %d candidates, %d holdout looks, %d passed", symbol, tf, res.trials, tried,
             looks_used, len(passed))
    del m, peers, res
    gc.collect()
    return summary


def main_loop(s: Settings) -> None:  # pragma: no cover - long-running service
    try:
        os.nice(19)
    except OSError:
        pass
    store, j = BarStore(s.bars_path), Journal(s.db_path)
    rng = np.random.default_rng(int(time.time()))
    j.event("research", "research worker started")
    while True:
        try:
            if research_once(s, store, j, rng) is None:
                log.info("research: not enough history yet; waiting for the trader to download it")
        except Exception as e:
            log.exception("research cycle failed")
            j.event("error", f"research cycle failed: {e}")
        time.sleep(s.research.sleep_between_cycles)


def history_report(store: BarStore) -> pd.DataFrame:
    rows = []
    for sym, tf, n in store.datasets():
        first, last, _ = store.span(sym, tf)
        rows.append({"symbol": sym, "tf": tf, "bars": n, "from": pd.Timestamp(first, unit="s"), "to": pd.Timestamp(last, unit="s")})
    return pd.DataFrame(rows)
