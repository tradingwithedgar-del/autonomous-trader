"""Offline rehearsal: the whole pipeline (research -> shadow -> promotion -> real trades) on synthetic
markets with the fake broker. Nothing touches TradeLocker. Used by tests and `gimenez simulate`
to preview the dashboard. Synthetic results say NOTHING about real markets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .broker.fake import FakeBroker
from .config import Settings
from .data.store import BarStore
from .data.synthetic import planted_breakout, random_walk
from .journal import Journal
from .live import engine as eng_mod
from .live import learning
from .live.engine import Engine
from .research.worker import research_once


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 30.0
        return self.t


def simulate(data_dir: Path | str, bars: int = 9000, live_bars: int = 2500, seed: int = 1, budget_s: float = 20,
             log=print) -> Settings:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    s = Settings()
    s.data_dir = data_dir
    s.mode = "demo"
    s.tl_environment = "https://demo.tradelocker.com"
    s.pipeline.shadow_min_trades = 12
    s.pipeline.probation_trades = 15
    total = bars + live_bars
    markets = {
        "SYNA": planted_breakout(n=total, seed=seed, price0=2000.0),
        "SYNB": planted_breakout(n=total, seed=seed + 1, price0=150.0, edge=0.2),
        "NOISE": random_walk(n=total, seed=seed + 2, price0=50.0),
    }
    store = BarStore(s.bars_path)
    j = Journal(s.db_path, clock=lambda: fb.now.isoformat())
    fb = FakeBroker(markets, "15m", spreads={k: float(v["close"].iloc[0]) * 0.0001 for k, v in markets.items()},
                    equity=10_000.0)
    fb.cursor = bars
    for sym, df in markets.items():
        store.upsert(sym, "15m", df.iloc[:bars])
        store.record_spread(sym, fb.spreads[sym], float(df["close"].iloc[bars - 1]))
        j.insert("watchlist", {"symbol": sym, "asset_class": "index", "score": 0.5, "spread": fb.spreads[sym], "atr": None,
                               "cost_ratio": 0.05, "hours_open": 24, "chosen": 1, "reason": "synthetic", "ts": j.clock()})
    j.set("last_screen", fb.now.isoformat())
    s.research.timeframes = ["15m"]
    rng = np.random.default_rng(seed)
    for sym in markets:
        log(f"research on {sym} ...")
        research_once(s, store, j, rng, budget_s=budget_s, dataset=(sym, "15m"))
    log(f"strategies after research: {[(x['name'], x['stage']) for x in j.strategies()]}")
    eng_mod.LOOKBACK = 1500
    e = Engine(s, fb, j, store, wall=_Clock())
    for k in range(live_bars - 1):
        e.step()
        fb.advance()
        if k % 50 == 0:
            for msg in learning.review(j, s):
                log(msg)
    log(f"done: {len(j.closed_trades(shadow=False))} real and {len(j.closed_trades(shadow=True))} shadow trades closed")
    return s


if __name__ == "__main__":  # pragma: no cover
    import sys

    simulate(sys.argv[1] if len(sys.argv) > 1 else "data/sim")
    print(pd.Timestamp.now())
