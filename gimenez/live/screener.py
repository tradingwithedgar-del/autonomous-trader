"""Market selection. Reads every instrument on the account, measures what trading it would cost
(spread vs typical hourly range), how many hours a day it trades and how active it is, then picks
a diverse watchlist. It runs a few instruments at a time so it never crowds out live trading."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..broker.base import Broker
from ..config import Settings
from ..data.store import BarStore
from ..journal import Journal
from ..research.indicators import atr

log = logging.getLogger(__name__)
CLASS_ORDER = ["index", "metal", "fx", "energy", "crypto", "stock", "other"]


def measure(broker: Broker, store: BarStore, symbol: str, days: int = 15) -> dict:
    q = broker.get_quote(symbol)
    store.record_spread(symbol, q.spread, q.mid)
    now = getattr(broker, "now", None) or pd.Timestamp.now(tz="UTC")
    end = int(now.timestamp() * 1000)
    df = broker.history(symbol, "1H", end - days * 86_400_000, end)
    if len(df) < 50 or q.bid <= 0 or q.spread < 0:
        return {"ok": False, "reason": "not enough data or no valid quote"}
    a = atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    med_atr = float(np.nanmedian(a[14:]))
    hours_open = len(df) / max(1.0, (df.index[-1] - df.index[0]).total_seconds() / 86400.0)
    vol = float(df["volume"].median()) if "volume" in df else 0.0
    spread = store.typical_spread(symbol, days=7) or q.spread
    return {"ok": med_atr > 0, "spread": spread, "atr": med_atr, "cost_ratio": spread / med_atr if med_atr > 0 else 9.9,
            "hours_open": min(24.0, hours_open * 1.0), "volume": vol, "price": q.mid}


def screen_step(broker: Broker, store: BarStore, j: Journal, s: Settings, per_step: int = 2) -> bool:
    """Measure the next few instruments. Returns True once a full pass finished and the watchlist was chosen."""
    queue = j.get("screen_queue")
    if not queue:
        inst = broker.instruments()
        queue = [[i.symbol, i.asset_class] for i in sorted(inst, key=lambda i: CLASS_ORDER.index(i.asset_class)
                                                           if i.asset_class in CLASS_ORDER else 99)]
        j.set("screen_results", {})
        j.event("screen", f"screening all {len(queue)} instruments on the account")
    results = j.get("screen_results", {})
    for _ in range(per_step):
        if not queue:
            break
        sym, cls = queue.pop(0)
        try:
            m = measure(broker, store, sym)
        except Exception as e:
            m = {"ok": False, "reason": f"error: {e}"[:200]}
        m["asset_class"] = cls
        results[sym] = m
    j.set("screen_results", results)
    j.set("screen_queue", queue)
    if queue:
        return False
    choose_watchlist(results, j, s)
    j.set("screen_queue", None)
    j.set("last_screen", j.clock())
    return True


def score(m: dict, s: Settings) -> tuple[float, str]:
    if not m.get("ok"):
        return -1.0, m.get("reason", "no data")
    if m["cost_ratio"] > s.screen.max_cost_ratio:
        return -1.0, f"too expensive: spread is {m['cost_ratio']:.0%} of an hourly range"
    if m["hours_open"] < s.screen.min_hours_open_per_day:
        return -1.0, f"trades only ~{m['hours_open']:.0f}h/day"
    cost = 1 - m["cost_ratio"] / s.screen.max_cost_ratio          # cheaper is better
    hours = min(1.0, m["hours_open"] / 20)
    return round(0.65 * cost + 0.35 * hours, 4), "ok"


def choose_watchlist(results: dict, j: Journal, s: Settings) -> list[str]:
    scored = []
    for sym, m in results.items():
        sc, why = score(m, s)
        scored.append((sym, m, sc, why))
    # liquidity tie-break inside each class: tick volume rank
    by_cls: dict[str, list] = {}
    for row in scored:
        by_cls.setdefault(row[1].get("asset_class", "other"), []).append(row)
    for rows in by_cls.values():
        vols = np.array([r[1].get("volume", 0.0) or 0.0 for r in rows])
        ranks = vols.argsort().argsort() / max(1, len(rows) - 1) if len(rows) > 1 else np.ones(len(rows))
        for k, r in enumerate(rows):
            if r[2] > 0:
                rows[k] = (r[0], r[1], round(r[2] + 0.1 * float(ranks[k]), 4), r[3])
    ranked = sorted((r for rows in by_cls.values() for r in rows), key=lambda r: r[2], reverse=True)
    chosen, per_cls = [], {}
    for sym, m, sc, why in ranked:
        cls = m.get("asset_class", "other")
        if sc > 0 and len(chosen) < s.screen.watchlist_size and per_cls.get(cls, 0) < s.screen.max_per_class:
            chosen.append(sym)
            per_cls[cls] = per_cls.get(cls, 0) + 1
    # keep markets that live strategies still trade
    running = {r["symbol"] for r in j.query("SELECT DISTINCT symbol FROM strategies WHERE stage IN "
                                             "('shadow','probation','active','proven')")}
    j.execute("DELETE FROM watchlist")
    for sym, m, sc, why in ranked:
        reason = why if sym in chosen or sc <= 0 else "good, but the watchlist is full / class quota reached"
        j.insert("watchlist", {"symbol": sym, "asset_class": m.get("asset_class"), "score": sc,
                               "spread": m.get("spread"), "atr": m.get("atr"), "cost_ratio": m.get("cost_ratio"),
                               "hours_open": m.get("hours_open"), "chosen": int(sym in chosen or sym in running),
                               "reason": reason, "ts": j.clock()})
    j.event("screen", f"watchlist chosen: {', '.join(chosen) or 'nothing passed'}", {"chosen": chosen})
    return chosen
