"""Compliance (PlexyTrade terms) and risk. Every real order passes both, or it is not sent.

PlexyTrade allows EAs, scalping, hedging and news trading. It forbids arbitrage of any kind
(including latency/feed arbitrage), trading on misquotes or price errors, platform manipulation,
Negative Balance Protection abuse and "abusive strategies". Gimenez only ever uses PlexyTrade's own
feed, and this module blocks: abnormal spreads, stale or off-market quotes, latency-style micro
stops, order spam, opposite positions on one symbol (self-hedging), adding to positions
(martingale/grid), and any increase in risk after a loss.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..broker.base import TIMEFRAME_SECONDS, InstrumentSpec, Position, Quote
from ..config import ComplianceConfig, RiskConfig
from ..research.stats import bootstrap_mean_ci, kelly_fraction


class Compliance:
    def __init__(self, cfg: ComplianceConfig, clock=time.monotonic) -> None:
        self.cfg = cfg
        self.clock = clock
        self.orders: deque[float] = deque()
        self.last_symbol_order: dict[str, float] = {}

    def check(self, *, symbol: str, side: str, entry: float, stop: float, quote: Quote, last_close: float, atr: float,
              typical_spread: float | None, last_bar_time: pd.Timestamp, tf: str, now: pd.Timestamp,
              positions: list[Position]) -> str | None:
        c = self.cfg
        if atr <= 0 or not np.isfinite(atr):
            return "ATR unavailable"
        if quote.bid <= 0 or quote.ask <= 0 or quote.spread < 0:
            return "invalid quote (possible price error) - not trading on it"
        if typical_spread and quote.spread > c.max_spread_vs_median * typical_spread:
            return f"abnormal spread {quote.spread:.5g} (usual {typical_spread:.5g})"
        risk = abs(entry - stop)
        if risk <= 0:
            return "invalid stop"
        if quote.spread > c.max_spread_vs_stop * risk:
            return f"spread is {quote.spread / risk:.0%} of the risk - too expensive"
        if abs(quote.mid - last_close) > c.max_quote_jump_atr * atr:
            return "quote far from the last close - possible misquote/spike"
        age = (now - (last_bar_time + pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf]))).total_seconds()
        if age > c.max_bar_age_factor * TIMEFRAME_SECONDS[tf]:
            return f"stale data: last bar closed {age / 60:.0f} min ago (market closed or feed problem)"
        if risk < c.min_stop_atr * atr:
            return "stop too tight (latency-scalping pattern not allowed)"
        for p in positions:
            if p.symbol == symbol:
                if p.side != side:
                    return "opposite position open on this symbol (no self-hedging)"
                return "already in a position on this symbol (no adding / grid / martingale)"
        t = self.clock()
        while self.orders and t - self.orders[0] > 3600:
            self.orders.popleft()
        if sum(1 for x in self.orders if t - x <= 60) >= c.max_orders_per_minute:
            return "order rate limit (per minute)"
        if len(self.orders) >= c.max_orders_per_hour:
            return "order rate limit (per hour)"
        if t - self.last_symbol_order.get(symbol, -1e9) < c.min_seconds_between_symbol_orders:
            return "too soon after the last order on this symbol"
        return None

    def record_order(self, symbol: str) -> None:
        t = self.clock()
        self.orders.append(t)
        self.last_symbol_order[symbol] = t


@dataclass
class RiskDecision:
    ok: bool
    qty: float = 0.0
    risk_pct: float = 0.0
    risk_amount: float = 0.0
    reason: str = ""
    why_this_risk: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def strategy_risk(self, stage: str, live_r: list[float]) -> tuple[float, str]:
        """Risk per trade earned by the strategy's proven record."""
        c = self.cfg
        if stage == "probation":
            return c.risk_probation, f"probation (new to real money): {c.risk_probation:.2%}"
        if stage == "active":
            return c.risk_active, f"positive live record, not yet proven: {c.risk_active:.2%}"
        if stage == "proven":
            k = kelly_fraction(np.asarray(live_r)) * c.kelly_fraction
            lo, _ = bootstrap_mean_ci(np.asarray(live_r), 0.9)
            if lo <= 0:
                return c.risk_active, "proven stage but recent evidence weakened: back to active risk"
            r = min(c.max_risk_per_trade, max(c.risk_proven_floor, k))
            return r, f"proven: quarter-Kelly {k:.2%} -> {r:.2%} (cap {c.max_risk_per_trade:.0%})"
        return 0.0, "not a live stage"

    def account_scale(self, equity: float, peak: float) -> tuple[float, str]:
        c = self.cfg
        dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
        if dd <= c.drawdown_scale_start:
            return 1.0, ""
        f = 1 - (dd - c.drawdown_scale_start) / (c.drawdown_halt - c.drawdown_scale_start)
        f = max(0.2, f)
        return f, f"account drawdown {dd:.1%}: risk x{f:.2f}"

    def limits(self, equity: float, peak: float, day_start: float) -> str | None:
        c = self.cfg
        if peak > 0 and (peak - equity) / peak >= c.drawdown_halt:
            return f"drawdown {c.drawdown_halt:.0%} from peak reached - halted until you run `gimenez resume`"
        if day_start > 0 and (day_start - equity) / day_start >= c.daily_loss_stop:
            return f"daily loss stop {c.daily_loss_stop:.0%} reached - no trading until tomorrow"
        return None

    def size(self, *, equity: float, peak: float, stage: str, live_r: list[float], entry: float, stop: float,
             spec: InstrumentSpec, open_risk_pct: float, n_open: int, last_closed: dict | None) -> RiskDecision:
        c = self.cfg
        if n_open >= c.max_positions:
            return RiskDecision(False, reason=f"max {c.max_positions} open positions")
        base, why = self.strategy_risk(stage, live_r)
        if base <= 0:
            return RiskDecision(False, reason=why)
        scale, why2 = self.account_scale(equity, peak)
        risk = base * scale
        notes = [why] + ([why2] if why2 else [])
        # never increase size after a loss (anti-martingale / NBP abuse rule)
        if last_closed and (last_closed.get("r") or 0) < 0 and last_closed.get("risk_pct"):
            if risk > last_closed["risk_pct"]:
                risk = float(last_closed["risk_pct"])
                notes.append(f"capped at the last losing trade's {risk:.2%} (no size-up after a loss)")
        room = c.max_total_open_risk - open_risk_pct
        if room < c.min_risk:
            return RiskDecision(False, reason=f"total open risk {open_risk_pct:.1%} at the {c.max_total_open_risk:.0%} cap")
        if risk > room:
            risk = room
            notes.append(f"trimmed to the remaining open-risk budget {room:.2%}")
        risk = min(risk, c.max_risk_per_trade)
        if risk < c.min_risk:
            return RiskDecision(False, reason="risk budget too small")
        dist = abs(entry - stop)
        if dist <= 0 or equity <= 0 or spec.value_per_point <= 0:
            return RiskDecision(False, reason="invalid stop distance, equity or instrument spec")
        amount = equity * risk
        raw = amount / (dist * spec.value_per_point)
        qty = min(np.floor(raw / spec.qty_step + 1e-9) * spec.qty_step, spec.max_qty)
        qty = round(float(qty), 8)
        if qty < spec.min_qty:
            return RiskDecision(False, reason=f"position too small (needs {raw:.4f}, min {spec.min_qty})")
        actual = qty * dist * spec.value_per_point
        if actual / equity > c.max_risk_per_trade + 1e-9:
            return RiskDecision(False, reason="rounded size would exceed the 2% cap")
        return RiskDecision(True, qty, actual / equity, actual, why_this_risk="; ".join(notes))
