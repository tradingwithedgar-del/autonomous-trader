"""In-memory broker for tests and dry runs: replays bars, fills at the replayed price +/- spread,
triggers stops/targets bar by bar."""
from __future__ import annotations

import itertools

import pandas as pd

from .base import TIMEFRAME_SECONDS, Broker, ClosedInfo, Instrument, InstrumentSpec, Position, Quote, classify, empty_bars


class FakeBroker(Broker):
    name = "fake"

    def __init__(self, data: dict[str, pd.DataFrame], timeframe: str = "15m", spreads: dict[str, float] | None = None,
                 equity: float = 10_000.0, value_per_point: float = 1.0) -> None:
        self.data = data   # symbol -> bars at `timeframe`
        self.tf = timeframe
        self.spreads = spreads or {}
        self.cash = equity
        self.vpp = value_per_point
        self.positions: dict[str, dict] = {}
        self.closed: dict[str, ClosedInfo] = {}
        self.orders: list[dict] = []
        self._ids = itertools.count(1)
        self.cursor = 0   # index (in every series) of the bar that is forming now
        self.drop_stops = False   # simulate a broker that "forgets" the stop

    # time control ------------------------------------------------------------------
    @property
    def now(self) -> pd.Timestamp:
        any_df = next(iter(self.data.values()))
        i = min(self.cursor, len(any_df) - 1)
        return any_df.index[i] + pd.Timedelta(seconds=1)

    def advance(self, bars: int = 1) -> None:
        for _ in range(bars):
            self.cursor += 1
            self._trigger()

    def _bar(self, symbol: str, i: int) -> pd.Series:
        return self.data[symbol].iloc[i]

    def _trigger(self) -> None:
        i = self.cursor - 1   # bar that just closed
        for pid, p in list(self.positions.items()):
            df = self.data[p["symbol"]]
            if i >= len(df):
                continue
            b = df.iloc[i]
            half = self.spreads.get(p["symbol"], 0.0) / 2
            if p["side"] == "buy":
                if p["stop"] is not None and b.low - half <= p["stop"]:
                    self._close(pid, min(p["stop"], b.open - half), df.index[i])
                elif p["tp"] is not None and b.high - half >= p["tp"]:
                    self._close(pid, p["tp"], df.index[i])
            else:
                if p["stop"] is not None and b.high + half >= p["stop"]:
                    self._close(pid, max(p["stop"], b.open + half), df.index[i])
                elif p["tp"] is not None and b.low + half <= p["tp"]:
                    self._close(pid, p["tp"], df.index[i])

    def _close(self, pid: str, price: float, t: pd.Timestamp) -> None:
        p = self.positions.pop(pid)
        sign = 1 if p["side"] == "buy" else -1
        self.cash += sign * (price - p["entry"]) * p["qty"] * self.vpp
        self.closed[pid] = ClosedInfo(price, t)

    # Broker API ----------------------------------------------------------------------
    def instruments(self) -> list[Instrument]:
        return [Instrument(s, "CFD", s, classify(s)) for s in self.data]

    def history(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        df = self.data.get(symbol)
        if df is None:
            return empty_bars()
        if timeframe != self.tf:
            rule = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            df = df.iloc[: self.cursor + 1].resample(rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        else:
            df = df.iloc[: self.cursor + 1]
        s, e = pd.Timestamp(start_ms, unit="ms", tz="UTC"), pd.Timestamp(end_ms, unit="ms", tz="UTC")
        return df[(df.index >= s) & (df.index <= e)]

    def get_quote(self, symbol: str) -> Quote:
        b = self._bar(symbol, min(self.cursor, len(self.data[symbol]) - 1))
        half = self.spreads.get(symbol, 0.0) / 2
        return Quote(b.open - half, b.open + half, self.now)

    def equity(self) -> float:
        eq = self.cash
        for p in self.positions.values():
            q = self.get_quote(p["symbol"])
            px = q.bid if p["side"] == "buy" else q.ask
            eq += (1 if p["side"] == "buy" else -1) * (px - p["entry"]) * p["qty"] * self.vpp
        return eq

    def balance(self) -> float:
        return self.cash

    def open_positions(self) -> list[Position]:
        return [Position(pid, p["symbol"], p["side"], p["qty"], p["entry"], p["stop"] is not None)
                for pid, p in self.positions.items()]

    def spec(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(self.vpp, 0.01, 0.01, 1e6)

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float | None, tag: str) -> str:
        q = self.get_quote(symbol)
        pid = str(next(self._ids))
        self.positions[pid] = {"symbol": symbol, "side": side, "qty": qty, "entry": q.ask if side == "buy" else q.bid,
                               "stop": None if self.drop_stops else stop, "tp": take_profit, "tag": tag}
        self.orders.append({"symbol": symbol, "side": side, "qty": qty, "stop": stop, "tp": take_profit, "tag": tag})
        return pid

    def close_position(self, position_id: str) -> None:
        p = self.positions[position_id]
        q = self.get_quote(p["symbol"])
        self._close(position_id, q.bid if p["side"] == "buy" else q.ask, self.now)

    def modify_stop(self, position_id: str, stop: float) -> bool:
        if position_id in self.positions:
            self.positions[position_id]["stop"] = stop
            return True
        return False

    def fill_price(self, position_id: str) -> float | None:
        p = self.positions.get(position_id)
        return p["entry"] if p else None

    def closed_info(self, position_id: str, symbol: str, side: str, stop: float, take_profit: float | None) -> ClosedInfo | None:
        return self.closed.get(position_id)
