"""PlexyTrade on TradeLocker via the official `tradelocker` package.

Adapted from TIIM's connector (github.com/tradingwithedgar-del/tradingbot-dev1,
tradingbot/broker/tradelocker_broker.py): login, symbol<->instrument id mapping, chunked
history, position sizing from lotSize and the quote->account currency rate, closed-trade
lookup in order history. Added here: rate limiting on every call, a bounded history cache,
and a check that every position really has a stop loss at the broker.

Sizing depends on value_per_point (account-currency P&L for qty 1 moving 1.0 in price).
Verify the first demo trades: the risk TradeLocker shows should match the journal's risk_amount.
Overrides go in data/instruments.json, e.g. {"XAUUSD": {"value_per_point": 100, "qty_step": 0.01}}.
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from ..config import Settings
from .base import BAR_COLUMNS, Broker, ClosedInfo, Instrument, InstrumentSpec, Position, Quote, classify, empty_bars
from .ratelimit import TokenBucket

log = logging.getLogger(__name__)
FX = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "SGD", "HKD", "MXN", "ZAR", "PLN", "TRY"}


class StopLossMissing(RuntimeError):
    pass


class TradeLockerBroker(Broker):
    name = "tradelocker"

    def __init__(self, settings: Settings) -> None:
        settings.validate()  # demo-only unless the owner explicitly enabled live
        from tradelocker import TLAPI  # lazy: backtests and tests never need credentials

        if not (settings.tl_email and settings.tl_password and settings.tl_server):
            raise RuntimeError("TL_EMAIL, TL_PASSWORD and TL_SERVER must be set in .env")
        self.settings = settings
        self.is_live = settings.is_live
        self.api = TLAPI(
            environment=settings.tl_environment,
            username=settings.tl_email,
            password=settings.tl_password,
            server=settings.tl_server,
            acc_num=settings.tl_acc_num if 0 < settings.tl_acc_num < 1000 else 0,
            account_id=settings.tl_acc_num if settings.tl_acc_num >= 1000 else 0,
            log_level="warning",
        )
        self.calls = TokenBucket(settings.requests_per_second, burst=3)
        self.hist_calls = TokenBucket(settings.requests_per_second / 2, burst=1)
        self._apply_published_limits()
        self._ids: dict[str, int] = {}
        self._specs: dict[str, InstrumentSpec] = {}
        path = settings.data_dir / "instruments.json"
        self._overrides = json.loads(path.read_text()) if path.exists() else {}
        self._account_ccy = self._detect_account_currency()

    # --- plumbing ------------------------------------------------------------------
    def _apply_published_limits(self) -> None:
        """Use at most half of what TradeLocker says each route allows."""
        for route, bucket in (("QUOTES_HISTORY", self.hist_calls), ("QUOTES", self.calls)):
            try:
                lim = self.api.get_route_rate_limit(route)
                per = float(lim["intervalNum"]) * (60 if lim["measure"] == "MINUTES" else 1)
                published = float(lim["limit"]) / per
                bucket.set_rate(min(bucket.rate, published / 2))
                log.info("TradeLocker %s limit %s/%ss -> using %.2f req/s", route, lim["limit"], per, bucket.rate)
            except Exception:  # pragma: no cover - depends on the server config
                log.info("no published rate limit for %s; using %.2f req/s", route, bucket.rate)

    def _call(self, fn, *args, **kwargs):
        self.calls.acquire()
        return fn(*args, **kwargs)

    def _iid(self, symbol: str) -> int:
        if symbol not in self._ids:
            self._ids[symbol] = int(self._call(self.api.get_instrument_id_from_symbol_name, symbol))
        return self._ids[symbol]

    def _symbol(self, iid: int) -> str:
        for s, i in self._ids.items():
            if i == iid:
                return s
        name = str(self._call(self.api.get_symbol_name_from_instrument_id, iid))
        self._ids[name] = iid
        return name

    def _detect_account_currency(self) -> str:
        try:
            accounts = self._call(self.api.get_trade_accounts)
            wanted = int(getattr(self.api, "acc_num", 0) or 0)
            match = [a for a in accounts if int(a.get("accNum", -1)) == wanted] or accounts[:1]
            if match:
                return str(match[0].get("currency", "USD"))
        except Exception:  # pragma: no cover - network
            log.exception("could not detect account currency, assuming USD")
        return "USD"

    def _quote_ccy(self, symbol: str) -> str:
        ov = self._overrides.get(symbol, {})
        if "quote_currency" in ov:
            return str(ov["quote_currency"]).upper()
        s = symbol.upper().replace(".", "")
        if len(s) >= 6 and s[3:6] in FX and (s[:3] in FX or s[:3] in {"XAU", "XAG", "XPT", "XPD"}):
            return s[3:6]
        return "USD"   # US indices/stocks/oil and USD-quoted crypto; others: set quote_currency in instruments.json

    def _to_account_rate(self, symbol: str) -> float:
        q, a = self._quote_ccy(symbol), self._account_ccy.upper()
        if q == a:
            return 1.0
        for pair, invert in ((q + a, False), (a + q, True)):
            try:
                mid = self.get_quote(pair).mid
                return 1.0 / mid if invert else mid
            except Exception:
                continue
        raise RuntimeError(f"cannot convert {q} to {a} for {symbol}; add it to data/instruments.json")

    # --- market data -----------------------------------------------------------------
    def instruments(self) -> list[Instrument]:
        df = self._call(self.api.get_all_instruments)
        out = []
        for _, r in df.iterrows():
            name = str(r.get("name"))
            iid = r.get("tradableInstrumentId")
            if iid is not None and not pd.isna(iid):
                self._ids.setdefault(name, int(iid))
            kind, desc = str(r.get("type") or ""), str(r.get("description") or "")
            out.append(Instrument(name, kind, desc, classify(name, kind, desc)))
        return out

    def max_history_rows(self) -> int:
        try:
            return int(self._call(self.api.max_price_history_rows))
        except Exception:
            return 5000

    def history(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        self.hist_calls.acquire()
        iid = self._iid(symbol)
        raw = self.api.get_price_history(iid, resolution=timeframe, start_timestamp=int(start_ms), end_timestamp=int(end_ms))
        # The package keeps every history response in an in-memory LRU cache (up to 128 large
        # responses). On a 1 GB server that adds up, so we clear it after each call.
        cached = getattr(self.api, "__cached__request_history_cacheable", None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
        if raw is None or raw.empty:
            return empty_bars()
        df = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        if "volume" not in df:
            df["volume"] = 0.0
        return df[BAR_COLUMNS].astype(float).sort_index()

    def get_quote(self, symbol: str) -> Quote:
        q = self._call(self.api.get_quotes, self._iid(symbol))
        return Quote(float(q["bp"]), float(q["ap"]), pd.Timestamp.now(tz="UTC"))

    # --- account -----------------------------------------------------------------------
    def _state(self) -> dict:
        return self._call(self.api.get_account_state)

    def balance(self) -> float:
        return float(self._state().get("balance", 0.0))

    def equity(self) -> float:
        st = self._state()
        return float(st.get("projectedBalance", st.get("balance", 0.0)))

    def open_positions(self) -> list[Position]:
        df = self._call(self.api.get_all_positions)
        out = []
        for _, r in df.iterrows():
            sl = r.get("stopLossId") if "stopLossId" in df.columns else None
            has_stop = None if sl is None else bool(not pd.isna(sl) and int(sl) != 0)
            out.append(Position(str(int(r["id"])), self._symbol(int(r["tradableInstrumentId"])), str(r["side"]).lower(),
                                float(r["qty"]), float(r["avgPrice"]), has_stop))
        return out

    def spec(self, symbol: str) -> InstrumentSpec:
        ov = self._overrides.get(symbol, {})
        fields = {"value_per_point", "qty_step", "min_qty", "max_qty"}
        if "value_per_point" in ov:
            return InstrumentSpec(**{k: v for k, v in ov.items() if k in fields})
        if symbol not in self._specs:
            d = self._call(self.api.get_instrument_details, self._iid(symbol))
            lot = float(d.get("lotSize") or d.get("contractSize") or 100_000)
            step = float(d.get("lotStep") or 0.01)
            self._specs[symbol] = InstrumentSpec(lot, step, float(d.get("minLot") or d.get("minOrderSize") or step),
                                                 float(d.get("maxLot") or d.get("maxOrderSize") or 100))
        s = self._specs[symbol]
        return InstrumentSpec(s.value_per_point * self._to_account_rate(symbol), s.qty_step, s.min_qty, s.max_qty)

    # --- orders --------------------------------------------------------------------------
    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float | None, tag: str) -> str:
        kwargs = dict(quantity=qty, side=side, type_="market", stop_loss=float(stop), stop_loss_type="absolute",
                      strategy_id=tag[:31])
        if take_profit is not None:
            kwargs.update(take_profit=float(take_profit), take_profit_type="absolute")
        order_id = self._call(self.api.create_order, self._iid(symbol), **kwargs)
        if not order_id:
            raise RuntimeError(f"order rejected: {symbol} {side} {qty}")
        pid = None
        for _ in range(10):
            pid = self._call(self.api.get_position_id_from_order_id, int(order_id))
            if pid:
                break
            time.sleep(0.5)
        if not pid:
            raise RuntimeError(f"order {order_id} placed but no position id found; check TradeLocker")
        self._ensure_stop(str(pid), float(stop))
        return str(pid)

    def _ensure_stop(self, position_id: str, stop: float) -> None:
        """Hard rule: every trade has a stop loss at the broker. Re-attach it, or close the trade."""
        pos = next((p for p in self.open_positions() if p.id == position_id), None)
        if pos is None or pos.has_stop is not False:
            return
        log.warning("position %s has no stop at the broker - attaching it", position_id)
        if self.modify_stop(position_id, stop):
            return
        self.close_position(position_id)
        raise StopLossMissing(f"position {position_id} had no stop and one could not be set - closed it")

    def close_position(self, position_id: str) -> None:
        self._call(self.api.close_position, position_id=int(position_id))

    def modify_stop(self, position_id: str, stop: float) -> bool:
        return bool(self._call(self.api.modify_position, int(position_id), {"stopLoss": float(stop)}))

    def closed_info(self, position_id: str, symbol: str, side: str, stop: float, take_profit: float | None) -> ClosedInfo | None:
        try:
            hist = self._call(self.api.get_all_orders, history=True, lookback_period="7D")
            m = hist[(hist["positionId"].astype("Int64") == int(position_id)) & (hist["status"] == "Filled")]
            m = m[m["side"].str.lower() != side]
            if len(m):
                row = m.iloc[-1]
                t = pd.to_datetime(int(row.get("lastModified", 0) or 0), unit="ms", utc=True)
                return ClosedInfo(float(row["avgPrice"]), t)
        except Exception:
            log.exception("could not read order history for position %s", position_id)
        q = self.get_quote(symbol)
        px = q.bid if side == "buy" else q.ask
        guess = stop if take_profit is None or abs(px - stop) < abs(px - take_profit) else take_profit
        return ClosedInfo(guess, pd.Timestamp.now(tz="UTC"), estimated=True)
