from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "4H": 14400, "1D": 86400}
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)


@dataclass
class Instrument:
    symbol: str
    kind: str          # TradeLocker's own type field (e.g. FOREX, CFD, CRYPTO, EQUITY)
    description: str = ""
    asset_class: str = ""   # our classification: fx, index, metal, energy, crypto, stock, other


@dataclass
class Quote:
    bid: float
    ask: float
    time: pd.Timestamp   # when we received it (TradeLocker quotes carry no timestamp)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2


@dataclass
class Position:
    id: str
    symbol: str
    side: str
    qty: float
    entry: float
    has_stop: bool | None = None   # None = unknown


@dataclass
class InstrumentSpec:
    value_per_point: float   # account-currency P&L for qty 1 when price moves 1.0
    qty_step: float = 0.01
    min_qty: float = 0.01
    max_qty: float = 100.0


@dataclass
class ClosedInfo:
    exit_price: float
    exit_time: pd.Timestamp
    estimated: bool = False


class Broker:
    """What Gimenez needs from a broker. TradeLockerBroker talks to PlexyTrade; FakeBroker is for tests."""
    name = "base"
    is_live = False

    def instruments(self) -> list[Instrument]:
        raise NotImplementedError

    def history(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Bars in [start, end], columns BAR_COLUMNS, UTC index = bar open time. May include the forming bar."""
        raise NotImplementedError

    def max_history_rows(self) -> int:
        return 5000

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError

    def balance(self) -> float:
        raise NotImplementedError

    def open_positions(self) -> list[Position]:
        raise NotImplementedError

    def spec(self, symbol: str) -> InstrumentSpec:
        raise NotImplementedError

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float | None, tag: str) -> str:
        raise NotImplementedError

    def close_position(self, position_id: str) -> None:
        raise NotImplementedError

    def modify_stop(self, position_id: str, stop: float) -> bool:
        raise NotImplementedError

    def closed_info(self, position_id: str, symbol: str, side: str, stop: float, take_profit: float | None) -> ClosedInfo | None:
        raise NotImplementedError


def classify(symbol: str, kind: str = "", description: str = "") -> str:
    """Best-effort asset class from the name/type/description TradeLocker gives us."""
    s = symbol.upper().replace(".", "").replace("_", "").replace("/", "")
    k, d = (kind or "").upper(), (description or "").upper()
    fx = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR",
          "TRY", "PLN", "CZK", "HUF", "CNH", "ILS", "THB"}
    crypto = {"BTC", "ETH", "SOL", "LTC", "XRP", "ADA", "DOGE", "DOT", "BNB", "AVAX", "LINK", "BCH", "XLM", "TRX",
              "MATIC", "UNI", "SHIB", "ATOM", "NEAR", "TON", "PEPE", "ARB", "APT", "SUI"}
    if "CRYPTO" in k or s[:3] in crypto or s[:4] in crypto or s[:5] in {"MATIC", "DOGE"}:
        return "crypto"
    if s[:3] in {"XAU", "XAG", "XPT", "XPD"} or "GOLD" in s or "SILVER" in s:
        return "metal"
    if s[:3] in {"XTI", "XBR", "XNG", "WTI", "BRE", "USO", "UKO"} or "OIL" in s or "NGAS" in s or "NATGAS" in s:
        return "energy"
    if len(s) == 6 and s[:3] in fx and s[3:] in fx:
        return "fx"
    if "FOREX" in k or k == "FX":
        return "fx"
    idx = ("US30", "US500", "US100", "USTECH", "NAS", "SPX", "DJ", "DE30", "DE40", "GER", "UK100", "FTSE", "JP225",
           "NIK", "HK50", "AUS200", "EU50", "STOXX", "FRA40", "ESP35", "US2000", "RUSS", "VIX", "CHINA")
    if any(s.startswith(p) for p in idx) or "INDEX" in k or "INDEX" in d:
        return "index"
    if "EQUIT" in k or "STOCK" in k or "SHARE" in k or (s.isalpha() and len(s) <= 5):
        return "stock"
    return "other"
