"""Price history on disk (SQLite, WAL mode).

Only the trader process talks to TradeLocker. It keeps this store up to date and slowly
backfills older history within the rate limits; the research worker only reads from here.
That keeps one broker session, one rate limiter, and research can never slow trading down.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..broker.base import BAR_COLUMNS, TIMEFRAME_SECONDS, Broker, empty_bars

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT, tf TEXT, t INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, tf, t)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS spreads (symbol TEXT, t INTEGER, spread REAL, mid REAL);
CREATE INDEX IF NOT EXISTS spreads_sym ON spreads(symbol, t);
CREATE TABLE IF NOT EXISTS backfill (symbol TEXT, tf TEXT, done INTEGER, oldest INTEGER, PRIMARY KEY (symbol, tf));
"""

# How far back we try to backfill per timeframe (TradeLocker decides what it actually serves).
BACKFILL_DAYS = {"1m": 45, "5m": 240, "15m": 540, "30m": 720, "1H": 1100, "4H": 2000, "1D": 4000}


class BarStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def upsert(self, symbol: str, tf: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        t = (df.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
        rows = list(zip([symbol] * len(df), [tf] * len(df), np.asarray(t, dtype="int64").tolist(),
                        *(df[c].astype(float).tolist() for c in BAR_COLUMNS)))
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()
        return len(rows)

    def load(self, symbol: str, tf: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
             limit: int | None = None) -> pd.DataFrame:
        sql, params = "SELECT t, open, high, low, close, volume FROM bars WHERE symbol=? AND tf=?", [symbol, tf]
        if start is not None:
            sql += " AND t >= ?"
            params.append(int(start.timestamp()))
        if end is not None:
            sql += " AND t <= ?"
            params.append(int(end.timestamp()))
        if limit:
            sql = f"SELECT * FROM ({sql} ORDER BY t DESC LIMIT ?) ORDER BY t"
            params.append(int(limit))
        else:
            sql += " ORDER BY t"
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return empty_bars()
        arr = np.array(rows, dtype=float)
        idx = pd.to_datetime(arr[:, 0].astype("int64"), unit="s", utc=True)
        return pd.DataFrame(arr[:, 1:], index=idx, columns=BAR_COLUMNS)

    def span(self, symbol: str, tf: str) -> tuple[int, int, int]:
        with self.lock:
            r = self.conn.execute("SELECT MIN(t), MAX(t), COUNT(*) FROM bars WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()
        return (r[0] or 0, r[1] or 0, r[2] or 0)

    def datasets(self) -> list[tuple[str, str, int]]:
        with self.lock:
            return [tuple(r) for r in self.conn.execute("SELECT symbol, tf, COUNT(*) FROM bars GROUP BY symbol, tf")]

    def record_spread(self, symbol: str, spread: float, mid: float, t: int | None = None) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO spreads VALUES (?,?,?,?)", (symbol, int(t or time.time()), spread, mid))
            self.conn.commit()

    def typical_spread(self, symbol: str, days: int = 30) -> float | None:
        since = int(time.time()) - days * 86400
        with self.lock:
            vals = [r[0] for r in self.conn.execute("SELECT spread FROM spreads WHERE symbol=? AND t>=? AND spread>0",
                                                    (symbol, since))]
        return float(np.median(vals)) if vals else None

    def prune(self, keep_days: dict[str, int] | None = None) -> None:
        """Keep the database small: drop very old fast-timeframe bars and old spread samples."""
        keep = keep_days or BACKFILL_DAYS
        now = int(time.time())
        with self.lock:
            for tf, days in keep.items():
                self.conn.execute("DELETE FROM bars WHERE tf=? AND t < ?", (tf, now - int(days * 1.2) * 86400))
            self.conn.execute("DELETE FROM spreads WHERE t < ?", (now - 120 * 86400,))
            self.conn.commit()

    # --- downloading -------------------------------------------------------------------
    def update_recent(self, broker: Broker, symbol: str, tf: str, min_bars: int = 300) -> pd.DataFrame:
        """Fetch new closed bars since the last stored one; returns the latest `min_bars` closed bars."""
        secs = TIMEFRAME_SECONDS[tf]
        now = pd.Timestamp.now(tz="UTC") if not hasattr(broker, "now") else broker.now
        _, last, count = self.span(symbol, tf)
        end_ms = int(now.timestamp() * 1000)
        rows = max(10, broker.max_history_rows() - 10)
        if count < min_bars or not last:
            start_ms = end_ms - int(min(rows, min_bars * 3) * secs * 1000)
        else:
            start_ms = max((last - secs) * 1000, end_ms - rows * secs * 1000)
        df = broker.history(symbol, tf, start_ms, end_ms)
        closed = df[df.index + pd.Timedelta(seconds=secs) <= now]
        self.upsert(symbol, tf, closed)
        return self.load(symbol, tf, limit=min_bars)

    def backfill_step(self, broker: Broker, symbol: str, tf: str) -> bool:
        """Download one older chunk. Returns False when there is nothing more to get."""
        with self.lock:
            r = self.conn.execute("SELECT done, oldest FROM backfill WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()
        if r and r[0]:
            return False
        secs = TIMEFRAME_SECONDS[tf]
        first, _, count = self.span(symbol, tf)
        now_s = int((broker.now if hasattr(broker, "now") else pd.Timestamp.now(tz="UTC")).timestamp())
        oldest_wanted = now_s - BACKFILL_DAYS.get(tf, 365) * 86400
        end_s = (r[1] if r and r[1] else (first if count else now_s))
        if end_s <= oldest_wanted:
            self._mark(symbol, tf, True, end_s)
            return False
        span = int(max(10, broker.max_history_rows() - 10) * secs * 0.95)
        start_s = max(oldest_wanted, end_s - span)
        df = broker.history(symbol, tf, start_s * 1000, end_s * 1000)
        self.upsert(symbol, tf, df)
        # An empty chunk may just be a weekend/holiday; give up only after a long empty stretch.
        empty_runs = 0 if len(df) else (self._get_empty(symbol, tf) + 1)
        self._set_empty(symbol, tf, empty_runs)
        done = start_s <= oldest_wanted or empty_runs >= 4
        self._mark(symbol, tf, done, start_s)
        return not done

    def _mark(self, symbol: str, tf: str, done: bool, oldest: int) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO backfill VALUES (?,?,?,?) ON CONFLICT(symbol, tf) DO UPDATE SET "
                              "done=excluded.done, oldest=excluded.oldest", (symbol, tf, int(done), int(oldest)))
            self.conn.commit()

    def _get_empty(self, symbol: str, tf: str) -> int:
        return getattr(self, "_empty", {}).get((symbol, tf), 0)

    def _set_empty(self, symbol: str, tf: str, n: int) -> None:
        if not hasattr(self, "_empty"):
            self._empty = {}
        self._empty[(symbol, tf)] = n

    def backfill_status(self) -> dict[tuple[str, str], bool]:
        with self.lock:
            return {(r[0], r[1]): bool(r[2]) for r in self.conn.execute("SELECT symbol, tf, done FROM backfill")}
