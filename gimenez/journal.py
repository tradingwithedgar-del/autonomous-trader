"""Gimenez's memory (SQLite): strategies and their stage, every decision with a market snapshot,
every trade (real and shadow) with its post-mortem, equity, events, research bookkeeping.
The dashboard and `why` only read from here."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY, name TEXT, symbol TEXT, tf TEXT, family TEXT, description TEXT, genome TEXT,
    stage TEXT, created_at TEXT, stage_changed_at TEXT, report TEXT, notes TEXT, paused TEXT
);
CREATE INDEX IF NOT EXISTS strategies_stage ON strategies(stage);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, bar_time TEXT, strategy_id TEXT, symbol TEXT, tf TEXT, side TEXT,
    action TEXT, reason TEXT, odds TEXT, features TEXT, snapshot TEXT, trade_id INTEGER
);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT, symbol TEXT, tf TEXT, side TEXT, shadow INTEGER,
    stage TEXT, status TEXT, opened_at TEXT, closed_at TEXT, entry REAL, stop REAL, target REAL, stop_now REAL,
    qty REAL, risk_pct REAL, risk_amount REAL, exit_price REAL, pnl REAL, r REAL, mfe_r REAL, mae_r REAL,
    bars_held INTEGER, exit_reason TEXT, broker_id TEXT, decision_id INTEGER, features TEXT, postmortem TEXT,
    state TEXT, estimated INTEGER DEFAULT 0, expected_r REAL
);
CREATE INDEX IF NOT EXISTS trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS trades_strategy ON trades(strategy_id);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, balance REAL, equity REAL, peak REAL, drawdown REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, message TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS research_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, tf TEXT, bars INTEGER, trials INTEGER,
    candidates INTEGER, holdout_looks INTEGER, passed INTEGER, seconds REAL, summary TEXT
);
CREATE TABLE IF NOT EXISTS trial_stats (symbol TEXT, tf TEXT, n INTEGER, sum_inv_n REAL, PRIMARY KEY (symbol, tf));
CREATE TABLE IF NOT EXISTS holdout_looks (symbol TEXT, tf TEXT, window TEXT, looks INTEGER, PRIMARY KEY (symbol, tf, window));
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY, asset_class TEXT, score REAL, spread REAL, atr REAL, cost_ratio REAL,
    hours_open REAL, chosen INTEGER, reason TEXT, ts TEXT
);
"""

JSON_COLS = {"genome", "report", "odds", "features", "snapshot", "postmortem", "state", "data", "paused", "summary"}
STAGES_LIVE = ("probation", "active", "proven")
STAGES_RUNNING = ("shadow",) + STAGES_LIVE


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path | str, clock=now_iso) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.clock = clock

    # --- helpers ----------------------------------------------------------------------
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if d[k] is not None:
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, ValueError):
                    pass
        return d

    @staticmethod
    def _enc(v: Any) -> Any:
        return json.dumps(v, default=str) if isinstance(v, (dict, list)) else v

    def query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        with self.lock:
            return [self._row(r) for r in self.conn.execute(sql, params).fetchall()]

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def insert(self, table: str, row: dict) -> int:
        cols = list(row)
        return self.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [self._enc(row[c]) for c in cols])

    def update(self, table: str, key: str, key_val: Any, **fields) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", [self._enc(v) for v in fields.values()] + [key_val])

    # --- strategies ----------------------------------------------------------------------
    def add_strategy(self, sid: str, name: str, symbol: str, tf: str, family: str, description: str, genome: dict,
                     stage: str, report: dict) -> None:
        if self.strategy(sid):
            return
        now = self.clock()
        self.insert("strategies", {"id": sid, "name": name, "symbol": symbol, "tf": tf, "family": family,
                                   "description": description, "genome": genome, "stage": stage, "created_at": now,
                                   "stage_changed_at": now, "report": report, "notes": "", "paused": []})

    def strategy(self, sid: str) -> dict | None:
        r = self.query("SELECT * FROM strategies WHERE id=?", (sid,))
        return r[0] if r else None

    def strategies(self, stages: tuple | None = None) -> list[dict]:
        if stages:
            q = ",".join("?" * len(stages))
            return self.query(f"SELECT * FROM strategies WHERE stage IN ({q}) ORDER BY created_at", stages)
        return self.query("SELECT * FROM strategies ORDER BY created_at")

    def set_stage(self, sid: str, stage: str, why: str) -> None:
        s = self.strategy(sid)
        if not s or s["stage"] == stage:
            return
        notes = (s.get("notes") or "") + f"\n{self.clock()[:16]} {s['stage']} -> {stage}: {why}"
        self.update("strategies", "id", sid, stage=stage, stage_changed_at=self.clock(), notes=notes.strip())
        self.event("stage", f"{s['name']}: {s['stage']} -> {stage}. {why}", {"strategy": sid, "from": s["stage"], "to": stage})

    # --- decisions & trades -------------------------------------------------------------
    def decision(self, **row) -> int:
        row.setdefault("ts", self.clock())
        return self.insert("decisions", row)

    def decisions(self, since: str | None = None, limit: int = 500, strategy: str | None = None,
                  with_snapshot: bool = False) -> list[dict]:
        cols = "*" if with_snapshot else "id, ts, bar_time, strategy_id, symbol, tf, side, action, reason, odds, features, trade_id"
        sql, p = f"SELECT {cols} FROM decisions WHERE 1=1", []
        if since:
            sql += " AND ts >= ?"
            p.append(since)
        if strategy:
            sql += " AND strategy_id = ?"
            p.append(strategy)
        sql += " ORDER BY id DESC LIMIT ?"
        p.append(limit)
        return self.query(sql, p)

    def open_trade(self, **row) -> int:
        row.setdefault("opened_at", self.clock())
        row["status"] = "open"
        return self.insert("trades", row)

    def trade(self, tid: int) -> dict | None:
        r = self.query("SELECT * FROM trades WHERE id=?", (tid,))
        return r[0] if r else None

    def trades(self, where: str = "1=1", params: tuple | list = (), order: str = "id", limit: int | None = None) -> list[dict]:
        sql = f"SELECT * FROM trades WHERE {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, params)

    def open_trades(self, shadow: bool | None = None) -> list[dict]:
        if shadow is None:
            return self.trades("status='open'")
        return self.trades("status='open' AND shadow=?", (int(shadow),))

    def closed_trades(self, shadow: bool | None = False, strategy: str | None = None, since: str | None = None) -> list[dict]:
        where, p = "status='closed'", []
        if shadow is not None:
            where += " AND shadow=?"
            p.append(int(shadow))
        if strategy:
            where += " AND strategy_id=?"
            p.append(strategy)
        if since:
            where += " AND closed_at>=?"
            p.append(since)
        return self.trades(where, p, order="closed_at")

    # --- account / events / kv ------------------------------------------------------------
    def record_equity(self, balance: float, equity: float, peak: float) -> None:
        dd = (peak - equity) / peak if peak > 0 else 0.0
        self.execute("INSERT INTO equity VALUES (?,?,?,?,?)", (self.clock(), balance, equity, peak, dd))

    def equity_curve(self, since: str | None = None) -> list[dict]:
        if since:
            return self.query("SELECT * FROM equity WHERE ts>=? ORDER BY ts", (since,))
        return self.query("SELECT * FROM equity ORDER BY ts")

    def event(self, kind: str, message: str, data: Any = None) -> None:
        self.execute("INSERT INTO events (ts, kind, message, data) VALUES (?,?,?,?)",
                     (self.clock(), kind, message, json.dumps(data, default=str) if data is not None else None))

    def events(self, since: str | None = None, limit: int = 200, kinds: tuple | None = None) -> list[dict]:
        sql, p = "SELECT * FROM events WHERE 1=1", []
        if since:
            sql += " AND ts>=?"
            p.append(since)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            p += list(kinds)
        sql += " ORDER BY id DESC LIMIT ?"
        p.append(limit)
        return self.query(sql, p)

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            r = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value, default=str)))

    # --- research bookkeeping -------------------------------------------------------------
    def add_trials(self, symbol: str, tf: str, n: int, sum_inv_n: float) -> tuple[int, float]:
        self.execute("INSERT INTO trial_stats VALUES (?,?,?,?) ON CONFLICT(symbol, tf) DO UPDATE SET "
                     "n = n + excluded.n, sum_inv_n = sum_inv_n + excluded.sum_inv_n", (symbol, tf, n, sum_inv_n))
        return self.trial_stats(symbol, tf)

    def trial_stats(self, symbol: str, tf: str) -> tuple[int, float]:
        r = self.query("SELECT n, sum_inv_n FROM trial_stats WHERE symbol=? AND tf=?", (symbol, tf))
        return (r[0]["n"], r[0]["sum_inv_n"]) if r else (0, 0.0)

    def total_trials(self) -> int:
        r = self.query("SELECT COALESCE(SUM(n), 0) AS n FROM trial_stats")
        return int(r[0]["n"])

    def register_look(self, symbol: str, tf: str, window: str) -> int:
        self.execute("INSERT INTO holdout_looks VALUES (?,?,?,1) ON CONFLICT(symbol, tf, window) DO UPDATE SET "
                     "looks = looks + 1", (symbol, tf, window))
        return int(self.query("SELECT looks FROM holdout_looks WHERE symbol=? AND tf=? AND window=?",
                              (symbol, tf, window))[0]["looks"])

    def prune(self, keep_days: int = 60) -> None:
        """Drop chart snapshots of old passed-on setups (taken trades keep theirs forever)."""
        self.execute("UPDATE decisions SET snapshot=NULL WHERE trade_id IS NULL AND ts < datetime('now', ?)",
                     (f"-{int(keep_days)} days",))
