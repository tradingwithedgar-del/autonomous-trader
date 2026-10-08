"""Read-only dashboard (it cannot place or change trades). Password-protected with HTTP Basic auth:
any user name + DASHBOARD_PASSWORD from .env. The page layout and the bundled TradingView
Lightweight Charts follow TIIM's dashboard (github.com/tradingwithedgar-del/tradingbot-dev1)."""
from __future__ import annotations

import base64
import hmac
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from ..config import Settings
from ..data.store import BarStore
from ..journal import Journal
from ..scorecard import scorecard, strategy_table
from ..why import report

STATIC = Path(__file__).parent / "static"
TRADE_COLS = ["id", "strategy_id", "symbol", "tf", "side", "shadow", "stage", "status", "opened_at", "closed_at", "entry",
              "stop", "target", "stop_now", "qty", "risk_pct", "exit_price", "pnl", "r", "mfe_r", "mae_r", "bars_held",
              "exit_reason", "expected_r", "estimated"]


def create_app(s: Settings | None = None, require_password: bool = True) -> FastAPI:
    s = s or Settings()
    password = os.getenv("DASHBOARD_PASSWORD", "")
    if require_password and not password:
        raise SystemExit("Set DASHBOARD_PASSWORD in .env before exposing the dashboard on the network.")
    j = Journal(s.db_path)
    store = BarStore(s.bars_path)
    app = FastAPI(title="Gimenez", docs_url=None, redoc_url=None, openapi_url=None)

    if password:
        @app.middleware("http")
        async def auth(request: Request, call_next):
            h = request.headers.get("authorization", "")
            ok = False
            if h.startswith("Basic "):
                try:
                    _, _, given = base64.b64decode(h[6:]).decode().partition(":")
                    ok = hmac.compare_digest(given.encode(), password.encode())
                except Exception:
                    ok = False
            if not ok:
                return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Gimenez"'})
            return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/app.js")
    def app_js():
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    @app.get("/vendor/lightweight-charts.js")
    def lwc():
        # TradingView Lightweight Charts (Apache-2.0), bundled so no CDN is needed.
        return FileResponse(STATIC / "vendor" / "lightweight-charts.js", media_type="text/javascript")

    @app.get("/api/overview")
    def overview():
        sc = scorecard(j)
        acct = j.get("account") or {}
        open_real = [{k: t.get(k) for k in TRADE_COLS} for t in j.open_trades(shadow=False)]
        stages = {}
        for st in j.strategies():
            stages[st["stage"]] = stages.get(st["stage"], 0) + 1
        alive = acct.get("ts")
        return {"scorecard": sc, "account": acct, "open": open_real, "stages": stages, "last_seen": alive,
                "mode": s.mode, "stop_file": s.stop_file.exists(), "ideas_tested": j.total_trials(),
                "limits": {"max_risk_per_trade": s.risk.max_risk_per_trade, "max_open_risk": s.risk.max_total_open_risk,
                           "daily_stop": s.risk.daily_loss_stop, "halt": s.risk.drawdown_halt}}

    @app.get("/api/equity")
    def equity():
        curve = j.equity_curve()
        step = max(1, len(curve) // 1500)
        return [{"ts": c["ts"], "equity": c["equity"], "balance": c["balance"], "drawdown": c["drawdown"]} for c in curve[::step]]

    @app.get("/api/strategies")
    def strategies():
        return strategy_table(j)

    @app.get("/api/trades")
    def trades(kind: str = "real", limit: int = 200):
        where = {"real": "shadow=0", "shadow": "shadow=1"}.get(kind, "1=1")
        rows = j.trades(where, order="id DESC", limit=min(limit, 1000))
        names = {x["id"]: x["name"] for x in j.strategies()}
        out = []
        for t in rows:
            d = {k: t.get(k) for k in TRADE_COLS}
            d["strategy"] = names.get(t["strategy_id"], t["strategy_id"])
            d["tags"] = (t.get("postmortem") or {}).get("tags", [])
            out.append(d)
        return out

    @app.get("/api/trade/{tid}")
    def trade(tid: int):
        t = j.trade(tid)
        if not t:
            raise HTTPException(404)
        dec = j.query("SELECT * FROM decisions WHERE id=?", (t["decision_id"],)) if t.get("decision_id") else []
        st = j.strategy(t["strategy_id"]) or {}
        after = _after(t, (dec[0].get("snapshot") or {}) if dec else {})
        return {"trade": t, "decision": dec[0] if dec else None, "after": after,
                "strategy": {k: st.get(k) for k in ("id", "name", "description", "stage", "family", "report")}}

    def _after(t: dict, snap: dict) -> dict | None:
        """Bars from the snapshot's end until a while after the exit: 'what happened next'."""
        if not snap.get("t"):
            return None
        start = pd.Timestamp(snap["t"][-1]) + pd.Timedelta(seconds=1)
        end = pd.Timestamp(t["closed_at"]) + pd.Timedelta(hours=12) if t.get("closed_at") else None
        df = store.load(t["symbol"], t["tf"], start=start, end=end)
        df = df.iloc[:400]
        return {"t": [x.isoformat() for x in df.index], **{c: [round(float(v), 6) for v in df[c]] for c in
                                                           ("open", "high", "low", "close")}}

    @app.get("/api/decisions")
    def decisions(action: str = "", limit: int = 200):
        rows = j.decisions(limit=min(limit, 2000))
        if action:
            rows = [r for r in rows if r["action"] == action]
        names = {x["id"]: x["name"] for x in j.strategies()}
        for r in rows:
            r["strategy"] = names.get(r["strategy_id"], r["strategy_id"])
        return rows

    @app.get("/api/decision/{did}")
    def decision(did: int):
        r = j.query("SELECT * FROM decisions WHERE id=?", (did,))
        if not r:
            raise HTTPException(404)
        st = j.strategy(r[0]["strategy_id"]) or {}
        return {"decision": r[0], "strategy": {k: st.get(k) for k in ("id", "name", "description", "stage")}}

    @app.get("/api/research")
    def research():
        runs = j.query("SELECT * FROM research_runs ORDER BY id DESC LIMIT 100")
        tot = j.query("SELECT COUNT(*) AS runs, COALESCE(SUM(trials),0) AS ideas, COALESCE(SUM(candidates),0) AS candidates,"
                      " COALESCE(SUM(holdout_looks),0) AS looks, COALESCE(SUM(passed),0) AS passed FROM research_runs")[0]
        stages = {}
        for st in j.strategies():
            stages[st["stage"]] = stages.get(st["stage"], 0) + 1
        hist = [{"symbol": sym, "tf": tf, "bars": n} for sym, tf, n in store.datasets()]
        return {"totals": tot, "stages": stages, "runs": runs, "history": hist, "hints": j.get("research_hints", {})}

    @app.get("/api/watchlist")
    def watchlist():
        return {"rows": j.query("SELECT * FROM watchlist ORDER BY chosen DESC, score DESC"), "last_screen": j.get("last_screen"),
                "screening_left": len(j.get("screen_queue") or [])}

    @app.get("/api/events")
    def events(limit: int = 150):
        return j.events(limit=min(limit, 1000))

    @app.get("/api/why", response_class=PlainTextResponse)
    def why(hours: int = 24):
        return report(j, hours=hours)

    return app
