import base64

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gimenez.broker.fake import FakeBroker
from gimenez.config import Settings
from gimenez.data.store import BarStore
from gimenez.data.synthetic import random_walk
from gimenez.journal import Journal
from gimenez.live import screener
from gimenez.scorecard import scorecard
from gimenez.why import report


def _trades(j, rs, shadow=0, expected=0.2):
    t0 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    for k, r in enumerate(rs):
        tid = j.open_trade(strategy_id="g1", symbol="X", tf="15m", side="buy", shadow=shadow, stage="active", entry=1.0,
                           stop=0.9, target=1.2, qty=1, risk_pct=0.005, risk_amount=50, expected_r=expected,
                           opened_at=(t0 + pd.Timedelta(hours=k)).isoformat())
        j.update("trades", "id", tid, status="closed", r=float(r), closed_at=(t0 + pd.Timedelta(hours=k, minutes=30)).isoformat(),
                 postmortem={"tags": ["clean_win" if r > 0 else "immediately_wrong"]})


def test_verdicts_are_honest(tmp_path):
    j = Journal(tmp_path / "a.db")
    assert scorecard(j)["verdict"] == "no edge yet"
    _trades(j, [1.0, -1.0] * 5 + [2.0])          # 11 trades, positive but tiny sample
    assert scorecard(j)["verdict"] == "no edge yet"
    rng = np.random.default_rng(0)
    j2 = Journal(tmp_path / "b.db")
    _trades(j2, list(rng.choice([2.0, -1.0], size=150, p=[0.5, 0.5])))
    sc = scorecard(j2)
    assert sc["verdict"] == "proven" and sc["ci95"][0] > 0
    j3 = Journal(tmp_path / "c.db")
    _trades(j3, list(rng.choice([1.0, -1.0], size=40, p=[0.3, 0.7])))
    assert scorecard(j3)["verdict"] == "failing"
    j4 = Journal(tmp_path / "d.db")
    _trades(j4, [1.0, -1.0] * 2, shadow=1)
    sc4 = scorecard(j4)
    assert sc4["verdict"] == "no edge yet" and sc4["trades"] == 0, "shadow results never count as proof"
    j5 = Journal(tmp_path / "e.db")
    j5.set("halted", "drawdown 25% from peak reached")
    assert scorecard(j5)["verdict"] == "failing"


def test_why_report_runs(tmp_path):
    j = Journal(tmp_path / "a.db")
    _trades(j, [1.0, -1.0, 0.5])
    j.decision(strategy_id="g1", symbol="X", tf="15m", side="buy", action="passed", reason="abnormal spread 0.3 (usual 0.1)")
    j.event("error", "something broke")
    txt = report(j, 24 * 30)
    assert "abnormal spread" in txt and "something broke" in txt and "Verdict" in txt


def test_dashboard_requires_password_and_serves(tmp_path, monkeypatch):
    s = Settings()
    s.data_dir = tmp_path
    j = Journal(s.db_path)
    _trades(j, [1.0, -1.0, 0.5])
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    with pytest.raises(SystemExit):
        from gimenez.dashboard.app import create_app
        create_app(s, require_password=True)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
    from gimenez.dashboard.app import create_app
    c = TestClient(create_app(s))
    assert c.get("/api/overview").status_code == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"x:wrong").decode()}
    assert c.get("/api/overview", headers=bad).status_code == 401
    h = {"Authorization": "Basic " + base64.b64encode(b"anyone:secret").decode()}
    for url in ["/", "/app.js", "/vendor/lightweight-charts.js", "/api/overview", "/api/equity", "/api/strategies",
                "/api/trades?kind=all", "/api/trade/1", "/api/decisions", "/api/research", "/api/watchlist", "/api/events",
                "/api/why"]:
        assert c.get(url, headers=h).status_code == 200, url
    assert c.get("/api/overview", headers=h).json()["scorecard"]["trades"] == 3


def test_screener_prefers_cheap_liquid_markets(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.screen.watchlist_size = 2
    data = {"CHEAP": random_walk(n=3000, timeframe="1H", seed=1), "PRICEY": random_walk(n=3000, timeframe="1H", seed=2),
            "MID": random_walk(n=3000, timeframe="1H", seed=3)}
    fb = FakeBroker(data, "1H", spreads={"CHEAP": 0.01, "PRICEY": 0.5, "MID": 0.02})
    fb.cursor = 2999
    j, store = Journal(s.db_path), BarStore(s.bars_path)
    done = False
    for _ in range(10):
        done = screener.screen_step(fb, store, j, s, per_step=1)
        if done:
            break
    assert done
    chosen = {r["symbol"] for r in j.query("SELECT symbol FROM watchlist WHERE chosen=1")}
    assert chosen == {"CHEAP", "MID"}
    pricey = j.query("SELECT reason FROM watchlist WHERE symbol='PRICEY'")[0]["reason"]
    assert "expensive" in pricey
