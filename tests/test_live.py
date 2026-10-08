import numpy as np
import pandas as pd
import pytest

from gimenez.broker.base import InstrumentSpec, Position, Quote
from gimenez.broker.fake import FakeBroker
from gimenez.config import ComplianceConfig, RiskConfig, Settings
from gimenez.data.store import BarStore
from gimenez.data.synthetic import planted_breakout
from gimenez.journal import Journal
from gimenez.live import engine as eng_mod
from gimenez.live.engine import Engine
from gimenez.live.guards import Compliance, RiskManager
from gimenez.research import backtest as bt
from gimenez.research.genome import Genome
from gimenez.research.indicators import Data

GENOME = Genome("SYN", "15m", {"type": "donchian", "n": 20}, [],
                {"stop_mode": "atr", "sl_atr": 1.5, "swing_n": 10, "rr": 2.0, "trail": "chandelier", "trail_atr": 2.5,
                 "be_r": 1.0, "max_bars": 40}, "both")
REPORT = {"expected": {"n": 100, "mean": 0.3, "std": 1.2, "pf": 1.5, "win_rate": 0.45}, "expected_r": [0.5, -1, 2, -1, 1] * 20}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 30.0
        return self.t


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(eng_mod, "LOOKBACK", 1200)
    df = planted_breakout(n=2600, seed=8)
    s = Settings()
    s.data_dir = tmp_path
    s.mode = "demo"
    s.tl_environment = "https://demo.tradelocker.com"
    fb = FakeBroker({"SYN": df}, "15m", spreads={"SYN": 0.01}, equity=10_000.0, value_per_point=1.0)
    fb.cursor = 1301
    j = Journal(tmp_path / "g.db")
    j.set("last_screen", pd.Timestamp.now(tz="UTC").isoformat())   # skip screening in these tests
    store = BarStore(tmp_path / "bars.db")
    e = Engine(s, fb, j, store, wall=Clock())
    return s, fb, j, store, e, df


def _add(j, stage, g=GENOME):
    j.add_strategy(g.id, g.short_name(), g.symbol, g.tf, g.family, g.describe(), g.to_dict(), stage, REPORT)
    return g.id


def test_shadow_trades_reproduce_the_backtest(setup):
    s, fb, j, store, e, df = setup
    _add(j, "shadow")
    for _ in range(1250):
        e.step()
        fb.advance()
    live = j.closed_trades(shadow=True)
    assert len(live) >= 15
    tr = bt.run(GENOME, Data(df, "SYN", "15m"), bt.Costs(0.01, s.research.slippage_atr, s.research.swap_per_day))
    idx = df.index
    first = idx.get_loc(pd.Timestamp(live[0]["opened_at"]) - pd.Timedelta(seconds=1))
    bt_trades = [k for k in range(len(tr)) if tr.entry_i[k] >= first][:len(live)]
    assert len(bt_trades) == len(live)
    for t, k in zip(live, bt_trades):
        assert idx.get_loc(pd.Timestamp(t["opened_at"]) - pd.Timedelta(seconds=1)) == tr.entry_i[k]
        assert t["entry"] == pytest.approx(tr.entry[k])
        assert t["exit_price"] == pytest.approx(tr.exit[k])
        assert t["r"] == pytest.approx(tr.r[k], abs=1e-9)
        assert t["exit_reason"] == tr.reason[k]
    # every setup was journaled with a chart snapshot and the odds
    d = j.decisions(limit=5, with_snapshot=True)
    assert d and d[0]["snapshot"]["close"] and "expected_r" in d[0]["odds"]


def test_real_trades_have_broker_stops_and_correct_size(setup):
    s, fb, j, store, e, df = setup
    _add(j, "probation")
    for _ in range(400):
        e.step()
        fb.advance()
    real = j.trades("shadow=0")
    assert real, "expected some real trades"
    for o in fb.orders:
        assert o["stop"] is not None
    t = real[0]
    assert t["risk_pct"] == pytest.approx(s.risk.risk_probation, rel=0.05)
    assert t["risk_pct"] <= 0.02
    closed = [x for x in real if x["status"] == "closed"]
    assert closed and all(x["r"] is not None for x in closed)


def test_missing_broker_stop_is_repaired(setup):
    s, fb, j, store, e, df = setup
    _add(j, "probation")
    fb.drop_stops = True
    for _ in range(300):
        e.step()
        fb.advance()
        if fb.positions:
            break
    e.step()
    assert fb.positions and all(p["stop"] is not None for p in fb.positions.values())
    assert any(ev["kind"] == "repair" for ev in j.events())


def test_daily_loss_stop_flattens_and_blocks(setup):
    s, fb, j, store, e, df = setup
    _add(j, "probation")
    for _ in range(300):   # until a real trade is open
        e.step()
        if fb.positions:
            break
        fb.advance()
    assert fb.positions
    fb.cash *= 0.93   # -7% on the day
    e.step()
    assert j.get("daily_stop")
    assert not fb.positions and not j.trades("shadow=0 AND status='open'")
    day = fb.now.date()
    orders = len(fb.orders)
    while (fb.now + pd.Timedelta(minutes=15)).date() == day:
        fb.advance()
        e.step()
    assert len(fb.orders) == orders, "no new real orders for the rest of the day"


def test_drawdown_halt_needs_resume(setup):
    s, fb, j, store, e, df = setup
    e.step()
    fb.cash *= 0.70
    e.step()
    assert "halted" in (j.get("halted") or "")
    fb.cash /= 0.70
    e.step()
    assert j.get("halted"), "recovery alone must not lift the halt"


# --- guards --------------------------------------------------------------------------------
def _check(c, **kw):
    now = pd.Timestamp("2025-01-01 12:00:01", tz="UTC")
    args = dict(symbol="X", side="buy", entry=100.0, stop=99.0, quote=Quote(99.99, 100.01, now), last_close=100.0,
                atr=1.0, typical_spread=0.02, last_bar_time=pd.Timestamp("2025-01-01 11:45", tz="UTC"), tf="15m",
                now=now, positions=[])
    args.update(kw)
    return c.check(**args)


def test_compliance_blocks_abusive_patterns():
    c = Compliance(ComplianceConfig(), clock=lambda: 1000.0)
    now = pd.Timestamp("2025-01-01 12:00:01", tz="UTC")
    assert _check(c) is None
    assert "abnormal spread" in _check(c, quote=Quote(99.9, 100.1, now))
    assert "misquote" in _check(c, quote=Quote(104.99, 105.01, now))
    assert "stale" in _check(c, last_bar_time=pd.Timestamp("2025-01-01 10:00", tz="UTC"))
    assert "too tight" in _check(c, stop=99.8)
    assert "hedging" in _check(c, positions=[Position("1", "X", "sell", 1, 100)])
    assert "adding" in _check(c, positions=[Position("1", "X", "buy", 1, 100)])
    assert "invalid quote" in _check(c, quote=Quote(0, 100.01, now))
    for _ in range(4):
        c.record_order("Y" + str(_))
    assert "rate limit" in _check(c)


def test_risk_never_exceeds_hard_caps_and_never_sizes_up_after_loss():
    r = RiskManager(RiskConfig())
    spec = InstrumentSpec(1.0, 0.01, 0.01, 1e6)
    base = dict(equity=10_000, peak=10_000, entry=100.0, stop=99.0, spec=spec, open_risk_pct=0.0, n_open=0, last_closed=None)
    big_edge = list(np.tile([3.0, 3.0, -1.0], 60))
    d = r.size(stage="proven", live_r=big_edge, **base)
    assert d.ok and d.risk_pct <= 0.02 + 1e-9
    d2 = r.size(stage="proven", live_r=big_edge, **{**base, "last_closed": {"r": -1.0, "risk_pct": 0.005}})
    assert d2.risk_pct <= 0.005 + 1e-9
    d3 = r.size(stage="active", live_r=[], **{**base, "open_risk_pct": 0.058})
    assert d3.ok and d3.risk_pct <= 0.002 + 1e-9
    d4 = r.size(stage="active", live_r=[], **{**base, "open_risk_pct": 0.06})
    assert not d4.ok
    d5 = r.size(stage="active", live_r=[], **{**base, "peak": 12_500})   # 20% drawdown -> smaller
    assert d5.risk_pct < 0.005
    assert r.limits(7_400, 10_000, 7_500) and "drawdown" in r.limits(7_400, 10_000, 7_500)
    assert "daily" in r.limits(9_350, 10_000, 10_000)


def test_settings_refuse_live_and_loose_risk():
    s = Settings()
    s.tl_environment = "https://live.tradelocker.com"
    s.mode = "demo"
    with pytest.raises(RuntimeError):
        s.validate()
    s.mode = "live"
    s.allow_live = "NO"
    with pytest.raises(RuntimeError):
        s.validate()
    s2 = Settings()
    s2.tl_environment = "https://demo.tradelocker.com"
    s2.risk.max_risk_per_trade = 0.03
    with pytest.raises(ValueError):
        s2.validate()


# --- store -----------------------------------------------------------------------------------
def test_store_roundtrip_and_backfill(tmp_path):
    df = planted_breakout(n=3000, seed=1)
    fb = FakeBroker({"SYN": df}, "15m")
    fb.cursor = 2999
    store = BarStore(tmp_path / "b.db")
    store.upsert("SYN", "15m", df.iloc[:100])
    back = store.load("SYN", "15m")
    assert len(back) == 100 and np.allclose(back["close"].to_numpy(), df["close"].iloc[:100].to_numpy())
    store2 = BarStore(tmp_path / "c.db")
    fb.max_history_rows = lambda: 500
    store2.update_recent(fb, "SYN", "15m", min_bars=200)
    steps = 0
    while store2.backfill_step(fb, "SYN", "15m") and steps < 50:
        steps += 1
    assert store2.span("SYN", "15m")[2] >= 2990
