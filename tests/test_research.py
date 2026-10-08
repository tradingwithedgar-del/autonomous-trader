import numpy as np
import pytest

from gimenez.config import ResearchConfig
from gimenez.data.synthetic import planted_breakout, random_walk
from gimenez.research import backtest as bt
from gimenez.research.blocks import ENTRIES, FILTERS
from gimenez.research.genome import Genome
from gimenez.research.indicators import Data
from gimenez.research.search import Market, evolve, validate
from gimenez.research.stats import deflated_sharpe, expected_max_sharpe, norm_ppf, summarize


@pytest.fixture(scope="module")
def data():
    return Data(random_walk(n=2500, seed=3), "SYN", "15m")


def _genome(entry, filters=(), **exit_kw):
    ex = {"stop_mode": "atr", "sl_atr": 1.5, "swing_n": 10, "rr": 2.0, "trail": "none", "trail_atr": 2.0, "be_r": 1.0,
          "max_bars": 50}
    ex.update(exit_kw)
    return Genome("SYN", "15m", entry, list(filters), ex, "both")


@pytest.mark.parametrize("name", list(ENTRIES))
def test_entry_blocks_never_look_ahead(name, data):
    """A signal at bar i must be identical whether or not later bars exist."""
    rng = np.random.default_rng(1)
    for _ in range(3):
        g = Genome.random("SYN", "15m", rng, entry_type=name)
        g.filters = []
        full_l, full_s = g.signals(data)
        cut = 1700
        part = Data(data.df.iloc[:cut], "SYN", "15m")
        pl, ps = g.signals(part)
        assert np.array_equal(full_l[:cut], pl), name
        assert np.array_equal(full_s[:cut], ps), name


@pytest.mark.parametrize("name", list(FILTERS))
def test_filters_never_look_ahead(name, data):
    rng = np.random.default_rng(2)
    for _ in range(3):
        f = {"type": name, **{k: p.sample(rng) for k, p in FILTERS[name].params.items()}}
        cut = 1700
        part = Data(data.df.iloc[:cut], "SYN", "15m")
        fl, fs = FILTERS[name].fn(data, f)
        pl, ps = FILTERS[name].fn(part, f)
        assert np.array_equal(np.asarray(fl)[:cut], np.asarray(pl)), name
        assert np.array_equal(np.asarray(fs)[:cut], np.asarray(ps)), name


def test_step_exit_matches_vectorised_backtest(data):
    """Live trades are managed bar by bar with ExitState; it must agree exactly with the backtest."""
    rng = np.random.default_rng(5)
    costs = bt.Costs(spread=0.02, slip_atr=0.03, swap_per_day=0.0003)
    checked = 0
    for _ in range(40):
        g = Genome.random("SYN", "15m", rng)
        tr = bt.run(g, data, costs)
        a = data.atr(14)
        for k in range(min(len(tr), 15)):
            i = tr.entry_i[k] - 1
            side = tr.side[k]
            slip = costs.slip_atr * a[i]
            st = bt.ExitState(side, tr.entry[k], tr.stop[k], None if np.isnan(tr.target[k]) else tr.target[k], g.exit, a[i],
                              costs.spread / 2, slip, costs.swap_per_day * data.secs / 86400 * tr.entry[k])
            res = None
            j = i + 1
            while j < data.n and res is None:
                res = st.update(data.o[j], data.h[j], data.l[j], data.c[j])
                j += 1
            if res is None:
                continue
            assert j - 1 == tr.exit_i[k]
            assert res[0] == pytest.approx(tr.exit[k])
            assert res[1] == tr.reason[k]
            assert st.r_multiple(res[0]) == pytest.approx(tr.r[k])
            assert st.mfe == pytest.approx(tr.mfe[k])
            checked += 1
    assert checked > 50


def test_same_bar_stop_and_target_counts_as_a_loss():
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=400, freq="15min", tz="UTC")
    c = np.full(400, 100.0)
    df = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0}, index=idx)
    df.iloc[350, df.columns.get_loc("high")] = 110   # one huge bar touching both sides
    df.iloc[350, df.columns.get_loc("low")] = 90
    d = Data(df, "X", "15m")
    g = _genome({"type": "time_of_day", "hour": int(idx[348].hour)}, rr=1.0, sl_atr=1.0)
    g.side = "long"
    long_ = np.zeros(400, bool)
    long_[349] = True
    tr = bt.run(g, d, bt.Costs(0.0, 0.0, 0.0), signals=(long_, np.zeros(400, bool)))
    assert len(tr) == 1 and tr.reason[0] == "stop" and tr.r[0] < 0


def test_gap_through_stop_fills_at_the_open():
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=400, freq="15min", tz="UTC")
    c = np.full(400, 100.0)
    df = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0}, index=idx)
    df.iloc[351:, :4] = 95.0   # gap down far beyond the stop
    d = Data(df, "X", "15m")
    g = _genome({"type": "time_of_day", "hour": 0}, rr=3.0, sl_atr=1.0)
    long_ = np.zeros(400, bool)
    long_[349] = True
    tr = bt.run(g, d, bt.Costs(0.0, 0.0, 0.0), signals=(long_, np.zeros(400, bool)))
    assert tr.exit[0] == pytest.approx(95.0) and tr.r[0] < -1.5


def test_costs_make_results_worse(data):
    rng = np.random.default_rng(9)
    g = Genome.random("SYN", "15m", rng, entry_type="donchian")
    free = bt.run(g, data, bt.Costs(0.0, 0.0, 0.0)).R
    costly = bt.run(g, data, bt.Costs(0.05, 0.05, 0.001)).R
    assert len(free) and costly.mean() < free.mean()


def test_stats_basics():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01) > 0
    st = summarize(np.array([1.0, -1.0, 2.0, -1.0]))
    assert st["pf"] == pytest.approx(1.5) and st["win_rate"] == 0.5
    # the more ideas tried, the less a given Sharpe means
    assert deflated_sharpe(0.2, 200, 0, 3, 10, 1 / 200) > deflated_sharpe(0.2, 200, 0, 3, 100000, 1 / 200)


def _run_gauntlet(gen, seeds, cfg):
    looks = {"n": 0}

    def look():
        looks["n"] += 1
        return looks["n"]
    passed = tested = 0
    for seed in seeds:
        m = Market(Data(gen(n=16000, seed=seed), f"S{seed}", "15m"), bt.Costs(0.01))
        res = evolve(m, cfg, np.random.default_rng(seed))
        var = float(np.mean(res.inv_n)) if res.inv_n else 0.0
        for e in res.best[:3]:
            tested += 1
            passed += validate(e, m, cfg, np.random.default_rng(seed), res.trials, var, [], look).passed
    return passed, tested


def test_noise_is_rejected():
    """The core credibility test: on data with NO edge, nothing may pass validation."""
    cfg = ResearchConfig()
    cfg.generations = 6
    passed, tested = _run_gauntlet(random_walk, [21, 22, 23], cfg)
    assert tested >= 6 and passed == 0


def test_real_edge_is_found():
    cfg = ResearchConfig()
    cfg.generations = 6
    passed, tested = _run_gauntlet(planted_breakout, [31, 32], cfg)
    assert passed >= 2


@pytest.mark.parametrize("name", list(ENTRIES))
def test_every_block_actually_runs(name, data):
    """The search treats a crashing idea as a bad idea, so a bug could hide a whole family. Not here."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        g = Genome.random("SYN", "15m", rng, entry_type=name)
        bt.run(g, data, bt.Costs(0.01))   # must not raise
        g.overlays(data)
