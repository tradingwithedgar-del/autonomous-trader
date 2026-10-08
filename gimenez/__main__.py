"""Gimenez command line. Run `python -m gimenez --help` (on the server: `gimenez --help`)."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings


def _journal(s: Settings):
    from .journal import Journal

    return Journal(s.db_path)


def cmd_run(a) -> None:
    from .broker.tradelocker import TradeLockerBroker
    from .data.store import BarStore
    from .live.engine import Engine

    s = Settings()
    s.validate()
    eng = Engine(s, TradeLockerBroker(s), _journal(s), BarStore(s.bars_path))
    eng.run_forever()


def cmd_research(a) -> None:
    from .research.worker import main_loop

    main_loop(Settings())


def cmd_dashboard(a) -> None:
    import uvicorn

    from .dashboard.app import create_app

    uvicorn.run(create_app(Settings(), require_password=a.host not in {"127.0.0.1", "localhost"}),
                host=a.host, port=a.port, log_level="warning")


def cmd_doctor(a) -> None:
    """Connection check: logs in, reads the account, instruments, a quote and some history. Places no orders."""
    import pandas as pd

    from .broker.tradelocker import TradeLockerBroker

    s = Settings()
    print(f"mode={s.mode}  server={s.tl_server}  environment={s.tl_environment}")
    s.validate()
    print("safety check OK (demo only unless you explicitly enable live)")
    b = TradeLockerBroker(s)
    print(f"logged in. account currency {b._account_ccy}, balance {b.balance():,.2f}, equity {b.equity():,.2f}")
    print(f"rate limits used: {b.calls.rate:.2f} req/s (trading), {b.hist_calls.rate:.2f} req/s (history); "
          f"max {b.max_history_rows()} bars per history request")
    inst = b.instruments()
    by = pd.Series([i.asset_class for i in inst]).value_counts()
    print(f"{len(inst)} instruments: " + ", ".join(f"{k} {v}" for k, v in by.items()))
    sym = a.symbol or next((i.symbol for i in inst if i.asset_class == "metal"), inst[0].symbol)
    q = b.get_quote(sym)
    print(f"{sym}: bid {q.bid} ask {q.ask} spread {q.spread:.5g}")
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    for tf, days in (("5m", 2), ("1H", 10)):
        df = b.history(sym, tf, end - days * 86_400_000, end)
        print(f"{sym} {tf}: {len(df)} bars, last {df.index[-1] if len(df) else '-'}")
    print(f"open positions: {len(b.open_positions())}")
    print("ALL GOOD" if len(inst) else "connected, but no instruments returned")


def cmd_status(a) -> None:
    from .scorecard import scorecard, strategy_table

    s = Settings()
    j = _journal(s)
    sc = scorecard(j)
    print(f"VERDICT: {sc['verdict'].upper()}")
    for e in sc["evidence"]:
        print(f"  - {e}")
    print(f"  {sc['termination']}")
    if sc.get("equity") is not None:
        print(f"equity {sc['equity']:,.2f}  drawdown {sc['drawdown_pct'] or 0:.1%}  max drawdown {sc['max_drawdown_pct'] or 0:.1%}")
    print(f"real trades {sc['trades']}  expectancy {sc['expectancy_r']}R  PF {sc['profit_factor']}  win rate {sc['win_rate']}")
    tbl = strategy_table(j)
    counts = {}
    for t in tbl:
        counts[t["stage"]] = counts.get(t["stage"], 0) + 1
    print("strategies: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "none yet (research is still running)"))
    for t in j.open_trades(shadow=False):
        print(f"OPEN #{t['id']} {t['side']} {t['qty']} {t['symbol']} @ {t['entry']:.5g} stop {t['stop_now']:.5g}")
    if s.stop_file.exists():
        print("STOP file present: no new real trades (run `gimenez resume` to allow them again)")


def cmd_why(a) -> None:
    from .why import report

    print(report(_journal(Settings()), hours=a.hours))


def cmd_strategies(a) -> None:
    from .scorecard import strategy_table

    for t in strategy_table(_journal(Settings())):
        bt = t["backtest"]
        print(f"[{t['stage']:9s}] {t['name']}")
        print(f"    {t['description']}")
        print(f"    backtest(out-of-sample) {bt['n']} trades {bt['mean'] or 0:+.2f}R | real {t['real']['n']} "
              f"{(t['real']['mean'] or 0):+.2f}R | shadow {t['shadow']['n']} {(t['shadow']['mean'] or 0):+.2f}R")


def cmd_stop(a) -> None:
    s = Settings()
    s.stop_file.parent.mkdir(parents=True, exist_ok=True)
    s.stop_file.write_text("no new real trades\n")
    _journal(s).event("stop", "owner ran `gimenez stop`: no new real trades")
    print("Stopped: no NEW real trades. Open trades keep their stops at the broker. Undo with `gimenez resume`.")


def cmd_resume(a) -> None:
    s = Settings()
    j = _journal(s)
    if s.stop_file.exists():
        s.stop_file.unlink()
    was = j.get("halted")
    j.set("halted", None)
    if was:
        acct = j.get("account") or {}
        j.set("peak", acct.get("equity", 0.0))   # drawdown is measured from today's equity again
    j.event("resume", "owner resumed Gimenez" + (f" after halt: {was}" if was else ""))
    print("Resumed." + (f" (was halted: {was})" if was else ""))


def cmd_screen(a) -> None:
    j = _journal(Settings())
    j.set("last_screen", None)
    print("A fresh market screen will start within a minute.")


def cmd_selftest(a) -> None:
    """Honesty check on synthetic data: noise must be rejected, a planted edge should be found."""
    import numpy as np

    from .config import ResearchConfig
    from .data.synthetic import planted_breakout, random_walk
    from .research import backtest as bt
    from .research.indicators import Data
    from .research.search import Market, evolve, validate

    cfg = ResearchConfig()
    cfg.generations = 8
    looks = {"n": 0}

    def look():
        looks["n"] += 1
        return looks["n"]
    for label, gen in (("pure noise (no edge exists)", random_walk), ("planted breakout edge", planted_breakout)):
        passed = tested = 0
        for seed in range(a.seeds):
            m = Market(Data(gen(n=20000, seed=100 + seed), f"SYN{seed}", "15m"), bt.Costs(0.01))
            res = evolve(m, cfg, np.random.default_rng(seed))
            var = float(np.mean(res.inv_n)) if res.inv_n else 0.0
            for e in res.best[:3]:
                tested += 1
                passed += validate(e, m, cfg, np.random.default_rng(seed), res.trials, var, [], look).passed
        print(f"{label}: {passed}/{tested} top candidates passed validation")
    print("Expected: noise ~0 passes, planted edge mostly passes.")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("tradelocker").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="gimenez", description="Gimenez - autonomous self-learning trading agent (demo)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run the live trader (demo)").set_defaults(fn=cmd_run)
    sub.add_parser("research", help="run the strategy-discovery worker").set_defaults(fn=cmd_research)
    d = sub.add_parser("dashboard", help="serve the dashboard")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8001)
    d.set_defaults(fn=cmd_dashboard)
    doc = sub.add_parser("doctor", help="check the TradeLocker connection (places no orders)")
    doc.add_argument("--symbol", default="")
    doc.set_defaults(fn=cmd_doctor)
    sub.add_parser("status", help="scorecard, strategies, open trades").set_defaults(fn=cmd_status)
    w = sub.add_parser("why", help="what happened recently and why")
    w.add_argument("--hours", type=int, default=24)
    w.set_defaults(fn=cmd_why)
    sub.add_parser("strategies", help="every strategy it invented and how it is doing").set_defaults(fn=cmd_strategies)
    sub.add_parser("stop", help="no new real trades").set_defaults(fn=cmd_stop)
    sub.add_parser("resume", help="allow trading again (also clears a drawdown halt)").set_defaults(fn=cmd_resume)
    sub.add_parser("screen", help="re-screen all markets now").set_defaults(fn=cmd_screen)
    st = sub.add_parser("selftest", help="prove the overfitting filter works on synthetic data")
    st.add_argument("--seeds", type=int, default=3)
    st.set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
