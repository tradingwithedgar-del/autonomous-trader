"""The survival scorecard: is Gimenez actually working? Computed only from REAL demo trades
(shadow results are shown separately and never count as proof).

Verdicts
  no edge yet  - too few real trades, or results indistinguishable from zero
  promising    - 30+ real trades, positive expectancy and profit factor > 1, not yet significant
  proven       - 100+ real trades, 95% bootstrap confidence the expectancy is above zero,
                 profit factor >= 1.2, and live results not significantly below the backtests
  failing      - halted, drawdown >= 15%, or clearly negative expectancy

Termination test: if after TERMINATION_TRADES real trades or TERMINATION_DAYS days of real trading
the verdict is not "proven", the honest recommendation is to terminate the project.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .journal import Journal
from .research.stats import bootstrap_mean_ci, summarize

TERMINATION_TRADES = 200
TERMINATION_DAYS = 120


def _f(x, nd=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def scorecard(j: Journal) -> dict:
    real = j.closed_trades(shadow=False)
    shadow = j.closed_trades(shadow=True)
    r = np.array([t["r"] or 0.0 for t in real])
    st = summarize(r)
    rolling = summarize(r[-30:])
    lo95, hi95 = bootstrap_mean_ci(r, 0.95) if len(r) >= 5 else (None, None)
    exp = np.array([t["expected_r"] for t in real if t.get("expected_r") is not None])
    gap = float(r[[t.get("expected_r") is not None for t in real]].mean() - exp.mean()) if len(exp) else None
    curve = j.equity_curve()
    max_dd = max((c["drawdown"] for c in curve), default=0.0)
    acct = j.get("account") or {}
    cur_dd = (acct["peak"] - acct["equity"]) / acct["peak"] if acct.get("peak") else 0.0
    halted = j.get("halted")
    first = pd.Timestamp(real[0]["opened_at"]) if real else None
    days = (pd.Timestamp.now(tz="UTC") - first).days if first is not None else 0

    evidence = []
    n = st["n"]
    if halted or cur_dd >= 0.15 or (n >= 20 and hi95 is not None and hi95 < 0) or (n >= 30 and st["mean"] < -0.1):
        verdict = "failing"
        if halted:
            evidence.append(f"halted: {halted}")
        if cur_dd >= 0.15:
            evidence.append(f"account drawdown {cur_dd:.1%} (limit 25%)")
        if n >= 20 and hi95 is not None and hi95 < 0:
            evidence.append(f"95% confident expectancy is negative (at best {hi95:+.2f}R)")
        if n >= 30 and st["mean"] < -0.1:
            evidence.append(f"losing {st['mean']:+.2f}R per trade over {n} trades")
    elif n >= 100 and lo95 is not None and lo95 > 0 and st["pf"] >= 1.2 and (gap is None or gap > -0.15):
        verdict = "proven"
        evidence.append(f"{n} real trades, {st['mean']:+.2f}R each; 95% confidence interval {lo95:+.2f} to {hi95:+.2f}R")
        evidence.append(f"profit factor {st['pf']:.2f}")
    elif n >= 30 and st["mean"] > 0 and st["pf"] > 1:
        verdict = "promising"
        evidence.append(f"{n} real trades, {st['mean']:+.2f}R each, profit factor {st['pf']:.2f}")
        evidence.append(f"but the 95% confidence interval ({lo95:+.2f} to {hi95:+.2f}R) still includes zero"
                        if lo95 is not None and lo95 <= 0 else f"needs {max(0, 100 - n)} more trades to be called proven")
    else:
        verdict = "no edge yet"
        if n == 0:
            evidence.append("no real trades yet - nothing has earned real-money trading so far")
        elif n < 30:
            evidence.append(f"only {n} real trades: far too few to tell skill from luck")
        else:
            evidence.append(f"{n} real trades, {st['mean']:+.2f}R each: no measurable edge")
    if gap is not None and len(exp) >= 10:
        evidence.append(f"live vs backtest: {gap:+.2f}R per trade " + ("(live worse - a sign of overfitting)" if gap < -0.15
                                                                        else "(in line)"))
    if shadow:
        sh = summarize(np.array([t["r"] or 0 for t in shadow]))
        evidence.append(f"shadow (virtual, not proof): {sh['n']} trades, {sh['mean']:+.2f}R each")

    trades_left = max(0, TERMINATION_TRADES - n)
    days_left = max(0, TERMINATION_DAYS - days) if real else TERMINATION_DAYS
    if real and (trades_left == 0 or days_left == 0) and verdict != "proven":
        termination = (f"TERMINATION RECOMMENDED: {n} real trades over {days} days and still no proven edge.")
    elif verdict == "failing":
        termination = "At risk: failing now. Termination will be recommended if it doesn't recover."
    else:
        termination = (f"Deadline to prove an edge: {trades_left} more real trades or {days_left} days, whichever comes first."
                       if real else f"The clock starts with the first real trade ({TERMINATION_TRADES} trades / {TERMINATION_DAYS} days).")
    return {
        "verdict": verdict, "evidence": evidence, "termination": termination,
        "trades": n, "expectancy_r": _f(st["mean"]), "rolling30_r": _f(rolling["mean"]) if rolling["n"] else None,
        "profit_factor": _f(st["pf"], 2) if n else None, "win_rate": _f(st["win_rate"]) if n else None,
        "ci95": [_f(lo95), _f(hi95)] if lo95 is not None else None, "total_r": _f(st["total"], 2),
        "max_dd_r": _f(st["max_dd"], 2), "max_drawdown_pct": _f(max_dd), "drawdown_pct": _f(cur_dd),
        "live_vs_backtest_r": _f(gap), "shadow_trades": len(shadow),
        "equity": acct.get("equity"), "balance": acct.get("balance"), "peak": acct.get("peak"), "halted": halted,
        "days_trading": days,
    }


def strategy_table(j: Journal) -> list[dict]:
    """League table: every strategy it invented, its stage, backtest promise vs live reality."""
    out = []
    for s in j.strategies():
        rep = s.get("report") or {}
        exp = rep.get("expected") or {}
        real = [t["r"] or 0 for t in j.closed_trades(shadow=False, strategy=s["id"])]
        sh = [t["r"] or 0 for t in j.closed_trades(shadow=True, strategy=s["id"])]
        rs, ss = summarize(np.array(real)), summarize(np.array(sh))
        live_mean = rs["mean"] if real else (ss["mean"] if sh else None)
        gap = (live_mean - exp["mean"]) if live_mean is not None and exp.get("mean") is not None else None
        out.append({
            "id": s["id"], "name": s["name"], "symbol": s["symbol"], "tf": s["tf"], "family": s["family"],
            "description": s["description"], "stage": s["stage"], "since": s["stage_changed_at"], "created": s["created_at"],
            "backtest": {"n": exp.get("n"), "mean": exp.get("mean"), "pf": exp.get("pf"), "win_rate": exp.get("win_rate")},
            "holdout": rep.get("holdout"), "dsr": rep.get("dsr"), "plateau": rep.get("plateau"),
            "walk_forward": rep.get("walk_forward"), "peers": rep.get("peers"),
            "real": {"n": rs["n"], "mean": _f(rs["mean"]) if real else None, "pf": _f(rs["pf"], 2) if real else None,
                     "total": _f(rs["total"], 2)},
            "shadow": {"n": ss["n"], "mean": _f(ss["mean"]) if sh else None, "total": _f(ss["total"], 2)},
            "gap_r": _f(gap), "paused": s.get("paused") or [], "notes": s.get("notes"),
            "cum_r": list(np.round(np.cumsum(real), 3)) if real else [],
        })
    order = {"proven": 0, "active": 1, "probation": 2, "shadow": 3, "validated": 4, "retired": 5}
    return sorted(out, key=lambda x: (order.get(x["stage"], 9), -(x["real"]["total"] or 0)))
