"""Learning from every trade.

* Post-mortem: each closed trade (real or shadow) gets tags explaining what happened, then is
  watched for a while after the exit (was the stop just hunted? did the target leave a lot?).
* Pipeline review: promotes strategies whose live results match their backtest and retires the
  ones whose edge decayed or never existed (a big live-vs-backtest gap = overfitting).
* Regime pauses: a strategy that keeps losing in one market condition (live evidence) stops
  trading in that condition.
* Hints for research: recurring mistakes in a family (e.g. stops too tight) bias future searches.
"""
from __future__ import annotations

import numpy as np

from ..config import Settings
from ..journal import Journal
from ..research.stats import bootstrap_mean_ci, drawdown_quantile, gap_z, summarize

TAG_NOTES = {
    "clean_win": "Reached the target without drawing down more than 0.5R.",
    "survived_drawdown_win": "Won after first going more than 0.7R against.",
    "trailed_win": "Trailing stop locked in a profit.",
    "gave_back_profit": "Was at least +1R up before turning into a loss.",
    "immediately_wrong": "Never moved more than 0.3R in favour: entry idea or timing was wrong.",
    "time_exit": "Closed by the time limit before stop or target.",
    "regime_mismatch": "Taken in a market condition where the backtest showed no edge.",
    "costly_spread": "Spread at entry was more than 10% of the risk.",
    "slippage": "Filled noticeably worse than the planned price (>0.1R).",
    "stop_too_tight": "Stopped out, then price went on to the original target: the stop was probably too tight.",
    "thesis_invalid": "After the stop, price kept going against the trade: the stop did its job.",
    "target_too_close": "After the target, price ran another 2R+: the target may be too conservative.",
    "estimated_exit": "Exit price was estimated (not found in the broker's history).",
}


def initial_tags(t: dict) -> list[str]:
    r, mfe, mae = t.get("r") or 0.0, t.get("mfe_r") or 0.0, t.get("mae_r") or 0.0
    f = t.get("features") or {}
    tags = []
    if r > 0:
        tags.append("trailed_win" if t.get("exit_reason") == "trail" else ("clean_win" if mae < 0.5 else
                                                                            ("survived_drawdown_win" if mae > 0.7 else "win")))
    else:
        if mfe >= 1.0:
            tags.append("gave_back_profit")
        elif mfe < 0.3:
            tags.append("immediately_wrong")
    if t.get("exit_reason") == "time":
        tags.append("time_exit")
    if f.get("regime_expectancy") is not None and f["regime_expectancy"] <= 0:
        tags.append("regime_mismatch")
    if f.get("spread_vs_risk", 0) > 0.10:
        tags.append("costly_spread")
    if f.get("slippage_r", 0) > 0.1:
        tags.append("slippage")
    if t.get("estimated"):
        tags.append("estimated_exit")
    return tags


def new_watch(t: dict, bars: int = 40) -> dict:
    return {"bars_left": bars, "hit_target": False, "beyond_stop": False, "max_after_r": 0.0}


def update_watch(t: dict, w: dict, h: float, l: float) -> dict:
    side = 1 if t["side"] == "buy" else -1
    entry, stop = t["entry"], t["stop"]
    risk = abs(entry - stop) or 1e-12
    target = t.get("target")
    w["bars_left"] -= 1
    fav, adv = (h, l) if side > 0 else (l, h)
    if not w["beyond_stop"] and not w["hit_target"]:
        if (adv - stop) * side < -risk * 0.5:
            w["beyond_stop"] = True
        elif target and (fav - target) * side >= 0:
            w["hit_target"] = True
    ref = target if target else (t.get("exit_price") or entry)
    w["max_after_r"] = max(w["max_after_r"], (fav - ref) * side / risk)
    return w


def final_tags(t: dict, w: dict) -> list[str]:
    tags = list((t.get("postmortem") or {}).get("tags", []))
    if (t.get("r") or 0) < 0:
        if w["hit_target"]:
            tags.append("stop_too_tight")
        elif w["beyond_stop"]:
            tags.append("thesis_invalid")
    elif (t.get("r") or 0) > 0 and t.get("exit_reason") == "target" and w["max_after_r"] >= 2.0:
        tags.append("target_too_close")
    return tags


def explain(tags: list[str]) -> list[str]:
    return [TAG_NOTES[x] for x in tags if x in TAG_NOTES]


# --- pipeline -------------------------------------------------------------------------------
def expected(strategy: dict) -> tuple[float, float, list[float]]:
    rep = strategy.get("report") or {}
    e = rep.get("expected") or rep.get("holdout") or {}
    return float(e.get("mean") or 0.0), float(e.get("std") or 1.0), list(rep.get("expected_r") or [])


def review(j: Journal, s: Settings) -> list[str]:
    """Promote / retire strategies on the evidence. Returns what changed."""
    p = s.pipeline
    changes = []
    for st in j.strategies(("shadow", "probation", "active", "proven", "validated")):
        sid, stage = st["id"], st["stage"]
        bt_mean, bt_std, bt_r = expected(st)
        shadow = [t["r"] for t in j.closed_trades(shadow=True, strategy=sid)]
        real = [t["r"] for t in j.closed_trades(shadow=False, strategy=sid)]
        new, why = None, ""
        if stage == "validated":
            if len(j.strategies(("shadow",))) < p.max_shadow:
                new, why = "shadow", "a shadow slot opened up"
        elif stage == "shadow":
            n = len(shadow)
            z = gap_z(np.array(shadow), bt_mean, bt_std)
            if n >= p.shadow_min_trades and np.mean(shadow) > 0 and z > -2.0:
                if len(j.strategies(("probation", "active", "proven"))) < p.max_live:
                    new, why = "probation", f"{n} shadow trades, {np.mean(shadow):+.2f}R each, consistent with backtest (gap z={z:+.1f})"
            elif n >= 10 and z < p.gap_z_retire:
                new, why = "retired", f"shadow results far below backtest ({np.mean(shadow):+.2f}R vs {bt_mean:+.2f}R, z={z:+.1f}): overfit"
            elif n >= p.shadow_max_trades:
                new, why = "retired", f"{n} shadow trades without earning promotion ({np.mean(shadow):+.2f}R)"
        else:
            n = len(real)
            z = gap_z(np.array(real), bt_mean, bt_std)
            recent = real[-p.decay_window:]
            dd_limit = drawdown_quantile(np.array(bt_r), max(1, n), 0.95) if bt_r else float("inf")
            st_real = summarize(np.array(real))
            if n >= 10 and z < p.gap_z_retire:
                new, why = "retired", f"live {st_real['mean']:+.2f}R vs backtest {bt_mean:+.2f}R (z={z:+.1f}): the edge isn't real or has gone"
            elif n >= 10 and st_real["max_dd"] > dd_limit:
                new, why = "retired", f"live drawdown {st_real['max_dd']:.1f}R is worse than 95% of backtest scenarios ({dd_limit:.1f}R)"
            elif len(recent) >= p.decay_window and np.mean(recent) < 0 and bootstrap_mean_ci(np.array(recent), 0.9)[1] < 0.05:
                new, why = "retired", f"edge decayed: last {len(recent)} trades {np.mean(recent):+.2f}R"
            elif stage == "probation" and n >= p.probation_trades and st_real["mean"] > 0 and z > -2.0:
                new, why = "active", f"{n} real trades, {st_real['mean']:+.2f}R each, in line with backtest"
            elif stage == "active" and n >= p.proven_min_trades and bootstrap_mean_ci(np.array(real), 0.95)[0] > 0:
                new, why = "proven", f"{n} real trades, 95% confidence the edge is positive"
            elif stage == "proven" and len(recent) >= p.decay_window and np.mean(recent) < 0:
                new, why = "active", f"last {len(recent)} trades negative ({np.mean(recent):+.2f}R): risk reduced"
        if new:
            j.set_stage(sid, new, why)
            changes.append(f"{st['name']}: {stage} -> {new} ({why})")
    update_regime_pauses(j, s)
    update_hints(j)
    return changes


def regime_key(features: dict) -> str:
    return f"{features.get('vol_regime', '?')}|{features.get('session', '?')}"


def update_regime_pauses(j: Journal, s: Settings) -> None:
    p = s.pipeline
    for st in j.strategies(("shadow", "probation", "active", "proven")):
        cells: dict[str, list[float]] = {}
        for t in j.closed_trades(shadow=None, strategy=st["id"]):
            for k in ((t.get("features") or {}).get("vol_regime"), (t.get("features") or {}).get("session")):
                if k:
                    cells.setdefault(k, []).append(t["r"] or 0.0)
        paused = sorted(k for k, rs in cells.items()
                        if len(rs) >= p.regime_pause_min_trades and np.mean(rs) < p.regime_pause_expectancy)
        if paused != (st.get("paused") or []):
            j.update("strategies", "id", st["id"], paused=paused)
            if paused:
                j.event("learning", f"{st['name']} paused in: {', '.join(paused)} (keeps losing there live)",
                        {"strategy": st["id"], "paused": paused})


def update_hints(j: Journal) -> None:
    """If most losses of a family are 'stop too tight', future searches start with wider stops."""
    hints = {}
    fam: dict[str, list] = {}
    for t in j.closed_trades(shadow=None):
        if (t.get("r") or 0) < 0:
            st = j.strategy(t["strategy_id"])
            if st:
                fam.setdefault(st["family"], []).append(((t.get("postmortem") or {}).get("tags") or [], st))
    for f, rows in fam.items():
        if len(rows) >= 15:
            tight = sum(1 for tags, _ in rows if "stop_too_tight" in tags) / len(rows)
            if tight >= 0.4:
                sls = [st["genome"]["exit"].get("sl_atr", 1.5) for _, st in rows]
                hints[f] = {"sl_atr_min": round(float(np.median(sls)) * 1.25, 2), "evidence": f"{tight:.0%} of {len(rows)} losses"}
    if hints != j.get("research_hints", {}):
        j.set("research_hints", hints)
        if hints:
            j.event("learning", "research hints updated: " + "; ".join(f"{k}: wider stops ({v['evidence']} stopped too tight)"
                                                                       for k, v in hints.items()))
