"""`gimenez why`: plain-English summary of the last N hours."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .journal import Journal
from .scorecard import scorecard


def _bucket(reason: str) -> str:
    """Group similar pass reasons ('spread 0.31 abnormal' and 'spread 0.29 abnormal' are the same thing)."""
    r = (reason or "").lower()
    for key, label in [("abnormal spread", "abnormal spread"), ("spread is", "spread too expensive vs risk"),
                       ("stale", "stale data / market closed"), ("misquote", "possible misquote"),
                       ("self-hedging", "opposite position open"), ("already in a position", "already in that market"),
                       ("paused", "strategy paused in this market condition"), ("daily loss", "daily loss stop"),
                       ("halted", "drawdown halt"), ("stop file", "you ran `gimenez stop`"), ("open risk", "6% open-risk cap"),
                       ("max ", "max open positions"), ("too small", "position size below minimum"),
                       ("order rate", "order rate limit"), ("too soon", "order rate limit"),
                       ("too tight", "stop too tight/wide for current volatility"), ("order failed", "order rejected"),
                       ("2% cap", "2% risk cap")]:
        if key in r:
            return label
    return reason[:80]


def report(j: Journal, hours: int = 24) -> str:
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).isoformat()
    sc = scorecard(j)
    dec = j.decisions(since=since, limit=10000)
    taken = [d for d in dec if d["action"] == "taken"]
    shadow = [d for d in dec if d["action"] == "shadow"]
    passed = [d for d in dec if d["action"] == "passed"]
    closed = j.closed_trades(shadow=None, since=since)
    errors = j.events(since=since, kinds=("error",), limit=50)
    stages = j.events(since=since, kinds=("stage", "discovery", "learning"), limit=50)
    runs = j.query("SELECT * FROM research_runs WHERE ts>=? ORDER BY id", (since,))
    acct = j.get("account") or {}
    L = []
    L.append(f"GIMENEZ - last {hours}h")
    L.append(f"Verdict: {sc['verdict'].upper()}  ({'; '.join(sc['evidence'][:2])})")
    L.append(f"  {sc['termination']}")
    if acct:
        L.append(f"Account: equity {acct.get('equity', 0):,.2f}, peak {acct.get('peak', 0):,.2f}, "
                 f"drawdown {sc['drawdown_pct'] or 0:.1%}" + (f"  HALTED: {sc['halted']}" if sc["halted"] else ""))
    L.append("")
    L.append(f"Setups found: {len(dec)}  |  real trades taken: {len(taken)}  |  virtual (shadow): {len(shadow)}  |  passed on: {len(passed)}")
    for d in taken[:10]:
        L.append(f"  TAKEN  {d['ts'][5:16]} {d['side'].upper()} {d['symbol']} {d['tf']} - {d['reason']}")
    if passed:
        L.append("Passed on, by reason:")
        for reason, n in Counter(_bucket(d["reason"]) for d in passed).most_common(10):
            L.append(f"  {n:4d} x {reason}")
    if closed:
        real = [t for t in closed if not t["shadow"]]
        sh = [t for t in closed if t["shadow"]]
        L.append(f"Closed: {len(real)} real ({sum(t['r'] or 0 for t in real):+.2f}R), "
                 f"{len(sh)} shadow ({sum(t['r'] or 0 for t in sh):+.2f}R)")
        for t in real[-10:]:
            tags = ", ".join((t.get("postmortem") or {}).get("tags", []))
            L.append(f"  #{t['id']} {t['symbol']} {t['side']} {t['r']:+.2f}R ({t['exit_reason']}) {tags}")
    if runs:
        ideas = sum(r["trials"] for r in runs)
        looks = sum(r["holdout_looks"] for r in runs)
        passed_n = sum(r["passed"] for r in runs)
        L.append(f"Research: {len(runs)} runs, {ideas} ideas tested, {looks} reached the holdout, {passed_n} passed everything "
                 f"(all-time ideas tested: {j.total_trials()})")
        fails = Counter()
        for r in runs:
            for c in (r.get("summary") or {}).get("top", []):
                if c.get("why"):
                    fails[c["stage"]] += 1
        if fails:
            L.append("  best candidates were stopped at: " + ", ".join(f"{k} ({v})" for k, v in fails.most_common()))
    if stages:
        L.append("Pipeline / learning:")
        for e in stages[:10]:
            L.append(f"  {e['ts'][5:16]} {e['message']}")
    if errors:
        L.append(f"Errors ({len(errors)}):")
        for e in errors[:8]:
            L.append(f"  {e['ts'][5:16]} {e['message'][:160]}")
    else:
        L.append("Errors: none")
    return "\n".join(L)
