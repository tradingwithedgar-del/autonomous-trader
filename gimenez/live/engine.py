"""The live trader. One loop, one broker connection:

  account check (daily stop / drawdown halt) -> sync real positions -> for every market/timeframe
  whose bar just closed: manage open trades, look for setups, decide (real / shadow / pass) and
  journal it with a chart snapshot -> idle work (screening, history backfill, pipeline review).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ..broker.base import TIMEFRAME_SECONDS, Broker
from ..config import Settings
from ..data.store import BarStore
from ..journal import STAGES_LIVE, STAGES_RUNNING, Journal
from ..research import backtest as bt
from ..research.genome import Genome
from ..research.indicators import Data
from ..research.search import session_of, vol_bucket
from . import learning, screener
from .guards import Compliance, RiskManager

log = logging.getLogger(__name__)
LOOKBACK = 3000          # bars loaded per evaluation (enough warm-up for every block)
SNAPSHOT_BARS = 150
MAX_STREAMS = 24


def features_at(d: Data, i: int, genome: Genome | None = None, report: dict | None = None) -> dict:
    a = float(d.atr(14)[i])
    rank = float(d.atr_rank(500)[i])
    e200 = d.ema(200)[i]
    f = {"atr": a, "atr_pct": rank if np.isfinite(rank) else None, "vol_regime": vol_bucket(rank),
         "session": session_of(int(d.hour[i])), "hour_utc": int(d.hour[i]), "weekday": int(d.dow[i]),
         "trend": ("up" if d.c[i] > e200 else "down") if np.isfinite(e200) else "unknown",
         "efficiency": float(d.er(20)[i]) if np.isfinite(d.er(20)[i]) else None, "close": float(d.c[i])}
    regs = (report or {}).get("regimes") or {}
    cells = [regs.get("vol", {}).get(f["vol_regime"]), regs.get("session", {}).get(f["session"])]
    cells = [c for c in cells if c and c.get("n", 0) >= 5]
    if cells:
        f["regime_expectancy"] = float(min(c["mean"] for c in cells))
    return f


def snapshot(d: Data, i: int, genome: Genome, levels: dict) -> dict:
    lo = max(0, i - SNAPSHOT_BARS + 1)
    sl = slice(lo, i + 1)
    rnd = lambda arr: [None if not np.isfinite(x) else round(float(x), 6) for x in arr[sl]]
    over = {}
    for name, arr in genome.overlays(d).items():
        arr = np.asarray(arr, dtype=float)
        if len(arr) == d.n:
            over[name] = rnd(arr)
    return {"t": [ts.isoformat() for ts in d.index[sl]], "open": rnd(d.o), "high": rnd(d.h), "low": rnd(d.l),
            "close": rnd(d.c), "overlays": over, "levels": levels}


class Engine:
    def __init__(self, s: Settings, broker: Broker, j: Journal, store: BarStore, wall=time.monotonic) -> None:
        s.validate()
        self.s, self.broker, self.j, self.store = s, broker, j, store
        self.compliance = Compliance(s.compliance, clock=wall)
        self.risk = RiskManager(s.risk)
        self.wall = wall
        self.last_equity_rec = -1e9
        self.last_review = -1e9
        self.last_spread_sample = -1e9
        self.spread_rr = 0
        self.errors_in_row = 0

    # --- time -----------------------------------------------------------------------------
    def now(self) -> pd.Timestamp:
        return getattr(self.broker, "now", None) or pd.Timestamp.now(tz="UTC")

    # --- main loop ------------------------------------------------------------------------
    def run_forever(self) -> None:  # pragma: no cover - service
        self.j.event("start", f"Gimenez started ({self.s.mode}, {self.broker.name})")
        while True:
            t0 = self.wall()
            try:
                self.step()
                self.errors_in_row = 0
            except Exception as e:
                self.errors_in_row += 1
                log.exception("loop error")
                self.j.event("error", f"loop error: {e}"[:500])
                if self.errors_in_row > 5:
                    time.sleep(min(300, 15 * self.errors_in_row))
            time.sleep(max(1.0, self.s.poll_seconds - (self.wall() - t0)))

    def step(self) -> None:
        now = self.now()
        blocked = self.account(now)
        self.sync_real(now)
        did_work = False
        for (sym, tf), strats in self.streams().items():
            last = self.j.get(f"bar:{sym}:{tf}")
            secs = TIMEFRAME_SECONDS[tf]
            if last and pd.Timestamp(last) + pd.Timedelta(seconds=2 * secs) > now:
                continue   # the next bar hasn't closed yet
            bars = self.store.update_recent(self.broker, sym, tf, min_bars=LOOKBACK)
            if bars.empty or (last and bars.index[-1] <= pd.Timestamp(last)):
                continue
            self.j.set(f"bar:{sym}:{tf}", bars.index[-1].isoformat())
            self.on_bar(sym, tf, bars, strats, blocked, now)
            did_work = True
        if not did_work:
            self.idle(now)

    def streams(self) -> dict[tuple[str, str], list[dict]]:
        out: dict[tuple[str, str], list[dict]] = {}
        for st in self.j.strategies(STAGES_RUNNING):
            out.setdefault((st["symbol"], st["tf"]), []).append(st)
        for t in self.j.open_trades():   # keep managing trades of strategies that were just retired
            if (t["symbol"], t["tf"]) not in out:
                st = self.j.strategy(t["strategy_id"])
                if st:
                    out[(t["symbol"], t["tf"])] = []
        return dict(list(out.items())[:MAX_STREAMS])

    # --- account & safety -----------------------------------------------------------------
    def account(self, now: pd.Timestamp) -> str | None:
        """Returns a reason string if new real trades are not allowed."""
        eq, bal = self.broker.equity(), self.broker.balance()
        peak = max(float(self.j.get("peak", 0.0) or 0.0), eq)
        self.j.set("peak", peak)
        day = (now - pd.Timedelta(hours=self.s.risk.day_start_utc_hour)).date().isoformat()
        day_start = self.j.get(f"day_start:{day}")
        if day_start is None:
            day_start = eq
            self.j.set(f"day_start:{day}", eq)
        self.j.set("account", {"equity": eq, "balance": bal, "peak": peak, "day_start": day_start, "ts": now.isoformat()})
        if self.wall() - self.last_equity_rec > 300:
            self.j.record_equity(bal, eq, peak)
            self.last_equity_rec = self.wall()
        halted = self.j.get("halted")
        if halted:
            return halted
        lim = self.risk.limits(eq, peak, day_start)
        if lim:
            if "drawdown" in lim:
                self.j.set("halted", lim)
                self.j.event("halt", lim)
            elif self.j.get("daily_stop") != day:
                self.j.set("daily_stop", day)
                self.j.event("daily_stop", lim)
            self.flatten(lim)
            return lim
        if self.j.get("daily_stop") == day:
            return "daily loss stop hit today"
        if self.s.stop_file.exists():
            return "STOP file present (`gimenez stop`): no new real trades"
        return None

    def flatten(self, why: str) -> None:
        for t in self.j.open_trades(shadow=False):
            try:
                self.broker.close_position(t["broker_id"])
            except Exception as e:
                self.j.event("error", f"could not close {t['symbol']} position {t['broker_id']}: {e}")
        self.sync_real(self.now(), reason="flattened: " + why)

    def sync_real(self, now: pd.Timestamp, reason: str | None = None) -> None:
        open_real = self.j.open_trades(shadow=False)
        if not open_real:
            return
        live = {p.id: p for p in self.broker.open_positions()}
        for t in open_real:
            p = live.get(t["broker_id"])
            if p is not None:
                if p.has_stop is False:   # hard rule: a stop at the broker, always
                    ok = self.broker.modify_stop(p.id, t["stop_now"] or t["stop"])
                    self.j.event("error" if not ok else "repair", f"{t['symbol']} position had no stop at the broker; "
                                 + ("re-attached it" if ok else "could not re-attach, closing"))
                    if not ok:
                        self.broker.close_position(p.id)
                continue
            info = self.broker.closed_info(t["broker_id"], t["symbol"], t["side"], t["stop_now"] or t["stop"], t["target"])
            if info is None:
                continue
            st = t.get("state") or {}
            why = reason or self._exit_reason(t, info.exit_price)
            self.close_trade(t, info.exit_price, info.exit_time, why, st.get("mfe", 0.0), st.get("mae", 0.0),
                             st.get("bars", 0), estimated=info.estimated)

    @staticmethod
    def _exit_reason(t: dict, px: float) -> str:
        risk = abs(t["entry"] - t["stop"]) or 1e-12
        if t["target"] and abs(px - t["target"]) < 0.15 * risk:
            return "target"
        stop_now = t["stop_now"] or t["stop"]
        if abs(px - stop_now) < 0.25 * risk:
            return "trail" if (stop_now - t["stop"]) * (1 if t["side"] == "buy" else -1) > 0 else "stop"
        return "closed"

    # --- per bar ----------------------------------------------------------------------------
    def on_bar(self, sym: str, tf: str, bars: pd.DataFrame, strats: list[dict], blocked: str | None, now: pd.Timestamp) -> None:
        d = Data(bars, sym, tf)
        i = d.n - 1
        self.manage_open(sym, tf, d)
        self.watch_closed(sym, tf, d)
        for st in strats:
            try:
                self.evaluate(st, d, i, blocked, now)
            except Exception as e:
                log.exception("evaluate %s", st["id"])
                self.j.event("error", f"{st['name']}: {e}"[:500])

    def manage_open(self, sym: str, tf: str, d: Data) -> None:
        for t in self.j.trades("status='open' AND symbol=? AND tf=?", (sym, tf)):
            st = t.get("state") or {}
            ex = self._exit_state(t)
            last = pd.Timestamp(st["last_bar"]) if st.get("last_bar") else None
            new = [k for k in range(d.n) if last is None or d.index[k] > last]
            if last is None:
                new = [k for k in new if d.index[k] >= pd.Timestamp(t["opened_at"]) - pd.Timedelta(seconds=d.secs)][-1:]
            for k in new:
                res = ex.update(d.o[k], d.h[k], d.l[k], d.c[k])
                st.update(last_bar=d.index[k].isoformat(), bars=ex.bars, mfe=ex.mfe, mae=ex.mae, stop_now=ex.stop_now,
                          best=ex.best)
                if t["shadow"]:
                    if res:
                        self.close_trade(t, res[0], d.index[k] + pd.Timedelta(seconds=d.secs), res[1], ex.mfe, ex.mae, ex.bars)
                        break
                else:
                    if res and res[1] == "time":
                        try:
                            self.broker.close_position(t["broker_id"])
                            self.j.event("trade", f"#{t['id']} {sym}: time limit reached, closed")
                        except Exception as e:
                            self.j.event("error", f"#{t['id']} could not close at time limit: {e}")
                    elif not res and abs(ex.stop_now - (t["stop_now"] or t["stop"])) > 1e-12:
                        if self.broker.modify_stop(t["broker_id"], ex.stop_now):
                            self.j.update("trades", "id", t["id"], stop_now=ex.stop_now)
                            t["stop_now"] = ex.stop_now
                        else:
                            self.j.event("error", f"#{t['id']} could not move the stop to {ex.stop_now:.5g}")
            else:
                self.j.update("trades", "id", t["id"], state=st, mfe_r=ex.mfe, mae_r=ex.mae, bars_held=ex.bars,
                              **({"stop_now": ex.stop_now} if t["shadow"] else {}))
                continue
        if not self.j.trades("status='open' AND shadow=0 AND symbol=?", (sym,)):
            return
        self.sync_real(self.now())

    def _exit_state(self, t: dict) -> bt.ExitState:
        st = t.get("state") or {}
        side = 1 if t["side"] == "buy" else -1
        ex = bt.ExitState(side, t["entry"], t["stop"], t["target"], st["exit"], st["atr"], st["half"], st["slip"],
                          st["swap_per_bar"])
        ex.bars, ex.mfe, ex.mae = st.get("bars", 0), st.get("mfe", 0.0), st.get("mae", 0.0)
        ex.stop_now = st.get("stop_now", t["stop"])
        ex.best = st.get("best", -np.inf) if st.get("best") is not None else -np.inf
        return ex

    def watch_closed(self, sym: str, tf: str, d: Data) -> None:
        for t in self.j.trades("status='closed' AND symbol=? AND tf=? AND state LIKE '%\"watch\"%'", (sym, tf)):
            st = t.get("state") or {}
            w = st.get("watch")
            if not w or w["bars_left"] <= 0:
                continue
            last = pd.Timestamp(st.get("watch_last") or t["closed_at"])
            for k in range(d.n):
                if d.index[k] > last and w["bars_left"] > 0:
                    w = learning.update_watch(t, w, d.h[k], d.l[k])
                    st["watch_last"] = d.index[k].isoformat()
            st["watch"] = w
            pm = t.get("postmortem") or {}
            if w["bars_left"] <= 0:
                tags = learning.final_tags(t, w)
                pm = {**pm, "tags": tags, "notes": learning.explain(tags), "after_exit": w}
                st.pop("watch", None)
            self.j.update("trades", "id", t["id"], state=st, postmortem=pm)

    # --- decisions --------------------------------------------------------------------------
    def evaluate(self, st: dict, d: Data, i: int, blocked: str | None, now: pd.Timestamp) -> None:
        g = Genome.from_dict(st["genome"])
        long_, short = g.signals(d)
        if not (long_[i] or short[i]):
            return
        if self.j.trades("status='open' AND strategy_id=?", (st["id"],)):
            return   # one trade at a time per strategy, exactly as in the backtest
        last = self.j.trades("status='closed' AND strategy_id=?", (st["id"],), order="closed_at DESC", limit=1)
        if last and pd.Timestamp(last[0]["closed_at"]) > d.index[i]:
            return   # the backtest doesn't re-enter on the bar a trade exited either
        side = 1 if long_[i] else -1
        sname = "buy" if side > 0 else "sell"
        feats = features_at(d, i, g, st.get("report"))
        atr_i = float(d.atr(14)[i])
        q = self.broker.get_quote(d.symbol)
        slip = self.s.research.slippage_atr * atr_i
        entry_est = (q.ask if side > 0 else q.bid) + side * slip
        stop = bt.initial_stop(d, i, side, entry_est, g.exit, atr_i)
        odds = self.odds(st, feats)
        base = dict(bar_time=d.index[i].isoformat(), strategy_id=st["id"], symbol=d.symbol, tf=d.tf, side=sname,
                    odds=odds, features=feats)
        if stop is None:
            self.j.decision(**base, action="passed", reason="stop would be too tight or too wide for current volatility",
                            snapshot=snapshot(d, i, g, {}))
            return
        risk_px = abs(entry_est - stop)
        target = entry_est + side * g.exit["rr"] * risk_px if g.exit["rr"] else None
        levels = {"entry": entry_est, "stop": stop, "target": target}
        snap = snapshot(d, i, g, levels)
        feats["spread"] = q.spread
        feats["spread_vs_risk"] = q.spread / risk_px if risk_px > 0 else None
        typical = self.store.typical_spread(d.symbol)
        problem = self.compliance.check(symbol=d.symbol, side=sname, entry=entry_est, stop=stop, quote=q,
                                        last_close=float(d.c[i]), atr=atr_i, typical_spread=typical,
                                        last_bar_time=d.index[i], tf=d.tf, now=now,
                                        positions=[] if st["stage"] == "shadow" else self.broker.open_positions())
        if problem and ("position" not in problem or st["stage"] == "shadow"):
            self.j.decision(**base, action="passed", reason=problem, snapshot=snap)
            return
        paused = st.get("paused") or []
        hit = [p for p in paused if p in (feats["vol_regime"], feats["session"])]
        if hit:
            self.j.decision(**base, action="passed", reason=f"strategy paused in {', '.join(hit)} (it keeps losing there live)",
                            snapshot=snap)
            return
        state = {"exit": g.exit, "atr": atr_i, "half": q.spread / 2, "slip": slip,
                 "swap_per_bar": self.s.research.swap_per_day * d.secs / 86400.0 * entry_est}
        if st["stage"] in STAGES_LIVE:
            why_not = blocked or problem
            if not why_not:
                why_not = self.open_real(st, d, side, entry_est, stop, target, state, base, snap, feats, odds)
                if why_not is None:
                    return
            # couldn't trade it for real: follow it virtually so the evidence keeps coming
            did = self.j.decision(**base, action="passed", reason=why_not, snapshot=snap)
            self.open_shadow(st, d, side, entry_est, stop, target, state, did, feats, odds)
            return
        did = self.j.decision(**base, action="shadow", reason="shadow-testing: virtual trade on live prices", snapshot=snap)
        self.open_shadow(st, d, side, entry_est, stop, target, state, did, feats, odds)

    def odds(self, st: dict, feats: dict) -> dict:
        rep = st.get("report") or {}
        exp = rep.get("expected") or {}
        real = [t["r"] for t in self.j.closed_trades(shadow=False, strategy=st["id"])]
        shadow = [t["r"] for t in self.j.closed_trades(shadow=True, strategy=st["id"])]
        return {"expected_win_rate": exp.get("win_rate"), "expected_r": exp.get("mean"), "expected_pf": exp.get("pf"),
                "based_on_trades": exp.get("n"), "regime_expectancy": feats.get("regime_expectancy"),
                "live_trades": len(real), "live_r": float(np.mean(real)) if real else None,
                "shadow_trades": len(shadow), "shadow_r": float(np.mean(shadow)) if shadow else None,
                "stage": st["stage"]}

    def open_shadow(self, st, d, side, entry, stop, target, state, decision_id, feats, odds) -> int:
        state = dict(state, last_bar=d.index[-1].isoformat())
        tid = self.j.open_trade(strategy_id=st["id"], symbol=d.symbol, tf=d.tf, side="buy" if side > 0 else "sell",
                                shadow=1, stage=st["stage"], entry=entry, stop=stop, target=target, stop_now=stop,
                                qty=0, risk_pct=0, risk_amount=0, decision_id=decision_id, features=feats, state=state,
                                expected_r=odds.get("expected_r"), opened_at=self.now().isoformat())
        self.j.update("decisions", "id", decision_id, trade_id=tid)
        return tid

    def open_real(self, st, d, side, entry, stop, target, state, base, snap, feats, odds) -> str | None:
        """Place a real order. Returns None on success or the reason it wasn't placed."""
        acct = self.j.get("account") or {}
        eq, peak = float(acct.get("equity") or self.broker.equity()), float(acct.get("peak") or 0)
        open_real = self.j.open_trades(shadow=False)
        last_closed = self.j.trades("status='closed' AND shadow=0", order="closed_at DESC", limit=1)
        live_r = [t["r"] for t in self.j.closed_trades(shadow=False, strategy=st["id"])]
        spec = self.broker.spec(d.symbol)
        rd = self.risk.size(equity=eq, peak=peak, stage=st["stage"], live_r=live_r, entry=entry, stop=stop, spec=spec,
                            open_risk_pct=sum(t["risk_pct"] or 0 for t in open_real), n_open=len(open_real),
                            last_closed=last_closed[0] if last_closed else None)
        if not rd.ok:
            return rd.reason
        sname = "buy" if side > 0 else "sell"
        try:
            pid = self.broker.place_market(d.symbol, sname, rd.qty, stop, target, f"gz-{st['id']}")
        except Exception as e:
            self.j.event("error", f"order failed {d.symbol} {sname}: {e}"[:500])
            return f"order failed: {e}"[:200]
        self.compliance.record_order(d.symbol)
        fill = next((p.entry for p in self.broker.open_positions() if p.id == pid), entry)
        risk_px = abs(fill - stop)
        feats = dict(feats, slippage_r=(fill - (entry - side * state["slip"])) * side / risk_px if risk_px else 0.0)
        reason = f"{st['stage']} strategy signal; {rd.why_this_risk}"
        did = self.j.decision(**{**base, "features": feats}, action="taken", reason=reason, snapshot=snap)
        state = dict(state, last_bar=d.index[-1].isoformat())
        tid = self.j.open_trade(strategy_id=st["id"], symbol=d.symbol, tf=d.tf, side=sname, shadow=0, stage=st["stage"],
                                entry=fill, stop=stop, target=target, stop_now=stop, qty=rd.qty, risk_pct=rd.risk_pct,
                                risk_amount=rd.risk_amount, broker_id=pid, decision_id=did, features=feats, state=state,
                                expected_r=odds.get("expected_r"), opened_at=self.now().isoformat())
        self.j.update("decisions", "id", did, trade_id=tid)
        self.j.event("trade", f"OPEN #{tid} {sname.upper()} {rd.qty:g} {d.symbol} @ {fill:.5g}, stop {stop:.5g}"
                     + (f", target {target:.5g}" if target else "") + f", risking {rd.risk_pct:.2%} - {st['name']}",
                     {"trade": tid})
        return None

    def close_trade(self, t: dict, exit_px: float, exit_time, reason: str, mfe: float, mae: float, bars: int,
                    estimated: bool = False) -> None:
        side = 1 if t["side"] == "buy" else -1
        risk = abs(t["entry"] - t["stop"]) or 1e-12
        st = t.get("state") or {}
        swap = (st.get("swap_per_bar") or 0.0) * bars
        r = ((exit_px - t["entry"]) * side - (swap if t["shadow"] else 0.0)) / risk
        pnl = r * (t["risk_amount"] or 0.0)
        t2 = dict(t, r=r, exit_reason=reason, mfe_r=mfe, mae_r=mae, estimated=int(estimated), exit_price=exit_px)
        tags = learning.initial_tags(t2)
        st["watch"] = learning.new_watch(t2)
        st["watch_last"] = pd.Timestamp(exit_time).isoformat()
        self.j.update("trades", "id", t["id"], status="closed", closed_at=pd.Timestamp(exit_time).isoformat(),
                      exit_price=float(exit_px), r=float(r), pnl=float(pnl), mfe_r=float(mfe), mae_r=float(mae),
                      bars_held=int(bars), exit_reason=reason, estimated=int(estimated), state=st,
                      postmortem={"tags": tags, "notes": learning.explain(tags)})
        if not t["shadow"]:
            self.j.event("trade", f"CLOSE #{t['id']} {t['symbol']} {reason}: {r:+.2f}R ({pnl:+.2f})", {"trade": t["id"]})

    # --- background work --------------------------------------------------------------------
    def idle(self, now: pd.Timestamp) -> None:
        last_screen = self.j.get("last_screen")
        due = not last_screen or (now - pd.Timestamp(last_screen)).total_seconds() > self.s.screen.rescreen_hours * 3600
        if due or self.j.get("screen_queue"):
            screener.screen_step(self.broker, self.store, self.j, self.s, per_step=4)
            return
        if self.wall() - self.last_spread_sample > 60:
            wl = [r["symbol"] for r in self.j.query("SELECT symbol FROM watchlist WHERE chosen=1")]
            if wl:
                sym = wl[self.spread_rr % len(wl)]
                self.spread_rr += 1
                q = self.broker.get_quote(sym)
                self.store.record_spread(sym, q.spread, q.mid)
            self.last_spread_sample = self.wall()
        if self.wall() - self.last_review > 3600:
            for msg in learning.review(self.j, self.s):
                log.info(msg)
            self.store.prune()
            self.j.prune()
            self.last_review = self.wall()
            return
        for _ in range(3):
            if not self.backfill_one():
                break

    def backfill_one(self) -> bool:
        status = self.store.backfill_status()
        wl = [r["symbol"] for r in self.j.query("SELECT symbol FROM watchlist WHERE chosen=1 ORDER BY score DESC")]
        for sym in wl:
            for tf in self.s.research.timeframes:
                if not status.get((sym, tf)):
                    try:
                        self.store.backfill_step(self.broker, sym, tf)
                    except Exception as e:
                        self.j.event("error", f"history download {sym} {tf}: {e}"[:300])
                        self.store._mark(sym, tf, True, 0)
                    return True
        return False
