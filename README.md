# Gimenez

A fully autonomous, self-learning trading agent for a **PlexyTrade demo account on TradeLocker**.
It picks its own markets, invents and tests its own strategies, chooses its own timeframes, risk and
reward:risk, learns from every trade, and tells you honestly whether it has an edge.
Its existence depends on proving one. If it can't, the scorecard will say so.

Sister project: [TIIM](https://github.com/tradingwithedgar-del/tradingbot-dev1) (branch
`claude/clever-lamport-2atw36`). Gimenez reuses and adapts TIIM's plumbing, with credit in each file:
the TradeLocker connector, the deployment approach (systemd + test-gated auto-update), the
password-protected dashboard with TradingView Lightweight Charts, and the `why` command. It does
**not** use any of TIIM's strategy logic; every strategy Gimenez trades, it found itself.

Setup: see [SETUP.md](SETUP.md) (one step at a time).

---

## Realistic expectations (read this first)

- **Most likely outcome: no edge, or a small and fragile one.** Retail-accessible intraday edges after
  spread and slippage are rare. A strict system *should* reject almost everything it tries. Weeks
  with zero strategies passing are normal, and are the filter doing its job, not a failure of the code.
- **It cannot be fast.** "High frequency" here means decisions on closed 1m-5m (or slower) candles.
  TradeLocker is a REST API with rate limits; true HFT is impossible and latency arbitrage is forbidden.
- **History is limited** to what TradeLocker serves. Fast timeframes may only go back weeks or months,
  which limits how much can be proven from backtests. Live forward results are the real test.
- **Proof takes time.** "Proven" needs 100+ real trades with 95% confidence the expectancy is above
  zero. At a few trades a day that is weeks to months. Shadow (virtual) results never count as proof.
- **Demo is not live.** Demo fills can be kinder than live ones. Even a proven demo edge needs your
  decision (and an explicit setting) before anything touches real money. Gimenez refuses otherwise.
- **Termination rule:** if after 200 real trades or 120 days of real trading the verdict is not
  "proven", the scorecard recommends terminating the project.

## How it works

```
 every instrument on the account
          |  screener: spread vs hourly range, hours open, activity -> watchlist (re-screened weekly)
          v
 price history (downloaded slowly, within rate limits, kept on disk)
          |
          v
 RESEARCH (low CPU priority, capped memory)
   genetic search over building blocks:
     entries: breakout (Donchian), MA cross, trend pullback, momentum, RSI / band mean-reversion,
              squeeze breakout, NR-bar breakout, range expansion, swing-structure break,
              liquidity-sweep reversal, opening-range breakout, time-of-day effect
     filters: trend (EMA), higher-timeframe trend, volatility regime, trending/ranging,
              session window, weekday
     exits:   ATR or swing stop, fixed R target or none, chandelier trail or break-even, time stop
          |
          v
 VALIDATION GAUNTLET (cheapest first; the holdout is last and every look is counted)
   1. in-sample: min 80 trades, profitable in >= 3 of 4 time periods
   2. deflated Sharpe: better than the best result expected from luck given ALL ideas tried so far
   3. parameter plateau: >= 60% of nearby parameter sets also profitable (no knife-edge)
   4. walk-forward: re-pick parameters on the past only, trade the next period
   5. similar markets: must not lose money on sibling instruments
   6. untouched holdout (most recent 30%): >= 30 trades, >= +0.05R, PF >= 1.1, and
      p < 0.05 / (number of looks ever taken at this holdout)
          |
          v
 SHADOW: virtual trades on live prices (>= 20 trades, must match the backtest)
          v
 PROBATION: real demo trades at 0.25% risk (30 trades, must match the backtest)
          v
 ACTIVE (0.5%) -> PROVEN (quarter-Kelly from the live record, never above 2%)
          v
 RETIRED: when live falls significantly below the backtest (overfit), the drawdown exceeds 99% of
          backtest scenarios, or the edge decays
```

Costs in every backtest are pessimistic: entry at the next bar's open, full spread (recorded from
PlexyTrade's own quotes, x1.25), slippage on market orders, gap fills at the open, stop assumed first
when a bar touches both stop and target, swap per day held. The live engine manages trades with the
exact same exit logic as the backtest; a test checks they match trade for trade.

**Self-check:** `gimenez selftest` runs the whole search on synthetic data. On pure noise (no edge
exists) nothing should pass; on data with a planted edge it should find it. Both are also tests.

## Hard rules (enforced in code, tested)

| Rule | Where |
|---|---|
| Demo only. Refuses any non-demo server unless `GIMENEZ_MODE=live` **and** `GIMENEZ_ALLOW_LIVE=YES` | `config.py` |
| Max 2% risk per trade, 6% total open risk | `config.py`, `live/guards.py` |
| Daily loss 6%: flatten, no trading until tomorrow | `live/engine.py` |
| Drawdown 25% from peak: flatten and halt until you run `gimenez resume` | `live/engine.py` |
| Stop loss at the broker on every trade; verified, re-attached or the trade is closed | `broker/tradelocker.py`, `live/engine.py` |
| No size increase after a loss; no adding to positions (no martingale / grid) | `live/guards.py` |
| No self-hedging (opposite positions on one symbol) | `live/guards.py` |
| Abnormal spread, stale data, off-market quote -> no trade | `live/guards.py` |
| Order rate limits; TradeLocker's published API limits halved | `live/guards.py`, `broker/tradelocker.py` |
| Only PlexyTrade's own feed is used for decisions (no external prices: no arbitrage possible) | whole design |

PlexyTrade allows EAs, scalping, hedging and news trading. It forbids arbitrage of any kind, trading
on misquotes or price errors, platform manipulation, Negative Balance Protection abuse and "abusive
strategies". Gimenez trades only patterns in its own price data, with real stops and sane sizes.

## Commands (on the server)

| Command | What it does |
|---|---|
| `gimenez status` | verdict, scorecard, strategies, open trades |
| `gimenez why` | last 24h in plain English: setups found, taken, passed on and why, errors, research |
| `gimenez strategies` | everything it invented and how each is doing, live vs backtest |
| `gimenez stop` / `gimenez resume` | no new real trades / allow again (also lifts a drawdown halt) |
| `gimenez doctor` | checks the TradeLocker connection (places no orders) |
| `gimenez screen` | re-screen all markets now |
| `gimenez selftest` | proves the overfitting filter rejects noise |

Dashboard: `http://SERVER_IP:8001` (any user name + your dashboard password). Scorecard with the
honest verdict, equity and drawdown, strategy league (live vs backtest), every trade's replay
(what it saw, why, expected odds, what happened, post-mortem), every setup including the ones it
passed on, the research funnel, and the market screen.

## Server resources

Gimenez runs three processes: trader (~130 MB), dashboard (~100 MB) and research (peaks ~250 MB,
capped at 450 MB, lowest CPU priority, at most half a CPU). Together ~500 MB. Next to TIIM on a
1 GB droplet that is tight but workable with the 2 GB swap file; **2 GB is recommended**. The setup
script prints memory use at the end so you can decide.

## Development

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # all tests (about a minute)
.venv/bin/python -m gimenez selftest   # honesty check on synthetic data
.venv/bin/python -c "from gimenez.sim import simulate; simulate('data/sim')"   # offline rehearsal
GIMENEZ_DATA_DIR=data/sim DASHBOARD_PASSWORD=x .venv/bin/python -m gimenez dashboard
```
The offline rehearsal runs the whole pipeline on synthetic markets with a fake broker, only to
preview the dashboard. Synthetic results say nothing about real markets.
