"""Every number that controls money or credibility lives here.

Values marked HARD are PlexyTrade/owner rules: the code refuses settings that loosen them.
Most can be overridden with environment variables in .env (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent

DEMO_ENVIRONMENT = "https://demo.tradelocker.com"


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


# --- HARD limits (owner's non-negotiable rules) -------------------------------------
MAX_RISK_PER_TRADE = 0.02      # 2% of equity, absolute ceiling
MAX_TOTAL_OPEN_RISK = 0.06     # 6% of equity across all open trades
DAILY_LOSS_STOP = 0.06         # -6% on the day -> flatten and stop until tomorrow
DRAWDOWN_HALT = 0.25           # -25% from peak -> flatten and halt until `gimenez resume`


@dataclass
class RiskConfig:
    max_risk_per_trade: float = MAX_RISK_PER_TRADE
    max_total_open_risk: float = MAX_TOTAL_OPEN_RISK
    daily_loss_stop: float = DAILY_LOSS_STOP
    drawdown_halt: float = DRAWDOWN_HALT
    # Risk per trade by how much a strategy has proven. Unproven ideas risk very little.
    risk_probation: float = 0.0025   # first real trades of a strategy fresh out of shadow testing
    risk_active: float = 0.005       # positive live record, not yet statistically proven
    risk_proven_floor: float = 0.0075
    kelly_fraction: float = 0.25     # proven strategies: quarter-Kelly from the live record, capped at 2%
    # Risk shrinks linearly as the account drawdown grows (never grows after losses).
    drawdown_scale_start: float = 0.05
    min_risk: float = 0.001
    max_positions: int = 6
    max_positions_per_symbol: int = 1
    day_start_utc_hour: int = 0

    def validate(self) -> None:
        if not 0 < self.max_risk_per_trade <= MAX_RISK_PER_TRADE:
            raise ValueError(f"max_risk_per_trade must be in (0, {MAX_RISK_PER_TRADE:.0%}]")
        if not 0 < self.max_total_open_risk <= MAX_TOTAL_OPEN_RISK:
            raise ValueError(f"max_total_open_risk must be in (0, {MAX_TOTAL_OPEN_RISK:.0%}]")
        if not 0 < self.daily_loss_stop <= DAILY_LOSS_STOP:
            raise ValueError(f"daily_loss_stop must be in (0, {DAILY_LOSS_STOP:.0%}]")
        if not 0 < self.drawdown_halt <= DRAWDOWN_HALT:
            raise ValueError(f"drawdown_halt must be in (0, {DRAWDOWN_HALT:.0%}]")
        for name in ("risk_probation", "risk_active", "risk_proven_floor"):
            if getattr(self, name) > self.max_risk_per_trade:
                raise ValueError(f"{name} above max_risk_per_trade")


@dataclass
class ComplianceConfig:
    """PlexyTrade terms: no arbitrage, no trading on misquotes/price errors, no manipulation,
    no NBP abuse. Guards: abnormal spread, stale/off-market quote, order spam, self-hedging,
    martingale/grid, size increase after a loss."""
    max_spread_vs_median: float = 3.0       # spread > 3x its usual level -> abnormal, don't trade
    max_spread_vs_stop: float = 0.25        # spread eats > 25% of the risk -> not worth it
    max_quote_jump_atr: float = 3.0         # quote > 3 ATR from last close -> possible misquote
    max_bar_age_factor: float = 3.0         # last closed bar older than 3 bar lengths -> stale feed
    min_stop_atr: float = 0.5               # stops tighter than 0.5 ATR look like latency scalping
    max_orders_per_minute: int = 4
    max_orders_per_hour: int = 30
    min_seconds_between_symbol_orders: int = 60


@dataclass
class ResearchConfig:
    """Strategy discovery and the anti-overfitting gates."""
    timeframes: list[str] = field(default_factory=lambda: [t.strip() for t in os.getenv(
        "RESEARCH_TIMEFRAMES", "5m,15m,1H").split(",") if t.strip()])
    population: int = 40
    generations: int = 12
    elite: int = 4
    mutation_rate: float = 0.35
    holdout_fraction: float = 0.30      # most recent 30% of history is never seen by the search
    is_folds: int = 4                   # in-sample split into 4 time folds for consistency
    min_trades_is: int = 80
    min_trades_fold: int = 10
    min_trades_holdout: int = 30
    max_holdout_candidates_per_run: int = 3   # only the best few ever see the holdout
    alpha: float = 0.05                 # holdout significance, Bonferroni-split across every look
    min_dsr: float = 0.5                # deflated Sharpe: better than the best expected from luck
    min_plateau: float = 0.6            # >=60% of nearby parameter sets must also be profitable
    min_walkforward_efficiency: float = 0.3
    min_peer_expectancy: float = 0.0    # same idea on sibling markets must not lose on average
    min_holdout_expectancy: float = 0.05
    min_holdout_pf: float = 1.1
    max_bars_in_trade: int = 300
    # cost model (on top of the recorded spread)
    spread_multiplier: float = 1.25     # assume spreads are 25% wider than the typical recorded one
    slippage_atr: float = 0.02          # market-order fills slip by 2% of ATR
    swap_per_day: float = 0.0002        # holding cost per day as a fraction of price
    max_bars_per_dataset: int = 60000
    cpu_seconds_per_cycle: int = _env_int("RESEARCH_CPU_SECONDS", 900)
    sleep_between_cycles: int = _env_int("RESEARCH_SLEEP_SECONDS", 300)


@dataclass
class PipelineConfig:
    """Promotion: idea -> backtest -> holdout -> shadow -> probation -> active -> proven -> retired."""
    max_shadow: int = 30
    max_live: int = 8
    shadow_min_trades: int = 20
    shadow_max_trades: int = 60
    probation_trades: int = 30
    proven_min_trades: int = 100
    gap_z_retire: float = -2.5          # live significantly worse than backtest -> overfit, retire
    decay_window: int = 30
    regime_pause_min_trades: int = 15
    regime_pause_expectancy: float = -0.2


@dataclass
class ScreenConfig:
    watchlist_size: int = _env_int("WATCHLIST_SIZE", 12)
    max_per_class: int = 4
    rescreen_hours: int = 24 * 7
    max_cost_ratio: float = 0.15        # spread must be < 15% of a typical 1H ATR
    min_hours_open_per_day: float = 6.0


@dataclass
class Settings:
    mode: str = field(default_factory=lambda: os.getenv("GIMENEZ_MODE", "demo").lower())
    allow_live: str = field(default_factory=lambda: os.getenv("GIMENEZ_ALLOW_LIVE", "NO"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("GIMENEZ_DATA_DIR", ROOT / "data")))
    tl_environment: str = field(default_factory=lambda: os.getenv("TL_ENVIRONMENT", DEMO_ENVIRONMENT).rstrip("/"))
    tl_email: str = field(default_factory=lambda: os.getenv("TL_EMAIL", ""))
    tl_password: str = field(default_factory=lambda: os.getenv("TL_PASSWORD", ""))
    tl_server: str = field(default_factory=lambda: os.getenv("TL_SERVER", "PLEXY"))
    tl_acc_num: int = field(default_factory=lambda: _env_int("TL_ACC_NUM", 0))
    poll_seconds: int = field(default_factory=lambda: _env_int("POLL_SECONDS", 15))
    # TradeLocker publishes its rate limits; we stay well below them on top of that.
    requests_per_second: float = field(default_factory=lambda: _env_float("TL_REQUESTS_PER_SECOND", 1.0))
    risk: RiskConfig = field(default_factory=RiskConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "gimenez.db"

    @property
    def bars_path(self) -> Path:
        return self.data_dir / "bars.db"

    @property
    def stop_file(self) -> Path:
        return self.data_dir / "STOP"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def validate(self) -> None:
        """Refuse anything that could touch real money unless the owner explicitly enabled it."""
        self.risk.validate()
        if self.mode not in {"demo", "live", "backtest"}:
            raise ValueError(f"GIMENEZ_MODE must be demo or live, got {self.mode!r}")
        env = self.tl_environment.lower()
        if self.is_live:
            if self.allow_live != "YES":
                raise RuntimeError("Live trading is disabled. It needs GIMENEZ_MODE=live AND GIMENEZ_ALLOW_LIVE=YES.")
            if "demo" in env:
                raise RuntimeError("GIMENEZ_MODE=live but TL_ENVIRONMENT points at the demo server.")
        elif self.mode == "demo" and env != DEMO_ENVIRONMENT:
            raise RuntimeError(f"Demo mode only connects to {DEMO_ENVIRONMENT}; TL_ENVIRONMENT is {self.tl_environment}. "
                               "Refusing to start against a non-demo server.")
