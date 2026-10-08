"""The statistics that keep Gimenez honest. No scipy needed (keeps memory low)."""
from __future__ import annotations

import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation, |error| < 1.2e-9)."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1)
    if p > 1 - lo:
        return -norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def t_to_z(t: float, df: int) -> float:
    """Student-t statistic -> equivalent normal z (accurate for df >= 5)."""
    if df <= 0:
        return 0.0
    return t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))


def summarize(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "win_rate": 0.0, "pf": 0.0, "t": 0.0, "p": 1.0, "sharpe": 0.0,
                "total": 0.0, "max_dd": 0.0, "skew": 0.0, "kurt": 3.0, "avg_win": 0.0, "avg_loss": 0.0}
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    wins, losses = r[r > 0], r[r <= 0]
    gl = -losses.sum()
    pf = float(wins.sum() / gl) if gl > 0 else (float("inf") if len(wins) else 0.0)
    t = mean / (std / math.sqrt(n)) if std > 0 else 0.0
    z = t_to_z(t, n - 1)
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:] - eq)) if n else 0.0
    if std > 0 and n > 2:
        zz = (r - mean) / r.std(ddof=0)
        skew, kurt = float(np.mean(zz ** 3)), float(np.mean(zz ** 4))
    else:
        skew, kurt = 0.0, 3.0
    return {"n": n, "mean": mean, "std": std, "win_rate": float(len(wins) / n), "pf": pf, "t": t,
            "p": 1 - norm_cdf(z), "sharpe": mean / std if std > 0 else 0.0, "total": float(r.sum()),
            "max_dd": dd, "skew": skew, "kurt": kurt,
            "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0}


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Sharpe the BEST of n_trials worthless strategies is expected to show by luck alone
    (Bailey & Lopez de Prado, 'The Deflated Sharpe Ratio', 2014)."""
    if n_trials < 2 or var_trials <= 0:
        return 0.0
    sd = math.sqrt(var_trials)
    return sd * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n_trials) + EULER_GAMMA * norm_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sharpe: float, n: int, skew: float, kurt: float, n_trials: int, var_trials: float) -> float:
    """Probability that the true Sharpe beats the best-by-luck benchmark, given how many ideas were tried."""
    if n < 3:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, var_trials)
    denom = 1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2
    if denom <= 0:
        denom = 1e-6
    return norm_cdf((sharpe - sr0) * math.sqrt(n - 1) / math.sqrt(denom))


def bootstrap_mean_ci(r: np.ndarray, level: float = 0.95, reps: int = 2000, seed: int = 7) -> tuple[float, float]:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return (-math.inf, math.inf)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(reps, len(r)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def drawdown_quantile(r: np.ndarray, horizon: int, q: float = 0.95, reps: int = 1000, seed: int = 11) -> float:
    """How deep a drawdown (in R) a strategy with these trade results reaches in `horizon` trades,
    in the worst (1-q) of cases. A live drawdown beyond this says the backtest no longer describes it."""
    r = np.asarray(r, dtype=float)
    if len(r) < 5 or horizon < 1:
        return math.inf
    rng = np.random.default_rng(seed)
    sims = rng.choice(r, size=(reps, horizon), replace=True)
    eq = np.cumsum(sims, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((reps, 1)), eq], axis=1), axis=1)[:, 1:]
    return float(np.quantile((peak - eq).max(axis=1), q))


def gap_z(live: np.ndarray, bt_mean: float, bt_std: float) -> float:
    """How many standard errors live results sit below/above what the backtest promised."""
    live = np.asarray(live, dtype=float)
    if len(live) < 2 or bt_std <= 0:
        return 0.0
    return float((live.mean() - bt_mean) / (bt_std / math.sqrt(len(live))))


def kelly_fraction(r: np.ndarray) -> float:
    """Fraction of equity to risk per 1R for maximum growth (approximation for small edges)."""
    r = np.asarray(r, dtype=float)
    if len(r) < 10:
        return 0.0
    m2 = float(np.mean(r ** 2))
    return max(0.0, float(r.mean()) / m2) if m2 > 0 else 0.0
