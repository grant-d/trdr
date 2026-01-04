"""Composite scoring for SICA benchmarks.

Weighted scoring system prioritizing cash generation (CAGR, Calmar)
with risk-adjusted quality metrics (Sortino, PF, WR).

Scoring curves (see constants below for current values):

| Metric | Curve      | Constant              | Why                              |
| ------ | ---------- | --------------------- | -------------------------------- |
| CAGR   | Asymptotic | TARGET_CAGR           | Higher return always better      |
| Calmar | Asymptotic | TARGET_CALMAR         | Higher always better             |
| Sortino| Asymptotic | TARGET_SORTINO        | Higher always better             |
| PF     | Asymptotic | TARGET_PF             | Higher always better             |
| WR     | Asymptotic | TARGET_WR             | Higher generally better          |
| Trades | Quadratic  | TARGET_TRADES_PER_YEAR| Sweet spot; extremes are bad     |

Asymptotic: score = value / (value + target)
  - At target: 0.5, approaches 1.0 asymptotically
  - Use when "more is always better"

Quadratic: score = 1 - ((value - target) / delta)²
  - At target: 1.0, zero at target ± delta
  - Use when optimal range exists

References:
- https://www.quantifiedstrategies.com/trading-performance/
- https://www.dakotaridgecapital.com/fearless-investor/portfolio-risk-ratios-sharpe-sortino-calmar
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trdr.backtest import PaperExchangeResult
    from trdr.data.market import Bar

# === SCORING TARGETS ===
# Asymptotic targets: score = 0.5 at target value
TARGET_CAGR = 0.25  # 25% annual return
TARGET_CALMAR = 2.0  # CAGR / Max Drawdown
TARGET_SORTINO = 2.0  # Downside risk-adjusted return
TARGET_PF = 2.0  # Profit Factor
TARGET_WR = 0.51  # Win Rate

# Quadratic target: score = 1.0 at target, 0.0 at target ± delta
TARGET_TRADES_PER_YEAR = 70  # 252 stock trading days/yr
TRADES_DELTA = 95  # Zero score at 5/yr and 195/yr

# Weights (must sum to 1.0)
WEIGHT_CAGR = 0.30
WEIGHT_CALMAR = 0.30
WEIGHT_SORTINO = 0.15
WEIGHT_PF = 0.15
WEIGHT_WR = 0.05
WEIGHT_TRADES = 0.05

# Penalty thresholds
DD_PENALTY_THRESHOLD = 0.25  # Drawdown penalty starts here
DD_DECAY_RATE = 5  # Exponential decay rate
ALPHA_MEGATREND_THRESHOLD = 0.25  # Soften alpha penalty above this
ALPHA_FLOOR_NORMAL = 0.3  # Min alpha penalty in normal markets
ALPHA_FLOOR_MEGATREND = 0.5  # Min alpha penalty in mega-trends


def asymptotic(value: float, target: float) -> float:
    """Map [0, ∞) → [0, 1). At target, score = 0.5.

    More is always better, with diminishing returns.

    Args:
        value: Raw metric value (must be >= 0)
        target: Target value where score = 0.5

    Returns:
        Score between 0 and 1 (approaches 1 asymptotically)
    """
    if value <= 0:
        return 0.0
    return value / (value + target)


def quadratic(value: float, target: float, delta: float) -> float:
    """Parabola peaking at target, zero at target ± delta.

    Use when there's an optimal range and both extremes are bad.

    Args:
        value: Raw metric value
        target: Optimal value where score = 1.0
        delta: Distance from target where score = 0

    Returns:
        Score between 0 and 1, peaks at target

    Examples:
        quadratic(100, 100, 100) → 1.0  (at target)
        quadratic(50, 100, 100)  → 0.75 (halfway)
        quadratic(0, 100, 100)   → 0.0  (at target - delta)
        quadratic(200, 100, 100) → 0.0  (at target + delta)
    """
    return max(0.0, 1 - ((value - target) / delta) ** 2)


def compute_composite_score(
    profit_factor: float,
    sortino: float | None,
    pnl: float,
    win_rate: float,
    max_drawdown: float,
    total_trades: int,
    initial_capital: float = 10_000,
    buyhold_return: float | None = None,
    timeframe: str | None = None,
    bars: "list[Bar] | None" = None,
) -> tuple[float, list[str]]:
    """Compute weighted composite score prioritizing cash generation.

    Scoring weights:
    - Primary (60%): CAGR (30%), Calmar (30%) - cash generation
    - Secondary (40%): Sortino (15%), PF (15%), WR (10%) - quality

    Targets (score = 0.5 at these values):
    - CAGR: 25% annually
    - Calmar: 2.0 (excellent risk-adjusted return vs drawdown)
    - Sortino: 2.0 (excellent downside risk-adjusted return)
    - Profit Factor: 2.0
    - Win Rate: 50%

    Penalties:
    - Max Drawdown > 25%: exponential decay
    - Trades < 20: linear scale down (statistical significance)
    - Alpha < 1.0 (underperforming buy-hold): linear penalty

    Args:
        profit_factor: Strategy profit factor
        sortino: Sortino ratio (can be None or inf)
        pnl: Total P&L in dollars
        win_rate: Win rate as decimal (0-1)
        max_drawdown: Max drawdown as decimal (0-1)
        total_trades: Number of trades executed
        initial_capital: Starting capital for return calculation
        buyhold_return: Buy-hold return as decimal (e.g., 0.25 for 25%)
        timeframe: Timeframe string (e.g., "15m", "4h", "1d")
        bars: List of Bar objects for period calculation

    Returns:
        Tuple of (score 0-1, list of metric descriptions)
    """
    details = []

    # Calculate period info, CAGR, and Calmar
    total_return = pnl / initial_capital if initial_capital > 0 else 0
    days_span, years, cagr, calmar = 0, 0, 0.0, 0.0

    if bars and len(bars) >= 2 and timeframe:
        num_bars = len(bars)
        start_ts = datetime.fromisoformat(bars[0].timestamp.replace("Z", "+00:00"))
        end_ts = datetime.fromisoformat(bars[-1].timestamp.replace("Z", "+00:00"))
        days_span = (end_ts - start_ts).days
        years = days_span / 365.25

        # CAGR = (1 + total_return)^(1/years) - 1
        if years > 0:
            cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
            calmar = cagr / max(0.001, max_drawdown) if cagr > 0 else 0

        details.append(f"Period: {timeframe} x {num_bars} bars ({days_span} days)")

    # Clamp metrics
    pf = 10.0 if profit_factor == float("inf") else max(0, profit_factor)
    s = 0.0 if (sortino is None or sortino == float("inf")) else max(0, sortino)
    wr = max(0, min(1, win_rate))
    dd = max(0, min(1, max_drawdown))

    # === PRIMARY METRICS (60%) - Cash Generation ===
    cagr_score = asymptotic(max(0, cagr), TARGET_CAGR)
    calmar_score = asymptotic(calmar, TARGET_CALMAR)

    details.append(
        f"CAGR: {cagr:.1%} → {cagr_score:.2f} ({WEIGHT_CAGR:.0%}) [target: {TARGET_CAGR:.0%} for 0.50]"
    )
    details.append(
        f"Calmar: {calmar:.2f} → {calmar_score:.2f} ({WEIGHT_CALMAR:.0%}) [target: {TARGET_CALMAR} for 0.50]"
    )

    # === SECONDARY METRICS (40%) - Quality/Risk ===
    sortino_score = asymptotic(s, TARGET_SORTINO)
    pf_score = asymptotic(pf, TARGET_PF)
    wr_score = asymptotic(wr, TARGET_WR)

    # Trades/year - quadratic curve penalizing both too few and too many
    trades_per_year = total_trades / years if years > 0 else total_trades
    trades_score = quadratic(trades_per_year, TARGET_TRADES_PER_YEAR, TRADES_DELTA)
    trades_min = TARGET_TRADES_PER_YEAR - TRADES_DELTA
    trades_max = TARGET_TRADES_PER_YEAR + TRADES_DELTA

    details.append(
        f"Sortino: {s:.2f} → {sortino_score:.2f} ({WEIGHT_SORTINO:.0%}) [target: {TARGET_SORTINO} for 0.50]"
    )
    details.append(
        f"PF: {pf:.2f} → {pf_score:.2f} ({WEIGHT_PF:.0%}) [target: {TARGET_PF} for 0.50]"
    )
    details.append(
        f"WR: {wr:.1%} → {wr_score:.2f} ({WEIGHT_WR:.0%}) [target: {TARGET_WR:.0%} for 0.50]"
    )
    details.append(
        f"Trades: {total_trades} ({trades_per_year:.0f}/yr) → {trades_score:.2f} ({WEIGHT_TRADES:.0%}) [optimal: {TARGET_TRADES_PER_YEAR}/yr, zero at {trades_min} or {trades_max}]"
    )

    # Weighted sum
    weighted_score = (
        WEIGHT_CAGR * cagr_score
        + WEIGHT_CALMAR * calmar_score
        + WEIGHT_SORTINO * sortino_score
        + WEIGHT_PF * pf_score
        + WEIGHT_WR * wr_score
        + WEIGHT_TRADES * trades_score
    )

    # === PENALTIES ===

    # Drawdown penalty: exponential decay past threshold
    dd_penalty = (
        1.0
        if dd <= DD_PENALTY_THRESHOLD
        else math.exp(-DD_DECAY_RATE * (dd - DD_PENALTY_THRESHOLD))
    )
    details.append(
        f"DD: {dd:.1%} → penalty {dd_penalty:.2f} [threshold: {DD_PENALTY_THRESHOLD:.0%}]"
    )

    # Alpha penalty: penalize underperforming buy-hold
    alpha_penalty = 1.0
    if buyhold_return is not None and buyhold_return > 0:
        alpha = total_return / buyhold_return
        if alpha < 1.0:
            # Soften on mega-trends
            floor = (
                ALPHA_FLOOR_MEGATREND
                if buyhold_return > ALPHA_MEGATREND_THRESHOLD
                else ALPHA_FLOOR_NORMAL
            )
            # Only sqrt if alpha is positive (strategy made money)
            if alpha > 0:
                alpha_penalty = max(
                    floor, math.sqrt(alpha) if buyhold_return > ALPHA_MEGATREND_THRESHOLD else alpha
                )
            else:
                alpha_penalty = floor  # Strategy lost money, use floor penalty
        details.append(f"Alpha: {alpha:.2f}x buy-hold → penalty {alpha_penalty:.2f}")

    # Final composite
    composite = weighted_score * dd_penalty * alpha_penalty
    penalty_str = f"{dd_penalty:.2f}" + (
        f" × {alpha_penalty:.2f}" if buyhold_return and buyhold_return > 0 else ""
    )
    details.append(f"Score: {weighted_score:.3f} × {penalty_str} = {composite:.3f}")

    return composite, details


def score_result(
    result: "PaperExchangeResult",
    buyhold_return: float | None = None,
) -> tuple[float, list[str]]:
    """Compute composite score from PaperExchangeResult.

    Args:
        result: Backtest result with metrics
        buyhold_return: Buy-hold return for alpha comparison (optional)

    Returns:
        Tuple of (score 0-1, list of metric descriptions)
    """
    details = []

    # Extract and clamp metrics
    cagr = result.cagr or 0.0
    calmar = result.calmar_ratio or 0.0
    sortino = (
        0.0
        if (result.sortino_ratio is None or result.sortino_ratio == float("inf"))
        else max(0, result.sortino_ratio)
    )
    pf = 10.0 if result.profit_factor == float("inf") else max(0, result.profit_factor)
    wr = max(0, min(1, result.win_rate))
    dd = max(0, min(1, result.max_drawdown))

    # Period info
    if result.start_time and result.end_time:
        start = datetime.fromisoformat(result.start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(result.end_time.replace("Z", "+00:00"))
        days_span = (end - start).days
        details.append(
            f"Period: {days_span} days, {result.total_trades} trades ({result.trades_per_year:.0f}/yr)"
        )

    # Score components
    cagr_score = asymptotic(max(0, cagr), TARGET_CAGR)
    calmar_score = asymptotic(calmar, TARGET_CALMAR)
    sortino_score = asymptotic(sortino, TARGET_SORTINO)
    pf_score = asymptotic(pf, TARGET_PF)
    wr_score = asymptotic(wr, TARGET_WR)
    trades_score = quadratic(result.trades_per_year, TARGET_TRADES_PER_YEAR, TRADES_DELTA)

    trades_min = TARGET_TRADES_PER_YEAR - TRADES_DELTA
    trades_max = TARGET_TRADES_PER_YEAR + TRADES_DELTA

    details.append(
        f"CAGR: {cagr:.1%} → {cagr_score:.2f} ({WEIGHT_CAGR:.0%}) [target: {TARGET_CAGR:.0%} for 0.50]"
    )
    details.append(
        f"Calmar: {calmar:.2f} → {calmar_score:.2f} ({WEIGHT_CALMAR:.0%}) [target: {TARGET_CALMAR} for 0.50]"
    )
    details.append(
        f"Sortino: {sortino:.2f} → {sortino_score:.2f} ({WEIGHT_SORTINO:.0%}) [target: {TARGET_SORTINO} for 0.50]"
    )
    details.append(
        f"PF: {pf:.2f} → {pf_score:.2f} ({WEIGHT_PF:.0%}) [target: {TARGET_PF} for 0.50]"
    )
    details.append(
        f"WR: {wr:.1%} → {wr_score:.2f} ({WEIGHT_WR:.0%}) [target: {TARGET_WR:.0%} for 0.50]"
    )
    details.append(
        f"Trades: {result.trades_per_year:.0f}/yr → {trades_score:.2f} ({WEIGHT_TRADES:.0%}) [target: {trades_min}-{trades_max}]"
    )

    # Weighted sum
    weighted_score = (
        WEIGHT_CAGR * cagr_score
        + WEIGHT_CALMAR * calmar_score
        + WEIGHT_SORTINO * sortino_score
        + WEIGHT_PF * pf_score
        + WEIGHT_WR * wr_score
        + WEIGHT_TRADES * trades_score
    )

    # Drawdown penalty
    dd_penalty = (
        1.0
        if dd <= DD_PENALTY_THRESHOLD
        else math.exp(-DD_DECAY_RATE * (dd - DD_PENALTY_THRESHOLD))
    )
    details.append(
        f"DD: {dd:.1%} → penalty {dd_penalty:.2f} [threshold: {DD_PENALTY_THRESHOLD:.0%}]"
    )

    # Alpha penalty
    alpha_penalty = 1.0
    if buyhold_return is not None and buyhold_return > 0:
        alpha = result.total_return / buyhold_return
        if alpha < 1.0:
            floor = (
                ALPHA_FLOOR_MEGATREND
                if buyhold_return > ALPHA_MEGATREND_THRESHOLD
                else ALPHA_FLOOR_NORMAL
            )
            # Only sqrt if alpha is positive (strategy made money)
            if alpha > 0:
                alpha_penalty = max(
                    floor, math.sqrt(alpha) if buyhold_return > ALPHA_MEGATREND_THRESHOLD else alpha
                )
            else:
                alpha_penalty = floor  # Strategy lost money, use floor penalty
        details.append(f"Alpha: {alpha:.2f}x → penalty {alpha_penalty:.2f}")

    # Final composite
    composite = weighted_score * dd_penalty * alpha_penalty
    penalty_str = f"{dd_penalty:.2f}" + (
        f" × {alpha_penalty:.2f}" if buyhold_return and buyhold_return > 0 else ""
    )
    details.append(f"Score: {weighted_score:.3f} × {penalty_str} = {composite:.3f}")

    return composite, details
