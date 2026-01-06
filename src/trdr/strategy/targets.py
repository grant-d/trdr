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

Quadratic: score = 1 - ((value - target) / target)²
  - At target: 1.0, zero at 0 or 2*target
  - Use when optimal range exists (trades: 0 to 140, sweet spot 70)

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

# === SCORING TARGETS ===
# Asymptotic targets: score = 0.5 at target value
TARGET_CAGR = 0.25  # 25% annual return
TARGET_CALMAR = 2.0  # CAGR / Max Drawdown
TARGET_SORTINO = 2.0  # Downside risk-adjusted return
TARGET_PF = 2.0  # Profit Factor
TARGET_WR = 0.51  # Win Rate

# Quadratic target: score = 1.0 at target, 0.0 at 0 or 2*target
TARGET_TRADES_PER_YEAR = 70  # Sweet spot; range is 0 to 140

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
            f"Period: {days_span} days, {result.total_trades} trades "
            f"({result.trades_per_year:.0f}/yr)"
        )

    # Score components
    cagr_score = asymptotic(max(0, cagr), TARGET_CAGR)
    calmar_score = asymptotic(calmar, TARGET_CALMAR)
    sortino_score = asymptotic(sortino, TARGET_SORTINO)
    pf_score = asymptotic(pf, TARGET_PF)
    wr_score = asymptotic(wr, TARGET_WR)
    trades_score = quadratic(result.trades_per_year, TARGET_TRADES_PER_YEAR, TARGET_TRADES_PER_YEAR)

    details.append(
        f"CAGR: {cagr:.1%} → {cagr_score:.2f} ({WEIGHT_CAGR:.0%}) "
        f"[target: {TARGET_CAGR:.0%} for 0.50]"
    )
    details.append(
        f"Calmar: {calmar:.2f} → {calmar_score:.2f} ({WEIGHT_CALMAR:.0%}) "
        f"[target: {TARGET_CALMAR} for 0.50]"
    )
    details.append(
        f"Sortino: {sortino:.2f} → {sortino_score:.2f} ({WEIGHT_SORTINO:.0%}) "
        f"[target: {TARGET_SORTINO} for 0.50]"
    )
    details.append(
        f"PF: {pf:.2f} → {pf_score:.2f} ({WEIGHT_PF:.0%}) [target: {TARGET_PF} for 0.50]"
    )
    details.append(
        f"WR: {wr:.1%} → {wr_score:.2f} ({WEIGHT_WR:.0%}) [target: {TARGET_WR:.0%} for 0.50]"
    )
    details.append(
        f"Trades: {result.trades_per_year:.0f}/yr → {trades_score:.2f} "
        f"({WEIGHT_TRADES:.0%}) [target: {TARGET_TRADES_PER_YEAR}/yr]"
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
    if buyhold_return is not None and buyhold_return != 0:
        excess_return = result.total_return - buyhold_return
        if buyhold_return > 0:
            # Positive market: use ratio (alpha = strategy / buyhold)
            alpha = result.total_return / buyhold_return
            if alpha < 1.0:
                floor = (
                    ALPHA_FLOOR_MEGATREND
                    if buyhold_return > ALPHA_MEGATREND_THRESHOLD
                    else ALPHA_FLOOR_NORMAL
                )
                if alpha > 0:
                    alpha_penalty = max(
                        floor,
                        math.sqrt(alpha) if buyhold_return > ALPHA_MEGATREND_THRESHOLD else alpha,
                    )
                else:
                    alpha_penalty = floor
            details.append(f"Alpha: {alpha:.2f}x buy-hold → penalty {alpha_penalty:.2f}")
        else:
            # Negative market: show excess return (ratio is meaningless)
            details.append(
                f"Alpha: +{excess_return:.1%} vs buy-hold ({buyhold_return:.1%}) → no penalty"
            )

    # Final composite
    composite = weighted_score * dd_penalty * alpha_penalty
    penalty_str = f"{dd_penalty:.2f}" + (f" × {alpha_penalty:.2f}" if alpha_penalty != 1.0 else "")
    details.append(f"Score: {weighted_score:.3f} × {penalty_str} = {composite:.3f}")

    return composite, details


def score_from_objectives(
    objectives: dict[str, float],
    period_days: float | None = None,
    buyhold_return: float | None = None,
) -> tuple[float, list[str]]:
    """Compute composite score from objective metrics.

    Args:
        objectives: Objective dict from MooResult.get_objectives_dict()
        period_days: Total backtest span in days
        buyhold_return: Buy-hold return for alpha penalty (optional)

    Returns:
        Tuple of (score 0-1, list of metric descriptions)
    """
    details = []

    cagr = objectives.get("cagr") or 0.0
    calmar = objectives.get("calmar") or 0.0
    sortino = objectives.get("sortino") or 0.0
    pf = objectives.get("profit_factor") or 0.0
    wr = objectives.get("win_rate") or 0.0
    dd = objectives.get("max_drawdown") or 0.0
    total_trades = int(objectives.get("total_trades") or 0)
    alpha = objectives.get("alpha")

    if sortino == float("inf"):
        sortino = 0.0
    if pf == float("inf"):
        pf = 10.0

    wr = max(0, min(1, wr))
    dd = max(0, min(1, dd))

    years = (period_days or 0.0) / 365.0
    trades_per_year = total_trades / years if years > 0 else 0.0

    if period_days:
        details.append(
            f"Period: {int(period_days)} days, {total_trades} trades "
            f"({trades_per_year:.0f}/yr)"
        )

    cagr_score = asymptotic(max(0, cagr), TARGET_CAGR)
    calmar_score = asymptotic(calmar, TARGET_CALMAR)
    sortino_score = asymptotic(sortino, TARGET_SORTINO)
    pf_score = asymptotic(pf, TARGET_PF)
    wr_score = asymptotic(wr, TARGET_WR)
    trades_score = quadratic(trades_per_year, TARGET_TRADES_PER_YEAR, TARGET_TRADES_PER_YEAR)

    weighted_score = (
        WEIGHT_CAGR * cagr_score
        + WEIGHT_CALMAR * calmar_score
        + WEIGHT_SORTINO * sortino_score
        + WEIGHT_PF * pf_score
        + WEIGHT_WR * wr_score
        + WEIGHT_TRADES * trades_score
    )

    dd_penalty = (
        1.0
        if dd <= DD_PENALTY_THRESHOLD
        else math.exp(-DD_DECAY_RATE * (dd - DD_PENALTY_THRESHOLD))
    )

    alpha_penalty = 1.0
    if buyhold_return is not None and buyhold_return != 0 and alpha is not None:
        if buyhold_return > 0:
            if alpha < 1.0:
                floor = (
                    ALPHA_FLOOR_MEGATREND
                    if buyhold_return > ALPHA_MEGATREND_THRESHOLD
                    else ALPHA_FLOOR_NORMAL
                )
                if alpha > 0:
                    alpha_penalty = max(
                        floor,
                        math.sqrt(alpha) if buyhold_return > ALPHA_MEGATREND_THRESHOLD else alpha,
                    )
                else:
                    alpha_penalty = floor

    composite = weighted_score * dd_penalty * alpha_penalty
    return composite, details
