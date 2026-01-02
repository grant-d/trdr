"""Composite scoring for SICA benchmarks.

Uses asymptotic scaling for smooth gradients. Geometric mean ensures
all metrics must improve together.
"""

import math


def asymptotic(value: float, target: float) -> float:
    """Map [0, ∞) → [0, 1). At target, score = 0.5.

    Args:
        value: Raw metric value (must be >= 0)
        target: Target value where score = 0.5

    Returns:
        Score between 0 and 1 (approaches 1 asymptotically)
    """
    if value <= 0:
        return 0.0
    return value / (value + target)


def compute_composite_score(
    profit_factor: float,
    sortino: float | None,
    pnl: float,
    win_rate: float,
    max_drawdown: float,
    total_trades: int,
) -> tuple[float, list[str]]:
    """Compute composite score with geometric mean and DD penalty.

    Targets (score = 0.5 at these values):
    - Profit Factor: 1.5
    - Sortino: 1.0
    - P&L: $1000
    - Win Rate: 40%
    - Min Trades: 10

    Returns:
        Tuple of (score 0-1, list of metric descriptions)
    """
    details = []

    # Handle edge cases
    pf = max(0, profit_factor) if profit_factor != float("inf") else 10.0
    s = max(0, sortino) if sortino and sortino != float("inf") else 0.0
    wr = max(0, min(1, win_rate))
    pl = max(0, pnl)
    dd = max(0, min(1, max_drawdown))

    # Individual asymptotic scores
    pf_score = asymptotic(pf, 1.5)
    sortino_score = asymptotic(s, 1.0)
    pnl_score = asymptotic(pl, 1000)
    wr_score = asymptotic(wr, 0.40)
    trades_score = asymptotic(total_trades, 10)

    details.append(f"PF: {pf:.2f} → {pf_score:.2f}")
    details.append(f"Sortino: {s:.2f} → {sortino_score:.2f}")
    details.append(f"P&L: ${pl:.0f} → {pnl_score:.2f}")
    details.append(f"WR: {wr:.1%} → {wr_score:.2f}")
    details.append(f"Trades: {total_trades} → {trades_score:.2f}")

    # Geometric mean of all scores (all must improve together)
    scores = [pf_score, sortino_score, pnl_score, wr_score, trades_score]
    # Avoid log(0) by adding small epsilon
    scores = [max(0.001, s) for s in scores]
    geomean = math.exp(sum(math.log(s) for s in scores) / len(scores))

    # Drawdown penalty: exponential decay past 20%
    if dd <= 0.20:
        dd_penalty = 1.0
    else:
        # Decay factor: at 25% DD → 0.61, at 30% DD → 0.37
        dd_penalty = math.exp(-10 * (dd - 0.20))
    details.append(f"DD: {dd:.1%} → penalty {dd_penalty:.2f}")

    # Final composite
    composite = geomean * dd_penalty
    details.append(f"Composite: {geomean:.3f} × {dd_penalty:.2f} = {composite:.3f}")

    return composite, details
