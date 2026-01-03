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


# TODO: Add CAGR calculation that handles arbitrary timeframes
# - Accept bar timestamps or duration
# - Compute annualized returns for strategy vs buy-hold
# - Display in scoring breakdown


def compute_composite_score(
    profit_factor: float,
    sortino: float | None,
    pnl: float,
    win_rate: float,
    max_drawdown: float,
    total_trades: int,
    initial_capital: float = 10_000,
    buyhold_return: float | None = None,
) -> tuple[float, list[str]]:
    """Compute composite score with geometric mean, DD penalty, and alpha check.

    Targets (score = 0.5 at these values):
    - Profit Factor: 1.5
    - Sortino: 1.0
    - P&L: $1000
    - Win Rate: 40%
    - Min Trades: 10

    Penalties:
    - Max Drawdown > 20%: exponential decay
    - Alpha < 1.0 (underperforming buy-hold): linear penalty

    Args:
        profit_factor: Strategy profit factor
        sortino: Sortino ratio (can be None or inf)
        pnl: Total P&L in dollars
        win_rate: Win rate as decimal (0-1)
        max_drawdown: Max drawdown as decimal (0-1)
        total_trades: Number of trades
        initial_capital: Starting capital for return calculation
        buyhold_return: Buy-hold return as decimal (e.g., 1.09 for 109%). None to skip.

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

    # Alpha penalty: penalize underperforming buy-hold
    # On mega-trends (buyhold > 20%), soften penalty since absolute returns matter more
    alpha_penalty = 1.0
    if buyhold_return is not None and buyhold_return > 0:
        strategy_return = pl / initial_capital
        alpha = strategy_return / buyhold_return
        if alpha < 1.0:
            # The scoring function uses this to soften the alpha penalty.
            # On mega-trends, it's unfair to penalize a strategy for not matching
            # a 600% buy-hold return - the strategy might still be good (high WR, Sortino)
            # even if absolute returns are lower.
            # Soften penalty on mega-trends: if buyhold > 20%, use sqrt dampening
            # Normal: alpha_penalty = max(0.1, alpha)
            # Mega-trend: alpha_penalty = max(0.3, sqrt(alpha))
            is_megatrend = buyhold_return > 0.20
            if is_megatrend:
                alpha_penalty = max(0.3, math.sqrt(alpha))
            else:
                alpha_penalty = max(0.1, alpha)
        details.append(f"Alpha: {alpha:.2f}x buy-hold → penalty {alpha_penalty:.2f}")

    # Final composite
    composite = geomean * dd_penalty * alpha_penalty
    if buyhold_return is not None:
        details.append(f"Composite: {geomean:.3f} × {dd_penalty:.2f} × {alpha_penalty:.2f} = {composite:.3f}")
    else:
        details.append(f"Composite: {geomean:.3f} × {dd_penalty:.2f} = {composite:.3f}")

    return composite, details
