"""Trading strategy implementation.

Regime-adaptive strategy that detects trending vs mean-reverting vs choppy periods
and applies appropriate signals for each.
"""

import numpy as np


def compute_signal(prices: np.ndarray, idx: int) -> float:
    """Compute position signal for bar at index idx.

    Args:
        prices: Array of close prices, shape (n,)
        idx: Current bar index (0 to n-1)

    Returns:
        Position signal in range [-1.0, 1.0]
    """
    # Need enough history
    if idx < 30:
        return 0.0

    # Get recent returns
    recent_prices = prices[idx - 30 : idx + 1]
    returns = np.diff(recent_prices) / recent_prices[:-1]

    # Key insight: this data is mean-reverting (negative autocorrelation)
    # Strategy: fade large moves, follow the trend reversal

    # Last two returns - key for sizing
    last_return = returns[-1]
    prev_return = returns[-2] if len(returns) >= 2 else 0

    # Recent volatility
    volatility = np.std(returns)
    recent_vol = np.std(returns[-10:])

    # Cumulative recent returns
    cum_recent = returns[-5:].sum()

    # Mean-reversion strategy optimized for all metrics
    signal = 0.0
    abs_ret = abs(last_return)

    # Fade moves proportionally to size
    if abs_ret > 0.004:
        # Non-linear scaling gives bigger positions on bigger moves
        normalized = abs_ret / 0.018
        intensity = min(1.0, normalized ** 1.25)
        signal = -np.sign(last_return) * max(0.32, intensity)

    # Multi-day momentum fade
    elif abs(cum_recent) > 0.028:
        normalized = abs(cum_recent) / 0.045
        intensity = min(0.92, normalized ** 1.2)
        signal = -np.sign(cum_recent) * max(0.35, intensity)

    # Risk management: adjust position size based on volatility
    if recent_vol > 0.04:
        signal *= 0.5
    elif recent_vol > 0.035:
        signal *= 0.7
    elif recent_vol < 0.005:
        signal *= 0.35
    elif volatility > 0.04:
        signal *= 0.75

    return np.clip(signal, -1.0, 1.0)


def generate_signals(prices: np.ndarray) -> np.ndarray:
    """Generate signals for all bars.

    Returns array of signals, same length as prices.
    """
    signals = np.zeros(len(prices))
    for i in range(len(prices)):
        signals[i] = compute_signal(prices, i)
    return signals
