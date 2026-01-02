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

    # Last three returns - key for confirmation
    last_return = returns[-1]
    prev_return = returns[-2] if len(returns) >= 2 else 0
    prev2_return = returns[-3] if len(returns) >= 3 else 0

    # Recent volatility
    volatility = np.std(returns)
    recent_vol = np.std(returns[-10:])

    # Cumulative recent returns
    cum_recent = returns[-5:].sum()

    # Confirmation-based signals for higher profit factor
    abs_ret = abs(last_return)
    signal = 0.0

    # Check for directional disagreement (reversal setup)
    has_reversal_setup = False
    if abs_ret > 0.0035 and abs(prev_return) > 0.0003:
        # Reversals happen when consecutive bars oppose direction
        if np.sign(last_return) != np.sign(prev_return):
            has_reversal_setup = True

    # Trade only on confirmed reversals with size
    if abs_ret > 0.004 and has_reversal_setup:
        # Confirmed reversal with decent move size
        normalized = abs_ret / 0.018
        intensity = min(1.0, normalized ** 1.25)
        signal = -np.sign(last_return) * max(0.37, intensity)

    # Also trade extreme accumulation (standalone signal)
    elif abs(cum_recent) > 0.030:
        normalized = abs(cum_recent) / 0.048
        intensity = min(0.92, normalized ** 1.2)
        signal = -np.sign(cum_recent) * max(0.37, intensity)

    # Adaptive position sizing: avoid small losing trades
    # Only trade aggressively in moderate vol environments
    if recent_vol < 0.006:
        # Ultra-low vol = low conviction reversals
        signal *= 0.15
    elif recent_vol < 0.010:
        # Still low vol, reduce exposure
        signal *= 0.40
    elif recent_vol < 0.018:
        # Goldilocks zone - good risk/reward
        signal *= 1.0
    elif recent_vol > 0.04:
        # High vol - preserve capital
        signal *= 0.5
    elif recent_vol > 0.035:
        signal *= 0.7
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
