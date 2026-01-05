"""Williams Vix Fix volatility indicator for market bottoms."""

import numpy as np

from ..data import Bar


def williams_vix_fix(
    bars: list[Bar],
    pd: int = 22,
    bbl: int = 20,
    mult: float = 2.0,
    lb: int = 50,
    ph: float = 0.85,
) -> tuple[float, str]:
    """Calculate Williams Vix Fix volatility indicator.

    Developed by Larry Williams, gives VIX-like readings for any asset class.
    Measures how far current low is from recent highest close, indicating
    potential market bottoms and high volatility periods.

    Original author: Larry Williams
    Credit: CM_Williams_Vix_Fix by ChrisMoody on TradingView

    Args:
        bars: List of OHLCV bars
        pd: Lookback period for standard deviation high (highest close)
        bbl: Bollinger Band length
        mult: Bollinger Band std deviation multiplier
        lb: Lookback period for percentile calculations
        ph: Highest percentile threshold (0.85 = 85th percentile)

    Returns:
        Tuple of (wvf_value, alert_state)
        - wvf_value: Williams Vix Fix value (0-100+)
        - alert_state: "high" if above threshold, "normal" otherwise
    """
    min_bars = max(pd, bbl, lb)
    if len(bars) < min_bars:
        return (0.0, "normal")

    # Calculate WVF: ((highest(close, pd) - low) / highest(close, pd)) * 100
    recent_bars = bars[-pd:]
    closes = [b.close for b in recent_bars]
    highest_close = max(closes)
    current_low = bars[-1].low

    if highest_close == 0:
        return (0.0, "normal")

    wvf = ((highest_close - current_low) / highest_close) * 100.0

    # Calculate Bollinger Bands around WVF
    # Need historical WVF values
    wvf_history = []
    for i in range(len(bars) - lb, len(bars)):
        if i < pd:
            continue
        recent = bars[max(0, i - pd + 1) : i + 1]
        closes_i = [b.close for b in recent]
        highest_i = max(closes_i) if closes_i else 0
        low_i = bars[i].low
        if highest_i > 0:
            wvf_i = ((highest_i - low_i) / highest_i) * 100.0
            wvf_history.append(wvf_i)

    if len(wvf_history) < bbl:
        return (float(wvf), "normal")

    # Bollinger Bands
    wvf_recent = wvf_history[-bbl:]
    mid_line = float(np.mean(wvf_recent))
    std_dev = float(np.std(wvf_recent))
    upper_band = mid_line + (mult * std_dev)

    # Percentile range high
    range_high = max(wvf_history) * ph

    # Determine alert state
    if wvf >= upper_band or wvf >= range_high:
        alert_state = "high"
    else:
        alert_state = "normal"

    return (float(wvf), alert_state)
