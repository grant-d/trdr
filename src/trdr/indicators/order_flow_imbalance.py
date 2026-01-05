"""Order Flow Imbalance indicator for buy/sell pressure analysis."""

from ..data import Bar


def order_flow_imbalance(bars: list[Bar], lookback: int = 5) -> float:
    """Compute Order Flow Imbalance (OFI) based on volume direction.

    Simplified OFI: tracks buy vs sell volume based on price direction.
    Positive OFI = more buying pressure, Negative = more selling pressure.

    Args:
        bars: List of OHLCV bars
        lookback: Period to analyze

    Returns:
        OFI score (-1.0 to 1.0)
    """
    if len(bars) < lookback + 1:
        return 0.0

    recent_bars = bars[-lookback:]
    buy_volume = 0.0
    sell_volume = 0.0

    for i in range(len(recent_bars)):
        if i == 0:
            continue
        # If price went up, volume on that bar = buy volume
        # If price went down, volume on that bar = sell volume
        if recent_bars[i].close > recent_bars[i - 1].close:
            buy_volume += recent_bars[i].volume
        elif recent_bars[i].close < recent_bars[i - 1].close:
            sell_volume += recent_bars[i].volume
        else:
            # Split neutral bars
            buy_volume += recent_bars[i].volume * 0.5
            sell_volume += recent_bars[i].volume * 0.5

    total_volume = buy_volume + sell_volume
    if total_volume == 0:
        return 0.0

    # OFI score: buy_volume / total - 0.5 (0 = balanced, 0.5 = all buys, -0.5 = all sells)
    ofi = (buy_volume / total_volume) - 0.5
    return float(ofi)
