"""Technical indicator implementations.

All functions accept bars: list[Bar] as first parameter.
Returns float for single values, tuple for multiple.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..data import Bar


class _AggBar(NamedTuple):
    """Aggregated bar for multi-timeframe calculations."""

    high: float
    low: float
    close: float
    volume: float


# =============================================================================
# Moving Averages
# =============================================================================


def sma(bars: list[Bar], period: int) -> float:
    """Calculate Simple Moving Average.

    Args:
        bars: List of OHLCV bars
        period: SMA period

    Returns:
        Current SMA value
    """
    if len(bars) < period:
        return bars[-1].close if bars else 0.0
    closes = [b.close for b in bars[-period:]]
    return float(np.mean(closes))


def ema(bars: list[Bar], period: int) -> float:
    """Calculate Exponential Moving Average (current value).

    Args:
        bars: List of OHLCV bars
        period: EMA period

    Returns:
        Current EMA value
    """
    if len(bars) < period:
        return bars[-1].close if bars else 0.0

    closes = [b.close for b in bars]
    alpha = 2 / (period + 1)
    ema_values = [0.0] * len(closes)

    # Start with SMA for first period
    ema_values[period - 1] = np.mean(closes[:period])

    # Calculate EMA for rest
    for i in range(period, len(closes)):
        ema_values[i] = alpha * closes[i] + (1 - alpha) * ema_values[i - 1]

    return float(ema_values[-1])


def ema_series(values: list[float], period: int) -> list[float]:
    """Calculate EMA series from raw values.

    Args:
        values: List of price values
        period: EMA period

    Returns:
        List of EMA values (0s for insufficient data)
    """
    if len(values) < period:
        return [0.0] * len(values)

    alpha = 2 / (period + 1)
    ema_values = [0.0] * len(values)

    # Start with SMA for first period
    ema_values[period - 1] = np.mean(values[:period])

    # Calculate EMA for rest
    for i in range(period, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]

    return ema_values


def wma(bars: list[Bar], period: int) -> float:
    """Calculate Weighted Moving Average.

    Args:
        bars: List of OHLCV bars
        period: WMA period

    Returns:
        Current WMA value
    """
    if len(bars) < period:
        return bars[-1].close if bars else 0.0

    closes = np.array([b.close for b in bars[-period:]])
    weights = np.arange(1, period + 1)
    return float(np.sum(closes * weights) / np.sum(weights))


def hma(bars: list[Bar], period: int = 9) -> float:
    """Calculate Hull Moving Average for trend confirmation.

    Args:
        bars: List of OHLCV bars
        period: HMA period

    Returns:
        Current HMA value
    """
    if len(bars) < period:
        return 0.0

    closes = np.array([b.close for b in bars[-period:]])
    half_period = period // 2

    # Weighted MA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.sum(closes[-half_period:] * weights_half) / np.sum(weights_half)

    # Weighted MA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.sum(closes * weights_full) / np.sum(weights_full)

    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    ema_input = 2 * wma_half - wma_full

    return float(ema_input)


def hma_slope(bars: list[Bar], period: int = 9, lookback: int = 3) -> float:
    """Calculate HMA slope over lookback period.

    Args:
        bars: List of OHLCV bars
        period: HMA period
        lookback: Number of bars to measure slope

    Returns:
        HMA slope (positive = uptrend, negative = downtrend)
    """
    if len(bars) < period + lookback:
        return 0.0

    hma_current = hma(bars, period)
    hma_prev = hma(bars[:-lookback], period)

    return hma_current - hma_prev


# =============================================================================
# Volatility
# =============================================================================


def atr(bars: list[Bar], period: int = 14) -> float:
    """Calculate Average True Range.

    Args:
        bars: List of OHLCV bars
        period: ATR period

    Returns:
        Current ATR value
    """
    if len(bars) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Wilder's smoothed ATR
    atr_val = np.mean(true_ranges[-period:])
    return float(atr_val)


def bollinger_bands(
    bars: list[Bar], period: int = 20, std_mult: float = 2.0
) -> tuple[float, float, float]:
    """Calculate Bollinger Bands.

    Args:
        bars: List of OHLCV bars
        period: SMA period
        std_mult: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(bars) < period:
        price = bars[-1].close if bars else 0
        return (price, price, price)

    closes = np.array([b.close for b in bars[-period:]])
    middle = float(np.mean(closes))
    std = float(np.std(closes))

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return (upper, middle, lower)


def volatility_regime(bars: list[Bar], lookback: int = 50) -> str:
    """Classify volatility regime using rolling realized volatility.

    Uses three-regime classification: low-vol (mean reversion), med-vol (transition),
    high-vol (momentum).

    Args:
        bars: List of OHLCV bars
        lookback: Period for volatility calculation

    Returns:
        Regime string: "low", "medium", "high"
    """
    if len(bars) < lookback + 1:
        return "medium"

    # Calculate realized volatility over lookback period
    recent_bars = bars[-lookback:]
    close_prices = [b.close for b in recent_bars]

    rv_values = []
    for i in range(1, len(close_prices)):
        ret = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
        rv_values.append(abs(ret))

    if not rv_values:
        return "medium"

    current_rv = np.mean(rv_values[-20:]) if len(rv_values) >= 20 else np.mean(rv_values)
    hist_rv = np.mean(rv_values)
    hist_std = np.std(rv_values) if len(rv_values) > 1 else 0.01

    # Three-regime classification based on historical percentiles
    low_threshold = hist_rv - hist_std
    high_threshold = hist_rv + hist_std

    if current_rv < low_threshold:
        return "low"
    elif current_rv > high_threshold:
        return "high"
    else:
        return "medium"


# =============================================================================
# Momentum
# =============================================================================


def rsi(bars: list[Bar], period: int = 14) -> float:
    """Calculate Relative Strength Index.

    Args:
        bars: List of OHLCV bars
        period: RSI period

    Returns:
        RSI value (0-100)
    """
    if len(bars) < period + 1:
        return 50.0  # Neutral default

    closes = [b.close for b in bars[-(period + 1) :]]
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def macd(
    bars: list[Bar], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        bars: List of OHLCV bars
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    min_bars = slow + signal + 1
    if len(bars) < min_bars:
        return (0.0, 0.0, 0.0)

    closes = [b.close for b in bars]

    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)

    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema_series(macd_line, signal)

    macd_current = macd_line[-1]
    signal_current = signal_line[-1]
    histogram = macd_current - signal_current

    return (macd_current, signal_current, histogram)


def mss(bars: list[Bar], lookback: int = 20) -> float:
    """Calculate Market Structure Score for regime detection.

    Args:
        bars: List of OHLCV bars
        lookback: Period for calculations

    Returns:
        MSS value (-100 to +100). >30 = bullish, <-30 = bearish, -30 to 30 = neutral
    """
    if len(bars) < lookback:
        return 0.0

    recent_bars = bars[-lookback:]
    closes = [b.close for b in recent_bars]

    # Trend: simple linear regression slope
    x = np.arange(lookback)
    y = np.array(closes)
    slope = np.polyfit(x, y, 1)[0]
    trend_pct = (slope / closes[-1] * 100) if closes[-1] != 0 else 0

    # Volatility: ATR as percentage (inverted: higher vol = lower score)
    atr_val = atr(bars, lookback)
    # Normalize by a reasonable ATR level
    volatility_pct = max(0, 80 - (atr_val / closes[-1] * 100 * 3)) if closes[-1] != 0 else 40

    # Exhaustion: price deviation from recent high/low
    recent_high = max(b.high for b in recent_bars)
    recent_low = min(b.low for b in recent_bars)
    recent_range = recent_high - recent_low
    if recent_range > 0:
        exhaustion = ((closes[-1] - recent_low) / recent_range * 100) - 50  # -50 to +50
    else:
        exhaustion = 0

    # Combine with weights favoring trend and exhaustion
    mss_val = (trend_pct * 0.5) + (volatility_pct * 0.2) + (exhaustion * 0.3)
    return float(np.clip(mss_val, -100, 100))


# =============================================================================
# Volume
# =============================================================================


@dataclass(frozen=True)
class VolumeProfile:
    """Calculated volume profile with key levels."""

    poc: float  # Point of Control (highest volume price)
    vah: float  # Value Area High
    val: float  # Value Area Low
    hvns: list[float]  # High Volume Nodes
    lvns: list[float]  # Low Volume Nodes
    price_levels: list[float]  # All price level midpoints
    volumes: list[float]  # Volume at each level
    total_volume: float


def volume_profile(
    bars: list[Bar],
    num_levels: int = 40,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Calculate Volume Profile from OHLCV bars.

    Args:
        bars: List of OHLCV bars
        num_levels: Number of price buckets
        value_area_pct: Percentage for Value Area (default 70%)

    Returns:
        VolumeProfile with PoC, VA, HVNs, LVNs
    """
    if not bars:
        raise ValueError("No bars provided for volume profile calculation")

    # Find price range
    all_highs = [b.high for b in bars]
    all_lows = [b.low for b in bars]
    price_min = min(all_lows)
    price_max = max(all_highs)

    if price_max == price_min:
        # Flat market - return minimal profile
        return VolumeProfile(
            poc=price_min,
            vah=price_min,
            val=price_min,
            hvns=[price_min],
            lvns=[],
            price_levels=[price_min],
            volumes=[sum(b.volume for b in bars)],
            total_volume=sum(b.volume for b in bars),
        )

    bucket_size = (price_max - price_min) / num_levels
    volumes = [0.0] * num_levels
    price_levels = [price_min + (i + 0.5) * bucket_size for i in range(num_levels)]

    # Distribute volume across price levels
    for bar in bars:
        bar_low = bar.low
        bar_high = bar.high
        bar_volume = bar.volume

        # Find buckets this bar spans (clamp to valid range)
        start_bucket = max(0, min(num_levels - 1, int((bar_low - price_min) / bucket_size)))
        end_bucket = max(0, min(num_levels - 1, int((bar_high - price_min) / bucket_size)))

        # Ensure valid bucket range (handles edge cases like bar_high < bar_low)
        if end_bucket < start_bucket:
            start_bucket, end_bucket = end_bucket, start_bucket

        # Distribute volume evenly across spanned buckets
        buckets_spanned = end_bucket - start_bucket + 1
        volume_per_bucket = bar_volume / buckets_spanned

        for bucket in range(start_bucket, end_bucket + 1):
            volumes[bucket] += volume_per_bucket

    total_volume = sum(volumes)

    # Find PoC (bucket with maximum volume)
    poc_bucket = int(np.argmax(volumes))
    poc = price_levels[poc_bucket]

    # Calculate Value Area (expand from PoC until 70% captured)
    va_volume = volumes[poc_bucket]
    va_buckets = {poc_bucket}
    va_target = total_volume * value_area_pct

    above_idx = poc_bucket + 1
    below_idx = poc_bucket - 1

    while va_volume < va_target and (above_idx < num_levels or below_idx >= 0):
        above_vol = volumes[above_idx] if above_idx < num_levels else 0
        below_vol = volumes[below_idx] if below_idx >= 0 else 0

        if above_vol >= below_vol and above_idx < num_levels:
            va_buckets.add(above_idx)
            va_volume += above_vol
            above_idx += 1
        elif below_idx >= 0:
            va_buckets.add(below_idx)
            va_volume += below_vol
            below_idx -= 1
        else:
            break

    vah = price_levels[max(va_buckets)]
    val = price_levels[min(va_buckets)]

    # Identify HVNs and LVNs
    avg_volume = total_volume / num_levels
    hvn_threshold = 1.5 * avg_volume
    lvn_threshold = 0.5 * avg_volume

    hvns = [price_levels[i] for i, v in enumerate(volumes) if v > hvn_threshold]
    lvns = [price_levels[i] for i, v in enumerate(volumes) if 0 < v < lvn_threshold]

    return VolumeProfile(
        poc=poc,
        vah=vah,
        val=val,
        hvns=hvns,
        lvns=lvns,
        price_levels=price_levels,
        volumes=volumes,
        total_volume=total_volume,
    )


def volume_trend(bars: list[Bar], lookback: int = 5) -> str:
    """Analyze recent volume trend.

    Args:
        bars: List of OHLCV bars
        lookback: Number of recent bars to analyze

    Returns:
        "increasing", "declining", or "neutral"
    """
    if len(bars) < lookback:
        return "neutral"

    recent_volumes = [b.volume for b in bars[-lookback:]]
    earlier_volumes = [b.volume for b in bars[-lookback * 2 : -lookback]]

    if not earlier_volumes:
        return "neutral"

    recent_avg = np.mean(recent_volumes)
    earlier_avg = np.mean(earlier_volumes)

    if recent_avg > earlier_avg * 1.2:
        return "increasing"
    elif recent_avg < earlier_avg * 0.8:
        return "declining"
    return "neutral"


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


def multi_timeframe_poc(bars: list[Bar]) -> tuple[float, float, float]:
    """Calculate PoC at multiple aggregation levels.

    Simulates different timeframes by aggregating bars:
    - TF1: Current bars (native resolution)
    - TF2: 4-bar aggregation (4x longer timeframe)
    - TF3: 12-bar aggregation (12x longer timeframe)

    Returns:
        Tuple of (poc_tf1, poc_tf2, poc_tf3)
    """
    if len(bars) < 12:
        current_poc = volume_profile(bars[-min(20, len(bars)) :]).poc
        return current_poc, current_poc, current_poc

    # TF1: Current resolution
    poc_tf1 = volume_profile(bars[-20:]).poc

    # TF2: 4-bar aggregation (aggregate last 20 bars into 5 "super bars")
    agg4_bars = []
    for i in range(0, min(20, len(bars)), 4):
        chunk = bars[-20 + i : -20 + i + 4]
        if chunk:
            agg4_bars.append(
                _AggBar(
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                )
            )

    poc_tf2 = volume_profile(agg4_bars).poc if agg4_bars else poc_tf1

    # TF3: 12-bar aggregation
    agg12_bars = []
    for i in range(0, min(20, len(bars)), 12):
        chunk = bars[-20 + i : -20 + i + 12]
        if chunk:
            agg12_bars.append(
                _AggBar(
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                )
            )

    poc_tf3 = volume_profile(agg12_bars).poc if agg12_bars else poc_tf1

    return float(poc_tf1), float(poc_tf2), float(poc_tf3)


def hvn_support_strength(bars: list[Bar], val_level: float, lookback: int = 30) -> float:
    """Detect strength of support at HVN levels using historical touches.

    Counts how many times price bounced from or consolidated at VAL level,
    indicating accumulated liquidity and institutional interest.

    Args:
        bars: List of OHLCV bars
        val_level: VAL price level to analyze
        lookback: Historical bars to check

    Returns:
        Support strength score (0.0 to 1.0)
    """
    if len(bars) < lookback:
        return 0.0

    recent_bars = bars[-lookback:]
    touches = 0
    bounces = 0

    for i in range(1, len(recent_bars)):
        low = recent_bars[i].low
        high = recent_bars[i].high

        # Bar touched the VAL level
        if low <= val_level <= high:
            touches += 1
            # Bar bounced from VAL (closed above after touching)
            if recent_bars[i].close > val_level:
                bounces += 1

    if touches == 0:
        return 0.0

    # Support strength = historical bounce rate at this level
    bounce_rate = bounces / touches if touches > 0 else 0
    # Also factor in frequency of touches (more touches = more tested level)
    touch_frequency = touches / lookback

    return float(min(bounce_rate * 0.6 + touch_frequency * 0.4, 1.0))


# =============================================================================
# Pattern Recognition
# =============================================================================


def sax_pattern(bars: list[Bar], window: int = 20, segments: int = 5) -> str:
    """Convert price series to SAX symbolic pattern.

    Args:
        bars: List of OHLCV bars
        window: Lookback window for pattern
        segments: Number of PAA segments (alphabet size)

    Returns:
        SAX pattern string (e.g., "aabcd")
    """
    if len(bars) < window:
        return ""

    closes = np.array([b.close for b in bars[-window:]])

    # Z-normalize
    mean = np.mean(closes)
    std = np.std(closes)
    if std == 0:
        return "ccccc"  # Flat market
    normalized = (closes - mean) / std
    normalized = np.clip(normalized, -3, 3)  # Clip outliers

    # PAA: Piecewise Aggregate Approximation
    segment_size = window // segments
    paa = []
    for i in range(segments):
        start = i * segment_size
        end = start + segment_size
        paa.append(np.mean(normalized[start:end]))

    # Discretize to alphabet {a, b, c, d, e} using Gaussian breakpoints
    # Breakpoints for 5 symbols: -0.84, -0.25, 0.25, 0.84
    breakpoints = [-0.84, -0.25, 0.25, 0.84]
    alphabet = "abcde"
    pattern = ""
    for val in paa:
        idx = 0
        for bp in breakpoints:
            if val > bp:
                idx += 1
        pattern += alphabet[idx]

    return pattern


def sax_bullish_reversal(pattern: str) -> bool:
    """Detect bullish reversal patterns in SAX string.

    Balanced detection: identifies reversal patterns with quality filtering.

    Args:
        pattern: SAX pattern string

    Returns:
        True if bullish reversal pattern detected
    """
    if len(pattern) < 4:
        return False

    first_half = pattern[: len(pattern) // 2]
    second_half = pattern[len(pattern) // 2 :]

    # First half: bearish (mostly a/b)
    first_bearish = sum(1 for c in first_half if c in "ab") >= len(first_half) // 2

    # Second half: bullish (has d/e) and ends high
    second_bullish = sum(1 for c in second_half if c in "de") >= 1
    ends_high = pattern[-1] in "de"

    # Upward momentum in second half
    momentum = all(pattern[i] >= pattern[i - 1] for i in range(len(pattern) // 2, len(pattern)))

    return first_bearish and second_bullish and ends_high and momentum


def heikin_ashi(bars: list[Bar]) -> list[dict]:
    """Transform bars into Heikin-Ashi representation.

    HA smooths price action by using:
    - HA Close = (O + H + L + C) / 4 (average of all prices)
    - HA Open = (prior HA Open + prior HA Close) / 2
    - HA High = max(H, HA Open, HA Close)
    - HA Low = min(L, HA Open, HA Close)

    Args:
        bars: List of original OHLCV bars

    Returns:
        List of dicts with Heikin-Ashi values
    """
    if len(bars) < 1:
        return []

    ha_bars = []

    for i, bar in enumerate(bars):
        # HA Close is average of all prices
        ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0

        # HA Open: average of prior HA bar's open/close
        if i == 0:
            ha_open = (bar.open + bar.close) / 2.0
        else:
            prev_bar = ha_bars[-1]
            ha_open = (prev_bar["open"] + prev_bar["close"]) / 2.0

        # HA High/Low: covers all prices
        ha_high = max(bar.high, ha_open, ha_close)
        ha_low = min(bar.low, ha_open, ha_close)

        ha_bar = {
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "volume": bar.volume,
            "timestamp": getattr(bar, "timestamp", None),
        }
        ha_bars.append(ha_bar)

    return ha_bars
