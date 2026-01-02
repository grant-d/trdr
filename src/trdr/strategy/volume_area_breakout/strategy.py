"""VolumeAreaBreakout strategy implementation."""

from dataclasses import dataclass

import numpy as np

from ...data.market import Bar
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import Position, Signal, SignalAction, VolumeProfile


@dataclass
class VolumeAreaBreakoutConfig(StrategyConfig):
    """Configuration for VolumeAreaBreakout strategy.

    Args:
        symbol: Trading symbol (e.g., "crypto:BTC/USD", "stock:AAPL")
        timeframe: Bar timeframe (e.g., "1h", "4h", "1d")
        atr_threshold: ATR multiplier for entry threshold
        stop_loss_multiplier: Multiplier for stop loss distance
    """

    atr_threshold: float = 2.0
    stop_loss_multiplier: float = 1.75


# -----------------------------------------------------------------------------
# Indicator Functions
# -----------------------------------------------------------------------------


def calculate_atr(bars: list[Bar], period: int = 14) -> float:
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
    atr = np.mean(true_ranges[-period:])
    return float(atr)


def calculate_volume_profile(
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


def analyze_volume_trend(bars: list[Bar], lookback: int = 5) -> str:
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


def calculate_mss(bars: list[Bar], lookback: int = 20) -> float:
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
    atr = calculate_atr(bars, lookback)
    # Normalize by a reasonable ATR level
    volatility_pct = max(0, 80 - (atr / closes[-1] * 100 * 3)) if closes[-1] != 0 else 40

    # Exhaustion: price deviation from recent high/low
    recent_high = max(b.high for b in recent_bars)
    recent_low = min(b.low for b in recent_bars)
    recent_range = recent_high - recent_low
    if recent_range > 0:
        exhaustion = ((closes[-1] - recent_low) / recent_range * 100) - 50  # -50 to +50
    else:
        exhaustion = 0

    # Combine with weights favoring trend and exhaustion
    mss = (trend_pct * 0.5) + (volatility_pct * 0.2) + (exhaustion * 0.3)
    return float(np.clip(mss, -100, 100))


def calculate_hma(bars: list[Bar], period: int = 9) -> float:
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
    sqrt_period = int(np.sqrt(period))
    ema_input = 2 * wma_half - wma_full

    # Simple approximation: use last value
    return float(ema_input)


def compute_sax_pattern(bars: list[Bar], window: int = 20, segments: int = 5) -> str:
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


def detect_sax_bullish_reversal(pattern: str) -> bool:
    """Detect bullish reversal patterns in SAX string.

    Balanced detection: identifies reversal patterns with quality filtering via confidence threshold.

    Args:
        pattern: SAX pattern string

    Returns:
        True if bullish reversal pattern detected
    """
    if len(pattern) < 4:
        return False

    first_half = pattern[:len(pattern) // 2]
    second_half = pattern[len(pattern) // 2:]

    # First half: bearish (mostly a/b)
    first_bearish = sum(1 for c in first_half if c in "ab") >= len(first_half) // 2

    # Second half: bullish (has d/e) and ends high
    second_bullish = sum(1 for c in second_half if c in "de") >= 1
    ends_high = pattern[-1] in "de"

    # Upward momentum in second half
    momentum = all(pattern[i] >= pattern[i - 1] for i in range(len(pattern) // 2, len(pattern)))

    return first_bearish and second_bullish and ends_high and momentum


def classify_volatility_regime(bars: list[Bar], lookback: int = 50) -> str:
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

    # Calculate 5-bar rolling returns (simulating 5-minute bars on hourly data)
    returns = []
    for i in range(len(bars) - 4, len(bars)):
        if i >= 1:
            ret = (bars[i].close - bars[i - 1].close) / bars[i - 1].close
            returns.append(ret)

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


def compute_heikin_ashi(bars: list[Bar]) -> list:
    """Transform bars into Heikin-Ashi representation.

    HA smooths price action by using:
    - HA Close = (O + H + L + C) / 4 (average of all prices)
    - HA Open = (prior HA Open + prior HA Close) / 2
    - HA High = max(H, HA Open, HA Close)
    - HA Low = min(L, HA Open, HA Close)

    This removes false wicks and reduces noise in breakouts/bounces.

    Args:
        bars: List of original OHLCV bars

    Returns:
        List of synthetic bars with Heikin-Ashi values
    """
    if len(bars) < 1:
        return bars

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


def calculate_multi_timeframe_poc(bars: list[Bar]) -> tuple[float, float, float]:
    """Calculate PoC at multiple aggregation levels.

    Simulates different timeframes by aggregating bars:
    - TF1: Current bars (native resolution)
    - TF2: 4-bar aggregation (4x longer timeframe)
    - TF3: 12-bar aggregation (12x longer timeframe)

    Returns:
        Tuple of (poc_tf1, poc_tf2, poc_tf3)
    """
    if len(bars) < 12:
        current_poc = calculate_volume_profile(bars[-min(20, len(bars)) :]).poc
        return current_poc, current_poc, current_poc

    # TF1: Current resolution
    poc_tf1 = calculate_volume_profile(bars[-20:]).poc

    # TF2: 4-bar aggregation (aggregate last 20 bars into 5 "super bars")
    agg4_bars = []
    for i in range(0, min(20, len(bars)), 4):
        chunk = bars[-20 + i : -20 + i + 4]
        if chunk:
            high = max(b.high for b in chunk)
            low = min(b.low for b in chunk)
            close = chunk[-1].close
            volume = sum(b.volume for b in chunk)
            # Create synthetic bar for profile calculation
            agg_bar = type("Bar", (), {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })()
            agg4_bars.append(agg_bar)

    poc_tf2 = calculate_volume_profile(agg4_bars).poc if agg4_bars else poc_tf1

    # TF3: 12-bar aggregation
    agg12_bars = []
    for i in range(0, min(20, len(bars)), 12):
        chunk = bars[-20 + i : -20 + i + 12]
        if chunk:
            high = max(b.high for b in chunk)
            low = min(b.low for b in chunk)
            close = chunk[-1].close
            volume = sum(b.volume for b in chunk)
            agg_bar = type("Bar", (), {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })()
            agg12_bars.append(agg_bar)

    poc_tf3 = calculate_volume_profile(agg12_bars).poc if agg12_bars else poc_tf1

    return float(poc_tf1), float(poc_tf2), float(poc_tf3)


def detect_hvn_support_strength(bars: list[Bar], val_level: float, lookback: int = 30) -> float:
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
        prev_close = recent_bars[i - 1].close

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


def compute_order_flow_imbalance(bars: list[Bar], lookback: int = 5) -> float:
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


# -----------------------------------------------------------------------------
# Strategy Class
# -----------------------------------------------------------------------------


class VolumeAreaBreakoutStrategy(BaseStrategy):
    """VAH breakout + VAL bounce strategy with POC target.

    Entry paths:
    1. VAH Breakout: Price breaks above VAH with volume (bullish regime)
    2. VAL Bounce: Price bounces from VAL with any regime tolerance

    Exit rules:
    - Target: POC level
    - Stop: 1.2x ATR for bounce, 0.4x ATR below VAL for breakout
    """

    def __init__(self, config: VolumeAreaBreakoutConfig):
        """Initialize strategy.

        Args:
            config: Strategy configuration with symbol and parameters
        """
        super().__init__(config)
        self.config: VolumeAreaBreakoutConfig = config

    @property
    def name(self) -> str:
        return "VolumeAreaBreakout"

    def generate_signal(
        self,
        bars: list[Bar],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal based on Volume Profile analysis.

        Args:
            bars: Historical bars (oldest first)
            position: Current position or None

        Returns:
            Trading signal with action, stops, and targets
        """
        if len(bars) < 20:
            return Signal(
                action=SignalAction.HOLD,
                price=bars[-1].close if bars else 0,
                confidence=0.0,
                reason="Insufficient data for analysis",
            )

        # Minimal bar requirement - only need basics for volume profile
        if len(bars) < 50:
            return Signal(
                action=SignalAction.HOLD,
                price=bars[-1].close,
                confidence=0.0,
                reason="Insufficient data for analysis",
            )

        # Calculate indicators (using original bars for robustness)
        profile = calculate_volume_profile(bars)
        atr = calculate_atr(bars)
        mss = calculate_mss(bars)

        current_bar = bars[-1]
        current_price = current_bar.close
        prev_close = bars[-2].close if len(bars) > 1 else current_price

        # Check exit conditions first if we have a position
        if position and position.side == "long":
            # Check stop loss
            if current_price <= position.stop_loss:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=1.0,
                    reason=f"Stop loss hit at {position.stop_loss:.2f}",
                )

            # Check take profit (PoC reached)
            if current_price >= profile.poc:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=0.9,
                    reason=f"PoC target reached at {profile.poc:.2f}",
                )

            # Hold position
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.5,
                reason="Holding position, awaiting target or stop",
            )

        # Entry logic (no position)
        if position and position.side != "none":
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason="Position already open",
            )

        # Regime Filter: Allow neutral to bullish
        if mss < -25:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"Bearish regime (MSS={mss:.0f})",
            )

        # Calculate volume metrics
        recent_volumes = [b.volume for b in bars[-20:]]
        avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1
        volume_ratio = current_bar.volume / avg_recent_volume if avg_recent_volume > 0 else 0
        volume_trend = analyze_volume_trend(bars)

        # Calculate historical support strength at VAL level
        hvn_strength = detect_hvn_support_strength(bars, profile.val, lookback=30)

        # Calculate multi-timeframe PoC confluence
        poc_tf1, poc_tf2, poc_tf3 = calculate_multi_timeframe_poc(bars)
        # Multi-TF confluence: higher TF POCs cluster together = stronger support
        poc_cluster_width = abs(poc_tf2 - poc_tf3)
        poc_clustered = poc_cluster_width < (atr * 0.5)  # POCs within 0.5 ATR = strong confluence

        # Path 1: VAH Breakout (moderate: requires strong volume AND bullish/neutral regime)
        above_vah = current_price > profile.vah
        above_vah_prev = bars[-2].close <= profile.vah if len(bars) > 1 else False
        volume_strong = volume_ratio >= 1.2  # Relax from 1.3 to 1.2 (120%+ volume)
        regime_bullish_plus = mss > -5  # Bullish or neutral

        vah_breakout = above_vah and above_vah_prev and volume_strong and regime_bullish_plus

        # Path 2: VAL Bounce (primary entry: price bounces from below VAL)
        below_val = current_price < profile.val
        below_val_prev = bars[-2].close >= profile.val if len(bars) > 1 else False
        # Accept all volume trends for VAL bounce (volume declining is bonus not requirement)
        regime_ok_val = mss > -35  # Further relaxed to allow weak bearish regime bounces

        val_bounce = below_val and below_val_prev and regime_ok_val

        # Accept VAH breakout OR VAL bounce
        entry_signal = vah_breakout or val_bounce

        if not entry_signal:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"No signal: vah={vah_breakout} val={val_bounce}",
            )

        # Calculate stops and targets based on which signal triggered
        if vah_breakout:
            # Breakout target: PoC level
            take_profit = profile.poc
            # Stop: below VAL with buffer
            stop_loss = profile.val - atr * 0.4
            signal_type = "VAH_breakout"
            confidence_base = 0.65
        else:  # val_bounce
            # Bounce target: PoC (mean reversion)
            take_profit = profile.poc
            # Stop: 1.2 ATR below entry
            stop_loss = current_price - atr * 1.2
            signal_type = "VAL_bounce"
            confidence_base = 0.70

        confidence = confidence_base

        # Volume bonus
        if vah_breakout and volume_ratio > 1.5:
            confidence += 0.1

        # Strong declining volume bonus for bounces (critical signal)
        if val_bounce and volume_trend == "declining":
            confidence += 0.25  # Very strong bonus - most reliable setup

        # HVN strength bonus for historically validated support on VAL bounces
        if val_bounce and hvn_strength > 0.70:
            confidence += 0.12  # Extra confidence on well-tested support levels

        # Regime bonus
        if mss > 5:
            confidence += 0.12

        # In VA bonus (better context for entry)
        if profile.val <= current_price <= profile.vah:
            confidence += 0.08

        confidence = min(confidence, 1.0)

        # Standard threshold: balance quality and opportunity
        min_confidence_threshold = 0.75

        if confidence < min_confidence_threshold:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=confidence,
                reason=f"Low confidence {confidence:.2f} < {min_confidence_threshold} threshold",
            )

        return Signal(
            action=SignalAction.BUY,
            price=current_price,
            confidence=confidence,
            reason=f"Entry: price={current_price:.2f}, target={take_profit:.2f}",
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_ratio=1.0,
        )
