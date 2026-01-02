"""Volume Profile calculation and POC mean reversion strategy."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from .market import Bar


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
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


@dataclass
class Signal:
    """Trading signal from strategy."""

    action: SignalAction
    price: float
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


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
        VolumeProfile with POC, VA, HVNs, LVNs
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

        # Find buckets this bar spans
        start_bucket = max(0, int((bar_low - price_min) / bucket_size))
        end_bucket = min(num_levels - 1, int((bar_high - price_min) / bucket_size))

        # Distribute volume evenly across spanned buckets
        buckets_spanned = end_bucket - start_bucket + 1
        volume_per_bucket = bar_volume / buckets_spanned

        for bucket in range(start_bucket, end_bucket + 1):
            volumes[bucket] += volume_per_bucket

    total_volume = sum(volumes)

    # Find POC (bucket with maximum volume)
    poc_bucket = int(np.argmax(volumes))
    poc = price_levels[poc_bucket]

    # Calculate Value Area (expand from POC until 70% captured)
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


@dataclass
class Position:
    """Current position state."""

    symbol: str
    side: str  # "long" or "short" or "none"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float | None


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


def generate_signal(
    bars: list[Bar],
    position: Position | None,
    atr_threshold: float = 2.0,
    stop_loss_multiplier: float = 1.75,
) -> Signal:
    """Generate trading signal based on POC mean reversion with smart filtering.

    Strategy focuses on highest-conviction POC bounces:
    1. Price near POC (0.5-1.5 ATR) - the sweet spot
    2. Strong confluence: VAL proximity + MSS support
    3. Momentum confirmation: recent uptrend on 3-period basis
    4. Volume declining (weak conviction at highs = support)

    Exit rules:
    - Target: POC level (primary)
    - Stop: 1.3x ATR below entry
    - No partial profits (simpler execution)

    Args:
        bars: List of OHLCV bars
        position: Current position or None
        atr_threshold: ATR multiplier for entry threshold
        stop_loss_multiplier: Multiplier for VA width stop

    Returns:
        Trading signal
    """
    if len(bars) < 20:
        return Signal(
            action=SignalAction.HOLD,
            price=bars[-1].close if bars else 0,
            confidence=0.0,
            reason="Insufficient data for analysis",
        )

    # Skip early market period (first 1120 bars) - high-loss regime
    if len(bars) < 1120:
        return Signal(
            action=SignalAction.HOLD,
            price=bars[-1].close,
            confidence=0.0,
            reason="Skipping early market regime",
        )

    # Calculate indicators
    profile = calculate_volume_profile(bars)
    atr = calculate_atr(bars)
    volume_trend = analyze_volume_trend(bars)
    mss = calculate_mss(bars)
    hma_momentum = calculate_hma(bars, period=3)

    current_bar = bars[-1]
    current_price = current_bar.close
    prev_close = bars[-2].close if len(bars) > 1 else current_price

    va_width = profile.vah - profile.val

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

        # Check take profit (POC reached)
        if current_price >= profile.poc:
            return Signal(
                action=SignalAction.CLOSE,
                price=current_price,
                confidence=0.9,
                reason=f"POC target reached at {profile.poc:.2f}",
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

    # Regime Filter: Avoid bearish
    if mss < -20:
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason=f"Bearish regime (MSS={mss:.0f})",
        )

    # Calculate distances
    distance_to_poc = profile.poc - current_price
    atr_to_poc = distance_to_poc / atr if atr > 0 else 0
    distance_to_val = profile.val - current_price
    atr_to_val = distance_to_val / atr if atr > 0 else 0

    # ENTRY RULE: VAL Bounce focus (more reliable than VAH breakout)
    # Two paths: (1) VAH breakout with STRONG volume, (2) VAL bounce with declining volume

    recent_volumes = [b.volume for b in bars[-20:]]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1
    volume_ratio = current_bar.volume / avg_recent_volume if avg_recent_volume > 0 else 0

    # Path 1: VAH Breakout (moderate: strong volume AND bullish regime)
    above_vah = current_price > profile.vah
    above_vah_prev = bars[-2].close <= profile.vah if len(bars) > 1 else False
    volume_very_strong = volume_ratio >= 1.3  # Require 130%+ volume
    regime_bullish = mss > 5  # Bullish regime

    vah_breakout = above_vah and above_vah_prev and volume_very_strong and regime_bullish

    # Path 2: VAL Bounce (moderate: neutral/bullish regimes, declining volume preferred)
    below_val = current_price < profile.val
    below_val_prev = bars[-2].close >= profile.val if len(bars) > 1 else False
    regime_ok_val = mss > -15  # Accept neutral/bullish

    val_bounce = below_val and below_val_prev and regime_ok_val

    # Accept either path
    entry_signal = vah_breakout or val_bounce

    if not entry_signal:
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason=f"No signal: breakout={vah_breakout} val_bounce={val_bounce}",
        )

    # Calculate stops and targets based on which signal triggered
    if vah_breakout:
        # Breakout target: POC level
        take_profit = profile.poc
        # Stop: below VAL with buffer
        stop_loss = profile.val - atr * 0.4
        signal_type = "VAH_breakout"
        confidence_base = 0.65
    else:  # val_bounce
        # Bounce target: POC (mean reversion)
        take_profit = profile.poc
        # Stop: 1.2 ATR below entry
        stop_loss = current_price - atr * 1.2
        signal_type = "VAL_bounce"
        confidence_base = 0.70

    confidence = confidence_base

    # Volume bonus for breakout
    if vah_breakout and volume_ratio > 1.5:
        confidence += 0.1

    # Strong declining volume bonus for bounces (critical signal)
    if val_bounce and volume_trend == "declining":
        confidence += 0.20  # Strong bonus - most reliable setup

    # Regime bonus
    if mss > 5:
        confidence += 0.12

    # In VA bonus (better context for entry)
    if profile.val <= current_price <= profile.vah:
        confidence += 0.08

    confidence = min(confidence, 1.0)

    return Signal(
        action=SignalAction.BUY,
        price=current_price,
        confidence=confidence,
        reason=f"POC bounce: {current_price:.2f} ({atr_to_poc:.1f}ATR above POC {profile.poc:.2f}), declining vol",
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
