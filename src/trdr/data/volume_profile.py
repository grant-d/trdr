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


def generate_signal(
    bars: list[Bar],
    position: Position | None,
    atr_threshold: float = 2.0,
    stop_loss_multiplier: float = 1.75,
) -> Signal:
    """Generate trading signal based on POC mean reversion.

    Entry rules (long):
    1. Price outside Value Area by > 2 ATR below VAL
    2. Volume during move is declining (weak conviction)
    3. Price returning toward POC (close > previous close)
    4. Enter when price crosses back into VA

    Exit rules:
    - Target: POC
    - Stop: Beyond LVN (1.75x VA width below entry)

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

    # Calculate indicators
    profile = calculate_volume_profile(bars)
    atr = calculate_atr(bars)
    volume_trend = analyze_volume_trend(bars)

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

    # Check long entry conditions
    distance_below_val = profile.val - current_price
    atr_distance = distance_below_val / atr if atr > 0 else 0

    # Condition 1: Price significantly below VAL
    if atr_distance < atr_threshold:
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason=f"Price not far enough from VA ({atr_distance:.1f}/{atr_threshold} ATR)",
        )

    # Condition 2: Volume declining (weak selling pressure)
    if volume_trend != "declining":
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.2,
            reason=f"Volume trend is {volume_trend}, need declining for mean reversion",
        )

    # Condition 3: Price returning toward POC
    if current_price <= prev_close:
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.3,
            reason="Price not yet returning toward POC",
        )

    # Condition 4: Price crossing back into VA (or close to it)
    if current_price < profile.val - atr:
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.4,
            reason="Waiting for price to approach Value Area",
        )

    # All conditions met - generate BUY signal
    stop_loss = current_price - (va_width * stop_loss_multiplier)
    take_profit = profile.poc

    # Calculate confidence
    confidence = 0.5
    if volume_trend == "declining":
        confidence += 0.15
    if any(abs(current_price - lvn) < atr for lvn in profile.lvns):
        confidence += 0.1  # Near LVN support
    if atr_distance > atr_threshold * 1.5:
        confidence += 0.1  # Extended move

    return Signal(
        action=SignalAction.BUY,
        price=current_price,
        confidence=min(confidence, 1.0),
        reason=f"POC mean reversion: price {atr_distance:.1f} ATR below VA, volume declining",
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
