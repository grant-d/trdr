"""Volume Profile indicator for market structure analysis."""

from dataclasses import dataclass

import numpy as np

from ..data import Bar


@dataclass
class VolumeProfile:
    """Volume Profile data structure."""

    poc: float  # Point of Control (highest volume price)
    vah: float  # Value Area High (top of 70% volume)
    val: float  # Value Area Low (bottom of 70% volume)
    hvns: list[float]  # High Volume Nodes (1.5x avg volume)
    lvns: list[float]  # Low Volume Nodes (<0.5x avg volume)
    price_levels: list[float]  # All price level midpoints
    volumes: list[float]  # Volume at each level
    total_volume: float


def _volume_profile_calculate(
    bars: list[Bar],
    num_levels: int = 40,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    if not bars:
        raise ValueError("No bars provided for volume profile calculation")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive for volume profile calculation")

    # Find price range and total volume in a single pass.
    price_min = bars[0].low
    price_max = bars[0].high
    total_bar_volume = 0.0
    for bar in bars:
        if bar.low < price_min:
            price_min = bar.low
        if bar.high > price_max:
            price_max = bar.high
        total_bar_volume += bar.volume

    if price_max == price_min:
        # Flat market - return minimal profile
        return VolumeProfile(
            poc=price_min,
            vah=price_min,
            val=price_min,
            hvns=[price_min],
            lvns=[],
            price_levels=[price_min],
            volumes=[total_bar_volume],
            total_volume=total_bar_volume,
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


class VolumeProfileIndicator:
    """Streaming Volume Profile calculator."""

    def __init__(self, num_levels: int = 40, value_area_pct: float = 0.70) -> None:
        self.num_levels = num_levels
        self.value_area_pct = value_area_pct
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar], num_levels: int = 40, value_area_pct: float = 0.70
    ) -> VolumeProfile:
        return _volume_profile_calculate(bars, num_levels=num_levels, value_area_pct=value_area_pct)

    def update(self, bar: Bar) -> VolumeProfile:
        self._bars.append(bar)
        return _volume_profile_calculate(
            self._bars, num_levels=self.num_levels, value_area_pct=self.value_area_pct
        )

    @property
    def value(self) -> VolumeProfile | None:
        if not self._bars:
            return None
        return _volume_profile_calculate(
            self._bars, num_levels=self.num_levels, value_area_pct=self.value_area_pct
        )
