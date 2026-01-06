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


def _build_profile(
    price_levels: list[float],
    volumes: list[float],
    value_area_pct: float,
    total_volume: float | None = None,
) -> VolumeProfile:
    if not price_levels:
        raise ValueError("No price levels provided for volume profile calculation")
    if total_volume is None:
        total_volume = sum(volumes)

    if len(price_levels) == 1:
        return VolumeProfile(
            poc=price_levels[0],
            vah=price_levels[0],
            val=price_levels[0],
            hvns=[price_levels[0]] if total_volume > 0 else [],
            lvns=[],
            price_levels=price_levels,
            volumes=volumes,
            total_volume=total_volume,
        )

    num_levels = len(price_levels)
    poc_bucket = int(np.argmax(volumes))
    poc = price_levels[poc_bucket]

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

    avg_volume = total_volume / num_levels if num_levels > 0 else 0.0
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


def _volume_profile_calculate(
    bars: list[Bar],
    num_levels: int = 40,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    if not bars:
        raise ValueError("No bars provided for volume profile calculation")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive for volume profile calculation")

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
        return _build_profile(
            price_levels=[price_min],
            volumes=[total_bar_volume],
            value_area_pct=value_area_pct,
            total_volume=total_bar_volume,
        )

    bucket_size = (price_max - price_min) / num_levels
    volumes = [0.0] * num_levels
    price_levels = [price_min + (i + 0.5) * bucket_size for i in range(num_levels)]

    for bar in bars:
        bar_low = bar.low
        bar_high = bar.high
        bar_volume = bar.volume

        start_bucket = max(0, min(num_levels - 1, int((bar_low - price_min) / bucket_size)))
        end_bucket = max(0, min(num_levels - 1, int((bar_high - price_min) / bucket_size)))

        if end_bucket < start_bucket:
            start_bucket, end_bucket = end_bucket, start_bucket

        buckets_spanned = end_bucket - start_bucket + 1
        volume_per_bucket = bar_volume / buckets_spanned

        for bucket in range(start_bucket, end_bucket + 1):
            volumes[bucket] += volume_per_bucket

    return _build_profile(
        price_levels=price_levels,
        volumes=volumes,
        value_area_pct=value_area_pct,
    )


class VolumeProfileIndicator:
    """Streaming Volume Profile calculator."""

    def __init__(self, num_levels: int = 40, value_area_pct: float = 0.70) -> None:
        self.num_levels = num_levels
        self.value_area_pct = value_area_pct
        self._bars: list[Bar] = []
        self._price_min: float | None = None
        self._price_max: float | None = None
        self._bucket_size: float = 0.0
        self._price_levels: list[float] = []
        self._volumes: list[float] = []
        self._total_volume: float = 0.0
        self._profile: VolumeProfile | None = None

    @staticmethod
    def calculate(
        bars: list[Bar], num_levels: int = 40, value_area_pct: float = 0.70
    ) -> VolumeProfile:
        return _volume_profile_calculate(bars, num_levels=num_levels, value_area_pct=value_area_pct)

    def update(self, bar: Bar) -> VolumeProfile:
        self._bars.append(bar)
        if self._price_min is None or self._price_max is None:
            self._price_min = bar.low
            self._price_max = bar.high
            if self._price_max == self._price_min:
                self._price_levels = [self._price_min]
                self._volumes = [bar.volume]
                self._total_volume = bar.volume
                self._profile = _build_profile(
                    price_levels=self._price_levels,
                    volumes=self._volumes,
                    value_area_pct=self.value_area_pct,
                    total_volume=self._total_volume,
                )
                return self._profile

        range_changed = False
        if bar.low < self._price_min:
            self._price_min = bar.low
            range_changed = True
        if bar.high > self._price_max:
            self._price_max = bar.high
            range_changed = True

        if range_changed or self._bucket_size == 0.0:
            self._profile = _volume_profile_calculate(
                self._bars, num_levels=self.num_levels, value_area_pct=self.value_area_pct
            )
            self._price_levels = self._profile.price_levels
            self._volumes = list(self._profile.volumes)
            self._total_volume = self._profile.total_volume
            if self._price_levels:
                self._bucket_size = (
                    (self._price_max - self._price_min) / self.num_levels
                    if self._price_max != self._price_min
                    else 0.0
                )
            return self._profile

        start_bucket = max(
            0, min(self.num_levels - 1, int((bar.low - self._price_min) / self._bucket_size))
        )
        end_bucket = max(
            0, min(self.num_levels - 1, int((bar.high - self._price_min) / self._bucket_size))
        )
        if end_bucket < start_bucket:
            start_bucket, end_bucket = end_bucket, start_bucket

        buckets_spanned = end_bucket - start_bucket + 1
        volume_per_bucket = bar.volume / buckets_spanned
        for bucket in range(start_bucket, end_bucket + 1):
            self._volumes[bucket] += volume_per_bucket

        self._total_volume += bar.volume
        self._profile = _build_profile(
            price_levels=self._price_levels,
            volumes=self._volumes,
            value_area_pct=self.value_area_pct,
            total_volume=self._total_volume,
        )
        return self._profile

    @property
    def value(self) -> VolumeProfile | None:
        if self._profile is None and self._bars:
            self._profile = _volume_profile_calculate(
                self._bars, num_levels=self.num_levels, value_area_pct=self.value_area_pct
            )
        return self._profile
