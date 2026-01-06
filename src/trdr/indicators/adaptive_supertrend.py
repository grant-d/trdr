"""ML Adaptive SuperTrend using k-means volatility clustering."""

from dataclasses import dataclass

import numpy as np

from ..data import Bar
from .volatility_regime import _atr_series


@dataclass(frozen=True)
class VolatilityCluster:
    """Volatility cluster information from k-means classification."""

    centroid: float  # ATR value at cluster center
    size: int  # Number of data points in cluster
    level: str  # "high", "medium", "low"


class AdaptiveSupertrendIndicator:
    """Streaming adaptive SuperTrend calculator."""

    def __init__(
        self,
        atr_period: int = 10,
        st_factor: float = 3.0,
        training_period: int = 100,
        high_vol_percentile: float = 0.75,
        mid_vol_percentile: float = 0.50,
        low_vol_percentile: float = 0.25,
        max_iterations: int = 100,
    ) -> None:
        self.atr_period = max(1, atr_period)
        self.st_factor = st_factor
        self.training_period = max(1, training_period)
        self.high_vol_percentile = high_vol_percentile
        self.mid_vol_percentile = mid_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        self.max_iterations = max_iterations
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar],
        atr_period: int = 10,
        st_factor: float = 3.0,
        training_period: int = 100,
        high_vol_percentile: float = 0.75,
        mid_vol_percentile: float = 0.50,
        low_vol_percentile: float = 0.25,
        max_iterations: int = 100,
    ) -> tuple[float, int, str, list[VolatilityCluster]]:
        min_bars = max(atr_period, training_period) + 1
        if len(bars) < min_bars:
            return (bars[-1].close if bars else 0.0, 1, "medium", [])

        atr_series = _atr_series(bars, atr_period)
        atr_values = [v for v in atr_series[-training_period:] if v > 0]

        if len(atr_values) < 10:
            return (bars[-1].close if bars else 0.0, 1, "medium", [])

        atr_array = np.array(atr_values)
        current_atr = atr_series[-1] if atr_series else 0.0

        atr_min = float(np.min(atr_array))
        atr_max = float(np.max(atr_array))
        atr_range = atr_max - atr_min

        centroids = np.array(
            [
                atr_min + atr_range * high_vol_percentile,
                atr_min + atr_range * mid_vol_percentile,
                atr_min + atr_range * low_vol_percentile,
            ]
        )

        prev_centroids = np.zeros(3)
        iterations = 0

        while iterations < max_iterations and not np.allclose(centroids, prev_centroids, rtol=1e-6):
            prev_centroids = centroids.copy()

            distances = np.abs(atr_array[:, np.newaxis] - centroids)
            assignments = np.argmin(distances, axis=1)

            for k in range(3):
                cluster_points = atr_array[assignments == k]
                if len(cluster_points) > 0:
                    centroids[k] = float(np.mean(cluster_points))

            iterations += 1

        cluster_sizes = [int(np.sum(assignments == k)) for k in range(3)]
        cluster_labels = ["high", "medium", "low"]

        clusters = [
            VolatilityCluster(
                centroid=float(centroids[k]), size=cluster_sizes[k], level=cluster_labels[k]
            )
            for k in range(3)
        ]

        current_distances = np.abs(current_atr - centroids)
        assigned_cluster = int(np.argmin(current_distances))
        assigned_centroid = centroids[assigned_cluster]
        cluster_level = cluster_labels[assigned_cluster]

        src = (bars[-1].high + bars[-1].low) / 2.0
        upper_band = src + st_factor * assigned_centroid
        lower_band = src - st_factor * assigned_centroid

        trend = 1
        st_upper = upper_band
        st_lower = lower_band

        lookback = min(len(bars), atr_period * 2)
        for i in range(len(bars) - lookback, len(bars)):
            bar = bars[i]
            prev_close = bars[i - 1].close if i > 0 else bar.close

            bar_src = (bar.high + bar.low) / 2.0

            if i < len(atr_series) and atr_series[i] > 0:
                bar_atr = atr_series[i]
            else:
                bar_atr = assigned_centroid
            bar_distances = np.abs(bar_atr - centroids)
            bar_centroid = centroids[int(np.argmin(bar_distances))]

            new_upper = bar_src + st_factor * bar_centroid
            new_lower = bar_src - st_factor * bar_centroid

            if prev_close > st_lower:
                st_lower = max(new_lower, st_lower)
            else:
                st_lower = new_lower

            if prev_close < st_upper:
                st_upper = min(new_upper, st_upper)
            else:
                st_upper = new_upper

            if trend == -1 and bar.close > st_upper:
                trend = 1
            elif trend == 1 and bar.close < st_lower:
                trend = -1

        st_value = st_lower if trend == 1 else st_upper
        return (float(st_value), trend, cluster_level, clusters)

    def update(self, bar: Bar) -> tuple[float, int, str, list[VolatilityCluster]]:
        self._bars.append(bar)
        return self.calculate(
            self._bars,
            atr_period=self.atr_period,
            st_factor=self.st_factor,
            training_period=self.training_period,
            high_vol_percentile=self.high_vol_percentile,
            mid_vol_percentile=self.mid_vol_percentile,
            low_vol_percentile=self.low_vol_percentile,
            max_iterations=self.max_iterations,
        )

    @property
    def value(self) -> tuple[float, int, str, list[VolatilityCluster]]:
        if not self._bars:
            return (0.0, 1, "medium", [])
        return self.calculate(
            self._bars,
            atr_period=self.atr_period,
            st_factor=self.st_factor,
            training_period=self.training_period,
            high_vol_percentile=self.high_vol_percentile,
            mid_vol_percentile=self.mid_vol_percentile,
            low_vol_percentile=self.low_vol_percentile,
            max_iterations=self.max_iterations,
        )
