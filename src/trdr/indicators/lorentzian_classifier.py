"""Lorentzian Distance Classifier for price direction prediction."""

import numpy as np

from ..data import Bar
from .adx import AdxIndicator
from .cci import CciIndicator
from .rsi import RsiIndicator


def _rsi_series(bars: list[Bar], period: int) -> list[float]:
    """Compute RSI series using simple moving average of gains/losses."""
    n = len(bars)
    if n == 0:
        return []
    if n < period + 1:
        return [50.0] * n

    deltas = [0.0] * n
    for i in range(1, n):
        deltas[i] = bars[i].close - bars[i - 1].close

    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    rsi_values = [50.0] * n
    sum_gain = sum(gains[1 : period + 1])
    sum_loss = sum(losses[1 : period + 1])

    for i in range(period, n):
        avg_gain = sum_gain / period
        avg_loss = sum_loss / period
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = float(100 - (100 / (1 + rs)))

        if i + 1 < n:
            sum_gain += gains[i + 1] - gains[i - period + 1]
            sum_loss += losses[i + 1] - losses[i - period + 1]

    return rsi_values


def _cci_series(bars: list[Bar], period: int) -> list[float]:
    """Compute CCI series with rolling window."""
    n = len(bars)
    if n == 0:
        return []
    if n < period:
        return [0.0] * n

    typical_prices = [(b.high + b.low + b.close) / 3 for b in bars]
    cci_values = [0.0] * n

    for i in range(period - 1, n):
        window = typical_prices[i - period + 1 : i + 1]
        sma_tp = float(np.mean(window))
        mean_dev = float(np.mean([abs(tp - sma_tp) for tp in window]))
        if mean_dev == 0:
            cci_values[i] = 0.0
        else:
            cci_values[i] = float((typical_prices[i] - sma_tp) / (0.015 * mean_dev))

    return cci_values


def _adx_series(bars: list[Bar], period: int) -> list[float]:
    """Compute ADX series using Wilder smoothing."""
    n = len(bars)
    if n < period + 1:
        return [0.0] * n

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close
        high_diff = high - bars[i - 1].high
        low_diff = bars[i - 1].low - low

        tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0.0

    atr_val = float(np.mean(tr[1 : period + 1]))
    plus_dm_s = float(np.mean(plus_dm[1 : period + 1]))
    minus_dm_s = float(np.mean(minus_dm[1 : period + 1]))

    adx_values = [0.0] * n
    dx_values: list[float] = []
    adx_val = 0.0

    for i in range(period, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        plus_dm_s = (plus_dm_s * (period - 1) + plus_dm[i]) / period
        minus_dm_s = (minus_dm_s * (period - 1) + minus_dm[i]) / period

        if atr_val == 0:
            dx = 0.0
        else:
            plus_di = (plus_dm_s / atr_val) * 100
            minus_di = (minus_dm_s / atr_val) * 100
            denom = plus_di + minus_di
            dx = abs(plus_di - minus_di) / denom * 100 if denom != 0 else 0.0

        dx_values.append(dx)

        if len(dx_values) == period:
            adx_val = float(np.mean(dx_values))
        elif len(dx_values) > period:
            adx_val = (adx_val * (period - 1) + dx) / period

        if len(dx_values) >= period:
            adx_values[i] = adx_val

    return adx_values


def _extract_features(bars: list[Bar]) -> np.ndarray:
    """Extract normalized features for Lorentzian classifier.

    Uses RSI, CCI, and ADX normalized to [0, 1] range.

    Args:
        bars: List of OHLCV bars

    Returns:
        Feature vector as numpy array
    """
    if len(bars) < 20:
        return np.array([0.5, 0.5, 0.5])

    # Feature 1: RSI (14) normalized to [0, 1]
    rsi_val = RsiIndicator.calculate(bars, 14) / 100.0

    # Feature 2: CCI (20) normalized to [-1, 1] then to [0, 1]
    cci_val = CciIndicator.calculate(bars, 20)
    cci_norm = np.clip(cci_val / 200.0, -1, 1)  # Clip to [-1, 1]
    cci_norm = (cci_norm + 1) / 2.0  # Map to [0, 1]

    # Feature 3: ADX (14) normalized to [0, 1]
    adx_val = AdxIndicator.calculate(bars, 14) / 100.0

    return np.array([rsi_val, cci_norm, adx_val])


def _lorentzian_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """Calculate Lorentzian distance between two feature vectors.

    Lorentzian distance: sum(log(1 + |x_i - y_i|))

    More robust to outliers than Euclidean distance, which is important
    for financial data affected by major market events.

    Args:
        features1: First feature vector
        features2: Second feature vector

    Returns:
        Lorentzian distance
    """
    distance = np.sum(np.log(1 + np.abs(features1 - features2)))
    return float(distance)


class LorentzianClassifierIndicator:
    """Streaming Lorentzian classifier wrapper."""

    def __init__(
        self,
        neighbors: int = 8,
        max_bars_back: int = 2000,
        training_window: int = 4,
    ) -> None:
        self.neighbors = neighbors
        self.max_bars_back = max_bars_back
        self.training_window = training_window
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar],
        neighbors: int = 8,
        max_bars_back: int = 2000,
        training_window: int = 4,
    ) -> int:
        if len(bars) < max(neighbors, training_window) + 10:
            return 0

        rsi_series = _rsi_series(bars, 14)
        cci_series = _cci_series(bars, 20)
        adx_series = _adx_series(bars, 14)

        def features_at(idx: int) -> np.ndarray:
            rsi_val = (rsi_series[idx] / 100.0) if idx < len(rsi_series) else 0.5
            cci_val = cci_series[idx] if idx < len(cci_series) else 0.0
            cci_norm = np.clip(cci_val / 200.0, -1, 1)
            cci_norm = (cci_norm + 1) / 2.0
            adx_val = (adx_series[idx] / 100.0) if idx < len(adx_series) else 0.0
            return np.array([rsi_val, cci_norm, adx_val])

        current_features = features_at(len(bars) - 1)

        labels = []
        feature_history = []

        lookback = min(max_bars_back, len(bars) - training_window)

        for i in range(lookback - 1, 0, -4):
            if i < training_window + 10:
                continue

            future_close = bars[min(i + training_window, len(bars) - 1)].close
            current_close = bars[i].close

            if future_close > current_close * 1.001:
                label = 1
            elif future_close < current_close * 0.999:
                label = -1
            else:
                label = 0

            labels.append(label)
            feature_history.append(features_at(i))

            if len(labels) >= neighbors * 4:
                break

        if len(labels) < neighbors:
            return 0

        distances = []
        for hist_features in feature_history:
            dist = _lorentzian_distance(current_features, hist_features)
            distances.append(dist)

        sorted_indices = np.argsort(distances)
        nearest_labels = [labels[i] for i in sorted_indices[:neighbors]]

        prediction_sum = sum(nearest_labels)

        if prediction_sum > 0:
            return 1
        if prediction_sum < 0:
            return -1
        return 0

    def update(self, bar: Bar) -> int:
        self._bars.append(bar)
        return self.calculate(
            self._bars,
            neighbors=self.neighbors,
            max_bars_back=self.max_bars_back,
            training_window=self.training_window,
        )

    @property
    def value(self) -> int:
        if not self._bars:
            return 0
        return self.calculate(
            self._bars,
            neighbors=self.neighbors,
            max_bars_back=self.max_bars_back,
            training_window=self.training_window,
        )
