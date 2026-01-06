"""Lorentzian Distance Classifier for price direction prediction."""

import heapq

import numpy as np

from ..data import Bar
from .adx import AdxIndicator
from .cci import CciIndicator
from .rsi import RsiIndicator


def _lorentzian_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """Calculate Lorentzian distance between two feature vectors.

    Lorentzian distance: sum(log(1 + |x_i - y_i|))
    """
    # Using simple loop for scalar speed or numpy if vectors
    # dist = sum(math.log(1 + abs(f1 - f2)) for f1, f2 in zip(features1, features2))
    return float(np.sum(np.log(1 + np.abs(features1 - features2))))


class LorentzianClassifierIndicator:
    """Streaming Lorentzian classifier wrapper.

    Optimized for incremental updates using stateful sub-indicators.
    """

    def __init__(
        self,
        neighbors: int = 8,
        max_bars_back: int = 2000,
        training_window: int = 4,
    ) -> None:
        self.neighbors = neighbors
        self.max_bars_back = max_bars_back
        self.training_window = training_window

        # Stateful indicators
        self._rsi = RsiIndicator(period=14)
        self._cci = CciIndicator(period=20)
        self._adx = AdxIndicator(period=14)

        # History storage
        # We need history of features and prices to determine labels and distances
        self._history_features: list[np.ndarray] = []
        self._history_closes: list[float] = []
        self._bars_seen = 0
        self._last_prediction = 0

    @staticmethod
    def calculate(
        bars: list[Bar],
        neighbors: int = 8,
        max_bars_back: int = 2000,
        training_window: int = 4,
    ) -> int:
        """Calculate on a full list of bars (slow, for backward compatibility)."""
        ind = LorentzianClassifierIndicator(neighbors, max_bars_back, training_window)
        val = 0
        for bar in bars:
            val = ind.update(bar)
        return val

    def update(self, bar: Bar) -> int:
        """Update with a new bar and return the prediction."""
        self._bars_seen += 1

        # Update sub-indicators
        rsi_val = self._rsi.update(bar)
        cci_val = self._cci.update(bar)
        adx_val = self._adx.update(bar)

        # Normalize features
        # RSI: [0, 100] -> [0, 1]
        feat_rsi = (rsi_val / 100.0) if rsi_val is not None else 0.5

        # CCI: [-200, 200] roughly -> [-1, 1] -> [0, 1]
        raw_cci = cci_val if cci_val is not None else 0.0
        feat_cci = np.clip(raw_cci / 200.0, -1, 1)
        feat_cci = (feat_cci + 1) / 2.0

        # ADX: [0, 100] -> [0, 1]
        feat_adx = (adx_val / 100.0) if adx_val is not None else 0.0

        current_features = np.array([feat_rsi, feat_cci, feat_adx])

        # Store history
        self._history_features.append(current_features)
        self._history_closes.append(bar.close)

        # Prune history if too long (keep enough for lookback + training window)
        # We need max_bars_back for search, plus some buffer?
        # Actually max_bars_back is the search range.
        if len(self._history_features) > self.max_bars_back + self.training_window + 100:
            # Keep the last max_bars_back + training_window
            keep = self.max_bars_back + self.training_window
            self._history_features = self._history_features[-keep:]
            self._history_closes = self._history_closes[-keep:]

        # Need enough history to have at least one labeled point
        # A point at index `i` is labeled if we know the future price at `i + training_window`
        # So we need len > training_window
        n = len(self._history_features)
        if n < self.training_window + 10:
            self._last_prediction = 0
            return 0

        # We search for neighbors in the past.
        # A candidate point at index `i` (in stored history) is valid if:
        # 1. It is far enough back to have a label: i + training_window < n - 1 (current bar is n-1)
        #    So max `i` = n - 1 - training_window - 1?
        #    Actually, we predict for the current bar by comparing to past features.
        #    Past feature at `i` has a known outcome at `i + training_window`,
        #    so we need `i + training_window <= current_index`.

        # Maintain a fixed-size max-heap of nearest neighbors: (-dist, label)
        nearest_heap: list[tuple[float, int]] = []

        # Search range:
        # We want `i` such that `i + training_window < n`
        # Iterate backwards from most recent valid training sample
        # Step of 4 as in original implementation? "range(lookback - 1, 0, -4)"

        # Current index in history is n-1.
        # We need i such that history_closes[i + training_window] exists.
        # Max i = n - 1 - training_window.

        max_i = n - 1 - self.training_window
        min_i = max(0, max_i - self.max_bars_back)

        # Using a step of 1 gives better results but is slower. Original used 4.
        # Original: for i in range(lookback - 1, 0, -4)
        # To balance accuracy and speed, use step 2.
        # If we need a closer match to the original speed profile, revert to step 4.

        cnt = 0
        for i in range(max_i, min_i, -2):
            # Calculate distance
            hist_features = self._history_features[i]
            dist = _lorentzian_distance(current_features, hist_features)

            # Determine label for this historical point
            future_close = self._history_closes[i + self.training_window]
            past_close = self._history_closes[i]

            label = 0
            if future_close > past_close * 1.001:
                label = 1
            elif future_close < past_close * 0.999:
                label = -1

            if len(nearest_heap) < self.neighbors:
                heapq.heappush(nearest_heap, (-dist, label))
            else:
                if dist < -nearest_heap[0][0]:
                    heapq.heapreplace(nearest_heap, (-dist, label))

            cnt += 1
            if cnt >= self.max_bars_back:  # Safety break
                break

        if len(nearest_heap) < self.neighbors:
            self._last_prediction = 0
            return 0

        prediction_sum = sum(label for _, label in nearest_heap)

        if prediction_sum > 0:
            self._last_prediction = 1
        elif prediction_sum < 0:
            self._last_prediction = -1
        else:
            self._last_prediction = 0
        return self._last_prediction

    @property
    def value(self) -> int:
        return self._last_prediction
