"""Squeeze Momentum Indicator (John Carter's TTM Squeeze)."""

import numpy as np

from ..data import Bar
from .atr import AtrIndicator


class SqueezeMomentumIndicator:
    """Streaming Squeeze Momentum calculator."""

    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        use_true_range: bool = True,
    ) -> None:
        self.bb_length = max(1, bb_length)
        self.bb_mult = bb_mult
        self.kc_length = max(1, kc_length)
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar],
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        use_true_range: bool = True,
    ) -> tuple[float, str]:
        min_bars = max(bb_length, kc_length) + 1
        if len(bars) < min_bars:
            return (0.0, "no_squeeze")

        closes = np.array([b.close for b in bars[-bb_length:]])

        # Calculate Bollinger Bands
        bb_basis = float(np.mean(closes))
        bb_dev = bb_mult * float(np.std(closes))
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev

        # Calculate Keltner Channel
        kc_closes = np.array([b.close for b in bars[-kc_length:]])
        kc_ma = float(np.mean(kc_closes))

        if use_true_range:
            range_val = AtrIndicator.calculate(bars[-(kc_length + 1) :], kc_length)
        else:
            recent_bars = bars[-kc_length:]
            ranges = [b.high - b.low for b in recent_bars]
            range_val = float(np.mean(ranges))

        upper_kc = kc_ma + range_val * kc_mult
        lower_kc = kc_ma - range_val * kc_mult

        sqz_on = (lower_bb > lower_kc) and (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) and (upper_bb > upper_kc)

        if sqz_on:
            squeeze_state = "squeeze_on"
        elif sqz_off:
            squeeze_state = "squeeze_off"
        else:
            squeeze_state = "no_squeeze"

        lookback = bars[-kc_length:]
        highs = np.array([b.high for b in lookback])
        lows = np.array([b.low for b in lookback])
        closes_kc = np.array([b.close for b in lookback])

        highest_high = float(np.max(highs))
        lowest_low = float(np.min(lows))
        donchian_mid = (highest_high + lowest_low) / 2.0

        close_sma = float(np.mean(closes_kc))

        mid_avg = (donchian_mid + close_sma) / 2.0

        deviations = closes_kc - mid_avg

        x = np.arange(len(deviations))
        slope, intercept = np.polyfit(x, deviations, 1)
        momentum = float(slope * (len(deviations) - 1) + intercept)

        return (momentum, squeeze_state)

    def update(self, bar: Bar) -> tuple[float, str]:
        self._bars.append(bar)
        return self.calculate(
            self._bars,
            bb_length=self.bb_length,
            bb_mult=self.bb_mult,
            kc_length=self.kc_length,
            kc_mult=self.kc_mult,
            use_true_range=self.use_true_range,
        )

    @property
    def value(self) -> tuple[float, str]:
        if not self._bars:
            return (0.0, "no_squeeze")
        return self.calculate(
            self._bars,
            bb_length=self.bb_length,
            bb_mult=self.bb_mult,
            kc_length=self.kc_length,
            kc_mult=self.kc_mult,
            use_true_range=self.use_true_range,
        )
