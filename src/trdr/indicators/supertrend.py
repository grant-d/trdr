"""SuperTrend ATR-based trailing stop indicator."""

from ..data import Bar
from collections import deque

from .atr import AtrIndicator
from .sma import sma_series


class SupertrendIndicator:
    """Streaming SuperTrend calculator."""

    def __init__(self, period: int = 10, multiplier: float = 3.0, use_atr: bool = True) -> None:
        self.period = max(1, period)
        self.multiplier = multiplier
        self.use_atr = use_atr

        # Streaming state
        self._atr_ind = AtrIndicator(period) if use_atr else None
        self._tr_values: deque[float] | None = None
        self._tr_sum = 0.0
        if not use_atr:
            self._tr_values = deque(maxlen=self.period)
        self._prev_close: float | None = None
        self._up_band = 0.0
        self._dn_band = 0.0
        self._trend = 1
        self._value = 0.0

    @staticmethod
    def calculate(
        bars: list[Bar],
        period: int = 10,
        multiplier: float = 3.0,
        use_atr: bool = True,
    ) -> tuple[float, int]:
        if len(bars) < period + 1:
            return (bars[-1].close if bars else 0.0, 1)

        if use_atr:
            ind = AtrIndicator(period)
            atr_series = [ind.update(bar) for bar in bars]
        else:
            tr_series = []
            for i in range(len(bars)):
                if i == 0:
                    tr_series.append(0.0)
                else:
                    high = bars[i].high
                    low = bars[i].low
                    prev_close = bars[i - 1].close
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    tr_series.append(tr)
            atr_series = sma_series(tr_series, period)

        atr_val = atr_series[-1] if atr_series else 0.0

        src = (bars[-1].high + bars[-1].low) / 2.0

        basic_up = src - (multiplier * atr_val)
        basic_dn = src + (multiplier * atr_val)

        trend = 1
        up_band = basic_up
        dn_band = basic_dn

        lookback = min(len(bars), period * 2)
        for i in range(len(bars) - lookback, len(bars)):
            bar = bars[i]
            prev_close = bars[i - 1].close if i > 0 else bar.close

            bar_src = (bar.high + bar.low) / 2.0

            bar_atr = atr_series[i] if i < len(atr_series) and atr_series[i] > 0 else atr_val

            new_up = bar_src - (multiplier * bar_atr)
            new_dn = bar_src + (multiplier * bar_atr)

            if prev_close > up_band:
                up_band = max(new_up, up_band)
            else:
                up_band = new_up

            if prev_close < dn_band:
                dn_band = min(new_dn, dn_band)
            else:
                dn_band = new_dn

            if trend == -1 and bar.close > dn_band:
                trend = 1
            elif trend == 1 and bar.close < up_band:
                trend = -1

        st_value = up_band if trend == 1 else dn_band
        return (float(st_value), trend)

    def update(self, bar: Bar) -> tuple[float, int]:
        # Update ATR/SMA
        if self.use_atr:
            atr_val = self._atr_ind.update(bar)
        else:
            # For non-ATR mode, compute true range and use SMA
            if self._prev_close is not None:
                high = bar.high
                low = bar.low
                tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
            else:
                tr = 0.0
            if self._tr_values is None:
                self._tr_values = deque(maxlen=self.period)
            if len(self._tr_values) == self.period:
                self._tr_sum -= self._tr_values[0]
            self._tr_values.append(tr)
            self._tr_sum += tr
            if len(self._tr_values) < self.period:
                atr_val = tr
            else:
                atr_val = self._tr_sum / self.period

        src = (bar.high + bar.low) / 2.0
        basic_up = src - (self.multiplier * atr_val)
        basic_dn = src + (self.multiplier * atr_val)

        # Update bands
        if self._prev_close is not None and self._prev_close > self._up_band:
            self._up_band = max(basic_up, self._up_band)
        else:
            self._up_band = basic_up

        if self._prev_close is not None and self._prev_close < self._dn_band:
            self._dn_band = min(basic_dn, self._dn_band)
        else:
            self._dn_band = basic_dn

        # Update trend
        if self._trend == -1 and bar.close > self._dn_band:
            self._trend = 1
        elif self._trend == 1 and bar.close < self._up_band:
            self._trend = -1

        self._value = self._up_band if self._trend == 1 else self._dn_band
        self._prev_close = bar.close

        return (float(self._value), self._trend)

    @property
    def value(self) -> tuple[float, int]:
        return (float(self._value), self._trend)
