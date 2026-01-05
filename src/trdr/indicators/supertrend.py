"""SuperTrend ATR-based trailing stop indicator."""

from ..data import Bar
from .volatility_regime import _atr_series, _sma_series, _true_ranges


def supertrend(
    bars: list[Bar],
    period: int = 10,
    multiplier: float = 3.0,
    use_atr: bool = True,
) -> tuple[float, int]:
    """Calculate SuperTrend indicator (ATR-based trailing stop).

    SuperTrend is a trend-following indicator that uses ATR to create
    dynamic support/resistance levels. Generates buy signals when price
    crosses above the indicator, sell signals when crossing below.

    Original indicator by Olivier Seban, popularized by various traders.
    This implementation based on common TradingView versions.

    Args:
        bars: List of OHLCV bars
        period: ATR calculation period
        multiplier: ATR multiplier for band width
        use_atr: Use ATR (True) vs simple TR average (False)

    Returns:
        Tuple of (supertrend_value, trend_direction)
        - supertrend_value: Current ST level (support in uptrend, resistance in downtrend)
        - trend_direction: 1 for uptrend, -1 for downtrend
    """
    return SupertrendIndicator.calculate(
        bars,
        period=period,
        multiplier=multiplier,
        use_atr=use_atr,
    )


class SupertrendIndicator:
    """Streaming SuperTrend calculator."""

    def __init__(self, period: int = 10, multiplier: float = 3.0, use_atr: bool = True) -> None:
        self.period = max(1, period)
        self.multiplier = multiplier
        self.use_atr = use_atr
        self._bars: list[Bar] = []

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
            atr_series = _atr_series(bars, period)
        else:
            tr_series = _true_ranges(bars)
            atr_series = _sma_series(tr_series, period)

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
        self._bars.append(bar)
        return self.calculate(
            self._bars,
            period=self.period,
            multiplier=self.multiplier,
            use_atr=self.use_atr,
        )

    @property
    def value(self) -> tuple[float, int]:
        if not self._bars:
            return (0.0, 1)
        return self.calculate(
            self._bars,
            period=self.period,
            multiplier=self.multiplier,
            use_atr=self.use_atr,
        )
