"""Heikin-Ashi smoothed candlestick transformation."""

from ..data import Bar


class HeikinAshiIndicator:
    """Streaming Heikin-Ashi calculator."""

    def __init__(self) -> None:
        self._prev_open: float | None = None
        self._prev_close: float | None = None
        self._last_bar: dict | None = None

    @staticmethod
    def calculate(bars: list[Bar]) -> list[dict]:
        if len(bars) < 1:
            return []

        ha_bars = []

        for i, bar in enumerate(bars):
            ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0

            if i == 0:
                ha_open = (bar.open + bar.close) / 2.0
            else:
                prev_bar = ha_bars[-1]
                ha_open = (prev_bar["open"] + prev_bar["close"]) / 2.0

            ha_high = max(bar.high, ha_open, ha_close)
            ha_low = min(bar.low, ha_open, ha_close)

            ha_bar = {
                "open": ha_open,
                "high": ha_high,
                "low": ha_low,
                "close": ha_close,
                "volume": bar.volume,
                "timestamp": getattr(bar, "timestamp", None),
            }
            ha_bars.append(ha_bar)

        return ha_bars

    def update(self, bar: Bar) -> dict:
        ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0
        if self._prev_open is None or self._prev_close is None:
            ha_open = (bar.open + bar.close) / 2.0
        else:
            ha_open = (self._prev_open + self._prev_close) / 2.0

        ha_high = max(bar.high, ha_open, ha_close)
        ha_low = min(bar.low, ha_open, ha_close)

        ha_bar = {
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "volume": bar.volume,
            "timestamp": getattr(bar, "timestamp", None),
        }

        self._prev_open = ha_open
        self._prev_close = ha_close
        self._last_bar = ha_bar
        return ha_bar

    @property
    def value(self) -> dict | None:
        return self._last_bar
