"""Tests for technical indicators."""

import pytest

from trdr.data.market import Bar
from trdr.indicators import (
    VolumeProfile,
    atr,
    bollinger_bands,
    ema,
    ema_series,
    hma,
    hma_slope,
    macd,
    mss,
    rsi,
    sma,
    volume_profile,
    volume_trend,
    wma,
)


def make_bars(closes: list[float], volume: float = 1000.0) -> list[Bar]:
    """Create bars from close prices."""
    return [
        Bar(
            timestamp="2024-01-01T00:00:00Z",
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=volume,
        )
        for c in closes
    ]


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_basic(self):
        bars = make_bars([10, 20, 30, 40, 50])
        assert sma(bars, 3) == 40.0  # (30+40+50)/3

    def test_insufficient_data(self):
        bars = make_bars([10, 20])
        result = sma(bars, 5)
        assert result == 20.0  # Returns last close

    def test_period_1(self):
        bars = make_bars([10, 20, 30])
        assert sma(bars, 1) == 30.0  # Just the last value

    def test_empty_bars(self):
        assert sma([], 5) == 0.0


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_basic(self):
        bars = make_bars([10, 20, 30, 40, 50])
        result = ema(bars, 3)
        assert result > 30  # EMA weights recent prices more

    def test_insufficient_data(self):
        bars = make_bars([10, 20])
        result = ema(bars, 5)
        assert result == 20.0  # Returns last close

    def test_period_1(self):
        bars = make_bars([10, 20, 30])
        result = ema(bars, 1)
        assert result == 30.0  # Just the last value

    def test_empty_bars(self):
        assert ema([], 5) == 0.0


class TestEmaSeries:
    """Tests for EMA series calculation."""

    def test_returns_list(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = ema_series(values, 3)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_insufficient_data(self):
        values = [10.0, 20.0]
        result = ema_series(values, 5)
        assert result == [0.0, 0.0]

    def test_period_1(self):
        values = [10.0, 20.0, 30.0]
        result = ema_series(values, 1)
        # Period 1 EMA = just the values themselves
        assert result[-1] == 30.0


class TestWMA:
    """Tests for Weighted Moving Average."""

    def test_basic(self):
        bars = make_bars([10, 20, 30])
        result = wma(bars, 3)
        # WMA = (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6 = 23.33
        assert abs(result - 23.33) < 0.1

    def test_period_1(self):
        bars = make_bars([10, 20, 30])
        assert wma(bars, 1) == 30.0

    def test_insufficient_data(self):
        bars = make_bars([10, 20])
        result = wma(bars, 5)
        assert result == 20.0


class TestHMA:
    """Tests for Hull Moving Average."""

    def test_basic(self):
        bars = make_bars([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        result = hma(bars, 9)
        assert result > 0

    def test_insufficient_data(self):
        bars = make_bars([10, 20, 30])
        result = hma(bars, 9)
        assert result == 0.0

    def test_period_1(self):
        # Period 1 causes half_period=0, resulting in NaN (expected edge case)
        bars = make_bars([10, 20, 30])
        result = hma(bars, 1)
        # HMA with period 1 is degenerate - returns NaN
        import math

        assert math.isnan(result)


class TestHMASlope:
    """Tests for HMA slope."""

    def test_uptrend_positive_slope(self):
        bars = make_bars(list(range(10, 100, 5)))  # Uptrend
        result = hma_slope(bars, 9, 3)
        assert result > 0

    def test_downtrend_negative_slope(self):
        bars = make_bars(list(range(100, 10, -5)))  # Downtrend
        result = hma_slope(bars, 9, 3)
        assert result < 0

    def test_insufficient_data(self):
        bars = make_bars([10, 20, 30])
        result = hma_slope(bars, 9, 3)
        assert result == 0.0


class TestATR:
    """Tests for Average True Range."""

    def test_basic(self):
        bars = make_bars([100, 102, 98, 105, 103] * 4)
        result = atr(bars, 14)
        assert result > 0

    def test_insufficient_data(self):
        bars = make_bars([100, 102, 98])
        result = atr(bars, 14)
        assert result == 0.0

    def test_period_1(self):
        # Create bars with explicit high/low for true range calc
        bars = [
            Bar(
                timestamp="2024-01-01",
                open=100,
                high=105,
                low=95,
                close=102,
                volume=1000,
            ),
            Bar(
                timestamp="2024-01-02",
                open=102,
                high=108,
                low=98,
                close=106,
                volume=1000,
            ),
            Bar(
                timestamp="2024-01-03",
                open=106,
                high=110,
                low=102,
                close=108,
                volume=1000,
            ),
        ]
        result = atr(bars, 1)
        assert result > 0


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_basic(self):
        bars = make_bars([100] * 20)  # Flat prices
        upper, middle, lower = bollinger_bands(bars, 20)
        assert upper == middle == lower == 100  # No std dev

    def test_with_variance(self):
        bars = make_bars([90, 95, 100, 105, 110] * 4)
        upper, middle, lower = bollinger_bands(bars, 20)
        assert upper > middle > lower

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        upper, middle, lower = bollinger_bands(bars, 20)
        assert upper == middle == lower == 102

    def test_period_1(self):
        bars = make_bars([100, 110, 105])
        upper, middle, lower = bollinger_bands(bars, 1)
        # With period 1, std=0
        assert upper == middle == lower == 105


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_overbought(self):
        # Strong uptrend should give high RSI
        bars = make_bars(list(range(50, 80)))
        result = rsi(bars, 14)
        assert result > 70

    def test_oversold(self):
        # Strong downtrend should give low RSI
        bars = make_bars(list(range(80, 50, -1)))
        result = rsi(bars, 14)
        assert result < 30

    def test_neutral_on_insufficient_data(self):
        bars = make_bars([100, 102, 98])
        result = rsi(bars, 14)
        assert result == 50.0

    def test_period_1(self):
        bars = make_bars([100, 110])  # One up move
        result = rsi(bars, 1)
        assert result == 100.0  # All gains


class TestMACD:
    """Tests for MACD."""

    def test_basic(self):
        bars = make_bars(list(range(50, 100)))
        macd_line, signal_line, histogram = macd(bars, 12, 26, 9)
        assert macd_line != 0
        assert signal_line != 0

    def test_insufficient_data(self):
        bars = make_bars([100, 102, 98])
        macd_line, signal_line, histogram = macd(bars, 12, 26, 9)
        assert macd_line == signal_line == histogram == 0.0


class TestMSS:
    """Tests for Market Structure Score."""

    def test_bullish_trend(self):
        bars = make_bars(list(range(50, 80)))
        result = mss(bars, 20)
        assert result > 0

    def test_bearish_trend(self):
        bars = make_bars(list(range(80, 50, -1)))
        result = mss(bars, 20)
        assert result < 0

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = mss(bars, 20)
        assert result == 0.0


class TestVolumeProfile:
    """Tests for Volume Profile."""

    def test_basic(self):
        bars = make_bars([100, 102, 101, 103, 100])
        result = volume_profile(bars)
        assert isinstance(result, VolumeProfile)
        assert result.poc > 0
        assert result.vah >= result.val

    def test_flat_market(self):
        # Note: make_bars adds 1% high/low variance, so not truly flat
        bars = make_bars([100] * 10)
        result = volume_profile(bars)
        # PoC should be near 100 (within the high/low range)
        assert 99 < result.poc < 101
        assert result.vah >= result.val

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            volume_profile([])


class TestVolumeTrend:
    """Tests for volume trend detection."""

    def test_increasing(self):
        # Create bars with increasing volume
        bars = [
            Bar(
                timestamp="2024-01-01",
                open=100,
                high=101,
                low=99,
                close=100,
                volume=v,
            )
            for v in [100, 110, 120, 130, 140, 200, 250, 300, 350, 400]
        ]
        result = volume_trend(bars, 5)
        assert result == "increasing"

    def test_declining(self):
        bars = [
            Bar(
                timestamp="2024-01-01",
                open=100,
                high=101,
                low=99,
                close=100,
                volume=v,
            )
            for v in [400, 350, 300, 250, 200, 100, 90, 80, 70, 60]
        ]
        result = volume_trend(bars, 5)
        assert result == "declining"

    def test_neutral_on_insufficient(self):
        bars = make_bars([100, 102])
        result = volume_trend(bars, 5)
        assert result == "neutral"
