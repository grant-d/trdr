"""Tests for technical indicators."""

import pytest

from trdr.data import Bar
from trdr.indicators import (
    VolumeProfile,
    atr,
    bollinger_bands,
    ema,
    ema_series,
    heikin_ashi,
    hma,
    hma_slope,
    hvn_support_strength,
    kalman,
    kalman_series,
    macd,
    mss,
    multi_timeframe_poc,
    order_flow_imbalance,
    rsi,
    sax_bullish_reversal,
    sax_pattern,
    sma,
    volatility_regime,
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


class TestKalman:
    """Tests for Kalman Filter."""

    def test_basic(self):
        bars = make_bars([100, 102, 104, 106, 108])
        result = kalman(bars)
        assert result > 100  # Should track price trend

    def test_single_bar(self):
        bars = make_bars([100])
        result = kalman(bars)
        assert result == 100.0

    def test_empty_bars(self):
        result = kalman([])
        assert result == 0.0

    def test_smoothing(self):
        # Noisy data should be smoothed
        bars = make_bars([100, 105, 95, 110, 90, 115])
        result = kalman(bars, measurement_noise=1.0)
        # Result should be less volatile than raw price
        assert 90 < result < 115


class TestKalmanSeries:
    """Tests for Kalman Filter series."""

    def test_returns_list(self):
        bars = make_bars([100, 102, 104, 106, 108])
        result = kalman_series(bars)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_empty_bars(self):
        result = kalman_series([])
        assert result == []

    def test_first_value_preserved(self):
        bars = make_bars([100, 102, 104])
        result = kalman_series(bars)
        assert result[0] == 100.0


class TestVolatilityRegime:
    """Tests for volatility regime detection."""

    def test_returns_valid_regime(self):
        bars = make_bars([100, 102, 101, 103, 100] * 10)
        result = volatility_regime(bars, lookback=50)
        assert result in ["low", "medium", "high"]

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = volatility_regime(bars, lookback=50)
        assert result == "medium"

    def test_high_volatility(self):
        # Very volatile prices
        bars = make_bars([100, 120, 80, 130, 70, 140] * 10)
        result = volatility_regime(bars, lookback=50)
        assert result in ["medium", "high"]


class TestOrderFlowImbalance:
    """Tests for order flow imbalance."""

    def test_buying_pressure(self):
        # Steadily rising prices = buying pressure
        bars = make_bars(list(range(100, 110)))
        result = order_flow_imbalance(bars, lookback=5)
        assert result > 0  # Positive = buying

    def test_selling_pressure(self):
        # Falling prices = selling pressure
        bars = make_bars(list(range(110, 100, -1)))
        result = order_flow_imbalance(bars, lookback=5)
        assert result < 0  # Negative = selling

    def test_insufficient_data(self):
        bars = make_bars([100])
        result = order_flow_imbalance(bars, lookback=5)
        assert result == 0.0

    def test_range_bound(self):
        result = order_flow_imbalance(make_bars([100, 102, 101, 100, 102]), lookback=5)
        assert -1.0 <= result <= 1.0


class TestMultiTimeframePOC:
    """Tests for multi-timeframe Point of Control."""

    def test_returns_three_values(self):
        bars = make_bars([100, 102, 101, 103, 100] * 5)
        poc1, poc2, poc3 = multi_timeframe_poc(bars)
        assert isinstance(poc1, float)
        assert isinstance(poc2, float)
        assert isinstance(poc3, float)

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        poc1, poc2, poc3 = multi_timeframe_poc(bars)
        # All should return same value when insufficient data
        assert poc1 == poc2 == poc3


class TestHVNSupportStrength:
    """Tests for HVN support strength."""

    def test_strong_support(self):
        # Price repeatedly bounces from 100
        bars = [
            Bar(timestamp="2024-01-01", open=100, high=105, low=100, close=103, volume=1000)
            for _ in range(10)
        ]
        result = hvn_support_strength(bars, val_level=100, lookback=10)
        assert result > 0.5  # Strong support

    def test_no_touches(self):
        bars = make_bars([120, 121, 122, 123, 124] * 6)
        result = hvn_support_strength(bars, val_level=100, lookback=30)
        assert result == 0.0  # No support at that level

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = hvn_support_strength(bars, val_level=100, lookback=30)
        assert result == 0.0


class TestSaxPattern:
    """Tests for SAX pattern generation."""

    def test_returns_string(self):
        bars = make_bars([100, 102, 104, 106, 108] * 4)
        result = sax_pattern(bars, window=20, segments=5)
        assert isinstance(result, str)
        assert len(result) == 5  # 5 segments

    def test_alphabet_range(self):
        bars = make_bars([100, 102, 104, 106, 108] * 4)
        result = sax_pattern(bars, window=20, segments=5)
        # Should only contain letters a-e
        assert all(c in "abcde" for c in result)

    def test_flat_market(self):
        bars = make_bars([100] * 20)
        result = sax_pattern(bars, window=20, segments=5)
        assert result == "ccccc"  # All middle values

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = sax_pattern(bars, window=20, segments=5)
        assert result == ""


class TestSaxBullishReversal:
    """Tests for SAX bullish reversal detection."""

    def test_bullish_pattern(self):
        # Pattern: starts low, ends high
        assert sax_bullish_reversal("aabde") is True
        assert sax_bullish_reversal("abbde") is True

    def test_bearish_pattern(self):
        # Pattern: starts high, stays high
        assert sax_bullish_reversal("ddddd") is False
        assert sax_bullish_reversal("eeeee") is False

    def test_insufficient_length(self):
        assert sax_bullish_reversal("ab") is False
        assert sax_bullish_reversal("") is False

    def test_no_momentum(self):
        # Has low and high but no momentum
        assert sax_bullish_reversal("aabaa") is False


class TestHeikinAshi:
    """Tests for Heikin-Ashi transformation."""

    def test_returns_list_of_dicts(self):
        bars = make_bars([100, 102, 104, 106, 108])
        result = heikin_ashi(bars)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(bar, dict) for bar in result)

    def test_has_required_keys(self):
        bars = make_bars([100, 102, 104])
        result = heikin_ashi(bars)
        for ha_bar in result:
            assert "open" in ha_bar
            assert "high" in ha_bar
            assert "low" in ha_bar
            assert "close" in ha_bar
            assert "volume" in ha_bar

    def test_empty_bars(self):
        result = heikin_ashi([])
        assert result == []

    def test_smoothing(self):
        # HA close is average of OHLC
        bars = make_bars([100, 102, 104])
        result = heikin_ashi(bars)
        # HA close should be average of OHLC
        for i, bar in enumerate(bars):
            ha_close = result[i]["close"]
            expected = (bar.open + bar.high + bar.low + bar.close) / 4.0
            assert abs(ha_close - expected) < 0.01
