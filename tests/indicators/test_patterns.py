"""Tests for pattern indicators."""


from trdr.data import Bar
from trdr.indicators import heikin_ashi, mss, sax_bullish_reversal, sax_pattern


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
