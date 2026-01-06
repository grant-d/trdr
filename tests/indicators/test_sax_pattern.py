"""Tests for SAX pattern indicator."""

from trdr.data import Bar
from trdr.indicators.sax_pattern import SaxPatternIndicator, sax_bullish_reversal


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


class TestSaxPattern:
    """Tests for SAX pattern generation."""

    def test_returns_string(self) -> None:
        bars = make_bars([100, 102, 104, 106, 108] * 4)
        result = SaxPatternIndicator.calculate(bars, window=20, segments=5)
        assert isinstance(result, str)
        assert len(result) == 5

    def test_alphabet_range(self) -> None:
        bars = make_bars([100, 102, 104, 106, 108] * 4)
        result = SaxPatternIndicator.calculate(bars, window=20, segments=5)
        assert all(c in "abcde" for c in result)

    def test_flat_market(self) -> None:
        bars = make_bars([100] * 20)
        result = SaxPatternIndicator.calculate(bars, window=20, segments=5)
        assert result == "ccccc"

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        result = SaxPatternIndicator.calculate(bars, window=20, segments=5)
        assert result == ""


class TestSaxBullishReversal:
    """Tests for SAX bullish reversal detection."""

    def test_bullish_pattern(self) -> None:
        assert sax_bullish_reversal("aabde") is True
        assert sax_bullish_reversal("abbde") is True

    def test_bearish_pattern(self) -> None:
        assert sax_bullish_reversal("ddddd") is False
        assert sax_bullish_reversal("eeeee") is False

    def test_insufficient_length(self) -> None:
        assert sax_bullish_reversal("ab") is False
        assert sax_bullish_reversal("") is False

    def test_no_momentum(self) -> None:
        assert sax_bullish_reversal("aabaa") is False


class TestSaxPatternIndicator:
    """Tests for SaxPatternIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = SaxPatternIndicator(window=20, segments=5)
        for bar in make_bars([100, 102, 104, 106, 108] * 4):
            value = ind.update(bar)
        assert isinstance(value, str)
        assert isinstance(ind.value, str)
