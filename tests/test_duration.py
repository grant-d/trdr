"""Tests for Duration class and Duration support in strategy types."""

import pytest

from trdr.core import Duration, Symbol, Timeframe
from trdr.strategy.types import DataRequirement

crypto = Symbol.parse("crypto:BTC/USD")
stock = Symbol.parse("stock:AAPL")


class TestDurationParsing:
    """Tests for Duration.parse()."""

    def test_hours(self) -> None:
        """Parse hour durations."""
        d = Duration.parse("24h")
        assert d.amount == 24
        assert d.unit == "h"

    def test_days(self) -> None:
        """Parse day durations."""
        d = Duration.parse("30d")
        assert d.amount == 30
        assert d.unit == "d"

    def test_weeks(self) -> None:
        """Parse week durations."""
        d = Duration.parse("2w")
        assert d.amount == 2
        assert d.unit == "w"

    def test_months(self) -> None:
        """Parse month durations (capital M)."""
        d = Duration.parse("3M")
        assert d.amount == 3
        assert d.unit == "M"

    def test_years(self) -> None:
        """Parse year durations."""
        d = Duration.parse("1y")
        assert d.amount == 1
        assert d.unit == "y"

    def test_case_insensitive_except_months(self) -> None:
        """Lowercase units work except M."""
        assert Duration.parse("24H").unit == "h"
        assert Duration.parse("30D").unit == "d"
        assert Duration.parse("2W").unit == "w"
        assert Duration.parse("1Y").unit == "y"

    def test_invalid_unit_raises(self) -> None:
        """Invalid unit raises ValueError."""
        with pytest.raises(ValueError):
            Duration.parse("30x")

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError):
            Duration.parse("days30")

    def test_zero_amount_raises(self) -> None:
        """Zero amount raises ValueError."""
        with pytest.raises(ValueError):
            Duration.parse("0d")

    def test_str_representation(self) -> None:
        """String representation matches input."""
        assert str(Duration.parse("30d")) == "30d"
        assert str(Duration.parse("3M")) == "3M"


class TestDurationEquality:
    """Tests for Duration equality and hashing."""

    def test_same_duration(self) -> None:
        """Same durations are equal."""
        assert Duration.parse("30d") == Duration.parse("30d")
        assert Duration.parse("1M") == Duration.parse("1M")
        assert Duration.parse("1y") == Duration.parse("1y")

    def test_equivalent_durations(self) -> None:
        """Equivalent durations are equal (7d == 1w, 24h == 1d)."""
        assert Duration.parse("24h") == Duration.parse("1d")
        assert Duration.parse("7d") == Duration.parse("1w")
        assert Duration.parse("168h") == Duration.parse("1w")  # 7 * 24 = 168

    def test_different_durations(self) -> None:
        """Different durations are not equal."""
        assert Duration.parse("29d") != Duration.parse("1M")  # 29d != 30d
        assert Duration.parse("1w") != Duration.parse("1M")
        assert Duration.parse("1d") != Duration.parse("1w")

    def test_month_equals_30d(self) -> None:
        """1M equals 30d (both are 720 hours)."""
        assert Duration.parse("1M") == Duration.parse("30d")

    def test_hash_equivalent_durations(self) -> None:
        """Equivalent durations have same hash."""
        assert hash(Duration.parse("24h")) == hash(Duration.parse("1d"))
        assert hash(Duration.parse("7d")) == hash(Duration.parse("1w"))
        # Can use in set
        s = {Duration.parse("1d"), Duration.parse("24h")}
        assert len(s) == 1

    def test_compare_with_duration_object(self) -> None:
        """Compare with Duration object directly."""
        assert Duration.parse("30d") == Duration(30, "d")
        assert Duration.parse("1w") == Duration(7, "d")


class TestDurationToBars:
    """Tests for Duration.to_bars()."""

    def test_crypto_15m_30d(self) -> None:
        """30 days of 15m bars for crypto (24/7)."""
        d = Duration.parse("30d")
        bars = d.to_bars(Timeframe.parse("15m"), crypto)
        # 30 days * 24 hours * 4 bars/hour = 2880
        assert bars == 2880

    def test_stock_15m_30d(self) -> None:
        """30 days of 15m bars for stock (6.5h/day, 252 days/yr)."""
        d = Duration.parse("30d")
        bars = d.to_bars(Timeframe.parse("15m"), stock)
        # 30 * 252/365 = ~20.7 trading days
        # 6.5h = 390min, 390/15 = 26 bars/day
        # ~20.7 * 26 = ~538
        assert 500 <= bars <= 600

    def test_crypto_1h_1M(self) -> None:
        """1 month of 1h bars for crypto."""
        d = Duration.parse("1M")
        bars = d.to_bars(Timeframe.parse("1h"), Symbol.parse("crypto:ETH/USD"))
        # 30 days * 24 hours = 720
        assert bars == 720

    def test_crypto_1d_1y(self) -> None:
        """1 year of daily bars for crypto."""
        d = Duration.parse("1y")
        bars = d.to_bars(Timeframe.parse("1d"), crypto)
        # 365 days
        assert bars == 365

    def test_stock_1d_1y(self) -> None:
        """1 year of daily bars for stock."""
        d = Duration.parse("1y")
        bars = d.to_bars(Timeframe.parse("1d"), stock)
        # 252 trading days
        assert bars == 252

    def test_hours_duration(self) -> None:
        """Hour-based duration."""
        d = Duration.parse("48h")
        bars = d.to_bars(Timeframe.parse("15m"), crypto)
        # 48 hours * 4 bars/hour = 192
        assert bars == 192

    def test_weeks_duration(self) -> None:
        """Week-based duration."""
        d = Duration.parse("2w")
        bars = d.to_bars(Timeframe.parse("1d"), crypto)
        # 14 days
        assert bars == 14


class TestDataRequirementDuration:
    """Tests for DataRequirement with Timeframe and Duration."""

    def test_duration_lookback(self) -> None:
        """Duration lookback resolves to bars."""
        req = DataRequirement(
            crypto,
            Timeframe.parse("15m"),
            Duration.parse("1M"),
        )
        assert req.lookback_bars == 2880  # 30 days * 24 * 4

    def test_duration_respects_symbol(self) -> None:
        """Duration resolution accounts for market type."""
        crypto_req = DataRequirement(
            crypto,
            Timeframe.parse("15m"),
            Duration.parse("30d"),
        )
        stock_req = DataRequirement(
            Symbol.parse("stock:AAPL"),
            Timeframe.parse("15m"),
            Duration.parse("30d"),
        )
        # Crypto should have more bars (24/7 vs 6.5h/day)
        assert crypto_req.lookback_bars > stock_req.lookback_bars

    def test_duration_respects_timeframe(self) -> None:
        """Duration resolution accounts for timeframe."""
        req_15m = DataRequirement(
            crypto,
            Timeframe.parse("15m"),
            Duration.parse("1d"),
        )
        req_1h = DataRequirement(
            crypto,
            Timeframe.parse("1h"),
            Duration.parse("1d"),
        )
        # 15m should have 4x more bars than 1h
        assert req_15m.lookback_bars == req_1h.lookback_bars * 4

    def test_key_property(self) -> None:
        """Key property uses timeframe string."""
        req = DataRequirement(
            Symbol.parse("crypto:ETH/USD"),
            Timeframe.parse("15m"),
            Duration.parse("1M"),
        )
        assert req.key == "crypto:ETH/USD:15m"

    def test_role_validation(self) -> None:
        """Role validation still works."""
        with pytest.raises(ValueError):
            DataRequirement(
                crypto,
                Timeframe.parse("15m"),
                Duration.parse("1M"),
                role="invalid",
            )
