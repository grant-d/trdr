"""Tests for trading calendar."""

from trdr.backtest.calendar import (
    filter_trading_bars,
    get_trading_days_in_year,
    is_market_hours,
    is_trading_day,
)
from trdr.data.market import Bar


def make_bar(timestamp: str) -> Bar:
    """Create test bar with given timestamp."""
    return Bar(
        timestamp=timestamp,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000,
    )


class TestIsTradingDay:
    """Tests for is_trading_day."""

    def test_crypto_always_trading(self) -> None:
        """Crypto trades every day."""
        # Weekend
        assert is_trading_day("2024-01-06T12:00:00Z", "crypto")  # Saturday
        assert is_trading_day("2024-01-07T12:00:00Z", "crypto")  # Sunday

        # Weekday
        assert is_trading_day("2024-01-08T12:00:00Z", "crypto")  # Monday

        # Holiday
        assert is_trading_day("2024-12-25T12:00:00Z", "crypto")  # Christmas

    def test_stock_weekday_trading(self) -> None:
        """Stocks trade on weekdays."""
        assert is_trading_day("2024-01-08T12:00:00Z", "stock")  # Monday
        assert is_trading_day("2024-01-09T12:00:00Z", "stock")  # Tuesday
        assert is_trading_day("2024-01-10T12:00:00Z", "stock")  # Wednesday
        assert is_trading_day("2024-01-11T12:00:00Z", "stock")  # Thursday
        assert is_trading_day("2024-01-12T12:00:00Z", "stock")  # Friday

    def test_stock_no_weekend_trading(self) -> None:
        """Stocks don't trade on weekends."""
        assert not is_trading_day("2024-01-06T12:00:00Z", "stock")  # Saturday
        assert not is_trading_day("2024-01-07T12:00:00Z", "stock")  # Sunday

    def test_stock_no_holiday_trading(self) -> None:
        """Stocks don't trade on major holidays."""
        assert not is_trading_day("2024-01-01T12:00:00Z", "stock")  # New Year's
        assert not is_trading_day("2024-07-04T12:00:00Z", "stock")  # Independence Day
        assert not is_trading_day("2024-12-25T12:00:00Z", "stock")  # Christmas


class TestIsMarketHours:
    """Tests for is_market_hours."""

    def test_crypto_always_open(self) -> None:
        """Crypto markets 24/7."""
        assert is_market_hours("2024-01-06T03:00:00Z", "crypto")  # 3 AM Saturday
        assert is_market_hours("2024-01-08T23:00:00Z", "crypto")  # 11 PM Monday

    def test_stock_closed_on_weekend(self) -> None:
        """Stock market closed on weekends."""
        assert not is_market_hours("2024-01-06T15:00:00Z", "stock")  # Saturday


class TestGetTradingDaysInYear:
    """Tests for get_trading_days_in_year."""

    def test_crypto_365(self) -> None:
        """Crypto trades 365 days."""
        assert get_trading_days_in_year("crypto") == 365

    def test_stock_252(self) -> None:
        """Stocks trade ~252 days."""
        assert get_trading_days_in_year("stock") == 252


class TestFilterTradingBars:
    """Tests for filter_trading_bars."""

    def test_crypto_no_filter(self) -> None:
        """Crypto bars not filtered."""
        bars = [
            make_bar("2024-01-06T10:00:00Z"),  # Saturday
            make_bar("2024-01-07T10:00:00Z"),  # Sunday
            make_bar("2024-01-08T10:00:00Z"),  # Monday
        ]
        filtered = filter_trading_bars(bars, "crypto")
        assert len(filtered) == 3

    def test_stock_filter_weekends(self) -> None:
        """Stock bars filter out weekends."""
        bars = [
            make_bar("2024-01-05T10:00:00Z"),  # Friday
            make_bar("2024-01-06T10:00:00Z"),  # Saturday
            make_bar("2024-01-07T10:00:00Z"),  # Sunday
            make_bar("2024-01-08T10:00:00Z"),  # Monday
        ]
        filtered = filter_trading_bars(bars, "stock")
        assert len(filtered) == 2
        assert filtered[0].timestamp == "2024-01-05T10:00:00Z"
        assert filtered[1].timestamp == "2024-01-08T10:00:00Z"

    def test_stock_filter_holidays(self) -> None:
        """Stock bars filter out holidays."""
        bars = [
            make_bar("2024-12-24T10:00:00Z"),  # Christmas Eve (trading day)
            make_bar("2024-12-25T10:00:00Z"),  # Christmas (holiday)
            make_bar("2024-12-26T10:00:00Z"),  # Day after (trading day)
        ]
        filtered = filter_trading_bars(bars, "stock")
        assert len(filtered) == 2
