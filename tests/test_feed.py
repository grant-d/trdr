"""Tests for Feed class."""

import pytest

from trdr.core import Feed, Symbol, Timeframe


class TestFeedConstruction:
    """Tests for Feed construction."""

    def test_construct_with_objects(self) -> None:
        """Construct Feed from Symbol and Timeframe objects."""
        symbol = Symbol.parse("crypto:BTC/USD")
        timeframe = Timeframe.parse("15m")
        feed = Feed(symbol, timeframe)

        assert feed.symbol == symbol
        assert feed.timeframe == timeframe

    def test_construct_crypto(self) -> None:
        """Construct crypto feed."""
        feed = Feed(Symbol.parse("crypto:ETH/USD"), Timeframe.parse("1h"))
        assert feed.symbol.asset_type == "crypto"
        assert feed.symbol.raw == "ETH/USD"
        assert feed.timeframe.canonical.amount == 1
        assert feed.timeframe.canonical.unit == "h"

    def test_construct_stock(self) -> None:
        """Construct stock feed."""
        feed = Feed(Symbol.parse("stock:AAPL"), Timeframe.parse("1d"))
        assert feed.symbol.asset_type == "stock"
        assert feed.symbol.raw == "AAPL"
        assert feed.timeframe.canonical.amount == 1
        assert feed.timeframe.canonical.unit == "d"


class TestFeedParsing:
    """Tests for Feed.parse()."""

    def test_parse_crypto_15m(self) -> None:
        """Parse crypto:BTC/USD:15m."""
        feed = Feed.parse("crypto:BTC/USD:15m")
        assert str(feed.symbol) == "crypto:BTC/USD"
        assert str(feed.timeframe) == "15m"

    def test_parse_stock_1d(self) -> None:
        """Parse stock:AAPL:1d."""
        feed = Feed.parse("stock:AAPL:1d")
        assert str(feed.symbol) == "stock:AAPL"
        assert str(feed.timeframe) == "1d"

    def test_parse_crypto_4h(self) -> None:
        """Parse crypto:ETH/USD:4h."""
        feed = Feed.parse("crypto:ETH/USD:4h")
        assert feed.symbol.raw == "ETH/USD"
        assert feed.timeframe.canonical.amount == 4
        assert feed.timeframe.canonical.unit == "h"

    def test_parse_invalid_format_missing_colon(self) -> None:
        """Parse fails without colon separator."""
        with pytest.raises(ValueError, match="Invalid timeframe format"):
            Feed.parse("crypto:BTC/USD15m")

    def test_parse_invalid_format_no_timeframe(self) -> None:
        """Parse fails without timeframe."""
        with pytest.raises(ValueError, match="Invalid timeframe format"):
            Feed.parse("crypto:BTC/USD:")

    def test_parse_only_one_component(self) -> None:
        """Parse fails with only one component."""
        with pytest.raises(ValueError, match="Invalid feed format"):
            Feed.parse("15m")


class TestFeedStringRepresentation:
    """Tests for Feed string conversion."""

    def test_str_crypto(self) -> None:
        """String representation of crypto feed."""
        feed = Feed(Symbol.parse("crypto:BTC/USD"), Timeframe.parse("15m"))
        assert str(feed) == "crypto:BTC/USD:15m"

    def test_str_stock(self) -> None:
        """String representation of stock feed."""
        feed = Feed(Symbol.parse("stock:AAPL"), Timeframe.parse("1d"))
        assert str(feed) == "stock:AAPL:1d"

    def test_str_roundtrip(self) -> None:
        """String representation can be parsed back."""
        original = Feed.parse("crypto:ETH/USD:4h")
        string = str(original)
        parsed = Feed.parse(string)
        assert str(parsed) == str(original)


class TestFeedEquality:
    """Tests for Feed equality."""

    def test_same_feeds_equal(self) -> None:
        """Same feeds are equal."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:BTC/USD:15m")
        assert feed1 == feed2

    def test_different_symbols_not_equal(self) -> None:
        """Different symbols are not equal."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:ETH/USD:15m")
        assert feed1 != feed2

    def test_different_timeframes_not_equal(self) -> None:
        """Different timeframes are not equal."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:BTC/USD:1h")
        assert feed1 != feed2

    def test_equivalent_timeframes_equal(self) -> None:
        """Equivalent timeframes are equal (60m == 1h)."""
        feed1 = Feed.parse("crypto:BTC/USD:60m")
        feed2 = Feed.parse("crypto:BTC/USD:1h")
        assert feed1 == feed2


class TestFeedHash:
    """Tests for Feed hashing."""

    def test_hash_consistent(self) -> None:
        """Same feed has consistent hash."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:BTC/USD:15m")
        assert hash(feed1) == hash(feed2)

    def test_usable_in_set(self) -> None:
        """Feed can be used in set."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:BTC/USD:15m")
        feed3 = Feed.parse("crypto:ETH/USD:15m")

        feeds = {feed1, feed2, feed3}
        assert len(feeds) == 2  # feed1 and feed2 are same

    def test_usable_as_dict_key(self) -> None:
        """Feed can be used as dict key."""
        feed1 = Feed.parse("crypto:BTC/USD:15m")
        feed2 = Feed.parse("crypto:ETH/USD:1h")

        data = {feed1: "btc_data", feed2: "eth_data"}
        assert data[feed1] == "btc_data"
        assert data[feed2] == "eth_data"
