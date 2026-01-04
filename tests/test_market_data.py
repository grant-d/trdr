import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trdr.core import Timeframe
from trdr.data.market import Bar, MarketDataClient, Quote, Symbol


class FakeQuote:
    def __init__(self, bid_price: float, ask_price: float, timestamp: datetime) -> None:
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.timestamp = timestamp


class FakeClient:
    def __init__(self, quote) -> None:
        self._quote = quote

    def get_crypto_latest_quote(self, request):
        return SimpleNamespace(data={"BTC/USD": self._quote})

    def get_stock_latest_quote(self, request):
        return SimpleNamespace(data={"AAPL": self._quote})


def test_symbol_cache_key_format():
    """Cache key includes asset type and is lowercase."""
    crypto = Symbol.parse("crypto:BTC/USD")
    assert crypto.cache_key == "crypto:btc_usd"

    stock = Symbol.parse("AAPL")
    assert stock.cache_key == "stock:aapl"

    stock_explicit = Symbol.parse("stock:MSFT")
    assert stock_explicit.cache_key == "stock:msft"


def test_symbol_equality():
    """Symbols are equal if type and raw match."""
    assert Symbol.parse("crypto:BTC/USD") == Symbol.parse("crypto:BTC/USD")
    assert Symbol.parse("stock:AAPL") == Symbol.parse("AAPL")  # Default is stock
    assert Symbol.parse("crypto:BTC/USD") != Symbol.parse("crypto:ETH/USD")
    assert Symbol.parse("crypto:BTC/USD") != Symbol.parse("stock:BTC/USD")


def test_symbol_equality_case_insensitive_type():
    """Asset type comparison is case-insensitive."""
    assert Symbol(asset_type="CRYPTO", raw="BTC/USD") == Symbol(asset_type="crypto", raw="BTC/USD")
    assert Symbol(asset_type="Stock", raw="AAPL") == Symbol(asset_type="stock", raw="AAPL")


def test_symbol_hash():
    """Symbols are hashable and usable in sets."""
    s = {Symbol.parse("crypto:BTC/USD"), Symbol.parse("crypto:BTC/USD")}
    assert len(s) == 1

    # Case-insensitive type hashing
    s2 = {Symbol(asset_type="CRYPTO", raw="BTC/USD"), Symbol(asset_type="crypto", raw="BTC/USD")}
    assert len(s2) == 1


@pytest.mark.asyncio
async def test_get_current_price_handles_data_wrapper():
    ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    quote = FakeQuote(100.0, 102.0, ts)
    client = MarketDataClient.__new__(MarketDataClient)
    client._crypto_client = FakeClient(quote)
    client._stock_client = FakeClient(quote)

    result = await MarketDataClient.get_current_price(client, Symbol.parse("crypto:BTC/USD"))
    assert isinstance(result, Quote)
    assert result.price == 101.0
    assert result.bid == 100.0
    assert result.ask == 102.0


class FakeBar:
    """Fake Alpaca bar for testing."""

    def __init__(
        self, timestamp: datetime, open_: float, high: float, low: float, close: float, volume: int
    ):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class FakeBarsClient:
    """Fake Alpaca bars client that returns specified bars."""

    def __init__(self, bars: list[FakeBar]):
        self._bars = bars

    def get_crypto_bars(self, request):
        return SimpleNamespace(data={"BTC/USD": self._bars})


def make_bar(hour_offset: int, price: float) -> Bar:
    """Create a bar with timestamp offset from base time."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=hour_offset)
    return Bar(
        timestamp=ts.isoformat(),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=100,
    )


def make_fake_bar(hour_offset: int, price: float) -> FakeBar:
    """Create a fake Alpaca bar for API response."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=hour_offset)
    return FakeBar(ts, price, price + 1, price - 1, price, 100)


def write_cache_csv(cache_file: Path, bars: list[Bar]) -> None:
    """Write bars to cache file in CSV format."""
    with open(cache_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow([bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume])


@pytest.mark.asyncio
async def test_cache_append_preserves_order_no_duplicates(tmp_path: Path):
    """Cache append merges correctly: no duplicates, maintains chronological order."""

    # Create initial cache with 20 bars (hours 0-19)
    initial_bars = [make_bar(i, 100.0 + i) for i in range(20)]
    cache_file = tmp_path / "crypto:btc_usd:1hour.csv"
    write_cache_csv(cache_file, initial_bars)

    # Fake API returns bars for hours 10-25 (overlapping last 10, plus 6 new)
    # Hours 10-19 have UPDATED prices (150 instead of 110-119)
    api_bars = [make_fake_bar(i, 150.0 + i) for i in range(10, 26)]

    # Create client with fake Alpaca client
    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = FakeBarsClient(api_bars)

    # Force cache to be stale by modifying the last bar timestamp to be old
    stale_bars = initial_bars.copy()
    old_ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stale_bars[-1] = Bar(
        timestamp=old_ts.isoformat(),
        open=119.0,
        high=120.0,
        low=118.0,
        close=119.0,
        volume=100,
    )
    write_cache_csv(cache_file, stale_bars)

    # Call get_bars with lookback <= cache size to test overlap behavior
    # (lookback > cache triggers full fetch instead of overlap)
    result = await client.get_bars("crypto:BTC/USD", lookback=20, timeframe=Timeframe.parse("1h"))

    # Cache has 20 bars, API provides 16 bars (hours 10-25)
    # Overlap removes last 10 from cache, keeping bars 0-9
    # Merge: bars 0-9 (cache) + bars 10-25 (API) = 26 bars total
    # Return last 20: bars 6-25
    assert len(result) == 20, f"Expected 20 bars, got {len(result)}"

    # Verify chronological order
    timestamps = [bar.timestamp for bar in result]
    assert timestamps == sorted(timestamps), "Bars not in chronological order"

    # Verify no duplicate timestamps
    assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"

    # First 4 bars (indices 0-3) are from cache (hours 6-9, prices 106-109)
    for i in range(4):
        expected = 100.0 + (6 + i)  # hours 6-9
        assert result[i].close == expected, f"Bar {i}: {result[i].close} != {expected}"

    # Remaining 16 bars (indices 4-19) are from API (hours 10-25, prices 160-175)
    for i in range(4, 20):
        expected = 150.0 + (6 + i)  # hours 10-25
        assert result[i].close == expected, f"Bar {i}: {result[i].close} != {expected}"


@pytest.mark.asyncio
async def test_cache_gap_behavior(tmp_path: Path):
    """Gaps within overlap window get filled; older gaps persist."""

    # Create cache with 30 bars but DELETE hour 17 (gap in overlap zone)
    all_bars = [make_bar(i, 100.0 + i) for i in range(30)]
    bars_with_gap = all_bars[:17] + all_bars[18:]  # Skip hour 17

    cache_file = tmp_path / "crypto:btc_usd:1hour.csv"
    write_cache_csv(cache_file, bars_with_gap)

    assert len(bars_with_gap) == 29, "Should have 29 bars (30 - 1 deleted)"

    # API returns bars 20-35 (overlap zone, includes hour 17 replacement if needed)
    api_bars = [make_fake_bar(i, 100.0 + i) for i in range(20, 36)]

    # Force cache to be stale
    stale_bars = bars_with_gap.copy()
    old_ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stale_bars[-1] = Bar(
        timestamp=old_ts.isoformat(),
        open=129.0, high=130.0, low=128.0, close=129.0, volume=100,
    )
    write_cache_csv(cache_file, stale_bars)

    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = FakeBarsClient(api_bars)

    # Request 25 bars (less than 29 cached to use overlap logic)
    result = await client.get_bars("crypto:BTC/USD", lookback=25, timeframe=Timeframe.parse("1h"))

    timestamps = [bar.timestamp for bar in result]

    # Check no duplicates
    assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"

    # Check chronological order
    assert timestamps == sorted(timestamps), "Bars not in chronological order"

    # Should get 25 bars
    assert len(result) == 25, f"Expected 25 bars, got {len(result)}"


@pytest.mark.asyncio
async def test_cache_backfill_when_lookback_exceeds_cache(tmp_path: Path):
    """Requesting more bars than cached triggers full historical fetch."""

    # Create cache with only 20 bars
    initial_bars = [make_bar(i, 100.0 + i) for i in range(20)]
    cache_file = tmp_path / "crypto:btc_usd:1hour.csv"
    write_cache_csv(cache_file, initial_bars)

    # API returns 100 bars (hours 0-99) for full fetch
    api_bars = [make_fake_bar(i, 200.0 + i) for i in range(100)]

    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = FakeBarsClient(api_bars)

    # Request 50 bars - more than the 20 cached
    # Should trigger full fetch, not overlap fetch
    result = await client.get_bars("crypto:BTC/USD", lookback=50, timeframe=Timeframe.parse("1h"))

    # Should get 50 bars from the API (full fetch replaces cache)
    assert len(result) == 50, f"Expected 50 bars, got {len(result)}"

    # All bars should be from API (prices 200+), not from old cache (100+)
    # The last 50 of 100 API bars
    for i, bar in enumerate(result):
        expected_price = 200.0 + (50 + i)  # API bars 50-99
        assert bar.close == expected_price, f"Bar {i}: {bar.close} != {expected_price}"


@pytest.mark.asyncio
async def test_cache_fresh_returns_cached_bars(tmp_path: Path):
    """Fresh cache with sufficient bars returns from cache without API call."""

    # Create fresh cache (last bar timestamp within 1 hour of now)
    now = datetime.now(timezone.utc)
    fresh_bars = []
    for i in range(30):
        # Last bar is 30 minutes ago (clearly fresh)
        ts = now - timedelta(minutes=30) - timedelta(hours=29 - i)
        fresh_bars.append(Bar(
            timestamp=ts.isoformat(),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=100,
        ))

    cache_file = tmp_path / "crypto:btc_usd:1hour.csv"
    write_cache_csv(cache_file, fresh_bars)

    # Create client with None API client (should not be called)
    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = None  # Would error if called

    # Request 20 bars - cache has 30 fresh bars, should return from cache
    result = await client.get_bars("crypto:BTC/USD", lookback=20, timeframe=Timeframe.parse("1h"))

    assert len(result) == 20, f"Expected 20 bars, got {len(result)}"
    # Should be last 20 bars from cache
    assert result[0].close == 110.0  # Bar index 10 (30 - 20)
    assert result[-1].close == 129.0  # Bar index 29
